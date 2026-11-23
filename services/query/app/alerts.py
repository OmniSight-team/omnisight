"""The alerts inbox (OMNI-907): list, confirm, dismiss, assign, and the unread count.

Alerts are raised by `pfm` (rules in slice 9, the PFM in slice 10), not through this API. Here
they are read and reviewed. The lifecycle: raised (`new`) -> reviewed -> `confirmed` or
`dismissed`, once, with the reviewing user and the time. Reading is open to every role that has
the Alerts screen (`/nav`); confirming, dismissing and assigning only to operators and analysts
(the plan's reviewing roles, NFR-7). Severity levels and "unread" are D-51's.
"""

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from omnisight_common import AlertSeverity, AlertSource, AlertStatus, Role
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session, aliased

from app.db import get_db
from app.deps import get_current_user, require_role
from app.models import Alert, Camera, Segment, User, Video
from app.videos import _thumb_url

router = APIRouter(prefix="/alerts", tags=["alerts"])

REVIEWERS = (Role.OPERATOR, Role.ANALYST)
ALERT_NOT_FOUND_MESSAGE = "alert not found"
ALREADY_REVIEWED_MESSAGE = "alert already reviewed"
NOT_A_REVIEWER_MESSAGE = "an alert can only be assigned to an active operator or analyst"
MAX_LIMIT = 500


class AlertOut(BaseModel):
    id: int
    camera_id: int | None
    camera_name: str | None
    video_id: int | None
    video_name: str | None
    segment_id: int | None
    # The moment in the footage the alert points at: seconds into the video, or the live frame's time.
    timestamp_s: float | None
    thumb_url: str | None
    event_type: str
    confidence: float | None
    expected_in_s: float | None
    severity: AlertSeverity
    status: AlertStatus
    source: AlertSource
    reasoning: str | None
    tier_reached: int | None
    raised_at: datetime
    reviewed_by: int | None
    reviewed_by_email: str | None
    reviewed_at: datetime | None
    assigned_to: int | None
    assigned_to_email: str | None


class AssignIn(BaseModel):
    user_id: int


class UnreadOut(BaseModel):
    unread: int


Reviewer, Assignee = aliased(User), aliased(User)


def _query():
    return (
        select(Alert, Camera.name, Video.filename, Segment.timestamp_s, Segment.thumb_path, Reviewer.email, Assignee.email)
        .outerjoin(Camera, Camera.id == Alert.camera_id)
        .outerjoin(Video, Video.id == Alert.video_id)
        .outerjoin(Segment, Segment.id == Alert.segment_id)
        .outerjoin(Reviewer, Reviewer.id == Alert.reviewed_by)
        .outerjoin(Assignee, Assignee.id == Alert.assigned_to)
    )


def _out(row) -> AlertOut:
    a, camera_name, video_name, timestamp_s, thumb_path, reviewer, assignee = row
    return AlertOut(
        id=a.id, camera_id=a.camera_id, camera_name=camera_name, video_id=a.video_id, video_name=video_name,
        segment_id=a.segment_id, timestamp_s=timestamp_s, thumb_url=_thumb_url(thumb_path),
        event_type=a.event_type, confidence=a.confidence, expected_in_s=a.expected_in_s, severity=a.severity,
        status=a.status, source=a.source, reasoning=a.reasoning, tier_reached=a.tier_reached,
        raised_at=a.raised_at, reviewed_by=a.reviewed_by, reviewed_by_email=reviewer, reviewed_at=a.reviewed_at,
        assigned_to=a.assigned_to, assigned_to_email=assignee,
    )


def _one(db: Session, alert_id: int) -> AlertOut:
    row = db.execute(_query().where(Alert.id == alert_id)).first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=ALERT_NOT_FOUND_MESSAGE)
    return _out(row)


def _new_alert(db: Session, alert_id: int) -> Alert:
    """The alert, locked for this transaction; 404 if there is none, 409 if it was already reviewed.
    The lock means two reviewers deciding at once cannot both win."""
    alert = db.execute(select(Alert).where(Alert.id == alert_id).with_for_update()).scalar_one_or_none()
    if alert is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=ALERT_NOT_FOUND_MESSAGE)
    if alert.status != AlertStatus.NEW:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=ALREADY_REVIEWED_MESSAGE)
    return alert


@router.get("", response_model=list[AlertOut])
def list_alerts(
    status_: AlertStatus | None = Query(None, alias="status"),
    camera_id: int | None = None,
    video_id: int | None = None,
    severity: AlertSeverity | None = None,
    source: AlertSource | None = None,
    sort: Literal["newest", "soonest"] = "newest",
    limit: int = Query(100, ge=1, le=MAX_LIMIT),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> list[AlertOut]:
    """Newest first, or soonest expected first: the moment the event is expected
    (`raised_at + expected_in_s`); alerts with no expected time (rules) come last, newest first."""
    query = _query()
    for column, value in ((Alert.status, status_), (Alert.camera_id, camera_id), (Alert.video_id, video_id),
                          (Alert.severity, severity), (Alert.source, source)):
        if value is not None:
            query = query.where(column == value)
    if sort == "soonest":
        expected_at = Alert.raised_at + func.make_interval(0, 0, 0, 0, 0, 0, Alert.expected_in_s)
        query = query.order_by(expected_at.asc().nulls_last(), Alert.raised_at.desc(), Alert.id.desc())
    else:
        query = query.order_by(Alert.raised_at.desc(), Alert.id.desc())
    return [_out(row) for row in db.execute(query.limit(limit)).all()]


@router.get("/unread_count", response_model=UnreadOut)
def unread_count(db: Session = Depends(get_db), _user: User = Depends(get_current_user)) -> UnreadOut:
    """Alerts not reviewed yet (status `new`), for the sidebar's badge (D-51)."""
    return UnreadOut(unread=db.scalar(select(func.count()).select_from(Alert).where(Alert.status == AlertStatus.NEW)))


def _review(alert_id: int, outcome: AlertStatus, db: Session, user: User) -> AlertOut:
    alert = _new_alert(db, alert_id)
    alert.status = outcome
    alert.reviewed_by = user.id
    alert.reviewed_at = datetime.now(UTC)
    db.commit()
    return _one(db, alert_id)


@router.post("/{alert_id}/confirm", response_model=AlertOut)
def confirm_alert(alert_id: int, db: Session = Depends(get_db), user: User = Depends(require_role(*REVIEWERS))) -> AlertOut:
    return _review(alert_id, AlertStatus.CONFIRMED, db, user)


@router.post("/{alert_id}/dismiss", response_model=AlertOut)
def dismiss_alert(alert_id: int, db: Session = Depends(get_db), user: User = Depends(require_role(*REVIEWERS))) -> AlertOut:
    return _review(alert_id, AlertStatus.DISMISSED, db, user)


@router.post("/{alert_id}/assign", response_model=AlertOut)
def assign_alert(
    alert_id: int, payload: AssignIn, db: Session = Depends(get_db), _user: User = Depends(require_role(*REVIEWERS))
) -> AlertOut:
    """Hand a new alert to a colleague who can review it. Assigning again replaces the assignee."""
    alert = _new_alert(db, alert_id)
    assignee = db.get(User, payload.user_id)
    if assignee is None or not assignee.is_active or assignee.role not in REVIEWERS:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=NOT_A_REVIEWER_MESSAGE)
    alert.assigned_to = assignee.id
    db.commit()
    return _one(db, alert_id)
