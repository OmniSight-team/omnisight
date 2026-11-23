"""OMNI-907: the alerts table, the API and the roles (FR-6, FR-9, NFR-7; D-51).

Requires Postgres (`docker compose up -d`); runs on the test database (conftest). Alerts are
inserted directly, as `pfm` will raise them; the API only reads and reviews them.
"""

from datetime import UTC, datetime, timedelta

import pytest
from app.db import SessionLocal
from app.main import app
from app.models import Alert, Camera, Segment, User, Video
from app.security import hash_password
from fastapi.testclient import TestClient
from omnisight_common import (
    AlertSeverity,
    AlertSource,
    AlertStatus,
    CameraStatus,
    Domain,
    Role,
    SegmentSourceType,
    VideoSource,
    VideoStatus,
)
from sqlalchemy.exc import IntegrityError

client = TestClient(app)

TEST_PASSWORD = "Omni907Pass!"
EMAILS = {
    Role.OPERATOR: "omni907-operator@omnisight.local",
    Role.ANALYST: "omni907-analyst@omnisight.local",
    Role.ADMIN: "omni907-admin@omnisight.local",
}
INACTIVE = "omni907-inactive@omnisight.local"
T0 = datetime(2026, 11, 23, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module", autouse=True)
def users():
    with SessionLocal() as s:
        for role, email in EMAILS.items():
            s.add(User(email=email, password_hash=hash_password(TEST_PASSWORD), role=role, is_active=True))
        s.add(User(email=INACTIVE, password_hash=hash_password(TEST_PASSWORD), role=Role.ANALYST, is_active=False))
        s.commit()
    yield
    with SessionLocal() as s:
        s.query(Alert).delete()
        s.query(User).filter(User.email.in_([*EMAILS.values(), INACTIVE])).delete(synchronize_session=False)
        s.commit()


def _headers(role: Role) -> dict[str, str]:
    resp = client.post("/auth/login", json={"email": EMAILS[role], "password": TEST_PASSWORD})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _user_id(email: str) -> int:
    with SessionLocal() as s:
        return s.query(User.id).filter(User.email == email).scalar()


@pytest.fixture
def footage():
    """A camera with one live frame, and a video with one frame; four alerts on them:
    rule/camera old, model/video new, model/video expected sooner, rule/camera dismissed."""
    with SessionLocal() as s:
        s.query(Alert).delete()
        camera = Camera(name="omni907 gate", rtsp_url="rtsp://x/omni907", location="gate", group_name="g",
                        domain=Domain.SURVEILLANCE, status=CameraStatus.REGISTERED)
        video = Video(filename="omni907.mp4", path="/data/videos/omni907.mp4", source=VideoSource.UPLOAD,
                      size_bytes=1, status=VideoStatus.READY)
        s.add_all([camera, video])
        s.flush()
        live = Segment(camera_id=camera.id, frame_index=0, timestamp_s=1_000.0, thumb_path="/data/thumbs/omni907/l.jpg",
                       admitted=True, source_type=SegmentSourceType.CAMERA)
        frame = Segment(video_id=video.id, frame_index=30, timestamp_s=12.5, thumb_path="/data/thumbs/omni907/v.jpg",
                        admitted=True, source_type=SegmentSourceType.UPLOAD)
        s.add_all([live, frame])
        s.flush()
        alerts = {
            "rule_old": Alert(camera_id=camera.id, segment_id=live.id, event_type="sustained motion in zone",
                              severity=AlertSeverity.LOW, source=AlertSource.RULE, reasoning="motion 12 s in zone",
                              raised_at=T0),
            "model_late": Alert(video_id=video.id, segment_id=frame.id, event_type="Robbery", confidence=0.81,
                                expected_in_s=30.0, severity=AlertSeverity.HIGH, source=AlertSource.MODEL,
                                raised_at=T0 + timedelta(minutes=2)),
            "model_soon": Alert(video_id=video.id, segment_id=frame.id, event_type="Crash", confidence=0.66,
                                expected_in_s=3.0, severity=AlertSeverity.MEDIUM, source=AlertSource.MODEL,
                                raised_at=T0 + timedelta(minutes=1)),
            "dismissed": Alert(camera_id=camera.id, event_type="activity outside hours", severity=AlertSeverity.LOW,
                               source=AlertSource.RULE, status=AlertStatus.DISMISSED, raised_at=T0 - timedelta(hours=1),
                               reviewed_by=_user_id(EMAILS[Role.OPERATOR]), reviewed_at=T0),
        }
        s.add_all(alerts.values())
        s.commit()
        ids = {name: a.id for name, a in alerts.items()} | {"camera": camera.id, "video": video.id}
        segments = [live.id, frame.id]
    yield ids
    with SessionLocal() as s:
        s.query(Alert).delete()
        s.query(Segment).filter(Segment.id.in_(segments)).delete(synchronize_session=False)
        s.query(Video).filter(Video.id == ids["video"]).delete()
        s.query(Camera).filter(Camera.id == ids["camera"]).delete()
        s.commit()


def _ids(resp) -> list[int]:
    assert resp.status_code == 200, resp.text
    return [a["id"] for a in resp.json()]


def test_list_is_newest_first_with_camera_video_moment_and_thumbnail(footage):
    resp = client.get("/alerts", headers=_headers(Role.OPERATOR))

    assert _ids(resp) == [footage["model_late"], footage["model_soon"], footage["rule_old"], footage["dismissed"]]
    by_id = {a["id"]: a for a in resp.json()}
    rule, model = by_id[footage["rule_old"]], by_id[footage["model_late"]]
    assert (rule["camera_name"], rule["video_id"], rule["timestamp_s"]) == ("omni907 gate", None, 1_000.0)
    assert rule["thumb_url"].endswith("/omni907/l.jpg") and rule["reasoning"] == "motion 12 s in zone"
    assert (model["video_name"], model["camera_id"], model["timestamp_s"]) == ("omni907.mp4", None, 12.5)
    assert (model["confidence"], model["expected_in_s"], model["severity"], model["source"]) == (0.81, 30.0, "high", "model")
    assert model["status"] == "new" and model["tier_reached"] is None


def test_soonest_expected_first_and_rules_last(footage):
    # model_soon is expected at T0+1 min+3 s, model_late at T0+2 min+30 s; rules have no expected time.
    resp = client.get("/alerts?sort=soonest", headers=_headers(Role.ANALYST))

    assert _ids(resp) == [footage["model_soon"], footage["model_late"], footage["rule_old"], footage["dismissed"]]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("status=new", ["model_late", "model_soon", "rule_old"]),
        ("status=dismissed", ["dismissed"]),
        ("source=rule", ["rule_old", "dismissed"]),
        ("severity=low", ["rule_old", "dismissed"]),
        ("severity=high&source=model", ["model_late"]),
        ("limit=1", ["model_late"]),
    ],
)
def test_filters(footage, query, expected):
    assert _ids(client.get(f"/alerts?{query}", headers=_headers(Role.ADMIN))) == [footage[n] for n in expected]


def test_filters_by_camera_and_by_video(footage):
    h = _headers(Role.OPERATOR)
    assert _ids(client.get(f"/alerts?camera_id={footage['camera']}", headers=h)) == [footage["rule_old"], footage["dismissed"]]
    assert _ids(client.get(f"/alerts?video_id={footage['video']}", headers=h)) == [footage["model_late"], footage["model_soon"]]


def test_a_bad_filter_value_is_refused(footage):
    assert client.get("/alerts?severity=extreme", headers=_headers(Role.OPERATOR)).status_code == 422
    assert client.get("/alerts?sort=random", headers=_headers(Role.OPERATOR)).status_code == 422


def test_unread_counts_the_new_ones(footage):
    resp = client.get("/alerts/unread_count", headers=_headers(Role.ADMIN))
    assert resp.status_code == 200 and resp.json() == {"unread": 3}


@pytest.mark.parametrize(("action", "role", "outcome"), [("confirm", Role.OPERATOR, "confirmed"), ("dismiss", Role.ANALYST, "dismissed")])
def test_a_review_is_recorded_with_who_and_when_and_survives(footage, action, role, outcome):
    before = datetime.now(UTC)
    resp = client.post(f"/alerts/{footage['model_late']}/{action}", headers=_headers(role))

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["status"], body["reviewed_by_email"]) == (outcome, EMAILS[role])
    assert before <= datetime.fromisoformat(body["reviewed_at"]) <= datetime.now(UTC)
    # Read back from the database in a new session: what a restart would see.
    with SessionLocal() as s:
        stored = s.get(Alert, footage["model_late"])
        assert (stored.status, stored.reviewed_by) == (AlertStatus(outcome), _user_id(EMAILS[role]))
    assert client.get("/alerts/unread_count", headers=_headers(role)).json() == {"unread": 2}


def test_an_alert_is_reviewed_once(footage):
    h = _headers(Role.OPERATOR)
    assert client.post(f"/alerts/{footage['model_soon']}/confirm", headers=h).status_code == 200

    for action in ("confirm", "dismiss"):
        resp = client.post(f"/alerts/{footage['model_soon']}/{action}", headers=_headers(Role.ANALYST))
        assert resp.status_code == 409
    with SessionLocal() as s:
        stored = s.get(Alert, footage["model_soon"])
        assert (stored.status, stored.reviewed_by) == (AlertStatus.CONFIRMED, _user_id(EMAILS[Role.OPERATOR]))


def test_an_unknown_alert_is_404(footage):
    h = _headers(Role.OPERATOR)
    for action in ("confirm", "dismiss"):
        assert client.post(f"/alerts/999999/{action}", headers=h).status_code == 404
    assert client.post("/alerts/999999/assign", json={"user_id": 1}, headers=h).status_code == 404


def test_assign_to_a_colleague_who_can_review(footage):
    analyst = _user_id(EMAILS[Role.ANALYST])
    resp = client.post(f"/alerts/{footage['rule_old']}/assign", json={"user_id": analyst}, headers=_headers(Role.OPERATOR))

    assert resp.status_code == 200
    assert (resp.json()["assigned_to"], resp.json()["assigned_to_email"], resp.json()["status"]) == (analyst, EMAILS[Role.ANALYST], "new")


@pytest.mark.parametrize("who", ["admin", "inactive", "nobody"])
def test_assign_refuses_someone_who_cannot_review(footage, who):
    user_id = {"admin": _user_id(EMAILS[Role.ADMIN]), "inactive": _user_id(INACTIVE), "nobody": 999999}[who]
    resp = client.post(f"/alerts/{footage['rule_old']}/assign", json={"user_id": user_id}, headers=_headers(Role.OPERATOR))
    assert resp.status_code == 422


def test_a_reviewed_alert_cannot_be_assigned(footage):
    resp = client.post(f"/alerts/{footage['dismissed']}/assign", json={"user_id": _user_id(EMAILS[Role.ANALYST])},
                       headers=_headers(Role.OPERATOR))
    assert resp.status_code == 409


# NFR-7, tested like TC-10: the right role 200, the wrong role 403, no or a bad token 401.
@pytest.mark.parametrize("action", ["confirm", "dismiss", "assign"])
def test_only_operators_and_analysts_may_review_or_assign(footage, action):
    body = {"user_id": _user_id(EMAILS[Role.ANALYST])} if action == "assign" else None
    resp = client.post(f"/alerts/{footage['rule_old']}/{action}", json=body, headers=_headers(Role.ADMIN))
    assert resp.status_code == 403
    with SessionLocal() as s:
        stored = s.get(Alert, footage["rule_old"])
        assert (stored.status, stored.assigned_to) == (AlertStatus.NEW, None)


@pytest.mark.parametrize("role", list(EMAILS))
def test_every_role_with_the_alerts_screen_may_read(footage, role):
    h = _headers(role)
    assert client.get("/alerts", headers=h).status_code == 200
    assert client.get("/alerts/unread_count", headers=h).status_code == 200


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer not-a-token"}], ids=["no token", "bad token"])
@pytest.mark.parametrize(
    ("method", "path"),
    [("get", "/alerts"), ("get", "/alerts/unread_count"), ("post", "/alerts/{id}/confirm"),
     ("post", "/alerts/{id}/dismiss"), ("post", "/alerts/{id}/assign")],
)
def test_no_or_a_bad_token_is_401(footage, headers, method, path):
    resp = getattr(client, method)(path.format(id=footage["rule_old"]), headers=headers, **({"json": {"user_id": 1}} if method == "post" else {}))
    assert resp.status_code == 401


def test_the_table_holds_an_alert_on_a_camera_or_a_video_never_both_or_neither(footage):
    for where in ({}, {"camera_id": footage["camera"], "video_id": footage["video"]}):
        with SessionLocal() as s:
            s.add(Alert(event_type="x", severity=AlertSeverity.LOW, source=AlertSource.RULE, **where))
            with pytest.raises(IntegrityError):
                s.commit()


def test_a_reviewed_status_needs_its_time(footage):
    with SessionLocal() as s:
        s.add(Alert(camera_id=footage["camera"], event_type="x", severity=AlertSeverity.LOW, source=AlertSource.RULE,
                    status=AlertStatus.CONFIRMED))
        with pytest.raises(IntegrityError):
            s.commit()
