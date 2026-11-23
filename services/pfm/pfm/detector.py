"""The rule detector's job (OMNI-908): read each camera's newest frames, raise an alert per run.

Stateless between looks: every look reads the last `rules_lookback_s` of each camera's motion
frames from `segments`, follows a run that reaches the window's start back to its real start, and
asks the rules what fired. Whether a run already has its alert is read from `alerts` (an alert of
the same rule on the same camera, on a frame inside the run), so a restart never raises the same
run twice and never misses one that fired while the service was down (within the lookback).

Raw SQL over the tables `query` owns (its Alembic migrations): this service writes `alerts` rows
only, as `query`'s model describes them.
"""

import logging
import time
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from sqlalchemy import Engine, text

from pfm import rules

log = logging.getLogger(__name__)

CAMERAS = text("""
    SELECT id, zone_left, zone_top, zone_right, zone_bottom, sustained_motion_s, active_from, active_until
    FROM cameras WHERE status <> 'disabled' ORDER BY id
""")
# A live camera's frames have no video. Motion and heartbeats are both read: a heartbeat
# between two motion frames ends a run as "no motion here".
FRAMES = text("""
    SELECT id, timestamp_s, admission_reason = 'motion' AS motion, motion_cells, motion_grid
    FROM segments
    WHERE camera_id = :camera AND video_id IS NULL AND admitted AND timestamp_s >= :since
    ORDER BY timestamp_s, id
""")
ALREADY = text("""
    SELECT 1 FROM alerts a JOIN segments s ON s.id = a.segment_id
    WHERE a.camera_id = :camera AND a.event_type = :rule AND a.source = 'rule'
      AND s.timestamp_s BETWEEN :start AND :end
    LIMIT 1
""")
RAISE = text("""
    INSERT INTO alerts (camera_id, segment_id, event_type, severity, source, reasoning)
    VALUES (:camera, :segment, :rule, :severity, 'rule', :reason)
    RETURNING id
""")


@dataclass
class Detector:
    engine: Engine
    lookback_s: float = 120.0
    max_gap_s: float = 2.0
    quiet_s: float = 60.0
    max_history_s: float = 7200.0
    tz: ZoneInfo = field(default_factory=lambda: ZoneInfo("Asia/Jerusalem"))
    raised: int = 0
    last_look_at: float | None = None
    last_error: str | None = None

    def _frames(self, conn, camera: int, now: float) -> list[rules.Frame]:
        """The window's frames; while a motion frame sits at the window's start, the window
        reaches further back, so a run is always seen from its start."""
        since = now - self.lookback_s
        while True:
            rows = conn.execute(FRAMES, {"camera": camera, "since": since}).all()
            first_motion = next((r.timestamp_s for r in rows if r.motion), None)
            reach = max(self.max_gap_s, self.quiet_s)
            if first_motion is None or first_motion - since > reach or since <= now - self.max_history_s:
                break
            since -= self.lookback_s
        return [rules.Frame(r.id, r.timestamp_s, bool(r.motion), r.motion_cells, r.motion_grid) for r in rows]

    def look(self, now: float | None = None) -> list[int]:
        """One look at every camera; returns the ids of the alerts raised."""
        now = time.time() if now is None else now
        raised = []
        with self.engine.begin() as conn:
            for cam in conn.execute(CAMERAS).all():
                frames = self._frames(conn, cam.id, now)
                if not frames:
                    continue
                zone = rules.Zone(cam.zone_left, cam.zone_top, cam.zone_right, cam.zone_bottom)
                firings = rules.sustained_motion(frames, zone, cam.sustained_motion_s, self.max_gap_s, self.tz)
                if cam.active_from is not None:
                    firings += rules.activity_outside_hours(frames, cam.active_from, cam.active_until, self.quiet_s, self.tz)
                for f in firings:
                    seen = {"camera": cam.id, "rule": f.rule, "start": f.run_start_s, "end": f.run_end_s}
                    if conn.execute(ALREADY, seen).first() is not None:
                        continue
                    alert_id = conn.execute(RAISE, {
                        "camera": cam.id, "segment": f.segment_id, "rule": f.rule,
                        "severity": rules.SEVERITY[f.rule], "reason": f.reason,
                    }).scalar_one()
                    raised.append(alert_id)
                    log.info("alert %s: camera %s, %s (%s)", alert_id, cam.id, f.rule, f.reason)
        self.raised += len(raised)
        self.last_look_at = now
        return raised

    def run_forever(self, interval_s: float, stop) -> None:
        """Look every `interval_s` until `stop` (a threading.Event) is set. A failed look is
        logged and kept for /health; the next one tries again."""
        while not stop.is_set():
            try:
                self.look()
                self.last_error = None
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.exception("rule look failed")
            stop.wait(interval_s)
