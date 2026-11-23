"""OMNI-908: the detector's job against the test database: alerts raised, once per run.

Cameras and their live frames are inserted directly, as the consumer stores them; the detector is
run with an explicit `now`. Every row a test makes is removed after it, by id.
"""

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest
from pfm import rules
from pfm.detector import Detector
from pfm.settings import settings
from sqlalchemy import create_engine, text

TZ = ZoneInfo("Asia/Jerusalem")
NOON = datetime(2026, 11, 23, 12, 0, tzinfo=TZ).timestamp()
NIGHT = datetime(2026, 11, 23, 2, 0, tzinfo=TZ).timestamp()
ALL = -1  # every bit of the 64-bit mask: all 64 cells moved

engine = create_engine(settings.database_url)
assert settings.postgres_db == "omnisight_test", "tests must never run on the stack's database"


@pytest.fixture
def world():
    made = {"cameras": [], "segments": []}

    def camera(**rules_settings) -> int:
        cols = {"sustained_motion_s": 5.0, **rules_settings}
        with engine.begin() as c:
            cam = c.execute(text(
                "INSERT INTO cameras (name, rtsp_url, location, group_name, domain, status, "
                + ", ".join(cols) + ") VALUES ('omni908 cam', 'rtsp://x/908', 'here', 'g', 'surveillance', "
                "'connected', " + ", ".join(f":{k}" for k in cols) + ") RETURNING id"
            ), cols).scalar_one()
        made["cameras"].append(cam)
        return cam

    def frames(cam: int, start: float, times, motion=True, cells=ALL) -> list[int]:
        ids = []
        with engine.begin() as c:
            for t in times:
                ids.append(c.execute(text(
                    "INSERT INTO segments (camera_id, frame_index, timestamp_s, thumb_path, admitted, source_type, "
                    "admission_reason, motion_score, motion_cells, motion_grid) VALUES (:cam, 0, :ts, '/data/thumbs/908.jpg', "
                    "true, 'camera', :reason, 64, :cells, 8) RETURNING id"
                ), {"cam": cam, "ts": start + t, "reason": "motion" if motion else "heartbeat", "cells": cells}).scalar_one())
        made["segments"] += ids
        return ids

    yield camera, frames
    with engine.begin() as c:
        if made["cameras"]:
            c.execute(text("DELETE FROM alerts WHERE camera_id = ANY(:ids)"), {"ids": made["cameras"]})
        if made["segments"]:
            c.execute(text("DELETE FROM segments WHERE id = ANY(:ids)"), {"ids": made["segments"]})
        if made["cameras"]:
            c.execute(text("DELETE FROM cameras WHERE id = ANY(:ids)"), {"ids": made["cameras"]})


def _detector(**kw) -> Detector:
    return Detector(engine, **{"lookback_s": 60.0, "max_gap_s": 2.0, "tz": TZ, **kw})


def _alerts(cam: int) -> list:
    with engine.connect() as c:
        return c.execute(text(
            "SELECT a.event_type, a.severity, a.source, a.status, a.segment_id, a.reasoning, a.video_id, s.timestamp_s "
            "FROM alerts a JOIN segments s ON s.id = a.segment_id WHERE a.camera_id = :cam ORDER BY a.id"
        ), {"cam": cam}).all()


def test_sustained_motion_raises_one_alert_with_its_frame_and_reason(world):
    camera, frames = world
    cam = camera()
    ids = frames(cam, NOON, range(8))  # motion for 7 s, N = 5

    raised = _detector().look(now=NOON + 8)

    got = _alerts(cam)
    assert len(raised) == 1 and len(got) == 1
    a = got[0]
    assert (a.event_type, a.severity, a.source, a.status, a.video_id) == (rules.SUSTAINED_MOTION, "medium", "rule", "new", None)
    assert a.segment_id == ids[5] and a.timestamp_s == NOON + 5
    assert "for 5 s in a row" in a.reasoning


def test_the_same_run_never_raises_twice_even_after_a_restart_or_as_it_goes_on(world):
    camera, frames = world
    cam = camera()
    frames(cam, NOON, range(8))
    _detector().look(now=NOON + 8)
    frames(cam, NOON, range(8, 30))  # the same run carries on

    assert _detector().look(now=NOON + 30) == []  # a new Detector: as after a restart
    assert len(_alerts(cam)) == 1


def test_a_run_longer_than_the_lookback_is_followed_back_to_its_start(world):
    camera, frames = world
    cam = camera()
    frames(cam, NOON, range(8))
    _detector(lookback_s=10.0).look(now=NOON + 8)
    frames(cam, NOON, range(8, 40))

    assert _detector(lookback_s=10.0).look(now=NOON + 40) == []  # the window alone starts mid-run
    assert len(_alerts(cam)) == 1


def test_a_new_run_after_a_pause_raises_again(world):
    camera, frames = world
    cam = camera()
    frames(cam, NOON, range(8))
    frames(cam, NOON, range(20, 28))  # 12 s of nothing between

    assert len(_detector().look(now=NOON + 28)) == 2


def test_motion_outside_the_zone_or_too_short_raises_nothing(world):
    camera, frames = world
    right_edge = camera(zone_left=0.9)
    frames(right_edge, NOON, range(10), cells=1)  # only the top-left cell moves
    short = camera()
    frames(short, NOON, range(4))
    heartbeats = camera()
    frames(heartbeats, NOON, range(20), motion=False)

    assert _detector().look(now=NOON + 20) == []


def test_activity_outside_hours_and_only_where_hours_are_set(world):
    camera, frames = world
    watched = camera(active_from=time(8, 0), active_until=time(20, 0), sustained_motion_s=600)
    unwatched = camera(sustained_motion_s=600)
    frames(watched, NIGHT, [0, 1, 12, 13, 30])  # bursts with pauses: one stretch of activity
    frames(unwatched, NIGHT, [0, 1])

    _detector().look(now=NIGHT + 31)

    got = _alerts(watched)
    assert [(a.event_type, a.severity) for a in got] == [(rules.OUTSIDE_HOURS, "high")]
    assert "outside the camera's hours 08:00-20:00" in got[0].reasoning
    assert _alerts(unwatched) == []


def test_a_disabled_camera_is_not_watched(world):
    camera, frames = world
    cam = camera()
    frames(cam, NOON, range(8))
    with engine.begin() as c:
        c.execute(text("UPDATE cameras SET status = 'disabled' WHERE id = :id"), {"id": cam})

    assert _detector().look(now=NOON + 8) == []
