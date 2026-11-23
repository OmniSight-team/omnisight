"""OMNI-908: the two rules, on frames built by hand and on two short clips through the real gate.

The clips go through ingestion's own Algorithm 1 (`pipeline.py`, loaded by path), timestamped as a
live camera stamps them (Unix time), and only admitted frames are kept, as a live camera stores them.
"""

import importlib.util
from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from pfm import rules

ROOT = Path(__file__).resolve().parents[3]
TZ = ZoneInfo("Asia/Jerusalem")
ALL = (1 << 64) - 1  # every cell of an 8 x 8 grid
NOON = datetime(2026, 11, 23, 12, 0, tzinfo=TZ).timestamp()

_spec = importlib.util.spec_from_file_location("omnisight_ingestion_pipeline_pfm", ROOT / "services" / "ingestion" / "app" / "pipeline.py")
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)


def _cell(row: int, col: int, g: int = 8) -> int:
    return 1 << (row * g + col)


def _frames(times, cells=ALL, motion=True, start=NOON):
    return [rules.Frame(i + 1, start + t, motion, cells, 8) for i, t in enumerate(times)]


def test_a_moved_cell_counts_when_its_centre_is_inside_the_zone():
    left_half = rules.Zone(0.0, 0.0, 0.5, 1.0)
    assert rules.moved_in_zone(_cell(3, 0), 8, left_half)
    assert not rules.moved_in_zone(_cell(3, 7), 8, left_half)
    assert rules.moved_in_zone(_cell(7, 7) - (1 << 64), 8, rules.Zone(0.9, 0.9, 1.0, 1.0))  # signed BIGINT
    assert not rules.moved_in_zone(None, None, rules.Zone())


@pytest.mark.parametrize(
    ("clock", "start", "end", "outside"),
    [((12, 0), (8, 0), (20, 0), False), ((21, 0), (8, 0), (20, 0), True), ((7, 59), (8, 0), (20, 0), True),
     ((23, 0), (22, 0), (6, 0), False), ((3, 0), (22, 0), (6, 0), False), ((12, 0), (22, 0), (6, 0), True)],
)
def test_outside_hours_local_time_and_overnight_spans(clock, start, end, outside):
    t = datetime(2026, 11, 23, *clock, tzinfo=TZ).timestamp()
    assert rules.outside_hours(t, time(*start), time(*end), TZ) is outside


def test_runs_split_only_at_a_gap_longer_than_the_pause():
    # 0-2, a 3 s gap, 5-6, a heartbeat at 6.5 (does not count, no split on its own), 7, a 4 s gap, 11.
    frames = _frames([0, 1, 2, 5, 6]) + [rules.Frame(99, NOON + 6.5, False, ALL, 8)]
    frames += [rules.Frame(100, NOON + 7, True, ALL, 8), rules.Frame(101, NOON + 11, True, ALL, 8)]
    got = [[f.timestamp_s - NOON for f in run] for run in rules.runs(frames, lambda f: f.motion, 2.0)]
    assert got == [[0, 1, 2], [5, 6, 7], [11]]


def test_sustained_motion_fires_once_at_the_frame_that_reaches_n():
    frames = _frames([0, 1, 2, 3, 4, 5, 6])

    firings = rules.sustained_motion(frames, rules.Zone(), 4.0, 2.0, TZ)

    assert len(firings) == 1
    f = firings[0]
    assert (f.rule, f.segment_id, f.run_start_s - NOON, f.run_end_s - NOON) == (rules.SUSTAINED_MOTION, 5, 0, 6)
    assert "for 4 s in a row" in f.reason and "in the frame" in f.reason


def test_shorter_than_n_or_outside_the_zone_does_not_fire():
    assert rules.sustained_motion(_frames([0, 1, 2]), rules.Zone(), 4.0, 2.0, TZ) == []
    right_only = _frames([0, 1, 2, 3, 4, 5], cells=_cell(2, 7))
    assert rules.sustained_motion(right_only, rules.Zone(0.0, 0.0, 0.5, 1.0), 4.0, 2.0, TZ) == []


def test_heartbeats_alone_never_fire():
    assert rules.sustained_motion(_frames(range(20), motion=False), rules.Zone(), 4.0, 2.0, TZ) == []


def test_activity_outside_hours_is_one_alert_per_stretch_of_activity_until_a_quiet_minute():
    night = datetime(2026, 11, 23, 2, 0, tzinfo=TZ).timestamp()
    # Bursts 9-15 s apart (as on the simulator's cam08), then 90 s of quiet, then more.
    frames = _frames([0, 1, 10, 11, 25, 26, 116, 117], start=night)

    firings = rules.activity_outside_hours(frames, time(8, 0), time(20, 0), 60.0, TZ)

    assert [(f.segment_id, f.rule) for f in firings] == [(1, rules.OUTSIDE_HOURS), (7, rules.OUTSIDE_HOURS)]
    assert "02:00:00" in firings[0].reason and "08:00-20:00" in firings[0].reason
    assert rules.activity_outside_hours(_frames([0, 1, 2]), time(8, 0), time(20, 0), 60.0, TZ) == []


# The ticket's two clips: one that must raise an alert and one that must not.

def _clip(seconds: float, fps: float, moving: bool) -> list[np.ndarray]:
    """320 x 240; a block crossing the left half when `moving`, else a still scene."""
    out = []
    for i in range(int(seconds * fps)):
        frame = np.full((240, 320, 3), 50, np.uint8)
        frame[100:140, 260:300] = 120  # a parked object on the right, never moving
        if moving:
            x = 10 + int(i * 3) % 120
            frame[60:180, x:x + 40] = 230
        out.append(frame)
    return out


def _live_frames(clip: list[np.ndarray], fps: float, start: float) -> list[rules.Frame]:
    gate = pipeline.HeartbeatGate(
        pipeline.MotionGate(grid_size=8, cell_threshold=12.0, min_active_cells=2, stride_s=0.5, blur_kernel=5),
        heartbeat_interval_s=1.0, stride_s=0.5, start_s=start,
    )
    kept = []
    for i, frame in enumerate(clip):
        d = gate.evaluate(frame, start + i / fps)
        if d.admitted:
            kept.append(rules.Frame(i + 1, start + i / fps, d.admission_reason == "motion", d.motion_cells, d.motion_grid))
    return kept


def test_the_moving_clip_raises_sustained_motion_and_the_still_clip_does_not():
    start = datetime(2026, 11, 23, 12, 0, tzinfo=UTC).timestamp()
    moving = _live_frames(_clip(8.0, 15.0, moving=True), 15.0, start)
    still = _live_frames(_clip(8.0, 15.0, moving=False), 15.0, start)
    left_half = rules.Zone(0.0, 0.0, 0.5, 1.0)

    assert len(rules.sustained_motion(moving, left_half, 5.0, 2.0, TZ)) == 1
    assert rules.sustained_motion(still, left_half, 5.0, 2.0, TZ) == []
    assert still and not any(f.motion for f in still)  # the still clip keeps only heartbeats
    # The same motion, but the zone is the right edge where nothing moves: no alert.
    assert rules.sustained_motion(moving, rules.Zone(0.9, 0.0, 1.0, 1.0), 5.0, 2.0, TZ) == []
