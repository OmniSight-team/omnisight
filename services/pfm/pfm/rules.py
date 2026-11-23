"""Slice 9's two deliberately simple rules (OMNI-908, D-52), as plain functions over frames.

A camera stores its admitted frames (Algorithm 1). A frame counts as motion when the gate admitted
it for motion (`admission_reason = motion`), not as a heartbeat. A **run** is the frames that count,
in time order, with no gap longer than the rule's own pause between neighbours (a heartbeat is
itself a sign of a gap: it is admitted only after a second without motion).

- **Sustained motion in zone:** a run of motion frames that each moved at least one cell inside
  the camera's zone, with no pause over `max_gap_s` (2 s: "in a row"), lasting at least N seconds
  (last frame's time minus first's). It fires at the first frame that reaches N.
- **Activity outside hours:** motion frames whose local time is outside the camera's watched
  hours; the run ends after `quiet_s` (60 s) without motion. Real activity comes in bursts with
  pauses between them (the simulator's cam08: 1-1.5 s of motion, 7-15 s apart), and one person
  moving about at night is one alert, not one per burst. It fires at the run's first frame.

One alert per run: a run that already has its alert never raises another (the detector checks).
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

SUSTAINED_MOTION = "Sustained motion in zone"
OUTSIDE_HOURS = "Activity outside hours"
# D-52's fixed table: activity at a time nobody should be there is the more serious of the two.
SEVERITY = {SUSTAINED_MOTION: "medium", OUTSIDE_HOURS: "high"}


@dataclass(frozen=True)
class Zone:
    left: float = 0.0
    top: float = 0.0
    right: float = 1.0
    bottom: float = 1.0


@dataclass(frozen=True)
class Frame:
    segment_id: int
    timestamp_s: float  # Unix time, as a live camera's segments store it
    motion: bool  # admitted for motion, not as a heartbeat
    cells: int | None  # Algorithm 1's moved cells, bit r*G + c
    grid: int | None


@dataclass(frozen=True)
class Firing:
    rule: str
    run_start_s: float
    run_end_s: float  # the run's last frame so far
    segment_id: int  # the frame the alert points at
    reason: str


def moved_in_zone(cells: int | None, grid: int | None, zone: Zone) -> bool:
    """Whether any moved cell's centre lies inside the zone. An unscored frame has no cells."""
    if cells is None or not grid:
        return False
    mask = cells & ((1 << 64) - 1)  # stored as a signed BIGINT
    for i in range(grid * grid):
        if mask >> i & 1:
            row, col = divmod(i, grid)
            x, y = (col + 0.5) / grid, (row + 0.5) / grid
            if zone.left <= x <= zone.right and zone.top <= y <= zone.bottom:
                return True
    return False


def outside_hours(timestamp_s: float, start: time, end: time, tz: ZoneInfo) -> bool:
    """Whether a moment falls outside the watched hours `[start, end)`; `start` after `end`
    spans midnight."""
    now = datetime.fromtimestamp(timestamp_s, UTC).astimezone(tz).time()
    inside = start <= now < end if start < end else (now >= start or now < end)
    return not inside


def runs(frames: Iterable[Frame], counts: Callable[[Frame], bool], max_gap_s: float) -> list[list[Frame]]:
    """Frames that count, in time order, split wherever two neighbours are more than `max_gap_s` apart."""
    out: list[list[Frame]] = []
    current: list[Frame] = []
    for frame in sorted(frames, key=lambda f: (f.timestamp_s, f.segment_id)):
        if not counts(frame):
            continue
        if current and frame.timestamp_s - current[-1].timestamp_s > max_gap_s:
            out.append(current)
            current = []
        current.append(frame)
    if current:
        out.append(current)
    return out


def _clock(timestamp_s: float, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(timestamp_s, UTC).astimezone(tz).strftime("%H:%M:%S")


def sustained_motion(frames: Iterable[Frame], zone: Zone, min_s: float, max_gap_s: float, tz: ZoneInfo) -> list[Firing]:
    whole = zone == Zone()
    where = "in the frame" if whole else "in the zone"
    found = []
    for run in runs(frames, lambda f: f.motion and moved_in_zone(f.cells, f.grid, zone), max_gap_s):
        start = run[0].timestamp_s
        reached = next((f for f in run if f.timestamp_s - start >= min_s), None)
        if reached is not None:
            found.append(Firing(
                SUSTAINED_MOTION, start, run[-1].timestamp_s, reached.segment_id,
                f"motion {where} for {min_s:g} s in a row, from {_clock(start, tz)}",
            ))
    return found


def activity_outside_hours(frames: Iterable[Frame], start: time, end: time, quiet_s: float, tz: ZoneInfo) -> list[Firing]:
    found = []
    for run in runs(frames, lambda f: f.motion and outside_hours(f.timestamp_s, start, end, tz), quiet_s):
        first = run[0]
        found.append(Firing(
            OUTSIDE_HOURS, first.timestamp_s, run[-1].timestamp_s, first.segment_id,
            f"motion at {_clock(first.timestamp_s, tz)}, outside the camera's hours "
            f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}",
        ))
    return found
