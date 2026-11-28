"""OMNI-X23 (D-66): held-out UCF-Crime's event windows placed as training's, 8 s before the event.

Train and validation windows end 8 s before their (UCA) event (D-49); held-out's end at the event. This rebuilds
every held-out UCF-Crime positive window of `datasets/windows/ucf_crime.csv` by D-49's rule, the event time being
the window's end there:
- 30 s: `[event − 38, event − 8)`, 60 s: `[event − 68, event − 8)`; too little footage: it starts at 0 and is
  flagged `shortened`; an event 8 s or less into its video: no window (listed)
- flagged `shift_8s`: the event is at `window_end_s + 8` (`pfm_samples.event_start_s`)
- the grid as OMNI-902's: a point every 1 s back from the window's last frame. 8 s is a whole number of grid steps
  at the videos' 30 fps, so every point is on the video's existing grid (checked)
- **missing points:** the shifted windows' grid points that the video's existing windows (OMNI-902's folder) lack:
  the 60 s windows' up to 8 s before the old 60 s window. Encoded on Colab into `internvideo2_shift8/`

Normal windows and CCD are not touched. From the repo root (no database, no video):

    uv run python scripts/build_heldout_shift.py     # writes the two CSVs below, prints counts and SHA-256

Writes `datasets/windows/ucf_crime_heldout_shift8.csv` (the shifted windows) and
`datasets/samples/ucf_crime_heldout_shift8.csv` (their samples, `pfm_samples.build`'s rules).
"""

import csv
import hashlib
import io
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pfm_samples
from encode_pfm_windows import STEPS_S, grid_frames, read_windows

ROOT = Path(__file__).resolve().parents[1]
SHIFT_S = pfm_samples.UCA_MARGIN_S
WINDOWS = ROOT / "datasets" / "windows" / "ucf_crime.csv"
OUT_WINDOWS = ROOT / "datasets" / "windows" / "ucf_crime_heldout_shift8.csv"
OUT_SAMPLES = ROOT / "datasets" / "samples" / "ucf_crime_heldout_shift8.csv"
HEADER = ("clip_key", "video_relpath", "kind", "nominal_length_s", "window_start_s", "window_end_s", "length_s",
          "event_class", "split", "fps", "flags")


def _num(x: float) -> str:  # as the windows file writes numbers
    return f"{x:.3f}".rstrip("0").rstrip(".")


def shift(row: dict, shift_s: float = SHIFT_S) -> dict | None:
    """The held-out positive window `row` rebuilt to end `shift_s` before its event; None if the event is
    `shift_s` or less into the video."""
    flags = [f for f in row["flags"].split(";") if f]
    if row["kind"] != "positive" or row["split"] != "held_out" or "uca_margin" in flags:
        raise ValueError(f"{row['clip_key']}: not a held-out positive window with an official time")
    end = round(float(row["window_end_s"]) - shift_s, 3)
    if end <= 0:
        return None
    length = float(row["nominal_length_s"])
    start = round(max(0.0, end - length), 3)
    out = dict(row)
    out.update({"window_start_s": _num(start), "window_end_s": _num(end), "length_s": _num(round(end - start, 3)),
                "flags": ";".join((["shortened"] if end < length else []) + [pfm_samples.SHIFT_FLAG])})
    return out


def shifted(windows: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], list[dict]]:
    """Every held-out UCF-Crime positive window shifted, by video; and the windows that get none."""
    out, none = {}, []
    for key, rows in windows.items():
        for row in rows:
            if row["split"] != "held_out" or row["kind"] != "positive":
                continue
            s = shift(row)
            if s is None:
                none.append(row)
            else:
                out.setdefault(key, []).append(s)
    return out, none


def missing(windows: dict[str, list[dict]], moved: dict[str, list[dict]], step_s: float) -> dict[str, list[int]]:
    """Per video, the shifted windows' grid frames its existing windows' grid lacks (those videos only). Raises if
    a shifted window's points are off the existing grid (the shift is not a whole number of steps)."""
    out = {}
    for key, rows in moved.items():
        fps = float(rows[0]["fps"])
        old = set(grid_frames(windows[key], fps, step_s, context_s=0.0))
        new = set(grid_frames(rows, fps, step_s, context_s=0.0))
        step = round(step_s * fps)
        anchor = min(old)
        if any((f - anchor) % step for f in new):
            raise ValueError(f"{key}: shifted grid points off the existing grid")
        if new - old:
            out[key] = sorted(new - old)
    return out


def to_csv(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=HEADER, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def main() -> int:
    windows = read_windows(WINDOWS)
    moved, none = shifted(windows)
    rows = [r for rs in moved.values() for r in rs]
    samples = pfm_samples.build(moved, "ucf_crime")
    need = missing(windows, moved, STEPS_S["ucf_crime"])
    OUT_WINDOWS.write_bytes(to_csv(rows))
    OUT_SAMPLES.write_bytes(pfm_samples.to_csv(samples))
    print(f"held-out positive windows {len(rows) + len(none)}: shifted {len(rows)}, none (event ≤ 8 s in) {len(none)}")
    print("  per length:", dict(sorted(Counter(r["nominal_length_s"] for r in rows).items())),
          "shortened:", dict(sorted(Counter(r["nominal_length_s"] for r in rows if "shortened" in r["flags"]).items())))
    print(f"  videos {len(moved)}, samples {len(samples)}, missing grid points {sum(map(len, need.values()))} "
          f"in {len(need)} videos")
    for path in (OUT_WINDOWS, OUT_SAMPLES):
        print(f"  {path.relative_to(ROOT)}  {hashlib.sha256(path.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
