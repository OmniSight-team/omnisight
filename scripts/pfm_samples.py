"""OMNI-903: the PFM's training samples, from the windows and the grid (D-54, D-55).

A sample is one grid point `t` inside a window: the sequence of grid clip vectors from the window's
start to `t`, its domain, its class and, on a positive, the time to the event `tau = event_start - t`.

- **Positives:** every grid point inside a lead-up window, so the model sees every distance to the
  event (CCD: 0.1 s to 4.6 s)
- **Negatives:** every grid point inside a negative window; `tau` empty
- **The sequence starts at the window's start (D-55),** not at the video's: CCD's crash windows all
  start at 0 s and its normal windows later, so "all the footage before `t`" would make short
  sequences mostly Crash. From the window's start, both classes have the same lengths (D-24 drew the
  negatives' lengths from the positives'). On a crash window the two are the same
- A sample keeps its source video's split (D-22)
- **No sample at the event's own first frame:** a window's end is rounded to the millisecond (D-24),
  so when the event starts on a frame (UCF-Crime's official times are frame numbers), the rounded
  end can fall just after that frame and the grid's last point is the event's first frame
  (`tau` 0.0003 s). A positive grid point less than half a frame before the event is dropped;
  the window's other points are unchanged (32 UCF-Crime samples in 16 videos, none on CCD)

`datasets/samples/{dataset}.csv` is committed: one row per sample, frames only, no vectors. The
vectors are read from the encoder's folder (`encode_pfm_windows.py`) by `Store`, which keeps each
video's grid vectors once and pads a batch at the end, with a mask.

    uv run python scripts/pfm_samples.py build ccd          # writes datasets/samples/ccd.csv, prints counts
    uv run python scripts/pfm_samples.py check ccd <dir>    # every sample against an encoder's folder
"""

import csv
import io
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encode_pfm_windows import (
    STEPS_S,
    grid_frames,
    out_file,
    read_windows,
)

ROOT = Path(__file__).resolve().parents[1]
UCA_MARGIN_S = 8.0  # D-49: a UCA-timed window ends this long before the event
HEADER = ("sample_id", "domain", "clip_key", "split", "label", "nominal_length_s", "window_start_s",
          "window_end_s", "start_frame", "end_frame", "length", "t_s", "tau_s")


def event_start_s(row: dict) -> float:
    """When a positive window's event starts: the window's end (D-24, half-open), or 8 s after it on
    a UCA-timed UCF-Crime window (D-49)."""
    end = float(row["window_end_s"])
    return end + UCA_MARGIN_S if "uca_margin" in row["flags"].split(";") else end


def build(windows: dict[str, list[dict]], dataset: str) -> list[dict]:
    """Every sample of every window, in the windows' order, then by `t`."""
    step = STEPS_S[dataset]
    samples = []
    for key, rows in windows.items():
        for row in rows:
            fps = float(row["fps"])
            frames = grid_frames([row], fps, step, context_s=0.0)  # this window's grid points
            event = event_start_s(row) if row["kind"] == "positive" else None
            if event is not None:  # only the last point can be on the event's first frame
                frames = [f for f in frames if event - f / fps >= 0.5 / fps]
            for i, frame in enumerate(frames):
                t = frame / fps
                samples.append({
                    "sample_id": f"{key}|{row['nominal_length_s']}|{frame}", "domain": dataset,
                    "clip_key": key, "split": row["split"], "label": row["event_class"],
                    "nominal_length_s": row["nominal_length_s"], "window_start_s": row["window_start_s"],
                    "window_end_s": row["window_end_s"], "start_frame": frames[0], "end_frame": frame,
                    "length": i + 1, "t_s": round(t, 6), "tau_s": "" if event is None else round(event - t, 6),
                })
    return samples


def to_csv(samples: list[dict]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=HEADER, lineterminator="\n")
    writer.writeheader()
    writer.writerows(samples)
    return buf.getvalue().encode("utf-8")


def read_samples(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def counts(samples: list[dict]) -> dict:
    """Samples per split and class, the videos they come from, and the lengths per class."""
    return {
        "samples": len(samples),
        "per_split_class": dict(sorted(Counter(f"{s['split']} {s['label']}" for s in samples).items())),
        "videos_per_split_class": dict(sorted(Counter(
            f"{s['split']} {s['label']}" for s in {(s["clip_key"], s["split"], s["label"]): s for s in samples}.values()
        ).items())),
        "length_per_class": {c: dict(sorted(Counter(int(s["length"]) for s in samples if s["label"] == c).items()))
                             for c in sorted({s["label"] for s in samples})},
        "max_length": max(int(s["length"]) for s in samples),
    }


class Store:
    """The samples' vectors: each video's grid vectors read once from an encoder's folder, and per
    sample the rows of its sequence's frames. A window's grid points are evenly spaced, but not always
    consecutive rows: a UCF-Crime normal video's 30 s and 60 s windows sit at independent positions,
    so the video's grid interleaves their points."""

    def __init__(self, samples: list[dict], vectors_dir: Path) -> None:
        blocks, offsets, starts = [], {}, 0
        self.length = np.array([int(s["length"]) for s in samples], np.int64)
        self.pointer = np.concatenate([[0], np.cumsum(self.length)])
        self.rows = np.empty(int(self.pointer[-1]), np.int64)
        frame_rows: dict[str, dict[int, int]] = {}
        for i, s in enumerate(samples):
            key = s["clip_key"]
            if key not in offsets:
                with np.load(out_file(vectors_dir, key)) as f:
                    vectors, frames = f["vectors"], f["frame_index"].tolist()
                blocks.append(vectors.astype(np.float32))
                offsets[key], frame_rows[key] = starts, {fr: r for r, fr in enumerate(frames)}
                starts += len(vectors)
            first, last, n = int(s["start_frame"]), int(s["end_frame"]), int(self.length[i])
            step = (last - first) // (n - 1) if n > 1 else 1
            sequence = list(range(first, last + 1, step))
            rows = [frame_rows[key].get(fr) for fr in sequence]
            if len(sequence) != n or sequence[-1] != last or None in rows:
                raise ValueError(f"{s['sample_id']}: frames {first}-{last} are not {n} grid vectors in {key}.npz")
            self.rows[self.pointer[i] : self.pointer[i + 1]] = np.asarray(rows) + offsets[key]
        self.vectors = np.concatenate(blocks)
        self.dim = self.vectors.shape[1]

    def batch(self, index, max_length: int) -> tuple[np.ndarray, np.ndarray]:
        """`(x, mask)` for these samples: `x` (n, max_length, dim), the sequence first and zeros after
        it; `mask` True on the sequence."""
        index = np.asarray(index)
        x = np.zeros((len(index), max_length, self.dim), np.float32)
        mask = np.zeros((len(index), max_length), bool)
        for row, i in enumerate(index):
            n = int(self.length[i])
            if n > max_length:
                raise ValueError(f"sample {i} has {n} vectors, more than {max_length}")
            x[row, :n] = self.vectors[self.rows[self.pointer[i] : self.pointer[i + 1]]]
            mask[row, :n] = True
        return x, mask


def main() -> int:
    import json

    command, dataset = sys.argv[1], sys.argv[2]
    path = ROOT / "datasets" / "samples" / f"{dataset}.csv"
    if command == "build":
        samples = build(read_windows(ROOT / "datasets" / "windows" / f"{dataset}.csv"), dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(to_csv(samples))
    elif command == "check":
        samples = read_samples(path)
        store = Store(samples, Path(sys.argv[3]))
        norms = np.linalg.norm(store.vectors, axis=1)
        print(f"{len(samples)} samples, {len(store.vectors)} vectors of {store.dim}-D, norms {norms.min():.4f}-{norms.max():.4f}")
    else:
        raise SystemExit(f"unknown command {command}")
    print(json.dumps(counts(samples), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
