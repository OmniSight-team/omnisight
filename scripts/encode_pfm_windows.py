"""OMNI-902 / OMNI-X16: clip vectors for the forecasting model's windows, on a fixed time grid.

Run from `notebooks/pfm_windows_gpu.ipynb` on Colab's GPU. The PFM (Algorithm 4) reads a sequence of
clip vectors ending at a time `t`; this makes them for every window of `datasets/windows/{dataset}.csv`
(positive and negative, every split), once per clip encoder.

**The grid.** Per window, a grid point every `step_s` seconds, anchored at the window's last frame
(the last frame before its end, which is the event's start on a positive) and going back to the
window's start minus `context_s` (None: to the video's first frame). CCD: step 0.5 s, the whole
footage before `t` (D-23: under 5 s). A video's grid is the union of its windows' grids, in frames,
so the step must be a whole number of frames.

**The clip** at a grid frame `f` is Algorithm 2's: 8 evenly spaced frames from the 1 s before
`t = f / fps` (`[t - 2Δ, t]`, Δ = 0.5 s), through ai_workers' own `window_span` and
`uniform_indices`, cut at 0 s at a video's start (a short span pads with its last frame, as there).

**Any encoder** with `encode_clips(clips) -> (n, dim)` unit vectors: InternVideo2 in the notebook's
process, Cosmos-Embed1 in its own process on transformers 4.51.3 (D-50), through this file's command
line (`cosmos`).

Output, one folder per encoder and dataset: `{out}/{clip_key}.npz` per video (`frame_index`, `t_s`,
`vectors`, `fps`, `frames_read`, `gpu_seconds`), written atomically, so a file that exists is
done and a rerun skips it. Videos are encoded in chunks, so batches are full although a CCD video
has only about 8 grid points. `{out}/manifest.json` (pins, grid, failures, totals) is rewritten
after every chunk.

Locally (correctness only): `uv run pytest scripts/tests/test_encode_pfm_windows.py`.
"""

import csv
import json
import math
import os
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

import numpy as np

CLIP_SPAN_S = 1.0  # Algorithm 2's clip, [t - 2Δ, t] with Δ = 0.5 s
STEPS_S = {"ccd": 0.5, "ucf_crime": 1.0}
BATCH = 16
CHUNK_CLIPS = 128  # clips decoded before they are encoded, across videos
READERS = 8  # videos decoded at once
AHEAD = 16  # videos decoded ahead of the encoder at most (UCF-Crime: about 100 MB of frames each)
MANIFEST = "manifest.json"


def read_windows(path: Path) -> dict[str, list[dict]]:
    """The windows of one dataset's CSV, grouped by video (`clip_key`), in file order."""
    videos: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            videos[row["clip_key"]].append(row)
    return dict(videos)


def _step_frames(step_s: float, fps: float) -> int:
    frames = step_s * fps
    if abs(frames - round(frames)) > 1e-6 or round(frames) < 1:
        raise ValueError(f"a {step_s} s step is {frames} frames at {fps} fps: not a whole number")
    return round(frames)


def grid_frames(windows: list[dict], fps: float, step_s: float, context_s: float | None = None) -> list[int]:
    """The grid frames of one video, ascending: per window, every `step_s` back from its last frame
    to its start minus `context_s` (None: to frame 0)."""
    step = _step_frames(step_s, fps)
    frames: set[int] = set()
    for w in windows:
        start_s, end_s = float(w["window_start_s"]), float(w["window_end_s"])
        last = math.ceil(end_s * fps - 1e-6) - 1  # the last frame whose time is before the end
        first = 0 if context_s is None else max(0, math.ceil((start_s - context_s) * fps - 1e-6))
        if last < first:
            raise ValueError(f"{w['clip_key']}: window [{start_s}, {end_s}) holds no frame")
        frames.update(range(last, first - 1, -step))
    return sorted(frames)


def clip_of(frame: int, fps: float, window) -> list[int]:
    """The 8 frame indices of the clip ending at `frame` (`window` is ai_workers' clip_encoder)."""
    t = frame / fps
    indices = window.uniform_indices(*window.window_span(max(0.0, t - CLIP_SPAN_S), t, fps))
    if max(indices) > frame or min(indices) < 0:
        raise RuntimeError(f"frame {frame}'s clip is {indices}")
    return indices


def read_clips(video: Path, grid: list[int], window, fps_expected: float) -> tuple[list[list[np.ndarray]], float, int]:
    """Every grid frame's clip from one video (RGB), its fps and the number of frames read (decoding stops after the last needed one).
    Only the frames some clip needs are kept."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError("could not open file")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if abs(fps - fps_expected) > 1e-3:
        raise RuntimeError(f"fps {fps}, the windows say {fps_expected}")
    wanted = {f: clip_of(f, fps, window) for f in grid}
    needed = {i for ix in wanted.values() for i in ix}
    held: dict[int, np.ndarray] = {}
    index = -1
    try:
        while capture.grab():
            index += 1
            if index in needed:
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    raise RuntimeError(f"could not decode frame {index}")
                held[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if index >= max(needed):
                break
    finally:
        capture.release()
    missing = sorted(needed - held.keys())
    if missing:
        raise RuntimeError(f"the video ends before frames {missing[:5]} (last decoded {index})")
    return [[held[i] for i in wanted[f]] for f in grid], fps, index + 1


class Timed:
    """Wraps an encoder to count the seconds spent in `encode_clips` (each call returns numpy, so
    the GPU has finished) and the clips encoded."""

    def __init__(self, encoder) -> None:
        self.encoder, self.seconds, self.clips = encoder, 0.0, 0

    def encode_clips(self, clips) -> np.ndarray:
        start = time.perf_counter()
        vectors = self.encoder.encode_clips(clips)
        self.seconds += time.perf_counter() - start
        self.clips += len(clips)
        return vectors


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def read_manifest(out: Path) -> dict:
    path = out / MANIFEST
    return json.loads(path.read_text()) if path.is_file() else {}


def out_file(out: Path, key: str) -> Path:
    return out / f"{key}.npz"


def run(
    windows: dict[str, list[dict]], data_dir: Path, out: Path, encoder, window, pins: dict, step_s: float,
    *, context_s: float | None = None, session: str = "", log=print,
) -> dict:
    """Every video of `windows` without a file in `out` yet, in order. `data_dir` holds the
    dataset's `video_relpath`s; `pins` name the encoder. One folder holds one encoder and grid."""
    out.mkdir(parents=True, exist_ok=True)
    grid = {"step_s": step_s, "context_s": context_s, "clip_span_s": CLIP_SPAN_S}
    manifest = read_manifest(out) or {"pins": pins, "grid": grid, "failed": {}, "sessions": []}
    if manifest["pins"] != pins or manifest["grid"] != grid:
        raise RuntimeError(f"{out} was made with {manifest['pins']}, {manifest['grid']}; a folder is never mixed")
    if session and session not in manifest["sessions"]:
        manifest["sessions"].append(session)
    todo = [k for k in windows if not out_file(out, k).is_file()]
    log(f"{len(windows) - len(todo)} of {len(windows)} videos done already, {len(todo)} to go")
    timed = Timed(encoder)
    started, done_now = time.perf_counter(), 0

    pending: list[tuple[str, list[int], float, int, int]] = []  # key, grid, fps, frames read, clips
    clips: list[list[np.ndarray]] = []

    def flush() -> None:
        nonlocal done_now
        if not pending:
            return
        before = timed.seconds
        try:
            vectors = np.concatenate([timed.encode_clips(clips[i : i + BATCH]) for i in range(0, len(clips), BATCH)])
        except Exception as exc:  # noqa: BLE001 - recorded per video; a rerun tries them again
            for key, *_ in pending:
                manifest["failed"][key] = f"encode: {type(exc).__name__}: {exc}"
            pending.clear()
            clips.clear()
            return
        per_clip = (timed.seconds - before) / len(clips)
        row = 0
        for key, frames, fps, decoded, n in pending:
            path = out_file(out, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.stem + ".tmp.npz")
            np.savez(tmp, frame_index=np.array(frames, np.int32), t_s=np.array(frames, np.float64) / fps,
                     vectors=vectors[row : row + n].astype(np.float32), fps=fps, frames_read=decoded,
                     gpu_seconds=per_clip * n)
            os.replace(tmp, path)
            manifest["failed"].pop(key, None)
            row += n
            done_now += 1
        pending.clear()
        clips.clear()
        manifest["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        _write_json(out / MANIFEST, manifest)
        rate = done_now / (time.perf_counter() - started)
        log(f"{done_now} of {len(todo)} videos, {timed.clips} clips, "
            f"{1000 * timed.seconds / max(timed.clips, 1):.0f} ms per clip in the encoder, "
            f"about {(len(todo) - done_now) / max(rate, 1e-9) / 60:.0f} min to go")

    def read(key: str):
        rows = windows[key]
        try:
            fps_expected = float(rows[0]["fps"])
            frames = grid_frames(rows, fps_expected, step_s, context_s)
            return key, frames, *read_clips(data_dir / rows[0]["video_relpath"], frames, window, fps_expected), None
        except Exception as exc:  # noqa: BLE001
            return key, None, None, None, None, exc

    # Videos are decoded by READERS threads, up to AHEAD videos before the encoder (OpenCV releases
    # the GIL), and taken in `todo`'s order, so the output is the same as one by one.
    # cv2 is imported here, once, before the threads: a first import racing in 8 threads can hand a
    # thread the half-loaded module ("module 'cv2' has no attribute 'VideoCapture'", OMNI-902's rerun).
    import cv2  # noqa: F401

    with ThreadPoolExecutor(READERS) as pool:
        upcoming = iter(todo)
        queue = deque(pool.submit(read, key) for key in islice(upcoming, AHEAD))
        while queue:
            key, frames, video_clips, fps, decoded, exc = queue.popleft().result()
            following = next(upcoming, None)
            if following is not None:
                queue.append(pool.submit(read, following))
            if exc is not None:
                manifest["failed"][key] = f"read: {type(exc).__name__}: {exc}"
                log(f"{key}: FAILED {manifest['failed'][key]}")
                continue
            pending.append((key, frames, fps, decoded, len(video_clips)))
            clips.extend(video_clips)
            if len(clips) >= CHUNK_CLIPS:
                flush()
    flush()
    _write_json(out / MANIFEST, manifest)
    return manifest


def summary(windows: dict[str, list[dict]], out: Path) -> dict:
    """What is in `out` for these videos: files, clips, encoder seconds, failures."""
    manifest = read_manifest(out)
    files = [out_file(out, k) for k in windows if out_file(out, k).is_file()]
    clips, seconds = 0, 0.0
    for path in files:
        with np.load(path) as data:
            clips += len(data["frame_index"])
            seconds += float(data["gpu_seconds"])
    return {
        "videos": len(windows), "done": len(files), "missing": len(windows) - len(files),
        "failed": manifest.get("failed", {}), "clips": clips, "encoder_seconds": round(seconds, 1),
        "ms_per_clip": round(1000 * seconds / clips, 1) if clips else None,
        "pins": manifest.get("pins"), "grid": manifest.get("grid"), "sessions": manifest.get("sessions", []),
    }


def main() -> int:
    """`cosmos`: the grid encoded with Cosmos-Embed1 in its own process (transformers 4.51.3 first
    on the path, D-50)."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["cosmos"])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(STEPS_S), required=True)
    parser.add_argument("--data", type=Path, required=True, help="the dataset's folder, holding its video_relpaths")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pins", required=True, help="the folder's pins, as JSON")
    parser.add_argument("--session", default="")
    args = parser.parse_args()

    import torch
    import transformers

    sys.path.insert(0, str(args.repo / "services" / "ai_workers"))
    sys.path.insert(0, str(args.repo / "scripts"))
    # ai_workers' window functions (torch, safetensors only)
    from app import clip_encoder as window
    from encode_virat_cosmos import CosmosEncoder

    print(f"transformers {transformers.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}", flush=True)
    encoder = CosmosEncoder(args.model_dir, device="cuda")
    windows = read_windows(args.repo / "datasets" / "windows" / f"{args.dataset}.csv")
    run(windows, args.data, args.out, encoder, window, json.loads(args.pins), STEPS_S[args.dataset],
        session=args.session, log=lambda m: print(m, flush=True))
    print(json.dumps({k: v for k, v in summary(windows, args.out).items() if k != "failed"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
