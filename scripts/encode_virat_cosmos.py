"""OMNI-X14: the VIRAT archive's clip stream again, with Cosmos-Embed1-336p, and SM-1 on it.

Run from `notebooks/cosmos_sm1_gpu.ipynb` on Colab's GPU. An experiment, not product code: the
product keeps InternVideo2 unless the rule in `docs/tickets/extra-cosmos-experiment.md` says switch.

**Only the encoder differs.** Nothing is re-gated: each video's kept frames, their timestamps and
clip windows `[t - 2Δ, t]` are read from the InternVideo2 archive (OMNI-708, `embeddings/virat/`),
the video is decoded again, and each kept frame's clip is 8 evenly spaced frames
(`clip_encoder.window_span`, `uniform_indices`), encoded by Cosmos-Embed1 in BF16 (its card), in
two spacings (`SPACINGS`):

- **A, `1s`:** the archive's own window `[t - 2Δ, t]`, the clip InternVideo2 got (Algorithm 2)
- **B, `4s`:** `[t - 4 s, t]`, cut at 0 s at a video's start: Cosmos is trained on 8 frames over 4-8 s

The SigLIP 2 frame vectors are copied from the archive unchanged, so `measure_sm1.load_archive`
reads each new archive as it reads the old one.

Output, one folder per spacing, one file per video, `{out}/{clip_key}.npz`: the archive's fields with `clip_vectors`
768-D Cosmos vectors. `{out}/manifest.json` is rewritten after every video with the pins and
each video's decode and GPU seconds; a rerun skips what is done, so a lost session resumes.

The fusion weights for Cosmos are chosen as D-40 chose InternVideo2's (`tune`, the tuning
queries only, `tune_fusion.grid`), then SM-1 is measured with them.

Locally (correctness only, a stand-in encoder): `uv run pytest scripts/tests/test_encode_virat_cosmos.py`.
"""

import json
import os
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

COSMOS_ID = "nvidia/Cosmos-Embed1-336p"
# The model card's commit, 2026-03-13 (OMNI-X14). Its code is loaded with trust_remote_code.
COSMOS_REVISION = "0e8a28f7bf370f2dcb3b9b61d23e167d0d6b0e6f"
COSMOS_DIM = 768
NUM_FRAMES = 8
BATCH = 16
# Seconds of video before `t` that a clip spreads its 8 frames over; None is the archive's own window.
SPACINGS = {"1s": None, "4s": 4.0}
MANIFEST = "manifest.json"
UNIT_TOLERANCE = 1e-2  # BF16 output, normalised again in float32 below


def _unit_rows(features) -> np.ndarray:
    import torch

    vectors = torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy()
    if vectors.ndim != 2 or vectors.shape[1] != COSMOS_DIM:
        raise ValueError(f"expected {COSMOS_DIM}-D embeddings, got shape {vectors.shape}")
    if not np.all(np.isfinite(vectors)) or np.any(np.abs(np.linalg.norm(vectors, axis=1) - 1.0) > UNIT_TOLERANCE):
        raise ValueError("embeddings are not finite unit vectors")
    return vectors.astype(np.float32)


def download(model_dir: Path) -> Path:
    """The pinned snapshot, kept on Drive so a new session does not download 2.4 GB again."""
    from huggingface_hub import snapshot_download

    snapshot_download(COSMOS_ID, revision=COSMOS_REVISION, local_dir=str(model_dir))
    return model_dir


class CosmosEncoder:
    """Cosmos-Embed1 as its model card runs it: `AutoModel` and `AutoProcessor` with
    trust_remote_code, BF16 on the GPU. The processor takes BTCHW uint8, resizes to 336 x 336
    and the model normalises (ImageNet mean and std) itself."""

    def __init__(self, model_dir: Path, device: str = "cuda", dtype=None) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.dtype = dtype or torch.bfloat16
        self.model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True).to(device, dtype=self.dtype).eval()
        self.processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        self.gpu_seconds = 0.0
        self.clips_encoded = 0

    def encode_clips(self, clips) -> np.ndarray:
        import torch

        for frames in clips:
            if len(frames) != NUM_FRAMES:
                raise ValueError(f"a clip is {NUM_FRAMES} frames, got {len(frames)}")
        batch = np.transpose(np.stack([np.stack(frames) for frames in clips]), (0, 1, 4, 2, 3))  # B, T, C, H, W
        start = time.perf_counter()
        with torch.inference_mode():
            inputs = self.processor(videos=batch).to(self.device, dtype=self.dtype)
            vectors = _unit_rows(self.model.get_video_embeddings(**inputs).visual_proj)
        self.gpu_seconds += time.perf_counter() - start
        self.clips_encoded += len(clips)
        return vectors

    def encode_texts(self, texts) -> np.ndarray:
        import torch

        with torch.inference_mode():
            inputs = self.processor(text=list(texts)).to(self.device, dtype=self.dtype)
            return _unit_rows(self.model.get_text_embeddings(**inputs).text_proj)


def clip_indices(frame_index: int, start_s: float, end_s: float, fps: float, window, span_s: float | None) -> list[int]:
    """One archived frame's 8 frame indices. `span_s` None: the archive's window, as OMNI-708 took it.
    Otherwise the `span_s` seconds up to `t` (`end_s`), from 0 s at the latest."""
    if span_s is not None:
        start_s = max(0.0, end_s - span_s)
    indices = window.uniform_indices(*window.window_span(start_s, end_s, fps))
    if max(indices) > frame_index:
        raise RuntimeError(f"frame {frame_index}'s clip reaches past it: {indices}")
    return indices


def encode_video(video: Path, archived: dict, encoder, window, span_s: float | None = None) -> np.ndarray:
    """The Cosmos vector of every archived frame's clip, in the archive's order. `window` is
    ai_workers' `clip_encoder` module (its window and sampling functions, as OMNI-708 used);
    `span_s` the spacing (`SPACINGS`)."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError("could not open file")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if abs(fps - float(archived["fps"])) > 1e-6:
        raise RuntimeError(f"fps {fps}, the archive says {float(archived['fps'])}")

    wanted = [
        (int(f), clip_indices(int(f), float(start_s), float(end_s), fps, window, span_s))
        for f, start_s, end_s in zip(archived["frame_index"], archived["clip_start_s"], archived["clip_end_s"], strict=True)
    ]
    reach = max((f - min(ix) for f, ix in wanted), default=0) + 3
    recent: deque[tuple[int, np.ndarray]] = deque(maxlen=reach)

    clips, parts, row, index = [], [], 0, -1
    try:
        while capture.grab():
            index += 1
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                raise RuntimeError(f"could not decode frame {index}")
            recent.append((index, frame))
            while row < len(wanted) and wanted[row][0] == index:
                held = dict(recent)
                missing = [i for i in wanted[row][1] if i not in held]
                if missing:
                    raise RuntimeError(f"frames {missing} of frame {index}'s window are no longer held")
                clips.append([cv2.cvtColor(held[i], cv2.COLOR_BGR2RGB) for i in wanted[row][1]])
                row += 1
                if len(clips) == BATCH:
                    parts.append(encoder.encode_clips(clips))
                    clips = []
        if clips:
            parts.append(encoder.encode_clips(clips))
    finally:
        capture.release()
    if index + 1 != int(archived["frames_decoded"]):
        raise RuntimeError(f"{index + 1} frames decoded, the archive decoded {int(archived['frames_decoded'])}")
    if row != len(wanted):
        raise RuntimeError(f"{len(wanted) - row} archived frames never reached")
    return np.concatenate(parts) if parts else np.zeros((0, COSMOS_DIM), np.float32)


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def read_manifest(out: Path) -> dict:
    path = out / MANIFEST
    return json.loads(path.read_text()) if path.is_file() else {}


def run(
    keys, video_dir: Path, source: Path, out: Path, encoder, window, pins: dict,
    *, span_s: float | None = None, session: str = "", log=print,
) -> dict:
    """Every video of `keys` not done yet, in order. `source` is the InternVideo2 archive;
    `pins` the new archive's (the frame pins of the source, Cosmos for the clip); `span_s` the
    spacing. One folder holds one spacing."""
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(out) or {"pins": pins, "span_s": span_s, "source": str(source), "videos": {}, "failed": {}}
    if manifest["pins"] != pins or manifest.get("span_s") != span_s:
        raise RuntimeError(f"{out} was made with {manifest['pins']}, span {manifest.get('span_s')}; an archive is never mixed")
    source_manifest = read_manifest(source)
    done = {k for k in manifest["videos"] if (out / f"{k}.npz").is_file()}
    todo = [k for k in keys if k not in done]
    log(f"{len(done)} of {len(keys)} done already, {len(todo)} to go")
    for n, key in enumerate(todo, 1):
        start, gpu_before, clips_before = time.perf_counter(), encoder.gpu_seconds, encoder.clips_encoded
        try:
            with np.load(source / f"{key}.npz") as data:
                archived = {name: data[name] for name in data.files}
            if len(archived["frame_index"]) != source_manifest["videos"][key]["admitted"]:
                raise RuntimeError("the source archive's file and manifest disagree")
            vectors = encode_video(video_dir / f"{key}.mp4", archived, encoder, window, span_s)
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden; a rerun tries it again
            manifest["failed"][key] = f"{type(exc).__name__}: {exc}"
            _write_json(out / MANIFEST, manifest)
            log(f"[{n}/{len(todo)}] {key}: FAILED {manifest['failed'][key]}")
            continue
        archived["clip_vectors"] = vectors
        tmp = out / f"{key}.tmp.npz"
        np.savez(tmp, **archived)
        os.replace(tmp, out / f"{key}.npz")
        manifest["failed"].pop(key, None)
        manifest["videos"][key] = {
            "admitted": len(vectors),
            "frames_decoded": int(archived["frames_decoded"]),
            "seconds": round(time.perf_counter() - start, 1),
            "gpu_seconds": round(encoder.gpu_seconds - gpu_before, 2),
            "clips": encoder.clips_encoded - clips_before,
            "session": session,
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        _write_json(out / MANIFEST, manifest)
        v = manifest["videos"][key]
        log(f"[{n}/{len(todo)}] {key}: {v['admitted']} clips, {v['seconds']:.0f} s, GPU {1000 * v['gpu_seconds'] / max(v['clips'], 1):.0f} ms per clip")
    return manifest


def timing(out: Path) -> dict:
    videos = read_manifest(out).get("videos", {})
    clips = sum(v["clips"] for v in videos.values())
    gpu = sum(v["gpu_seconds"] for v in videos.values())
    return {
        "videos": len(videos), "clips": clips, "gpu_seconds": round(gpu, 1),
        "ms_per_clip": round(1000 * gpu / clips, 1) if clips else None,
        "wall_seconds": round(sum(v["seconds"] for v in videos.values()), 1),
        "sessions": sorted({v["session"] for v in videos.values()}),
    }


class _Enough(Exception):
    pass


class _Collector:
    def __init__(self, limit: int) -> None:
        self.limit, self.clips = limit, []
        self.gpu_seconds, self.clips_encoded = 0.0, 0

    def encode_clips(self, clips) -> np.ndarray:
        self.clips += list(clips)
        if len(self.clips) >= self.limit:
            del self.clips[self.limit:]
            raise _Enough
        return np.zeros((len(clips), COSMOS_DIM), np.float32)


def sample_clips(video: Path, source: Path, key: str, window, span_s: float | None, limit: int = 64) -> list:
    """The first `limit` clips of one archived video, as the runner builds them: the timing input."""
    with np.load(source / f"{key}.npz") as data:
        archived = {name: data[name] for name in data.files}
    collector = _Collector(limit)
    try:
        encode_video(video, archived, collector, window, span_s)
    except _Enough:
        pass
    return collector.clips


def time_per_clip(encode_clips, clips, batch: int = BATCH) -> float:
    """Milliseconds per clip, encoding `clips` in batches of `batch` (preparation included, as
    the runner encodes), after one warm-up batch. Each call returns numpy, so the GPU has finished."""
    encode_clips(clips[:batch])
    start = time.perf_counter()
    for i in range(0, len(clips), batch):
        encode_clips(clips[i : i + batch])
    return 1000 * (time.perf_counter() - start) / len(clips)


def tune(dev_queries, archive, frame_text, clip_text, classifier, sm1, tune_fusion) -> dict:
    """D-40's choice made again for this clip encoder: `tune_fusion.grid` over list depth and
    `w_f` per kind, on the tuning queries only, the same order of preference."""
    depth = max(tune_fusion.DEPTHS)
    rows = [
        {
            "query_id": q["query_id"],
            "kind": q["kind"],
            "w_f": classifier.weights(f.tolist())[0],
            "frame": tune_fusion.ranked(archive.frame_vectors, f, depth),
            "clip": tune_fusion.ranked(archive.clip_vectors, c, depth),
            "correct": archive.correct[q["answer_rule"]],
            "near": archive.correct_near[q["answer_rule"]],
        }
        for q, f, c in zip(dev_queries, frame_text, clip_text, strict=True)
    ]
    grid = tune_fusion.grid(rows, sm1)
    best = next(r for r in grid if r["action_w_f"] is not None)
    return {
        "frame_only": tune_fusion.single(rows, "frame", sm1),
        "clip_only": tune_fusion.single(rows, "clip", sm1),
        "best": best,
        "grid": grid,
    }
