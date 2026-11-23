"""OMNI-X14: the Cosmos archive runner, with stand-in encoders (no model, no GPU).

A source archive is made by OMNI-708's runner on a synthetic video; the Cosmos runner must give
each archived frame the same 8 frames, keep the frame vectors and every other field, and resume.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The archive tests' gate, stand-ins and synthetic video, reused rather than copied.
helpers = _load("archive_test_helpers", ROOT / "scripts" / "tests" / "test_encode_virat_archive.py")
cosmos = _load("encode_virat_cosmos", ROOT / "scripts" / "encode_virat_cosmos.py")
archive, pipeline, SETTINGS, clip_encoder = helpers.archive, helpers.pipeline, helpers.SETTINGS, helpers.clip_encoder
PINS = {"frame": "test", "clip": "cosmos-test"}


class CosmosStandIn:
    def __init__(self):
        self.clips = []
        self.gpu_seconds = 0.0
        self.clips_encoded = 0

    def encode_clips(self, clips):
        for frames in clips:
            assert len(frames) == 8 and all(f.ndim == 3 and f.shape[2] == 3 for f in frames)
        self.clips += [[int(f[0, 0, 0]) for f in frames] for frames in clips]
        self.clips_encoded += len(clips)
        return np.stack([helpers._unit(np.stack(frames).tobytes(), cosmos.COSMOS_DIM) for frames in clips])


@pytest.fixture
def source(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    for key in ("V1", "V2"):
        helpers._video(videos / f"{key}.mp4", seconds=3.0)
    out = tmp_path / "iv2"
    iv2_clips = helpers.ClipStandIn()
    archive.run(["V1", "V2"], videos, out, pipeline, SETTINGS, {"frame": "test", "clip": "iv2"},
                helpers.FrameStandIn(), clip_encoder, iv2_clips, tmp_path / "s")
    return videos, out, iv2_clips


def test_every_archived_frame_gets_the_same_eight_frames_and_the_rest_is_kept(source, tmp_path):
    videos, iv2, iv2_clips = source
    encoder = CosmosStandIn()

    manifest = cosmos.run(["V1", "V2"], videos, iv2, tmp_path / "cosmos", encoder, clip_encoder, PINS, session="a")

    assert not manifest["failed"]
    assert encoder.clips == iv2_clips.clips  # the same frames, in the same order, video after video
    for key in ("V1", "V2"):
        with np.load(iv2 / f"{key}.npz") as old, np.load(tmp_path / "cosmos" / f"{key}.npz") as new:
            assert set(new.files) == set(old.files)
            for name in old.files:
                if name != "clip_vectors":
                    assert np.array_equal(old[name], new[name]), name
            assert new["clip_vectors"].shape == (len(old["frame_index"]), 768)
        assert manifest["videos"][key]["clips"] == manifest["videos"][key]["admitted"] > 0


def test_a_rerun_skips_finished_videos_and_mixing_pins_is_refused(source, tmp_path):
    videos, iv2, _ = source
    out = tmp_path / "cosmos"
    cosmos.run(["V1"], videos, iv2, out, CosmosStandIn(), clip_encoder, PINS, session="one")
    second = CosmosStandIn()

    cosmos.run(["V1", "V2"], videos, iv2, out, second, clip_encoder, PINS, session="two")

    manifest = json.loads((out / "manifest.json").read_text())
    assert [manifest["videos"][k]["session"] for k in ("V1", "V2")] == ["one", "two"]
    assert len(second.clips) == manifest["videos"]["V2"]["admitted"]
    assert cosmos.timing(out)["clips"] == sum(v["admitted"] for v in manifest["videos"].values())
    with pytest.raises(RuntimeError, match="never mixed"):
        cosmos.run(["V1"], videos, iv2, out, CosmosStandIn(), clip_encoder, {**PINS, "clip": "other"})


def test_a_video_that_does_not_match_its_archive_is_recorded_as_failed(source, tmp_path):
    videos, iv2, _ = source
    helpers._video(videos / "V2.mp4", seconds=2.0)  # shorter than when it was archived

    manifest = cosmos.run(["V1", "V2"], videos, iv2, tmp_path / "cosmos", CosmosStandIn(), clip_encoder, PINS)

    assert "V1" in manifest["videos"] and "V2" in manifest["failed"]


def test_spacing_b_spreads_eight_frames_over_the_four_seconds_up_to_each_frame(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    helpers._video(videos / "V1.mp4", seconds=7.0)
    iv2 = tmp_path / "iv2"
    archive.run(["V1"], videos, iv2, pipeline, SETTINGS, {"frame": "test", "clip": "iv2"},
                helpers.FrameStandIn(), clip_encoder, helpers.ClipStandIn(), tmp_path / "s")
    encoder = CosmosStandIn()

    cosmos.run(["V1"], videos, iv2, tmp_path / "4s", encoder, clip_encoder, PINS, span_s=4.0)

    with np.load(iv2 / "V1.npz") as data:
        fps, frames, ends = float(data["fps"]), data["frame_index"], data["clip_end_s"]
    assert len(encoder.clips) == len(frames)
    spans = []
    for read, frame_index, end_s in zip(encoder.clips, frames, ends, strict=True):
        first = max(0, int(np.ceil((float(end_s) - 4.0) * fps - 1e-6)))
        indices = clip_encoder.uniform_indices(first, int(frame_index))
        assert np.allclose(read, [i % 256 for i in indices], atol=4), (frame_index, read)
        spans.append((first, int(frame_index)))
    # Late frames reach back the full 4 s (120 frames at 30 fps), not to 0 and not only 1 s.
    late = [(a, b) for a, b in spans if b > 4 * fps]
    assert late and all(a > 0 and b - a >= 4 * fps - 1 for a, b in late)


def test_a_folder_holds_one_spacing(source, tmp_path):
    videos, iv2, _ = source
    cosmos.run(["V1"], videos, iv2, tmp_path / "out", CosmosStandIn(), clip_encoder, PINS)

    with pytest.raises(RuntimeError, match="never mixed"):
        cosmos.run(["V1"], videos, iv2, tmp_path / "out", CosmosStandIn(), clip_encoder, PINS, span_s=4.0)


def test_the_timing_sample_is_the_runners_first_clips(source, tmp_path):
    videos, iv2, _ = source
    encoder = CosmosStandIn()
    cosmos.run(["V1"], videos, iv2, tmp_path / "4s", encoder, clip_encoder, PINS, span_s=4.0)

    clips = cosmos.sample_clips(videos / "V1.mp4", iv2, "V1", clip_encoder, 4.0, limit=5)

    assert [[int(f[0, 0, 0]) for f in c] for c in clips] == encoder.clips[:5]
    assert cosmos.time_per_clip(lambda batch: np.zeros((len(batch), 768)), clips, batch=2) >= 0
