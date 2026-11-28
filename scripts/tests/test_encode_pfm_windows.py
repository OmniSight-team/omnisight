"""OMNI-902 / OMNI-X16: the PFM's grid runner, with a stand-in encoder (no model, no GPU)."""

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


# The archive tests' synthetic video (top-left pixel = frame index) and ai_workers' window functions.
helpers = _load("archive_test_helpers", ROOT / "scripts" / "tests" / "test_encode_virat_archive.py")
runner = _load("encode_pfm_windows", ROOT / "scripts" / "encode_pfm_windows.py")
window = helpers.clip_encoder
PINS = {"clip_model_id": "stand-in", "clip_model_revision": "0"}


class StandIn:
    def __init__(self, fail_on: int | None = None):
        self.clips, self.calls, self.fail_on = [], [], fail_on

    def encode_clips(self, clips):
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            self.calls.append(len(clips))
            raise RuntimeError("out of memory")
        self.calls.append(len(clips))
        self.clips += [[int(f[0, 0, 0]) for f in frames] for frames in clips]
        return np.stack([helpers._unit(np.stack(frames).tobytes(), 512) for frames in clips])


def _row(key, kind, start, end, fps=10):
    return {"clip_key": key, "video_relpath": f"videos/{key}.avi", "kind": kind, "window_start_s": str(start),
            "window_end_s": str(end), "fps": str(fps), "split": "train", "event_class": "Crash"}


@pytest.fixture
def ccd(tmp_path):
    """Three 5 s videos at 10 fps, as CCD's: two crashes and a normal clip."""
    (tmp_path / "videos" / "Crash").mkdir(parents=True)
    (tmp_path / "videos" / "Normal").mkdir(parents=True)
    windows = {
        "Crash/1": [_row("Crash/1", "positive", 0, 3.2)],
        "Crash/2": [_row("Crash/2", "positive", 0, 4.9)],
        "Normal/1": [_row("Normal/1", "negative", 1.956, 4.956)],
    }
    for key in windows:
        helpers._video(tmp_path / "videos" / f"{key}.avi", seconds=5.0, fps=10.0)
    return tmp_path, windows


def test_the_grid_steps_back_from_the_last_frame_before_the_end():
    # [0, 3.2): the last frame before the accident is 31 (3.1 s); every 5 frames back to 0.
    assert runner.grid_frames([_row("a", "positive", 0, 3.2)], 10.0, 0.5) == [1, 6, 11, 16, 21, 26, 31]
    # A normal window [1.956, 4.956): the whole footage before t (CCD), or only the window.
    normal = [_row("b", "negative", 1.956, 4.956)]
    assert runner.grid_frames(normal, 10.0, 0.5) == [4, 9, 14, 19, 24, 29, 34, 39, 44, 49]
    assert runner.grid_frames(normal, 10.0, 0.5, context_s=0.0) == [24, 29, 34, 39, 44, 49]
    with pytest.raises(ValueError, match="whole number"):
        runner.grid_frames(normal, 30.0, 0.55)


def test_a_clip_is_algorithm_2s_and_cut_at_the_videos_start():
    assert runner.clip_of(31, 10.0, window) == window.uniform_indices(21, 31)
    assert runner.clip_of(1, 10.0, window) == window.uniform_indices(0, 1)  # 0.1 s of video: padded
    assert all(max(runner.clip_of(f, 10.0, window)) <= f for f in range(50))


def test_every_video_gets_its_grid_clips_in_order_across_chunks(ccd, monkeypatch):
    data, windows = ccd
    monkeypatch.setattr(runner, "CHUNK_CLIPS", 10)  # chunks span videos; batches of BATCH inside
    monkeypatch.setattr(runner, "BATCH", 4)
    encoder = StandIn()

    manifest = runner.run(windows, data, data / "out", encoder, window, PINS, 0.5, session="s1")

    assert not manifest["failed"] and max(encoder.calls) <= 4
    expected = []
    for key, rows in windows.items():
        frames = runner.grid_frames(rows, 10.0, 0.5)
        with np.load(data / "out" / f"{key}.npz") as f:
            assert f["frame_index"].tolist() == frames
            assert np.allclose(f["t_s"], np.array(frames) / 10.0)
            assert f["vectors"].shape == (len(frames), 512) and float(f["fps"]) == 10.0
            assert float(f["gpu_seconds"]) >= 0
        expected += [runner.clip_of(fr, 10.0, window) for fr in frames]
    assert len(encoder.clips) == len(expected)
    for read, indices in zip(encoder.clips, expected, strict=True):
        assert np.allclose(read, indices, atol=4)  # MJPG moves a pixel value a little
    summary = runner.summary(windows, data / "out")
    assert (summary["done"], summary["clips"], summary["sessions"]) == (3, len(expected), ["s1"])


def test_only_encodes_the_given_grid_points_of_the_given_videos(ccd):
    # D-66: a second folder holds only the points another folder of the same grid lacks
    data, windows = ccd
    encoder = StandIn()
    only = {"Crash/1": [1, 6]}
    manifest = runner.run({"Crash/1": windows["Crash/1"]}, data, data / "extra", encoder, window, PINS, 0.5, only=only)

    assert not manifest["failed"]
    with np.load(data / "extra" / "Crash/1.npz") as f:
        assert f["frame_index"].tolist() == [1, 6] and f["vectors"].shape == (2, 512)
    assert len(encoder.clips) == 2
    assert np.allclose(encoder.clips[1], runner.clip_of(6, 10.0, window), atol=4)


def test_a_rerun_skips_done_videos_and_a_folder_holds_one_encoder(ccd):
    data, windows = ccd
    runner.run({"Crash/1": windows["Crash/1"]}, data, data / "out", StandIn(), window, PINS, 0.5)
    second = StandIn()

    runner.run(windows, data, data / "out", second, window, PINS, 0.5)

    assert len(second.clips) == len(runner.grid_frames(windows["Crash/2"], 10, 0.5)) + len(
        runner.grid_frames(windows["Normal/1"], 10, 0.5))
    with pytest.raises(RuntimeError, match="never mixed"):
        runner.run(windows, data, data / "out", StandIn(), window, {"clip_model_id": "other"}, 0.5)
    with pytest.raises(RuntimeError, match="never mixed"):
        runner.run(windows, data, data / "out", StandIn(), window, PINS, 1.0)


def test_a_bad_video_or_a_failed_batch_is_recorded_and_retried(ccd, monkeypatch):
    data, windows = ccd
    helpers._video(data / "videos" / "Crash" / "2.avi", seconds=2.0, fps=10.0)  # ends before its window
    monkeypatch.setattr(runner, "CHUNK_CLIPS", 1)  # one video per chunk

    manifest = runner.run(windows, data, data / "out", StandIn(fail_on=0), window, PINS, 0.5)

    assert manifest["failed"]["Crash/2"].startswith("read: ")
    assert manifest["failed"]["Crash/1"].startswith("encode: ")  # the first batch failed
    assert (data / "out" / "Normal" / "1.npz").is_file()
    assert json.loads((data / "out" / "manifest.json").read_text())["failed"].keys() == {"Crash/1", "Crash/2"}

    helpers._video(data / "videos" / "Crash" / "2.avi", seconds=5.0, fps=10.0)
    manifest = runner.run(windows, data, data / "out", StandIn(), window, PINS, 0.5)
    assert not manifest["failed"] and runner.summary(windows, data / "out")["done"] == 3


def test_the_ccd_windows_file_makes_a_grid_for_every_video():
    windows = runner.read_windows(ROOT / "datasets" / "windows" / "ccd.csv")
    assert len(windows) == 4500 and all(len(rows) == 1 for rows in windows.values())
    clips = sum(len(runner.grid_frames(rows, 10.0, runner.STEPS_S["ccd"])) for rows in windows.values())
    assert 30_000 < clips < 45_000, clips

def test_a_batch_that_mixes_frame_sizes_is_encoded_per_size_in_order():
    # CCD: a few videos have another resolution, and a batch spans videos (X16's 84 failed videos).
    cosmos = _load("encode_virat_cosmos", ROOT / "scripts" / "encode_virat_cosmos.py")
    small, large = np.zeros((4, 6, 3), np.uint8), np.zeros((8, 10, 3), np.uint8)
    clips = [[small + i] * 8 if i % 3 else [large + i] * 8 for i in range(7)]
    calls = []

    def encode(batch):  # one array per call, as Cosmos's processor needs
        calls.append(np.stack([np.stack(f) for f in batch]).shape)
        return np.array([[float(f[0][0, 0, 0])] for f in batch], np.float32)

    vectors = cosmos.by_frame_size(encode, clips)

    assert vectors[:, 0].tolist() == list(range(7)) and len(calls) == 2


def test_ucf_crimes_grid_is_one_second_inside_both_windows_only():
    # D-55: a sequence starts at its window's start, so UCF-Crime needs no footage before its windows.
    rows = [_row("v", "positive", 40.0, 70.0, fps=30), _row("v", "positive", 10.0, 70.0, fps=30)]
    frames = runner.grid_frames(rows, 30.0, runner.STEPS_S["ucf_crime"], context_s=0.0)
    assert frames[-1] == 2099 and frames[0] == 329 and len(frames) == 60  # 69.97 s back to 10.97 s
    assert set(np.diff(frames)) == {30}
    negatives = [_row("n", "negative", 5.0, 35.0, fps=30), _row("n", "negative", 100.0, 160.0, fps=30)]
    assert len(runner.grid_frames(negatives, 30.0, 1.0, context_s=0.0)) == 30 + 60  # two stretches, apart


def test_parallel_readers_give_the_same_files_as_one_reader(ccd, monkeypatch):
    data, windows = ccd
    runner.run(windows, data, data / "one", StandIn(), window, PINS, 0.5)
    monkeypatch.setattr(runner, "READERS", 1)
    monkeypatch.setattr(runner, "AHEAD", 1)
    runner.run(windows, data, data / "serial", StandIn(), window, PINS, 0.5)
    for key in windows:
        with np.load(data / "one" / f"{key}.npz") as a, np.load(data / "serial" / f"{key}.npz") as b:
            assert np.array_equal(a["vectors"], b["vectors"]) and np.array_equal(a["frame_index"], b["frame_index"])

