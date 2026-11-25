"""OMNI-903: the PFM's samples from the windows and the grid, and the store that pads them."""

import csv
import importlib.util
import sys
from collections import Counter
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


samples_mod = _load("pfm_samples", ROOT / "scripts" / "pfm_samples.py")
runner = sys.modules["encode_pfm_windows"]


def _row(key, kind, start, end, label="Crash", split="train", nominal="", flags="", fps=10):
    return {"clip_key": key, "video_relpath": f"videos/{key}.mp4", "kind": kind, "nominal_length_s": nominal,
            "window_start_s": str(start), "window_end_s": str(end), "event_class": label, "split": split,
            "fps": str(fps), "flags": flags}


def test_a_positive_window_gives_a_sample_at_every_grid_point_with_its_time_to_the_event():
    samples = samples_mod.build({"Crash/1": [_row("Crash/1", "positive", 0, 3.2)]}, "ccd")

    assert [s["end_frame"] for s in samples] == [1, 6, 11, 16, 21, 26, 31]
    assert [s["length"] for s in samples] == [1, 2, 3, 4, 5, 6, 7]
    assert {s["start_frame"] for s in samples} == {1}
    assert [s["tau_s"] for s in samples] == pytest.approx([3.1, 2.6, 2.1, 1.6, 1.1, 0.6, 0.1])


def test_a_negative_sequence_starts_at_its_windows_start_and_has_no_tau():
    samples = samples_mod.build({"Normal/1": [_row("Normal/1", "negative", 1.956, 4.956, "Normal")]}, "ccd")

    assert [s["end_frame"] for s in samples] == [24, 29, 34, 39, 44, 49]  # not from frame 4, the video's grid
    assert [s["length"] for s in samples] == [1, 2, 3, 4, 5, 6]
    assert {s["tau_s"] for s in samples} == {""} and {s["label"] for s in samples} == {"Normal"}


def test_a_uca_timed_window_measures_tau_to_the_event_8_s_after_its_end():
    row = _row("Abuse001", "positive", 0, 10.0, "Abuse", nominal="30", flags="shortened;uca_margin", fps=30)
    samples = samples_mod.build({"Abuse001": [row]}, "ucf_crime")

    assert samples[-1]["t_s"] == pytest.approx(9.966667, abs=1e-6)
    assert samples[-1]["tau_s"] == pytest.approx(18.0 - samples[-1]["t_s"])


def test_the_store_reads_each_sequence_from_the_videos_grid_and_pads_it(tmp_path):
    windows = {"Crash/1": [_row("Crash/1", "positive", 0, 3.2)],
               "Normal/1": [_row("Normal/1", "negative", 1.956, 4.956, "Normal")]}
    for key, rows in windows.items():  # as the runner writes them: the video's whole grid, from frame 0
        frames = runner.grid_frames(rows, 10.0, 0.5)
        (tmp_path / key).parent.mkdir(parents=True, exist_ok=True)
        np.savez(runner.out_file(tmp_path, key), frame_index=np.array(frames), vectors=np.array(
            [[f, 0, 0, 0] for f in frames], np.float32))
    samples = samples_mod.build(windows, "ccd")
    store = samples_mod.Store(samples, tmp_path)

    x, mask = store.batch([2, len(samples) - 1], max_length=10)

    assert x.shape == (2, 10, 4) and mask.sum(axis=1).tolist() == [3, 6]
    assert x[0, :3, 0].tolist() == [1, 6, 11] and not x[0, 3:].any()
    assert x[1, :6, 0].tolist() == [24, 29, 34, 39, 44, 49]
    with pytest.raises(ValueError, match="more than"):
        store.batch([len(samples) - 1], max_length=5)
    samples[0]["start_frame"] = 2  # not a grid frame
    with pytest.raises(ValueError, match="consecutive"):
        samples_mod.Store(samples, tmp_path)


def test_the_committed_ccd_samples_are_the_build_and_lengths_do_not_tell_the_class():
    windows = runner.read_windows(ROOT / "datasets" / "windows" / "ccd.csv")
    built = samples_mod.build(windows, "ccd")
    assert (ROOT / "datasets" / "samples" / "ccd.csv").read_bytes() == samples_mod.to_csv(built)

    with open(ROOT / "datasets" / "splits" / "ccd.csv", newline="") as f:
        split = {r["clip_key"]: r["split"] for r in csv.DictReader(f)}
    assert all(s["split"] == split[s["clip_key"]] for s in built)  # D-22
    assert {s["clip_key"] for s in built} == set(windows)
    lengths = {c: Counter(s["length"] for s in built if s["label"] == c) for c in ("Crash", "Normal")}
    for n in range(1, 11):
        share = [lengths[c][n] / sum(lengths[c].values()) for c in ("Crash", "Normal")]
        assert abs(share[0] - share[1]) < 0.005, (n, share)
    assert all(0 < s["tau_s"] < 5 for s in built if s["label"] == "Crash")
