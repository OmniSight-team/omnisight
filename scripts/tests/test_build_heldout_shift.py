"""OMNI-X23 (D-66): held-out UCF-Crime's event windows shifted 8 s earlier, their missing grid points, and the
store reading two folders together."""

import importlib.util
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


shift_mod = _load("build_heldout_shift", ROOT / "scripts" / "build_heldout_shift.py")
samples_mod = sys.modules["pfm_samples"]
runner = sys.modules["encode_pfm_windows"]


def _row(key, nominal, start, end, flags="", kind="positive", split="held_out"):
    return {"clip_key": key, "video_relpath": f"videos/{key}.mp4", "kind": kind, "nominal_length_s": nominal,
            "window_start_s": str(start), "window_end_s": str(end), "length_s": str(end - start),
            "event_class": "Arrest", "split": split, "fps": "30", "flags": flags}


def test_a_window_moves_8_s_before_its_event_by_d49s_rule():
    # event at 100 s: [70, 100) -> [62, 92); [40, 100) -> [32, 92)
    assert shift_mod.shift(_row("A", "30", 70, 100))[ "window_start_s"] == "62"
    s = shift_mod.shift(_row("A", "60", 40, 100))
    assert (s["window_start_s"], s["window_end_s"], s["length_s"], s["flags"]) == ("32", "92", "60", "shift_8s")
    # event at 50 s: the 60 s window starts at 0, shortened
    s = shift_mod.shift(_row("B", "60", 0, 50, "shortened"))
    assert (s["window_start_s"], s["window_end_s"], s["flags"]) == ("0", "42", "shortened;shift_8s")
    # event at 20 s: the 30 s window, fine before, is now shortened
    assert shift_mod.shift(_row("C", "30", 0, 20, "shortened"))["flags"] == "shortened;shift_8s"
    # an event 8 s or less into its video: no window
    assert shift_mod.shift(_row("D", "30", 0, 8.0, "shortened")) is None
    assert shift_mod.shift(_row("D", "30", 0, 8.5, "shortened"))["window_end_s"] == "0.5"
    with pytest.raises(ValueError):
        shift_mod.shift(_row("E", "30", 0, 20, "uca_margin"))  # a UCA time is already 8 s before
    with pytest.raises(ValueError):
        shift_mod.shift(_row("F", "30", 0, 20, kind="negative"))


def test_the_event_is_8_s_after_a_shifted_windows_end_and_close_and_far_both_fit():
    moved = {"A": [shift_mod.shift(_row("A", "30", 70, 100))]}
    samples = samples_mod.build(moved, "ucf_crime")
    taus = [float(s["tau_s"]) for s in samples]
    assert len(samples) == 30 and min(taus) == pytest.approx(8.0333, abs=1e-3) and max(taus) == pytest.approx(37.0333, abs=1e-3)
    assert {s["start_frame"] for s in samples} == {samples[0]["end_frame"]}  # each sequence from the window's start


def test_the_missing_points_are_the_60_s_windows_8_s_before_the_old_one_on_the_same_grid():
    old = {"A": [_row("A", "30", 70, 100), _row("A", "60", 40, 100)],   # event at 100 s: 8 s missing
           "B": [_row("B", "30", 20, 50), _row("B", "60", 0, 50, "shortened")]}  # event at 50 s: nothing missing
    moved, none = shift_mod.shifted(old)
    need = shift_mod.missing(old, moved, 1.0)

    assert not none and set(need) == {"A"}
    old_grid = runner.grid_frames(old["A"], 30.0, 1.0, context_s=0.0)
    assert need["A"] == list(range(old_grid[0] - 8 * 30, old_grid[0], 30))  # 8 points, 1 s apart, before the old
    new_grid = runner.grid_frames(moved["A"], 30.0, 1.0, context_s=0.0)
    assert set(new_grid) <= set(old_grid) | set(need["A"])


def test_the_store_reads_a_sequence_across_the_two_folders(tmp_path):
    old = {"A": [_row("A", "30", 70, 100), _row("A", "60", 40, 100)]}
    moved, _ = shift_mod.shifted(old)
    need = shift_mod.missing(old, moved, 1.0)
    for folder, frames in ((tmp_path / "old", runner.grid_frames(old["A"], 30.0, 1.0, context_s=0.0)),
                           (tmp_path / "extra", need["A"])):
        folder.mkdir()
        np.savez(runner.out_file(folder, "A"), frame_index=np.array(frames),
                 vectors=np.array([[f, 0] for f in frames], np.float32))
    samples = samples_mod.build(moved, "ucf_crime")
    with pytest.raises(ValueError, match="grid vectors"):
        samples_mod.Store(samples, tmp_path / "old")  # the 60 s window's first 8 s are not there
    store = samples_mod.Store(samples, tmp_path / "old", extra_dir=tmp_path / "extra")

    last_60 = max(i for i, s in enumerate(samples) if s["nominal_length_s"] == "60")
    x, mask = store.batch([last_60], max_length=60)
    frames = list(range(int(samples[last_60]["start_frame"]), int(samples[last_60]["end_frame"]) + 1, 30))
    assert int(mask.sum()) == 60 and x[0, :60, 0].tolist() == frames
    np.savez(runner.out_file(tmp_path / "extra", "A"), frame_index=np.array(need["A"][:1] + [frames[-1]]),
             vectors=np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError, match="share"):
        samples_mod.Store(samples, tmp_path / "old", extra_dir=tmp_path / "extra")


def test_the_committed_files_are_the_build():
    windows = runner.read_windows(ROOT / "datasets" / "windows" / "ucf_crime.csv")
    moved, none = shift_mod.shifted(windows)
    assert shift_mod.to_csv([r for rs in moved.values() for r in rs]) == shift_mod.OUT_WINDOWS.read_bytes()
    assert samples_mod.to_csv(samples_mod.build(moved, "ucf_crime")) == shift_mod.OUT_SAMPLES.read_bytes()
    assert len(none) == 74 and sum(map(len, moved.values())) == 206
