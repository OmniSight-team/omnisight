"""OMNI-X24 (D-67): the rise rule on hand-made windows with known answers."""

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


train_mod = _load("train_pfm", ROOT / "scripts" / "train_pfm.py")
rise_mod = _load("pfm_rise", ROOT / "scripts" / "pfm_rise.py")


def _data(windows):
    """`windows`: (clip, label, length, [score per second from t = 0]); an event window ends 8 s before its event,
    so its last second has τ = 8."""
    samples, scores = [], []
    for clip, label, length, points in windows:
        n = len(points)
        for t, score in enumerate(points):
            tau = "" if label == "Normal" else str(8 + n - 1 - t)
            samples.append({"label": label, "tau_s": tau, "clip_key": clip, "nominal_length_s": length, "t_s": str(t)})
            scores.append(score)
    s = np.array(scores, float)
    return train_mod.Data(None, samples, range(len(samples)), "ucf_crime").for_labels("event"), np.c_[1 - s, s]


JUMP = [0.2] * 20 + [0.9] * 10  # 30 s, the score jumps at t = 20


def test_the_rise_is_the_score_over_the_mean_of_20_to_5_s_before_and_needs_10_s_of_it():
    data, p = _data([("a", "Abuse", "30", JUMP)])
    r = rise_mod.rise(data, p)
    # t = 0..14: the baseline [t - 20, t - 5) holds at most 9 points: no rise
    assert np.isnan(r[:15]).all() and not np.isnan(r[15:]).any()
    assert r[15] == pytest.approx(0.0)  # baseline t 0..9, all 0.2; score 0.2
    assert r[20] == pytest.approx(0.7)  # the jump: 0.9 over t 0..14, all 0.2
    assert r[26] == pytest.approx(0.9 - np.mean([0.2] * 14 + [0.9]))  # t 6..20: one 0.9 in the baseline
    assert r[29] == pytest.approx(0.9 - np.mean([0.2] * 11 + [0.9] * 4))  # t 9..23


def test_a_flat_window_never_warns_a_jump_warns_at_the_jump_and_too_little_history_never_does():
    data, p = _data([("flat", "Normal", "30", [0.6] * 30),
                     ("jump", "Abuse", "30", JUMP),
                     ("short", "Arson", "30", [0.0] * 5 + [1.0] * 9)])  # 14 s: never 10 s of baseline
    r = rise_mod.rise(data, p)
    assert np.isnan(r[data.window == "short|30"]).all()
    m = rise_mod._at(data, r, 0.5)
    assert (m["false_alarm_windows"], m["warned_windows"], m["detection_rate"]) == (0.0, 1, 0.5)
    assert m["mtta_s"] == 17.0  # the first warning is the jump, t = 20: τ = 8 + 29 - 20
    # the level line at 0.5 warns on the flat normal window at once (its score is 0.6)
    assert rise_mod._at(data, 1 - p[:, 0], 0.5)["false_alarm_windows"] == 1.0


def test_delta_lets_at_most_the_share_of_normal_windows_warn_and_a_flat_line_sets_it_above_zero():
    normals = [(f"n{i}", "Normal", "30", [0.3] * 20 + [0.3 + 0.01 * i] * 10) for i in range(20)]  # max rise 0.01 i
    data, p = _data(normals + [("a", "Abuse", "30", JUMP)])
    r = rise_mod.rise(data, p)
    d = rise_mod.line(data, r, 0.05)  # 1 of 20 may warn: just above the second highest, 0.18
    assert d == pytest.approx(0.18) and d > 0.18
    assert rise_mod._at(data, r, d)["false_alarm_windows"] == pytest.approx(0.05)
    assert rise_mod._at(data, r, rise_mod.line(data, r, 0.10))["false_alarm_windows"] == pytest.approx(0.10)
    flat, p = _data([("n", "Normal", "30", [0.6] * 30)])
    assert rise_mod.line(flat, rise_mod.rise(flat, p), 0.0) > 0.0  # a flat window does not warn even at F = 0


def test_the_report_fits_on_one_split_and_measures_another_with_the_same_lines():
    val, pv = _data([(f"n{i}", "Normal", "30", [0.3] * 20 + [0.3 + 0.01 * i] * 10) for i in range(20)]
                    + [("a", "Abuse", "30", JUMP)])
    lines = rise_mod.fit(val, pv)
    assert set(lines) == {"0.02", "0.05", "0.1"} and lines["0.05"]["delta"] == pytest.approx(0.18)
    held, ph = _data([("h", "Abuse", "30", [0.5] * 20 + [0.6] * 10),  # high but rises only 0.1
                      ("j", "Abuse", "60", [0.0] * 50 + [0.5] * 10),   # low but jumps 0.5
                      ("m", "Normal", "60", [0.55] * 60)])
    out = rise_mod.report(held, ph, lines)
    at5 = out["levels"]["0.05"]
    assert at5["rise"]["warned_windows"] == 1 and at5["rise"]["false_alarm_windows"] == 0.0
    assert at5["level"]["line"] == lines["0.05"]["theta"]
    assert set(out["per_length"]) == {"30", "60"}
    assert out["per_length"]["60"]["levels"]["0.05"]["rise"]["warned_windows"] == 1
    assert out["rise_per_window_auc"] == 1.0  # largest rises: j 0.5 and h 0.1, both above m 0.0
    w = rise_mod.wins(out)
    assert w["rise_warned"] == 1 and w["rise_wins"] == (1 > w["level_warned"])
