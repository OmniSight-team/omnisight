"""OMNI-X18: the forecaster's eight measures (`pfm_measures.py`, D-65) on hand-made windows with known answers."""

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
m = _load("pfm_measures", ROOT / "scripts" / "pfm_measures.py")


def _data(domain, windows):
    """`windows`: (clip, label, [(t, τ or None, score)]); returns the data and `p` with `p(Normal) = 1 - score`."""
    samples, scores = [], []
    for clip, label, points in windows:
        for t, tau, score in points:
            samples.append({"label": label, "tau_s": "" if tau is None else str(tau), "clip_key": clip,
                            "nominal_length_s": "30" if domain == "ucf_crime" else "", "t_s": str(t)})
            scores.append(score)
    s = np.array(scores)
    return train_mod.Data(None, samples, range(len(samples)), domain).for_labels("event"), np.c_[1 - s, s]


def test_the_ticket_example_a_rising_window_scores_one_and_a_flat_one_a_half():
    # 10% -> 30% -> 70% as the event nears: τ 25 and 21 are far, τ 10 is close
    rising, p = _data("ucf_crime", [("a", "Abuse", [(5, 25, 0.1), (9, 21, 0.3), (20, 10, 0.7)])])
    assert m.within_window(rising, p) == {"auc": 1.0, "windows": 1, "skipped": 0}
    flat, p = _data("ucf_crime", [("a", "Abuse", [(5, 25, 0.4), (9, 21, 0.4), (20, 10, 0.4)])])
    assert m.within_window(flat, p)["auc"] == 0.5


def test_within_window_uses_close_8_to_13_s_on_ucf_and_skips_a_window_without_both_parts():
    data, p = _data("ucf_crime", [
        ("a", "Abuse", [(5, 25, 0.2), (8, 22, 0.3), (18, 12, 0.5), (22, 8, 0.3), (28, 2, 0.0)]),  # close .5 .3, far .2 .3
        ("b", "Arson", [(0, 15, 0.9), (10, 5, 0.1)]),  # lead-up under 20 s: no far
        ("n", "Normal", [(0, None, 0.1), (29, None, 0.1)]),
    ])
    # τ 2 (score 0) is outside [8, 13): counting it would add two losing pairs
    assert m.within_window(data, p) == {"auc": pytest.approx(3.5 / 4), "windows": 1, "skipped": 1}
    assert m.within_window(data, p, close=m.HELDOUT_CLOSE) == {"auc": 0.0, "windows": 1, "skipped": 1}


def test_held_out_adds_the_0_to_8_s_within_window_auc_separately():
    data, p = _data("ucf_crime", [("a", "Abuse", [(5, 25, 0.2), (20, 10, 0.5), (27, 3, 0.9)]),
                                  ("n", "Normal", [(0, None, 0.1), (29, None, 0.1)])])
    out = m.measures(data, p, theta=0.5, heldout=True)
    assert out["within_window_auc"] == 1.0 and out["within_window_auc_0_8"] == 1.0
    assert "within_window_auc_0_8" not in m.measures(data, p, theta=0.5)


def test_the_flat_line_check_counts_time_back_from_the_last_second_of_a_normal_window():
    data, p = _data("ucf_crime", [
        ("flat", "Normal", [(0, None, 0.1), (5, None, 0.1), (18, None, 0.1), (29, None, 0.1)]),  # all ties: 0.5
        ("climb", "Normal", [(29, None, 0.8), (0, None, 0.1), (18, None, 0.7), (5, None, 0.2)]),  # out of order: 1.0
        ("short", "Normal", [(0, None, 0.5), (9, None, 0.5)]),  # 9 s: no far part
        ("a", "Abuse", [(0, 25, 0.9), (17, 8, 0.1)]),
    ])
    # τ' = 29 - t: 29 and 24 far, 11 close (0 is in neither)
    assert m.flat_line(data, p) == {"auc": pytest.approx(0.75), "windows": 2, "skipped": 1}
    assert m.FLAT_LINE_MAX == 0.60


def test_theta_is_the_lowest_value_at_which_at_most_10_percent_of_normal_windows_warn():
    windows = [(f"n{i}", "Normal", [(0, None, 0.05), (1, None, s)]) for i, s in enumerate(np.arange(1, 11) / 10)]
    data, p = _data("ccd", windows + [("c", "Crash", [(0, 1.0, 0.95)])])
    theta = m.threshold(data, p)
    assert 0.9 < theta < 0.9 + 1e-9  # 1.0 warns (1 of 10), 0.9 does not
    assert m.at_threshold(data, p, theta)["false_alarm_windows"] == pytest.approx(0.1)


def test_lead_time_detection_and_false_alarms_at_theta():
    data, p = _data("ucf_crime", [
        ("a", "Abuse", [(2, 3.0, 0.9), (0, 5.0, 0.2), (1, 4.0, 0.7)]),  # out of order; first warning at τ 4
        ("b", "Arson", [(0, 6.0, 0.6), (1, 5.0, 0.9)]),  # first warning at τ 6
        ("c", "Fighting", [(0, 2.0, 0.1)]),  # never warns
        ("n", "Normal", [(3, None, 0.8), (0, None, 0.9), (1, None, 0.1), (2, None, 0.8), (4, None, 0.2), (5, None, 0.9)]),
        ("q", "Normal", [(0, None, 0.1), (1, None, 0.2)]),
    ])
    out = m.at_threshold(data, p, 0.5)
    assert out["mtta_s"] == pytest.approx(5.0) and out["warned_windows"] == 2
    assert out["detection_rate"] == pytest.approx(2 / 3)
    # n in time order: warn, no, warn, warn, no, warn -> 3 alarms over 8 seconds of normal footage
    assert out["false_alarm_windows"] == 0.5 and out["false_alarms"] == 3
    assert out["normal_hours"] == pytest.approx(8 / 3600) and out["false_alarms_per_hour"] == pytest.approx(1350.0)


def test_ccd_alarms_per_hour_count_each_second_as_half_a_second_of_footage():
    data, p = _data("ccd", [("n", "Normal", [(0.0, None, 0.9), (0.5, None, 0.9)]), ("c", "Crash", [(0.0, 1.0, 0.9)])])
    out = m.at_threshold(data, p, 0.5)
    assert out["false_alarms"] == 1 and out["normal_hours"] == pytest.approx(1 / 3600)


def _ccd():
    return _data("ccd", [
        ("c1", "Crash", [(1.0, 2.5, 0.2), (0.5, 3.0, 0.9), (1.5, 2.0, 0.7)]),
        ("c2", "Crash", [(0.5, 3.0, 0.1), (1.5, 2.0, 0.3), (1.0, 2.5, 0.6)]),
        ("n1", "Normal", [(0.5, None, 0.7), (1.0, None, 0.2)]),
        ("n2", "Normal", [(0.5, None, 0.1), (1.0, None, 0.3)]),
    ])


def test_per_window_auc_and_ap_sweep_every_window_maximum():
    data, p = _ccd()
    assert m.per_window_auc(data, p) == pytest.approx(0.75)  # .9 .6 against .7 .3
    # lines .9 .7 .6 .3: recall .5 at precision 1, then 1 at 2/3
    assert m.average_precision(data, p) == pytest.approx(0.5 + 0.5 * 2 / 3)


def test_all_eight_in_one_call_and_a_model_with_classes_scores_the_same_as_a_merged_one():
    data, p = _ccd()
    merged = m.measures(data, p)
    split = np.c_[p[:, 0], p[:, 1] * 0.25, p[:, 1] * 0.75]
    data.names = ["Normal", "A", "B"]
    assert m.measures(data, split) == merged
    for key in ("within_window_auc", "flat_line", "mtta_s", "detection_rate", "false_alarm_windows",
                "false_alarms_per_hour", "ap", "per_window_auc", "pooled_auc", "theta"):
        assert key in merged, key
    assert merged["pooled_auc"] == pytest.approx(train_mod._roc_auc(data.y != 0, p[:, 1]))
    assert merged["within_window_skipped"] == 2  # CCD's close is τ under 1 s: none here
    assert m.measures(data, p, theta=0.65)["theta"] == 0.65  # held-out is given validation's θ_d


def test_kept_counts_the_windows_each_range_keeps():
    data, _ = _data("ucf_crime", [
        ("a", "Abuse", [(5, 25, 0.2), (18, 12, 0.5)]),  # both
        ("b", "Arson", [(0, 15, 0.9), (10, 10, 0.1)]),  # close only
        ("n", "Normal", [(0, None, 0.1), (18, None, 0.1), (29, None, 0.1)]),  # both
    ])
    out = m.kept(data)
    assert out["event"] == {"windows": 2, "close (8.0, 13.0)": 2, "far (20.0, 30.0)": 1, "both (close)": 1}
    assert out["normal"]["both (close)"] == 1
