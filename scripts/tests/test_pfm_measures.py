"""OMNI-917: the forecasting measures (`pfm_measures.py`, D-64) on hand-made windows with known answers."""

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
    return train_mod.Data(None, samples, range(len(samples)), domain), np.c_[1 - s, s]


def test_within_window_compares_close_with_far_inside_one_window_and_skips_a_window_without_far():
    data, p = _data("ucf_crime", [
        ("a", "Abuse", [(5, 25, 0.2), (8, 22, 0.3), (22, 8, 0.5), (28, 2, 0.3)]),  # close .5 .3, far .2 .3
        ("b", "Arson", [(0, 15, 0.9), (10, 5, 0.1)]),  # lead-up under 20 s: no far
        ("n", "Normal", [(0, None, 0.1), (5, None, 0.1), (25, None, 0.1), (29, None, 0.1)]),
    ])
    assert m.within_window(data, p) == {"auc": pytest.approx(3.5 / 4), "windows": 1, "skipped": 1}


def test_the_flat_line_check_uses_time_before_the_last_sample_of_a_normal_window():
    data, p = _data("ucf_crime", [
        ("flat", "Normal", [(0, None, 0.1), (5, None, 0.1), (25, None, 0.1), (29, None, 0.1)]),  # all ties: 0.5
        ("climb", "Normal", [(29, None, 0.8), (0, None, 0.1), (25, None, 0.7), (5, None, 0.2)]),  # out of order: 1.0
        ("short", "Normal", [(0, None, 0.5), (9, None, 0.5)]),  # 9 s: no far part
        ("a", "Abuse", [(0, 25, 0.9), (20, 5, 0.1)]),
    ])
    assert m.flat_line(data, p) == {"auc": pytest.approx(0.75), "windows": 2, "skipped": 1}


def _ccd():
    return _data("ccd", [
        ("c1", "Crash", [(1.0, 2.5, 0.2), (0.5, 3.0, 0.9), (1.5, 2.0, 0.7)]),
        ("c2", "Crash", [(0.5, 3.0, 0.1), (1.5, 2.0, 0.3), (1.0, 2.5, 0.6)]),
        ("n1", "Normal", [(0.5, None, 0.7), (1.0, None, 0.2)]),
        ("n2", "Normal", [(0.5, None, 0.1), (1.0, None, 0.3)]),
    ])


def test_per_window_auc_ap_and_mtta_sweep_every_window_score():
    data, p = _ccd()
    assert m.per_window_auc(data, p) == pytest.approx(0.75)  # .9 .6 against .7 .3
    # lines .9 .7 .6 .3: recall .5 at precision 1, then 1 at 2/3; TTA 3.0, 3.0, 2.75, 2.75
    assert m.ap_mtta(data, p) == {"ap": pytest.approx(0.5 + 0.5 * 2 / 3), "mtta_s": pytest.approx(2.875), "lines": 4}


def test_a_model_with_classes_and_a_merged_one_score_the_same_when_p_normal_is_the_same():
    data, p = _ccd()
    merged = m.measures(data, p)
    split = np.c_[p[:, 0], p[:, 1] * 0.25, p[:, 1] * 0.75]
    data.names = ["Normal", "A", "B"]
    assert m.measures(data, split) == merged
    assert merged["pooled_auc"] == pytest.approx(train_mod._roc_auc(data.y != 0, p[:, 1]))
    assert merged["within_window_skipped"] == 2  # CCD's close is τ under 1 s: none here
