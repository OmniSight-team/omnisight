"""OMNI-906: the held-out measures (`pfm_heldout.py`) on hand-made predictions."""

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
held = _load("pfm_heldout", ROOT / "scripts" / "pfm_heldout.py")


def _ccd():
    """Two crash windows of 3 samples (τ 3.0, 2.5, 2.0) and two normal windows of 2, out of time order."""
    samples = []
    for key in ("c1", "c2"):
        for t, tau in ((1.5, 2.0), (0.5, 3.0), (1.0, 2.5)):
            samples.append({"label": "Crash", "tau_s": str(tau), "clip_key": key, "nominal_length_s": "", "t_s": str(t)})
    for key in ("n1", "n2"):
        for t in (0.5, 1.0):
            samples.append({"label": "Normal", "tau_s": "", "clip_key": key, "nominal_length_s": "", "t_s": str(t)})
    return train_mod.Data(None, samples, range(len(samples)), "ccd")


def test_lead_time_is_the_first_warning_in_time_order_and_a_normal_window_warns_once():
    data = _ccd()
    crash = [0.9, 0.2, 0.7,   0.4, 0.3, 0.45,   0.1, 0.6,   0.2, 0.3]  # c1 warns at t=1.0 first; c2 never
    p = np.array([[1 - c, c] for c in crash])
    rows = held.windows(data, p, theta=0.5)

    c1, c2, n1, n2 = rows
    assert (c1["lead_s"], c1["lead_correct_s"], c1["first_warning"], c1["ceiling_s"]) == (2.5, 2.5, "Crash", 3.0)
    assert c1["top3_last"] and c1["top1_last"] and c1["p_event_last"] == pytest.approx(0.9)  # last = t 1.5
    assert c2["lead_s"] is None and not c2["warned"] and not c2["top1_last"]
    assert n1["warned"] and not n2["warned"]

    s = held.summary(data, p, np.zeros(len(p)), rows)
    assert s["false_alarm_windows"] == 0.5 and s["warned_windows"] == 1 and s["unwarned_windows"] == 1
    assert s["lead_mean_s"] == 2.5 and s["lead_ceiling_s"] == 3.0 and s["top3_windows"] == 1.0
    assert s["top1_windows"] == 0.5


def test_ucf_top3_at_the_last_point_and_the_first_warning_of_the_true_class():
    samples = [{"label": "Abuse", "tau_s": str(tau), "clip_key": "a", "nominal_length_s": "30", "t_s": str(t)}
               for t, tau in ((0.0, 9.0), (1.0, 8.0))]
    data = train_mod.Data(None, samples, range(2), "ucf_crime")
    p = np.full((2, 14), 0.01)
    p[0, 2] = 0.8  # first: Arrest warns
    p[1, [1, 3, 4, 5]] = [0.3, 0.35, 0.4, 0.45]  # last: Abuse (1) is 4th, but warns as Burglary
    p /= p.sum(axis=1, keepdims=True)

    (row,) = held.windows(data, p, theta=0.1)
    assert (row["lead_s"], row["first_warning"], row["lead_correct_s"]) == (9.0, "Arrest", None)
    assert not row["top3_last"]


def test_the_control_sees_only_the_first_vector_and_the_timeline_bins_time_to_event():
    class Store:
        dim = 2

        def batch(self, index, max_length):
            x = np.ones((len(index), max_length, 2), np.float32)
            return x, np.ones((len(index), max_length), bool)

    x, mask = held.FirstVector(Store()).batch([0, 1], 4)
    assert mask.tolist() == [[True, False, False, False]] * 2 and x[:, 1:].sum() == 0 and x[:, 0].sum() == 4

    data = _ccd()
    p = np.array([[1 - c, c] for c in [0.9, 0.1, 0.5, 0.7, 0.3, 0.5, 0.2, 0.2, 0.4, 0.4]])
    tl = held.timeline(data, p, step=1.0)
    assert tl["tau_bin_start_s"] == [2.0, 3.0] and tl["positive_count"] == [4, 2]
    assert tl["positive_mean"] == pytest.approx([(0.9 + 0.5 + 0.7 + 0.5) / 4, 0.2])
    assert tl["negative_mean"] == pytest.approx(0.3)


def test_measure_splits_ucf_by_window_length():
    samples = [{"label": label, "tau_s": "5" if label != "Normal" else "", "clip_key": key, "nominal_length_s": length,
                "t_s": "1"} for key, label in (("a", "Abuse"), ("n", "Normal")) for length in ("30", "60")]
    data = train_mod.Data(None, samples, range(4), "ucf_crime")
    p = np.full((4, 14), 0.01)
    p[[0, 1], 1], p[[2, 3], 0] = 0.9, 0.9
    out = held.measure(data, p / p.sum(axis=1, keepdims=True), np.full(4, 5.0), theta=0.5, step=5.0)
    assert sorted(out["per_length"]) == ["30", "60"]
    assert out["per_length"]["30"]["positive_windows"] == 1 and out["per_length"]["30"]["lead_mean_s"] == 5.0
    assert out["by_class"]["Abuse"]["top3_last"] == 1.0 and out["summary"]["roc_auc"] == 1.0


def test_rise_is_p_event_near_the_event_minus_far_from_it():
    samples = [{"label": "Abuse", "tau_s": str(tau), "clip_key": "a", "nominal_length_s": "30", "t_s": str(30 - tau)}
               for tau in (2, 8, 15, 22, 28)]
    samples.append({"label": "Normal", "tau_s": "", "clip_key": "n", "nominal_length_s": "30", "t_s": "1"})
    data = train_mod.Data(None, samples, range(6), "ucf_crime")
    event = np.array([0.9, 0.7, 0.5, 0.3, 0.1, 0.99])
    p = np.zeros((6, 14))
    p[:, 0], p[:, 1] = 1 - event, event
    assert held.rise(data, p) == pytest.approx(0.8 - 0.2)  # the normal sample is not counted


def test_a_merged_model_keeps_the_crime_names_per_window_and_false_alarms_are_counted_per_hour():
    samples = [{"label": "Abuse", "tau_s": str(3 - t), "clip_key": "a", "nominal_length_s": "30", "t_s": str(t)}
               for t in (0, 1, 2)]
    samples += [{"label": "Normal", "tau_s": "", "clip_key": "n", "nominal_length_s": "30", "t_s": str(t)}
                for t in (3, 0, 1, 2, 4, 5)]  # out of time order
    data = train_mod.Data(None, samples, range(len(samples)), "ucf_crime").for_labels("event")
    event = np.array([0.2, 0.7, 0.9,   0.8, 0.9, 0.1, 0.8, 0.2, 0.9])  # n in time order: warn, no, warn, warn, no, warn
    p = np.stack([1 - event, event], axis=1)
    out = held.measure(data, p, np.zeros(len(p)), theta=0.5, step=5.0)
    assert out["windows"][0]["label"] == "Abuse" and out["windows"][0]["first_warning"] == "Event"
    assert out["windows"][0]["lead_s"] == 2.0 and sorted(out["by_class"]) == ["Abuse", "Normal"]
    s = out["summary"]
    assert s["false_alarms"] == 3 and s["normal_hours"] == pytest.approx(6 / 3600)
    assert s["false_alarms_per_hour"] == pytest.approx(1800.0)
