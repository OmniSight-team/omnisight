"""OMNI-904: the PFM's shapes, mask, κ and loss, and its overfit and resume proofs, on the CPU with
stand-in vectors."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


model_mod = _load("pfm_model", ROOT / "scripts" / "pfm_model.py")
train_mod = _load("train_pfm", ROOT / "scripts" / "train_pfm.py")
Config, PFM = model_mod.Config, model_mod.PFM


class FakeStore:
    """`Store`'s interface over given sequences."""

    def __init__(self, sequences: list[np.ndarray]) -> None:
        self.sequences, self.dim = sequences, sequences[0].shape[1]

    def batch(self, index, max_length):
        x = np.zeros((len(index), max_length, self.dim), np.float32)
        mask = np.zeros((len(index), max_length), bool)
        for row, i in enumerate(index):
            n = len(self.sequences[i])
            x[row, :n], mask[row, :n] = self.sequences[i], True
        return x, mask


def _ccd(n: int, dim: int = 16, seed: int = 0):
    """`n` CCD-like samples: windows of 4 samples, alternately Crash and Normal, random vectors."""
    rng = np.random.default_rng(seed)
    samples, sequences = [], []
    for i in range(n):
        crash = (i // 4) % 2 == 0
        k = i % 4 + 1
        samples.append({"label": "Crash" if crash else "Normal", "tau_s": str(2.0 - 0.5 * k) if crash else "",
                        "clip_key": f"v{i // 4}", "nominal_length_s": "", "t_s": str(0.5 * k), "split": "train"})
        sequences.append(rng.normal(size=(k, dim)).astype(np.float32))
    return train_mod.Data(FakeStore(sequences), samples, range(n), "ccd")


def _config(**kw):
    return Config(**{"input_dim": 16, "domains": ["ccd", "ucf_crime"], "max_length": 10, "width": 32,
                     "heads": 2, "ff": 64, "batch": 8, "epochs": 4, **kw})


def test_each_domain_head_gives_its_classes_and_a_time_ahead():
    torch.manual_seed(0)
    model = PFM(_config()).eval()
    x, mask = torch.randn(5, 10, 16), torch.ones(5, 10, dtype=torch.bool)

    logits, dt = model(x, mask, "ccd")
    assert logits.shape == (5, 2) and dt.shape == (5,) and (dt >= 0).all()
    assert model(x, mask, "ucf_crime")[0].shape == (5, 14)
    with pytest.raises(ValueError, match="at most"):
        model(torch.randn(1, 11, 16), torch.ones(1, 11, dtype=torch.bool), "ccd")


def test_padding_is_ignored_and_positions_count_back_from_t():
    torch.manual_seed(0)
    model = PFM(_config()).eval()
    seq = torch.randn(1, 3, 16)
    short = model(seq, torch.ones(1, 3, dtype=torch.bool), "ccd")[0]
    padded_x = torch.cat([seq, 100 * torch.randn(1, 7, 16)], dim=1)  # garbage after the sequence
    padded_mask = torch.tensor([[True] * 3 + [False] * 7])

    assert torch.allclose(short, model(padded_x, padded_mask, "ccd")[0], atol=1e-5)


def test_kappa_is_zero_on_a_uniform_prediction_and_one_on_a_certain_one():
    p = torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.9, 0.1]])
    k = model_mod.kappa(p)
    assert k[0] == pytest.approx(0.0, abs=1e-6) and k[1] == pytest.approx(1.0, abs=1e-6)
    assert k[2] == pytest.approx(1 - (-(0.9 * math.log(0.9) + 0.1 * math.log(0.1))) / math.log(2), abs=1e-6)
    assert model_mod.kappa(torch.full((1, 14), 1 / 14))[0] == pytest.approx(0.0, abs=1e-6)


def test_the_loss_is_ce_plus_alpha_mse_on_positives_weighted_per_domain():
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    dt = torch.tensor([1.0, 3.0, 50.0])
    y = torch.tensor([0, 1, 1])
    tau = torch.tensor([float("nan"), 1.0, 48.0])  # the negative's Δt (1.0) is not compared
    config = _config(alpha=0.5, lambdas={"ccd": 2.0})

    total, parts = model_mod.loss({"ccd": (logits, dt)}, {"ccd": (y, tau)}, config)

    ce = torch.nn.functional.cross_entropy(logits, y)
    assert float(total) == pytest.approx(2.0 * (float(ce) + 0.5 * (4.0 + 4.0) / 2), rel=1e-6)
    assert parts["ccd"]["mse"] == pytest.approx(4.0)
    no_positive = model_mod.domain_loss(logits[:1], dt[:1], y[:1], tau[:1], 0.5)[0]
    assert float(no_positive) == pytest.approx(float(torch.nn.functional.cross_entropy(logits[:1], y[:1])))


def test_overfit_proof_32_samples_drive_the_loss_near_zero(tmp_path):
    data = _ccd(32)
    config = _config(domains=["ccd"], epochs=150, batch=32, dropout=0.0, lr=3e-3)

    model, history = train_mod.train(config, {"ccd": data}, tmp_path, seed=1, log=lambda m: None)

    assert history[0]["loss"] > 0.5 and history[-1]["loss"] < 0.05, (history[0]["loss"], history[-1]["loss"])
    p, _ = train_mod.predict(model, data)
    assert (p.argmax(axis=1) == data.y).all()


def test_resume_proof_a_stopped_run_continues_with_the_same_loss_curve(tmp_path):
    data = _ccd(48)
    config = _config(domains=["ccd"], epochs=4)
    _, straight = train_mod.train(config, {"ccd": data}, tmp_path / "a", seed=3, log=lambda m: None)

    _, first = train_mod.train(config, {"ccd": data}, tmp_path / "b", seed=3, stop_after=2, log=lambda m: None)
    _, resumed = train_mod.train(config, {"ccd": data}, tmp_path / "b", seed=3, log=lambda m: None)

    assert len(first) == 2 and len(resumed) == 4
    assert [h["loss"] for h in resumed] == pytest.approx([h["loss"] for h in straight], rel=1e-5)
    with pytest.raises(RuntimeError, match="was made with"):
        train_mod.train(config, {"ccd": data}, tmp_path / "b", seed=4, log=lambda m: None)

    # A finished 2-epoch run extended to 4 makes the same epochs as the straight 4; other changes are refused.
    train_mod.train(_config(domains=["ccd"], epochs=2), {"ccd": data}, tmp_path / "c", seed=3, log=lambda m: None)
    _, extended = train_mod.train(config, {"ccd": data}, tmp_path / "c", seed=3, log=lambda m: None)
    assert [h["loss"] for h in extended] == pytest.approx([h["loss"] for h in straight], rel=1e-5)
    with pytest.raises(RuntimeError, match="was made with"):
        train_mod.train(_config(domains=["ccd"], epochs=4, lr=1e-3), {"ccd": data}, tmp_path / "c", seed=3, log=lambda m: None)


def test_the_threshold_allows_10_percent_of_normal_windows_and_lead_time_is_the_first_warning():
    # 10 normal windows (1 sample each) and 2 crash windows of 3 samples.
    samples = [{"label": "Normal", "tau_s": "", "clip_key": f"n{i}", "nominal_length_s": "", "t_s": "1"}
               for i in range(10)]
    samples += [{"label": "Crash", "tau_s": str(tau), "clip_key": key, "nominal_length_s": "", "t_s": str(t)}
                for key in ("c1", "c2") for t, tau in ((0.5, 3.0), (1.0, 2.5), (1.5, 2.0))]
    data = train_mod.Data(None, samples, range(len(samples)), "ccd")
    crash = [0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9]  # normals: only 0.55+ are "Crash" argmax
    crash += [0.3, 0.75, 0.95, 0.6, 0.65, 0.78]
    p = np.array([[1 - c, c] for c in crash])
    dt = np.array([0.0] * 10 + [3.0, 2.5, 1.0, 3.0, 2.5, 2.0])

    m = train_mod.evaluate(data, p, dt)

    assert m["theta"] == pytest.approx(0.8, abs=1e-6) and m["false_alarm_windows"] == pytest.approx(0.1)
    assert m["warned_windows"] == 1 and m["lead_time_s"] == pytest.approx(2.0)  # c1 at 0.95; c2 never above 0.8
    assert m["dt_mae_s"] == pytest.approx(1.0 / 6)
    assert m["roc_auc"] == pytest.approx(train_mod._roc_auc(data.y != 0, p[:, 1]))


def test_roc_auc_and_the_x16_rule():
    assert train_mod._roc_auc(np.array([True, True, False, False]), np.array([0.9, 0.8, 0.1, 0.2])) == 1.0
    assert train_mod._roc_auc(np.array([True, False]), np.array([0.5, 0.5])) == 0.5

    def runs(aucs, lead):
        return [{"roc_auc": a, "lead_time_s": lead, "top1": 0.9, "dt_mae_s": 0.5} for a in aucs]

    clear = train_mod.x16_rule({"internvideo2": runs([0.80, 0.81, 0.82], 2.0), "cosmos": runs([0.86, 0.87, 0.88], 2.1)})
    assert clear["winner"] == "cosmos" and clear["margin"] == pytest.approx(0.02)
    within = train_mod.x16_rule({"internvideo2": runs([0.80, 0.81, 0.86], 2.0), "cosmos": runs([0.84, 0.85, 0.86], 2.1)})
    assert within["winner"] == "internvideo2"  # 0.0267 more, spread 0.06
    earlier = train_mod.x16_rule({"internvideo2": runs([0.80, 0.80, 0.80], 2.0), "cosmos": runs([0.9, 0.9, 0.9], 1.9)})
    assert earlier["winner"] == "internvideo2"  # better AUC, lower lead time


def test_validation_keeps_the_best_epoch_and_a_resumed_run_chooses_the_same(tmp_path):
    data = _ccd(48)
    config = _config(domains=["ccd"], epochs=5)
    scores = iter([0.5, 0.9, 0.7, 0.9, 0.6] * 2)  # epoch 2 is the best; epoch 4's tie does not replace it

    def validate(model):
        return {"score": next(scores)}

    _, straight = train_mod.train(config, {"ccd": data}, tmp_path / "a", seed=0, validate=validate, log=lambda m: None)
    train_mod.train(config, {"ccd": data}, tmp_path / "b", seed=0, validate=validate, stop_after=3, log=lambda m: None)
    _, resumed = train_mod.train(config, {"ccd": data}, tmp_path / "b", seed=0, validate=validate, log=lambda m: None)

    assert [h["validation"]["score"] for h in straight] == [0.5, 0.9, 0.7, 0.9, 0.6]
    for folder in ("a", "b"):
        model, state = train_mod.load(tmp_path / folder / "best.pt")
        assert (state["epoch"], state["score"], state["seed"]) == (2, 0.9, 0)
        assert not model.training
    assert [h["loss"] for h in resumed] == pytest.approx([h["loss"] for h in straight], rel=1e-5)


def test_top3_counts_a_positive_sample_whose_class_is_among_the_three_most_likely():
    samples = [{"label": label, "tau_s": "1" if label != "Normal" else "", "clip_key": f"v{i}",
                "nominal_length_s": "30", "t_s": "1"} for i, label in enumerate(["Abuse", "Arrest", "Normal"])]
    data = train_mod.Data(None, samples, range(3), "ucf_crime")
    p = np.full((3, 14), 0.01)
    p[0, [0, 2, 3, 1]] = [0.4, 0.3, 0.2, 0.05]  # Abuse (1) is 4th: a miss
    p[1, [0, 2]] = [0.5, 0.4]  # Arrest (2) is 2nd: a hit
    p[2, 0] = 0.9

    m = train_mod.evaluate(data, p / p.sum(axis=1, keepdims=True), np.ones(3))
    assert m["top3_positive"] == pytest.approx(0.5)


def test_a_domain_can_weigh_its_time_term_on_its_own_scale():
    logits, y = torch.tensor([[0.0, 1.0]]), torch.tensor([1])
    dt, tau = torch.tensor([10.0]), torch.tensor([30.0])  # squared error 400
    config = _config(alphas={"ucf_crime": 0.005})
    ce = float(torch.nn.functional.cross_entropy(logits, y))

    total, _ = model_mod.loss({"ucf_crime": (logits, dt), "ccd": (logits, dt)}, {"ucf_crime": (y, tau), "ccd": (y, tau)}, config)
    assert float(total) == pytest.approx((ce + 0.005 * 400) + (ce + 1.0 * 400), rel=1e-6)  # CCD keeps alpha
    assert Config(**config.to_dict()).alpha_of("ucf_crime") == 0.005


def test_minus_first_ignores_what_the_whole_window_shares_and_old_configs_stay_raw():
    torch.manual_seed(0)
    model = PFM(_config(input="minus_first")).eval()
    x, mask = torch.randn(2, 5, 16), torch.tensor([[True] * 5, [True] * 3 + [False] * 2])
    scene = torch.randn(2, 1, 16)  # the same vector added to every step of each window
    with torch.no_grad():
        a, b = model(x, mask, "ccd"), model(x + scene, mask, "ccd")
    assert torch.allclose(a[0], b[0], atol=1e-5) and torch.allclose(a[1], b[1], atol=1e-5)

    raw = PFM(_config()).eval()
    with torch.no_grad():
        assert not torch.allclose(raw(x, mask, "ccd")[0], raw(x + scene, mask, "ccd")[0], atol=1e-5)
    old = {k: v for k, v in _config().to_dict().items() if k != "input"}  # a checkpoint from before D-61
    assert Config(**old).input == "raw"
    with pytest.raises(ValueError):
        _config(input="minus_last")


def test_early_stopping_ends_after_patience_epochs_without_a_better_score_and_stays_stopped(tmp_path):
    data = _ccd(48)
    config = _config(domains=["ccd"], epochs=10)
    scores = iter([0.5, 0.9, 0.7, 0.9, 0.6, 0.95, 0.8, 0.8, 0.8, 0.8])
    messages = []

    def validate(model):
        return {"score": next(scores)}

    _, history = train_mod.train(config, {"ccd": data}, tmp_path, seed=0, validate=validate, patience=3,
                                 log=messages.append)
    # best at 2, then 3 epochs not better (the tie at 4 does not count), stop: epochs 1-5 trained
    assert [h["epoch"] for h in history] == [1, 2, 3, 4, 5] and "stopped early" in messages[-1]
    _, again = train_mod.train(config, {"ccd": data}, tmp_path, seed=0, validate=validate, patience=3, log=lambda m: None)
    assert len(again) == 5  # a resumed run that had stopped trains no further


def test_event_labels_merge_ucf_crime_into_one_event_class_and_old_configs_keep_their_classes():
    model = PFM(_config(labels="event")).eval()
    x, mask = torch.randn(3, 4, 16), torch.ones(3, 4, dtype=torch.bool)
    with torch.no_grad():
        assert model(x, mask, "ucf_crime")[0].shape == (3, 2) and model(x, mask, "ccd")[0].shape == (3, 2)
    old = {k: v for k, v in _config().to_dict().items() if k != "labels"}  # a checkpoint from before D-62
    assert Config(**old).labels == "classes" and PFM(Config(**old)).heads["ucf_crime"].out_features == 15
    with pytest.raises(ValueError):
        _config(labels="binary")

    samples = [{"label": label, "tau_s": "3" if label != "Normal" else "", "clip_key": label, "nominal_length_s": "30",
                "t_s": "1"} for label in ("Normal", "Abuse", "Vandalism")]
    data = train_mod.Data(None, samples, range(3), "ucf_crime")
    events = data.for_labels("event")
    assert events.names == ["Normal", "Event"] and events.y.tolist() == [0, 1, 1]
    assert events.label.tolist() == ["Normal", "Abuse", "Vandalism"] and data.y.tolist() == [0, 1, 13]  # unchanged
    assert data.for_labels("classes").y.tolist() == [0, 1, 13] and events.subset([2]).names == ["Normal", "Event"]
    ccd = _ccd(8)
    assert ccd.for_labels("event").y.tolist() == ccd.y.tolist() and ccd.for_labels("event").names == ["Normal", "Crash"]


def _ucf_validation():
    """One Abuse window (τ 25, 22, 8, 2) and one normal window of 4 samples."""
    samples = [{"label": "Abuse", "tau_s": str(tau), "clip_key": "a", "nominal_length_s": "30", "t_s": str(t)}
               for t, tau in ((5, 25), (8, 22), (22, 8), (28, 2))]
    samples += [{"label": "Normal", "tau_s": "", "clip_key": "n", "nominal_length_s": "30", "t_s": str(t)}
                for t in (0, 5, 25, 29)]
    return train_mod.Data(None, samples, range(8), "ucf_crime").for_labels("event")


def _p(scores):
    s = np.array(scores, float)
    return np.c_[1 - s, s]


def test_best_pt_is_chosen_by_the_within_window_auc_and_every_epoch_keeps_the_five_measures(tmp_path, monkeypatch):
    val = {"ucf_crime": _ucf_validation()}
    # epoch 1: every event sample above every normal one (pooled AUC 1.0) but falling as the event nears (within 0);
    # epoch 2: close above far inside the window (within 3.5 / 4) though the pooled AUC is low (0.25)
    epochs = iter([_p([0.9, 0.9, 0.6, 0.6] + [0.1] * 4), _p([0.2, 0.3, 0.5, 0.3] + [0.4] * 4)])
    monkeypatch.setattr(train_mod, "predict", lambda model, data, device="cpu": (next(epochs), np.zeros(len(data))))

    config = _config(domains=["ccd"], epochs=2)
    _, history = train_mod.train(config, {"ccd": _ccd(16)}, tmp_path, seed=0, log=lambda m: None,
                                 validate=lambda model: train_mod.validation(model, val))

    first, second = (h["validation"] for h in history)
    assert first["ucf_crime"]["pooled_auc"] == 1.0 and first["score"] == 0.0
    assert second["ucf_crime"]["pooled_auc"] == pytest.approx(0.25) and second["score"] == pytest.approx(0.875)
    assert second["chosen_by"] == "ucf_crime within_window_auc"
    for key in ("within_window_auc", "flat_line", "per_window_auc", "ap", "mtta_s", "pooled_auc", "detection_rate",
                "theta"):
        assert key in first["ucf_crime"] and key in second["ucf_crime"], key
    assert train_mod.load(tmp_path / "best.pt")[1]["epoch"] == 2


def test_a_split_without_a_window_that_has_both_parts_scores_minus_infinity(monkeypatch):
    samples = [{"label": "Abuse", "tau_s": str(tau), "clip_key": "a", "nominal_length_s": "30", "t_s": str(t)}
               for t, tau in ((0, 15), (10, 5))]
    samples += [{"label": "Normal", "tau_s": "", "clip_key": "n", "nominal_length_s": "30", "t_s": "0"}]
    data = train_mod.Data(None, samples, range(3), "ucf_crime").for_labels("event")
    monkeypatch.setattr(train_mod, "predict", lambda model, d, device="cpu": (_p([0.5, 0.6, 0.1]), np.zeros(3)))
    assert train_mod.validation(None, {"ucf_crime": data})["score"] == -math.inf

