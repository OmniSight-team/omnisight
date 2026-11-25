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
