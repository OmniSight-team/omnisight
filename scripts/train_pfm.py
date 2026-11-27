"""OMNI-904 / OMNI-X16 / OMNI-905: train the PFM (`pfm_model.py`) on the samples (`pfm_samples.py`), resumably,
and measure it on a split.

**Training.** Each step draws one batch per domain and sums the domains' losses with `λ_d`
(Appendix B.4). The order of an epoch and its dropout come from `(seed, epoch)`, so a run stopped
after an epoch and resumed from its checkpoint makes the same epochs as one that never stopped.
`{out}/last.pt` (model, optimiser, epoch, history, config, seed) is written atomically after every
epoch; a run with a checkpoint resumes from it, and refuses one made with another config or seed;
only `epochs` may grow, so a finished run can be extended and makes the same epochs as a longer one.

**Choosing the epoch** (OMNI-905, D-59): with `validate`, each epoch ends with a validation measure
whose `score` decides; the best epoch so far is kept in `{out}/best.pt` (the first one on a tie), and
its epoch and score live in `last.pt` too, so a resumed run keeps choosing the same way.
`validation` (OMNI-917, D-64) is that measure for the event-only model trained again: per domain the X16
measures and `pfm_measures`' five, so every epoch's are in the history; its `score` is UCF-Crime's
within-window AUC, not the pooled AUC.

**Measuring** (X16, on validation; written before training):
- ROC-AUC of "an event" against Normal: the score is `1 - p(Normal)`, over every sample
- top-1 accuracy over every sample; the mean absolute error of `Δt` over positive samples
- **lead time:** a sample warns when its most likely class is an event and `max(p) ≥ θ`. `θ` is the
  lowest value at which at most 10% of the split's negative windows warn at any of their samples
  (OMNI-905's example rule). Per positive window, the first warning in time order: its `τ` is the
  lead time. Mean over the windows that warn, and how many do

Run from `notebooks/pfm_train_gpu.ipynb` on Colab. Locally: `uv run pytest scripts/tests/test_pfm_model.py`.
"""

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfm_model import CLASSES, PFM, Config, class_names, loss
from pfm_samples import Store

FALSE_ALARM = 0.10


class Data:
    """One domain's samples of one split, with their vectors in `store`. `y` indexes `names`; `label` keeps
    each sample's own class (for the per-class analysis when the classes are merged, D-62)."""

    def __init__(self, store: Store, samples: list[dict], rows, domain: str) -> None:
        rows = np.asarray(rows, np.int64)
        names = CLASSES[domain]
        self.store, self.domain, self.rows, self.names = store, domain, rows, names
        self.label = np.array([samples[r]["label"] for r in rows])
        self.y = np.array([names.index(samples[r]["label"]) for r in rows], np.int64)
        self.tau = np.array([float(samples[r]["tau_s"]) if samples[r]["tau_s"] != "" else np.nan for r in rows],
                            np.float32)
        self.window = np.array([f"{samples[r]['clip_key']}|{samples[r]['nominal_length_s']}" for r in rows])
        self.t_s = np.array([float(samples[r]["t_s"]) for r in rows])

    def __len__(self) -> int:
        return len(self.rows)

    @classmethod
    def split(cls, store: Store, samples: list[dict], split: str, domain: str) -> "Data":
        return cls(store, samples, [i for i, s in enumerate(samples) if s["split"] == split], domain)

    def subset(self, which) -> "Data":
        part = object.__new__(Data)
        part.store, part.domain, part.names = self.store, self.domain, self.names
        for name in ("rows", "y", "label", "tau", "window", "t_s"):
            setattr(part, name, getattr(self, name)[which])
        return part

    def for_labels(self, labels: str) -> "Data":
        """The same samples with `y` in the classes a model with `Config.labels = labels` answers (D-62)."""
        names = class_names(self.domain, labels)
        part = self.subset(slice(None))
        part.names = names
        part.y = np.array([names.index(l) if l in names else names.index("Event") for l in part.label], np.int64)
        return part

    def tensors(self, which, max_length: int, device) -> tuple[torch.Tensor, ...]:
        x, mask = self.store.batch(self.rows[which], max_length)
        return (torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device),
                torch.from_numpy(self.y[which]).to(device), torch.from_numpy(self.tau[which]).to(device))


def _save(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _but_epochs(config: dict) -> dict:
    """A config without its number of epochs: a finished run may be extended (OMNI-914), nothing else may change."""
    return {k: v for k, v in config.items() if k != "epochs"}


def train(config: Config, data: dict[str, Data], out: Path, seed: int, *, device="cpu", stop_after: int | None = None,
          validate=None, patience: int | None = None, log=print) -> tuple[PFM, list[dict]]:
    """Train to `config.epochs` (or stop after epoch `stop_after`), resuming from `{out}/last.pt`.
    `validate(model) -> dict` with a `score`, if given, runs after every epoch: its result goes in the
    history and the best-scoring epoch is saved to `{out}/best.pt`. With `patience` (and `validate`),
    training stops early once `patience` epochs in a row have not beaten the best (OMNI-914, D-61); a
    resumed run that had already stopped trains no further."""
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "last.pt"
    torch.manual_seed(seed)
    model = PFM(config).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    start, history, best = 0, [], {"epoch": 0, "score": -math.inf}
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if _but_epochs(Config(**state["config"]).to_dict()) != _but_epochs(config.to_dict()) or state["seed"] != seed:
            raise RuntimeError(f"{checkpoint} was made with {state['config']}, seed {state['seed']}")
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        start, history, best = state["epoch"], state["history"], state.get("best", best)
        log(f"resumed after epoch {start}")
    for epoch in range(start, config.epochs):
        if stop_after is not None and epoch >= stop_after:
            break
        if patience is not None and validate is not None and epoch - best["epoch"] >= patience:
            log(f"stopped early: no better validation for {patience} epochs since epoch {best['epoch']}")
            break
        began = time.perf_counter()
        torch.manual_seed(seed * 10_007 + epoch)  # this epoch's dropout
        rng = np.random.default_rng([seed, epoch])
        orders = {d: rng.permutation(len(part)) for d, part in data.items()}
        batches = {d: math.ceil(len(part) / config.batch) for d, part in data.items()}
        model.train()
        sums: dict[str, float] = {}
        steps = max(batches.values())
        for step in range(steps):
            outputs, targets = {}, {}
            for d, part in data.items():
                b = step % batches[d]
                x, mask, y, tau = part.tensors(orders[d][b * config.batch : (b + 1) * config.batch], config.max_length, device)
                outputs[d], targets[d] = model(x, mask, d), (y, tau)
            total, parts = loss(outputs, targets, config)
            optimiser.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            sums["loss"] = sums.get("loss", 0.0) + float(total.detach())
            for d, part in parts.items():
                for name, value in part.items():
                    sums[f"{d}_{name}"] = sums.get(f"{d}_{name}", 0.0) + value
        record = {"epoch": epoch + 1, **{k: v / steps for k, v in sums.items()}, "seconds": time.perf_counter() - began}
        if validate is not None:
            record["validation"] = validate(model)
            if record["validation"]["score"] > best["score"]:
                best = {"epoch": epoch + 1, "score": record["validation"]["score"]}
                _save(out / "best.pt", {"model": model.state_dict(), "epoch": epoch + 1, "score": best["score"],
                                        "config": config.to_dict(), "seed": seed})
        history.append(record)
        _save(checkpoint, {"model": model.state_dict(), "optimiser": optimiser.state_dict(), "epoch": epoch + 1,
                           "history": history, "config": config.to_dict(), "seed": seed, "best": best})
        score = f", validation {record['validation']['score']:.4f}" if validate is not None else ""
        log(f"epoch {epoch + 1}: loss {record['loss']:.4f}{score} ({record['seconds']:.1f} s)")
    return model, history


def load(path: Path, device="cpu") -> tuple[PFM, dict]:
    """A model from a checkpoint (`best.pt` or `last.pt`), in eval mode, and the checkpoint's other fields."""
    state = torch.load(path, map_location=device, weights_only=False)
    model = PFM(Config(**state["config"])).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), {k: v for k, v in state.items() if k not in ("model", "optimiser")}


@torch.no_grad()
def predict(model: PFM, data: Data, device="cpu", batch: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """`(p, Δt)` for every sample of `data`, in order."""
    model.eval()
    probs, times = [], []
    for start in range(0, len(data), batch):
        x, mask, _, _ = data.tensors(np.arange(start, min(start + batch, len(data))), model.config.max_length, device)
        logits, dt = model(x, mask, data.domain)
        probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        times.append(dt.float().cpu().numpy())
    return np.concatenate(probs), np.concatenate(times)


def _roc_auc(positive: np.ndarray, score: np.ndarray) -> float:
    """The probability that a random positive scores above a random negative (ties count half)."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    i = 0
    while i < len(score):  # average ranks over ties
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate(data: Data, p: np.ndarray, dt: np.ndarray, false_alarm: float = FALSE_ALARM) -> dict:
    """The X16 measures (the module docstring) for one split."""
    positive = data.y != 0
    event = p.argmax(axis=1) != 0
    alarm = np.where(event, p.max(axis=1), 0.0)  # a sample warns when alarm >= θ (and > 0)

    negative_windows = sorted(set(data.window[~positive]))
    worst = np.array([alarm[data.window == w].max() for w in negative_windows])
    allowed = math.floor(false_alarm * len(worst))
    ranked = np.sort(worst)[::-1]
    theta = float(np.nextafter(ranked[allowed], np.inf)) if allowed < len(ranked) else 0.0
    theta = max(theta, float(np.nextafter(0.0, 1.0)))
    warns = alarm >= theta

    leads, windows = [], sorted(set(data.window[positive]))
    for w in windows:
        rows = np.flatnonzero(data.window == w)
        rows = rows[np.argsort(data.t_s[rows])]
        hit = rows[warns[rows]]
        if len(hit):
            leads.append(float(data.tau[hit[0]]))
    return {
        "samples": len(data), "positive_samples": int(positive.sum()),
        "roc_auc": _roc_auc(positive, 1.0 - p[:, 0]),
        "top1": float((p.argmax(axis=1) == data.y).mean()),
        "top3_positive": float((np.argsort(-p, axis=1)[positive, :3] == data.y[positive, None]).any(axis=1).mean()),
        "dt_mae_s": float(np.abs(dt[positive] - data.tau[positive]).mean()),
        "theta": theta, "false_alarm_windows": float((worst >= theta).mean()), "negative_windows": len(worst),
        "positive_windows": len(windows), "warned_windows": len(leads),
        "lead_time_s": float(np.mean(leads)) if leads else None,
    }


def validation(model: PFM, data: dict[str, Data], device="cpu", choose_by: str = "ucf_crime") -> dict:
    """OMNI-917 (D-64): per domain `evaluate`'s measures, `pfm_measures.measures` and the detection rate at
    `θ`; `score` is `choose_by`'s within-window AUC (`-inf` if no window has both parts)."""
    import pfm_measures  # here, not at the top: pfm_measures imports this module

    out = {}
    for d, part in data.items():
        p, dt = predict(model, part, device)
        e = evaluate(part, p, dt)
        out[d] = {**e, **pfm_measures.measures(part, p),
                  "detection_rate": e["warned_windows"] / e["positive_windows"] if e["positive_windows"] else None}
    score = out[choose_by]["within_window_auc"]
    out["score"] = -math.inf if score is None else score
    out["chosen_by"] = f"{choose_by} within_window_auc"
    return out


def x16_rule(results: dict[str, list[dict]]) -> dict:
    """The ticket's rule: Cosmos wins only if its mean AUC beats InternVideo2's by more than the larger
    seed spread (max - min) and its mean lead time is not lower. Otherwise InternVideo2 stays."""
    summary = {}
    for encoder, runs in results.items():
        aucs = [r["roc_auc"] for r in runs]
        leads = [r["lead_time_s"] for r in runs if r["lead_time_s"] is not None]
        summary[encoder] = {"auc_mean": float(np.mean(aucs)), "auc_spread": float(max(aucs) - min(aucs)),
                            "lead_mean_s": float(np.mean(leads)) if leads else 0.0,
                            **{f"{k}_mean": float(np.mean([r[k] for r in runs])) for k in ("top1", "dt_mae_s")}}
    c, i = summary["cosmos"], summary["internvideo2"]
    margin = max(c["auc_spread"], i["auc_spread"])
    wins = c["auc_mean"] - i["auc_mean"] > margin and c["lead_mean_s"] >= i["lead_mean_s"]
    return {"summary": summary, "margin": margin, "winner": "cosmos" if wins else "internvideo2"}
