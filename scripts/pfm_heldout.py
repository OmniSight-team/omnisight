"""OMNI-906: the chosen PFM on the held-out split: SM-2, SM-3, false alarms, TC-5 (first half), the failure
analysis and the scene control (rules in D-60, written before the run).

- **The model and `θ_d`** come from the run record (`pfm_runs.py`), unchanged: nothing is refit on held-out
- **A sample warns** when its most likely class is an event and `max(p) ≥ θ_d` (X16, D-59)
- **Per window** (`windows`): a positive window's class is in the top 3 at its **last** grid point (the most
  lead-up seen, just before the event): SM-2 and TC-5's top-3. Its lead time is the `τ` of its first warning
  (any event), and `lead_correct_s` that of its first warning of the true class. The ceiling is the `τ` of
  its first sample (a warning at once); the summary's ceiling is over the windows that warned, beside their
  lead. A negative window is a false alarm if any of its samples warns
- **The scene control** (`FirstVector`): the same model, each sample given only its window's first grid
  vector, so it sees the scene but none of the lead-up. Measured exactly as the model
- **False alarms per hour** (`false_alarms_per_hour`, D-63): on the negative windows, each run of consecutive
  warning samples is one alarm; divided by the hours of normal footage the samples cover (one grid step each). The
  30 s and 60 s windows of a video overlap, so some footage counts twice, in both the alarms and the hours
- **The timeline** (`timeline`): the mean `1 - p(Normal)` of the positive samples per bin of time to the
  event, beside that of the negative samples: a forecaster's rises as the event nears

Run from `notebooks/pfm_heldout_gpu.ipynb` on Colab. Locally: `uv run pytest scripts/tests/test_pfm_heldout.py`.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pfm import Data, _roc_auc


class FirstVector:
    """A store's view where every sample is its window's first grid vector alone (the scene control)."""

    def __init__(self, store) -> None:
        self.store, self.dim = store, store.dim

    def batch(self, index, max_length: int) -> tuple[np.ndarray, np.ndarray]:
        x, mask = self.store.batch(index, max_length)
        x[:, 1:], mask[:, 1:] = 0.0, False
        return x, mask


def control(data: Data) -> Data:
    """`data` with the scene control's store."""
    part = data.subset(slice(None))
    part.store = FirstVector(data.store)
    return part


def warnings(p: np.ndarray, theta: float) -> np.ndarray:
    """Per sample: True when its most likely class is an event and `max(p) ≥ θ`."""
    return (p.argmax(axis=1) != 0) & (p.max(axis=1) >= theta)


def windows(data: Data, p: np.ndarray, theta: float) -> list[dict]:
    """One row per window (TC-5's per-clip table), its samples in time order."""
    names = data.names
    warns, top3 = warnings(p, theta), np.argsort(-p, axis=1)[:, :3]
    out = []
    for w in dict.fromkeys(data.window):
        rows = np.flatnonzero(data.window == w)
        rows = rows[np.argsort(data.t_s[rows])]
        y, last = int(data.y[rows[0]]), rows[-1]
        if (data.y[rows] != y).any():
            raise ValueError(f"window {w} mixes classes")
        clip, length = w.rsplit("|", 1)
        row = {"clip_key": clip, "length_s": length, "label": str(data.label[rows[0]]), "samples": len(rows),
               "warned": bool(warns[rows].any()), "p_event_first": float(1 - p[rows[0], 0]),
               "p_event_last": float(1 - p[last, 0])}
        if y:
            hit = rows[warns[rows]]
            right = hit[p[hit].argmax(axis=1) == y]
            row.update({"top3_last": bool(y in top3[last]), "top1_last": bool(p[last].argmax() == y),
                        "ceiling_s": float(data.tau[rows[0]]),
                        "lead_s": float(data.tau[hit[0]]) if len(hit) else None,
                        "lead_correct_s": float(data.tau[right[0]]) if len(right) else None,
                        "first_warning": names[int(p[hit[0]].argmax())] if len(hit) else ""})
        out.append(row)
    return out


def summary(data: Data, p: np.ndarray, dt: np.ndarray, rows: list[dict]) -> dict:
    """The domain's measures from its samples and its window rows."""
    positive = data.y != 0
    pos = [r for r in rows if r["label"] != "Normal"]
    neg = [r for r in rows if r["label"] == "Normal"]
    leads = np.array([r["lead_s"] for r in pos if r["lead_s"] is not None])
    right = [r["lead_correct_s"] for r in pos if r["lead_correct_s"] is not None]
    out = {
        "samples": len(data), "positive_windows": len(pos), "negative_windows": len(neg),
        "roc_auc": _roc_auc(positive, 1.0 - p[:, 0]),
        "top1": float((p.argmax(axis=1) == data.y).mean()),
        "top3_windows": float(np.mean([r["top3_last"] for r in pos])),
        "top1_windows": float(np.mean([r["top1_last"] for r in pos])),
        "top3_positive_samples": float((np.argsort(-p, axis=1)[positive, :3] == data.y[positive, None]).any(axis=1).mean()),
        "dt_mae_s": float(np.abs(dt[positive] - data.tau[positive]).mean()),
        "dt_mean_baseline_s": float(np.abs(data.tau[positive].mean() - data.tau[positive]).mean()),
        "false_alarm_windows": float(np.mean([r["warned"] for r in neg])),
        "warned_windows": len(leads), "unwarned_windows": len(pos) - len(leads),
        "lead_mean_s": float(leads.mean()) if len(leads) else None,
        "lead_quartiles_s": [float(q) for q in np.percentile(leads, [25, 50, 75])] if len(leads) else None,
        "lead_ceiling_s": float(np.mean([r["ceiling_s"] for r in pos if r["lead_s"] is not None])) if len(leads) else None,
        "warned_correct_windows": len(right), "lead_correct_mean_s": float(np.mean(right)) if right else None,
    }
    if len(leads):
        edges = np.arange(0, np.ceil(leads.max()) + 1.0, 0.5 if data.domain == "ccd" else 5.0)
        out["lead_histogram"] = {"edges": edges.tolist(), "counts": np.histogram(leads, edges)[0].tolist()}
    return out


def by_class(rows: list[dict]) -> dict:
    """The failure analysis: per class, its windows' top-3, warnings, lead and `p(event)` first and last."""
    out = {}
    for label in sorted({r["label"] for r in rows}):
        part = [r for r in rows if r["label"] == label]
        leads = [r["lead_s"] for r in part if r.get("lead_s") is not None]
        out[label] = {"windows": len(part), "warned": float(np.mean([r["warned"] for r in part])),
                      "p_event_first": float(np.mean([r["p_event_first"] for r in part])),
                      "p_event_last": float(np.mean([r["p_event_last"] for r in part]))}
        if label != "Normal":
            out[label].update({"top3_last": float(np.mean([r["top3_last"] for r in part])),
                               "lead_mean_s": float(np.mean(leads)) if leads else None,
                               "ceiling_s": float(np.mean([r["ceiling_s"] for r in part]))})
    return out


def timeline(data: Data, p: np.ndarray, step: float) -> dict:
    """The mean `1 - p(Normal)` of the positive samples per `step`-second bin of time to the event (bin
    starts), and over all negative samples."""
    positive = data.y != 0
    score = 1.0 - p[:, 0]
    bins = np.floor(data.tau[positive] / step).astype(int)
    starts = sorted(set(bins.tolist()))
    return {"tau_bin_start_s": [b * step for b in starts],
            "positive_mean": [float(score[positive][bins == b].mean()) for b in starts],
            "positive_count": [int((bins == b).sum()) for b in starts],
            "negative_mean": float(score[~positive].mean())}


def rise(data: Data, p: np.ndarray, near=(0.0, 10.0), far=(20.0, 30.0)) -> float:
    """D-61's forecasting check: the mean `1 - p(Normal)` of the positive samples `near` the event (τ in
    [near)) minus that of those `far` from it (τ in [far)). A scene recogniser's is about 0 or below."""
    positive, score = data.y != 0, 1.0 - p[:, 0]
    def mean(lo, hi):
        which = positive & (data.tau >= lo) & (data.tau < hi)
        return float(score[which].mean()) if which.any() else float("nan")
    return mean(*near) - mean(*far)


def false_alarms_per_hour(data: Data, p: np.ndarray, theta: float) -> dict:
    """D-63: alarms on normal footage per hour of it; an alarm is a run of consecutive warning samples."""
    warns, alarms, seconds = warnings(p, theta), 0, 0.0
    for w in dict.fromkeys(data.window[data.y == 0]):
        rows = np.flatnonzero(data.window == w)
        order = np.argsort(data.t_s[rows])
        rows, t = rows[order], data.t_s[rows][order]
        on = warns[rows]
        alarms += int(on[0]) + int((on[1:] & ~on[:-1]).sum())
        seconds += len(rows) * (float(np.median(np.diff(t))) if len(t) > 1 else 0.0)
    hours = seconds / 3600
    return {"false_alarms": alarms, "normal_hours": hours, "false_alarms_per_hour": alarms / hours if hours else None}


def measure(data: Data, p: np.ndarray, dt: np.ndarray, theta: float, step: float) -> dict:
    """Everything for one domain and one input (the model or the control): the window rows, the summary
    (overall and per nominal length when there are several), the classes and the timeline."""
    rows = windows(data, p, theta)
    out = {"summary": {**summary(data, p, dt, rows), "rise": rise(data, p), **false_alarms_per_hour(data, p, theta)},
           "by_class": by_class(rows), "timeline": timeline(data, p, step),
           "windows": rows}
    lengths = sorted({r["length_s"] for r in rows})
    if len(lengths) > 1:
        out["per_length"] = {}
        for length in lengths:
            which = np.array([w.rsplit("|", 1)[1] == length for w in data.window])
            part = data.subset(which)
            out["per_length"][length] = {**summary(part, p[which], dt[which], [r for r in rows if r["length_s"] == length]),
                                         "rise": rise(part, p[which]), **false_alarms_per_hour(part, p[which], theta)}
    return out
