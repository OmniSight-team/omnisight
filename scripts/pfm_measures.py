"""OMNI-917: measures that fit forecasting (D-64, written before any of these numbers was computed).

Every measure scores a sample by `1 - p(Normal)`, the same for a model with classes and a merged one (D-62).

1. **Within-window AUC** (`within_window`): inside each event window, every `close` sample against every `far`
   sample of the same window, the share where `close` scores higher (ties half); the mean over the windows that
   have both, and how many were skipped. Places never meet: both sides are the same video
2. **Flat-line check** (`flat_line`): the same inside each normal window, with `τ' = t_last - t` (time before the
   window's last sample) for `τ`. About 0.5 is right; well above means the score climbs with time anyway
3. **Per-window AUC** (`per_window_auc`): each window's highest score, event windows against normal windows
4. **AP and mTTA** (`ap_mtta`): every distinct window score is a warning line. An event window is a hit if any
   sample reaches it, its TTA the `τ` of the first that does (time order); a normal window is a false alarm if
   any sample does. AP is the step area under precision-recall, mTTA the hits' mean TTA averaged over the lines
   with a hit
5. **The pooled AUC** (`train_pfm._roc_auc` over every sample), unchanged

`RANGES` holds D-64's `close` / `far` per domain. Run from `notebooks/pfm_measures_gpu.ipynb` (validation) and
`notebooks/pfm_heldout_gpu.ipynb` (held-out). Locally: `uv run pytest scripts/tests/test_pfm_measures.py`.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pfm import Data, _roc_auc

RANGES = {"ccd": {"close": (0.0, 1.0), "far": (2.5, 5.0)},
          "ucf_crime": {"close": (0.0, 10.0), "far": (20.0, 30.0)}}


def _groups(data: Data, positive: bool):
    """Each window of one kind: its rows in time order."""
    which = (data.y != 0) == positive
    for w in dict.fromkeys(data.window[which]):
        rows = np.flatnonzero(data.window == w)
        yield rows[np.argsort(data.t_s[rows], kind="stable")]


def _higher(a: np.ndarray, b: np.ndarray) -> float:
    """The share of pairs where `a` is above `b`, ties half."""
    d = a[:, None] - b[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def _inside(data: Data, p: np.ndarray, positive: bool, close, far) -> dict:
    score, scores, skipped = 1.0 - p[:, 0], [], 0
    for rows in _groups(data, positive):
        tau = data.tau[rows] if positive else data.t_s[rows].max() - data.t_s[rows]
        near = rows[(tau >= close[0]) & (tau < close[1])]
        early = rows[(tau >= far[0]) & (tau < far[1])]
        if len(near) and len(early):
            scores.append(_higher(score[near], score[early]))
        else:
            skipped += 1
    return {"auc": float(np.mean(scores)) if scores else None, "windows": len(scores), "skipped": skipped}


def within_window(data: Data, p: np.ndarray, close=None, far=None) -> dict:
    """Measure 1 on the event windows."""
    r = RANGES[data.domain]
    return _inside(data, p, True, close or r["close"], far or r["far"])


def flat_line(data: Data, p: np.ndarray, close=None, far=None) -> dict:
    """Measure 2 on the normal windows."""
    r = RANGES[data.domain]
    return _inside(data, p, False, close or r["close"], far or r["far"])


def _window_max(data: Data, p: np.ndarray):
    score = 1.0 - p[:, 0]
    pos = [score[rows].max() for rows in _groups(data, True)]
    neg = [score[rows].max() for rows in _groups(data, False)]
    return np.array(pos), np.array(neg)


def per_window_auc(data: Data, p: np.ndarray) -> float:
    """Measure 3."""
    pos, neg = _window_max(data, p)
    return _roc_auc(np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)], np.r_[pos, neg])


def ap_mtta(data: Data, p: np.ndarray) -> dict:
    """Measure 4."""
    score = 1.0 - p[:, 0]
    pos, neg = _window_max(data, p)
    climbs = []  # per event window: the running maximum in time order and the τ of each sample
    for rows in _groups(data, True):
        climbs.append((np.maximum.accumulate(score[rows]), data.tau[rows]))
    lines = np.unique(np.r_[pos, neg])[::-1]
    ap, recall_before, ttas = 0.0, 0.0, []
    for line in lines:
        hits = int((pos >= line).sum())
        alarms = int((neg >= line).sum())
        recall = hits / len(pos) if len(pos) else 0.0
        if hits:
            ap += (recall - recall_before) * hits / (hits + alarms)
            first = [tau[np.searchsorted(run, line)] for run, tau in climbs if run[-1] >= line]
            ttas.append(float(np.mean(first)))
        recall_before = recall
    return {"ap": float(ap), "mtta_s": float(np.mean(ttas)) if ttas else None, "lines": len(lines)}


def measures(data: Data, p: np.ndarray) -> dict:
    """All five for one domain (the detection rate at `θ_d` comes from the caller's warnings)."""
    inside, flat = within_window(data, p), flat_line(data, p)
    return {"within_window_auc": inside["auc"], "within_window_windows": inside["windows"],
            "within_window_skipped": inside["skipped"],
            "flat_line": flat["auc"], "flat_line_windows": flat["windows"], "flat_line_skipped": flat["skipped"],
            "per_window_auc": per_window_auc(data, p), **ap_mtta(data, p),
            "pooled_auc": _roc_auc(data.y != 0, 1.0 - p[:, 0])}
