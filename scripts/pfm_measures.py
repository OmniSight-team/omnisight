"""OMNI-X18: the forecaster's eight measures (D-65, written before any training of OMNI-X21).

Every measure scores a grid point ("second": 1 s on UCF-Crime, 0.5 s on CCD) by `score = 1 - p(Normal)`, the
same for a model with classes and a merged one (D-62).

1. **Within-window AUC** (`within_window`, the main one): inside each event window, every `close` second against
   every `far` second of the same window, the share where `close` scores higher (ties half); the mean over the
   windows that have both, and how many were skipped. Places never meet: both sides are the same video.
   Held-out UCF-Crime also gets it with `close` τ in [0, 8) s (`HELDOUT_CLOSE`), reported separately
2. **Flat-line check** (`flat_line`): the same inside each normal window, with `τ' = t_last - t` (time back from
   the window's last second) for `τ`. About 0.5 is right; **≤ 0.60 is required** (`FLAT_LINE_MAX`)
3. **Lead time** (`mtta_s`): over the event windows warned at `θ_d`, the mean `τ` of each one's first second
   (time order) with `score ≥ θ_d`
4. **Detection rate**: the share of event windows with any second at `score ≥ θ_d`
5. **False alarms**: the share of normal windows with any warning second; and alarms per hour of normal footage,
   a run of consecutive warning seconds being one alarm, each second standing for one grid step of footage
6. **AP** (`average_precision`): every distinct window maximum is a warning line; at each, an event window is a hit
   if any second reaches it, a normal window a false alarm if any does; the step area under precision-recall
7. **Per-window AUC** (`per_window_auc`): each window's highest score, event windows against normal windows
8. **Pooled AUC**: every second of every window in one pile (`train_pfm._roc_auc`); for comparison only

**`θ_d`** (`threshold`, the 10% rule, D-59): the lowest value at which at most 10% of the split's normal windows
have a second at `score ≥ θ_d`. Fit on validation only; held-out is given validation's (`measures(..., theta=)`).

`RANGES` holds D-65's `close` / `far` per domain. Locally: `uv run pytest scripts/tests/test_pfm_measures.py`.
How many windows each range keeps, per dataset and split (no model needed):

    uv run python scripts/pfm_measures.py kept validation      # or: kept held_out
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pfm import FALSE_ALARM, Data, _roc_auc

RANGES = {"ccd": {"close": (0.0, 1.0), "far": (2.5, 5.0)},
          "ucf_crime": {"close": (8.0, 13.0), "far": (20.0, 30.0)}}
HELDOUT_CLOSE = (0.0, 8.0)  # held-out UCF-Crime only: closer than train and validation reach (D-65)
STEP_S = {"ccd": 0.5, "ucf_crime": 1.0}  # the grid (D-54)
FLAT_LINE_MAX = 0.60


def _score(p: np.ndarray) -> np.ndarray:
    return 1.0 - p[:, 0]


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


def _parts(data: Data, rows: np.ndarray, positive: bool, close, far):
    tau = data.tau[rows] if positive else data.t_s[rows].max() - data.t_s[rows]
    return rows[(tau >= close[0]) & (tau < close[1])], rows[(tau >= far[0]) & (tau < far[1])]


def _inside(data: Data, p: np.ndarray, positive: bool, close, far) -> dict:
    score, scores, skipped = _score(p), [], 0
    for rows in _groups(data, positive):
        near, early = _parts(data, rows, positive, close, far)
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
    score = _score(p)
    pos = [score[rows].max() for rows in _groups(data, True)]
    neg = [score[rows].max() for rows in _groups(data, False)]
    return np.array(pos), np.array(neg)


def threshold(data: Data, p: np.ndarray, false_alarm: float = FALSE_ALARM) -> float:
    """`θ_d`: the lowest value at which at most `false_alarm` of the normal windows warn (D-59's rule)."""
    _, worst = _window_max(data, p)
    allowed = math.floor(false_alarm * len(worst))
    ranked = np.sort(worst)[::-1]
    theta = float(np.nextafter(ranked[allowed], np.inf)) if allowed < len(ranked) else 0.0
    return max(theta, float(np.nextafter(0.0, 1.0)))


def at_threshold(data: Data, p: np.ndarray, theta: float) -> dict:
    """Measures 3 to 5 at `θ_d`."""
    warns = _score(p) >= theta
    leads, events = [], 0
    for rows in _groups(data, True):
        events += 1
        hit = rows[warns[rows]]
        if len(hit):
            leads.append(float(data.tau[hit[0]]))
    alarmed, normals, alarms, seconds = 0, 0, 0, 0.0
    for rows in _groups(data, False):
        normals += 1
        on = warns[rows]
        alarmed += int(on.any())
        alarms += int(on[0]) + int((on[1:] & ~on[:-1]).sum())
        seconds += len(rows) * STEP_S[data.domain]
    hours = seconds / 3600
    return {"theta": theta, "event_windows": events, "warned_windows": len(leads),
            "mtta_s": float(np.mean(leads)) if leads else None,
            "detection_rate": len(leads) / events if events else None,
            "normal_windows": normals, "false_alarm_windows": alarmed / normals if normals else None,
            "false_alarms": alarms, "normal_hours": hours, "false_alarms_per_hour": alarms / hours if hours else None}


def average_precision(data: Data, p: np.ndarray) -> float | None:
    """Measure 6."""
    pos, neg = _window_max(data, p)
    if not len(pos):
        return None
    ap, recall_before = 0.0, 0.0
    for line in np.unique(np.r_[pos, neg])[::-1]:
        hits, alarms = int((pos >= line).sum()), int((neg >= line).sum())
        recall = hits / len(pos)
        if hits:
            ap += (recall - recall_before) * hits / (hits + alarms)
        recall_before = recall
    return float(ap)


def per_window_auc(data: Data, p: np.ndarray) -> float | None:
    """Measure 7."""
    pos, neg = _window_max(data, p)
    if not len(pos) or not len(neg):
        return None
    return _roc_auc(np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)], np.r_[pos, neg])


def measures(data: Data, p: np.ndarray, theta: float | None = None, heldout: bool = False) -> dict:
    """All eight for one domain and split. `theta=None` fits `θ_d` on this split (validation only); held-out is
    given validation's. `heldout` adds UCF-Crime's τ [0, 8) s within-window AUC."""
    inside, flat = within_window(data, p), flat_line(data, p)
    positive = data.y != 0
    out = {"within_window_auc": inside["auc"], "within_window_windows": inside["windows"],
           "within_window_skipped": inside["skipped"],
           "flat_line": flat["auc"], "flat_line_windows": flat["windows"], "flat_line_skipped": flat["skipped"],
           **at_threshold(data, p, threshold(data, p) if theta is None else theta),
           "ap": average_precision(data, p), "per_window_auc": per_window_auc(data, p),
           "pooled_auc": _roc_auc(positive, _score(p)) if positive.any() and (~positive).any() else None}
    if heldout and data.domain == "ucf_crime":
        near = within_window(data, p, close=HELDOUT_CLOSE)
        out.update({"within_window_auc_0_8": near["auc"], "within_window_0_8_windows": near["windows"],
                    "within_window_0_8_skipped": near["skipped"]})
    return out


def kept(data: Data, close_ranges: dict[str, tuple] | None = None) -> dict:
    """How many windows each range keeps (no model): event and normal windows with a `close` part, a `far` part,
    and both (the ones measures 1 and 2 use)."""
    r = RANGES[data.domain]
    closes = close_ranges or {"close": r["close"]}
    out = {}
    for kind, positive in (("event", True), ("normal", False)):
        groups = list(_groups(data, positive))
        out[kind] = {"windows": len(groups)}
        for name, close in closes.items():
            parts = [_parts(data, rows, positive, close, r["far"]) for rows in groups]
            out[kind][f"{name} {close}"] = sum(1 for near, _ in parts if len(near))
            out[kind][f"far {r['far']}"] = sum(1 for _, early in parts if len(early))
            out[kind][f"both ({name})"] = sum(1 for near, early in parts if len(near) and len(early))
    return out


def main() -> int:
    import json

    from pfm_samples import ROOT, read_samples

    command, split = sys.argv[1], sys.argv[2]
    if command != "kept":
        raise SystemExit(f"unknown command {command}")
    for domain in ("ccd", "ucf_crime"):
        samples = read_samples(ROOT / "datasets" / "samples" / f"{domain}.csv")
        data = Data(None, samples, [i for i, s in enumerate(samples) if s["split"] == split], domain).for_labels("event")
        closes = {"close": RANGES[domain]["close"]}
        if domain == "ucf_crime" and split == "held_out":
            closes["close 0-8"] = HELDOUT_CLOSE
        print(domain, split, json.dumps(kept(data, closes), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
