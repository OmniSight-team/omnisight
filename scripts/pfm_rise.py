"""OMNI-X24 (D-67): the rise rule. UCF-Crime warns when a camera's score jumps, not when it is high.

- **Score** of a second: `1 - p(Normal)` (`pfm_measures`). **The rise** at a second `t` (`rise`):
  `score(t) - mean(score of the same window's seconds in [t - 20 s, t - 5 s))`. It exists only if that baseline
  holds at least 10 s of footage (points × grid step); otherwise it is NaN and the second cannot warn
- **A second warns** when `rise(t) ≥ δ`. **`δ`** (`line`): the lowest value at which at most a share `F` of the
  split's normal windows have a warning second. Set on validation, `F` = 5% main, 2% and 10% beside (`LEVELS`)
- **The level line** for comparison (and CCD's rule): `θ` by the same share on `score` itself
  (`pfm_measures.threshold`)
- **The measures** (`report`): per `F`, the rise's and the level's detection rate, lead time and false alarms
  (`pfm_measures.at_threshold` on the rise in place of the score); the rise's per-window AUC and AP (each window's
  largest rise; a window with none counts as the lowest); overall and per window length

The live version (OMNI-911) keeps each camera's last 20 s of scores and computes the same rise.

**The run** (`run`, Colab; the existing vectors, no encoding, no training): for the served `pfm-v5a-s1-e14` and B's
reference `pfm-v5b-s1-e14`, the lines fit on validation (UCF-Crime: `δ` and `θ`; CCD: `θ`), then validation and
held-out measured with them. Held-out UCF-Crime is D-66's shifted event windows plus the unchanged normal windows.
Writes `results/x24_rise.json` on Drive. On Colab, after cloning the branch and mounting Drive:

    import pfm_rise; pfm_rise.run("/content/drive/MyDrive/omnisight", "/content/omni-sight", "/content/pfm", "cuda")

Locally: `uv run pytest scripts/tests/test_pfm_rise.py`.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pfm_measures
from train_pfm import Data

BASELINE_S = (5.0, 20.0)  # the baseline: [t - 20 s, t - 5 s)
MIN_HISTORY_S = 10.0  # at least this much footage in the baseline
LEVELS = (0.02, 0.05, 0.10)  # false-alarm shares; 0.05 is the main one
MAIN = 0.05
_EPS = 1e-6


def rise(data: Data, p: np.ndarray, baseline=BASELINE_S, min_history_s=MIN_HISTORY_S) -> np.ndarray:
    """Per sample, its rise over its own window's baseline; NaN where the baseline is too short."""
    score = 1.0 - p[:, 0]
    step = pfm_measures.STEP_S[data.domain]
    out = np.full(len(score), np.nan)
    for w in dict.fromkeys(data.window):
        rows = np.flatnonzero(data.window == w)
        rows = rows[np.argsort(data.t_s[rows], kind="stable")]
        t = data.t_s[rows]
        for i, r in enumerate(rows):
            back = t[i] - t
            base = (back > baseline[0] + _EPS) & (back < baseline[1] + _EPS)  # t - 20 <= t_j < t - 5
            if base.sum() * step >= min_history_s - _EPS:
                out[r] = score[r] - score[rows[base]].mean()
    return out


def as_p(values: np.ndarray) -> np.ndarray:
    """A stand-in `p` whose `1 - p(Normal)` is `values` (NaN: never warns), so `pfm_measures` reads the rise."""
    v = np.where(np.isnan(values), -np.inf, values)
    return np.c_[1.0 - v, v]


def line(data: Data, values: np.ndarray, false_alarm: float) -> float:
    """The lowest value at which at most `false_alarm` of the normal windows have a second at or above it."""
    worst = np.array([np.max(np.where(np.isnan(values[rows]), -np.inf, values[rows]))
                      for rows in pfm_measures._groups(data, False)])
    allowed = math.floor(false_alarm * len(worst))
    ranked = np.sort(worst)[::-1]
    return float(np.nextafter(ranked[allowed], np.inf)) if allowed < len(ranked) else -math.inf


def fit(data: Data, p: np.ndarray, levels=LEVELS, use_rise: bool = True) -> dict:
    """On validation: per level, `δ` (the rise, if `use_rise`) and `θ` (the level line)."""
    r = rise(data, p) if use_rise else None
    score = 1.0 - p[:, 0]
    return {str(f): {"delta": line(data, r, f) if use_rise else None, "theta": line(data, score, f)} for f in levels}


def _at(data: Data, values: np.ndarray, value: float) -> dict:
    m = pfm_measures.at_threshold(data, as_p(values), value)
    m["line"] = m.pop("theta")
    return m


def _one(data: Data, p: np.ndarray, lines: dict, use_rise: bool) -> dict:
    score = 1.0 - p[:, 0]
    out = {"levels": {}}
    r = rise(data, p) if use_rise else None
    for f, l in lines.items():
        out["levels"][f] = {"level": _at(data, score, l["theta"])}
        if use_rise:
            out["levels"][f]["rise"] = _at(data, r, l["delta"])
    out["level_per_window_auc"] = pfm_measures.per_window_auc(data, p)
    if use_rise:
        out["rise_per_window_auc"] = pfm_measures.per_window_auc(data, as_p(r))
        out["rise_ap"] = pfm_measures.average_precision(data, as_p(r))
        out["rise_seconds"] = int((~np.isnan(r)).sum())
    return out


def report(data: Data, p: np.ndarray, lines: dict, use_rise: bool = True) -> dict:
    """Everything for one domain and split at the given lines, overall and per window length (UCF-Crime)."""
    out = _one(data, p, lines, use_rise)
    lengths = sorted({w.rsplit("|", 1)[1] for w in data.window} - {""})
    if len(lengths) > 1:
        out["per_length"] = {}
        for length in lengths:
            which = np.array([w.rsplit("|", 1)[1] == length for w in data.window])
            out["per_length"][length] = _one(data.subset(which), p[which], lines, use_rise)
    return out


def wins(held: dict) -> dict:
    """D-67's test on held-out UCF-Crime at F = 5%: the rise warns on more crime windows than the level line."""
    m = held["levels"][str(MAIN)]
    rise_n, level_n = m["rise"]["warned_windows"], m["level"]["warned_windows"]
    return {"rise_warned": rise_n, "level_warned": level_n, "event_windows": m["rise"]["event_windows"],
            "rise_false_alarm_windows": m["rise"]["false_alarm_windows"],
            "level_false_alarm_windows": m["level"]["false_alarm_windows"], "rise_wins": rise_n > level_n}


RUNS = {"pfm-v5a-s1-e14": "served (D-65's rule)", "pfm-v5b-s1-e14": "reference only, NOT chosen by the rule"}


def _copy(samples: list[dict], source: Path, target: Path, optional: bool = False) -> Path:
    """The samples' videos' vector files copied once (temporary name, then renamed); `optional`: skip missing."""
    import os
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    from pfm_samples import out_file

    def one(key):
        src, dst = out_file(source, key), out_file(target, key)
        if dst.is_file() or (optional and not src.is_file()):
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + ".part")
        shutil.copyfile(src, part)
        os.replace(part, dst)

    target.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(one, sorted({s["clip_key"] for s in samples})))
    return target


def _line(name: str, m: dict) -> str:
    def v(x, spec):
        return "-" if x is None else format(x, spec)
    return (f"{name} detection {v(m['detection_rate'], '.3f')} ({m['warned_windows']}/{m['event_windows']})  "
            f"lead {v(m['mtta_s'], '.2f')} s  false alarms {v(m['false_alarm_windows'], '.3f')} of "
            f"{m['normal_windows']}, {v(m['false_alarms_per_hour'], '.1f')}/h")


def run(drive, repo_dir, local, device="cuda", log=print) -> dict:
    """D-67's run: see the module's docstring."""
    import hashlib
    import json
    import subprocess
    import time

    import torch

    import pfm_runs
    import train_pfm
    from pfm_samples import Store, read_samples

    drive, repo_dir, local = Path(drive), Path(repo_dir), Path(local)
    emb, samples_dir = drive / "embeddings" / "pfm", repo_dir / "datasets" / "samples"
    ucf, ccd = read_samples(samples_dir / "ucf_crime.csv"), read_samples(samples_dir / "ccd.csv")
    shifted = read_samples(samples_dir / "ucf_crime_heldout_shift8.csv")
    split = {("ucf_crime", "validation"): [s for s in ucf if s["split"] == "validation"],
             ("ucf_crime", "held_out"): [s for s in ucf if s["split"] == "held_out" and s["label"] == "Normal"] + shifted,
             ("ccd", "validation"): [s for s in ccd if s["split"] == "validation"],
             ("ccd", "held_out"): [s for s in ccd if s["split"] == "held_out"]}
    start = time.time()
    folders = {"ucf_crime": _copy(split["ucf_crime", "validation"] + split["ucf_crime", "held_out"],
                                  emb / "ucf_crime" / "internvideo2", local / "ucf_crime"),
               "ccd": _copy(split["ccd", "validation"] + split["ccd", "held_out"], emb / "ccd" / "internvideo2",
                            local / "ccd")}
    extra = _copy(shifted, emb / "ucf_crime" / "internvideo2_shift8", local / "ucf_crime_shift8", optional=True)
    data = {}
    for (d, s), rows in split.items():
        store = Store(rows, folders[d], extra_dir=extra if (d, s) == ("ucf_crime", "held_out") else None)
        data[d, s] = train_pfm.Data(store, rows, range(len(rows)), d).for_labels("event")
        log(f"{d} {s}: {len(rows):,} samples, {len(set(data[d, s].window)):,} windows")
    log(f"vectors ready ({time.time() - start:.0f} s)")
    assert pfm_runs.download(drive, None, target=local / "chosen")["run_id"] == "pfm-v5a-s1-e14"

    results = {}
    for rid, role in RUNS.items():
        record = pfm_runs.download(drive, rid, target=local / rid)
        model, state = train_pfm.load(local / rid / pfm_runs.CHECKPOINT, device)
        p = {k: train_pfm.predict(model, x, device)[0] for k, x in data.items()}
        lines = {"ucf_crime": fit(data["ucf_crime", "validation"], p["ucf_crime", "validation"]),
                 "ccd": fit(data["ccd", "validation"], p["ccd", "validation"], use_rise=False)}
        res = {"role": role, "epoch": state["epoch"], "checkpoint_sha256": record["checkpoint_sha256"], "lines": lines}
        log(f"\n{rid} ({role}), epoch {state['epoch']}")
        for (d, s), x in data.items():
            r = report(x, p[d, s], lines[d], use_rise=d == "ucf_crime")
            res[f"{d} {s}"] = r
            head = f"  {d} {s}"
            if d == "ucf_crime":
                head += (f": rise per-window AUC {r['rise_per_window_auc']:.4f}, AP {r['rise_ap']:.4f} "
                         f"(level per-window AUC {r['level_per_window_auc']:.4f})")
            log(head)
            for f, l in lines[d].items():
                m = r["levels"][f]
                if d == "ucf_crime":
                    log(f"    F {float(f):.0%}  δ {l['delta']:.4f}  " + _line("rise: ", m["rise"]))
                    log(f"           θ {l['theta']:.4f}  " + _line("level:", m["level"]))
                    for length, part in r.get("per_length", {}).items():
                        q = part["levels"][f]
                        log(f"           {length} s: rise {q['rise']['warned_windows']}/{q['rise']['event_windows']}, "
                            f"level {q['level']['warned_windows']}/{q['level']['event_windows']}")
                else:
                    log(f"    F {float(f):.0%}  θ {l['theta']:.4f}  " + _line("level:", m["level"]))
        w = res["d67"] = wins(res["ucf_crime held_out"])
        results[rid] = res
        verdict = "the rise rule wins" if w["rise_wins"] else "the level line stays"
        log(f"  D-67 at F 5%, held-out UCF-Crime: rise {w['rise_warned']} against level {w['level_warned']} of "
            f"{w['event_windows']} crime windows (false alarms {w['rise_false_alarm_windows']:.3f} against "
            f"{w['level_false_alarm_windows']:.3f}): {verdict}"
            + ("" if rid == "pfm-v5a-s1-e14" else "  [reference only: decides nothing]"))

    commit = subprocess.run(["git", "-C", str(repo_dir), "log", "-1", "--format=%h %s"], capture_output=True,
                            text=True).stdout.strip()
    out = drive / "results" / "x24_rise.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "commit": commit, "device": torch.cuda.get_device_name(0) if device == "cuda" else device,
        "torch": torch.__version__, "baseline_s": BASELINE_S, "min_history_s": MIN_HISTORY_S, "levels": LEVELS,
        "samples_sha256": {n: hashlib.sha256((samples_dir / n).read_bytes()).hexdigest()
                           for n in ("ucf_crime.csv", "ccd.csv", "ucf_crime_heldout_shift8.csv")},
        "results": results}, indent=1, default=float))
    log(f"\n{out}")
    return results
