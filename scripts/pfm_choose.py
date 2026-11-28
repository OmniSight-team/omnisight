"""OMNI-X22: D-65's choice from OMNI-X21's records, the two comparisons, the validation table, and the record of
the served run. Nothing is trained or refit: every number comes from the histories in `results/x21_v5{a,b,c}.json`
(each seed's best epoch, the one in its `best.pt`; `θ_d` is that epoch's 10% rule on validation).

- **The served model:** among the seeds of A and B with a passing epoch, the highest UCF-Crime validation
  within-window AUC; on a tie the lower seed, then A. None passed: nothing new is served
- **What Fix 2 adds** (A against B) and **does sharing help** (B against C): each version's mean and spread (best
  minus worst seed) of the UCF-Crime within-window AUC; a difference is claimed only if larger than the larger spread

    uv run python scripts/pfm_choose.py              # print the choice, the comparisons and the table
    uv run python scripts/pfm_choose.py --register   # also: best.pt -> models/pfm/runs/{run_id}/model.pt, its
                                                     # run.json, and model_runs.json names it chosen
    ... --drive "<path to MyDrive/omnisight>"        # default G:/My Drive/omnisight

Writes `results/x22_choice.json` on Drive. The held-out run is `notebooks/pfm_heldout_gpu.ipynb` (Colab).
"""

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pfm_runs
import train_pfm

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ("A", "B", "C")
SERVABLE = ("A", "B")  # C has no CCD head
DOMAINS = ("ccd", "ucf_crime")
MAIN = "within_window_auc"
KEYS = ("within_window_auc", "flat_line", "mtta_s", "detection_rate", "false_alarm_windows", "false_alarms_per_hour",
        "ap", "per_window_auc", "pooled_auc", "theta", "dt_mae_s")


def run_id(version: str, seed: int, epoch: int) -> str:
    return f"pfm-v5{version.lower()}-s{seed}-e{epoch}"


def choose(best: dict[tuple[str, int], float | None]) -> tuple[str, int] | None:
    """`best[(version, seed)]` is that seed's best UCF-Crime within-window AUC, None if no epoch passed. The served
    (version, seed): the highest of A and B; on a tie the lower seed, then A. None if none passed."""
    ok = [(v, s) for (v, s), score in best.items() if v in SERVABLE and score is not None]
    if not ok:
        return None
    return max(ok, key=lambda k: (best[k], -k[1], -SERVABLE.index(k[0])))


def spread(scores: list[float]) -> dict:
    return {"mean": float(np.mean(scores)), "spread": float(max(scores) - min(scores)), "seeds": scores}


def compare(first: list[float], second: list[float]) -> dict:
    """The difference of the means (second minus first), claimed only if larger than the larger spread."""
    a, b = spread(first), spread(second)
    diff = b["mean"] - a["mean"]
    return {"first": a, "second": b, "difference": diff, "larger_spread": max(a["spread"], b["spread"]),
            "claimed": abs(diff) > max(a["spread"], b["spread"])}


def read(drive: Path) -> dict[str, dict]:
    return {v: json.loads((drive / "results" / f"x21_v5{v.lower()}.json").read_text(encoding="utf-8")) for v in VERSIONS}


def bests(records: dict[str, dict]) -> dict[tuple[str, int], dict | None]:
    return {(v, int(s)): b for v, r in records.items() for s, b in r["best_per_seed"].items()}


def main() -> int:
    args = sys.argv[1:]
    drive = pfm_runs.DRIVE
    if "--drive" in args:
        i = args.index("--drive")
        drive, args = Path(args[i + 1]), args[:i] + args[i + 2:]
    register = "--register" in args

    records = read(drive)
    best = bests(records)
    score = {k: None if b is None else b["validation"]["ucf_crime"][MAIN] for k, b in best.items()}
    chosen = choose(score)
    ucf = {v: [score[v, s] for s in sorted(int(x) for x in records[v]["best_per_seed"]) if score[v, s] is not None]
           for v in VERSIONS}
    fix2, sharing = compare(ucf["A"], ucf["B"]), compare(ucf["B"], ucf["C"])

    print("validation, each seed's best epoch (the one in best.pt)")
    for d in DOMAINS:
        print(f"\n{d}\n  version seed epoch  " + "  ".join(KEYS) + "  windows (within / flat)")
        for (v, s), b in sorted(best.items()):
            if b is None:
                print(f"  {v}       {s}    failed (no passing epoch)")
            elif d in b["validation"]:
                m = b["validation"][d]
                print(f"  {v}       {s}    {b['epoch']:<5d}  " + "  ".join(f"{m[k]:.4f}" for k in KEYS)
                      + f"  {m['within_window_windows']}+{m['within_window_skipped']} / "
                        f"{m['flat_line_windows']}+{m['flat_line_skipped']}")
    for name, c, (x, y) in (("What Fix 2 adds (A -> B)", fix2, "AB"), ("Does sharing help (B -> C)", sharing, "BC")):
        print(f"\n{name}: {x} mean {c['first']['mean']:.4f} (spread {c['first']['spread']:.4f}), {y} mean "
              f"{c['second']['mean']:.4f} (spread {c['second']['spread']:.4f}); difference {c['difference']:+.4f} "
              f"against the larger spread {c['larger_spread']:.4f}: {'claimed' if c['claimed'] else 'not claimed'}")

    out = {"chosen": None, "fix2": fix2, "sharing": sharing,
           "best_per_seed": {f"{v}{s}": None if b is None else {"epoch": b["epoch"], "validation": b["validation"]}
                             for (v, s), b in sorted(best.items())}}
    if chosen is None:
        print("\nno seed of A or B passed: nothing new is served")
    else:
        v, s = chosen
        b = best[chosen]
        rid = run_id(v, s, b["epoch"])
        out["chosen"] = {"run_id": rid, "version": v, "seed": s, "epoch": b["epoch"], "score": score[chosen],
                         "thresholds": {d: b["validation"][d]["theta"] for d in DOMAINS}}
        print(f"\nserved: {rid} (version {v}, seed {s}, epoch {b['epoch']}), UCF-Crime within-window AUC "
              f"{score[chosen]:.4f}, θ {out['chosen']['thresholds']}")
        if register:
            source = drive / "models" / "pfm" / f"v5-{v.lower()}" / f"seed{s}" / "best.pt"
            _, state = train_pfm.load(source, "cpu")
            if state["epoch"] != b["epoch"]:
                raise SystemExit(f"{source} holds epoch {state['epoch']}, not {b['epoch']}")
            folder = drive / pfm_runs.RUNS / rid
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, folder / pfm_runs.CHECKPOINT)
            r = records[v]
            run = pfm_runs.record(
                drive, rid, config=r["config"], epochs_trained=len(r["history"][str(s)]), epoch=b["epoch"], seed=s,
                thresholds=out["chosen"]["thresholds"], validation=b["validation"], commit=r["commit"], gpu=r["gpu"],
                torch=r["torch"], encoder="internvideo2", false_alarm_rule=train_pfm.FALSE_ALARM,
                warning_rule="1 - p(Normal) >= theta (D-65)",
                samples_sha256={d: hashlib.sha256((ROOT / "datasets" / "samples" / f"{d}.csv").read_bytes()).hexdigest()
                                for d in DOMAINS},
                created_at=time.strftime("%Y-%m-%d %H:%M"))
            pfm_runs.register(drive, run, chosen=True)
            print(f"registered {rid} as the chosen run ({drive / pfm_runs.INDEX})")
    path = drive / "results" / "x22_choice.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
