"""OMNI-905: the PFM's `model_runs` record, and the download of the chosen checkpoint (D-59).

A trained model lives on Drive in its own folder, `MyDrive/omnisight/models/pfm/runs/{run_id}/`:
`model.pt` (the chosen epoch's weights and config) and `run.json`, the run's record: run id, config,
domains, epochs (trained and chosen), seed, the thresholds `θ_d`, the validation measures, the commit,
the samples' SHA-256 and the checkpoint's. `MyDrive/omnisight/models/pfm/model_runs.json` lists every
run's record and names the chosen one. Both are written by `notebooks/pfm_train_joint_gpu.ipynb`.

The product runs no training and Colab cannot reach its database, so the record is a file next to the
checkpoint, not a table. The serving side (OMNI-910) reads `run.json` for `model_run_id` and `θ_d`.

    uv run python scripts/pfm_runs.py list                 # every run on Drive, the chosen one marked
    uv run python scripts/pfm_runs.py download             # the chosen run -> models/trained/pfm/
    uv run python scripts/pfm_runs.py download <run_id>    # another run
    ... --drive "<path to MyDrive/omnisight>"              # default G:/My Drive/omnisight

`models/trained/` is never committed. The download checks the checkpoint's SHA-256 against its record.
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVE = Path("G:/My Drive/omnisight")
RUNS = Path("models") / "pfm" / "runs"
INDEX = Path("models") / "pfm" / "model_runs.json"
TARGET = ROOT / "models" / "trained" / "pfm"
CHECKPOINT, RECORD = "model.pt", "run.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def record(drive: Path, run_id: str, *, config: dict, epochs_trained: int, epoch: int, seed: int,
           thresholds: dict[str, float], validation: dict, **extra) -> dict:
    """Write `run.json` beside the run's `model.pt` (which must exist) and return it."""
    folder = drive / RUNS / run_id
    checkpoint = folder / CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    run = {"run_id": run_id, "domains": config["domains"], "config": config, "epochs_trained": epochs_trained,
           "epoch": epoch, "seed": seed, "thresholds": thresholds, "validation": validation,
           "path": (RUNS / run_id / CHECKPOINT).as_posix(), "checkpoint_sha256": sha256(checkpoint), **extra}
    _write_json(folder / RECORD, run)
    return run


def register(drive: Path, run: dict, chosen: bool) -> dict:
    """Add (or replace) the run in `model_runs.json`; `chosen` makes it the one to download."""
    path = drive / INDEX
    index = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"chosen": None, "runs": []}
    index["runs"] = [r for r in index["runs"] if r["run_id"] != run["run_id"]] + [run]
    if chosen:
        index["chosen"] = run["run_id"]
    _write_json(path, index)
    return index


def download(drive: Path, run_id: str | None = None, target: Path = TARGET) -> dict:
    """Copy a run's `model.pt` and `run.json` into `target` (replacing what is there), checked by SHA-256."""
    index = json.loads((drive / INDEX).read_text(encoding="utf-8"))
    run_id = run_id or index["chosen"]
    run = next((r for r in index["runs"] if r["run_id"] == run_id), None)
    if run is None:
        raise SystemExit(f"no run {run_id!r} in {drive / INDEX}")
    target.mkdir(parents=True, exist_ok=True)
    tmp = target / (CHECKPOINT + ".part")
    shutil.copyfile(drive / run["path"], tmp)
    if sha256(tmp) != run["checkpoint_sha256"]:
        tmp.unlink()
        raise SystemExit(f"{run['path']}: SHA-256 differs from its record; nothing replaced")
    os.replace(tmp, target / CHECKPOINT)
    _write_json(target / RECORD, run)
    return run


def main() -> int:
    args = sys.argv[1:]
    drive = DRIVE
    if "--drive" in args:
        i = args.index("--drive")
        drive, args = Path(args[i + 1]), args[:i] + args[i + 2 :]
    if args[:1] == ["list"]:
        index = json.loads((drive / INDEX).read_text(encoding="utf-8"))
        for r in index["runs"]:
            mark = "*" if r["run_id"] == index["chosen"] else " "
            print(f"{mark} {r['run_id']}  epoch {r['epoch']}/{r['epochs_trained']}  seed {r['seed']}  "
                  f"θ {r['thresholds']}  score {r['validation'].get('score')}")
    elif args[:1] == ["download"]:
        run = download(drive, args[1] if len(args) > 1 else None)
        print(f"{run['run_id']} -> {TARGET} (SHA-256 checked)")
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
