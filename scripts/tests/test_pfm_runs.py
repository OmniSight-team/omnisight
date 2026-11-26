"""OMNI-905: the PFM's run record on Drive and the download of the chosen checkpoint."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("pfm_runs", ROOT / "scripts" / "pfm_runs.py")
runs = importlib.util.module_from_spec(spec)
sys.modules["pfm_runs"] = runs
spec.loader.exec_module(runs)


def _run(drive: Path, run_id: str, content: bytes) -> dict:
    folder = drive / runs.RUNS / run_id
    folder.mkdir(parents=True)
    (folder / runs.CHECKPOINT).write_bytes(content)
    return runs.record(drive, run_id, config={"domains": ["ccd", "ucf_crime"], "epochs": 30}, epochs_trained=30,
                       epoch=17, seed=1, thresholds={"ccd": 0.5, "ucf_crime": 0.6}, validation={"score": 0.8})


def test_a_run_is_recorded_beside_its_checkpoint_and_the_chosen_one_is_downloaded(tmp_path):
    drive, target = tmp_path / "drive", tmp_path / "models" / "trained" / "pfm"
    first = _run(drive, "joint-s0", b"weights 0")
    runs.register(drive, first, chosen=True)
    second = _run(drive, "joint-s1", b"weights 1")
    index = runs.register(drive, second, chosen=False)

    assert index["chosen"] == "joint-s0" and [r["run_id"] for r in index["runs"]] == ["joint-s0", "joint-s1"]
    assert json.loads((drive / runs.RUNS / "joint-s0" / runs.RECORD).read_text())["domains"] == ["ccd", "ucf_crime"]
    assert first["path"] == "models/pfm/runs/joint-s0/model.pt" and first["epoch"] == 17

    assert runs.download(drive, target=target)["run_id"] == "joint-s0"
    assert (target / runs.CHECKPOINT).read_bytes() == b"weights 0"
    assert json.loads((target / runs.RECORD).read_text())["thresholds"] == {"ccd": 0.5, "ucf_crime": 0.6}
    runs.download(drive, "joint-s1", target=target)
    assert (target / runs.CHECKPOINT).read_bytes() == b"weights 1"


def test_a_checkpoint_that_differs_from_its_record_replaces_nothing(tmp_path):
    drive, target = tmp_path / "drive", tmp_path / "trained"
    runs.register(drive, _run(drive, "joint-s0", b"weights 0"), chosen=True)
    runs.download(drive, target=target)
    (drive / runs.RUNS / "joint-s0" / runs.CHECKPOINT).write_bytes(b"changed")

    with pytest.raises(SystemExit, match="SHA-256"):
        runs.download(drive, target=target)
    assert (target / runs.CHECKPOINT).read_bytes() == b"weights 0" and not (target / "model.pt.part").exists()
