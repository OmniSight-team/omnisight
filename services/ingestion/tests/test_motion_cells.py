"""OMNI-908: Algorithm 1 records which of its cells moved, without changing what it decides.

pipeline.py is loaded by file path, as in test_motion_gate.py.
"""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "omnisight_ingestion_pipeline_cells", Path(__file__).resolve().parents[1] / "app" / "pipeline.py"
)
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)

G = 8


def _gate(**overrides):
    settings = {"grid_size": G, "cell_threshold": 10.0, "min_active_cells": 2, "stride_s": 0.5, "blur_kernel": 3}
    return pipeline.MotionGate(**{**settings, **overrides})


def _frame(bright: tuple[int, int, int, int] | None = None, size: int = 160) -> np.ndarray:
    """Grey, with an optional white block at (row0, col0, row1, col1) in pixels."""
    frame = np.full((size, size, 3), 60, np.uint8)
    if bright:
        r0, c0, r1, c1 = bright
        frame[r0:r1, c0:c1] = 255
    return frame


def _decide(gate, frames: list[np.ndarray]):
    """Each frame Δ apart; the last decision."""
    decision = None
    for i, frame in enumerate(frames):
        decision = gate.evaluate(frame, i * 0.5)
    return decision


def _cells(mask: int) -> set[tuple[int, int]]:
    mask &= (1 << 64) - 1
    return {divmod(i, G) for i in range(G * G) if mask >> i & 1}


def test_the_mask_names_the_cells_that_moved_and_counts_like_the_score():
    # A block appears over cells rows 1-2, columns 5-6 (each cell is 20 x 20 px).
    still = _frame()
    moved = _frame((20, 100, 60, 140))

    decision = _decide(_gate(), [still, still, moved])

    assert decision.admitted and decision.admission_reason == "motion"
    assert decision.motion_grid == G
    assert _cells(decision.motion_cells) == {(1, 5), (1, 6), (2, 5), (2, 6)}
    assert len(_cells(decision.motion_cells)) == decision.motion_score


def test_a_discarded_frame_keeps_its_mask_and_warmup_has_none():
    gate = _gate(min_active_cells=10)
    warm = gate.evaluate(_frame(), 0.0)
    assert (warm.motion_cells, warm.motion_grid) == (None, None)

    decision = _decide(gate, [_frame(), _frame(), _frame((0, 0, 20, 20))])

    assert not decision.admitted
    assert _cells(decision.motion_cells) == {(0, 0)} and decision.motion_score == 1


def test_the_bottom_right_cell_fits_a_signed_bigint():
    decision = _decide(_gate(min_active_cells=1), [_frame(), _frame(), _frame((140, 140, 160, 160))])

    assert decision.motion_cells < 0  # bit 63 set: stored as a negative BIGINT
    assert -(1 << 63) <= decision.motion_cells < 1 << 63
    assert _cells(decision.motion_cells) == {(7, 7)}


def test_a_grid_over_64_cells_records_no_mask():
    assert pipeline.cells_mask(np.ones((9, 9), bool)) is None
    assert pipeline.cells_mask(np.zeros((8, 8), bool)) == 0


def test_a_heartbeat_carries_the_inner_gates_mask():
    inner = _gate(min_active_cells=10)
    gate = pipeline.HeartbeatGate(inner, heartbeat_interval_s=1.0, stride_s=0.5)

    decision = _decide(gate, [_frame(), _frame(), _frame((0, 0, 20, 20))])

    assert decision.admission_reason == "heartbeat"
    assert _cells(decision.motion_cells) == {(0, 0)} and decision.motion_grid == G
