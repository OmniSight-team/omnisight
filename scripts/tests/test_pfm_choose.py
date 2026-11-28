"""OMNI-X22: D-65's choice of the served model and the two comparisons."""

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("pfm_choose", Path(__file__).resolve().parents[2] / "scripts" / "pfm_choose.py")
pfm_choose = importlib.util.module_from_spec(_spec)
sys.modules["pfm_choose"] = pfm_choose
_spec.loader.exec_module(pfm_choose)


def test_the_highest_seed_of_a_or_b_is_served_never_c():
    best = {("A", 0): 0.70, ("A", 1): 0.80, ("B", 0): 0.75, ("C", 0): 0.95}
    assert pfm_choose.choose(best) == ("A", 1)


def test_a_tie_goes_to_the_lower_seed_then_to_a():
    assert pfm_choose.choose({("A", 2): 0.8, ("B", 1): 0.8}) == ("B", 1)
    assert pfm_choose.choose({("B", 1): 0.8, ("A", 1): 0.8}) == ("A", 1)


def test_a_failed_seed_is_skipped_and_none_passing_serves_nothing():
    assert pfm_choose.choose({("A", 0): None, ("B", 0): 0.6}) == ("B", 0)
    assert pfm_choose.choose({("A", 0): None, ("B", 0): None, ("C", 0): 0.9}) is None


def test_a_difference_is_claimed_only_above_the_larger_spread():
    c = pfm_choose.compare([0.70, 0.80], [0.85, 0.87])  # means 0.75, 0.86; spreads 0.10, 0.02
    assert abs(c["difference"] - 0.11) < 1e-9 and abs(c["larger_spread"] - 0.10) < 1e-9 and c["claimed"]
    c = pfm_choose.compare([0.70, 0.80], [0.79, 0.81])  # difference 0.05 < 0.10
    assert not c["claimed"]
