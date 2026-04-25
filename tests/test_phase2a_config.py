"""Lightweight config validation for Phase 2A frozen baseline."""

from pathlib import Path
import sys

# Resolve project root and inject so module imports work when running via pytest.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Ensure project root is at the very front to avoid shadowing by e.g.
# /Volumes/t7/backtest.py when pytest adds parent directories to sys.path.
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

from configs.settings import load_settings
from optimize.phase2a_search import run_phase2a


BASELINE_CONFIG = _PROJECT_ROOT / "configs" / "phase2a_baseline.yaml"


def test_baseline_config_exists():
    """Ensure the Phase 2A baseline config file exists."""
    assert BASELINE_CONFIG.exists(), f"Baseline config not found: {BASELINE_CONFIG}"


def test_baseline_config_loadable():
    """Ensure the config loads without error and contains required keys."""
    settings = load_settings(BASELINE_CONFIG)
    assert "strategy" in settings
    assert "backtest" in settings


def test_baseline_frozen_parameters():
    """Verify the exact frozen parameters from Phase 2A review."""
    settings = load_settings(BASELINE_CONFIG)
    strategy = settings["strategy"]

    # Trend filter
    assert strategy["trend"]["lookback_bars"] == 24
    assert strategy["trend"]["min_close_ratio"] == 0.80

    # Pivots
    assert strategy["pivots"]["left_bars"] == 4
    assert strategy["pivots"]["right_bars"] == 3

    # Risk
    assert strategy["risk"]["rr_target"] == 2.0

    # Stochastic
    assert strategy["stochastic"]["oversold"] == 20
    assert strategy["stochastic"]["overbought"] == 80


def test_run_phase2a_importable():
    """Ensure run_phase2a is importable (smoke-test the module init)."""
    assert callable(run_phase2a)
