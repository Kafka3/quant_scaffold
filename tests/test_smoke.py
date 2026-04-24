from pathlib import Path
import sys

import pandas as pd
import yaml

# Ensure the repository root is first in sys.path so our local packages are imported.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategy.signal_builder import build_signals
from backtest.vectorbt_engine import run_backtest
from data.loaders.csv_loader import load_ohlcv_csv


def _load_cfg() -> dict:
    """Load base config directly via yaml to avoid depending on configs.settings stability."""
    cfg_path = ROOT / "configs" / "base.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_load_config():
    """Config must contain strategy and backtest sections."""
    cfg = _load_cfg()
    assert "strategy" in cfg
    assert "backtest" in cfg


def test_build_signals_smoke():
    """Signal building must complete and index-align with input df."""
    df = load_ohlcv_csv(Path("example_data/sample_ohlcv.csv"))
    cfg = _load_cfg()
    bundle = build_signals(df, cfg["strategy"])

    assert bundle.entries_long.index.equals(df.index)
    assert bundle.entries_short.index.equals(df.index)
    assert bundle.long_entry_price.index.equals(df.index)
    assert bundle.short_entry_price.index.equals(df.index)


def test_run_backtest_smoke():
    """Backtest must run without error and return required fields, even with 0 trades."""
    df = load_ohlcv_csv(Path("example_data/sample_ohlcv.csv"))
    cfg = _load_cfg()
    bundle = build_signals(df, cfg["strategy"])
    result = run_backtest(df, bundle, cfg["backtest"])

    assert hasattr(result, "trades")
    assert hasattr(result, "equity")
    assert hasattr(result, "summary")

    assert "total_trades" in result.summary
    assert "total_return" in result.summary
    assert "win_rate" in result.summary

    assert isinstance(result.trades, pd.DataFrame)
    assert isinstance(result.equity, pd.Series)
    assert isinstance(result.summary, dict)
