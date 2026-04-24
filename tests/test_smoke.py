from pathlib import Path
import sys

import pandas as pd

# Ensure the repository root is first in sys.path so our local backtest package is imported.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.settings import load_settings
from strategy.signal_builder import build_signals
from backtest.vectorbt_engine import run_backtest
from data.loaders.csv_loader import load_ohlcv_csv


def test_load_config():
    """Test that configuration can be loaded successfully."""
    cfg = load_settings(Path("configs/base.yaml"))
    assert "strategy" in cfg
    assert "backtest" in cfg


def test_build_signals_smoke():
    """Test that signal building completes end-to-end without errors."""
    df = load_ohlcv_csv(Path("example_data/sample_ohlcv.csv"))
    cfg = load_settings(Path("configs/base.yaml"))
    bundle = build_signals(df, cfg["strategy"])

    # Check return object is not None
    assert bundle is not None
    
    # Check index consistency with entries_long and entries_short
    assert bundle.entries_long.index.equals(df.index)
    assert bundle.entries_short.index.equals(df.index)
    
    # Check index consistency with entry prices
    assert bundle.long_entry_price.index.equals(df.index)
    assert bundle.short_entry_price.index.equals(df.index)


def test_run_backtest_smoke():
    """Test that backtest runs end-to-end without errors and returns required fields."""
    df = load_ohlcv_csv(Path("example_data/sample_ohlcv.csv"))
    cfg = load_settings(Path("configs/base.yaml"))
    bundle = build_signals(df, cfg["strategy"])
    result = run_backtest(df, bundle, cfg["backtest"])

    # Check return object contains required attributes
    assert result is not None
    assert hasattr(result, "trades")
    assert hasattr(result, "equity")
    assert hasattr(result, "summary")
    
    # Check summary contains required fields
    assert "total_trades" in result.summary
    assert "total_return" in result.summary
    assert "win_rate" in result.summary
    
    # Verify types are correct
    assert isinstance(result.trades, pd.DataFrame)
    assert isinstance(result.equity, pd.Series)
    assert isinstance(result.summary, dict)
