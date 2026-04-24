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
    cfg = load_settings(Path("configs/base.yaml"))
    assert "strategy" in cfg
    assert "backtest" in cfg


def test_build_signals_smoke():
    df = load_ohlcv_csv(Path("example_data/sample_ohlcv.csv"))
    cfg = load_settings(Path("configs/base.yaml"))
    bundle = build_signals(df, cfg["strategy"])

    assert bundle is not None
    assert bundle.entries_long.index.equals(df.index)
    assert bundle.entries_short.index.equals(df.index)
    assert bundle.exits_long.index.equals(df.index)
    assert bundle.exits_short.index.equals(df.index)
    assert bundle.features.index.equals(df.index)

    required_cols = [
        "Close",
        "osc",
        "atr",
        "ema_high",
        "ema_low",
        "trend_long",
        "trend_short",
        "pivot_high",
        "pivot_low",
        "bullish_div",
        "bearish_div",
        "bullish_trigger_price",
        "bearish_trigger_price",
        "bullish_stop_anchor",
        "bearish_stop_anchor",
        "long_entry_price",
        "short_entry_price",
        "long_stop_price",
        "short_stop_price",
        "long_target_price",
        "short_target_price",
    ]
    for col in required_cols:
        assert col in bundle.features.columns


def test_run_backtest_smoke():
    df = load_ohlcv_csv(Path("example_data/sample_ohlcv.csv"))
    cfg = load_settings(Path("configs/base.yaml"))
    bundle = build_signals(df, cfg["strategy"])
    result = run_backtest(df, bundle, cfg["backtest"])

    assert result is not None
    assert "total_trades" in result.summary
    assert "total_return" in result.summary
