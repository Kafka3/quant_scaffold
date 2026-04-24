from pathlib import Path
import pandas as pd
from configs.settings import load_settings
from strategy.signal_builder import build_signals
from data.loaders.csv_loader import load_ohlcv_csv


def test_can_load_settings():
    cfg = load_settings(Path("configs/base.yaml"))
    assert "strategy" in cfg
    assert "backtest" in cfg


def test_build_signals_returns_valid_bundle():
    cfg = load_settings(Path("configs/base.yaml"))
    df = load_ohlcv_csv(cfg["data"]["path"])
    bundle = build_signals(df, cfg["strategy"])
    
    # Check types
    assert isinstance(bundle.entries_long, pd.Series)
    assert isinstance(bundle.entries_short, pd.Series)
    assert isinstance(bundle.exits_long, pd.Series)
    assert isinstance(bundle.exits_short, pd.Series)
    assert isinstance(bundle.features, pd.DataFrame)
    
    # Check indices match
    assert bundle.entries_long.index.equals(df.index)
    assert bundle.entries_short.index.equals(df.index)
    assert bundle.exits_long.index.equals(df.index)
    assert bundle.exits_short.index.equals(df.index)
    assert bundle.features.index.equals(df.index)
    
    # Check features columns
    required_cols = ["close", "osc", "pivot_high", "pivot_low", "bullish_div", "bearish_div", "ema_high", "ema_low", "trend_long", "trend_short"]
    for col in required_cols:
        assert col in bundle.features.columns
