from pathlib import Path
import sys

import pandas as pd
import yaml

# Ensure the repository root is first in sys.path so our local packages are imported.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tempfile
import os

from strategy.signal_builder import build_signals, SignalBundle
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


def test_divergence_signal_is_after_pivot_confirmation():
    """
    Pivot lookahead guard: divergence signal must NOT appear on the pivot2 bar itself.
    It must be marked on the confirmation bar (pivot2_pos + right_bars) or later.
    """
    df = load_ohlcv_csv(Path("example_data/sample_ohlcv.csv"))
    cfg = _load_cfg()

    from features.indicators import stochastic_d
    from features.divergence import detect_regular_divergence
    from features.trend_filter import build_trend_filter

    stoch_cfg = cfg["strategy"]["stochastic"]
    osc = stochastic_d(
        df,
        k_period=stoch_cfg["k_period"],
        d_period=stoch_cfg["d_period"],
        smooth=stoch_cfg["smooth"],
    )
    trend = build_trend_filter(df, cfg["strategy"]["trend"])
    div = detect_regular_divergence(df, osc, cfg["strategy"], trend)
    right_bars = cfg["strategy"]["pivots"]["right_bars"]

    # Check bullish signals
    bullish_signals = df.index[div.bullish]
    for signal_idx in bullish_signals:
        pivot2_idx = div.bullish_pivot2_idx.loc[signal_idx]
        assert pivot2_idx is not None and not pd.isna(pivot2_idx)
        pivot2_pos = df.index.get_loc(pivot2_idx)
        signal_pos = df.index.get_loc(signal_idx)
        assert signal_pos >= pivot2_pos + right_bars, (
            f"Bullish signal at {signal_idx} (pos {signal_pos}) appears before "
            f"pivot2 {pivot2_idx} (pos {pivot2_pos}) + right_bars {right_bars}"
        )

    # Check bearish signals
    bearish_signals = df.index[div.bearish]
    for signal_idx in bearish_signals:
        pivot2_idx = div.bearish_pivot2_idx.loc[signal_idx]
        assert pivot2_idx is not None and not pd.isna(pivot2_idx)
        pivot2_pos = df.index.get_loc(pivot2_idx)
        signal_pos = df.index.get_loc(signal_idx)
        assert signal_pos >= pivot2_pos + right_bars, (
            f"Bearish signal at {signal_idx} (pos {signal_pos}) appears before "
            f"pivot2 {pivot2_idx} (pos {pivot2_pos}) + right_bars {right_bars}"
        )


def test_end_of_data_liquidation_updates_equity():
    """
    If a position is still open at the last bar, end-of-data forced liquidation
    must update the final equity value to reflect realized cash.
    """
    idx = pd.date_range("2024-01-01 00:00", periods=4, freq="5min")
    df = pd.DataFrame({
        "Open":  [100, 101, 102, 103],
        "High":  [101, 102, 103, 104],
        "Low":   [99,  100, 101, 102],
        "Close": [100, 101, 102, 103],
    }, index=idx)

    entries_long = pd.Series(False, index=idx)
    entries_long.iloc[0] = True

    na_float = pd.Series(index=idx, dtype=float)
    na_obj = pd.Series(index=idx, dtype=object)

    bundle = SignalBundle(
        entries_long=entries_long,
        exits_long=pd.Series(False, index=idx),
        entries_short=pd.Series(False, index=idx),
        exits_short=pd.Series(False, index=idx),
        long_entry_price=pd.Series([100.0, pd.NA, pd.NA, pd.NA], index=idx),
        short_entry_price=na_float,
        long_stop_price=pd.Series([90.0, pd.NA, pd.NA, pd.NA], index=idx),
        short_stop_price=na_float,
        long_target_price=pd.Series([110.0, pd.NA, pd.NA, pd.NA], index=idx),
        short_target_price=na_float,
        long_setup_pivot2_time=pd.Series([idx[0], pd.NA, pd.NA, pd.NA], index=idx),
        short_setup_pivot2_time=na_obj,
        long_setup_confirm_time=pd.Series([idx[0], pd.NA, pd.NA, pd.NA], index=idx),
        short_setup_confirm_time=na_obj,
        long_trigger_price_raw=pd.Series([100.0, pd.NA, pd.NA, pd.NA], index=idx),
        short_trigger_price_raw=na_float,
        features=pd.DataFrame(index=idx),
        entry_prices_long=na_float,
        entry_prices_short=na_float,
        stop_prices_long=na_float,
        stop_prices_short=na_float,
        target_prices_long=na_float,
        target_prices_short=na_float,
    )

    config = {"initial_cash": 100000, "fee_per_trade": 0.0, "slippage": 0.0, "allow_short": True}
    result = run_backtest(df, bundle, config)

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "end_of_data"
    expected_pnl = 3.0  # exit 103 - entry 100
    assert abs(result.trades.iloc[0]["pnl"] - expected_pnl) < 1e-9
    assert abs(result.equity.iloc[-1] - (100000 + expected_pnl)) < 1e-9
    assert abs(result.equity.iloc[-1] - (100000 + result.trades["pnl"].sum())) < 1e-9


def test_load_numeric_timestamp_ms():
    """CSV with millisecond integer timestamps must be parsed correctly."""
    import time
    import csv

    base_ts = int(time.time() * 1000)
    rows = [
        ["timestamp", "Open", "High", "Low", "Close", "Volume"],
        [base_ts, 100, 101, 99, 100, 1000],
        [base_ts + 300000, 101, 102, 100, 101, 1100],
        [base_ts + 600000, 102, 103, 101, 102, 1200],
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        tmp_path = f.name

    try:
        df = load_ohlcv_csv(tmp_path)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == 3
        # 300000 ms = 5 min interval
        assert (df.index[1] - df.index[0]).total_seconds() == 300
    finally:
        os.unlink(tmp_path)


def test_trade_timing_is_after_confirmation():
    """
    Timing audit: every trade must satisfy
    setup_pivot2_time <= setup_confirm_time < entry_time.
    This guards against pivot lookahead leaking into live signals.
    """
    df = load_ohlcv_csv(Path("example_data/sample_ohlcv.csv"))
    cfg = _load_cfg()
    bundle = build_signals(df, cfg["strategy"])
    result = run_backtest(df, bundle, cfg["backtest"])

    if result.trades.empty:
        import pytest
        pytest.skip("No trades to audit timing")

    for _, row in result.trades.iterrows():
        pivot2 = row["setup_pivot2_time"]
        confirm = row["setup_confirm_time"]
        entry = row["entry_time"]

        assert pd.notna(pivot2), "setup_pivot2_time must not be empty"
        assert pd.notna(confirm), "setup_confirm_time must not be empty"
        assert confirm >= pivot2, (
            f"setup_confirm_time {confirm} is before setup_pivot2_time {pivot2}"
        )
        assert entry > confirm, (
            f"entry_time {entry} is not after setup_confirm_time {confirm}"
        )


def test_prior_trend_semantics_for_continuation():
    """
    prior_uptrend serves bullish continuation;
    prior_downtrend serves bearish continuation.
    """
    df = load_ohlcv_csv(Path("example_data/sample_ohlcv.csv"))
    cfg = _load_cfg()
    bundle = build_signals(df, cfg["strategy"])

    # If bullish setups exist, they must sit in a region where prior_uptrend is True.
    bullish_setups = bundle.features[bundle.features["bullish_setup_active"] == True]
    for _, row in bullish_setups.iterrows():
        assert row["prior_uptrend"] == 1, (
            "bullish setup must be in a prior_uptrend region"
        )

    # If bearish setups exist, they must sit in a region where prior_downtrend is True.
    bearish_setups = bundle.features[bundle.features["bearish_setup_active"] == True]
    for _, row in bearish_setups.iterrows():
        assert row["prior_downtrend"] == 1, (
            "bearish setup must be in a prior_downtrend region"
        )


def test_prior_trend_uses_shift_one():
    """
    prior_uptrend / prior_downtrend must be based on shift(1) so the current bar
    is never counted in its own trend lookback window.
    """
    df = load_ohlcv_csv(Path("example_data/sample_ohlcv.csv"))
    cfg = _load_cfg()

    from features.trend_filter import build_trend_filter

    trend = build_trend_filter(df, cfg["strategy"]["trend"])
    above_channel = df["Close"] > trend["ema_high"]
    below_channel = df["Close"] < trend["ema_low"]

    lookback = cfg["strategy"]["trend"]["lookback_bars"]
    above_ratio_manual = above_channel.shift(1).rolling(lookback).mean()
    below_ratio_manual = below_channel.shift(1).rolling(lookback).mean()

    # Verify our manual reconstruction matches the returned series.
    pd.testing.assert_series_equal(
        trend["above_ratio"].dropna(),
        above_ratio_manual.dropna(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        trend["below_ratio"].dropna(),
        below_ratio_manual.dropna(),
        check_names=False,
    )


def test_bullish_divergence_requires_prior_uptrend_and_progressive_channel():
    """
    Every bullish divergence signal must satisfy progressive pullback logic:
      - prior_uptrend condition is True at pivot1
      - pivot1 is an early pullback: Close < ema_high (inside_or_below_high)
      - pivot2 is a deep pullback: Close < ema_low (below_channel)
      - signal is marked on confirmation bar (>= pivot2_pos + right_bars)
    """
    df = load_ohlcv_csv(Path("example_data/sample_ohlcv.csv"))
    cfg = _load_cfg()

    from features.indicators import stochastic_d
    from features.divergence import detect_regular_divergence
    from features.trend_filter import build_trend_filter

    stoch_cfg = cfg["strategy"]["stochastic"]
    osc = stochastic_d(
        df,
        k_period=stoch_cfg["k_period"],
        d_period=stoch_cfg["d_period"],
        smooth=stoch_cfg["smooth"],
    )
    trend = build_trend_filter(df, cfg["strategy"]["trend"])
    div = detect_regular_divergence(df, osc, cfg["strategy"], trend)
    right_bars = cfg["strategy"]["pivots"]["right_bars"]

    bullish_signals = df.index[div.bullish]
    if len(bullish_signals) == 0:
        import pytest
        pytest.skip("No bullish divergence signals in sample data")

    for signal_idx in bullish_signals:
        assert div.bullish_prior_trend_ok.loc[signal_idx], (
            f"bullish signal at {signal_idx} lacks prior_uptrend"
        )
        assert div.bullish_channel_break_ok.loc[signal_idx], (
            f"bullish signal at {signal_idx} lacks channel break"
        )
        pivot1_idx = div.bullish_pivot1_idx.loc[signal_idx]
        pivot2_idx = div.bullish_pivot2_idx.loc[signal_idx]
        pivot2_pos = df.index.get_loc(pivot2_idx)
        signal_pos = df.index.get_loc(signal_idx)

        assert trend["inside_or_below_high"].loc[pivot1_idx], (
            f"bullish pivot1 {pivot1_idx} is not inside_or_below_high"
        )
        assert trend["below_channel"].loc[pivot2_idx], (
            f"bullish pivot2 {pivot2_idx} is not below_channel"
        )
        assert signal_pos >= pivot2_pos + right_bars, (
            f"bullish signal at {signal_idx} (pos {signal_pos}) before "
            f"pivot2 confirmation (pos {pivot2_pos + right_bars})"
        )


def test_bearish_divergence_requires_prior_downtrend_and_progressive_channel():
    """
    Every bearish divergence signal must satisfy progressive pullback logic:
      - prior_downtrend condition is True at pivot1
      - pivot1 is an early pullback: Close > ema_low (inside_or_above_low)
      - pivot2 is a deep pullback: Close > ema_high (above_channel)
      - signal is marked on confirmation bar (>= pivot2_pos + right_bars)
    """
    df = load_ohlcv_csv(Path("example_data/sample_ohlcv.csv"))
    cfg = _load_cfg()

    from features.indicators import stochastic_d
    from features.divergence import detect_regular_divergence
    from features.trend_filter import build_trend_filter

    stoch_cfg = cfg["strategy"]["stochastic"]
    osc = stochastic_d(
        df,
        k_period=stoch_cfg["k_period"],
        d_period=stoch_cfg["d_period"],
        smooth=stoch_cfg["smooth"],
    )
    trend = build_trend_filter(df, cfg["strategy"]["trend"])
    div = detect_regular_divergence(df, osc, cfg["strategy"], trend)
    right_bars = cfg["strategy"]["pivots"]["right_bars"]

    bearish_signals = df.index[div.bearish]
    if len(bearish_signals) == 0:
        import pytest
        pytest.skip("No bearish divergence signals in sample data")

    for signal_idx in bearish_signals:
        assert div.bearish_prior_trend_ok.loc[signal_idx], (
            f"bearish signal at {signal_idx} lacks prior_downtrend"
        )
        assert div.bearish_channel_break_ok.loc[signal_idx], (
            f"bearish signal at {signal_idx} lacks channel break"
        )
        pivot1_idx = div.bearish_pivot1_idx.loc[signal_idx]
        pivot2_idx = div.bearish_pivot2_idx.loc[signal_idx]
        pivot2_pos = df.index.get_loc(pivot2_idx)
        signal_pos = df.index.get_loc(signal_idx)

        assert trend["inside_or_above_low"].loc[pivot1_idx], (
            f"bearish pivot1 {pivot1_idx} is not inside_or_above_low"
        )
        assert trend["above_channel"].loc[pivot2_idx], (
            f"bearish pivot2 {pivot2_idx} is not above_channel"
        )
        assert signal_pos >= pivot2_pos + right_bars, (
            f"bearish signal at {signal_idx} (pos {signal_pos}) before "
            f"pivot2 confirmation (pos {pivot2_pos + right_bars})"
        )
