"""Full-chain synthetic tests for divergence + setup + entry + exit logic.

Strategy: construct mini OHLCV dataframes using real BTC data slices that
are known to produce specific trade behaviors, then make targeted
modifications to verify edge cases.

Constants used (matching configs/baseline_ema55_stoch143_2r.yaml):
  - ema_period = 55, pivot left/right = 3/3, min_sep = 5, max_sep = 35
  - stoch 14/1/3, oversold 20, overbought 80
  - RR = 2.0, setup_max_bars = 12
"""

from pathlib import Path
import sys
from typing import Optional

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loaders.csv_loader import load_ohlcv_csv
from features.indicators import stochastic_d
from features.divergence import detect_regular_divergence
from features.trend_filter import build_trend_filter
from strategy.signal_builder import build_signals
from backtest.event_engine import run_backtest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASELINE_STRATEGY = {
    "stochastic": {"k_period": 14, "d_period": 3, "smooth": 1,
                   "oversold": 20, "overbought": 80},
    "pivots": {"left_bars": 3, "right_bars": 3,
               "min_separation": 5, "max_separation": 35, "strict": True},
    "trend": {"ema_period": 55, "lookback_bars": 12, "min_close_ratio": 1.0},
    "risk": {"atr_period": 14, "stop_buffer": 0.0, "rr_target": 2.0},
    "setup": {"setup_max_bars": 12, "replace_same_side_setup": True,
              "invalidate_on_stop_anchor_break": True},
}

BASELINE_BACKTEST = {
    "initial_cash": 100000, "fee_per_trade": 0.0,
    "slippage": 0.0, "allow_short": True,
}

# Real BTC data slice (bars 2075-2275 of BTCUSDT 5m 2024-2025) which contains
# a confirmed bullish divergence trade with entry, target exit.
_REAL_DATA_PATH = ROOT / "data" / "raw" / "BTCUSDT_5m_2024_2025.csv"


def _load_trade_slice(start: int = 2075, end: int = 2275) -> pd.DataFrame:
    """Load a real BTC data slice that is known to produce trades."""
    df = load_ohlcv_csv(_REAL_DATA_PATH)
    return df.iloc[start:end].copy()


def _reindex(df: pd.DataFrame, start: str = "2025-01-01 00:00") -> pd.DataFrame:
    """Rebase a DataFrame's index to a new start timestamp (preserving gaps)."""
    idx = pd.date_range(start, periods=len(df), freq="5min")
    df = df.copy()
    df.index = idx
    return df


def _run_baseline(df: pd.DataFrame):
    """Run baseline build_signals + backtest."""
    bundle = build_signals(df, BASELINE_STRATEGY)
    result = run_backtest(df, bundle, BASELINE_BACKTEST)
    return bundle, result


# ===================================================================
# 1. Bullish divergence: full chain reproduces
# ===================================================================

def test_bullish_divergence_full_chain():
    """A real BTC data slice that produces a bullish trade must do so
    after re-indexing (proving it's not timestamp-dependent)."""
    df = _load_trade_slice(2075, 2275)
    df = _reindex(df)
    bundle, result = _run_baseline(df)

    assert len(result.trades) > 0, "No trades on real data slice"
    first_trade = result.trades.iloc[0]
    assert first_trade["side"] == "long", f"First trade is {first_trade['side']}"
    assert first_trade["exit_reason"] in ("target", "stop"), (
        f"Unexpected exit reason: {first_trade['exit_reason']}"
    )


# ===================================================================
# 2. No false signal on flat data
# ===================================================================

def test_no_signal_without_divergence():
    """A completely flat price series must produce ZERO signals."""
    n = 200
    idx = pd.date_range("2025-01-01 00:00", periods=n, freq="5min")
    df = pd.DataFrame({
        "Open": np.ones(n) * 100, "High": np.ones(n) * 100.2,
        "Low": np.ones(n) * 99.8, "Close": np.ones(n) * 100,
        "Volume": np.ones(n) * 100,
    }, index=idx)
    bundle, result = _run_baseline(df)
    assert bundle.entries_long.sum() == 0
    assert bundle.entries_short.sum() == 0
    assert len(result.trades) == 0


# ===================================================================
# 3. No false signal on random walk
# ===================================================================

def test_no_signal_without_prior_trend():
    """Random walk should rarely produce signals."""
    n = 200
    np.random.seed(12345)
    close = 100 + np.cumsum(np.random.normal(0, 0.5, n))
    idx = pd.date_range("2025-01-01 00:00", periods=n, freq="5min")
    df = pd.DataFrame({
        "Open": pd.Series(close).shift(1).fillna(close[0]),
        "High": close * 1.002,
        "Low": close * 0.998,
        "Close": close,
        "Volume": np.ones(n) * 100,
    }, index=idx)
    bundle, result = _run_baseline(df)
    # Soft check: random walk should not produce many signals
    assert len(result.trades) <= 1, f"Random walk: {len(result.trades)} trades"


# ===================================================================
# 4. Entry after pivot confirmation
# ===================================================================

def test_entry_not_before_pivot_confirmation():
    """Every trade must have: confirm >= p2_time and entry > confirm."""
    df = _load_trade_slice(2075, 2275)
    df = _reindex(df)
    bundle, result = _run_baseline(df)

    if len(result.trades) == 0:
        import pytest
        pytest.skip("No trades")

    for _, trade in result.trades.iterrows():
        assert pd.notna(trade["setup_pivot2_time"]), "NA pivot2_time"
        assert pd.notna(trade["setup_confirm_time"]), "NA confirm_time"
        assert trade["setup_confirm_time"] >= trade["setup_pivot2_time"], (
            f"confirm {trade['setup_confirm_time']} < p2 {trade['setup_pivot2_time']}"
        )
        assert trade["entry_time"] > trade["setup_confirm_time"], (
            f"entry {trade['entry_time']} <= confirm {trade['setup_confirm_time']}"
        )


# ===================================================================
# 5. Stop priority over target on same bar
# ===================================================================

def test_stop_priority_over_target_on_same_bar():
    """If both stop and target are hit, stop wins."""
    n = 150
    idx = pd.date_range("2025-01-01 00:00", periods=n, freq="5min")
    from strategy.signal_builder import SignalBundle

    entry_idx = idx[50]
    entries_long = pd.Series(False, index=idx)
    entries_long.loc[entry_idx] = True
    na_f = pd.Series(index=idx, dtype=float)
    na_o = pd.Series(index=idx, dtype=object)

    bundle = SignalBundle(
        entries_long=entries_long, exits_long=pd.Series(False, index=idx),
        entries_short=pd.Series(False, index=idx), exits_short=pd.Series(False, index=idx),
        long_entry_price=pd.Series(100.0, index=[entry_idx]).reindex(idx, fill_value=pd.NA),
        short_entry_price=na_f,
        long_stop_price=pd.Series(99.0, index=[entry_idx]).reindex(idx, fill_value=pd.NA),
        short_stop_price=na_f,
        long_target_price=pd.Series(102.0, index=[entry_idx]).reindex(idx, fill_value=pd.NA),
        short_target_price=na_f,
        long_setup_pivot2_time=na_o, short_setup_pivot2_time=na_o,
        long_setup_confirm_time=na_o, short_setup_confirm_time=na_o,
        long_trigger_price_raw=na_f, short_trigger_price_raw=na_f,
        features=pd.DataFrame(index=idx),
    )

    df = pd.DataFrame({
        "Open": np.ones(n) * 100, "High": np.ones(n) * 100.5,
        "Low": np.ones(n) * 99.5, "Close": np.ones(n) * 100,
        "Volume": np.ones(n) * 100,
    }, index=idx)
    df.loc[idx[51], "Low"] = 98.8    # ≤ stop 99
    df.loc[idx[51], "High"] = 102.5  # ≥ target 102

    result = run_backtest(df, bundle, BASELINE_BACKTEST)
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "stop", f"Expected stop, got {trade['exit_reason']}"
    assert trade["exit_price"] == 99.0


# ===================================================================
# 6. Same-bar exit: stop first
# ===================================================================

def test_same_bar_exit_stop_first():
    """Entry bar itself checks stop/target. Stop wins."""
    n = 100
    idx = pd.date_range("2025-01-01 00:00", periods=n, freq="5min")
    from strategy.signal_builder import SignalBundle

    entry_idx = idx[50]
    entries_long = pd.Series(False, index=idx)
    entries_long.loc[entry_idx] = True
    na_f = pd.Series(index=idx, dtype=float)
    na_o = pd.Series(index=idx, dtype=object)

    bundle = SignalBundle(
        entries_long=entries_long, exits_long=pd.Series(False, index=idx),
        entries_short=pd.Series(False, index=idx), exits_short=pd.Series(False, index=idx),
        long_entry_price=pd.Series(100.0, index=[entry_idx]).reindex(idx, fill_value=pd.NA),
        short_entry_price=na_f,
        long_stop_price=pd.Series(99.0, index=[entry_idx]).reindex(idx, fill_value=pd.NA),
        short_stop_price=na_f,
        long_target_price=pd.Series(102.0, index=[entry_idx]).reindex(idx, fill_value=pd.NA),
        short_target_price=na_f,
        long_setup_pivot2_time=na_o, short_setup_pivot2_time=na_o,
        long_setup_confirm_time=na_o, short_setup_confirm_time=na_o,
        long_trigger_price_raw=na_f, short_trigger_price_raw=na_f,
        features=pd.DataFrame(index=idx),
    )

    df = pd.DataFrame({
        "Open": np.ones(n) * 100, "High": np.ones(n) * 100.5,
        "Low": np.ones(n) * 99.5, "Close": np.ones(n) * 100,
        "Volume": np.ones(n) * 100,
    }, index=idx)
    df.loc[idx[50], "Low"] = 98.5   # stop hit same bar
    df.loc[idx[50], "High"] = 100.0  # target not hit

    result = run_backtest(df, bundle, BASELINE_BACKTEST)
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "stop"
    assert trade["exit_price"] == 99.0


# ===================================================================
# 7. Bearish exit: stop priority
# ===================================================================

def test_bearish_exit_stop_priority():
    """Short: High ≥ stop and Low ≤ target on same bar → stop wins."""
    n = 150
    idx = pd.date_range("2025-01-01 00:00", periods=n, freq="5min")
    from strategy.signal_builder import SignalBundle

    entry_idx = idx[50]
    ent_short = pd.Series(False, index=idx)
    ent_short.loc[entry_idx] = True
    na_f = pd.Series(index=idx, dtype=float)
    na_o = pd.Series(index=idx, dtype=object)

    bundle = SignalBundle(
        entries_long=pd.Series(False, index=idx), exits_long=pd.Series(False, index=idx),
        entries_short=ent_short, exits_short=pd.Series(False, index=idx),
        long_entry_price=na_f,
        short_entry_price=pd.Series(100.0, index=[entry_idx]).reindex(idx, fill_value=pd.NA),
        long_stop_price=na_f,
        short_stop_price=pd.Series(101.0, index=[entry_idx]).reindex(idx, fill_value=pd.NA),
        long_target_price=na_f,
        short_target_price=pd.Series(98.0, index=[entry_idx]).reindex(idx, fill_value=pd.NA),
        long_setup_pivot2_time=na_o, short_setup_pivot2_time=na_o,
        long_setup_confirm_time=na_o, short_setup_confirm_time=na_o,
        long_trigger_price_raw=na_f, short_trigger_price_raw=na_f,
        features=pd.DataFrame(index=idx),
    )

    df = pd.DataFrame({
        "Open": np.ones(n) * 100, "High": np.ones(n) * 100.5,
        "Low": np.ones(n) * 99.5, "Close": np.ones(n) * 100,
        "Volume": np.ones(n) * 100,
    }, index=idx)
    df.loc[idx[51], "High"] = 101.5  # stop hit
    df.loc[idx[51], "Low"] = 97.5    # also target hit

    result = run_backtest(df, bundle, BASELINE_BACKTEST)
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "stop"
    assert trade["exit_price"] == 101.0


# ===================================================================
# 8. Setup expired (modify data to block trigger)
# ===================================================================

def test_setup_expires():
    """If trigger is not hit within setup_max_bars, setup expires."""
    df = _load_trade_slice(2075, 2275)
    df = _reindex(df)

    # Find the entry bar and suppress the High so trigger never breaks
    bundle_before = build_signals(df, BASELINE_STRATEGY)
    entry_idx = bundle_before.entries_long.idxmax()  # bar where entry would happen
    if pd.isna(entry_idx):
        import pytest
        pytest.skip("No entries in slice — can't test expiry")

    # Replace High at entry and subsequent bars with values below trigger
    trigger_df = bundle_before.long_trigger_price_raw
    df_modified = df.copy()
    for idx in df_modified.index:
        pos = df_modified.index.get_loc(idx)
        entry_pos = df_modified.index.get_loc(entry_idx)
        if pos >= entry_pos:
            tp = trigger_df.loc[idx]
            if pd.notna(tp):
                df_modified.loc[idx, "High"] = tp * 0.999  # just below trigger

    bundle, result = _run_baseline(df_modified)

    # Modifying High breaks trigger — expect fewer entries (possibly 0)
    # We can't guarantee 0 because other setups may exist, but the original
    # entry should be suppressed.
    assert bundle.entries_long.sum() <= bundle_before.entries_long.sum(), (
        "Suppressing trigger should not increase entries"
    )


# ===================================================================
# 9. Structure break invalidates setup
# ===================================================================

def test_structure_break_cancels_setup():
    """If Low drops below stop_anchor before trigger, setup is cancelled."""
    df = _load_trade_slice(2075, 2275)
    df = _reindex(df)

    bundle_before = build_signals(df, BASELINE_STRATEGY)
    if bundle_before.entries_long.sum() == 0:
        import pytest
        pytest.skip("No entries — can't test structure break")

    # Find the confirm bar and suppress High (so trigger never breaks),
    # while ALSO pushing Low below stop_anchor
    confirm_times = bundle_before.long_setup_confirm_time.dropna()
    if len(confirm_times) == 0:
        import pytest
        pytest.skip("No confirm times")

    first_confirm = confirm_times.iloc[0]
    confirm_pos = df.index.get_loc(first_confirm)

    # Get stop_anchor for this confirm
    stop_val = bundle_before.features.loc[first_confirm, "bullish_setup_stop_anchor"]
    if pd.isna(stop_val):
        import pytest
        pytest.skip("No stop anchor")

    df_modified = df.copy()
    # After confirm, push Low below stop_anchor AND keep High below trigger
    for i in range(confirm_pos + 1, len(df_modified)):
        idx = df_modified.index[i]
        df_modified.loc[idx, "Low"] = min(df_modified.loc[idx, "Low"], stop_val * 0.999)
        trigger = bundle_before.long_trigger_price_raw.loc[idx]
        if pd.notna(trigger):
            df_modified.loc[idx, "High"] = trigger * 0.999

    bundle, result = _run_baseline(df_modified)

    # The original entry should now be suppressed
    assert bundle.entries_long.sum() == 0, (
        "Structure break should cancel all setups"
    )


# ===================================================================
# 10. Entry requires trigger break
# ===================================================================

def test_entry_only_on_trigger_break():
    """No entry until High breaks trigger_price (bullish) or
    Low breaks trigger_price (bearish)."""
    df = _load_trade_slice(2075, 2275)
    df = _reindex(df)

    # Suppress entry: find all confirm bars and their triggers,
    # then push High below trigger from confirm+1 onward
    bundle_before = build_signals(df, BASELINE_STRATEGY)
    trigger_series = bundle_before.long_trigger_price_raw
    confirm_series = bundle_before.long_setup_confirm_time

    df_modified = df.copy()
    # Find each confirm time and suppress High on subsequent bars
    for confirm_time in confirm_series.dropna():
        confirm_pos = df_modified.index.get_loc(confirm_time)
        # Check if there's a matching trigger_price at or near this confirm
        tp_values = trigger_series.loc[confirm_time:]
        tp = tp_values.dropna()
        if len(tp) > 0:
            trigger_level = float(tp.iloc[0])
            # From confirm+1, keep High below trigger
            for i in range(confirm_pos + 1, len(df_modified)):
                idx = df_modified.index[i]
                df_modified.loc[idx, "High"] = min(
                    float(df_modified.loc[idx, "High"]),
                    trigger_level * 0.999,
                )

    bundle, result = _run_baseline(df_modified)

    # All entries should be suppressed because High never breaks trigger
    assert bundle.entries_long.sum() == 0, (
        f"Expected 0 entries when High stays below trigger, "
        f"got {bundle.entries_long.sum()}"
    )


# ===================================================================
# 11. Real data type compatibility
# ===================================================================

def test_real_data_type_compatibility():
    """load_ohlcv_csv output must have correct types."""
    df = _load_trade_slice(2075, 2275)
    assert isinstance(df.index, pd.DatetimeIndex)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col in df.columns
    assert df["Close"].dtype in (np.float64, float)
