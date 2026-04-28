"""Phase 6A — Feature engineering for signal-level dataset.

Extracts features at each rule-strategy signal point (entry timestamp)
from bundle.features DataFrame.

All features are computed from:
  - df:         Raw OHLCV data (for volatility/temporal features not in bundle)
  - bundle:     SignalBundle with entries_long/short, prices, and features DataFrame
  - df_stoch:   Stochastic oscillator values
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategy.signal_builder import SignalBundle


# Column names in bundle.features that we use
_F_CLOSE = "Close"
_F_OSC = "osc"
_F_ATR = "atr"
_F_EMA_HIGH = "ema_high"
_F_EMA_LOW = "ema_low"
_F_ABOVE_CHANNEL = "above_channel"
_F_BELOW_CHANNEL = "below_channel"
_F_ABOVE_RATIO = "above_ratio"
_F_BELOW_RATIO = "below_ratio"


def extract_features(
    df: pd.DataFrame,
    bundle: SignalBundle,
    entry_timestamps: List[pd.Timestamp],
    action_name: Optional[str] = None,
    trade_history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Extract feature vectors at each signal entry timestamp.

    Args:
        df:                Full OHLCV DataFrame for the period.
        bundle:            SignalBundle built with the action's config.
        entry_timestamps:  Timestamps where the rule strategy has a signal.
        action_name:       Optional label (used when appending action outcomes).
        trade_history:     DataFrame of past trades (for recent-performance features).

    Returns:
        DataFrame with one row per entry_timestamp.
    """
    rows: List[Dict] = []
    features_df = bundle.features

    # Pre-compute recent trade stats if available
    trade_stats = _build_trade_stats(trade_history) if trade_history is not None else {}

    for ts in entry_timestamps:
        if ts not in df.index or ts not in features_df.index:
            continue

        row: Dict = {"timestamp": ts}

        # Determine signal side
        side = _signal_side(bundle, ts)
        if side is None:
            continue

        row["side"] = side
        row["entry_price"] = _entry_price(bundle, ts, side)
        row["stop_price"] = _stop_price(bundle, ts, side)
        row["target_price"] = _target_price(bundle, ts, side)

        if row["entry_price"] is not None and row["stop_price"] is not None and row["entry_price"] > 0:
            stop_dist = abs(row["entry_price"] - row["stop_price"])
            row["stop_distance_pct"] = stop_dist / row["entry_price"]
            row["required_leverage"] = 0.01 / row["stop_distance_pct"] if row["stop_distance_pct"] > 0 else 999.0
        else:
            row["stop_distance_pct"] = None
            row["required_leverage"] = None

        # --- Features from bundle.features ---
        f = features_df.loc[ts]

        # Price / trend
        row["ema_slope"] = _ema_slope(features_df, ts, _F_EMA_HIGH, 12)

        # Price position vs channel
        ema_h = float(f[_F_EMA_HIGH]) if pd.notna(f[_F_EMA_HIGH]) else None
        ema_l = float(f[_F_EMA_LOW]) if pd.notna(f[_F_EMA_LOW]) else None
        close = float(df.loc[ts, "Close"])
        if ema_h is not None and ema_l is not None and (ema_h - ema_l) > 0:
            row["price_position_vs_channel"] = (close - ema_l) / (ema_h - ema_l)
            row["channel_width"] = (ema_h - ema_l) / close
        else:
            row["price_position_vs_channel"] = 0.5
            row["channel_width"] = 0.0

        # Above/below ratio (prior trend signal quality)
        row["above_ratio"] = float(f[_F_ABOVE_RATIO]) if pd.notna(f[_F_ABOVE_RATIO]) else 0.5
        row["below_ratio"] = float(f[_F_BELOW_RATIO]) if pd.notna(f[_F_BELOW_RATIO]) else 0.5

        # Stochastic
        osc_val = float(f[_F_OSC]) if pd.notna(f[_F_OSC]) else 50.0
        row["stoch_d"] = osc_val
        row["stoch_slope"] = _stoch_slope(features_df, ts, _F_OSC, 3)

        # Volatility
        atr_val = float(f[_F_ATR]) if pd.notna(f[_F_ATR]) else 0.0
        row["atr_14"] = atr_val
        row["atr_pct"] = atr_val / close if close > 0 else 0.0

        # Volume z-score
        row["volume_zscore"] = _volume_zscore(df, ts, 20)

        # Temporal
        row["hour_of_day"] = ts.hour
        row["day_of_week"] = ts.dayofweek

        # Recent trade performance
        row["recent_5_trade_avg_r"] = trade_stats.get("recent_5_avg_r", 0.0)
        row["recent_10_trade_avg_r"] = trade_stats.get("recent_10_avg_r", 0.0)
        row["recent_5_win_rate"] = trade_stats.get("recent_5_win_rate", 0.50)
        row["recent_drawdown_pct"] = trade_stats.get("recent_drawdown_pct", 0.0)

        # Additional useful features
        row["close"] = close

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _signal_side(bundle: SignalBundle, ts: pd.Timestamp) -> Optional[str]:
    try:
        if ts in bundle.entries_long.index and bool(bundle.entries_long.loc[ts]):
            return "long"
        if ts in bundle.entries_short.index and bool(bundle.entries_short.loc[ts]):
            return "short"
    except Exception:
        pass
    return None


def _entry_price(bundle: SignalBundle, ts: pd.Timestamp, side: str) -> Optional[float]:
    col = bundle.long_entry_price if side == "long" else bundle.short_entry_price
    try:
        v = col.loc[ts]
        return float(v) if pd.notna(v) else None
    except Exception:
        return None


def _stop_price(bundle: SignalBundle, ts: pd.Timestamp, side: str) -> Optional[float]:
    col = bundle.long_stop_price if side == "long" else bundle.short_stop_price
    try:
        v = col.loc[ts]
        return float(v) if pd.notna(v) else None
    except Exception:
        return None


def _target_price(bundle: SignalBundle, ts: pd.Timestamp, side: str) -> Optional[float]:
    col = bundle.long_target_price if side == "long" else bundle.short_target_price
    try:
        v = col.loc[ts]
        return float(v) if pd.notna(v) else None
    except Exception:
        return None


def _ema_slope(features_df: pd.DataFrame, ts: pd.Timestamp, col: str, lookback: int) -> float:
    try:
        idx = features_df.index.get_loc(ts)
        start = max(0, idx - lookback)
        col_vals = features_df.iloc[start: idx + 1][col].dropna()
        if len(col_vals) >= 2:
            return float((col_vals.iloc[-1] - col_vals.iloc[0]) / max(abs(col_vals.iloc[0]), 0.001))
    except Exception:
        pass
    return 0.0


def _stoch_slope(features_df: pd.DataFrame, ts: pd.Timestamp, col: str, lookback: int) -> float:
    try:
        idx = features_df.index.get_loc(ts)
        start = max(0, idx - lookback)
        col_vals = features_df.iloc[start: idx + 1][col].dropna()
        if len(col_vals) >= 2:
            return float(col_vals.iloc[-1] - col_vals.iloc[0])
    except Exception:
        pass
    return 0.0


def _volume_zscore(df: pd.DataFrame, ts: pd.Timestamp, window: int = 20) -> float:
    try:
        idx = df.index.get_loc(ts)
        start = max(0, idx - window)
        vol_slice = df.iloc[start: idx + 1]["Volume"]
        if len(vol_slice) > 1:
            mu = float(vol_slice.mean())
            sd = float(vol_slice.std())
            cv = float(df.loc[ts, "Volume"])
            return (cv - mu) / sd if sd > 0 else 0.0
    except Exception:
        pass
    return 0.0


def _build_trade_stats(trade_history: pd.DataFrame) -> Dict:
    if trade_history is None or len(trade_history) == 0:
        return {}

    trades = trade_history.sort_values("exit_time").copy()
    stats: Dict = {}

    r_vals = trades["r_multiple"].dropna().values

    last5 = r_vals[-5:] if len(r_vals) >= 5 else r_vals
    stats["recent_5_avg_r"] = float(np.mean(last5)) if len(last5) > 0 else 0.0
    stats["recent_5_win_rate"] = float(np.mean(last5 > 0)) if len(last5) > 0 else 0.5

    last10 = r_vals[-10:] if len(r_vals) >= 10 else r_vals
    stats["recent_10_avg_r"] = float(np.mean(last10)) if len(last10) > 0 else 0.0

    if "equity_after" in trades.columns:
        eq = trades["equity_after"].values
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        stats["recent_drawdown_pct"] = float(np.min(dd) * 100) if len(dd) > 0 else 0.0
    else:
        stats["recent_drawdown_pct"] = 0.0

    return stats
