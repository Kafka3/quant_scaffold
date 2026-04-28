"""Phase 6A — Signal-level dataset builder.

For each action (except skip), independently:
  1. Build signals + backtest for a given period
  2. Record each signal's outcome (net_r_multiple, max_adverse/favorable excursion)
  3. Extract features at each signal point
  4. Label with action_name and outcome metrics

The result is a signal-level dataset where each row = (features, action, reward).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rl.phase6a_actions import (
    ActionDef,
    ACTIONS,
    NON_SKIP_ACTIONS,
    build_strategy_config,
)


def build_dataset_for_action(
    action: ActionDef,
    df_period: pd.DataFrame,
    base_config: dict,
    risk_cost_config: dict,
    trade_history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build a signal-level dataset for a single action over a period.

    For each trade executed by this action, records:
      - entry timestamp, side, prices
      - feature vector (from extract_features)
      - outcome: net_r_multiple, net_pnl, max_adverse/favorable excursion
      - reward (currently = net_r_multiple for non-skip trades)

    Args:
        action:        The action definition.
        df_period:     Subset of OHLCV data for the period.
        base_config:   Loaded tpe_trial_490 config.
        risk_cost_config: Risk/cost config dict.
        trade_history: Past trades for recent-performance features.

    Returns:
        DataFrame with signal-level rows.  Empty if no trades.
    """
    # This is a placeholder — the actual backtesting + signal extraction
    # is done in phase6a_run.py by calling the full pipeline for each action.
    # Here we just define the schema.
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Dataset schema (column metadata)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    # Price/trend
    "ema_slope",
    "price_position_vs_channel",
    "channel_width",
    # Prior trend
    "above_ratio",
    "below_ratio",
    # Stochastic
    "stoch_d",
    "stoch_slope",
    # Volatility
    "atr_14",
    "atr_pct",
    "volume_zscore",
    # Temporal
    "hour_of_day",
    "day_of_week",
    # Recent performance
    "recent_5_trade_avg_r",
    "recent_10_trade_avg_r",
    "recent_5_win_rate",
    "recent_drawdown_pct",
]

SIGNAL_META_COLUMNS = [
    "timestamp",
    "side",
    "entry_price",
    "stop_price",
    "target_price",
    "stop_distance_pct",
    "required_leverage",
    "close",
]

LABEL_COLUMNS = [
    "action",
    "action_name",
    "net_r_multiple",
    "net_pnl",
    "max_adverse_excursion_r",
    "max_favorable_excursion_r",
    "reward",
]

ALL_DATASET_COLUMNS = SIGNAL_META_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS


def compute_reward(
    net_r_multiple: float,
    action_name: str,
    max_adverse_excursion_r: Optional[float] = None,
) -> float:
    """Compute reward for a signal-action pair.

    Phase 6A v1: simple R-multiple reward.
    Skip actions get 0.
    """
    if action_name == "skip":
        return 0.0
    return net_r_multiple


def compute_excursions(
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute max adverse/favorable excursion for each trade.

    Requires:
        - entry_price_filled, stop_price, target_price, side
        - For max excursion, would need intra-bar data (not available from
          simple backtest result). For v1, we estimate from bars_held and
          the low/high of entry/exit bars.

    Phase 6A v1: computes a simplified version from available data.
    Returns updated DataFrame with max_adverse_excursion_r, max_favorable_excursion_r.
    """
    if trades_df.empty:
        return trades_df

    df = trades_df.copy()

    # Simplified: estimate adverse/favorable from exit reason
    # If stopped out, adverse ≈ stop distance; if hit target, favorable ≈ target distance
    df["max_adverse_excursion_r"] = np.where(
        df["exit_reason"] == "stop",
        -1.0,  # actual R is -1 at stop
        df.apply(
            lambda r: -abs(r["r_multiple"]) if r["r_multiple"] < 0 else 0.0,
            axis=1,
        ),
    )

    df["max_favorable_excursion_r"] = np.where(
        df["exit_reason"] == "target",
        df["r_multiple"],  # full R achieved at target
        df.apply(
            lambda r: r["r_multiple"] if r["r_multiple"] > 0 else 0.0,
            axis=1,
        ),
    )

    return df
