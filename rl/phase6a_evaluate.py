"""Phase 6A — Evaluation and comparison.

Simulates each policy on a given period by:
  1. Loading the pre-built signal-level dataset (with action outcomes).
  2. For each signal, the policy chooses an action.
  3. The chosen action's outcome is applied to track equity/trades/summary.
  4. Produces a policy_comparison.csv row for the period.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rl.phase6a_policy import BasePolicy


def evaluate_policy(
    policy: BasePolicy,
    dataset: pd.DataFrame,
    period_name: str,
) -> Dict:
    """Evaluate a policy on a signal-level dataset.

    For each signal timestamp, the policy decides which action to take.
    The action determines:
      - If 'skip': no trade, equity unchanged.
      - If a trading action: we look up the outcome (net_r_multiple, etc.)
        from the dataset row for that action.

    The dataset must contain, for each signal, one row per action
    with fields: timestamp, action_name, net_r_multiple, net_pnl, etc.

    Args:
        policy:     A BasePolicy instance.
        dataset:    Signal-level dataset with action outcomes.
        period_name: Label for the period being evaluated.

    Returns:
        Dict with performance metrics (matching policy_comparison.csv schema).
    """
    if dataset.empty:
        return _empty_metrics(period_name, policy.name)

    # Get policy's chosen actions per signal
    policy_actions = policy.predict_actions(dataset)

    # For each signal, find the corresponding outcome row for chosen action
    # Dataset has one row per (timestamp, action). We pivot to get
    # the chosen action's outcome.
    outcomes = _collect_outcomes(policy_actions, dataset)

    if outcomes.empty:
        return _empty_metrics(period_name, policy.name)

    # Compute summary metrics
    return _compute_summary(outcomes, period_name, policy.name)


def _collect_outcomes(
    policy_actions: pd.DataFrame,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Collect the outcome for each signal based on the policy's chosen action.

    Args:
        policy_actions: DataFrame with columns ['timestamp', 'chosen_action'].
        dataset: Full dataset with columns ['timestamp', 'action_name', ...].

    Returns:
        DataFrame with one row per executed signal (skips excluded),
        containing outcome fields: net_r_multiple, net_pnl, etc.
    """
    results = []

    # Merge: for each row in policy_actions, find the matching action outcome
    for _, row in policy_actions.iterrows():
        ts = row["timestamp"]
        chosen_action = row["chosen_action"]

        if chosen_action == "skip":
            continue

        # Find the matching action row in dataset
        match = dataset[
            (dataset["timestamp"] == ts) &
            (dataset["action_name"] == chosen_action)
        ]

        if match.empty:
            continue

        r = match.iloc[0]
        results.append({
            "timestamp": ts,
            "chosen_action": chosen_action,
            "side": r.get("side"),
            "entry_price": r.get("entry_price"),
            "exit_price": None,
            "net_r_multiple": r.get("net_r_multiple", 0.0),
            "net_pnl": r.get("net_pnl", 0.0),
            "max_adverse_excursion_r": r.get("max_adverse_excursion_r", 0.0),
            "max_favorable_excursion_r": r.get("max_favorable_excursion_r", 0.0),
        })

    return pd.DataFrame(results)


def _compute_summary(
    outcomes: pd.DataFrame,
    period_name: str,
    policy_name: str,
) -> Dict:
    """Compute comprehensive metrics from executed trade outcomes."""
    total_signals = len(outcomes)
    if total_signals == 0:
        return _empty_metrics(period_name, policy_name)

    r_values = outcomes["net_r_multiple"].dropna().values
    pnl_values = outcomes["net_pnl"].dropna().values

    total_trades = len(r_values)
    if total_trades == 0:
        return _empty_metrics(period_name, policy_name)

    # Win rate
    wins = r_values > 0
    win_count = int(wins.sum())
    win_rate = win_count / total_trades

    # Profit factor
    gross_profit = float(pnl_values[pnl_values > 0].sum())
    gross_loss = float(abs(pnl_values[pnl_values < 0].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    # R-multiple stats
    avg_r = float(r_values.mean())
    expectancy_r = float(r_values.mean())  # net expectancy = mean R

    # Max drawdown (simple equity-based)
    equity = 100000.0 + np.cumsum(pnl_values)
    peak = np.maximum.accumulate(equity)
    dd_pct = ((equity - peak) / peak * 100)
    max_dd = float(np.min(dd_pct))

    # Total return
    total_return = float((equity[-1] / 100000.0 - 1) * 100)

    # Side breakdown
    long_trades = int((outcomes["side"] == "long").sum())
    short_trades = int((outcomes["side"] == "short").sum())

    # Fees (estimate: $5 per trade)
    total_fees = total_trades * 5.0

    return {
        "policy_name": policy_name,
        "period": period_name,
        "total_signals": total_signals,
        "executed_trades": total_trades,
        "skipped_trades": total_signals - total_trades,
        "execution_rate": total_trades / total_signals if total_signals > 0 else 0.0,
        "total_return": total_return,
        "profit_factor": profit_factor,
        "sharpe_ratio": _sharpe_from_r(r_values),
        "max_drawdown_pct": max_dd,
        "avg_r": avg_r,
        "expectancy_r": expectancy_r,
        "win_rate": win_rate,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "total_fees": total_fees,
        "total_slippage_cost": 0.0,  # Not tracked at signal level
    }


def _sharpe_from_r(r_values: np.ndarray) -> float:
    """Approximate Sharpe from R-multiple series."""
    if len(r_values) < 5:
        return 0.0
    std = float(np.std(r_values))
    if std == 0:
        return 0.0
    mean_r = float(np.mean(r_values))
    # Assume ~187 trades over 2 years ≈ 93.5 per year
    annual_factor = np.sqrt(93.5)
    return mean_r / std * annual_factor


def _empty_metrics(period_name: str, policy_name: str) -> Dict:
    return {
        "policy_name": policy_name,
        "period": period_name,
        "total_signals": 0,
        "executed_trades": 0,
        "skipped_trades": 0,
        "execution_rate": 0.0,
        "total_return": 0.0,
        "profit_factor": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "avg_r": 0.0,
        "expectancy_r": 0.0,
        "win_rate": 0.0,
        "long_trades": 0,
        "short_trades": 0,
        "total_fees": 0.0,
        "total_slippage_cost": 0.0,
    }


def generate_policy_comparison_table(
    results: List[Dict],
) -> pd.DataFrame:
    """Generate the policy_comparison.csv DataFrame from a list of evaluation results."""
    return pd.DataFrame(results)


def generate_feature_importance(
    model, feature_columns: List[str],
) -> pd.DataFrame:
    """Extract feature importance from a trained RandomForest model.

    If model is a dict of per-action models, averages importance across all.
    """
    if hasattr(model, "feature_importances_"):
        # Single model
        importances = model.feature_importances_
        return pd.DataFrame({
            "feature": feature_columns,
            "importance": importances,
        }).sort_values("importance", ascending=False)

    if isinstance(model, dict):
        # Per-action models
        all_importances = {}
        count = 0
        for action_name, m in model.items():
            if m is not None and hasattr(m, "feature_importances_"):
                for i, col in enumerate(feature_columns):
                    all_importances[col] = all_importances.get(col, 0.0) + m.feature_importances_[i]
                count += 1

        if count > 0:
            avg = {k: v / count for k, v in all_importances.items()}
            return pd.DataFrame({
                "feature": list(avg.keys()),
                "importance": list(avg.values()),
            }).sort_values("importance", ascending=False)

    return pd.DataFrame({"feature": feature_columns, "importance": [0.0] * len(feature_columns)})
