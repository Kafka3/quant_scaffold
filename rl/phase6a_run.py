#!/usr/bin/env python3
"""Phase 6A — Offline Top-4 Dynamic Controller.

Main entry point.  Runs the full pipeline:
  1. Load data & configs
  2. For each non-skip action, build signals + backtest on train/val/test splits
  3. Build signal-level dataset with features + outcomes
  4. Train learned_policy on train set
  5. Evaluate all 3 policies on train, val, test
  6. Generate reports

Usage:
    python rl/phase6a_run.py --data data/raw/BTCUSDT_5m_2024_2025.csv

Outputs:
    reports/phase6a_top4_dynamic_dataset.csv
    reports/phase6a_policy_comparison.csv
    reports/phase6a_test_trades.csv
    reports/phase6a_feature_importance.csv
    reports/phase6a_summary.md
"""

from __future__ import annotations

import sys
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from optimize.utils import slice_dataframe
from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.cost_model import apply_slippage, calculate_fees, calculate_slippage_cost
from backtest.risk_model import calculate_position_size
from backtest.event_engine import (
    _run_event_driven_loop,
    _build_summary_ps,
)

from rl.phase6a_actions import (
    ACTIONS,
    NON_SKIP_ACTIONS,
    build_action_configs,
    ActionDef,
    get_action,
)
from rl.phase6a_features import extract_features
from rl.phase6a_dataset import compute_excursions, compute_reward, FEATURE_COLUMNS
from rl.phase6a_policy import (
    AlwaysExecutePolicy,
    SimpleRuleFilterPolicy,
    LearnedPolicy,
    create_policy,
)
from rl.phase6a_evaluate import (
    evaluate_policy,
    generate_policy_comparison_table,
    generate_feature_importance,
)

# =========================================================================
# Constants
# =========================================================================

REPORT_DIR = _PROJECT_ROOT / "reports"

# Fixed risk/cost config (Phase 5 Pre-Live)
RISK_COST_CONFIG = {
    "account": {"initial_cash": 100000},
    "risk": {
        "risk_per_trade_pct": 0.01,
        "max_position_value_pct": 1.0,
        "max_leverage": 10,
        "min_qty": 0.0001,
        "qty_step": 0.0001,
    },
    "cost": {
        "fixed_fee_per_trade": 5.0,
        "slippage_per_side": 5.0,
        "fee_rate": 0.0,
    },
    "position_modes": {
        "capped_10x": {"max_position_value_pct": 1.0, "max_leverage": 10},
    },
    "execution": {
        "allow_short": True,
        "same_bar_stop_first": True,
    },
}

MIN_STOP_DISTANCE_PCT = 0.001

# Time splits
TRAIN = ("train", "2024-01-01", "2025-07-01")
VAL = ("val", "2025-07-01", "2025-10-01")
TEST = ("test", "2025-10-01", "2026-01-01")
ALL_SPLITS = [TRAIN, VAL, TEST]

# For summary report, also run full period
FULL_PERIOD = ("2024-2025-Full", "2024-01-01", "2026-01-01")


# =========================================================================
# Core: Backtest an action on a time slice
# =========================================================================


def backtest_action(
    action: ActionDef,
    df_slice: pd.DataFrame,
    action_config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """Run a full backtest for a single action on a time slice.

    Returns:
        (trades_df, skipped_df, equity_series, summary_dict)
    """
    bundle = build_signals(df_slice, action_config["strategy"])

    risk_cfg = RISK_COST_CONFIG["risk"]
    cost_cfg = RISK_COST_CONFIG["cost"]
    exec_cfg = RISK_COST_CONFIG["execution"]
    initial_cash = 100000.0
    fee_rate = cost_cfg.get("fee_rate", 0.0)
    fixed_fee = cost_cfg.get("fixed_fee_per_trade", 5.0)
    slippage = cost_cfg.get("slippage_per_side", 5.0)
    allow_short = exec_cfg.get("allow_short", True)
    target_risk_pct = risk_cfg["risk_per_trade_pct"]
    max_lev = risk_cfg["max_leverage"]
    max_pos_pct = risk_cfg["max_position_value_pct"]
    min_qty = risk_cfg["min_qty"]
    qty_step = risk_cfg["qty_step"]

    skipped: List[dict] = []

    def try_enter(idx, side, cash):
        if side == "long":
            entry_price_raw = float(bundle.long_entry_price.loc[idx])
            stop_price = float(bundle.long_stop_price.loc[idx])
            target_price = float(bundle.long_target_price.loc[idx])
        else:
            entry_price_raw = float(bundle.short_entry_price.loc[idx])
            stop_price = float(bundle.short_stop_price.loc[idx])
            target_price = float(bundle.short_target_price.loc[idx])

        if pd.notna(entry_price_raw) and pd.notna(stop_price) and pd.notna(target_price):
            stop_dist = abs(entry_price_raw - stop_price)
            stop_dist_pct = stop_dist / entry_price_raw

            # Filter: stop too tight
            if stop_dist_pct < MIN_STOP_DISTANCE_PCT:
                return None, {"time": idx, "side": side, "skip_reason": "stop_too_tight",
                              "entry_price": entry_price_raw, "stop_price": stop_price,
                              "equity": cash}

            # Filter: required leverage > max
            required_lev = target_risk_pct / stop_dist_pct if stop_dist_pct > 0 else 999
            if required_lev > max_lev:
                return None, {"time": idx, "side": side, "skip_reason": "leverage_required_exceeds_cap",
                              "entry_price": entry_price_raw, "stop_price": stop_price,
                              "required_leverage": required_lev, "max_leverage": max_lev,
                              "equity": cash}

            sizing = calculate_position_size(
                equity=cash, entry_price=entry_price_raw, stop_price=stop_price,
                risk_per_trade_pct=target_risk_pct,
                max_position_value_pct=max_pos_pct, max_leverage=max_lev,
                min_qty=min_qty, qty_step=qty_step,
            )

            if sizing["skip_trade"]:
                return None, {"time": idx, "side": side, "skip_reason": sizing["skip_reason"],
                              "entry_price": entry_price_raw, "stop_price": stop_price,
                              "equity": cash}

            entry_filled = apply_slippage(entry_price_raw, side, "entry", slippage)
            return {
                "entry_time": idx, "side": side,
                "entry_price_raw": entry_price_raw, "entry_price_filled": entry_filled,
                "stop_price": stop_price, "target_price": target_price,
                "qty": sizing["qty"], "notional": sizing["notional"],
                "target_risk_amount": sizing["target_risk_amount"],
                "actual_risk_amount": sizing["actual_risk_amount"],
                "target_risk_pct": sizing["target_risk_pct"],
                "actual_risk_pct": sizing["actual_risk_pct"],
                "stop_distance": sizing["stop_distance"],
                "stop_distance_pct": stop_dist_pct,
                "required_leverage": required_lev,
                "raw_qty": sizing["raw_qty"], "max_qty": sizing["max_qty"],
                "cap_hit": sizing.get("cap_hit", False),
                "equity_before": cash, "bars_held": 0,
            }, None
        return None, None

    def finalize_exit(trade, idx, exit_price_raw, exit_reason, cash):
        exit_filled = apply_slippage(exit_price_raw, trade["side"], "exit", slippage)
        if trade["side"] == "long":
            gross_pnl = (exit_filled - trade["entry_price_filled"]) * trade["qty"]
        else:
            gross_pnl = (trade["entry_price_filled"] - exit_filled) * trade["qty"]
        fees = calculate_fees(trade["entry_price_filled"], exit_filled, trade["qty"], fee_rate, fixed_fee)
        slip_cost = calculate_slippage_cost(trade["entry_price_raw"], trade["entry_price_filled"],
                                             exit_price_raw, exit_filled, trade["qty"], trade["side"])
        net_pnl = gross_pnl - fees
        actual_risk = trade.get("actual_risk_amount", 0.0)
        r_multiple = net_pnl / actual_risk if actual_risk > 0 else float("nan")
        equity_after = trade["equity_before"] + net_pnl
        trade.update({
            "exit_time": idx, "exit_price_raw": exit_price_raw, "exit_price_filled": exit_filled,
            "gross_pnl": gross_pnl, "fees": fees, "slippage_cost": slip_cost,
            "net_pnl": net_pnl, "r_multiple": r_multiple, "equity_after": equity_after,
            "exit_reason": exit_reason,
        })
        return equity_after

    def finalize_eod(trade, last_idx, last_close, cash):
        return finalize_exit(trade, last_idx, last_close, "end_of_data", cash)

    def compute_equity(pos, trade, cash, close):
        if pos == "flat" or trade is None:
            return cash
        elif pos == "long":
            return cash + (close - trade["entry_price_filled"]) * trade["qty"]
        else:
            return cash + (trade["entry_price_filled"] - close) * trade["qty"]

    trades, equity_values, warnings = _run_event_driven_loop(
        df_slice, bundle, allow_short, initial_cash,
        try_enter=try_enter, finalize_exit=finalize_exit,
        finalize_eod=finalize_eod, compute_equity=compute_equity,
        skipped=skipped,
    )

    equity = pd.Series(equity_values, index=df_slice.index)

    if trades:
        trades_df = pd.DataFrame(trades)
    else:
        trades_df = pd.DataFrame()

    if skipped:
        skipped_df = pd.DataFrame(skipped)
    else:
        skipped_df = pd.DataFrame()

    summary = _build_summary_ps(trades_df, equity, initial_cash, skipped_df, 5, 0.0)

    return trades_df, skipped_df, equity, summary


# =========================================================================
# Dataset builder
# =========================================================================


def build_signal_dataset(
    df_slice: pd.DataFrame,
    action_configs: Dict[str, dict],
    period_label: str,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Build the full signal-level dataset for a time slice.

    For each non-skip action:
      1. Backtest to get trades
      2. Extract features at each entry timestamp
      3. Label with action_name and outcome reward

    Skip action gets synthetic rows (features extracted, reward=0).

    Returns:
        DataFrame with columns: timestamp, features, action_name, reward.
    """
    all_rows = []

    for action in NON_SKIP_ACTIONS:
        print(f"    Action '{action.name}'...", end=" ", flush=True)
        trades, skipped, equity, summary = backtest_action(
            action, df_slice, action_configs[action.name]
        )

        # Build signals just to get the bundle for feature extraction
        bundle = build_signals(df_slice, action_configs[action.name]["strategy"])

        # Extract features at signal entry timestamps
        if len(trades) > 0:
            entry_timestamps = [pd.Timestamp(t, tz='UTC') for t in trades["entry_time"].values]
        else:
            entry_timestamps = []

        # Also include signals that were skipped
        skipped_timestamps = [pd.Timestamp(t, tz='UTC') for t in skipped["time"].values] if len(skipped) > 0 else []

        # Merge: features at both executed and skipped signal timestamps
        all_entry_ts = sorted(set(entry_timestamps + skipped_timestamps))

        if not all_entry_ts:
            print("no signals")
            continue

        # Extract features
        features_df = extract_features(
            df_slice, bundle, all_entry_ts,
            action_name=action.name,
        )

        if features_df.empty:
            print("no features")
            continue

        # Merge outcome for executed trades
        for _, feat_row in features_df.iterrows():
            ts = feat_row["timestamp"]
            row = {
                "timestamp": ts,
                "action_name": action.name,
                **{col: feat_row.get(col, 0.0) for col in FEATURE_COLUMNS},
                **{col: feat_row.get(col) for col in [
                    "side", "entry_price", "stop_price", "target_price",
                    "stop_distance_pct", "required_leverage", "close",
                ]},
            }

            # Ensure trades entry_time is timezone-naive for matching
            if len(trades) > 0 and hasattr(trades["entry_time"].dt, 'tz') and trades["entry_time"].dt.tz is not None:
                trades["entry_time"] = trades["entry_time"].dt.tz_localize(None)

            # Check if this was an executed trade
            ts_naive = ts.tz_localize(None) if hasattr(ts, 'tz') and ts.tz is not None else ts
            trade_match = trades[trades["entry_time"] == ts_naive]
            if len(trade_match) > 0:
                t = trade_match.iloc[0]
                row["net_r_multiple"] = float(t.get("r_multiple", 0.0))
                row["net_pnl"] = float(t.get("net_pnl", 0.0))
                row["reward"] = compute_reward(row["net_r_multiple"], action.name)
                row["max_adverse_excursion_r"] = -1.0 if t.get("exit_reason") == "stop" else 0.0
                row["max_favorable_excursion_r"] = float(t.get("r_multiple", 0.0)) if t.get("exit_reason") == "target" else 0.0
            else:
                row["net_r_multiple"] = 0.0
                row["net_pnl"] = 0.0
                row["reward"] = 0.0
                row["max_adverse_excursion_r"] = 0.0
                row["max_favorable_excursion_r"] = 0.0

            all_rows.append(row)

        print(f"{len(all_entry_ts)} signals")

    if not all_rows:
        print("  ⚠️  No signals generated for any action")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # Add period label
    df["period"] = period_label
    return df


# =========================================================================
# Per-action summary (for report)
# =========================================================================


def action_full_summary(action: ActionDef, df_slice: pd.DataFrame, config: dict) -> Dict:
    """Run full backtest for an action and return summary metrics."""
    trades, skipped, equity, summary = backtest_action(action, df_slice, config)
    total_return = float(summary.get("total_return", 0.0))
    total_trades = int(summary.get("total_trades", 0))
    pf = summary.get("profit_factor", 0.0)
    if pf is None or (isinstance(pf, float) and math.isinf(pf)):
        pf = 999.0
    dd = float(summary.get("max_drawdown_pct", 0.0))
    return {
        "action": action.name,
        "total_trades": total_trades,
        "total_return": total_return,
        "profit_factor": pf,
        "max_drawdown": dd,
    }


# =========================================================================
# Main
# =========================================================================


def main():
    parser = ArgumentParser(description="Phase 6A — Offline Top-4 Dynamic Controller")
    parser.add_argument("--data", dest="data_path", required=True)
    args = parser.parse_args()

    # 1. Load data
    print("=" * 65)
    print("Phase 6A — Offline Top-4 Dynamic Controller")
    print("=" * 65)

    df_full = load_ohlcv_csv(args.data_path)
    print(f"\nData: {df_full.index[0]} ~ {df_full.index[-1]}, {len(df_full):,} bars")

    # 2. Load base config
    base_path = _PROJECT_ROOT / "configs/candidates/tpe_trial_490.yaml"
    with open(base_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    print(f"Base config: {base_path}")

    # 3. Pre-build action configs
    action_configs = build_action_configs(base_config)
    print(f"Actions: {[a.name for a in ACTIONS]}")

    # 4. Build dataset per split
    print("\n--- Building signal-level dataset ---")
    all_dataset_rows = []
    for split_name, start_str, end_str in ALL_SPLITS:
        print(f"\n[{split_name}] {start_str} ~ {end_str}")
        df_slice = slice_dataframe(df_full, start_str, end_str)
        print(f"  Bars: {len(df_slice):,}")

        split_dataset = build_signal_dataset(df_slice, action_configs, split_name)
        if not split_dataset.empty:
            all_dataset_rows.append(split_dataset)
            print(f"  Total signal rows: {len(split_dataset)}")
        else:
            print(f"  ⚠️  No signal rows generated for {split_name}")

    if not all_dataset_rows:
        print("\n❌ No dataset generated. Check signal generation.")
        sys.exit(1)

    full_dataset = pd.concat(all_dataset_rows, ignore_index=True)
    print(f"\nTotal dataset rows: {len(full_dataset)}")

    # Save dataset
    dataset_path = REPORT_DIR / "phase6a_top4_dynamic_dataset.csv"
    full_dataset.to_csv(dataset_path, index=False)
    print(f"✅ Dataset saved: {dataset_path}")

    # 5. Split into train/val/test
    train_dataset = full_dataset[full_dataset["period"] == "train"].copy()
    val_dataset = full_dataset[full_dataset["period"] == "val"].copy()
    test_dataset = full_dataset[full_dataset["period"] == "test"].copy()

    print(f"\nTrain rows: {len(train_dataset)}")
    print(f"Val rows:   {len(val_dataset)}")
    print(f"Test rows:  {len(test_dataset)}")

    # 6. Train learned policy
    print("\n--- Training policies ---")

    always_execute = AlwaysExecutePolicy(default_action="base_tpe490")
    simple_rule = SimpleRuleFilterPolicy(recent_n=5, min_avg_r=0.0, default_action="base_tpe490")

    # Build feature columns list (actual columns that exist in the dataset)
    available_features = [c for c in FEATURE_COLUMNS if c in full_dataset.columns]
    print(f"Available features ({len(available_features)}): {available_features}")

    learned = LearnedPolicy(
        feature_columns=available_features,
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        min_predicted_r=0.0,
    )
    if not train_dataset.empty:
        print("  Fitting learned_policy on train set...")
        learned.fit(train_dataset, val_dataset if not val_dataset.empty else None)
    else:
        print("  ⚠️  Empty train set, learned_policy not fitted")

    # 7. Evaluate all policies on train, val, test
    print("\n--- Evaluating policies ---")
    all_results = []

    for split_name, dataset in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
        if dataset.empty:
            print(f"  [{split_name}] Empty dataset, skipping")
            continue

        print(f"\n  [{split_name}] Evaluating...")

        for policy in [always_execute, simple_rule, learned]:
            if not learned.is_fitted and policy.name == "learned_policy":
                continue
            result = evaluate_policy(policy, dataset, split_name)
            all_results.append(result)
            print(f"    {policy.name:20s}: "
                  f"trades={result['executed_trades']:>3} "
                  f"ret={result['total_return']:>6.2f}% "
                  f"pf={result['profit_factor']:.2f} "
                  f"dd={result['max_drawdown_pct']:.2f}% "
                  f"exec_rate={result['execution_rate']:.0%}")

    # Also evaluate on full dataset
    print("\n  [full] Evaluating...")
    for policy in [always_execute, simple_rule, learned]:
        if not learned.is_fitted and policy.name == "learned_policy":
            continue
        result = evaluate_policy(policy, full_dataset, "2024-2025-Full")
        all_results.append(result)
        print(f"    {policy.name:20s}: "
              f"trades={result['executed_trades']:>3} "
              f"ret={result['total_return']:>6.2f}% "
              f"pf={result['profit_factor']:.2f} "
              f"dd={result['max_drawdown_pct']:.2f}% "
              f"exec_rate={result['execution_rate']:.0%}")

    # 8. Save policy comparison
    comparison_df = generate_policy_comparison_table(all_results)
    comparison_path = REPORT_DIR / "phase6a_policy_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✅ Policy comparison saved: {comparison_path}")

    # 9. Feature importance
    if learned.is_fitted:
        importances = generate_feature_importance(learned.models, available_features)
        imp_path = REPORT_DIR / "phase6a_feature_importance.csv"
        importances.to_csv(imp_path, index=False)
        print(f"✅ Feature importance saved: {imp_path}")
    else:
        # Save empty
        pd.DataFrame({"feature": available_features, "importance": [0.0] * len(available_features)})\
            .to_csv(REPORT_DIR / "phase6a_feature_importance.csv", index=False)

    # 10. Test trades
    test_trades_df = test_dataset[test_dataset["action_name"] == "base_tpe490"].copy()
    test_trades_path = REPORT_DIR / "phase6a_test_trades.csv"
    test_trades_df.to_csv(test_trades_path, index=False)
    print(f"✅ Test trades saved: {test_trades_path} ({len(test_trades_df)} rows)")

    # 11. Summary markdown
    _write_summary(all_results, comparison_df, learned, available_features, full_dataset)

    print(f"\n{'='*65}")
    print("Phase 6A — Complete")
    print(f"{'='*65}")


def _write_summary(
    all_results: List[Dict],
    comparison_df: pd.DataFrame,
    learned: LearnedPolicy,
    feature_cols: List[str],
    dataset: pd.DataFrame,
) -> None:
    """Write the phase6a_summary.md report."""
    lines = []

    def w(s):
        lines.append(s)

    w("# Phase 6A — Offline Top-4 Dynamic Controller Summary\n")
    w(f"> 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    w("> 候选: tpe_trial_490\n")
    w("> 动态参数: right_bars, rr_target, min_separation, min_close_ratio\n")
    w("> Policy: always_execute / simple_rule_filter / learned_policy (RandomForest)\n")
    w("\n---\n")

    # Action space overview
    w("## 动作空间\n\n")
    w("| Action | right_bars | rr_target | min_separation | min_close_ratio |\n")
    w("|--------|:----------:|:---------:|:--------------:|:---------------:|\n")
    for a in ACTIONS:
        if a.name == "skip":
            w(f"| {a.label:25s} | — | — | — | — |\n")
        else:
            w(f"| {a.label:25s} | {a.params.get('right_bars', '—'):>2d} | {a.params.get('rr_target', ''):>5} | {a.params.get('min_separation', ''):>2d} | {a.params.get('min_close_ratio', ''):>5} |\n")

    w("\n---\n")

    # Dataset stats
    w("## 数据集\n\n")
    w(f"- 总信号行数: {len(dataset)}\n")
    w(f"- 特征数: {len(feature_cols)}\n")
    if len(dataset) > 0:
        w(f"- 时间段: {dataset['period'].unique()}\n")
        w(f"- 各 action 信号分布:\n\n")
        action_counts = dataset.groupby("action_name").size().sort_values(ascending=False)
        for name, count in action_counts.items():
            w(f"  - {name}: {count} 条\n")

    w("\n---\n")

    # Policy comparison
    w("## Policy 对比\n\n")
    for period in ["train", "val", "test", "2024-2025-Full"]:
        sub = comparison_df[comparison_df["period"] == period]
        if sub.empty:
            continue
        w(f"### {period}\n\n")
        w("| Policy | 执行数 | 跳过数 | 执行率 | 总收益 | PF | Sharpe | DD | avg_r | WinRate |\n")
        w("|-------|:------:|:------:|:-----:|:-----:|:--:|:------:|:--:|:----:|:-------:|\n")
        for _, r in sub.iterrows():
            w(
                f"| {r['policy_name']:20s} "
                f"| {r['executed_trades']:>3} "
                f"| {r['skipped_trades']:>3} "
                f"| {r['execution_rate']:.0%} "
                f"| {r['total_return']:>6.2f}% "
                f"| {r['profit_factor']:.2f} "
                f"| {r['sharpe_ratio']:.2f} "
                f"| {r['max_drawdown_pct']:.2f}% "
                f"| {r['avg_r']:.2f} "
                f"| {r['win_rate']:.1%} |\n"
            )
        w("\n")

    w("---\n")

    # Feature importance
    w("## Top-10 特征重要性\n\n")
    imp_path = REPORT_DIR / "phase6a_feature_importance.csv"
    if imp_path.exists():
        imp_df = pd.read_csv(imp_path)
        imp_df = imp_df.sort_values("importance", ascending=False).head(10)
        w("| 特征 | 重要性 |\n")
        w("|------|:------:|\n")
        for _, r in imp_df.iterrows():
            w(f"| {r['feature']} | {r['importance']:.4f} |\n")

    w("\n---\n")

    # Test period results (critical)
    w("## Test 结果 (2025-Q4, 严格样本外)\n\n")
    test_results = comparison_df[comparison_df["period"] == "test"]
    if not test_results.empty:
        w("| 指标 | always_execute | simple_rule_filter | learned_policy |\n")
        w("|------|:--------------:|:------------------:|:--------------:|\n")

        metrics = ["total_return", "profit_factor", "sharpe_ratio",
                    "max_drawdown_pct", "expectancy_r", "win_rate",
                    "executed_trades", "execution_rate"]
        for m in metrics:
            row = "| " + m
            for policy_name in ["always_execute", "simple_rule_filter", "learned_policy"]:
                match = test_results[test_results["policy_name"] == policy_name]
                if not match.empty:
                    val = match.iloc[0][m]
                    if isinstance(val, float):
                        row += f" | {val:.2f}"
                    else:
                        row += f" | {val}"
                else:
                    row += " | —"
            w(row + " |\n")

        # Pass/fail judgment
        ae = test_results[test_results["policy_name"] == "always_execute"]
        lp = test_results[test_results["policy_name"] == "learned_policy"]

        if not ae.empty and not lp.empty:
            ae_r = ae.iloc[0]
            lp_r = lp.iloc[0]
            checks = []
            checks.append(("PF更高", lp_r["profit_factor"] >= ae_r["profit_factor"]))
            checks.append(("ExpectR更高", lp_r["expectancy_r"] >= ae_r["expectancy_r"]))
            checks.append(("DD不恶化", lp_r["max_drawdown_pct"] >= ae_r["max_drawdown_pct"]))
            checks.append(("执行数>=50%", lp_r["executed_trades"] >= ae_r["executed_trades"] * 0.5))

            all_pass = all(c for _, c in checks)
            w(f"\n### 通过检查: {'✅ 全部通过' if all_pass else '❌ 未全部通过'}\n")
            for label, ok in checks:
                w(f"- {'✅' if ok else '❌'} {label}\n")

            if all_pass:
                w("\n**结论: ✅ Phase 6A 通过 — learned_policy 在样本外优于 always_execute**\n")
            else:
                w("\n**结论: ❌ Phase 6A 未完全通过 — 需要进一步调优**\n")
    else:
        w("(无 test 结果)\n")

    w("\n---\n")
    w("### 下一阶段\n\n")
    w("- Phase 6B: 扩展动作空间 / 引入 PPO\n")
    w("- Phase 5A: Freqtrade Adapter + Backtest Parity\n")

    text = "".join(lines)
    summary_path = REPORT_DIR / "phase6a_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
