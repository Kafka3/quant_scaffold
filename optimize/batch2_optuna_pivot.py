#!/usr/bin/env python
"""Batch 2: Optuna + TPE local refinement around pivot separation parameters.

Searches min_separation (5-10) and max_separation (30-50) with the constraint
max_separation >= min_separation + 10.  All other parameters are fixed to the
robust_ema55 baseline values.

Each trial is evaluated across 8 quarterly periods (2024-Q1 to 2025-Q4) in both
cost and no-cost modes.  The Optuna objective maximises the cost-mode batch2_score.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler

# ------------------------------------------------------------------
# Ensure project root is on PYTHONPATH
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from optimize.utils import safe_profit_factor, slice_dataframe, compute_extra_metrics
from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.vectorbt_engine import run_backtest

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
PERIODS = [
    ("2024-Q1", "2024-01-01", "2024-04-01"),
    ("2024-Q2", "2024-04-01", "2024-07-01"),
    ("2024-Q3", "2024-07-01", "2024-10-01"),
    ("2024-Q4", "2024-10-01", "2025-01-01"),
    ("2025-Q1", "2025-01-01", "2025-04-01"),
    ("2025-Q2", "2025-04-01", "2025-07-01"),
    ("2025-Q3", "2025-07-01", "2025-10-01"),
    ("2025-Q4", "2025-10-01", "2026-01-01"),
]

COST_MODES = ["no-cost", "cost"]

N_TRIALS = 100

# Baseline & Batch1 reference combos (for final comparison)
BASELINE_COMBO = (3, 20)
BATCH1_BEST_COMBO = (8, 40)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def build_strategy_config(min_sep, max_sep):
    return {
        "stochastic": {
            "k_period": 14,
            "d_period": 3,
            "smooth": 1,
            "oversold": 15,
            "overbought": 85,
        },
        "pivots": {
            "left_bars": 4,
            "right_bars": 3,
            "min_separation": min_sep,
            "max_separation": max_sep,
            "strict": True,
        },
        "trend": {
            "ema_period": 55,
            "lookback_bars": 24,
            "min_close_ratio": 0.80,
        },
        "risk": {
            "atr_period": 14,
            "stop_buffer": 0.0,
            "rr_target": 2.2,
        },
        "setup": {
            "setup_max_bars": 12,
            "replace_same_side_setup": True,
            "invalidate_on_stop_anchor_break": True,
        },
    }


def run_single(cfg, df_full, cost_mode, period_name, start, end):
    bt_cfg = {
        "initial_cash": 100000.0,
        "fee_per_trade": 0.0 if cost_mode == "no-cost" else 5.0,
        "slippage": 0.0 if cost_mode == "no-cost" else 5.0,
        "allow_short": True,
    }

    df_slice = slice_dataframe(df_full, start, end)
    if len(df_slice) < 100:
        return None

    bundle = build_signals(df_slice, cfg)
    result = run_backtest(df_slice, bundle, bt_cfg)
    summary = result.summary
    extras = compute_extra_metrics(result)

    return {
        "cost_mode": cost_mode,
        "min_separation": cfg["pivots"]["min_separation"],
        "max_separation": cfg["pivots"]["max_separation"],
        "period": period_name,
        "start_time": start,
        "end_time": end,
        "total_bars": len(df_slice),
        "total_return": summary.get("total_return", 0.0),
        "total_trades": summary.get("total_trades", 0),
        "win_rate": summary.get("win_rate", 0.0),
        "profit_factor": safe_profit_factor(summary.get("profit_factor")),
        "max_drawdown": summary.get("max_drawdown", 0.0),
        "avg_trade": summary.get("avg_trade", 0.0),
        "expectancy": summary.get("expectancy", 0.0),
        **extras,
    }


def compute_batch2_score(row):
    """Compute batch2_score from summary row (same formula as Batch 1)."""
    if row["min_trades_per_period"] < 8:
        return -9999.0

    avg_pf_score = min(row["avg_profit_factor"], 5.0) / 5.0
    min_pf_score = min(row["min_profit_factor"], 5.0) / 5.0
    avg_return_score = max(-1.0, min(1.0, row["avg_quarter_return"] / 5.0))
    trade_dist_score = min(row["min_trades_per_period"] / 20.0, 1.0)
    dd_penalty = abs(row["max_quarter_drawdown_worst"]) / 20.0

    score = (
        0.25 * avg_pf_score
        + 0.20 * row["positive_ratio"]
        + 0.20 * min_pf_score
        + 0.15 * avg_return_score
        + 0.10 * trade_dist_score
        - 0.10 * dd_penalty
    )
    return score


def build_summary(period_rows):
    """Aggregate a list of per-period result dicts into a summary dict."""
    sub = pd.DataFrame(period_rows)
    if len(sub) == 0:
        return {
            "total_periods": 0,
            "positive_periods": 0,
            "positive_ratio": 0.0,
            "total_return_sum": 0.0,
            "avg_quarter_return": 0.0,
            "median_quarter_return": 0.0,
            "min_quarter_return": 0.0,
            "total_trades_sum": 0,
            "min_trades_per_period": 0,
            "avg_trades_per_period": 0.0,
            "avg_profit_factor": 0.0,
            "median_profit_factor": 0.0,
            "min_profit_factor": 0.0,
            "max_quarter_drawdown_worst": 0.0,
            "long_trades_total": 0,
            "short_trades_total": 0,
        }

    total_periods = len(sub)
    positive_periods = int((sub["total_return"] > 0).sum())
    positive_ratio = positive_periods / total_periods if total_periods > 0 else 0.0

    total_return_sum = float(sub["total_return"].sum())
    avg_quarter_return = float(sub["total_return"].mean())
    median_quarter_return = float(sub["total_return"].median())
    min_quarter_return = float(sub["total_return"].min())

    total_trades_sum = int(sub["total_trades"].sum())
    min_trades_per_period = int(sub["total_trades"].min())
    avg_trades_per_period = float(sub["total_trades"].mean())

    avg_profit_factor = float(sub["profit_factor"].mean())
    median_profit_factor = float(sub["profit_factor"].median())
    min_profit_factor = float(sub["profit_factor"].min())

    max_quarter_drawdown_worst = float(sub["max_drawdown"].min())

    long_trades_total = int(sub["long_trades"].sum())
    short_trades_total = int(sub["short_trades"].sum())

    return {
        "total_periods": total_periods,
        "positive_periods": positive_periods,
        "positive_ratio": positive_ratio,
        "total_return_sum": total_return_sum,
        "avg_quarter_return": avg_quarter_return,
        "median_quarter_return": median_quarter_return,
        "min_quarter_return": min_quarter_return,
        "total_trades_sum": total_trades_sum,
        "min_trades_per_period": min_trades_per_period,
        "avg_trades_per_period": avg_trades_per_period,
        "avg_profit_factor": avg_profit_factor,
        "median_profit_factor": median_profit_factor,
        "min_profit_factor": min_profit_factor,
        "max_quarter_drawdown_worst": max_quarter_drawdown_worst,
        "long_trades_total": long_trades_total,
        "short_trades_total": short_trades_total,
    }


# ------------------------------------------------------------------
# Evaluation cache (keyed by param combo)
# ------------------------------------------------------------------
_param_cache = {}


def evaluate_combo(min_sep, max_sep, df_full):
    """Run all periods & cost modes for a single parameter combo.

    Returns
    -------
    val_results : list[dict]
        Per-period raw results (no trial_id).
    summary_results : dict[str, dict]
        cost_mode -> summary dict (no trial_id, no batch2_score yet).
    """
    key = (min_sep, max_sep)
    if key in _param_cache:
        return _param_cache[key]

    cfg = build_strategy_config(min_sep, max_sep)
    val_results = []
    summary_results = {}

    for cost_mode in COST_MODES:
        period_rows = []
        for period_name, start, end in PERIODS:
            res = run_single(cfg, df_full, cost_mode, period_name, start, end)
            if res is not None:
                val_results.append(res)
                period_rows.append(res)

        summary = build_summary(period_rows)
        summary["cost_mode"] = cost_mode
        summary["min_separation"] = min_sep
        summary["max_separation"] = max_sep
        summary["batch2_score"] = compute_batch2_score(summary)
        summary_results[cost_mode] = summary

    _param_cache[key] = (val_results, summary_results)
    return val_results, summary_results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch 2: Optuna TPE pivot separation search")
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS, help="Optuna trials")
    parser.add_argument("--seed", type=int, default=42, help="TPESampler seed")
    args = parser.parse_args()

    print("Loading data...")
    df_full = load_ohlcv_csv(args.data)
    print(f"Loaded {len(df_full)} bars, {df_full.index[0]} to {df_full.index[-1]}")

    # Global collectors for report generation
    validation_rows = []
    summary_rows = []

    # ------------------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------------------
    def objective(trial: optuna.Trial):
        min_sep = trial.suggest_int("min_separation", 5, 10)
        max_sep = trial.suggest_int("max_separation", 30, 50)

        val_results, summary_results = evaluate_combo(min_sep, max_sep, df_full)

        # Attach trial_id and store for reporting
        for v in val_results:
            row = v.copy()
            row["trial_id"] = trial.number
            validation_rows.append(row)

        for cost_mode, s in summary_results.items():
            row = s.copy()
            row["trial_id"] = trial.number
            summary_rows.append(row)

        return summary_results["cost"]["batch2_score"]

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print(f"\nOptuna study complete. Best trial: {study.best_trial.number}")
    print(f"  params = {study.best_trial.params}")
    print(f"  value  = {study.best_trial.value:.6f}")

    # ------------------------------------------------------------------
    # Explicitly evaluate reference combos for comparison
    # ------------------------------------------------------------------
    def eval_reference(min_sep, max_sep, label):
        """Evaluate a reference combo and return its cost summary dict."""
        val_results, summary_results = evaluate_combo(min_sep, max_sep, df_full)
        # Append to global collectors with a string trial_id so they appear in CSVs
        for v in val_results:
            row = v.copy()
            row["trial_id"] = label
            validation_rows.append(row)
        for cost_mode, s in summary_results.items():
            row = s.copy()
            row["trial_id"] = label
            summary_rows.append(row)
        return summary_results["cost"]

    baseline_cost = eval_reference(*BASELINE_COMBO, "baseline")
    batch1_cost = eval_reference(*BATCH1_BEST_COMBO, "batch1_best")

    # ------------------------------------------------------------------
    # Build DataFrames
    # ------------------------------------------------------------------
    val_df = pd.DataFrame(validation_rows)
    summary_df = pd.DataFrame(summary_rows)

    # Ensure numeric profit_factor in validation
    val_df["profit_factor"] = pd.to_numeric(val_df["profit_factor"], errors="coerce").fillna(0.0)

    # Sort / top10
    cost_summary = summary_df[summary_df["cost_mode"] == "cost"].copy()
    nocost_summary = summary_df[summary_df["cost_mode"] == "no-cost"].copy()

    top10_cost = cost_summary.sort_values("batch2_score", ascending=False).head(10)
    top10_nocost = nocost_summary.sort_values("batch2_score", ascending=False).head(10)

    # ------------------------------------------------------------------
    # Save reports
    # ------------------------------------------------------------------
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    val_path = reports_dir / "batch2_optuna_pivot_validation.csv"
    summary_path = reports_dir / "batch2_optuna_pivot_summary.csv"
    top10_cost_path = reports_dir / "batch2_optuna_pivot_top10_cost.csv"
    top10_nocost_path = reports_dir / "batch2_optuna_pivot_top10_nocost.csv"

    val_df.to_csv(val_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    top10_cost.to_csv(top10_cost_path, index=False)
    top10_nocost.to_csv(top10_nocost_path, index=False)

    print(f"\nReports saved:")
    print(f"  {val_path}")
    print(f"  {summary_path}")
    print(f"  {top10_cost_path}")
    print(f"  {top10_nocost_path}")

    # ------------------------------------------------------------------
    # Final analysis output
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Batch 2 — Optuna Pivot Separation Final Analysis")
    print("=" * 60)

    print(f"\n1. Optuna 搜索完成: YES ({args.n_trials} trials)")
    print(f"2. 独立参数组合数: {len(_param_cache)} (含 reference combos)")
    print(f"3. 报告文件生成:")
    print(f"   - batch2_optuna_pivot_validation.csv:   {val_path.exists()}")
    print(f"   - batch2_optuna_pivot_summary.csv:      {summary_path.exists()}")
    print(f"   - batch2_optuna_pivot_top10_cost.csv:   {top10_cost_path.exists()}")
    print(f"   - batch2_optuna_pivot_top10_nocost.csv: {top10_nocost_path.exists()}")

    # Top10 cost table
    print("\n4. Cost 版本 Top 10 (含 reference combos):")
    cols = [
        "trial_id", "min_separation", "max_separation",
        "batch2_score", "positive_ratio", "avg_profit_factor", "min_profit_factor",
        "avg_quarter_return", "min_quarter_return", "max_quarter_drawdown_worst",
        "min_trades_per_period",
    ]
    print(top10_cost[cols].to_string(index=False))

    # Top10 no-cost table
    print("\n5. No-cost 版本 Top 10 (含 reference combos):")
    print(top10_nocost[cols].to_string(index=False))

    # Comparison table
    best_trial = study.best_trial
    best_cost = cost_summary[cost_summary["trial_id"] == best_trial.number]
    if len(best_cost) == 0:
        # fallback: rebuild best from cache
        _, best_sums = evaluate_combo(best_trial.params["min_separation"], best_trial.params["max_separation"], df_full)
        best_cost_row = best_sums["cost"]
    else:
        best_cost_row = best_cost.iloc[0].to_dict()

    print("\n6. 三组参数对比 (cost 版本):")
    print("-" * 80)
    print(f"{'名称':<15} {'min_sep':>8} {'max_sep':>8} {'avg_pf':>8} {'pos_ratio':>10} {'batch2_score':>12}")
    print("-" * 80)

    def fmt_row(name, row):
        return (
            f"{name:<15} "
            f"{row['min_separation']:>8} "
            f"{row['max_separation']:>8} "
            f"{row['avg_profit_factor']:>8.3f} "
            f"{row['positive_ratio']:>10.2f} "
            f"{row['batch2_score']:>12.4f}"
        )

    print(fmt_row("Baseline", baseline_cost))
    print(fmt_row("Batch1 Best", batch1_cost))
    print(fmt_row("Batch2 Best", best_cost_row))
    print("-" * 80)

    # Conclusion
    best_score = best_cost_row["batch2_score"]
    batch1_score = batch1_cost["batch2_score"]
    baseline_score = baseline_cost["batch2_score"]

    print("\n7. 结论:")
    if best_score > batch1_score + 0.05 and best_cost_row["positive_ratio"] >= batch1_cost["positive_ratio"]:
        print("   ✅ Batch2 找到的组合稳定优于 Batch1 8/40 与 baseline 3/20")
    elif best_score > batch1_score + 0.02:
        print("   🟡 Batch2 最优组合略优于 Batch1，但优势不明显")
    elif best_score > baseline_score + 0.05:
        print("   🟡 Batch2 最优组合优于 baseline，但未超越 Batch1 8/40")
    else:
        print("   ❌ Batch2 未找到优于 Batch1 8/40 的组合；baseline 3/20 仍稳健")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
