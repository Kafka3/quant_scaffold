#!/usr/bin/env python
"""Batch 1: Stochastic period + Pivot separation local sensitivity test.

24 param combos x 8 quarters x 2 cost modes = 384 runs.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

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
STOCH_COMBOS = [
    (9, 1, 3),
    (14, 1, 3),
    (40, 1, 4),
    (40, 4, 4),
]

PIVOT_COMBOS = [
    (3, 15),
    (3, 20),
    (3, 30),
    (5, 20),
    (5, 30),
    (8, 40),
]

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

BASELINE = {
    "k_period": 14,
    "smooth": 1,
    "d_period": 3,
    "min_separation": 3,
    "max_separation": 20,
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def build_strategy_config(k_period, smooth, d_period, min_sep, max_sep):
    return {
        "stochastic": {
            "k_period": k_period,
            "d_period": d_period,
            "smooth": smooth,
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
        "k_period": cfg["stochastic"]["k_period"],
        "smooth": cfg["stochastic"]["smooth"],
        "d_period": cfg["stochastic"]["d_period"],
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


def compute_batch1_score(row):
    """Compute batch1_score from summary row."""
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


def main():
    parser = argparse.ArgumentParser(description="Batch 1: Stoch + Pivot sensitivity")
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--n-jobs", type=int, default=-1, help="joblib parallel workers")
    args = parser.parse_args()

    print("Loading data...")
    df_full = load_ohlcv_csv(args.data)
    print(f"Loaded {len(df_full)} bars, {df_full.index[0]} to {df_full.index[-1]}")

    # Build task list
    tasks = []
    for k_period, smooth, d_period in STOCH_COMBOS:
        for min_sep, max_sep in PIVOT_COMBOS:
            cfg = build_strategy_config(k_period, smooth, d_period, min_sep, max_sep)
            for cost_mode in COST_MODES:
                for period_name, start, end in PERIODS:
                    tasks.append((cfg, cost_mode, period_name, start, end))

    total_tasks = len(tasks)
    print(f"Total runs: {total_tasks} ({len(STOCH_COMBOS)} stoch x {len(PIVOT_COMBOS)} pivot x {len(COST_MODES)} cost x {len(PERIODS)} periods)")

    # Run in parallel
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_single)(cfg, df_full, cost_mode, period_name, start, end)
        for cfg, cost_mode, period_name, start, end in tasks
    )

    # Filter out None (empty slices)
    results = [r for r in results if r is not None]
    print(f"Completed {len(results)} / {total_tasks} runs")

    # Build validation DataFrame
    val_df = pd.DataFrame(results)
    # Ensure all profit_factor are numeric (replace inf with NaN then 0 for summary)
    val_df["profit_factor"] = pd.to_numeric(val_df["profit_factor"], errors="coerce").fillna(0.0)

    # Build summary DataFrame
    summary_rows = []
    for cost_mode in COST_MODES:
        for k_period, smooth, d_period in STOCH_COMBOS:
            for min_sep, max_sep in PIVOT_COMBOS:
                sub = val_df[
                    (val_df["cost_mode"] == cost_mode)
                    & (val_df["k_period"] == k_period)
                    & (val_df["smooth"] == smooth)
                    & (val_df["d_period"] == d_period)
                    & (val_df["min_separation"] == min_sep)
                    & (val_df["max_separation"] == max_sep)
                ]
                if len(sub) == 0:
                    continue

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

                is_baseline = (
                    k_period == BASELINE["k_period"]
                    and smooth == BASELINE["smooth"]
                    and d_period == BASELINE["d_period"]
                    and min_sep == BASELINE["min_separation"]
                    and max_sep == BASELINE["max_separation"]
                )

                row = {
                    "cost_mode": cost_mode,
                    "k_period": k_period,
                    "smooth": smooth,
                    "d_period": d_period,
                    "min_separation": min_sep,
                    "max_separation": max_sep,
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
                    "is_current_baseline": is_baseline,
                }
                row["batch1_score"] = compute_batch1_score(row)
                summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Sort for top20
    top20_cost = summary_df[summary_df["cost_mode"] == "cost"].sort_values("batch1_score", ascending=False).head(20)
    top20_nocost = summary_df[summary_df["cost_mode"] == "no-cost"].sort_values("batch1_score", ascending=False).head(20)

    # Save reports
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    val_path = reports_dir / "batch1_stoch_pivot_validation.csv"
    summary_path = reports_dir / "batch1_stoch_pivot_summary.csv"
    top20_cost_path = reports_dir / "batch1_stoch_pivot_top20_cost.csv"
    top20_nocost_path = reports_dir / "batch1_stoch_pivot_top20_nocost.csv"

    val_df.to_csv(val_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    top20_cost.to_csv(top20_cost_path, index=False)
    top20_nocost.to_csv(top20_nocost_path, index=False)

    print(f"\nReports saved:")
    print(f"  {val_path}")
    print(f"  {summary_path}")
    print(f"  {top20_cost_path}")
    print(f"  {top20_nocost_path}")

    # ------------------------------------------------------------------
    # Final analysis output
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Batch 1 — Stochastic + Pivot Sensitivity Final Analysis")
    print("=" * 60)

    print(f"\n1. 是否成功运行: YES ({len(results)} runs completed)")
    print(f"2. 是否完成 384 次验证: {'YES' if len(results) == 384 else 'NO'} ({len(results)}/384)")
    print(f"3. 报告文件生成:")
    print(f"   - batch1_stoch_pivot_validation.csv: {val_path.exists()}")
    print(f"   - batch1_stoch_pivot_summary.csv:    {summary_path.exists()}")
    print(f"   - batch1_stoch_pivot_top20_cost.csv: {top20_cost_path.exists()}")
    print(f"   - batch1_stoch_pivot_top20_nocost.csv: {top20_nocost_path.exists()}")

    def fmt_top20(df, title):
        print(f"\n4. {title}")
        cols = ["k_period", "smooth", "d_period", "min_separation", "max_separation",
                "batch1_score", "positive_ratio", "avg_profit_factor", "min_profit_factor",
                "avg_quarter_return", "min_quarter_return", "max_quarter_drawdown_worst",
                "min_trades_per_period", "is_current_baseline"]
        print(df[cols].to_string(index=False))

    fmt_top20(top20_cost, "cost 版本 Top 20")
    fmt_top20(top20_nocost, "no-cost 版本 Top 20")

    # Baseline ranks
    cost_df = summary_df[summary_df["cost_mode"] == "cost"].sort_values("batch1_score", ascending=False).reset_index(drop=True)
    nocost_df = summary_df[summary_df["cost_mode"] == "no-cost"].sort_values("batch1_score", ascending=False).reset_index(drop=True)

    baseline_cost_rank = cost_df[cost_df["is_current_baseline"]].index
    baseline_nocost_rank = nocost_df[nocost_df["is_current_baseline"]].index
    baseline_cost_rank = int(baseline_cost_rank[0]) + 1 if len(baseline_cost_rank) > 0 else -1
    baseline_nocost_rank = int(baseline_nocost_rank[0]) + 1 if len(baseline_nocost_rank) > 0 else -1

    print(f"\n5. 当前 baseline 在 cost 版本中的排名: {baseline_cost_rank} / {len(cost_df)}")
    print(f"6. 当前 baseline 在 no-cost 版本中的排名: {baseline_nocost_rank} / {len(nocost_df)}")

    # Stochastic stability
    print("\n7. Stochastic 组合整体稳定性 (cost):")
    for kp, sm, dp in STOCH_COMBOS:
        sub = cost_df[(cost_df["k_period"] == kp) & (cost_df["smooth"] == sm) & (cost_df["d_period"] == dp)]
        if len(sub) == 0:
            continue
        print(f"   ({kp},{sm},{dp}): avg_score={sub['batch1_score'].mean():.4f}  "
              f"avg_pf={sub['avg_profit_factor'].mean():.3f}  "
              f"min_pf={sub['min_profit_factor'].mean():.3f}  "
              f"pos_ratio={sub['positive_ratio'].mean():.3f}")

    # Pivot stability
    print("\n8. Pivot 间隔整体稳定性 (cost):")
    for ms, mx in PIVOT_COMBOS:
        sub = cost_df[(cost_df["min_separation"] == ms) & (cost_df["max_separation"] == mx)]
        if len(sub) == 0:
            continue
        print(f"   {ms}/{mx}: avg_score={sub['batch1_score'].mean():.4f}  "
              f"avg_pf={sub['avg_profit_factor'].mean():.3f}  "
              f"min_pf={sub['min_profit_factor'].mean():.3f}  "
              f"pos_ratio={sub['positive_ratio'].mean():.3f}")

    # Cost vs no-cost consistency
    print("\n9. cost vs no-cost 一致性:")
    merged = pd.merge(
        cost_df[["k_period", "smooth", "d_period", "min_separation", "max_separation", "batch1_score"]],
        nocost_df[["k_period", "smooth", "d_period", "min_separation", "max_separation", "batch1_score"]],
        on=["k_period", "smooth", "d_period", "min_separation", "max_separation"],
        suffixes=("_cost", "_nocost"),
    )
    corr = merged["batch1_score_cost"].corr(merged["batch1_score_nocost"])
    print(f"   评分相关系数: {corr:.4f}")

    # Best combo by stoch family
    print("\n10. 每个 Stochastic 组合最佳 pivot (cost):")
    for kp, sm, dp in STOCH_COMBOS:
        sub = cost_df[(cost_df["k_period"] == kp) & (cost_df["smooth"] == sm) & (cost_df["d_period"] == dp)]
        if len(sub) == 0:
            continue
        best = sub.iloc[0]
        print(f"   ({kp},{sm},{dp}) best={best['min_separation']}/{best['max_separation']} "
              f"score={best['batch1_score']:.4f} pos={best['positive_ratio']:.2f} "
              f"min_pf={best['min_profit_factor']:.2f}")

    # Satisfying combos
    print("\n11. 满足筛选条件的组合 (cost):")
    qualified = cost_df[
        (cost_df["positive_ratio"] >= 0.75)
        & (cost_df["min_trades_per_period"] >= 8)
        & (cost_df["avg_profit_factor"] >= 1.3)
        & (cost_df["min_profit_factor"] >= 1.0)
    ]
    if len(qualified) == 0:
        print("   无组合满足全部条件")
    else:
        print(f"   共 {len(qualified)} 个组合满足:")
        for _, r in qualified.iterrows():
            print(f"   Stoch({r['k_period']},{r['smooth']},{r['d_period']}) "
                  f"Pivot({r['min_separation']}/{r['max_separation']}) -> "
                  f"score={r['batch1_score']:.4f} pos={r['positive_ratio']:.2f} "
                  f"avg_pf={r['avg_profit_factor']:.2f} min_pf={r['min_profit_factor']:.2f} "
                  f"dd={r['max_quarter_drawdown_worst']:.2f}% "
                  f"baseline={r['is_current_baseline']}")

    # Recommendation
    print("\n12. 是否建议替换当前 baseline:")
    baseline_cost = cost_df[cost_df["is_current_baseline"]]
    baseline_score = float(baseline_cost["batch1_score"].iloc[0]) if len(baseline_cost) > 0 else -9999
    best_cost = cost_df.iloc[0]
    best_score = float(best_cost["batch1_score"])

    print(f"   baseline score = {baseline_score:.4f}")
    print(f"   best score     = {best_score:.4f}  (Stoch {best_cost['k_period']},{best_cost['smooth']},{best_cost['d_period']} + Pivot {best_cost['min_separation']}/{best_cost['max_separation']})")

    if best_score > baseline_score + 0.05 and best_cost["positive_ratio"] >= baseline_cost["positive_ratio"].iloc[0]:
        print("   ✅ 建议替换 baseline")
    elif best_score > baseline_score + 0.02:
        print("   🟡 可考虑替换，但优势不明显")
    else:
        print("   ❌ 不建议替换，baseline 仍是最稳选择")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
