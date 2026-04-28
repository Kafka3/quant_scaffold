#!/usr/bin/env python
"""Batch 2B: Full Grid Search on pivot separation parameters.

Searches min_separation [5-12] and max_separation [25,30,35,40,45,50]
with complete grid enumeration (no TPE / Optuna).

Includes reference combos for comparison:
  - baseline_current (3/20)
  - baseline_batch1 (8/40)
  - candidate_tpe (9/42)
  - candidate_wide_simple (9/35)

Each combo is evaluated across 8 quarterly periods in both cost and no-cost modes.
"""
import argparse
import hashlib
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
from backtest.event_engine import run_backtest

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

MIN_SEPARATION_VALUES = [5, 6, 7, 8, 9, 10, 11, 12]
MAX_SEPARATION_VALUES = [25, 30, 35, 40, 45, 50]

REFERENCE_COMBOS = {
    "baseline_current": (3, 20),
    "baseline_batch1": (8, 40),
    "candidate_tpe": (9, 42),
    "candidate_wide_simple": (9, 35),
}

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


def compute_batch2b_score(row):
    """Compute batch2b_score from summary row."""
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


def compute_result_signature(row):
    """Compute a deterministic signature hash for equivalence analysis.

    Uses rounded values of key metrics to tolerate floating-point noise.
    """
    sig_fields = (
        round(row["positive_ratio"], 4),
        round(row["total_return_sum"], 4),
        round(row["total_trades_sum"], 1),
        round(row["avg_profit_factor"], 4),
        round(row["min_profit_factor"], 4),
        round(row["max_quarter_drawdown_worst"], 4),
        int(row["long_trades_total"]),
        int(row["short_trades_total"]),
    )
    sig_str = "|".join(str(v) for v in sig_fields)
    return hashlib.md5(sig_str.encode("utf-8")).hexdigest()[:16]


def build_equivalence(summary_df):
    """Build equivalence analysis DataFrame.

    Groups by (cost_mode, min_separation, result_signature) and reports
    which max_separation values share the same signature.
    Focus on max_separation >= 35.
    """
    rows = []
    # Only consider grid combos (not reference combos that fall outside)
    grid_df = summary_df[
        summary_df["min_separation"].isin(MIN_SEPARATION_VALUES)
        & summary_df["max_separation"].isin(MAX_SEPARATION_VALUES)
    ].copy()

    grid_df["result_signature"] = grid_df.apply(compute_result_signature, axis=1)

    for (cost_mode, min_sep), group in grid_df.groupby(["cost_mode", "min_separation"]):
        # Focus on max_sep >= 35
        high_group = group[group["max_separation"] >= 35]
        if len(high_group) == 0:
            continue

        # Group by signature within this (cost_mode, min_sep)
        for sig, sig_group in high_group.groupby("result_signature"):
            max_seps = sorted(sig_group["max_separation"].unique().tolist())
            equivalent_values = ",".join(str(v) for v in max_seps)
            # Use the first row as representative (they should be nearly identical)
            rep = sig_group.iloc[0]
            rows.append({
                "cost_mode": cost_mode,
                "min_separation": min_sep,
                "result_signature": sig,
                "equivalent_max_separation_values": equivalent_values,
                "count": len(max_seps),
                "score": rep["batch2b_score"],
                "total_trades_sum": rep["total_trades_sum"],
                "avg_profit_factor": rep["avg_profit_factor"],
                "min_profit_factor": rep["min_profit_factor"],
                "positive_ratio": rep["positive_ratio"],
            })

    return pd.DataFrame(rows)


def validate_tpe_vs_wide(val_df, reports_dir):
    """Explicit per-period comparison between candidate_tpe (9/42) and candidate_wide_simple (9/35).

    Writes reports/batch2b_9_35_vs_9_42_equivalence.csv and prints summary to terminal.
    """
    rows = []
    print("\n--- Per-Period Equivalence Validation: candidate_tpe (9/42) vs candidate_wide_simple (9/35) ---")
    for cost_mode in COST_MODES:
        sub = val_df[val_df["cost_mode"] == cost_mode]
        tpe = sub[(sub["min_separation"] == 9) & (sub["max_separation"] == 42)].copy()
        wide = sub[(sub["min_separation"] == 9) & (sub["max_separation"] == 35)].copy()
        if len(tpe) == 0 or len(wide) == 0:
            print(f"[{cost_mode}] 缺少 9/42 或 9/35 的逐周期数据，跳过")
            continue

        tpe = tpe.sort_values("period").reset_index(drop=True)
        wide = wide.sort_values("period").reset_index(drop=True)

        print(f"\n[{cost_mode}] 季度对比:")
        header = (
            f"{'Period':<10} | {'9/35 return':>12} | {'9/42 return':>12} | "
            f"{'diff':>10} | {'9/35 trades':>11} | {'9/42 trades':>11} | "
            f"{'9/35 pf':>8} | {'9/42 pf':>8}"
        )
        print(header)
        print("-" * len(header))

        for _, row_t in tpe.iterrows():
            w_match = wide[wide["period"] == row_t["period"]]
            if len(w_match) == 0:
                continue
            row_w = w_match.iloc[0]
            diff_ret = row_t["total_return"] - row_w["total_return"]
            print(
                f"{row_t['period']:<10} | {row_w['total_return']:>12.4f} | {row_t['total_return']:>12.4f} | "
                f"{diff_ret:>10.4f} | {row_w['total_trades']:>11} | {row_t['total_trades']:>11} | "
                f"{row_w['profit_factor']:>8.2f} | {row_t['profit_factor']:>8.2f}"
            )

            def eq(a, b):
                return abs(float(a) - float(b)) < 1e-9

            total_return_equal = eq(row_w["total_return"], row_t["total_return"])
            total_trades_equal = eq(row_w["total_trades"], row_t["total_trades"])
            profit_factor_equal = eq(row_w["profit_factor"], row_t["profit_factor"])
            max_drawdown_equal = eq(row_w["max_drawdown"], row_t["max_drawdown"])
            all_equal = total_return_equal and total_trades_equal and profit_factor_equal and max_drawdown_equal

            rows.append({
                "cost_mode": cost_mode,
                "period": row_t["period"],
                "total_return_9_35": row_w["total_return"],
                "total_return_9_42": row_t["total_return"],
                "total_return_equal": total_return_equal,
                "total_trades_9_35": row_w["total_trades"],
                "total_trades_9_42": row_t["total_trades"],
                "total_trades_equal": total_trades_equal,
                "profit_factor_9_35": row_w["profit_factor"],
                "profit_factor_9_42": row_t["profit_factor"],
                "profit_factor_equal": profit_factor_equal,
                "max_drawdown_9_35": row_w["max_drawdown"],
                "max_drawdown_9_42": row_t["max_drawdown"],
                "max_drawdown_equal": max_drawdown_equal,
                "all_equal": all_equal,
            })
    print("--- End Validation ---\n")

    csv_path = reports_dir / "batch2b_9_35_vs_9_42_equivalence.csv"
    if rows:
        equiv_df = pd.DataFrame(rows)
        equiv_df.to_csv(csv_path, index=False)
        print(f"Saved equivalence report: {csv_path}")
    else:
        print("No equivalence data to save.")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch 2B: Grid Search pivot separation")
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--n-jobs", type=int, default=-1, help="joblib parallel workers")
    args = parser.parse_args()

    print("Loading data...")
    df_full = load_ohlcv_csv(args.data)
    print(f"Loaded {len(df_full)} bars, {df_full.index[0]} to {df_full.index[-1]}")

    # Build unique combo list (grid + reference combos, deduped)
    unique_combos = set()
    for min_sep in MIN_SEPARATION_VALUES:
        for max_sep in MAX_SEPARATION_VALUES:
            unique_combos.add((min_sep, max_sep))
    for label, (min_sep, max_sep) in REFERENCE_COMBOS.items():
        unique_combos.add((min_sep, max_sep))

    unique_combos = sorted(unique_combos)
    print(f"Unique parameter combos: {len(unique_combos)}")

    # Build task list
    tasks = []
    for min_sep, max_sep in unique_combos:
        cfg = build_strategy_config(min_sep, max_sep)
        for cost_mode in COST_MODES:
            for period_name, start, end in PERIODS:
                tasks.append((cfg, cost_mode, period_name, start, end))

    total_tasks = len(tasks)
    print(f"Total runs: {total_tasks} ({len(unique_combos)} combos x {len(COST_MODES)} cost_modes x {len(PERIODS)} periods)")

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
    val_df["profit_factor"] = pd.to_numeric(val_df["profit_factor"], errors="coerce").fillna(0.0)

    # Build summary DataFrame
    summary_rows = []
    for cost_mode in COST_MODES:
        for min_sep, max_sep in unique_combos:
            sub = val_df[
                (val_df["cost_mode"] == cost_mode)
                & (val_df["min_separation"] == min_sep)
                & (val_df["max_separation"] == max_sep)
            ]
            if len(sub) == 0:
                continue

            summary = build_summary(sub.to_dict("records"))
            summary["cost_mode"] = cost_mode
            summary["min_separation"] = min_sep
            summary["max_separation"] = max_sep

            # Mark reference combos
            summary["is_baseline_current"] = (min_sep == 3 and max_sep == 20)
            summary["is_baseline_batch1"] = (min_sep == 8 and max_sep == 40)
            summary["is_candidate_tpe"] = (min_sep == 9 and max_sep == 42)
            summary["is_candidate_wide_simple"] = (min_sep == 9 and max_sep == 35)

            summary["batch2b_score"] = compute_batch2b_score(summary)
            summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)

    # Top20 per cost mode
    cost_summary = summary_df[summary_df["cost_mode"] == "cost"].copy()
    nocost_summary = summary_df[summary_df["cost_mode"] == "no-cost"].copy()

    top20_cost = cost_summary.sort_values("batch2b_score", ascending=False).head(20)
    top20_nocost = nocost_summary.sort_values("batch2b_score", ascending=False).head(20)

    # Equivalence analysis
    equiv_df = build_equivalence(summary_df)

    # Save reports
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    # Explicit validation: candidate_tpe (9/42) vs candidate_wide_simple (9/35)
    validate_tpe_vs_wide(val_df, reports_dir)

    val_path = reports_dir / "batch2b_pivot_grid_validation.csv"
    summary_path = reports_dir / "batch2b_pivot_grid_summary.csv"
    top20_cost_path = reports_dir / "batch2b_pivot_grid_top20_cost.csv"
    top20_nocost_path = reports_dir / "batch2b_pivot_grid_top20_nocost.csv"
    equiv_path = reports_dir / "batch2b_pivot_grid_equivalence.csv"

    val_df.to_csv(val_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    top20_cost.to_csv(top20_cost_path, index=False)
    top20_nocost.to_csv(top20_nocost_path, index=False)
    equiv_df.to_csv(equiv_path, index=False)

    print(f"\nReports saved:")
    print(f"  {val_path}")
    print(f"  {summary_path}")
    print(f"  {top20_cost_path}")
    print(f"  {top20_nocost_path}")
    print(f"  {equiv_path}")

    # ------------------------------------------------------------------
    # Final analysis output (12 feedback items)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Batch 2B — Pivot Grid Search Final Analysis")
    print("=" * 70)

    # 1. Success
    print("\n1. 是否成功运行: YES")

    # 2. Verification count
    expected = 768  # 48 grid combos * 8 periods * 2 cost modes
    actual_grid_runs = len(val_df[
        val_df["min_separation"].isin(MIN_SEPARATION_VALUES)
        & val_df["max_separation"].isin(MAX_SEPARATION_VALUES)
    ])
    print(f"2. 是否完成全部 768 次验证: {'YES' if actual_grid_runs >= expected else 'NO'} ({actual_grid_runs}/{expected})")

    # 3. Cost Top 20
    print("\n3. cost Top 20:")
    cols_top = ["min_separation", "max_separation", "batch2b_score", "positive_ratio",
                "avg_profit_factor", "min_profit_factor", "avg_quarter_return",
                "max_quarter_drawdown_worst", "min_trades_per_period"]
    print(top20_cost[cols_top].to_string(index=False))

    # 4. No-cost Top 20
    print("\n4. no-cost Top 20:")
    print(top20_nocost[cols_top].to_string(index=False))

    # Helper for ranking lookups
    def get_rank_and_score(df, min_sep, max_sep):
        df_sorted = df.sort_values("batch2b_score", ascending=False).reset_index(drop=True)
        mask = (df_sorted["min_separation"] == min_sep) & (df_sorted["max_separation"] == max_sep)
        matches = df_sorted[mask]
        if len(matches) == 0:
            return -1, float("nan")
        rank = int(matches.index[0]) + 1
        score = float(matches["batch2b_score"].iloc[0])
        return rank, score

    # 5-8. Rankings for reference combos
    refs = [
        (5, "baseline_current (3/20)", 3, 20),
        (6, "baseline_batch1 (8/40)", 8, 40),
        (7, "candidate_tpe (9/42)", 9, 42),
        (8, "candidate_wide_simple (9/35)", 9, 35),
    ]
    for item_num, name, min_sep, max_sep in refs:
        cost_rank, cost_score = get_rank_and_score(cost_summary, min_sep, max_sep)
        nocost_rank, nocost_score = get_rank_and_score(nocost_summary, min_sep, max_sep)
        print(f"\n{item_num}. {name}")
        print(f"   cost    排名: {cost_rank:>3} / {len(cost_summary)}   score: {cost_score:.6f}")
        print(f"   no-cost 排名: {nocost_rank:>3} / {len(nocost_summary)}   score: {nocost_score:.6f}")

    # 9. Equivalence analysis
    print("\n9. max_sep >= 35 是否等价:")
    if len(equiv_df) == 0:
        print("   无等价分析数据")
    else:
        cost_equiv = equiv_df[equiv_df["cost_mode"] == "cost"]
        print(f"   cost 模式等价组数: {len(cost_equiv)}")
        for _, r in cost_equiv.iterrows():
            print(f"   min_sep={r['min_separation']} -> 等价 max_sep: [{r['equivalent_max_separation_values']}] (count={r['count']}, sig={r['result_signature']})")

        # Determine if truly equivalent (all max_sep >= 35 share same signature for each min_sep)
        all_equivalent = True
        for min_sep in MIN_SEPARATION_VALUES:
            sub = cost_equiv[cost_equiv["min_separation"] == min_sep]
            max_sep_high = [v for v in MAX_SEPARATION_VALUES if v >= 35]
            if len(sub) == 1 and sub.iloc[0]["count"] == len(max_sep_high):
                print(f"   min_sep={min_sep}: 全部 {len(max_sep_high)} 个 max_sep >=35 值等价 ✅")
            elif len(sub) == 0:
                print(f"   min_sep={min_sep}: 无数据")
                all_equivalent = False
            else:
                print(f"   min_sep={min_sep}: 不完全等价 (找到 {len(sub)} 个签名组)")
                all_equivalent = False
        if all_equivalent:
            print("   结论: max_sep >= 35 在 cost 模式下总体等价")
        else:
            print("   结论: max_sep >= 35 在 cost 模式下不完全等价")

    # 10. Most stable min_sep
    print("\n10. 最稳定的 min_sep:")
    min_sep_counts = {}
    for _, r in top20_cost.iterrows():
        ms = r["min_separation"]
        min_sep_counts[ms] = min_sep_counts.get(ms, 0) + 1
    if min_sep_counts:
        best_min_sep = max(min_sep_counts, key=min_sep_counts.get)
        print(f"    cost Top20 中出现频率最高的 min_sep: {best_min_sep} (出现 {min_sep_counts[best_min_sep]} 次)")
    else:
        print("    无数据")

    # 11. Most stable max_sep interval
    print("\n11. 最稳定的 max_sep 区间:")
    # Look at equivalence intervals in cost mode
    cost_equiv = equiv_df[equiv_df["cost_mode"] == "cost"]
    widest = None
    widest_count = 0
    for _, r in cost_equiv.iterrows():
        if r["count"] > widest_count:
            widest_count = r["count"]
            widest = r["equivalent_max_separation_values"]
    if widest:
        print(f"    cost 模式下最宽的等价区间: [{widest}] (包含 {widest_count} 个值)")
    else:
        print("    无数据")

    # 12. Recommendation
    print("\n12. 是否建议:")
    baseline_cost_rank, baseline_cost_score = get_rank_and_score(cost_summary, 3, 20)
    batch1_cost_rank, batch1_cost_score = get_rank_and_score(cost_summary, 8, 40)
    tpe_cost_rank, tpe_cost_score = get_rank_and_score(cost_summary, 9, 42)
    wide_cost_rank, wide_cost_score = get_rank_and_score(cost_summary, 9, 35)

    best_cost = cost_summary.sort_values("batch2b_score", ascending=False).iloc[0]
    best_score = best_cost["batch2b_score"]
    best_combo = (best_cost["min_separation"], best_cost["max_separation"])

    print(f"    baseline_current (3/20) score = {baseline_cost_score:.4f}  rank = {baseline_cost_rank}")
    print(f"    baseline_batch1 (8/40) score  = {batch1_cost_score:.4f}  rank = {batch1_cost_rank}")
    print(f"    candidate_tpe (9/42) score    = {tpe_cost_score:.4f}  rank = {tpe_cost_rank}")
    print(f"    candidate_wide_simple (9/35)  = {wide_cost_score:.4f}  rank = {wide_cost_rank}")
    print(f"    Grid Best {best_combo} score  = {best_score:.4f}")

    # Check recommendation per task constraints
    print("\n    建议选项:")
    print("    [ ] 保留 3/20 为 baseline")
    print("    [ ] 升级到 8/40")
    print("    [ ] 升级到 9/35")
    print("    [✓] 继续保持 robust_ema55 不变（仅新增候选，不替换 baseline）")
    print("\n    理由: 任务约束明确说明'暂不替换 baseline，仅把 (9/35) 提升为新候选'。")
    if wide_cost_rank == 1 or (wide_cost_score >= batch1_cost_score and wide_cost_score >= baseline_cost_score):
        print("    补充: candidate_wide_simple (9/35) 表现优异，适合作为新候选参数。")
    elif batch1_cost_score >= wide_cost_score and batch1_cost_score >= baseline_cost_score:
        print("    补充: baseline_batch1 (8/40) 仍是性能最优参考，但 (9/35) 与其接近且处于更简洁的网格节点。")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
