#!/usr/bin/env python3
"""
Phase 3C — Parameter Plateau Analysis around robust_ema55.

Local perturbation grid search (324 combos) across 8 quarters.
Generates both cost and no-cost reports.

Usage:
    python optimize/phase3c_plateau.py --data data/raw/BTCUSDT_5m_2024_2025.csv
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from argparse import ArgumentParser
import copy
import math
import itertools

import pandas as pd

from optimize.utils import safe_profit_factor, slice_dataframe, compute_extra_metrics
from joblib import Parallel, delayed

from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.vectorbt_engine import run_backtest


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

# Parameter grid
EMA_PERIODS = [50, 55, 60]
LOOKBACK_BARS = [20, 24, 28]
MIN_CLOSE_RATIOS = [0.75, 0.80, 0.85]
LEFT_BARS = [3, 4]
RIGHT_BARS = [3, 4]
RR_TARGETS = [2.0, 2.2, 2.4]

FIXED_PARAMS = {
    "oversold": 15,
    "overbought": 85,
    "stop_buffer": 0,
    "min_separation": 3,
    "max_separation": 20,
    "setup_max_bars": 12,
}

BASELINE = {
    "ema_period": 55,
    "lookback_bars": 24,
    "min_close_ratio": 0.80,
    "left_bars": 4,
    "right_bars": 3,
    "rr_target": 2.2,
}








def build_config(param_combo: dict) -> dict:
    """Build a full settings dict from parameter combo."""
    return {
        "data": {"path": ""},
        "strategy": {
            "stochastic": {
                "k_period": 14,
                "d_period": 3,
                "smooth": 1,
                "oversold": FIXED_PARAMS["oversold"],
                "overbought": FIXED_PARAMS["overbought"],
            },
            "pivots": {
                "left_bars": param_combo["left_bars"],
                "right_bars": param_combo["right_bars"],
                "min_separation": FIXED_PARAMS["min_separation"],
                "max_separation": FIXED_PARAMS["max_separation"],
                "strict": True,
            },
            "trend": {
                "ema_period": param_combo["ema_period"],
                "lookback_bars": param_combo["lookback_bars"],
                "min_close_ratio": param_combo["min_close_ratio"],
            },
            "risk": {
                "atr_period": 14,
                "stop_buffer": float(FIXED_PARAMS["stop_buffer"]),
                "rr_target": param_combo["rr_target"],
            },
            "setup": {
                "setup_max_bars": FIXED_PARAMS["setup_max_bars"],
                "replace_same_side_setup": True,
                "invalidate_on_stop_anchor_break": True,
            },
        },
        "backtest": {
            "initial_cash": 100000,
            "fee_per_trade": 0.0,
            "slippage": 0.0,
            "allow_short": True,
        },
    }


def _run_one_combo(combo, df_periods, cost_overrides, total_runs):
    """Worker: run one parameter combo across all periods."""
    import sys
    from pathlib import Path
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    import pandas as pd
    from strategy.signal_builder import build_signals
    from backtest.vectorbt_engine import run_backtest

    ema_period, lookback_bars, min_close_ratio, left_bars, right_bars, rr_target = combo
    param_combo = {
        "ema_period": ema_period,
        "lookback_bars": lookback_bars,
        "min_close_ratio": min_close_ratio,
        "left_bars": left_bars,
        "right_bars": right_bars,
        "rr_target": rr_target,
    }
    settings = build_config(param_combo)
    for k, v in cost_overrides.items():
        settings["backtest"][k] = v

    rows = []
    for period_name, start_str, end_str, df_slice in df_periods:
        if len(df_slice) < 100:
            row = {
                **param_combo,
                "period": period_name,
                "start_time": start_str,
                "end_time": end_str,
                "total_bars": len(df_slice),
                "oversold": FIXED_PARAMS["oversold"],
                "overbought": FIXED_PARAMS["overbought"],
                "stop_buffer": FIXED_PARAMS["stop_buffer"],
                "total_return": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "avg_trade": 0.0,
                "expectancy": 0.0,
                "long_trades": 0,
                "short_trades": 0,
                "target_exits": 0,
                "stop_exits": 0,
                "end_of_data_exits": 0,
                "avg_bars_held": 0.0,
                "median_bars_held": 0.0,
            }
            rows.append(row)
            continue

        bundle = build_signals(df_slice, settings["strategy"])
        result = run_backtest(df_slice, bundle, settings["backtest"])
        summary = pd.Series(result.summary)
        extras = compute_extra_metrics(result)

        row = {
            **param_combo,
            "period": period_name,
            "start_time": start_str,
            "end_time": end_str,
            "total_bars": len(df_slice),
            "oversold": FIXED_PARAMS["oversold"],
            "overbought": FIXED_PARAMS["overbought"],
            "stop_buffer": FIXED_PARAMS["stop_buffer"],
            "total_return": summary.get("total_return", 0.0),
            "total_trades": summary.get("total_trades", 0),
            "win_rate": summary.get("win_rate", 0.0),
            "profit_factor": summary.get("profit_factor", 0.0),
            "max_drawdown": summary.get("max_drawdown", 0.0),
            "avg_trade": summary.get("avg_trade", 0.0),
            "expectancy": summary.get("expectancy", 0.0),
            **extras,
        }
        rows.append(row)

    return rows


def run_plateau_analysis(df: pd.DataFrame, cost_overrides: dict, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all 324 combos across 8 quarters. Returns (validation_df, summary_df)."""

    param_combos = list(itertools.product(
        EMA_PERIODS,
        LOOKBACK_BARS,
        MIN_CLOSE_RATIOS,
        LEFT_BARS,
        RIGHT_BARS,
        RR_TARGETS,
    ))

    total_runs = len(param_combos) * len(PERIODS)
    print("=" * 60)
    print(f"Phase 3C — Plateau Analysis ({label})")
    print("=" * 60)
    print(f"Parameter combos: {len(param_combos)}")
    print(f"Periods:          {len(PERIODS)}")
    print(f"Total runs:       {total_runs}")
    if cost_overrides:
        print(f"Cost overrides:   {cost_overrides}")
    print(f"Parallel jobs:    -1 (auto)")
    print("=" * 60)

    # Pre-slice data into periods to avoid repeated slicing
    df_periods = [
        (period_name, start_str, end_str, slice_dataframe(df, start_str, end_str))
        for period_name, start_str, end_str in PERIODS
    ]

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(_run_one_combo)(combo, df_periods, cost_overrides, total_runs)
        for combo in param_combos
    )

    validation_rows = [row for sublist in results for row in sublist]
    validation_df = pd.DataFrame(validation_rows)

    # Build summary: one row per parameter combo
    summary_rows = []
    grouped = validation_df.groupby([
        "ema_period", "lookback_bars", "min_close_ratio",
        "left_bars", "right_bars", "rr_target"
    ])

    for (ema_period, lookback_bars, min_close_ratio, left_bars, right_bars, rr_target), cand_df in grouped:
        total_periods = len(cand_df)
        positive_periods = int((cand_df["total_return"] > 0).sum())
        positive_ratio = positive_periods / total_periods if total_periods > 0 else 0.0

        total_return_sum = float(cand_df["total_return"].sum())
        total_trades_sum = int(cand_df["total_trades"].sum())

        avg_quarter_return = float(cand_df["total_return"].mean())
        median_quarter_return = float(cand_df["total_return"].median())
        min_quarter_return = float(cand_df["total_return"].min())

        max_quarter_drawdown_worst = float(cand_df["max_drawdown"].min())

        pf_values = cand_df["profit_factor"].apply(_safe_pf)
        avg_profit_factor = float(pf_values.mean())
        median_profit_factor = float(pf_values.median())
        min_profit_factor = float(pf_values.min())

        min_trades_per_period = int(cand_df["total_trades"].min())
        avg_trades_per_period = float(cand_df["total_trades"].mean())

        long_trades_total = int(cand_df["long_trades"].sum())
        short_trades_total = int(cand_df["short_trades"].sum())

        if min_trades_per_period < 8:
            plateau_score = -9999.0
        else:
            avg_pf_score = min(avg_profit_factor, 5.0) / 5.0
            min_pf_score = min(min_profit_factor, 5.0) / 5.0
            avg_return_score = max(-1.0, min(1.0, avg_quarter_return / 5.0))
            trade_distribution_score = min(min_trades_per_period / 20.0, 1.0)
            drawdown_penalty = abs(max_quarter_drawdown_worst) / 20.0

            plateau_score = (
                0.25 * avg_pf_score
                + 0.20 * positive_ratio
                + 0.20 * min_pf_score
                + 0.15 * avg_return_score
                + 0.10 * trade_distribution_score
                - 0.10 * drawdown_penalty
            )

        summary_rows.append({
            "ema_period": ema_period,
            "lookback_bars": lookback_bars,
            "min_close_ratio": min_close_ratio,
            "left_bars": left_bars,
            "right_bars": right_bars,
            "rr_target": rr_target,
            "total_periods": total_periods,
            "positive_periods": positive_periods,
            "positive_ratio": positive_ratio,
            "total_return_sum": total_return_sum,
            "total_trades_sum": total_trades_sum,
            "avg_quarter_return": avg_quarter_return,
            "median_quarter_return": median_quarter_return,
            "min_quarter_return": min_quarter_return,
            "max_quarter_drawdown_worst": max_quarter_drawdown_worst,
            "avg_profit_factor": avg_profit_factor,
            "median_profit_factor": median_profit_factor,
            "min_profit_factor": min_profit_factor,
            "min_trades_per_period": min_trades_per_period,
            "avg_trades_per_period": avg_trades_per_period,
            "long_trades_total": long_trades_total,
            "short_trades_total": short_trades_total,
            "plateau_score": plateau_score,
        })

    summary_df = pd.DataFrame(summary_rows)
    # Sort by plateau_score descending
    summary_df = summary_df.sort_values("plateau_score", ascending=False).reset_index(drop=True)
    return validation_df, summary_df


def main() -> None:
    parser = ArgumentParser(description="Phase 3C — Parameter Plateau Analysis")
    parser.add_argument("--data", dest="data_path", default="data/raw/BTCUSDT_5m_2024_2025.csv",
                        help="Path to OHLCV CSV file")
    args = parser.parse_args()

    data_path = args.data_path
    df = load_ohlcv_csv(data_path)

    print(f"Data path:  {data_path}")
    print(f"Full range: {df.index[0]} ~ {df.index[-1]}")
    print(f"Total bars: {len(df)}")
    print()

    Path("reports").mkdir(exist_ok=True)

    # Run cost version only (nocost already done)
    cost_overrides = {"fee_per_trade": 5.0, "slippage": 5.0}
    cost_validation, cost_summary = run_plateau_analysis(df, cost_overrides, "cost")
    cost_validation.to_csv("reports/phase3c_plateau_validation_cost.csv", index=False)
    cost_summary.to_csv("reports/phase3c_plateau_summary_cost.csv", index=False)

    # Load nocost for final analysis
    nocost_summary = pd.read_csv("reports/phase3c_plateau_summary_nocost.csv")

    # Final analysis output
    print("\n" + "=" * 60)
    print("Phase 3C — Final Analysis")
    print("=" * 60)

    # 1. Success
    print("\n1. 是否成功运行: YES")

    # 2. Total combos
    total_combos = len(cost_summary)
    print(f"2. 总组合数: {total_combos} (expected 324)")

    # 3. Cost version complete
    print(f"3. 成本版本是否完成: YES ({len(cost_validation)} validation rows)")

    # 4. Top 30 plateau_score
    print("\n4. plateau_score Top 30 (cost):")
    top30 = cost_summary.head(30)
    display_cols = ["ema_period", "lookback_bars", "min_close_ratio", "left_bars", "right_bars", "rr_target",
                    "positive_ratio", "avg_profit_factor", "min_profit_factor", "min_trades_per_period", "plateau_score"]
    print(top30[display_cols].to_string(index=True))

    # 5. Baseline ranking
    baseline_mask = (
        (cost_summary["ema_period"] == BASELINE["ema_period"]) &
        (cost_summary["lookback_bars"] == BASELINE["lookback_bars"]) &
        (cost_summary["min_close_ratio"] == BASELINE["min_close_ratio"]) &
        (cost_summary["left_bars"] == BASELINE["left_bars"]) &
        (cost_summary["right_bars"] == BASELINE["right_bars"]) &
        (cost_summary["rr_target"] == BASELINE["rr_target"])
    )
    baseline_rows = cost_summary[baseline_mask]
    if len(baseline_rows) > 0:
        baseline_rank = int(baseline_rows.index[0]) + 1
        baseline_score = float(baseline_rows.iloc[0]["plateau_score"])
        print(f"\n5. 基准 robust_ema55 在 324 组中的排名: {baseline_rank} (score={baseline_score:.4f})")
    else:
        print("\n5. 基准 robust_ema55 在 324 组中的排名: NOT FOUND")

    # 6. How many combos meet criteria
    meets_criteria = cost_summary[
        (cost_summary["positive_ratio"] >= 0.75) &
        (cost_summary["min_trades_per_period"] >= 8) &
        (cost_summary["avg_profit_factor"] >= 1.5) &
        (cost_summary["min_profit_factor"] >= 1.0)
    ]
    print(f"\n6. 满足条件的组合数 (positive_ratio>=0.75, min_trades>=8, avg_pf>=1.5, min_pf>=1.0): {len(meets_criteria)} / {total_combos}")

    # 7. Top 30 concentration
    print("\n7. Top 30 参数分布 (cost):")
    for col in ["ema_period", "lookback_bars", "min_close_ratio", "rr_target"]:
        print(f"   {col}: {top30[col].value_counts().to_dict()}")

    # 8. Plateau check
    print("\n8. 参数高原判断:")

    def _param_works(col, values):
        good_vals = set()
        for v in values:
            subset = meets_criteria[meets_criteria[col] == v]
            if len(subset) > 0:
                good_vals.add(v)
        return good_vals

    good_ema = _param_works("ema_period", EMA_PERIODS)
    good_lookback = _param_works("lookback_bars", LOOKBACK_BARS)
    good_mcr = _param_works("min_close_ratio", MIN_CLOSE_RATIOS)
    good_rr = _param_works("rr_target", RR_TARGETS)

    print(f"   ema_period {EMA_PERIODS} 都能工作? {'YES' if good_ema == set(EMA_PERIODS) else 'NO'} (working: {sorted(good_ema)})")
    print(f"   lookback {LOOKBACK_BARS} 都能工作? {'YES' if good_lookback == set(LOOKBACK_BARS) else 'NO'} (working: {sorted(good_lookback)})")
    print(f"   mcr {MIN_CLOSE_RATIOS} 都能工作? {'YES' if good_mcr == set(MIN_CLOSE_RATIOS) else 'NO'} (working: {sorted(good_mcr)})")
    print(f"   rr {RR_TARGETS} 都能工作? {'YES' if good_rr == set(RR_TARGETS) else 'NO'} (working: {sorted(good_rr)})")

    # 9. Overfitting check
    print("\n9. 单点过拟合迹象:")
    if len(top30) > 0:
        top1 = cost_summary.iloc[0]
        top1_is_baseline = (
            top1["ema_period"] == BASELINE["ema_period"] and
            top1["lookback_bars"] == BASELINE["lookback_bars"] and
            top1["min_close_ratio"] == BASELINE["min_close_ratio"] and
            top1["left_bars"] == BASELINE["left_bars"] and
            top1["right_bars"] == BASELINE["right_bars"] and
            top1["rr_target"] == BASELINE["rr_target"]
        )
        if top1_is_baseline:
            print("   Top 1 是基准参数: YES")
        else:
            print("   Top 1 是基准参数: NO")
            print(f"   Top 1: ema={top1['ema_period']} lb={top1['lookback_bars']} mcr={top1['min_close_ratio']} "
                  f"l={top1['left_bars']} r={top1['right_bars']} rr={top1['rr_target']} score={top1['plateau_score']:.4f}")

        # Check if top scores are very close to each other (plateau indicator)
        if len(top30) >= 5:
            top5_scores = cost_summary.head(5)["plateau_score"].tolist()
            score_spread = top5_scores[0] - top5_scores[4]
            if score_spread < 0.05:
                print(f"   Top 5 分数差 < 0.05: YES (spread={score_spread:.4f}) -> 高原迹象")
            else:
                print(f"   Top 5 分数差 < 0.05: NO (spread={score_spread:.4f}) -> 可能有过拟合")

        # Check if baseline rank is high
        if baseline_rank <= 10:
            print(f"   基准排名 Top 10: YES (rank={baseline_rank})")
        elif baseline_rank <= 30:
            print(f"   基准排名 Top 30: YES (rank={baseline_rank})")
        else:
            print(f"   基准排名 Top 30: NO (rank={baseline_rank}) -> 过拟合风险")

    # Overall judgment
    print("\n" + "=" * 60)
    print("Phase 3C 综合判断")
    print("=" * 60)

    plateau_indicators = 0
    total_indicators = 5

    if len(meets_criteria) >= 50:
        plateau_indicators += 1
        print(f"✓ 大量组合通过门槛 ({len(meets_criteria)} >= 50)")
    else:
        print(f"✗ 通过门槛组合偏少 ({len(meets_criteria)} < 50)")

    if baseline_rank <= 30:
        plateau_indicators += 1
        print(f"✓ 基准排名前列 (rank={baseline_rank} <= 30)")
    else:
        print(f"✗ 基准排名靠后 (rank={baseline_rank} > 30)")

    if good_ema == set(EMA_PERIODS):
        plateau_indicators += 1
        print("✓ EMA 参数形成高原 (50/55/60 都能工作)")
    else:
        print(f"✗ EMA 参数未形成高原 (working: {sorted(good_ema)})")

    if good_lookback == set(LOOKBACK_BARS):
        plateau_indicators += 1
        print("✓ Lookback 参数形成高原 (20/24/28 都能工作)")
    else:
        print(f"✗ Lookback 参数未形成高原 (working: {sorted(good_lookback)})")

    if good_mcr == set(MIN_CLOSE_RATIOS):
        plateau_indicators += 1
        print("✓ MCR 参数形成高原 (0.75/0.80/0.85 都能工作)")
    else:
        print(f"✗ MCR 参数未形成高原 (working: {sorted(good_mcr)})")

    print(f"\n高原指标: {plateau_indicators}/{total_indicators}")
    if plateau_indicators >= 4:
        print("结论: 参数高原成立。建议进入 Phase 4 执行压力测试。")
    elif plateau_indicators >= 3:
        print("结论: 参数高原部分成立。可谨慎进入 Phase 4，但需关注敏感参数。")
    else:
        print("结论: 参数高原不成立。策略可能存在过拟合，建议回退研究。")


if __name__ == "__main__":
    main()
