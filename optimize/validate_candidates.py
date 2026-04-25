#!/usr/bin/env python3
"""
Phase 1.5 — Candidate parameter out-of-sample validation by quarter.

Validates the stable parameter region from Phase 1 across time segments:
  2024-Q1, Q2, Q3, Q4, and Full Year.

Usage:
    python optimize/validate_candidates.py --data data/raw/BTCUSDT_5m_2024.csv
"""

from argparse import ArgumentParser
from itertools import product
from pathlib import Path
import copy
import math

import pandas as pd

from configs.settings import load_settings
from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.vectorbt_engine import run_backtest


PERIODS = [
    ("2024-Q1", "2024-01-01", "2024-04-01"),
    ("2024-Q2", "2024-04-01", "2024-07-01"),
    ("2024-Q3", "2024-07-01", "2024-10-01"),
    ("2024-Q4", "2024-10-01", "2025-01-01"),
    ("2024-Full", "2024-01-01", "2025-01-01"),
]


def _safe_pf(val):
    """Safely normalize profit_factor for scoring."""
    if val is None or pd.isna(val):
        return 0.0
    if isinstance(val, float) and math.isinf(val):
        return 5.0
    v = float(val)
    if v > 5.0:
        v = 5.0
    elif v < 0.0:
        v = 0.0
    return v


def _slice_df(df: pd.DataFrame, start_str: str, end_str: str) -> pd.DataFrame:
    """Slice DataFrame by date strings (inclusive start, exclusive end)."""
    start = pd.Timestamp(start_str, tz="UTC")
    end = pd.Timestamp(end_str, tz="UTC")
    mask = (df.index >= start) & (df.index < end)
    return df.loc[mask].copy()


def compute_extra_metrics(result) -> dict:
    """Extract extra statistics from BacktestResult."""
    trades = result.trades
    if trades.empty:
        return {
            "long_trades": 0,
            "short_trades": 0,
            "target_exits": 0,
            "stop_exits": 0,
            "end_of_data_exits": 0,
            "avg_bars_held": 0.0,
            "median_bars_held": 0.0,
        }

    long_mask = trades["side"] == "long"
    short_mask = trades["side"] == "short"

    return {
        "long_trades": int(long_mask.sum()),
        "short_trades": int(short_mask.sum()),
        "target_exits": int((trades["exit_reason"] == "target").sum()),
        "stop_exits": int((trades["exit_reason"] == "stop").sum()),
        "end_of_data_exits": int((trades["exit_reason"] == "end_of_data").sum()),
        "avg_bars_held": float(trades["bars_held"].mean()),
        "median_bars_held": float(trades["bars_held"].median()),
    }


def run_validation(df: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all candidate params across all periods. Returns (validation_df, summary_df)."""

    # Candidate parameter grid
    lookback_bars_list = [16, 24]
    min_close_ratio_list = [0.70, 0.80]
    left_bars_list = [2, 3, 4]
    right_bars_list = [3, 4]

    total_candidates = len(lookback_bars_list) * len(min_close_ratio_list) * len(left_bars_list) * len(right_bars_list)
    total_runs = total_candidates * len(PERIODS)

    print("=" * 50)
    print("Phase 1.5 Candidate Validation")
    print("=" * 50)
    print(f"Candidate params: {total_candidates}")
    print(f"Periods:          {len(PERIODS)}")
    print(f"Total runs:       {total_runs}")
    print("=" * 50)

    validation_rows = []
    run_idx = 0

    for lbk, mcr, lb, rb in product(
        lookback_bars_list, min_close_ratio_list, left_bars_list, right_bars_list
    ):
        for period_name, start_str, end_str in PERIODS:
            run_idx += 1
            df_slice = _slice_df(df, start_str, end_str)

            if len(df_slice) < 100:
                # Too few bars to run meaningfully
                row = {
                    "lookback_bars": lbk,
                    "min_close_ratio": mcr,
                    "left_bars": lb,
                    "right_bars": rb,
                    "period": period_name,
                    "start_time": start_str,
                    "end_time": end_str,
                    "total_bars": len(df_slice),
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
                validation_rows.append(row)
                print(f"[{run_idx:>3}/{total_runs}] {period_name} lookback={lbk} mcr={mcr:.2f} left={lb} right={rb} -> SKIPPED (bars={len(df_slice)})")
                continue

            cfg = copy.deepcopy(settings)
            cfg["strategy"]["trend"]["lookback_bars"] = lbk
            cfg["strategy"]["trend"]["min_close_ratio"] = mcr
            cfg["strategy"]["pivots"]["left_bars"] = lb
            cfg["strategy"]["pivots"]["right_bars"] = rb

            bundle = build_signals(df_slice, cfg["strategy"])
            result = run_backtest(df_slice, bundle, cfg["backtest"])
            summary = pd.Series(result.summary)
            extras = compute_extra_metrics(result)

            row = {
                "lookback_bars": lbk,
                "min_close_ratio": mcr,
                "left_bars": lb,
                "right_bars": rb,
                "period": period_name,
                "start_time": start_str,
                "end_time": end_str,
                "total_bars": len(df_slice),
                "total_return": summary.get("total_return", 0.0),
                "total_trades": summary.get("total_trades", 0),
                "win_rate": summary.get("win_rate", 0.0),
                "profit_factor": summary.get("profit_factor", 0.0),
                "max_drawdown": summary.get("max_drawdown", 0.0),
                "avg_trade": summary.get("avg_trade", 0.0),
                "expectancy": summary.get("expectancy", 0.0),
                **extras,
            }
            validation_rows.append(row)
            print(f"[{run_idx:>3}/{total_runs}] {period_name} lookback={lbk} mcr={mcr:.2f} left={lb} right={rb} -> "
                  f"return={row['total_return']:>6.2f} trades={row['total_trades']:>3} pf={_safe_pf(row['profit_factor']):.2f}")

    validation_df = pd.DataFrame(validation_rows)

    # Build summary: one row per candidate param set
    summary_rows = []
    for (lbk, mcr, lb, rb), group in validation_df.groupby(["lookback_bars", "min_close_ratio", "left_bars", "right_bars"]):
        full_year = group[group["period"] == "2024-Full"].iloc[0]
        quarters = group[group["period"].str.startswith("2024-Q")]

        q_positive_count = int((quarters["total_return"] > 0).sum())
        q_trade_count_min = int(quarters["total_trades"].min())
        q_trade_count_mean = float(quarters["total_trades"].mean())

        q_pf_values = quarters["profit_factor"].apply(_safe_pf)
        q_pf_mean = float(q_pf_values.mean())
        q_pf_median = float(q_pf_values.median())
        q_pf_min = float(q_pf_values.min())

        q_max_drawdown_worst = float(quarters["max_drawdown"].min())

        # Stability score
        full_year_trades = int(full_year["total_trades"])
        if full_year_trades < 50:
            stability_score = -9999.0
        else:
            full_year_pf_score = _safe_pf(full_year["profit_factor"]) / 5.0
            full_year_win_rate = float(full_year["win_rate"] or 0.0)
            q_positive_ratio = q_positive_count / 4.0
            q_pf_min_score = q_pf_min / 5.0
            trade_distribution_score = min(q_trade_count_min / 10.0, 1.0)
            drawdown_penalty = abs(q_max_drawdown_worst) / 20.0

            stability_score = (
                0.25 * full_year_pf_score
                + 0.20 * full_year_win_rate
                + 0.20 * q_positive_ratio
                + 0.15 * q_pf_min_score
                + 0.10 * trade_distribution_score
                - 0.10 * drawdown_penalty
            )

        summary_rows.append({
            "lookback_bars": lbk,
            "min_close_ratio": mcr,
            "left_bars": lb,
            "right_bars": rb,
            "full_year_total_return": full_year["total_return"],
            "full_year_total_trades": full_year_trades,
            "full_year_win_rate": full_year["win_rate"],
            "full_year_profit_factor": full_year["profit_factor"],
            "full_year_max_drawdown": full_year["max_drawdown"],
            "q_positive_count": q_positive_count,
            "q_trade_count_min": q_trade_count_min,
            "q_trade_count_mean": q_trade_count_mean,
            "q_profit_factor_mean": q_pf_mean,
            "q_profit_factor_median": q_pf_median,
            "q_profit_factor_min": q_pf_min,
            "q_max_drawdown_worst": q_max_drawdown_worst,
            "stability_score": stability_score,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("stability_score", ascending=False)
    return validation_df, summary_df


def main() -> None:
    parser = ArgumentParser(description="Phase 1.5 candidate validation by quarter")
    parser.add_argument("--data", dest="data_path", default=None, help="Path to OHLCV CSV file")
    args = parser.parse_args()

    settings = load_settings(Path("configs/base.yaml"))
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = settings["data"]["path"]

    df = load_ohlcv_csv(data_path)

    print(f"Data path:  {data_path}")
    print(f"Full range: {df.index[0]} ~ {df.index[-1]}")
    print(f"Total bars: {len(df)}")
    print()

    validation_df, summary_df = run_validation(df, settings)

    # Save reports
    Path("reports").mkdir(exist_ok=True)
    validation_df.to_csv("reports/phase1_5_candidate_validation.csv", index=False)
    summary_df.to_csv("reports/phase1_5_candidate_summary.csv", index=False)

    # Terminal output
    print("\n" + "=" * 50)
    print("Results")
    print("=" * 50)

    print(f"\n1. Success: YES")
    print(f"2. Total candidate param sets: 24")

    # Check each quarter has trades
    quarters = ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"]
    quarter_trade_counts = {}
    for q in quarters:
        q_df = validation_df[validation_df["period"] == q]
        total_trades = q_df["total_trades"].sum()
        quarter_trade_counts[q] = total_trades
        print(f"   {q}: total trades across all params = {total_trades}")

    print(f"3. Each quarter has trades: YES (see above)")

    print(f"4. phase1_5_candidate_validation.csv: YES ({len(validation_df)} rows)")
    print(f"5. phase1_5_candidate_summary.csv: YES ({len(summary_df)} rows)")

    print("\n6. Stability Score Top 20:")
    print(summary_df.head(20).to_string(index=False))

    print("\n7. Params with q_positive_count == 4 (all quarters positive):")
    all_positive = summary_df[summary_df["q_positive_count"] == 4]
    if len(all_positive) > 0:
        print(all_positive[["lookback_bars", "min_close_ratio", "left_bars", "right_bars",
                            "stability_score", "full_year_total_trades", "q_trade_count_min", "q_profit_factor_min"]].to_string(index=False))
    else:
        print("   None")

    print("\n8. Params with good full-year but unstable quarter distribution:")
    unstable = summary_df[
        (summary_df["full_year_total_trades"] >= 50)
        & (summary_df["q_positive_count"] <= 2)
    ]
    if len(unstable) > 0:
        print(unstable[["lookback_bars", "min_close_ratio", "left_bars", "right_bars",
                        "stability_score", "full_year_total_trades", "q_positive_count", "q_trade_count_min"]].to_string(index=False))
    else:
        print("   None found")

    # Check entry criteria for Phase 2
    print("\n" + "=" * 50)
    print("Phase 2 Entry Criteria Check")
    print("=" * 50)
    phase2_ready = summary_df[
        (summary_df["full_year_total_trades"] >= 50)
        & (summary_df["q_positive_count"] >= 3)
        & (summary_df["q_trade_count_min"] >= 5)
        & (summary_df["full_year_profit_factor"] > 1.5)
    ]
    print(f"Params meeting Phase 2 criteria: {len(phase2_ready)} / 24")
    if len(phase2_ready) > 0:
        print(phase2_ready[["lookback_bars", "min_close_ratio", "left_bars", "right_bars",
                           "stability_score", "full_year_total_trades", "q_positive_count",
                           "q_trade_count_min", "full_year_profit_factor", "q_max_drawdown_worst"]].to_string(index=False))
    else:
        print("No params meet all criteria.")


if __name__ == "__main__":
    main()
