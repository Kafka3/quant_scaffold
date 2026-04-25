#!/usr/bin/env python3
"""
Phase 2A — Fixed trend structure, fine-tune RR and Stochastic thresholds.

Searches:
  rr_target: [1.5, 1.8, 2.0, 2.2, 2.5]
  oversold/overbought: [(25,75), (20,80), (15,85)]

Validates each param set across quarters.

Usage:
    python optimize/phase2a_search.py --data data/raw/BTCUSDT_5m_2024.csv
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


# Time periods for validation
PERIODS = [
    ("2024-Q1", "2024-01-01", "2024-04-01"),
    ("2024-Q2", "2024-04-01", "2024-07-01"),
    ("2024-Q3", "2024-07-01", "2024-10-01"),
    ("2024-Q4", "2024-10-01", "2025-01-01"),
    ("2024-Full", "2024-01-01", "2025-01-01"),
]

# Fixed trend structure from Phase 1.5
FIXED_TREND = {
    "ema_period": 55,
    "lookback_bars": 24,
    "min_close_ratio": 0.80,
}

FIXED_PIVOTS = {
    "left_bars": 4,
    "right_bars": 3,
}


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


def run_phase2a(df: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Phase 2A search. Returns (validation_df, summary_df)."""

    rr_targets = [1.5, 1.8, 2.0, 2.2, 2.5]
    threshold_pairs = [(25, 75), (20, 80), (15, 85)]

    total_combos = len(rr_targets) * len(threshold_pairs)
    total_runs = total_combos * len(PERIODS)

    print("=" * 50)
    print("Phase 2A — RR & Stochastic Threshold Search")
    print("=" * 50)
    print(f"Fixed trend: lookback=24, mcr=0.80, left=4, right=3")
    print(f"Search combos: {total_combos}")
    print(f"Periods: {len(PERIODS)}")
    print(f"Total runs: {total_runs}")
    print("=" * 50)

    validation_rows = []
    run_idx = 0

    for rr, (os_val, ob_val) in product(rr_targets, threshold_pairs):
        for period_name, start_str, end_str in PERIODS:
            run_idx += 1
            df_slice = _slice_df(df, start_str, end_str)

            if len(df_slice) < 100:
                row = {
                    "period": period_name,
                    "start_time": start_str,
                    "end_time": end_str,
                    "total_bars": len(df_slice),
                    "rr_target": rr,
                    "oversold": os_val,
                    "overbought": ob_val,
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
                print(f"[{run_idx:>3}/{total_runs}] {period_name} rr={rr} os={os_val} ob={ob_val} -> SKIPPED")
                continue

            cfg = copy.deepcopy(settings)
            cfg["strategy"]["trend"].update(FIXED_TREND)
            cfg["strategy"]["pivots"].update(FIXED_PIVOTS)
            cfg["strategy"]["risk"]["rr_target"] = rr
            cfg["strategy"]["stochastic"]["oversold"] = os_val
            cfg["strategy"]["stochastic"]["overbought"] = ob_val

            bundle = build_signals(df_slice, cfg["strategy"])
            result = run_backtest(df_slice, bundle, cfg["backtest"])
            summary = pd.Series(result.summary)
            extras = compute_extra_metrics(result)

            row = {
                "period": period_name,
                "start_time": start_str,
                "end_time": end_str,
                "total_bars": len(df_slice),
                "rr_target": rr,
                "oversold": os_val,
                "overbought": ob_val,
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
            print(f"[{run_idx:>3}/{total_runs}] {period_name} rr={rr} os={os_val} ob={ob_val} -> "
                  f"return={row['total_return']:>6.2f} trades={row['total_trades']:>3} pf={_safe_pf(row['profit_factor']):.2f}")

    validation_df = pd.DataFrame(validation_rows)

    # Build summary
    summary_rows = []
    for (rr, os_val, ob_val), group in validation_df.groupby(["rr_target", "oversold", "overbought"]):
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

        full_year_trades = int(full_year["total_trades"])
        if full_year_trades < 80:
            stability_score = -9999.0
        else:
            full_year_pf_score = _safe_pf(full_year["profit_factor"]) / 5.0
            full_year_win_rate = float(full_year["win_rate"] or 0.0)
            q_positive_ratio = q_positive_count / 4.0
            q_pf_min_score = q_pf_min / 5.0
            return_score = max(min(full_year["total_return"] / 20.0, 1.0), -1.0)
            drawdown_penalty = abs(q_max_drawdown_worst) / 20.0

            stability_score = (
                0.25 * full_year_pf_score
                + 0.20 * full_year_win_rate
                + 0.20 * q_positive_ratio
                + 0.15 * q_pf_min_score
                + 0.10 * return_score
                - 0.10 * drawdown_penalty
            )

        is_baseline = (rr == 2.0 and os_val == 20 and ob_val == 80)

        summary_rows.append({
            "rr_target": rr,
            "oversold": os_val,
            "overbought": ob_val,
            "is_baseline": is_baseline,
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
    parser = ArgumentParser(description="Phase 2A — RR & Stochastic threshold search")
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

    validation_df, summary_df = run_phase2a(df, settings)

    # Save reports
    Path("reports").mkdir(exist_ok=True)
    validation_df.to_csv("reports/phase2a_validation.csv", index=False)
    summary_df.to_csv("reports/phase2a_summary.csv", index=False)

    # Terminal output
    print("\n" + "=" * 50)
    print("Phase 2A Results")
    print("=" * 50)

    print(f"\n1. Success: YES")
    print(f"2. Total combos: 15")

    baseline = summary_df[summary_df["is_baseline"] == True]
    if len(baseline) > 0:
        b = baseline.iloc[0]
        print(f"\n3. Baseline (rr=2.0, os=20, ob=80):")
        print(f"   return={b['full_year_total_return']:.2f}% trades={b['full_year_total_trades']:.0f} "
              f"pf={_safe_pf(b['full_year_profit_factor']):.2f} q+={b['q_positive_count']:.0f}/4 "
              f"q_min_pf={b['q_profit_factor_min']:.2f} stability={b['stability_score']:.4f}")
    else:
        print("\n3. Baseline: NOT FOUND")

    print(f"\n4. Stability Score Top 15:")
    print(summary_df.to_string(index=False))

    # Check if any combo beats baseline on all criteria
    print(f"\n5. Combos beating baseline on all criteria:")
    if len(baseline) > 0:
        b = baseline.iloc[0]
        beaters = summary_df[
            (summary_df["q_positive_count"] == 4)
            & (summary_df["q_trade_count_min"] >= 20)
            & (summary_df["full_year_profit_factor"] >= _safe_pf(b["full_year_profit_factor"]))
            & (summary_df["full_year_total_return"] >= b["full_year_total_return"])
            & (summary_df["q_profit_factor_min"] >= 1.3)
        ]
        if len(beaters) > 0:
            print(beaters[["rr_target", "oversold", "overbought", "is_baseline",
                           "stability_score", "full_year_total_return", "full_year_total_trades",
                           "full_year_profit_factor", "q_positive_count", "q_trade_count_min",
                           "q_profit_factor_min"]].to_string(index=False))
        else:
            print("   None found")
    else:
        print("   Baseline not available for comparison")

    print(f"\n6. phase2a_validation.csv: YES ({len(validation_df)} rows)")
    print(f"7. phase2a_summary.csv: YES ({len(summary_df)} rows)")

    # Phase 2B entry check
    print("\n" + "=" * 50)
    print("Phase 2B Entry Criteria Check")
    print("=" * 50)
    phase2b_ready = summary_df[
        (summary_df["q_positive_count"] == 4)
        & (summary_df["q_trade_count_min"] >= 20)
        & (summary_df["full_year_profit_factor"] >= 2.0)
        & (summary_df["q_profit_factor_min"] >= 1.3)
        & (summary_df["full_year_total_trades"] >= 80)
    ]
    print(f"Params meeting Phase 2B criteria: {len(phase2b_ready)} / 15")
    if len(phase2b_ready) > 0:
        print(phase2b_ready[["rr_target", "oversold", "overbought", "is_baseline",
                             "stability_score", "full_year_total_trades", "q_positive_count",
                             "q_trade_count_min", "full_year_profit_factor", "q_profit_factor_min"]].to_string(index=False))
    else:
        print("No params meet all criteria. Keep baseline: rr=2.0, os=20, ob=80")


if __name__ == "__main__":
    main()
