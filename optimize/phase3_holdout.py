#!/usr/bin/env python3
"""
Phase 3A — 2025 Out-of-Sample Holdout Validation + Cost Stress Test.

Reads configs/candidates/*.yaml, runs each on 2025-Q1~Q4 + 2025-Full.
Supports --fee-per-trade and --slippage overrides for cost pressure testing.

Usage:
    python optimize/phase3_holdout.py --data data/raw/BTCUSDT_5m_2025.csv
    python optimize/phase3_holdout.py --data data/raw/BTCUSDT_5m_2025.csv --fee-per-trade 5 --slippage 5
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from argparse import ArgumentParser
import copy
import math
import glob

import pandas as pd

from configs.settings import load_settings
from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.vectorbt_engine import run_backtest


PERIODS = [
    ("2025-Q1", "2025-01-01", "2025-04-01"),
    ("2025-Q2", "2025-04-01", "2025-07-01"),
    ("2025-Q3", "2025-07-01", "2025-10-01"),
    ("2025-Q4", "2025-10-01", "2026-01-01"),
    ("2025-Full", "2025-01-01", "2026-01-01"),
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


def _safe_win_rate(val):
    """Safely normalize win_rate."""
    if val is None or pd.isna(val):
        return 0.0
    return float(val)


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


def run_holdout(df: pd.DataFrame, candidate_configs: list, backtest_overrides: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all candidates across all periods. Returns (validation_df, summary_df)."""

    total_runs = len(candidate_configs) * len(PERIODS)
    print("=" * 60)
    print("Phase 3A — 2025 Out-of-Sample Holdout Validation")
    print("=" * 60)
    print(f"Candidates: {len(candidate_configs)}")
    print(f"Periods:    {len(PERIODS)}")
    print(f"Total runs: {total_runs}")
    if backtest_overrides:
        print(f"Overrides:  {backtest_overrides}")
    print("=" * 60)

    validation_rows = []
    run_idx = 0

    for candidate_name, config_path, settings in candidate_configs:
        strategy_cfg = settings["strategy"]
        trend_cfg = strategy_cfg["trend"]
        risk_cfg = strategy_cfg["risk"]
        stoch_cfg = strategy_cfg["stochastic"]
        pivots_cfg = strategy_cfg["pivots"]

        for period_name, start_str, end_str in PERIODS:
            run_idx += 1
            df_slice = _slice_df(df, start_str, end_str)

            if len(df_slice) < 100:
                row = {
                    "candidate_name": candidate_name,
                    "config_path": str(config_path),
                    "period": period_name,
                    "start_time": start_str,
                    "end_time": end_str,
                    "total_bars": len(df_slice),
                    "ema_period": trend_cfg["ema_period"],
                    "lookback_bars": trend_cfg["lookback_bars"],
                    "min_close_ratio": trend_cfg["min_close_ratio"],
                    "left_bars": pivots_cfg["left_bars"],
                    "right_bars": pivots_cfg["right_bars"],
                    "rr_target": risk_cfg["rr_target"],
                    "oversold": stoch_cfg["oversold"],
                    "overbought": stoch_cfg["overbought"],
                    "stop_buffer": risk_cfg["stop_buffer"],
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
                print(f"[{run_idx:>3}/{total_runs}] {candidate_name} {period_name} -> SKIPPED (bars={len(df_slice)})")
                continue

            cfg = copy.deepcopy(settings)
            # Apply cost overrides
            for k, v in backtest_overrides.items():
                cfg["backtest"][k] = v

            bundle = build_signals(df_slice, cfg["strategy"])
            result = run_backtest(df_slice, bundle, cfg["backtest"])
            summary = pd.Series(result.summary)
            extras = compute_extra_metrics(result)

            row = {
                "candidate_name": candidate_name,
                "config_path": str(config_path),
                "period": period_name,
                "start_time": start_str,
                "end_time": end_str,
                "total_bars": len(df_slice),
                "ema_period": trend_cfg["ema_period"],
                "lookback_bars": trend_cfg["lookback_bars"],
                "min_close_ratio": trend_cfg["min_close_ratio"],
                "left_bars": pivots_cfg["left_bars"],
                "right_bars": pivots_cfg["right_bars"],
                "rr_target": risk_cfg["rr_target"],
                "oversold": stoch_cfg["oversold"],
                "overbought": stoch_cfg["overbought"],
                "stop_buffer": risk_cfg["stop_buffer"],
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
            print(
                f"[{run_idx:>3}/{total_runs}] {candidate_name} {period_name} -> "
                f"return={row['total_return']:>6.2f} trades={row['total_trades']:>3} pf={_safe_pf(row['profit_factor']):.2f}"
            )

    validation_df = pd.DataFrame(validation_rows)

    # Build summary: one row per candidate
    summary_rows = []
    for candidate_name, config_path, _settings in candidate_configs:
        cand_df = validation_df[validation_df["candidate_name"] == candidate_name]
        full_year = cand_df[cand_df["period"] == "2025-Full"].iloc[0]
        quarters = cand_df[cand_df["period"].str.startswith("2025-Q")]

        q_positive_count = int((quarters["total_return"] > 0).sum())
        q_trade_count_min = int(quarters["total_trades"].min())
        q_trade_count_mean = float(quarters["total_trades"].mean())

        q_pf_values = quarters["profit_factor"].apply(_safe_pf)
        q_pf_mean = float(q_pf_values.mean())
        q_pf_median = float(q_pf_values.median())
        q_pf_min = float(q_pf_values.min())

        q_max_drawdown_worst = float(quarters["max_drawdown"].min())

        full_year_trades = int(full_year["total_trades"])
        if full_year_trades < 50:
            holdout_score = -9999.0
        else:
            full_year_pf_score = _safe_pf(full_year["profit_factor"]) / 5.0
            full_year_win_rate_val = _safe_win_rate(full_year["win_rate"])
            q_positive_ratio = q_positive_count / 4.0
            q_pf_min_score = q_pf_min / 5.0
            return_score = max(-1.0, min(1.0, full_year["total_return"] / 20.0))
            drawdown_penalty = abs(q_max_drawdown_worst) / 20.0

            holdout_score = (
                0.25 * full_year_pf_score
                + 0.20 * full_year_win_rate_val
                + 0.20 * q_positive_ratio
                + 0.15 * q_pf_min_score
                + 0.10 * return_score
                - 0.10 * drawdown_penalty
            )

        summary_rows.append({
            "candidate_name": candidate_name,
            "config_path": str(config_path),
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
            "long_trades": full_year["long_trades"],
            "short_trades": full_year["short_trades"],
            "holdout_score": holdout_score,
        })

    summary_df = pd.DataFrame(summary_rows)
    return validation_df, summary_df


def main() -> None:
    parser = ArgumentParser(description="Phase 3A — 2025 Out-of-Sample Holdout Validation")
    parser.add_argument("--data", dest="data_path", default=None,
                        help="Path to OHLCV CSV file (overrides config data.path)")
    parser.add_argument("--fee-per-trade", dest="fee_per_trade", type=float, default=None,
                        help="Override backtest fee_per_trade (default: use config)")
    parser.add_argument("--slippage", dest="slippage", type=float, default=None,
                        help="Override backtest slippage (default: use config)")
    args = parser.parse_args()

    # Load all candidate configs
    candidate_dir = Path("configs/candidates")
    config_paths = sorted(candidate_dir.glob("*.yaml"))
    if not config_paths:
        print("ERROR: No candidate configs found in configs/candidates/")
        sys.exit(1)

    backtest_overrides = {}
    if args.fee_per_trade is not None:
        backtest_overrides["fee_per_trade"] = args.fee_per_trade
    if args.slippage is not None:
        backtest_overrides["slippage"] = args.slippage

    candidate_configs = []
    for cp in config_paths:
        settings = load_settings(cp)
        candidate_name = cp.stem
        # Override data path if --data is provided
        if args.data_path:
            settings = copy.deepcopy(settings)
            settings["data"]["path"] = args.data_path
        # Apply backtest overrides
        for k, v in backtest_overrides.items():
            settings["backtest"][k] = v
        candidate_configs.append((candidate_name, cp, settings))

    # Determine data path from first candidate (all should be same after override)
    data_path = candidate_configs[0][2]["data"]["path"]
    df = load_ohlcv_csv(data_path)

    print(f"Data path:  {data_path}")
    print(f"Full range: {df.index[0]} ~ {df.index[-1]}")
    print(f"Total bars: {len(df)}")
    print()

    validation_df, summary_df = run_holdout(df, candidate_configs, backtest_overrides)

    # Save reports with explicit nocost/cost naming
    Path("reports").mkdir(exist_ok=True)
    if not backtest_overrides:
        validation_path = "reports/phase3_holdout_validation_nocost.csv"
        summary_path = "reports/phase3_holdout_summary_nocost.csv"
    else:
        validation_path = "reports/phase3_holdout_validation_cost.csv"
        summary_path = "reports/phase3_holdout_summary_cost.csv"

    validation_df.to_csv(validation_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # Also keep legacy filenames for backward compatibility
    validation_df.to_csv("reports/phase3_holdout_validation.csv", index=False)
    summary_df.to_csv("reports/phase3_holdout_summary.csv", index=False)

    # Terminal output
    print("\n" + "=" * 60)
    print("Phase 3A Results")
    print("=" * 60)

    print(f"\n1. Success: YES")
    print(f"2. Total candidates: {len(candidate_configs)}")
    for cn, cp, _ in candidate_configs:
        print(f"   - {cn}: {cp}")

    print(f"\n3. phase3_holdout_validation.csv: YES ({len(validation_df)} rows)")
    print(f"4. phase3_holdout_summary.csv: YES ({len(summary_df)} rows)")

    if not backtest_overrides:
        print(f"5. phase3_holdout_summary_nocost.csv: YES")
    else:
        print(f"5. phase3_holdout_summary_cost.csv: YES")

    print("\n6. Candidate Summary:")
    print(summary_df.to_string(index=False))

    print("\n7. Per-candidate full-year metrics:")
    for _, row in summary_df.iterrows():
        print(
            f"   {row['candidate_name']}: return={row['full_year_total_return']:.2f}%  "
            f"trades={row['full_year_total_trades']}  win_rate={row['full_year_win_rate']:.2%}  "
            f"pf={_safe_pf(row['full_year_profit_factor']):.2f}  mdd={row['full_year_max_drawdown']:.2f}%  "
            f"holdout_score={row['holdout_score']:.4f}"
        )

    print("\n8. Per-candidate quarter metrics:")
    for _, row in summary_df.iterrows():
        print(
            f"   {row['candidate_name']}: q_positive={row['q_positive_count']}  "
            f"q_trade_min={row['q_trade_count_min']}  q_pf_min={row['q_profit_factor_min']:.2f}  "
            f"q_mdd_worst={row['q_max_drawdown_worst']:.2f}%"
        )

    # Pass/fail analysis
    print("\n9. Phase 3B Pass/Fail Analysis:")
    for _, row in summary_df.iterrows():
        checks = []
        if row["full_year_total_trades"] >= 50:
            checks.append("trades>=50: PASS")
        else:
            checks.append("trades>=50: FAIL")

        if row["q_positive_count"] >= 3:
            checks.append("q+>=3: PASS")
        else:
            checks.append("q+>=3: FAIL")

        if row["q_trade_count_min"] >= 10:
            checks.append("q_trade_min>=10: PASS")
        else:
            checks.append("q_trade_min>=10: FAIL")

        if _safe_pf(row["full_year_profit_factor"]) >= 1.3:
            checks.append("fy_pf>=1.3: PASS")
        else:
            checks.append("fy_pf>=1.3: FAIL")

        if row["q_profit_factor_min"] >= 1.0:
            checks.append("q_pf_min>=1.0: PASS")
        else:
            checks.append("q_pf_min>=1.0: FAIL")

        long_pct = row["long_trades"] / max(row["long_trades"] + row["short_trades"], 1)
        if 0.2 <= long_pct <= 0.8:
            checks.append("balance_ok: PASS")
        else:
            checks.append("balance_ok: FAIL")

        print(f"   {row['candidate_name']}: {' | '.join(checks)}")

    best_score = summary_df.loc[summary_df["holdout_score"].idxmax()]
    best_return = summary_df.loc[summary_df["full_year_total_return"].idxmax()]
    best_q_pf = summary_df.loc[summary_df["q_profit_factor_min"].idxmax()]

    print(f"\n10. Highest holdout_score: {best_score['candidate_name']} ({best_score['holdout_score']:.4f})")
    print(f"11. Highest return: {best_return['candidate_name']} ({best_return['full_year_total_return']:.2f}%)")
    print(f"12. Highest q_min_pf: {best_q_pf['candidate_name']} ({best_q_pf['q_profit_factor_min']:.2f})")


if __name__ == "__main__":
    main()
