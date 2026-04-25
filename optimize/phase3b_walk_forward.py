#!/usr/bin/env python3
"""
Phase 3B — Walk-Forward Validation (2024-Q1 ~ 2025-Q4).

Fixed candidates, rolling quarterly windows, no parameter optimization.

Usage:
    python optimize/phase3b_walk_forward.py --data data/raw/BTCUSDT_5m_2024_2025.csv
    python optimize/phase3b_walk_forward.py --data data/raw/BTCUSDT_5m_2024_2025.csv --fee-per-trade 5 --slippage 5
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
    ("2024-Q1", "2024-01-01", "2024-04-01"),
    ("2024-Q2", "2024-04-01", "2024-07-01"),
    ("2024-Q3", "2024-07-01", "2024-10-01"),
    ("2024-Q4", "2024-10-01", "2025-01-01"),
    ("2025-Q1", "2025-01-01", "2025-04-01"),
    ("2025-Q2", "2025-04-01", "2025-07-01"),
    ("2025-Q3", "2025-07-01", "2025-10-01"),
    ("2025-Q4", "2025-10-01", "2026-01-01"),
]

PRIMARY_CANDIDATE = "robust_ema55"
SECONDARY_CANDIDATES = ["aggressive_ema34_A", "stable_ema34_B"]


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


def run_walk_forward(df: pd.DataFrame, candidate_configs: list, backtest_overrides: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all candidates across all quarterly periods. Returns (validation_df, summary_df)."""

    total_runs = len(candidate_configs) * len(PERIODS)
    print("=" * 60)
    print("Phase 3B — Walk-Forward Validation")
    print("=" * 60)
    print(f"Primary candidate:   {PRIMARY_CANDIDATE}")
    print(f"Secondary candidates: {', '.join(SECONDARY_CANDIDATES)}")
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

        if min_trades_per_period < 10:
            walk_forward_score = -9999.0
        else:
            avg_pf_score = min(avg_profit_factor, 5.0) / 5.0
            min_pf_score = min(min_profit_factor, 5.0) / 5.0
            avg_return_score = max(-1.0, min(1.0, avg_quarter_return / 5.0))
            trade_distribution_score = min(min_trades_per_period / 20.0, 1.0)
            drawdown_penalty = abs(max_quarter_drawdown_worst) / 20.0

            walk_forward_score = (
                0.25 * avg_pf_score
                + 0.20 * positive_ratio
                + 0.20 * min_pf_score
                + 0.15 * avg_return_score
                + 0.10 * trade_distribution_score
                - 0.10 * drawdown_penalty
            )

        summary_rows.append({
            "candidate_name": candidate_name,
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
            "walk_forward_score": walk_forward_score,
        })

    summary_df = pd.DataFrame(summary_rows)
    return validation_df, summary_df


def main() -> None:
    parser = ArgumentParser(description="Phase 3B — Walk-Forward Validation")
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

    validation_df, summary_df = run_walk_forward(df, candidate_configs, backtest_overrides)

    # Save reports with explicit nocost/cost naming
    Path("reports").mkdir(exist_ok=True)
    if not backtest_overrides:
        validation_path = "reports/phase3b_walk_forward_validation_nocost.csv"
        summary_path = "reports/phase3b_walk_forward_summary_nocost.csv"
    else:
        validation_path = "reports/phase3b_walk_forward_validation_cost.csv"
        summary_path = "reports/phase3b_walk_forward_summary_cost.csv"

    validation_df.to_csv(validation_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # Terminal output
    print("\n" + "=" * 60)
    print("Phase 3B Results")
    print("=" * 60)

    print(f"\n1. Success: YES")
    print(f"2. Total candidates: {len(candidate_configs)}")
    for cn, cp, _ in candidate_configs:
        tag = "[PRIMARY]" if cn == PRIMARY_CANDIDATE else "[SECONDARY]"
        print(f"   {tag} {cn}: {cp}")

    print(f"\n3. {validation_path}: YES ({len(validation_df)} rows)")
    print(f"4. {summary_path}: YES ({len(summary_df)} rows)")

    print("\n5. Candidate Summary:")
    print(summary_df.to_string(index=False))

    print("\n6. Per-candidate quarter metrics:")
    for _, row in summary_df.iterrows():
        print(
            f"   {row['candidate_name']}: positive={row['positive_periods']}/{row['total_periods']} "
            f"min_trades={row['min_trades_per_period']} avg_pf={row['avg_profit_factor']:.2f} "
            f"min_pf={row['min_profit_factor']:.2f} mdd_worst={row['max_quarter_drawdown_worst']:.2f}% "
            f"wf_score={row['walk_forward_score']:.4f}"
        )

    # Pass/fail analysis
    print("\n7. Phase 3B Pass/Fail Analysis:")
    for _, row in summary_df.iterrows():
        checks = []
        if row["positive_ratio"] >= 0.75:
            checks.append("positive_ratio>=0.75: PASS")
        else:
            checks.append("positive_ratio>=0.75: FAIL")

        if row["min_trades_per_period"] >= 10:
            checks.append("min_trades>=10: PASS")
        else:
            checks.append("min_trades>=10: FAIL")

        if row["avg_profit_factor"] >= 1.5:
            checks.append("avg_pf>=1.5: PASS")
        else:
            checks.append("avg_pf>=1.5: FAIL")

        if row["min_profit_factor"] >= 1.0:
            checks.append("min_pf>=1.0: PASS")
        else:
            checks.append("min_pf>=1.0: FAIL")

        total_ls = row["long_trades_total"] + row["short_trades_total"]
        long_pct = row["long_trades_total"] / max(total_ls, 1)
        if 0.2 <= long_pct <= 0.8:
            checks.append("balance_ok: PASS")
        else:
            checks.append("balance_ok: FAIL")

        tag = "[PRIMARY]" if row["candidate_name"] == PRIMARY_CANDIDATE else "        "
        print(f"   {tag} {row['candidate_name']}: {' | '.join(checks)}")

    best_score = summary_df.loc[summary_df["walk_forward_score"].idxmax()]
    print(f"\n8. Highest walk_forward_score: {best_score['candidate_name']} ({best_score['walk_forward_score']:.4f})")


if __name__ == "__main__":
    main()
