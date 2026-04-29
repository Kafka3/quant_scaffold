#!/usr/bin/env python3
"""
plateau_check.py — Small-scale parameter plateau analysis for baseline strategy.

Runs a cartesian product of parameter variations around baseline defaults,
evaluates each combination on BTC 2024-2025 data, and saves results to
reports/plateau_check.csv.

Search space (deliberately narrow — no TPE):
  - ema_period:     45, 55, 65
  - stoch_k:         9, 14, 21
  - pivot_left:      2, 3, 4, 5  (combined with pivot_right for balanced windows)
  - pivot_right:     2, 3, 4, 5
  - rr:              1.5, 2.0, 2.5
  - setup_max_bars:  8, 12, 16

To keep runtime manageable, the full grid runs in a single sequential
process.  Estimated runtime: ~3-4 hours for 1296 combos on 210k bars.
Use --quick for a 50-combo smoke test.

Output columns:
  ema_period, stoch_k, pivot_left, pivot_right, rr, setup_max_bars,
  total_return, trade_count, win_rate, profit_factor, max_drawdown,
  avg_hold_bars

Usage:
  python scripts/plateau_check.py [--data data/raw/BTCUSDT_5m_2024_2025.csv]
                                   [--quick]  # runs first 50 combos for testing
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.event_engine import run_backtest


# Search space
EMA_PERIODS = [45, 55, 65]
STOCH_K = [9, 14, 21]
PIVOT_LEFT = [2, 3, 4, 5]
PIVOT_RIGHT = [2, 3, 4, 5]
RR = [1.5, 2.0, 2.5]
SETUP_MAX_BARS = [8, 12, 16]

# Baseline (for reference)
BASELINE = {"ema": 55, "stoch_k": 14, "p_left": 3, "p_right": 3, "rr": 2.0, "max_bars": 12}
BASELINE_CONFIG = ROOT / "configs" / "baseline_ema55_stoch143_2r.yaml"


def _build_config(ema_period: int, stoch_k: int, pivot_left: int,
                  pivot_right: int, rr: float, setup_max_bars: int) -> dict:
    """Build a strategy config dict for the given parameters."""
    return {
        "stochastic": {
            "k_period": stoch_k,
            "d_period": 3,
            "smooth": 1,
            "oversold": 20,
            "overbought": 80,
        },
        "pivots": {
            "left_bars": pivot_left,
            "right_bars": pivot_right,
            "min_separation": 5,
            "max_separation": 35,
            "strict": True,
        },
        "trend": {
            "ema_period": ema_period,
            "lookback_bars": 12,
            "min_close_ratio": 1.0,
        },
        "risk": {
            "atr_period": 14,
            "stop_buffer": 0.0,
            "rr_target": rr,
        },
        "setup": {
            "setup_max_bars": setup_max_bars,
            "replace_same_side_setup": True,
            "invalidate_on_stop_anchor_break": True,
        },
    }


def _backtest_single(df: pd.DataFrame, strategy_cfg: dict) -> dict:
    """Run backtest for one parameter combination and return summary."""
    try:
        bundle = build_signals(df, strategy_cfg)
        bt_config = {"initial_cash": 100000, "fee_per_trade": 0.0,
                     "slippage": 0.0, "allow_short": True}
        result = run_backtest(df, bundle, bt_config)

        s = result.summary
        avg_hold = result.trades["bars_held"].mean() if len(result.trades) > 0 else 0.0

        return {
            "total_return": round(s.get("total_return", 0.0), 4),
            "trade_count": s.get("total_trades", 0),
            "win_rate": round(s.get("win_rate", 0.0), 4),
            "profit_factor": (
                round(s.get("profit_factor", 0.0), 4)
                if s.get("profit_factor") is not None
                else "inf"
            ),
            "max_drawdown": round(s.get("max_drawdown", 0.0), 4),
            "avg_hold_bars": round(avg_hold, 1),
        }
    except Exception as e:
        return {
            "total_return": None,
            "trade_count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_hold_bars": 0.0,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Small-scale parameter plateau analysis"
    )
    parser.add_argument(
        "--data",
        default=str(ROOT / "data" / "raw" / "BTCUSDT_5m_2024_2025.csv"),
        help="Path to OHLCV CSV",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "reports" / "plateau_check.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only first 50 combos on a 2-month subset for testing",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading data: {args.data}")
    df = load_ohlcv_csv(Path(args.data))
    print(f"Loaded {len(df)} bars")

    # Generate all parameter combinations
    combos = list(itertools.product(
        EMA_PERIODS, STOCH_K, PIVOT_LEFT, PIVOT_RIGHT, RR, SETUP_MAX_BARS
    ))
    total = min(len(combos), 50) if args.quick else len(combos)

    # For quick mode, use a subset of data (3 months instead of 2 years)
    df_bt = df
    if args.quick:
        df_bt = df[df.index >= pd.Timestamp("2025-01-01", tz="UTC")]
        df_bt = df_bt[df_bt.index < pd.Timestamp("2025-03-01", tz="UTC")]
        if len(df_bt) < 1000:
            df_bt = df.iloc[:20000]  # fallback
        print(f"Quick mode: using {len(df_bt)} bars (2025-01 to 2025-02)")

    print(f"Total combos: {len(combos)}, running: {total}")
    print(f"Baseline: {BASELINE}\n")

    records = []
    start = time.time()

    for i, (ema_p, stoch_k, p_left, p_right, rr, max_bars) in enumerate(combos[:total]):
        is_baseline = (
            ema_p == BASELINE["ema"] and stoch_k == BASELINE["stoch_k"]
            and p_left == BASELINE["p_left"] and p_right == BASELINE["p_right"]
            and rr == BASELINE["rr"] and max_bars == BASELINE["max_bars"]
        )

        strategy_cfg = _build_config(ema_p, stoch_k, p_left, p_right, rr, max_bars)
        result = _backtest_single(df_bt, strategy_cfg)

        records.append({
            "combo_id": i,
            "is_baseline": is_baseline,
            "ema_period": ema_p,
            "stoch_k": stoch_k,
            "pivot_left": p_left,
            "pivot_right": p_right,
            "rr": rr,
            "setup_max_bars": max_bars,
            **result,
        })

        if (i + 1) % 20 == 0 or i == 0 or i == total - 1:
            elapsed = time.time() - start
            pct = (i + 1) / total * 100
            print(f"  [{i+1}/{total}] ({pct:.0f}%) "
                  f"ema={ema_p} k={stoch_k} p={p_left}/{p_right} "
                  f"rr={rr} max={max_bars} → "
                  f"ret={result['total_return']}% trades={result['trade_count']} "
                  f"({elapsed:.1f}s)")

    # Save
    report_df = pd.DataFrame(records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(out_path, index=False)

    total_time = time.time() - start
    print(f"\nSaved {len(report_df)} results to {out_path} ({total_time:.1f}s)")

    # Print baseline and top performers
    baseline_row = report_df[report_df["is_baseline"] == True]
    if len(baseline_row) > 0:
        print(f"\n--- Baseline ---")
        for col in baseline_row.columns:
            val = baseline_row.iloc[0][col]
            print(f"  {col}: {val}")

    # Top by profit_factor (with at least 5 trades)
    valid = report_df[
        (report_df["trade_count"] >= 5)
        & (report_df["error"].isna() if "error" in report_df.columns else True)
    ].copy()
    if len(valid) > 0:
        valid["pf_num"] = pd.to_numeric(valid["profit_factor"], errors="coerce")
        top5 = valid.nlargest(5, "pf_num")
        print(f"\n--- Top 5 by Profit Factor (≥5 trades) ---")
        cols = ["ema_period", "stoch_k", "pivot_left", "pivot_right",
                "rr", "setup_max_bars", "total_return", "trade_count",
                "profit_factor", "max_drawdown"]
        print(top5[cols].to_string(index=False))

        top_by_return = valid.nlargest(5, "total_return")
        print(f"\n--- Top 5 by Total Return (≥5 trades) ---")
        print(top_by_return[cols].to_string(index=False))


if __name__ == "__main__":
    main()
