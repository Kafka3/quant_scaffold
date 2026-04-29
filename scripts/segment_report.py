#!/usr/bin/env python3
"""
segment_report.py — Per-quarter performance report for baseline strategy.

Splits data by calendar quarter, runs baseline backtest on each segment,
and outputs reports/segment_report.csv.

Columns:
  segment, bar_count, trade_count, win_rate, expectancy_R,
  profit_factor, max_drawdown_R, avg_R, long_count, short_count,
  avg_hold_bars, max_consecutive_losses

Usage:
  python scripts/segment_report.py [--data data/raw/BTCUSDT_5m_2024_2025.csv]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.event_engine import run_backtest

BASELINE_CONFIG = ROOT / "configs" / "baseline_ema55_stoch143_2r.yaml"


def _quarter_label(dt) -> str:
    """Return e.g. '2024Q1' for a timestamp."""
    return f"{dt.year}Q{(dt.month - 1) // 3 + 1}"


def _segment_dataframe(df: pd.DataFrame) -> list:
    """Split DataFrame by calendar quarter, return list of (label, df) tuples."""
    segments = {}
    for idx in df.index:
        q = _quarter_label(idx)
        if q not in segments:
            segments[q] = []
        segments[q].append(idx)

    result = []
    for q in sorted(segments.keys()):
        idx_list = segments[q]
        result.append((q, df.loc[idx_list].copy()))
    return result


def _consecutive_losses(trades_df: pd.DataFrame, side: str = "net") -> int:
    """Max consecutive losing trades."""
    if "pnl" not in trades_df.columns or trades_df.empty:
        return 0
    if side == "long":
        subset = trades_df[trades_df["side"] == "long"]
    elif side == "short":
        subset = trades_df[trades_df["side"] == "short"]
    else:
        subset = trades_df

    if subset.empty:
        return 0

    losses = (subset["pnl"] <= 0).astype(int)
    streaks = (losses.groupby((losses != losses.shift()).cumsum()).cumsum() + 1) * losses
    return int(streaks.max()) if len(streaks) > 0 else 0


def main():
    parser = argparse.ArgumentParser(
        description="Per-quarter segment report for baseline strategy"
    )
    parser.add_argument(
        "--data",
        default=str(ROOT / "data" / "raw" / "BTCUSDT_5m_2024_2025.csv"),
        help="Path to OHLCV CSV",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "reports" / "segment_report.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading data: {args.data}")
    df = load_ohlcv_csv(Path(args.data))
    print(f"Loaded {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    # Load config
    with open(BASELINE_CONFIG) as f:
        cfg = yaml.safe_load(f)

    segments = _segment_dataframe(df)
    print(f"Segments: {[s[0] for s in segments]}")

    records = []

    for label, seg_df in segments:
        # Need enough bars for EMA 55 + pivot lookback
        if len(seg_df) < 80:
            print(f"  {label}: {len(seg_df)} bars — too few, skip")
            continue

        bundle = build_signals(seg_df, cfg["strategy"])
        result = run_backtest(seg_df, bundle, cfg["backtest"])

        trades = result.trades
        trade_count = len(trades)
        total_pnl = trades["pnl"].sum() if trade_count > 0 else 0.0

        if trade_count == 0:
            records.append({
                "segment": label,
                "bar_count": len(seg_df),
                "trade_count": 0,
                "win_rate": 0.0,
                "expectancy_R": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_R": 0.0,
                "avg_R": 0.0,
                "long_count": 0,
                "short_count": 0,
                "avg_hold_bars": 0.0,
                "max_consecutive_losses": 0,
            })
            continue

        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] < 0]
        win_count = len(wins)
        loss_count = len(losses)

        win_rate = win_count / trade_count

        # R-multiple: R = entry price - stop price (long) or stop - entry (short)
        def _calc_r(row):
            if row["side"] == "long":
                return row["entry_price"] - row["stop_price"]
            else:
                return row["stop_price"] - row["entry_price"]

        trades = trades.copy()
        trades["r"] = trades.apply(_calc_r, axis=1)
        trades["r_multiple"] = trades.apply(
            lambda r: r["pnl"] / r["r"] if r["r"] != 0 else 0.0, axis=1
        )

        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] < 0]
        avg_r_trade = trades["r_multiple"].mean()
        expectancy_r = (
            win_rate * wins["r_multiple"].mean() + (1 - win_rate) * losses["r_multiple"].mean()
            if win_count > 0 and loss_count > 0 else
            wins["r_multiple"].mean() if win_count > 0 else
            losses["r_multiple"].mean()
        )

        gross_profit = float(wins["pnl"].sum()) if win_count > 0 else 0.0
        gross_loss = float(abs(losses["pnl"].sum())) if loss_count > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        )

        # Max drawdown in R units
        equity = result.equity
        peak = equity.cummax()
        drawdown_pct = ((equity - peak) / peak).min() if peak.iloc[-1] > 0 else 0.0

        avg_r_val = avg_r_trade  # reuse
        max_dd_r = drawdown_pct / (avg_r_val / 100) if avg_r_val != 0 else 0.0

        long_count = int((trades["side"] == "long").sum())
        short_count = int((trades["side"] == "short").sum())
        avg_hold = float(trades["bars_held"].mean())

        max_cons_loss = _consecutive_losses(trades)

        records.append({
            "segment": label,
            "bar_count": len(seg_df),
            "trade_count": trade_count,
            "win_rate": round(win_rate, 4),
            "expectancy_R": round(expectancy_r, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
            "max_drawdown_R": round(max_dd_r, 4),
            "avg_R": round(avg_r_val, 4),
            "long_count": long_count,
            "short_count": short_count,
            "avg_hold_bars": round(avg_hold, 1),
            "max_consecutive_losses": max_cons_loss,
        })

    report_df = pd.DataFrame(records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(out_path, index=False)
    print(f"\nSaved segment report to {out_path}")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
