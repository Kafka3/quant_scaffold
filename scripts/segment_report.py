#!/usr/bin/env python3
"""
segment_report.py — Per-quarter performance report for baseline strategy.

Splits data by calendar quarter, runs baseline backtest on each segment,
and outputs reports/segment_report.csv.

Columns:
  segment, bar_count, trade_count, win_rate, expectancy_R,
  profit_factor, max_drawdown_pct, avg_R, long_count, short_count,
  avg_hold_bars, max_consecutive_losses, r_drawdown_pct

Notes:
  - max_consecutive_losses: count of consecutive losing trades (pnl <= 0).
    The streak restarts on each winner. No off-by-one: first loser is count=1.
  - max_drawdown_pct: peak-to-trough drawdown of equity curve.
  - r_drawdown_pct: drawdown computed on R-multiple cumulative curve instead
    of dollar equity, giving a risk-normalized drawdown metric.

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
    return f"{dt.year}Q{(dt.month - 1) // 3 + 1}"


def _segment_dataframe(df: pd.DataFrame) -> list:
    segments = {}
    for idx in df.index:
        q = _quarter_label(idx)
        if q not in segments:
            segments[q] = []
        segments[q].append(idx)
    result = []
    for q in sorted(segments.keys()):
        result.append((q, df.loc[segments[q]].copy()))
    return result


def _consecutive_losses(trades_df: pd.DataFrame) -> int:
    """Max consecutive trades with pnl <= 0. First loser = count 1."""
    if trades_df.empty:
        return 0
    losses = (trades_df["pnl"] <= 0).astype(int)
    # Group consecutive same values by comparing to shifted
    streak_groups = (losses != losses.shift()).cumsum()
    streaks = losses.groupby(streak_groups).cumsum()
    return int(streaks.max()) if len(streaks) > 0 else 0


def _drawdown(series: pd.Series) -> float:
    """Peak-to-trough drawdown as a fraction (0.0 to 1.0)."""
    if len(series) == 0:
        return 0.0
    peak = series.cummax()
    dd = (series - peak) / peak
    safe_dd = dd.dropna()
    if len(safe_dd) == 0:
        return 0.0
    return abs(float(safe_dd.min()))


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

    print(f"Loading data: {args.data}")
    df = load_ohlcv_csv(Path(args.data))
    print(f"Loaded {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    with open(BASELINE_CONFIG) as f:
        cfg = yaml.safe_load(f)

    segments = _segment_dataframe(df)
    print(f"Segments: {[s[0] for s in segments]}")

    records = []

    for label, seg_df in segments:
        if len(seg_df) < 80:
            print(f"  {label}: {len(seg_df)} bars — too few, skip")
            continue

        bundle = build_signals(seg_df, cfg["strategy"])
        result = run_backtest(seg_df, bundle, cfg["backtest"])
        trades = result.trades
        trade_count = len(trades)

        if trade_count == 0:
            records.append({
                "segment": label, "bar_count": len(seg_df),
                "trade_count": 0, "win_rate": 0.0,
                "expectancy_R": 0.0, "profit_factor": 0.0,
                "max_drawdown_pct": 0.0, "avg_R": 0.0,
                "long_count": 0, "short_count": 0,
                "avg_hold_bars": 0.0, "max_consecutive_losses": 0,
                "r_drawdown_pct": 0.0,
            })
            continue

        # Compute per-trade R-multiple
        trades = trades.copy()
        trades["R"] = trades.apply(
            lambda r: (r["entry_price"] - r["stop_price"]) if r["side"] == "long"
            else (r["stop_price"] - r["entry_price"]),
            axis=1,
        )
        trades["r_multiple"] = trades.apply(
            lambda r: r["pnl"] / r["R"] if r["R"] != 0 else 0.0, axis=1
        )

        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] < 0]
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / trade_count if trade_count > 0 else 0.0

        # R stats
        avg_r = trades["r_multiple"].mean()
        avg_r_win = wins["r_multiple"].mean() if win_count > 0 else 0.0
        avg_r_loss = losses["r_multiple"].mean() if loss_count > 0 else 0.0
        r_win_rate = win_count / trade_count
        expectancy_r = (r_win_rate * avg_r_win + (1 - r_win_rate) * avg_r_loss
                        if trade_count > 0 else 0.0)

        # Profit factor
        gross_profit = float(wins["pnl"].sum()) if win_count > 0 else 0.0
        gross_loss = float(abs(losses["pnl"].sum())) if loss_count > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        )

        # Max drawdown (equity curve)
        equity = result.equity
        max_dd_pct = _drawdown(equity) * 100

        # R-drawdown: cumulative R curve drawdown (absolute R units)
        cum_r = trades["r_multiple"].cumsum()
        running_max = cum_r.cummax()
        dd_r = running_max - cum_r
        r_dd_abs = float(dd_r.max()) if len(dd_r) > 0 and not dd_r.isna().all() else 0.0

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
            "max_drawdown_pct": round(max_dd_pct, 4),
            "avg_R": round(avg_r, 4),
            "long_count": long_count,
            "short_count": short_count,
            "avg_hold_bars": round(avg_hold, 1),
            "max_consecutive_losses": max_cons_loss,
            "r_drawdown_abs": round(r_dd_abs, 4),
        })

    report_df = pd.DataFrame(records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(out_path, index=False)
    print(f"\nSaved segment report to {out_path}")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
