#!/usr/bin/env python3
"""
segment_report.py — Per-quarter performance report for baseline strategy.

Splits data by calendar quarter, runs baseline backtest on each segment,
and outputs reports/segment_report.csv.

Columns:
  segment, bar_count, trade_count,
  win_rate, avg_win_R, avg_loss_R, expectancy_R,
  expected_R_from_winrate,
  profit_factor_price_pnl, profit_factor_R,
  target_exit_count, stop_exit_count, other_exit_count,
  max_drawdown_pct, avg_R, long_count, short_count,
  avg_hold_bars, max_consecutive_losses, r_drawdown_abs

Notes:
  - profit_factor_price_pnl: sum(pnl>0) / abs(sum(pnl<0)) — uses dollar PnL
  - profit_factor_R: sum(realized_R>0) / abs(sum(realized_R<0)) — uses R multiples
  - expected_R_from_winrate: win_rate * avg_win_R - (1-win_rate) * abs(avg_loss_R)
    This validates that expectancy_R equals the formula.
  - max_consecutive_losses: count of consecutive trades with pnl <= 0.
    First loser = count 1, streak resets on each winner.
  - r_drawdown_abs: peak-to-trough of cumulative R-multiple curve (absolute R units)

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
        segments.setdefault(q, []).append(idx)
    return [(q, df.loc[segments[q]].copy()) for q in sorted(segments)]


def _consecutive_losses(trades_df: pd.DataFrame) -> int:
    """Max consecutive trades with pnl <= 0. First loser = count 1."""
    if trades_df.empty:
        return 0
    losses = (trades_df["pnl"] <= 0).astype(int)
    streak_groups = (losses != losses.shift()).cumsum()
    streaks = losses.groupby(streak_groups).cumsum()
    return int(streaks.max()) if len(streaks) > 0 else 0


def _drawdown(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    dd = (series - series.cummax()) / series.cummax()
    safe_dd = dd.dropna()
    return abs(float(safe_dd.min())) if len(safe_dd) > 0 else 0.0


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
            print(f"  {label}: {len(seg_df)} bars — skip")
            continue

        bundle = build_signals(seg_df, cfg["strategy"])
        result = run_backtest(seg_df, bundle, cfg["backtest"])
        trades = result.trades

        if len(trades) == 0:
            records.append({
                "segment": label, "bar_count": len(seg_df),
                "trade_count": 0,
                "win_rate": 0.0, "avg_win_R": 0.0, "avg_loss_R": 0.0,
                "expectancy_R": 0.0, "expected_R_from_winrate": 0.0,
                "profit_factor_price_pnl": 0.0, "profit_factor_R": 0.0,
                "target_exit_count": 0, "stop_exit_count": 0, "other_exit_count": 0,
                "max_drawdown_pct": 0.0, "avg_R": 0.0,
                "long_count": 0, "short_count": 0,
                "avg_hold_bars": 0.0, "max_consecutive_losses": 0,
                "r_drawdown_abs": 0.0,
            })
            continue

        # Compute R-multiple
        trades = trades.copy()
        trades["R"] = trades.apply(
            lambda r: (r["entry_price"] - r["stop_price"]) if r["side"] == "long"
            else (r["stop_price"] - r["entry_price"]),
            axis=1,
        )
        trades["realized_R"] = trades.apply(
            lambda r: r["pnl"] / r["R"] if r["R"] != 0 else 0.0, axis=1
        )

        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] < 0]
        win_count = len(wins)
        loss_count = len(losses)
        trade_count = len(trades)
        win_rate = win_count / trade_count if trade_count > 0 else 0.0

        # R stats
        avg_r = trades["realized_R"].mean()
        avg_win_R = wins["realized_R"].mean() if win_count > 0 else 0.0
        avg_loss_R = losses["realized_R"].mean() if loss_count > 0 else 0.0
        expectancy_R = (
            win_rate * avg_win_R + (1 - win_rate) * avg_loss_R
            if trade_count > 0 else 0.0
        )
        expected_R_from_winrate = (
            win_rate * avg_win_R - (1 - win_rate) * abs(avg_loss_R)
            if trade_count > 0 else 0.0
        )

        # Profit factor: price PnL
        gross_profit_pnl = float(wins["pnl"].sum()) if win_count > 0 else 0.0
        gross_loss_pnl = float(abs(losses["pnl"].sum())) if loss_count > 0 else 0.0
        pf_price = gross_profit_pnl / gross_loss_pnl if gross_loss_pnl > 0 else (
            float("inf") if gross_profit_pnl > 0 else 0.0
        )

        # Profit factor: R-multiple
        r_wins = trades[trades["realized_R"] > 0]
        r_losses = trades[trades["realized_R"] < 0]
        gross_win_R = float(r_wins["realized_R"].sum()) if len(r_wins) > 0 else 0.0
        gross_loss_R = float(abs(r_losses["realized_R"].sum())) if len(r_losses) > 0 else 0.0
        pf_R = gross_win_R / gross_loss_R if gross_loss_R > 0 else (
            float("inf") if gross_win_R > 0 else 0.0
        )

        # Exit reason counts
        target_count = int((trades["exit_reason"] == "target").sum())
        stop_count = int((trades["exit_reason"] == "stop").sum())
        other_count = trade_count - target_count - stop_count

        # Drawdowns
        equity = result.equity
        max_dd_pct = _drawdown(equity) * 100

        cum_r = trades["realized_R"].cumsum()
        r_dd = (cum_r.cummax() - cum_r).max()
        r_dd_abs = float(r_dd) if not pd.isna(r_dd) else 0.0

        long_count = int((trades["side"] == "long").sum())
        short_count = int((trades["side"] == "short").sum())
        avg_hold = float(trades["bars_held"].mean())
        max_cons_loss = _consecutive_losses(trades)

        records.append({
            "segment": label,
            "bar_count": len(seg_df),
            "trade_count": trade_count,
            "win_rate": round(win_rate, 4),
            "avg_win_R": round(avg_win_R, 4),
            "avg_loss_R": round(avg_loss_R, 4),
            "expectancy_R": round(expectancy_R, 4),
            "expected_R_from_winrate": round(expected_R_from_winrate, 4),
            "profit_factor_price_pnl": round(pf_price, 4) if pf_price != float("inf") else "inf",
            "profit_factor_R": round(pf_R, 4) if pf_R != float("inf") else "inf",
            "target_exit_count": target_count,
            "stop_exit_count": stop_count,
            "other_exit_count": other_count,
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

    # Cross-validation: check expectancy_R ≈ expected_R_from_winrate
    print("\n--- Consistency check ---")
    for _, row in report_df.iterrows():
        e_r = row["expectancy_R"]
        e_f = row["expected_R_from_winrate"]
        match = "✓" if abs(e_r - e_f) < 0.001 else "✗"
        print(f"  {row['segment']}: expectancy_R={e_r} ≈ formula={e_f} {match}")


if __name__ == "__main__":
    main()
