#!/usr/bin/env python3
"""
audit_trades.py — Per-trade audit for baseline strategy.

Loads BTC 5m data and the baseline config, runs full pipeline,
and exports reports/audit_trades.csv with per-trade breakdown:

  trade_id, side, pivot1_time, pivot2_time, confirm_time,
  entry_time, entry_price, stop_price, target_price,
  rr, stoch_p1, stoch_p2,
  channel_state_p1, channel_state_p2,
  reason, valid_flag

Usage:
  python scripts/audit_trades.py [--data data/raw/BTCUSDT_5m_2024_2025.csv]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loaders.csv_loader import load_ohlcv_csv
from features.indicators import stochastic_d
from features.divergence import detect_regular_divergence
from features.trend_filter import build_trend_filter
from strategy.signal_builder import build_signals
from backtest.event_engine import run_backtest

BASELINE_CONFIG = ROOT / "configs" / "baseline_ema55_stoch143_2r.yaml"


def main():
    parser = argparse.ArgumentParser(
        description="Audit trades for baseline_ema55_stoch143_2r"
    )
    parser.add_argument(
        "--data",
        default=str(ROOT / "data" / "raw" / "BTCUSDT_5m_2024_2025.csv"),
        help="Path to OHLCV CSV",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "reports" / "audit_trades.csv"),
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

    strategy_cfg = cfg["strategy"]

    # Run signal builder to get intermediate state
    bundle = build_signals(df, strategy_cfg)

    # Run backtest
    result = run_backtest(df, bundle, cfg["backtest"])

    if result.trades.empty:
        print("No trades — nothing to audit.")
        # Still write an empty CSV with correct columns
        columns = [
            "trade_id", "side",
            "pivot1_time", "pivot2_time", "confirm_time",
            "entry_time", "entry_price", "stop_price", "target_price",
            "rr", "stoch_p1", "stoch_p2",
            "channel_state_p1", "channel_state_p2",
            "exit_reason", "valid_flag",
        ]
        empty = pd.DataFrame(columns=columns)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(out_path, index=False)
        print(f"Saved empty audit to {out_path}")
        return

    # Compute oscillator and trend for pivot-level analysis
    stoch_cfg = strategy_cfg["stochastic"]
    osc = stochastic_d(df, k_period=stoch_cfg["k_period"],
                        d_period=stoch_cfg["d_period"],
                        smooth=stoch_cfg["smooth"])
    trend = build_trend_filter(df, strategy_cfg["trend"])

    # Also run divergence detection to access pivot1 details
    div = detect_regular_divergence(df, osc, strategy_cfg, trend)

    records = []

    for trade_id, (_, trade) in enumerate(result.trades.iterrows()):
        side = trade["side"]
        entry_time = trade["entry_time"]
        p2_time = trade["setup_pivot2_time"]
        confirm_time = trade["setup_confirm_time"]
        entry_price = trade["entry_price"]
        stop_price = trade["stop_price"]
        target_price = trade["target_price"]
        exit_reason = trade["exit_reason"]

        # R-multiple (R = entry - stop for long, stop - entry for short)
        if side == "long":
            r = entry_price - stop_price
        else:
            r = stop_price - entry_price
        rr = (target_price - entry_price) / r if r != 0 else 0.0
        if side == "short":
            rr = (entry_price - target_price) / r if r != 0 else 0.0

        # Look up pivot1 time from divergence result
        # pivot1_time is stored at the confirm bar
        p1_idx = None
        if side == "long" and pd.notna(confirm_time):
            p1_idx = div.bullish_pivot1_idx.loc[confirm_time] if confirm_time in div.bullish_pivot1_idx.index else None
            p2_idx = div.bullish_pivot2_idx.loc[confirm_time] if confirm_time in div.bullish_pivot2_idx.index else None
        elif side == "short" and pd.notna(confirm_time):
            p1_idx = div.bearish_pivot1_idx.loc[confirm_time] if confirm_time in div.bearish_pivot1_idx.index else None
            p2_idx = div.bearish_pivot2_idx.loc[confirm_time] if confirm_time in div.bearish_pivot2_idx.index else None

        p1_time = p1_idx if pd.notna(p1_idx) else pd.NA

        # Stochastic values at pivot1 and pivot2
        stoch_p1 = osc.loc[p1_idx] if pd.notna(p1_idx) and p1_idx in osc.index else pd.NA
        stoch_p2 = osc.loc[p2_time] if pd.notna(p2_time) and p2_time in osc.index else pd.NA

        # Channel state at pivot1 and pivot2
        if pd.notna(p1_idx) and p1_idx in trend.get("inside_or_below_high", pd.Series()).index:
            if side == "long":
                cs_p1 = "inside_or_below_high" if trend["inside_or_below_high"].loc[p1_idx] else "other"
            else:
                cs_p1 = "inside_or_above_low" if trend["inside_or_above_low"].loc[p1_idx] else "other"
        else:
            cs_p1 = "N/A"

        if pd.notna(p2_time) and p2_time in trend.get("below_channel", pd.Series()).index:
            if side == "long":
                cs_p2 = "below_channel" if trend["below_channel"].loc[p2_time] else "other"
            else:
                cs_p2 = "above_channel" if trend["above_channel"].loc[p2_time] else "other"
        else:
            cs_p2 = "N/A"

        # Validity check
        valid_checks = []
        if pd.notna(confirm_time) and pd.notna(p2_time):
            valid_checks.append(confirm_time >= p2_time)
        if pd.notna(entry_time) and pd.notna(confirm_time):
            valid_checks.append(entry_time > confirm_time)
        if pd.notna(p1_time) and pd.notna(p2_time):
            valid_checks.append(p2_time >= p1_time)
        valid_flag = all(valid_checks) if valid_checks else True

        records.append({
            "trade_id": trade_id,
            "side": side,
            "pivot1_time": p1_time,
            "pivot2_time": p2_time,
            "confirm_time": confirm_time,
            "entry_time": entry_time,
            "entry_price": round(entry_price, 2) if pd.notna(entry_price) else pd.NA,
            "stop_price": round(stop_price, 2) if pd.notna(stop_price) else pd.NA,
            "target_price": round(target_price, 2) if pd.notna(target_price) else pd.NA,
            "rr": round(rr, 2) if rr != 0 else pd.NA,
            "stoch_p1": round(stoch_p1, 1) if pd.notna(stoch_p1) else pd.NA,
            "stoch_p2": round(stoch_p2, 1) if pd.notna(stoch_p2) else pd.NA,
            "channel_state_p1": cs_p1,
            "channel_state_p2": cs_p2,
            "exit_reason": exit_reason,
            "valid_flag": valid_flag,
        })

    audit_df = pd.DataFrame(records)

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(out_path, index=False)
    print(f"Saved {len(audit_df)} trade audits to {out_path}")

    # Summary
    valid_count = audit_df["valid_flag"].sum()
    print(f"Valid trades: {valid_count}/{len(audit_df)}")


if __name__ == "__main__":
    main()
