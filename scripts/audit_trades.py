#!/usr/bin/env python3
"""
audit_trades.py — Per-trade audit for baseline strategy.

Loads BTC 5m data and the baseline config, runs full pipeline,
and exports reports/audit_trades.csv with per-trade breakdown.

Each trade receives independent validity checks:
  - valid_time_order:     confirm >= p2 AND entry > confirm
  - valid_price_divergence:   p2 price < p1 price (bullish) / p2 > p1 (bearish)
  - valid_osc_divergence:     osc[p2] > osc[p1] (bullish) / osc[p2] < osc[p1] (bearish)
  - valid_channel_state:      p1 inside channel, p2 outside channel
  - valid_trigger_break:      entry price crosses trigger price in correct direction
  - valid_stop_structure:     stop_price is on correct side of entry (no invalid stop)
  - valid_target_rr:          actual R-multiple ≈ configured RR target
  - valid_all:                all of the above True

Usage:
  python scripts/audit_trades.py [--data data/raw/BTCUSDT_5m_2024_2025.csv]
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
    stoch_cfg = strategy_cfg["stochastic"]
    piv_cfg = strategy_cfg["pivots"]

    # Compute components
    osc = stochastic_d(df, k_period=stoch_cfg["k_period"],
                        d_period=stoch_cfg["d_period"],
                        smooth=stoch_cfg["smooth"])
    trend = build_trend_filter(df, strategy_cfg["trend"])
    div = detect_regular_divergence(df, osc, strategy_cfg, trend)

    # Run pipeline
    bundle = build_signals(df, strategy_cfg)
    result = run_backtest(df, bundle, cfg["backtest"])

    if result.trades.empty:
        columns = [
            "trade_id", "side",
            "pivot1_time", "pivot2_time", "confirm_time",
            "entry_time", "entry_price", "stop_price", "target_price",
            "pivot1_price", "pivot2_price",
            "stoch_p1", "stoch_p2",
            "channel_state_p1", "channel_state_p2",
            "exit_reason",
            "valid_time_order", "valid_price_divergence", "valid_osc_divergence",
            "valid_channel_state", "valid_trigger_break", "valid_stop_structure",
            "valid_target_rr", "valid_all",
        ]
        empty = pd.DataFrame(columns=columns)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(out_path, index=False)
        print(f"Saved empty audit to {out_path}")
        return

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

        # Look up pivot times from divergence result
        p1_idx = None
        p1_price = None
        p2_price = None
        stoch_p1 = pd.NA
        stoch_p2 = pd.NA
        cs_p1 = "N/A"
        cs_p2 = "N/A"

        if pd.notna(confirm_time) and confirm_time in div.bullish.index:
            p1_idx = div.bullish_pivot1_idx.loc[confirm_time] if side == "long" else div.bearish_pivot1_idx.loc[confirm_time]
            p2_idx_check = div.bullish_pivot2_idx.loc[confirm_time] if side == "long" else div.bearish_pivot2_idx.loc[confirm_time]
            p1_price = div.bullish_pivot1_price.loc[confirm_time] if side == "long" else div.bearish_pivot1_price.loc[confirm_time]
            p2_price = div.bullish_pivot2_price.loc[confirm_time] if side == "long" else div.bearish_pivot2_price.loc[confirm_time]

            if pd.notna(p1_idx) and p1_idx in osc.index:
                stoch_p1 = osc.loc[p1_idx]
            if pd.notna(p2_time) and p2_time in osc.index:
                stoch_p2 = osc.loc[p2_time]

            # Channel states
            if side == "long" and pd.notna(p1_idx) and p1_idx in trend["inside_or_below_high"].index:
                cs_p1 = "inside_or_below_high" if trend["inside_or_below_high"].loc[p1_idx] else "not_inside_or_below_high"
            elif side == "short" and pd.notna(p1_idx) and p1_idx in trend["inside_or_above_low"].index:
                cs_p1 = "inside_or_above_low" if trend["inside_or_above_low"].loc[p1_idx] else "not_inside_or_above_low"

            if side == "long" and pd.notna(p2_time) and p2_time in trend["below_channel"].index:
                cs_p2 = "below_channel" if trend["below_channel"].loc[p2_time] else "not_below_channel"
            elif side == "short" and pd.notna(p2_time) and p2_time in trend["above_channel"].index:
                cs_p2 = "above_channel" if trend["above_channel"].loc[p2_time] else "not_above_channel"

        # ---- Validity checks ----
        vals = {}

        # valid_time_order: confirm >= p2, entry > confirm
        vals["valid_time_order"] = bool(
            pd.notna(confirm_time) and pd.notna(p2_time) and pd.notna(entry_time)
            and confirm_time >= p2_time and entry_time > confirm_time
        )

        # valid_price_divergence: p2 price is deeper than p1
        if pd.notna(p1_price) and pd.notna(p2_price):
            p1v = float(p1_price)
            p2v = float(p2_price)
            if side == "long":
                vals["valid_price_divergence"] = bool(p2v < p1v)  # lower low
            else:
                vals["valid_price_divergence"] = bool(p2v > p1v)  # higher high
        else:
            vals["valid_price_divergence"] = False

        # valid_osc_divergence: osc[p2] moves opposite to price
        if pd.notna(stoch_p1) and pd.notna(stoch_p2):
            o1 = float(stoch_p1)
            o2 = float(stoch_p2)
            if side == "long":
                vals["valid_osc_divergence"] = bool(o2 > o1)  # higher osc at lower low
            else:
                vals["valid_osc_divergence"] = bool(o2 < o1)  # lower osc at higher high
        else:
            vals["valid_osc_divergence"] = False

        # valid_channel_state: p1 inside channel, p2 outside
        if side == "long":
            p1_ok = cs_p1.startswith("inside")
            p2_ok = cs_p2 == "below_channel"
        else:
            p1_ok = cs_p1.startswith("inside")
            p2_ok = cs_p2 == "above_channel"
        vals["valid_channel_state"] = bool(p1_ok and p2_ok) if cs_p1 != "N/A" and cs_p2 != "N/A" else False

        # valid_trigger_break: entry price crosses trigger in correct direction
        trigger_raw = (bundle.long_trigger_price_raw if side == "long"
                       else bundle.short_trigger_price_raw)
        if pd.notna(entry_time) and entry_time in trigger_raw.index:
            tp_val = trigger_raw.loc[entry_time]
            if pd.notna(tp_val):
                if side == "long":
                    vals["valid_trigger_break"] = bool(entry_price >= float(tp_val))
                else:
                    vals["valid_trigger_break"] = bool(entry_price <= float(tp_val))
            else:
                # Trigger might be on previous bar; check confirm_time
                if pd.notna(confirm_time) and confirm_time in trigger_raw.index:
                    tp_at_confirm = trigger_raw.loc[confirm_time]
                    if pd.notna(tp_at_confirm):
                        if side == "long":
                            vals["valid_trigger_break"] = bool(entry_price >= float(tp_at_confirm))
                        else:
                            vals["valid_trigger_break"] = bool(entry_price <= float(tp_at_confirm))
                    else:
                        vals["valid_trigger_break"] = False
                else:
                    vals["valid_trigger_break"] = False
        else:
            vals["valid_trigger_break"] = False

        # valid_stop_structure: stop is on the correct side
        if pd.notna(stop_price) and pd.notna(entry_price):
            stop_v = float(stop_price)
            entry_v = float(entry_price)
            if side == "long":
                vals["valid_stop_structure"] = bool(stop_v < entry_v)
            else:
                vals["valid_stop_structure"] = bool(stop_v > entry_v)
        else:
            vals["valid_stop_structure"] = False

        # valid_target_rr: actual R-multiple ≈ configured RR (2.0)
        if (pd.notna(entry_price) and pd.notna(stop_price)
                and pd.notna(target_price) and float(stop_price) != float(entry_price)):
            if side == "long":
                r = float(entry_price) - float(stop_price)
                actual_rr = (float(target_price) - float(entry_price)) / r if r != 0 else 0.0
            else:
                r = float(stop_price) - float(entry_price)
                actual_rr = (float(entry_price) - float(target_price)) / r if r != 0 else 0.0
            vals["valid_target_rr"] = bool(abs(actual_rr - 2.0) < 0.01)
        else:
            vals["valid_target_rr"] = False

        # valid_all
        vals["valid_all"] = all(vals.values())

        records.append({
            "trade_id": trade_id,
            "side": side,
            "pivot1_time": p1_idx,
            "pivot2_time": p2_time,
            "confirm_time": confirm_time,
            "entry_time": entry_time,
            "entry_price": round(entry_price, 2) if pd.notna(entry_price) else pd.NA,
            "stop_price": round(stop_price, 2) if pd.notna(stop_price) else pd.NA,
            "target_price": round(target_price, 2) if pd.notna(target_price) else pd.NA,
            "pivot1_price": round(p1_price, 2) if pd.notna(p1_price) else pd.NA,
            "pivot2_price": round(p2_price, 2) if pd.notna(p2_price) else pd.NA,
            "stoch_p1": round(stoch_p1, 1) if pd.notna(stoch_p1) else pd.NA,
            "stoch_p2": round(stoch_p2, 1) if pd.notna(stoch_p2) else pd.NA,
            "channel_state_p1": cs_p1,
            "channel_state_p2": cs_p2,
            "exit_reason": exit_reason,
            **vals,
        })

    audit_df = pd.DataFrame(records)

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(out_path, index=False)
    print(f"Saved {len(audit_df)} trade audits to {out_path}")

    # Summary
    for col in [c for c in audit_df.columns if c.startswith("valid_")]:
        passed = audit_df[col].sum()
        print(f"  {col}: {passed}/{len(audit_df)}")


if __name__ == "__main__":
    main()
