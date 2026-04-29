#!/usr/bin/env python3
"""
audit_trades.py — Per-trade audit for baseline strategy.

Exports reports/audit_trades.csv with per-trade breakdown including:
  entry_price, exit_price, stop_price, target_price,
  risk_per_unit (R), realized_R (pnl / R),
  pnl_price, pnl_pct,
  gross_win_R, gross_loss_R
  + 8 valid_* flags

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

    print(f"Loading data: {args.data}")
    df = load_ohlcv_csv(Path(args.data))
    print(f"Loaded {len(df)} bars")

    with open(BASELINE_CONFIG) as f:
        cfg = yaml.safe_load(f)

    strategy_cfg = cfg["strategy"]
    stoch_cfg = strategy_cfg["stochastic"]

    osc = stochastic_d(df, k_period=stoch_cfg["k_period"],
                        d_period=stoch_cfg["d_period"],
                        smooth=stoch_cfg["smooth"])
    trend = build_trend_filter(df, strategy_cfg["trend"])
    div = detect_regular_divergence(df, osc, strategy_cfg, trend)

    bundle = build_signals(df, strategy_cfg)
    result = run_backtest(df, bundle, cfg["backtest"])

    base_columns = [
        "trade_id", "side",
        "pivot1_time", "pivot2_time", "confirm_time",
        "entry_time", "exit_time",
        "entry_price", "exit_price", "stop_price", "target_price",
        "pivot1_price", "pivot2_price",
        "risk_per_unit", "realized_R",
        "pnl_price", "pnl_pct",
        "gross_win_R", "gross_loss_R",
        "stoch_p1", "stoch_p2",
        "channel_state_p1", "channel_state_p2",
        "exit_reason",
        "valid_time_order", "valid_price_divergence", "valid_osc_divergence",
        "valid_channel_state", "valid_trigger_break", "valid_stop_structure",
        "valid_target_rr", "valid_all",
    ]

    if result.trades.empty:
        empty = pd.DataFrame(columns=base_columns)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(out_path, index=False)
        print(f"Saved empty audit to {out_path}")
        return

    records = []

    for trade_id, (_, trade) in enumerate(result.trades.iterrows()):
        side = trade["side"]
        entry_time = trade["entry_time"]
        exit_time = trade["exit_time"]
        p2_time = trade["setup_pivot2_time"]
        confirm_time = trade["setup_confirm_time"]
        entry_price = float(trade["entry_price"])
        exit_price = float(trade["exit_price"])
        stop_price = float(trade["stop_price"])
        target_price = float(trade["target_price"])
        pnl_price = float(trade["pnl"])
        exit_reason = trade["exit_reason"]

        # Risk per unit (R)
        if side == "long":
            risk_per_unit = entry_price - stop_price
            pnl_pct = pnl_price / entry_price * 100 if entry_price != 0 else 0.0
        else:
            risk_per_unit = stop_price - entry_price
            pnl_pct = pnl_price / entry_price * 100 if entry_price != 0 else 0.0

        realized_R = pnl_price / risk_per_unit if risk_per_unit != 0 else 0.0
        gross_win_R = realized_R if realized_R > 0 else 0.0
        gross_loss_R = realized_R if realized_R < 0 else 0.0

        # Pivot info
        p1_idx = None
        p1_price = None
        p2_price = None
        stoch_p1 = pd.NA
        stoch_p2 = pd.NA
        cs_p1 = "N/A"
        cs_p2 = "N/A"

        if pd.notna(confirm_time) and confirm_time in div.bullish.index:
            if side == "long":
                p1_idx = div.bullish_pivot1_idx.loc[confirm_time]
                p2_idx_chk = div.bullish_pivot2_idx.loc[confirm_time]
                p1_price = div.bullish_pivot1_price.loc[confirm_time]
                p2_price = div.bullish_pivot2_price.loc[confirm_time]
            else:
                p1_idx = div.bearish_pivot1_idx.loc[confirm_time]
                p2_idx_chk = div.bearish_pivot2_idx.loc[confirm_time]
                p1_price = div.bearish_pivot1_price.loc[confirm_time]
                p2_price = div.bearish_pivot2_price.loc[confirm_time]

            if pd.notna(p1_idx) and p1_idx in osc.index:
                stoch_p1 = osc.loc[p1_idx]
            if pd.notna(p2_time) and p2_time in osc.index:
                stoch_p2 = osc.loc[p2_time]

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

        vals["valid_time_order"] = bool(
            pd.notna(confirm_time) and pd.notna(p2_time) and pd.notna(entry_time)
            and confirm_time >= p2_time and entry_time > confirm_time
        )

        if pd.notna(p1_price) and pd.notna(p2_price):
            p1v, p2v = float(p1_price), float(p2_price)
            if side == "long":
                vals["valid_price_divergence"] = bool(p2v < p1v)
            else:
                vals["valid_price_divergence"] = bool(p2v > p1v)
        else:
            vals["valid_price_divergence"] = False

        if pd.notna(stoch_p1) and pd.notna(stoch_p2):
            o1, o2 = float(stoch_p1), float(stoch_p2)
            if side == "long":
                vals["valid_osc_divergence"] = bool(o2 > o1)
            else:
                vals["valid_osc_divergence"] = bool(o2 < o1)
        else:
            vals["valid_osc_divergence"] = False

        if side == "long":
            p1_ok = cs_p1.startswith("inside")
            p2_ok = cs_p2 == "below_channel"
        else:
            p1_ok = cs_p1.startswith("inside")
            p2_ok = cs_p2 == "above_channel"
        vals["valid_channel_state"] = bool(p1_ok and p2_ok) if cs_p1 != "N/A" and cs_p2 != "N/A" else False

        trigger_raw = (bundle.long_trigger_price_raw if side == "long"
                       else bundle.short_trigger_price_raw)
        if pd.notna(entry_time) and entry_time in trigger_raw.index:
            tpv = trigger_raw.loc[entry_time]
            if pd.notna(tpv):
                if side == "long":
                    vals["valid_trigger_break"] = bool(entry_price >= float(tpv))
                else:
                    vals["valid_trigger_break"] = bool(entry_price <= float(tpv))
            elif pd.notna(confirm_time) and confirm_time in trigger_raw.index:
                tpc = trigger_raw.loc[confirm_time]
                if pd.notna(tpc):
                    vals["valid_trigger_break"] = (entry_price >= float(tpc)) if side == "long" else (entry_price <= float(tpc))
                else:
                    vals["valid_trigger_break"] = False
            else:
                vals["valid_trigger_break"] = False
        else:
            vals["valid_trigger_break"] = False

        vals["valid_stop_structure"] = bool(
            (side == "long" and stop_price < entry_price) or
            (side == "short" and stop_price > entry_price)
        ) if (pd.notna(stop_price) and pd.notna(entry_price)) else False

        if side == "long":
            implied_rr = (target_price - entry_price) / risk_per_unit if risk_per_unit != 0 else 0.0
        else:
            implied_rr = (entry_price - target_price) / risk_per_unit if risk_per_unit != 0 else 0.0
        vals["valid_target_rr"] = bool(abs(implied_rr - 2.0) < 0.01) if risk_per_unit != 0 else False
        vals["valid_all"] = all(vals.values())

        records.append({
            "trade_id": trade_id,
            "side": side,
            "pivot1_time": p1_idx,
            "pivot2_time": p2_time,
            "confirm_time": confirm_time,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "stop_price": round(stop_price, 2),
            "target_price": round(target_price, 2),
            "pivot1_price": round(p1_price, 2) if pd.notna(p1_price) else pd.NA,
            "pivot2_price": round(p2_price, 2) if pd.notna(p2_price) else pd.NA,
            "risk_per_unit": round(risk_per_unit, 2),
            "realized_R": round(realized_R, 2),
            "pnl_price": round(pnl_price, 2),
            "pnl_pct": round(pnl_pct, 4),
            "gross_win_R": round(gross_win_R, 2),
            "gross_loss_R": round(gross_loss_R, 2),
            "stoch_p1": round(stoch_p1, 1) if pd.notna(stoch_p1) else pd.NA,
            "stoch_p2": round(stoch_p2, 1) if pd.notna(stoch_p2) else pd.NA,
            "channel_state_p1": cs_p1,
            "channel_state_p2": cs_p2,
            "exit_reason": exit_reason,
            **vals,
        })

    audit_df = pd.DataFrame(records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(out_path, index=False)
    print(f"Saved {len(audit_df)} trade audits to {out_path}")

    for col in [c for c in audit_df.columns if c.startswith("valid_")]:
        passed = audit_df[col].sum()
        print(f"  {col}: {passed}/{len(audit_df)}")


if __name__ == "__main__":
    main()
