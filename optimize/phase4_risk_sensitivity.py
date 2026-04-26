#!/usr/bin/env python3
"""
Phase 4 — Risk Sensitivity and Cost Analysis for robust_ema55.

Tests 6 risk-per-trade levels × 2 position modes across 9 periods.
Generates trades, equity curves, summary statistics, and skipped records.

Usage:
    python optimize/phase4_risk_sensitivity.py \
      --config configs/candidates/robust_ema55.yaml \
      --risk-config configs/risk_cost_sensitivity.yaml \
      --data data/raw/BTCUSDT_5m_2024_2025.csv
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from argparse import ArgumentParser
from typing import Tuple

import pandas as pd
import yaml

from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.vectorbt_engine import run_backtest_with_position_sizing_and_costs


PERIODS = [
    ("2024-Q1", "2024-01-01", "2024-04-01"),
    ("2024-Q2", "2024-04-01", "2024-07-01"),
    ("2024-Q3", "2024-07-01", "2024-10-01"),
    ("2024-Q4", "2024-10-01", "2025-01-01"),
    ("2025-Q1", "2025-01-01", "2025-04-01"),
    ("2025-Q2", "2025-04-01", "2025-07-01"),
    ("2025-Q3", "2025-07-01", "2025-10-01"),
    ("2025-Q4", "2025-10-01", "2026-01-01"),
    ("2024-2025-Full", "2024-01-01", "2026-01-01"),
]


def _slice_df(df: pd.DataFrame, start_str: str, end_str: str) -> pd.DataFrame:
    start = pd.Timestamp(start_str, tz="UTC")
    end = pd.Timestamp(end_str, tz="UTC")
    mask = (df.index >= start) & (df.index < end)
    return df.loc[mask].copy()


def _run_one(
    position_mode: str,
    risk_pct: float,
    period_name: str,
    start_str: str,
    end_str: str,
    df_full: pd.DataFrame,
    strategy_config: dict,
    risk_cost_config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Run backtest for one position_mode + risk level + one period."""
    df_slice = _slice_df(df_full, start_str, end_str)
    bundle = build_signals(df_slice, strategy_config["strategy"])

    mode_cfg = risk_cost_config["position_modes"][position_mode]
    # Temporarily override the risk config for this mode
    rc_copy = dict(risk_cost_config)
    rc_copy["risk"] = dict(risk_cost_config.get("risk", {}))
    rc_copy["risk"]["max_position_value_pct"] = mode_cfg["max_position_value_pct"]
    rc_copy["risk"]["max_leverage"] = mode_cfg["max_leverage"]

    result = run_backtest_with_position_sizing_and_costs(
        df_slice,
        bundle,
        strategy_config,
        rc_copy,
        risk_per_trade_pct=risk_pct,
        position_mode=position_mode,
    )

    trades = result.trades.copy()
    trades["position_mode"] = position_mode
    trades["risk_per_trade_pct"] = risk_pct
    trades["period"] = period_name

    equity = result.equity.to_frame(name="equity")
    equity["position_mode"] = position_mode
    equity["risk_per_trade_pct"] = risk_pct
    equity["period"] = period_name
    equity["drawdown"] = equity["equity"] - equity["equity"].cummax()
    equity["drawdown_pct"] = equity["drawdown"] / equity["equity"].cummax() * 100
    equity["bar_return"] = equity["equity"].pct_change()
    equity = equity.reset_index().rename(columns={"index": "time"})

    skipped = result.skipped.copy()
    skipped["position_mode"] = position_mode
    skipped["risk_per_trade_pct"] = risk_pct
    skipped["period"] = period_name

    summary = result.summary.copy()
    summary["position_mode"] = position_mode
    summary["risk_per_trade_pct"] = risk_pct
    summary["period"] = period_name
    summary["start_time"] = start_str
    summary["end_time"] = end_str
    summary["initial_cash"] = risk_cost_config.get("account", {}).get("initial_cash", 100000)

    return trades, equity, skipped, summary


def main() -> None:
    parser = ArgumentParser(description="Phase 4 — Risk Sensitivity Analysis")
    parser.add_argument("--config", required=True, help="Path to candidate strategy config YAML")
    parser.add_argument("--risk-config", required=True, help="Path to risk/cost config YAML")
    parser.add_argument("--data", dest="data_path", required=True, help="Path to OHLCV CSV")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        strategy_config = yaml.safe_load(f)
    with open(args.risk_config, "r", encoding="utf-8") as f:
        risk_cost_config = yaml.safe_load(f)

    df_full = load_ohlcv_csv(args.data_path)
    print(f"Data loaded: {args.data_path}")
    print(f"Full range:  {df_full.index[0]} ~ {df_full.index[-1]}")
    print(f"Total bars:  {len(df_full)}")

    risk_list = risk_cost_config["risk"]["risk_per_trade_pct_list"]
    mode_list = list(risk_cost_config["position_modes"].keys())
    total_runs = len(mode_list) * len(risk_list) * len(PERIODS)
    print(f"\nPosition modes: {mode_list}")
    print(f"Risk levels:    {risk_list}")
    print(f"Periods:        {len(PERIODS)}")
    print(f"Total runs:     {total_runs}")
    print("=" * 60)

    all_trades = []
    all_equity = []
    all_skipped = []
    all_summaries = []

    run_idx = 0
    for position_mode in mode_list:
        for risk_pct in risk_list:
            for period_name, start_str, end_str in PERIODS:
                run_idx += 1
                print(f"[{run_idx:>3}/{total_runs}] mode={position_mode} risk={risk_pct:.1%}  {period_name}")

                trades, equity, skipped, summary = _run_one(
                    position_mode, risk_pct, period_name, start_str, end_str,
                    df_full, strategy_config, risk_cost_config,
                )
                all_trades.append(trades)
                all_equity.append(equity)
                all_skipped.append(skipped)
                all_summaries.append(summary)

    # Combine results
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_df = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    skipped_df = pd.concat(all_skipped, ignore_index=True) if all_skipped else pd.DataFrame()
    summary_df = pd.DataFrame(all_summaries)

    # Compute ending_equity and net_profit
    if "initial_cash" in summary_df.columns:
        ic = summary_df["initial_cash"]
    else:
        ic = 100000
        summary_df["initial_cash"] = ic

    if "ending_equity" not in summary_df.columns or summary_df["ending_equity"].isna().all():
        summary_df["ending_equity"] = ic * (1 + summary_df.get("total_return", 0) / 100)
    if "net_profit" not in summary_df.columns or summary_df["net_profit"].isna().all():
        summary_df["net_profit"] = summary_df["ending_equity"] - ic

    # Column order
    summary_cols = [
        "position_mode", "risk_per_trade_pct", "period", "start_time", "end_time",
        "initial_cash", "ending_equity", "net_profit", "total_return",
        "total_trades", "winning_trades", "losing_trades", "win_rate", "profit_factor",
        "sharpe_ratio",
        "avg_r", "median_r", "total_r", "expectancy_r",
        "max_drawdown", "max_drawdown_pct",
        "long_trades", "short_trades", "long_pnl", "short_pnl",
        "total_fees", "total_slippage_cost", "gross_profit", "gross_loss",
        "cost_as_pct_of_gross_profit",
        "avg_qty", "avg_notional", "max_notional", "avg_notional_pct", "max_notional_pct",
        "skipped_trades",
        "cap_hit_count", "cap_hit_rate",
        "avg_actual_risk_pct", "max_actual_risk_pct",
        "avg_target_risk_pct", "max_target_risk_pct",
    ]
    for col in summary_cols:
        if col not in summary_df.columns:
            if col in ["winning_trades", "losing_trades", "long_trades", "short_trades", "skipped_trades", "cap_hit_count"]:
                summary_df[col] = 0
            elif col == "max_drawdown_pct":
                summary_df[col] = summary_df.get("max_drawdown", 0.0)
            elif col in ["cap_hit_rate", "avg_actual_risk_pct", "max_actual_risk_pct", "avg_target_risk_pct", "max_target_risk_pct"]:
                summary_df[col] = 0.0
            else:
                summary_df[col] = 0.0

    available_cols = [c for c in summary_cols if c in summary_df.columns]
    summary_df = summary_df[available_cols]

    # Save
    Path("reports").mkdir(exist_ok=True)
    trades_df.to_csv("reports/phase4_risk_sensitivity_trades.csv", index=False)
    summary_df.to_csv("reports/phase4_risk_sensitivity_summary.csv", index=False)
    equity_df.to_csv("reports/phase4_risk_sensitivity_equity.csv", index=False)
    skipped_df.to_csv("reports/phase4_risk_sensitivity_skipped.csv", index=False)

    print("\n" + "=" * 60)
    print("Reports saved:")
    print("  reports/phase4_risk_sensitivity_trades.csv")
    print("  reports/phase4_risk_sensitivity_summary.csv")
    print("  reports/phase4_risk_sensitivity_equity.csv")
    print("  reports/phase4_risk_sensitivity_skipped.csv")
    print("=" * 60)

    # Final analysis
    print("\n" + "=" * 60)
    print("Phase 4 — Risk Sensitivity Final Analysis")
    print("=" * 60)

    full_period = "2024-2025-Full"
    full_df = summary_df[summary_df["period"] == full_period].sort_values(["position_mode", "risk_per_trade_pct"])

    print(f"\n1. 是否成功运行: YES ({len(summary_df)} summary rows)")
    print(f"2. 是否完成 108 次运行: {'YES' if len(summary_df) == 108 else 'NO'} ({len(summary_df)} rows)")
    print(f"3. 报告文件是否生成: YES (4 files)")

    print("\n4. uncapped 模式下 2024-2025 Full 表现:")
    uncapped = full_df[full_df["position_mode"] == "uncapped"]
    disp = ["risk_per_trade_pct", "ending_equity", "total_return", "total_trades",
            "win_rate", "profit_factor", "sharpe_ratio", "max_drawdown_pct",
            "expectancy_r", "avg_actual_risk_pct", "max_actual_risk_pct"]
    print(uncapped[[c for c in disp if c in uncapped.columns]].to_string(index=False))

    print("\n5. capped_1x 模式下 2024-2025 Full 表现:")
    capped = full_df[full_df["position_mode"] == "capped_1x"]
    disp2 = ["risk_per_trade_pct", "ending_equity", "total_return", "cap_hit_count",
             "cap_hit_rate", "avg_actual_risk_pct", "max_actual_risk_pct", "profit_factor"]
    print(capped[[c for c in disp2 if c in capped.columns]].to_string(index=False))

    print("\n6. 各风险档季度表现 (uncapped):")
    q_uncapped = summary_df[(summary_df["period"] != full_period) & (summary_df["position_mode"] == "uncapped")]
    for risk in sorted(q_uncapped["risk_per_trade_pct"].unique()):
        sub = q_uncapped[q_uncapped["risk_per_trade_pct"] == risk]
        pos = len(sub[sub["total_return"] > 0])
        print(f"  risk={risk:.1%}: 盈利季度={pos}/8  最差return={sub['total_return'].min():.2f}%  最差DD={sub['max_drawdown_pct'].min():.2f}%  最差PF={sub['profit_factor'].min():.2f}")

    print("\n7. 各风险档季度表现 (capped_1x):")
    q_capped = summary_df[(summary_df["period"] != full_period) & (summary_df["position_mode"] == "capped_1x")]
    for risk in sorted(q_capped["risk_per_trade_pct"].unique()):
        sub = q_capped[q_capped["risk_per_trade_pct"] == risk]
        pos = len(sub[sub["total_return"] > 0])
        print(f"  risk={risk:.1%}: 盈利季度={pos}/8  最差return={sub['total_return'].min():.2f}%  最差DD={sub['max_drawdown_pct'].min():.2f}%  cap_hit={sub['cap_hit_count'].sum()}")

    print("\n8. 推荐:")
    for _, row in full_df.iterrows():
        mode = row["position_mode"]
        risk = row["risk_per_trade_pct"]
        pf = row.get("profit_factor", 0)
        exp_r = row.get("expectancy_r", 0)
        sharpe = row.get("sharpe_ratio", float("nan"))
        dd = row.get("max_drawdown_pct", 0)
        q_sub = summary_df[(summary_df["position_mode"] == mode) & (summary_df["risk_per_trade_pct"] == risk) & (summary_df["period"] != full_period)]
        pos_q = len(q_sub[q_sub["total_return"] > 0])

        if pd.isna(pf) or (isinstance(pf, float) and pf == float("inf")):
            pf = 999.0

        if pf >= 1.6 and exp_r >= 0.2 and sharpe >= 1.0 and dd >= -6 and pos_q >= 7:
            rec = "✅ 推荐 paper trading"
        elif pf >= 1.3 and exp_r > 0 and sharpe > 0 and dd >= -10 and pos_q >= 6:
            rec = "🟡 可用于 paper trading"
        elif dd < -15 or sharpe < 0 or pos_q < 6:
            rec = "❌ 仅压力测试"
        else:
            rec = "🟡 边缘"
        print(f"  {mode} risk={risk:.1%}: {rec}")

    print("\n" + "=" * 60)
    print("确认:")
    print("  - uncapped 模式用于理论压力测试")
    print("  - capped_1x 模式用于现实 1x 仓位约束测试")
    print("=" * 60)


if __name__ == "__main__":
    main()
