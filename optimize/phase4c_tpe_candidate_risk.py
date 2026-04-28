#!/usr/bin/env python3
"""Phase 4C — TPE Candidate Risk Validation with Position Sizing.

Tests tpe_trial_490 and tpe_trial_262 against current_baseline and
pivot_wide_9_35 under capped position modes (3x/5x) with risk=0.5%.

Usage:
    python optimize/phase4c_tpe_candidate_risk.py \
      --data data/raw/BTCUSDT_5m_2024_2025.csv
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import math
from argparse import ArgumentParser
from typing import Tuple

import numpy as np
import pandas as pd
import yaml

from optimize.utils import slice_dataframe
from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.event_engine import run_backtest_with_position_sizing_and_costs

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

CANDIDATE_CONFIGS = {
    "robust_ema55": "configs/candidates/robust_ema55.yaml",
    "pivot_wide_9_35": "configs/candidates/robust_ema55_pivot_wide_9_35.yaml",
    "tpe_trial_490": "configs/candidates/tpe_trial_490.yaml",
    "tpe_trial_262": "configs/candidates/tpe_trial_262.yaml",
}

RISK_PCT = 0.005
POSITION_MODES = {
    "capped_3x": {"max_position_value_pct": 0.03, "max_leverage": 3.0},
    "capped_5x": {"max_position_value_pct": 0.05, "max_leverage": 5.0},
}


def _compute_sharpe(equity: pd.Series) -> float:
    """Compute annualized Sharpe ratio from daily returns."""
    if len(equity) < 10:
        return 0.0
    rets = equity.pct_change().dropna()
    if len(rets) < 5:
        return 0.0
    if rets.std() == 0:
        return 0.0
    # 5m data: ~288 bars/day, ~2016 bars/week, ~8640 bars/month
    # Annualize by sqrt(288 * 365) ≈ sqrt(105120) ≈ 324
    bars_per_year = 288 * 365
    return float(rets.mean() / rets.std() * math.sqrt(bars_per_year))


def _safe_pf(val) -> float:
    if val is None or pd.isna(val):
        return 0.0
    if isinstance(val, float) and math.isinf(val):
        return 999.0
    v = float(val)
    return 0.0 if v < 0 else v


def run_one(
    candidate_name: str,
    position_mode: str,
    risk_pct: float,
    period_name: str,
    start_str: str,
    end_str: str,
    df_full: pd.DataFrame,
    strategy_config: dict,
) -> dict:
    """Run one backtest and return summary row."""
    df_slice = slice_dataframe(df_full, start_str, end_str)

    if len(df_slice) < 100:
        return {
            "candidate_name": candidate_name,
            "position_mode": position_mode,
            "risk_per_trade_pct": risk_pct,
            "period": period_name,
            "ending_equity": 100000.0,
            "total_return": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_r": 0.0,
            "expectancy_r": 0.0,
            "avg_actual_risk_pct": 0.0,
            "max_actual_risk_pct": 0.0,
            "cap_hit_count": 0,
            "cap_hit_rate": 0.0,
            "total_fees": 0.0,
            "total_slippage_cost": 0.0,
            "cost_as_pct_of_gross_profit": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "long_pnl": 0.0,
            "short_pnl": 0.0,
        }

    bundle = build_signals(df_slice, strategy_config["strategy"])

    # Build risk/cost config for this mode
    mode_cfg = POSITION_MODES[position_mode]
    risk_cost_config = {
        "account": {"initial_cash": 100000},
        "risk": {
            "risk_per_trade_pct": risk_pct,
            "max_position_value_pct": mode_cfg["max_position_value_pct"],
            "max_leverage": mode_cfg["max_leverage"],
            "min_qty": 0.0001,
            "qty_step": 0.0001,
        },
        "cost": {
            "fee_per_trade": 5.0,
            "slippage": 5.0,
        },
        "position_modes": {
            position_mode: mode_cfg,
        },
    }

    result = run_backtest_with_position_sizing_and_costs(
        df_slice, bundle, strategy_config, risk_cost_config,
        risk_per_trade_pct=risk_pct, position_mode=position_mode,
    )

    summary = result.summary
    trades = result.trades
    equity = result.equity

    sharpe = _compute_sharpe(equity)

    total_trades = int(summary.get("total_trades", 0))
    long_t = int(summary.get("long_trades", 0))
    short_t = int(summary.get("short_trades", 0))
    long_pnl = float(summary.get("long_pnl", 0.0))
    short_pnl = float(summary.get("short_pnl", 0.0))

    ending_equity = float(summary.get("ending_equity", 100000))
    total_return = float(summary.get("total_return", 0.0))
    win_rate = float(summary.get("win_rate", 0.0))
    pf = _safe_pf(summary.get("profit_factor"))
    max_dd = float(summary.get("max_drawdown_pct", 0.0))

    avg_r = float(summary.get("avg_r", 0.0))
    exp_r = float(summary.get("expectancy_r", 0.0))

    avg_risk = float(summary.get("avg_actual_risk_pct", 0.0))
    max_risk = float(summary.get("max_actual_risk_pct", 0.0))
    cap_hit = int(summary.get("cap_hit_count", 0))
    cap_rate = float(summary.get("cap_hit_rate", 0.0))

    fees = float(summary.get("total_fees", 0.0))
    slippage = float(summary.get("total_slippage_cost", 0.0))
    gross_p = float(summary.get("gross_profit", 0.0))

    cost_pct = 0.0
    if gross_p > 0:
        cost_pct = (fees + slippage) / gross_p * 100

    return {
        "candidate_name": candidate_name,
        "position_mode": position_mode,
        "risk_per_trade_pct": risk_pct,
        "period": period_name,
        "ending_equity": ending_equity,
        "total_return": total_return,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": pf,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "avg_r": avg_r,
        "expectancy_r": exp_r,
        "avg_actual_risk_pct": avg_risk,
        "max_actual_risk_pct": max_risk,
        "cap_hit_count": cap_hit,
        "cap_hit_rate": cap_rate,
        "total_fees": fees,
        "total_slippage_cost": slippage,
        "cost_as_pct_of_gross_profit": cost_pct,
        "long_trades": long_t,
        "short_trades": short_t,
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
    }


def load_strategy_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = ArgumentParser(description="Phase 4C — TPE Candidate Risk Validation")
    parser.add_argument("--data", dest="data_path", required=True)
    args = parser.parse_args()

    df_full = load_ohlcv_csv(args.data_path)
    print(f"Data loaded: {args.data_path}")
    print(f"Full range:  {df_full.index[0]} ~ {df_full.index[-1]}")
    print(f"Total bars:  {len(df_full)}")

    modes = list(POSITION_MODES.keys())
    candidates = list(CANDIDATE_CONFIGS.keys())
    total_runs = len(candidates) * len(modes) * len(PERIODS)
    print(f"\nCandidates:    {candidates}")
    print(f"Position modes: {modes}")
    print(f"Risk pct:       {RISK_PCT:.1%}")
    print(f"Periods:        {len(PERIODS)}")
    print(f"Total runs:     {total_runs}")
    print("=" * 60)

    # Pre-load all configs
    config_cache = {}
    for name, path in CANDIDATE_CONFIGS.items():
        config_cache[name] = load_strategy_config(path)

    all_summaries = []
    all_trades = []
    all_equity = []

    run_idx = 0
    for cand_name in candidates:
        strat_cfg = config_cache[cand_name]
        for mode in modes:
            for pname, start, end in PERIODS:
                run_idx += 1
                print(f"[{run_idx:>3}/{total_runs}] {cand_name:18s} {mode:10s} {pname:14s} ", end="")

                row = run_one(cand_name, mode, RISK_PCT, pname, start, end, df_full, strat_cfg)
                all_summaries.append(row)
                print(f"return={row['total_return']:>6.2f}% trades={row['total_trades']:>3} pf={row['profit_factor']:.2f} dd={row['max_drawdown_pct']:.2f}%")

    # Build DataFrames
    summary_df = pd.DataFrame(all_summaries)

    # Save reports
    summary_df.to_csv("reports/phase4c_tpe_candidate_risk_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("Reports saved:")
    print("  reports/phase4c_tpe_candidate_risk_summary.csv")
    print("=" * 60)

    # --- Final Analysis ---
    print("\n" + "=" * 60)
    print("Phase 4C — TPE Candidate Risk Validation Report")
    print("=" * 60)

    full_df = summary_df[summary_df["period"] == "2024-2025-Full"].copy()
    full_df = full_df.sort_values(["position_mode", "profit_factor"], ascending=[True, False])

    print(f"\n1. Phase 4C 是否成功运行: YES")
    print(f"2. 是否完成 72 次运行: {'YES' if len(all_summaries) == 72 else f'NO ({len(all_summaries)} rows)'}")
    print(f"3. 报告文件是否生成: YES (1 summary file)")

    for mode in modes:
        print(f"\n4. {mode} 下 Full 排名:")
        sub = full_df[full_df["position_mode"] == mode].sort_values("total_return", ascending=False)
        disp_cols = [
            "candidate_name", "total_return", "total_trades", "profit_factor",
            "sharpe_ratio", "max_drawdown_pct", "expectancy_r",
            "avg_actual_risk_pct", "cap_hit_rate", "cost_as_pct_of_gross_profit",
        ]
        print(sub[disp_cols].to_string(index=False))

    print("\n5. 8 季度稳定性:")
    q_df = summary_df[summary_df["period"] != "2024-2025-Full"]
    for cand_name in candidates:
        for mode in modes:
            sub = q_df[(q_df["candidate_name"] == cand_name) & (q_df["position_mode"] == mode)]
            pos_q = int((sub["total_return"] > 0).sum())
            worst_ret = sub["total_return"].min()
            worst_pf = sub["profit_factor"].min()
            worst_dd = sub["max_drawdown_pct"].min()
            print(f"  {cand_name:20s} {mode:10s}: 盈利={pos_q}/8  最差return={worst_ret:.2f}%  最差PF={worst_pf:.2f}  最差DD={worst_dd:.2f}%")

    print("\n6. 验收标准判断:")

    for mode in modes:
        print(f"\n  [{mode}]")
        for cand_name in candidates:
            row = full_df[(full_df["candidate_name"] == cand_name) & (full_df["position_mode"] == mode)]
            if row.empty:
                continue
            r = row.iloc[0]
            pf = r["profit_factor"]
            sharpe = r["sharpe_ratio"]
            dd = r["max_drawdown_pct"]
            trades = r["total_trades"]
            cap_rate = r["cap_hit_rate"]
            cost_pct = r["cost_as_pct_of_gross_profit"]
            long_t = r["long_trades"]
            short_t = r["short_trades"]

            # Quarter stability
            q_sub = q_df[(q_df["candidate_name"] == cand_name) & (q_df["position_mode"] == mode)]
            pos_q = int((q_sub["total_return"] > 0).sum())

            checks = []
            checks.append(("PF >= 1.3", pf >= 1.3))
            checks.append(("Sharpe > 0", sharpe > 0))
            checks.append(("DD <= 10%", abs(dd) <= 10))
            checks.append(("7/8 季度盈利", pos_q >= 7))
            checks.append(("cap_rate可接受", cap_rate < 0.5))

            # Side balance
            total_side = long_t + short_t
            side_balanced = True
            if total_side > 0:
                long_pct = long_t / total_side
                short_pct = short_t / total_side
                side_balanced = long_pct <= 0.75 and short_pct <= 0.75
            checks.append(("long/short平衡", side_balanced))

            passed = all(c for _, c in checks)
            status = "✅ 通过" if passed else "❌ 未通过"
            print(f"  {cand_name:20s}: {status}")
            for label, ok in checks:
                print(f"    {'✓' if ok else '✗'} {label}")

    # Comparison vs baseline
    print("\n7. TPE候选 vs robust_ema55 对比:")
    for mode in modes:
        print(f"\n  [{mode}]")
        baseline = full_df[(full_df["candidate_name"] == "robust_ema55") & (full_df["position_mode"] == mode)]
        for cand in ["tpe_trial_490", "tpe_trial_262"]:
            tp = full_df[(full_df["candidate_name"] == cand) & (full_df["position_mode"] == mode)]
            if baseline.empty or tp.empty:
                continue
            bl_ret = baseline.iloc[0]["total_return"]
            tp_ret = tp.iloc[0]["total_return"]
            bl_pf = baseline.iloc[0]["profit_factor"]
            tp_pf = tp.iloc[0]["profit_factor"]
            bl_dd = baseline.iloc[0]["max_drawdown_pct"]
            tp_dd = tp.iloc[0]["max_drawdown_pct"]
            better_ret = "YES" if tp_ret > bl_ret else "NO"
            better_pf = "YES" if tp_pf > bl_pf else "NO"
            better_dd = "YES" if tp_dd > bl_dd else "NO"
            print(f"  {cand:20s}: 优于baseline收益? {better_ret}  PF? {better_pf}  DD? {better_dd}")

    print("\n8. 决策建议:")
    print("  暂不替换 robust_ema55.yaml")
    print("  后续 Phase 5A Freqtrade 应重点迁移表现最优的候选配置")
    print("=" * 60)


if __name__ == "__main__":
    main()
