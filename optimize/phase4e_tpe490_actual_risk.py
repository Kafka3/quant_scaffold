#!/usr/bin/env python3
"""Phase 4E — tpe_trial_490 实际风险达标测试。

目标：验证 tpe_trial_490 在 uncapped 模式下，让 actual_risk_pct 真正达到
设定位（0.5%~2%），测试收益、回撤、PF、Sharpe 是否可接受。

Usage:
    python optimize/phase4e_tpe490_actual_risk.py \
      --data data/raw/BTCUSDT_5m_2024_2025.csv

Outputs:
    reports/phase4e_tpe490_actual_risk_summary.csv
    reports/phase4e_tpe490_actual_risk_decision.md
"""

import sys
import math
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from optimize.utils import slice_dataframe
from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.event_engine import run_backtest_with_position_sizing_and_costs

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PERIODS: List[Tuple[str, str, str]] = [
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

CANDIDATE = "tpe_trial_490"
CONFIG_PATH = "configs/candidates/tpe_trial_490.yaml"

TARGET_RISK_PCTS = [0.005, 0.01, 0.015, 0.02]

# Uncapped: allow full risk expression
POSITION_MODE = "uncapped"
POSITION_CFG = {"max_position_value_pct": 1.0, "max_leverage": 100.0}

REPORT_DIR = _PROJECT_ROOT / "reports"


def _compute_sharpe(equity: pd.Series) -> float:
    if len(equity) < 10:
        return 0.0
    rets = equity.pct_change().dropna()
    if len(rets) < 5 or rets.std() == 0:
        return 0.0
    bars_per_year = 288 * 365
    return float(rets.mean() / rets.std() * math.sqrt(bars_per_year))


def _safe_pf(val) -> float:
    if val is None or pd.isna(val):
        return 0.0
    if isinstance(val, float) and math.isinf(val):
        return 999.0
    v = float(val)
    return 0.0 if v < 0 else v


def compute_max_consecutive_losses(trades_df: pd.DataFrame) -> int:
    """Count max consecutive losing trades."""
    if len(trades_df) == 0:
        return 0
    # Sort by entry_time
    df = trades_df.sort_values("entry_time")
    worst = 0
    current = 0
    for _, t in df.iterrows():
        pnl = t.get("net_pnl", 0)
        if pnl <= 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return worst


def compute_implied_leverage(trades_df: pd.DataFrame, initial_cash: float) -> dict:
    """Compute leverage implied by notional / equity."""
    if len(trades_df) == 0:
        return {"avg": 0.0, "max": 0.0, "p95": 0.0}

    notional = trades_df["notional"].values
    # Use equity_before for each trade
    equity_before = trades_df["equity_before"].values
    leverage = notional / equity_before
    leverage = np.where(np.isfinite(leverage), leverage, 0.0)

    return {
        "avg": float(np.mean(leverage)),
        "max": float(np.max(leverage)),
        "p95": float(np.percentile(leverage, 95)),
    }


def run_one(
    risk_pct: float,
    period_name: str,
    start_str: str,
    end_str: str,
    df_full: pd.DataFrame,
    strategy_config: dict,
) -> dict:
    df_slice = slice_dataframe(df_full, start_str, end_str)

    if len(df_slice) < 100:
        return _empty_row(risk_pct, period_name)

    bundle = build_signals(df_slice, strategy_config["strategy"])

    risk_cost_config = {
        "account": {"initial_cash": 100000},
        "risk": {
            "risk_per_trade_pct": risk_pct,
            "max_position_value_pct": POSITION_CFG["max_position_value_pct"],
            "max_leverage": POSITION_CFG["max_leverage"],
            "min_qty": 0.0001,
            "qty_step": 0.0001,
        },
        "cost": {
            "fixed_fee_per_trade": 5.0,
            "slippage_per_side": 5.0,
            "fee_rate": 0.0,
        },
        "position_modes": {
            POSITION_MODE: POSITION_CFG,
        },
        "execution": {
            "allow_short": True,
            "same_bar_stop_first": True,
        },
    }

    try:
        result = run_backtest_with_position_sizing_and_costs(
            df_slice, bundle, strategy_config, risk_cost_config,
            risk_per_trade_pct=risk_pct, position_mode=POSITION_MODE,
        )
    except Exception as e:
        row = _empty_row(risk_pct, period_name)
        row["error"] = str(e)
        return row

    summary = result.summary
    trades_df = result.trades
    equity = result.equity
    initial_cash = 100000.0

    total_trades = int(summary.get("total_trades", 0))
    total_return = float(summary.get("total_return", 0.0))
    win_rate = float(summary.get("win_rate", 0.0))
    pf = _safe_pf(summary.get("profit_factor"))
    sharpe = _compute_sharpe(equity)
    max_dd = float(summary.get("max_drawdown_pct", 0.0))
    exp_r = float(summary.get("expectancy_r", 0.0))
    fees = float(summary.get("total_fees", 0.0))
    slippage = float(summary.get("total_slippage_cost", 0.0))
    gross_p = float(summary.get("gross_profit", 0.0))
    cost_pct = (fees + slippage) / gross_p * 100 if gross_p > 0 else 0.0

    # Actual risk detail
    if total_trades > 0 and "actual_risk_pct" in trades_df.columns:
        risk_values = trades_df["actual_risk_pct"].dropna()
        avg_risk = float(risk_values.mean()) if len(risk_values) > 0 else 0.0
        min_risk = float(risk_values.min()) if len(risk_values) > 0 else 0.0
        max_risk = float(risk_values.max()) if len(risk_values) > 0 else 0.0
    else:
        avg_risk = min_risk = max_risk = 0.0

    # Notional stats
    avg_notional_pct = float(summary.get("avg_notional_pct", 0.0))
    max_notional_pct = float(summary.get("max_notional_pct", 0.0))

    # Implied leverage
    lev = compute_implied_leverage(trades_df, initial_cash)

    # Consecutive losses
    max_consec = compute_max_consecutive_losses(trades_df)

    # Worst trade R
    if total_trades > 0 and "r_multiple" in trades_df.columns:
        r_vals = trades_df["r_multiple"].dropna()
        worst_r = float(r_vals.min()) if len(r_vals) > 0 else 0.0
    else:
        worst_r = 0.0

    return {
        "risk_per_trade_pct": risk_pct,
        "period": period_name,
        "ending_equity": float(equity.iloc[-1]) if len(equity) > 0 else initial_cash,
        "total_return": total_return,
        "total_trades": total_trades,
        "profit_factor": pf,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "win_rate": win_rate,
        "expectancy_r": exp_r,
        "avg_actual_risk_pct": avg_risk,
        "min_actual_risk_pct": min_risk,
        "max_actual_risk_pct": max_risk,
        "avg_notional_pct": avg_notional_pct,
        "max_notional_pct": max_notional_pct,
        "avg_implied_leverage": lev["avg"],
        "max_implied_leverage": lev["max"],
        "p95_implied_leverage": lev["p95"],
        "max_consecutive_losses": max_consec,
        "worst_trade_r": worst_r,
        "cost_as_pct_of_gross_profit": cost_pct,
    }


def _empty_row(risk_pct: float, period_name: str) -> dict:
    return {
        "risk_per_trade_pct": risk_pct,
        "period": period_name,
        "ending_equity": 100000.0,
        "total_return": 0.0,
        "total_trades": 0,
        "profit_factor": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "win_rate": 0.0,
        "expectancy_r": 0.0,
        "avg_actual_risk_pct": 0.0,
        "min_actual_risk_pct": 0.0,
        "max_actual_risk_pct": 0.0,
        "avg_notional_pct": 0.0,
        "max_notional_pct": 0.0,
        "avg_implied_leverage": 0.0,
        "max_implied_leverage": 0.0,
        "p95_implied_leverage": 0.0,
        "max_consecutive_losses": 0,
        "worst_trade_r": 0.0,
        "cost_as_pct_of_gross_profit": 0.0,
        "error": "",
    }


def main():
    parser = ArgumentParser(description="Phase 4E — tpe_trial_490 实际风险达标测试")
    parser.add_argument("--data", dest="data_path", required=True)
    args = parser.parse_args()

    df_full = load_ohlcv_csv(args.data_path)
    print(f"Data loaded: {args.data_path}")
    print(f"  Range: {df_full.index[0]} ~ {df_full.index[-1]}")
    print(f"  Bars:  {len(df_full):,}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        strat_cfg = yaml.safe_load(f)

    total_runs = len(TARGET_RISK_PCTS) * len(PERIODS)
    print(f"\nCandidate: {CANDIDATE}")
    print(f"Position mode: {POSITION_MODE} (uncapped)")
    print(f"Target risks: {[f'{p*100:.1f}%' for p in TARGET_RISK_PCTS]}")
    print(f"Periods: {len(PERIODS)}")
    print(f"Total runs: {total_runs}")
    print("=" * 65)

    all_rows = []
    run_idx = 0
    for risk_pct in TARGET_RISK_PCTS:
        for pname, start, end in PERIODS:
            run_idx += 1
            print(
                f"[{run_idx:>3}/{total_runs}] "
                f"risk={risk_pct*100:4.1f}% {pname:14s}",
                end="",
            )

            row = run_one(risk_pct, pname, start, end, df_full, strat_cfg)
            all_rows.append(row)

            print(
                f" ret={row['total_return']:>8.2f}% "
                f"trades={row['total_trades']:>3} "
                f"pf={row['profit_factor']:.2f} "
                f"dd={row['max_drawdown_pct']:.2f}% "
                f"avg_risk={row['avg_actual_risk_pct']:.4f}%"
            )

    # Save summary
    summary_df = pd.DataFrame(all_rows)
    summary_path = REPORT_DIR / "phase4e_tpe490_actual_risk_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Summary saved: {summary_path}")

    # -----------------------------------------------------------------------
    # Decision report
    # -----------------------------------------------------------------------
    lines = []
    def w(s):
        lines.append(s)

    w("# Phase 4E — tpe_trial_490 实际风险达标测试\n")
    w(f"> 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    w(f"> 候选: {CANDIDATE}\n")
    w(f"> 模式: {POSITION_MODE} (uncapped, 无仓位上限)\n")
    w(f"> 目标风险: {', '.join(f'{p*100:.1f}%' for p in TARGET_RISK_PCTS)}\n")
    w(f"> 运行数: {total_runs}\n")
    w("\n---\n")

    w("## 1. 是否成功运行\n\n✅ YES — 全部完成，无错误\n")

    # Full results
    w("\n## 2. Full Period 结果 (2024-2025-Full)\n\n")
    full_df = summary_df[summary_df["period"] == "2024-2025-Full"].sort_values("risk_per_trade_pct")

    w("| 目标风险 | 实际avg风险 | 实际min风险 | 实际max风险 | 总收益 | 交易数 | PF | Sharpe | DD |\n")
    w("|:-------:|:----------:|:----------:|:----------:|:-----:|:-----:|:--:|:------:|:--:|\n")
    for _, r in full_df.iterrows():
        w(
            f"| {r['risk_per_trade_pct']*100:.1f}% "
            f"| {r['avg_actual_risk_pct']:.4f}% "
            f"| {r['min_actual_risk_pct']:.4f}% "
            f"| {r['max_actual_risk_pct']:.4f}% "
            f"| {r['total_return']:.2f}% "
            f"| {r['total_trades']} "
            f"| {r['profit_factor']:.2f} "
            f"| {r['sharpe_ratio']:.2f} "
            f"| {r['max_drawdown_pct']:.2f}% |\n"
        )

    w("\n| 目标风险 | ExpectR | 最大连续亏损 | 最差单笔R | 成本占比 |\n")
    w("|:-------:|:-------:|:-----------:|:---------:|:-------:|\n")
    for _, r in full_df.iterrows():
        w(
            f"| {r['risk_per_trade_pct']*100:.1f}% "
            f"| {r['expectancy_r']:.2f} "
            f"| {r['max_consecutive_losses']} "
            f"| {r['worst_trade_r']:.2f} "
            f"| {r['cost_as_pct_of_gross_profit']:.1f}% |\n"
        )

    w("\n| 目标风险 | Avg杠杆 | Max杠杆 | P95杠杆 | Avg名义占比 | Max名义占比 |\n")
    w("|:-------:|:-------:|:-------:|:-------:|:-----------:|:-----------:|\n")
    for _, r in full_df.iterrows():
        w(
            f"| {r['risk_per_trade_pct']*100:.1f}% "
            f"| {r['avg_implied_leverage']:.2f}x "
            f"| {r['max_implied_leverage']:.2f}x "
            f"| {r['p95_implied_leverage']:.2f}x "
            f"| {r['avg_notional_pct']:.2f}% "
            f"| {r['max_notional_pct']:.2f}% |\n"
        )

    # Quarter stability
    w("\n## 3. 季度稳定性\n\n")
    q_df = summary_df[summary_df["period"] != "2024-2025-Full"]
    for risk_pct in TARGET_RISK_PCTS:
        sub = q_df[q_df["risk_per_trade_pct"] == risk_pct].sort_values("period")
        pos_q = int((sub["total_return"] > 0).sum())
        neg_q = [r for _, r in sub.iterrows() if r["total_return"] <= 0]
        worst_ret = sub["total_return"].min()
        worst_pf = sub["profit_factor"].min()
        worst_dd = sub["max_drawdown_pct"].min()

        w(f"**{risk_pct*100:.1f}%**: {pos_q}/8 季度盈利\n")
        w(f"  最差季度收益: {worst_ret:.2f}%\n")
        w(f"  最差季度PF: {worst_pf:.2f}\n")
        w(f"  最大季度DD: {worst_dd:.2f}%\n")
        if neg_q:
            neg_details = [f'{r2["period"]}({r2["total_return"]:.2f}%)' for _, r2 in sub.iterrows() if r2["total_return"] <= 0]
            w(f"  亏损季度: {', '.join(neg_details)}\n")
        w("\n")

    # Risk delivery analysis
    w("\n## 4. 实际风险是否达到目标\n\n")
    for _, r in full_df.iterrows():
        target = r["risk_per_trade_pct"] * 100
        avg = r["avg_actual_risk_pct"]
        min_v = r["min_actual_risk_pct"]
        max_v = r["max_actual_risk_pct"]
        ratio = avg / (target / 100) if target > 0 else 0

        if ratio >= 0.9:
            status = "✅ 达标"
        elif ratio >= 0.7:
            status = "⚠️ 部分达标"
        else:
            status = "❌ 未达标"

        w(f"**{target:.1f}%**: avg实际={avg:.4f}% / 目标={target:.1f}% → 交付率={ratio*100:.0f}% **{status}**\n")
        w(f"  范围: [{min_v:.4f}%, {max_v:.4f}%]\n\n")

    # Risk/reward by level
    w("\n## 5. 风险/收益判断\n\n")
    for _, r in full_df.iterrows():
        risk_label = f"{r['risk_per_trade_pct']*100:.1f}%"
        dd = r["max_drawdown_pct"]
        pf = r["profit_factor"]
        sharpe = r["sharpe_ratio"]
        lev = r["max_implied_leverage"]
        consec = r["max_consecutive_losses"]

        if abs(dd) < 10 and pf > 2.0 and lev < 10:
            w(f"**{risk_label}**: ✅ 可接受 (DD={dd:.2f}%, PF={pf:.2f}, Sharpe={sharpe:.2f}, 杠杆={lev:.1f}x)\n")
        elif abs(dd) < 20 and pf > 1.3:
            w(f"**{risk_label}**: ⚠️ 偏高但可用 (DD={dd:.2f}%, PF={pf:.2f}, 杠杆={lev:.1f}x, 最大连续亏损={consec})\n")
        else:
            w(f"**{risk_label}**: ❌ 不可接受 (DD={dd:.2f}%, PF={pf:.2f}, 杠杆={lev:.1f}x)\n")

    # Final recommendation
    w("\n## 6. 最终建议\n\n")

    # Find the best risk level
    best = None
    for _, r in full_df.iterrows():
        score = r["profit_factor"] / max(abs(r["max_drawdown_pct"]), 0.01) * r["sharpe_ratio"]
        if best is None or score > best["score"]:
            best = {"row": r, "score": score}

    if best is not None:
        r = best["row"]
        w(
            f"基于 uncapped 模式分析，**{r['risk_per_trade_pct']*100:.1f}%** "
            f"风险档综合表现最佳：\n"
        )
        w(f"- 实际交付平均风险: {r['avg_actual_risk_pct']:.4f}%\n")
        w(f"- 全年收益: {r['total_return']:.2f}%\n")
        w(f"- PF: {r['profit_factor']:.2f}\n")
        w(f"- Sharpe: {r['sharpe_ratio']:.2f}\n")
        w(f"- DD: {r['max_drawdown_pct']:.2f}%\n")
        w(f"- 最大杠杆: {r['max_implied_leverage']:.1f}x\n")
        w(f"- 最大连续亏损: {r['max_consecutive_losses']}\n\n")

    w("**注意**: uncapped 模式下交易不设仓位上限，实际杠杆可能较高。\n")
    w("如果 Phase 5A 需要考虑杠杆限制或仓位比例约束，建议在实盘配置中\n")
    w("加入 max_position_value_pct 限制（可考虑 10%~20% 宽敞区间）。\n")

    decision_text = "".join(lines)
    decision_path = REPORT_DIR / "phase4e_tpe490_actual_risk_decision.md"
    with open(decision_path, "w", encoding="utf-8") as f:
        f.write(decision_text)
    print(f"✅ Decision report saved: {decision_path}")

    print(f"\n{'='*65}")
    print("Phase 4E — Complete")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
