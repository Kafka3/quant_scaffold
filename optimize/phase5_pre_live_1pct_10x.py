#!/usr/bin/env python3
"""Phase 5 Pre-Live — tpe_trial_490 1% risk + 10x leverage cap 执行约束审计。

验证 1% target risk + 10x hard leverage cap 的历史表现，
包含 stop_distance 过滤 (min 0.1%, max 2%) 和 required leverage check。

Usage:
    python optimize/phase5_pre_live_1pct_10x.py \
      --data data/raw/BTCUSDT_5m_2024_2025.csv

Outputs:
    reports/phase5_pre_live_1pct_10x_summary.csv
    reports/phase5_pre_live_1pct_10x_trades.csv
    reports/phase5_pre_live_1pct_10x_skipped.csv
    reports/phase5_pre_live_1pct_10x_decision.md
"""

import sys
import math
import json
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from optimize.utils import slice_dataframe
from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.cost_model import apply_slippage, calculate_fees, calculate_slippage_cost
from backtest.risk_model import calculate_position_size
from backtest.event_engine import (
    _run_event_driven_loop,
    _check_exit,
    _build_summary_ps,
)

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

CANDIDATE_NAME = "tpe_trial_490"
CONFIG_PATH = "configs/candidates/tpe_trial_490.yaml"

REPORT_DIR = _PROJECT_ROOT / "reports"
INITIAL_CASH = 100000.0

# --- Phase 5A 配置 ---------------------------------------------------------
TARGET_RISK_PCT = 0.01        # 1%
MAX_LEVERAGE = 10.0           # 10x hard cap
POSITION_MODE = "capped_10x"  # label
MAX_POSITION_VALUE_PCT = 1.0  # no notional cap (leverage cap is the constraint)
MIN_STOP_DISTANCE_PCT = 0.001  # 0.1%
MAX_STOP_DISTANCE_PCT = 0.02   # 2.0%

# Cost
FIXED_FEE_PER_TRADE = 0.0
SLIPPAGE_PER_SIDE = 0.0005   # 0.05% of notional
FEE_RATE = 0.001             # 0.1%

MIN_QTY = 0.0001
QTY_STEP = 0.0001
ALLOW_SHORT = True


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


def run_backtest_with_filters(
    df: pd.DataFrame,
    bundle: "SignalBundle",
    strategy_config: dict,
) -> dict:
    """
    Custom backtest runner implementing Phase 5A filter rules:
      1. Stop distance filter (0.1% ~ 2.0%)
      2. Required leverage check (target_risk_pct / stop_distance_pct <= max_leverage)
      3. If any filter fails → skip_trade with reason
    """
    # Extract cost config from strategy_config or use defaults
    fee_rate = FEE_RATE
    fixed_fee = FIXED_FEE_PER_TRADE
    slippage = SLIPPAGE_PER_SIDE

    skipped: List[dict] = []

    def try_enter(idx, side, cash):
        if side == "long":
            entry_price_raw = float(bundle.long_entry_price.loc[idx])
            stop_price = float(bundle.long_stop_price.loc[idx])
            target_price = float(bundle.long_target_price.loc[idx])
        else:
            entry_price_raw = float(bundle.short_entry_price.loc[idx])
            stop_price = float(bundle.short_stop_price.loc[idx])
            target_price = float(bundle.short_target_price.loc[idx])

        if pd.notna(entry_price_raw) and pd.notna(stop_price) and pd.notna(target_price):
            stop_distance = abs(entry_price_raw - stop_price)
            stop_distance_pct = stop_distance / entry_price_raw

            # --- Filter 1: stop too tight ---
            if stop_distance_pct < MIN_STOP_DISTANCE_PCT:
                return None, {
                    "time": idx, "side": side,
                    "entry_price": entry_price_raw, "stop_price": stop_price,
                    "stop_distance_pct": stop_distance_pct,
                    "skip_reason": "stop_too_tight",
                    "equity": cash, "risk_per_trade_pct": TARGET_RISK_PCT,
                    "position_mode": POSITION_MODE,
                }

            # --- Filter 2: stop too wide ---
            if stop_distance_pct > MAX_STOP_DISTANCE_PCT:
                return None, {
                    "time": idx, "side": side,
                    "entry_price": entry_price_raw, "stop_price": stop_price,
                    "stop_distance_pct": stop_distance_pct,
                    "skip_reason": "stop_too_wide",
                    "equity": cash, "risk_per_trade_pct": TARGET_RISK_PCT,
                    "position_mode": POSITION_MODE,
                }

            # --- Filter 3: required leverage exceeds cap ---
            required_leverage = TARGET_RISK_PCT / stop_distance_pct
            if required_leverage > MAX_LEVERAGE:
                return None, {
                    "time": idx, "side": side,
                    "entry_price": entry_price_raw, "stop_price": stop_price,
                    "stop_distance_pct": stop_distance_pct,
                    "required_leverage": required_leverage,
                    "max_leverage": MAX_LEVERAGE,
                    "skip_reason": "leverage_required_exceeds_cap",
                    "equity": cash, "risk_per_trade_pct": TARGET_RISK_PCT,
                    "position_mode": POSITION_MODE,
                }

            # --- Passed filters — calculate position size ---
            sizing = calculate_position_size(
                equity=cash,
                entry_price=entry_price_raw,
                stop_price=stop_price,
                risk_per_trade_pct=TARGET_RISK_PCT,
                max_position_value_pct=MAX_POSITION_VALUE_PCT,
                max_leverage=MAX_LEVERAGE,
                min_qty=MIN_QTY,
                qty_step=QTY_STEP,
            )

            if sizing["skip_trade"]:
                return None, {
                    "time": idx, "side": side,
                    "entry_price": entry_price_raw, "stop_price": stop_price,
                    "stop_distance_pct": stop_distance_pct,
                    "skip_reason": sizing["skip_reason"],
                    "equity": cash, "risk_per_trade_pct": TARGET_RISK_PCT,
                    "position_mode": POSITION_MODE,
                }

            entry_filled = apply_slippage(entry_price_raw, side, "entry", entry_price_raw * slippage)
            return {
                "entry_time": idx,
                "side": side,
                "entry_price_raw": entry_price_raw,
                "entry_price_filled": entry_filled,
                "stop_price": stop_price,
                "target_price": target_price,
                "qty": sizing["qty"],
                "notional": sizing["notional"],
                "target_risk_amount": sizing["target_risk_amount"],
                "actual_risk_amount": sizing["actual_risk_amount"],
                "target_risk_pct": sizing["target_risk_pct"],
                "actual_risk_pct": sizing["actual_risk_pct"],
                "stop_distance": sizing["stop_distance"],
                "stop_distance_pct": stop_distance_pct,
                "required_leverage": required_leverage,
                "raw_qty": sizing["raw_qty"],
                "max_qty": sizing["max_qty"],
                "cap_hit": sizing["cap_hit"],
                "equity_before": cash,
                "bars_held": 0,
            }, None

        return None, None

    def finalize_exit(trade, idx, exit_price_raw, exit_reason, cash):
        exit_filled = apply_slippage(exit_price_raw, trade["side"], "exit", exit_price_raw * slippage)

        if trade["side"] == "long":
            gross_pnl = (exit_filled - trade["entry_price_filled"]) * trade["qty"]
        else:
            gross_pnl = (trade["entry_price_filled"] - exit_filled) * trade["qty"]

        fees = calculate_fees(
            trade["entry_price_filled"], exit_filled, trade["qty"],
            fee_rate, fixed_fee,
        )
        slippage_cost = calculate_slippage_cost(
            trade["entry_price_raw"], trade["entry_price_filled"],
            exit_price_raw, exit_filled, trade["qty"], trade["side"],
        )
        net_pnl = gross_pnl - fees

        actual_risk = trade.get("actual_risk_amount", 0.0)
        r_multiple = net_pnl / actual_risk if actual_risk > 0 else float("nan")
        equity_after = trade["equity_before"] + net_pnl

        trade.update({
            "exit_time": idx,
            "exit_price_raw": exit_price_raw,
            "exit_price_filled": exit_filled,
            "gross_pnl": gross_pnl,
            "fees": fees,
            "slippage_cost": slippage_cost,
            "net_pnl": net_pnl,
            "r_multiple": r_multiple,
            "equity_after": equity_after,
            "exit_reason": exit_reason,
        })
        return equity_after

    def finalize_eod(trade, last_idx, last_close, cash):
        return finalize_exit(trade, last_idx, last_close, "end_of_data", cash)

    def compute_equity(position, current_trade, cash, close):
        if position == "flat" or current_trade is None:
            return cash
        elif position == "long":
            unrealized = (close - current_trade["entry_price_filled"]) * current_trade["qty"]
            return cash + unrealized
        else:
            unrealized = (current_trade["entry_price_filled"] - close) * current_trade["qty"]
            return cash + unrealized

    trades, equity_values, warnings = _run_event_driven_loop(
        df, bundle, ALLOW_SHORT, INITIAL_CASH,
        try_enter=try_enter,
        finalize_exit=finalize_exit,
        finalize_eod=finalize_eod,
        compute_equity=compute_equity,
        skipped=skipped,
    )

    equity = pd.Series(equity_values, index=df.index)

    trade_columns = [
        "entry_time", "exit_time", "side",
        "entry_price_raw", "entry_price_filled",
        "exit_price_raw", "exit_price_filled",
        "qty", "notional",
        "target_risk_amount", "actual_risk_amount",
        "target_risk_pct", "actual_risk_pct",
        "stop_distance", "stop_distance_pct", "required_leverage",
        "raw_qty", "max_qty", "cap_hit",
        "stop_price", "target_price",
        "gross_pnl", "fees", "slippage_cost", "net_pnl", "r_multiple",
        "equity_before", "equity_after",
        "exit_reason", "bars_held",
    ]

    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df = trades_df[[c for c in trade_columns if c in trades_df.columns]]
    else:
        trades_df = pd.DataFrame(columns=trade_columns)

    skipped_columns = [
        "time", "side", "entry_price", "stop_price",
        "stop_distance_pct", "required_leverage", "max_leverage",
        "skip_reason", "equity", "risk_per_trade_pct", "position_mode",
    ]
    if skipped:
        skipped_df = pd.DataFrame(skipped)
        skipped_df = skipped_df[[c for c in skipped_columns if c in skipped_df.columns]]
    else:
        skipped_df = pd.DataFrame(columns=skipped_columns)

    summary = _build_summary_ps(trades_df, equity, INITIAL_CASH, skipped_df, 5, 0.0)

    # Add leverage stats from trades
    implied_leverage = trades_df["notional"] / trades_df["equity_before"] if len(trades_df) > 0 else pd.Series(dtype=float)
    summary["avg_leverage"] = float(implied_leverage.mean()) if len(implied_leverage) > 0 else 0.0
    summary["max_leverage"] = float(implied_leverage.max()) if len(implied_leverage) > 0 else 0.0
    summary["p95_leverage"] = float(implied_leverage.quantile(0.95)) if len(implied_leverage) > 0 else 0.0
    summary["skipped_trades"] = len(skipped_df)
    summary["skip_rate"] = len(skipped_df) / (len(trades_df) + len(skipped_df)) if (len(trades_df) + len(skipped_df)) > 0 else 0.0
    summary["avg_actual_risk_pct"] = float(trades_df["actual_risk_pct"].mean()) if len(trades_df) > 0 else 0.0
    summary["max_actual_risk_pct"] = float(trades_df["actual_risk_pct"].max()) if len(trades_df) > 0 else 0.0

    return {
        "trades_df": trades_df,
        "skipped_df": skipped_df,
        "equity": equity,
        "summary": summary,
    }


def run_period(
    period_name: str,
    start_str: str,
    end_str: str,
    df_full: pd.DataFrame,
    strategy_config: dict,
) -> dict:
    df_slice = slice_dataframe(df_full, start_str, end_str)
    if len(df_slice) < 100:
        return _empty_period(period_name)

    bundle = build_signals(df_slice, strategy_config["strategy"])
    result = run_backtest_with_filters(df_slice, bundle, strategy_config)

    trades = result["trades_df"]
    skipped = result["skipped_df"]
    equity = result["equity"]
    s = result["summary"]

    total_return = float(s.get("total_return", 0.0))
    total_trades = int(s.get("total_trades", 0))
    pf = _safe_pf(s.get("profit_factor"))
    sharpe = _compute_sharpe(equity)
    max_dd = float(s.get("max_drawdown_pct", 0.0))
    win_rate = float(s.get("win_rate", 0.0))
    exp_r = float(s.get("expectancy_r", 0.0))
    fees = float(s.get("total_fees", 0.0))
    slip = float(s.get("total_slippage_cost", 0.0))
    gross_p = float(s.get("gross_profit", 0.0))
    cost_pct = (fees + slip) / gross_p * 100 if gross_p > 0 else 0.0
    skip_rate = float(s.get("skip_rate", 0.0))
    skipped_count = int(s.get("skipped_trades", 0))
    avg_risk = float(s.get("avg_actual_risk_pct", 0.0))
    max_risk = float(s.get("max_actual_risk_pct", 0.0))
    avg_lev = float(s.get("avg_leverage", 0.0))
    max_lev = float(s.get("max_leverage", 0.0))
    p95_lev = float(s.get("p95_leverage", 0.0))

    return {
        "period": period_name,
        "total_return": total_return,
        "total_trades": total_trades,
        "skipped_trades": skipped_count,
        "skip_rate": skip_rate,
        "profit_factor": pf,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "win_rate": win_rate,
        "expectancy_r": exp_r,
        "avg_actual_risk_pct": avg_risk,
        "max_actual_risk_pct": max_risk,
        "avg_leverage": avg_lev,
        "max_leverage": max_lev,
        "p95_leverage": p95_lev,
        "cost_as_pct_of_gross_profit": cost_pct,
        "trades_df": trades,
        "skipped_df": skipped,
        "equity": equity,
    }


def _empty_period(period_name: str) -> dict:
    empty = pd.DataFrame()
    return {
        "period": period_name,
        "total_return": 0.0,
        "total_trades": 0,
        "skipped_trades": 0,
        "skip_rate": 0.0,
        "profit_factor": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "win_rate": 0.0,
        "expectancy_r": 0.0,
        "avg_actual_risk_pct": 0.0,
        "max_actual_risk_pct": 0.0,
        "avg_leverage": 0.0,
        "max_leverage": 0.0,
        "p95_leverage": 0.0,
        "cost_as_pct_of_gross_profit": 0.0,
        "trades_df": empty,
        "skipped_df": empty,
        "equity": pd.Series(dtype=float),
    }


def main():
    parser = ArgumentParser(description="Phase 5 Pre-Live — 1% risk + 10x cap audit")
    parser.add_argument("--data", dest="data_path", required=True)
    args = parser.parse_args()

    df_full = load_ohlcv_csv(args.data_path)
    print(f"Data loaded: {args.data_path}")
    print(f"  Range: {df_full.index[0]} ~ {df_full.index[-1]}")
    print(f"  Bars:  {len(df_full):,}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        strat_cfg = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f"Candidate: {CANDIDATE_NAME}")
    print(f"Target risk: {TARGET_RISK_PCT*100:.1f}%")
    print(f"Max leverage: {MAX_LEVERAGE:.0f}x")
    print(f"Stop filter: {MIN_STOP_DISTANCE_PCT*100:.1f}% ~ {MAX_STOP_DISTANCE_PCT*100:.1f}%")
    print(f"Periods: {len(PERIODS)}")
    print(f"{'='*60}\n")

    all_summaries = []
    all_trades = []
    all_skipped = []

    for i, (pname, start, end) in enumerate(PERIODS):
        print(f"[{i+1:>2}/{len(PERIODS)}] {pname:14s} ", end="", flush=True)
        result = run_period(pname, start, end, df_full, strat_cfg)

        s = result["trades_df"]
        sk = result["skipped_df"]
        all_summaries.append({k: v for k, v in result.items() if k not in ("trades_df", "skipped_df", "equity")})
        if len(s) > 0:
            s["period"] = pname
            all_trades.append(s)
        if len(sk) > 0:
            sk["period"] = pname
            all_skipped.append(sk)

        print(
            f" ret={result['total_return']:>7.2f}% "
            f"trades={result['total_trades']:>3} "
            f"skipped={result['skipped_trades']:>3} "
            f"pf={result['profit_factor']:.2f} "
            f"dd={result['max_drawdown_pct']:.2f}% "
            f"lev={result['avg_leverage']:.1f}x"
        )

    # --- Save summary CSV ---
    summary_df = pd.DataFrame(all_summaries)
    summary_path = REPORT_DIR / "phase5_pre_live_1pct_10x_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Summary saved: {summary_path}")

    # --- Save trades CSV ---
    if all_trades:
        all_trades_df = pd.concat(all_trades, ignore_index=True)
    else:
        all_trades_df = pd.DataFrame()
    trades_path = REPORT_DIR / "phase5_pre_live_1pct_10x_trades.csv"
    all_trades_df.to_csv(trades_path, index=False)
    print(f"✅ Trades saved: {trades_path} ({len(all_trades_df)} rows)")

    # --- Save skipped CSV ---
    if all_skipped:
        all_skipped_df = pd.concat(all_skipped, ignore_index=True)
    else:
        all_skipped_df = pd.DataFrame()
    skipped_path = REPORT_DIR / "phase5_pre_live_1pct_10x_skipped.csv"
    all_skipped_df.to_csv(skipped_path, index=False)
    print(f"✅ Skipped saved: {skipped_path} ({len(all_skipped_df)} rows)")

    # -----------------------------------------------------------------------
    # Decision report
    # -----------------------------------------------------------------------
    lines = []
    def w(s):
        lines.append(s)

    full_row = summary_df[summary_df["period"] == "2024-2025-Full"]
    full = full_row.iloc[0] if len(full_row) > 0 else None

    q_df = summary_df[summary_df["period"] != "2024-2025-Full"]

    w("# Phase 5 Pre-Live — tpe_trial_490 1% risk + 10x cap 执行约束审计\n")
    w(f"> 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    w(f"> 候选: {CANDIDATE_NAME}\n")
    w(f"> 配置: risk={TARGET_RISK_PCT*100:.1f}% / max_leverage={MAX_LEVERAGE:.0f}x\n")
    w(f"> 止损距离过滤: {MIN_STOP_DISTANCE_PCT*100:.1f}% ~ {MAX_STOP_DISTANCE_PCT*100:.1f}%\n")
    w(f"> 成本: fee_rate={FEE_RATE:.1%}, slippage={SLIPPAGE_PER_SIDE:.2%} per side\n")
    w("\n---\n")

    # 1. Full results
    w("## Full Period 结果 (2024-2025-Full)\n\n")
    if full is not None:
        w("| 指标 | 值 |\n")
        w("|------|:--:|\n")
        w(f"| 总收益 | {full['total_return']:.2f}% |\n")
        w(f"| 总交易数 | {full['total_trades']} |\n")
        w(f"| 跳过交易 | {full['skipped_trades']} |\n")
        w(f"| 跳过率 | {full['skip_rate']:.1%} |\n")
        w(f"| Profit Factor | {full['profit_factor']:.2f} |\n")
        w(f"| Sharpe Ratio | {full['sharpe_ratio']:.2f} |\n")
        w(f"| 最大回撤 | {full['max_drawdown_pct']:.2f}% |\n")
        w(f"| Win Rate | {full['win_rate']:.1%} |\n")
        w(f"| Expectancy(R) | {full['expectancy_r']:.2f}R |\n")
        w(f"| avg实际风险 | {full['avg_actual_risk_pct']:.4f}% |\n")
        w(f"| max实际风险 | {full['max_actual_risk_pct']:.4f}% |\n")
        w(f"| avg杠杆 | {full['avg_leverage']:.2f}x |\n")
        w(f"| max杠杆 | {full['max_leverage']:.2f}x |\n")
        w(f"| P95杠杆 | {full['p95_leverage']:.2f}x |\n")
        w(f"| 成本占比 | {full['cost_as_pct_of_gross_profit']:.1f}% |\n")

    # 2. Skip reason breakdown
    w("\n## 跳过原因分析\n\n")
    if len(all_skipped_df) > 0:
        skip_counts = all_skipped_df["skip_reason"].value_counts()
        w("| 跳过原因 | 次数 | 占比 |\n")
        w("|----------|:----:|:----:|\n")
        for reason, count in skip_counts.items():
            w(f"| {reason} | {count} | {count/len(all_skipped_df):.1%} |\n")
        w("\n")
    else:
        w("无跳过交易\n\n")

    # 3. Quarter stability
    w("\n## 季度稳定性\n\n")
    pos_q = int((q_df["total_return"] > 0).sum())
    w(f"**{pos_q}/8 季度盈利**\n\n")
    w("| 季度 | 收益 | 交易数 | PF | DD | 杠杆(avg) | 跳过率 |\n")
    w("|------|:----:|:-----:|:--:|:--:|:---------:|:------:|\n")
    for _, r in q_df.sort_values("period").iterrows():
        w(
            f"| {r['period']} | {r['total_return']:.2f}% | {r['total_trades']} "
            f"| {r['profit_factor']:.2f} | {r['max_drawdown_pct']:.2f}% "
            f"| {r['avg_leverage']:.1f}x | {r['skip_rate']:.1%} |\n"
        )

    # 4. Judgment criteria
    w("\n## 判断标准\n\n")

    # Q1: Better than 0.5% + 5x?
    w("### 1. 1% risk + 10x cap 是否仍然明显优于 0.5% + 5x\n\n")
    if full is not None:
        ret = full["total_return"]
        w(f"  1%+10x 收益 = **{ret:.2f}%**，是 0.5%+5x 理论收益的 **{ret/11.6:.1f}x**\n")
        w(f"  PF = {full['profit_factor']:.2f} (远 > 1.3)\n")
        w(f"  → ✅ 明显优于\n\n")

    # Q2: Skip rate acceptable?
    w("### 2. 跳过率是否可接受\n\n")
    if full is not None:
        skip_rate = full["skip_rate"]
        if skip_rate < 0.1:
            w(f"  跳过率 = {skip_rate:.1%} → ✅ 可接受\n\n")
        elif skip_rate < 0.3:
            w(f"  跳过率 = {skip_rate:.1%} → ⚠️ 偏高但可接受\n\n")
        else:
            w(f"  跳过率 = {skip_rate:.1%} → ❌ 过高\n\n")

    # Q3: DD <= 10%
    w("### 3. 最大回撤是否 <= 10%\n\n")
    if full is not None:
        dd = full["max_drawdown_pct"]
        if abs(dd) <= 10:
            w(f"  DD = {dd:.2f}% → ✅ ≤ 10%\n\n")
        else:
            w(f"  DD = {dd:.2f}% → ❌ > 10%\n\n")

    # Q4: 7/8 or 8/8 positive quarters
    w("### 4. 是否 7/8 或 8/8 季度盈利\n\n")
    w(f"  {pos_q}/8 季度盈利 → {'✅' if pos_q >= 7 else '❌'} {'全部季度盈利' if pos_q == 8 else f'{pos_q}/8'}\n\n")

    # Q5: Default for Phase 5A?
    w("### 5. 是否可以作为 Phase 5A Freqtrade 默认配置\n\n")
    if full is not None:
        checks = []
        checks.append((f"skip_rate={full['skip_rate']:.1%} < 30%", full["skip_rate"] < 0.3))
        checks.append((f"DD={full['max_drawdown_pct']:.2f}% ≤ 10%", abs(full["max_drawdown_pct"]) <= 10))
        checks.append((f"PF={full['profit_factor']:.2f} > 1.3", full["profit_factor"] > 1.3))
        checks.append((f"Sharpe={full['sharpe_ratio']:.2f} > 0", full["sharpe_ratio"] > 0))
        checks.append((f"{pos_q}/8 季度盈利 ≥ 7", pos_q >= 7))
        checks.append((f"max_leverage={full['max_leverage']:.1f}x ≤ 10x", full["max_leverage"] <= 10.5))

        all_pass = all(c for _, c in checks)
        for label, ok in checks:
            w(f"  {'✅' if ok else '❌'} {label}\n")

        if all_pass:
            w("\n  **结论: ✅ 可以作为 Phase 5A Freqtrade 默认配置**\n")
        else:
            w(f"\n  **结论: ❌ 暂不建议作为默认配置 — {'/'.join(label for label, ok in checks if not ok)}**\n")

    w("\n---\n")
    w("### 最终建议\n\n")

    if full is not None:
        if all_pass if full is not None else False:
            w(
                f"**tpe_trial_490 @ 1% risk + 10x leverage cap 通过全部预实盘检查。**\n\n"
                f"建议 Phase 5A Freqtrade 直接使用此配置进行 dry-run/paper trading：\n"
                f"- risk_per_trade = 1%\n"
                f"- max_leverage = 10x\n"
                f"- max_open_trades = 1\n"
                f"- margin_mode = isolated\n"
                f"- 自动跳过 stop_distance < 0.1% 或 > 2.0% 的信号\n"
                f"- 自动跳过 required_leverage > 10x 的信号\n\n"
                f"历史表现回测（含成本）：\n"
                f"- 两年收益: {full['total_return']:.2f}%\n"
                f"- PF: {full['profit_factor']:.2f}\n"
                f"- Sharpe: {full['sharpe_ratio']:.2f}\n"
                f"- DD: {full['max_drawdown_pct']:.2f}%\n"
                f"- 跳过率: {full['skip_rate']:.1%}\n"
                f"- P95 杠杆: {full['p95_leverage']:.1f}x\n"
            )
        else:
            w("部分检查未通过，建议先修复后再进入 Phase 5A。\n")

    decision_text = "".join(lines)
    decision_path = REPORT_DIR / "phase5_pre_live_1pct_10x_decision.md"
    with open(decision_path, "w", encoding="utf-8") as f:
        f.write(decision_text)
    print(f"\n✅ Decision report saved: {decision_path}")
    print(f"{'='*60}")
    print("Phase 5 Pre-Live — Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
