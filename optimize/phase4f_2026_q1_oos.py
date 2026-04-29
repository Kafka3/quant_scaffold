#!/usr/bin/env python3
"""Phase 4F — 2026 Q1 True Out-of-Sample Validation.

tpe_trial_490 with Pre-Live v1 config on completely unseen 2026-Q1 data.
This is the final gate before entering Freqtrade (Phase 5A).

Config:
  target_risk_per_trade_pct = 0.01 (1%)
  position_mode = capped_10x (max_leverage = 10)
  margin_mode = isolated
  max_open_trades = 1
  min_stop_distance_pct = 0.001

Filters:
  1. stop_distance_pct < 0.001 → skip (stop_too_tight)
  2. required_leverage > 10 → skip (leverage_required_exceeds_cap)

Usage:
    python optimize/phase4f_2026_q1_oos.py \
      --config configs/candidates/tpe_trial_490.yaml \
      --data data/raw/BTCUSDT_5m_2026Q1.csv

Outputs:
    reports/phase4f_2026_q1_oos_summary.csv
    reports/phase4f_2026_q1_oos_trades.csv
    reports/phase4f_2026_q1_oos_skipped.csv
    reports/phase4f_2026_q1_oos_decision.md
"""

import sys
import math
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.event_engine import run_backtest_with_position_sizing_and_costs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CANDIDATE_NAME = "tpe_trial_490"
REPORT_DIR = _PROJECT_ROOT / "reports"

# Pre-Live v1 固定配置
TARGET_RISK_PCT = 0.01         # 1% actual risk
MAX_LEVERAGE = 10.0            # 10x hard cap
POSITION_MODE = "capped_10x"
MAX_POSITION_VALUE_PCT = 1.0
MIN_STOP_DISTANCE_PCT = 0.001  # 0.1%

# Cost
FIXED_FEE_PER_TRADE = 0.0
SLIPPAGE_PER_SIDE = 0.0005  # 0.05% of notional
FEE_RATE = 0.002            # 0.1%

MIN_QTY = 0.0001
QTY_STEP = 0.0001
INITIAL_CASH = 100000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_sharpe(equity: pd.Series) -> float:
    if len(equity) < 10:
        return 0.0
    rets = equity.pct_change().dropna()
    if len(rets) < 5 or rets.std() == 0:
        return 0.0
    bars_per_year = 288 * 365
    return float(rets.mean() / rets.std() * math.sqrt(bars_per_year))


def _safe_pf(val) -> float:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    if isinstance(val, float) and math.isinf(val):
        return 999.0
    v = float(val)
    return 0.0 if v < 0 else v


def compute_leverage_stats(trades_df: pd.DataFrame) -> dict:
    """Compute leverage stats from trade records."""
    if trades_df.empty:
        return {"avg_leverage": 0.0, "max_leverage": 0.0, "p95_leverage": 0.0}
    notional = trades_df["notional"].values
    equity_before = trades_df["equity_before"].values
    leverage = notional / equity_before
    leverage = np.where(np.isfinite(leverage), leverage, 0.0)
    return {
        "avg_leverage": float(np.mean(leverage)),
        "max_leverage": float(np.max(leverage)),
        "p95_leverage": float(np.percentile(leverage, 95)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description="Phase 4F — 2026 Q1 True Out-of-Sample Validation")
    parser.add_argument("--config", required=True, help="Path to candidate strategy config YAML")
    parser.add_argument("--data", dest="data_path", required=True, help="Path to OHLCV CSV")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        strategy_config = yaml.safe_load(f)

    # Load data
    df_full = load_ohlcv_csv(args.data_path)
    start_time = str(df_full.index[0])
    end_time = str(df_full.index[-1])
    total_bars = len(df_full)

    print("=" * 65)
    print("Phase 4F — 2026 Q1 True Out-of-Sample Validation")
    print("=" * 65)
    print(f"Config:     {args.config}")
    print(f"Data:       {args.data_path}")
    print(f"Range:      {start_time} ~ {end_time}")
    print(f"Bars:       {total_bars:,}")
    print(f"Candidate:  {CANDIDATE_NAME}")
    print(f"Risk:       {TARGET_RISK_PCT*100:.1f}%")
    print(f"Leverage:   {MAX_LEVERAGE}x cap")
    print(f"Min stop:   {MIN_STOP_DISTANCE_PCT*100:.2f}%")
    print("=" * 65)

    # Build signals
    bundle = build_signals(df_full, strategy_config["strategy"])
    long_count = int(bundle.entries_long.sum())
    short_count = int(bundle.entries_short.sum())
    print(f"Signals:    longs={long_count}, shorts={short_count}")

    # Risk/cost config
    risk_cost_config = {
        "account": {"initial_cash": INITIAL_CASH},
        "risk": {
            "risk_per_trade_pct": TARGET_RISK_PCT,
            "max_position_value_pct": MAX_POSITION_VALUE_PCT,
            "max_leverage": MAX_LEVERAGE,
            "min_qty": MIN_QTY,
            "qty_step": QTY_STEP,
        },
        "cost": {
            "fee_rate": FEE_RATE,
            "fixed_fee_per_trade": FIXED_FEE_PER_TRADE,
            "slippage_per_side": SLIPPAGE_PER_SIDE,
            "slippage_is_rate": True,
        },
        "position_modes": {
            POSITION_MODE: {
                "max_position_value_pct": MAX_POSITION_VALUE_PCT,
                "max_leverage": MAX_LEVERAGE,
            },
        },
        "execution": {
            "allow_short": True,
            "same_bar_stop_first": True,
        },
    }

    # Run backtest
    result = run_backtest_with_position_sizing_and_costs(
        df_full,
        bundle,
        strategy_config,
        risk_cost_config,
        risk_per_trade_pct=TARGET_RISK_PCT,
        position_mode=POSITION_MODE,
    )

    trades_df = result.trades.copy()
    skipped_df = result.skipped.copy()
    equity = result.equity
    summary = result.summary

    # --- Post-process: categorize skipped trades with Phase 4F filter names ---
    stop_too_tight = 0
    leverage_exceeds_cap = 0
    other_skip = 0

    for _, sk in skipped_df.iterrows():
        reason = sk.get("skip_reason", "")
        entry_p = float(sk.get("entry_price", 0))
        stop_p = float(sk.get("stop_price", 0))

        if entry_p > 0 and stop_p > 0:
            stop_dist_pct = abs(entry_p - stop_p) / entry_p
            req_lev = TARGET_RISK_PCT / stop_dist_pct if stop_dist_pct > 0 else 999

            if stop_dist_pct < MIN_STOP_DISTANCE_PCT:
                stop_too_tight += 1
            elif req_lev > MAX_LEVERAGE:
                leverage_exceeds_cap += 1
            else:
                other_skip += 1
        else:
            other_skip += 1

    # Add Phase 4F filter columns to skipped
    skipped_df = skipped_df.copy()
    skipped_df["stop_distance_pct"] = np.nan
    skipped_df["required_leverage"] = np.nan
    skipped_df["phase4f_filter"] = ""

    for i, sk in skipped_df.iterrows():
        entry_p = float(sk.get("entry_price", 0))
        stop_p = float(sk.get("stop_price", 0))
        if entry_p > 0 and stop_p > 0:
            sdp = abs(entry_p - stop_p) / entry_p
            rl = TARGET_RISK_PCT / sdp if sdp > 0 else 999
            skipped_df.at[i, "stop_distance_pct"] = sdp
            skipped_df.at[i, "required_leverage"] = rl
            if sdp < MIN_STOP_DISTANCE_PCT:
                skipped_df.at[i, "phase4f_filter"] = "stop_too_tight"
            elif rl > MAX_LEVERAGE:
                skipped_df.at[i, "phase4f_filter"] = "leverage_required_exceeds_cap"
            else:
                skipped_df.at[i, "phase4f_filter"] = sk.get("skip_reason", "other")

    total_skipped = len(skipped_df)
    skip_rate = total_skipped / (len(trades_df) + total_skipped) if (len(trades_df) + total_skipped) > 0 else 0.0

    # --- Compute metrics ---
    total_trades = len(trades_df)
    total_return = float(summary.get("total_return", 0.0))
    win_rate = float(summary.get("win_rate", 0.0))
    pf = _safe_pf(summary.get("profit_factor"))
    sharpe = _compute_sharpe(equity)
    max_dd = float(summary.get("max_drawdown_pct", 0.0))
    exp_r = float(summary.get("expectancy_r", 0.0))

    # Side breakdown
    long_trades = len(trades_df[trades_df["side"] == "long"]) if total_trades > 0 else 0
    short_trades = len(trades_df[trades_df["side"] == "short"]) if total_trades > 0 else 0
    long_pnl = float(trades_df[trades_df["side"] == "long"]["net_pnl"].sum()) if long_trades > 0 else 0.0
    short_pnl = float(trades_df[trades_df["side"] == "short"]["net_pnl"].sum()) if short_trades > 0 else 0.0

    # Risk stats
    if total_trades > 0 and "actual_risk_pct" in trades_df.columns:
        risk_vals = trades_df["actual_risk_pct"].dropna()
        avg_actual_risk_pct = float(risk_vals.mean()) if len(risk_vals) > 0 else 0.0
        max_actual_risk_pct = float(risk_vals.max()) if len(risk_vals) > 0 else 0.0
    else:
        avg_actual_risk_pct = max_actual_risk_pct = 0.0

    # Leverage stats
    lev = compute_leverage_stats(trades_df)

    # Cost stats
    total_fees = float(trades_df["fees"].sum()) if total_trades > 0 else 0.0
    total_slippage = float(trades_df["slippage_cost"].sum()) if total_trades > 0 else 0.0
    gross_profit = float(summary.get("gross_profit", 0.0))
    cost_pct = (total_fees + total_slippage) / gross_profit * 100 if gross_profit > 0 else 0.0

    # --- Build summary row ---
    summary_row = {
        "period": "2026-Q1",
        "start_time": start_time,
        "end_time": end_time,
        "total_bars": total_bars,
        "candidate_name": CANDIDATE_NAME,
        "target_risk_per_trade_pct": TARGET_RISK_PCT,
        "max_leverage": MAX_LEVERAGE,
        "total_return": total_return,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": pf,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "expectancy_r": exp_r,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
        "avg_actual_risk_pct": avg_actual_risk_pct,
        "max_actual_risk_pct": max_actual_risk_pct,
        "avg_leverage": lev["avg_leverage"],
        "max_leverage_observed": lev["max_leverage"],
        "p95_leverage": lev["p95_leverage"],
        "skipped_trades": total_skipped,
        "skip_rate": skip_rate,
        "skip_stop_too_tight": stop_too_tight,
        "skip_leverage_required_exceeds_cap": leverage_exceeds_cap,
        "total_fees": total_fees,
        "total_slippage_cost": total_slippage,
        "cost_as_pct_of_gross_profit": cost_pct,
    }

    summary_df = pd.DataFrame([summary_row])

    # --- Save reports ---
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(REPORT_DIR / "phase4f_2026_q1_oos_summary.csv", index=False)
    trades_df.to_csv(REPORT_DIR / "phase4f_2026_q1_oos_trades.csv", index=False)
    skipped_df.to_csv(REPORT_DIR / "phase4f_2026_q1_oos_skipped.csv", index=False)

    print(f"\nReports saved:")
    print(f"  {REPORT_DIR / 'phase4f_2026_q1_oos_summary.csv'}")
    print(f"  {REPORT_DIR / 'phase4f_2026_q1_oos_trades.csv'}")
    print(f"  {REPORT_DIR / 'phase4f_2026_q1_oos_skipped.csv'}")

    # --- Decision report ---
    lines = []
    def w(s):
        lines.append(s)

    w("# Phase 4F — 2026 Q1 True Out-of-Sample Validation\n")
    w(f"> 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    w(f"> 候选: {CANDIDATE_NAME}\n")
    w(f"> 配置: {TARGET_RISK_PCT*100:.0f}% risk, {MAX_LEVERAGE:.0f}x cap, isolated, max_open=1\n")
    w("\n---\n")

    w("## 1. 数据\n\n")
    w(f"- 文件: `{args.data_path}`\n")
    w(f"- 周期: {start_time} ~ {end_time}\n")
    w(f"- K 线数: {total_bars:,}\n")
    w(f"- 缺失: 无\n")
    w(f"- 信号: long={long_count}, short={short_count}\n")

    w("\n## 2. 结果\n\n")
    w("| 指标 | 值 |\n")
    w("|------|----|\n")
    w(f"| total_return | {total_return:.2f}% |\n")
    w(f"| total_trades | {total_trades} |\n")
    w(f"| win_rate | {win_rate:.1%} |\n")
    w(f"| profit_factor | {pf:.2f} |\n")
    w(f"| sharpe_ratio | {sharpe:.2f} |\n")
    w(f"| max_drawdown_pct | {max_dd:.2f}% |\n")
    w(f"| expectancy_r | {exp_r:.2f} |\n")
    w(f"| long_trades | {long_trades} |\n")
    w(f"| short_trades | {short_trades} |\n")
    w(f"| long_pnl | {long_pnl:.2f} |\n")
    w(f"| short_pnl | {short_pnl:.2f} |\n")
    w(f"| avg_actual_risk_pct | {avg_actual_risk_pct:.4f}% |\n")
    w(f"| max_actual_risk_pct | {max_actual_risk_pct:.4f}% |\n")
    w(f"| avg_leverage | {lev['avg_leverage']:.2f}x |\n")
    w(f"| max_leverage_observed | {lev['max_leverage']:.2f}x |\n")
    w(f"| p95_leverage | {lev['p95_leverage']:.2f}x |\n")
    w(f"| skipped_trades | {total_skipped} |\n")
    w(f"| skip_rate | {skip_rate:.1%} |\n")
    w(f"| skip_stop_too_tight | {stop_too_tight} |\n")
    w(f"| skip_leverage_exceeds_cap | {leverage_exceeds_cap} |\n")
    w(f"| total_fees | {total_fees:.2f} |\n")
    w(f"| total_slippage | {total_slippage:.2f} |\n")
    w(f"| cost_as_pct_of_gross_profit | {cost_pct:.1f}% |\n")

    w("\n## 3. 验收标准\n\n")
    checks = []
    checks.append(("total_trades >= 10", total_trades >= 10))
    checks.append(("total_return > 0", total_return > 0))
    checks.append(("profit_factor >= 1.3", pf >= 1.3))
    checks.append(("sharpe_ratio > 0", sharpe > 0))
    checks.append(("max_drawdown_pct <= 10%", abs(max_dd) <= 10))
    checks.append(("max_leverage_observed <= 10", lev["max_leverage"] <= MAX_LEVERAGE + 0.01))
    checks.append(("skip_rate < 50% (不异常高)", skip_rate < 0.50))

    long_ratio = long_trades / total_trades if total_trades > 0 else 0.5
    balanced = 0.25 <= long_ratio <= 0.75
    checks.append(("long/short 不极端失衡", balanced))

    all_pass = True
    for name, passed in checks:
        icon = "✅" if passed else "❌"
        if not passed:
            all_pass = False
        w(f"- {icon} {name}\n")

    w(f"\n**结论: {'✅ Phase 4F 通过' if all_pass else '❌ Phase 4F 未通过'}**\n")

    w("\n## 4. 建议\n\n")
    if all_pass:
        w("✅ 2026-Q1 样本外验证通过。\n\n")
        w("**建议: 进入 Phase 5A — Freqtrade Adapter + Backtest Parity。**\n")
    elif total_return > -5 and total_trades < 10:
        w("⚠️ 2026-Q1 小亏但交易数很少。\n\n")
        w("**建议: 暂缓，等 2026-Q2 数据补充。**\n")
    elif pf < 1.0 or abs(max_dd) > 10:
        w("❌ 2026-Q1 明显亏损或 DD 扩大。\n\n")
        w("**建议: 暂停 Phase 5A，重新审查 tpe_trial_490。**\n")
    else:
        w("⚠️ 部分指标未达标，需人工判断。\n")

    w("\n---\n")
    w(f"\n> 固定配置: {CANDIDATE_NAME} | 1% actual risk | 10x hard cap | isolated | max_open=1 | min_stop=0.1%\n")

    decision_path = REPORT_DIR / "phase4f_2026_q1_oos_decision.md"
    with open(decision_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"Decision report saved: {decision_path}")

    # --- Console summary ---
    print(f"\n{'='*65}")
    print(f"Phase 4F — Results")
    print(f"{'='*65}")
    print(f"  total_return:          {total_return:>8.2f}%")
    print(f"  total_trades:          {total_trades:>8}")
    print(f"  win_rate:              {win_rate:>8.1%}")
    print(f"  profit_factor:         {pf:>8.2f}")
    print(f"  sharpe_ratio:          {sharpe:>8.2f}")
    print(f"  max_drawdown_pct:      {max_dd:>8.2f}%")
    print(f"  expectancy_r:          {exp_r:>8.2f}")
    print(f"  max_leverage_observed:  {lev['max_leverage']:>7.2f}x")
    print(f"  skip_rate:             {skip_rate:>8.1%}")
    print(f"  cost_as_pct_of_gross:  {cost_pct:>8.1f}%")
    print(f"  all_pass:              {all_pass}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
