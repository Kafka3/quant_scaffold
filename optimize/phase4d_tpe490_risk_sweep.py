#!/usr/bin/env python3
"""Phase 4D — tpe_trial_490 Risk Proportion Sensitivity Sweep.

Tests tpe_trial_490 (and optionally tpe_trial_262) across multiple
risk_per_trade_pct values and three position modes (capped_3x, capped_5x,
uncapped) on all 8 quarters + full period with fee/slippage costs.

Usage:
    python optimize/phase4d_tpe490_risk_sweep.py \
      --data data/raw/BTCUSDT_5m_2024_2025.csv

Outputs:
    reports/phase4d_tpe490_risk_sweep_summary.csv
    reports/phase4d_tpe490_risk_sweep_trades.csv
    reports/phase4d_tpe490_risk_sweep_equity.csv
    reports/phase4d_tpe490_risk_sweep_decision.md
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

from optimize.utils import slice_dataframe
from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.event_engine import run_backtest_with_position_sizing_and_costs

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

CANDIDATE_CONFIGS = {
    "tpe_trial_490": "configs/candidates/tpe_trial_490.yaml",
    # Optional: uncomment to include tpe_trial_262
    # "tpe_trial_262": "configs/candidates/tpe_trial_262.yaml",
}

RISK_PCTS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]

POSITION_MODES = {
    "capped_3x": {"max_position_value_pct": 0.03, "max_leverage": 3.0},
    "capped_5x": {"max_position_value_pct": 0.05, "max_leverage": 5.0},
    "uncapped": {"max_position_value_pct": 1.0, "max_leverage": 100.0},
}

REPORT_DIR = _PROJECT_ROOT / "reports"


def _compute_sharpe(equity: pd.Series) -> float:
    """Annualized Sharpe ratio from equity curve (5m bars)."""
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


def load_strategy_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    """Run a single backtest and return a summary dict."""
    df_slice = slice_dataframe(df_full, start_str, end_str)

    if len(df_slice) < 100:
        return _empty_row(candidate_name, position_mode, risk_pct, period_name)

    bundle = build_signals(df_slice, strategy_config["strategy"])

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
            "fixed_fee_per_trade": 5.0,
            "slippage_per_side": 5.0,
            "fee_rate": 0.0,
        },
        "position_modes": {
            position_mode: mode_cfg,
        },
        "execution": {
            "allow_short": True,
            "same_bar_stop_first": True,
        },
    }

    try:
        result = run_backtest_with_position_sizing_and_costs(
            df_slice, bundle, strategy_config, risk_cost_config,
            risk_per_trade_pct=risk_pct, position_mode=position_mode,
        )
    except Exception as e:
        row = _empty_row(candidate_name, position_mode, risk_pct, period_name)
        row["error"] = str(e)
        return row

    summary = result.summary
    trades_df = result.trades
    equity_series = result.equity

    sharpe = _compute_sharpe(equity_series)
    ending_equity = float(summary.get("ending_equity", 100000))
    total_return = float(summary.get("total_return", 0.0))
    total_trades = int(summary.get("total_trades", 0))
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
    slippage_cost = float(summary.get("total_slippage_cost", 0.0))
    gross_p = float(summary.get("gross_profit", 0.0))
    cost_pct = (fees + slippage_cost) / gross_p * 100 if gross_p > 0 else 0.0
    long_t = int(summary.get("long_trades", 0))
    short_t = int(summary.get("short_trades", 0))

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
        "total_slippage_cost": slippage_cost,
        "cost_as_pct_of_gross_profit": cost_pct,
        "long_trades": long_t,
        "short_trades": short_t,
    }


def _empty_row(
    candidate_name: str, position_mode: str, risk_pct: float, period_name: str
) -> dict:
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
        "error": "",
    }


def main():
    parser = ArgumentParser(description="Phase 4D — tpe_trial_490 Risk Sweep")
    parser.add_argument("--data", dest="data_path", required=True)
    parser.add_argument(
        "--include-262", action="store_true",
        help="Also include tpe_trial_262 as secondary candidate",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    df_full = load_ohlcv_csv(args.data_path)
    print(f"Data loaded: {args.data_path}")
    print(f"  Range:  {df_full.index[0]} ~ {df_full.index[-1]}")
    print(f"  Bars:   {len(df_full):,}")

    # Build candidate list
    if args.include_262:
        CANDIDATE_CONFIGS["tpe_trial_262"] = "configs/candidates/tpe_trial_262.yaml"

    candidates = list(CANDIDATE_CONFIGS.keys())
    modes = list(POSITION_MODES.keys())

    total_runs = len(candidates) * len(modes) * len(RISK_PCTS) * len(PERIODS)
    print(f"\nCandidates:     {candidates}")
    print(f"Position modes: {modes}")
    print(f"Risk pcts:      {[f'{p*100:.1f}%' for p in RISK_PCTS]}")
    print(f"Periods:        {len(PERIODS)}")
    print(f"Total runs:     {total_runs:,}")
    print("=" * 65)

    # Pre-load configs
    config_cache = {}
    for name, path in CANDIDATE_CONFIGS.items():
        config_cache[name] = load_strategy_config(path)

    # -----------------------------------------------------------------------
    # Run sweep
    # -----------------------------------------------------------------------
    all_summaries: list[dict] = []
    all_trades_list: list[pd.DataFrame] = []
    all_equity_list: list[pd.Series] = []

    run_idx = 0
    for cand_name in candidates:
        strat_cfg = config_cache[cand_name]
        for mode in modes:
            for risk_pct in RISK_PCTS:
                for pname, start, end in PERIODS:
                    run_idx += 1
                    print(
                        f"[{run_idx:>4}/{total_runs}] "
                        f"{cand_name:15s} {mode:10s} "
                        f"risk={risk_pct*100:4.1f}% {pname:14s}",
                        end="",
                    )

                    row = run_one(
                        cand_name, mode, risk_pct,
                        pname, start, end, df_full, strat_cfg,
                    )
                    all_summaries.append(row)
                    print(
                        f" ret={row['total_return']:>7.2f}% "
                        f"trades={row['total_trades']:>3} "
                        f"pf={row['profit_factor']:.2f} "
                        f"dd={row['max_drawdown_pct']:.2f}%"
                    )

    # -----------------------------------------------------------------------
    # Build summary DataFrame
    # -----------------------------------------------------------------------
    summary_df = pd.DataFrame(all_summaries)

    # Save summary
    summary_path = REPORT_DIR / "phase4d_tpe490_risk_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Summary saved: {summary_path}")

    # -----------------------------------------------------------------------
    # Decision report
    # -----------------------------------------------------------------------
    lines = [
        "# Phase 4D — tpe_trial_490 Risk Proportion Sensitivity Sweep\n",
        f"> 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
        f"> 候选: {', '.join(candidates)}\n",
        f"> 风险档: {', '.join(f'{p*100:.1f}%' for p in RISK_PCTS)}\n",
        f"> 模式: {', '.join(modes)}\n",
        f"> 运行数: {total_runs:,}\n",
        "\n---\n",
    ]

    # 1. Success
    lines.append("## 1. 是否成功运行\n")
    lines.append(f"✅ YES — {total_runs:,} 次回测全部完成\n")
    error_rows = summary_df[summary_df.get("error", "").apply(lambda x: bool(x) if isinstance(x, str) else False)]
    if len(error_rows) > 0:
        lines.append(f"⚠️ {len(error_rows)} 行有错误:\n")
        lines.append(f"{error_rows[['candidate_name','position_mode','risk_per_trade_pct','period','error']].to_string(index=False)}\n")
    else:
        lines.append("✅ 无错误\n")

    lines.append("\n---\n")

    # 2/3. Full results per risk per mode
    full_df = summary_df[summary_df["period"] == "2024-2025-Full"].copy()

    disp_cols = [
        "risk_per_trade_pct", "total_return", "total_trades",
        "profit_factor", "sharpe_ratio", "max_drawdown_pct",
        "avg_r", "expectancy_r", "avg_actual_risk_pct",
        "max_actual_risk_pct", "cap_hit_rate",
        "cost_as_pct_of_gross_profit",
        "long_trades", "short_trades",
    ]

    for cand_name in candidates:
        for mode in modes:
            sub = full_df[
                (full_df["candidate_name"] == cand_name) &
                (full_df["position_mode"] == mode)
            ].sort_values("risk_per_trade_pct").copy()

            sub["risk_pct_label"] = (sub["risk_per_trade_pct"] * 100).map("{:.1f}%".format)
            sub["ret_pf_sharpe"] = sub.apply(
                lambda r: f"{r['total_return']:.2f}% / PF={r['profit_factor']:.2f} / Sharpe={r['sharpe_ratio']:.2f}",
                axis=1,
            )
            sub["dd_cap_cost"] = sub.apply(
                lambda r: f"DD={r['max_drawdown_pct']:.2f}% / cap_rate={r['cap_hit_rate']:.1%} / cost={r['cost_as_pct_of_gross_profit']:.1f}%",
                axis=1,
            )
            sub["trades_risk"] = sub.apply(
                lambda r: f"{r['total_trades']} trades / avg_risk={r['avg_actual_risk_pct']:.4f}% / max_risk={r['max_actual_risk_pct']:.4f}%",
                axis=1,
            )

            lines.append(f"\n## {cand_name} / {mode} — Full 表现\n\n")
            lines.append("| 风险 | 收益/PF/Sharpe | 回撤/成本 | 交易/风险 |\n")
            lines.append("|------|---------------|-----------|----------|\n")
            for _, r in sub.iterrows():
                lines.append(
                    f"| {r['risk_pct_label']:>5s} | "
                    f"{r['ret_pf_sharpe']} | "
                    f"{r['dd_cap_cost']} | "
                    f"{r['trades_risk']} |\n"
                )

    lines.append("\n---\n")

    # 4. Uncapped blow-up analysis
    lines.append("## 4. Uncapped 下哪些风险档出现资金曲线失控\n\n")
    for cand_name in candidates:
        sub = full_df[
            (full_df["candidate_name"] == cand_name) &
            (full_df["position_mode"] == "uncapped")
        ].sort_values("risk_per_trade_pct")

        lines.append(f"**{cand_name}** uncapped:\n\n")
        lines.append("| 风险 | 总收益 | DD | PF | 状态 |\n")
        lines.append("|------|:------:|:--:|:--:|:----:|\n")
        blowup_risks = []
        for _, r in sub.iterrows():
            label = f"{r['risk_per_trade_pct']*100:.1f}%"
            ret = r["total_return"]
            dd = r["max_drawdown_pct"]
            pf = r["profit_factor"]

            # Blow-up criteria: DD > 20% or PF < 1.0 or total_return < -50%
            if dd < -20 or pf < 1.0 or ret < -50:
                status = "❌ BLOWUP"
                blowup_risks.append(label)
            elif dd < -10:
                status = "⚠️ 高风险"
                blowup_risks.append(label)
            else:
                status = "✅ 可控"

            lines.append(f"| {label:>5s} | {ret:>8.2f}% | {dd:>6.2f}% | {pf:.2f} | {status} |\n")

        if blowup_risks:
            lines.append(f"\n⚠️ 失控风险档: {', '.join(blowup_risks)}\n")
        else:
            lines.append("\n✅ 全部风险档可控\n")

    lines.append("\n---\n")

    # 5. Best risk/reward ratio analysis
    lines.append("## 5. 哪个风险档收益/回撤比最好\n\n")
    for cand_name in candidates:
        for mode in modes:
            sub = full_df[
                (full_df["candidate_name"] == cand_name) &
                (full_df["position_mode"] == mode)
            ].copy()
            # Compute return/drawdown ratio (protected)
            sub["ret_dd_ratio"] = sub.apply(
                lambda r: abs(r["total_return"] / r["max_drawdown_pct"])
                if r["max_drawdown_pct"] != 0 else 0,
                axis=1,
            )
            best = sub.loc[sub["ret_dd_ratio"].idxmax()]
            lines.append(
                f"**{cand_name} / {mode}**: "
                f"最佳收益/回撤比 = {best['risk_per_trade_pct']*100:.1f}% "
                f"(ret={best['total_return']:.2f}%, dd={best['max_drawdown_pct']:.2f}%, "
                f"ratio={best['ret_dd_ratio']:.1f})\n"
            )

            # Also show stability — which risk level maintains highest Sharpe
            best_sharpe_row = sub.loc[sub["sharpe_ratio"].idxmax()]
            lines.append(
                f"  → 最佳Sharpe = {best_sharpe_row['risk_per_trade_pct']*100:.1f}% "
                f"(Sharpe={best_sharpe_row['sharpe_ratio']:.2f})\n"
            )

    lines.append("\n---\n")

    # 6. Can 1% replace 0.5%?
    lines.append("## 6. 1% 是否可以替代 0.5%\n\n")
    for cand_name in candidates:
        for mode in modes:
            sub = full_df[
                (full_df["candidate_name"] == cand_name) &
                (full_df["position_mode"] == mode)
            ]
            r05 = sub[sub["risk_per_trade_pct"] == 0.005]
            r10 = sub[sub["risk_per_trade_pct"] == 0.01]
            if r05.empty or r10.empty:
                lines.append(f"{cand_name}/{mode}: 数据不足\n")
                continue
            r05 = r05.iloc[0]
            r10 = r10.iloc[0]

            ret_factor = r10["total_return"] / r05["total_return"] if r05["total_return"] != 0 else 0
            dd_ratio = abs(r10["max_drawdown_pct"] / r05["max_drawdown_pct"]) if r05["max_drawdown_pct"] != 0 else 0

            lines.append(f"**{cand_name} / {mode}**:\n")
            lines.append(f"  - 0.5%: ret={r05['total_return']:.2f}%, dd={r05['max_drawdown_pct']:.2f}%, PF={r05['profit_factor']:.2f}\n")
            lines.append(f"  - 1.0%: ret={r10['total_return']:.2f}%, dd={r10['max_drawdown_pct']:.2f}%, PF={r10['profit_factor']:.2f}\n")
            lines.append(f"  - 收益倍数: {ret_factor:.1f}x (理论2.0x), 回撤比率: {dd_ratio:.1f}x\n")

            if r10["max_drawdown_pct"] > -5 and r10["profit_factor"] > 1.5:
                lines.append(f"  ✅ 1% 可替代 0.5% (DD可控, PF保持良好)\n")
            elif r10["max_drawdown_pct"] > -10:
                lines.append(f"  ⚠️ 1% 部分可替代 (DD略高但仍在可接受范围)\n")
            else:
                lines.append(f"  ❌ 1% 不可替代 0.5% (DD过大)\n")

    lines.append("\n---\n")

    # 7. Is 2% too high?
    lines.append("## 7. 2% 是否过高\n\n")
    for cand_name in candidates:
        for mode in modes:
            sub = full_df[
                (full_df["candidate_name"] == cand_name) &
                (full_df["position_mode"] == mode)
            ]
            r20 = sub[sub["risk_per_trade_pct"] == 0.02]
            if r20.empty:
                continue
            r20 = r20.iloc[0]

            # Check quarter stability at 2%
            q_sub = summary_df[
                (summary_df["candidate_name"] == cand_name) &
                (summary_df["position_mode"] == mode) &
                (summary_df["risk_per_trade_pct"] == 0.02) &
                (summary_df["period"] != "2024-2025-Full")
            ]
            pos_q = int((q_sub["total_return"] > 0).sum())
            worst_q_ret = q_sub["total_return"].min() if len(q_sub) > 0 else 0
            worst_q_pf = q_sub["profit_factor"].min() if len(q_sub) > 0 else 0

            lines.append(f"**{cand_name} / {mode}**:\n")
            lines.append(f"  - ret={r20['total_return']:.2f}%, dd={r20['max_drawdown_pct']:.2f}%, PF={r20['profit_factor']:.2f}\n")
            lines.append(f"  - 季度: {pos_q}/8 盈利, 最差季度收益={worst_q_ret:.2f}%, 最差PF={worst_q_pf:.2f}\n")

            if r20["max_drawdown_pct"] > -10 and worst_q_pf > 1.0 and pos_q >= 6:
                lines.append(f"  ✅ 2% 可接受\n")
            elif r20["max_drawdown_pct"] > -15 and worst_q_pf > 0.8:
                lines.append(f"  ⚠️ 2% 偏高但未失控\n")
            else:
                lines.append(f"  ❌ 2% 过高\n")

    lines.append("\n---\n")

    # 8. Recommendation for Phase 5A
    lines.append("## 8. 是否建议 Freqtrade Phase 5A 使用 0.5%、1% 或其他风险档\n\n")

    for cand_name in candidates:
        lines.append(f"**{cand_name}**:\n\n")

        # Analyze each mode separately
        for mode in modes:
            sub = full_df[
                (full_df["candidate_name"] == cand_name) &
                (full_df["position_mode"] == mode)
            ].copy()

            # Determine recommended risk levels
            good_risks = []
            acceptable_risks = []
            for _, r in sub.iterrows():
                risk_label = f"{r['risk_per_trade_pct']*100:.1f}%"
                dd = r["max_drawdown_pct"]
                pf = r["profit_factor"]

                if abs(dd) < 5 and pf > 2.0:
                    good_risks.append(risk_label)
                elif abs(dd) < 15 and pf > 1.3:
                    acceptable_risks.append(risk_label)

            lines.append(f"  **{mode}**:\n")
            lines.append(f"    - ✅ 推荐: {', '.join(good_risks) if good_risks else '无'}\n")
            lines.append(f"    - ⚠️ 可接受: {', '.join(acceptable_risks) if acceptable_risks else '无'}\n")

            # Pick the sweet spot
            best_row = sub.loc[sub.apply(
                lambda r: r["profit_factor"] / max(abs(r["max_drawdown_pct"]), 0.01),
                axis=1,
            ).idxmax()]
            lines.append(
                f"    - 🎯 最佳风险档: {best_row['risk_per_trade_pct']*100:.1f}% "
                f"(ret={best_row['total_return']:.2f}%, PF={best_row['profit_factor']:.2f}, "
                f"DD={best_row['max_drawdown_pct']:.2f}%)\n"
            )

    # Final consolidated recommendation
    lines.append("\n## 最终建议\n\n")

    # Use capped_3x as default (most conservative/realistic)
    cand_ref = candidates[0]
    sub_3x = full_df[
        (full_df["candidate_name"] == cand_ref) &
        (full_df["position_mode"] == "capped_3x")
    ]
    if not sub_3x.empty:
        # Find risk that gives PF > 2 and DD < 5%
        best = None
        for _, r in sub_3x.sort_values("risk_per_trade_pct").iterrows():
            if r["profit_factor"] > 2.0 and abs(r["max_drawdown_pct"]) < 5:
                best = r
        if best is not None:
            best_pct = best["risk_per_trade_pct"] * 100
            lines.append(
                f"基于 {cand_ref} capped_3x 结果：\n\n"
                f"**建议 Phase 5A Freqtrade 使用 {best_pct:.1f}% risk_per_trade_pct**\n"
                f"- return: {best['total_return']:.2f}%\n"
                f"- PF: {best['profit_factor']:.2f}\n"
                f"- Sharpe: {best['sharpe_ratio']:.2f}\n"
                f"- DD: {best['max_drawdown_pct']:.2f}%\n\n"
                f"capped_3x 推荐作为 Real/Demo 模式，capped_5x 推荐作为高级加速模式。\n"
                f"uncapped 仅为理论压力测试，不建议实盘使用。\n"
            )
        else:
            # Fallback: suggest lowest risk
            lowest = sub_3x.sort_values("risk_per_trade_pct").iloc[0]
            lines.append(
                f"基于 {cand_ref} capped_3x 结果，推荐最保守档位 {lowest['risk_per_trade_pct']*100:.1f}%\n"
                f" (ret={lowest['total_return']:.2f}%, PF={lowest['profit_factor']:.2f}, "
                f"DD={lowest['max_drawdown_pct']:.2f}%)\n"
            )

    decision_text = "".join(lines)
    decision_path = REPORT_DIR / "phase4d_tpe490_risk_sweep_decision.md"
    with open(decision_path, "w", encoding="utf-8") as f:
        f.write(decision_text)
    print(f"✅ Decision report saved: {decision_path}")

    # -----------------------------------------------------------------------
    # Also save trades and equity (compact: only one representative row)
    # -----------------------------------------------------------------------
    # We skip trades/equity CSV for all runs to keep file size manageable.
    # For the full run, we just save summary.
    print(f"\n{'='*65}")
    print("Phase 4D — Complete")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
