#!/usr/bin/env python3
"""
TPE Top Candidate Validation

Validates TPE top parameters against current_baseline and pivot_wide_9_35
across 2024-Q1~Q4 and 2025-Q1~Q4, plus Full period.
Each period independently builds signals (no full-build-then-slice).

Usage:
    python optimize/tpe_candidate_validation.py \
        --data data/raw/BTCUSDT_5m_2024_2025.csv \
        --top50 reports/tpe_top50.csv
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import copy
import math

import numpy as np
import pandas as pd

from optimize.utils import safe_profit_factor, slice_dataframe, compute_extra_metrics
from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.event_engine import run_backtest

# ------------------------------------------------------------------
# Periods
# ------------------------------------------------------------------
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

BT_CFG_COST = {
    "initial_cash": 100000.0,
    "fee_per_trade": 5.0,
    "slippage": 5.0,
    "allow_short": True,
}

BT_CFG_NOCOST = {
    "initial_cash": 100000.0,
    "fee_per_trade": 0.0,
    "slippage": 0.0,
    "allow_short": True,
}


def _safe_pf(val) -> float:
    if val is None or pd.isna(val):
        return 0.0
    if isinstance(val, float) and math.isinf(val):
        return 5.0
    v = float(val)
    if v > 5.0:
        v = 5.0
    elif v < 0.0:
        v = 0.0
    return v


def compute_expectancy_r(trades: pd.DataFrame) -> float:
    """Compute average R-multiple from trades."""
    if trades.empty:
        return 0.0
    r_list = []
    for _, row in trades.iterrows():
        if row["side"] == "long":
            risk = row["entry_price"] - row["stop_price"]
        else:
            risk = row["stop_price"] - row["entry_price"]
        if risk <= 0 or pd.isna(risk):
            continue
        r_list.append(row["pnl"] / risk)
    if not r_list:
        return 0.0
    return float(np.mean(r_list))


def build_strategy_config(params: dict) -> dict:
    """Build strategy config dict from flat param dict."""
    return {
        "stochastic": {
            "k_period": int(params["k_period"]),
            "d_period": int(params["d_period"]),
            "smooth": int(params["smooth"]),
            "oversold": int(params["oversold"]),
            "overbought": int(params["overbought"]),
        },
        "pivots": {
            "left_bars": int(params["left_bars"]),
            "right_bars": int(params["right_bars"]),
            "min_separation": int(params["min_separation"]),
            "max_separation": int(params["max_separation"]),
            "strict": True,
        },
        "trend": {
            "ema_period": int(params["ema_period"]),
            "lookback_bars": int(params.get("lookback_bars", 24)),
            "min_close_ratio": float(params["min_close_ratio"]),
        },
        "risk": {
            "atr_period": 14,
            "stop_buffer": float(params.get("stop_buffer", 0.0)),
            "rr_target": float(params["rr_target"]),
        },
        "setup": {
            "setup_max_bars": int(params.get("setup_max_bars", 12)),
            "replace_same_side_setup": True,
            "invalidate_on_stop_anchor_break": True,
        },
    }


def load_candidates(top50_path: str):
    """Build candidate list from fixed params + top50 CSV."""
    candidates = []

    # 1. current_baseline
    candidates.append({
        "candidate_name": "current_baseline",
        "source_trial": "fixed",
        "k_period": 14,
        "d_period": 3,
        "smooth": 1,
        "oversold": 15,
        "overbought": 85,
        "left_bars": 4,
        "right_bars": 3,
        "min_separation": 3,
        "max_separation": 20,
        "ema_period": 55,
        "lookback_bars": 24,
        "min_close_ratio": 0.80,
        "rr_target": 2.2,
        "stop_buffer": 0.0,
        "setup_max_bars": 12,
    })

    # 2. pivot_wide_9_35
    candidates.append({
        "candidate_name": "pivot_wide_9_35",
        "source_trial": "fixed",
        "k_period": 14,
        "d_period": 3,
        "smooth": 1,
        "oversold": 15,
        "overbought": 85,
        "left_bars": 4,
        "right_bars": 3,
        "min_separation": 9,
        "max_separation": 35,
        "ema_period": 55,
        "lookback_bars": 24,
        "min_close_ratio": 0.80,
        "rr_target": 2.2,
        "stop_buffer": 0.0,
        "setup_max_bars": 12,
    })

    # 3. tpe_trial_9
    candidates.append({
        "candidate_name": "tpe_trial_9",
        "source_trial": "9",
        "k_period": 11,
        "d_period": 2,
        "smooth": 1,
        "oversold": 12,
        "overbought": 88,
        "left_bars": 6,
        "right_bars": 4,
        "min_separation": 14,
        "max_separation": 49,
        "ema_period": 33,
        "lookback_bars": 24,
        "min_close_ratio": 0.90,
        "rr_target": 2.58,
        "stop_buffer": 0.0,
        "setup_max_bars": 12,
    })

    # Load top50 CSV
    top50_df = pd.read_csv(top50_path)

    # Helper to add trial by ID
    def _add_trial(trial_id: int):
        row = top50_df[top50_df["trial_id"] == trial_id]
        if row.empty:
            print(f"Warning: trial {trial_id} not found in {top50_path}")
            return
        r = row.iloc[0]
        candidates.append({
            "candidate_name": f"tpe_trial_{trial_id}",
            "source_trial": str(trial_id),
            "k_period": int(r["k_period"]),
            "d_period": int(r["d_period"]),
            "smooth": int(r["smooth"]),
            "oversold": int(r["oversold"]),
            "overbought": int(r["overbought"]),
            "left_bars": int(r["left_bars"]),
            "right_bars": int(r["right_bars"]),
            "min_separation": int(r["min_separation"]),
            "max_separation": int(r["max_separation"]),
            "ema_period": int(r["ema_period"]),
            "lookback_bars": 24,
            "min_close_ratio": float(r["min_close_ratio"]),
            "rr_target": float(r["rr_target"]),
            "stop_buffer": 0.0,
            "setup_max_bars": 12,
        })

    # 4. Specified trials
    for tid in (421, 251, 199, 302):
        _add_trial(tid)

    # 5. Filter additional from top50
    # Exclude already-added trials (9, 421, 251, 199, 302)
    added_ids = {9, 421, 251, 199, 302}
    filtered = top50_df[
        (~top50_df["trial_id"].isin(added_ids)) &
        (top50_df["total_trades"] >= 100) &
        (top50_df["profit_factor"] >= 2.0) &
        (top50_df["quarterly_positive_ratio"] >= 0.75) &
        (top50_df["expectancy_R"] > 0)
    ].copy()
    # Sort by original score descending, take top 8
    filtered = filtered.sort_values("score", ascending=False).head(8)
    for _, r in filtered.iterrows():
        tid = int(r["trial_id"])
        _add_trial(tid)

    return candidates, top50_df


def run_single_backtest(df_slice: pd.DataFrame, cfg: dict, bt_cfg: dict):
    """Run build_signals + backtest on a sliced dataframe."""
    if len(df_slice) < 100:
        return None
    bundle = build_signals(df_slice, cfg)
    result = run_backtest(df_slice, bundle, bt_cfg)
    return result


def run_validation(df: pd.DataFrame, candidates: list) -> pd.DataFrame:
    """Run all candidates across all periods and both cost modes."""
    total_runs = len(candidates) * len(PERIODS) * 2
    print("=" * 60)
    print("TPE Candidate Validation")
    print("=" * 60)
    print(f"Candidates: {len(candidates)}")
    print(f"Periods:    {len(PERIODS)}")
    print(f"Cost modes: 2 (no-cost, cost)")
    print(f"Total runs: {total_runs}")
    print("=" * 60)

    rows = []
    run_idx = 0

    for cand in candidates:
        cfg = build_strategy_config(cand)
        for cost_mode, bt_cfg in (("no-cost", BT_CFG_NOCOST), ("cost", BT_CFG_COST)):
            for period_name, start_str, end_str in PERIODS:
                run_idx += 1
                df_slice = slice_dataframe(df, start_str, end_str)

                if len(df_slice) < 100:
                    row = {
                        "candidate_name": cand["candidate_name"],
                        "source_trial": cand["source_trial"],
                        "cost_mode": cost_mode,
                        "period": period_name,
                        "start_time": start_str,
                        "end_time": end_str,
                        "total_bars": len(df_slice),
                        "total_return": 0.0,
                        "total_trades": 0,
                        "win_rate": 0.0,
                        "profit_factor": 0.0,
                        "max_drawdown": 0.0,
                        "avg_trade": 0.0,
                        "expectancy": 0.0,
                        "expectancy_r": 0.0,
                        "long_trades": 0,
                        "short_trades": 0,
                    }
                    rows.append(row)
                    print(f"[{run_idx:>4}/{total_runs}] {cand['candidate_name']:20s} {cost_mode:6s} {period_name:14s} -> SKIPPED (bars={len(df_slice)})")
                    continue

                result = run_single_backtest(df_slice, cfg, bt_cfg)
                if result is None:
                    row = {
                        "candidate_name": cand["candidate_name"],
                        "source_trial": cand["source_trial"],
                        "cost_mode": cost_mode,
                        "period": period_name,
                        "start_time": start_str,
                        "end_time": end_str,
                        "total_bars": len(df_slice),
                        "total_return": 0.0,
                        "total_trades": 0,
                        "win_rate": 0.0,
                        "profit_factor": 0.0,
                        "max_drawdown": 0.0,
                        "avg_trade": 0.0,
                        "expectancy": 0.0,
                        "expectancy_r": 0.0,
                        "long_trades": 0,
                        "short_trades": 0,
                    }
                    rows.append(row)
                    print(f"[{run_idx:>4}/{total_runs}] {cand['candidate_name']:20s} {cost_mode:6s} {period_name:14s} -> SKIPPED (bars={len(df_slice)})")
                    continue

                summary = result.summary
                extras = compute_extra_metrics(result)
                exp_r = compute_expectancy_r(result.trades)

                row = {
                    "candidate_name": cand["candidate_name"],
                    "source_trial": cand["source_trial"],
                    "cost_mode": cost_mode,
                    "period": period_name,
                    "start_time": start_str,
                    "end_time": end_str,
                    "total_bars": len(df_slice),
                    "total_return": summary.get("total_return", 0.0),
                    "total_trades": summary.get("total_trades", 0),
                    "win_rate": summary.get("win_rate", 0.0),
                    "profit_factor": summary.get("profit_factor", 0.0),
                    "max_drawdown": summary.get("max_drawdown", 0.0),
                    "avg_trade": summary.get("avg_trade", 0.0),
                    "expectancy": summary.get("expectancy", 0.0),
                    "expectancy_r": exp_r,
                    "long_trades": extras.get("long_trades", 0),
                    "short_trades": extras.get("short_trades", 0),
                }
                rows.append(row)
                pf_disp = safe_profit_factor(row["profit_factor"])
                print(f"[{run_idx:>4}/{total_runs}] {cand['candidate_name']:20s} {cost_mode:6s} {period_name:14s} -> "
                      f"return={row['total_return']:>6.2f} trades={row['total_trades']:>3} pf={pf_disp:.2f}")

    return pd.DataFrame(rows)


def build_summary(validation_df: pd.DataFrame, candidates: list) -> pd.DataFrame:
    """Build summary with scoring and risk flags."""
    summary_rows = []

    for cand in candidates:
        cname = cand["candidate_name"]
        for cost_mode in ("no-cost", "cost"):
            sub = validation_df[
                (validation_df["candidate_name"] == cname) &
                (validation_df["cost_mode"] == cost_mode)
            ]
            if sub.empty:
                continue

            full = sub[sub["period"] == "2024-2025-Full"]
            if full.empty:
                continue
            full = full.iloc[0]

            quarters = sub[sub["period"].str.match(r"^\d{4}-Q\d$")]
            if quarters.empty:
                continue

            positive_periods = int((quarters["total_return"] > 0).sum())
            positive_ratio = positive_periods / len(quarters) if len(quarters) > 0 else 0.0

            full_total_return = float(full["total_return"])
            full_total_trades = int(full["total_trades"])
            full_win_rate = float(full["win_rate"])
            full_profit_factor = float(full["profit_factor"]) if pd.notna(full["profit_factor"]) else 0.0
            full_max_drawdown = float(full["max_drawdown"])

            avg_quarter_return = float(quarters["total_return"].mean())
            min_quarter_return = float(quarters["total_return"].min())

            total_trades_sum = int(quarters["total_trades"].sum())
            min_trades_per_period = int(quarters["total_trades"].min())
            avg_trades_per_period = float(quarters["total_trades"].mean())

            q_pf = quarters["profit_factor"].apply(_safe_pf)
            avg_profit_factor = float(q_pf.mean())
            min_profit_factor = float(q_pf.min())
            max_quarter_drawdown_worst = float(quarters["max_drawdown"].min())

            long_trades_total = int(full["long_trades"])
            short_trades_total = int(full["short_trades"])

            # Scoring
            if min_trades_per_period < 8:
                validation_score = -9999.0
            else:
                def _clip(x, lo, hi):
                    return max(lo, min(hi, x))
                score = (
                    0.25 * min(avg_profit_factor, 5.0) / 5.0
                    + 0.20 * positive_ratio
                    + 0.20 * min(min_profit_factor, 5.0) / 5.0
                    + 0.15 * _clip(avg_quarter_return / 5.0, -1.0, 1.0)
                    + 0.10 * min(min_trades_per_period / 20.0, 1.0)
                    - 0.10 * abs(max_quarter_drawdown_worst) / 20.0
                )
                validation_score = score

            summary_rows.append({
                "candidate_name": cname,
                "source_trial": cand["source_trial"],
                "cost_mode": cost_mode,
                "ema_period": cand["ema_period"],
                "lookback_bars": cand["lookback_bars"],
                "min_close_ratio": cand["min_close_ratio"],
                "k_period": cand["k_period"],
                "smooth": cand["smooth"],
                "d_period": cand["d_period"],
                "oversold": cand["oversold"],
                "overbought": cand["overbought"],
                "left_bars": cand["left_bars"],
                "right_bars": cand["right_bars"],
                "min_separation": cand["min_separation"],
                "max_separation": cand["max_separation"],
                "rr_target": cand["rr_target"],
                "stop_buffer": cand["stop_buffer"],
                "setup_max_bars": cand["setup_max_bars"],
                "positive_periods": positive_periods,
                "positive_ratio": positive_ratio,
                "full_total_return": full_total_return,
                "full_total_trades": full_total_trades,
                "full_win_rate": full_win_rate,
                "full_profit_factor": full_profit_factor,
                "full_max_drawdown": full_max_drawdown,
                "avg_quarter_return": avg_quarter_return,
                "min_quarter_return": min_quarter_return,
                "total_trades_sum": total_trades_sum,
                "min_trades_per_period": min_trades_per_period,
                "avg_trades_per_period": avg_trades_per_period,
                "avg_profit_factor": avg_profit_factor,
                "min_profit_factor": min_profit_factor,
                "max_quarter_drawdown_worst": max_quarter_drawdown_worst,
                "long_trades_total": long_trades_total,
                "short_trades_total": short_trades_total,
                "validation_score": validation_score,
                "risk_flag": "",  # filled later
            })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def assign_risk_flags(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Assign risk flags per candidate per cost_mode."""
    # We need to compare cost vs no-cost rankings for COST_FRAGILE
    cost_df = summary_df[summary_df["cost_mode"] == "cost"].copy()
    nocost_df = summary_df[summary_df["cost_mode"] == "no-cost"].copy()

    # Build ranking maps
    cost_rank = {}
    nocost_rank = {}
    if not cost_df.empty:
        cost_sorted = cost_df.sort_values("validation_score", ascending=False).reset_index(drop=True)
        for i, row in cost_sorted.iterrows():
            cost_rank[row["candidate_name"]] = i + 1
    if not nocost_df.empty:
        nocost_sorted = nocost_df.sort_values("validation_score", ascending=False).reset_index(drop=True)
        for i, row in nocost_sorted.iterrows():
            nocost_rank[row["candidate_name"]] = i + 1

    flags_list = []
    for _, row in summary_df.iterrows():
        flags = []
        cname = row["candidate_name"]

        # LOW_TRADE_COUNT
        if row["full_total_trades"] < 80 or row["min_trades_per_period"] < 8:
            flags.append("LOW_TRADE_COUNT")

        # LOW_FREQ_HIGH_SCORE
        top3 = False
        mode_df = summary_df[summary_df["cost_mode"] == row["cost_mode"]]
        if not mode_df.empty:
            top3_names = mode_df.sort_values("validation_score", ascending=False).head(3)["candidate_name"].tolist()
            if cname in top3_names and row["full_total_trades"] < 80:
                top3 = True
        if top3:
            flags.append("LOW_FREQ_HIGH_SCORE")

        # COST_FRAGILE
        if row["cost_mode"] == "cost":
            if row["full_profit_factor"] < 1.3:
                flags.append("COST_FRAGILE")
            elif cname in cost_rank and cname in nocost_rank:
                # If cost rank is >=3 worse than no-cost rank
                if cost_rank[cname] - nocost_rank[cname] >= 3:
                    flags.append("COST_FRAGILE")

        # QUARTER_UNSTABLE
        if row["positive_ratio"] < 0.75 or row["min_profit_factor"] < 1.0:
            flags.append("QUARTER_UNSTABLE")

        # SIDE_IMBALANCE
        total_side = row["long_trades_total"] + row["short_trades_total"]
        if total_side > 0:
            long_pct = row["long_trades_total"] / total_side
            short_pct = row["short_trades_total"] / total_side
            if long_pct > 0.75 or short_pct > 0.75:
                flags.append("SIDE_IMBALANCE")

        flags_list.append("; ".join(flags) if flags else "")

    summary_df = summary_df.copy()
    summary_df["risk_flag"] = flags_list
    return summary_df


def build_top_tables(summary_df: pd.DataFrame, validation_df: pd.DataFrame) -> tuple:
    """Build top_cost and top_nocost tables."""
    # Top cost
    cost_summary = summary_df[summary_df["cost_mode"] == "cost"].sort_values("validation_score", ascending=False)
    top_cost = cost_summary.head(10).copy()

    # Top no-cost
    nocost_summary = summary_df[summary_df["cost_mode"] == "no-cost"].sort_values("validation_score", ascending=False)
    top_nocost = nocost_summary.head(10).copy()

    return top_cost, top_nocost


def build_risk_flags_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build risk flags table with all flagged candidates."""
    flagged = summary_df[summary_df["risk_flag"] != ""].copy()
    if flagged.empty:
        # Return empty frame with right columns
        return summary_df.head(0)[[
            "candidate_name", "source_trial", "cost_mode",
            "full_total_trades", "min_trades_per_period",
            "positive_ratio", "min_profit_factor",
            "validation_score", "risk_flag",
        ]].copy()
    return flagged[[
        "candidate_name", "source_trial", "cost_mode",
        "full_total_trades", "min_trades_per_period",
        "positive_ratio", "min_profit_factor",
        "validation_score", "risk_flag",
    ]].copy()


def main():
    parser = argparse.ArgumentParser(description="TPE Top Candidate Validation")
    parser.add_argument("--data", default=str(_PROJECT_ROOT / "data" / "raw" / "BTCUSDT_5m_2024_2025.csv"),
                        help="Path to OHLCV CSV")
    parser.add_argument("--top50", default=str(_PROJECT_ROOT / "reports" / "tpe_top50.csv"),
                        help="Path to tpe_top50.csv")
    args = parser.parse_args()

    print("Loading data...")
    df = load_ohlcv_csv(args.data)
    print(f"Loaded {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    print("Loading candidates from top50...")
    candidates, top50_df = load_candidates(args.top50)
    print(f"Total candidates to validate: {len(candidates)}")
    for c in candidates:
        print(f"  - {c['candidate_name']} (trial {c['source_trial']})")

    # Run validation
    validation_df = run_validation(df, candidates)

    # Build summary
    summary_df = build_summary(validation_df, candidates)
    summary_df = assign_risk_flags(summary_df)

    # Build top tables
    top_cost, top_nocost = build_top_tables(summary_df, validation_df)
    risk_flags_df = build_risk_flags_table(summary_df)

    # Save outputs
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)

    validation_path = out_dir / "tpe_candidate_validation.csv"
    summary_path = out_dir / "tpe_candidate_summary.csv"
    top_cost_path = out_dir / "tpe_candidate_top_cost.csv"
    top_nocost_path = out_dir / "tpe_candidate_top_nocost.csv"
    risk_flags_path = out_dir / "tpe_candidate_risk_flags.csv"

    validation_df.to_csv(validation_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    top_cost.to_csv(top_cost_path, index=False)
    top_nocost.to_csv(top_nocost_path, index=False)
    risk_flags_df.to_csv(risk_flags_path, index=False)

    # Terminal report (15 items)
    print("\n" + "=" * 60)
    print("TPE Candidate Validation Report")
    print("=" * 60)

    print(f"\n1. Data loaded: {len(df)} bars ({df.index[0]} ~ {df.index[-1]})")
    print(f"2. Total candidates: {len(candidates)}")
    for c in candidates:
        print(f"   - {c['candidate_name']} (source: {c['source_trial']})")

    print(f"3. Periods validated: {len(PERIODS)} (8 quarters + Full)")
    for p, _, _ in PERIODS:
        print(f"   - {p}")

    print(f"4. Cost modes: no-cost, cost (fee=5, slippage=5)")
    print(f"5. tpe_candidate_validation.csv saved: {len(validation_df)} rows")
    print(f"6. tpe_candidate_summary.csv saved: {len(summary_df)} rows")
    print(f"7. tpe_candidate_top_cost.csv saved: {len(top_cost)} rows")
    print(f"8. tpe_candidate_top_nocost.csv saved: {len(top_nocost)} rows")
    print(f"9. tpe_candidate_risk_flags.csv saved: {len(risk_flags_df)} rows")

    # Cost top 5
    print("\n10. Top 5 by validation_score (cost):")
    if not top_cost.empty:
        for _, r in top_cost.head(5).iterrows():
            print(f"    {r['candidate_name']:20s} score={r['validation_score']:.4f}  "
                  f"return={r['full_total_return']:>6.2f}%  trades={r['full_total_trades']:>3}  "
                  f"pf={safe_profit_factor(r['full_profit_factor']):.2f}  mdd={r['full_max_drawdown']:.2f}%  "
                  f"flag={r['risk_flag']}")
    else:
        print("    (none)")

    # No-cost top 5
    print("\n11. Top 5 by validation_score (no-cost):")
    if not top_nocost.empty:
        for _, r in top_nocost.head(5).iterrows():
            print(f"    {r['candidate_name']:20s} score={r['validation_score']:.4f}  "
                  f"return={r['full_total_return']:>6.2f}%  trades={r['full_total_trades']:>3}  "
                  f"pf={safe_profit_factor(r['full_profit_factor']):.2f}  mdd={r['full_max_drawdown']:.2f}%  "
                  f"flag={r['risk_flag']}")
    else:
        print("    (none)")

    # Compare baseline vs top TPE (cost)
    baseline_cost = summary_df[(summary_df["candidate_name"] == "current_baseline") & (summary_df["cost_mode"] == "cost")]
    pivot_cost = summary_df[(summary_df["candidate_name"] == "pivot_wide_9_35") & (summary_df["cost_mode"] == "cost")]
    print("\n12. Baseline vs pivot_wide (cost mode):")
    for label, sub in [("current_baseline", baseline_cost), ("pivot_wide_9_35", pivot_cost)]:
        if not sub.empty:
            r = sub.iloc[0]
            print(f"    {label:20s} score={r['validation_score']:.4f}  return={r['full_total_return']:>6.2f}%  "
                  f"trades={r['full_total_trades']:>3}  pf={safe_profit_factor(r['full_profit_factor']):.2f}  "
                  f"min_q_pf={r['min_profit_factor']:.2f}  positive_ratio={r['positive_ratio']:.2f}")

    # Best TPE trial (cost)
    tpe_only = summary_df[
        (summary_df["cost_mode"] == "cost") &
        (~summary_df["candidate_name"].isin(["current_baseline", "pivot_wide_9_35"]))
    ].sort_values("validation_score", ascending=False)
    print("\n13. Best TPE trial (cost mode):")
    if not tpe_only.empty:
        r = tpe_only.iloc[0]
        print(f"    {r['candidate_name']:20s} score={r['validation_score']:.4f}  return={r['full_total_return']:>6.2f}%  "
              f"trades={r['full_total_trades']:>3}  pf={safe_profit_factor(r['full_profit_factor']):.2f}  "
              f"min_q_pf={r['min_profit_factor']:.2f}  positive_ratio={r['positive_ratio']:.2f}  "
              f"flag={r['risk_flag']}")
    else:
        print("    (none)")

    # Risk flags summary
    print("\n14. Risk flags summary:")
    if not risk_flags_df.empty:
        for _, r in risk_flags_df.iterrows():
            print(f"    {r['candidate_name']:20s} {r['cost_mode']:6s}  {r['risk_flag']}")
    else:
        print("    No risk flags raised.")

    # Overall conclusion
    print("\n15. Overall conclusion:")
    if not top_cost.empty:
        best = top_cost.iloc[0]
        baseline_score = baseline_cost.iloc[0]["validation_score"] if not baseline_cost.empty else -9999
        pivot_score = pivot_cost.iloc[0]["validation_score"] if not pivot_cost.empty else -9999
        print(f"    Best candidate (cost): {best['candidate_name']} (score={best['validation_score']:.4f})")
        print(f"    Baseline score: {baseline_score:.4f}")
        print(f"    Pivot wide score: {pivot_score:.4f}")
        if best["candidate_name"] not in ("current_baseline", "pivot_wide_9_35"):
            print(f"    TPE top candidate beats baseline: {'YES' if best['validation_score'] > baseline_score else 'NO'}")
            print(f"    TPE top candidate beats pivot_wide: {'YES' if best['validation_score'] > pivot_score else 'NO'}")
        else:
            print("    Baseline/pivot_wide remains best among tested candidates.")
    else:
        print("    No cost-mode results available.")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
