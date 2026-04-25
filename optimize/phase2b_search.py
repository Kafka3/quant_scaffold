#!/usr/bin/env python3
"""
Phase 2B — EMA 周期与 Stop Buffer 稳健性验证。

固定趋势结构，验证 3 组候选参数 × 3 EMA 周期 × 3 stop_buffer
 across 5 个时间分段。

Usage:
    python optimize/phase2b_search.py \
      --config configs/phase2a_baseline.yaml \
      --data data/raw/BTCUSDT_5m_2024.csv
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from argparse import ArgumentParser
from itertools import product
import copy
import math

import pandas as pd

from configs.settings import load_settings
from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.vectorbt_engine import run_backtest


PERIODS = [
    ("2024-Q1", "2024-01-01", "2024-04-01"),
    ("2024-Q2", "2024-04-01", "2024-07-01"),
    ("2024-Q3", "2024-07-01", "2024-10-01"),
    ("2024-Q4", "2024-10-01", "2025-01-01"),
    ("2024-Full", "2024-01-01", "2025-01-01"),
]

CANDIDATES = [
    {"candidate_name": "baseline", "rr_target": 2.0, "oversold": 20, "overbought": 80},
    {"candidate_name": "candidate_A_return", "rr_target": 2.2, "oversold": 15, "overbought": 85},
    {"candidate_name": "candidate_B_stability", "rr_target": 2.0, "oversold": 15, "overbought": 85},
]

EMA_PERIODS = [34, 55, 89]
STOP_BUFFERS = [0, 10, 20]

FIXED_TREND = {
    "ema_period": 55,
    "lookback_bars": 24,
    "min_close_ratio": 0.80,
}

FIXED_PIVOTS = {
    "left_bars": 4,
    "right_bars": 3,
    "min_separation": 3,
    "max_separation": 20,
    "strict": True,
}

FIXED_SETUP = {
    "setup_max_bars": 12,
    "replace_same_side_setup": True,
    "invalidate_on_stop_anchor_break": True,
}


def _safe_pf(val):
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


def _slice_df(df: pd.DataFrame, start_str: str, end_str: str) -> pd.DataFrame:
    start = pd.Timestamp(start_str, tz="UTC")
    end = pd.Timestamp(end_str, tz="UTC")
    mask = (df.index >= start) & (df.index < end)
    return df.loc[mask].copy()


def compute_extra_metrics(result) -> dict:
    trades = result.trades
    if trades.empty:
        return {
            "long_trades": 0,
            "short_trades": 0,
            "target_exits": 0,
            "stop_exits": 0,
            "end_of_data_exits": 0,
            "avg_bars_held": 0.0,
            "median_bars_held": 0.0,
        }
    long_mask = trades["side"] == "long"
    short_mask = trades["side"] == "short"
    return {
        "long_trades": int(long_mask.sum()),
        "short_trades": int(short_mask.sum()),
        "target_exits": int((trades["exit_reason"] == "target").sum()),
        "stop_exits": int((trades["exit_reason"] == "stop").sum()),
        "end_of_data_exits": int((trades["exit_reason"] == "end_of_data").sum()),
        "avg_bars_held": float(trades["bars_held"].mean()),
        "median_bars_held": float(trades["bars_held"].median()),
    }


def run_phase2b(df: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_combos = len(CANDIDATES) * len(EMA_PERIODS) * len(STOP_BUFFERS)
    total_runs = total_combos * len(PERIODS)

    print("=" * 50)
    print("Phase 2B — EMA Period & Stop Buffer Robustness")
    print("=" * 50)
    print(f"Candidates:  {len(CANDIDATES)}")
    print(f"EMA periods: {EMA_PERIODS}")
    print(f"Stop buffers: {STOP_BUFFERS}")
    print(f"Total combos: {total_combos}")
    print(f"Periods: {len(PERIODS)}")
    print(f"Total runs: {total_runs}")
    print("=" * 50)

    validation_rows = []
    run_idx = 0

    for cand, ema_p, buf in product(CANDIDATES, EMA_PERIODS, STOP_BUFFERS):
        for period_name, start_str, end_str in PERIODS:
            run_idx += 1
            df_slice = _slice_df(df, start_str, end_str)

            if len(df_slice) < 100:
                row = {
                    "candidate_name": cand["candidate_name"],
                    "rr_target": cand["rr_target"],
                    "oversold": cand["oversold"],
                    "overbought": cand["overbought"],
                    "ema_period": ema_p,
                    "stop_buffer": buf,
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
                    "long_trades": 0,
                    "short_trades": 0,
                    "target_exits": 0,
                    "stop_exits": 0,
                    "end_of_data_exits": 0,
                    "avg_bars_held": 0.0,
                    "median_bars_held": 0.0,
                }
                validation_rows.append(row)
                print(f"[{run_idx:>4}/{total_runs}] {period_name} {cand['candidate_name']} ema={ema_p} buf={buf} -> SKIPPED")
                continue

            cfg = copy.deepcopy(settings)
            cfg["strategy"]["trend"].update(FIXED_TREND)
            cfg["strategy"]["trend"]["ema_period"] = ema_p
            cfg["strategy"]["pivots"].update(FIXED_PIVOTS)
            cfg["strategy"]["setup"].update(FIXED_SETUP)
            cfg["strategy"]["risk"]["rr_target"] = cand["rr_target"]
            cfg["strategy"]["risk"]["stop_buffer"] = buf
            cfg["strategy"]["stochastic"]["oversold"] = cand["oversold"]
            cfg["strategy"]["stochastic"]["overbought"] = cand["overbought"]

            bundle = build_signals(df_slice, cfg["strategy"])
            result = run_backtest(df_slice, bundle, cfg["backtest"])
            summary = pd.Series(result.summary)
            extras = compute_extra_metrics(result)

            row = {
                "candidate_name": cand["candidate_name"],
                "rr_target": cand["rr_target"],
                "oversold": cand["oversold"],
                "overbought": cand["overbought"],
                "ema_period": ema_p,
                "stop_buffer": buf,
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
                **extras,
            }
            validation_rows.append(row)
            print(f"[{run_idx:>4}/{total_runs}] {period_name} {cand['candidate_name']} ema={ema_p} buf={buf} -> "
                  f"return={row['total_return']:>6.2f} trades={row['total_trades']:>3} pf={_safe_pf(row['profit_factor']):.2f}")

    validation_df = pd.DataFrame(validation_rows)

    summary_rows = []
    for (cand_name, rr, os_val, ob_val, ema_p, buf), group in validation_df.groupby(
        ["candidate_name", "rr_target", "oversold", "overbought", "ema_period", "stop_buffer"]
    ):
        full_year = group[group["period"] == "2024-Full"].iloc[0]
        quarters = group[group["period"].str.startswith("2024-Q")]

        q_positive_count = int((quarters["total_return"] > 0).sum())
        q_trade_count_min = int(quarters["total_trades"].min())
        q_trade_count_mean = float(quarters["total_trades"].mean())

        q_pf_values = quarters["profit_factor"].apply(_safe_pf)
        q_pf_mean = float(q_pf_values.mean())
        q_pf_median = float(q_pf_values.median())
        q_pf_min = float(q_pf_values.min())

        q_max_drawdown_worst = float(quarters["max_drawdown"].min())

        full_year_trades = int(full_year["total_trades"])
        if full_year_trades < 80:
            stability_score = -9999.0
        else:
            full_year_pf_score = _safe_pf(full_year["profit_factor"]) / 5.0
            full_year_win_rate = float(full_year["win_rate"] or 0.0)
            q_positive_ratio = q_positive_count / 4.0
            q_pf_min_score = q_pf_min / 5.0
            return_score = max(min(full_year["total_return"] / 20.0, 1.0), -1.0)
            drawdown_penalty = abs(q_max_drawdown_worst) / 20.0

            stability_score = (
                0.25 * full_year_pf_score
                + 0.20 * full_year_win_rate
                + 0.20 * q_positive_ratio
                + 0.15 * q_pf_min_score
                + 0.10 * return_score
                - 0.10 * drawdown_penalty
            )

        summary_rows.append({
            "candidate_name": cand_name,
            "rr_target": rr,
            "oversold": os_val,
            "overbought": ob_val,
            "ema_period": ema_p,
            "stop_buffer": buf,
            "full_year_total_return": full_year["total_return"],
            "full_year_total_trades": full_year_trades,
            "full_year_win_rate": full_year["win_rate"],
            "full_year_profit_factor": full_year["profit_factor"],
            "full_year_max_drawdown": full_year["max_drawdown"],
            "q_positive_count": q_positive_count,
            "q_trade_count_min": q_trade_count_min,
            "q_trade_count_mean": q_trade_count_mean,
            "q_profit_factor_mean": q_pf_mean,
            "q_profit_factor_median": q_pf_median,
            "q_profit_factor_min": q_pf_min,
            "q_max_drawdown_worst": q_max_drawdown_worst,
            "long_trades": full_year["long_trades"],
            "short_trades": full_year["short_trades"],
            "stability_score": stability_score,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("stability_score", ascending=False)
    return validation_df, summary_df


def main() -> None:
    parser = ArgumentParser(description="Phase 2B — EMA Period & Stop Buffer Robustness")
    parser.add_argument("--config", dest="config_path", default="configs/phase2a_baseline.yaml",
                        help="Path to config YAML file (default: configs/phase2a_baseline.yaml)")
    parser.add_argument("--data", dest="data_path", default=None,
                        help="Path to OHLCV CSV file")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    settings = load_settings(config_path)
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = settings["data"]["path"]

    df = load_ohlcv_csv(data_path)

    print(f"Config:     {config_path}")
    print(f"Data path:  {data_path}")
    print(f"Full range: {df.index[0]} ~ {df.index[-1]}")
    print(f"Total bars: {len(df)}")
    print()

    validation_df, summary_df = run_phase2b(df, settings)

    Path("reports").mkdir(exist_ok=True)
    validation_df.to_csv("reports/phase2b_validation.csv", index=False)
    summary_df.to_csv("reports/phase2b_summary.csv", index=False)

    print("\n" + "=" * 50)
    print("Phase 2B Results")
    print("=" * 50)

    print(f"\n1. Success: YES")
    print(f"2. Total combos: {len(summary_df)}")
    print(f"3. Total runs: {len(validation_df)} (Q1-Q4+Full)")

    print(f"\n4. reports/phase2b_validation.csv: YES ({len(validation_df)} rows)")
    print(f"5. reports/phase2b_summary.csv: YES ({len(summary_df)} rows)")

    print(f"\n6. Stability Score Top 20:")
    print(summary_df.head(20).to_string(index=False))

    print(f"\n7. Best combo per candidate:")
    for cand_name in [c["candidate_name"] for c in CANDIDATES]:
        cand_df = summary_df[summary_df["candidate_name"] == cand_name]
        if len(cand_df) > 0:
            best = cand_df.iloc[0]
            print(f"   {cand_name}: ema={best['ema_period']} buf={best['stop_buffer']} "
                  f"score={best['stability_score']:.4f} return={best['full_year_total_return']:.2f}% "
                  f"trades={best['full_year_total_trades']:.0f} q+={best['q_positive_count']:.0f}/4")

    print(f"\n8. Combos meeting Phase 2B criteria:")
    phase2b_ready = summary_df[
        (summary_df["q_positive_count"] == 4)
        & (summary_df["q_trade_count_min"] >= 20)
        & (summary_df["full_year_profit_factor"] >= 2.0)
        & (summary_df["q_profit_factor_min"] >= 1.3)
        & (summary_df["full_year_total_trades"] >= 80)
    ]
    print(f"   Count: {len(phase2b_ready)} / {len(summary_df)}")
    if len(phase2b_ready) > 0:
        print(phase2b_ready[["candidate_name", "ema_period", "stop_buffer",
                              "stability_score", "full_year_total_return", "full_year_total_trades",
                              "full_year_profit_factor", "q_positive_count", "q_trade_count_min",
                              "q_profit_factor_min"]].to_string(index=False))
    else:
        print("   None")

    print(f"\n9. EMA period analysis:")
    for ema_p in EMA_PERIODS:
        ema_group = summary_df[summary_df["ema_period"] == ema_p]
        avg_score = ema_group["stability_score"].mean()
        avg_trades = ema_group["full_year_total_trades"].mean()
        avg_pf = ema_group["full_year_profit_factor"].mean()
        ready_count = len(ema_group[
            (ema_group["q_positive_count"] == 4)
            & (ema_group["q_trade_count_min"] >= 20)
            & (ema_group["full_year_profit_factor"] >= 2.0)
            & (ema_group["q_profit_factor_min"] >= 1.3)
            & (ema_group["full_year_total_trades"] >= 80)
        ])
        print(f"   EMA {ema_p:>2}: avg_score={avg_score:.4f} avg_trades={avg_trades:.0f} avg_pf={avg_pf:.2f} ready={ready_count}/9")

    print(f"\n10. Stop buffer analysis:")
    for buf in STOP_BUFFERS:
        buf_group = summary_df[summary_df["stop_buffer"] == buf]
        avg_score = buf_group["stability_score"].mean()
        avg_trades = buf_group["full_year_total_trades"].mean()
        avg_pf = buf_group["full_year_profit_factor"].mean()
        ready_count = len(buf_group[
            (buf_group["q_positive_count"] == 4)
            & (buf_group["q_trade_count_min"] >= 20)
            & (buf_group["full_year_profit_factor"] >= 2.0)
            & (buf_group["q_profit_factor_min"] >= 1.3)
            & (buf_group["full_year_total_trades"] >= 80)
        ])
        print(f"   buf {buf:>2}: avg_score={avg_score:.4f} avg_trades={avg_trades:.0f} avg_pf={avg_pf:.2f} ready={ready_count}/9")

    print(f"\n11. Long/Short balance:")
    summary_df["balance_ratio"] = summary_df["long_trades"] / summary_df[["long_trades", "short_trades"]].sum(axis=1).clip(lower=1)
    for cand_name in [c["candidate_name"] for c in CANDIDATES]:
        cand_df = summary_df[summary_df["candidate_name"] == cand_name]
        avg_long = cand_df["long_trades"].mean()
        avg_short = cand_df["short_trades"].mean()
        print(f"   {cand_name}: avg_long={avg_long:.1f} avg_short={avg_short:.1f}")

    print(f"\n12. End-of-data exits:")
    eod_by_period = validation_df.groupby("period")["end_of_data_exits"].sum()
    for period_name, total_eod in eod_by_period.items():
        print(f"   {period_name}: {total_eod} end-of-data exits")


if __name__ == "__main__":
    main()
