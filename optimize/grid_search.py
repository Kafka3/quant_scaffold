import sys
from pathlib import Path

# Resolve project root from __file__ and inject into sys.path so imports work
# regardless of the directory the script is executed from.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from argparse import ArgumentParser
from itertools import product
import copy
import math

import pandas as pd

from configs.settings import load_settings
from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.vectorbt_engine import run_backtest


def phase1_score(summary: pd.Series) -> float:
    """
    Phase-1 grid-search scoring function.

    Rules:
      - total_trades < 20  ->  score = -9999
      - profit_factor None/NaN -> 0, inf -> 5, capped at 5
      - pf_score       = min(pf, 5) / 5
      - win_rate_score = win_rate
      - expectancy_score = clip(expectancy / 10, -1, 1)
      - return_score     = clip(total_return / 10, -1, 1)
      - drawdown_penalty = abs(max_drawdown) / 20
      - balance_penalty  = abs(long_trades - short_trades) / max(total_trades, 1)
      - frequency_penalty = 0 if total_trades <= 120 else (total_trades - 120) / 120
      - score = 0.25*pf + 0.20*wr + 0.20*exp + 0.15*ret - 0.10*dd - 0.05*balance - 0.05*freq
    """
    trades = int(summary.get("total_trades", 0) or 0)
    if trades < 20:
        return -9999.0

    # profit_factor safe handling
    pf = summary.get("profit_factor", 0.0)
    if pf is None or pd.isna(pf):
        pf = 0.0
    elif isinstance(pf, float) and math.isinf(pf):
        pf = 5.0
    pf = float(pf)
    if pf > 5.0:
        pf = 5.0
    elif pf < 0.0:
        pf = 0.0

    win_rate = float(summary.get("win_rate", 0.0) or 0.0)
    expectancy = float(summary.get("expectancy", 0.0) or 0.0)
    total_return = float(summary.get("total_return", 0.0) or 0.0)
    max_drawdown = float(summary.get("max_drawdown", 0.0) or 0.0)
    long_trades = int(summary.get("long_trades", 0) or 0)
    short_trades = int(summary.get("short_trades", 0) or 0)

    pf_score = pf / 5.0
    win_rate_score = win_rate
    expectancy_score = max(min(expectancy / 10.0, 1.0), -1.0)
    return_score = max(min(total_return / 10.0, 1.0), -1.0)
    drawdown_penalty = abs(max_drawdown) / 20.0
    balance_penalty = abs(long_trades - short_trades) / max(trades, 1)
    frequency_penalty = 0.0 if trades <= 120 else (trades - 120) / 120.0

    return (
        0.25 * pf_score
        + 0.20 * win_rate_score
        + 0.20 * expectancy_score
        + 0.15 * return_score
        - 0.10 * drawdown_penalty
        - 0.05 * balance_penalty
        - 0.05 * frequency_penalty
    )


def compute_extra_metrics(result) -> dict:
    """Extract extra statistics from BacktestResult for grid-search logging."""
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


def extract_diagnostics(bundle) -> dict:
    """Extract signal-funnel diagnostics from SignalBundle features."""
    f = bundle.features
    return {
        "raw_bullish_divergence_candidates": int(f["bullish_raw_divergence"].sum()),
        "raw_bearish_divergence_candidates": int(f["bearish_raw_divergence"].sum()),
        "bullish_prior_trend_pass": int(f["bullish_prior_trend_ok"].sum()),
        "bearish_prior_trend_pass": int(f["bearish_prior_trend_ok"].sum()),
        "bullish_pivot1_channel_pass": int(f["bullish_pivot1_channel_ok"].sum()),
        "bullish_pivot2_channel_pass": int(f["bullish_pivot2_channel_ok"].sum()),
        "bearish_pivot1_channel_pass": int(f["bearish_pivot1_channel_ok"].sum()),
        "bearish_pivot2_channel_pass": int(f["bearish_pivot2_channel_ok"].sum()),
        "final_bullish_divergences": int(f["bullish_div"].sum()),
        "final_bearish_divergences": int(f["bearish_div"].sum()),
    }


def _run_single_combo(args_tuple):
    """Worker function for multiprocessing. Must be top-level for pickle."""
    (lbk, mcr, lb, rb, settings, df) = args_tuple

    cfg = copy.deepcopy(settings)
    cfg["strategy"]["trend"]["lookback_bars"] = lbk
    cfg["strategy"]["trend"]["min_close_ratio"] = mcr
    cfg["strategy"]["pivots"]["left_bars"] = lb
    cfg["strategy"]["pivots"]["right_bars"] = rb

    bundle = build_signals(df, cfg["strategy"])
    result = run_backtest(df, bundle, cfg["backtest"])
    summary = pd.Series(result.summary)
    extras = compute_extra_metrics(result)
    diagnostics = extract_diagnostics(bundle)

    merged_summary = summary.copy()
    merged_summary["long_trades"] = extras["long_trades"]
    merged_summary["short_trades"] = extras["short_trades"]
    score = phase1_score(merged_summary)

    return {
        "lookback_bars": lbk,
        "min_close_ratio": mcr,
        "left_bars": lb,
        "right_bars": rb,
        "score": score,
        "total_return": summary.get("total_return", 0.0),
        "total_trades": summary.get("total_trades", 0),
        "win_rate": summary.get("win_rate", 0.0),
        "profit_factor": summary.get("profit_factor", 0.0),
        "max_drawdown": summary.get("max_drawdown", 0.0),
        "avg_trade": summary.get("avg_trade", 0.0),
        "expectancy": summary.get("expectancy", 0.0),
        **extras,
        **diagnostics,
    }


def main() -> None:
    parser = ArgumentParser(description="BTC 5m continuation divergence Phase 1 grid search")
    parser.add_argument("--config", dest="config_path", default="configs/base.yaml",
                        help="Path to config YAML file (default: configs/base.yaml)")
    parser.add_argument("--data", dest="data_path", default=None,
                        help="Path to OHLCV CSV file (overrides config)")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    settings = load_settings(config_path)
    if args.data_path:
        settings["data"]["path"] = args.data_path

    data_path = settings["data"]["path"]
    df = load_ohlcv_csv(data_path)

    # Data summary
    start_time = df.index[0]
    end_time = df.index[-1]
    total_bars = len(df)

    # Phase 1 search space
    lookback_bars_list = [6, 8, 12, 16, 24]
    min_close_ratio_list = [0.60, 0.70, 0.80, 0.90, 1.00]
    left_bars_list = [2, 3, 4]
    right_bars_list = [2, 3, 4]

    total_combos = (
        len(lookback_bars_list)
        * len(min_close_ratio_list)
        * len(left_bars_list)
        * len(right_bars_list)
    )

    print("=" * 50)
    print("Phase 1 Grid Search")
    print("=" * 50)
    print(f"Data path:       {data_path}")
    print(f"Start time:      {start_time}")
    print(f"End time:        {end_time}")
    print(f"Total bars:      {total_bars}")
    print(f"Total parameter combinations: {total_combos}")
    print("=" * 50)

    # Build task list for multiprocessing.
    tasks = [
        (lbk, mcr, lb, rb, settings, df)
        for lbk, mcr, lb, rb in product(
            lookback_bars_list, min_close_ratio_list, left_bars_list, right_bars_list
        )
    ]

    import multiprocessing as mp
    # macOS: use fewer workers to avoid Bus error (memory pressure).
    workers = 2
    print(f"Using {workers} parallel workers")
    print("=" * 50)

    rows = []
    checkpoint_every = 50
    next_checkpoint = checkpoint_every

    with mp.Pool(processes=workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_run_single_combo, tasks), start=1):
            rows.append(row)
            print(f"[{i:>3}/{total_combos}] lookback={row['lookback_bars']:>2} mcr={row['min_close_ratio']:.2f} "
                  f"left={row['left_bars']} right={row['right_bars']} -> "
                  f"score={row['score']:>8.4f} trades={row['long_trades'] + row['short_trades']:>3}")

            if i >= next_checkpoint:
                pd.DataFrame(rows).to_csv("reports/_grid_search_checkpoint.csv", index=False)
                print(f"  -> checkpoint saved ({i} combos)")
                next_checkpoint += checkpoint_every

    out = pd.DataFrame(rows).sort_values("score", ascending=False)

    # Terminal output: top 20
    print("\n" + "=" * 50)
    print("Top 20 by score")
    print("=" * 50)
    print(out.head(20).to_string(index=False))

    # Save full results
    Path("reports").mkdir(exist_ok=True)
    out.to_csv("reports/grid_search_phase1.csv", index=False)
    print(f"\nSaved {len(out)} rows to reports/grid_search_phase1.csv")

    # Save top 50
    top50 = out.head(50)
    top50.to_csv("reports/grid_search_phase1_top50.csv", index=False)
    print(f"Saved top 50 to reports/grid_search_phase1_top50.csv")

    # Parameter region summary
    print("\n" + "=" * 50)
    print("Parameter Region Summary")
    print("=" * 50)
    param_summary_rows = []
    for param_name in ["lookback_bars", "min_close_ratio", "left_bars", "right_bars"]:
        for val, group in out.groupby(param_name):
            param_summary_rows.append({
                "parameter": param_name,
                "value": val,
                "count": len(group),
                "avg_score": group["score"].mean(),
                "avg_total_trades": group["total_trades"].mean(),
                "avg_profit_factor": group["profit_factor"].mean(),
                "avg_total_return": group["total_return"].mean(),
                "avg_max_drawdown": group["max_drawdown"].mean(),
            })

    param_summary = pd.DataFrame(param_summary_rows)
    param_summary = param_summary.sort_values(["parameter", "avg_score"], ascending=[True, False])
    param_summary.to_csv("reports/grid_search_phase1_param_summary.csv", index=False)
    print(param_summary.to_string(index=False))
    print(f"\nSaved param summary to reports/grid_search_phase1_param_summary.csv")


if __name__ == "__main__":
    main()
