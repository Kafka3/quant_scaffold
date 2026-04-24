from itertools import product
from pathlib import Path
import copy
import pandas as pd

from configs.settings import load_settings
from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.vectorbt_engine import run_backtest
from backtest.metrics import summarize_portfolio
from backtest.validation import simple_objective


def main() -> None:
    settings = load_settings(Path("configs/base.yaml"))
    df = load_ohlcv_csv(settings["data"]["path"])

    ema_periods = [34, 55, 89]
    rr_targets = [1.5, 2.0, 2.5]
    left_bars = [2, 3, 4]

    rows = []
    for ema_p, rr, lb in product(ema_periods, rr_targets, left_bars):
        cfg = copy.deepcopy(settings)
        cfg["strategy"]["trend"]["ema_period"] = ema_p
        cfg["strategy"]["risk"]["rr_target"] = rr
        cfg["strategy"]["pivots"]["left_bars"] = lb
        cfg["strategy"]["pivots"]["right_bars"] = lb

        bundle = build_signals(df, cfg["strategy"])
        pf = run_backtest(df, bundle, cfg["backtest"])
        summary = summarize_portfolio(pf)
        score = simple_objective(summary)

        rows.append({
            "ema_period": ema_p,
            "rr_target": rr,
            "pivot_bars": lb,
            "score": score,
            **summary.to_dict(),
        })

    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    print(out.head(10).to_string(index=False))
    out.to_csv("reports/grid_search_results.csv", index=False)


if __name__ == "__main__":
    main()
