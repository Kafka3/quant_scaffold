from pathlib import Path
import copy
import pandas as pd

from configs.settings import load_settings
from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.vectorbt_engine import run_backtest
from backtest.metrics import summarize_portfolio
from backtest.validation import simple_objective


def rolling_windows(index: pd.Index, train_size: int, test_size: int, step: int):
    start = 0
    while start + train_size + test_size <= len(index):
        train_idx = index[start : start + train_size]
        test_idx = index[start + train_size : start + train_size + test_size]
        yield train_idx, test_idx
        start += step


def select_params_on_train(df_train: pd.DataFrame, settings: dict) -> dict:
    candidates = [34, 55, 89]
    best_score = float("-inf")
    best_cfg = copy.deepcopy(settings)
    for ema_p in candidates:
        cfg = copy.deepcopy(settings)
        cfg["strategy"]["trend"]["ema_period"] = ema_p
        bundle = build_signals(df_train, cfg["strategy"])
        pf = run_backtest(df_train, bundle, cfg["backtest"])
        score = simple_objective(summarize_portfolio(pf))
        if score > best_score:
            best_score = score
            best_cfg = cfg
    return best_cfg


def main() -> None:
    settings = load_settings(Path("configs/base.yaml"))
    df = load_ohlcv_csv(settings["data"]["path"])

    rows = []
    for train_idx, test_idx in rolling_windows(df.index, train_size=500, test_size=100, step=100):
        df_train = df.loc[train_idx]
        df_test = df.loc[test_idx]
        best_cfg = select_params_on_train(df_train, settings)
        bundle = build_signals(df_test, best_cfg["strategy"])
        pf = run_backtest(df_test, bundle, best_cfg["backtest"])
        summary = summarize_portfolio(pf)
        rows.append({
            "train_start": str(train_idx[0]),
            "train_end": str(train_idx[-1]),
            "test_start": str(test_idx[0]),
            "test_end": str(test_idx[-1]),
            "ema_period": best_cfg["strategy"]["trend"]["ema_period"],
            **summary.to_dict(),
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv("reports/walk_forward_results.csv", index=False)


if __name__ == "__main__":
    main()
