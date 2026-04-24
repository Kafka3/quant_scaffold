from pathlib import Path
import copy
import pandas as pd

from configs.settings import load_settings
from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.vectorbt_engine import run_backtest
from backtest.validation import simple_objective


def rolling_windows(index: pd.Index, train_size: int, test_size: int, step: int):
    start = 0
    while start + train_size + test_size <= len(index):
        train_idx = index[start : start + train_size]
        test_idx = index[start + train_size : start + train_size + test_size]
        yield train_idx, test_idx
        start += step


def normalize_backtest_summary(result) -> pd.Series:
    if hasattr(result, "summary"):
        summary = pd.Series(result.summary)
    elif isinstance(result, pd.Series):
        summary = result
    elif isinstance(result, dict):
        summary = pd.Series(result)
    else:
        raise ValueError("Unsupported backtest result type")

    summary = summary.copy()
    trades = getattr(result, "trades", None)
    equity = getattr(result, "equity", None)

    if "total_trades" not in summary:
        summary["total_trades"] = len(trades) if trades is not None else 0

    if "win_rate" not in summary:
        if trades is not None and len(trades) > 0:
            summary["win_rate"] = len(trades[trades["pnl"] > 0]) / len(trades)
        else:
            summary["win_rate"] = 0.0

    if "total_return" not in summary and equity is not None and len(equity) > 0:
        start = float(equity.iloc[0])
        end = float(equity.iloc[-1])
        summary["total_return"] = (end / start - 1) * 100

    if "profit_factor" not in summary and trades is not None:
        gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
        gross_loss = float(-trades.loc[trades["pnl"] < 0, "pnl"].sum())
        if gross_loss > 0:
            summary["profit_factor"] = gross_profit / gross_loss
        elif gross_profit > 0:
            summary["profit_factor"] = float("inf")
        else:
            summary["profit_factor"] = 0.0

    if "max_drawdown" not in summary and equity is not None and len(equity) > 0:
        peak = equity.cummax()
        drawdown = (equity - peak) / peak.replace(0, 1)
        summary["max_drawdown"] = float(drawdown.min() * 100)

    return summary


def select_params_on_train(df_train: pd.DataFrame, settings: dict) -> dict:
    candidates = [34, 55, 89]
    best_score = float("-inf")
    best_cfg = copy.deepcopy(settings)
    for ema_p in candidates:
        cfg = copy.deepcopy(settings)
        cfg["strategy"]["trend"]["ema_period"] = ema_p
        bundle = build_signals(df_train, cfg["strategy"])
        result = run_backtest(df_train, bundle, cfg)
        summary = normalize_backtest_summary(result)
        score = simple_objective(summary)
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
        result = run_backtest(df_test, bundle, best_cfg)
        summary = normalize_backtest_summary(result)
        rows.append({
            "train_start": str(train_idx[0]),
            "train_end": str(train_idx[-1]),
            "test_start": str(test_idx[0]),
            "test_end": str(test_idx[-1]),
            "ema_period": best_cfg["strategy"]["trend"]["ema_period"],
            "total_return": summary["total_return"],
            "total_trades": summary["total_trades"],
            "win_rate": summary["win_rate"],
            "profit_factor": summary["profit_factor"],
            "max_drawdown": summary["max_drawdown"],
            "score": simple_objective(summary),
            **summary.to_dict(),
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv("reports/walk_forward_results.csv", index=False)


if __name__ == "__main__":
    main()
