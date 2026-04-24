from itertools import product
from pathlib import Path
import copy
import pandas as pd

from configs.settings import load_settings
from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.vectorbt_engine import run_backtest
from backtest.validation import simple_objective


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
        result = run_backtest(df, bundle, cfg)
        summary = normalize_backtest_summary(result)
        score = simple_objective(summary)

        rows.append({
            "ema_period": ema_p,
            "rr_target": rr,
            "left_bars": lb,
            "right_bars": lb,
            "score": score,
            "total_return": summary["total_return"],
            "total_trades": summary["total_trades"],
            "win_rate": summary["win_rate"],
            "profit_factor": summary["profit_factor"],
            "max_drawdown": summary["max_drawdown"],
            **summary.to_dict(),
        })

    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    print(out.head(10).to_string(index=False))
    out.to_csv("reports/grid_search_results.csv", index=False)


if __name__ == "__main__":
    main()
