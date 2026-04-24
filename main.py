from pathlib import Path
import pandas as pd

from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.custom_engine import run_custom_backtest
from configs.settings import load_settings


def main() -> None:
    settings = load_settings(Path("configs/base.yaml"))
    df = load_ohlcv_csv(settings["data"]["path"])
    bundle = build_signals(df, settings["strategy"])
    result = run_custom_backtest(df, bundle, settings)
    print("Backtest Summary:")
    for k, v in result.summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
