from pathlib import Path

from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.custom_engine import run_custom_backtest
from configs.settings import load_settings


def main() -> None:
    settings = load_settings(Path("configs/base.yaml"))
    example_path = Path("example_data/sample_ohlcv.csv")
    data_path = example_path if example_path.exists() else Path(settings["data"]["path"])

    df = load_ohlcv_csv(data_path)
    bundle = build_signals(df, settings["strategy"])
    result = run_custom_backtest(df, bundle, settings)

    print("Backtest Summary:")
    for k, v in result.summary.items():
        print(f"{k}: {v}")

    if not result.trades.empty:
        print("\nFirst trades:")
        print(result.trades.head().to_string(index=False))


if __name__ == "__main__":
    main()
