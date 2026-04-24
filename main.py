from argparse import ArgumentParser
from pathlib import Path

from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.vectorbt_engine import run_backtest
from configs.settings import load_settings


def main() -> None:
    parser = ArgumentParser(description="Run BTC 5m strategy validation")
    parser.add_argument("--data", dest="data_path", default=None, help="Path to BTC 5m OHLCV CSV file")
    args = parser.parse_args()

    settings = load_settings(Path("configs/base.yaml"))
    example_path = Path("example_data/sample_ohlcv.csv")
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = example_path if example_path.exists() else Path(settings["data"]["path"])

    df = load_ohlcv_csv(data_path)
    bundle = build_signals(df, settings["strategy"])
    result = run_backtest(df, bundle, settings)

    print("Backtest Summary:")
    for k, v in result.summary.items():
        print(f"{k}: {v}")

    if not result.trades.empty:
        columns = [
            "entry_time",
            "exit_time",
            "side",
            "setup_pivot2_time",
            "entry_price",
            "trigger_price",
            "stop_price",
            "target_price",
            "exit_reason",
            "pnl",
        ]
        available_columns = [c for c in columns if c in result.trades.columns]
        print("\nRecent trades:")
        print(result.trades[available_columns].tail(20).to_string(index=False))
    else:
        print("\nNo trades were generated.")


if __name__ == "__main__":
    main()
