import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.vectorbt_engine import run_backtest
from configs.settings import load_settings


def _print_data_quality(df: pd.DataFrame) -> None:
    print("=" * 50)
    print("Data Quality")
    print("=" * 50)
    print(f"total_bars:      {len(df)}")
    print(f"start_time:      {df.index[0]}")
    print(f"end_time:        {df.index[-1]}")
    print(f"duration:        {(df.index[-1] - df.index[0]).days} days")

    # Missing values
    missing = df.isna().sum()
    if missing.any():
        print(f"missing_values:  {missing.to_dict()}")
    else:
        print("missing_values:  none")

    # Time continuity (5m interval = 300s)
    expected_interval = pd.Timedelta(minutes=5)
    gaps = df.index.to_series().diff().dropna()
    gap_mask = gaps > expected_interval
    gap_count = gap_mask.sum()
    if gap_count > 0:
        max_gap = gaps[gap_mask].max()
        print(f"time_gaps:       {gap_count} gaps detected (max={max_gap})")
    else:
        print("time_gaps:       none")
    print()


def _print_signal_diagnostics(bundle) -> None:
    print("=" * 50)
    print("Signal Diagnostics")
    print("=" * 50)

    f = bundle.features

    # Raw divergence candidates (price + osc + spacing only)
    raw_bull = int(f["bullish_raw_divergence"].sum())
    raw_bear = int(f["bearish_raw_divergence"].sum())
    print(f"raw_bullish_divergence_candidates:  {raw_bull}")
    print(f"raw_bearish_divergence_candidates:  {raw_bear}")

    # Layer-1: prior trend filter
    bull_trend = int(f["bullish_prior_trend_ok"].sum())
    bear_trend = int(f["bearish_prior_trend_ok"].sum())
    print(f"bullish_prior_trend_pass:           {bull_trend} / {raw_bull}")
    print(f"bearish_prior_trend_pass:           {bear_trend} / {raw_bear}")

    # Layer-2: progressive channel filter
    bull_p1_ch = int(f["bullish_pivot1_channel_ok"].sum())
    bull_p2_ch = int(f["bullish_pivot2_channel_ok"].sum())
    bear_p1_ch = int(f["bearish_pivot1_channel_ok"].sum())
    bear_p2_ch = int(f["bearish_pivot2_channel_ok"].sum())
    print(f"bullish_pivot1_channel_pass:        {bull_p1_ch} / {raw_bull}")
    print(f"bullish_pivot2_channel_pass:        {bull_p2_ch} / {raw_bull}")
    print(f"bearish_pivot1_channel_pass:        {bear_p1_ch} / {raw_bear}")
    print(f"bearish_pivot2_channel_pass:        {bear_p2_ch} / {raw_bear}")

    # Final divergence signals (all filters passed)
    final_bull = int(f["bullish_div"].sum())
    final_bear = int(f["bearish_div"].sum())
    print(f"final_bullish_divergences:          {final_bull}")
    print(f"final_bearish_divergences:          {final_bear}")

    # Setups created
    setups_long = f["bullish_setup_active"].any()
    setups_short = f["bearish_setup_active"].any()
    print(f"long_setups_created:                {setups_long}")
    print(f"short_setups_created:               {setups_short}")

    # Triggered entries
    entries_long = f["long_entry_price"].notna().sum()
    entries_short = f["short_entry_price"].notna().sum()
    print(f"long_entries:                       {int(entries_long)}")
    print(f"short_entries:                      {int(entries_short)}")

    # Expired setups
    expired_long = int(f["bullish_setup_expired"].sum())
    expired_short = int(f["bearish_setup_expired"].sum())
    print(f"bullish_setups_expired:             {expired_long}")
    print(f"bearish_setups_expired:             {expired_short}")
    print()


def main() -> None:
    parser = ArgumentParser(description="Run BTC 5m strategy validation")
    parser.add_argument("--config", dest="config_path", default="configs/base.yaml",
                        help="Path to config YAML file (default: configs/base.yaml)")
    parser.add_argument("--data", dest="data_path", default=None,
                        help="Path to BTC 5m OHLCV CSV file")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    settings = load_settings(config_path)
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = Path(settings["data"]["path"])

    df = load_ohlcv_csv(data_path)
    print(f"Config:     {config_path}")
    print(f"Data path:  {data_path}")
    print()
    _print_data_quality(df)

    bundle = build_signals(df, settings["strategy"])
    _print_signal_diagnostics(bundle)

    result = run_backtest(df, bundle, settings["backtest"])

    print("=" * 50)
    print("Backtest Summary")
    print("=" * 50)
    for k, v in result.summary.items():
        print(f"{k}: {v}")
    print()

    if not result.trades.empty:
        columns = [
            "entry_time",
            "exit_time",
            "side",
            "setup_pivot2_time",
            "setup_confirm_time",
            "entry_price",
            "trigger_price",
            "stop_price",
            "target_price",
            "exit_reason",
            "pnl",
        ]
        available_columns = [c for c in columns if c in result.trades.columns]
        print("=" * 50)
        print("Recent Trades")
        print("=" * 50)
        print(result.trades[available_columns].tail(20).to_string(index=False))
    else:
        print("=" * 50)
        print("Recent Trades")
        print("=" * 50)
        print("no trades generated")


if __name__ == "__main__":
    main()
