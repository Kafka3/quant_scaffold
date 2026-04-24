import pandas as pd
import vectorbt as vbt

from strategy.signal_builder import SignalBundle
from strategy.cost_model import get_costs


def run_backtest(df: pd.DataFrame, bundle: SignalBundle, config: dict):
    fees, slippage = get_costs(config)
    portfolio = vbt.Portfolio.from_signals(
        close=df["Close"],
        entries=bundle.entries_long,
        exits=bundle.exits_long,
        short_entries=bundle.entries_short,
        short_exits=bundle.exits_short,
        init_cash=config.get("init_cash", 100000),
        fees=fees,
        slippage=slippage,
        freq=config.get("freq", None),
    )
    return portfolio
