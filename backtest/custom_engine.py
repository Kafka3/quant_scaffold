from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    summary: dict


def run_custom_backtest(df: pd.DataFrame, bundle, config: dict) -> BacktestResult:
    """
    Simple bar-by-bar backtester for dynamic stops/targets.
    Assumes only one position at a time, long or short.
    """
    cash = config["backtest"]["init_cash"]
    position = 0  # 1 for long, -1 for short, 0 for flat
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    equity = [cash]
    trades = []

    for i, idx in enumerate(df.index):
        high = df.loc[idx, 'High']
        low = df.loc[idx, 'Low']
        close = df.loc[idx, 'Close']

        # Check exits
        if position == 1:  # Long
            if low <= stop_price:  # Hit stop
                exit_price = stop_price
                pnl = exit_price - entry_price
                cash += pnl
                trades.append({'type': 'long', 'entry': entry_price, 'exit': exit_price, 'pnl': pnl, 'exit_reason': 'stop'})
                position = 0
            elif high >= target_price:  # Hit target
                exit_price = target_price
                pnl = exit_price - entry_price
                cash += pnl
                trades.append({'type': 'long', 'entry': entry_price, 'exit': exit_price, 'pnl': pnl, 'exit_reason': 'target'})
                position = 0
        elif position == -1:  # Short
            if high >= stop_price:  # Hit stop
                exit_price = stop_price
                pnl = entry_price - exit_price
                cash += pnl
                trades.append({'type': 'short', 'entry': entry_price, 'exit': exit_price, 'pnl': pnl, 'exit_reason': 'stop'})
                position = 0
            elif low <= target_price:  # Hit target
                exit_price = target_price
                pnl = entry_price - exit_price
                cash += pnl
                trades.append({'type': 'short', 'entry': entry_price, 'exit': exit_price, 'pnl': pnl, 'exit_reason': 'target'})
                position = 0

        # Check entries if flat
        if position == 0:
            if bundle.entries_long.loc[idx] and not pd.isna(bundle.entry_prices_long.loc[idx]):
                position = 1
                entry_price = bundle.entry_prices_long.loc[idx]
                stop_price = bundle.stop_prices_long.loc[idx]
                target_price = bundle.target_prices_long.loc[idx]
                trades.append({'type': 'long', 'entry': entry_price, 'exit': None, 'pnl': None, 'exit_reason': None})
            elif bundle.entries_short.loc[idx] and not pd.isna(bundle.entry_prices_short.loc[idx]):
                position = -1
                entry_price = bundle.entry_prices_short.loc[idx]
                stop_price = bundle.stop_prices_short.loc[idx]
                target_price = bundle.target_prices_short.loc[idx]
                trades.append({'type': 'short', 'entry': entry_price, 'exit': None, 'pnl': None, 'exit_reason': None})

        equity.append(cash)

    equity = pd.Series(equity[1:], index=df.index)
    returns = equity.pct_change().fillna(0)
    trades_df = pd.DataFrame(trades)

    # Summary
    if trades_df.empty:
        summary = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'final_equity': cash,
        }
    else:
        completed_trades = trades_df.dropna(subset=['pnl'])
        total_trades = len(completed_trades)
        winning_trades = len(completed_trades[completed_trades['pnl'] > 0])
        losing_trades = len(completed_trades[completed_trades['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = completed_trades['pnl'].sum()
        avg_win = completed_trades[completed_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = completed_trades[completed_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

        summary = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_equity': cash,
        }

    return BacktestResult(equity=equity, returns=returns, trades=trades_df, summary=summary)