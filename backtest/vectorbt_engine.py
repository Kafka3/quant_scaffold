from dataclasses import dataclass
from typing import Optional

import pandas as pd

from strategy.signal_builder import SignalBundle


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    summary: dict


def run_backtest(df: pd.DataFrame, bundle: SignalBundle, config: dict) -> BacktestResult:
    init_cash = config.get("init_cash", 100000)
    cash = float(init_cash)
    position: str = "flat"
    current_trade: Optional[dict] = None

    equity_values = []
    trades = []

    for idx in df.index:
        high = df.loc[idx, "High"]
        low = df.loc[idx, "Low"]
        close = df.loc[idx, "Close"]

        # Exit logic for an existing position before new entries on the same bar.
        if position == "long" and current_trade is not None:
            if low <= current_trade["stop_price"]:
                exit_price = current_trade["stop_price"]
                exit_reason = "stop"
            elif high >= current_trade["target_price"]:
                exit_price = current_trade["target_price"]
                exit_reason = "target"
            else:
                exit_price = None
                exit_reason = None

            if exit_price is not None:
                pnl = exit_price - current_trade["entry_price"]
                return_pct = pnl / current_trade["entry_price"]
                cash += pnl
                current_trade.update(
                    {
                        "exit_time": idx,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "return_pct": return_pct,
                        "exit_reason": exit_reason,
                    }
                )
                trades.append(current_trade)
                current_trade = None
                position = "flat"

        elif position == "short" and current_trade is not None:
            if high >= current_trade["stop_price"]:
                exit_price = current_trade["stop_price"]
                exit_reason = "stop"
            elif low <= current_trade["target_price"]:
                exit_price = current_trade["target_price"]
                exit_reason = "target"
            else:
                exit_price = None
                exit_reason = None

            if exit_price is not None:
                pnl = current_trade["entry_price"] - exit_price
                return_pct = pnl / current_trade["entry_price"]
                cash += pnl
                current_trade.update(
                    {
                        "exit_time": idx,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "return_pct": return_pct,
                        "exit_reason": exit_reason,
                    }
                )
                trades.append(current_trade)
                current_trade = None
                position = "flat"

        # Entry logic only when flat.
        if position == "flat":
            if bundle.entries_long.loc[idx] and bundle.entries_short.loc[idx]:
                # Both long and short signals on the same bar are ambiguous.
                # Skip entry in this bar to avoid conflicting positions.
                pass
            elif bundle.entries_long.loc[idx]:
                entry_price = bundle.long_entry_price.loc[idx]
                stop_price = bundle.long_stop_price.loc[idx]
                target_price = bundle.long_target_price.loc[idx]
                if pd.notna(entry_price) and pd.notna(stop_price) and pd.notna(target_price):
                    position = "long"
                    current_trade = {
                        "entry_time": idx,
                        "side": "long",
                        "entry_price": entry_price,
                        "stop_price": stop_price,
                        "target_price": target_price,
                    }
                    # Same-bar exit after entry is possible and will be evaluated
                    # in the next loop iteration of this bar.
            elif bundle.entries_short.loc[idx]:
                entry_price = bundle.short_entry_price.loc[idx]
                stop_price = bundle.short_stop_price.loc[idx]
                target_price = bundle.short_target_price.loc[idx]
                if pd.notna(entry_price) and pd.notna(stop_price) and pd.notna(target_price):
                    position = "short"
                    current_trade = {
                        "entry_time": idx,
                        "side": "short",
                        "entry_price": entry_price,
                        "stop_price": stop_price,
                        "target_price": target_price,
                    }

        # Compute equity curve with mark-to-market on the current bar.
        if position == "flat" or current_trade is None:
            equity_values.append(cash)
        elif position == "long":
            equity_values.append(cash + (close - current_trade["entry_price"]))
        else:
            equity_values.append(cash + (current_trade["entry_price"] - close))

    equity = pd.Series(equity_values, index=df.index)
    trades_df = pd.DataFrame(trades)

    total_trades = len(trades_df)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]
    gross_profit = wins["pnl"].sum() if total_trades > 0 else 0.0
    gross_loss = -losses["pnl"].sum() if total_trades > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
    avg_trade = trades_df["pnl"].mean() if total_trades > 0 else 0.0
    expectancy = win_rate * (wins["pnl"].mean() if len(wins) > 0 else 0.0) + (1 - win_rate) * (losses["pnl"].mean() if len(losses) > 0 else 0.0)
    peak = equity.cummax()
    drawdown = (equity - peak) / peak.replace(0, 1)
    max_drawdown = drawdown.min() * 100
    total_return = (equity.iloc[-1] / init_cash - 1) * 100 if len(equity) > 0 else 0.0

    summary = {
        "total_return": total_return,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "avg_trade": avg_trade,
        "expectancy": expectancy,
    }

    return BacktestResult(equity=equity, trades=trades_df, summary=summary)
