from dataclasses import dataclass
from typing import Optional, List, Tuple

import pandas as pd

from strategy.signal_builder import SignalBundle


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    summary: dict
    warnings: List[str]


def run_backtest(df: pd.DataFrame, bundle: SignalBundle, config: dict) -> BacktestResult:
    """
    Custom event-driven backtester.

    Rules:
      - Flat only entry.
      - If long & short signal on same bar → skip, record warning.
      - Same-bar exit checked immediately after entry.
      - Stop takes priority over target on same-bar collisions.
      - Subsequent bars: stop > target priority maintained.
    """
    init_cash = float(config.get("init_cash", 100000))
    cash = init_cash
    position: str = "flat"
    current_trade: Optional[dict] = None

    equity_values: List[float] = []
    trades: List[dict] = []
    warnings: List[str] = []

    # Pre-defined columns so 0-trade DataFrame stays structurally stable.
    trade_columns = [
        "entry_time",
        "exit_time",
        "side",
        "setup_pivot2_time",
        "entry_price",
        "trigger_price",
        "stop_price",
        "target_price",
        "exit_price",
        "pnl",
        "return_pct",
        "exit_reason",
        "bars_held",
    ]

    for idx in df.index:
        high = float(df.loc[idx, "High"])
        low = float(df.loc[idx, "Low"])
        close = float(df.loc[idx, "Close"])

        bar_entered = False

        # ------------------------------------------------------------------
        # 1. Entry logic: only when flat
        # ------------------------------------------------------------------
        if position == "flat":
            long_entry = bool(bundle.entries_long.loc[idx])
            short_entry = bool(bundle.entries_short.loc[idx])

            if long_entry and short_entry:
                warnings.append(f"ambiguous long/short entry on {idx}, skip this bar")
            elif long_entry:
                entry_price = bundle.long_entry_price.loc[idx]
                stop_price = bundle.long_stop_price.loc[idx]
                target_price = bundle.long_target_price.loc[idx]
                if pd.notna(entry_price) and pd.notna(stop_price) and pd.notna(target_price):
                    position = "long"
                    current_trade = {
                        "entry_time": idx,
                        "side": "long",
                        "entry_price": float(entry_price),
                        "stop_price": float(stop_price),
                        "target_price": float(target_price),
                        "setup_pivot2_time": bundle.long_setup_pivot2_time.loc[idx],
                        "trigger_price": bundle.long_trigger_price_raw.loc[idx],
                        "bars_held": 0,
                    }
                    bar_entered = True
            elif short_entry:
                entry_price = bundle.short_entry_price.loc[idx]
                stop_price = bundle.short_stop_price.loc[idx]
                target_price = bundle.short_target_price.loc[idx]
                if pd.notna(entry_price) and pd.notna(stop_price) and pd.notna(target_price):
                    position = "short"
                    current_trade = {
                        "entry_time": idx,
                        "side": "short",
                        "entry_price": float(entry_price),
                        "stop_price": float(stop_price),
                        "target_price": float(target_price),
                        "setup_pivot2_time": bundle.short_setup_pivot2_time.loc[idx],
                        "trigger_price": bundle.short_trigger_price_raw.loc[idx],
                        "bars_held": 0,
                    }
                    bar_entered = True

        # ------------------------------------------------------------------
        # 2. Same-bar exit check (immediately after entry)
        #    Priority: stop > target
        # ------------------------------------------------------------------
        if bar_entered and current_trade is not None:
            exit_price, exit_reason = _check_exit(current_trade, high, low)
            if exit_price is not None:
                pnl = _finalize_trade(current_trade, idx, exit_price, exit_reason)
                cash += pnl
                trades.append(current_trade)
                current_trade = None
                position = "flat"

        # ------------------------------------------------------------------
        # 3. Subsequent-bar exit check
        #    Priority: stop > target
        # ------------------------------------------------------------------
        if not bar_entered and current_trade is not None:
            exit_price, exit_reason = _check_exit(current_trade, high, low)
            if exit_price is not None:
                pnl = _finalize_trade(current_trade, idx, exit_price, exit_reason)
                cash += pnl
                trades.append(current_trade)
                current_trade = None
                position = "flat"

        # ------------------------------------------------------------------
        # 4. bars_held counting (entry bar itself is NOT counted)
        # ------------------------------------------------------------------
        if current_trade is not None and idx != current_trade["entry_time"]:
            current_trade["bars_held"] += 1

        # ------------------------------------------------------------------
        # 5. Mark-to-market equity
        # ------------------------------------------------------------------
        if position == "flat" or current_trade is None:
            equity_values.append(cash)
        elif position == "long":
            equity_values.append(cash + (close - current_trade["entry_price"]))
        else:  # short
            equity_values.append(cash + (current_trade["entry_price"] - close))

    # ------------------------------------------------------------------
    # Assemble results
    # ------------------------------------------------------------------
    equity = pd.Series(equity_values, index=df.index)

    if trades:
        trades_df = pd.DataFrame(trades)
        # Reorder columns to the canonical list; ignore any missing (defensive).
        trades_df = trades_df[[c for c in trade_columns if c in trades_df.columns]]
    else:
        trades_df = pd.DataFrame(columns=trade_columns)

    summary = _build_summary(trades_df, equity, init_cash)

    return BacktestResult(
        trades=trades_df,
        equity=equity,
        summary=summary,
        warnings=warnings,
    )


def _check_exit(trade: dict, high: float, low: float) -> Tuple[Optional[float], Optional[str]]:
    """
    Determine whether the bar hits stop or target.
    Stop always takes priority when both are hit on the same bar.
    """
    if trade["side"] == "long":
        if low <= trade["stop_price"]:
            return trade["stop_price"], "stop"
        elif high >= trade["target_price"]:
            return trade["target_price"], "target"
    else:  # short
        if high >= trade["stop_price"]:
            return trade["stop_price"], "stop"
        elif low <= trade["target_price"]:
            return trade["target_price"], "target"
    return None, None


def _finalize_trade(trade: dict, exit_time, exit_price: float, exit_reason: str) -> float:
    """
    Populate exit fields on the trade dict and return the PnL.
    """
    if trade["side"] == "long":
        pnl = exit_price - trade["entry_price"]
    else:
        pnl = trade["entry_price"] - exit_price

    entry_price = trade["entry_price"]
    return_pct = pnl / entry_price if entry_price != 0 else 0.0

    trade.update({
        "exit_time": exit_time,
        "exit_price": exit_price,
        "pnl": pnl,
        "return_pct": return_pct,
        "exit_reason": exit_reason,
    })
    return pnl


def _build_summary(trades_df: pd.DataFrame, equity: pd.Series, init_cash: float) -> dict:
    """
    Build summary statistics with robust handling for edge cases:
      - 0 trades
      - all winners / all losers
      - zero init_cash (avoid division by zero)
    """
    total_trades = len(trades_df)

    if total_trades == 0:
        return {
            "total_return": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_trade": 0.0,
            "expectancy": 0.0,
        }

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]
    win_count = len(wins)
    loss_count = len(losses)

    gross_profit = float(wins["pnl"].sum()) if win_count > 0 else 0.0
    gross_loss = float(abs(losses["pnl"].sum())) if loss_count > 0 else 0.0

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = None  # all winners, undefined
    else:
        profit_factor = 0.0

    win_rate = win_count / total_trades
    avg_trade = float(trades_df["pnl"].mean())

    avg_win = float(wins["pnl"].mean()) if win_count > 0 else 0.0
    avg_loss = float(losses["pnl"].mean()) if loss_count > 0 else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    # Total return
    if init_cash > 0:
        total_return = (equity.iloc[-1] / init_cash - 1) * 100
    else:
        total_return = 0.0

    # Max drawdown: protect against non-positive peaks
    peak = equity.cummax()
    safe_peak = peak.where(peak > 0, pd.NA)
    drawdown = (equity - safe_peak) / safe_peak
    if drawdown.isna().all():
        max_drawdown = 0.0
    else:
        max_drawdown = float(drawdown.min() * 100)

    return {
        "total_return": total_return,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "avg_trade": avg_trade,
        "expectancy": expectancy,
    }
