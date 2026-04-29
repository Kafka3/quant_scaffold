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


def _run_event_driven_loop(
    df: pd.DataFrame,
    bundle: SignalBundle,
    allow_short: bool,
    initial_cash: float,
    *,
    try_enter: callable,
    finalize_exit: callable,
    finalize_eod: callable,
    compute_equity: callable,
    skipped: Optional[list] = None,
) -> Tuple[List[dict], List[float], List[str]]:
    """
    Shared event-driven loop used by both run_backtest and
    run_backtest_with_position_sizing_and_costs.

    Callbacks:
      - try_enter(idx, side, cash) -> (trade_dict, skip_info)
          trade_dict is None if no entry; skip_info is None unless skipped.
      - finalize_exit(trade, idx, exit_price_raw, exit_reason, cash) -> new_cash
      - finalize_eod(trade, last_idx, last_close, cash) -> new_cash
      - compute_equity(position, current_trade, cash, close) -> float
    """
    cash = initial_cash
    position: str = "flat"
    current_trade: Optional[dict] = None
    equity_values: List[float] = []
    trades: List[dict] = []
    warnings: List[str] = []

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
            short_entry = bool(bundle.entries_short.loc[idx]) and allow_short

            if long_entry and short_entry:
                warnings.append(f"ambiguous long/short entry on {idx}, skip this bar")
            elif long_entry:
                trade, skip_info = try_enter(idx, "long", cash)
                if skip_info is not None and skipped is not None:
                    skipped.append(skip_info)
                if trade is not None:
                    current_trade = trade
                    position = "long"
                    bar_entered = True
            elif short_entry:
                trade, skip_info = try_enter(idx, "short", cash)
                if skip_info is not None and skipped is not None:
                    skipped.append(skip_info)
                if trade is not None:
                    current_trade = trade
                    position = "short"
                    bar_entered = True

        # ------------------------------------------------------------------
        # 2. Same-bar exit check (immediately after entry)
        #    Priority: stop > target
        # ------------------------------------------------------------------
        if bar_entered and current_trade is not None:
            exit_price_raw, exit_reason = _check_exit(current_trade, high, low)
            if exit_price_raw is not None:
                cash = finalize_exit(current_trade, idx, exit_price_raw, exit_reason, cash)
                trades.append(current_trade)
                current_trade = None
                position = "flat"

        # ------------------------------------------------------------------
        # 3. Subsequent-bar exit check
        #    Priority: stop > target
        # ------------------------------------------------------------------
        if not bar_entered and current_trade is not None:
            exit_price_raw, exit_reason = _check_exit(current_trade, high, low)
            if exit_price_raw is not None:
                cash = finalize_exit(current_trade, idx, exit_price_raw, exit_reason, cash)
                trades.append(current_trade)
                current_trade = None
                position = "flat"

        # ------------------------------------------------------------------
        # 4. bars_held counting (entry bar itself is NOT counted)
        # ------------------------------------------------------------------
        if current_trade is not None and idx != current_trade["entry_time"]:
            current_trade["bars_held"] += 1

        # ------------------------------------------------------------------
        # 5. Mark-to-market equity (using raw close, no slippage)
        # ------------------------------------------------------------------
        equity_values.append(compute_equity(position, current_trade, cash, close))

    # ------------------------------------------------------------------
    # 6. End-of-data forced liquidation
    # ------------------------------------------------------------------
    if current_trade is not None:
        last_close = float(df["Close"].iloc[-1])
        last_idx = df.index[-1]
        cash = finalize_eod(current_trade, last_idx, last_close, cash)
        trades.append(current_trade)
        current_trade = None
        position = "flat"
        if equity_values:
            equity_values[-1] = cash

    return trades, equity_values, warnings


def run_backtest(df: pd.DataFrame, bundle: SignalBundle, config: dict) -> BacktestResult:
    """
    Custom event-driven backtester.

    Rules:
      - Flat only entry.
      - If long & short signal on same bar → skip, record warning.
      - Same-bar exit checked immediately after entry.
      - Stop takes priority over target on same-bar collisions.
      - Subsequent bars: stop > target priority maintained.
      - End-of-data forced liquidation if still in position.
    """
    initial_cash = float(config.get("initial_cash", config.get("init_cash", 100000)))
    fee_per_trade = float(config.get("fee_per_trade", 0.0))
    slippage = float(config.get("slippage", 0.0))
    allow_short = bool(config.get("allow_short", True))

    def try_enter(idx, side, cash):
        if side == "long":
            entry_price_raw = bundle.long_entry_price.loc[idx]
            stop_price = bundle.long_stop_price.loc[idx]
            target_price = bundle.long_target_price.loc[idx]
            if pd.notna(entry_price_raw) and pd.notna(stop_price) and pd.notna(target_price):
                entry_price = float(entry_price_raw) + slippage
                return {
                    "entry_time": idx,
                    "side": "long",
                    "entry_price": entry_price,
                    "stop_price": float(stop_price),
                    "target_price": float(target_price),
                    "setup_pivot2_time": bundle.long_setup_pivot2_time.loc[idx],
                    "setup_confirm_time": bundle.long_setup_confirm_time.loc[idx],
                    "trigger_price": bundle.long_trigger_price_raw.loc[idx],
                    "bars_held": 0,
                }, None
        else:  # short
            entry_price_raw = bundle.short_entry_price.loc[idx]
            stop_price = bundle.short_stop_price.loc[idx]
            target_price = bundle.short_target_price.loc[idx]
            if pd.notna(entry_price_raw) and pd.notna(stop_price) and pd.notna(target_price):
                entry_price = float(entry_price_raw) - slippage
                return {
                    "entry_time": idx,
                    "side": "short",
                    "entry_price": entry_price,
                    "stop_price": float(stop_price),
                    "target_price": float(target_price),
                    "setup_pivot2_time": bundle.short_setup_pivot2_time.loc[idx],
                    "setup_confirm_time": bundle.short_setup_confirm_time.loc[idx],
                    "trigger_price": bundle.short_trigger_price_raw.loc[idx],
                    "bars_held": 0,
                }, None
        return None, None

    def finalize_exit(trade, idx, exit_price_raw, exit_reason, cash):
        exit_price = exit_price_raw - slippage if trade["side"] == "long" else exit_price_raw + slippage
        pnl = _finalize_trade(trade, idx, exit_price, exit_reason, fee_per_trade)
        return cash + pnl

    def finalize_eod(trade, last_idx, last_close, cash):
        exit_price = last_close - slippage if trade["side"] == "long" else last_close + slippage
        pnl = _finalize_trade(trade, last_idx, exit_price, "end_of_data", fee_per_trade)
        return cash + pnl

    def compute_equity(position, current_trade, cash, close):
        if position == "flat" or current_trade is None:
            return cash
        elif position == "long":
            return cash + (close - current_trade["entry_price"])
        else:  # short
            return cash + (current_trade["entry_price"] - close)

    trades, equity_values, warnings = _run_event_driven_loop(
        df, bundle, allow_short, initial_cash,
        try_enter=try_enter,
        finalize_exit=finalize_exit,
        finalize_eod=finalize_eod,
        compute_equity=compute_equity,
    )

    # ------------------------------------------------------------------
    # Assemble results
    # ------------------------------------------------------------------
    equity = pd.Series(equity_values, index=df.index)

    trade_columns = [
        "entry_time",
        "exit_time",
        "side",
        "setup_pivot2_time",
        "setup_confirm_time",
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

    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df = trades_df[[c for c in trade_columns if c in trades_df.columns]]
    else:
        trades_df = pd.DataFrame(columns=trade_columns)

    summary = _build_summary(trades_df, equity, initial_cash)

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
    Returns raw exit price (slippage applied by caller).
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


def _finalize_trade(trade: dict, exit_time, exit_price: float, exit_reason: str, fee_per_trade: float) -> float:
    """
    Populate exit fields on the trade dict and return the PnL.
    PnL is net of slippage (already baked into entry/exit prices) and fees.
    """
    if trade["side"] == "long":
        pnl = exit_price - trade["entry_price"]
    else:
        pnl = trade["entry_price"] - exit_price

    pnl -= 2 * fee_per_trade

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


def _build_summary(trades_df: pd.DataFrame, equity: pd.Series, initial_cash: float) -> dict:
    """
    Build summary statistics with robust handling for edge cases:
      - 0 trades
      - all winners / all losers
      - zero initial_cash (avoid division by zero)
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
    if initial_cash > 0:
        total_return = (equity.iloc[-1] / initial_cash - 1) * 100
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


# =====================================================================
# Phase 4 — Position sizing and cost-aware backtester
# =====================================================================

from dataclasses import dataclass as _dc
from typing import Dict, Any

from backtest.risk_model import calculate_position_size
from backtest.cost_model import apply_slippage, calculate_fees, calculate_slippage_cost
from backtest.performance_metrics import calculate_sharpe_ratio


@_dc
class PositionSizingBacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    summary: dict
    skipped: pd.DataFrame
    warnings: List[str]


def run_backtest_with_position_sizing_and_costs(
    df: pd.DataFrame,
    bundle: SignalBundle,
    strategy_config: dict,
    risk_cost_config: dict,
    risk_per_trade_pct: float,
    position_mode: str = "capped",
) -> PositionSizingBacktestResult:
    """
    Event-driven backtester with position sizing and cost model.

    Preserves all entry/exit logic from run_backtest, but adds:
      - Risk-based position sizing
      - Slippage and fee model
      - Full trade record with qty, notional, costs, R-multiple
    """
    account_cfg = risk_cost_config.get("account", {})
    risk_cfg = risk_cost_config.get("risk", {})
    cost_cfg = risk_cost_config.get("cost", {})
    exec_cfg = risk_cost_config.get("execution", {})

    initial_cash = float(account_cfg.get("initial_cash", 100000))
    allow_short = bool(exec_cfg.get("allow_short", True))
    same_bar_stop_first = bool(exec_cfg.get("same_bar_stop_first", True))

    max_position_value_pct = float(risk_cfg.get("max_position_value_pct", 1.0))
    max_leverage = float(risk_cfg.get("max_leverage", 1.0))
    min_qty = float(risk_cfg.get("min_qty", 0.0001))
    qty_step = float(risk_cfg.get("qty_step", 0.0001))

    fee_rate = float(cost_cfg.get("fee_rate", 0.0))
    fixed_fee_per_trade = float(cost_cfg.get("fixed_fee_per_trade", 0.0))
    slippage = float(cost_cfg.get("slippage_per_side", 0.0))
    slippage_is_rate = bool(cost_cfg.get("slippage_is_rate", False))

    metrics_cfg = risk_cost_config.get("metrics", {})
    timeframe_minutes = int(metrics_cfg.get("timeframe_minutes", 5))
    risk_free_rate_annual = float(metrics_cfg.get("risk_free_rate_annual", 0.0))

    skipped: List[dict] = []

    def try_enter(idx, side, cash):
        if side == "long":
            entry_price_raw = float(bundle.long_entry_price.loc[idx])
            stop_price = float(bundle.long_stop_price.loc[idx])
            target_price = float(bundle.long_target_price.loc[idx])
        else:  # short
            entry_price_raw = float(bundle.short_entry_price.loc[idx])
            stop_price = float(bundle.short_stop_price.loc[idx])
            target_price = float(bundle.short_target_price.loc[idx])

        if pd.notna(entry_price_raw) and pd.notna(stop_price) and pd.notna(target_price):
            sizing = calculate_position_size(
                equity=cash,
                entry_price=entry_price_raw,
                stop_price=stop_price,
                risk_per_trade_pct=risk_per_trade_pct,
                max_position_value_pct=max_position_value_pct,
                max_leverage=max_leverage,
                min_qty=min_qty,
                qty_step=qty_step,
            )
            if sizing["skip_trade"]:
                return None, {
                    "time": idx,
                    "side": side,
                    "entry_price": entry_price_raw,
                    "stop_price": stop_price,
                    "skip_reason": sizing["skip_reason"],
                    "equity": cash,
                    "risk_per_trade_pct": risk_per_trade_pct,
                    "position_mode": position_mode,
                }
            else:
                entry_slippage = entry_price_raw * slippage if slippage_is_rate else slippage
                entry_filled = apply_slippage(entry_price_raw, side, "entry", entry_slippage)
                return {
                    "entry_time": idx,
                    "side": side,
                    "entry_price_raw": entry_price_raw,
                    "entry_price_filled": entry_filled,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "qty": sizing["qty"],
                    "notional": sizing["notional"],
                    "target_risk_amount": sizing["target_risk_amount"],
                    "actual_risk_amount": sizing["actual_risk_amount"],
                    "target_risk_pct": sizing["target_risk_pct"],
                    "actual_risk_pct": sizing["actual_risk_pct"],
                    "stop_distance": sizing["stop_distance"],
                    "raw_qty": sizing["raw_qty"],
                    "max_qty": sizing["max_qty"],
                    "cap_hit": sizing["cap_hit"],
                    "equity_before": cash,
                    "bars_held": 0,
                }, None
        return None, None

    def finalize_exit(trade, idx, exit_price_raw, exit_reason, cash):
        exit_slippage = exit_price_raw * slippage if slippage_is_rate else slippage
        _finalize_trade_ps(
            trade, idx, exit_price_raw, exit_reason,
            exit_slippage, fee_rate, fixed_fee_per_trade,
        )
        return trade["equity_after"]

    def finalize_eod(trade, last_idx, last_close, cash):
        exit_slippage = last_close * slippage if slippage_is_rate else slippage
        _finalize_trade_ps(
            trade, last_idx, last_close, "end_of_data",
            exit_slippage, fee_rate, fixed_fee_per_trade,
        )
        return trade["equity_after"]

    def compute_equity(position, current_trade, cash, close):
        if position == "flat" or current_trade is None:
            return cash
        elif position == "long":
            unrealized = (close - current_trade["entry_price_filled"]) * current_trade["qty"]
            return cash + unrealized
        else:  # short
            unrealized = (current_trade["entry_price_filled"] - close) * current_trade["qty"]
            return cash + unrealized

    trades, equity_values, warnings = _run_event_driven_loop(
        df, bundle, allow_short, initial_cash,
        try_enter=try_enter,
        finalize_exit=finalize_exit,
        finalize_eod=finalize_eod,
        compute_equity=compute_equity,
        skipped=skipped,
    )

    # ------------------------------------------------------------------
    # Assemble results
    # ------------------------------------------------------------------
    equity = pd.Series(equity_values, index=df.index)

    trade_columns = [
        "entry_time", "exit_time", "side",
        "entry_price_raw", "entry_price_filled",
        "exit_price_raw", "exit_price_filled",
        "qty", "notional",
        "target_risk_amount", "actual_risk_amount",
        "target_risk_pct", "actual_risk_pct",
        "stop_distance", "raw_qty", "max_qty", "cap_hit",
        "stop_price", "target_price",
        "gross_pnl", "fees", "slippage_cost", "net_pnl", "r_multiple",
        "equity_before", "equity_after",
        "exit_reason", "bars_held", "skip_reason",
    ]

    skipped_columns = [
        "time", "side", "entry_price", "stop_price",
        "skip_reason", "equity", "risk_per_trade_pct", "position_mode",
    ]

    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df = trades_df[[c for c in trade_columns if c in trades_df.columns]]
    else:
        trades_df = pd.DataFrame(columns=trade_columns)

    if skipped:
        skipped_df = pd.DataFrame(skipped)
        skipped_df = skipped_df[[c for c in skipped_columns if c in skipped_df.columns]]
    else:
        skipped_df = pd.DataFrame(columns=skipped_columns)

    summary = _build_summary_ps(
        trades_df, equity, initial_cash, skipped_df,
        timeframe_minutes, risk_free_rate_annual,
    )

    return PositionSizingBacktestResult(
        trades=trades_df,
        equity=equity,
        summary=summary,
        skipped=skipped_df,
        warnings=warnings,
    )


def _finalize_trade_ps(
    trade: dict,
    exit_time,
    exit_price_raw: float,
    exit_reason: str,
    slippage: float,
    fee_rate: float,
    fixed_fee_per_trade: float,
) -> None:
    """Finalize a trade with position sizing and cost model."""
    side = trade["side"]
    qty = trade["qty"]
    entry_raw = trade["entry_price_raw"]
    entry_filled = trade["entry_price_filled"]

    exit_filled = apply_slippage(exit_price_raw, side, "exit", slippage)

    if side == "long":
        gross_pnl = (exit_filled - entry_filled) * qty
    else:
        gross_pnl = (entry_filled - exit_filled) * qty

    fees = calculate_fees(entry_filled, exit_filled, qty, fee_rate, fixed_fee_per_trade)
    slippage_cost = calculate_slippage_cost(entry_raw, entry_filled, exit_price_raw, exit_filled, qty, side)
    net_pnl = gross_pnl - fees

    actual_risk_amount = trade.get("actual_risk_amount", 0.0)
    if actual_risk_amount > 0:
        r_multiple = net_pnl / actual_risk_amount
    else:
        r_multiple = float("nan")

    equity_after = trade["equity_before"] + net_pnl

    trade.update({
        "exit_time": exit_time,
        "exit_price_raw": exit_price_raw,
        "exit_price_filled": exit_filled,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "slippage_cost": slippage_cost,
        "net_pnl": net_pnl,
        "r_multiple": r_multiple,
        "equity_after": equity_after,
        "exit_reason": exit_reason,
    })


def _build_summary_ps(
    trades_df: pd.DataFrame,
    equity: pd.Series,
    initial_cash: float,
    skipped_df: pd.DataFrame,
    timeframe_minutes: int,
    risk_free_rate_annual: float,
) -> dict:
    """Build summary for position-sizing backtest."""
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
            "sharpe_ratio": float("nan"),
            "avg_r": 0.0,
            "median_r": 0.0,
            "total_r": 0.0,
            "expectancy_r": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "long_pnl": 0.0,
            "short_pnl": 0.0,
            "total_fees": 0.0,
            "total_slippage_cost": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "cost_as_pct_of_gross_profit": 0.0,
            "avg_qty": 0.0,
            "avg_notional": 0.0,
            "max_notional": 0.0,
            "avg_notional_pct": 0.0,
            "max_notional_pct": 0.0,
            "skipped_trades": len(skipped_df),
            "cap_hit_count": 0,
            "cap_hit_rate": 0.0,
            "avg_actual_risk_pct": 0.0,
            "max_actual_risk_pct": 0.0,
            "avg_target_risk_pct": 0.0,
            "max_target_risk_pct": 0.0,
        }

    wins = trades_df[trades_df["net_pnl"] > 0]
    losses = trades_df[trades_df["net_pnl"] < 0]
    win_count = len(wins)
    loss_count = len(losses)

    gross_profit = float(wins["net_pnl"].sum()) if win_count > 0 else 0.0
    gross_loss = float(abs(losses["net_pnl"].sum())) if loss_count > 0 else 0.0

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    win_rate = win_count / total_trades
    avg_trade = float(trades_df["net_pnl"].mean())

    avg_win = float(wins["net_pnl"].mean()) if win_count > 0 else 0.0
    avg_loss = float(losses["net_pnl"].mean()) if loss_count > 0 else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    # R-multiple stats
    r_values = trades_df["r_multiple"].dropna()
    avg_r = float(r_values.mean()) if len(r_values) > 0 else 0.0
    median_r = float(r_values.median()) if len(r_values) > 0 else 0.0
    total_r = float(r_values.sum()) if len(r_values) > 0 else 0.0

    r_wins = trades_df[trades_df["r_multiple"] > 0]["r_multiple"]
    r_losses = trades_df[trades_df["r_multiple"] < 0]["r_multiple"]
    r_win_rate = len(r_wins) / len(r_values) if len(r_values) > 0 else 0.0
    avg_r_win = float(r_wins.mean()) if len(r_wins) > 0 else 0.0
    avg_r_loss = float(r_losses.mean()) if len(r_losses) > 0 else 0.0
    expectancy_r = r_win_rate * avg_r_win + (1 - r_win_rate) * avg_r_loss

    # Total return
    if initial_cash > 0:
        total_return = (equity.iloc[-1] / initial_cash - 1) * 100
    else:
        total_return = 0.0

    # Max drawdown
    peak = equity.cummax()
    safe_peak = peak.where(peak > 0, pd.NA)
    drawdown = (equity - safe_peak) / safe_peak
    if drawdown.isna().all():
        max_drawdown = 0.0
    else:
        max_drawdown = float(drawdown.min() * 100)

    # Sharpe ratio
    sharpe = calculate_sharpe_ratio(equity, timeframe_minutes, risk_free_rate_annual)

    # Side breakdown
    long_trades = len(trades_df[trades_df["side"] == "long"])
    short_trades = len(trades_df[trades_df["side"] == "short"])
    long_pnl = float(trades_df[trades_df["side"] == "long"]["net_pnl"].sum())
    short_pnl = float(trades_df[trades_df["side"] == "short"]["net_pnl"].sum())

    # Costs
    total_fees = float(trades_df["fees"].sum())
    total_slippage_cost = float(trades_df["slippage_cost"].sum())
    total_gross = gross_profit + gross_loss
    cost_as_pct = (total_fees + total_slippage_cost) / gross_profit * 100 if gross_profit > 0 else 0.0

    # Notional stats
    avg_qty = float(trades_df["qty"].mean())
    avg_notional = float(trades_df["notional"].mean())
    max_notional = float(trades_df["notional"].max())
    avg_notional_pct = (avg_notional / initial_cash * 100) if initial_cash > 0 else 0.0
    max_notional_pct = (max_notional / initial_cash * 100) if initial_cash > 0 else 0.0

    return {
        "total_return": total_return,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown,
        "avg_trade": avg_trade,
        "expectancy": expectancy,
        "winning_trades": win_count,
        "losing_trades": loss_count,
        "sharpe_ratio": sharpe,
        "avg_r": avg_r,
        "median_r": median_r,
        "total_r": total_r,
        "expectancy_r": expectancy_r,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
        "total_fees": total_fees,
        "total_slippage_cost": total_slippage_cost,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "cost_as_pct_of_gross_profit": cost_as_pct,
        "avg_qty": avg_qty,
        "avg_notional": avg_notional,
        "max_notional": max_notional,
        "avg_notional_pct": avg_notional_pct,
        "max_notional_pct": max_notional_pct,
        "skipped_trades": len(skipped_df),
        "cap_hit_count": int(trades_df["cap_hit"].sum()) if "cap_hit" in trades_df.columns else 0,
        "cap_hit_rate": float(trades_df["cap_hit"].mean()) if "cap_hit" in trades_df.columns and len(trades_df) > 0 else 0.0,
        "avg_actual_risk_pct": float(trades_df["actual_risk_pct"].mean()) if "actual_risk_pct" in trades_df.columns and len(trades_df) > 0 else 0.0,
        "max_actual_risk_pct": float(trades_df["actual_risk_pct"].max()) if "actual_risk_pct" in trades_df.columns and len(trades_df) > 0 else 0.0,
        "avg_target_risk_pct": float(trades_df["target_risk_pct"].mean()) if "target_risk_pct" in trades_df.columns and len(trades_df) > 0 else 0.0,
        "max_target_risk_pct": float(trades_df["target_risk_pct"].max()) if "target_risk_pct" in trades_df.columns and len(trades_df) > 0 else 0.0,
    }
