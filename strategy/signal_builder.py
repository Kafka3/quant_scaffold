from dataclasses import dataclass
from typing import Optional

import pandas as pd

from features.indicators import stochastic_d, atr
from features.divergence import detect_regular_divergence
from features.trend_filter import build_trend_filter


@dataclass
class PendingSetup:
    pivot2_idx: object
    trigger_price: float
    stop_anchor: float


@dataclass
class SignalBundle:
    entries_long: pd.Series
    exits_long: pd.Series
    entries_short: pd.Series
    exits_short: pd.Series
    long_entry_price: pd.Series
    short_entry_price: pd.Series
    long_stop_price: pd.Series
    short_stop_price: pd.Series
    long_target_price: pd.Series
    short_target_price: pd.Series
    features: pd.DataFrame
    # backward-compatible aliases
    entry_prices_long: pd.Series
    entry_prices_short: pd.Series
    stop_prices_long: pd.Series
    stop_prices_short: pd.Series
    target_prices_long: pd.Series
    target_prices_short: pd.Series


def build_signals(df: pd.DataFrame, config: dict) -> SignalBundle:
    stoch_cfg = config["stochastic"]
    risk_cfg = config["risk"]

    osc = stochastic_d(
        df,
        k_period=stoch_cfg["k_period"],
        d_period=stoch_cfg["d_period"],
        smooth=stoch_cfg["smooth"],
    )
    div = detect_regular_divergence(df, osc, config)
    trend = build_trend_filter(df, config["trend"])
    current_atr = atr(df, risk_cfg["atr_period"])

    entries_long = pd.Series(False, index=df.index)
    entries_short = pd.Series(False, index=df.index)
    exits_long = pd.Series(False, index=df.index)
    exits_short = pd.Series(False, index=df.index)

    long_entry_price = pd.Series(index=df.index, dtype=float)
    short_entry_price = pd.Series(index=df.index, dtype=float)
    long_stop_price = pd.Series(index=df.index, dtype=float)
    short_stop_price = pd.Series(index=df.index, dtype=float)
    long_target_price = pd.Series(index=df.index, dtype=float)
    short_target_price = pd.Series(index=df.index, dtype=float)

    pending_long: Optional[PendingSetup] = None
    pending_short: Optional[PendingSetup] = None

    for idx in df.index:
        # First process any pending setup from earlier divergence bars.
        if pending_long is not None:
            if df.loc[idx, "High"] > pending_long.trigger_price:
                entries_long.loc[idx] = True
                long_entry_price.loc[idx] = pending_long.trigger_price
                long_stop_price.loc[idx] = pending_long.stop_anchor - risk_cfg["stop_buffer"]
                r = long_entry_price.loc[idx] - long_stop_price.loc[idx]
                long_target_price.loc[idx] = long_entry_price.loc[idx] + risk_cfg["rr_target"] * r
                pending_long = None

        if pending_short is not None:
            if df.loc[idx, "Low"] < pending_short.trigger_price:
                entries_short.loc[idx] = True
                short_entry_price.loc[idx] = pending_short.trigger_price
                short_stop_price.loc[idx] = pending_short.stop_anchor + risk_cfg["stop_buffer"]
                r = short_stop_price.loc[idx] - short_entry_price.loc[idx]
                short_target_price.loc[idx] = short_entry_price.loc[idx] - risk_cfg["rr_target"] * r
                pending_short = None

        # Then register new setups on the current bar if divergence and trend are valid.
        if div.bullish.loc[idx] and trend["trend_long"].loc[idx]:
            trigger_price = div.bullish_trigger_price.loc[idx]
            stop_anchor = div.bullish_stop_anchor.loc[idx]
            if not pd.isna(trigger_price) and not pd.isna(stop_anchor):
                # Newer bullish setup overrides any existing pending bullish setup.
                pending_long = PendingSetup(
                    pivot2_idx=idx,
                    trigger_price=trigger_price,
                    stop_anchor=stop_anchor,
                )

        if div.bearish.loc[idx] and trend["trend_short"].loc[idx]:
            trigger_price = div.bearish_trigger_price.loc[idx]
            stop_anchor = div.bearish_stop_anchor.loc[idx]
            if not pd.isna(trigger_price) and not pd.isna(stop_anchor):
                # Newer bearish setup overrides any existing pending bearish setup.
                pending_short = PendingSetup(
                    pivot2_idx=idx,
                    trigger_price=trigger_price,
                    stop_anchor=stop_anchor,
                )

    # Backward-compatible alias fields for existing custom backtest code.
    entry_prices_long = long_entry_price
    entry_prices_short = short_entry_price
    stop_prices_long = long_stop_price
    stop_prices_short = short_stop_price
    target_prices_long = long_target_price
    target_prices_short = short_target_price

    features = pd.DataFrame(
        {
            "Close": df["Close"],
            "osc": osc,
            "atr": current_atr,
            "ema_high": trend["ema_high"],
            "ema_low": trend["ema_low"],
            "trend_long": trend["trend_long"].astype(int),
            "trend_short": trend["trend_short"].astype(int),
            "pivot_high": div.pivot_high.astype(int),
            "pivot_low": div.pivot_low.astype(int),
            "bullish_div": div.bullish.astype(int),
            "bearish_div": div.bearish.astype(int),
            "bullish_trigger_price": div.bullish_trigger_price,
            "bearish_trigger_price": div.bearish_trigger_price,
            "bullish_stop_anchor": div.bullish_stop_anchor,
            "bearish_stop_anchor": div.bearish_stop_anchor,
            "long_entry_price": long_entry_price,
            "short_entry_price": short_entry_price,
            "long_stop_price": long_stop_price,
            "short_stop_price": short_stop_price,
            "long_target_price": long_target_price,
            "short_target_price": short_target_price,
        },
        index=df.index,
    )

    return SignalBundle(
        entries_long=entries_long.fillna(False),
        exits_long=exits_long.fillna(False),
        entries_short=entries_short.fillna(False),
        exits_short=exits_short.fillna(False),
        long_entry_price=long_entry_price,
        short_entry_price=short_entry_price,
        long_stop_price=long_stop_price,
        short_stop_price=short_stop_price,
        long_target_price=long_target_price,
        short_target_price=short_target_price,
        features=features,
        entry_prices_long=entry_prices_long,
        entry_prices_short=entry_prices_short,
        stop_prices_long=stop_prices_long,
        stop_prices_short=stop_prices_short,
        target_prices_long=target_prices_long,
        target_prices_short=target_prices_short,
    )
