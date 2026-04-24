from dataclasses import dataclass
from typing import Optional

import pandas as pd

from features.indicators import stochastic_d, atr
from features.divergence import detect_regular_divergence
from features.trend_filter import build_trend_filter


@dataclass
class PendingSetup:
    setup_bar_idx: object
    pivot2_idx: object
    trigger_price: float
    stop_anchor: float
    bars_waited: int = 0


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
    long_setup_pivot2_time: pd.Series
    short_setup_pivot2_time: pd.Series
    long_trigger_price_raw: pd.Series
    short_trigger_price_raw: pd.Series
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
    setup_cfg = config.get("setup", {})
    setup_max_bars = setup_cfg.get("setup_max_bars", 12)
    replace_same_side_setup = setup_cfg.get("replace_same_side_setup", True)
    invalidate_on_stop_anchor_break = setup_cfg.get("invalidate_on_stop_anchor_break", True)

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
    long_setup_pivot2_time = pd.Series(index=df.index, dtype=object)
    short_setup_pivot2_time = pd.Series(index=df.index, dtype=object)
    long_trigger_price_raw = pd.Series(index=df.index, dtype=float)
    short_trigger_price_raw = pd.Series(index=df.index, dtype=float)

    pending_long: Optional[PendingSetup] = None
    pending_short: Optional[PendingSetup] = None

    bullish_setup_active = []
    bearish_setup_active = []
    bullish_setup_trigger = []
    bearish_setup_trigger = []
    bullish_setup_stop_anchor = []
    bearish_setup_stop_anchor = []
    bullish_setup_pivot2_pos = []
    bearish_setup_pivot2_pos = []

    for idx in df.index:
        # First process any pending setup from earlier divergence bars.
        # setup only becomes eligible to trigger from the next bar after pivot2.
        if pending_long is not None:
            if invalidate_on_stop_anchor_break and df.loc[idx, "Low"] < pending_long.stop_anchor:
                # stop_anchor 被破坏则 setup 作废
                pending_long = None
            elif idx > pending_long.setup_bar_idx and df.loc[idx, "High"] > pending_long.trigger_price:
                entries_long.loc[idx] = True
                long_entry_price.loc[idx] = pending_long.trigger_price
                long_stop_price.loc[idx] = pending_long.stop_anchor - risk_cfg["stop_buffer"]
                r = long_entry_price.loc[idx] - long_stop_price.loc[idx]
                long_target_price.loc[idx] = long_entry_price.loc[idx] + risk_cfg["rr_target"] * r
                long_setup_pivot2_time.loc[idx] = pending_long.pivot2_idx
                long_trigger_price_raw.loc[idx] = pending_long.trigger_price
                pending_long = None
            else:
                if idx > pending_long.setup_bar_idx:
                    pending_long.bars_waited += 1
                # setup 最多等待多少根 bar
                if pending_long.bars_waited > setup_max_bars:
                    pending_long = None

        if pending_short is not None:
            if invalidate_on_stop_anchor_break and df.loc[idx, "High"] > pending_short.stop_anchor:
                # stop_anchor 被破坏则 setup 作废
                pending_short = None
            elif idx > pending_short.setup_bar_idx and df.loc[idx, "Low"] < pending_short.trigger_price:
                entries_short.loc[idx] = True
                short_entry_price.loc[idx] = pending_short.trigger_price
                short_stop_price.loc[idx] = pending_short.stop_anchor + risk_cfg["stop_buffer"]
                r = short_stop_price.loc[idx] - short_entry_price.loc[idx]
                short_target_price.loc[idx] = short_entry_price.loc[idx] - risk_cfg["rr_target"] * r
                short_setup_pivot2_time.loc[idx] = pending_short.pivot2_idx
                short_trigger_price_raw.loc[idx] = pending_short.trigger_price
                pending_short = None
            else:
                if idx > pending_short.setup_bar_idx:
                    pending_short.bars_waited += 1
                # setup 最多等待多少根 bar
                if pending_short.bars_waited > setup_max_bars:
                    pending_short = None

        # Then register new setups on the current bar if divergence and trend are valid.
        if div.bullish.loc[idx] and trend["trend_long"].loc[idx]:
            trigger_price = div.bullish_trigger_price.loc[idx]
            stop_anchor = div.bullish_stop_anchor.loc[idx]
            if not pd.isna(trigger_price) and not pd.isna(stop_anchor):
                # bullish divergence only creates a setup; it does not trigger on the same bar.
                if replace_same_side_setup or pending_long is None:
                    pending_long = PendingSetup(
                        setup_bar_idx=idx,
                        pivot2_idx=idx,
                        trigger_price=trigger_price,
                        stop_anchor=stop_anchor,
                    )

        if div.bearish.loc[idx] and trend["trend_short"].loc[idx]:
            trigger_price = div.bearish_trigger_price.loc[idx]
            stop_anchor = div.bearish_stop_anchor.loc[idx]
            if not pd.isna(trigger_price) and not pd.isna(stop_anchor):
                # bearish divergence only creates a setup; it does not trigger on the same bar.
                if replace_same_side_setup or pending_short is None:
                    pending_short = PendingSetup(
                        setup_bar_idx=idx,
                        pivot2_idx=idx,
                        trigger_price=trigger_price,
                        stop_anchor=stop_anchor,
                    )

        bullish_setup_active.append(pending_long is not None)
        bearish_setup_active.append(pending_short is not None)
        bullish_setup_trigger.append(pending_long.trigger_price if pending_long is not None else pd.NA)
        bearish_setup_trigger.append(pending_short.trigger_price if pending_short is not None else pd.NA)
        bullish_setup_stop_anchor.append(pending_long.stop_anchor if pending_long is not None else pd.NA)
        bearish_setup_stop_anchor.append(pending_short.stop_anchor if pending_short is not None else pd.NA)
        bullish_setup_pivot2_pos.append(pending_long.pivot2_idx if pending_long is not None else pd.NA)
        bearish_setup_pivot2_pos.append(pending_short.pivot2_idx if pending_short is not None else pd.NA)

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
            "bullish_setup_active": bullish_setup_active,
            "bearish_setup_active": bearish_setup_active,
            "bullish_setup_trigger": bullish_setup_trigger,
            "bearish_setup_trigger": bearish_setup_trigger,
            "bullish_setup_stop_anchor": bullish_setup_stop_anchor,
            "bearish_setup_stop_anchor": bearish_setup_stop_anchor,
            "bullish_setup_pivot2_pos": bullish_setup_pivot2_pos,
            "bearish_setup_pivot2_pos": bearish_setup_pivot2_pos,
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
        long_setup_pivot2_time=long_setup_pivot2_time,
        short_setup_pivot2_time=short_setup_pivot2_time,
        long_trigger_price_raw=long_trigger_price_raw,
        short_trigger_price_raw=short_trigger_price_raw,
        features=features,
        entry_prices_long=entry_prices_long,
        entry_prices_short=entry_prices_short,
        stop_prices_long=stop_prices_long,
        stop_prices_short=stop_prices_short,
        target_prices_long=target_prices_long,
        target_prices_short=target_prices_short,
    )
