from dataclasses import dataclass
import pandas as pd

from features.indicators import stochastic_d, atr
from features.divergence import detect_regular_divergence
from features.trend_filter import build_trend_filter


@dataclass
class SignalBundle:
    entries_long: pd.Series
    exits_long: pd.Series
    entries_short: pd.Series
    exits_short: pd.Series
    features: pd.DataFrame
    # For custom backtester
    entry_prices_long: pd.Series
    entry_prices_short: pd.Series
    stop_prices_long: pd.Series
    stop_prices_short: pd.Series
    target_prices_long: pd.Series
    target_prices_short: pd.Series


def build_signals(df: pd.DataFrame, config: dict) -> SignalBundle:
    stoch_cfg = config["stochastic"]
    piv_cfg = config["pivots"]
    risk_cfg = config["risk"]

    osc = stochastic_d(
        df,
        k_period=stoch_cfg["k_period"],
        d_period=stoch_cfg["d_period"],
        smooth=stoch_cfg["smooth"],
    )
    div = detect_regular_divergence(
        price=df["Close"],
        osc=osc,
        left_bars=piv_cfg["left_bars"],
        right_bars=piv_cfg["right_bars"],
        min_separation=piv_cfg["min_separation"],
        max_separation=piv_cfg["max_separation"],
        overbought=stoch_cfg["overbought"],
        oversold=stoch_cfg["oversold"],
    )
    trend = build_trend_filter(df, config["trend"])

    # Entries: divergence + trend + break of pivot2
    bullish_div = div.bullish & trend["trend_long"]
    bearish_div = div.bearish & trend["trend_short"]

    # Entry triggers: break of pivot2 high/low
    entries_long = pd.Series(False, index=df.index)
    entries_short = pd.Series(False, index=df.index)
    entry_prices_long = pd.Series(index=df.index, dtype=float)
    entry_prices_short = pd.Series(index=df.index, dtype=float)

    for idx in df.index:
        if bullish_div.loc[idx]:
            pivot2_price = div.pivot2_low.loc[idx]['price'] if div.pivot2_low.loc[idx] else None
            if pivot2_price:
                # Trigger on break of pivot2 low's high (simplified)
                trigger_price = df.loc[idx, 'High']  # Use the bar's high as trigger
                if df.loc[idx, 'Close'] > pivot2_price:
                    entries_long.loc[idx] = True
                    entry_prices_long.loc[idx] = trigger_price

        if bearish_div.loc[idx]:
            pivot2_price = div.pivot2_high.loc[idx]['price'] if div.pivot2_high.loc[idx] else None
            if pivot2_price:
                trigger_price = df.loc[idx, 'Low']  # Use the bar's low as trigger
                if df.loc[idx, 'Close'] < pivot2_price:
                    entries_short.loc[idx] = True
                    entry_prices_short.loc[idx] = trigger_price

    # Stops and targets
    stop_prices_long = pd.Series(index=df.index, dtype=float)
    stop_prices_short = pd.Series(index=df.index, dtype=float)
    target_prices_long = pd.Series(index=df.index, dtype=float)
    target_prices_short = pd.Series(index=df.index, dtype=float)

    for idx in df.index:
        if entries_long.loc[idx]:
            pivot2_price = div.pivot2_low.loc[idx]['price']
            stop_prices_long.loc[idx] = pivot2_price - risk_cfg["stop_buffer"]
            r = entry_prices_long.loc[idx] - stop_prices_long.loc[idx]
            target_prices_long.loc[idx] = entry_prices_long.loc[idx] + risk_cfg["rr_target"] * r

        if entries_short.loc[idx]:
            pivot2_price = div.pivot2_high.loc[idx]['price']
            stop_prices_short.loc[idx] = pivot2_price + risk_cfg["stop_buffer"]
            r = stop_prices_short.loc[idx] - entry_prices_short.loc[idx]
            target_prices_short.loc[idx] = entry_prices_short.loc[idx] - risk_cfg["rr_target"] * r

    # Placeholder exits for vectorbt compatibility
    exits_long = entries_short  # Simplified
    exits_short = entries_long

    features = pd.DataFrame({
        "close": df["Close"],
        "osc": osc,
        "pivot_high": div.pivot_high.astype(int),
        "pivot_low": div.pivot_low.astype(int),
        "bullish_div": div.bullish.astype(int),
        "bearish_div": div.bearish.astype(int),
        "ema_high": trend["ema_high"],
        "ema_low": trend["ema_low"],
        "trend_long": trend["trend_long"].astype(int),
        "trend_short": trend["trend_short"].astype(int),
    }, index=df.index)

    return SignalBundle(
        entries_long=entries_long.fillna(False),
        exits_long=exits_long.fillna(False),
        entries_short=entries_short.fillna(False),
        exits_short=exits_short.fillna(False),
        features=features,
        entry_prices_long=entry_prices_long,
        entry_prices_short=entry_prices_short,
        stop_prices_long=stop_prices_long,
        stop_prices_short=stop_prices_short,
        target_prices_long=target_prices_long,
        target_prices_short=target_prices_short,
    )
