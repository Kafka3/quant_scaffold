import pandas as pd
from features.indicators import ema


def build_trend_filter(df: pd.DataFrame, config: dict) -> dict[str, pd.Series]:
    """
    Build trend filter with EMA channel and prior-trend state.

    Returns:
      - ema_high / ema_low: EMA of High/Low
      - above_channel / below_channel: current bar close strictly outside channel
      - inside_or_below_high: Close < ema_high (inside channel OR below)
      - inside_or_above_low: Close > ema_low (inside channel OR above)
      - above_ratio / below_ratio: proportion of recent bars above/below channel
      - prior_uptrend / prior_downtrend: trend state based on historical bars only
      - trend_long / trend_short: backward-compatible aliases (same as above/below_channel)

    Notes:
      - prior_uptrend is intended for bullish continuation divergence setups
        (looking for dips within an established uptrend).
      - prior_downtrend is intended for bearish continuation divergence setups
        (looking for rallies within an established downtrend).
    """
    period = config["ema_period"]
    lookback_bars = config.get("lookback_bars", 20)
    min_close_ratio = config.get("min_close_ratio", 0.6)

    ema_high = ema(df["High"], period)
    ema_low = ema(df["Low"], period)

    # Current bar channel position.
    above_channel = df["Close"] > ema_high
    below_channel = df["Close"] < ema_low

    # Progressive pullback channel checks.
    # pivot1 (first pullback) may only reach the channel boundary.
    # pivot2 (deep pullback) must break through the channel.
    inside_or_below_high = df["Close"] < ema_high
    inside_or_above_low = df["Close"] > ema_low

    # Prior trend uses only historical bars (shift(1) excludes current bar)
    # to avoid lookahead.
    above_ratio = above_channel.shift(1).rolling(lookback_bars).mean()
    below_ratio = below_channel.shift(1).rolling(lookback_bars).mean()

    prior_uptrend = above_ratio >= min_close_ratio
    prior_downtrend = below_ratio >= min_close_ratio

    # Backward-compatible aliases used by signal_builder.
    trend_long = above_channel
    trend_short = below_channel

    return {
        "ema_high": ema_high,
        "ema_low": ema_low,
        "above_channel": above_channel,
        "below_channel": below_channel,
        "inside_or_below_high": inside_or_below_high,
        "inside_or_above_low": inside_or_above_low,
        "above_ratio": above_ratio,
        "below_ratio": below_ratio,
        "prior_uptrend": prior_uptrend.fillna(False),
        "prior_downtrend": prior_downtrend.fillna(False),
        # backward-compatible aliases
        "trend_long": trend_long.fillna(False),
        "trend_short": trend_short.fillna(False),
    }
