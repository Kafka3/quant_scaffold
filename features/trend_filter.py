import pandas as pd
from features.indicators import ema


def build_trend_filter(df: pd.DataFrame, config: dict) -> dict[str, pd.Series]:
    period = config["ema_period"]
    ema_high = ema(df["High"], period)
    ema_low = ema(df["Low"], period)

    trend_long = df["Close"] > ema_high
    trend_short = df["Close"] < ema_low

    return {
        "ema_high": ema_high,
        "ema_low": ema_low,
        "trend_long": trend_long.fillna(False),
        "trend_short": trend_short.fillna(False),
    }
