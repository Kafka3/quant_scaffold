import pandas as pd


def structure_stop_long(df: pd.DataFrame, pivot_low: pd.Series, buffer: float = 0.0) -> pd.Series:
    last_pivot_low = df["Low"].where(pivot_low.astype(bool)).ffill()
    return last_pivot_low * (1 - buffer)


def structure_stop_short(df: pd.DataFrame, pivot_high: pd.Series, buffer: float = 0.0) -> pd.Series:
    last_pivot_high = df["High"].where(pivot_high.astype(bool)).ffill()
    return last_pivot_high * (1 + buffer)
