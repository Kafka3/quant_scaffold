import pandas as pd


def apply_allowed_regimes(signal: pd.Series, regimes: pd.Series, allowed: set[int]) -> pd.Series:
    return signal & regimes.isin(allowed)
