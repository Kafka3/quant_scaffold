"""Performance metrics including Sharpe Ratio."""

import math
from typing import Optional

import numpy as np
import pandas as pd


def calculate_sharpe_ratio(
    equity: pd.Series,
    timeframe_minutes: int = 5,
    risk_free_rate_annual: float = 0.0,
) -> Optional[float]:
    """
    Calculate annualized Sharpe Ratio from equity curve.

    Uses bar-by-bar returns. Annualization assumes crypto 24/7 markets.
    """
    bars_per_year = int(365 * 24 * 60 / timeframe_minutes)

    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    if len(returns) < 2:
        return float("nan")

    risk_free_per_bar = risk_free_rate_annual / bars_per_year
    excess_returns = returns - risk_free_per_bar

    std = excess_returns.std(ddof=1)
    if pd.isna(std) or std == 0:
        return float("nan")

    sharpe = excess_returns.mean() / std * np.sqrt(bars_per_year)
    return float(sharpe)
