from dataclasses import dataclass
import pandas as pd


@dataclass
class DivergenceSignals:
    bullish: pd.Series
    bearish: pd.Series
    pivot_high: pd.Series
    pivot_low: pd.Series
    pivot1_high: pd.Series  # First pivot high for bearish divergence
    pivot2_high: pd.Series  # Second pivot high for bearish divergence
    pivot1_low: pd.Series   # First pivot low for bullish divergence
    pivot2_low: pd.Series   # Second pivot low for bullish divergence


def _pivot_high(series: pd.Series, left: int, right: int) -> pd.Series:
    out = pd.Series(False, index=series.index)
    for i in range(left, len(series) - right):
        window = series.iloc[i - left : i + right + 1]
        out.iloc[i] = series.iloc[i] == window.max()
    return out


def _pivot_low(series: pd.Series, left: int, right: int) -> pd.Series:
    out = pd.Series(False, index=series.index)
    for i in range(left, len(series) - right):
        window = series.iloc[i - left : i + right + 1]
        out.iloc[i] = series.iloc[i] == window.min()
    return out


def detect_regular_divergence(
    price: pd.Series,
    osc: pd.Series,
    left_bars: int,
    right_bars: int,
    min_separation: int,
    max_separation: int,
    overbought: float = 80,
    oversold: float = 20,
) -> DivergenceSignals:
    pivot_high = _pivot_high(price, left_bars, right_bars)
    pivot_low = _pivot_low(price, left_bars, right_bars)

    bullish = pd.Series(False, index=price.index)
    bearish = pd.Series(False, index=price.index)
    pivot1_high = pd.Series(index=price.index, dtype=object)
    pivot2_high = pd.Series(index=price.index, dtype=object)
    pivot1_low = pd.Series(index=price.index, dtype=object)
    pivot2_low = pd.Series(index=price.index, dtype=object)

    low_idx = list(price.index[pivot_low])
    for i in range(1, len(low_idx)):
        idx1, idx2 = low_idx[i - 1], low_idx[i]
        pos1, pos2 = price.index.get_loc(idx1), price.index.get_loc(idx2)
        sep = pos2 - pos1
        if min_separation <= sep <= max_separation:
            if price.loc[idx2] < price.loc[idx1] and osc.loc[idx2] > osc.loc[idx1] and osc.loc[idx1] <= oversold:
                bullish.loc[idx2] = True
                pivot1_low.loc[idx2] = {'price': price.loc[idx1], 'osc': osc.loc[idx1], 'pos': pos1}
                pivot2_low.loc[idx2] = {'price': price.loc[idx2], 'osc': osc.loc[idx2], 'pos': pos2}

    high_idx = list(price.index[pivot_high])
    for i in range(1, len(high_idx)):
        idx1, idx2 = high_idx[i - 1], high_idx[i]
        pos1, pos2 = price.index.get_loc(idx1), price.index.get_loc(idx2)
        sep = pos2 - pos1
        if min_separation <= sep <= max_separation:
            if price.loc[idx2] > price.loc[idx1] and osc.loc[idx2] < osc.loc[idx1] and osc.loc[idx1] >= overbought:
                bearish.loc[idx2] = True
                pivot1_high.loc[idx2] = {'price': price.loc[idx1], 'osc': osc.loc[idx1], 'pos': pos1}
                pivot2_high.loc[idx2] = {'price': price.loc[idx2], 'osc': osc.loc[idx2], 'pos': pos2}

    return DivergenceSignals(
        bullish=bullish,
        bearish=bearish,
        pivot_high=pivot_high,
        pivot_low=pivot_low,
        pivot1_high=pivot1_high,
        pivot2_high=pivot2_high,
        pivot1_low=pivot1_low,
        pivot2_low=pivot2_low,
    )
