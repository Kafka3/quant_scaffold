from dataclasses import dataclass
import pandas as pd


@dataclass
class DivergenceResult:
    bullish: pd.Series
    bearish: pd.Series
    pivot_high: pd.Series
    pivot_low: pd.Series
    bullish_pivot1_idx: pd.Series
    bullish_pivot2_idx: pd.Series
    bearish_pivot1_idx: pd.Series
    bearish_pivot2_idx: pd.Series
    bullish_pivot1_price: pd.Series
    bullish_pivot2_price: pd.Series
    bearish_pivot1_price: pd.Series
    bearish_pivot2_price: pd.Series
    bullish_trigger_price: pd.Series
    bearish_trigger_price: pd.Series
    bullish_stop_anchor: pd.Series
    bearish_stop_anchor: pd.Series


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


def detect_regular_divergence(df: pd.DataFrame, osc: pd.Series, config: dict) -> DivergenceResult:
    """
    Detect regular bullish / bearish divergence using high/low pivots.

    Note: 背离成立 bar != 实际入场 bar，入场要等后续突破触发。
    """
    high = df["High"]
    low = df["Low"]

    piv_cfg = config["pivots"]
    stoch_cfg = config["stochastic"]

    pivot_high = _pivot_high(high, piv_cfg["left_bars"], piv_cfg["right_bars"])
    pivot_low = _pivot_low(low, piv_cfg["left_bars"], piv_cfg["right_bars"])

    bullish = pd.Series(False, index=df.index)
    bearish = pd.Series(False, index=df.index)
    bullish_pivot1_idx = pd.Series(index=df.index, dtype=object)
    bullish_pivot2_idx = pd.Series(index=df.index, dtype=object)
    bearish_pivot1_idx = pd.Series(index=df.index, dtype=object)
    bearish_pivot2_idx = pd.Series(index=df.index, dtype=object)
    bullish_pivot1_price = pd.Series(index=df.index, dtype=float)
    bullish_pivot2_price = pd.Series(index=df.index, dtype=float)
    bearish_pivot1_price = pd.Series(index=df.index, dtype=float)
    bearish_pivot2_price = pd.Series(index=df.index, dtype=float)
    bullish_trigger_price = pd.Series(index=df.index, dtype=float)
    bearish_trigger_price = pd.Series(index=df.index, dtype=float)
    bullish_stop_anchor = pd.Series(index=df.index, dtype=float)
    bearish_stop_anchor = pd.Series(index=df.index, dtype=float)

    low_idx = list(df.index[pivot_low])
    for i in range(1, len(low_idx)):
        idx1, idx2 = low_idx[i - 1], low_idx[i]
        pos1, pos2 = df.index.get_loc(idx1), df.index.get_loc(idx2)
        sep = pos2 - pos1
        if piv_cfg["min_separation"] <= sep <= piv_cfg["max_separation"]:
            if low.loc[idx2] < low.loc[idx1] and osc.loc[idx2] > osc.loc[idx1] and osc.loc[idx1] <= stoch_cfg["oversold"]:
                bullish.loc[idx2] = True
                bullish_pivot1_idx.loc[idx2] = idx1
                bullish_pivot2_idx.loc[idx2] = idx2
                bullish_pivot1_price.loc[idx2] = low.loc[idx1]
                bullish_pivot2_price.loc[idx2] = low.loc[idx2]
                bullish_trigger_price.loc[idx2] = high.loc[idx2]
                bullish_stop_anchor.loc[idx2] = low.loc[idx2]

    high_idx = list(df.index[pivot_high])
    for i in range(1, len(high_idx)):
        idx1, idx2 = high_idx[i - 1], high_idx[i]
        pos1, pos2 = df.index.get_loc(idx1), df.index.get_loc(idx2)
        sep = pos2 - pos1
        if piv_cfg["min_separation"] <= sep <= piv_cfg["max_separation"]:
            if high.loc[idx2] > high.loc[idx1] and osc.loc[idx2] < osc.loc[idx1] and osc.loc[idx1] >= stoch_cfg["overbought"]:
                bearish.loc[idx2] = True
                bearish_pivot1_idx.loc[idx2] = idx1
                bearish_pivot2_idx.loc[idx2] = idx2
                bearish_pivot1_price.loc[idx2] = high.loc[idx1]
                bearish_pivot2_price.loc[idx2] = high.loc[idx2]
                bearish_trigger_price.loc[idx2] = low.loc[idx2]
                bearish_stop_anchor.loc[idx2] = high.loc[idx2]

    return DivergenceResult(
        bullish=bullish,
        bearish=bearish,
        pivot_high=pivot_high,
        pivot_low=pivot_low,
        bullish_pivot1_idx=bullish_pivot1_idx,
        bullish_pivot2_idx=bullish_pivot2_idx,
        bearish_pivot1_idx=bearish_pivot1_idx,
        bearish_pivot2_idx=bearish_pivot2_idx,
        bullish_pivot1_price=bullish_pivot1_price,
        bullish_pivot2_price=bullish_pivot2_price,
        bearish_pivot1_price=bearish_pivot1_price,
        bearish_pivot2_price=bearish_pivot2_price,
        bullish_trigger_price=bullish_trigger_price,
        bearish_trigger_price=bearish_trigger_price,
        bullish_stop_anchor=bullish_stop_anchor,
        bearish_stop_anchor=bearish_stop_anchor,
    )
