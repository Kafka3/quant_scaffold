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
    bullish_confirm_idx: pd.Series
    bearish_confirm_idx: pd.Series
    bullish_confirm_pos: pd.Series
    bearish_confirm_pos: pd.Series
    bullish_raw_divergence: pd.Series
    bearish_raw_divergence: pd.Series
    bullish_prior_trend_ok: pd.Series
    bearish_prior_trend_ok: pd.Series
    bullish_pivot1_channel_ok: pd.Series
    bullish_pivot2_channel_ok: pd.Series
    bearish_pivot1_channel_ok: pd.Series
    bearish_pivot2_channel_ok: pd.Series
    bullish_channel_break_ok: pd.Series
    bearish_channel_break_ok: pd.Series


def _pivot_high(series: pd.Series, left: int, right: int, strict: bool = True) -> pd.Series:
    out = pd.Series(False, index=series.index)
    for i in range(left, len(series) - right):
        if strict:
            left_max = series.iloc[i - left : i].max()
            right_max = series.iloc[i + 1 : i + right + 1].max()
            out.iloc[i] = series.iloc[i] > left_max and series.iloc[i] > right_max
        else:
            window = series.iloc[i - left : i + right + 1]
            out.iloc[i] = series.iloc[i] == window.max()
    return out


def _pivot_low(series: pd.Series, left: int, right: int, strict: bool = True) -> pd.Series:
    out = pd.Series(False, index=series.index)
    for i in range(left, len(series) - right):
        if strict:
            left_min = series.iloc[i - left : i].min()
            right_min = series.iloc[i + 1 : i + right + 1].min()
            out.iloc[i] = series.iloc[i] < left_min and series.iloc[i] < right_min
        else:
            window = series.iloc[i - left : i + right + 1]
            out.iloc[i] = series.iloc[i] == window.min()
    return out


def detect_regular_divergence(
    df: pd.DataFrame, osc: pd.Series, config: dict, trend: dict
) -> DivergenceResult:
    """
    Detect continuation divergence using progressive pullback logic.

    Bullish continuation (long):
      - Uses pivot lows.
      - Price: pivot2 Low < pivot1 Low (lower low).
      - Oscillator: osc[pivot2] > osc[pivot1] (higher oscillator).
      - osc[pivot1] <= oversold.
      - Prior uptrend existed before pivot1.
      - pivot1 is an early pullback: Close < ema_high (inside_or_below_high).
      - pivot2 is a deep pullback: Close < ema_low (below_channel).
      - Signal marked on confirmation bar (pivot2_pos + right_bars).

    Bearish continuation (short):
      - Uses pivot highs.
      - Price: pivot2 High > pivot1 High (higher high).
      - Oscillator: osc[pivot2] < osc[pivot1] (lower oscillator).
      - osc[pivot1] >= overbought.
      - Prior downtrend existed before pivot1.
      - pivot1 is an early pullback: Close > ema_low (inside_or_above_low).
      - pivot2 is a deep pullback: Close > ema_high (above_channel).
      - Signal marked on confirmation bar (pivot2_pos + right_bars).
    """
    high = df["High"]
    low = df["Low"]

    piv_cfg = config["pivots"]
    stoch_cfg = config["stochastic"]
    right_bars = piv_cfg["right_bars"]

    strict = piv_cfg.get("strict", True)
    pivot_high = _pivot_high(high, piv_cfg["left_bars"], right_bars, strict=strict)
    pivot_low = _pivot_low(low, piv_cfg["left_bars"], right_bars, strict=strict)

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
    bullish_confirm_idx = pd.Series(index=df.index, dtype=object)
    bearish_confirm_idx = pd.Series(index=df.index, dtype=object)
    bullish_confirm_pos = pd.Series(index=df.index, dtype=float)
    bearish_confirm_pos = pd.Series(index=df.index, dtype=float)
    bullish_raw_divergence = pd.Series(False, index=df.index)
    bearish_raw_divergence = pd.Series(False, index=df.index)
    bullish_prior_trend_ok = pd.Series(False, index=df.index)
    bearish_prior_trend_ok = pd.Series(False, index=df.index)
    bullish_pivot1_channel_ok = pd.Series(False, index=df.index)
    bullish_pivot2_channel_ok = pd.Series(False, index=df.index)
    bearish_pivot1_channel_ok = pd.Series(False, index=df.index)
    bearish_pivot2_channel_ok = pd.Series(False, index=df.index)
    bullish_channel_break_ok = pd.Series(False, index=df.index)
    bearish_channel_break_ok = pd.Series(False, index=df.index)

    low_idx = list(df.index[pivot_low])
    for i in range(1, len(low_idx)):
        idx1, idx2 = low_idx[i - 1], low_idx[i]
        pos1, pos2 = df.index.get_loc(idx1), df.index.get_loc(idx2)
        sep = pos2 - pos1
        if piv_cfg["min_separation"] <= sep <= piv_cfg["max_separation"]:
            if low.loc[idx2] < low.loc[idx1] and osc.loc[idx2] > osc.loc[idx1] and osc.loc[idx1] <= stoch_cfg["oversold"]:
                confirm_pos = pos2 + right_bars
                if confirm_pos < len(df):
                    confirm_idx = df.index[confirm_pos]

                    # Progressive channel checks for bullish continuation.
                    p1_ch_ok = bool(trend["inside_or_below_high"].loc[idx1])
                    p2_ch_ok = bool(trend["below_channel"].loc[idx2])
                    prior_ok = bool(trend["prior_uptrend"].loc[idx1])
                    channel_ok = p1_ch_ok and p2_ch_ok

                    bullish_raw_divergence.loc[confirm_idx] = True
                    bullish_prior_trend_ok.loc[confirm_idx] = prior_ok
                    bullish_pivot1_channel_ok.loc[confirm_idx] = p1_ch_ok
                    bullish_pivot2_channel_ok.loc[confirm_idx] = p2_ch_ok
                    bullish_channel_break_ok.loc[confirm_idx] = channel_ok

                    if prior_ok and channel_ok:
                        bullish.loc[confirm_idx] = True
                        bullish_pivot1_idx.loc[confirm_idx] = idx1
                        bullish_pivot2_idx.loc[confirm_idx] = idx2
                        bullish_pivot1_price.loc[confirm_idx] = low.loc[idx1]
                        bullish_pivot2_price.loc[confirm_idx] = low.loc[idx2]
                        bullish_trigger_price.loc[confirm_idx] = high.loc[idx2]
                        bullish_stop_anchor.loc[confirm_idx] = low.loc[idx2]
                        bullish_confirm_idx.loc[confirm_idx] = confirm_idx
                        bullish_confirm_pos.loc[confirm_idx] = confirm_pos

    high_idx = list(df.index[pivot_high])
    for i in range(1, len(high_idx)):
        idx1, idx2 = high_idx[i - 1], high_idx[i]
        pos1, pos2 = df.index.get_loc(idx1), df.index.get_loc(idx2)
        sep = pos2 - pos1
        if piv_cfg["min_separation"] <= sep <= piv_cfg["max_separation"]:
            if high.loc[idx2] > high.loc[idx1] and osc.loc[idx2] < osc.loc[idx1] and osc.loc[idx1] >= stoch_cfg["overbought"]:
                confirm_pos = pos2 + right_bars
                if confirm_pos < len(df):
                    confirm_idx = df.index[confirm_pos]

                    # Progressive channel checks for bearish continuation.
                    p1_ch_ok = bool(trend["inside_or_above_low"].loc[idx1])
                    p2_ch_ok = bool(trend["above_channel"].loc[idx2])
                    prior_ok = bool(trend["prior_downtrend"].loc[idx1])
                    channel_ok = p1_ch_ok and p2_ch_ok

                    bearish_raw_divergence.loc[confirm_idx] = True
                    bearish_prior_trend_ok.loc[confirm_idx] = prior_ok
                    bearish_pivot1_channel_ok.loc[confirm_idx] = p1_ch_ok
                    bearish_pivot2_channel_ok.loc[confirm_idx] = p2_ch_ok
                    bearish_channel_break_ok.loc[confirm_idx] = channel_ok

                    if prior_ok and channel_ok:
                        bearish.loc[confirm_idx] = True
                        bearish_pivot1_idx.loc[confirm_idx] = idx1
                        bearish_pivot2_idx.loc[confirm_idx] = idx2
                        bearish_pivot1_price.loc[confirm_idx] = high.loc[idx1]
                        bearish_pivot2_price.loc[confirm_idx] = high.loc[idx2]
                        bearish_trigger_price.loc[confirm_idx] = low.loc[idx2]
                        bearish_stop_anchor.loc[confirm_idx] = high.loc[idx2]
                        bearish_confirm_idx.loc[confirm_idx] = confirm_idx
                        bearish_confirm_pos.loc[confirm_idx] = confirm_pos

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
        bullish_confirm_idx=bullish_confirm_idx,
        bearish_confirm_idx=bearish_confirm_idx,
        bullish_confirm_pos=bullish_confirm_pos,
        bearish_confirm_pos=bearish_confirm_pos,
        bullish_raw_divergence=bullish_raw_divergence,
        bearish_raw_divergence=bearish_raw_divergence,
        bullish_prior_trend_ok=bullish_prior_trend_ok,
        bearish_prior_trend_ok=bearish_prior_trend_ok,
        bullish_pivot1_channel_ok=bullish_pivot1_channel_ok,
        bullish_pivot2_channel_ok=bullish_pivot2_channel_ok,
        bearish_pivot1_channel_ok=bearish_pivot1_channel_ok,
        bearish_pivot2_channel_ok=bearish_pivot2_channel_ok,
        bullish_channel_break_ok=bullish_channel_break_ok,
        bearish_channel_break_ok=bearish_channel_break_ok,
    )
