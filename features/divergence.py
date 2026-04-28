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
    if strict:
        rolled_left = series.rolling(window=left, min_periods=left).max().shift(1)
        rolled_right = series[::-1].rolling(window=right, min_periods=right).max()[::-1].shift(-right)
        result = (series > rolled_left) & (series > rolled_right)
    else:
        rolled = series.rolling(window=left + right + 1, min_periods=left + right + 1, center=True).max()
        result = series == rolled
    return result


def _pivot_low(series: pd.Series, left: int, right: int, strict: bool = True) -> pd.Series:
    if strict:
        rolled_left = series.rolling(window=left, min_periods=left).min().shift(1)
        rolled_right = series[::-1].rolling(window=right, min_periods=right).min()[::-1].shift(-right)
        result = (series < rolled_left) & (series < rolled_right)
    else:
        rolled = series.rolling(window=left + right + 1, min_periods=left + right + 1, center=True).min()
        result = series == rolled
    return result


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

    # Bullish loop: collect confirm indices and data, then batch-assign.
    bullish_raw_list = []
    bullish_prior_list = []
    bullish_p1_ch_list = []
    bullish_p2_ch_list = []
    bullish_ch_br_list = []
    bullish_list = []

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
                    p1_ch_ok = bool(trend["inside_or_below_high"].iloc[pos1])
                    p2_ch_ok = bool(trend["below_channel"].iloc[pos2])
                    prior_ok = bool(trend["prior_uptrend"].iloc[pos1])
                    channel_ok = p1_ch_ok and p2_ch_ok

                    bullish_raw_list.append(confirm_idx)
                    bullish_prior_list.append((confirm_idx, prior_ok))
                    bullish_p1_ch_list.append((confirm_idx, p1_ch_ok))
                    bullish_p2_ch_list.append((confirm_idx, p2_ch_ok))
                    bullish_ch_br_list.append((confirm_idx, channel_ok))

                    if prior_ok and channel_ok:
                        bullish_list.append({
                            "idx": confirm_idx,
                            "p1_idx": idx1,
                            "p2_idx": idx2,
                            "p1_price": low.loc[idx1],
                            "p2_price": low.loc[idx2],
                            "trigger_price": high.loc[idx2],
                            "stop_anchor": low.loc[idx2],
                            "confirm_pos": confirm_pos,
                        })

    if bullish_raw_list:
        bullish_raw_divergence.loc[bullish_raw_list] = True
        bullish_prior_trend_ok.loc[[x[0] for x in bullish_prior_list]] = [x[1] for x in bullish_prior_list]
        bullish_pivot1_channel_ok.loc[[x[0] for x in bullish_p1_ch_list]] = [x[1] for x in bullish_p1_ch_list]
        bullish_pivot2_channel_ok.loc[[x[0] for x in bullish_p2_ch_list]] = [x[1] for x in bullish_p2_ch_list]
        bullish_channel_break_ok.loc[[x[0] for x in bullish_ch_br_list]] = [x[1] for x in bullish_ch_br_list]

    if bullish_list:
        idxs = [d["idx"] for d in bullish_list]
        bullish.loc[idxs] = True
        bullish_pivot1_idx.loc[idxs] = [d["p1_idx"] for d in bullish_list]
        bullish_pivot2_idx.loc[idxs] = [d["p2_idx"] for d in bullish_list]
        bullish_pivot1_price.loc[idxs] = [d["p1_price"] for d in bullish_list]
        bullish_pivot2_price.loc[idxs] = [d["p2_price"] for d in bullish_list]
        bullish_trigger_price.loc[idxs] = [d["trigger_price"] for d in bullish_list]
        bullish_stop_anchor.loc[idxs] = [d["stop_anchor"] for d in bullish_list]
        bullish_confirm_idx.loc[idxs] = idxs
        bullish_confirm_pos.loc[idxs] = [d["confirm_pos"] for d in bullish_list]

    # Bearish loop: collect confirm indices and data, then batch-assign.
    bearish_raw_list = []
    bearish_prior_list = []
    bearish_p1_ch_list = []
    bearish_p2_ch_list = []
    bearish_ch_br_list = []
    bearish_list = []

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
                    p1_ch_ok = bool(trend["inside_or_above_low"].iloc[pos1])
                    p2_ch_ok = bool(trend["above_channel"].iloc[pos2])
                    prior_ok = bool(trend["prior_downtrend"].iloc[pos1])
                    channel_ok = p1_ch_ok and p2_ch_ok

                    bearish_raw_list.append(confirm_idx)
                    bearish_prior_list.append((confirm_idx, prior_ok))
                    bearish_p1_ch_list.append((confirm_idx, p1_ch_ok))
                    bearish_p2_ch_list.append((confirm_idx, p2_ch_ok))
                    bearish_ch_br_list.append((confirm_idx, channel_ok))

                    if prior_ok and channel_ok:
                        bearish_list.append({
                            "idx": confirm_idx,
                            "p1_idx": idx1,
                            "p2_idx": idx2,
                            "p1_price": high.loc[idx1],
                            "p2_price": high.loc[idx2],
                            "trigger_price": low.loc[idx2],
                            "stop_anchor": high.loc[idx2],
                            "confirm_pos": confirm_pos,
                        })

    if bearish_raw_list:
        bearish_raw_divergence.loc[bearish_raw_list] = True
        bearish_prior_trend_ok.loc[[x[0] for x in bearish_prior_list]] = [x[1] for x in bearish_prior_list]
        bearish_pivot1_channel_ok.loc[[x[0] for x in bearish_p1_ch_list]] = [x[1] for x in bearish_p1_ch_list]
        bearish_pivot2_channel_ok.loc[[x[0] for x in bearish_p2_ch_list]] = [x[1] for x in bearish_p2_ch_list]
        bearish_channel_break_ok.loc[[x[0] for x in bearish_ch_br_list]] = [x[1] for x in bearish_ch_br_list]

    if bearish_list:
        idxs = [d["idx"] for d in bearish_list]
        bearish.loc[idxs] = True
        bearish_pivot1_idx.loc[idxs] = [d["p1_idx"] for d in bearish_list]
        bearish_pivot2_idx.loc[idxs] = [d["p2_idx"] for d in bearish_list]
        bearish_pivot1_price.loc[idxs] = [d["p1_price"] for d in bearish_list]
        bearish_pivot2_price.loc[idxs] = [d["p2_price"] for d in bearish_list]
        bearish_trigger_price.loc[idxs] = [d["trigger_price"] for d in bearish_list]
        bearish_stop_anchor.loc[idxs] = [d["stop_anchor"] for d in bearish_list]
        bearish_confirm_idx.loc[idxs] = idxs
        bearish_confirm_pos.loc[idxs] = [d["confirm_pos"] for d in bearish_list]

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
