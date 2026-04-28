# -----------------------------------------------------------------------------
# RobustEma55Divergence
# Freqtrade Strategy Adapter for quant_scaffold robust_ema55 baseline
#
# Original logic:
#   - Trend: EMA(55) channel on High/Low, prior trend via 24-bar ratio
#   - Oscillator: Stochastic %D(14,1,3)
#   - Divergence: Regular continuation divergence with progressive pullback
#   - Setup: 12-bar lifetime, replaceable, invalidated on stop-anchor break
#   - Entry: trigger price = High/Low of pivot2 (stop order)
#   - Stop: Low/High of pivot2 +/- buffer
#   - Target: RR = 2.2
# -----------------------------------------------------------------------------
import logging
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)


class RobustEma55Divergence(IStrategy):
    """
    Freqtrade strategy mirroring quant_scaffold robust_ema55 baseline.
    """

    # ------------------------------------------------------------------
    # Freqtrade metadata
    # ------------------------------------------------------------------
    timeframe = "5m"
    can_short = True
    startup_candle_count = 100
    process_only_new_candles = True

    # Disable default ROI / stoploss — we use custom_exit + custom_stoploss
    minimal_roi = {"0": 1.0}
    stoploss = -0.99

    # ------------------------------------------------------------------
    # Strategy constants (from configs/candidates/robust_ema55.yaml)
    # ------------------------------------------------------------------
    # trend
    ema_period = 55
    lookback_bars = 24
    min_close_ratio = 0.80

    # stochastic
    k_period = 14
    smooth = 1
    d_period = 3
    oversold = 15
    overbought = 85

    # pivots
    left_bars = 4
    right_bars = 3
    min_separation = 3
    max_separation = 20
    strict_pivots = True

    # risk / setup
    rr_target = 2.2
    stop_buffer = 0.0
    setup_max_bars = 12
    replace_same_side_setup = True
    invalidate_on_stop_anchor_break = True

    # ------------------------------------------------------------------
    # Indicator helpers (same math as quant_scaffold)
    # ------------------------------------------------------------------
    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def _stochastic_d(self, dataframe: DataFrame) -> pd.Series:
        low_min = dataframe["low"].rolling(self.k_period).min()
        high_max = dataframe["high"].rolling(self.k_period).max()
        raw_k = 100 * (dataframe["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
        fast_k = raw_k.rolling(max(self.smooth, 1)).mean()
        d = fast_k.rolling(self.d_period).mean()
        return d

    def _pivot_high(self, series: pd.Series) -> pd.Series:
        left = self.left_bars
        right = self.right_bars
        strict = self.strict_pivots
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

    def _pivot_low(self, series: pd.Series) -> pd.Series:
        left = self.left_bars
        right = self.right_bars
        strict = self.strict_pivots
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

    def _build_trend_filter(self, dataframe: DataFrame) -> dict:
        ema_high = self._ema(dataframe["high"], self.ema_period)
        ema_low = self._ema(dataframe["low"], self.ema_period)

        above_channel = dataframe["close"] > ema_high
        below_channel = dataframe["close"] < ema_low
        inside_or_below_high = dataframe["close"] < ema_high
        inside_or_above_low = dataframe["close"] > ema_low

        above_ratio = above_channel.shift(1).rolling(self.lookback_bars).mean()
        below_ratio = below_channel.shift(1).rolling(self.lookback_bars).mean()

        prior_uptrend = above_ratio >= self.min_close_ratio
        prior_downtrend = below_ratio >= self.min_close_ratio

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
        }

    # ------------------------------------------------------------------
    # Populate indicators (vectorised + lightweight loops for setups)
    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # --- 1. EMA channel ---
        dataframe["ema55"] = self._ema(dataframe["close"], self.ema_period)
        trend = self._build_trend_filter(dataframe)
        dataframe["ema_high"] = trend["ema_high"]
        dataframe["ema_low"] = trend["ema_low"]
        dataframe["prior_uptrend"] = trend["prior_uptrend"]
        dataframe["prior_downtrend"] = trend["prior_downtrend"]
        dataframe["inside_or_below_high"] = trend["inside_or_below_high"]
        dataframe["inside_or_above_low"] = trend["inside_or_above_low"]
        dataframe["below_channel"] = trend["below_channel"]
        dataframe["above_channel"] = trend["above_channel"]

        # --- 2. Stochastic ---
        dataframe["stoch_d"] = self._stochastic_d(dataframe)

        # --- 3. Pivots ---
        pivot_high = self._pivot_high(dataframe["high"])
        pivot_low = self._pivot_low(dataframe["low"])
        dataframe["pivot_high"] = pivot_high
        dataframe["pivot_low"] = pivot_low

        # --- 4. Detect regular continuation divergence ---
        osc = dataframe["stoch_d"]
        low = dataframe["low"]
        high = dataframe["high"]
        close = dataframe["close"]
        idx = dataframe.index

        low_idx = list(idx[pivot_low])
        high_idx = list(idx[pivot_high])

        bullish_setups = []   # dicts: confirm_pos, trigger_price, stop_anchor
        bearish_setups = []

        # Bullish (long): pivot lows, lower low price, higher osc, oversold pivot1
        for i in range(1, len(low_idx)):
            idx1, idx2 = low_idx[i - 1], low_idx[i]
            pos1, pos2 = idx.get_loc(idx1), idx.get_loc(idx2)
            sep = pos2 - pos1
            if not (self.min_separation <= sep <= self.max_separation):
                continue
            if not (low.loc[idx2] < low.loc[idx1] and osc.loc[idx2] > osc.loc[idx1] and osc.loc[idx1] <= self.oversold):
                continue
            confirm_pos = pos2 + self.right_bars
            if confirm_pos >= len(dataframe):
                continue

            p1_ch_ok = bool(trend["inside_or_below_high"].loc[idx1])
            p2_ch_ok = bool(trend["below_channel"].loc[idx2])
            prior_ok = bool(trend["prior_uptrend"].loc[idx1])

            if prior_ok and p1_ch_ok and p2_ch_ok:
                bullish_setups.append({
                    "confirm_pos": confirm_pos,
                    "trigger_price": high.loc[idx2],
                    "stop_anchor": low.loc[idx2],
                })

        # Bearish (short): pivot highs, higher high price, lower osc, overbought pivot1
        for i in range(1, len(high_idx)):
            idx1, idx2 = high_idx[i - 1], high_idx[i]
            pos1, pos2 = idx.get_loc(idx1), idx.get_loc(idx2)
            sep = pos2 - pos1
            if not (self.min_separation <= sep <= self.max_separation):
                continue
            if not (high.loc[idx2] > high.loc[idx1] and osc.loc[idx2] < osc.loc[idx1] and osc.loc[idx1] >= self.overbought):
                continue
            confirm_pos = pos2 + self.right_bars
            if confirm_pos >= len(dataframe):
                continue

            p1_ch_ok = bool(trend["inside_or_above_low"].loc[idx1])
            p2_ch_ok = bool(trend["above_channel"].loc[idx2])
            prior_ok = bool(trend["prior_downtrend"].loc[idx1])

            if prior_ok and p1_ch_ok and p2_ch_ok:
                bearish_setups.append({
                    "confirm_pos": confirm_pos,
                    "trigger_price": low.loc[idx2],
                    "stop_anchor": high.loc[idx2],
                })

        # --- 5. Apply setup lifecycle (state machine simulation) ---
        n = len(dataframe)
        enter_long = np.zeros(n, dtype=int)
        enter_short = np.zeros(n, dtype=int)
        long_stop = np.full(n, np.nan)
        long_target = np.full(n, np.nan)
        short_stop = np.full(n, np.nan)
        short_target = np.full(n, np.nan)
        long_trigger_price = np.full(n, np.nan)
        short_trigger_price = np.full(n, np.nan)

        # For each bar, record the *latest* setup that should govern it.
        # This implements replace_same_side_setup=True.
        latest_bullish = [-1] * n
        for s_idx, setup in enumerate(bullish_setups):
            cp = setup["confirm_pos"]
            start = cp + 1
            end = min(cp + 1 + self.setup_max_bars, n)
            for bar in range(start, end):
                latest_bullish[bar] = s_idx

        latest_bearish = [-1] * n
        for s_idx, setup in enumerate(bearish_setups):
            cp = setup["confirm_pos"]
            start = cp + 1
            end = min(cp + 1 + self.setup_max_bars, n)
            for bar in range(start, end):
                latest_bearish[bar] = s_idx

        # Now resolve each setup: walk forward until trigger, invalidation, or replaced.
        # Long
        for s_idx, setup in enumerate(bullish_setups):
            cp = setup["confirm_pos"]
            trigger = setup["trigger_price"]
            stop_anchor = setup["stop_anchor"]
            start = cp + 1
            end = min(cp + 1 + self.setup_max_bars, n)

            for bar in range(start, end):
                if latest_bullish[bar] != s_idx:
                    break  # replaced by newer setup

                if self.invalidate_on_stop_anchor_break and dataframe["low"].iloc[bar] < stop_anchor:
                    break  # stop anchor broken

                if dataframe["high"].iloc[bar] > trigger:
                    enter_long[bar] = 1
                    entry_price = trigger
                    stop_price = stop_anchor - self.stop_buffer
                    long_stop[bar] = stop_price
                    long_target[bar] = entry_price + self.rr_target * (entry_price - stop_price)
                    long_trigger_price[bar] = trigger
                    break

        # Short
        for s_idx, setup in enumerate(bearish_setups):
            cp = setup["confirm_pos"]
            trigger = setup["trigger_price"]
            stop_anchor = setup["stop_anchor"]
            start = cp + 1
            end = min(cp + 1 + self.setup_max_bars, n)

            for bar in range(start, end):
                if latest_bearish[bar] != s_idx:
                    break

                if self.invalidate_on_stop_anchor_break and dataframe["high"].iloc[bar] > stop_anchor:
                    break

                if dataframe["low"].iloc[bar] < trigger:
                    enter_short[bar] = 1
                    entry_price = trigger
                    stop_price = stop_anchor + self.stop_buffer
                    short_stop[bar] = stop_price
                    short_target[bar] = entry_price - self.rr_target * (stop_price - entry_price)
                    short_trigger_price[bar] = trigger
                    break

        dataframe["enter_long"] = enter_long
        dataframe["enter_short"] = enter_short
        dataframe["long_stop"] = long_stop
        dataframe["long_target"] = long_target
        dataframe["short_stop"] = short_stop
        dataframe["short_target"] = short_target
        dataframe["long_trigger_price"] = long_trigger_price
        dataframe["short_trigger_price"] = short_trigger_price

        return dataframe

    # ------------------------------------------------------------------
    # Entry trend (signals already computed in populate_indicators)
    # ------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ensure columns exist even if empty
        if "enter_long" not in dataframe.columns:
            dataframe["enter_long"] = 0
        if "enter_short" not in dataframe.columns:
            dataframe["enter_short"] = 0
        return dataframe

    # ------------------------------------------------------------------
    # Exit trend (unused; we rely on custom_exit + custom_stoploss)
    # ------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    # ------------------------------------------------------------------
    # Custom entry price = trigger price (stop-order emulation)
    # ------------------------------------------------------------------
    def custom_entry_price(self, pair, current_time, proposed_rate, entry_tag, side, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_rate
        last_candle = dataframe.iloc[-1]
        if side == "long":
            price = last_candle.get("long_trigger_price", proposed_rate)
        else:
            price = last_candle.get("short_trigger_price", proposed_rate)
        return price if pd.notna(price) else proposed_rate

    # ------------------------------------------------------------------
    # Custom stoploss = absolute stop anchor
    # ------------------------------------------------------------------
    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        # Try to read stop from trade custom_info (set in confirm_trade_entry)
        stop_price = None
        if hasattr(trade, "custom_info") and trade.custom_info:
            stop_price = trade.custom_info.get("stop_price", None)

        if stop_price is None or pd.isna(stop_price):
            # Fallback: lookup from dataframe at trade open time
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return 1.0
            mask = dataframe["date"] <= pd.Timestamp(trade.open_date_utc)
            if not mask.any():
                return 1.0
            entry_candle = dataframe.loc[mask].iloc[-1]
            if trade.is_short:
                stop_price = entry_candle.get("short_stop", None)
            else:
                stop_price = entry_candle.get("long_stop", None)

        if stop_price is None or pd.isna(stop_price):
            return 1.0

        stop_price = float(stop_price)

        if trade.is_short:
            if current_rate >= stop_price:
                return -0.999
            return (current_rate - stop_price) / current_rate
        else:
            if current_rate <= stop_price:
                return -0.999
            return (stop_price - current_rate) / current_rate

    # ------------------------------------------------------------------
    # Custom exit = target hit
    # ------------------------------------------------------------------
    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        target_price = None
        if hasattr(trade, "custom_info") and trade.custom_info:
            target_price = trade.custom_info.get("target_price", None)

        if target_price is None or pd.isna(target_price):
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return None
            mask = dataframe["date"] <= pd.Timestamp(trade.open_date_utc)
            if not mask.any():
                return None
            entry_candle = dataframe.loc[mask].iloc[-1]
            if trade.is_short:
                target_price = entry_candle.get("short_target", None)
            else:
                target_price = entry_candle.get("long_target", None)

        if target_price is None or pd.isna(target_price):
            return None

        target_price = float(target_price)

        if trade.is_short:
            if current_rate <= target_price:
                return "target"
        else:
            if current_rate >= target_price:
                return "target"

        return None

    # ------------------------------------------------------------------
    # Stake amount — use configured value (unlimited)
    # ------------------------------------------------------------------
    def custom_stake_amount(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_stake: float,
        min_stake,
        max_stake: float,
        leverage: float,
        entry_tag,
        side: str,
        **kwargs,
    ) -> float:
        return proposed_stake

    # ------------------------------------------------------------------
    # Leverage — fixed 1x
    # ------------------------------------------------------------------
    def leverage(self, pair: str, current_time, current_rate: float, proposed_leverage: float, **kwargs) -> float:
        return 1.0

    # ------------------------------------------------------------------
    # Persist stop/target on the trade object so callbacks can read them
    # ------------------------------------------------------------------
    def confirm_trade_entry(
        self,
        pair,
        order_type,
        amount,
        rate,
        time_in_force,
        current_time,
        entry_tag,
        side,
        **kwargs,
    ) -> bool:
        trade = kwargs.get("trade")
        if trade is None:
            return True

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return True
        last_candle = dataframe.iloc[-1]

        if side == "long":
            stop = last_candle.get("long_stop", None)
            target = last_candle.get("long_target", None)
        else:
            stop = last_candle.get("short_stop", None)
            target = last_candle.get("short_target", None)

        try:
            if not hasattr(trade, "custom_info") or trade.custom_info is None:
                trade.custom_info = {}
            if pd.notna(stop):
                trade.custom_info["stop_price"] = float(stop)
            if pd.notna(target):
                trade.custom_info["target_price"] = float(target)
            trade.custom_info["trigger_price"] = float(rate)
        except Exception:
            pass

        return True
