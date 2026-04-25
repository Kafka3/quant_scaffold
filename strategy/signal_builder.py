from dataclasses import dataclass
from typing import Optional

import pandas as pd

from features.indicators import stochastic_d, atr
from features.divergence import detect_regular_divergence
from features.trend_filter import build_trend_filter


@dataclass
class PendingSetup:
    setup_bar_idx: object
    setup_pos: int
    setup_time: object
    pivot2_idx: object
    pivot2_pos: int
    pivot2_time: object
    trigger_price: float
    stop_anchor: float
    bars_waited: int = 0


@dataclass
class SignalBundle:
    entries_long: pd.Series
    exits_long: pd.Series
    entries_short: pd.Series
    exits_short: pd.Series
    long_entry_price: pd.Series
    short_entry_price: pd.Series
    long_stop_price: pd.Series
    short_stop_price: pd.Series
    long_target_price: pd.Series
    short_target_price: pd.Series
    long_setup_pivot2_time: pd.Series
    short_setup_pivot2_time: pd.Series
    long_setup_confirm_time: pd.Series
    short_setup_confirm_time: pd.Series
    long_trigger_price_raw: pd.Series
    short_trigger_price_raw: pd.Series
    features: pd.DataFrame
    # backward-compatible aliases
    entry_prices_long: pd.Series
    entry_prices_short: pd.Series
    stop_prices_long: pd.Series
    stop_prices_short: pd.Series
    target_prices_long: pd.Series
    target_prices_short: pd.Series


def build_signals(df: pd.DataFrame, config: dict) -> SignalBundle:
    stoch_cfg = config["stochastic"]
    risk_cfg = config["risk"]
    setup_cfg = config.get("setup", {})
    setup_max_bars = setup_cfg.get("setup_max_bars", 12)
    replace_same_side_setup = setup_cfg.get("replace_same_side_setup", True)
    invalidate_on_stop_anchor_break = setup_cfg.get("invalidate_on_stop_anchor_break", True)

    osc = stochastic_d(
        df,
        k_period=stoch_cfg["k_period"],
        d_period=stoch_cfg["d_period"],
        smooth=stoch_cfg["smooth"],
    )
    trend = build_trend_filter(df, config["trend"])
    div = detect_regular_divergence(df, osc, config, trend)
    current_atr = atr(df, risk_cfg["atr_period"])

    entries_long = pd.Series(False, index=df.index)
    entries_short = pd.Series(False, index=df.index)
    exits_long = pd.Series(False, index=df.index)
    exits_short = pd.Series(False, index=df.index)

    long_entry_price = pd.Series(index=df.index, dtype=float)
    short_entry_price = pd.Series(index=df.index, dtype=float)
    long_stop_price = pd.Series(index=df.index, dtype=float)
    short_stop_price = pd.Series(index=df.index, dtype=float)
    long_target_price = pd.Series(index=df.index, dtype=float)
    short_target_price = pd.Series(index=df.index, dtype=float)
    long_setup_pivot2_time = pd.Series(index=df.index, dtype=object)
    short_setup_pivot2_time = pd.Series(index=df.index, dtype=object)
    long_setup_confirm_time = pd.Series(index=df.index, dtype=object)
    short_setup_confirm_time = pd.Series(index=df.index, dtype=object)
    long_trigger_price_raw = pd.Series(index=df.index, dtype=float)
    short_trigger_price_raw = pd.Series(index=df.index, dtype=float)

    pending_long: Optional[PendingSetup] = None
    pending_short: Optional[PendingSetup] = None

    bullish_setup_active = []
    bearish_setup_active = []
    bullish_setup_trigger = []
    bearish_setup_trigger = []
    bullish_setup_stop_anchor = []
    bearish_setup_stop_anchor = []
    bullish_setup_pivot2_pos = []
    bearish_setup_pivot2_pos = []
    bullish_setup_pivot2_time = []
    bearish_setup_pivot2_time = []
    bullish_setup_expired = []
    bearish_setup_expired = []
    bullish_setup_confirm_time = []
    bearish_setup_confirm_time = []

    for idx in df.index:
        current_pos = df.index.get_loc(idx)
        expired_long = False
        expired_short = False

        # ------------------------------------------------------------------
        # 1. 处理已存在的 pending setup（等待、触发或失效）
        # ------------------------------------------------------------------
        # 规则明确：
        #   - 背离确认 bar 只创建 setup，同 bar 不触发。
        #   - 触发只能从确认 bar 的下一根 K 线开始（current_pos > setup_pos）。
        #   - setup 可能因结构破坏（stop_anchor 被击穿）或超时（bars_waited > max）而失效。
        # ------------------------------------------------------------------

        if pending_long is not None:
            # bars_waited 从确认 bar 后的第一根 K 线开始计数
            if current_pos > pending_long.setup_pos:
                pending_long.bars_waited += 1

            # 失效 1：结构破坏 — Low 跌破 stop_anchor（仅当配置启用时）
            if invalidate_on_stop_anchor_break and df.loc[idx, "Low"] < pending_long.stop_anchor:
                expired_long = True
                pending_long = None
            # 失效 2：超时
            elif pending_long.bars_waited > setup_max_bars:
                expired_long = True
                pending_long = None
            # 触发：只能从确认 bar 下一根开始，High 上穿 trigger_price，只触发一次
            elif current_pos > pending_long.setup_pos and df.loc[idx, "High"] > pending_long.trigger_price:
                entries_long.loc[idx] = True
                long_entry_price.loc[idx] = pending_long.trigger_price
                long_stop_price.loc[idx] = pending_long.stop_anchor - risk_cfg["stop_buffer"]
                r = long_entry_price.loc[idx] - long_stop_price.loc[idx]
                long_target_price.loc[idx] = long_entry_price.loc[idx] + risk_cfg["rr_target"] * r
                long_setup_pivot2_time.loc[idx] = pending_long.pivot2_time
                long_setup_confirm_time.loc[idx] = pending_long.setup_time
                long_trigger_price_raw.loc[idx] = pending_long.trigger_price
                pending_long = None
            # 否则继续持有 pending setup（等待中）

        if pending_short is not None:
            # bars_waited 从确认 bar 后的第一根 K 线开始计数
            if current_pos > pending_short.setup_pos:
                pending_short.bars_waited += 1

            # 失效 1：结构破坏 — High 涨破 stop_anchor（仅当配置启用时）
            if invalidate_on_stop_anchor_break and df.loc[idx, "High"] > pending_short.stop_anchor:
                expired_short = True
                pending_short = None
            # 失效 2：超时
            elif pending_short.bars_waited > setup_max_bars:
                expired_short = True
                pending_short = None
            # 触发：只能从确认 bar 下一根开始，Low 跌破 trigger_price，只触发一次
            elif current_pos > pending_short.setup_pos and df.loc[idx, "Low"] < pending_short.trigger_price:
                entries_short.loc[idx] = True
                short_entry_price.loc[idx] = pending_short.trigger_price
                short_stop_price.loc[idx] = pending_short.stop_anchor + risk_cfg["stop_buffer"]
                r = short_stop_price.loc[idx] - short_entry_price.loc[idx]
                short_target_price.loc[idx] = short_entry_price.loc[idx] - risk_cfg["rr_target"] * r
                short_setup_pivot2_time.loc[idx] = pending_short.pivot2_time
                short_setup_confirm_time.loc[idx] = pending_short.setup_time
                short_trigger_price_raw.loc[idx] = pending_short.trigger_price
                pending_short = None
            # 否则继续持有 pending setup（等待中）

        # ------------------------------------------------------------------
        # 2. 在背离确认 bar 上创建新 setup
        # ------------------------------------------------------------------
        # 注意：setup 在确认 bar 上创建，但同 bar 不触发。
        # 如果 replace_same_side_setup=True，新 setup 会覆盖旧 setup。
        # ------------------------------------------------------------------

        if div.bullish.loc[idx]:
            trigger_price = div.bullish_trigger_price.loc[idx]
            stop_anchor = div.bullish_stop_anchor.loc[idx]
            pivot2_idx = div.bullish_pivot2_idx.loc[idx]
            if not pd.isna(trigger_price) and not pd.isna(stop_anchor) and not pd.isna(pivot2_idx):
                if replace_same_side_setup or pending_long is None:
                    setup_pos = current_pos
                    pivot2_pos = df.index.get_loc(pivot2_idx)
                    pending_long = PendingSetup(
                        setup_bar_idx=idx,
                        setup_pos=setup_pos,
                        setup_time=idx,
                        pivot2_idx=pivot2_idx,
                        pivot2_pos=pivot2_pos,
                        pivot2_time=pivot2_idx,
                        trigger_price=trigger_price,
                        stop_anchor=stop_anchor,
                        bars_waited=0,
                    )

        if div.bearish.loc[idx]:
            trigger_price = div.bearish_trigger_price.loc[idx]
            stop_anchor = div.bearish_stop_anchor.loc[idx]
            pivot2_idx = div.bearish_pivot2_idx.loc[idx]
            if not pd.isna(trigger_price) and not pd.isna(stop_anchor) and not pd.isna(pivot2_idx):
                if replace_same_side_setup or pending_short is None:
                    setup_pos = current_pos
                    pivot2_pos = df.index.get_loc(pivot2_idx)
                    pending_short = PendingSetup(
                        setup_bar_idx=idx,
                        setup_pos=setup_pos,
                        setup_time=idx,
                        pivot2_idx=pivot2_idx,
                        pivot2_pos=pivot2_pos,
                        pivot2_time=pivot2_idx,
                        trigger_price=trigger_price,
                        stop_anchor=stop_anchor,
                        bars_waited=0,
                    )

        # 记录当前 bar 的 setup 状态到 features 列表
        bullish_setup_active.append(pending_long is not None)
        bearish_setup_active.append(pending_short is not None)
        bullish_setup_trigger.append(pending_long.trigger_price if pending_long is not None else pd.NA)
        bearish_setup_trigger.append(pending_short.trigger_price if pending_short is not None else pd.NA)
        bullish_setup_stop_anchor.append(pending_long.stop_anchor if pending_long is not None else pd.NA)
        bearish_setup_stop_anchor.append(pending_short.stop_anchor if pending_short is not None else pd.NA)
        bullish_setup_pivot2_pos.append(pending_long.pivot2_pos if pending_long is not None else pd.NA)
        bearish_setup_pivot2_pos.append(pending_short.pivot2_pos if pending_short is not None else pd.NA)
        bullish_setup_pivot2_time.append(pending_long.pivot2_time if pending_long is not None else pd.NA)
        bearish_setup_pivot2_time.append(pending_short.pivot2_time if pending_short is not None else pd.NA)
        bullish_setup_expired.append(expired_long)
        bearish_setup_expired.append(expired_short)
        bullish_setup_confirm_time.append(pending_long.setup_time if pending_long is not None else pd.NA)
        bearish_setup_confirm_time.append(pending_short.setup_time if pending_short is not None else pd.NA)

    # Backward-compatible alias fields for existing custom backtest code.
    entry_prices_long = long_entry_price
    entry_prices_short = short_entry_price
    stop_prices_long = long_stop_price
    stop_prices_short = short_stop_price
    target_prices_long = long_target_price
    target_prices_short = short_target_price

    features = pd.DataFrame(
        {
            "Close": df["Close"],
            "osc": osc,
            "atr": current_atr,
            "ema_high": trend["ema_high"],
            "ema_low": trend["ema_low"],
            "above_channel": trend["above_channel"].astype(int),
            "below_channel": trend["below_channel"].astype(int),
            "inside_or_below_high": trend["inside_or_below_high"].astype(int),
            "inside_or_above_low": trend["inside_or_above_low"].astype(int),
            "above_ratio": trend["above_ratio"],
            "below_ratio": trend["below_ratio"],
            "prior_uptrend": trend["prior_uptrend"].astype(int),
            "prior_downtrend": trend["prior_downtrend"].astype(int),
            "trend_long": trend["trend_long"].astype(int),
            "trend_short": trend["trend_short"].astype(int),
            "pivot_high": div.pivot_high.astype(int),
            "pivot_low": div.pivot_low.astype(int),
            "bullish_raw_divergence": div.bullish_raw_divergence.astype(int),
            "bearish_raw_divergence": div.bearish_raw_divergence.astype(int),
            "bullish_prior_trend_ok": div.bullish_prior_trend_ok.astype(int),
            "bearish_prior_trend_ok": div.bearish_prior_trend_ok.astype(int),
            "bullish_pivot1_channel_ok": div.bullish_pivot1_channel_ok.astype(int),
            "bullish_pivot2_channel_ok": div.bullish_pivot2_channel_ok.astype(int),
            "bearish_pivot1_channel_ok": div.bearish_pivot1_channel_ok.astype(int),
            "bearish_pivot2_channel_ok": div.bearish_pivot2_channel_ok.astype(int),
            "bullish_channel_break_ok": div.bullish_channel_break_ok.astype(int),
            "bearish_channel_break_ok": div.bearish_channel_break_ok.astype(int),
            "bullish_div": div.bullish.astype(int),
            "bearish_div": div.bearish.astype(int),
            "bullish_trigger_price": div.bullish_trigger_price,
            "bearish_trigger_price": div.bearish_trigger_price,
            "bullish_stop_anchor": div.bullish_stop_anchor,
            "bearish_stop_anchor": div.bearish_stop_anchor,
            "bullish_setup_active": bullish_setup_active,
            "bearish_setup_active": bearish_setup_active,
            "bullish_setup_trigger": bullish_setup_trigger,
            "bearish_setup_trigger": bearish_setup_trigger,
            "bullish_setup_stop_anchor": bullish_setup_stop_anchor,
            "bearish_setup_stop_anchor": bearish_setup_stop_anchor,
            "bullish_setup_pivot2_pos": bullish_setup_pivot2_pos,
            "bearish_setup_pivot2_pos": bearish_setup_pivot2_pos,
            "bullish_setup_pivot2_time": bullish_setup_pivot2_time,
            "bearish_setup_pivot2_time": bearish_setup_pivot2_time,
            "bullish_setup_expired": bullish_setup_expired,
            "bearish_setup_expired": bearish_setup_expired,
            "bullish_setup_confirm_time": bullish_setup_confirm_time,
            "bearish_setup_confirm_time": bearish_setup_confirm_time,
            "long_entry_price": long_entry_price,
            "short_entry_price": short_entry_price,
            "long_stop_price": long_stop_price,
            "short_stop_price": short_stop_price,
            "long_target_price": long_target_price,
            "short_target_price": short_target_price,
        },
        index=df.index,
    )

    return SignalBundle(
        entries_long=entries_long.fillna(False),
        exits_long=exits_long.fillna(False),
        entries_short=entries_short.fillna(False),
        exits_short=exits_short.fillna(False),
        long_entry_price=long_entry_price,
        short_entry_price=short_entry_price,
        long_stop_price=long_stop_price,
        short_stop_price=short_stop_price,
        long_target_price=long_target_price,
        short_target_price=short_target_price,
        long_setup_pivot2_time=long_setup_pivot2_time,
        short_setup_pivot2_time=short_setup_pivot2_time,
        long_setup_confirm_time=long_setup_confirm_time,
        short_setup_confirm_time=short_setup_confirm_time,
        long_trigger_price_raw=long_trigger_price_raw,
        short_trigger_price_raw=short_trigger_price_raw,
        features=features,
        entry_prices_long=entry_prices_long,
        entry_prices_short=entry_prices_short,
        stop_prices_long=stop_prices_long,
        stop_prices_short=stop_prices_short,
        target_prices_long=target_prices_long,
        target_prices_short=target_prices_short,
    )
