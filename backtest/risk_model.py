"""Position sizing by risk percent."""

import math
from typing import Dict, Union


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_per_trade_pct: float,
    max_position_value_pct: float = 1.0,
    max_leverage: float = 1.0,
    min_qty: float = 0.0001,
    qty_step: float = 0.0001,
) -> Dict[str, Union[float, bool, str]]:
    """
    Calculate position size based on risk percent.

    Returns dict with:
        qty, notional, target_risk_amount, actual_risk_amount,
        target_risk_pct, actual_risk_pct, stop_distance,
        raw_qty, max_qty, cap_hit,
        skip_trade, skip_reason
    """
    result = {
        "qty": 0.0,
        "notional": 0.0,
        "target_risk_amount": 0.0,
        "actual_risk_amount": 0.0,
        "target_risk_pct": 0.0,
        "actual_risk_pct": 0.0,
        "stop_distance": 0.0,
        "raw_qty": 0.0,
        "max_qty": 0.0,
        "cap_hit": False,
        "skip_trade": True,
        "skip_reason": "",
    }

    if equity <= 0:
        result["skip_reason"] = "insufficient_equity"
        return result

    if entry_price <= 0 or stop_price <= 0:
        result["skip_reason"] = "invalid_price"
        return result

    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        result["skip_reason"] = "invalid_stop_distance"
        return result

    target_risk_amount = equity * risk_per_trade_pct
    raw_qty = target_risk_amount / stop_distance

    max_notional = equity * max_position_value_pct * max_leverage
    max_qty = max_notional / entry_price if entry_price > 0 else 0.0

    qty = min(raw_qty, max_qty)
    cap_hit = raw_qty > max_qty

    # Round down to qty_step
    if qty_step > 0:
        qty = math.floor(qty / qty_step) * qty_step

    if qty < min_qty:
        result["skip_reason"] = "qty_below_min"
        return result

    notional = qty * entry_price
    actual_risk_amount = qty * stop_distance
    actual_risk_pct = actual_risk_amount / equity if equity > 0 else 0.0

    result["qty"] = qty
    result["notional"] = notional
    result["target_risk_amount"] = target_risk_amount
    result["actual_risk_amount"] = actual_risk_amount
    result["target_risk_pct"] = risk_per_trade_pct
    result["actual_risk_pct"] = actual_risk_pct
    result["stop_distance"] = stop_distance
    result["raw_qty"] = raw_qty
    result["max_qty"] = max_qty
    result["cap_hit"] = cap_hit
    result["skip_trade"] = False
    result["skip_reason"] = ""
    return result
