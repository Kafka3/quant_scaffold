"""Cost model: slippage and fees."""

from typing import Dict


def apply_slippage(price: float, side: str, action: str, slippage: float, is_rate: bool = False) -> float:
    """
    Apply slippage to a price.

    Parameters
    ----------
    price : float
        Raw price before slippage.
    side : str
        'long' or 'short'.
    action : str
        'entry' or 'exit'.
    slippage : float
        Absolute slippage amount (in price units) or rate (fraction of price).
    is_rate : bool
        When True, `slippage` is treated as a rate and multiplied by `price`.
    """
    if is_rate:
        slippage = price * slippage

    if side == "long":
        if action == "entry":
            return price + slippage
        else:  # exit
            return price - slippage
    else:  # short
        if action == "entry":
            return price - slippage
        else:  # exit
            return price + slippage


def calculate_fees(
    entry_price: float,
    exit_price: float,
    qty: float,
    fee_rate: float,
    fixed_fee_per_trade: float = 0.0,
) -> float:
    """Calculate total fees for a round trip."""
    fee = entry_price * qty * fee_rate
    fee += exit_price * qty * fee_rate
    fee += 2 * fixed_fee_per_trade
    return fee


def calculate_slippage_cost(
    entry_price_raw: float,
    entry_price_filled: float,
    exit_price_raw: float,
    exit_price_filled: float,
    qty: float,
    side: str,
) -> float:
    """
    Calculate the dollar cost of slippage.

    For long:  slippage_cost = (entry_filled - entry_raw) * qty + (exit_raw - exit_filled) * qty
    For short: slippage_cost = (entry_raw - entry_filled) * qty + (exit_filled - exit_raw) * qty
    """
    if side == "long":
        entry_slip = (entry_price_filled - entry_price_raw) * qty
        exit_slip = (exit_price_raw - exit_price_filled) * qty
    else:
        entry_slip = (entry_price_raw - entry_price_filled) * qty
        exit_slip = (exit_price_filled - exit_price_raw) * qty
    return entry_slip + exit_slip
