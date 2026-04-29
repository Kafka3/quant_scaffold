"""Tests for risk_model and cost_model."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from backtest.risk_model import calculate_position_size
from backtest.cost_model import apply_slippage, calculate_fees, calculate_slippage_cost


def test_calculate_position_size_basic():
    result = calculate_position_size(
        equity=100000,
        entry_price=100000,
        stop_price=99000,
        risk_per_trade_pct=0.005,
    )
    assert not result["skip_trade"]
    assert result["target_risk_amount"] == 500
    assert result["actual_risk_amount"] == 500
    assert result["target_risk_pct"] == 0.005
    assert result["actual_risk_pct"] == 0.005
    assert result["stop_distance"] == 1000
    assert result["qty"] == 0.5
    assert result["raw_qty"] == 0.5
    assert result["max_qty"] > 0.5
    assert result["cap_hit"] is False


def test_calculate_position_size_cap_hit():
    result = calculate_position_size(
        equity=100000,
        entry_price=50000,
        stop_price=40000,
        risk_per_trade_pct=0.20,
        max_position_value_pct=0.5,
        max_leverage=1.0,
    )
    assert not result["skip_trade"]
    assert result["cap_hit"] is True
    assert result["target_risk_amount"] == 20000
    assert result["actual_risk_amount"] < result["target_risk_amount"]
    assert result["actual_risk_pct"] < result["target_risk_pct"]


def test_skip_zero_stop_distance():
    result = calculate_position_size(
        equity=100000,
        entry_price=100000,
        stop_price=100000,
        risk_per_trade_pct=0.01,
    )
    assert result["skip_trade"]
    assert result["skip_reason"] == "invalid_stop_distance"


def test_skip_invalid_price():
    result = calculate_position_size(
        equity=100000,
        entry_price=0,
        stop_price=99000,
        risk_per_trade_pct=0.01,
    )
    assert result["skip_trade"]
    assert result["skip_reason"] == "invalid_price"


def test_skip_qty_below_min():
    result = calculate_position_size(
        equity=100,
        entry_price=100000,
        stop_price=99999,
        risk_per_trade_pct=0.005,
        min_qty=1.0,
    )
    assert result["skip_trade"]
    assert result["skip_reason"] == "qty_below_min"


def test_apply_slippage_long_entry():
    assert apply_slippage(100000, "long", "entry", 5.0) == 100005.0


def test_apply_slippage_long_exit():
    assert apply_slippage(100000, "long", "exit", 5.0) == 99995.0


def test_apply_slippage_short_entry():
    assert apply_slippage(100000, "short", "entry", 5.0) == 99995.0


def test_apply_slippage_short_exit():
    assert apply_slippage(100000, "short", "exit", 5.0) == 100005.0


def test_calculate_fees():
    fee = calculate_fees(
        entry_price=100000,
        exit_price=102000,
        qty=0.1,
        fee_rate=0.002,
        fixed_fee_per_trade=0.0,
    )
    expected = 100000 * 0.1 * 0.002 + 102000 * 0.1 * 0.002
    assert fee == pytest.approx(expected)


def test_calculate_slippage_cost_long():
    cost = calculate_slippage_cost(
        entry_price_raw=100000,
        entry_price_filled=100005,
        exit_price_raw=102000,
        exit_price_filled=101995,
        qty=0.1,
        side="long",
    )
    expected = (100005 - 100000) * 0.1 + (102000 - 101995) * 0.1
    assert cost == pytest.approx(expected)


def test_calculate_slippage_cost_short():
    cost = calculate_slippage_cost(
        entry_price_raw=100000,
        entry_price_filled=99995,
        exit_price_raw=98000,
        exit_price_filled=98005,
        qty=0.1,
        side="short",
    )
    expected = (100000 - 99995) * 0.1 + (98005 - 98000) * 0.1
    assert cost == pytest.approx(expected)
