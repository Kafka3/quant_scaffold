"""Tests for performance_metrics module."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import math

import numpy as np
import pandas as pd
import pytest

from backtest.performance_metrics import calculate_sharpe_ratio


def test_sharpe_ratio_rising():
    equity = pd.Series([100, 101, 102, 103, 104])
    sr = calculate_sharpe_ratio(equity, timeframe_minutes=5, risk_free_rate_annual=0.0)
    assert sr is not None and not math.isnan(sr)
    assert sr > 0


def test_sharpe_ratio_falling():
    equity = pd.Series([100, 99, 98, 97, 96])
    sr = calculate_sharpe_ratio(equity, timeframe_minutes=5, risk_free_rate_annual=0.0)
    assert sr is not None and not math.isnan(sr)
    assert sr < 0


def test_sharpe_ratio_flat():
    equity = pd.Series([100, 100, 100, 100])
    sr = calculate_sharpe_ratio(equity, timeframe_minutes=5, risk_free_rate_annual=0.0)
    assert math.isnan(sr)


def test_sharpe_ratio_insufficient_data():
    equity = pd.Series([100])
    sr = calculate_sharpe_ratio(equity, timeframe_minutes=5, risk_free_rate_annual=0.0)
    assert math.isnan(sr)


def test_sharpe_ratio_no_error():
    # Should not raise even with zeros or nans
    equity = pd.Series([100, np.nan, 100, 100])
    sr = calculate_sharpe_ratio(equity, timeframe_minutes=5, risk_free_rate_annual=0.0)
    # All returns are 0 after dropna -> std=0 -> NaN; that's fine, just don't crash
    assert math.isnan(sr)
