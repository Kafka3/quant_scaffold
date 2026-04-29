"""Synthetic pivot tests verifying right-side rolling window semantics.

The critical bug being guarded against: the strict-mode pivot detection used
    rolled_right = series[::-1].rolling(window=right).max()[::-1].shift(-right)
which checks max(s[i+right : i+2*right]) instead of max(s[i+1 : i+1+right]),
effectively skipping the first `right` bars immediately to the right of the pivot.

The correct implementation is shift(-1):
    rolled_right = series[::-1].rolling(window=right).max()[::-1].shift(-1)
which checks max(s[i+1 : i+1+right]), i.e. bars [i+1, i+right].
"""

from pathlib import Path
import sys

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.divergence import _pivot_high, _pivot_low


def test_pivot_high_right_shift_one_semantics():
    """
    Verify that _pivot_high with strict=True uses shift(-1) so that
    a pivot is only marked when every bar in [i+1, i+right] is strictly lower.

    Synthetic data: a peak (value 10) at position 5, right=3.
    Positions 6,7,8 are all strictly lower than 10.
    Position 5 MUST be detected as a pivot.

    This was broken with the old shift(-right) which checked
    s[8:11] = [2,2,1] (passing) but also allowed s[9:12] = [2,1,0]
    to affect the pivot decision, which caused the wrong
    right-side window to be checked.
    """
    series = pd.Series([1, 2, 3, 4, 5, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    left, right = 3, 3

    pivots = _pivot_high(series, left, right, strict=True)

    # Position 5 (value=10) must be a pivot: left side [2,3,4] < 10, right side [9,8,7] < 10
    assert pivots.iloc[5], (
        f"Position 5 (value=10) should be a pivot high with left={left}, right={right}. "
        f"Detected pivots: {list(series[pivots].index)}"
    )

    # Position 18 *right after* pivot must NOT be a pivot (it's lower)
    for pos in [6, 7, 8]:
        assert not pivots.iloc[pos], (
            f"Position {pos} (value={series.iloc[pos]}) should NOT be a pivot"
        )

    # Last bar must not be a pivot (insufficient right-side bars)
    for pos in range(len(series) - right, len(series)):
        assert not pivots.iloc[pos], (
            f"Position {pos} should not be a pivot (fewer than {right} bars to the right)"
        )


def test_pivot_low_right_shift_one_semantics():
    """
    Symmetric test for _pivot_low: a trough (value 1) at position 5, right=3.
    Positions 6,7,8 are all strictly higher than 1.
    Position 5 MUST be detected as a pivot low.
    """
    series = pd.Series([9, 8, 7, 6, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    left, right = 3, 3

    pivots = _pivot_low(series, left, right, strict=True)

    # Position 5 (value=1) must be a pivot low
    assert pivots.iloc[5], (
        f"Position 5 (value=1) should be a pivot low with left={left}, right={right}. "
        f"Detected pivots: {list(series[pivots].index)}"
    )

    # Positions immediately after pivot must NOT be pivots
    for pos in [6, 7, 8]:
        assert not pivots.iloc[pos], (
            f"Position {pos} (value={series.iloc[pos]}) should NOT be a pivot"
        )


def test_pivot_high_second_bar_on_right_not_skipped():
    """
    CRITICAL TEST: Verify that the right-side window includes
    bar [i+1] (the FIRST bar immediately after the pivot).

    Create a scenario where bar [i+1] has a value that is LOWER than
    the pivot value but HIGHER than bar [i+right]. If shift(-right)
    was used (the old bug), the window checked would miss bar [i+1],
    potentially making a false pivot.

    With the CORRECT shift(-1), the right-side window is s[i+1:i+1+right],
    which includes the critical first-right bar.
    """
    left, right = 2, 2

    # Bar i=2 (value=10) is a potential pivot.
    # Bar i+1=3 has value 8 (lower than 10, good)
    # Bar i+2=4 has value 5 (lower than 10, good)
    # This should be a clear pivot.
    series = pd.Series([3, 4, 10, 8, 5, 3, 2, 1])
    pivots = _pivot_high(series, left, right, strict=True)
    assert pivots.iloc[2], "Position 2 (value=10) should be a pivot high"

    # EDGE CASE: bar [i+1] has value equal to the pivot
    # The strict condition is series[i] > rolled_right, so equal values must NOT be a pivot
    series2 = pd.Series([3, 4, 10, 10, 5, 3, 2, 1])
    pivots2 = _pivot_high(series2, left, right, strict=True)
    assert not pivots2.iloc[2], (
        "Position 2 should NOT be a pivot because bar[i+1] has equal value"
    )

    # EDGE CASE: bar [i+1] is the highest of the right window
    # E.g. i=2 value=10, i+1=3 value=10.1 (higher!)
    # This must NOT be a pivot.
    series3 = pd.Series([3, 4, 10, 10.1, 9, 8, 7, 6])
    pivots3 = _pivot_high(series3, left, right, strict=True)
    assert not pivots3.iloc[2], (
        "Position 2 should NOT be a pivot because bar[i+1]=10.1 is higher"
    )


def test_pivot_low_first_bar_on_right_not_skipped():
    """
    Symmetric low-pivot edge case test.
    """
    left, right = 2, 2

    # Clear trough at position 2
    series = pd.Series([7, 6, 1, 3, 5, 7, 9])
    pivots = _pivot_low(series, left, right, strict=True)
    assert pivots.iloc[2], "Position 2 (value=1) should be a pivot low"

    # Bar [i+1] equals pivot value — should NOT be a pivot
    series2 = pd.Series([7, 6, 1, 1, 5, 7, 9])
    pivots2 = _pivot_low(series2, left, right, strict=True)
    assert not pivots2.iloc[2], "Position 2 should NOT be pivot because bar[i+1] equals it"

    # Bar [i+1] lower than pivot — should NOT be a pivot
    series3 = pd.Series([7, 6, 1, 0.5, 5, 7, 9])
    pivots3 = _pivot_low(series3, left, right, strict=True)
    assert not pivots3.iloc[2], "Position 2 should NOT be pivot because bar[i+1] is lower"


def test_pivot_high_right_edge_exclusion():
    """
    Verify that positions near the end of data with fewer than `right`
    bars to their right are correctly excluded from being pivots.
    """
    series = pd.Series([1, 2, 3, 4, 5, 10, 9, 8, 7, 6, 5, 4, 3])
    left, right = 3, 3

    pivots = _pivot_high(series, left, right, strict=True)

    # The last `right` positions (10, 11, 12) should NOT be pivots
    # even if they are local maxima, because there aren't enough
    # subsequent bars to confirm.
    for pos in range(len(series) - right, len(series)):
        assert not pivots.iloc[pos], (
            f"Position {pos} should not be a pivot (in right-edge exclusion zone)"
        )


def test_pivot_low_right_edge_exclusion():
    series = pd.Series([10, 9, 8, 7, 6, 1, 2, 3, 4, 5, 6, 7, 8])
    left, right = 3, 3

    pivots = _pivot_low(series, left, right, strict=True)

    for pos in range(len(series) - right, len(series)):
        assert not pivots.iloc[pos], (
            f"Position {pos} should not be a pivot low (in right-edge exclusion zone)"
        )


def test_pivot_high_detects_multiple_pivots():
    """
    With correct shift(-1), both peaks in a double-top pattern
    must be detected as pivots.
    """
    series = pd.Series([
        5, 6, 7, 8, 9, 10, 9, 8, 7, 8,
        9, 10, 9, 8, 7, 6, 5, 4, 3, 2,
    ])
    left, right = 3, 3
    pivots = _pivot_high(series, left, right, strict=True)

    # Position 5 (first peak, value=10) must be a pivot
    assert pivots.iloc[5], "Position 5 (value=10) should be first peak pivot"

    # Position 11 (second peak, value=10) must be a pivot
    assert pivots.iloc[11], "Position 11 (value=10) should be second peak pivot"

    # The valley between peaks should not be a pivot high
    assert not pivots.iloc[8], "Position 8 in the valley should not be a pivot high"


def test_flat_top_is_not_pivot():
    """
    A flat top (plateau) must NOT produce a pivot in strict mode,
    because the condition is series[i] > rolled_right, not >=.
    """
    series = pd.Series([1, 2, 10, 10, 10, 9, 8, 7])
    left, right = 2, 2
    pivots = _pivot_high(series, left, right, strict=True)
    # Neither of the plateau bars should be pivots
    for pos in range(2, 5):
        assert not pivots.iloc[pos], (
            f"Position {pos} is part of a flat top and should not be a strict pivot"
        )


def test_flat_bottom_is_not_pivot():
    """Symmetric: flat bottom should not be a pivot low in strict mode."""
    series = pd.Series([10, 9, 1, 1, 1, 2, 3, 4])
    left, right = 2, 2
    pivots = _pivot_low(series, left, right, strict=True)
    for pos in range(2, 5):
        assert not pivots.iloc[pos], (
            f"Position {pos} is part of a flat bottom and should not be a strict pivot"
        )


def test_pivot_matches_config_defaults():
    """
    Smoke test using default config values (left=3, right=3).
    A clear peak at position 10 with value 10.
    """
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, 25)
    trend = [i * 0.3 for i in range(25)]
    signal = pd.Series(trend + noise).rolling(3).mean().bfill().ffill()

    # Inject a clear pivot
    signal_padded = signal.copy()
    signal_padded.iloc[15] = signal.iloc[15:20].max() + 3.0

    left, right = 3, 3
    pivots = _pivot_high(signal_padded, left, right, strict=True)

    # At least one pivot should be detected
    assert pivots.any(), "At least one pivot should be detected in the modified signal"
