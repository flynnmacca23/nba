"""Fail-closed validation for realized betting returns."""

import numpy as np
import pytest

from nba_sim.backtest.metrics import flat_roi


def test_roi_uses_recorded_decimal_prices_and_binary_settlements():
    assert flat_roi([1, 0], [2.5, 1.8]) == pytest.approx(0.25)


def test_empty_roi_is_nan():
    assert np.isnan(flat_roi([], []))


@pytest.mark.parametrize(
    "outcomes, prices",
    [
        ([1, 0], [2.0]),
        ([[1]], [[2.0]]),
        ([1, 0], [[2.0, 2.0]]),
        ([1, np.nan], [2.0, 2.0]),
        ([1, np.inf], [2.0, 2.0]),
        ([1, 0.5], [2.0, 2.0]),
        ([2], [2.0]),
        ([1], [np.nan]),
        ([1], [np.inf]),
        ([1], [1.0]),
        ([1], [0.9]),
    ],
)
def test_roi_rejects_invalid_inputs(outcomes, prices):
    with pytest.raises(ValueError):
        flat_roi(outcomes, prices)
