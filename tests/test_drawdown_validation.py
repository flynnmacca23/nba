"""Regression tests for drawdown accounting and validation."""

import numpy as np
import pytest

from nba_sim.backtest.metrics import max_drawdown


def test_first_bet_loss_counts_as_drawdown():
    assert max_drawdown([-1.0]) == pytest.approx(1.0)
    assert max_drawdown([-1.0, 0.5, -0.25]) == pytest.approx(1.0)


def test_drawdown_uses_running_high_water_mark():
    assert max_drawdown([2.0, -1.0, -2.0, 3.0]) == pytest.approx(3.0)
    assert max_drawdown([1.0, 2.0]) == pytest.approx(0.0)
    assert max_drawdown([]) == pytest.approx(0.0)


@pytest.mark.parametrize("profits", [
    [np.nan], [np.inf], [-np.inf], [[1.0, -1.0]], 1.0,
])
def test_drawdown_rejects_invalid_profit_series(profits):
    with pytest.raises(ValueError, match="Drawdown requires"):
        max_drawdown(profits)
