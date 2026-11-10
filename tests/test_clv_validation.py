"""Regression tests for paired closing-line value observations."""

import numpy as np
import pytest

from nba_sim.backtest.metrics import closing_line_value


def test_clv_uses_only_complete_recorded_pairs():
    assert closing_line_value([0.4, np.nan, 0.6], [0.5, 0.8, 0.55]) == pytest.approx(0.025)
    assert closing_line_value([np.nan], [0.5]) is None
    assert closing_line_value([], []) is None


@pytest.mark.parametrize("bet,close", [
    ([0.4, 0.5], [0.6]),
    ([[0.4]], [[0.5]]),
    ([0.4], [[0.5]]),
    ([1.1], [0.5]),
    ([-0.1], [0.5]),
    ([0.5], [1.1]),
    ([0.5], [-0.1]),
    ([np.inf], [0.5]),
    ([0.5], [-np.inf]),
])
def test_clv_rejects_invalid_or_unpaired_quotes(bet, close):
    with pytest.raises(ValueError):
        closing_line_value(bet, close)
