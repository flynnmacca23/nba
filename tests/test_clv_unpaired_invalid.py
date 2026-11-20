"""CLV quote validity must not depend on the availability of its pair."""

import numpy as np
import pytest

from nba_sim.backtest.metrics import closing_line_value


@pytest.mark.parametrize(
    "bet, close",
    [
        ([1.1], [np.nan]),
        ([-0.01], [np.nan]),
        ([np.nan], [1.01]),
        ([np.nan], [-0.01]),
        ([0.4, np.nan], [0.5, 1.5]),
        ([0.4, -0.1], [0.5, np.nan]),
    ],
)
def test_clv_rejects_invalid_quote_with_missing_counterpart(bet, close):
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        closing_line_value(bet, close)


def test_clv_ignores_genuinely_missing_matched_quotes():
    assert closing_line_value([0.4, np.nan], [0.5, 0.6]) == pytest.approx(0.1)
    assert closing_line_value([np.nan], [0.5]) is None
