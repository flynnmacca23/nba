"""Validation parity for direct night-minute draws."""

import numpy as np
import pytest

from nba_sim.features.minutes import MinutePlayer, draw_night_minutes


@pytest.mark.parametrize("n_sims", [0, -1, 1.5, True, np.bool_(False)])
def test_draw_night_rejects_invalid_simulation_counts(n_sims):
    with pytest.raises(ValueError, match="n_sims"):
        draw_night_minutes([MinutePlayer(1, 30.0, 2.0, True)], n_sims, np.random.default_rng(1))


@pytest.mark.parametrize("mean,std", [
    (float("nan"), 1.0),
    (float("inf"), 1.0),
    (-1.0, 1.0),
    (30.0, float("nan")),
    (30.0, float("inf")),
    (30.0, -1.0),
])
@pytest.mark.parametrize("available", [True, False])
def test_draw_night_rejects_invalid_priors_even_for_absences(mean, std, available):
    with pytest.raises(ValueError, match="priors"):
        draw_night_minutes([MinutePlayer(1, mean, std, available)], 2, np.random.default_rng(1))


def test_draw_night_all_unavailable_returns_zero_minutes():
    players = [MinutePlayer(1, 30.0, 2.0, False)]
    actual = draw_night_minutes(players, 3, np.random.default_rng(1))
    np.testing.assert_array_equal(actual, np.zeros((3, 1)))
