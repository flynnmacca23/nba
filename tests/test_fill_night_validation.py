"""Fail-closed validation for minutes filling inputs."""

import numpy as np
import pytest

from nba_sim.features.minutes import fill_night


@pytest.mark.parametrize("target", [0.0, -1.0, float("nan"), float("inf")])
def test_fill_night_rejects_invalid_target(target):
    with pytest.raises(ValueError, match="target"):
        fill_night(np.array([30.0, 20.0]), target=target)


@pytest.mark.parametrize("means", [
    [30.0, float("nan")],
    [30.0, float("inf")],
    [30.0, float("-inf")],
])
def test_fill_night_rejects_nonfinite_means(means):
    with pytest.raises(ValueError, match="means"):
        fill_night(np.asarray(means))


@pytest.mark.parametrize("ranking", [
    [30.0, float("nan")],
    [30.0, float("inf")],
    [30.0, float("-inf")],
])
def test_fill_night_rejects_nonfinite_ranking(ranking):
    with pytest.raises(ValueError, match="ranking"):
        fill_night(np.array([30.0, 20.0]), order_by=np.asarray(ranking))


def test_fill_night_preserves_valid_batched_allocations():
    result = fill_night(np.array([[150.0, 130.0], [30.0, 20.0]]))
    np.testing.assert_allclose(result, [[150.0, 90.0], [30.0, 20.0]])
