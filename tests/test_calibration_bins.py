"""Regression coverage for vectorized calibration bucket assignment."""

import numpy as np
import pytest

from nba_sim.backtest.metrics import calibration_table


def test_calibration_boundaries_and_empty_buckets():
    rows = calibration_table([0.0, 0.25, 0.5, 0.75, 1.0], [0, 1, 0, 1, 1], bins=4)
    assert [row["n"] for row in rows] == [1, 1, 1, 2]
    assert [row["frequency"] for row in rows] == pytest.approx([0, 1, 0, 1])
    assert rows[-1]["mean_prob"] == pytest.approx(0.875)


def test_calibration_matches_reference_mask_implementation():
    rng = np.random.default_rng(42)
    probabilities = np.r_[rng.random(1000), 0.0, 1.0, 0.3]
    outcomes = rng.integers(0, 2, len(probabilities))
    bins = 10
    edges = np.linspace(0, 1, bins + 1)
    expected = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (probabilities >= low) & (
            probabilities < high if high < 1 else probabilities <= high
        )
        if selected.any():
            expected.append((int(selected.sum()), probabilities[selected].mean(), outcomes[selected].mean()))
    actual = calibration_table(probabilities, outcomes, bins=bins)
    assert len(actual) == len(expected)
    for row, (count, probability, frequency) in zip(actual, expected):
        assert row["n"] == count
        assert row["mean_prob"] == pytest.approx(probability)
        assert row["frequency"] == pytest.approx(frequency)


@pytest.mark.parametrize("bins", [0, -1, True, 2.5])
def test_calibration_rejects_invalid_bin_counts(bins):
    with pytest.raises(ValueError):
        calibration_table([0.5], [1], bins=bins)


def test_calibration_empty_inputs():
    assert calibration_table([], [], bins=8) == []
