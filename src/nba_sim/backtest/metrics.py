"""Accuracy and betting metrics for a walk-forward window."""

from __future__ import annotations

import numpy as np


def _binary_forecasts(prob, outcome) -> tuple[np.ndarray, np.ndarray]:
    """Validate paired forecasts before computing proper scoring rules."""
    p = np.asarray(prob, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if p.ndim != 1 or y.ndim != 1 or len(p) != len(y):
        raise ValueError("Scores require equally sized one-dimensional arrays")
    if not (np.isfinite(p).all() and np.isfinite(y).all()):
        raise ValueError("Scores require finite probabilities and outcomes")
    if ((p < 0) | (p > 1)).any() or (~np.isin(y, [0.0, 1.0])).any():
        raise ValueError("Scores require probabilities in [0, 1] and binary outcomes")
    return p, y


def log_loss(prob, outcome) -> float:
    p, y = _binary_forecasts(prob, outcome)
    if len(y) == 0:
        return float("nan")
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(prob, outcome) -> float:
    p, y = _binary_forecasts(prob, outcome)
    if len(y) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def flat_roi(wins, decimal_odds) -> float:
    wins = np.asarray(wins, dtype=bool)
    prices = np.asarray(decimal_odds, dtype=float)
    if len(wins) == 0:
        return float("nan")
    profit = np.where(wins, prices - 1.0, -1.0)
    return float(profit.mean())


def max_drawdown(profits) -> float:
    series = np.asarray(profits, dtype=float)
    if len(series) == 0:
        return 0.0
    equity = np.cumsum(series)
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


def calibration_table(prob, outcome, bins: int = 8) -> list[dict]:
    p = np.asarray(prob, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if p.ndim != 1 or y.ndim != 1 or len(p) != len(y):
        raise ValueError("Calibration requires equally sized one-dimensional arrays")
    if not isinstance(bins, (int, np.integer)) or isinstance(bins, (bool, np.bool_)) or bins < 1:
        raise ValueError("Calibration bins must be a positive integer")
    if not (np.isfinite(p).all() and np.isfinite(y).all()):
        raise ValueError("Calibration inputs must be finite")
    if ((p < 0) | (p > 1)).any() or (~np.isin(y, [0.0, 1.0])).any():
        raise ValueError("Calibration requires probabilities in [0, 1] and binary outcomes")
    if len(p) == 0:
        return []
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p >= left) & (p < right if right < 1 else p <= right)
        if not mask.any():
            continue
        rows.append(
            {
                "bin_low": float(left),
                "bin_high": float(right),
                "n": int(mask.sum()),
                "mean_prob": float(p[mask].mean()),
                "frequency": float(y[mask].mean()),
            }
        )
    return rows


def closing_line_value(bet_implied, close_implied) -> float | None:
    bet = np.asarray(bet_implied, dtype=float)
    close = np.asarray(close_implied, dtype=float)
    valid = np.isfinite(bet) & np.isfinite(close)
    if not valid.any():
        return None
    return float(np.mean(close[valid] - bet[valid]))
