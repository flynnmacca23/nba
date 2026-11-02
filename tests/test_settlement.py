"""Regression coverage for historical market pushes."""
from types import SimpleNamespace

import numpy as np

from nba_sim.backtest.walkforward import _grade, _metrics


def _game(total: float):
    return SimpleNamespace(date="2026-01-01", home="den", away="mia",
        home_score=120, away_score=105, home_spread=-4.5,
        total=total, moneyline_home=-180, moneyline_away=155)


def _result():
    return SimpleNamespace(home_points=np.array([120., 130., 115.]),
        away_points=np.array([105., 100., 110.]))


def test_exact_total_push_is_not_a_losing_bet():
    import pandas as pd
    rows = _grade(_game(225.0), _result())
    total = next(row for row in rows if row["bet_market"] == "total")
    assert total["settlement"] == "push"
    assert total["profit"] == 0.0
    assert not total["bet"]
    assert np.isnan(total["won"])
    metrics = _metrics(pd.DataFrame(rows))
    assert metrics["total"]["n"] == 0
    assert metrics["total"]["log_loss"] is None


def test_non_push_total_is_still_graded():
    rows = _grade(_game(224.5), _result())
    total = next(row for row in rows if row["bet_market"] == "total")
    assert total.get("settlement") != "push"
    assert np.isfinite(total["won"])
