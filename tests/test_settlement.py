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


def test_spread_push_is_excluded_from_wins_and_bets():
    import pandas as pd
    game = _game(224.5)
    game.home_spread = -15.0
    spread = next(row for row in _grade(game, _result()) if row["bet_market"] == "spread")
    assert spread["settlement"] == "push"
    assert not spread["bet"]
    assert np.isnan(spread["won"])
    metrics = _metrics(pd.DataFrame([spread]))
    assert metrics["spread"]["n_pushes"] == 1
    assert metrics["spread"]["log_loss"] is None


def test_missing_side_specific_prices_never_create_fictional_roi():
    import pandas as pd
    rows = _grade(_game(224.5), _result())
    for market in ("spread", "total"):
        candidate = next(row for row in rows if row["bet_market"] == market)
        assert candidate["price_status"] == "missing"
        assert not candidate["bet"]
        assert np.isnan(candidate["american"])
    metrics = _metrics(pd.DataFrame(rows))
    for market in ("spread", "total"):
        assert metrics[market]["n"] == 0
        assert metrics[market]["n_unpriced"] == 1
        assert metrics[market]["roi"] is None


def test_recorded_side_prices_are_used_for_returns():
    import pandas as pd
    from nba_sim.markets.devig import american_to_decimal

    game = _game(224.5)
    game.spread_home_american = -115
    game.total_over_american = -105
    rows = _grade(game, _result())
    spread = next(row for row in rows if row["bet_market"] == "spread")
    total = next(row for row in rows if row["bet_market"] == "total")
    assert spread["american"] == -115
    assert total["american"] == -105
    assert spread["decimal_odds"] == american_to_decimal(-115)
    assert total["decimal_odds"] == american_to_decimal(-105)
    assert spread["bet"] and total["bet"]
    metrics = _metrics(pd.DataFrame(rows))
    assert metrics["spread"]["n"] == 1
    assert metrics["total"]["n"] == 1


def test_parsed_historical_lines_keep_explicit_market_prices():
    import pandas as pd
    from nba_sim.ingest.markets import parse_kaggle

    raw = pd.DataFrame([{
        "date": "2026-01-01", "home": "den", "away": "mia",
        "score_home": 120, "score_away": 105,
        "whos_favored": "home", "spread": 4.5, "total": 224.5,
        "moneyline_home": -180, "moneyline_away": 155,
        "spread_home_american": -115, "total_over_american": -105,
    }])
    parsed = parse_kaggle(raw)
    assert parsed.iloc[0]["spread_home_american"] == -115
    assert parsed.iloc[0]["total_over_american"] == -105


def test_moneyline_market_baseline_is_aligned_to_the_predicted_side():
    import pandas as pd
    from nba_sim.markets.devig import devig_american

    game = _game(224.5)
    rows = _grade(game, _result())
    moneyline = next(row for row in rows if row["bet_market"] == "moneyline")
    fair_home, _ = devig_american([-180, 155])
    assert moneyline["market_prob"] == fair_home
    metrics = _metrics(pd.DataFrame(rows))
    assert metrics["moneyline"]["market_brier"] is not None
    assert metrics["moneyline"]["market_log_loss"] is not None
    assert metrics["moneyline"]["brier_improvement_vs_market"] == (
        metrics["moneyline"]["market_brier"] - metrics["moneyline"]["brier"]
    )
