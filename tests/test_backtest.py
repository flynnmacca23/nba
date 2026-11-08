from datetime import date

import pandas as pd
import pytest

from nba_sim.backtest.walkforward import empty_player_props, run_backtest
from nba_sim.features.prepare import (
    _lineup_book,
    _logs_before,
    _minutes,
    _roster_for_game,
    _roster_on_night,
    build_players,
    build_team_model,
    league_priors,
    season_for,
)
from nba_sim.sim.specs import LeaguePriors, TeamModel
from nba_sim.teams import TEAM_ID_BY_ABBR


def test_a_past_season_uses_the_players_who_were_on_the_team():
    players = pd.DataFrame(
        [
            {"season": "2025-26", "player_name": "Bam Adebayo", "team_abbr": "mia", "min": 34.0, "position": "C"},
            {"season": "2025-26", "player_name": "Jimmy Butler", "team_abbr": "mia", "min": 32.0, "position": "F"},
            {"season": "2026-27", "player_name": "Giannis Antetokounmpo", "team_abbr": "mia", "min": 30.0, "position": "F"},
        ]
    )
    chart = pd.DataFrame(
        [
            {
                "team_abbr": "mia",
                "player_name": "Giannis Antetokounmpo",
                "position": "F",
                "depth_rank": 0,
                "roster_status": "Active",
            }
        ]
    )
    past = _roster_for_game("2025-26", chart, players)
    past_names = list(past.loc[past["team_abbr"] == "mia", "player_name"])
    assert past_names == ["Bam Adebayo", "Jimmy Butler"]
    current = _roster_for_game("2026-27", chart, players)
    assert list(current["player_name"]) == ["Giannis Antetokounmpo"]


def test_membership_runs_from_the_first_game_through_the_last():
    cle = TEAM_ID_BY_ABBR["cle"]
    lac = TEAM_ID_BY_ABBR["lac"]
    players = pd.DataFrame(
        [
            {"season": "2025-26", "player_id": 1, "player_name": "Star", "team_abbr": "cle", "position": "G"},
            {"season": "2025-26", "player_id": 2, "player_name": "Callup", "team_abbr": "cle", "position": "G"},
            {"season": "2025-26", "player_id": 3, "player_name": "Visitor", "team_abbr": "cle", "position": "G"},
        ]
    )
    logs = pd.DataFrame(
        [
            {"season": "2025-26", "entity_type": "P", "player_id": 1, "team_id": cle, "date": "2025-10-22", "min": 34, "game_id": "a"},
            {"season": "2025-26", "entity_type": "P", "player_id": 1, "team_id": cle, "date": "2026-01-15", "min": 40, "game_id": "a2"},
            {"season": "2025-26", "entity_type": "P", "player_id": 1, "team_id": cle, "date": "2026-04-10", "min": 30, "game_id": "b"},
            {"season": "2025-26", "entity_type": "P", "player_id": 2, "team_id": cle, "date": "2026-04-01", "min": 36, "game_id": "c"},
            {"season": "2025-26", "entity_type": "P", "player_id": 2, "team_id": cle, "date": "2026-04-08", "min": 36, "game_id": "d"},
            {"season": "2025-26", "entity_type": "P", "player_id": 3, "team_id": lac, "date": "2026-01-02", "min": 28, "game_id": "e"},
            {"season": "2025-26", "entity_type": "P", "player_id": 3, "team_id": lac, "date": "2026-01-05", "min": 28, "game_id": "f"},
        ]
    )

    def names(on: date, team: str) -> list[str]:
        roster = _roster_on_night("2025-26", None, players, logs, on)
        return list(roster.loc[roster["team_abbr"] == team, "player_name"])

    assert names(date(2025, 10, 22), "cle") == ["Star"]
    april = names(date(2026, 4, 5), "cle")
    assert april[0] == "Star"
    assert "Callup" in april
    assert "Visitor" not in april
    assert names(date(2026, 4, 12), "cle") == []
    assert names(date(2026, 1, 3), "lac") == ["Visitor"]
    assert names(date(2026, 1, 8), "lac") == []


def test_the_chart_drops_a_player_whose_games_are_elsewhere():
    lal = TEAM_ID_BY_ABBR["lal"]
    phi = TEAM_ID_BY_ABBR["phi"]
    players = pd.DataFrame(
        [
            {"season": "2026-27", "player_id": 10, "player_name": "Wing", "team_abbr": "lal"},
            {"season": "2026-27", "player_id": 11, "player_name": "Rookie", "team_abbr": "phi"},
            {"season": "2026-27", "player_id": 12, "player_name": "Arrival", "team_abbr": "phi"},
        ]
    )
    chart = pd.DataFrame(
        [
            {"team_abbr": "phi", "player_name": "Wing", "position": "F", "depth_rank": 0, "roster_status": "Active"},
            {"team_abbr": "phi", "player_name": "Rookie", "position": "G", "depth_rank": 4, "roster_status": "Active"},
            {"team_abbr": "phi", "player_name": "Arrival", "position": "F", "depth_rank": 3, "roster_status": "Active"},
            {"team_abbr": "lal", "player_name": "Wing", "position": "F", "depth_rank": 0, "roster_status": "Active"},
        ]
    )
    logs = pd.DataFrame(
        [
            {"season": "2026-27", "entity_type": "P", "player_id": 10, "team_id": lal, "date": "2026-10-20", "min": 34, "game_id": "1"},
            {"season": "2026-27", "entity_type": "P", "player_id": 10, "team_id": lal, "date": "2026-10-22", "min": 33, "game_id": "2"},
            {"season": "2026-27", "entity_type": "P", "player_id": 12, "team_id": phi, "date": "2026-12-01", "min": 20, "game_id": "3"},
        ]
    )
    roster = _roster_on_night("2026-27", chart, players, logs, date(2026, 10, 24))
    phi_names = set(roster.loc[roster["team_abbr"] == "phi", "player_name"])
    lal_names = set(roster.loc[roster["team_abbr"] == "lal", "player_name"])
    assert "Wing" not in phi_names
    assert "Rookie" in phi_names
    assert "Arrival" not in phi_names
    assert "Wing" in lal_names


def test_two_games_count_as_a_share_of_the_team_schedule():
    team_id = TEAM_ID_BY_ABBR["cle"]
    team = TeamModel(team_id=team_id, abbr="cle", name="Cleveland")
    roster = pd.DataFrame(
        [
            {
                "team_abbr": "cle",
                "espn_id": "11",
                "player_name": "Regular",
                "position": "G",
                "depth_rank": 0,
                "roster_status": "Active",
            },
            {
                "team_abbr": "cle",
                "espn_id": "12",
                "player_name": "Callup",
                "position": "G",
                "depth_rank": 1,
                "roster_status": "Active",
            },
        ]
    )
    rows = []
    for day in range(1, 11):
        game_id = f"g{day}"
        played_on = f"2026-01-{day:02d}"
        rows.append(
            {
                "season": "2025-26",
                "entity_type": "T",
                "team_id": team_id,
                "player_id": 0,
                "date": played_on,
                "min": 240,
                "game_id": game_id,
            }
        )
        rows.append(
            {
                "season": "2025-26",
                "entity_type": "P",
                "team_id": team_id,
                "player_id": 11,
                "date": played_on,
                "min": 34,
                "game_id": game_id,
            }
        )
    rows.extend(
        [
            {
                "season": "2025-26",
                "entity_type": "P",
                "team_id": team_id,
                "player_id": 12,
                "date": "2026-01-09",
                "min": 36,
                "game_id": "g9",
            },
            {
                "season": "2025-26",
                "entity_type": "P",
                "team_id": team_id,
                "player_id": 12,
                "date": "2026-01-10",
                "min": 36,
                "game_id": "g10",
            },
        ]
    )
    built = build_players(
        "cle",
        "2025-26",
        team,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        pd.DataFrame(rows),
        None,
        LeaguePriors.flat(),
        roster,
        date(2026, 1, 11),
    )
    by_name = {player.name: player for player in built}
    callup = by_name["Callup"]
    regular = by_name["Regular"]
    # Top depth slots keep a coach floor under thin court noise; his games stay the source.
    assert regular.minutes_mean >= 34.0
    assert regular.minutes_mean == pytest.approx(max(34.0, float(regular.minutes_depth) * 0.95), abs=0.05)
    assert regular.minutes_source == "His games"
    assert callup.minutes_team_games == pytest.approx(10)
    assert callup.minutes_appearances == pytest.approx(2)
    assert callup.minutes_mean == pytest.approx(max(36.0, float(callup.minutes_depth) * 0.95), abs=0.05)
    assert callup.minutes_source == "His games"


def test_missing_line_file_explains_itself(tmp_path):
    result = run_backtest(path=tmp_path / "missing.csv")
    assert result["status"] == "missing_lines"
    assert "nba_2008-2026.csv" in result["message"] or "Historical team lines" in result["message"]
    assert result["player_props"]["status"] == "empty"


def test_player_props_without_lines_are_an_empty_result():
    result = empty_player_props(None)
    assert result["n_bets"] == 0
    assert "player prop" in result["message"].lower()


def test_walk_forward_scores_team_markets_and_leaves_props_empty():
    rows = []
    for day in range(1, 7):
        rows.append(
            {
                "date": date(2026, 1, day),
                "home": "den",
                "away": "mia",
                "home_score": 120,
                "away_score": 108,
                "home_spread": -4.5,
                "total": 225.0,
                "moneyline_home": -180,
                "moneyline_away": 155,
            }
        )
    result = run_backtest(lines=pd.DataFrame(rows), last_n=1, n_sims=80, seed=2)
    assert result["status"] == "ok"
    assert result["n_games"] == 1
    assert set(result["metrics"]) >= {"moneyline", "spread", "total", "note"}
    assert result["player_props"]["status"] == "empty"
    assert result["metrics"]["moneyline"]["log_loss"] is not None
    assert result["validation"]["replay"] == "synthetic_team_only"
    assert result["validation"]["input_provenance"] == "unverified_historical_lines"


def test_a_players_own_pick_and_roll_sample_survives_shrinkage():
    team_id = 1610612743
    play_rows = []
    for entity, entity_id, ppp, poss in (
        ("T", team_id, 1.00, 2000),
        ("T", 1610612748, 0.95, 2000),
        ("P", 15, 1.30, 500),
    ):
        play_rows.append(
            {
                "season": "2025-26",
                "entity_type": entity,
                "entity_id": entity_id,
                "team_id": team_id,
                "side": "offensive",
                "play_type": "PRBallHandler",
                "poss": poss,
                "poss_pct": 0.4,
                "ppp": ppp,
                "fg_pct": 0.48,
                "efg_pct": 0.52,
                "ft_poss_pct": 0.08,
                "tov_poss_pct": 0.12,
                "score_poss_pct": 0.5,
                "percentile": 80,
                "pts": ppp * poss,
                "gp": 70,
            }
        )
    plays = pd.DataFrame(play_rows)
    players = pd.DataFrame(
        [
            {
                "season": "2025-26",
                "player_id": 15,
                "player_name": "Handler",
                "team_id": team_id,
                "team_abbr": "den",
                "gp": 70,
                "min": 34,
                "pts": 27,
                "reb": 8,
                "ast": 6,
                "stl": 1,
                "blk": 0.6,
                "position": "G",
                "oreb_pct": 0.02,
                "dreb_pct": 0.1,
                "ft_pct": 0.8,
                "usg_pct": 0.3,
            }
        ]
    )
    teams = pd.DataFrame(
        [
            {
                "season": "2025-26",
                "team_id": team_id,
                "team_abbr": "den",
                "gp": 70,
                "pace": 100,
                "off_rating": 114,
                "def_rating": 112,
                "oreb_pct": 0.28,
                "dreb_pct": 0.73,
            }
        ]
    )
    league = league_priors(plays)
    # League is the possession-weighted team rate, below the player's 1.30.
    assert league.ppp_by_type["PRBallHandler"] < 1.05
    team = build_team_model("den", "2025-26", teams, plays, None, None, league)
    built = build_players(
        "den",
        "2025-26",
        team,
        players,
        plays,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        league,
    )
    assert len(built) == 1
    assert built[0].offense["PRBallHandler"].ppp > 1.20
    assert season_for(date(2026, 10, 22)) == "2026-27"


def test_a_missing_dashboard_uses_the_team_play_sample():
    team_id = 1610612743
    plays = pd.DataFrame(
        [
            {
                "season": "2025-26",
                "entity_type": "T",
                "entity_id": team_id,
                "team_id": team_id,
                "side": side,
                "play_type": "Spotup",
                "poss": 4000,
                "poss_pct": 0.4,
                "ppp": ppp,
                "fg_pct": 0.48,
                "efg_pct": 0.52,
                "ft_poss_pct": 0.08,
                "tov_poss_pct": 0.12,
                "score_poss_pct": 0.5,
                "percentile": 60,
                "pts": ppp * 4000,
                "gp": 78,
            }
            for side, ppp in (("offensive", 1.08), ("defensive", 0.94))
        ]
    )
    league = league_priors(plays)
    team = build_team_model("den", "2025-26", None, plays, None, None, league)
    assert team.off_rating == pytest.approx(108.0)
    assert team.def_rating == pytest.approx(94.0)
    assert team.games == pytest.approx(78.0)


def test_last_season_dashboard_fills_pace_and_the_glass():
    team_id = 1610612743
    teams = pd.DataFrame(
        [
            {
                "season": "2025-26",
                "team_id": team_id,
                "team_abbr": "den",
                "gp": 82,
                "pace": 98.4,
                "off_rating": 118.0,
                "def_rating": 112.5,
                "oreb_pct": 0.31,
                "dreb_pct": 0.72,
            },
            {
                "season": "2026-27",
                "team_id": team_id,
                "team_abbr": "den",
                "gp": 4,
                "pace": 110.0,
                "off_rating": 100.0,
                "def_rating": 120.0,
                "oreb_pct": 0.20,
                "dreb_pct": 0.80,
            },
        ]
    )
    league = LeaguePriors.flat()
    carried = build_team_model("den", "2026-27", teams.iloc[:1], None, None, None, league)
    assert carried.pace == pytest.approx(98.4)
    assert carried.oreb_pct == pytest.approx(0.31)
    assert carried.def_rating == pytest.approx(112.5)
    assert carried.games == pytest.approx(82)
    early = build_team_model("den", "2026-27", teams, None, None, None, league)
    assert early.pace == pytest.approx(98.4 * 14 / 18 + 110.0 * 4 / 18)
    assert early.off_rating == pytest.approx(118.0 * 14 / 18 + 100.0 * 4 / 18)
    assert early.games == pytest.approx(82)


def test_recent_minutes_stop_before_the_game():
    logs = pd.DataFrame(
        {
            "entity_type": ["P", "P", "P", "P"],
            "player_id": [7, 7, 7, 7],
            "date": pd.to_datetime(["2026-10-01", "2026-10-10", "2026-10-18", "2026-10-22"]),
            "min": [10, 12, 14, 40],
        }
    )
    mean, _std, source, count = _minutes(_logs_before(logs, date(2026, 10, 22)), 7, 30)
    assert count == 3
    assert source == "Last 12 games"
    assert mean == pytest.approx(12)


def _five(ids: tuple[int, ...], season: str, minutes: float, pace: float) -> dict:
    return {
        "season": season,
        "team_id": 1,
        "group_id": "-".join(str(player) for player in ids),
        "min": minutes,
        "pace": pace,
    }


def test_this_season_lineups_count_more_and_a_short_group_stays_behind():
    last = (1, 2, 3, 4, 5)
    fresh = (6, 7, 8, 9, 10)
    rows = pd.DataFrame(
        [
            _five(last, "2025-26", 400, 100),
            _five(fresh, "2026-27", 18, 110),
            _five((11, 12, 13, 14, 15), "2026-27", 4, 108),
        ]
    )
    _pace, minutes, shared = _lineup_book(rows, "2026-27", 1)
    assert (11, 12, 13, 14, 15) not in minutes
    assert minutes[last] > minutes[fresh]
    played = pd.DataFrame([_five(last, "2025-26", 400, 100), _five(fresh, "2026-27", 40, 110)])
    pace, minutes, shared = _lineup_book(played, "2026-27", 1)
    assert minutes[fresh] > minutes[last]
    assert shared[(6, 7)] > shared[(1, 2)]
    assert pace[fresh] == pytest.approx(110)
    continued = pd.DataFrame([_five(last, "2025-26", 400, 100), _five(last, "2026-27", 40, 112)])
    pace, _minutes, _shared = _lineup_book(continued, "2026-27", 1)
    trust = 40.0 / 76.0
    assert pace[last] == pytest.approx((1.0 - trust) * 100 + trust * 112)
    assert pace[last] > 106


def test_same_day_results_never_affect_other_pregame_ratings():
    from nba_sim.backtest.walkforward import _pregame_ratings

    base = {
        "date": "2026-01-02", "home_spread": -2.5, "total": 220,
        "moneyline_home": -130, "moneyline_away": 110,
    }
    first = {**base, "home": "den", "away": "mia", "home_score": 180, "away_score": 90}
    second = {**base, "home": "den", "away": "bos", "home_score": 100, "away_score": 101}
    games = pd.DataFrame([first, second])
    rated = _pregame_ratings(games)
    assert rated["pred_margin"].tolist() == pytest.approx([2.2, 2.2])
    assert rated["pred_total"].tolist() == pytest.approx([224.0, 224.0])
    reversed_rated = _pregame_ratings(games.iloc[::-1])
    assert reversed_rated["pred_margin"].tolist() == pytest.approx([2.2, 2.2])
    assert reversed_rated["pred_total"].tolist() == pytest.approx([224.0, 224.0])


def test_next_day_ratings_are_independent_of_same_day_row_order():
    from nba_sim.backtest.walkforward import _pregame_ratings

    base = {
        "home_spread": -2.5, "total": 220,
        "moneyline_home": -130, "moneyline_away": 110,
    }
    first = {**base, "date": "2026-01-02", "home": "den", "away": "mia",
             "home_score": 180, "away_score": 90}
    second = {**base, "date": "2026-01-02", "home": "den", "away": "bos",
              "home_score": 100, "away_score": 101}
    tomorrow = {**base, "date": "2026-01-03", "home": "den", "away": "bos",
                "home_score": 99, "away_score": 98}
    forward = _pregame_ratings(pd.DataFrame([first, second, tomorrow]))
    reversed_day = _pregame_ratings(pd.DataFrame([second, first, tomorrow]))
    assert forward.iloc[-1]["pred_margin"] == pytest.approx(reversed_day.iloc[-1]["pred_margin"])
    assert forward.iloc[-1]["pred_total"] == pytest.approx(reversed_day.iloc[-1]["pred_total"])
    assert forward.iloc[-1]["pred_margin"] != pytest.approx(2.2)


def test_invalid_game_dates_fail_instead_of_disappearing_from_backtest():
    from nba_sim.backtest.walkforward import _pregame_ratings

    with pytest.raises(ValueError, match="invalid game date"):
        _pregame_ratings(pd.DataFrame([{
            "date": None, "home": "den", "away": "mia", "home_score": 110,
            "away_score": 100, "home_spread": -3, "total": 220,
            "moneyline_home": -140, "moneyline_away": 120,
        }]))


def test_player_prop_scoring_requires_pregame_provenance():
    props = pd.DataFrame([{"market": "player_points", "model_prob": 0.7, "outcome": 1}])
    result = empty_player_props(props)
    assert result["status"] == "unevaluable"
    assert result["n_bets"] == 0
    assert "provenance" in result["message"]


def test_player_prop_scoring_accepts_verified_timing_and_rejects_late_prices():
    props = pd.DataFrame([{
        "market": "player_points", "game_id": "g1", "source": "archived_book",
        "snapshot_id": "s1", "model_version": "v1",
        "available_at": "2026-01-02T17:00:00Z",
        "prediction_at": "2026-01-02T17:10:00Z",
        "game_start_at": "2026-01-02T18:00:00Z",
        "model_prob": 0.7, "outcome": 1,
    }])
    scored = empty_player_props(props)
    assert scored["status"] == "ok"
    assert scored["n_bets"] == 1
    assert scored["brier"] == pytest.approx(0.09)
    props.loc[0, "available_at"] = "2026-01-02T17:30:00Z"
    assert empty_player_props(props)["status"] == "unevaluable"
    props.loc[0, "available_at"] = "2026-01-02T17:00:00"
    assert empty_player_props(props)["status"] == "unevaluable"


def test_historical_simulation_seed_is_stable_across_order_and_processes():
    from types import SimpleNamespace
    from nba_sim.backtest.walkforward import _game_seed

    game = SimpleNamespace(date="2026-01-02", home="den", away="mia")
    other = SimpleNamespace(date="2026-01-02", home="bos", away="mia")
    assert _game_seed(17, game) == _game_seed(17, game)
    assert _game_seed(17, game) != _game_seed(17, other)
    assert _game_seed(17, game) != _game_seed(18, game)
    assert 0 <= _game_seed(17, game) < 2**32


def test_replay_rejects_empty_windows_and_invalid_simulation_counts():
    import pytest
    from nba_sim.backtest.walkforward import run_backtest

    # Validate before attempting to load external historical line files.
    for kwargs in (
        {"last_n": 0}, {"last_n": -1}, {"last_n": True},
        {"last_n": 1.5}, {"n_sims": 0}, {"n_sims": -10},
        {"n_sims": True}, {"n_sims": 2.5},
        {"seed": -1}, {"seed": True}, {"seed": 1.5},
    ):
        with pytest.raises(ValueError):
            run_backtest(**kwargs)


@pytest.mark.parametrize(
    "prob,outcome,bins",
    [
        ([0.2, 0.8], [1.0], 8),
        ([[0.2]], [1.0], 8),
        ([float("nan")], [1.0], 8),
        ([float("inf")], [1.0], 8),
        ([1.1], [1.0], 8),
        ([-0.1], [0.0], 8),
        ([0.5], [0.5], 8),
        ([0.5], [1.0], 0),
        ([0.5], [1.0], 2.5),
        ([0.5], [1.0], True),
    ],
)
def test_calibration_rejects_invalid_inputs(prob, outcome, bins):
    from nba_sim.backtest.metrics import calibration_table

    with pytest.raises(ValueError):
        calibration_table(prob, outcome, bins=bins)


def test_calibration_accepts_empty_inputs_and_probability_one():
    from nba_sim.backtest.metrics import calibration_table

    assert calibration_table([], []) == []
    rows = calibration_table([0.0, 1.0], [0, 1], bins=2)
    assert [row["n"] for row in rows] == [1, 1]
    assert rows[-1]["frequency"] == 1.0
