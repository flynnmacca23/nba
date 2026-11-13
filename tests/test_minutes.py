import numpy as np
import pandas as pd
import pytest

from nba_sim.features.minutes import MinutePlayer, allocate_minutes, draw_night_minutes, fill_night
from datetime import date

from nba_sim.features.prepare import (
    _hygiene_current_slate_minutes,
    _is_current_slate,
    _minutes,
    _recent_fg_multiplier,
)
from nba_sim.report import _minutes_table, _player_bins
from nba_sim.sim.engine import SimResult
from nba_sim.sim.specs import GameContext, GameSpec, LeaguePriors, PlayerModel, TeamModel


def test_minutes_sum_to_a_full_game_and_an_absence_is_redistributed():
    players = [MinutePlayer(i, 30.0, 0.0, i != 7) for i in range(8)]
    minutes = allocate_minutes(players, n_sims=40, seed=3)
    assert minutes.shape == (40, 8)
    np.testing.assert_allclose(minutes.sum(axis=1), 240.0, atol=1e-6)
    assert np.all(minutes[:, 7] == 0)
    # Seven available players split 240. With the injured 30 minutes removed,
    # each of the others plays more than the raw 30.
    assert minutes[0, 0] > 30


def test_minutes_draw_is_seeded():
    players = [MinutePlayer(i, 28 - i, 2.0, True) for i in range(8)]
    first = allocate_minutes(players, n_sims=25, seed=9)
    second = allocate_minutes(players, n_sims=25, seed=9)
    other = allocate_minutes(players, n_sims=25, seed=10)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_the_night_fills_240_from_the_largest_roles():
    means = np.array([36, 34, 32, 30, 28, 24, 20, 18, 16, 14], dtype=float)
    filled = fill_night(means)
    assert filled.sum() == pytest.approx(240.0)
    assert filled[0] == pytest.approx(36.0)
    assert filled[2] == pytest.approx(32.0)
    assert filled[-1] < 14.0
    light = np.array([30.0, 24.0, 18.0])
    assert np.allclose(fill_night(light), light)
    players = [MinutePlayer(index, float(value), 0.0, True) for index, value in enumerate(means)]
    players.append(MinutePlayer(99, 40.0, 0.0, False))
    drawn = draw_night_minutes(players, n_sims=6, rng=np.random.default_rng(3))
    np.testing.assert_allclose(drawn.sum(axis=1), 240.0, atol=1e-6)
    assert np.all(drawn[:, -1] == 0.0)
    assert drawn[0, 0] == pytest.approx(36.0)


def test_a_hot_bench_draw_does_not_steal_a_star_slot():
    # Prior: star 36, then nine 28s (over 240). A soft star and a hot last man
    # would hand the last man 44 first if fill ordered by the noise. Prior order
    # keeps the star ahead, so the last man takes only what is left.
    means = np.array([36.0] + [28.0] * 9, dtype=float)
    targets = fill_night(means)
    swapped = np.full(len(means), 28.0)
    swapped[0] = 8.0
    swapped[-1] = 44.0
    prior_order = fill_night(np.broadcast_to(swapped, (1, len(means))), order_by=targets)
    draw_order = fill_night(np.broadcast_to(swapped, (1, len(means))))
    assert prior_order[0, 0] == pytest.approx(8.0)
    assert draw_order[0, -1] == pytest.approx(44.0)
    assert prior_order[0, -1] < 20.0
    np.testing.assert_allclose(prior_order.sum(axis=1), 240.0, atol=1e-6)


def test_recent_games_set_the_minutes_prior():
    logs = pd.DataFrame(
        {
            "entity_type": ["P", "P", "P"],
            "player_id": [7, 7, 7],
            "date": ["2026-10-01", "2026-10-03", "2026-10-05"],
            "min": [30, 36, 24],
        }
    )
    mean, _std, source, sample = _minutes(logs, 7, 18)
    assert source == "Last 12 games"
    assert sample == 3
    assert mean == 30
    assert _minutes(None, 7, 18)[2] == "Season average"
    assert _minutes(None, 7, 0)[2] == "Depth chart"
    with_zeros = pd.DataFrame(
        {
            "entity_type": ["P", "P", "P", "P"],
            "player_id": [7, 7, 7, 7],
            "date": ["2026-10-01", "2026-10-03", "2026-10-05", "2026-10-07"],
            "min": [30, 0, 36, 0],
        }
    )
    mean_played, _std, source, sample = _minutes(with_zeros, 7, 18)
    assert source == "Last 12 games"
    assert sample == 2
    assert mean_played == 33.0


def test_distributions_include_the_deeper_rotation():
    players = [
        PlayerModel(
            player_id=i,
            name=f"P{i}",
            team_id=1,
            position="G",
            minutes_mean=float(30 - i),
            minutes_std=0.0,
            minutes_source="Depth chart",
            minutes_base=float(30 - i),
            minutes_slot=i + 1,
        )
        for i in range(10)
    ]
    n = 6
    stats = {}
    for player in players:
        zeros = np.zeros(n)
        stats[player.player_id] = {key: zeros.copy() for key in ("pts", "reb", "ast", "tpm", "pra")}
        stats[player.player_id]["minutes"] = np.full(n, player.minutes_mean)
    result = SimResult(
        home_points=np.zeros(n),
        away_points=np.zeros(n),
        player_stats=stats,
        home_ids=[player.player_id for player in players],
        away_ids=[],
        second_chance_home=np.zeros(n),
        second_chance_away=np.zeros(n),
        n_sims=n,
        seed=1,
    )
    team = TeamModel(team_id=1, abbr="phi", name="Philadelphia 76ers")
    other = TeamModel(team_id=2, abbr="cle", name="Cleveland Cavaliers")
    spec = GameSpec(
        game_id="1",
        date="2026-10-22",
        home=team,
        away=other,
        home_players=players,
        away_players=[],
        context=GameContext(),
        league=LeaguePriors.flat(),
    )
    names = {row["player"] for row in _player_bins(spec, result)}
    assert "P8" in names and "P9" in names
    table = _minutes_table(spec, result)
    assert len(table) == 10
    assert table[0]["how"] == "Depth chart, slot 1"
    assert abs(sum(row["scaled"] for row in table) - 240) < 0.2


def test_finished_season_hygiene_is_skipped():
    # 2025-26 is a finished log-roster season under the current SETTINGS slate.
    assert _is_current_slate(date(2026, 1, 15)) is False
    players = [
        PlayerModel(
            player_id=1,
            name="Phantom",
            team_id=1,
            position="G",
            minutes_mean=22.0,
            minutes_sample=0,
            minutes_depth=18.0,
            available=True,
        ),
        PlayerModel(
            player_id=2,
            name="Star",
            team_id=1,
            position="C",
            minutes_mean=36.0,
            minutes_sample=40,
            available=True,
        ),
    ]
    before = [float(player.minutes_mean) for player in players]
    _hygiene_current_slate_minutes(players, date(2026, 1, 15))
    assert [float(player.minutes_mean) for player in players] == before


def test_stod_rescale_protects_star_minutes():
    # One 36-minute star + eleven 24-minute depth men = 300 > 280. Depth
    # stays under the 32+ protect band so only they absorb the cut.
    players = [
        PlayerModel(
            player_id=1,
            name="Star",
            team_id=1,
            position="C",
            minutes_mean=36.0,
            minutes_appearance_avg=36.0,
            minutes_sample=10,
            available=True,
        )
    ]
    for index in range(2, 13):
        players.append(
            PlayerModel(
                player_id=index,
                name=f"Depth{index}",
                team_id=1,
                position="G",
                minutes_mean=24.0,
                minutes_sample=2,
                available=True,
            )
        )
    from nba_sim.features.prepare import _rescale_available_to_240

    _rescale_available_to_240(players, only_if_over=280.0, protect_stars=True)
    assert players[0].minutes_mean == pytest.approx(36.0)
    assert sum(float(player.minutes_mean) for player in players) == pytest.approx(240.0, abs=0.5)


def test_recent_fg_multiplier_tracks_cold_scoring_per_fga():
    rows = []
    for i in range(10):
        rows.append(
            {
                "entity_type": "P",
                "player_id": 7,
                "date": f"2026-10-{i + 1:02d}",
                "min": 30,
                "pts": 8 if i >= 6 else 18,
                "fga": 12,
            }
        )
    mult, n = _recent_fg_multiplier(pd.DataFrame(rows), 7, season_eff=1.5, season="2026-27")
    assert n == 10
    assert mult < 0.95


def test_all_unavailable_roster_never_resurrects_injured_players():
    players = [MinutePlayer(i, 30.0, 2.0, False) for i in range(8)]
    minutes = allocate_minutes(players, n_sims=20, seed=3)
    assert minutes.shape == (20, 8)
    np.testing.assert_array_equal(minutes, np.zeros((20, 8)))


def test_empty_roster_has_zero_minutes_without_division_by_zero():
    minutes = allocate_minutes([], n_sims=4, seed=1)
    assert minutes.shape == (4, 0)


@pytest.mark.parametrize("n_sims", [0, -1, 1.5, True])
def test_allocate_minutes_rejects_invalid_simulation_counts(n_sims):
    with pytest.raises(ValueError, match="n_sims"):
        allocate_minutes([MinutePlayer(1, 30.0, 2.0, True)], n_sims=n_sims)


@pytest.mark.parametrize("target", [0, -240, float("nan"), float("inf")])
def test_allocate_minutes_rejects_invalid_target(target):
    with pytest.raises(ValueError, match="target"):
        allocate_minutes([MinutePlayer(1, 30.0, 2.0, True)], n_sims=2, target=target)


@pytest.mark.parametrize("mean,std", [(float("nan"), 1), (30, float("inf")), (-1, 1), (30, -1)])
def test_allocate_minutes_rejects_invalid_priors(mean, std):
    with pytest.raises(ValueError, match="priors"):
        allocate_minutes([MinutePlayer(1, mean, std, True)], n_sims=2)
