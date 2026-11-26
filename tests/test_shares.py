"""Play-type shares follow the player, the coach, and the team."""

import pandas as pd
import pytest

from nba_sim.constants import PLAY_TYPES
from nba_sim.features.prepare import _template_player
from nba_sim.features.shares import apply_play_shares, role_shares
from nba_sim.sim.specs import LeaguePriors, TeamModel


def _team(abbr: str, team_id: int) -> TeamModel:
    return TeamModel(
        team_id=team_id,
        abbr=abbr,
        name=abbr,
        off_ppp={play: 1.05 for play in PLAY_TYPES},
        off_poss={play: 0.0 for play in PLAY_TYPES},
        def_ppp={play: 1.05 for play in PLAY_TYPES},
        def_poss={play: 0.0 for play in PLAY_TYPES},
    )


def test_the_roll_man_spelling_counts_as_the_roll():
    league = LeaguePriors.flat()
    team = _team("phi", 1610612755)
    center = _template_player({"position": "C", "espn_id": "9", "player_name": "Joel Embiid"}, team, league, 34)
    rows = [
        {
            "season": "2025-26",
            "entity_type": "P",
            "entity_id": 9,
            "player_name": "Joel Embiid",
            "team_id": team.team_id,
            "side": "offensive",
            "play_type": "PRRollMan",
            "poss": 196,
            "poss_pct": 0.205,
            "ppp": 1.18,
            "fg_pct": 0.58,
            "ft_poss_pct": 0.22,
            "tov_poss_pct": 0.10,
            "gp": 38,
        }
    ]
    apply_play_shares([center], team, pd.DataFrame(rows), "2026-27")
    assert center.offense["PRRollman"].poss > 4


def test_the_lead_guard_handles_and_a_lower_guard_spots_up():
    lead = role_shares("G", 0, "okc")
    third = role_shares("G", 2, "okc")
    assert lead["PRBallHandler"] > third["PRBallHandler"] + 0.08
    assert third["Spotup"] > lead["Spotup"]
    assert abs(sum(lead.values()) - 1.0) < 1e-9


def test_kerr_spots_up_where_oklahoma_city_handles():
    kerr = role_shares("G", 0, "gsw")
    thunder = role_shares("G", 0, "okc")
    kerr_motion = kerr["Spotup"] + kerr["Cut"] + kerr["OffScreen"]
    thunder_motion = thunder["Spotup"] + thunder["Cut"] + thunder["OffScreen"]
    assert kerr_motion > thunder_motion
    assert thunder["PRBallHandler"] > kerr["PRBallHandler"]


def test_a_measured_player_keeps_his_possessions():
    league = LeaguePriors.flat()
    team = _team("phi", 1610612755)
    star = _template_player({"position": "G", "espn_id": "1", "player_name": "Tyrese Maxey"}, team, league, 34)
    wing = _template_player({"position": "G", "espn_id": "2", "player_name": "VJ Edgecombe"}, team, league, 32)
    rows = []
    for play, poss in (
        ("PRBallHandler", 900),
        ("Isolation", 280),
        ("Transition", 220),
        ("Spotup", 140),
    ):
        rows.append(
            {
                "season": "2025-26",
                "entity_type": "P",
                "entity_id": 101,
                "player_name": "Tyrese Maxey",
                "team_id": team.team_id,
                "side": "offensive",
                "play_type": play,
                "poss": poss,
                "poss_pct": 0.2,
                "ppp": 1.12,
                "fg_pct": 0.48,
                "ft_poss_pct": 0.14,
                "tov_poss_pct": 0.10,
            }
        )
    for play, pct in (
        ("Isolation", 0.08),
        ("Transition", 0.16),
        ("PRBallHandler", 0.22),
        ("PRRollman", 0.08),
        ("Postup", 0.06),
        ("Spotup", 0.18),
        ("Handoff", 0.06),
        ("Cut", 0.06),
        ("OffScreen", 0.04),
        ("OffRebound", 0.03),
        ("Misc", 0.03),
    ):
        rows.append(
            {
                "season": "2025-26",
                "entity_type": "T",
                "entity_id": team.team_id,
                "player_name": None,
                "team_id": team.team_id,
                "side": "offensive",
                "play_type": play,
                "poss": 1000,
                "poss_pct": pct,
                "ppp": 1.04,
                "fg_pct": 0.46,
                "ft_poss_pct": 0.08,
                "tov_poss_pct": 0.12,
            }
        )
    apply_play_shares([star, wing], team, pd.DataFrame(rows), "2026-27")
    assert star.offense["PRBallHandler"].poss > wing.offense["PRBallHandler"].poss * 2
    assert star.offense["PRBallHandler"].ppp > 1.08
    star_total = sum(skill.poss for skill in star.offense.values())
    wing_total = sum(skill.poss for skill in wing.offense.values())
    star_handler = star.offense["PRBallHandler"].poss / star_total
    wing_handler = wing.offense["PRBallHandler"].poss / wing_total
    assert star_handler > wing_handler + 0.15


def test_a_thin_action_keeps_the_season_before_it():
    league = LeaguePriors.flat()
    team = _team("ind", 1610612754)
    guard = _template_player({"position": "G", "espn_id": "1", "player_name": "Tyrese Haliburton"}, team, league, 34)
    rows = []
    for season, poss, gp, ppp in (("2025-26", 40, 10, 1.40), ("2024-25", 200, 70, 1.00)):
        rows.append(
            {
                "season": season,
                "entity_type": "P",
                "entity_id": 1,
                "player_name": "Tyrese Haliburton",
                "team_id": team.team_id,
                "side": "offensive",
                "play_type": "Isolation",
                "poss": poss,
                "poss_pct": 0.1,
                "ppp": ppp,
                "fg_pct": 0.45,
                "ft_poss_pct": 0.1,
                "tov_poss_pct": 0.12,
                "gp": gp,
            }
        )
    apply_play_shares([guard], team, pd.DataFrame(rows), "2026-27")
    assert guard.offense["Isolation"].poss == pytest.approx(240 / 80)


def test_a_full_slate_action_replaces_last_year():
    league = LeaguePriors.flat()
    team = _team("ind", 1610612754)
    guard = _template_player({"position": "G", "espn_id": "1", "player_name": "Tyrese Haliburton"}, team, league, 34)
    rows = []
    for season, poss, gp, ppp in (("2026-27", 200, 4, 1.20), ("2025-26", 400, 70, 0.90)):
        rows.append(
            {
                "season": season,
                "entity_type": "P",
                "entity_id": 1,
                "player_name": "Tyrese Haliburton",
                "team_id": team.team_id,
                "side": "offensive",
                "play_type": "Isolation",
                "poss": poss,
                "poss_pct": 0.1,
                "ppp": ppp,
                "fg_pct": 0.45,
                "ft_poss_pct": 0.1,
                "tov_poss_pct": 0.12,
                "gp": gp,
            }
        )
    apply_play_shares([guard], team, pd.DataFrame(rows), "2026-27")
    assert guard.offense["Isolation"].poss == pytest.approx(200 / 4)


def test_a_thin_new_season_action_counts_a_little_more_than_last_year():
    from nba_sim.features.shares import _measured_with_prior

    rows = []
    for season, poss, ppp in (("2026-27", 20, 1.40), ("2025-26", 200, 1.00)):
        rows.append(
            {
                "season": season,
                "entity_type": "P",
                "entity_id": 1,
                "player_name": "Tester",
                "team_id": 1,
                "side": "offensive",
                "play_type": "Spotup",
                "poss": poss,
                "ppp": ppp,
                "fg_pct": 0.45,
                "ft_poss_pct": 0.08,
                "tov_poss_pct": 0.12,
                "gp": 2 if season == "2026-27" else 70,
            }
        )
    found = _measured_with_prior(pd.DataFrame(rows), "Tester", "2026-27")
    plain = (20 * 1.40 + 200 * 1.00) / 220
    assert found["Spotup"]["poss"] == pytest.approx(220)
    assert found["Spotup"]["ppp"] > plain
    assert found["Spotup"]["ppp"] < 1.08


def test_synergy_cache_retains_source_frame_identity():
    from nba_sim.features.shares import _CACHE_OWNERS, _measured_player

    frame = pd.DataFrame([{
        "season": "2026-27", "entity_type": "P", "entity_id": 987654,
        "player_name": "Cache Identity Example", "side": "offensive",
        "play_type": "Isolation", "poss": 90, "gp": 3, "ppp": 1.1,
    }])
    measured = _measured_player(frame, "Cache Identity Example", "2026-27")
    assert measured["Isolation"]["poss"] == 90
    assert _CACHE_OWNERS[id(frame)] is frame
