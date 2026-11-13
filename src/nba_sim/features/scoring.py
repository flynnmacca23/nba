"""Each player's own usage, shot profile, free throws, and glass.

The slate season is the line. A new season is a change of role, so the
games already played count a little more than they would in the middle of
a year. A short sample still stays close to the season before it, and each
new game moves that blend. A name missing from this season keeps the last
line he has. Minutes still decide who is on the floor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_sim.constants import AST_PCT_FROM_PER_GAME, DEFAULT_ZONE_FG, DEFAULT_ZONE_FREQ
from nba_sim.names import norm_person
from nba_sim.sim.specs import PlayerModel
from nba_sim.warehouse import Warehouse

_SEASONS = ("2023-24", "2024-25", "2025-26", "2026-27")
# Five games ties last season. Forty games is this season's line.
_BLEND_GAMES = 8.0
_FULL_SEASON_GAMES = 40.0
_BLEND_KEYS = (
    "usg",
    "ft_pct",
    "oreb_pct",
    "dreb_pct",
    "ts",
    "ast",
    "ast_pct",
    "stl",
    "blk",
    "pf",
    "min",
    "fga",
    "fg3a",
    "fg_pct",
    "fg3_pct",
    "fta",
    "three_fga",
    "paint_pts",
    "mid_pts",
)
# A 25% usage player in 36 minutes uses about 19 possessions. Per-game
# possessions divided by 75 is the claim when no scoring line exists.
_POSS_PER_USAGE = 75.0
_LEAGUE_TS = 0.578
# Used only when he has no play mix to imply a free-throw rate.
_LEAGUE_FTA = 0.25
# Eight games ties the residual to 1. Sixty games is his own rate.
_FT_PRIOR_GAMES = 8.0
_FT_FULL_GAMES = 60.0
_THREE_PLAYS = ("Spotup", "OffScreen", "Handoff")
_RIM_PLAYS = ("PRRollman", "Postup", "Cut", "OffRebound")


_SCORING_SEASON: dict[tuple, dict[str, dict]] = {}
_SCORING_TABLE: dict[tuple, dict[str, dict]] = {}


def _scoring_feed_stamp(warehouse: Warehouse, season: str) -> tuple[int, ...]:
    stamps = []
    for measure in ("Base", "Advanced", "Scoring"):
        path = warehouse.path(f"raw_player_{measure}_{season}")
        try:
            stamps.append(path.stat().st_mtime_ns if path.exists() else 0)
        except OSError:
            stamps.append(0)
    return tuple(stamps)


def scoring_table(warehouse: Warehouse | None = None, through: str | None = None) -> dict[str, dict]:
    """One row per player through the slate season. A short year leans on the one before it."""
    warehouse = warehouse or Warehouse()
    seasons = tuple(season for season in _SEASONS if through is None or season <= through)
    key = (str(warehouse.root), through or "", tuple(_scoring_feed_stamp(warehouse, season) for season in seasons))
    cached = _SCORING_TABLE.get(key)
    if cached is not None:
        return cached
    found: dict[str, dict] = {}
    for season in seasons:
        for name, row in _scoring_season(warehouse, season).items():
            found[name] = _blend_scoring(found.get(name), row)
    _SCORING_TABLE[key] = found
    if len(_SCORING_TABLE) > 8:
        oldest = next(iter(_SCORING_TABLE))
        if oldest != key:
            _SCORING_TABLE.pop(oldest, None)
    return found


def _blend_scoring(older: dict | None, newer: dict | None) -> dict | None:
    """This year's per-game line gains weight with each game. A full season stands alone."""
    if newer is None:
        return older
    if older is None:
        return newer
    games = float(newer.get("gp") or 0.0)
    if games >= _FULL_SEASON_GAMES:
        return newer
    trust = games / (games + _BLEND_GAMES) if games > 0 else 0.0
    if trust <= 0:
        return older
    blended = dict(newer)
    for key in _BLEND_KEYS:
        left = older.get(key)
        right = newer.get(key)
        left_ok = isinstance(left, (int, float)) and np.isfinite(left)
        right_ok = isinstance(right, (int, float)) and np.isfinite(right)
        if left_ok and right_ok:
            blended[key] = (1.0 - trust) * float(left) + trust * float(right)
        elif left_ok:
            blended[key] = left
    blended["gp"] = games
    return blended


def _scoring_season(warehouse: Warehouse, season: str) -> dict[str, dict]:
    """Minutes-weighted line for one season. An empty feed contributes nothing."""
    key = (str(warehouse.root), season, _scoring_feed_stamp(warehouse, season))
    cached = _SCORING_SEASON.get(key)
    if cached is not None:
        return cached
    base = warehouse.read(f"raw_player_Base_{season}")
    advanced = warehouse.read(f"raw_player_Advanced_{season}")
    if base is None or base.empty or advanced is None or advanced.empty:
        _SCORING_SEASON[key] = {}
        return {}
    scoring = warehouse.read(f"raw_player_Scoring_{season}")
    base = _ids(base)
    advanced = _ids(advanced)
    keep = [
        column
        for column in ("PLAYER_ID", "TEAM_ID", "USG_PCT", "OREB_PCT", "DREB_PCT", "TS_PCT", "AST_PCT")
        if column in advanced.columns
    ]
    merged = base.merge(advanced[keep], on=["PLAYER_ID", "TEAM_ID"], how="left")
    if scoring is not None and not scoring.empty:
        scoring = _ids(scoring)
        extra = [
            column
            for column in ("PLAYER_ID", "TEAM_ID", "PCT_FGA_3PT", "PCT_PTS_PAINT", "PCT_PTS_2PT_MR")
            if column in scoring.columns
        ]
        merged = merged.merge(scoring[extra], on=["PLAYER_ID", "TEAM_ID"], how="left")
    found: dict[str, dict] = {}
    for _player_id, group in merged.groupby("PLAYER_ID"):
        record = _collapse(group)
        name_key = norm_person(record.get("name"))
        if not name_key:
            continue
        previous = found.get(name_key)
        if previous is None or record["minutes"] > previous["minutes"]:
            found[name_key] = record
    _SCORING_SEASON[key] = found
    if len(_SCORING_SEASON) > 16:
        oldest = next(iter(_SCORING_SEASON))
        if oldest != key:
            _SCORING_SEASON.pop(oldest, None)
    return found


def apply_player_scoring(
    players: list[PlayerModel],
    table: dict[str, dict] | None = None,
    season: str | None = None,
) -> None:
    """Write this player's scoring line onto his model and keep his play mix."""
    rows = table if table is not None else scoring_table(through=season)
    for player in players:
        if player.minutes_mean <= 0:
            continue
        row = rows.get(norm_person(player.name))
        if row:
            _write_box(player, row)
        else:
            player.usg = _usage_from_poss(player)
            _zones_from_mix(player)
            _glass_from_mix(player)


def apply_season_minutes(players: list[PlayerModel]) -> None:
    """Move the depth slot by the share of the team's games he has played.

    The slot stands until that share is real. Six games out of seventy-six
    leave it where the depth chart put him. A player who has been in most of
    the team's games takes the average of the games he played.
    """
    for player in players:
        if not player.available:
            continue
        on_roster = int(player.minutes_slot) > 0
        depth = float(player.minutes_depth)
        team_games = float(player.minutes_team_games)
        games = float(player.minutes_appearances)
        average = float(player.minutes_appearance_avg)
        if team_games <= 0.0 or games <= 0.0 or average <= 0.0:
            if on_roster:
                player.minutes_mean = depth
                player.minutes_base = depth
                player.minutes_source = "Depth chart"
            continue
        share = min(1.0, games / team_games)
        prior = depth if on_roster else float(player.minutes_mean)
        player.minutes_mean = share * average + (1.0 - share) * prior
        player.minutes_base = player.minutes_mean
        player.minutes_source = "Season average" if share >= 0.5 else "Depth chart"


def _ids(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("PLAYER_ID", "TEAM_ID"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("int64")
    return frame


def _collapse(group: pd.DataFrame) -> dict:
    minutes = pd.to_numeric(group.get("MIN"), errors="coerce").fillna(0.0)
    games = pd.to_numeric(group.get("GP"), errors="coerce").fillna(0.0)
    weight = (minutes * games).to_numpy(dtype=float)
    if not np.isfinite(weight).any() or float(np.nansum(weight)) <= 0:
        weight = np.ones(len(group), dtype=float)
    name_row = group.iloc[int(np.nanargmax(weight))]

    def average(column: str) -> float | None:
        if column not in group.columns:
            return None
        values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(values) & np.isfinite(weight) & (weight > 0)
        if not ok.any():
            return None
        return float(np.average(values[ok], weights=weight[ok]))

    return {
        "name": str(name_row.get("PLAYER_NAME") or ""),
        "usg": average("USG_PCT"),
        "ft_pct": average("FT_PCT"),
        "oreb_pct": average("OREB_PCT"),
        "dreb_pct": average("DREB_PCT"),
        "ts": average("TS_PCT"),
        "ast": average("AST"),
        "ast_pct": average("AST_PCT"),
        "stl": average("STL"),
        "blk": average("BLK"),
        "pf": average("PF"),
        "min": average("MIN"),
        "gp": float(games.sum()),
        "fga": average("FGA"),
        "fg3a": average("FG3A"),
        "fg_pct": average("FG_PCT"),
        "fg3_pct": average("FG3_PCT"),
        "fta": average("FTA"),
        "three_fga": average("PCT_FGA_3PT"),
        "paint_pts": average("PCT_PTS_PAINT"),
        "mid_pts": average("PCT_PTS_2PT_MR"),
        "minutes": float(np.nansum(weight)),
    }


def _write_box(player: PlayerModel, row: dict) -> None:
    usg = row.get("usg")
    if usg is not None and np.isfinite(usg) and usg > 0:
        player.usg = float(np.clip(usg, 0.06, 0.45))
    else:
        player.usg = _usage_from_poss(player)
    if row.get("ft_pct") is not None:
        player.ft_pct = float(np.clip(row["ft_pct"], 0.40, 0.95))
    if row.get("oreb_pct") is not None:
        player.oreb_pct = float(np.clip(row["oreb_pct"], 0.004, 0.20))
    if row.get("dreb_pct") is not None:
        player.dreb_pct = float(np.clip(row["dreb_pct"], 0.02, 0.40))
    minutes = float(row.get("min") or 0.0)
    ast_pct = row.get("ast_pct")
    if ast_pct is not None and np.isfinite(ast_pct):
        player.ast_pct = float(np.clip(ast_pct, 0.0, 0.65))
    elif minutes > 8:
        per_game = float(row.get("ast") or 0.0)
        player.ast_pct = float(np.clip(per_game * AST_PCT_FROM_PER_GAME, 0.0, 0.65))
    if minutes > 8:
        player.ast_share = max(0.2, float(row.get("ast") or 0.0) * 36.0 / minutes)
        player.stl_weight = max(0.2, float(row.get("stl") or 0.0) * 36.0 / minutes)
        player.blk_weight = max(0.1, float(row.get("blk") or 0.0) * 36.0 / minutes)
        fouls = float(row.get("pf") or 0.0)
        if fouls > 0:
            player.foul_rate = (fouls * 36.0 / minutes) / 3.2
    games = float(row.get("gp") or 0.0)
    if games > 0:
        player.sample_games = games
    if minutes > 8 and games >= 8:
        player.minutes_season = minutes
        player.minutes_season_games = games
    three = row.get("three_fga")
    fga = float(row.get("fga") or 0.0)
    fg3a = float(row.get("fg3a") or 0.0)
    if three is None and fga > 0.5:
        three = fg3a / fga
    if three is not None and np.isfinite(three) and player.zone_freq == DEFAULT_ZONE_FREQ:
        player.zone_freq, player.zone_fg = _zones(
            float(three),
            row.get("fg_pct"),
            row.get("fg3_pct"),
            fga,
            fg3a,
            row.get("paint_pts"),
            row.get("mid_pts"),
        )
    if not _has_sample(player):
        if three is not None and np.isfinite(three):
            _tilt_mix(player, float(three))
        _scale_efficiency(player, row.get("ts"))
    _set_ft_draw(
        player,
        float(row.get("fga") or 0.0),
        float(row.get("fta") or 0.0),
        float(row.get("gp") or 0.0),
    )


def _has_sample(player: PlayerModel) -> bool:
    return any(skill.poss_pct > 0 for skill in player.offense.values())


def _usage_from_poss(player: PlayerModel) -> float:
    total = sum(max(skill.poss, 0.0) for skill in player.offense.values())
    if total <= 0:
        return 0.12
    return float(np.clip(total / _POSS_PER_USAGE, 0.07, 0.42))


def _tilt_mix(player: PlayerModel, three_rate: float) -> None:
    """Shot profile sets how much of an unmeasured role is a spot-up or a rim action."""
    total = sum(max(skill.poss, 0.0) for skill in player.offense.values())
    if total <= 0:
        return
    three_now = _group_poss(player, _THREE_PLAYS)
    rim_now = _group_poss(player, _RIM_PLAYS)
    other_share = max(0.0, total - three_now - rim_now) / total
    want_three = float(np.clip(three_rate * 0.90, 0.08, 0.62))
    want_rim = float(np.clip(0.72 - three_rate * 0.85, 0.10, 0.72))
    budget = max(0.0, 1.0 - other_share)
    if want_three + want_rim > budget > 0:
        fit = budget / (want_three + want_rim)
        want_three *= fit
        want_rim *= fit
    _scale_group(player, _THREE_PLAYS, three_now, want_three * total)
    _scale_group(player, _RIM_PLAYS, rim_now, want_rim * total)


def _group_poss(player: PlayerModel, plays: tuple[str, ...]) -> float:
    total = 0.0
    for play in plays:
        skill = player.offense.get(play)
        if skill is not None:
            total += max(skill.poss, 0.0)
    return total


def _scale_group(player: PlayerModel, plays: tuple[str, ...], now: float, target: float) -> None:
    if now <= 1e-6:
        return
    factor = target / now
    for play in plays:
        skill = player.offense.get(play)
        if skill is not None:
            skill.poss *= factor


def _scale_efficiency(player: PlayerModel, ts: float | None) -> None:
    if ts is None or not np.isfinite(ts) or ts <= 0:
        return
    scale = float(np.clip(ts / _LEAGUE_TS, 0.88, 1.12))
    for skill in player.offense.values():
        skill.ppp = float(np.clip(skill.ppp * scale, 0.70, 1.50))


def _implied_fta_per_fga(player: PlayerModel) -> float:
    """Free throws per shot already inside his play mix.

    A shooting-foul possession is that action's free-throw frequency. What is
    left after turnovers is a shot. The foul itself is two free throws.
    """
    weight = 0.0
    ft = 0.0
    tov = 0.0
    for skill in player.offense.values():
        poss = max(float(skill.poss), 0.0)
        if poss <= 0:
            continue
        weight += poss
        ft += poss * float(skill.ft_poss_pct)
        tov += poss * float(skill.tov_poss_pct)
    if weight <= 1e-6:
        return _LEAGUE_FTA
    ft_rate = ft / weight
    tov_rate = tov / weight
    shots = max(1.0 - tov_rate - ft_rate, 0.20)
    return max((ft_rate * 2.0) / shots, 0.05)


def _set_ft_draw(player: PlayerModel, fga: float, fta: float, games: float = 0.0) -> None:
    """How far his free throws sit from the rate his own actions already imply.

    A full season stands on that residual. A short season stays near 1, so the
    play mix is not scaled a second time.
    """
    if fga <= 0.5 or fta <= 0:
        return
    raw = (fta / fga) / _implied_fta_per_fga(player)
    if games >= _FT_FULL_GAMES:
        trust = 1.0
    elif games > 0:
        trust = games / (games + _FT_PRIOR_GAMES)
    else:
        trust = 0.0
    mixed = (1.0 - trust) + trust * raw
    player.ft_draw = float(np.clip(mixed, 0.45, 1.85))


def _zones(
    three: float,
    fg_pct: float | None,
    fg3_pct: float | None,
    fga: float,
    fg3a: float,
    paint_pts: float | None,
    mid_pts: float | None,
) -> tuple[dict[str, float], dict[str, float]]:
    three = float(np.clip(three, 0.0, 0.85))
    two = 1.0 - three
    paint = float(paint_pts or 0.0)
    mid = float(mid_pts or 0.0)
    rim_of_twos = paint / (paint + mid) if paint + mid > 0.05 else 0.62
    freq = {
        "ra": two * rim_of_twos * 0.70,
        "paint": two * rim_of_twos * 0.30,
        "mid": two * (1.0 - rim_of_twos),
        "corner3": three * 0.30,
        "atb3": three * 0.70,
    }
    fg2 = _two_point_fg(fg_pct, fg3_pct, fga, fg3a)
    fg3 = float(np.clip(fg3_pct if fg3_pct is not None else DEFAULT_ZONE_FG["atb3"], 0.20, 0.50))
    if fg2 is None:
        made = dict(DEFAULT_ZONE_FG)
        made["corner3"] = float(np.clip(fg3 * 1.06, 0.22, 0.52))
        made["atb3"] = fg3
        return freq, made
    made = {
        "ra": float(np.clip(fg2 * 1.18, 0.48, 0.80)),
        "paint": float(np.clip(fg2 * 0.92, 0.32, 0.62)),
        "mid": float(np.clip(fg2 * 0.78, 0.28, 0.55)),
        "corner3": float(np.clip(fg3 * 1.06, 0.22, 0.52)),
        "atb3": fg3,
    }
    return freq, made


def _two_point_fg(fg_pct: float | None, fg3_pct: float | None, fga: float, fg3a: float) -> float | None:
    if fg_pct is None or fg3_pct is None or fga <= fg3a + 0.15:
        return None
    made = (fg_pct * fga - fg3_pct * fg3a) / (fga - fg3a)
    if not np.isfinite(made):
        return None
    return float(np.clip(made, 0.32, 0.78))


def _zones_from_mix(player: PlayerModel) -> None:
    total = sum(max(skill.poss, 0.0) for skill in player.offense.values())
    if total <= 0:
        return
    three = _group_poss(player, _THREE_PLAYS) / total
    rim = _group_poss(player, _RIM_PLAYS) / total
    rate = float(np.clip(0.12 + 0.70 * three - 0.28 * rim, 0.04, 0.72))
    player.zone_freq, player.zone_fg = _zones(rate, None, None, 0.0, 0.0, rim, three * 0.35)


def _glass_from_mix(player: PlayerModel) -> None:
    total = sum(max(skill.poss, 0.0) for skill in player.offense.values())
    if total <= 0:
        return
    skill = player.offense.get("OffRebound")
    share = 0.0 if skill is None else max(skill.poss, 0.0) / total
    player.oreb_pct = float(np.clip(0.012 + 0.50 * share, 0.008, 0.16))
