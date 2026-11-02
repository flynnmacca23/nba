"""Build player and team models from the warehouse tables."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from nba_sim.config import SETTINGS
from nba_sim.constants import DEFAULT_ZONE_FG, DEFAULT_ZONE_FREQ, LEAGUE_RATING, PLAY_TYPES, canon_play
from nba_sim.features.coaches import coach_for, depth_minutes
from nba_sim.features.context import TeamTrip, build_context
from nba_sim.features.play_priors import position_key, position_rates
from nba_sim.features.scoring import apply_player_scoring
from nba_sim.features.shares import apply_play_shares
from nba_sim.features.shrink import FILE_PRIOR, LEAGUE_PRIOR, blend_ppp, shrink
from nba_sim.geo import load_arenas
from nba_sim.sim.specs import (
    GameSpec,
    LeaguePriors,
    MarketQuote,
    PlayerModel,
    PlaySkill,
    TeamModel,
)
from nba_sim.ingest.espn import injury_decision
from nba_sim.sim.adjust import _zone_ratio, play_shot_freq
from nba_sim.names import norm_person
from nba_sim.teams import ABBR_BY_TEAM_ID, TEAM_ID_BY_ABBR, normalize_team, team_name

from nba_sim.warehouse import Warehouse


def season_for(on: date) -> str:
    year = on.year if on.month >= 8 else on.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


def previous_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


def _seasons_back(season: str, depth: int = 4) -> list[str]:
    """This season, then the ones behind it. The newest row wins when several exist."""
    found = []
    current = str(season)
    for _ in range(depth):
        found.append(current)
        current = previous_season(current)
    return found


def _finite(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _rate(value: object, default: float) -> float:
    number = _finite(value, default)
    if number > 1.5:
        return number / 100.0
    return number


def _ppp(value: object, default: float = 1.02) -> float:
    number = _finite(value, default)
    if number > 3:
        return number / 100.0
    return number


def _weighted_ppp(frame: pd.DataFrame) -> float:
    if frame is None or frame.empty:
        return 1.02
    poss = pd.to_numeric(frame["poss"], errors="coerce").fillna(0)
    ppp = frame["ppp"].map(_ppp)
    total = float(poss.sum())
    if total <= 0:
        return float(np.nanmean(ppp)) if len(ppp) else 1.02
    return float(np.average(ppp, weights=poss))


def _weighted_rate(frame: pd.DataFrame, column: str) -> float:
    if frame is None or frame.empty or column not in frame.columns or "poss" not in frame.columns:
        return 0.0
    poss = pd.to_numeric(frame["poss"], errors="coerce").fillna(0.0)
    rate = pd.to_numeric(frame[column], errors="coerce")
    rate = rate.map(lambda value: _rate(value, float("nan")))
    keep = poss.gt(0) & np.isfinite(rate) & rate.gt(0) & rate.lt(1.0)
    if not bool(keep.any()):
        return 0.0
    return float(np.average(rate[keep], weights=poss[keep]))


def league_priors(play_types: pd.DataFrame | None) -> LeaguePriors:
    priors = {}
    shooting = {}
    if play_types is None or play_types.empty:
        return LeaguePriors.flat()
    offensive = play_types[play_types["side"].astype(str).str.lower().eq("offensive")]
    source = offensive[offensive["entity_type"].astype(str).eq("T")]
    if source.empty:
        source = offensive
    if "season" in source.columns and not source.empty:
        latest = source["season"].astype(str).max()
        source = source[source["season"].astype(str).eq(latest)]
    for play in PLAY_TYPES:
        rows = source[source["play_type"] == play]
        priors[play] = _weighted_ppp(rows) if not rows.empty else 1.02
        shooting[play] = _weighted_rate(rows, "fg_pct")
    return LeaguePriors(ppp_by_type=priors, fg_by_type=shooting)


_PLAY_INDEX: dict[int, dict[tuple, pd.DataFrame]] = {}
_POSITION_PPP: dict[tuple, float] = {}
_POSITION_IDS: dict[tuple, set[int]] = {}


def _play_index(play_types: pd.DataFrame) -> dict[tuple, pd.DataFrame]:
    """One pass over Synergy rows. Later slices are dict lookups."""
    key = id(play_types)
    cached = _PLAY_INDEX.get(key)
    if cached is not None:
        return cached
    work = play_types
    seasons = work["season"].astype(str)
    types = work["entity_type"].astype(str)
    entities = pd.to_numeric(work["entity_id"], errors="coerce")
    sides = work["side"].astype(str).str.lower()
    plays = work["play_type"].map(canon_play) if "play_type" in work.columns else work.get("play_type")
    indexed: dict[tuple, list[int]] = {}
    for index, (season, entity_type, entity_id, side) in enumerate(zip(seasons, types, entities, sides)):
        if not np.isfinite(entity_id):
            continue
        bucket = (season, entity_type, int(entity_id), side)
        indexed.setdefault(bucket, []).append(index)
    out: dict[tuple, pd.DataFrame] = {}
    for bucket, rows in indexed.items():
        frame = work.iloc[rows]
        if plays is not None:
            frame = frame.copy()
            frame["play_type"] = plays.iloc[rows].to_numpy()
        out[bucket] = frame
    if len(_PLAY_INDEX) > 4:
        _PLAY_INDEX.clear()
    _PLAY_INDEX[key] = out
    return out


def _slice_plays(
    play_types: pd.DataFrame | None,
    season: str,
    entity_type: str,
    entity_id: int,
    side: str,
) -> pd.DataFrame:
    if play_types is None or play_types.empty:
        return pd.DataFrame()
    found = _play_index(play_types).get((str(season), str(entity_type), int(entity_id), str(side).lower()))
    return found if found is not None else pd.DataFrame()


def _position_player_ids(players: pd.DataFrame, season: str, position: str) -> set[int]:
    key = (id(players), str(season), str(position))
    cached = _POSITION_IDS.get(key)
    if cached is not None:
        return cached
    rows = players[
        (players["season"].astype(str) == season) & (players["position"].astype(str) == position)
    ]
    ids = set(pd.to_numeric(rows["player_id"], errors="coerce").dropna().astype(int))
    if len(_POSITION_IDS) > 256:
        _POSITION_IDS.clear()
    _POSITION_IDS[key] = ids
    return ids


def _position_ppp(
    play_types: pd.DataFrame | None,
    players: pd.DataFrame | None,
    season: str,
    position: str,
    play: str,
) -> float:
    if play_types is None or players is None or play_types.empty or players.empty:
        return 1.02
    cache_key = (id(play_types), id(players), str(season), str(position), str(play))
    cached = _POSITION_PPP.get(cache_key)
    if cached is not None:
        return cached
    ids = _position_player_ids(players, season, position)
    if not ids:
        _POSITION_PPP[cache_key] = 1.02
        return 1.02
    index = _play_index(play_types)
    chunks = [
        index[(str(season), "P", pid, "offensive")]
        for pid in ids
        if (str(season), "P", pid, "offensive") in index
    ]
    if not chunks:
        _POSITION_PPP[cache_key] = 1.02
        return 1.02
    rows = pd.concat(chunks, ignore_index=True)
    rows = rows[rows["play_type"].astype(str).eq(play)]
    value = _weighted_ppp(rows) if not rows.empty else 1.02
    if len(_POSITION_PPP) > 2048:
        _POSITION_PPP.clear()
    _POSITION_PPP[cache_key] = value
    return value


def _fold_skill(latest: pd.Series | None, older: pd.Series | None, default_ppp: float) -> PlaySkill | None:
    """A thin action keeps the year before it. A full action stays on its own possessions."""
    current = _skill_from_row(latest, default_ppp) if latest is not None else None
    prior = _skill_from_row(older, default_ppp) if older is not None else None
    if current is None or current.poss <= 0:
        return prior if prior is not None and prior.poss > 0 else current
    if prior is None or prior.poss <= 0 or current.poss >= 80:
        return current
    poss = current.poss + prior.poss

    def mix(left: float, right: float) -> float:
        return (left * current.poss + right * prior.poss) / poss

    current.ppp = mix(current.ppp, prior.ppp)
    current.tov_poss_pct = mix(current.tov_poss_pct, prior.tov_poss_pct)
    current.ft_poss_pct = mix(current.ft_poss_pct, prior.ft_poss_pct)
    current.fg_pct = mix(current.fg_pct, prior.fg_pct)
    current.fg_sample = current.fg_sample + prior.fg_sample
    current.poss = poss
    return current


def _skill_from_row(row: pd.Series | None, default_ppp: float) -> PlaySkill:
    if row is None:
        return PlaySkill(poss=0.0, ppp=default_ppp)
    poss = _finite(row.get("poss"), 0.0)
    raw_fg = row.get("fg_pct")
    has_fg = False
    try:
        has_fg = raw_fg is not None and math.isfinite(float(raw_fg))
    except (TypeError, ValueError):
        has_fg = False
    return PlaySkill(
        poss=poss,
        poss_pct=_rate(row.get("poss_pct"), 0.0),
        ppp=_ppp(row.get("ppp"), default_ppp),
        fg_pct=_rate(raw_fg, 0.46),
        fg_sample=poss if has_fg and poss > 0 else 0.0,
        efg_pct=_rate(row.get("efg_pct"), 0.52),
        ft_poss_pct=_rate(row.get("ft_poss_pct"), 0.08),
        tov_poss_pct=_rate(row.get("tov_poss_pct"), 0.12),
        score_poss_pct=_rate(row.get("score_poss_pct"), 0.48),
        percentile=_finite(row.get("percentile"), 50),
    )


# A game from the slate season counts this much next to a game from a prior year.
_SEASON_WEIGHT = 1.5


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total = float(sum(weights))
    if total <= 0:
        return float(np.mean(values)) if values else 0.0
    return float(sum(value * weight for value, weight in zip(values, weights)) / total)


def _recent_multiplier(
    logs: pd.DataFrame | None,
    player_id: int,
    season_ppm: float,
    season: str | None = None,
) -> tuple[float, float]:
    if logs is None or logs.empty or season_ppm <= 0:
        return 1.0, 0.0
    rows = logs[
        (logs["entity_type"] == "P")
        & (pd.to_numeric(logs["player_id"], errors="coerce") == player_id)
    ].sort_values("date")
    if rows.empty:
        return 1.0, 0.0
    minutes = pd.to_numeric(rows["min"], errors="coerce").fillna(0)
    points = pd.to_numeric(rows["pts"], errors="coerce").fillna(0)
    usable = minutes > 0
    if not usable.any():
        return 1.0, 0.0
    frame = rows.loc[usable, ["date"]].copy()
    frame["ppm"] = points[usable] / minutes[usable]
    frame = frame.tail(12)
    days = pd.to_datetime(frame["date"], errors="coerce").dt.date
    alpha = 2 / (min(8, len(frame)) + 1)
    value = float(frame["ppm"].iloc[0])
    for point, day in zip(frame["ppm"].iloc[1:], days.iloc[1:]):
        step = alpha * (_SEASON_WEIGHT if _in_season(day, season) else 1.0)
        step = min(step, 0.85)
        value = step * float(point) + (1.0 - step) * value
    return value / season_ppm, float(len(frame))


def _recent_fg_multiplier(
    logs: pd.DataFrame | None,
    player_id: int,
    season_eff: float,
    season: str | None = None,
) -> tuple[float, float]:
    """Recent scoring per FGA vs the season card — personal make-form residual.

    Game logs carry FGA and PTS (not FGM). Points per attempt tracks shooting
    inefficiency without a second PPP stack on the rate card.
    """
    if logs is None or logs.empty or season_eff <= 0 or "fga" not in logs.columns:
        return 1.0, 0.0
    rows = logs[
        (logs["entity_type"] == "P")
        & (pd.to_numeric(logs["player_id"], errors="coerce") == player_id)
    ].sort_values("date")
    if rows.empty:
        return 1.0, 0.0
    minutes = pd.to_numeric(rows["min"], errors="coerce").fillna(0)
    points = pd.to_numeric(rows["pts"], errors="coerce").fillna(0)
    fga = pd.to_numeric(rows["fga"], errors="coerce").fillna(0)
    usable = (minutes > 0) & (fga > 0)
    if not usable.any():
        return 1.0, 0.0
    frame = rows.loc[usable, ["date"]].copy()
    frame["eff"] = points[usable].to_numpy() / fga[usable].to_numpy()
    # Prefer this season's look once two games are on the board — prior-year
    # heat should not cancel early-season cold boxes.
    if season is not None:
        in_year = frame["date"].map(lambda day: _in_season(day, season))
        if int(in_year.sum()) >= 2:
            frame = frame.loc[in_year]
    frame = frame.tail(12)
    days = pd.to_datetime(frame["date"], errors="coerce").dt.date
    alpha = 2 / (min(8, len(frame)) + 1)
    value = float(frame["eff"].iloc[0])
    for point, day in zip(frame["eff"].iloc[1:], days.iloc[1:]):
        step = alpha * (_SEASON_WEIGHT if _in_season(day, season) else 1.0)
        step = min(step, 0.85)
        value = step * float(point) + (1.0 - step) * value
    return value / season_eff, float(len(frame))


def _in_season(day: object, season: str | None) -> bool:
    if season is None or day is None:
        return False
    try:
        if isinstance(day, float) and pd.isna(day):
            return False
        parsed = day if hasattr(day, "year") and hasattr(day, "month") else pd.to_datetime(day, errors="coerce")
        if parsed is None or pd.isna(parsed):
            return False
        return season_for(parsed) == season
    except (TypeError, ValueError, AttributeError):
        return False


def _ewma(series: pd.Series) -> float:
    alpha = 2 / (min(8, len(series)) + 1)
    value = float(series.iloc[0])
    for point in series.iloc[1:]:
        value = alpha * float(point) + (1.0 - alpha) * value
    return value


def build_team_model(
    abbr: str,
    season: str,
    team_stats: pd.DataFrame | None,
    play_types: pd.DataFrame | None,
    tracking: pd.DataFrame | None,
    lineups: pd.DataFrame | None,
    league: LeaguePriors,
    logs: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
) -> TeamModel:
    team_id = TEAM_ID_BY_ABBR[abbr]
    row = _team_row(team_stats, season, team_id)
    prev = _team_row(team_stats, previous_season(season), team_id)
    # A short current season stays close to last year's dashboard. With no
    # current row at all, pace, defense, and the glass come from that season.
    off_rating = _blend_dash(row, prev, "off_rating", 114.0)
    def_rating = _blend_dash(row, prev, "def_rating", off_rating)
    pace = _blend_dash(row, prev, "pace", 99.5)
    current_games = _dash_number(row, "gp")
    if current_games is not None and current_games >= _TEAM_STAND:
        games = current_games
    else:
        games = _dash_number(prev, "gp")
        if games is None:
            games = current_games or 0.0
    # The dashboard row is the rating. Without it, the team's own play sample
    # is the rating, and the game count is that sample so a lineup can be
    # pulled back to the team. A one-game fallback never moves.
    if row is None and prev is None:
        off_level, off_games = _synergy_level(play_types, season, team_id, "offensive")
        def_level, _def_games = _synergy_level(play_types, season, team_id, "defensive")
        if off_level is not None:
            off_rating = off_level * 100.0
        if def_level is not None:
            def_rating = def_level * 100.0
        elif off_level is not None:
            def_rating = off_rating
        if off_games >= 8:
            games = off_games
    off_ppp: dict[str, float] = {}
    off_poss: dict[str, float] = {}
    def_ppp: dict[str, float] = {}
    def_poss: dict[str, float] = {}
    def_tov: dict[str, float] = {}
    def_fg: dict[str, float] = {}
    def_fg_poss: dict[str, float] = {}
    for play in PLAY_TYPES:
        league_ppp = league.ppp_by_type.get(play, 1.02)
        off_row = _one(_slice_plays(play_types, season, "T", team_id, "offensive"), play)
        prev_off = _one(_slice_plays(play_types, previous_season(season), "T", team_id, "offensive"), play)
        older_off = _one(
            _slice_plays(play_types, previous_season(previous_season(season)), "T", team_id, "offensive"),
            play,
        )
        def_row = _play_row(play_types, _seasons_back(season, 3), "T", team_id, "defensive", play)
        current = _ppp(None if off_row is None else off_row.get("ppp"), league_ppp)
        current_n = _finite(None if off_row is None else off_row.get("poss"), 0)
        prev_v = None if prev_off is None else _ppp(prev_off.get("ppp"), league_ppp)
        prev_n = 0 if prev_off is None else _finite(prev_off.get("poss"), 0)
        if older_off is not None and prev_n < 80:
            older_n = _finite(older_off.get("poss"), 0)
            older_v = _ppp(older_off.get("ppp"), league_ppp)
            if prev_n <= 0:
                prev_v, prev_n = older_v, older_n
            elif older_n > 0 and prev_v is not None:
                prev_v = (prev_v * prev_n + older_v * older_n) / (prev_n + older_n)
                prev_n = prev_n + older_n
        off_ppp[play] = blend_ppp(
            current,
            current_n,
            league_ppp,
            200,
            league_ppp,
            league_ppp,
            prev_ppp=prev_v,
            prev_poss=prev_n,
        )
        off_poss[play] = current_n + prev_n
        # Defensive rating is points per 100. Play points are per possession,
        # so the prior is the league rate for this action times the rating.
        rating_index = (def_rating or LEAGUE_RATING) / LEAGUE_RATING
        if not math.isfinite(rating_index) or rating_index <= 0:
            rating_index = 1.0
        anchor = league_ppp * rating_index
        if def_row is None:
            def_ppp[play] = anchor
            def_poss[play] = 0.0
        else:
            sample = _finite(def_row.get("poss"), 0)
            def_ppp[play] = shrink(
                _ppp(def_row.get("ppp"), anchor),
                sample,
                anchor,
                LEAGUE_PRIOR,
            )
            def_poss[play] = sample
            forced = _rate(def_row.get("tov_poss_pct"), float("nan"))
            if math.isfinite(forced) and 0.0 < forced < 0.6:
                def_tov[play] = forced
            allowed_fg = _rate(def_row.get("fg_pct"), float("nan"))
            if math.isfinite(allowed_fg) and 0.0 < allowed_fg < 1.0 and sample > 0:
                def_fg[play] = allowed_fg
                def_fg_poss[play] = sample
    zone_d, zone_n = _team_zones(tracking, season, team_id)
    pace_book, minute_book, shared_book = _lineup_book(lineups, season, team_id)
    off_book, def_book, oreb_book = _lineup_rating_book(lineups, season, team_id)
    return TeamModel(
        team_id=team_id,
        abbr=abbr,
        name=team_name(abbr),
        pace=pace if pace > 80 else 99.5,
        off_rating=off_rating,
        def_rating=def_rating,
        oreb_pct=_rate(_blend_dash(row, prev, "oreb_pct", 0.27), 0.27),
        dreb_pct=_rate(_blend_dash(row, prev, "dreb_pct", 0.73), 0.73),
        games=max(games, 1),
        off_ppp=off_ppp,
        off_poss=off_poss,
        def_ppp=def_ppp,
        def_poss=def_poss,
        def_tov=def_tov,
        def_fg=def_fg,
        def_fg_poss=def_fg_poss,
        zone_d_fg=zone_d,
        zone_normal_fg=zone_n,
        lineup_pace=pace_book,
        lineup_minutes=minute_book,
        shared_minutes=shared_book,
        lineup_off=off_book,
        lineup_def=def_book,
        lineup_oreb=oreb_book,
        coach=coach_for(abbr).name,
        rotation_size=coach_for(abbr).players,
        foul_patience=coach_for(abbr).foul_patience,
        fta_allowed=_fta_allowed(row, prev),
        oreb_allowed=_oreb_allowed(row, prev),
        ast_allowed=_ast_allowed(logs, season, team_id),
        reb_share_allowed=_reb_share_own(logs, players, season, team_id),
        ast_share_allowed=_ast_share_allowed(logs, players, season, team_id),
    )


def build_players(
    abbr: str,
    season: str,
    team: TeamModel,
    players: pd.DataFrame | None,
    play_types: pd.DataFrame | None,
    zones: pd.DataFrame | None,
    tracking: pd.DataFrame | None,
    hustle: pd.DataFrame | None,
    clutch: pd.DataFrame | None,
    on_off: pd.DataFrame | None,
    logs: pd.DataFrame | None,
    matchups: pd.DataFrame | None,
    league: LeaguePriors,
    roster: pd.DataFrame | None = None,
    on: date | None = None,
) -> list[PlayerModel]:
    logs = _logs_before(logs, on)
    membership = _roster_members(roster, abbr)
    on_night = roster is not None and "night_window" in roster.columns
    if on_night or membership:
        built = _players_from_roster(
            membership,
            abbr,
            season,
            team,
            players,
            play_types,
            zones,
            tracking,
            hustle,
            clutch,
            on_off,
            logs,
            matchups,
            league,
            on=on,
        )
    elif players is None or players.empty:
        built = _synthetic_rotation(team, league)
    else:
        current = _team_player_rows(players, abbr, season)
        if current.empty:
            built = _synthetic_rotation(team, league)
        else:
            built = [
                _compose_player(
                    row, team, season, players, play_types, zones, tracking, hustle, clutch, on_off, logs, matchups, league
                )
                for row in current.to_dict("records")
            ]
    _stamp_shared(built, team)
    _stamp_pace(built, team)
    _attach_court(built, team, logs, on)
    return built


def _roster_members(roster: pd.DataFrame | None, abbr: str) -> list[dict]:
    if roster is None or roster.empty or "team_abbr" not in roster.columns:
        return []
    rows = roster[roster["team_abbr"].astype(str) == abbr]
    return rows.to_dict("records")


def _team_player_rows(players: pd.DataFrame, abbr: str, season: str) -> pd.DataFrame:
    current = players[
        (players["season"].astype(str) == season) & (players["team_abbr"].astype(str) == abbr)
    ]
    if current.empty:
        current = players[players["team_abbr"].astype(str) == abbr]
    return current


def _players_from_roster(
    membership: list[dict],
    abbr: str,
    season: str,
    team: TeamModel,
    players: pd.DataFrame | None,
    play_types: pd.DataFrame | None,
    zones: pd.DataFrame | None,
    tracking: pd.DataFrame | None,
    hustle: pd.DataFrame | None,
    clutch: pd.DataFrame | None,
    on_off: pd.DataFrame | None,
    logs: pd.DataFrame | None,
    matchups: pd.DataFrame | None,
    league: LeaguePriors,
    on: date | None = None,
) -> list[PlayerModel]:
    ranked = sorted(
        membership,
        key=lambda row: (
            _finite(row.get("depth_rank"), 99),
            -_finite(row.get("salary"), 0),
            -_finite(row.get("experience"), 0),
            str(row.get("player_name") or ""),
        ),
    )
    stats = _stats_by_name(players, abbr, season)
    coach = coach_for(team.abbr)
    active_ranks = [
        index
        for index, member in enumerate(ranked)
        if str(member.get("roster_status") or "").lower() != "inactive"
    ]
    curve = depth_minutes(coach, len(active_ranks) or 1)
    by_rank = {index: minutes for index, minutes in zip(active_ranks, curve)}
    built = []
    for index, member in enumerate(ranked):
        default_minutes = by_rank.get(index, 0.0)
        stats_row = stats.get(norm_person(member.get("player_name")))
        if stats_row is None:
            player = _template_player(member, team, league, default_minutes)
        else:
            player = _compose_player(
                stats_row,
                team,
                season,
                players,
                play_types,
                zones,
                tracking,
                hustle,
                clutch,
                on_off,
                logs,
                matchups,
                league,
                fallback_minutes=default_minutes,
            )
            player.name = str(member.get("player_name") or player.name)
            if member.get("position"):
                player.position = str(member["position"])
        player.espn_id = str(member.get("espn_id") or "")
        player.minutes_slot = index + 1
        player.minutes_depth = float(default_minutes)
        if player.minutes_base <= 0:
            player.minutes_base = player.minutes_mean
        if str(member.get("roster_status") or "").lower() == "inactive":
            if player.minutes_base <= 0:
                player.minutes_base = player.minutes_mean
            player.injury_label = "Inactive"
            player.injury_factor = 0.0
            player.available = False
            player.minutes_mean = 0
        built.append(player)
    _stamp_schedule(built, logs, team.team_id, season)
    apply_play_shares(
        built,
        team,
        play_types,
        season,
        current_slate=_is_current_slate(on) if on is not None else False,
    )
    apply_player_scoring(built, season=season)
    return built


def _position_skills(position: str, team: TeamModel, league: LeaguePriors) -> tuple[dict[str, PlaySkill], dict[str, PlaySkill]]:
    """Scale a position's play mix to this team's rating.

    A real defensive sample keeps the team rate. With none, the lineup's
    position prior is what the possession model uses.
    """
    rates = position_rates(position)
    off_target = _weighted_level(team.off_ppp, team.off_poss, team.off_rating / 100.0)
    def_anchor = {
        play: shrink(
            float(team.def_ppp.get(play, team.def_rating / 100.0)),
            float(team.def_poss.get(play, 0.0)),
            float(league.ppp_by_type.get(play, 1.02) or 1.02),
            40.0,
        )
        for play in PLAY_TYPES
    }
    def_target = _weighted_level(def_anchor, team.def_poss, off_target)
    off_level = sum(share * ppp for share, ppp, _allowed in rates.values()) or 1.0
    def_level = sum(share * allowed for share, _ppp, allowed in rates.values()) or 1.0
    off_scale = off_target / off_level
    def_scale = def_target / def_level
    measured_def = sum(float(value) for value in team.def_poss.values()) > 1.0
    offense = {}
    defense = {}
    for play, (share, ppp, allowed) in rates.items():
        team_ppp = team.off_ppp.get(play)
        team_n = team.off_poss.get(play, 0.0)
        raw = ppp * off_scale
        blended = raw if not team_n else shrink(float(team_ppp or raw), team_n, raw, FILE_PRIOR)
        offense[play] = PlaySkill(poss=share * 100.0, ppp=blended)
        defense[play] = PlaySkill(poss=0.0 if measured_def else share * 100.0, ppp=allowed * def_scale)
    return offense, defense


def _foul_rate(row: dict) -> float:
    """Personal fouls per 36 minutes, relative to a league average of 3.2."""
    minutes = _finite(row.get("min"), 0.0)
    fouls = _finite(row.get("pf"), 0.0)
    if minutes <= 8 or fouls <= 0:
        return 0.0
    return (fouls * 36.0 / minutes) / 3.2


def _mean_or(values: dict[str, float], fallback: float) -> float:
    if not values:
        return fallback
    return float(sum(values.values()) / len(values))


def _weighted_level(values: dict[str, float], weights: dict[str, float], fallback: float) -> float:
    """Possession-weighted mean. A rare play does not count as much as the offense."""
    num = 0.0
    den = 0.0
    for key, value in values.items():
        weight = float(weights.get(key, 0.0) or 0.0)
        if weight <= 0 or not math.isfinite(float(value)):
            continue
        num += weight * float(value)
        den += weight
    if den <= 0:
        return _mean_or(values, fallback)
    return num / den


def _synergy_level(
    play_types: pd.DataFrame | None,
    season: str,
    team_id: int,
    side: str,
) -> tuple[float | None, float]:
    """Possession-weighted points per possession, and the games behind that sample."""
    frame = _slice_plays(play_types, season, "T", team_id, side)
    if frame.empty:
        frame = _slice_plays(play_types, previous_season(season), "T", team_id, side)
    if frame.empty:
        return None, 0.0
    poss = pd.to_numeric(frame["poss"], errors="coerce").fillna(0.0)
    total = float(poss.sum())
    if total <= 0:
        return None, 0.0
    ppp = frame["ppp"].map(lambda value: _ppp(value, 1.0))
    level = float(np.average(ppp.to_numpy(dtype=float), weights=poss.to_numpy(dtype=float)))
    games = pd.to_numeric(frame.get("gp"), errors="coerce")
    sample = float(games.max()) if games.notna().any() else 0.0
    return level, sample


def _stats_by_name(players: pd.DataFrame | None, abbr: str, season: str) -> dict[str, dict]:
    """Newest season for this name. The same team wins a tie, then minutes."""
    if players is None or players.empty:
        return {}
    chain = _seasons_back(season)
    best: dict[str, tuple[tuple, dict]] = {}
    for row in players.to_dict("records"):
        key = norm_person(row.get("player_name"))
        row_season = str(row.get("season") or "")
        if not key or row_season not in chain:
            continue
        same_team = str(row.get("team_abbr") or "") == abbr
        score = (chain.index(row_season), 0 if same_team else 1, -_finite(row.get("min"), 0))
        current = best.get(key)
        if current is None or score < current[0]:
            best[key] = (score, row)
    return {key: item[1] for key, item in best.items()}


def _template_player(member: dict, team: TeamModel, league: LeaguePriors, minutes: float) -> PlayerModel:
    position = str(member.get("position") or "F")
    offense, defense = _position_skills(position, team, league)
    oreb = {"C": 0.09, "F": 0.05, "G": 0.02}.get(position, 0.04)
    dreb = {"C": 0.22, "F": 0.14, "G": 0.08}.get(position, 0.12)
    espn_id = str(member.get("espn_id") or "0")
    try:
        player_id = int(espn_id)
    except ValueError:
        player_id = abs(hash(espn_id)) % 1_000_000_000
    return PlayerModel(
        player_id=player_id,
        name=str(member.get("player_name") or espn_id),
        team_id=team.team_id,
        position=position,
        minutes_mean=minutes,
        offense=offense,
        defense=defense,
        oreb_pct=oreb,
        dreb_pct=dreb,
        espn_id=espn_id,
        minutes_source="Depth chart",
        minutes_base=minutes,
    )


def _compose_player(
    row: dict,
    team: TeamModel,
    season: str,
    players: pd.DataFrame | None,
    play_types: pd.DataFrame | None,
    zones: pd.DataFrame | None,
    tracking: pd.DataFrame | None,
    hustle: pd.DataFrame | None,
    clutch: pd.DataFrame | None,
    on_off: pd.DataFrame | None,
    logs: pd.DataFrame | None,
    matchups: pd.DataFrame | None,
    league: LeaguePriors,
    fallback_minutes: float = 18,
) -> PlayerModel:
    player_id = int(row["player_id"])
    position = str(row.get("position") or "F")
    offense = {}
    season_ppm = _finite(row.get("pts"), 10) / max(_finite(row.get("min"), 20), 1)
    recent_mult, recent_n = _recent_multiplier(logs, player_id, season_ppm, season)
    season_eff = _finite(row.get("pts"), 10) / max(_finite(row.get("fga"), 8), 1)
    make_form, make_form_games = _recent_fg_multiplier(logs, player_id, season_eff, season)
    for play in PLAY_TYPES:
        league_ppp = league.ppp_by_type.get(play, 1.02)
        team_ppp = team.off_ppp.get(play, league_ppp)
        team_poss = team.off_poss.get(play, 100)
        cur = _one(_slice_plays(play_types, season, "P", player_id, "offensive"), play)
        prev = _one(
            _slice_plays(play_types, previous_season(season), "P", player_id, "offensive"),
            play,
        )
        older = _one(
            _slice_plays(play_types, previous_season(previous_season(season)), "P", player_id, "offensive"),
            play,
        )
        skill = _skill_from_row(cur, team_ppp)
        prev_skill = _fold_skill(prev, older, team_ppp)
        skill.ppp = blend_ppp(
            skill.ppp,
            skill.poss,
            team_ppp,
            team_poss,
            _position_ppp(play_types, players, season, position, play),
            league_ppp,
            recent_mult=recent_mult,
            recent_games=recent_n,
            prev_ppp=None if prev_skill is None else prev_skill.ppp,
            prev_poss=0 if prev_skill is None else prev_skill.poss,
        )
        if skill.poss <= 0 and prev_skill is not None:
            skill.poss = prev_skill.poss * 0.25
            skill.tov_poss_pct = prev_skill.tov_poss_pct
            skill.ft_poss_pct = prev_skill.ft_poss_pct
            skill.fg_pct = prev_skill.fg_pct
        offense[play] = skill
    defense = _defensive_profile(play_types, season, player_id, team)
    zone_freq, zone_fg = _player_zones(zones, season, player_id)
    zone_d, zone_n, zone_fga = _player_tracking(tracking, season, player_id)
    hustle_row = _entity_row(hustle, season, "P", player_id)
    clutch_mult = _clutch_mult(clutch, logs, season, player_id, row)
    off_impact, def_impact = _on_court_impact(on_off, season, player_id, team.off_rating, team.def_rating)
    _complete_defense(defense, team, zone_d, zone_n, zone_fga, def_impact, league)
    minutes_mean, minutes_std, minutes_source, minutes_sample = _minutes(
        logs, player_id, _finite(row.get("min"), fallback_minutes)
    )
    if minutes_mean <= 0:
        minutes_mean = fallback_minutes
        minutes_source = "Depth chart"
        minutes_sample = 0
    return PlayerModel(
        player_id=player_id,
        name=str(row.get("player_name") or player_id),
        team_id=team.team_id,
        position=position,
        minutes_mean=minutes_mean,
        minutes_std=minutes_std,
        offense=offense,
        defense=defense,
        zone_freq=zone_freq,
        zone_fg=zone_fg,
        zone_d_fg=zone_d,
        zone_normal_fg=zone_n,
        zone_d_fga=zone_fga,
        oreb_pct=_rate(row.get("oreb_pct"), 0.04),
        dreb_pct=_rate(row.get("dreb_pct"), 0.14),
        ast_share=max(0.2, _finite(row.get("ast"), 2)),
        stl_weight=max(0.2, _finite(row.get("stl"), 0.7)),
        blk_weight=max(0.1, _finite(row.get("blk"), 0.4)),
        ft_pct=_rate(row.get("ft_pct"), 0.78),
        foul_rate=_foul_rate(row),
        usg=_rate(row.get("usg_pct"), 0.18),
        sample_games=_finite(row.get("gp"), 0),
        clutch_mult=clutch_mult,
        screen_assists=_finite(None if hustle_row is None else hustle_row.get("screen_assists"), 0),
        box_outs=_finite(None if hustle_row is None else hustle_row.get("box_outs"), 0),
        matchup_fg=_matchups_for(matchups, player_id),
        off_impact=off_impact,
        def_impact=def_impact,
        minutes_source=minutes_source,
        minutes_base=minutes_mean,
        minutes_sample=minutes_sample,
        make_form=make_form,
        make_form_games=make_form_games,
    )


def _pre_tip_reports(
    injuries: pd.DataFrame | None,
    abbr: str,
    game_on: date,
    include_game_day: bool = False,
) -> pd.DataFrame:
    """Reports filed before tip for this team.

    The chart team is the team on the report. A report for another team stays
    with that team. Same-day notes stay off a finished night (they are often
    written after tip); an upcoming game keeps today's report.
    """
    if injuries is None or injuries.empty or "team_abbr" not in injuries.columns or "report_date" not in injuries.columns:
        return pd.DataFrame()
    frame = injuries.copy()
    frame["team_abbr"] = frame["team_abbr"].map(lambda value: normalize_team(value) or "")
    team = normalize_team(abbr) or ""
    frame = frame[frame["team_abbr"] == team]
    stamps = pd.to_datetime(frame["report_date"], errors="coerce", utc=True)
    days = stamps.dt.date
    if include_game_day:
        return frame.loc[stamps.notna() & (days <= game_on)].copy()
    return frame.loc[stamps.notna() & (days < game_on)].copy()


def mark_projected_starters(players: list[PlayerModel]) -> None:
    """Tag the opening five from the ESPN depth chart after outs are applied.

    ``minutes_slot`` is depth order on the chart. Available men keep their
    slots; the first five still standing are tonight's projected starters.
    """
    for player in players:
        player.projected_starter = False
    candidates = [
        player
        for player in players
        if player.available
        and float(player.injury_factor or 0.0) > 0.0
        and float(player.minutes_mean or 0.0) > 0.0
    ]
    candidates.sort(
        key=lambda player: (
            int(player.minutes_slot) if int(player.minutes_slot or 0) > 0 else 99,
            -float(player.minutes_mean or 0.0),
            str(player.name),
        )
    )
    for player in candidates[:5]:
        player.projected_starter = True
        profile = getattr(player, "court", None)
        if profile is not None:
            profile.starts = True


def apply_injuries(
    players: list[PlayerModel],
    injuries: pd.DataFrame | None,
    abbr: str,
    game_on: date | None = None,
    include_game_day: bool = False,
) -> tuple[list[str], list[dict]]:
    """Apply a pre-tip report for this team and return flags plus a display row for every report line."""
    flags: list[str] = []
    display: list[dict] = []
    game_on = game_on or date.today()
    rows = _pre_tip_reports(injuries, abbr, game_on, include_game_day=include_game_day)
    if rows.empty:
        _hygiene_current_slate_minutes(players, game_on)
        mark_projected_starters(players)
        return flags, display
    by_id = {}
    by_name = {}
    for row in rows.to_dict("records"):
        athlete_id = str(row.get("athlete_id") or "")
        if athlete_id:
            by_id[athlete_id] = row
        key = norm_person(row.get("player_name"))
        if key:
            by_name.setdefault(key, row)
    matched_keys: set[str] = set()
    for player in players:
        row = by_id.get(player.espn_id) if player.espn_id else None
        if row is None:
            row = by_name.get(norm_person(player.name))
        if row is None:
            row = _unique_last_name(player, rows, players)
        if row is None:
            continue
        matched_keys.add(_injury_key(row))
        label, multiplier = injury_decision(row, game_on)
        _apply_multiplier(player, label, multiplier)
        flags.append(_injury_text(player.name, label, row))
        display.append(_injury_display(abbr, player.name, label, row, matched=True))
    _absorb_known_outs(players)
    _hygiene_current_slate_minutes(players, game_on)
    mark_projected_starters(players)
    for row in rows.to_dict("records"):
        if _injury_key(row) in matched_keys:
            continue
        label, _multiplier = injury_decision(row, game_on)
        display.append(_injury_display(abbr, str(row.get("player_name") or ""), label, row, matched=False))
    return flags, display


def _injury_key(row: dict) -> str:
    athlete_id = str(row.get("athlete_id") or "")
    return athlete_id or norm_person(row.get("player_name"))


def _last_name(value: object) -> str:
    suffixes = {"jr", "sr", "ii", "iii", "iv", "v"}
    tokens = [norm_person(part) for part in str(value or "").replace(".", " ").split()]
    tokens = [token for token in tokens if token and token not in suffixes]
    return tokens[-1] if tokens else ""


def _unique_last_name(player: PlayerModel, rows: pd.DataFrame, players: list[PlayerModel]) -> dict | None:
    last = _last_name(player.name)
    if len(last) < 4:
        return None
    if sum(1 for teammate in players if _last_name(teammate.name) == last) != 1:
        return None
    hits = [row for row in rows.to_dict("records") if _last_name(row.get("player_name")) == last]
    if len(hits) == 1:
        return hits[0]
    return None


def _apply_multiplier(player: PlayerModel, label: str, multiplier: float) -> None:
    if player.minutes_base <= 0:
        player.minutes_base = player.minutes_mean
    player.injury_label = label
    player.injury_factor = multiplier
    profile = getattr(player, "court", None)
    if multiplier <= 0:
        # Leave profile.minutes for absorb. Clearing them here donated nothing.
        player.available = False
        player.minutes_mean = 0
        if profile is not None:
            profile.starts = False
        return
    if profile is not None and int(getattr(profile, "sample", 0) or 0) > 0:
        # The tag is the chance he plays. The minutes, when he plays, stay his
        # (already depth-floored on the profile when that floor applied).
        player.minutes_mean = float(profile.minutes)
        return
    player.minutes_mean = player.minutes_base * multiplier
    if label == "Out":
        player.available = False
        player.minutes_mean = 0


def _is_current_slate(on: date) -> bool:
    """Finished seasons use log rosters; only the live slate needs chart hygiene."""
    season = season_for(on)
    slate = SETTINGS.seasons[-1] if SETTINGS.seasons else season
    return str(season) >= str(slate)


def _hygiene_current_slate_minutes(players: list[PlayerModel], on: date) -> None:
    """Cap chart phantoms and collapse bloated Σ minutes — current slate only.

    Applying these to finished log-roster seasons compressed 32+ stars in the
    post-eval full grade. Keep the hygiene on STOD chart nights only, and only
    rescale when the sheet is clearly oversubscribed.
    """
    if not _is_current_slate(on):
        return
    if len(players) >= 5:\n        _cap_unsampled_minutes(players, cap=6.0)
    _rescale_available_to_240(players, only_if_over=280.0, protect_stars=True)


def _cap_unsampled_minutes(players: list[PlayerModel], cap: float = 6.0) -> None:
    """Chart-only / age-coach men without a same-team sample stay at depth minutes."""
    for player in players:
        if not player.available:
            continue
        sample = int(getattr(player, "minutes_sample", 0) or 0)
        profile = getattr(player, "court", None)
        if profile is not None:
            sample = max(sample, int(getattr(profile, "sample", 0) or 0))
        if sample > 0:
            continue
        depth = float(getattr(player, "minutes_depth", 0.0) or 0.0)
        limit = min(cap, depth) if depth > 0 else cap
        if float(player.minutes_mean or 0.0) <= limit:
            continue
        player.minutes_mean = limit
        if profile is not None:
            profile.minutes = min(float(profile.minutes or 0.0), limit)
            if float(profile.minutes_close or 0.0) > limit:
                profile.minutes_close = limit
            if float(profile.minutes_blowout or 0.0) > limit:
                profile.minutes_blowout = limit


def _is_star_minutes(player: PlayerModel) -> bool:
    mean = float(player.minutes_mean or 0.0)
    avg = float(getattr(player, "minutes_appearance_avg", 0.0) or 0.0)
    return mean >= 32.0 or avg >= 32.0


def _rescale_available_to_240(
    players: list[PlayerModel],
    target: float = 240.0,
    only_if_over: float | None = None,
    protect_stars: bool = False,
) -> None:
    """Soft defense when the available sheet is bloated past a regulation night.

    On the live slate, cut depth/phantoms first so STOD 32+ stars keep their
    log minutes instead of sharing a proportional crush.
    """
    avail = [player for player in players if player.available and float(player.minutes_mean or 0.0) > 0]
    total = sum(float(player.minutes_mean) for player in avail)
    if total <= 1e-6:
        return
    if only_if_over is not None and total <= only_if_over:
        return
    if abs(total - target) < 1.0:
        return

    def _apply(scale_map: dict[int, float]) -> None:
        for player in avail:
            scale = scale_map.get(id(player), 1.0)
            if abs(scale - 1.0) < 1e-9:
                continue
            player.minutes_mean = float(player.minutes_mean) * scale
            profile = getattr(player, "court", None)
            if profile is None:
                continue
            profile.minutes = float(player.minutes_mean)
            if float(profile.minutes_close or 0.0) > 0:
                profile.minutes_close = float(profile.minutes_close) * scale
            if float(profile.minutes_blowout or 0.0) > 0:
                profile.minutes_blowout = float(profile.minutes_blowout) * scale

    if protect_stars:
        stars = [player for player in avail if _is_star_minutes(player)]
        others = [player for player in avail if player not in stars]
        star_total = sum(float(player.minutes_mean) for player in stars)
        other_total = sum(float(player.minutes_mean) for player in others)
        if stars and others and other_total > 1e-6 and star_total < target - 1.0:
            need = max(target - star_total, 0.0)
            scale = need / other_total
            _apply({id(player): scale for player in others})
            return
    scale = target / total
    _apply({id(player): scale for player in avail})


def _absorb_known_outs(players: list[PlayerModel]) -> None:
    """An out player's own minutes move onto the positions that have absorbed them.

    Roster inactives stay zeroed without dumping their historical minutes onto
    tonight's actives. Injury and box outs donate, then their profiles clear.
    """
    from nba_sim.features.court import absorb

    profiles = {int(player.player_id): player.court for player in players if getattr(player, "court", None) is not None}
    matrix = getattr(players[0], "_sub_matrix", None) if players else None
    if not profiles or not matrix:
        return
    donate: list[int] = []
    for player in players:
        if player.available or int(player.player_id) not in profiles:
            continue
        label = str(getattr(player, "injury_label", "") or "").strip().lower()
        profile = profiles[int(player.player_id)]
        if label == "inactive":
            profile.minutes = 0.0
            profile.minutes_close = 0.0
            profile.minutes_blowout = 0.0
            profile.starts = False
            player.minutes_mean = 0.0
            continue
        donate.append(int(player.player_id))
    if donate:
        absorb(profiles, donate, matrix)
    for player in players:
        profile = profiles.get(int(player.player_id))
        if profile is None:
            continue
        if not player.available:
            profile.minutes = 0.0
            profile.minutes_close = 0.0
            profile.minutes_blowout = 0.0
            profile.starts = False
            player.minutes_mean = 0.0
            continue
        if int(getattr(profile, "sample", 0) or 0) > 0 or float(profile.minutes) > 0:
            player.minutes_mean = float(profile.minutes)


def _injury_text(name: str, label: str, row: dict) -> str:
    body = str(row.get("body_part") or "").strip()
    detail = f"{name} {label}"
    if body:
        detail = f"{detail} ({body.lower()})"
    return detail


def _injury_display(team: str, name: str, label: str, row: dict, matched: bool) -> dict:
    return {
        "team": team.upper(),
        "player": name,
        "label": label,
        "body_part": str(row.get("body_part") or ""),
        "comment": str(row.get("comment") or row.get("detail") or ""),
        "return_date": str(row.get("return_date") or ""),
        "matched": matched,
    }


def _synthetic_rotation(team: TeamModel, league: LeaguePriors) -> list[PlayerModel]:
    from nba_sim.sim.demo import rotation

    players = rotation(team.team_id, team.abbr.upper(), team.team_id % 100000 * 10)
    for player in players:
        for play, skill in player.offense.items():
            skill.ppp = team.off_ppp.get(play, league.ppp_by_type.get(play, 1.02))
        player.oreb_pct = team.oreb_pct / 5.0
        player.dreb_pct = team.dreb_pct / 5.0
    return players


def _dash_number(row: pd.Series | None, key: str) -> float | None:
    if row is None:
        return None
    try:
        number = float(row.get(key))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _fta_allowed(current: pd.Series | None, previous: pd.Series | None) -> float:
    """How often this defense sends a shot to the line, against a league rate of 0.26."""
    blended = _blend_dash(current, previous, "opp_fta_rate", 0.26)
    if blended <= 0.05:
        return 1.0
    return float(np.clip(blended / 0.26, 0.86, 1.18))


def _oreb_allowed(current: pd.Series | None, previous: pd.Series | None) -> float:
    """Opponent offensive rebound rate. A short season leans on last year."""
    blended = _blend_dash(current, previous, "opp_oreb_pct", 0.30)
    if blended <= 0.05:
        return 0.30
    return float(np.clip(blended, 0.20, 0.40))


_LEAGUE_AST_RATE = 0.60  # AST / estimated FGM
_REB_SHARE_PRIOR = {"G": 0.22, "F": 0.38, "C": 0.40}
_AST_SHARE_PRIOR = {"G": 0.55, "F": 0.35, "C": 0.10}


def _ast_allowed(logs: pd.DataFrame | None, season: str, team_id: int) -> float:
    """Opponent AST/FGM vs this defense, as a multiplier around 1.0."""
    if logs is None or logs.empty:
        return 1.0
    teams = logs[
        (logs["entity_type"].astype(str) == "T")
        & (logs["season"].astype(str) == str(season))
    ]
    if teams.empty or "game_id" not in teams.columns:
        return 1.0
    own_ids = set(
        teams.loc[pd.to_numeric(teams["team_id"], errors="coerce") == team_id, "game_id"].astype(str)
    )
    if len(own_ids) < 4:
        return 1.0
    opp = teams[
        teams["game_id"].astype(str).isin(own_ids)
        & (pd.to_numeric(teams["team_id"], errors="coerce") != team_id)
    ]
    if opp.empty:
        return 1.0
    rates: list[float] = []
    for _, row in opp.iterrows():
        ast = float(pd.to_numeric(row.get("ast"), errors="coerce") or 0.0)
        fga = float(pd.to_numeric(row.get("fga"), errors="coerce") or 0.0)
        # No FGM in the log; estimate from league FG% × FGA.
        fgm = max(1.0, 0.47 * fga) if fga > 0 else 0.0
        if fgm <= 0:
            continue
        rates.append(ast / fgm)
    if len(rates) < 4:
        return 1.0
    measured = float(np.mean(rates))
    pooled = shrink(measured, float(len(rates)), _LEAGUE_AST_RATE, 8.0)
    return float(np.clip(pooled / _LEAGUE_AST_RATE, 0.88, 1.12))


def _position_lookup(players: pd.DataFrame | None, season: str) -> dict[int, str]:
    """Map player_id → G/F/C. Prefer ESPN roster labels; stats positions often collapse F→G."""
    out: dict[int, str] = {}
    try:
        roster = Warehouse().read("espn_rosters")
    except Exception:
        roster = None
    name_pos: dict[str, str] = {}
    if roster is not None and not roster.empty and "position" in roster.columns:
        for _, row in roster.iterrows():
            key = norm_person(row.get("player_name"))
            if key:
                name_pos[key] = position_key(str(row.get("position") or "F"))
    if players is not None and not players.empty and "player_id" in players.columns:
        rows = players
        if "season" in players.columns:
            matched = players[players["season"].astype(str) == str(season)]
            if not matched.empty:
                rows = matched
        for _, row in rows.iterrows():
            pid = pd.to_numeric(row.get("player_id"), errors="coerce")
            if not np.isfinite(pid):
                continue
            key = norm_person(row.get("player_name"))
            if key and key in name_pos:
                out[int(pid)] = name_pos[key]
            else:
                out[int(pid)] = position_key(str(row.get("position") or "F"))
    return out


def _share_dict(totals: dict[str, float], prior: dict[str, float], games: int) -> dict[str, float]:
    raw = {key: float(totals.get(key, 0.0)) for key in ("G", "F", "C")}
    mass = sum(raw.values())
    if mass <= 0 or games < 4:
        return dict(prior)
    # Floor empty buckets before normalize so a missing F label cannot zero a slot.
    floored = {key: max(raw[key], 0.05 * mass) for key in raw}
    floored_mass = sum(floored.values()) or 1.0
    shares = {key: floored[key] / floored_mass for key in floored}
    # Need ~8 games before the defense's own shape stands alone; never drop the prior entirely.
    trust = min(0.75, max(0.0, (games - 4) / 8.0))
    blended = {
        key: float(trust * shares[key] + (1.0 - trust) * prior[key])
        for key in prior
    }
    total = sum(blended.values()) or 1.0
    return {key: blended[key] / total for key in blended}


def _reb_share_own(
    logs: pd.DataFrame | None,
    players: pd.DataFrame | None,
    season: str,
    team_id: int,
) -> dict[str, float]:
    """How this team's G/F/C share the glass (applied to their DREB weights)."""
    return _pos_stat_shares(logs, players, season, team_id, "reb", own=True, prior=_REB_SHARE_PRIOR)


def _ast_share_allowed(
    logs: pd.DataFrame | None,
    players: pd.DataFrame | None,
    season: str,
    team_id: int,
) -> dict[str, float]:
    """Opponent AST by G/F/C when facing this defense."""
    return _pos_stat_shares(logs, players, season, team_id, "ast", own=False, prior=_AST_SHARE_PRIOR)


def _pos_stat_shares(
    logs: pd.DataFrame | None,
    players: pd.DataFrame | None,
    season: str,
    team_id: int,
    column: str,
    *,
    own: bool,
    prior: dict[str, float],
) -> dict[str, float]:
    if logs is None or logs.empty or column not in logs.columns:
        return dict(prior)
    positions = _position_lookup(players, season)
    if not positions:
        return dict(prior)
    frame = logs[
        (logs["entity_type"].astype(str) == "P")
        & (logs["season"].astype(str) == str(season))
    ].copy()
    if frame.empty:
        return dict(prior)
    frame["team_id"] = pd.to_numeric(frame["team_id"], errors="coerce")
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce")
    frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["min"] = pd.to_numeric(frame.get("min"), errors="coerce").fillna(0.0)
    team_games = logs[
        (logs["entity_type"].astype(str) == "T")
        & (logs["season"].astype(str) == str(season))
        & (pd.to_numeric(logs["team_id"], errors="coerce") == team_id)
    ]
    game_ids = set(team_games["game_id"].astype(str)) if not team_games.empty else set()
    if len(game_ids) < 4:
        return dict(prior)
    in_games = frame[frame["game_id"].astype(str).isin(game_ids)]
    if own:
        rows = in_games[in_games["team_id"] == team_id]
    else:
        rows = in_games[in_games["team_id"] != team_id]
    rows = rows[rows["min"] > 0]
    if rows.empty:
        return dict(prior)
    totals = {"G": 0.0, "F": 0.0, "C": 0.0}
    for _, row in rows.iterrows():
        pid = int(row["player_id"]) if np.isfinite(row["player_id"]) else 0
        pos = positions.get(pid, "F")
        totals[pos] = totals.get(pos, 0.0) + float(row[column])
    return _share_dict(totals, prior, len(game_ids))


# Ten games ties last season's dashboard. Fourteen games is this year's team.
_TEAM_PRIOR = 14.0
_TEAM_STAND = 14.0


def _blend_dash(current: pd.Series | None, previous: pd.Series | None, key: str, default: float) -> float:
    """A new season moves team strength a little faster. Fourteen games stands alone."""
    now = _dash_number(current, key)
    old = _dash_number(previous, key)
    if now is None and old is None:
        return default
    if now is None:
        return float(old)
    if old is None:
        return now
    games = _dash_number(current, "gp") or 0.0
    if games >= _TEAM_STAND:
        return now
    trust = games / (games + _TEAM_PRIOR)
    return trust * now + (1.0 - trust) * float(old)


def _team_row(frame: pd.DataFrame | None, season: str, team_id: int) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    rows = frame[
        (frame["season"].astype(str) == season)
        & (pd.to_numeric(frame["team_id"], errors="coerce") == team_id)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def _play_row(
    play_types: pd.DataFrame | None,
    seasons: list[str],
    entity_type: str,
    entity_id: int,
    side: str,
    play: str,
) -> pd.Series | None:
    """The newest season that actually has this action."""
    for candidate in seasons:
        row = _one(_slice_plays(play_types, candidate, entity_type, entity_id, side), play)
        if row is not None and _finite(row.get("poss"), 0) > 0:
            return row
    return None


def _one(frame: pd.DataFrame, play: str) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    rows = frame[frame["play_type"] == play]
    if rows.empty:
        return None
    return rows.iloc[0]


def _entity_row(frame: pd.DataFrame | None, season: str, entity: str, entity_id: int) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    rows = frame[
        (frame["season"].astype(str) == season)
        & (frame["entity_type"].astype(str) == entity)
        & (pd.to_numeric(frame["entity_id"], errors="coerce") == entity_id)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def _team_zones(tracking: pd.DataFrame | None, season: str, team_id: int) -> tuple[dict, dict]:
    for candidate in _seasons_back(season):
        allowed, normal = _team_zones_one(tracking, candidate, team_id)
        if allowed:
            return allowed, normal
    return {}, {}


def _team_zones_one(tracking: pd.DataFrame | None, season: str, team_id: int) -> tuple[dict, dict]:
    allowed: dict[str, float] = {}
    normal: dict[str, float] = {}
    if tracking is None or tracking.empty:
        return allowed, normal
    rows = tracking[
        (tracking["season"].astype(str) == season)
        & (tracking["entity_type"].astype(str) == "T")
        & (pd.to_numeric(tracking["entity_id"], errors="coerce") == team_id)
        & (tracking["zone"].astype(str) != "overall")
    ]
    for row in rows.to_dict("records"):
        zone = row["zone"]
        allowed[zone] = _rate(row.get("d_fg_pct"), DEFAULT_ZONE_FG.get(zone, 0.45))
        normal[zone] = _rate(row.get("normal_fg_pct"), allowed[zone])
    return allowed, normal


def _player_zones(zones: pd.DataFrame | None, season: str, player_id: int) -> tuple[dict, dict]:
    if zones is None or zones.empty:
        return dict(DEFAULT_ZONE_FREQ), dict(DEFAULT_ZONE_FG)
    current = _player_zones_one(zones, season, player_id)
    prior = _player_zones_one(zones, previous_season(season), player_id)
    if current[0] is None and prior[0] is None:
        for candidate in _seasons_back(season)[2:]:
            freq, fg = _player_zones_one(zones, candidate, player_id)
            if freq is not None:
                return freq, fg
        return dict(DEFAULT_ZONE_FREQ), dict(DEFAULT_ZONE_FG)
    if current[0] is None:
        return prior  # type: ignore[return-value]
    if prior[0] is None:
        return current  # type: ignore[return-value]
    # Blend prior-year zones toward this season once both books exist.
    trust = 0.55
    freq = {
        zone: (1.0 - trust) * float(prior[0].get(zone, DEFAULT_ZONE_FREQ.get(zone, 0.0)))
        + trust * float(current[0].get(zone, DEFAULT_ZONE_FREQ.get(zone, 0.0)))
        for zone in set(current[0]) | set(prior[0]) | set(DEFAULT_ZONE_FREQ)
    }
    fg = {
        zone: (1.0 - trust) * float(prior[1].get(zone, DEFAULT_ZONE_FG.get(zone, 0.4)))
        + trust * float(current[1].get(zone, DEFAULT_ZONE_FG.get(zone, 0.4)))
        for zone in set(current[1]) | set(prior[1]) | set(DEFAULT_ZONE_FG)
    }
    total = sum(freq.values()) or 1.0
    freq = {zone: value / total for zone, value in freq.items()}
    return freq, fg


def _player_zones_one(zones: pd.DataFrame, season: str, player_id: int) -> tuple[dict, dict] | tuple[None, None]:
    rows = zones[
        (zones["season"].astype(str) == season)
        & (zones["entity_type"].astype(str) == "P")
        & (pd.to_numeric(zones["entity_id"], errors="coerce") == player_id)
    ]
    if rows.empty:
        return None, None
    freq = {row["zone"]: _finite(row["freq"], 0) for row in rows.to_dict("records")}
    fg = {row["zone"]: _rate(row["fg_pct"], 0.4) for row in rows.to_dict("records")}
    total = sum(freq.values())
    if total <= 0:
        return None, None
    freq = {zone: value / total for zone, value in freq.items()}
    for zone, share in DEFAULT_ZONE_FREQ.items():
        freq.setdefault(zone, share * 0.01)
        fg.setdefault(zone, DEFAULT_ZONE_FG[zone])
    total = sum(freq.values())
    freq = {zone: value / total for zone, value in freq.items()}
    return freq, fg


def _player_tracking(tracking, season, player_id) -> tuple[dict, dict, dict]:
    if tracking is None or tracking.empty:
        return {}, {}, {}
    for candidate in _seasons_back(season):
        allowed, normal, fga = _player_tracking_one(tracking, candidate, player_id)
        if allowed:
            return allowed, normal, fga
    return {}, {}, {}


def _player_tracking_one(tracking, season, player_id) -> tuple[dict, dict, dict]:
    allowed, normal, fga = {}, {}, {}
    rows = tracking[
        (tracking["season"].astype(str) == season)
        & (tracking["entity_type"].astype(str) == "P")
        & (pd.to_numeric(tracking["entity_id"], errors="coerce") == player_id)
        & (tracking["zone"].astype(str) != "overall")
    ]
    for row in rows.to_dict("records"):
        zone = row["zone"]
        allowed[zone] = _rate(row.get("d_fg_pct"), 0.45)
        normal[zone] = _rate(row.get("normal_fg_pct"), allowed[zone])
        fga[zone] = _finite(row.get("d_fga"), 0)
    return allowed, normal, fga


def _clutch_mult(clutch, logs, season, player_id, row) -> float:
    if clutch is None or clutch.empty:
        return 1.0
    rows = clutch.iloc[0:0]
    for candidate in _seasons_back(season):
        rows = clutch[
            (clutch["season"].astype(str) == candidate)
            & (clutch["entity_type"].astype(str) == "P")
            & (pd.to_numeric(clutch["entity_id"], errors="coerce") == player_id)
        ]
        if not rows.empty:
            break
    if rows.empty:
        return 1.0
    item = rows.iloc[0]
    season_min = max(_finite(row.get("min"), 1), 1)
    season_ppm = _finite(row.get("pts"), 0) / season_min
    clutch_min = max(_finite(item.get("min"), 0), 0.1)
    clutch_ppm = _finite(item.get("pts"), 0) / clutch_min
    if season_ppm <= 0:
        return 1.0
    return float(np.clip(shrink(clutch_ppm / season_ppm, _finite(item.get("gp"), 0), 1.0, 12), 0.9, 1.12))


def _defensive_profile(
    play_types: pd.DataFrame | None,
    season: str,
    player_id: int,
    team: TeamModel,
) -> dict[str, PlaySkill]:
    """Every action he has actually defended. A thin year stays tied to the one before it."""
    seasons = _seasons_back(season, 3)
    frame = _player_plays(play_types, player_id, "defensive", seasons)
    found: dict[str, PlaySkill] = {}
    if frame.empty:
        return found
    for play in PLAY_TYPES:
        default = float(team.def_ppp.get(play, 1.02) or 1.02)
        skill = None
        for candidate in reversed(seasons):
            rows = frame[(frame["season"].astype(str) == candidate) & (frame["play_type"] == play)]
            if rows.empty:
                continue
            incoming = _skill_from_row(rows.iloc[0], default)
            skill = _pool_skill(incoming, skill)
        if skill is not None and skill.poss > 0:
            found[play] = skill
    return found


def _player_plays(
    play_types: pd.DataFrame | None,
    player_id: int,
    side: str,
    seasons: list[str],
) -> pd.DataFrame:
    if play_types is None or play_types.empty:
        return pd.DataFrame()
    frame = play_types[
        play_types["season"].astype(str).isin(seasons)
        & play_types["entity_type"].astype(str).eq("P")
        & pd.to_numeric(play_types["entity_id"], errors="coerce").eq(int(player_id))
        & play_types["side"].astype(str).str.lower().eq(side)
    ]
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["play_type"] = frame["play_type"].map(canon_play)
    return frame


def _pool_skill(incoming: PlaySkill, prior: PlaySkill | None) -> PlaySkill:
    """A full sample replaces the prior. A short one keeps the prior in the rate."""
    if incoming.poss <= 0:
        return prior if prior is not None else incoming
    if prior is None or prior.poss <= 0 or incoming.poss >= 80:
        return incoming
    poss = incoming.poss + prior.poss

    def mix(left: float, right: float) -> float:
        return (left * incoming.poss + right * prior.poss) / poss

    incoming.ppp = mix(incoming.ppp, prior.ppp)
    incoming.tov_poss_pct = mix(incoming.tov_poss_pct, prior.tov_poss_pct)
    incoming.ft_poss_pct = mix(incoming.ft_poss_pct, prior.ft_poss_pct)
    incoming.fg_pct = mix(incoming.fg_pct, prior.fg_pct)
    incoming.fg_sample = incoming.fg_sample + prior.fg_sample
    incoming.poss = poss
    return incoming


def _complete_defense(
    defense: dict[str, PlaySkill],
    team: TeamModel,
    allowed: dict[str, float],
    normal: dict[str, float],
    fga: dict[str, float],
    def_impact: float,
    league: LeaguePriors,
) -> None:
    """Fill actions synergy does not cover, from his own level and the shots he contests.

    A real sample is left alone. The filled action carries no turnover rate, so
    only a measured defensive column moves the steal. Possessions stay small
    enough that a measured teammate still outweighs the fill.
    """
    measured = [skill for skill in defense.values() if skill.poss > 0]
    anchor = None
    if measured:
        weight = sum(skill.poss for skill in measured)
        anchor = sum(skill.poss * skill.ppp for skill in measured) / weight
    contested = _contested_freq(fga)
    overall = None
    if allowed and contested:
        overall = _zone_ratio(contested, allowed, normal, league)
    for play in PLAY_TYPES:
        current = defense.get(play)
        if current is not None and current.poss > 0:
            continue
        team_ppp = float(team.def_ppp.get(play, 1.02) or 1.02)
        shape = 1.0
        if allowed:
            action = _zone_ratio(play_shot_freq(play), allowed, normal, league)
            if overall and overall > 0:
                shape = action / overall
            elif anchor is None:
                shape = action
        if anchor is not None:
            ppp = anchor * shape
            # The fill borrows his other actions. It does not outvote the team file.
            sample = min(FILE_PRIOR, max(8.0, sum(skill.poss for skill in measured) * 0.05))
        elif allowed and sum(fga.values()) >= 20:
            ppp = team_ppp * shape
            # Contested shots are a shape, not a counted season of this action.
            sample = min(FILE_PRIOR, max(8.0, sum(fga.values()) * 0.04))
        elif abs(def_impact - 1.0) > 0.005:
            ppp = team_ppp * def_impact
            sample = min(FILE_PRIOR, 12.0)
        else:
            continue
        defense[play] = PlaySkill(poss=sample, ppp=float(ppp), tov_poss_pct=0.0, fg_pct=0.0, fg_sample=0.0)


def _contested_freq(fga: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 0.0) for value in fga.values())
    if total <= 0:
        return {}
    return {zone: max(value, 0.0) / total for zone, value in fga.items()}


def _on_court_impact(
    on_off: pd.DataFrame | None,
    season: str,
    player_id: int,
    team_off: float,
    team_def: float,
) -> tuple[float, float]:
    """How the team scores and how many points it allows while he is on the floor."""
    if on_off is None or on_off.empty:
        return 1.0, 1.0
    rows = on_off.iloc[0:0]
    for candidate in _seasons_back(season):
        rows = on_off[
            (on_off["season"].astype(str) == candidate)
            & (pd.to_numeric(on_off["player_id"], errors="coerce") == player_id)
        ]
        if not rows.empty:
            break
    if rows.empty:
        return 1.0, 1.0
    item = rows.iloc[0]
    minutes = _finite(item.get("min"), 0)
    return (
        _rating_impact(item.get("off_rating_on"), team_off, minutes),
        _rating_impact(item.get("def_rating_on"), team_def, minutes),
    )


def _rating_impact(rating: object, team_rating: float, minutes: float) -> float:
    if team_rating <= 0:
        return 1.0
    raw = _finite(rating, team_rating) / team_rating
    shrunk = shrink(raw, minutes, 1.0, FILE_PRIOR)
    if not np.isfinite(shrunk) or shrunk <= 0:
        return 1.0
    return float(shrunk)


def _logs_before(logs: pd.DataFrame | None, on: date | None) -> pd.DataFrame | None:
    """Games already played. The slate date itself stays out of the ratings."""
    if logs is None or logs.empty or on is None or "date" not in logs.columns:
        return logs
    dates = pd.to_datetime(logs["date"], errors="coerce")
    return logs.loc[dates.notna() & (dates < pd.Timestamp(on))].copy()


def _minutes(logs, player_id, season_min) -> tuple[float, float, str, int]:
    """Recent playing time, then the season average, then nothing for the depth chart to fill.

    Only games he actually played count. A coach's decision zero would pull a
    star prior down and inflate the next man; the court profile already uses
    the same played-only mean.
    """
    season_min = max(season_min, 0)
    if logs is None or logs.empty:
        if season_min > 0:
            return season_min, 3.0, "Season average", 0
        return 0.0, 3.0, "Depth chart", 0
    rows = logs[
        (logs["entity_type"] == "P")
        & (pd.to_numeric(logs["player_id"], errors="coerce") == player_id)
    ].sort_values("date")
    minutes = pd.to_numeric(rows["min"], errors="coerce").dropna()
    recent = minutes[minutes > 0].tail(12)
    if recent.empty:
        if season_min > 0:
            return season_min, 3.0, "Season average", 0
        return 0.0, 3.0, "Depth chart", 0
    return float(recent.mean()), float(recent.std(ddof=0) or 2.5), "Last 12 games", int(len(recent))


# Minutes of this year's best five that tie last year's book. A short year still leans back.
_LINEUP_PRIOR = 36.0
_LINEUP_NOW_FLOOR = 8.0
_LINEUP_PRIOR_FLOOR = 48.0


def _lineup_book(lineups, season, team_id):
    """Pace, minutes, and pair minutes for the fives this team actually played.

    This season's groups count more than the same minutes last year. A handful
    of minutes does not erase last year's fives. Pace still needs 30 minutes.
    Last year's book ignores a five under 48. This year's book keeps a five
    once it has played 8.
    """
    empty = {}, {}, {}
    if lineups is None or lineups.empty:
        return empty
    current = _lineup_book_one(lineups, season, team_id, _LINEUP_NOW_FLOOR)
    previous = _lineup_book_one(lineups, previous_season(season), team_id, _LINEUP_PRIOR_FLOOR)
    if not _book_has(current) and not _book_has(previous):
        for candidate in _seasons_back(season)[2:]:
            older = _lineup_book_one(lineups, candidate, team_id, _LINEUP_PRIOR_FLOOR)
            if _book_has(older):
                return older
        return empty
    return _blend_lineup_books(current, previous)


def _lineup_rating_book(lineups, season: str, team_id: int):
    """Offensive rating, defensive rating, and the glass for each five.

    This season's group keeps its own number. A five only on last year's file
    keeps that number until this season has seen it.
    """
    current = _rating_one(lineups, season, team_id)
    previous = _rating_one(lineups, previous_season(season), team_id)
    off = dict(previous[0])
    deff = dict(previous[1])
    oreb = dict(previous[2])
    off.update(current[0])
    deff.update(current[1])
    oreb.update(current[2])
    return off, deff, oreb


def _rating_one(lineups, season: str, team_id: int):
    off: dict[tuple[int, ...], float] = {}
    deff: dict[tuple[int, ...], float] = {}
    oreb: dict[tuple[int, ...], float] = {}
    if lineups is None or getattr(lineups, "empty", True) or "season" not in lineups.columns:
        return off, deff, oreb
    rows = lineups[
        (lineups["season"].astype(str) == str(season))
        & (pd.to_numeric(lineups["team_id"], errors="coerce") == team_id)
    ]
    for row in rows.to_dict("records"):
        played = _finite(row.get("min"), 0)
        if played < 8:
            continue
        parts = str(row.get("group_id") or "").split("-")
        try:
            ids = tuple(sorted(int(float(part)) for part in parts if part))
        except ValueError:
            continue
        if len(ids) != 5:
            continue
        offense = _finite(row.get("off_rating"), 0)
        defense = _finite(row.get("def_rating"), 0)
        glass = _finite(row.get("oreb_pct"), 0)
        if offense > 70:
            off[ids] = offense
        if defense > 70:
            deff[ids] = defense
        if glass > 0:
            oreb[ids] = glass
    return off, deff, oreb


def _squad_frame(team: TeamModel) -> pd.DataFrame:
    rows = []
    keys = set(team.lineup_minutes) | set(team.lineup_off) | set(team.lineup_pace)
    for key in keys:
        if len(key) != 5:
            continue
        rows.append(
            {
                "group_id": "-".join(str(int(pid)) for pid in key),
                "min": float(team.lineup_minutes.get(key, 0.0) or 0.0),
                "off_rating": float(team.lineup_off.get(key, 0.0) or 0.0),
                "def_rating": float(team.lineup_def.get(key, 0.0) or 0.0),
                "pace": float(team.lineup_pace.get(key, 0.0) or 0.0),
                "oreb_pct": float(team.lineup_oreb.get(key, 0.0) or 0.0),
            }
        )
    return pd.DataFrame(rows)


def _attach_court(players: list[PlayerModel], team: TeamModel, logs: pd.DataFrame | None, on) -> None:
    """One profile per player on this team, from his games before the slate."""
    from nba_sim.features.court import fit_court

    stamped = _with_team_margin(logs)
    participation, subs, possessions = _court_history(stamped, int(team.team_id))
    profiles, whistle, squads, matrix = fit_court(
        stamped,
        players,
        on=on,
        participation=participation,
        subs=subs,
        possessions=possessions,
        lineups=_squad_frame(team),
        ages=_age_table(),
        coach=str(team.coach or ""),
        team_id=int(team.team_id),
        current_slate=_is_current_slate(on) if on is not None else False,
    )
    team.squads = squads
    team.sub_matrix = matrix
    team.court_whistle = whistle
    if not any(int(profile.sample) > 0 for profile in profiles.values()):
        return
    for player in players:
        profile = profiles.get(int(player.player_id))
        if profile is None:
            continue
        player.court = profile
        player._sub_matrix = matrix
        if not player.available:
            continue
        if int(profile.sample) <= 0:
            continue
        court_minutes = float(profile.minutes)
        depth = float(getattr(player, "minutes_depth", 0.0) or 0.0)
        slot = int(getattr(player, "minutes_slot", 0) or 0)
        # Coach depth is a floor for the first three roles so a thin court pool
        # cannot sit a star below the staff's planned minutes. Profiled lineups
        # owe close/blowout clocks, so the floor must land on those too.
        if slot in (1, 2, 3) and depth > 0:
            floor = depth * 0.95
            court_minutes = max(court_minutes, floor)
            profile.minutes = max(float(profile.minutes), floor)
            if float(profile.minutes_close or 0.0) > 0:
                profile.minutes_close = max(float(profile.minutes_close), floor)
        player.minutes_mean = court_minutes
        player.minutes_std = float(profile.minutes_std) if profile.minutes_std > 0 else player.minutes_std
        player.minutes_source = profile.source
        player.minutes_sample = int(profile.sample)
        if profile.usage and abs(profile.usage - 0.2) > 0.02:
            player.usg = float(profile.usage)


_MARGIN_LOGS: dict[int, pd.DataFrame] = {}


def _with_team_margin(logs: pd.DataFrame | None) -> pd.DataFrame | None:
    """His plus-minus is not the game margin. The team row is."""
    if logs is None or getattr(logs, "empty", True):
        return logs
    if "entity_type" not in logs.columns or "plus_minus" not in logs.columns:
        return logs
    cached = _MARGIN_LOGS.get(id(logs))
    if cached is not None:
        return cached
    team = logs[logs["entity_type"].astype(str) == "T"]
    if team.empty or "game_id" not in team.columns or "team_id" not in team.columns:
        return logs
    margin = team.loc[:, ["game_id", "team_id", "plus_minus"]].copy()
    margin["team_id"] = pd.to_numeric(margin["team_id"], errors="coerce")
    margin["plus_minus"] = pd.to_numeric(margin["plus_minus"], errors="coerce")
    margin = margin.dropna(subset=["team_id"]).drop_duplicates(["game_id", "team_id"])
    margin = margin.rename(columns={"plus_minus": "_team_margin"})
    work = logs.copy()
    work["team_id"] = pd.to_numeric(work["team_id"], errors="coerce")
    work = work.drop(columns=[column for column in ("margin",) if column in work.columns])
    work = work.merge(margin, on=["game_id", "team_id"], how="left")
    work["margin"] = work["_team_margin"].fillna(0.0)
    out = work.drop(columns=["_team_margin"])
    if len(_MARGIN_LOGS) > 4:
        _MARGIN_LOGS.clear()
    _MARGIN_LOGS[id(logs)] = out
    return out


_EVENT_CACHE: dict[tuple, pd.DataFrame] = {}


def _read_events(name: str) -> pd.DataFrame:
    from nba_sim.config import warehouse_dir
    from nba_sim.features.court import _game_token

    path = warehouse_dir() / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        stamp = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    except OSError:
        return pd.DataFrame()
    cached = _EVENT_CACHE.get(stamp)
    if cached is not None:
        return cached
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    if "game_id" in frame.columns and not frame.empty:
        frame = frame.copy()
        frame["game_id"] = frame["game_id"].map(_game_token)
        frame.attrs["game_tokens"] = True
    _EVENT_CACHE[stamp] = frame
    return frame


def _court_history(logs: pd.DataFrame | None, team_id: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Boxes, whistles, and possessions from games already in the pre-slate logs."""
    from nba_sim.features.court import _events_before

    empty = pd.DataFrame()
    if logs is None or getattr(logs, "empty", True):
        return empty, empty, empty
    participation = _team_events(_events_before(_read_events("participation"), logs), team_id)
    subs = _stamp_decided(_team_events(_events_before(_read_events("substitutions"), logs), team_id), logs)
    possessions = _team_events(_events_before(_read_events("possessions"), logs), team_id)
    return participation, subs, possessions


def _team_events(frame: pd.DataFrame, team_id: int) -> pd.DataFrame:
    if frame is None or frame.empty or "team_id" not in frame.columns:
        return frame
    ids = pd.to_numeric(frame["team_id"], errors="coerce")
    return frame.loc[ids == int(team_id)].copy()


def _stamp_decided(subs: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """A fourth-quarter whistle in a game that ended decided is that coach's blowout exit."""
    if subs is None or subs.empty or "period" not in subs.columns:
        return subs
    if logs is None or getattr(logs, "empty", True) or "entity_type" not in logs.columns or "plus_minus" not in logs.columns:
        return subs
    team = logs[logs["entity_type"].astype(str) == "T"]
    if team.empty:
        return subs
    margin = team.loc[:, ["game_id", "team_id", "plus_minus"]].copy()
    from nba_sim.features.court import _game_token

    margin["game_id"] = margin["game_id"].map(_game_token)
    margin["team_id"] = pd.to_numeric(margin["team_id"], errors="coerce")
    margin["plus_minus"] = pd.to_numeric(margin["plus_minus"], errors="coerce")
    margin = margin.dropna(subset=["team_id"]).drop_duplicates(["game_id", "team_id"])
    work = subs.copy()
    work["game_id"] = work["game_id"].map(_game_token)
    work["team_id"] = pd.to_numeric(work["team_id"], errors="coerce")
    work = work.merge(margin, on=["game_id", "team_id"], how="left")
    period = pd.to_numeric(work["period"], errors="coerce")
    work["margin"] = np.where(period >= 4, work["plus_minus"].fillna(0.0), 0.0)
    return work.drop(columns=["plus_minus"])


_AGE_CACHE: pd.DataFrame | None = None


def _age_table() -> pd.DataFrame:
    """Age from seasons of experience when the roster file has no birth date. One table."""
    global _AGE_CACHE
    if _AGE_CACHE is not None:
        return _AGE_CACHE
    from nba_sim.config import warehouse_dir
    from nba_sim.names import norm_person

    empty = pd.DataFrame(columns=["player_id", "age"])
    stats_path = warehouse_dir() / "player_stats.parquet"
    roster_path = warehouse_dir() / "espn_rosters.parquet"
    if not stats_path.exists() or not roster_path.exists():
        _AGE_CACHE = empty
        return empty
    try:
        stats = pd.read_parquet(stats_path, columns=["player_id", "player_name"])
        roster = pd.read_parquet(roster_path, columns=["player_name", "experience"])
    except Exception:
        _AGE_CACHE = empty
        return empty
    if stats.empty or roster.empty:
        _AGE_CACHE = empty
        return empty
    names = stats.dropna(subset=["player_id"]).drop_duplicates("player_id").copy()
    names["key"] = names["player_name"].map(norm_person)
    roster = roster.copy()
    roster["key"] = roster["player_name"].map(norm_person)
    roster["years"] = roster["experience"].map(_experience_years)
    roster = roster.dropna(subset=["years"]).drop_duplicates("key")
    merged = names.merge(roster[["key", "years"]], on="key", how="inner")
    merged["age"] = 19.0 + merged["years"].astype(float)
    _AGE_CACHE = merged.loc[:, ["player_id", "age"]]
    return _AGE_CACHE


def _experience_years(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    text = str(value or "").lower()
    if "rook" in text:
        return 0.0
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return float(digits)


def _book_has(book: tuple[dict, dict, dict]) -> bool:
    return bool(book[0] or book[1] or book[2])


def _blend_lineup_books(current, previous):
    """Put this year's fives on last year's minute scale, then give them the season's share."""
    cur_pace, cur_min, cur_shared = current
    prev_pace, prev_min, prev_shared = previous
    if not _book_has(current):
        return previous
    if not _book_has(previous):
        return current
    peak = max(cur_min.values(), default=0.0)
    trust = peak / (peak + _LINEUP_PRIOR) if peak > 0 else 0.0
    last_peak = max(prev_min.values(), default=0.0)
    scale = last_peak / peak if peak > 0 and last_peak > 0 else 1.0
    minutes: dict[tuple[int, ...], float] = {}
    for group, played in prev_min.items():
        minutes[group] = (1.0 - trust) * float(played)
    for group, played in cur_min.items():
        minutes[group] = minutes.get(group, 0.0) + trust * float(played) * scale
    pair_peak = max(cur_shared.values(), default=0.0)
    last_pair = max(prev_shared.values(), default=0.0)
    pair_scale = last_pair / pair_peak if pair_peak > 0 and last_pair > 0 else 1.0
    shared: dict[tuple[int, int], float] = {}
    for pair, played in prev_shared.items():
        shared[pair] = (1.0 - trust) * float(played)
    for pair, played in cur_shared.items():
        shared[pair] = shared.get(pair, 0.0) + trust * float(played) * pair_scale
    paces: dict[tuple[int, ...], float] = {}
    for group in set(prev_pace) | set(cur_pace):
        old = prev_pace.get(group)
        new = cur_pace.get(group)
        if old is None:
            paces[group] = new
        elif new is None:
            paces[group] = old
        else:
            paces[group] = (1.0 - trust) * float(old) + trust * float(new)
    return paces, minutes, shared


def _lineup_book_one(lineups, season, team_id, floor: float):
    paces: dict[tuple[int, ...], float] = {}
    minutes: dict[tuple[int, ...], float] = {}
    shared: dict[tuple[int, int], float] = {}
    rows = lineups[
        (lineups["season"].astype(str) == season)
        & (pd.to_numeric(lineups["team_id"], errors="coerce") == team_id)
    ]
    for row in rows.to_dict("records"):
        played = _finite(row.get("min"), 0)
        if played < min(30.0, floor):
            continue
        parts = str(row.get("group_id") or "").split("-")
        try:
            ids = tuple(sorted(int(float(part)) for part in parts if part))
        except ValueError:
            continue
        if len(ids) != 5:
            continue
        pace = _finite(row.get("pace"), 0)
        if pace > 80 and played >= 30:
            paces[ids] = pace
        if played < floor:
            continue
        minutes[ids] = played
        for left in range(5):
            for right in range(left + 1, 5):
                pair = (ids[left], ids[right])
                shared[pair] = shared.get(pair, 0.0) + played
    return paces, minutes, shared


def _stamp_pace(players: list[PlayerModel], team: TeamModel) -> None:
    """Fit each player's pace from every five he has played, then keep it on him."""
    from nba_sim.sim.clock import _fit_pace_impacts

    found = _fit_pace_impacts(team)
    team.pace_impact = found
    for player in players:
        player.pace_impact = float(found.get(int(player.player_id), 0.0))


def _stamp_shared(players: list[PlayerModel], team: TeamModel) -> None:
    """Give each player the minutes he has shared a five with each teammate."""
    pairs = team.shared_minutes
    if not pairs:
        return
    for player in players:
        mine: dict[int, float] = {}
        pid = int(player.player_id)
        for (left, right), played in pairs.items():
            if left == pid:
                mine[right] = played
            elif right == pid:
                mine[left] = played
        player.shared_minutes = mine


def _matchups_for(matchups: pd.DataFrame | None, player_id: int) -> dict[int, tuple[float, float]]:
    if matchups is None or matchups.empty:
        return {}
    off_col = _first_col(matchups, "personIdOff", "PERSONIDOFF")
    def_col = _first_col(matchups, "personIdDef", "PERSONIDDEF")
    fga_col = _first_col(matchups, "matchupFieldGoalsAttempted", "MATCHUPFIELDGOALSATTEMPTED")
    fgm_col = _first_col(matchups, "matchupFieldGoalsMade", "MATCHUPFIELDGOALSMADE")
    if not all([off_col, def_col, fga_col, fgm_col]):
        return {}
    rows = matchups[pd.to_numeric(matchups[off_col], errors="coerce") == player_id]
    if rows.empty:
        return {}
    grouped = rows.groupby(pd.to_numeric(rows[def_col], errors="coerce"))
    found = {}
    for defender, group in grouped:
        if not math.isfinite(defender):
            continue
        fga = float(pd.to_numeric(group[fga_col], errors="coerce").fillna(0).sum())
        fgm = float(pd.to_numeric(group[fgm_col], errors="coerce").fillna(0).sum())
        if fga >= 8:
            found[int(defender)] = (fgm / fga, fga)
    return found


def _first_col(frame: pd.DataFrame, *names: str) -> str | None:
    lookup = {str(col).upper(): col for col in frame.columns}
    for name in names:
        if name.upper() in lookup:
            return lookup[name.upper()]
    return None


def _norm_name(value: object) -> str:
    return norm_person(value)


def _stamp_schedule(players: list[PlayerModel], logs: pd.DataFrame | None, team_id: int, season: str) -> None:
    """Games this team has played, and the games he has played for them."""
    team_games, by_player = _shares_before(logs, team_id, season)
    for player in players:
        games, average = by_player.get(int(player.player_id), (0.0, 0.0))
        player.minutes_team_games = float(team_games)
        player.minutes_appearances = float(games)
        player.minutes_appearance_avg = float(average)


def _shares_before(logs: pd.DataFrame | None, team_id: int, season: str) -> tuple[float, dict[int, tuple[float, float]]]:
    if logs is None or getattr(logs, "empty", True):
        return 0.0, {}
    team_rows = _log_slice(logs, season, "T", team_id)
    if team_rows.empty:
        team_rows = _log_slice(logs, season, "P", team_id)
        team_games = float(team_rows["game_id"].nunique()) if "game_id" in team_rows.columns else float(len(team_rows))
    else:
        team_games = float(team_rows["game_id"].nunique()) if "game_id" in team_rows.columns else float(len(team_rows))
    player_rows = _log_slice(logs, season, "P", team_id)
    found: dict[int, tuple[float, float]] = {}
    if player_rows.empty or "player_id" not in player_rows.columns:
        return team_games, found
    played = player_rows.copy()
    played["player_id"] = pd.to_numeric(played["player_id"], errors="coerce")
    played["min"] = pd.to_numeric(played["min"], errors="coerce") if "min" in played.columns else 0.0
    played = played.dropna(subset=["player_id"])
    played = played[played["min"].fillna(0) > 0]
    for player_id, group in played.groupby(played["player_id"].astype(int)):
        minutes = pd.to_numeric(group["min"], errors="coerce").dropna()
        if minutes.empty:
            continue
        found[int(player_id)] = (float(len(minutes)), float(minutes.mean()))
    return team_games, found


def _log_slice(logs: pd.DataFrame, season: str, entity: str, team_id: int) -> pd.DataFrame:
    frame = logs
    if "entity_type" in frame.columns:
        frame = frame[frame["entity_type"].astype(str) == entity]
    if "season" in frame.columns:
        frame = frame[frame["season"].astype(str) == str(season)]
    if "team_id" not in frame.columns:
        return frame.iloc[0:0]
    ids = pd.to_numeric(frame["team_id"], errors="coerce")
    return frame.loc[ids == int(team_id)]


def _roster_from_playing_time(players: pd.DataFrame | None, season: str) -> pd.DataFrame | None:
    """Who played for each team that season, with the heaviest minutes first.

    A stat line is one team for the year. The depth chart is not consulted,
    because that chart is a different season's roster.
    """
    if players is None or players.empty or "season" not in players.columns:
        return None
    frame = players[players["season"].astype(str) == str(season)].copy()
    if frame.empty or "player_name" not in frame.columns:
        return None
    if "team_abbr" not in frame.columns:
        return None
    frame["team_abbr"] = frame["team_abbr"].map(normalize_team)
    frame = frame.dropna(subset=["team_abbr"])
    frame["player_name"] = frame["player_name"].astype(str)
    frame = frame[frame["player_name"].str.strip() != ""]
    if frame.empty:
        return None
    if "min" not in frame.columns:
        frame["min"] = 0.0
    frame["min"] = pd.to_numeric(frame["min"], errors="coerce").fillna(0.0)
    if "position" not in frame.columns:
        frame["position"] = "F"
    frame = frame.sort_values(["team_abbr", "min", "player_name"], ascending=[True, False, True])
    frame = frame.drop_duplicates(["team_abbr", "player_name"], keep="first")
    frame["depth_rank"] = frame.groupby("team_abbr").cumcount()
    frame["position"] = frame["position"].fillna("F").astype(str).replace({"": "F"})
    frame["roster_status"] = "Active"
    frame["espn_id"] = ""
    keep = ["team_abbr", "player_name", "position", "depth_rank", "roster_status", "espn_id"]
    return frame[keep].reset_index(drop=True)


def _roster_for_game(
    season: str,
    chart: pd.DataFrame | None,
    players: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """The depth chart is the current season. A past season is who played there."""
    slate = SETTINGS.seasons[-1] if SETTINGS.seasons else season
    if season >= slate and chart is not None and not getattr(chart, "empty", True):
        return chart
    played = _roster_from_playing_time(players, season)
    if played is not None and not played.empty:
        return played
    return chart


def _roster_on_night(
    season: str,
    chart: pd.DataFrame | None,
    players: pd.DataFrame | None,
    logs: pd.DataFrame | None,
    on: date,
) -> pd.DataFrame | None:
    """Who is on the team for this date.

    A finished season runs from a player's first game for that team through
    his last. The current season keeps the depth chart, and a player leaves
    it when his games are for another team.
    """
    slate = SETTINGS.seasons[-1] if SETTINGS.seasons else season
    if str(season) >= str(slate) and chart is not None and not getattr(chart, "empty", True):
        return _chart_on_night(chart, players, logs, str(season), on)
    played = _log_roster(players, logs, str(season), on)
    if played is not None:
        return played
    return _roster_for_game(season, chart, players)


def _stints(logs: pd.DataFrame | None, season: str) -> pd.DataFrame | None:
    if logs is None or logs.empty:
        return None
    if "player_id" not in logs.columns or "team_id" not in logs.columns or "date" not in logs.columns:
        return None
    frame = logs
    if "entity_type" in frame.columns:
        frame = frame[frame["entity_type"].astype(str) == "P"]
    if "season" in frame.columns:
        frame = frame[frame["season"].astype(str) == str(season)]
    if frame.empty:
        return None
    frame = frame.copy()
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce")
    frame["team_id"] = pd.to_numeric(frame["team_id"], errors="coerce")
    frame["min"] = pd.to_numeric(frame["min"], errors="coerce") if "min" in frame.columns else 1.0
    frame = frame.dropna(subset=["player_id", "team_id"])
    frame = frame[frame["min"].fillna(0) > 0]
    frame["team_abbr"] = frame["team_id"].astype(int).map(ABBR_BY_TEAM_ID)
    frame = frame.dropna(subset=["team_abbr"])
    days = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.loc[days.notna()].copy()
    if frame.empty:
        return None
    frame["day"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    grouped = frame.groupby(["player_id", "team_abbr"], as_index=False).agg(
        first=("day", "min"),
        last=("day", "max"),
        total=("min", "sum"),
        games=("min", "size"),
    )
    grouped["player_id"] = grouped["player_id"].astype(int)
    return grouped


def _player_labels(players: pd.DataFrame | None, season: str) -> pd.DataFrame:
    columns = ["player_id", "player_name", "position"]
    if players is None or players.empty or "player_id" not in players.columns or "player_name" not in players.columns:
        return pd.DataFrame(columns=columns)
    frame = players.copy()
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce")
    frame = frame.dropna(subset=["player_id"])
    frame["player_id"] = frame["player_id"].astype(int)
    frame["player_name"] = frame["player_name"].astype(str)
    frame = frame[frame["player_name"].str.strip() != ""]
    if "position" not in frame.columns:
        frame["position"] = "F"
    frame["position"] = frame["position"].fillna("F").astype(str).replace({"": "F"})
    if "season" in frame.columns:
        this_year = frame[frame["season"].astype(str) == str(season)]
        other = frame[frame["season"].astype(str) != str(season)]
        frame = pd.concat([this_year, other], ignore_index=True)
    frame = frame.drop_duplicates("player_id", keep="first")
    return frame[columns]


def _log_roster(
    players: pd.DataFrame | None,
    logs: pd.DataFrame | None,
    season: str,
    on: date,
) -> pd.DataFrame | None:
    """Players whose games for that team cover this date, heaviest total minutes first."""
    stints = _stints(logs, season)
    columns = ["team_abbr", "player_name", "position", "depth_rank", "roster_status", "espn_id", "night_window"]
    if stints is None:
        return None
    stamp = pd.Timestamp(on)
    window = stints[(stints["first"] <= stamp) & (stints["last"] >= stamp)].copy()
    if window.empty:
        empty = pd.DataFrame(columns=columns)
        empty["night_window"] = True
        return empty
    labels = _player_labels(players, season)
    window = window.merge(labels, on="player_id", how="left")
    window = window.dropna(subset=["player_name"])
    if window.empty:
        empty = pd.DataFrame(columns=columns)
        empty["night_window"] = True
        return empty
    window = window.sort_values(["team_abbr", "total", "player_name"], ascending=[True, False, True])
    window["depth_rank"] = window.groupby("team_abbr").cumcount()
    window["roster_status"] = "Active"
    window["espn_id"] = ""
    window["night_window"] = True
    return window[columns].reset_index(drop=True)


def _chart_on_night(
    chart: pd.DataFrame,
    players: pd.DataFrame | None,
    logs: pd.DataFrame | None,
    season: str,
    on: date,
) -> pd.DataFrame:
    """The depth chart, without a player whose games are for another team."""
    frame = chart.copy()
    if "team_abbr" in frame.columns:
        frame["team_abbr"] = frame["team_abbr"].map(lambda value: normalize_team(value) or "")
        frame = frame[frame["team_abbr"] != ""]
    stints = _stints(logs, season)
    if stints is None or players is None or players.empty or "player_id" not in players.columns:
        frame["night_window"] = True
        return frame
    played = set(int(value) for value in stints["player_id"])
    by_player: dict[int, list[tuple[str, pd.Timestamp, pd.Timestamp]]] = {}
    for row in stints.itertuples(index=False):
        by_player.setdefault(int(row.player_id), []).append((str(row.team_abbr), row.first, row.last))
    ids: dict[str, int] = {}
    id_frame = players.copy()
    id_frame["player_id"] = pd.to_numeric(id_frame["player_id"], errors="coerce")
    for row in id_frame.dropna(subset=["player_id"]).to_dict("records"):
        key = norm_person(row.get("player_name"))
        if not key:
            continue
        player_id = int(row["player_id"])
        current = ids.get(key)
        if current is None or (current not in played and player_id in played):
            ids[key] = player_id
    stamp = pd.Timestamp(on)
    teams_with_logs = {str(team) for team in stints["team_abbr"].astype(str)}
    kept = []
    for row in frame.to_dict("records"):
        team = str(row.get("team_abbr") or "")
        player_id = ids.get(norm_person(row.get("player_name")))
        if player_id is None or player_id not in by_player:
            # After a team has STOD log appearances, drop chart-only phantoms
            # (no matched id / no stint). Opening night before any logs still
            # keeps the depth chart.
            # Keep an unobserved chart player until there is evidence of a\n            # later stint elsewhere. A rookie may have no NBA log yet.\n            if player_id is not None and player_id in by_player:\n                continue\n            if player_id is None and team in teams_with_logs:\n                continue\n            kept.append(row)
            continue
        mine = [item for item in by_player[player_id] if item[0] == team]
        others = [item for item in by_player[player_id] if item[0] != team]
        if not mine:
            continue
        first = min(item[1] for item in mine)
        last = max(item[2] for item in mine)
        if stamp < first:
            continue
        if stamp > last and any(item[1] > last for item in others):
            continue
        kept.append(row)
    if not kept:
        empty = frame.iloc[0:0].copy()
        empty["night_window"] = True
        return empty
    out = pd.DataFrame(kept)
    out["night_window"] = True
    return out


def build_specs_for_date(
    on: date,
    warehouse: Warehouse | None = None,
    game_id: str | None = None,
) -> list[GameSpec]:
    warehouse = warehouse or Warehouse()
    players = warehouse.read("player_stats")
    teams = warehouse.read("team_stats")
    play_types = warehouse.read("play_types")
    zones = warehouse.read("shot_zones")
    tracking = warehouse.read("tracking_defense")
    hustle = warehouse.read("hustle")
    lineups = warehouse.read("lineups")
    clutch = warehouse.read("clutch")
    on_off = warehouse.read("on_off")
    logs = warehouse.read("game_logs")
    injuries = warehouse.read("injuries")
    markets = warehouse.read("markets")
    schedule = warehouse.read("schedule")
    if schedule is None:
        schedule = warehouse.read("scoreboard")
    season = season_for(on)
    rosters = _roster_on_night(season, warehouse.read("espn_rosters"), players, logs, on)
    matchups = _all_matchups(warehouse)
    league = league_priors(play_types)
    games = _games_on(on, schedule, logs)
    if not games:
        return []
    specs = []
    for game in games:
        gid = str(game.get("game_id") or "")
        if game_id is not None and gid != str(game_id):
            continue
        home = normalize_team(game["home"])
        away = normalize_team(game["away"])
        if home is None or away is None:
            continue
        home_team = build_team_model(
            home, season, teams, play_types, tracking, lineups, league, logs=logs, players=players,
        )
        away_team = build_team_model(
            away, season, teams, play_types, tracking, lineups, league, logs=logs, players=players,
        )
        home_players = build_players(
            home, season, home_team, players, play_types, zones, tracking, hustle, clutch, on_off, logs, matchups, league, rosters, on
        )
        away_players = build_players(
            away, season, away_team, players, play_types, zones, tracking, hustle, clutch, on_off, logs, matchups, league, rosters, on
        )
        status = str(game.get("status") or "").lower()
        upcoming = "final" not in status
        home_flags, home_report = apply_injuries(
            home_players, injuries, home, on, include_game_day=upcoming
        )
        away_flags, away_report = apply_injuries(
            away_players, injuries, away, on, include_game_day=upcoming
        )
        home_trip = _trip(home, on, logs)
        away_trip = _trip(away, on, logs)
        if _is_current_slate(on):
            _blend_live_rating_card(home_team, home_trip)
            _blend_live_rating_card(away_team, away_trip)
        context = build_context(
            home_team,
            away_team,
            on,
            home_trip,
            away_trip,
        )
        for flag in home_flags + away_flags:
            context.factors.append(
                {
                    "name": "injury",
                    "team_ids": [home_team.team_id, away_team.team_id],
                    "player_ids": [],
                    "sample_size": 1,
                    "value": 0,
                    "detail": flag,
                }
            )
        specs.append(
            GameSpec(
                game_id=str(game.get("game_id") or f"{on.isoformat()}-{away}-{home}"),
                date=on.isoformat(),
                home=home_team,
                away=away_team,
                home_players=home_players,
                away_players=away_players,
                context=context,
                league=league,
                quotes=_quotes(markets, on, home, away, home_players + away_players),
                source="warehouse",
                venue=_cell(game.get("venue")),
                city=_cell(game.get("city")),
                start_utc=_cell(game.get("start_utc")),
                tip_local=_cell(game.get("tip_local")),
                tip_melbourne=_cell(game.get("tip_melbourne")),
                game_status=_cell(game.get("status")),
                injuries=home_report + away_report,
            )
        )
    return specs


def _all_matchups(warehouse: Warehouse) -> pd.DataFrame | None:
    frames = []
    stored = warehouse.read("matchups")
    if stored is not None:
        frames.append(stored)
    for name in warehouse.tables():
        if name.startswith("matchups_"):
            frame = warehouse.read(name)
            if frame is not None:
                frames.append(frame)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _cell(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _games_on(on: date, scoreboard: pd.DataFrame | None, logs: pd.DataFrame | None) -> list[dict]:
    iso = on.isoformat()
    if scoreboard is not None and not scoreboard.empty and "date" in scoreboard.columns:
        rows = scoreboard[scoreboard["date"].astype(str).str[:10] == iso]
        if "venue" in scoreboard.columns or "start_utc" in scoreboard.columns:
            return rows.to_dict("records")
        if not rows.empty:
            return rows.to_dict("records")
    if logs is None or logs.empty:
        return []
    team_logs = logs[(logs["entity_type"] == "T") & (pd.to_datetime(logs["date"]).dt.date == on)]
    games = {}
    for row in team_logs.to_dict("records"):
        game_id = row.get("game_id")
        matchup = str(row.get("matchup") or "")
        abbr = normalize_team(
            next(
                (
                    value
                    for value, team_id in TEAM_ID_BY_ABBR.items()
                    if int(team_id) == int(row["team_id"])
                ),
                "",
            )
        )
        slot = games.setdefault(game_id, {"game_id": game_id, "home": None, "away": None})
        if " vs" in matchup.lower():
            slot["home"] = abbr
        elif "@" in matchup:
            slot["away"] = abbr
    return [game for game in games.values() if game["home"] and game["away"]]


def _trip(abbr: str, on: date, logs: pd.DataFrame | None) -> TeamTrip:
    team_id = TEAM_ID_BY_ABBR[abbr]
    arenas = load_arenas()
    if logs is None or logs.empty:
        return TeamTrip()
    rows = logs[
        (logs["entity_type"] == "T")
        & (pd.to_numeric(logs["team_id"], errors="coerce") == team_id)
        & (pd.to_datetime(logs["date"]).dt.date < on)
    ].sort_values("date")
    if rows.empty:
        return TeamTrip()
    previous = rows.iloc[-1]
    prev_date = pd.to_datetime(previous["date"]).date()
    arena = _matchup_arena(str(previous.get("matchup") or ""), abbr, arenas)
    # Four nights ending tonight: past games on [on-3, on) plus tonight itself.
    # Counting only past games missed classic three-in-four when tonight is game 3.
    window_start = on - timedelta(days=3)
    dates = pd.to_datetime(rows["date"], errors="coerce").dt.date
    dense = rows[dates.notna() & (dates >= window_start) & (dates < on)]
    home_pm, away_pm = [], []
    home_w, away_w = [], []
    current = season_for(on)
    for row in rows.to_dict("records"):
        margin = _finite(row.get("plus_minus"), float("nan"))
        if not math.isfinite(margin):
            continue
        text = str(row.get("matchup") or "").lower()
        weight = _SEASON_WEIGHT if _in_season(row.get("date"), current) else 1.0
        if " vs" in text:
            home_pm.append(margin)
            home_w.append(weight)
        elif "@" in text:
            away_pm.append(margin)
            away_w.append(weight)
    measured = None
    samples = 0
    if home_pm and away_pm:
        measured = 0.5 * (_weighted_mean(home_pm, home_w) - _weighted_mean(away_pm, away_w))
        samples = int(round(sum(home_w) + sum(away_w)))
    points = pd.to_numeric(rows["pts"], errors="coerce")
    days = pd.to_datetime(rows["date"], errors="coerce").dt.date
    valid = points.notna() & days.notna()
    this_year = points[valid & days.map(lambda day: _in_season(day, current))]
    last_year = points[valid & days.map(lambda day: _in_season(day, previous_season(current)))]
    history = points[valid]
    form_mult, form_games = 1.0, 0
    if len(this_year) >= 2 and len(last_year) >= 8:
        baseline = float(last_year.mean())
        tail = this_year.tail(8)
        if baseline > 0:
            form_mult = _ewma(tail) / baseline
            form_games = int(len(tail))
    elif len(history) >= 4:
        season_mean = float(history.mean())
        tail = history.tail(8)
        if season_mean > 0:
            form_mult = _ewma(tail) / season_mean
            form_games = int(len(tail))
    off_form, def_form, pace_form, rating_games = _rating_trends(rows, current)
    return TeamTrip(
        prev_date=prev_date,
        prev_lat=None if arena is None else arena.lat,
        prev_lon=None if arena is None else arena.lon,
        prev_tz=None if arena is None else arena.offset(prev_date),
        games_in_last_4=int(len(dense)) + 1,
        measured_home_court=measured,
        home_court_games=samples,
        form_mult=form_mult,
        form_games=form_games,
        off_form=off_form,
        def_form=def_form,
        pace_form=pace_form,
        rating_games=rating_games,
    )


def _blend_live_rating_card(team: TeamModel, trip: TeamTrip) -> None:
    """On the live slate, soft-blend recent off/def/pace into the residual card.

    Finished seasons leave the dashboard alone. Thin samples (<4) stay put.
    """
    games = int(getattr(trip, "rating_games", 0) or 0)
    if games < 4:
        return
    trust = 0.25
    off = float(getattr(trip, "off_form", 1.0) or 1.0)
    defense = float(getattr(trip, "def_form", 1.0) or 1.0)
    pace = float(getattr(trip, "pace_form", 1.0) or 1.0)
    team.off_rating = float(team.off_rating) * ((1.0 - trust) + trust * off)
    team.def_rating = float(team.def_rating) * ((1.0 - trust) + trust * defense)
    if float(team.pace or 0.0) > 80.0:
        team.pace = float(team.pace) * ((1.0 - trust) + trust * pace)


def _rating_trends(rows: pd.DataFrame, current: str) -> tuple[float, float, float, int]:
    """Recent off / def / pace vs this season's mean — no PPP stack onto makes.

    Off uses pts / (fga + tov). Def uses opponent points (pts − plus_minus).
    Pace uses fga as a tempo proxy from the box columns we already store.
    """
    if rows is None or rows.empty or "fga" not in rows.columns:
        return 1.0, 1.0, 1.0, 0
    frame = rows.copy()
    frame["pts"] = pd.to_numeric(frame["pts"], errors="coerce")
    frame["fga"] = pd.to_numeric(frame["fga"], errors="coerce")
    frame["tov"] = pd.to_numeric(frame.get("tov"), errors="coerce").fillna(0.0)
    frame["plus_minus"] = pd.to_numeric(frame.get("plus_minus"), errors="coerce")
    days = pd.to_datetime(frame["date"], errors="coerce").dt.date
    usable = (
        frame["pts"].notna()
        & frame["fga"].notna()
        & (frame["fga"] > 0)
        & days.notna()
        & days.map(lambda day: _in_season(day, current))
    )
    frame = frame.loc[usable].sort_values("date")
    if len(frame) < 2:
        return 1.0, 1.0, 1.0, 0
    denom = (frame["fga"] + frame["tov"]).clip(lower=1.0)
    frame["off"] = frame["pts"] / denom
    frame["opp"] = frame["pts"] - frame["plus_minus"].fillna(0.0)
    frame["pace"] = frame["fga"]
    tail = frame.tail(8)
    n = int(len(tail))

    def ratio(column: str) -> float:
        baseline = float(frame[column].mean())
        if baseline <= 0:
            return 1.0
        return float(_ewma(tail[column]) / baseline)

    return ratio("off"), ratio("opp"), ratio("pace"), n


def _matchup_arena(matchup: str, team_abbr: str, arenas: dict):
    lowered = matchup.lower()
    if "@" in lowered:
        opponent = normalize_team(matchup.split("@")[-1])
        return arenas.get(opponent or "")
    return arenas.get(team_abbr)


def _quotes(markets, on: date, home: str, away: str, players: list[PlayerModel]) -> list[MarketQuote]:
    if markets is None or markets.empty:
        return []
    iso = on.isoformat()
    rows = markets.copy()
    if "date" in rows.columns:
        dated = rows["date"].astype(str).str[:10]
        rows = rows[(dated == iso) | (dated == "") | dated.isna()]
    home_rows = rows["home"].map(normalize_team) if "home" in rows.columns else None
    away_rows = rows["away"].map(normalize_team) if "away" in rows.columns else None
    if home_rows is not None and away_rows is not None:
        pair = ((home_rows == home) & (away_rows == away)) | ((home_rows == away) & (away_rows == home))
        rows = rows[pair.fillna(False)]
    names = {_norm_name(player.name): player for player in players}
    quotes = []
    for row in rows.to_dict("records"):
        market = str(row.get("market") or "")
        side = str(row.get("side") or "")
        if side == home:
            side = "home"
        elif side == away:
            side = "away"
        player = names.get(_norm_name(row.get("player")))
        american = row.get("american")
        if american is None or (isinstance(american, float) and math.isnan(american)):
            continue
        quotes.append(
            MarketQuote(
                source=str(row.get("source") or "market"),
                market=market,
                side=side,
                american=float(american),
                line=None if row.get("line") is None or (isinstance(row.get("line"), float) and math.isnan(row.get("line"))) else float(row["line"]),
                player_id=None if player is None else player.player_id,
                player_name=None if player is None else player.name,
                book=str(row.get("book") or ""),
            )
        )
    return quotes
