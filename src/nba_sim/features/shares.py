"""Play-type shares for a player, a lineup, a team, and a coach.

A player's own Synergy possessions win. A teammate with no sample gets a
role in that coach's offense: the lead guard handles and isolates, a
lower guard spots up. The five on the floor are the lineup mix, because
the possession draw adds whoever is out there. Once the team file has a
real season, that mix pulls the roster. A short file shares the pull
with the coach's scheme. The league position mix is the shape underneath
an empty role.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_sim.constants import PLAY_TYPES, canon_play
from nba_sim.features.play_priors import position_key, position_rates
from nba_sim.features.shrink import PLAYER_PRIOR, shrink
from nba_sim.names import norm_person
from nba_sim.sim.specs import PlayerModel, TeamModel

# How far a coach's scheme pulls a roster that does not yet have a team file.
_COACH_PULL = 0.22
# Possessions before this season's Synergy FG stands alone (was 80).
_MEASURED_POSS = 48.0
# Current-season weight vs prior year when pooling a thin action.
_CURRENT_FG_WEIGHT = 2.0
_CURRENT_FG_WEIGHT_THIN = 3.0
_THIN_STOD_APPEARANCES = 12

# Scheme weights on the position mix. Used for players without a sample,
# and as the target for a first-year staff. A measured player keeps his
# own possessions.
_TILT: dict[str, dict[str, float]] = {
    "gsw": {"Isolation": 0.60, "PRBallHandler": 0.84, "Spotup": 1.30, "Cut": 1.40, "OffScreen": 1.35, "Handoff": 1.15, "Postup": 0.70},
    "okc": {"Transition": 1.25, "PRBallHandler": 1.30, "Isolation": 0.90, "Postup": 0.70, "Spotup": 1.08},
    "phi": {"Isolation": 1.20, "PRBallHandler": 1.15, "Postup": 1.15, "Spotup": 0.90},
    "cle": {"Handoff": 1.25, "OffScreen": 1.20, "Cut": 1.15, "PRBallHandler": 1.10, "Isolation": 0.85},
    "bos": {"Spotup": 1.28, "Transition": 1.20, "OffScreen": 1.15, "Isolation": 0.80, "Postup": 0.75},
    "mia": {"Handoff": 1.25, "Cut": 1.20, "OffScreen": 1.15, "Isolation": 0.90},
    "nyk": {"Isolation": 1.25, "Postup": 1.20, "Transition": 0.85, "PRBallHandler": 1.08},
    "hou": {"Isolation": 1.20, "Transition": 1.15, "Postup": 1.12, "PRBallHandler": 1.05},
    "lac": {"Isolation": 1.25, "PRBallHandler": 1.20, "Spotup": 0.90},
    "ind": {"OffScreen": 1.25, "Handoff": 1.20, "PRBallHandler": 1.12},
    "den": {"PRBallHandler": 1.15, "Postup": 1.20, "Cut": 1.12, "OffRebound": 1.10},
    "min": {"PRBallHandler": 1.15, "Isolation": 1.10, "Postup": 1.08},
    "mil": {"Transition": 1.15, "PRBallHandler": 1.10, "Spotup": 1.08},
    "sas": {"Cut": 1.20, "Spotup": 1.15, "OffScreen": 1.15, "Isolation": 0.80},
    "uta": {"Handoff": 1.15, "Spotup": 1.12, "Isolation": 0.85, "PRBallHandler": 1.08},
    "atl": {"Isolation": 1.10, "PRBallHandler": 1.12, "Handoff": 1.10},
}

_GUARD_CREATION = {"Isolation", "Transition", "PRBallHandler"}
_FORWARD_CREATION = {"Isolation", "PRBallHandler", "Postup", "Transition"}
_BIG_CREATION = {"Postup", "PRRollman"}
_SPACING = {"Spotup", "Cut", "OffScreen", "Handoff", "OffRebound"}


def role_shares(position: str, rank: int, abbr: str) -> dict[str, float]:
    """Possession mix for one role on this coach's team. Sums to 1."""
    key = position_key(position)
    creator = _creator(key, rank)
    tilt = _TILT.get((abbr or "").lower(), {})
    raw: dict[str, float] = {}
    for play, (share, _ppp, _allowed) in position_rates(position).items():
        scheme = float(tilt.get(play, 1.0))
        if _is_creation(key, play):
            role = creator
        elif play in _SPACING:
            role = min(1.7, max(0.40, 1.55 - 0.60 * creator))
        else:
            role = 1.0
        raw[play] = share * scheme * role
    total = sum(raw.values()) or 1.0
    return {play: value / total for play, value in raw.items()}


def _is_role_flip(player: PlayerModel) -> bool:
    """Minutes jumped vs depth chart, or a thin sample already in a big role."""
    mean = float(player.minutes_mean or 0.0)
    depth = float(getattr(player, "minutes_depth", 0.0) or 0.0)
    sample = int(getattr(player, "minutes_sample", 0) or 0)
    if depth > 0 and abs(mean - depth) >= 6.0:
        return True
    if sample < 8 and mean >= 28.0:
        return True
    return False


def _thin_stod(player: PlayerModel, current_slate: bool) -> bool:
    """Live slate with few appearances: trust this season's FG book harder."""
    if not current_slate:
        return False
    appearances = float(getattr(player, "minutes_appearances", 0.0) or 0.0)
    return appearances < _THIN_STOD_APPEARANCES


def apply_play_shares(
    players: list[PlayerModel],
    team: TeamModel,
    play_types: pd.DataFrame | None,
    season: str,
    current_slate: bool = False,
) -> None:
    """Replace league-sized possession weights with player, team, and coach shares.

    The slate season is the sample. A thin action keeps the year before it,
    so a new game moves the rate without throwing out the last full season.
    A clear role flip drops the prior-year pool so the FG book can catch usage.
    """
    ranks = _ranks(players)
    measured = [
        _measured_with_prior(
            play_types,
            player.name,
            season,
            role_flip=_is_role_flip(player),
            current_weight=(
                _CURRENT_FG_WEIGHT_THIN if _thin_stod(player, current_slate) else _CURRENT_FG_WEIGHT
            ),
            solo_threshold=_MEASURED_POSS * (0.55 if _thin_stod(player, current_slate) else 1.0),
        )
        for player in players
    ]
    rate = _poss_per_game(measured)
    for index, player in enumerate(players):
        if player.minutes_mean <= 0:
            continue
        rows = measured[index]
        thin = _is_role_flip(player) or _thin_stod(player, current_slate)
        threshold = _MEASURED_POSS * (0.55 if thin else 1.0)
        if sum(row["poss"] for row in rows.values()) >= threshold:
            _write_measured(player, rows)
            continue
        if sum(skill.poss for skill in player.offense.values()) >= 250:
            continue
        rank = ranks.get(index, 3)
        shares = role_shares(player.position, rank, team.abbr)
        # Possessions per game keep the mix and the sample. Usage is the claim
        # once he is on the floor, and minutes decide whether he is there.
        volume = rate * _volume_scale(player.position, rank)
        for play in PLAY_TYPES:
            skill = player.offense.get(play)
            if skill is not None:
                skill.poss = shares.get(play, 0.0) * volume
    _pull_team_mix(players, team, play_types, season)


def _creator(position: str, rank: int) -> float:
    if position == "G":
        return (1.55, 0.82, 0.52, 0.36)[min(rank, 3)]
    if position == "C":
        return (1.40, 0.72, 0.48)[min(rank, 2)]
    return (1.25, 0.78, 0.52, 0.38)[min(rank, 3)]


def _volume_scale(position: str, rank: int) -> float:
    """Lead role matches a measured teammate's possessions per game. Deeper roles touch less."""
    creator = _creator(position_key(position), rank)
    peak = _creator(position_key(position), 0)
    return creator / peak


def _is_creation(position: str, play: str) -> bool:
    if position == "C":
        return play in _BIG_CREATION
    if position == "F":
        return play in _FORWARD_CREATION
    return play in _GUARD_CREATION


def _ranks(players: list[PlayerModel]) -> dict[int, int]:
    grouped: dict[str, list[int]] = {}
    for index, player in enumerate(players):
        if player.minutes_mean <= 0 or not player.available:
            continue
        grouped.setdefault(position_key(player.position), []).append(index)
    ranks: dict[int, int] = {}
    for indexes in grouped.values():
        indexes.sort(key=lambda index: players[index].minutes_mean, reverse=True)
        for order, index in enumerate(indexes):
            ranks[index] = order
    return ranks


def _poss_per_game(measured: list[dict[str, dict]]) -> float:
    rates = []
    for rows in measured:
        total = sum(row["poss"] for row in rows.values())
        if total < _MEASURED_POSS:
            continue
        gp = max(float(row.get("gp") or 0) for row in rows.values()) or 1.0
        rates.append(total / gp)
    if not rates:
        return 16.0
    return sum(rates) / len(rates)


def _write_measured(player: PlayerModel, rows: dict[str, dict]) -> None:
    gp = max(float(row.get("gp") or 0) for row in rows.values()) or 1.0
    for play in PLAY_TYPES:
        skill = player.offense.get(play)
        if skill is None:
            continue
        row = rows.get(play)
        if row is None or row["poss"] <= 0:
            skill.poss = 0.0
            skill.fg_sample = 0.0
            continue
        evidence = float(row["poss"])
        skill.ppp = shrink(float(row["ppp"]), evidence, skill.ppp, PLAYER_PRIOR)
        # The mix is possessions per game. The field-goal sample is the season.
        skill.fg_sample = evidence
        skill.poss = evidence / gp
        skill.poss_pct = skill.poss
        skill.tov_poss_pct = float(row["tov"])
        skill.ft_poss_pct = float(row["ft"])
        skill.fg_pct = float(row["fg"])


def _pull_team_mix(
    players: list[PlayerModel],
    team: TeamModel,
    play_types: pd.DataFrame | None,
    season: str,
) -> None:
    current = _minute_shares(players)
    if not current:
        return
    target, poss = _measured_team_mix(play_types, team.team_id, season)
    if not target or poss <= 0:
        # No team file. Empty roles already carry the coach's scheme.
        return
    # The file's own possessions set the pull. A short file only nudges.
    data_pull = min(0.70, poss / (poss + 2200.0))
    abbr = (team.abbr or "").lower()
    if poss < 800 and abbr in _TILT:
        scheme = _coach_target(players, abbr)
        # Trust the scheme has not earned yet stays with the coach.
        scheme_pull = _COACH_PULL * (1.0 - poss / (poss + 800.0))
        weight = data_pull + scheme_pull
        blended: dict[str, float] = {}
        for play in PLAY_TYPES:
            now = current.get(play, 0.0)
            blended[play] = (
                data_pull * float(target.get(play, now)) + scheme_pull * float(scheme.get(play, now))
            ) / weight
        target = blended
        pull = min(0.70, weight)
    else:
        pull = data_pull
    if pull <= 0:
        return
    for play in PLAY_TYPES:
        now = current.get(play, 0.0)
        if now <= 1e-6:
            continue
        goal = (1.0 - pull) * now + pull * float(target.get(play, now))
        scale = min(2.2, max(0.45, goal / now))
        for player in players:
            skill = player.offense.get(play)
            if skill is not None:
                skill.poss *= scale


def _measured_team_mix(
    play_types: pd.DataFrame | None, team_id: int, season: str
) -> tuple[dict[str, float], float]:
    """Newest season with a real file. A thinner year is kept until that exists."""
    current = season
    best: dict[str, float] = {}
    best_poss = 0.0
    for _ in range(3):
        shares, poss = _team_shares_one(play_types, team_id, current)
        if shares and poss > best_poss:
            best, best_poss = shares, poss
        if shares and poss >= 800:
            return shares, poss
        current = _previous_season(current)
    return best, best_poss


def _minute_shares(players: list[PlayerModel]) -> dict[str, float]:
    weight = {play: 0.0 for play in PLAY_TYPES}
    for player in players:
        minutes = float(player.minutes_mean)
        if minutes <= 0:
            continue
        for play in PLAY_TYPES:
            skill = player.offense.get(play)
            if skill is not None:
                weight[play] += minutes * float(skill.poss)
    total = sum(weight.values())
    if total <= 0:
        return {}
    return {play: value / total for play, value in weight.items()}


def _coach_target(players: list[PlayerModel], abbr: str) -> dict[str, float]:
    ranks = _ranks(players)
    weight = {play: 0.0 for play in PLAY_TYPES}
    for index, player in enumerate(players):
        minutes = float(player.minutes_mean)
        if minutes <= 0:
            continue
        shares = role_shares(player.position, ranks.get(index, 3), abbr)
        for play, share in shares.items():
            weight[play] += minutes * share
    total = sum(weight.values()) or 1.0
    return {play: value / total for play, value in weight.items()}


def _team_shares_one(
    play_types: pd.DataFrame | None, team_id: int, season: str
) -> tuple[dict[str, float], float]:
    frame = _offensive(play_types, season)
    if frame.empty:
        return {}, 0.0
    frame = frame[frame["entity_type"].astype(str).eq("T")]
    frame = frame[pd.to_numeric(frame["entity_id"], errors="coerce").eq(int(team_id))]
    if frame.empty or "poss_pct" not in frame.columns:
        return {}, 0.0
    shares: dict[str, float] = {}
    poss_total = 0.0
    frame = frame.copy()
    frame["play_type"] = frame["play_type"].map(canon_play)
    for play, group in frame.groupby(frame["play_type"].astype(str)):
        poss = pd.to_numeric(group["poss"], errors="coerce").fillna(0.0)
        weight = float(poss.sum())
        if weight <= 0:
            continue
        poss_total += weight
        pct = pd.to_numeric(group["poss_pct"], errors="coerce").fillna(0.0)
        shares[str(play)] = float((pct * poss).sum() / weight)
    total = sum(shares.values())
    if total <= 0.2:
        return {}, poss_total
    return {play: value / total for play, value in shares.items()}, poss_total


def _previous_season(season: str) -> str:
    start = int(str(season)[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


_MEASURED_CACHE: dict[tuple, dict[str, dict]] = {}
_OFFENSE_SEASON: dict[tuple, pd.DataFrame] = {}


def _offense_season(play_types: pd.DataFrame, season: str) -> pd.DataFrame:
    """One normalized offensive player slice per season for Synergy lookups."""
    key = (id(play_types), str(season))
    cached = _OFFENSE_SEASON.get(key)
    if cached is not None:
        return cached
    if play_types is None or play_types.empty or "player_name" not in play_types.columns:
        empty = pd.DataFrame()
        _OFFENSE_SEASON[key] = empty
        return empty
    frame = play_types
    frame = frame[frame["season"].astype(str).eq(str(season))]
    if "side" in frame.columns:
        frame = frame[frame["side"].astype(str).str.lower().eq("offensive")]
    frame = frame[frame["entity_type"].astype(str).eq("P")]
    if frame.empty:
        _OFFENSE_SEASON[key] = frame
        return frame
    work = frame.copy()
    work["player_key"] = work["player_name"].map(norm_person)
    work["poss"] = pd.to_numeric(work["poss"], errors="coerce").fillna(0.0)
    work["play_type"] = work["play_type"].map(canon_play)
    work["entity_id"] = pd.to_numeric(work["entity_id"], errors="coerce")
    for column in ("ppp", "tov_poss_pct", "ft_poss_pct", "fg_pct", "gp"):
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    if len(_OFFENSE_SEASON) > 24:
        _OFFENSE_SEASON.clear()
    _OFFENSE_SEASON[key] = work
    return work


def _measured_with_prior(
    play_types: pd.DataFrame | None,
    name: str,
    season: str,
    depth: int = 2,
    role_flip: bool = False,
    current_weight: float = _CURRENT_FG_WEIGHT,
    solo_threshold: float | None = None,
) -> dict[str, dict]:
    """A thin action adds the season before it. Possessions per game stay per game."""
    latest = _measured_player(play_types, name, season)
    if not latest:
        if depth <= 0:
            return {}
        return _measured_with_prior(
            play_types,
            name,
            _previous_season(season),
            depth - 1,
            role_flip=False,  # No current-season evidence for a role change.
            current_weight=current_weight,
            solo_threshold=solo_threshold,
        )
    # A role flip trusts this season's FG book alone so prior-year Synergy
    # cannot keep a new starter on last year's make rates.
    if role_flip:
        return latest
    older = _measured_player(play_types, name, _previous_season(season)) if depth else {}
    if not older:
        return latest
    pooled: dict[str, dict] = {}
    solo = float(_MEASURED_POSS if solo_threshold is None else solo_threshold)
    for play in set(latest) | set(older):
        new = latest.get(play)
        old = older.get(play)
        if new is None:
            pooled[play] = old
            continue
        if old is None or float(new["poss"]) >= solo:
            pooled[play] = new
            continue
        poss = float(new["poss"]) + float(old["poss"])
        if poss <= 0:
            continue
        # A thin action from this season counts more than the same number of
        # possessions from the year before it.
        new_w = float(new["poss"]) * (current_weight if float(new["poss"]) < solo else 1.0)
        old_w = float(old["poss"])
        weight = new_w + old_w

        def mix(key: str, default: float, new=new, old=old, new_w=new_w, old_w=old_w, weight=weight) -> float:
            return (float(new.get(key, default)) * new_w + float(old.get(key, default)) * old_w) / weight

        pooled[play] = {
            "poss": poss,
            "ppp": mix("ppp", 1.02),
            "tov": mix("tov", 0.12),
            "ft": mix("ft", 0.08),
            "fg": mix("fg", 0.46),
            # Pool the exposure window across both seasons for per-game rates.
            "gp": float(new.get("gp") or 0) + float(old.get("gp") or 0),
        }
    return pooled


def _measured_player(play_types: pd.DataFrame | None, name: str, season: str) -> dict[str, dict]:
    if play_types is None or play_types.empty:
        return {}
    player_key = norm_person(name)
    cache_key = (id(play_types), str(season), player_key)
    cached = _MEASURED_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    frame = _offense_season(play_types, season)
    if frame.empty or not player_key:
        _MEASURED_CACHE[cache_key] = {}
        return {}
    rows = frame[frame["player_key"].eq(player_key)]
    if rows.empty:
        _MEASURED_CACHE[cache_key] = {}
        return {}
    totals = rows.groupby("entity_id")["poss"].sum()
    if totals.empty or float(totals.max()) <= 0:
        _MEASURED_CACHE[cache_key] = {}
        return {}
    chosen = rows[rows["entity_id"] == totals.idxmax()]
    found: dict[str, dict] = {}
    for play, group in chosen.groupby(chosen["play_type"].astype(str), sort=False):
        poss = float(group["poss"].sum())
        if poss <= 0:
            continue
        w = group["poss"].to_numpy(dtype=float)
        gp = float(group["gp"].max()) if "gp" in group.columns else 1.0
        found[str(play)] = {
            "poss": poss,
            "ppp": _wavg_arr(group["ppp"].to_numpy(dtype=float) if "ppp" in group.columns else None, w, 1.02),
            "tov": _wavg_arr(
                group["tov_poss_pct"].to_numpy(dtype=float) if "tov_poss_pct" in group.columns else None, w, 0.12
            ),
            "ft": _wavg_arr(
                group["ft_poss_pct"].to_numpy(dtype=float) if "ft_poss_pct" in group.columns else None, w, 0.08
            ),
            "fg": _wavg_arr(group["fg_pct"].to_numpy(dtype=float) if "fg_pct" in group.columns else None, w, 0.46),
            "gp": gp if np.isfinite(gp) and gp > 0 else 1.0,
        }
    if len(_MEASURED_CACHE) > 4096:
        _MEASURED_CACHE.clear()
    _MEASURED_CACHE[cache_key] = found
    return dict(found)


def _offensive(play_types: pd.DataFrame | None, season: str) -> pd.DataFrame:
    if play_types is None or play_types.empty:
        return pd.DataFrame()
    frame = play_types
    frame = frame[frame["season"].astype(str).eq(str(season))]
    if "side" not in frame.columns:
        return pd.DataFrame()
    return frame[frame["side"].astype(str).str.lower().eq("offensive")].copy()


def _wavg_arr(values: np.ndarray | None, weights: np.ndarray, default: float) -> float:
    if values is None or weights is None or len(weights) == 0:
        return default
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not ok.any():
        return default
    return float(np.average(values[ok], weights=weights[ok]))


def _wavg(group: pd.DataFrame, column: str, poss: float, default: float) -> float:
    if column not in group.columns or poss <= 0:
        return default
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(group["poss"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return _wavg_arr(values, weights, default)
