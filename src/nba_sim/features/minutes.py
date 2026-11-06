"""Minutes allocation. The night fills 240 from the players on the team."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MinutePlayer:
    player_id: int
    mean: float
    std: float
    available: bool


def rotation_player_ids(players) -> set[int]:
    """Every available player with a minute prior.

    How many of them actually play is already in the prior. A coach who plays
    eight leaves the next man at a spot minute. A coach who plays eleven keeps
    that eleventh man. Neither decision is a league-wide cut after ten.
    """
    return {
        player.player_id
        for player in players
        if player.available and player.minutes_mean > 0
    }


def fill_night(
    means: np.ndarray,
    target: float = 240.0,
    order_by: np.ndarray | None = None,
) -> np.ndarray:
    """Give the 240 minutes to the largest roles first.

    A player keeps his own number until the minutes run out. The next man
    takes what is left. A sheet that already fits stays as it is. When
    ``order_by`` is set, the fill walks that ranking instead of the noisy
    amounts, so a soft star draw cannot lose his slot to a hot bench draw.
    """
    values = np.asarray(means, dtype=float)
    single = values.ndim == 1
    if single:
        values = values.reshape(1, -1)
    values = np.clip(values, 0.0, None)
    ranking = values if order_by is None else np.asarray(order_by, dtype=float)
    if ranking.ndim == 1:
        ranking = ranking.reshape(1, -1)
    if ranking.shape != values.shape:
        ranking = np.broadcast_to(ranking, values.shape)
    out = np.zeros_like(values)
    order = np.argsort(-ranking, axis=1, kind="stable")
    for row, indexes in enumerate(order):
        remaining = float(target)
        for index in indexes:
            role = float(values[row, index])
            if role <= 0.0 or remaining <= 1e-9:
                continue
            take = min(role, remaining)
            out[row, index] = take
            remaining -= take
    if single:
        return out[0]
    return out


def draw_night_minutes(
    players: list[MinutePlayer],
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """A night around the minutes the active list can actually play.

    The largest role is filled first, up to 240. The draw moves around that
    number, and the refill walks the prior role order so a bench spike cannot
    steal a star's slot in the 240.
    """
    means = np.array([p.mean if p.available else 0.0 for p in players], dtype=float)
    stds = np.array([max(0.0, p.std) if p.available else 0.0 for p in players], dtype=float)
    if float(means.sum()) <= 0:
        return np.zeros((n_sims, len(players)), dtype=float)
    targets = fill_night(means)
    raw = rng.normal(targets, stds, size=(n_sims, len(players)))
    raw = np.clip(raw, 0.0, 44.0)
    raw[:, targets <= 0] = 0.0
    return fill_night(raw, order_by=targets)


def allocate_minutes(
    players: list[MinutePlayer],
    n_sims: int,
    target: float = 240.0,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw minutes and rescale the available rotation to ``target``.

    An unavailable player stays at zero. Everyone else absorbs that playing time.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    n = len(players)
    available = np.array([p.available for p in players], dtype=bool)
    # Never resurrect injured/inactive players to fill an impossible slate.
    # An all-out roster has no legal minutes allocation; return zero minutes
    # rather than assigning 240 minutes to known absences.
    if not available.any():
        return np.zeros((n_sims, n), dtype=float)
    means = np.array(
        [p.mean if bool(available[i]) else 0.0 for i, p in enumerate(players)],
        dtype=float,
    )
    stds = np.array(
        [max(0.0, p.std) if bool(available[i]) else 0.0 for i, p in enumerate(players)],
        dtype=float,
    )
    if float(means.sum()) <= 0:
        means = np.where(available, target / int(available.sum()), 0.0)
        stds = np.zeros(n, dtype=float)
    raw = rng.normal(means, stds, size=(n_sims, n))
    raw = np.clip(raw, 0.0, 46.0)
    raw[:, ~available] = 0.0
    totals = raw.sum(axis=1, keepdims=True)
    totals = np.maximum(totals, 1e-6)
    out = raw * (target / totals)
    out[:, ~available] = 0.0
    return out
