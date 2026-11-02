"""Walk-forward team markets on historical lines, using the same simulator.

Ratings for a game are fit only on earlier games. Player props stay an explicit
empty result until a settled prop file is stored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nba_sim.backtest.metrics import brier, calibration_table, closing_line_value, flat_roi, log_loss, max_drawdown
from nba_sim.config import raw_dir
from nba_sim.constants import PLAY_TYPES
from nba_sim.ingest.markets import load_kaggle, parse_kaggle
from nba_sim.markets.devig import american_to_decimal, american_to_implied, devig_american
from nba_sim.sim.engine import simulate_game
from nba_sim.sim.specs import GameContext, GameSpec, LeaguePriors, PlayerModel, PlaySkill, TeamModel
from nba_sim.teams import TEAM_ID_BY_ABBR


def empty_player_props(lines: pd.DataFrame | None = None) -> dict:
    message = (
        "No historical player prop lines are in the warehouse. "
        "Team moneyline, spread, and total can still be scored."
    )
    if lines is not None and not lines.empty and "market" in lines.columns:
        props = lines[lines["market"].astype(str).str.startswith("player_")]
        if not props.empty and {"model_prob", "outcome"}.issubset(props.columns):
            return _score_ready_props(props)
        if not props.empty:
            return {
                "status": "empty",
                "market": "player_props",
                "n_bets": 0,
                "n_rows": int(len(props)),
                "message": (
                    "Player prop lines are stored, but they do not yet have a settled "
                    "outcome and a model probability from a prior-only simulation."
                ),
            }
    return {"status": "empty", "market": "player_props", "n_bets": 0, "message": message}


def _score_ready_props(props: pd.DataFrame) -> dict:
    """Score only settled props with auditable pregame source timestamps."""
    required = (
        "game_id", "source", "snapshot_id", "model_version",
        "available_at", "prediction_at", "game_start_at",
    )
    missing = [column for column in required if column not in props.columns]
    reason = None
    if missing:
        reason = f"Missing prop provenance columns: {', '.join(missing)}"
    else:
        for column in ("game_id", "source", "snapshot_id", "model_version"):
            if props[column].isna().any() or props[column].astype(str).str.strip().eq("").any():
                reason = f"Missing prop identity: {column}"
                break
    if reason is None:
        times = {}
        for column in ("available_at", "prediction_at", "game_start_at"):
            values = []
            for value in props[column]:
                try:
                    stamp = pd.Timestamp(value)
                except (TypeError, ValueError, OverflowError):
                    reason = f"Invalid prop timestamp: {column}"
                    break
                if pd.isna(stamp) or stamp.tzinfo is None:
                    reason = f"Prop timestamp must be timezone-aware: {column}"
                    break
                values.append(stamp.tz_convert("UTC"))
            if reason is not None:
                break
            times[column] = pd.DatetimeIndex(values)
        if reason is None and (
            (times["available_at"] > times["prediction_at"]).any()
            or (times["prediction_at"] >= times["game_start_at"]).any()
        ):
            reason = "Prop prices or predictions were not available before tipoff"
    prob = pd.to_numeric(props["model_prob"], errors="coerce").to_numpy(dtype=float)
    outcome = pd.to_numeric(props["outcome"], errors="coerce").to_numpy(dtype=float)
    if reason is None and (
        not np.isfinite(prob).all() or not np.isfinite(outcome).all()
        or ((prob < 0) | (prob > 1)).any()
        or (~np.isin(outcome, [0.0, 1.0])).any()
    ):
        reason = "Prop probabilities must be in [0,1] and settled outcomes must be binary"
    if reason is not None:
        return {
            "status": "unevaluable",
            "market": "player_props",
            "n_bets": 0,
            "n_rows": int(len(props)),
            "message": reason,
        }
    return {
        "status": "ok",
        "market": "player_props",
        "n_bets": int(len(props)),
        "log_loss": log_loss(prob, outcome),
        "brier": brier(prob, outcome),
        "message": "Scored from timestamped pregame probabilities and settled outcomes.",
    }


def run_backtest(
    path: Path | None = None,
    last_n: int = 8,
    n_sims: int = 800,
    seed: int = 1,
    lines: pd.DataFrame | None = None,
) -> dict:
    if lines is None:
        csv_path = path or _default_lines_path()
        if csv_path is None or not Path(csv_path).exists():
            return {
                "status": "missing_lines",
                "message": (
                    "Historical team lines were not found at "
                    f"{path or raw_dir() / 'nba_2008-2026.csv'}. "
                    "Download the public Kaggle file or copy the CSV into data/raw. "
                    "The slate can still be simulated without it."
                ),
                "player_props": empty_player_props(),
                "n_games": 0,
            }
        lines = load_kaggle(Path(csv_path))
    if lines is None or lines.empty:
        return {
            "status": "missing_lines",
            "message": "The line file did not contain any scored games.",
            "player_props": empty_player_props(),
            "n_games": 0,
        }
    rated = _pregame_ratings(lines)
    window = rated.tail(last_n)
    bets = []
    for index, game in enumerate(window.itertuples(index=False)):
        spec = _composite_spec(game)
        result = simulate_game(spec, n_sims=n_sims, seed=seed + index)
        bets.extend(_grade(game, result))
    frame = pd.DataFrame(bets)
    n_games = 0
    if not frame.empty and {"date", "home", "away"}.issubset(frame.columns):
        n_games = int(frame.groupby(["date", "home", "away"]).ngroups)
    settled_ml = frame.loc[frame["bet_market"].eq("moneyline") & frame["won"].notna()] if not frame.empty else frame
    return {
        "status": "ok",
        "message": f"Walk-forward on the last {n_games} games with ratings fit only on earlier games.",
        "n_games": n_games,
        "metrics": _metrics(frame),
        "calibration": calibration_table(
            settled_ml["model_prob"] if "model_prob" in settled_ml else [],
            settled_ml["won"] if "won" in settled_ml else [],
        ),
        "player_props": empty_player_props(lines if "market" in lines.columns else None),
        "bets": frame.to_dict("records"),
    }


def _default_lines_path() -> Path | None:
    raw = raw_dir()
    for name in ("nba_2008-2026.csv", "nba_2008-2025.csv"):
        if (raw / name).exists():
            return raw / name
    matches = sorted(raw.glob("nba_*.csv"))
    return matches[0] if matches else None


def _pregame_ratings(lines: pd.DataFrame) -> pd.DataFrame:
    """Predict an entire day's slate before learning any result from that day.

    Same-day outcomes are pooled into commutative updates. This prevents CSV
    ordering from changing either the current slate or the following day.
    """
    strength: dict[str, float] = {}
    scoring: dict[str, float] = {}
    hca = 2.2
    rows = []
    ordered = lines.copy()
    ordered["_rating_date"] = pd.to_datetime(ordered["date"]).dt.normalize()
    if ordered["_rating_date"].isna().any():
        raise ValueError("Backtest contains an invalid game date")
    ordered = ordered.sort_values("_rating_date", kind="stable")
    for _, day in ordered.groupby("_rating_date", sort=True):
        strength_delta: dict[str, float] = {}
        score_observations: dict[str, list[float]] = {}
        hca_residuals = []
        for game in day.drop(columns=["_rating_date"]).itertuples(index=False):
            home, away = game.home, game.away
            hs = strength.get(home, 0.0)
            aws = strength.get(away, 0.0)
            ht = scoring.get(home, 112.0)
            at = scoring.get(away, 112.0)
            pred_margin = hs - aws + hca
            rows.append(
                {
                    "date": game.date,
                    "home": home,
                    "away": away,
                    "home_score": game.home_score,
                    "away_score": game.away_score,
                    "home_spread": game.home_spread,
                    "total": game.total,
                    "moneyline_home": game.moneyline_home,
                    "moneyline_away": game.moneyline_away,
                    "pred_margin": pred_margin,
                    "pred_total": ht + at,
                    **{
                        name: getattr(game, name, None)
                        for name in (
                            "spread_home_american", "spread_away_american",
                            "total_over_american", "total_under_american",
                        )
                    },
                }
            )
            home_score = float(game.home_score)
            away_score = float(game.away_score)
            margin = home_score - away_score
            err = margin - pred_margin
            strength_delta[home] = strength_delta.get(home, 0.0) + 0.12 * err
            strength_delta[away] = strength_delta.get(away, 0.0) - 0.12 * err
            score_observations.setdefault(home, []).append(home_score)
            score_observations.setdefault(away, []).append(away_score)
            hca_residuals.append(margin - (hs - aws))
        for team, delta in strength_delta.items():
            strength[team] = strength.get(team, 0.0) + delta
        for team, scores in score_observations.items():
            scoring[team] = 0.9 * scoring.get(team, 112.0) + 0.1 * float(np.mean(scores))
        hca = 0.98 * hca + 0.02 * float(np.mean(hca_residuals))
    return pd.DataFrame(rows)


def _composite_spec(game) -> GameSpec:
    home_mean = (float(game.pred_total) + float(game.pred_margin)) / 2
    away_mean = (float(game.pred_total) - float(game.pred_margin)) / 2
    league = LeaguePriors.flat(1.02)
    home_players, home_team = _side(game.home, home_mean, 1)
    away_players, away_team = _side(game.away, away_mean, 2)
    return GameSpec(
        game_id=f"bt-{game.date}-{game.away}-{game.home}",
        date=str(game.date)[:10],
        home=home_team,
        away=away_team,
        home_players=home_players,
        away_players=away_players,
        context=GameContext(),
        league=league,
        source="backtest",
    )


def _side(abbr: str, mean_points: float, slot: int) -> tuple[list[PlayerModel], TeamModel]:
    team_id = TEAM_ID_BY_ABBR.get(abbr, 1_000_000 + slot)
    ppp = max(0.85, mean_points / 100.0)
    players = []
    for index, position in enumerate(("G", "G", "F", "F", "C")):
        offense = {
            play: PlaySkill(poss=80.0 if play == "Misc" else 4.0, ppp=ppp, tov_poss_pct=0.11, ft_poss_pct=0.08)
            for play in PLAY_TYPES
        }
        players.append(
            PlayerModel(
                player_id=team_id * 10 + index,
                name=f"{abbr.upper()} {position}{index}",
                team_id=team_id,
                position=position,
                minutes_mean=48,
                minutes_std=0.2,
                offense=offense,
                oreb_pct=0.012,
                dreb_pct=0.16,
                ast_share=1.0,
            )
        )
    team = TeamModel(
        team_id=team_id,
        abbr=abbr,
        name=abbr.upper(),
        pace=100,
        off_rating=ppp * 100,
        def_rating=102,
        games=40,
        off_ppp={play: ppp for play in PLAY_TYPES},
        off_poss={play: 400.0 for play in PLAY_TYPES},
        def_ppp={play: 1.02 for play in PLAY_TYPES},
        def_poss={play: 800.0 for play in PLAY_TYPES},
    )
    return players, team


def _grade(game, result) -> list[dict]:
    home = result.home_points
    away = result.away_points
    total = home + away
    home_win = float(np.mean(home > away) + 0.5 * np.mean(home == away))
    base = {
        "date": str(game.date)[:10],
        "home": game.home,
        "away": game.away,
        "home_score": game.home_score,
        "away_score": game.away_score,
    }
    rows = []
    ml_home = _recorded_price(game.moneyline_home)
    ml_away = _recorded_price(game.moneyline_away)
    if ml_home is not None and ml_away is not None:
        fair_home, fair_away = devig_american([ml_home, ml_away])
        home_won = float(game.home_score) > float(game.away_score)
        if home_win - fair_home >= (1 - home_win) - fair_away and home_win - fair_home > 0.02:
            rows.append({**base, **_bet("moneyline", home_win, home_won, ml_home)})
        elif (1 - home_win) - fair_away > 0.02:
            rows.append({**base, **_bet("moneyline", 1 - home_win, not home_won, ml_away)})
        else:
            rows.append({**base, **_watch("moneyline", home_win, home_won)})
    spread = _num(game.home_spread)
    if spread is not None:
        cover = float(np.mean((home - away + spread) > 0) + 0.5 * np.mean((home - away + spread) == 0))
        margin = float(game.home_score) - float(game.away_score) + spread
        if margin == 0:
            rows.append({**base, **_push("spread", cover)})
        elif cover - 0.5 > 0.03:
            price = _recorded_price(getattr(game, "spread_home_american", None))
            selection = _bet("spread", cover, margin > 0, price) if price is not None else _unpriced("spread", cover, margin > 0)
            rows.append({**base, **selection})
        elif 0.5 - cover > 0.03:
            price = _recorded_price(getattr(game, "spread_away_american", None))
            selection = _bet("spread", 1 - cover, margin < 0, price) if price is not None else _unpriced("spread", 1 - cover, margin < 0)
            rows.append({**base, **selection})
        else:
            rows.append({**base, **_watch("spread", cover, margin > 0)})
    total_line = _num(game.total)
    if total_line is not None:
        over = float(np.mean(total > total_line) + 0.5 * np.mean(total == total_line))
        actual_points = float(game.home_score) + float(game.away_score)
        if actual_points == total_line:
            rows.append({**base, **_push("total", over)})
            return rows
        actual_over = actual_points > total_line
        if over - 0.5 > 0.03:
            price = _recorded_price(getattr(game, "total_over_american", None))
            selection = _bet("total", over, actual_over, price) if price is not None else _unpriced("total", over, actual_over)
            rows.append({**base, **selection})
        elif 0.5 - over > 0.03:
            price = _recorded_price(getattr(game, "total_under_american", None))
            selection = _bet("total", 1 - over, not actual_over, price) if price is not None else _unpriced("total", 1 - over, not actual_over)
            rows.append({**base, **selection})
        else:
            rows.append({**base, **_watch("total", over, actual_over)})
    return rows


def _recorded_price(value) -> float | None:
    """Require a plausible, explicitly recorded American price."""
    price = _num(value)
    return price if price is not None and (price <= -100 or price >= 100) else None


def _unpriced(market: str, prob: float, won: bool) -> dict:
    """Keep the forecast for calibration, but never fabricate betting returns."""
    return {**_watch(market, prob, won), "price_status": "missing"}


def _push(market: str, prob: float) -> dict:
    """A settled push returns the stake and is excluded from binary calibration."""
    return {
        "bet_market": market,
        "model_prob": prob,
        "won": np.nan,
        "decimal_odds": np.nan,
        "profit": 0.0,
        "american": np.nan,
        "implied": np.nan,
        "bet": False,
        "settlement": "push",
    }


def _watch(market: str, prob: float, won: bool) -> dict:
    return {
        "bet_market": market,
        "model_prob": prob,
        "won": float(won),
        "decimal_odds": np.nan,
        "profit": 0.0,
        "american": np.nan,
        "implied": np.nan,
        "bet": False,
    }


def _bet(market: str, prob: float, won: bool, american: float) -> dict:
    decimal = american_to_decimal(american)
    profit = decimal - 1 if won else -1
    return {
        "bet_market": market,
        "model_prob": prob,
        "won": float(won),
        "decimal_odds": decimal,
        "profit": profit,
        "american": american,
        "implied": american_to_implied(american),
        "bet": True,
        "price_status": "recorded",
    }


def _metrics(frame: pd.DataFrame) -> dict:
    output = {}
    if frame.empty or "bet_market" not in frame.columns:
        for market in ("moneyline", "spread", "total"):
            output[market] = {
                "n": 0,
                "log_loss": None,
                "brier": None,
                "roi": None,
                "drawdown": None,
                "clv": None,
            }
        output["note"] = "CLV is empty when the file has a single closing number and no earlier price."
        return output
    for market in ("moneyline", "spread", "total"):
        rows = frame[frame["bet_market"] == market]
        settled = rows.loc[rows["won"].notna()]
        bets = settled[settled["bet"] == True] if "bet" in settled.columns else settled.iloc[0:0]  # noqa: E712
        output[market] = {
            "n": int(len(bets)),
            "n_unpriced": int(rows["price_status"].eq("missing").sum()) if "price_status" in rows else 0,
            "n_pushes": int(rows["settlement"].eq("push").sum()) if "settlement" in rows else 0,
            "log_loss": None if settled.empty else log_loss(settled["model_prob"], settled["won"]),
            "brier": None if settled.empty else brier(settled["model_prob"], settled["won"]),
            "roi": None if bets.empty else flat_roi(bets["won"].astype(bool), bets["decimal_odds"]),
            "drawdown": None if bets.empty else max_drawdown(bets["profit"]),
            "clv": closing_line_value([], []),
        }
    output["note"] = "CLV is empty when the file has a single closing number and no earlier price."
    return output


def _num(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def parse_lines_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return parse_kaggle(frame)
