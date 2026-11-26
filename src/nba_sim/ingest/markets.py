"""Public market and availability feeds."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

from nba_sim.config import raw_dir
from nba_sim.ingest.common import get_json
from nba_sim.teams import normalize_team
from nba_sim.warehouse import Warehouse

logger = logging.getLogger(__name__)

KAGGLE_SLUG = "cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024"


def ingest_injuries(warehouse: Warehouse) -> pd.DataFrame:
    payload = get_json("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries")
    rows = []
    for block in payload.get("injuries", []) if isinstance(payload, dict) else []:
        team_name = block.get("displayName") or (block.get("team") or {}).get("displayName")
        team = normalize_team(team_name)
        for item in block.get("injuries", []):
            athlete = item.get("athlete") or {}
            details = item.get("details") or item.get("detail") or {}
            if isinstance(details, str):
                detail = details
            else:
                detail = details.get("detail") or details.get("shortComment") or ""
            rows.append(
                {
                    "team_abbr": team,
                    "player_name": athlete.get("displayName") or item.get("displayName"),
                    "status": item.get("status") or "",
                    "detail": detail,
                }
            )
    frame = pd.DataFrame(rows)
    warehouse.write("injuries", frame)
    return frame


def ingest_scoreboard(warehouse: Warehouse, day: str | None = None) -> pd.DataFrame:
    params = {}
    if day:
        params["dates"] = day.replace("-", "")
    payload = get_json(
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
        params,
    )
    rows = []
    for event in payload.get("events", []) if isinstance(payload, dict) else []:
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]
        home = away = None
        for team in comp.get("competitors", []):
            abbr = normalize_team((team.get("team") or {}).get("abbreviation"))
            if team.get("homeAway") == "home":
                home = abbr
            elif team.get("homeAway") == "away":
                away = abbr
        rows.append(
            {
                "date": str(event.get("date", ""))[:10],
                "game_id": str(event.get("id") or ""),
                "home": home,
                "away": away,
                "name": event.get("shortName") or event.get("name"),
            }
        )
    frame = pd.DataFrame(rows)
    warehouse.write("scoreboard", frame)
    return frame


def parse_kaggle(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data.columns = [str(col).strip().lower() for col in data.columns]
    favored = data.get("whos_favored", pd.Series("", index=data.index)).astype(str).str.lower()
    spread = pd.to_numeric(data.get("spread"), errors="coerce").abs()
    home_favored = favored.str.startswith("home") | favored.eq("h")
    away_favored = favored.str.startswith("away") | favored.eq("a")
    home_spread = pd.Series(pd.NA, index=data.index, dtype="Float64")
    home_spread = home_spread.mask(home_favored, -spread)
    home_spread = home_spread.mask(away_favored, spread)
    parsed = pd.DataFrame(
        {
            "date": pd.to_datetime(data.get("date"), errors="coerce"),
            "season": data.get("season"),
            "home": data.get("home", pd.Series(index=data.index)).map(normalize_team),
            "away": data.get("away", pd.Series(index=data.index)).map(normalize_team),
            "home_score": pd.to_numeric(data.get("score_home"), errors="coerce"),
            "away_score": pd.to_numeric(data.get("score_away"), errors="coerce"),
            "home_spread": pd.to_numeric(home_spread, errors="coerce"),
            "total": pd.to_numeric(data.get("total"), errors="coerce"),
            "moneyline_home": pd.to_numeric(data.get("moneyline_home"), errors="coerce"),
            "moneyline_away": pd.to_numeric(data.get("moneyline_away"), errors="coerce"),
        }
    )
    # Historical files often contain only a line and no executable price.
    # Preserve side-specific odds only when the source explicitly provides them;
    # never synthesize -110 for spread/total return calculations.
    for column in (
        "spread_home_american", "spread_away_american",
        "total_over_american", "total_under_american",
    ):
        if column in data.columns:
            parsed[column] = pd.to_numeric(data[column], errors="coerce")
    return parsed.dropna(subset=["date", "home", "away", "home_score", "away_score"])


def load_kaggle(path: Path | None = None, warehouse: Warehouse | None = None) -> pd.DataFrame | None:
    csv_path = path or _find_kaggle()
    if csv_path is None or not csv_path.exists():
        return None
    frame = parse_kaggle(pd.read_csv(csv_path))
    if warehouse is not None:
        warehouse.write("team_lines", frame)
    return frame


def _find_kaggle() -> Path | None:
    raw = raw_dir()
    for name in ("nba_2008-2026.csv", "nba_2008-2025.csv", "nba_betting.csv"):
        candidate = raw / name
        if candidate.exists():
            return candidate
    matches = list(raw.glob("nba_*.csv"))
    return matches[0] if matches else None


def try_download_kaggle() -> Path | None:
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        return _find_kaggle()
    dest = raw_dir()
    try:
        import subprocess

        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(dest), "--unzip"],
            check=False,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("kaggle download failed: %s", exc)
    return _find_kaggle()


def parse_manual_odds(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data.columns = [str(col).strip().lower() for col in data.columns]
    return pd.DataFrame(
        {
            "source": "manual",
            "date": pd.to_datetime(data.get("date"), errors="coerce").dt.strftime("%Y-%m-%d"),
            "home": data.get("home", pd.Series(index=data.index)).map(normalize_team),
            "away": data.get("away", pd.Series(index=data.index)).map(normalize_team),
            "market": data.get("market"),
            "side": data.get("side"),
            "line": pd.to_numeric(data.get("line"), errors="coerce"),
            "american": pd.to_numeric(data.get("odds", data.get("american")), errors="coerce"),
            "player": data.get("player"),
            "book": data.get("book", "manual"),
        }
    )


def load_manual_odds(warehouse: Warehouse | None = None) -> pd.DataFrame | None:
    path = raw_dir() / "slate_odds.csv"
    if not path.exists():
        return None
    frame = parse_manual_odds(pd.read_csv(path))
    if warehouse is not None:
        warehouse.write("markets_manual", frame)
    return frame


def ingest_polymarket(warehouse: Warehouse) -> pd.DataFrame:
    payload = get_json(
        "https://gamma-api.polymarket.com/markets",
        {"tag_id": 745, "closed": "false", "limit": 100},
    )
    markets = payload if isinstance(payload, list) else payload.get("data", payload.get("markets", []))
    rows = []
    for market in markets or []:
        question = str(market.get("question") or market.get("title") or "")
        teams = _teams_in_text(question)
        prices = _jsonish(market.get("outcomePrices"))
        outcomes = _jsonish(market.get("outcomes"))
        if not isinstance(prices, list) or not isinstance(outcomes, list):
            continue
        for outcome, price in zip(outcomes, prices):
            try:
                prob = float(price)
            except (TypeError, ValueError):
                continue
            if prob <= 0 or prob >= 1:
                continue
            american = _prob_to_american(prob)
            rows.append(
                {
                    "source": "polymarket",
                    "date": str(market.get("endDate") or market.get("gameStartTime") or "")[:10],
                    "home": teams[0] if len(teams) > 0 else None,
                    "away": teams[1] if len(teams) > 1 else None,
                    "market": _market_kind(question),
                    "side": normalize_team(outcome) or str(outcome).lower(),
                    "line": _line_from_text(question),
                    "american": american,
                    "player": None,
                    "book": "polymarket",
                    "question": question,
                }
            )
    frame = pd.DataFrame(rows)
    warehouse.write("markets_polymarket", frame)
    return frame


def ingest_kalshi(warehouse: Warehouse) -> pd.DataFrame:
    rows = []
    hosts = (
        "https://external-api.kalshi.com/trade-api/v2",
        "https://api.elections.kalshi.com/trade-api/v2",
    )
    series = ("KXNBAGAME", "KXNBA", "KXNBASPREAD", "KXNBATOTAL")
    for host in hosts:
        for ticker in series:
            try:
                payload = get_json(f"{host}/markets", {"series_ticker": ticker, "status": "open", "limit": 100})
            except Exception as exc:
                logger.info("kalshi %s %s skipped: %s", host, ticker, exc)
                continue
            markets = payload.get("markets", []) if isinstance(payload, dict) else []
            for market in markets:
                title = str(market.get("title") or market.get("yes_sub_title") or "")
                teams = _teams_in_text(title + " " + str(market.get("subtitle") or ""))
                yes = market.get("yes_ask") or market.get("last_price") or market.get("yes_bid")
                try:
                    cents = float(yes)
                except (TypeError, ValueError):
                    continue
                prob = cents / 100.0 if cents > 1 else cents
                if prob <= 0 or prob >= 1:
                    continue
                rows.append(
                    {
                        "source": "kalshi",
                        "date": str(market.get("close_time") or market.get("expiration_time") or "")[:10],
                        "home": teams[0] if teams else None,
                        "away": teams[1] if len(teams) > 1 else None,
                        "market": _market_kind(title),
                        "side": "yes",
                        "line": _line_from_text(title),
                        "american": _prob_to_american(prob),
                        "player": None,
                        "book": "kalshi",
                        "question": title,
                        "ticker": market.get("ticker"),
                    }
                )
        if rows:
            break
    frame = pd.DataFrame(rows)
    warehouse.write("markets_kalshi", frame)
    return frame


def ingest_odds_api(warehouse: Warehouse) -> pd.DataFrame | None:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return None
    payload = get_json(
        "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
        {
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "apiKey": key,
        },
    )
    rows = []
    for event in payload if isinstance(payload, list) else []:
        home = normalize_team(event.get("home_team"))
        away = normalize_team(event.get("away_team"))
        when = str(event.get("commence_time") or "")[:10]
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                key_name = {"h2h": "moneyline", "spreads": "spread", "totals": "total"}.get(market.get("key"))
                if key_name is None:
                    continue
                for outcome in market.get("outcomes", []):
                    side = normalize_team(outcome.get("name")) or str(outcome.get("name", "")).lower()
                    if key_name == "moneyline":
                        side = "home" if side == home else "away" if side == away else side
                    elif key_name == "spread":
                        side = "home" if side == home else "away" if side == away else side
                    rows.append(
                        {
                            "source": "odds_api",
                            "date": when,
                            "home": home,
                            "away": away,
                            "market": key_name,
                            "side": side,
                            "line": outcome.get("point"),
                            "american": outcome.get("price"),
                            "player": None,
                            "book": book.get("key"),
                            "question": "",
                        }
                    )
    frame = pd.DataFrame(rows)
    warehouse.write("markets_odds_api", frame)
    return frame


def combine_markets(warehouse: Warehouse) -> pd.DataFrame:
    frames = []
    for name in ("markets_manual", "markets_polymarket", "markets_kalshi", "markets_odds_api"):
        frame = warehouse.read(name)
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    warehouse.write("markets", combined)
    return combined


def _teams_in_text(text: str) -> list[str]:
    found = []
    lowered = text.lower()
    # Phrase scan is inside normalize_team, so slide a window of words.
    words = lowered.replace(".", " ").split()
    for size in (3, 2, 1):
        for index in range(len(words) - size + 1):
            abbr = normalize_team(" ".join(words[index : index + size]))
            if abbr and abbr not in found:
                found.append(abbr)
            if len(found) == 2:
                return found
    return found


def _market_kind(text: str) -> str:
    lowered = text.lower()
    if "spread" in lowered or "point spread" in lowered:
        return "spread"
    if "over" in lowered or "total" in lowered or "o/u" in lowered:
        return "total"
    return "moneyline"


def _line_from_text(text: str) -> float | None:
    import re

    match = re.search(r"([+-]?\d+\.\d)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _jsonish(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _prob_to_american(prob: float) -> int:
    if prob >= 0.5:
        return int(round(-100 * prob / (1 - prob)))
    return int(round(100 * (1 - prob) / prob))
