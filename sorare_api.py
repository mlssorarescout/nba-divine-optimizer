"""
Sorare GraphQL pulls for the NBA Divine optimizer.

* `fetch_gallery`      – every NBA card a user owns, cursor-paginated
                         (`pageInfo { endCursor hasNextPage }`), (common rarity = Divination),
                         with its edition bonus multiplier.
* `fetch_projections`  – per-player projected scores + L10 for all 30 teams
                         (same fields as nba_player_scoring_proj.py).

The key is never hard-coded: pass it in (Streamlit secrets / env var).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

ENDPOINT = "https://api.sorare.com/federation/graphql"
JWT_AUD = "scout_data_scrape"

TEAMS = [
    "atlanta-hawks", "boston-celtics", "brooklyn-nets", "charlotte-hornets",
    "chicago-bulls", "cleveland-cavaliers", "dallas-mavericks", "denver-nuggets",
    "detroit-pistons", "golden-state-warriors", "houston-rockets", "indiana-pacers",
    "la-clippers", "los-angeles-lakers", "memphis-grizzlies", "miami-heat",
    "milwaukee-bucks", "minnesota-timberwolves", "new-orleans-pelicans",
    "new-york-knicks", "oklahoma-city-thunder", "orlando-magic",
    "philadelphia-76ers", "phoenix-suns", "portland-trail-blazers",
    "sacramento-kings", "san-antonio-spurs", "toronto-raptors", "utah-jazz",
    "washington-wizards",
]

PAGE_SIZE = 200
REQUEST_DELAY = 0.15


class SorareError(RuntimeError):
    """Anything that stopped a pull."""


def _post(api_key: str, query: str, variables: dict, retries: int = 4) -> dict:
    headers = {"Content-Type": "application/json", "JWT-AUD": JWT_AUD}
    if api_key:
        headers["APIKEY"] = api_key
    for attempt in range(retries):
        resp = requests.post(
            ENDPOINT, json={"query": query, "variables": variables},
            headers=headers, timeout=30,
        )
        if resp.status_code == 429:
            time.sleep(float(resp.headers.get("Retry-After", 2 ** attempt)))
            continue
        if resp.status_code >= 400:
            raise SorareError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        if body.get("errors"):
            raise SorareError("; ".join(e.get("message", str(e)) for e in body["errors"]))
        return body["data"]
    raise SorareError("Rate limited too many times (429).")


# --------------------------------------------------------------------------
# Gallery
# --------------------------------------------------------------------------

_GALLERY = """
query NbaGallery($slug: String!, $first: Int!, $after: String) {
  user(slug: $slug) {
    slug
    cards(sport: NBA, rarities: [common], first: $first, after: $after) {
      ... on AnyCardInterfaceConnection {
        pageInfo { endCursor hasNextPage }
        nodes {
          slug
          name
          pictureUrl
          inSeasonUntil
          anyPlayer { slug }
          cardSet { name }
          skinEdition { bonus variantName }
        }
      }
    }
  }
}
"""

_COLS = ["card_slug", "player_slug", "bonus", "edition", "set_name", "in_season_until", "picture_url"]


def _to_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _multiplier(bonus) -> float:
    """skinEdition.bonus -> multiplier. Accepts 0.05 or 1.05 style; missing = 1.0."""
    b = _to_float(bonus, 0.0)
    return 1.0 + b if b < 0.5 else b


def fetch_gallery(username: str, api_key: str) -> pd.DataFrame:
    """Common NBA cards (Divination cards are common), one row per card."""
    username = (username or "").strip().lower()
    if not username:
        raise SorareError("Enter a Sorare username (slug).")

    rows, cursor = [], None
    while True:
        data = _post(api_key, _GALLERY, {"slug": username, "first": PAGE_SIZE, "after": cursor})
        user = data.get("user")
        if user is None:
            raise SorareError(f"No Sorare user found with slug '{username}'.")
        cards = user.get("cards") or {}
        for n in cards.get("nodes") or []:
            p = (n or {}).get("anyPlayer") or {}
            if not p.get("slug"):
                continue
            skin = n.get("skinEdition") or {}
            rows.append({
                "card_slug": n.get("slug"),
                "player_slug": p["slug"],
                "bonus": _multiplier(skin.get("bonus")),
                "edition": skin.get("variantName"),
                "set_name": (n.get("cardSet") or {}).get("name"),
                "in_season_until": n.get("inSeasonUntil"),
                "picture_url": n.get("pictureUrl"),
            })
        info = cards.get("pageInfo") or {}
        cursor = info.get("endCursor")
        if not info.get("hasNextPage"):
            break
        time.sleep(REQUEST_DELAY)

    return pd.DataFrame(rows, columns=_COLS)


# --------------------------------------------------------------------------
# Per-game projections
# --------------------------------------------------------------------------

_UPCOMING = """
query upcoming($slug: String!, $n: Int!) {
  team(slug: $slug) {
    upcomingAnyGames(first: $n) { id date }
  }
}
"""

_GAME = """
query game_projections($id: ID!) {
  anyGame(id: $id) {
    date
    playerGameScores {
      anyPlayer {
        slug
        displayName
        gameplayTierStars
        averageScore(type: LAST_TEN_PLAYED_SO5_AVERAGE_SCORE)
      }
      projectedScore
    }
  }
}
"""


def _upcoming_games(api_key: str, slug: str, n: int) -> list[dict]:
    time.sleep(REQUEST_DELAY)
    team = (_post(api_key, _UPCOMING, {"slug": slug, "n": n}).get("team")) or {}
    return [g for g in (team.get("upcomingAnyGames") or []) if g and g.get("id")]


def _game_rows(api_key: str, game: dict) -> list[dict]:
    time.sleep(REQUEST_DELAY)
    g = (_post(api_key, _GAME, {"id": game["id"]}).get("anyGame")) or {}
    rows = []
    for s in g.get("playerGameScores") or []:
        p = (s or {}).get("anyPlayer") or {}
        if p.get("slug"):
            rows.append({
                "game_id": game["id"],
                "game_date": g.get("date") or game.get("date"),
                "player_slug": p["slug"],
                "display_name": p.get("displayName"),
                "stars": p.get("gameplayTierStars"),
                "l10": p.get("averageScore"),
                "proj": s.get("projectedScore"),
            })
    return rows


def fetch_game_projections(api_key: str, max_days: int = 7) -> pd.DataFrame:
    """One row per (upcoming game, player) with that game's projected score.

    Pulls each team's next games, dedupes game ids across teams, and keeps games
    within `max_days` of now. Day windowing happens in the app.
    """
    with ThreadPoolExecutor(max_workers=4) as pool:
        per_team = list(pool.map(lambda t: _upcoming_games(api_key, t, max_days + 3), TEAMS))
    games = {g["id"]: g for chunk in per_team for g in chunk}
    if not games:
        raise SorareError("No upcoming NBA games found.")
    cutoff = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=max_days + 1)
    games = [g for g in games.values()
             if not g.get("date") or pd.to_datetime(g["date"], utc=True) <= cutoff]

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [r for chunk in pool.map(lambda g: _game_rows(api_key, g), games) for r in chunk]
    if not rows:
        raise SorareError("Projection pull returned no players.")
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"], utc=True, errors="coerce")
    for c in ("proj", "l10", "stars"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["game_date"]).drop_duplicates(["game_id", "player_slug"])
