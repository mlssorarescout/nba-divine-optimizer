"""
NBA Divine Optimizer
====================

Enter a Sorare username, hit Run: pulls the user's NBA cards and per-game
projections, then builds the 5-player, 16-Star lineup that maximizes projected
score x card bonus over the chosen date/time window (default: the rest of
today's games).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import os

import pandas as pd
import streamlit as st

import sorare_api
from optimizer import solve_many

TZ = "America/New_York"  # NBA slate days
MAX_DAYS = 7
LINEUP_SIZE = 5  # Divination lineup

st.set_page_config(page_title="NBA Divine Optimizer", layout="wide")
st.title("NBA Divine Optimizer")

st.markdown("""
<style>
.statrow { display: flex; justify-content: center; gap: 2.5rem; margin: 0.25rem 0 1rem; }
.statrow div { text-align: center; }
.statrow span { display: block; font-size: 0.75rem; opacity: 0.7; }
.statrow b { font-size: 1.1rem; }
.lineup-title, .lineup-totals { text-align: center; font-weight: 700; }
.lineup-totals { margin-bottom: 0.75rem; }
[data-testid="stSlider"] { padding: 0.5rem 1.5rem 1.5rem; }
[data-testid="stSlider"] label p { font-size: 1.15rem; font-weight: 700; }
[data-testid="stSlider"] div[role="slider"] { width: 1.75rem; height: 1.75rem; }
[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child { height: 0.6rem; }
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"] { font-size: 0.95rem; }
.player-info { text-align: center; font-size: 1.05rem; font-weight: 600; line-height: 1.5;
               opacity: 1; margin-top: 0.4rem; }
.player-info .pname { font-size: 1.15rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)


def _get_api_key() -> str:
    """Streamlit secrets (deployed) or SORARE_API_KEY env var (local)."""
    try:
        return st.secrets.get("SORARE_API_KEY", "")
    except Exception:
        return os.getenv("SORARE_API_KEY", "")


API_KEY = _get_api_key()


@st.cache_data(ttl=900, show_spinner=False)
def load_projections(api_key: str) -> pd.DataFrame:
    return sorare_api.fetch_game_projections(api_key, MAX_DAYS)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Run")
    username = st.text_input("Sorare username (slug)", value=st.session_state.get("username", ""),
                             placeholder="e.g. mls-sorare-scout")
    run = st.button("Run", type="primary", use_container_width=True)
    if st.button("Refresh projections", use_container_width=True):
        load_projections.clear()
        run = True

if run and not API_KEY:
    st.error("SORARE_API_KEY is not set. Add it to Streamlit secrets (or the environment).")
elif run:
    try:
        with st.spinner("Pulling gallery (paginating)..."):
            cards = sorare_api.fetch_gallery(username, API_KEY)
        with st.spinner("Pulling per-game projections for all 30 teams..."):
            games = load_projections(API_KEY)
        st.session_state.update(username=username, cards=cards, games=games)
    except sorare_api.SorareError as e:
        st.error(str(e))

if "cards" not in st.session_state:
    st.info("Enter a username and hit Run to get started.")
    st.stop()

cards: pd.DataFrame = st.session_state["cards"]
games: pd.DataFrame = st.session_state["games"].copy()

# Game start in Eastern time (naive, for the slider).
games["game_et"] = games["game_date"].dt.tz_convert(TZ).dt.tz_localize(None)
now_et = pd.Timestamp.now(tz=TZ).tz_localize(None).floor("15min")
tonight = now_et.normalize() + pd.Timedelta(days=1)
today_games = games.loc[games["game_et"] < tonight, "game_et"]
default_end = (today_games.max().ceil("15min") + pd.Timedelta(minutes=15)) if len(today_games) else tonight
max_end = now_et + pd.Timedelta(days=MAX_DAYS)

# ---------------------------------------------------------------- settings
with st.sidebar:
    st.header("2. Lineup settings")
    use_cap = st.checkbox("Star cap", value=True)
    cap = st.number_input("Max total Stars", 0.0, 100.0, 16.0, 1.0, disabled=not use_cap,
                          help="Divination: 5 players, 16 Stars max (Stars come from L10).")
    n_lineups = st.number_input("Lineups to build", 1, 10, 1,
                                help="Each extra lineup excludes cards already used.")
    min_bonus = st.number_input("Min card bonus", 0.5, 3.0, 1.0, 0.01, format="%.2f")

start, end = st.slider(
    "Game window (Eastern): drag to pick tip-off times", min_value=now_et.to_pydatetime(), max_value=max_end.to_pydatetime(),
    value=(now_et.to_pydatetime(), min(default_end, max_end).to_pydatetime()),
    step=pd.Timedelta(minutes=15).to_pytimedelta(), format="ddd MM/DD h:mm A",
    help="Games starting inside this window. Default is the rest of today's slate.")

window = games[(games["game_et"] >= start) & (games["game_et"] <= end)].dropna(subset=["proj"])
if window.empty:
    nxt = games.loc[games["game_et"] >= now_et, "game_et"].min()
    st.warning("No games with projections in this window."
               + (f" Next game starts {nxt:%a %m/%d %I:%M %p} ET; widen the slider." if pd.notna(nxt) else ""))
    st.stop()

# A player's value is their best single game in the window.
per_player = (window.groupby("player_slug")
                    .agg(proj=("proj", "max"), games=("game_id", "nunique"),
                         next_game=("game_et", "min"), display_name=("display_name", "first"),
                         stars=("stars", "first"), l10=("l10", "first"))
                    .reset_index())
per_player["next_game"] = per_player["next_game"].dt.strftime("%a %m/%d %I:%M%p")

pool = cards.merge(per_player, on="player_slug", how="inner")
pool = pool[pool["bonus"] >= min_bonus]

stats = [("Cards pulled", len(cards)), ("Unique players", cards["player_slug"].nunique()),
         ("Games in window", window["game_id"].nunique()), ("Cards playing", len(pool))]
st.markdown(
    '<div class="statrow">' + "".join(
        f'<div><span>{label}</span><b>{value}</b></div>' for label, value in stats) + "</div>",
    unsafe_allow_html=True)

names = pool.drop_duplicates("player_slug").set_index("player_slug")["display_name"].fillna("").to_dict()
fmt = lambda s: f"{names.get(s) or s}"
with st.sidebar:
    locked = st.multiselect("Lock in", sorted(names), format_func=fmt)
    excluded = st.multiselect("Exclude", sorted(names), format_func=fmt)

# ---------------------------------------------------------------- solve
lineups, reason = solve_many(
    pool, LINEUP_SIZE, int(n_lineups),
    cap=float(cap) if use_cap else None,
    locked=set(locked), excluded=set(excluded),
)

for i, lu in enumerate(lineups, 1):
    with st.container(border=True):
        st.markdown(f'<h3 class="lineup-title">Lineup {i}</h3>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="lineup-totals">Bonus-adjusted {lu["adj_proj"].sum():.2f} &nbsp;|&nbsp; '
            f'Raw {lu["proj"].sum():.2f} &nbsp;|&nbsp; '
            f'{pd.to_numeric(lu["stars"], errors="coerce").sum():.0f} Stars</div>',
            unsafe_allow_html=True)

        # Horizontal stack of the selected players' cards.
        for col, (_, r) in zip(st.columns(len(lu)), lu.iterrows()):
            if isinstance(r.get("picture_url"), str):
                col.image(r["picture_url"], use_container_width=True)
            col.markdown(
                f'<div class="player-info"><div class="pname">{html.escape(str(r["display_name"]))}</div>'
                f'{html.escape(str(r["edition"] or "Standard"))}<br>{r["adj_proj"]:.1f} pts &middot; '
                f'{r["stars"]:.0f} Stars<br>{r["next_game"]}</div>',
                unsafe_allow_html=True)

        st.download_button("Download CSV", lu.to_csv(index=False).encode("utf-8-sig"),
                           f"divine_lineup_{i}.csv", key=f"dl{i}")

if reason and len(lineups) < n_lineups:
    st.warning(f"Built {len(lineups)} of {int(n_lineups)} lineups: {reason}")

with st.expander("All candidate cards"):
    st.dataframe(pool.sort_values("proj", ascending=False), hide_index=True,
                 use_container_width=True)
