"""
Lineup optimizer: exact MILP (scipy/HiGHS).

Maximize   sum( projected_score_i * bonus_i )          bonus 1.05 = +5%
subject to exactly K cards,
           at most one card per player,
           optional cap on sum(Stars) (Divination: 16 max; Stars come from L10),
           optional locked / excluded players.

Input frame columns: card_slug, player_slug, proj, bonus, stars (+ anything else,
which is carried through).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


def solve_lineup(pool: pd.DataFrame, k: int, cap: float | None = None,
                 locked: set[str] | None = None, excluded: set[str] | None = None):
    """Return (lineup_df, None) or (None, reason)."""
    locked, excluded = set(locked or ()), set(excluded or ())
    df = pool[~pool["player_slug"].isin(excluded)].reset_index(drop=True)
    df = df.dropna(subset=["proj", "bonus"]).reset_index(drop=True)
    n = len(df)
    if n < k:
        return None, f"Only {n} eligible cards for {k} slots."

    value = (df["proj"] * df["bonus"]).to_numpy(float)
    stars = pd.to_numeric(df["stars"], errors="coerce").fillna(0).to_numpy(float)

    rows, lo, hi = [np.ones(n)], [k], [k]
    if cap is not None:
        rows.append(stars); lo.append(-np.inf); hi.append(cap)
    for _, idx in df.groupby("player_slug").indices.items():
        if len(idx) > 1:
            r = np.zeros(n); r[idx] = 1
            rows.append(r); lo.append(-np.inf); hi.append(1)

    lb = np.zeros(n)
    lb[df["player_slug"].isin(locked).to_numpy()] = 1  # lock every copy; the one-per-player row picks one

    res = milp(
        c=-value,
        constraints=LinearConstraint(np.vstack(rows), lo, hi),
        integrality=np.ones(n),
        bounds=Bounds(lb, np.ones(n)),
    )
    if not res.success:
        return None, "No feasible lineup under these constraints."

    out = df[res.x > 0.5].copy()
    out["adj_proj"] = out["proj"] * out["bonus"]
    return out.sort_values("adj_proj", ascending=False).reset_index(drop=True), None


def solve_many(pool: pd.DataFrame, k: int, n_lineups: int, **kw):
    """Greedy successive lineups; each card is used at most once overall."""
    lineups, reason = [], None
    remaining = pool
    for _ in range(n_lineups):
        lu, reason = solve_lineup(remaining, k, **kw)
        if lu is None:
            break
        lineups.append(lu)
        remaining = remaining[~remaining["card_slug"].isin(lu["card_slug"])]
    return lineups, reason
