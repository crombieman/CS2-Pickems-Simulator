"""Rating model: regularized Bradley-Terry fit on match results + market anchoring.

Pipeline (run via fit.py):
  1. Start from prior ratings (market-informed estimates on an Elo-like scale).
  2. MAP-fit a Bradley-Terry model on weighted match results, with a Gaussian
     prior pulling each team toward its prior rating (prevents sparse-sample
     blowups, e.g. a 3-0 Swiss run rocketing a team past its true level).
  3. Hard re-anchor pairs of teams to vig-free market-implied probabilities
     where liquid lines exist (symmetric rating shift per pair).

Win prob between teams a, b:  P(a beats b) = 1 / (1 + 10^(-(R_a - R_b)/400))
Ratings are calibrated to BO3 series outcomes (BO1s are downweighted in data).
"""

import csv
import json
import math
from pathlib import Path

from event_config import COLOGNE

DATA = Path(__file__).resolve().parent.parent / "data"

# Event facts now live in data/events/<event>.json (W5). The Cologne config
# holds the exact team order the locked fit + tables were generated under, so
# this is byte-identical to the former in-code literal. Seeds 1-8 are the
# Valve Global Standings invitees; 9-16 are seeded by Stage 2 Swiss + Buchholz.
STAGE3_TEAMS = COLOGNE.teams

# Prior means. Stage 3 teams: market-informed estimates (two GGbet BO3 lines +
# outright odds + VRS position) as of 2026-06-09. Connector teams (eliminated
# in Stages 1-2 but present in the match graph): weak tier-level priors.
PRIORS = {
    "Vitality": 1180, "Spirit": 1075, "NAVI": 1065, "Falcons": 1040,
    "FURIA": 995, "MOUZ": 990, "MongolZ": 985, "Aurora": 950,
    "PARIVISION": 945, "FUT": 935, "G2": 920, "BetBoom": 880,
    "Monte": 875, "Legacy": 870, "B8": 845, "9z": 790,
    # connectors
    "Astralis": 900, "GamerLegion": 900, "paiN": 860, "TYLOO": 860,
    "BIG": 860, "Liquid": 850, "MIBR": 840, "3DMAX": 840, "M80": 820,
    "FlyQuest": 820, "HOTU": 820, "RedCanids": 780, "GentleMates": 780,
    "PassionUA": 760,
}

LOG10_OVER_400 = math.log(10) / 400


def win_prob(ratings: dict, a: str, b: str) -> float:
    return 1.0 / (1.0 + 10 ** (-(ratings[a] - ratings[b]) / 400.0))


def load_matches(path: Path = DATA / "matches_2026.csv"):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [(r["winner"], r["loser"], float(r["weight"])) for r in rows]


def fit_bradley_terry(matches, priors=PRIORS, sigma_s3=70.0, sigma_other=50.0,
                      iters=4000, lr=2000.0):
    """MAP estimate: weighted BT log-likelihood + Gaussian prior per team.

    sigma controls how far data can move a team off its prior. 70 Elo for
    Stage 3 teams lets ~60 series move ratings meaningfully; 50 for connector
    teams keeps thin-sample teams pinned near tier priors.
    """
    ratings = dict(priors)
    sigma = {t: (sigma_s3 if t in STAGE3_TEAMS else sigma_other) for t in priors}
    for _ in range(iters):
        grad = {t: 0.0 for t in priors}
        for w, l, wt in matches:
            p = win_prob(ratings, w, l)
            g = wt * (1.0 - p) * LOG10_OVER_400
            grad[w] += g
            grad[l] -= g
        for t in priors:
            grad[t] -= (ratings[t] - priors[t]) / sigma[t] ** 2
        for t in priors:
            ratings[t] += lr * grad[t]
    # re-center Stage 3 mean to prior mean (BT is translation-invariant)
    shift = (sum(priors[t] for t in STAGE3_TEAMS)
             - sum(ratings[t] for t in STAGE3_TEAMS)) / len(STAGE3_TEAMS)
    return {t: r + shift for t, r in ratings.items()}


# Fraction of the market-vs-fit correction that propagates into ratings used
# for NON-anchored matchups (rounds 2-5 cross-pairings). The anchored pair
# itself always plays at the exact market prob via pair overrides in the
# simulator. 1.0 = old behavior (a line moves a team against everyone);
# 0.0 = lines only affect their own match. 0.5 because a line's deviation
# from the fit mixes global info (roster news, form) with matchup-specific
# info (H2H style, e.g. Spirit-NAVI) and a single number can't be decomposed
# per-pair without player-level modeling. Slate sensitivity to this knob is
# one advance slot (MongolZ vs G2 at lam ~0.6); everything else is stable
# across the full [0, 1] range.
ANCHOR_LAMBDA = 0.5


def apply_market_anchors(ratings: dict, anchors_path: Path = DATA / "market_anchors.json",
                         lam: float = ANCHOR_LAMBDA):
    """Shift ratings toward vig-free market lines via partial symmetric shifts.

    Markets aggregate information the historical fit can't see (roster news,
    prep state). The anchored matchup itself is played at the exact market
    prob (see load_pair_overrides / simulate.match_prob); only lam of the
    correction propagates to each team's OTHER matchups.
    """
    ratings = dict(ratings)
    for anc in json.load(open(anchors_path))["anchors"]:
        a, b, p = anc["a"], anc["b"], anc["p"]
        needed_gap = math.log10(p / (1 - p)) * 400.0
        delta = lam * (needed_gap - (ratings[a] - ratings[b])) / 2.0
        ratings[a] += delta
        ratings[b] -= delta
    return ratings


def load_pair_overrides(anchors_path: Path = DATA / "market_anchors.json") -> dict:
    """{(a, b): P(a beats b)} for every anchored pair, both orientations.
    Where the market priced a specific match, that prob is used verbatim
    whenever the two teams meet (R1 always; later rematches are rare under
    Swiss rematch avoidance)."""
    overrides = {}
    for anc in json.load(open(anchors_path))["anchors"]:
        overrides[(anc["a"], anc["b"])] = anc["p"]
        overrides[(anc["b"], anc["a"])] = 1.0 - anc["p"]
    return overrides


def devig(odds_a: float, odds_b: float) -> float:
    """Two-way decimal odds -> vig-free P(a) by proportional normalization."""
    ia, ib = 1.0 / odds_a, 1.0 / odds_b
    return ia / (ia + ib)
