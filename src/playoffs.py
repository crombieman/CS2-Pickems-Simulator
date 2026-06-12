"""Playoff (top-8 single-elimination) bracket model: exact enumeration.

Rules (verified 2026-06-12; final check = in-client pick'em UI when picks
open):
  - Seeds 1-8 = final Stage 3 seeds: W-L record, then Difficulty Score
    (Buchholz), then initial stage seed (Valve supplemental rulebook).
  - QFs: 1v8 + 4v5 in the top half, 2v7 + 3v6 in the bottom half; SF
    winners cross in the grand final.
  - QF/SF are BO3; grand final is BO5 per Liquipedia + the official event
    site (the rulebook's generic text says all-BO3 — GRAND_FINAL_BO5 covers
    either resolution).

The bracket has exactly 128 outcome branches (2^7), each with probability
a product of 7 series probs — so everything downstream is computed
EXACTLY, no Monte Carlo noise at the decision layer.

Playoff matches do NOT inherit Stage 3 pair overrides (pre-stage lines are
stale by playoffs). Ratings keep the lam-propagated anchor signal; exact
market probs come only from data/playoff_anchors.json, used verbatim for
the priced series (no BO5 re-conversion: the market prices the actual
format).

Usage:
    python src/playoffs.py
Bracket source: data/playoff_bracket.json {"seeds": [8 teams, seed order]}
if present (the real announced bracket always beats derivation), else
derived from data/live_state.json once Stage 3 is complete.
"""

import collections
import itertools
import json
from pathlib import Path

from model import STAGE3_TEAMS, win_prob
from simulate import SEED, make_state

DATA = Path(__file__).resolve().parent.parent / "data"

GRAND_FINAL_BO5 = True

# A finished 16-team Swiss always yields exactly this record multiset.
RECORD_MULTISET = {(3, 0): 2, (3, 1): 3, (3, 2): 3,
                   (2, 3): 3, (1, 3): 3, (0, 3): 2}


def stage3_final_state(completed):
    """Final records + Buchholz from the full Stage 3 results list.
    Validates via make_state (teams, dupes, impossible records) and then
    requires a complete stage. Returns (records, buchholz)."""
    state = make_state(completed)
    wins, losses = state["wins"], state["losses"]
    records = {t: (wins[t], losses[t]) for t in STAGE3_TEAMS}
    counts = collections.Counter(records.values())
    if counts != RECORD_MULTISET:
        raise ValueError(
            f"Stage 3 not complete: record multiset {dict(counts)} != "
            f"{RECORD_MULTISET} — playoff seeding needs all results")
    buchholz = {t: sum(wins[o] - losses[o] for o in state["opponents"][t])
                for t in STAGE3_TEAMS}
    return records, buchholz


def playoff_seeds(records, buchholz):
    """Teams in playoff seed order 1-8: fewest losses (all have 3 wins),
    then Buchholz desc, then initial stage seed asc — the rulebook's
    final-seed ordering."""
    qualified = [t for t, (w, _) in records.items() if w == 3]
    if len(qualified) != 8:
        raise ValueError(f"expected 8 qualified teams, got {len(qualified)}")
    return sorted(qualified,
                  key=lambda t: (records[t][1], -buchholz[t], SEED[t]))


def quarterfinals(seeds):
    """Rulebook bracket: top half 1v8 + 4v5, bottom half 2v7 + 3v6.
    SF1 = winners of the first two, SF2 = winners of the last two."""
    s = seeds
    return [(s[0], s[7]), (s[3], s[4]), (s[1], s[6]), (s[2], s[5])]


def map_prob(p3, tol=1e-15):
    """Invert p3 = q^2(3-2q): the single-map win prob implied by a
    BO3-series prob (ratings are calibrated to BO3 outcomes)."""
    p3 = min(max(p3, 1e-12), 1.0 - 1e-12)
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        q = (lo + hi) / 2.0
        if q * q * (3.0 - 2.0 * q) < p3:
            lo = q
        else:
            hi = q
    return (lo + hi) / 2.0


def series_prob_bo5(p3):
    """BO3-series prob -> BO5-series prob via the implied map prob.
    P(first to 3) = sum_k C(k+2, k) q^3 (1-q)^k for k opponent maps."""
    q = map_prob(p3)
    return (q ** 3) * (1.0 + 3.0 * (1.0 - q) + 6.0 * (1.0 - q) ** 2)


def bracket_distribution(seeds, prob, bo5_final=GRAND_FINAL_BO5):
    """All 128 outcome branches as (qf_winners, sf_winners, champion, p).

    prob(a, b, bo5=False) -> P(a beats b) in that series format.
    qf_winners ordered as quarterfinals(); sf_winners = (SF1, SF2)."""
    qfs = quarterfinals(seeds)
    branches = []
    for qf_w in itertools.product(*qfs):
        p_qf = 1.0
        for (a, b), w in zip(qfs, qf_w):
            pa = prob(a, b)
            p_qf *= pa if w == a else 1.0 - pa
        sf_matches = [(qf_w[0], qf_w[1]), (qf_w[2], qf_w[3])]
        for sf_w in itertools.product(*sf_matches):
            p_sf = p_qf
            for (a, b), w in zip(sf_matches, sf_w):
                pa = prob(a, b)
                p_sf *= pa if w == a else 1.0 - pa
            fa, fb = sf_w
            pf = prob(fa, fb, bo5=bo5_final)
            branches.append((qf_w, sf_w, fa, p_sf * pf))
            branches.append((qf_w, sf_w, fb, p_sf * (1.0 - pf)))
    return branches


def load_playoff_overrides(path=DATA / "playoff_anchors.json"):
    """{(a, b): P(a wins THE PRICED SERIES)} both orientations, or {}.
    Used verbatim — the market prices the actual series format, so no
    BO5 conversion is applied on top."""
    if not Path(path).exists():
        return {}
    overrides = {}
    for anc in json.load(open(path))["anchors"]:
        overrides[(anc["a"], anc["b"])] = anc["p"]
        overrides[(anc["b"], anc["a"])] = 1.0 - anc["p"]
    return overrides


def make_prob_fn(ratings, overrides):
    def prob(a, b, bo5=False):
        p = overrides.get((a, b))
        if p is not None:
            return p
        p = win_prob(ratings, a, b)
        return series_prob_bo5(p) if bo5 else p
    return prob


def load_bracket():
    """Real bracket file if present, else derive from completed Stage 3."""
    path = DATA / "playoff_bracket.json"
    if path.exists():
        seeds = json.load(open(path))["seeds"]
        if (len(seeds) != 8 or len(set(seeds)) != 8
                or any(t not in STAGE3_TEAMS for t in seeds)):
            raise ValueError(f"playoff_bracket.json needs 8 distinct known "
                             f"teams in seed order, got {seeds}")
        return seeds, "playoff_bracket.json (announced)"
    live = json.load(open(DATA / "live_state.json"))
    completed = [tuple(m) for m in live.get("completed", [])]
    records, buchholz = stage3_final_state(completed)
    return playoff_seeds(records, buchholz), "derived from live_state.json"


def reach_probs(branches, seeds):
    reach = {t: {"sf": 0.0, "final": 0.0, "champ": 0.0} for t in seeds}
    for qf_w, sf_w, champ, p in branches:
        for t in qf_w:
            reach[t]["sf"] += p
        for t in sf_w:
            reach[t]["final"] += p
        reach[champ]["champ"] += p
    return reach


def main():
    ratings = json.load(open(DATA / "ratings_fitted.json"))
    seeds, source = load_bracket()
    overrides = load_playoff_overrides()
    if not overrides:
        print("NOTE: no data/playoff_anchors.json — ratings only, "
              "fetch QF lines before locking.\n")
    prob = make_prob_fn(ratings, overrides)
    branches = bracket_distribution(seeds, prob)

    qfs = quarterfinals(seeds)
    print(f"Bracket ({source}):")
    for i, (a, b) in enumerate(qfs, 1):
        tag = " [anchored]" if (a, b) in overrides else ""
        print(f"  QF{i}: {a} vs {b}  (P({a}) = {prob(a, b):.3f}){tag}")
    print(f"  Grand final: BO5 = {GRAND_FINAL_BO5}\n")

    reach = reach_probs(branches, seeds)
    print(f"{'Team':12s} {'P(win QF)':>9s} {'P(final)':>9s} {'P(champ)':>9s}")
    for t in sorted(seeds, key=lambda t: -reach[t]["champ"]):
        r = reach[t]
        print(f"{t:12s} {r['sf']:9.3f} {r['final']:9.3f} {r['champ']:9.3f}")


if __name__ == "__main__":
    main()
