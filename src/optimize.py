"""Pick'em slate optimizer + CLI entry point.

Valve Stage 3 pick'em scoring:
  - 2 picks for exactly 3-0 (a 3-1 finish scores zero for a 3-0 pick)
  - 2 picks for exactly 0-3
  - 6 picks for "advance", which counts ONLY for 3-1 or 3-2 finishes
  - 5 of 10 correct passes the stage

Objective: maximize P(>= 5 correct), evaluated empirically on the stored
simulation outcomes so all Swiss correlations are respected (e.g. two R1
opponents can't both go 3-0; the loser of a top-team R1 clash is forced
onto the 3-1/3-2 path, which makes opposing-advance-slot pairs valuable).

Search: exhaustive over the top-k candidates per slot category. Slate-score
surfaces are flat near the optimum, so k=5/5/9 is more than sufficient.

Usage:
    python src/fit.py        # fit ratings -> data/ratings_fitted.json
    python src/optimize.py   # simulate + optimize -> printed slate
"""

import datetime
import itertools
import json
from pathlib import Path

from model import STAGE3_TEAMS
from simulate import run

DATA = Path(__file__).resolve().parent.parent / "data"


def score_slate(sims, picks_30, picks_03, picks_adv):
    """Return (P(>=5 correct), E[correct]) for a slate over stored sims."""
    hits = total = 0
    for result in sims:
        k = (sum(1 for t in picks_30 if result[t] == (3, 0))
             + sum(1 for t in picks_03 if result[t] == (0, 3))
             + sum(1 for t in picks_adv if result[t] in ((3, 1), (3, 2))))
        hits += k >= 5
        total += k
    n = len(sims)
    return hits / n, total / n


def optimize(sims, stats, k30=5, k03=5, kadv=9):
    cand30 = sorted(STAGE3_TEAMS, key=lambda t: -stats[t]["p30"])[:k30]
    cand03 = sorted(STAGE3_TEAMS, key=lambda t: -stats[t]["p03"])[:k03]
    candadv = sorted(STAGE3_TEAMS, key=lambda t: -stats[t]["padv"])[:kadv]
    best = None
    for c30 in itertools.combinations(cand30, 2):
        for c03 in itertools.combinations(cand03, 2):
            if set(c30) & set(c03):
                continue
            pool = [t for t in candadv if t not in c30 and t not in c03]
            if len(pool) < 6:
                continue
            for cadv in itertools.combinations(pool, 6):
                p5, ev = score_slate(sims, c30, c03, cadv)
                if best is None or (p5, ev) > (best[0], best[1]):
                    best = (p5, ev, c30, c03, cadv)
    return best


N_SIMS = 40000
SIM_SEED = 11


def main():
    ratings = json.load(open(DATA / "ratings_fitted.json"))
    sims, stats = run(ratings, n_sims=N_SIMS, seed=SIM_SEED)

    # Persist the probability table — this is what the postmortem grades.
    out = {
        "meta": {"n_sims": N_SIMS, "seed": SIM_SEED,
                 "generated": datetime.date.today().isoformat()},
        "probs": stats,
    }
    json.dump(out, open(DATA / "stage3_probs.json", "w"), indent=2)

    print(f"{'Team':12s} {'P(3-0)':>7s} {'P(3-1/3-2)':>11s} {'P(advance)':>11s} {'P(0-3)':>7s}")
    for t in sorted(STAGE3_TEAMS, key=lambda t: -stats[t]["pany"]):
        s = stats[t]
        print(f"{t:12s} {s['p30']:7.3f} {s['padv']:11.3f} {s['pany']:11.3f} {s['p03']:7.3f}")

    p5, ev, c30, c03, cadv = optimize(sims, stats)
    print(f"\nOptimal slate  (P(>=5 correct) = {p5:.3f}, E[correct] = {ev:.2f})")
    print(f"  3-0:     {', '.join(c30)}")
    print(f"  0-3:     {', '.join(c03)}")
    print(f"  Advance: {', '.join(cadv)}")


if __name__ == "__main__":
    main()
