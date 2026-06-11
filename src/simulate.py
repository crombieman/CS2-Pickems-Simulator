"""Monte Carlo simulator for the 16-team Valve Swiss stage (Stage 3 format).

Round 1 pairings are fixed (the real announced matchups). Rounds 2-5 are
simulated endogenously: teams grouped by record, Buchholz recomputed each
round (sum of opponents' W-L differential), pairing within each group by
Buchholz desc then initial seed asc, highest vs lowest, with rematch
avoidance via backtracking. All matches treated as BO3 single draws at the
model probability.

Known approximations (second-order for advance/3-0/0-3 probabilities):
  - Valve's exact pairing algorithm optimizes seeding-difference across the
    whole group; this uses greedy high-vs-low. Same opponent-strength
    distribution, occasionally different specific pairings.
  - Static ratings within the stage (no form updating round to round).
  - Scalar ratings assume transitivity: no map-pool/style matchup effects.
"""

import collections
import random

from model import STAGE3_TEAMS, load_pair_overrides, win_prob

# Anchored matchups play at the exact market prob whenever the pair meets;
# everything else runs off ratings (which carry only the lam-propagated
# share of the market corrections — see model.ANCHOR_LAMBDA).
PAIR_OVERRIDES = load_pair_overrides()


def match_prob(ratings, a: str, b: str) -> float:
    p = PAIR_OVERRIDES.get((a, b))
    return p if p is not None else win_prob(ratings, a, b)


ROUND1 = [
    ("Vitality", "FUT"), ("NAVI", "Spirit"), ("MOUZ", "Legacy"),
    ("Falcons", "G2"), ("MongolZ", "BetBoom"), ("Aurora", "Monte"),
    ("FURIA", "B8"), ("PARIVISION", "9z"),
]

SEED = {t: i for i, t in enumerate(STAGE3_TEAMS)}


def _pair_group(group, played, buchholz):
    """Pair one record group: sort by (Buchholz desc, seed asc), match top
    vs bottom, avoid rematches, backtrack if stuck (allow rematch as last
    resort, mirroring Valve's fallback)."""
    ordered = sorted(group, key=lambda t: (-buchholz[t], SEED[t]))

    def backtrack(remaining):
        if not remaining:
            return []
        first = remaining[0]
        for opp in reversed(remaining[1:]):
            if opp not in played[first]:
                rest = backtrack([t for t in remaining[1:] if t != opp])
                if rest is not None:
                    return [(first, opp)] + rest
        opp = remaining[-1]  # forced rematch
        rest = backtrack([t for t in remaining[1:] if t != opp])
        return ([(first, opp)] + rest) if rest is not None else None

    return backtrack(ordered)


def simulate_stage(ratings, rng: random.Random):
    """One full Swiss stage. Returns {team: (wins, losses)} final records."""
    wins = collections.Counter()
    losses = collections.Counter()
    played = {t: set() for t in STAGE3_TEAMS}
    opponents = {t: [] for t in STAGE3_TEAMS}
    final = {}
    matches = ROUND1
    while matches:
        for a, b in matches:
            w, l = (a, b) if rng.random() < match_prob(ratings, a, b) else (b, a)
            wins[w] += 1
            losses[l] += 1
            played[a].add(b)
            played[b].add(a)
            opponents[a].append(b)
            opponents[b].append(a)
        for t in STAGE3_TEAMS:
            if t not in final and (wins[t] == 3 or losses[t] == 3):
                final[t] = (wins[t], losses[t])
        buchholz = {t: sum(wins[o] - losses[o] for o in opponents[t])
                    for t in STAGE3_TEAMS}
        groups = collections.defaultdict(list)
        for t in STAGE3_TEAMS:
            if t not in final:
                groups[(wins[t], losses[t])].append(t)
        matches = []
        for grp in groups.values():
            matches += _pair_group(grp, played, buchholz)
    return final


def run(ratings, n_sims=40000, seed=11):
    """Run n simulations. Returns (list of per-sim results, per-team stats).

    stats[team] = dict with p30, padv (exactly 3-1/3-2), pany, p03.
    """
    rng = random.Random(seed)
    sims = []
    records = {t: collections.Counter() for t in STAGE3_TEAMS}
    for _ in range(n_sims):
        result = simulate_stage(ratings, rng)
        sims.append(result)
        for t, rec in result.items():
            records[t][rec] += 1
    stats = {}
    for t in STAGE3_TEAMS:
        n = n_sims
        p30 = records[t][(3, 0)] / n
        padv = (records[t][(3, 1)] + records[t][(3, 2)]) / n
        stats[t] = {"p30": p30, "padv": padv, "pany": p30 + padv,
                    "p03": records[t][(0, 3)] / n}
    return sims, stats
