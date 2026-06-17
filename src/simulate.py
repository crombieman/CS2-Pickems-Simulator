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

from event_config import COLOGNE
from model import STAGE3_TEAMS, load_pair_overrides, win_prob

# Anchored matchups play at the exact market prob whenever the pair meets;
# everything else runs off ratings (which carry only the lam-propagated
# share of the market corrections — see model.ANCHOR_LAMBDA).
PAIR_OVERRIDES = load_pair_overrides()


def match_prob(ratings, a: str, b: str) -> float:
    p = PAIR_OVERRIDES.get((a, b))
    return p if p is not None else win_prob(ratings, a, b)


# R1 pairings and initial seeds now come from the event config (W5) — the
# Cologne values are byte-identical to the former in-code literals. The seeds
# were derived 2026-06-12 by inverting the R1 bracket (1v9..8v16) against the
# post-R1 standings order, validated by reproducing the announced R2 pairings
# 8/8 under the rulebook's highest-vs-lowest rule. (The original guess —
# STAGE3_TEAMS order — got R2 pairings 1/8: MOUZ/Spirit/9z were misplaced.)
ROUND1 = COLOGNE.round1
SEED = COLOGNE.seeds

# Pre-fix seed order (list position in STAGE3_TEAMS). The locked v1-v3
# probability tables were generated under this mapping; the regression
# test pins it so those frozen artifacts stay reproducible.
LEGACY_SEED = {t: i for i, t in enumerate(STAGE3_TEAMS)}


# Valve Major Supplemental Rulebook: 6-team record groups (R4 2-1/1-2,
# R5 2-2... when applicable) pick the TOP-MOST priority row that avoids a
# rematch. Positions are 1-indexed into the group sorted by current seed.
PRIORITY_TABLE_6 = [
    ((1, 6), (2, 5), (3, 4)), ((1, 6), (2, 4), (3, 5)),
    ((1, 5), (2, 6), (3, 4)), ((1, 5), (2, 4), (3, 6)),
    ((1, 4), (2, 6), (3, 5)), ((1, 4), (2, 5), (3, 6)),
    ((1, 6), (2, 3), (4, 5)), ((1, 5), (2, 3), (4, 6)),
    ((1, 3), (2, 6), (4, 5)), ((1, 3), (2, 5), (4, 6)),
    ((1, 4), (2, 3), (5, 6)), ((1, 3), (2, 4), (5, 6)),
    ((1, 2), (3, 6), (4, 5)), ((1, 2), (3, 5), (4, 6)),
    ((1, 2), (3, 4), (5, 6)),
]

# Pin to False to reproduce sims generated before the table existed
# (the locked v1-v3 tables used greedy pairing for all group sizes).
USE_PRIORITY_TABLE = True


def _pair_group(group, played, buchholz):
    """Pair one record group, sorted by (Buchholz desc, seed asc).

    6-team groups use Valve's explicit 15-row priority table (first row
    with no rematch). Other sizes use highest-vs-lowest with rematch
    avoidance via backtracking (the rulebook's R2/R3 rule; allow rematch
    as last resort, mirroring Valve's fallback)."""
    ordered = sorted(group, key=lambda t: (-buchholz[t], SEED[t]))

    if USE_PRIORITY_TABLE and len(ordered) == 6:
        for row in PRIORITY_TABLE_6:
            if all(ordered[j - 1] not in played[ordered[i - 1]]
                   for i, j in row):
                return [(ordered[i - 1], ordered[j - 1]) for i, j in row]
        # all 15 rows blocked (needs 5+ rematch constraints) — fall through

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


def make_state(completed, upcoming=()):
    """Build a resume state from real results.

    completed: iterable of (winner, loser) for every finished match.
    upcoming: iterable of (a, b) scheduled-but-unplayed pairings (use this
      whenever the next round's real pairings are announced, or mid-round —
      Valve's exact seeding-difference pairing can differ from our greedy
      approximation, so real pairings always beat simulated ones).
    """
    # live_state.json is hand-edited — validate at the boundary, fail loud
    # and friendly rather than producing silently-wrong probabilities.
    known = set(STAGE3_TEAMS)
    for src_name, matches_in in (("completed", completed), ("upcoming", upcoming)):
        for m in matches_in:
            if len(tuple(m)) != 2:
                raise ValueError(f"{src_name}: each entry needs exactly 2 teams, got {m!r}")
            for t in m:
                if t not in known:
                    raise ValueError(
                        f"{src_name}: unknown team {t!r} — check spelling against "
                        f"model.STAGE3_TEAMS (e.g. 'MongolZ', 'NAVI', '9z')")
    pair_seen = collections.Counter(frozenset(m) for m in completed)
    pair_seen.update(frozenset(m) for m in upcoming)
    dupes = [set(p) for p, n in pair_seen.items() if n > 1]
    if dupes:
        raise ValueError(f"same pairing listed twice (rematches don't happen "
                         f"mid-Swiss): {dupes}")

    wins = collections.Counter()
    losses = collections.Counter()
    played = {t: set() for t in STAGE3_TEAMS}
    opponents = {t: [] for t in STAGE3_TEAMS}
    for w, l in completed:
        wins[w] += 1
        losses[l] += 1
        played[w].add(l)
        played[l].add(w)
        opponents[w].append(l)
        opponents[l].append(w)
    for t in STAGE3_TEAMS:
        if wins[t] > 3 or losses[t] > 3:
            raise ValueError(f"{t} has {wins[t]}-{losses[t]} — impossible in a "
                             f"best-of-3-wins Swiss; check 'completed' for errors "
                             f"(winner goes FIRST in each pair)")
    if not upcoming:
        # Without forced pairings we can only pair by record group, which
        # is wrong mid-round (a 1-0 team awaiting its R2 game would be
        # grouped against 1-1 teams). Require a round boundary.
        active = [t for t in STAGE3_TEAMS if wins[t] < 3 and losses[t] < 3]
        games = {wins[t] + losses[t] for t in active}
        assert len(games) <= 1, (
            "mid-round state: supply the remaining scheduled pairings "
            f"via 'upcoming' (games played varies: {sorted(games)})")
    return {"wins": wins, "losses": losses, "played": played,
            "opponents": opponents, "matches": [tuple(m) for m in upcoming]}


def simulate_stage(ratings, rng: random.Random, state=None):
    """One full Swiss stage (optionally resumed from a real mid-stage
    state — see make_state). Returns {team: (wins, losses)} final records."""
    if state is None:
        wins = collections.Counter()
        losses = collections.Counter()
        played = {t: set() for t in STAGE3_TEAMS}
        opponents = {t: [] for t in STAGE3_TEAMS}
        matches = ROUND1
    else:
        wins = state["wins"].copy()
        losses = state["losses"].copy()
        played = {t: set(s) for t, s in state["played"].items()}
        opponents = {t: list(o) for t, o in state["opponents"].items()}
        matches = list(state["matches"])
    final = {t: (wins[t], losses[t]) for t in STAGE3_TEAMS
             if wins[t] == 3 or losses[t] == 3}
    while True:
        if not matches:
            buchholz = {t: sum(wins[o] - losses[o] for o in opponents[t])
                        for t in STAGE3_TEAMS}
            groups = collections.defaultdict(list)
            for t in STAGE3_TEAMS:
                if t not in final:
                    groups[(wins[t], losses[t])].append(t)
            for grp in groups.values():
                matches += _pair_group(grp, played, buchholz)
            if not matches:
                return final
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
        matches = []


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
