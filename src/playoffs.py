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

from event_config import COLOGNE
from model import STAGE3_TEAMS, win_prob
from simulate import SEED, make_state

DATA = Path(__file__).resolve().parent.parent / "data"

# Playoff format from the event config (W5): Cologne grand final is BO5.
GRAND_FINAL_BO5 = COLOGNE.playoffs["grand_final_bo5"]

K_DRAWS = 200
K_ENVELOPE = 100
DRAW_SEED = 11
OBJECTIVE = "challenges"   # set per Will's coin status at lock time:
                           # challenges | champion | expected_correct | perfect
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)

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


def all_picks(seeds):
    """All 128 consistent pick brackets: (qf_winners, sf_winners, champion).
    SF picks must come from your QF winners, champion from your SF picks —
    the in-client bracket enforces the same consistency."""
    qfs = quarterfinals(seeds)
    picks = []
    for qf_w in itertools.product(*qfs):
        for sf_w in itertools.product((qf_w[0], qf_w[1]),
                                      (qf_w[2], qf_w[3])):
            for champ in sf_w:
                picks.append((qf_w, sf_w, champ))
    return picks


def score_pick(branches, pick):
    """All objectives for one pick, exactly, over the outcome branches.

    challenges: P(>=2 QF correct AND >=1 SF correct AND champion correct)
                — the coin-challenge joint event.
    A QF/SF pick is correct iff that team wins that match; each team has a
    fixed bracket path, so set intersection counts per-match agreement."""
    pq, ps, pc = set(pick[0]), set(pick[1]), pick[2]
    p_chal = p_champ = p_perfect = e_correct = 0.0
    for qf_w, sf_w, champ, p in branches:
        qf_c = len(pq.intersection(qf_w))
        sf_c = len(ps.intersection(sf_w))
        ch = champ == pc
        e_correct += p * (qf_c + sf_c + ch)
        if ch:
            p_champ += p
            if qf_c >= 2 and sf_c >= 1:
                p_chal += p
            if qf_c == 4 and sf_c == 2:
                p_perfect += p
    return {"challenges": p_chal, "champion": p_champ,
            "expected_correct": e_correct, "perfect": p_perfect}


def optimize_picks(seeds, overrides, draws, objective="challenges",
                   bo5_final=GRAND_FINAL_BO5):
    """Rank all picks by posterior-mean objective across rating draws.

    Each draw is scored EXACTLY (128 branches), so across-draw spread is
    pure parameter uncertainty — no MC noise. Returns a sorted list of
    {"pick", "means" (all objectives), "draw_values" (ranking objective
    per draw, paired across picks for margin SEs)}."""
    picks = all_picks(seeds)
    sums = [collections.defaultdict(float) for _ in picks]
    vals = [[] for _ in picks]
    for ratings in draws:
        prob = make_prob_fn(ratings, overrides)
        branches = bracket_distribution(seeds, prob, bo5_final)
        for i, pick in enumerate(picks):
            s = score_pick(branches, pick)
            for key, v in s.items():
                sums[i][key] += v
            vals[i].append(s[objective])
    n = len(draws)
    results = [{"pick": picks[i],
                "means": {key: v / n for key, v in sums[i].items()},
                "draw_values": vals[i]}
               for i in range(len(picks))]
    # Exact ties are structural, not numerical: "challenges" never sees the
    # SF pick on the non-champion side (champion correct already implies
    # >=1 SF correct), so every pick has a twin differing only there.
    # Tie-break by the secondary objectives so that slot is still chosen
    # to maximize what it CAN still win.
    secondary = [k for k in ("expected_correct", "champion", "perfect")
                 if k != objective]
    results.sort(key=lambda r: tuple(-r["means"][k]
                                     for k in [objective] + secondary))
    return results


def paired_margin(top_vals, runner_vals):
    """(mean diff, SE of mean diff) for the top pick vs runner-up, paired
    per draw — the honest 'how decided is this' number."""
    n = len(top_vals)
    diffs = [a - b for a, b in zip(top_vals, runner_vals)]
    mean = sum(diffs) / n
    if n < 2:
        return mean, 0.0
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    return mean, (var / n) ** 0.5


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

    # Posterior-predictive pick optimization: rank picks by E[objective]
    # over Laplace rating draws, not the MAP point — knife-edge picks
    # (the 0.003-0.007 margins) are exactly where this differs.
    from posterior import laplace_factor, rating_draws
    print(f"\nDrawing {K_DRAWS} rating vectors from the Laplace posterior...")
    factor = laplace_factor()
    draws = rating_draws(k=K_DRAWS, seed=DRAW_SEED, factor=factor)
    results = optimize_picks(seeds, overrides, draws, OBJECTIVE)

    map_branches = branches  # MAP = the fitted-ratings table above
    print(f"\nTop picks by posterior-mean P({OBJECTIVE}) "
          f"({K_DRAWS} draws, exact per draw):")
    print(f"{'#':>2s} {'champion':12s} {'finalists':24s} "
          f"{'E[P]':>7s} {'5-95%':>15s} {'MAP':>7s}")
    for rank, r in enumerate(results[:5], 1):
        qf_w, sf_w, champ = r["pick"]
        xs = sorted(r["draw_values"])
        lo = xs[min(int(0.05 * len(xs)), len(xs) - 1)]
        hi = xs[min(int(0.95 * len(xs)), len(xs) - 1)]
        map_v = score_pick(map_branches, r["pick"])[OBJECTIVE]
        print(f"{rank:2d} {champ:12s} {' + '.join(sf_w):24s} "
              f"{r['means'][OBJECTIVE]:7.3f} [{lo:.3f}-{hi:.3f}] {map_v:7.3f}")
        if rank == 1:
            print(f"   QF picks: {', '.join(qf_w)}")

    # Margin vs the best pick with a DIFFERENT primary value — exact ties
    # (the free non-champion-side SF slot) are resolved by tie-break, not
    # by evidence, and would report a meaningless 0.0 margin.
    top_mean = results[0]["means"][OBJECTIVE]
    j = next((i for i in range(1, len(results))
              if abs(results[i]["means"][OBJECTIVE] - top_mean) > 1e-12),
             1)
    if j > 1:
        print(f"\n({j} picks tied on P({OBJECTIVE}) -- the non-champion-side "
              f"SF slot is free under this objective; tie-broken by "
              f"E[correct], then P(champion).)")
    mean_d, se_d = paired_margin(results[0]["draw_values"],
                                 results[j]["draw_values"])
    decided = "decided" if mean_d > 2 * se_d else "KNIFE-EDGE"
    print(f"Margin #1 over first non-tied rival (#{j + 1}): "
          f"{mean_d:+.4f} +/- {se_d:.4f} (paired SE) -> {decided}")
    m = results[0]["means"]
    print(f"Top pick, all objectives: P(challenges)={m['challenges']:.3f}  "
          f"P(champion)={m['champion']:.3f}  E[correct]={m['expected_correct']:.2f}  "
          f"P(perfect)={m['perfect']:.4f}")

    # Structural envelope: does the best pick survive the lambda sweep?
    print(f"\nLambda envelope (top pick per anchor-propagation weight, "
          f"{K_ENVELOPE} draws each):")
    base_pick = results[0]["pick"]
    for lam in LAMBDAS:
        d = rating_draws(k=K_ENVELOPE, seed=DRAW_SEED, lam=lam, factor=factor)
        r = optimize_picks(seeds, overrides, d, OBJECTIVE)[0]
        flag = "" if r["pick"] == base_pick else "  <- PICK CHANGES"
        print(f"  lam={lam:.2f}: champ {r['pick'][2]:12s} "
              f"E[P]={r['means'][OBJECTIVE]:.3f}{flag}")


if __name__ == "__main__":
    main()
