"""Slate sensitivity analysis — the pre-lock ritual tool (roadmap 0.4b).

Usage: python src/sensitivity.py

Answers two questions before any pick'em lock:
  1. Which market anchors actually matter? (perturb each line +/- 3pts,
     see if the best slate changes or rivals overtake) -> these are the
     lines worth hunting better data for.
  2. Are the picks stable under the hand-set data weights? (scale each
     event group's weight x0.8 / x1.2, re-fit, re-score)

Method: run the full optimizer ONCE on the baseline to get the incumbent
slate and its nearest rival slates, then re-score only that shortlist
under each perturbation. A perturbation "flips" if any rival overtakes
the incumbent. Everything runs in memory off the current CSV + anchors —
committed lock artifacts are never touched.
"""

import collections
import itertools
import json
import math
import random
from pathlib import Path

import simulate
from model import (ANCHOR_LAMBDA, PRIORS, STAGE3_TEAMS, fit_bradley_terry,
                   load_matches)
from optimize import score_slate
from simulate import run

DATA = Path(__file__).resolve().parent.parent / "data"

N_SIMS_BASE = 40000   # baseline (incumbent + rival discovery)
N_SIMS_PERT = 15000   # per perturbation (flip detection, not precision)
SEED = 11
ANCHOR_DELTA = 0.03
WEIGHT_SCALES = (0.8, 1.2)
N_RIVALS = 12


def load_anchors():
    return json.load(open(DATA / "market_anchors.json"))["anchors"]


def build_ratings(matches, anchors, lam=ANCHOR_LAMBDA):
    fitted = fit_bradley_terry(matches)
    r = dict(fitted)
    for a in anchors:
        gap = math.log10(a["p"] / (1 - a["p"])) * 400.0
        d = lam * (gap - (r[a["a"]] - r[a["b"]])) / 2.0
        r[a["a"]] += d
        r[a["b"]] -= d
    return r


def set_overrides(anchors):
    simulate.PAIR_OVERRIDES.clear()
    for a in anchors:
        simulate.PAIR_OVERRIDES[(a["a"], a["b"])] = a["p"]
        simulate.PAIR_OVERRIDES[(a["b"], a["a"])] = 1.0 - a["p"]


def top_slates(sims, stats, n=N_RIVALS, k30=5, k03=5, kadv=9):
    """Same search space as optimize.optimize(), keeping the top n slates."""
    cand30 = sorted(STAGE3_TEAMS, key=lambda t: -stats[t]["p30"])[:k30]
    cand03 = sorted(STAGE3_TEAMS, key=lambda t: -stats[t]["p03"])[:k03]
    candadv = sorted(STAGE3_TEAMS, key=lambda t: -stats[t]["padv"])[:kadv]
    scored = []
    for c30 in itertools.combinations(cand30, 2):
        for c03 in itertools.combinations(cand03, 2):
            if set(c30) & set(c03):
                continue
            pool = [t for t in candadv if t not in c30 and t not in c03]
            if len(pool) < 6:
                continue
            for cadv in itertools.combinations(pool, 6):
                p5, ev = score_slate(sims, c30, c03, cadv)
                scored.append((p5, ev, c30, c03, cadv))
    scored.sort(reverse=True)
    return scored[:n]


def slate_key(s):
    return (frozenset(s[2]), frozenset(s[3]), frozenset(s[4]))


def rescore(ratings, slates, n_sims=N_SIMS_PERT, seed=SEED):
    sims, _ = run(ratings, n_sims=n_sims, seed=seed)
    return [(score_slate(sims, c30, c03, cadv)[0], c30, c03, cadv)
            for _, _, c30, c03, cadv in slates]


def fmt(c30, c03, cadv):
    return f"3-0:{'/'.join(c30)} 0-3:{'/'.join(c03)} adv:{'/'.join(sorted(cadv))}"


def main():
    matches = load_matches()
    anchors = load_anchors()
    set_overrides(anchors)
    base_ratings = build_ratings(matches, anchors)

    print(f"Baseline: full optimize, {N_SIMS_BASE} sims...")
    sims, stats = run(base_ratings, n_sims=N_SIMS_BASE, seed=SEED)
    shortlist = top_slates(sims, stats)
    inc_p5, inc_ev, *inc = shortlist[0]
    inc_key = slate_key(shortlist[0])
    print(f"Incumbent (p5={inc_p5:.3f}): {fmt(*inc)}")
    print(f"Shortlist: {len(shortlist)} slates, p5 range "
          f"{shortlist[-1][0]:.3f}-{inc_p5:.3f}\n")

    flips = []

    print(f"-- Anchor sensitivity (+/-{ANCHOR_DELTA} on each line, "
          f"{N_SIMS_PERT} sims each) --")
    print(f"{'Anchor':24s} {'dir':>5s} {'best slate':>42s} {'Dp5':>7s}")
    for i, anc in enumerate(anchors):
        for sign in (+1, -1):
            pert = [dict(a) for a in anchors]
            pert[i]["p"] = min(0.99, max(0.01, anc["p"] + sign * ANCHOR_DELTA))
            set_overrides(pert)
            r = build_ratings(matches, pert)
            res = sorted(rescore(r, shortlist), reverse=True)
            best = res[0]
            flipped = slate_key((None, None, *best[1:])) != inc_key
            label = fmt(*best[1:]) if flipped else "(incumbent holds)"
            inc_here = next(x[0] for x in res
                            if slate_key((None, None, *x[1:])) == inc_key)
            print(f"{anc['a']+'-'+anc['b']:24s} {('+' if sign>0 else '-')+f'{ANCHOR_DELTA}':>5s} "
                  f"{label:>42s} {best[0]-inc_here:+7.3f}")
            if flipped:
                flips.append((f"anchor {anc['a']}-{anc['b']} "
                              f"{'+' if sign > 0 else '-'}{ANCHOR_DELTA}", label))

    print(f"\n-- Weight sensitivity (event groups x{WEIGHT_SCALES}, "
          f"re-fit each) --")
    set_overrides(anchors)
    groups = sorted({m[3] if len(m) > 3 else "?" for m in
                     [ln.split(",") for ln in
                      open(DATA / "matches_2026.csv").read().splitlines()[1:]]})
    raw = [ln.split(",") for ln in
           open(DATA / "matches_2026.csv").read().splitlines()[1:]]
    for grp in groups:
        for scale in WEIGHT_SCALES:
            pert_matches = [(w, l, float(wt) * (scale if ev == grp else 1.0))
                            for w, l, wt, ev, *_ in raw]
            r = build_ratings(pert_matches, anchors)
            res = sorted(rescore(r, shortlist), reverse=True)
            best = res[0]
            flipped = slate_key((None, None, *best[1:])) != inc_key
            if flipped:
                flips.append((f"weights[{grp}] x{scale}", fmt(*best[1:])))
                print(f"{grp:24s} x{scale}: FLIP -> {fmt(*best[1:])}")
    print("(unlisted weight perturbations: incumbent holds)")

    print(f"\n== Verdict ==")
    if not flips:
        print("Slate is stable under every tested perturbation. Lock it.")
    else:
        print(f"{len(flips)} perturbation(s) flip the slate — these are the "
              "knife edges; get better data on them before locking:")
        for cause, slate in flips:
            print(f"  {cause}: {slate}")


if __name__ == "__main__":
    main()
