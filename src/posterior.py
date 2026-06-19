"""Posterior sampling over ratings (Laplace approximation). Pure stdlib.

Usage: python src/posterior.py

The pipeline's point estimates condition on fitted ratings being exactly
right. This propagates parameter uncertainty: Laplace-approximate the
Bradley-Terry posterior at the MAP (Hessian of negative log posterior ->
Gaussian), draw rating vectors, push each through the anchor transform,
simulate each, and report honest intervals instead of points.

Epistemic spread reported as the 5th-95th percentile of per-sample
quantities across K rating draws (each estimated from M sims, so the
interval includes ~+/-0.02 of MC noise on top of true parameter
uncertainty — conservative in width, stated plainly).
"""

import json
import math
import random
from pathlib import Path

from model import (ANCHOR_LAMBDA, PRIORS, STAGE3_TEAMS, apply_market_anchors,
                   fit_bradley_terry, load_matches, win_prob)
from simulate import simulate_stage
from live import SLATE_30, SLATE_03, SLATE_ADV, slate_ticks

DATA = Path(__file__).resolve().parent.parent / "data"

K_SAMPLES = 200   # rating vectors drawn from the Laplace posterior
M_SIMS = 2000     # Swiss sims per rating vector
SEED = 11

C = math.log(10) / 400.0


def hessian(matches, ratings, teams, sigma):
    """Hessian of the negative log posterior at the MAP. Positive definite:
    the BT likelihood alone is translation-invariant (singular), but each
    team's Gaussian prior adds 1/sigma^2 on the diagonal."""
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    H = [[0.0] * n for _ in range(n)]
    for w, l, wt in matches:
        p = win_prob(ratings, w, l)
        h = wt * C * C * p * (1.0 - p)
        i, j = idx[w], idx[l]
        H[i][i] += h
        H[j][j] += h
        H[i][j] -= h
        H[j][i] -= h
    for t, i in idx.items():
        H[i][i] += 1.0 / sigma[t] ** 2
    return H


def cholesky(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][i] = math.sqrt(A[i][i] - s)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def sample_offsets(L, rng):
    """y ~ N(0, H^-1) where H = L L^T: solve L^T y = z by back-substitution."""
    n = len(L)
    z = [rng.gauss(0.0, 1.0) for _ in range(n)]
    y = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(L[j][i] * y[j] for j in range(i + 1, n))
        y[i] = (z[i] - s) / L[i][i]
    return y


def pct(sorted_xs, q):
    i = min(int(q * len(sorted_xs)), len(sorted_xs) - 1)
    return sorted_xs[i]


def laplace_factor():
    """(map_ratings, teams, L): MAP fit + Cholesky factor of the posterior
    Hessian — everything needed to draw rating vectors. Factored out so
    other decision layers (playoffs.py) can reuse the draws."""
    matches = load_matches()
    map_ratings = fit_bradley_terry(matches)
    teams = sorted(PRIORS)
    sigma = {t: (70.0 if t in STAGE3_TEAMS else 50.0) for t in teams}
    return map_ratings, teams, cholesky(hessian(matches, map_ratings,
                                                teams, sigma))


def rating_draws(k=K_SAMPLES, seed=SEED, lam=ANCHOR_LAMBDA, factor=None):
    """k anchored rating dicts drawn from the Laplace posterior.
    Pass a precomputed laplace_factor() to amortize the fit (e.g. when
    sweeping lam: same seed -> same offsets, so the comparison is
    controlled)."""
    map_ratings, teams, L = factor if factor is not None else laplace_factor()
    rng = random.Random(seed)
    draws = []
    for _ in range(k):
        off = sample_offsets(L, rng)
        sampled = {t: map_ratings[t] + off[i] for i, t in enumerate(teams)}
        draws.append(apply_market_anchors(sampled, lam=lam))
    return draws


# -- W15 / U1: structural envelope + E.3 published intervals ------------------
# Three uncertainty sources fold into ONE honest interval (roadmap E.3):
#   (1) MC error    -- finite sims (~+/-0.003 at M_SIMS); a floor half-width.
#   (2) parameter   -- Laplace posterior draws (rating_draws, above).
#   (3) structural  -- model-shape choices we can't pin from data: the anchor-
#                      propagation weight lambda and the recency/format weighting.
#                      Swept over a small grid; the spread is the structural range.
# The published interval ENVELOPES (1)-(3) rather than adding variances in
# quadrature: structural spread is not a Gaussian sd, so a union is the honest,
# conservative statement (matches this module's "conservative in width" ethos).

LAMBDAS_STRUCT = (0.3, 0.5, 0.7)   # anchor-propagation sweep (ANCHOR_LAMBDA=0.5)
WEIGHT_SCALES = (0.7, 1.0, 1.3)    # recency/format sharpness sweep (1.0 = as-fit)
MC_FLOOR = 0.003                   # min interval half-width: finite-sim noise


def perturb_weights(matches, scale):
    """Re-weight matches to sweep recency/format sharpness (the structural recency
    axis; recency lives in the CSV weight column, not a runtime knob). scale=1.0
    leaves weights untouched; scale<1 flattens them toward 1.0 (less recency/format
    emphasis), scale>1 sharpens. w' = 1 - scale*(1 - w), clamped positive."""
    return [(w, l, max(1e-6, 1.0 - scale * (1.0 - wt))) for w, l, wt in matches]


def slate_point(anchored, rng, m_sims):
    """Point estimates from m_sims at fixed anchored ratings:
    (p_pass, {team: {"p30","padv","p03"}}). One structural-grid corner."""
    passes = 0
    counts = {t: {"p30": 0, "padv": 0, "p03": 0} for t in STAGE3_TEAMS}
    for _ in range(m_sims):
        result = simulate_stage(anchored, rng)
        passes += slate_ticks(result) >= 5
        for t, rec in result.items():
            if rec == (3, 0):
                counts[t]["p30"] += 1
            elif rec in ((3, 1), (3, 2)):
                counts[t]["padv"] += 1
            elif rec == (0, 3):
                counts[t]["p03"] += 1
    per_team = {t: {c: counts[t][c] / m_sims for c in ("p30", "padv", "p03")}
                for t in STAGE3_TEAMS}
    return passes / m_sims, per_team


def structural_points(seed=SEED, m_sims=M_SIMS, lambdas=LAMBDAS_STRUCT,
                      scales=WEIGHT_SCALES):
    """Point estimate of every slate quantity at each (weight_scale, lambda)
    corner. Re-fits per weight scale (recency moves the MAP), re-anchors per
    lambda (cheap). Returns a list of (p_pass, per_team) over the grid."""
    base_matches = load_matches()
    pts = []
    for scale in scales:
        matches = (base_matches if scale == 1.0
                   else perturb_weights(base_matches, scale))
        map_ratings = fit_bradley_terry(matches)
        for lam in lambdas:
            anchored = apply_market_anchors(dict(map_ratings), lam=lam)
            pts.append(slate_point(anchored, random.Random(seed), m_sims))
    return pts


def published_interval(point, param_lo, param_hi, structural_vals,
                       mc_floor=MC_FLOOR):
    """E.3 interval for one quantity: envelope of the parameter interval
    (param_lo..param_hi, already carrying MC noise), the structural spread
    (corner values), and a minimum MC half-width. Clamped to [0, 1]."""
    lo = min([param_lo, point - mc_floor] + structural_vals)
    hi = max([param_hi, point + mc_floor] + structural_vals)
    return max(0.0, lo), min(1.0, hi)


def fmt_interval(point, lo, hi):
    """E.3 display form, e.g. '0.42 [0.37-0.45]' — two decimals; three would
    overstate what we know by an order of magnitude (roadmap E.3)."""
    return f"{point:.2f} [{lo:.2f}-{hi:.2f}]"


def main():
    map_ratings, teams, L = laplace_factor()
    rng = random.Random(SEED)

    marginal_sd = {}  # quick visibility into posterior width per team
    # diag of H^-1 via solving for unit vectors is O(n^3); estimate from samples instead.

    per_sample_p5 = []
    per_sample_team = {t: {"p30": [], "padv": [], "p03": []} for t in STAGE3_TEAMS}
    pooled_pass = 0
    pooled_ticks = 0
    offsets_seen = {t: [] for t in STAGE3_TEAMS}

    for _ in range(K_SAMPLES):
        off = sample_offsets(L, rng)
        sampled = {t: map_ratings[t] + off[i] for i, t in enumerate(teams)}
        for t in STAGE3_TEAMS:
            offsets_seen[t].append(off[teams.index(t)])
        anchored = apply_market_anchors(sampled, lam=ANCHOR_LAMBDA)
        counts = {t: {"p30": 0, "padv": 0, "p03": 0} for t in STAGE3_TEAMS}
        passes = 0
        for _ in range(M_SIMS):
            result = simulate_stage(anchored, rng)
            k = slate_ticks(result)
            passes += k >= 5
            pooled_ticks += k
            for t, rec in result.items():
                if rec == (3, 0):
                    counts[t]["p30"] += 1
                elif rec in ((3, 1), (3, 2)):
                    counts[t]["padv"] += 1
                elif rec == (0, 3):
                    counts[t]["p03"] += 1
        pooled_pass += passes
        per_sample_p5.append(passes / M_SIMS)
        for t in STAGE3_TEAMS:
            for c in ("p30", "padv", "p03"):
                per_sample_team[t][c].append(counts[t][c] / M_SIMS)

    n_total = K_SAMPLES * M_SIMS
    p5_sorted = sorted(per_sample_p5)
    print(f"Posterior predictive over {K_SAMPLES} rating draws x {M_SIMS} sims "
          f"(lambda={ANCHOR_LAMBDA}, anchored pairs held at market):\n")
    sds = sorted((len(offsets_seen[t]) > 1 and
                  (sum(x * x for x in offsets_seen[t]) / len(offsets_seen[t])) ** 0.5
                  or 0.0) for t in STAGE3_TEAMS)
    print(f"Rating posterior sd (Stage 3 teams): median ~{sds[len(sds)//2]:.0f} Elo "
          f"(range {sds[0]:.0f}-{sds[-1]:.0f})\n")

    print(f"Locked v3 slate:")
    print(f"  P(>=5 correct) = {pooled_pass / n_total:.3f}   "
          f"[{pct(p5_sorted, 0.05):.3f} - {pct(p5_sorted, 0.95):.3f}]  (5th-95th pct"
          f" across rating draws; width includes ~+/-0.02 MC noise)")
    print(f"  E[ticks]       = {pooled_ticks / n_total:.2f}\n")

    print(f"{'Team':12s} {'P(3-0)':>7s} {'5-95%':>13s} {'P(adv)':>7s} {'5-95%':>13s} "
          f"{'P(0-3)':>7s} {'5-95%':>13s}")
    pooled = {t: {c: sum(v) / len(v) for c, v in d.items()}
              for t, d in per_sample_team.items()}
    for t in sorted(STAGE3_TEAMS,
                    key=lambda t: -(pooled[t]["p30"] + pooled[t]["padv"])):
        row = f"{t:12s}"
        for c in ("p30", "padv", "p03"):
            xs = sorted(per_sample_team[t][c])
            row += (f" {pooled[t][c]:7.3f} [{pct(xs, 0.05):.3f}-{pct(xs, 0.95):.3f}]")
        print(row)

    # -- W15 / U1: E.3 published intervals (parameter + structural + MC) -------
    n_corners = len(WEIGHT_SCALES) * len(LAMBDAS_STRUCT)
    print(f"\nStructural envelope: {len(WEIGHT_SCALES)} recency scales x "
          f"{len(LAMBDAS_STRUCT)} lambda = {n_corners} corners (re-fit per scale)...")
    struct = structural_points()
    p_point = pooled_pass / n_total
    lo, hi = published_interval(p_point, pct(p5_sorted, 0.05),
                               pct(p5_sorted, 0.95), [pt[0] for pt in struct])
    print("\nE.3 published intervals (parameter + structural + MC, enveloped):")
    print(f"  P(>=5 correct) = {fmt_interval(p_point, lo, hi)}")
    print(f"\n  {'Team':12s} {'P(3-0)':>16s} {'P(adv)':>16s} {'P(0-3)':>16s}")
    for t in sorted(STAGE3_TEAMS,
                    key=lambda t: -(pooled[t]["p30"] + pooled[t]["padv"])):
        row = f"  {t:12s}"
        for c in ("p30", "padv", "p03"):
            xs = sorted(per_sample_team[t][c])
            sv = [pt[1][t][c] for pt in struct]
            clo, chi = published_interval(pooled[t][c], pct(xs, 0.05),
                                          pct(xs, 0.95), sv)
            row += f" {fmt_interval(pooled[t][c], clo, chi):>16s}"
        print(row)


if __name__ == "__main__":
    main()
