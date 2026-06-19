"""W15 / U2 -- ONE-OFF Laplace tail certification (run once; NOT a pipeline dep).

posterior.py approximates the Bradley-Terry posterior as a Gaussian at the MAP
(Laplace: invert the Hessian of the negative log-posterior). That is exact only
if the log-posterior is quadratic. BT is curved, so the Gaussian can understate
tail mass and make U1's published intervals look more certain than the data
warrants. This script samples the TRUE posterior by component-wise Metropolis-
Hastings and compares its marginal spread + tails, per team, against the Laplace
Gaussian -- certifying whether Laplace is adequate for the U1 interval reporting.

  Run:    python src/u2_laplace_certification.py
  Output: prints a per-team table + writes the finding to
          docs/research/<date>-u2-laplace-tail-certification.md

NOTHING in the pipeline imports this module. It is a one-off validation: once the
finding is recorded, the certification stands on the written report, not on a
standing MCMC dependency (DoR U2: "one-off ... then discard").
"""

import datetime
import math
import random
from pathlib import Path

from model import PRIORS, STAGE3_TEAMS, load_matches, win_prob
from posterior import cholesky, hessian, laplace_factor

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs" / "research"

N_SAMPLES = 50_000     # post-burn samples
BURN = 10_000
STEP = 22.0            # component proposal sd (Elo); tuned for healthy acceptance
SEED = 11


def sigma_map(teams):
    return {t: (70.0 if t in STAGE3_TEAMS else 50.0) for t in teams}


def marginal_sd_from_L(L, teams):
    """Laplace marginal sd per team = sqrt(diag(H^-1)), via the Cholesky factor
    of H (H = L L^T). Solve H x = e_i for each team i; x[i] is its variance."""
    n = len(teams)
    sd = {}
    for i, t in enumerate(teams):
        e = [1.0 if j == i else 0.0 for j in range(n)]
        y = [0.0] * n                                   # forward: L y = e
        for r in range(n):
            y[r] = (e[r] - sum(L[r][k] * y[k] for k in range(r))) / L[r][r]
        x = [0.0] * n                                   # back: L^T x = y
        for r in range(n - 1, -1, -1):
            x[r] = (y[r] - sum(L[k][r] * x[k] for k in range(r + 1, n))) / L[r][r]
        sd[t] = math.sqrt(x[i])
    return sd


def metropolis(matches, priors, sigma, map_ratings, n, burn, step, seed):
    """Component-wise random-walk Metropolis over the BT log-posterior. Each
    sweep proposes a Gaussian move per team and accepts on the local delta
    (only that team's matches + prior change), so a sweep is O(sum of degrees)."""
    rng = random.Random(seed)
    teams = list(priors)
    by_team = {t: [] for t in teams}
    for m in matches:
        by_team[m[0]].append(m)
        by_team[m[1]].append(m)
    cur = dict(map_ratings)

    def contrib(t):
        s = -0.5 * (cur[t] - priors[t]) ** 2 / sigma[t] ** 2
        for w, l, wt in by_team[t]:
            s += wt * math.log(max(win_prob(cur, w, l), 1e-12))
        return s

    samples = {t: [] for t in teams}
    accepts = proposals = 0
    for it in range(n + burn):
        for t in teams:
            old, old_c = cur[t], contrib(t)
            cur[t] = old + rng.gauss(0.0, step)
            proposals += 1
            if math.log(rng.random() + 1e-300) < contrib(t) - old_c:
                accepts += 1
            else:
                cur[t] = old
        if it >= burn:
            for t in teams:
                samples[t].append(cur[t])
    return samples, accepts / proposals


def pct(xs, q):
    s = sorted(xs)
    return s[min(int(q * len(s)), len(s) - 1)]


def main():
    matches = load_matches()
    map_ratings, teams, L = laplace_factor()
    sigma = sigma_map(teams)
    lap_sd = marginal_sd_from_L(L, teams)

    print(f"Metropolis: {N_SAMPLES} samples (+{BURN} burn), step={STEP} Elo...")
    samples, acc = metropolis(matches, PRIORS, sigma, map_ratings,
                              N_SAMPLES, BURN, STEP, SEED)
    print(f"acceptance rate: {acc:.2f}\n")

    rows = []   # (team, lap_sd, mcmc_sd, sd_ratio, lap_95w, mcmc_95w, tail_ratio)
    for t in STAGE3_TEAMS:
        xs = samples[t]
        m = sum(xs) / len(xs)
        mcmc_sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
        # central 95% width: Laplace = 2*1.96*sd; MCMC = empirical 2.5..97.5 pct.
        lap_w = 2 * 1.959964 * lap_sd[t]
        mcmc_w = pct(xs, 0.975) - pct(xs, 0.025)
        rows.append((t, lap_sd[t], mcmc_sd, mcmc_sd / lap_sd[t],
                     lap_w, mcmc_w, mcmc_w / lap_w))

    print(f"{'Team':12s} {'LapSD':>6s} {'MCMCsd':>7s} {'sd_r':>5s} "
          f"{'Lap95w':>7s} {'MC95w':>7s} {'tail_r':>6s}")
    for t, ls, ms, sr, lw, mw, tr in rows:
        print(f"{t:12s} {ls:6.1f} {ms:7.1f} {sr:5.2f} {lw:7.1f} {mw:7.1f} {tr:6.2f}")

    sd_ratios = [r[3] for r in rows]
    tail_ratios = [r[6] for r in rows]
    max_tail = max(tail_ratios)
    med_tail = sorted(tail_ratios)[len(tail_ratios) // 2]
    # Verdict: Laplace is adequate if the true 95% width never exceeds the
    # Gaussian's by more than 15% (intervals are already enveloped + MC-floored).
    adequate = max_tail <= 1.15
    verdict = ("ADEQUATE -- Laplace does not materially understate the tails; "
               "U1's enveloped intervals already absorb the residual."
               if adequate else
               f"UNDERSTATES -- true 95% width exceeds Gaussian by up to "
               f"{(max_tail - 1) * 100:.0f}%; widen U1 or note the floor.")
    print(f"\nmedian tail ratio {med_tail:.2f}, max {max_tail:.2f} -> {verdict}")

    write_finding(rows, acc, med_tail, max_tail, sd_ratios, verdict)


def write_finding(rows, acc, med_tail, max_tail, sd_ratios, verdict):
    DOCS.mkdir(parents=True, exist_ok=True)
    date = datetime.date.today().isoformat()
    out = DOCS / f"{date}-u2-laplace-tail-certification.md"
    med_sd = sorted(sd_ratios)[len(sd_ratios) // 2]
    lines = [
        f"# U2 -- Laplace tail certification ({date})",
        "",
        "**One-off** (W15/U2). Certifies whether the Laplace (Gaussian-at-MAP)",
        "posterior in `posterior.py` adequately captures the BT posterior tails",
        "that U1's published intervals rest on. Method: component-wise Metropolis-",
        f"Hastings on the exact BT log-posterior, {N_SAMPLES} samples (+{BURN} burn,",
        f"step {STEP:.0f} Elo, seed {SEED}, acceptance {acc:.2f}), compared per team",
        "against the Laplace marginal sd (sqrt(diag(H^-1))).",
        "",
        f"**Verdict: {verdict}**",
        "",
        f"Median MCMC/Laplace sd ratio {med_sd:.2f}; median central-95% width ratio "
        f"{med_tail:.2f}, max {max_tail:.2f}.",
        "",
        "| Team | LaplaceSD | MCMC sd | sd ratio | Laplace 95%w | MCMC 95%w | tail ratio |",
        "|------|----------:|--------:|---------:|-------------:|----------:|-----------:|",
    ]
    for t, ls, ms, sr, lw, mw, tr in rows:
        lines.append(f"| {t} | {ls:.1f} | {ms:.1f} | {sr:.2f} | {lw:.1f} | "
                     f"{mw:.1f} | {tr:.2f} |")
    lines += [
        "",
        "`sd ratio` and `tail ratio` > 1 mean the true posterior is wider than the",
        "Gaussian (Laplace understates). U1 envelopes parameter + structural + an",
        "MC floor, so a small understatement is already absorbed; a large one would",
        "warrant widening the parameter component or raising the MC floor.",
        "",
        "_Reproduce: `python src/u2_laplace_certification.py`. Not imported by the",
        "pipeline; the certification stands on this report (DoR U2: one-off, then discard)._",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"\nfinding -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
