# Assumptions Register (E.6)

> Every model assumption → how it could fail → how we'd **detect** failure →
> test status. Rule: a limitation without a detection mechanism is a
> disclaimer, not coverage. Review at every postmortem (next: June 15,
> then post-playoffs). Created 2026-06-12, mid-Stage-3.

Status legend: **tested** (automated check exists) · **process** (ritual/
checklist gate) · **pending** (detection designed, measurement waiting on
data) · **accepted** (consciously unmitigated; revisit trigger named).

## Data layer

| # | Assumption | Failure mode | Detection | Status |
|---|-----------|--------------|-----------|--------|
| D1 | matches_2026.csv winners/losers are correct | Swapped direction or phantom row silently poisons ratings (4 instances caught in 2 days pre-audit) | E.1 two-source rule on entry; E.2 audit re-verified all w≥0.85 + Cologne rows (2026-06-11); CI fit-reproducibility gate pins CSV→ratings | tested + process |
| D2 | Recency weights (0.5/0.6/0.65/0.85/1.0) reflect information decay | Over/under-weighting form; ratings chase noise or lag | Phase 3 walk-forward backtest estimates decay (reparameterized as half-life per deep-dive) | pending (E.7: no mid-event changes) |
| D3 | BO1 discount 0.6 | BO1 upsets over/under-counted | Phase 3 backtest | pending |
| D4 | Dataset covers each team's 2026 series (no missing-series bias) | Missing series (e.g. the omitted Falcons>Spirit Rio loss) bias ratings invisibly | Per-team coverage audit vs team pages — done once (2026-06-11); repeat at ingestion (2.2) row-by-row vs two sources | process |

## Rating model

| # | Assumption | Failure mode | Detection | Status |
|---|-----------|--------------|-----------|--------|
| M1 | Scalar BT strength is transitive | Style/H2H effects (donk vs NAVI 11-2) make A>B>C>A cycles | Per-match postmortem (postmortem_matches.py) accumulates evidence; Phase 5 player model is the fix | pending |
| M2 | Gaussian prior σ=70 (S3) / 50 (connectors) | Too sticky: data can't correct a wrong prior; too loose: 3-0 runs blow up | fit.py prints prior/fit/anchored side by side (drift visible); Phase 3 estimates σ | pending |
| M3 | Priors (market-informed 2026-06-09) are unbiased starting points | Stale or vibe-biased priors persist in sparse data | Same fit.py drift table; large prior-vs-fit gaps flag review | process |
| M4 | Ratings static within stage | Form swings mid-event (9z?) unmodeled | Per-match postmortem split R1 vs R2+ (lock-foresight grading shows decay if real) | **new 2026-06-12: detectable** |
| M5 | Series = single Bernoulli draw (no maps) | Tail miscalibration; map variance compounds differently | Phase 4 re-scope: map-level FITTING (~220 obs); record-multiset calibration in backtest | pending |
| M6 | ANCHOR_LAMBDA=0.5 propagation split | Market info smeared globally (or wasted) — unidentifiable at team level | Lambda envelope in sensitivity.py + playoffs.py every lock; slate impact currently confined to one slot | tested (per-lock) |
| M7 | Elo/400 logistic link calibrated to BO3 outcomes | Scale miscalibration distorts all probs | Backtest calibration curve (Phase 3) | pending |

## Market layer

| # | Assumption | Failure mode | Detection | Status |
|---|-----------|--------------|-----------|--------|
| K1 | Polymarket mids ≈ true probs after proportional de-vig | Venue bias: compression-toward-50% (deep-dive), favorite-longshot under proportional de-vig | Venue-calibration curve from odds_archive (close vs outcome) — archive accumulating since 06-11; Phase 3 measures; E.4 haircut rule for slate-sensitive anchors meanwhile | pending + process |
| K2 | Anchor lines fresh at lock | v1 failure mode: stale manual lines changed picks | Pre-lock ritual: fetch → re-fit → sensitivity at T-12h and T-1h | process |
| K3 | p>0.95 fetch guard = "match finished, not forecast" | A genuine 0.95+ pre-match favorite would be dropped from anchors | Accepted: rating-implied prob used instead; error bounded by model-market gap on a near-certain match | accepted |
| K4 | Playoff anchors price the actual series format | Market prices BO3 while final is BO5 (or vice versa) | Lock-day step: read the market question text before writing playoff_anchors.json | process |

## Tournament structure

| # | Assumption | Failure mode | Detection | Status |
|---|-----------|--------------|-----------|--------|
| S1 | Swiss pairing = true seeds + highest-vs-lowest + Valve priority table | Valve manual overrides / rule deviation (the R2 seed-order error cost 1/8 predicted pairings, ~0.008 P(≥5)) | Validated 8/8 vs announced R2; R4 priority table exact w/ tests; routine compares announced pairings every round | tested |
| S2 | Playoff seeding = W-L → Buchholz → initial seed; 1v8+4v5 / 2v7+3v6 | Rulebook misread or supplemental change | Cross-check derived seeds vs announced bracket at lock (ritual step, playoffs.py prefers announced file) | process |
| S3 | Grand final is BO5 | Rulebook generic text says all-BO3 (conflict noted 06-12) | Official schedule at lock; GRAND_FINAL_BO5 flag covers either | process |
| S4 | Playoff pick'em = full bracket, single pre-QF lock, challenges ≥2QF/≥1SF/champ | Secondary sources wrong (confidence medium-high, ×3 agreement) | **OPEN: screenshot in-client rules when picks open (~June 17) before optimizing** | pending — blocking |
| S5 | BO5 prob from BO3 prob via iid map inversion q²(3−2q) | Map-level correlation (momentum, vetoes) breaks iid | Market override preferred wherever a BO5 line exists (K4); Phase 4 measures | accepted (bounded by K4) |

## Inference / decision layer

| # | Assumption | Failure mode | Detection | Status |
|---|-----------|--------------|-----------|--------|
| I1 | Laplace posterior (K=200, seed 11) captures rating uncertainty | Non-Gaussian tails; underdispersed draws | Across-draw 5-95% spread reported next to MAP; Phase 3 can compare to bootstrap | pending |
| I2 | MC error ±0.003 at 40k sims is decision-irrelevant | Knife-edge picks flip on noise | Seed-stability checks (7/11/42/123) at lock; playoff layer is EXACT (128 branches, no MC) | tested |
| I3 | Optimizer objective matches the actual scoring rule | Optimizing the wrong objective (e.g. challenges vs champion-only) | S4 verification + OBJECTIVE set per Will's coin status at lock — **input still open** | pending — blocking |
| I4 | Degenerate pick slots are identified per objective | Structural ties misread as evidence (challenges ignores non-champ-side SF — caught by smoke test 06-12) | Pinning test + tie-aware margin reporting in playoffs.py; design rule: enumerate invariant slots for ANY new scoring rule | tested |

## Meta

- The two **blocking-pending** rows (S4 rules verification, I3 objective)
  are lock-day gates — playoffs.py output is not pick-ready until both close.
- This file is reviewed (and statuses updated) at every postmortem; new
  assumptions enter when introduced, not when they bite.
