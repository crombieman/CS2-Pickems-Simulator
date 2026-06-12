# Deep Dive: Making the CS2 Pick'em Simulator More Realistic and Statistically Accurate

> Researched: 2026-06-11/12
> Analyst: Claude Deep-Dive Agent
> Confidence: see per-claim annotations | Sources: see Sources section
> Scope: state-of-the-art esports/CS2 win-probability modeling vs. our current
> Bradley-Terry + market-anchor + Swiss Monte Carlo stack; sequenced
> recommendations that respect E.7 (backtest-gated knob changes), the
> frozen-lock policy, and hobbyist-institutional scale.

## Executive Summary

The current stack (decayed MAP Bradley-Terry + market anchors + exact Valve
Swiss simulation + joint-distribution slate optimizer + Laplace intervals)
is architecturally correct by the standards of the literature and ahead of
every public CS2 pick'em artifact found. The literature actively *rejects*
the obvious "upgrades": in sparse-data regimes (teams playing 10-40
series/year), simple Elo/BT-class models empirically beat Glicko-2,
TrueSkill, and neural raters (all within ~0.3pp of each other on 10K pro
CS:GO matches), and estimated forecast-combination weights routinely lose
to a fixed 50/50 blend — so neither the rating engine nor ANCHOR_LAMBDA=0.5
is the weak point. The real gains, in order: (1) decision-layer fixes that
need no new data — optimize the slate over the posterior predictive instead
of the MAP fit, report pick margins with paired SEs, widen intervals with
the structural envelope; (2) extracting more information per match once
bo3.gg data lands — fitting on maps instead of series (~2.4x observations)
plus margin-relative-to-expectation round-diff weighting (literature: +0.5pp
accuracy, −0.7% Brier in the NBA analog), with Phase 4 re-scoped from
"map-level simulation" to "map-level fitting" since map scores never touch
the Swiss machinery; (3) better-posed Phase 3 estimation — one half-life
parameter instead of four bucket weights, an empirically measured BO1
discount, and a Polymarket venue-calibration curve instead of a Shin de-vig
switch (Polymarket's bias structure is not bookmaker favorite-longshot
bias). Phase 5's expected accuracy gain now has a literature prior (~+1pp
over team-level); its real payoff remains roster-change robustness.

## Key Insights

1. **The sparse-data literature inverts the intuition that fancier ratings
   are better.** Model misspecification is provably not the binding
   constraint at our n — online-learning regret is — so swapping BT for
   Glicko-2/TrueSkill/WHR buys ~nothing (arXiv 2502.10985; Bober-Irizar
   CS:GO numbers). The estimable weaknesses are the hand-set decay and
   sigma, exactly what Phase 3 already targets.
2. **Map scores never enter Buchholz, pairing, or advancement — so
   map-level work is a fitting upgrade, not a simulation upgrade.** Phase 4
   as written ("BO3s as map sequences") would change nothing downstream if
   series probs stayed fixed. Re-scope it to map-level *fitting*: 2.4x
   observations per series is the largest statistical gain available, and
   it absorbs the 2-0/2-1 margin signal for free.
3. **The optimizer maximizes the right objective under the wrong measure:**
   it argmaxes P(>=5) under the MAP rating vector while posterior.py proves
   ratings carry ±62 Elo of uncertainty. Optimizing E[P(>=5)] across
   posterior draws is the cheapest genuinely-new improvement found — no new
   data, existing machinery, and it targets exactly the 0.003-0.007 margins
   that decided MongolZ-vs-G2.
4. **Polymarket's miscalibration is not bookmaker favorite-longshot bias**
   (it compresses toward 50% / longshots resolve more often than priced in
   one study) — so the planned Shin/power de-vig comparison is measuring
   the wrong distortion. The fix is an empirical venue calibration curve
   from the 0.4 odds archive.
5. **At n=16-per-event, only match-level grading against a market baseline
   accumulates statistical power fast enough to ever change a knob.** One
   event = ~33 match forecasts; five events = a usable calibration log.
   Per-team tail Briers alone would take years to separate models — the
   measurement spec matters as much as the model.

## Current Approach — Baseline Audit (from code, verified 2026-06-11)

- **Rating fit** (`src/model.py`): MAP Bradley-Terry on ~92 weighted series,
  Gaussian prior toward market-informed PRIORS (sigma 70 Elo Stage-3 / 50
  connector), gradient ascent 4000 iters. Weights: recency 0.5-0.85 by date
  bucket (Cologne=1.0), BO1 x0.6. Translation re-centered to prior mean.
- **Market layer**: proportional de-vig of Polymarket two-sided mids; anchored
  pairs play at exact market prob (pair overrides); ANCHOR_LAMBDA=0.5 of the
  market-vs-fit correction propagates into ratings for non-anchored matchups.
- **Swiss sim** (`src/simulate.py`): full endogenous Swiss, Buchholz recomputed
  per round, Valve 15-row priority table for 6-team groups (validated 8/8 vs
  real R2 pairings), rematch avoidance, series as single Bernoulli draws.
- **Optimizer** (`src/optimize.py`): exhaustive over top-k candidates per slot
  (k=5/5/9), scores P(>=5 of 10) empirically on 40K stored sims — correlation-
  aware by construction.
- **Uncertainty** (`src/posterior.py`): Laplace approximation at MAP, 200
  rating draws x 2000 sims; P(>=5)=0.40 [0.32-0.48]; rating posterior sd ~62
  Elo (data barely tightens priors).
- **Live** (`src/live.py` + `make_state`): mid-stage resume with validation.

Status checklist (research progress):
- [x] Q1 Rating systems (dynamic BT / Glicko-2 / TrueSkill / Elo variants)
- [x] Q2 Map-level & round-level CS modeling
- [x] Q3 Margin/score information (2-0 vs 2-1, round diff)
- [x] Q4 Market integration & de-vig / favorite-longshot
- [x] Q5 Uncertainty quantification (Laplace vs bootstrap vs full Bayes)
- [x] Q6 Swiss-format literature (pairing fidelity vs match-prob fidelity)
- [x] Q7 Pick'em/portfolio optimization (DFS literature)
- [x] Q8 Calibration measurement with tiny n
- [x] Q9 Data scale implications (bo3.gg 71K matches)
- [x] Q10 Best public CS2 prediction projects / academic work

## Q1. Rating Systems: Dynamic vs Static-with-Recency-Weights

**The headline finding is counter-intuitive and directly load-bearing: in
sparse-data regimes, simple Elo/BT-class models empirically beat more complex
rating systems, even though the BT assumptions are statistically rejected by
the data.** "Is Elo Rating Reliable? A Study Under Model Misspecification"
(arXiv 2502.10985, 2025) decomposes total predictive loss into model
misspecification error + online-learning regret, and shows that when each
player has few games (our regime: top CS2 teams play ~10-40 series/year),
regret dominates — complex models (neural Elo2k, full pairwise) lose to plain
Elo across chess, Go, tennis, and StarCraft datasets. Likelihood-ratio tests
reject the BT model at p < 1e-10 in all 8 datasets tested, *and Elo still
wins on prediction*. [HIGH confidence — primary paper fetched and read]

Implications for us:
- Our static MAP-BT with recency weights is a discretized approximation of a
  dynamic model (recency weight ~ exponential forgetting). The literature's
  dynamic BT lineage — Fahrmeir & Tutz 1994 state-space, Glickman 1993/1998
  approximate Bayes (→ Glicko), Cattelan et al. 2013 EWMA dynamics, Bong et
  al. 2020 nonparametric kernel-smoothed BT — all fit time-varying strengths,
  but the measured prediction gains over well-tuned static/decayed fits in
  sparse data are consistently small (Cattelan reports dynamic ≈ static with
  EWMA weighting in basketball; the CMU/MLR work is about estimation
  guarantees, not accuracy gains). [MEDIUM — abstracts and summaries, not all
  full texts]
- Glicko-2 vs TrueSkill on ~10K pro CS:GO matches (attributed to
  Bober-Irizar et al. 2024 in secondary surveys): Glicko-2 63.1%, TrueSkill
  62.9%, Elo 62.8% — essentially tied; *player-level* granularity
  ("TrueSkillPlayers") reaches ~64.1%, i.e. **player-composition modeling
  buys roughly +1pp accuracy over team-level ratings in pro CS** — a
  realistic prior for what Phase 5 delivers on accuracy (its roster-change
  robustness is worth more than the raw pp). Ratings-only CS prediction
  ceiling: roughly 62-67%. [MEDIUM — two secondary summaries agree
  (emergentmind topic pages, cross-checked across two searches); primary
  PDF not independently fetched]
- TrueSkill2's famous 68% vs 52% gain over TrueSkill (Minka et al. 2018) came
  from *adding individual player statistics* (kills/deaths) and squad
  correlation in matchmade games — i.e., the gain is from richer features,
  not from fancier dynamics. That is an argument for Phase 5
  (player-composition), not for swapping the rating engine. [HIGH — widely
  replicated claim from the primary paper's own abstract]
- The principled middle ground if we ever want time-variation: Whole-History
  Rating (Coulom 2008) — batch-fit all matches with a Wiener-process prior on
  each team's strength trajectory; it is exactly our MAP-BT + Gaussian priors
  extended with a time axis, stays stdlib-feasible, and produces per-date
  ratings + uncertainty. But per the misspecification paper, at n≈92 series
  the expected gain over tuned recency weights is small; this is a
  Phase-3-backtest question ("does WHR beat decayed static BT out of
  sample?"), not an adopt-now item.

**Verdict: keep the engine; let the backtest estimate the decay.** The thing
the literature actually criticizes in setups like ours is *hand-set* decay
(0.5-0.85 buckets) and *hand-set* sigma — both already slated for Phase 3
estimation. The marginal value of Glicko-2/TrueSkill-style per-team
uncertainty is mostly already delivered by the Laplace posterior (which gives
per-team rating sd ~62 Elo). One genuine gap vs Glicko-class systems: our
posterior width does not grow for teams with *stale* data (a team unseen for
3 months has the same sigma as one seen last week). A cheap, principled fix
is inflating prior sigma with data staleness — backtestable, one line.

Also relevant: Valve's own VRS model (see Q10) self-reports miscalibration at
the extremes — "underestimates win rates at the low end and overestimates at
the high end" — i.e., even the official invisible rating is a worse
probability model than a tuned BT; VRS is a *ranking* device, not a
probability device. Do not be tempted to ingest VRS points as probabilities.
[HIGH — Valve's own repo documentation]

## Q2. Map-Level and Round-Level Modeling in CS

**Structural observation (from our own code, verified): map-level simulation
changes nothing about the Swiss machinery.** In `simulate.py`, Buchholz is
computed from opponents' W-L only; map scores never enter pairing, seeding,
or advancement. Therefore the entire value of Phase 4 map-level work flows
through exactly one channel: *better series win probabilities* (and,
secondarily, live mid-series conditioning). This reframes Phase 4: it is a
FITTING upgrade (map results as data, veto as a probability adjuster), not a
SIMULATION upgrade. Simulating BO3s as map sequences with a fixed series
prob would change nothing downstream. [HIGH — direct code inspection]

What the literature offers:
- **Veto modeling**: Xenopoulos et al., "Bandit Modeling of Map Selection in
  CS:GO" (arXiv 2106.08888) — 3,500 matches, 25K veto decisions, contextual
  bandits. Findings: team veto behavior is predictable and *suboptimal* —
  optimal vetoes would improve a team's predicted map win prob by up to 11%
  and match win prob by ~20% for evenly-matched teams. Implication: vetoes
  carry real probability mass, and because teams are predictable, a simple
  empirical veto model (each team bans its historically worst maps) captures
  most of it. [HIGH for the paper's existence/claims; MEDIUM for
  transferability to 2026 CS2 map pool]
- **Round-level / economy models**: Xenopoulos's round win-prob models
  (XGBoost on players-alive/equipment, arXiv 2011.01324) and the economy
  decision paper (arXiv 2109.12990) are *in-match live* models. They predict
  rounds given game state — useless for pre-match series probs. Round-level
  modeling is firmly overkill for pick'em simulation. [HIGH]
- Per-map team strengths: standard practice in CS betting models (map-
  specific win rates are a core feature of every commercial predictor
  surveyed, e.g. cs2bet.io's listed feature set). No published
  apples-to-apples measurement of series-prob accuracy gain from map-level
  vs series-level fitting was found for CS specifically. [Gap noted —
  unverifiable; the Phase 3 harness can measure it on bo3.gg data]

**The tennis analogy is the right mental model**: tennis forecasting fits
set/game-level models and aggregates upward (Klaassen & Magnus; the i.i.d.
assumption is mildly violated but the hierarchy still wins) — because lower
levels multiply the effective sample size. A BO3 series gives 1 series
observation but 2-3 map observations and ~30-60 round observations. At
n=92 series, that hierarchy is where the statistical gains live (see Q3).

## Q3. Margin / Score Information

**This is the cheapest real accuracy gain available to us, and the
literature consistently confirms margin information helps — but modestly.**
Measured magnitudes:
- MOVDA (arXiv 2506.00348, NBA, 13.6K games): Brier 0.2258 vs 0.2274
  (-0.7%), accuracy 63.32% vs 62.77% (+0.55pp) over binary Elo; also beat
  TrueSkill by 1.54% Brier. Notably, *naive* MOV scaling (K-factor times
  margin) conflates expected blowouts with surprises; the gain comes from
  margin-relative-to-expectation. [HIGH — primary paper fetched]
- Hvattum & Arntzen (soccer goal difference): "modest gains." Kovalchik
  2020 (Int. J. Forecasting, "Extension of the Elo rating system to margin
  of victory") found MOV-Elo variants beat win-only Elo in tennis. [MEDIUM —
  abstracts/secondary only]
- The faster-convergence result matters more than the accuracy result for
  us: MOVDA converged to stable ratings in 166 vs 193 games (13.5% faster).
  **With ~10-40 series per team per year, anything that extracts more
  information per match is worth proportionally more to us than to the
  NBA.** [HIGH for the number; the extrapolation to our regime is inference]

Concrete options ranked for our stack (all backtest-gated per E.7):
1. **Fit BT on maps, not series** (needs bo3.gg per-map data, Phase 2.2):
   ~92 series → ~220 map observations. If maps were independent, rating SEs
   shrink by ~sqrt(2.4) ≈ 1.55x; real gain is less (intra-series
   correlation — same day, same form, veto selection) so weight maps within
   a series at < 1.0, with the weight estimated in the backtest. This
   simultaneously absorbs the 2-0 vs 2-1 distinction (a 2-0 is two map wins,
   a 2-1 is two wins one loss) with zero extra model machinery — no
   Davidson/ordinal model needed. Davidson-style tie models are irrelevant
   (no draws in CS); ordinal series-score models add nothing over map-level
   Bernoulli fitting. [Analytic reasoning on verified data structure]
2. **Round-differential weighting within maps** (16-2 vs 16-14): the MOVDA
   lesson applies — use margin relative to expected margin, or simply a
   mild weight multiplier (e.g. logistic in round diff), estimated, not
   hand-set. Second-order vs option 1; bundle into the same backtest sweep.
3. **BO1 discount becomes estimable**: with 71K bo3.gg matches, the BO1
   vs BO3 upset-rate differential can be measured directly instead of the
   hand-set 0.6 (see Q9).

## Q4. Market Integration, De-Vig, Favorite-Longshot Bias

**Finding 1 — the lambda=0.5 blend is better-supported by literature than it
looks.** The forecast-combination literature's most robust result (the
"forecast combination puzzle"; Wang & Hyndman et al. 2022 review, arXiv
2205.04216; Clemen & Winkler shrinkage work) is that *estimated* optimal
weights routinely lose to the simple 50/50 average out of sample, because
weight-estimation variance eats the gains. At our data scale, a hand-set 0.5
is not an embarrassment — it is approximately what the literature converges
to when weight-estimation data is thin. [HIGH for the combination-puzzle
result; the application to our lambda is analytic inference]

Two genuine upgrades exist, both backtest-gated:
1. **Precision-weighted blending**: weight market vs model by inverse
   variance per matchup. We already *have* per-matchup model variance (the
   Laplace posterior gives rating sd → prob sd via the logistic derivative)
   and a market precision proxy (book volume + spread width). This turns the
   global scalar lambda into a principled per-matchup weight with zero new
   data. The combination literature supports precision weighting *when the
   precisions are known rather than estimated* — ours are, approximately.
   [MEDIUM — sound theory, no esports-specific empirical validation found]
2. **Market-as-prior in the fit** (replace post-hoc anchoring): set each
   anchored matchup's prior contribution as pseudo-observations at the
   market prob. Cleaner than the current shift-after-fit, and it makes
   lambda interpretable as pseudo-match count. Equivalent in the limit; not
   worth churn before the backtest can score it. [Analytic]

**Finding 2 — Polymarket's bias structure is NOT bookmaker favorite-longshot
bias, so Shin/power de-vig is solving the wrong problem for us.** Studies of
Polymarket calibration (Reichenbach & Walther, SSRN 5910522; arXiv
2602.19520 "Decomposing Crowd Wisdom"; aggregate trackers reporting Brier
~0.187 overall) find: (a) overall calibration is good and slightly *better*
than bookmakers in head-to-head comparisons; (b) the dominant distortion is
*compression toward 50%* at long horizons (favorites underpriced, longshots
overpriced in price space — sub-10% contracts resolving true ~14% of the
time in one study is the opposite sign; the literature is not unanimous on
direction, but agrees the structure differs from bookmaker vig loading);
(c) markets with professional traders are sharper than retail-dominated
ones. For 1-cent-spread two-way mids, classical de-vig method choice
(proportional vs power vs Shin) moves probabilities by well under 1pt —
confirmed negligible at our spreads. The actionable version of E.4 is not a
better de-vig formula; it is an *empirical Polymarket-vs-outcome calibration
curve* built from the 0.4 odds archive, which Phase 3 can fit once enough
events accumulate, plus the existing retail-skew haircut judgment on
fan-favorite teams. [MEDIUM-HIGH — multiple independent studies agree on
"calibrated overall, biased in segments"; direction-of-bias details differ]

**Finding 3 — markets beat models; anchoring is the right architecture.**
Every accuracy datapoint gathered (Q1's 62-67% ratings ceiling, Q10's
community models, the LLM project's 65%) sits at or below what closing-line
baselines achieve in liquid esports markets. The esports market-efficiency
picture: increasingly efficient since ~2020, with residual inefficiency in
low-liquidity/early lines (community + industry sources; no rigorous
CS2-specific academic study found — gap noted). Our architecture (market
verbatim where lines exist, model fills the gaps) matches best practice in
the sports-analytics literature. The model's job is the *joint* distribution
(Swiss correlations) that no market quotes — that's exactly what a pick'em
needs. [MEDIUM — architecture endorsement is analytic; market efficiency
specifics under-documented]

## Q5. Uncertainty Quantification

**Laplace is adequate here, and the literature says the gap is elsewhere.**
The BT posterior with Gaussian priors is log-concave (logistic likelihood ×
Gaussian prior), so it is unimodal and near-Gaussian — the regime where
Laplace approximations are excellent. Formal results on BTL uncertainty
quantification in sparse comparison graphs (Han et al., arXiv 2110.03874 /
Inf. & Inference 2023) establish valid MLE-based confidence intervals even
in sparse regimes; nothing there suggests Laplace materially misstates
interval width for a 30-team, 92-match, strongly-regularized fit. Full MCMC
(stdlib random-walk Metropolis over ~30 params) would be a half-session
validation exercise worth doing ONCE to confirm Laplace tails, then
discard. Bootstrap is dominated here: resampling 92 weighted matches gives
noisier intervals than the analytic posterior and handles the priors
awkwardly. [HIGH on log-concavity (mathematical fact); MEDIUM on "MCMC
won't move intervals" — verify once, cheaply]

Hierarchical Bayes is the real upgrade the literature points at: under
sparsity, hierarchical shrinkage (team strengths drawn from tier/region
hyperpriors) measurably beats flat MLE/MAP (multiple sources incl. the
emergentmind hierarchical-BT survey; standard result in sports modeling).
Our PRIORS dict *is* a hand-built hierarchy (tier levels hand-assigned);
with bo3.gg scale the hyperpriors become estimable (Q9). [MEDIUM]

What Laplace genuinely misses, in order of importance:
1. **Structural uncertainty** — lambda, sigma, recency, BO1 discount,
   data-entry quality. The roadmap's lambda sweeps are the right
   instrument; formalize as a small ensemble (e.g. 3 lambda x 2 recency
   settings), report the envelope. The interval [0.32-0.48] would honestly
   widen a bit more. Literature analogue: ensemble/multi-model spread in
   forecasting outperforms single-model parameter uncertainty alone. [HIGH
   conceptually]
2. **Staleness-blind sigma** — see Q1; Glicko's one good idea, one line
   to add.
3. Conformal prediction: designed for exchangeable per-instance prediction
   sets; awkward for a joint 16-team tournament functional with n=1 event.
   Overkill-now, likely overkill-forever for this use case. [MEDIUM]

## Q6. Swiss Format: Pairing Fidelity vs Match-Prob Fidelity

The academic literature (Sziklai, Biró & Csató, "The efficacy of tournament
designs", arXiv 2103.06023, Monte Carlo across 6 win-prob assumptions; Csató
et al. on Swiss unfairness, arXiv 2410.19333) establishes: Swiss is the most
*accurate* format class at ranking participants; its weaknesses are
tie-breaking reliability (four documented Buchholz shortcomings in chess
variants) and pairing-rule edge effects. But those papers optimize RANKING
fidelity. Our pick'em only needs the *advancement-class distribution*
(3-0 / advance / 0-3), which is much coarser and robust to pairing detail.
[HIGH for the papers' claims; the coarseness argument is analytic]

Internal evidence settles the budget question: the worst pairing error the
project ever had (entirely wrong seed mapping, R2 pairings 1/8 correct)
moved P(>=5) by only ~0.008 — while three missing/updated market anchors
*changed the picks* (v1→v2: Spirit 64.5% vs fitted 51% flipped 3-0 and 0-3
slots). With pairing now validated 8/8 plus the exact R4 priority table,
**the marginal value is overwhelmingly in match probabilities, not pairing
fidelity.** The parked "Valve exact seeding-difference pairing" item should
stay parked. [HIGH — project's own measured numbers]

One Swiss-specific subtlety worth keeping: the simulator's endogenous-path
correlation (R1 losers forced onto 3-1/3-2 paths) is a genuine structural
advantage over per-team independent-probability tools (most community
simulators expose only marginals; see Q10). The optimizer already exploits
it. No public tool found does this better. [MEDIUM — survey of found tools]

## Q7. Pick'em Slate Optimization as Portfolio Construction

The nearest literature — DFS portfolio optimization — is *less* applicable
than it first appears, and our current optimizer is already theoretically
correct for the actual objective:
- Hunter, Vielma & Zaman, "Picking Winners" (arXiv 1604.01455): maximize
  P(at least one of N entries clears a threshold) → submodular selection of
  high-variance, low-cross-correlation lineups. Applies to MULTI-entry
  contests. Valve pick'em is single-entry, fixed threshold (>=5 of 10), no
  opponents — the entire diversification apparatus collapses. [HIGH]
- Haugh & Singal ("How to Play Fantasy Sports Strategically") add opponent-
  behavior modeling (Dirichlet-multinomial over opponents' picks) — only
  matters for rank-payoff contests; irrelevant to a fixed pass threshold.
  [HIGH]
- Our `optimize.py` maximizes the exact empirical objective P(>=5) on the
  joint simulation distribution — this IS the correct formulation; the
  literature's approximations (jointly-Gaussian entry scores, pairwise-
  marginal bounds) exist because most settings can't simulate the joint
  distribution. We can. [HIGH — code inspection + literature comparison]

Three real gaps in the optimizer, in priority order:
1. **It optimizes the MAP model, not the posterior predictive.** The slate
   is argmax for one rating vector; under the Laplace posterior the optimal
   slate could differ (robust-optimization effect: picks that are near-ties
   at MAP can be dominated in posterior expectation when a team's rating is
   high-variance). Fix: score candidate slates on sims pooled ACROSS the
   200 posterior draws (machinery already exists in posterior.py) and pick
   argmax of E_theta[P(>=5 | theta)]. Cheap, principled, and uses only
   existing components. The MongolZ-vs-G2 margin (+0.003-0.007) is exactly
   the kind of decision this could flip or solidify. [Analytic — flagged as
   the highest-value optimizer change]
2. **Top-k truncation risk** (k=5/5/9): SmallRob's project brute-forces all
   ~10.1M valid slates (C(16,2)xC(14,2)xC(12,6)); at 40K sims that's
   infeasible in pure Python per-slate scoring, but a one-time exhaustive
   validation pass on a smaller sim store (e.g. 5K sims with bitset scoring)
   would empirically bound the truncation loss. Likely zero given flat
   optima, but currently unverified. [Inference — flat-surface claim is
   plausible but untested at full slate space]
3. **MC noise at the argmax margin**: 40K sims → SE(P(>=5)) ≈ 0.0025;
   observed pick margins of 0.003-0.007 are 1.2-2.8 SE. Common random
   numbers (same sim store for all slates — already done) cancels most of
   the noise on DIFFERENCES, and the multi-seed check (4 seeds) was the
   right ritual. Formalize: report the margin and its paired-difference SE
   (computable from per-sim hit indicators) so near-ties are visible at
   lock time. [HIGH — basic MC statistics]

## Q8. Calibration Measurement with Tiny n

The verification literature (meteorology, where this is most mature) gives
clear guidance:
- **Brier/Brier-skill sampling uncertainty is large at small n** — Bradley,
  Schwartz & Hashino 2008 (Weather & Forecasting) derive CIs for BS/BSS;
  related small-sample results show that with a handful of forecast-
  observation pairs, skill must exceed enormous thresholds (e.g. RPSS >
  0.42 in one 5-sample configuration) to be significant. n=16 correlated
  team-outcomes from one event separates nothing — E.7 is exactly right and
  the literature endorses it. [HIGH]
- **Skill scores need a baseline, and the right baseline is the market.**
  Grade every published probability against the market closing line's
  Brier on the same events (BSS with market reference). Beating
  climatology (uniform) is trivial; the market is the honest bar. Note the
  BSS is itself non-proper at small n (Murphy 1973 — asymptotically proper
  only), so report raw paired differences too. [HIGH]
- **Grow n by grading at the match level, not the event level.** One Swiss
  stage = ~33 match forecasts (and live re-forecasts add conditionals; map-
  level later multiplies again). After ~5-6 events the calibration log
  holds ~150-200 graded probabilities — enough for meaningful reliability
  curves and paired model-vs-market tests. Use paired per-match score
  differences with clustering by event (block bootstrap over events) since
  within-event outcomes are correlated through the Swiss structure. [HIGH
  methodology; standard practice]
- **Log score alongside Brier**: log score is more sensitive to tail
  errors — and our pick'em slots (3-0, 0-3) live in the tails. Brier
  under-penalizes a model that says 2% when truth is 6%. Both are free to
  log. [HIGH]
- Murphy decomposition (reliability/resolution): defer until n in the
  hundreds; at current n it's noise theater. [MEDIUM]

Concrete Phase 1/Phase 3 measurement spec this implies:
1. Postmortem grades v1/v2/v3 per-team (p30, padv, p03) Briers as LOGGED
   EVIDENCE, explicitly not as model selection (E.7 holds).
2. Calibration log schema: (date, event, match, p_model, p_market_close,
   p_published, outcome, model_version) — one row per match, plus per-team
   tail rows per event. The 0.4 odds archive supplies p_market_close.
3. Phase 3 harness primary metric: mean paired (Brier_model −
   Brier_market) per match with event-block bootstrap CI; secondary: log
   score, reliability curve once n>150.

## Q9. What 71K Matches of bo3.gg Data Unlocks

Model classes that flip from infeasible to feasible (all contingent on 2.2
ingestion with archive-first discipline):
1. **Estimated knobs replace hand-set knobs** (Phase 3 as planned) — and
   the parameterization should change: replace bucketed recency weights
   (0.5/0.65/0.85) with a single exponential half-life parameter (SmallRob
   uses 50-day half-life; estimate ours by walk-forward). One parameter
   estimates far better than four bucket weights. Same for BO1 discount:
   measurable directly from 71K matches as the BO1-vs-BO3 upset-rate gap
   conditional on rating difference. [Analytic + community precedent]
2. **Hierarchical priors** — team strength ~ N(tier/region mean, tau).
   Replaces the hand-built PRIORS dict with estimated hyperpriors; the
   sparse-BT literature consistently shows hierarchical shrinkage beating
   flat fits under sparsity (Q5). Caution: bo3.gg's 71K matches are mostly
   tier-2/3; without tier hierarchy they would actively pollute top-tier
   ratings through the connector graph. Hierarchy is not a luxury at that
   scale — it is the safety mechanism. [MEDIUM-HIGH]
3. **Map-level BT + empirical veto model** (Phase 4, reframed per Q2/Q3):
   per-map team strengths need ~3x the data per team — only viable at
   bo3.gg scale; veto frequencies per team are directly tabulable.
4. **Roster-aware fitting** (Phase 5): bo3.gg lineups per match make
   roster-change detection automatic (flag any match where lineup differs
   from current); even before full player-composition BT, a cheap
   "roster-discontinuity reset" (inflate sigma / reset to prior when 2+
   players change) captures most of the blind spot. TrueSkill2's measured
   gains came from player-level features — the ceiling for Phase 5 is
   real, but so is the role-confound work. [MEDIUM]
5. **Walk-forward eval at scale**: hundreds of past events → the Phase 3
   harness can actually rank model variants (static vs WHR vs map-level)
   with event-blocked significance, resolving every "backtest-gated"
   question in this report.

## Q10. What the Best Public CS2 Models Actually Do

Surveyed (READMEs fetched where listed; others from search descriptions):
- **Valve VRS** (github.com/ValveSoftware/counter-strike_regional_standings,
  fetched): rating = f(prize bounty offered/collected, opponent network,
  LAN wins), top-10 results in 6 months, age-weighted, normalized 400-2000.
  Valve's own evaluation: Spearman 0.98 between expected and observed win
  rates BUT with a shallow slope — overconfident at the top, underconfident
  at the bottom; community statistical review in repo issue #32. It is a
  ranking/invitation device, not a calibrated probability model. Do not use
  VRS points as probabilities (ndunnett's simulator does exactly this — a
  known weakness). [HIGH]
- **HLTV ranking**: points formula (form/achievements), never claimed to be
  predictive; no probability semantics. Manual reference only. [HIGH —
  long-standing public knowledge; not re-verified this session]
- **Community pick'em tools** (all fetched or search-described):
  - SmallRob/CS2_Major_Swiss (fetched): Elo with 50-day half-life decay,
    format multipliers (BO1 1.0 / BO3 1.2 / BO5 1.5 — NB: weights *up*
    BO3s rather than down-weighting BO1s, same idea as ours), adaptive K,
    HLTV-rating blending for new teams, Valve pairing rules, 100K sims,
    exhaustive 10.1M-slate brute force on P(>=5). No validation reported.
  - ndunnett/major-pickems-sim (fetched): VRS points → win prob heuristic,
    1M sims, Valve-documented seeding rules, explicit "not accurate"
    disclaimer. No optimizer.
  - claabs/cs-buchholz-simulator: user-supplied H2H odds matrix + sim —
    i.e., it outsources the hard part (probabilities) to the user. [MEDIUM
    — description only]
  - holygodly, x1aoqv, Foulest: plain Elo + MC variants. [LOW-MEDIUM —
    descriptions only]
- **LLM-based** (luizcieslak/cs2-match-prediction, fetched): HLTV stats +
  news → two-agent LLM pipeline; 65% match accuracy / 58.3% advancement
  accuracy at BLAST Austin Major 2025. Comparable to ratings-only models;
  no probability calibration, just picks. Interesting for *qualitative*
  inputs (roster news) — the thing our market anchors already price in.
- **Academic CS:GO/CS2 prediction**: ML on player stats/demos lands at
  ~62-70% accuracy (bachelor-thesis-grade work clustering player styles;
  IEEE round-prediction papers are in-match); the Glicko-2-on-10K-matches
  figure of 63.1% (Q1) is the cleanest ratings-only benchmark found.
  [MEDIUM]

**Positioning conclusion**: no public CS2 pick'em tool found combines (a)
market-anchored probabilities, (b) validated exact Valve pairing, (c) a
correlation-aware slate optimizer on the joint sim distribution, and (d)
parameter-uncertainty intervals. Our stack is ahead of every public
artifact surveyed on (a), (c), (d). The two ideas worth stealing are
SmallRob's exhaustive-slate validation pass and the single-parameter
half-life decay; the LLM project's only durable lesson is that news-aware
inputs matter (we get that via market anchors). [MEDIUM-HIGH — bounded by
"tools found in search"; private betting models are invisible to this
survey]

## Comparison Table: Technique -> Verdict

| Technique | Expected gain (evidence) | Data req | Cost at our scale | Verdict |
|---|---|---|---|---|
| Posterior-predictive slate optimization (optimize over Laplace draws, not MAP) | Robustness of picks at decision margins; exact-objective correctness (analytic; Q7) | none — reuses posterior.py | ~half session | **ADOPT** (no E.7 issue — decision layer, not a knob) |
| Paired-difference SE on slate margins + near-tie reporting | Honesty at lock time (basic MC stats) | none | hours | **ADOPT** |
| Exhaustive 10M-slate validation pass (a la SmallRob) | Bounds top-k truncation loss, likely ~0 (flat optima claim, untested) | none | ~half session, one-off | **ADOPT** (one-time validation) |
| Match-level calibration log + market-baseline BSS + log score, event-blocked bootstrap | Statistically honest grading at small n (Bradley et al. 2008; verification lit) | odds archive (0.4) | built into Phase 1/3 | **ADOPT** (measurement spec, Q8) |
| Single-parameter exponential recency half-life (replace weight buckets) | Better-estimable decay; 1 param vs 4 (estimation-variance argument; community precedent) | backtest events | trivial code; needs Phase 3 | **BACKTEST-THEN-ADOPT** |
| Staleness-inflated prior sigma (Glicko's RD-growth idea) | Honest uncertainty for stale teams (Glicko lineage) | match dates (have) | one line | **BACKTEST-THEN-ADOPT** |
| Map-level BT fitting (maps as observations, intra-series down-weight) | ~2.4x observations/series; rating SE shrink up to ~1.55x; absorbs 2-0/2-1 signal free (Q3 analytic; tennis-hierarchy precedent) | bo3.gg per-map (2.2) | Phase 4 as re-scoped | **BACKTEST-THEN-ADOPT** (highest expected model gain) |
| Round-diff margin weighting (margin-vs-expectation per MOVDA) | +0.5pp acc / −0.7% Brier in NBA analog; modest but real (arXiv 2506.00348) | bo3.gg round scores | small, bundle with above | **BACKTEST-THEN-ADOPT** |
| Empirical BO1 discount + format effects from 71K matches | Replaces hand-set 0.6 (no direct lit found) | bo3.gg (2.2) | free inside Phase 3 | **BACKTEST-THEN-ADOPT** |
| Precision-weighted market blend (per-matchup lambda from posterior var + book depth) | Principled per-matchup weighting; combination lit warns estimated weights often lose to 0.5 — hence gate | odds archive + posterior (have) | ~1 session | **BACKTEST-THEN-ADOPT** (fall back to 0.5 if no out-of-sample win) |
| Polymarket empirical calibration curve (E.4 made measurable) | Venue bias is real but direction differs from bookmaker FLB (Reichenbach & Walther; arXiv 2602.19520) | odds archive, multi-event | free inside Phase 3 | **BACKTEST-THEN-ADOPT** |
| Shin/power de-vig switch | <1pt at 1-cent two-way spreads (de-vig lit; arithmetic) | none | trivial | **REJECT as priority** (keep as free Phase 3 measurement, as planned) |
| Hierarchical tier/region priors over team strength | Beats flat fits under sparsity (sparse-BT lit); *required* safety once tier-2/3 data ingested | bo3.gg (2.2) | ~1 session inside Phase 3/5 | **BACKTEST-THEN-ADOPT** (mandatory before mass ingestion feeds the fit) |
| Whole-History Rating / dynamic BT / Glicko-2 engine swap | ~0 over tuned decayed BT at sparse n (arXiv 2502.10985; Cattelan; Bober-Irizar: all rating engines within 0.3pp) | none | medium | **OVERKILL-NOW** (revisit only if Phase 3 shows decay misfit) |
| Full MCMC over ratings | Laplace already near-exact for log-concave posterior (Q5) | none | half session | **OVERKILL** (do once as validation, then discard) |
| Structural-uncertainty ensemble (lambda x recency envelope in published intervals) | Wider, honest intervals; ensemble-spread precedent (forecasting lit) | none | ~half session | **ADOPT** (reporting change, not a knob change) |
| Veto model + per-map strengths in simulation | Veto behavior predictable, carries real prob mass (arXiv 2106.08888) | bo3.gg maps + veto logs | Phase 4 | **BACKTEST-THEN-ADOPT** (after map-level fitting proves out) |
| Player-composition BT (Phase 5) | ~+1pp accuracy over team-level (Bober-Irizar TrueSkillPlayers); roster-change robustness is the real prize | bo3.gg player data | 3+ sessions | **KEEP AS PLANNED** (expectations now calibrated: small accuracy delta, big blind-spot fix) |
| Round/economy-level in-match models | In-match live only; no pre-match value (Xenopoulos) | demos | large | **REJECT** for pick'em |
| Conformal prediction intervals | Wrong tool for joint tournament functionals (Q5) | n/a | n/a | **REJECT** |
| Exact Valve seeding-difference pairing refinements | Pairing error budget already ~0.008 P(>=5) at worst-ever; now validated 8/8 (internal evidence) | n/a | n/a | **REJECT** (stay parked, as roadmap says) |
| LLM news-analysis layer | 65% accuracy standalone (cieslak project) — below market anchors; redundant with them | HLTV scraping | large + fragile | **REJECT** |

## Strategic Recommendations: Roadmap Delta

The roadmap's architecture and sequencing survive contact with the
literature almost fully intact — the deltas below are re-scopings and
additions, not reversals. Everything parameter-touching routes through
Phase 3 per E.7; items marked "decision layer" or "reporting" don't touch
knobs and can land immediately.

**R1. Add now (decision/reporting layer, pre-playoff-lock if possible):**
- Posterior-predictive optimizer: pick the playoff slate (0.2) by argmax of
  E[P(pass)] across posterior draws, not MAP argmax. Reuses posterior.py;
  also satisfies 0.5's "do before playoff model" intent. (~half session)
- Margin + paired-SE reporting at every lock; publish near-ties explicitly.
- Structural envelope in published intervals: report the union across the
  existing lambda sweep settings alongside the Laplace interval (E.3+).

**R2. Phase 1 postmortem — adopt the Q8 measurement spec:** per-match
grading rows (not only per-team tails), market-close baseline column from
the 0.4 archive, log score + Brier, explicit "evidence not verdict" framing
(already E.7-compliant).

**R3. Phase 3 harness — three scope changes:**
  a. Reparameterize before estimating: exponential half-life (1 param)
     instead of bucket weights; estimated BO1 discount; staleness-sigma.
     Fewer, better-posed parameters = the small-n estimation the
     combination literature says actually works.
  b. Add the precision-weighted blend as a CANDIDATE against fixed
     lambda=0.5 — expect 0.5 to be hard to beat (forecast-combination
     puzzle); adopt only on out-of-sample win.
  c. Primary metric: paired per-match Brier difference vs market closing
     line, event-blocked bootstrap CI (not raw Brier).
  Keep the de-vig comparison as the free measurement it already is.

**R4. Phase 4 — re-scope from "map-level simulation" to "map-level
FITTING":** the sim needs nothing (Q2); the wins are (i) maps as fit
observations with intra-series down-weighting, (ii) margin-vs-expectation
round-diff weighting, (iii) only then an empirical veto/per-map layer for
series probs. Sequence (i) before (ii) before (iii); each gated on
walk-forward improvement. This is also the de-risked on-ramp to Phase 5.

**R5. Phase 5 — keep, with calibrated expectations:** literature prior is
~+1pp accuracy over team-level ratings; the justifying benefit is roster-
change robustness and emergent matchup effects, exactly as the roadmap
says. Before full player-BT, ship the cheap intermediate: roster-
discontinuity sigma reset (flag lineup changes from bo3.gg rosters).

**R6. Add one mandatory guard to 2.2/Phase 3:** hierarchical tier priors
BEFORE mass tier-2/3 data enters the fit. At 71K mixed-tier matches, flat
BT pollutes top-tier ratings through the connector graph; hierarchy is the
safety mechanism, not an enhancement.

**R7. Drop / keep parked (now with evidence):** rating-engine swaps
(Glicko-2/TrueSkill/WHR — sparse-regime literature says no gain), Shin
de-vig as a priority, exact-pairing refinements, round/economy models,
conformal intervals, LLM news layer.

**One-off validations worth a half session each:** exhaustive-slate pass to
bound top-k truncation; single MCMC run to certify Laplace tails.

## What Was NOT Investigated / Could Not Be Verified

- Bober-Irizar et al. 2024 (CS:GO rating-system comparison) — claims taken
  from two agreeing secondary summaries; primary PDF not fetched.
- Kovalchik 2020 MOV-Elo and Hvattum & Arntzen — abstracts/secondhand only
  (paywalled).
- No published measurement of map-level vs series-level fitting gain for
  CS specifically — the 2.4x-observations argument is analytic; Phase 3
  must measure it.
- Intra-series map correlation magnitude (momentum effects) in CS —
  no source found; the map-weight parameter must be estimated, not assumed.
- Polymarket *esports-specific* calibration — studies found cover politics/
  sports broadly; venue bias for CS2 markets specifically remains unmeasured
  (E.4 stands; the odds archive is the only path to an answer).
- Private/commercial betting models (Pinnacle esports pricing, GGbet
  models) — invisible by nature; the "our stack leads public tools" claim
  is bounded to artifacts discoverable via search.
- cieslak.dev benchmark blog post 403'd; used the GitHub README instead.
- Sziklai/Biró/Csató PDF parsed only via abstract page (PDF fetch returned
  binary); quantitative per-format numbers not extracted.
- claabs/holygodly/x1aoqv/Foulest simulators — search descriptions only.
- Valve playoff bracket seeding/scoring rules for Stage 4 (0.2 must verify
  from Liquipedia as planned — out of scope here).

## Sources

### Primary Sources (fetched/read)
- https://arxiv.org/html/2502.10985v1 — Is Elo Rating Reliable? A Study
  Under Model Misspecification (sparse-regime result)
- https://arxiv.org/html/2506.00348 — MOVDA margin-of-victory framework
  (NBA numbers)
- https://github.com/ValveSoftware/counter-strike_regional_standings —
  VRS methodology + self-reported calibration
- https://arxiv.org/abs/2103.06023 — Sziklai, Biró & Csató, The efficacy
  of tournament designs (abstract-level)
- https://github.com/SmallRob/CS2_Major_Swiss — community Elo+MC+brute-force
- https://github.com/ndunnett/major-pickems-sim — community VRS-based sim
- https://github.com/luizcieslak/cs2-match-prediction — LLM predictor,
  Austin 2025 accuracy
- Project code: src/model.py, simulate.py, optimize.py, posterior.py;
  README.md; docs/plans/2026-06-10-roadmap.md (direct inspection)

### Secondary Sources
- https://arxiv.org/abs/1604.01455 — Hunter, Vielma & Zaman, Picking
  Winners (DFS portfolio IP)
- http://www.columbia.edu/~mh2078/DFS_Revision_1_May2019.pdf — Haugh &
  Singal, How to Play Fantasy Sports Strategically
- https://arxiv.org/pdf/2205.04216 — Wang, Hyndman et al., Forecast
  combinations: a 50-year review
- https://arxiv.org/abs/2110.03874 — Han et al., Uncertainty
  quantification in the Bradley-Terry-Luce model
- https://arxiv.org/pdf/2106.08888 — Bandit Modeling of Map Selection in
  CS:GO (Xenopoulos et al.)
- https://arxiv.org/pdf/2011.01324 — Valuing Player Actions in CS:GO
- https://arxiv.org/abs/2109.12990 — Optimal Team Economic Decisions in
  Counter-Strike
- https://www.sciencedirect.com/science/article/abs/pii/S0169207020300157
  — Kovalchik, Extension of the Elo rating system to margin of victory
- https://journals.ametsoc.org/waf/article/23/5/992/39152 — Bradley,
  Schwartz & Hashino, Sampling Uncertainty and CIs for Brier Score/BSS
- https://journals.ametsoc.org/view/journals/mwre/135/1/mwr3280.1.xml —
  Discrete Brier and Ranked Probability Skill Scores (small-sample
  thresholds)
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5910522 —
  Reichenbach & Walther, Decentralized Prediction Markets: Accuracy,
  Skill, and Bias on Polymarket
- https://arxiv.org/html/2602.19520v1 — Decomposing Crowd Wisdom:
  Domain-Specific Calibration Dynamics in Prediction Markets
- https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9876.2012.01046.x
  — Cattelan et al., Dynamic Bradley-Terry modelling of sports tournaments
- https://www.glicko.net/research/glicko.pdf — Glickman, parameter
  estimation in large dynamic paired comparison experiments
- https://arxiv.org/html/2410.19333v3 — Swiss-system chess tournaments and
  unfairness
- https://cran.r-project.org/web/packages/implied/vignettes/introduction.html
  — de-vig methods reference (proportional/power/Shin)
- https://www.emergentmind.com/topics/glicko2-rating-system — secondary
  summary carrying the Bober-Irizar CS:GO comparison numbers

### Community & General
- https://www.hltv.org/news/36097/valves-swiss-system-under-the-microscope
  — NER0cs seeding-correlation analysis (fetched)
- https://github.com/claabs/cs-buchholz-simulator,
  https://github.com/holygodly/CS2-Major-ELO-PickEm-Predictor,
  https://github.com/x1aoqv/cs2-major-pickems-simulation,
  https://github.com/Foulest/Swiss — community simulators (descriptions)
- https://betherosports.com/blog/devigging-methods-explained,
  https://help.outlier.bet/en/articles/8208129 — de-vig practitioner guides
- https://en.wikipedia.org/wiki/Favourite-longshot_bias
- https://fensory.com/intelligence/predict/polymarket-accuracy-analysis-track-record-2026,
  https://www.tradetheoutcome.com/polymarket-accuracy-report-data/ —
  Polymarket calibration trackers

## Research Journal

- Read all five core source files + README + roadmap first; the single most
  consequential pre-search observation came from code, not literature:
  Buchholz never sees map scores, so map-level simulation cannot change the
  Swiss distribution given fixed series probs — which re-scopes Phase 4.
- Broad discovery (12 searches): rating systems, VRS, de-vig/FLB, CS
  academic work, DFS portfolio lit, market efficiency, dynamic BT lineage.
- Pivot 1: expected to recommend dynamic ratings; the misspecification
  paper (2502.10985) reversed that hypothesis — sparse-regime regret
  dominates, simple decayed BT is the right engine. Recommendation became
  "reparameterize the decay, keep the engine."
- Pivot 2: expected DFS literature to upgrade the optimizer; instead it
  validated the current exact-objective formulation and redirected effort
  to posterior-predictive optimization (a gap the literature doesn't even
  discuss because most settings can't simulate the joint).
- Dead ends: cieslak.dev blog 403; Sziklai PDF binary (recovered via
  abstract page); no CS-specific map-vs-series fitting gain measurement
  exists anywhere I could find; primary Bober-Irizar PDF not located in
  two attempts (numbers retained with MEDIUM confidence from agreeing
  secondary surveys).
- Surprise: Polymarket's bias literature points the opposite direction
  from bookmaker favorite-longshot bias (compression toward 50% /
  longshots resolving MORE often than priced in one study) — which kills
  "switch to Shin" as a meaningful improvement and elevates the empirical
  venue-calibration-curve approach.
- Community-tool survey confirmed positioning: nothing public combines
  market anchoring + exact pairing + joint-distribution optimization +
  honest intervals. SmallRob's exhaustive brute force was the one stealable
  validation idea.
- 2026-06-11 session start: read model.py, simulate.py, optimize.py,
  posterior.py, README, roadmap before any searching; baseline audit
  written first. ~16 searches, 9 page fetches total.
