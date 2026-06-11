# cs2-pickem

Market-calibrated Monte Carlo pick'em optimizer for Valve Major Swiss stages.
Built for IEM Cologne Major 2026 Stage 3 (June 11–15); the structure
generalizes to any 16-team Valve Swiss with fixed Round 1 pairings.

Pure stdlib Python, no dependencies.

## Usage

```bash
python src/fit.py        # fit ratings from data -> data/ratings_fitted.json
python src/optimize.py   # 40k Swiss sims -> per-team probs + optimal slate
```

## Methodology

Three calibration layers, in order of authority:

1. **Priors** (`model.PRIORS`): Elo-scale estimates seeded from the few
   market lines available pre-lock plus VRS position.
2. **Bradley-Terry fit** (`fit_bradley_terry`): MAP estimate on verified
   2026 series results (`data/matches_2026.csv`), weighted by recency
   (Cologne 1.0 → early season 0.5) and format (BO1 = 0.6× a BO3).
   Gaussian regularization (σ = 70 Elo) toward priors prevents
   sparse-sample blowups — e.g. a 3-0 Swiss run against a soft field
   moves a rating ~20 points, not 100.
3. **Market anchoring** (`apply_market_anchors`): pairwise probabilities
   forced to vig-free implied probs from liquid lines (bookmaker BO3
   lines, traded Polymarket prices). Markets see what historical fits
   can't (roster news, prep state); where a real line exists, it wins.
   Shifts propagate to all simulated matchups for the affected teams.

The simulator (`simulate.py`) plays the full Swiss endogenously: fixed
Round 1, then record groups with Buchholz recomputed per round, high-vs-low
pairing with rematch avoidance. Downstream path difficulty is therefore
integral to every probability, including the correlation structure (two R1
opponents can't both go 3-0; the loser is forced onto the 3-1/3-2 path).

The optimizer (`optimize.py`) maximizes **P(≥5 of 10 correct)** — the
pass threshold — evaluated empirically on stored sim outcomes, not on
independent per-team probabilities. This is what surfaces non-obvious
structure, e.g. placing *both* teams of a top-tier R1 clash in advance
slots (the loser, if they advance, must do it 3-1/3-2).

Note advance slots score **only** for 3-1/3-2 finishes; a 3-0 by an
advance pick scores zero. This is why the strongest non-lock teams often
belong in advance slots rather than 3-0 slots.

## Data provenance (as committed)

- `matches_2026.csv`: 87 verified series — complete Cologne Stage 2,
  complete IEM Rio 2026, plus confirmed results from IEM Atlanta,
  EPL S23, PGL Astana (full Stage-3-team coverage added in v2 refresh),
  CS Asia Championships, IEM Kraków, BLAST Bounty/Rotterdam/Spring,
  PGL Bucharest. Sourced from Liquipedia/HLTV/escharts coverage; v2
  additions cross-verified against 2+ independent pages (two series
  with contradictory sources were discarded rather than guessed).
- `market_anchors.json`: refreshed 2026-06-10 evening (v2). All 8 R1
  matches from Polymarket gamma API exact two-sided mids ($101K-$472K
  volume per market, 1-cent spreads); GGbet cross-checks agree within
  ~2pts. Stage 3 format verified via Liquipedia: **all matches BO3**
  (a first for Majors), so anchors are BO3 series probabilities — the
  scale ratings are calibrated to. The original 5-anchor set
  (2026-06-09/10, thinner books) is superseded.

## Final Stage 3 slate (v2 re-lock, 2026-06-10 evening)

- **3-0:** Vitality, Spirit
- **0-3:** B8, Monte
- **Advance:** NAVI, Falcons, FURIA, Aurora, MOUZ, G2
- Model P(≥5 correct) ≈ 0.43, E[ticks] ≈ 4.25
- Pipeline argmax, stable across sim seeds 7/11/42/123 (including the
  G2-vs-MongolZ last advance slot: G2 +0.003-0.006 P(≥5) on same sims).

Per-team probabilities as re-locked: `data/stage3_probs.json` (40k sims,
seed 11). The superseded v1 table is `data/stage3_probs_locked_v1.json`.

### v1 slate (original lock, 2026-06-10 afternoon — superseded)

- 3-0: Vitality, Falcons · 0-3: 9z, B8 ·
  Advance: Spirit, NAVI, FURIA, MOUZ, PARIVISION, Aurora
- Built on 5 anchors / 73 series; P(≥5) ≈ 0.39 under v1, ≈ 0.37 under v2.
- The v2 refresh repriced three unanchored R1 matches (Spirit 64.5% over
  NAVI vs fitted 51%; PARIVISION only 55.5% over 9z; FURIA 73.5% over B8)
  and filled dataset gaps (e.g. 9z's PGL Astana run). Original provenance
  note: v1's argmax preferred FURIA 3-0 over Falcons at Δ P(≥5) ≈ +0.002;
  the v1 lock was a manual tie-break toward E[ticks] (4.13 vs 4.10).

Log these against actuals: per-team Brier on (p30, padv, p03) for both
v1 and v2 (`src/postmortem.py` grades both) is the postmortem that
matters — including whether the refresh helped — not whether the slate
passed.

## Known limitations

- Greedy Buchholz pairing approximates Valve's exact seeding-difference
  algorithm (second-order for record-level probabilities).
- Scalar ratings assume transitivity — no map-pool intersection, veto
  modeling, or head-to-head style effects (e.g. donk vs NAVI).
- Static ratings within the stage; no round-to-round form updating.
- BO3s drawn as single Bernoulli events rather than map-level sequences.
- Roster changes are invisible to team-level fitting; only the market
  layer can price them. Known at v2 lock: Brollan's last event with MOUZ,
  karrigan reportedly starting for Falcons, BetBoom stand-in churn +
  visa uncertainty (likely why MGLZ-BB sits at a coin flip).
- Symmetric anchor propagation smears matchup-specific effects globally:
  the Spirit-NAVI line partly prices a Spirit-specific H2H edge (11-2 in
  the donk era), but the anchor moves NAVI -36 against *everyone*. The
  player-level extension is the real fix.

## Extension path (the real version)

- Ingest match/map/player data via the bo3.gg API into Postgres.
- Player-composition Bradley-Terry: team strength = f(five player form
  vectors) + team/IGL term, with market lines as the prior. Solves the
  roster-change blind spot and adds map-level granularity for veto
  modeling. Mind role confounds (entry vs AWP stat baselines).
- Re-fit between Swiss rounds: R2+ match markets go live during the
  stage, and mid-stage map data covers all 16 teams.
- Track calibration: log every published probability, Brier-score after
  each event.
