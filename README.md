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

- `matches_2026.csv`: 73 verified series — complete Cologne Stage 2,
  complete IEM Rio 2026, plus confirmed results from IEM Atlanta,
  EPL S23, PGL Astana, CS Asia Championships, BLAST Bounty/Rotterdam/
  Spring, PGL Bucharest. Sourced from Liquipedia/HLTV/escharts coverage.
- `market_anchors.json`: pre-lock lines, 2026-06-10. GGbet (de-vigged
  proportionally) + Polymarket traded mid prices. Polymarket markets with
  $0 volume were discarded as untraded market-maker ladders.

## Final Stage 3 slate (locked 2026-06-10)

- **3-0:** Vitality, Falcons
- **0-3:** 9z, B8
- **Advance:** Spirit, NAVI, FURIA, MOUZ, PARIVISION, Aurora
- Model P(≥5 correct) ≈ 0.39, E[ticks] ≈ 4.1

Provenance note: the committed pipeline's argmax swaps Falcons and FURIA
(FURIA 3-0, Falcons advance) at Δ P(≥5) ≈ +0.002 — inside Monte Carlo
noise, and the locked slate has the higher E[ticks] (4.13 vs 4.10).
The lock stands as a manual tie-break toward expected ticks. Per-team
probabilities as locked: `data/stage3_probs.json` (40k sims, seed 11).

Log these against actuals: per-team Brier score on (p30, padv, p03) is
the postmortem that matters, not whether the slate passed.

## Known limitations

- Greedy Buchholz pairing approximates Valve's exact seeding-difference
  algorithm (second-order for record-level probabilities).
- Scalar ratings assume transitivity — no map-pool intersection, veto
  modeling, or head-to-head style effects (e.g. donk vs NAVI).
- Static ratings within the stage; no round-to-round form updating.
- BO3s drawn as single Bernoulli events rather than map-level sequences.
- Roster changes (MOUZ's Brollan-for-jL, Major only) are invisible to
  team-level fitting; only the market layer can price them.

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
