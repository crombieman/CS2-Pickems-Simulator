"""Live mid-stage re-forecast + rooting guide.

Usage during the stage:
  1. Edit data/live_state.json as results come in:
       {
         "completed": [["Vitality", "FUT"], ...],   // winner FIRST
         "upcoming":  [["NAVI", "MOUZ"], ...]       // announced, unplayed
       }
     Always list announced pairings in "upcoming" — Valve's exact pairing
     algorithm can differ from our greedy Buchholz approximation, and the
     real bracket beats the simulated one. Mid-round states REQUIRE it.
  2. Optional: data/live_anchors.json (same shape as market_anchors.json)
     with market lines for upcoming matches — run fetch_anchors.py.
     These override ratings for those specific pairings.
  3. python src/live.py

Outputs: updated per-team (p30 / padv / p03), the locked v3 slate's live
P(>=5 correct) and E[ticks], and per upcoming match P(pass | A wins) vs
P(pass | B wins) — who to root for, quantified.

Ratings: data/ratings_fitted.json by default — the LIVING file, which
re-fits as the dataset grows (since 2026-06-12 that includes Stage 3
results, i.e. the rooting guide is mid-stage-refit). --locked runs on
the frozen v3 lock ratings instead (the pre-registered model's view).
The source + content hash go into every calibration-log entry, so model
breaks in the P(pass) time series are visible, never silent.

Every run APPENDS its published numbers to data/calibration_log.jsonl —
the forward-capture calibration log (README promise since v1): published
forecasts are unrecoverable later, exactly like odds.
"""

import argparse
import collections
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import simulate
from calibration import _git_sha, _sha, _src_dirty
from model import STAGE3_TEAMS
from simulate import make_state, simulate_stage

DATA = Path(__file__).resolve().parent.parent / "data"

N_SIMS = 40000
SEED = 11

# v3 final picks (must match postmortem.SLATES "FINAL PICKS" entry).
SLATE_30 = ["Vitality", "Spirit"]
SLATE_03 = ["B8", "Monte"]
SLATE_ADV = ["NAVI", "Falcons", "FURIA", "Aurora", "MOUZ", "MongolZ"]


def slate_ticks(result):
    return (sum(1 for t in SLATE_30 if result[t] == (3, 0))
            + sum(1 for t in SLATE_03 if result[t] == (0, 3))
            + sum(1 for t in SLATE_ADV if result[t] in ((3, 1), (3, 2))))


def run_from(ratings, state, n_sims=N_SIMS, seed=SEED):
    """Sims from a state. Returns (P(>=5), E[ticks], per-team stats)."""
    rng = random.Random(seed)
    records = {t: collections.Counter() for t in STAGE3_TEAMS}
    passes = total = 0
    for _ in range(n_sims):
        result = simulate_stage(ratings, rng, state)
        k = slate_ticks(result)
        passes += k >= 5
        total += k
        for t, rec in result.items():
            records[t][rec] += 1
    stats = {}
    for t in STAGE3_TEAMS:
        p30 = records[t][(3, 0)] / n_sims
        padv = (records[t][(3, 1)] + records[t][(3, 2)]) / n_sims
        stats[t] = {"p30": p30, "padv": padv, "pany": p30 + padv,
                    "p03": records[t][(0, 3)] / n_sims}
    return passes / n_sims, total / n_sims, stats


def build_log_entry(ts, ratings_source, ratings_sha, n_anchors, completed,
                    upcoming, p_pass, e_ticks, stats, rooting,
                    manifest=None, match_forecasts=None):
    """One calibration-log record: everything published by this run, as
    plain JSON. rooting rows arrive as (swing, a, b, p_pass_a, p_pass_b).

    manifest + match_forecasts (W2b) make a forward forecast replayable and
    thus adoption-eligible once graded (DoR §2.3). They default to None for
    backward compatibility; main() always supplies them on a live run."""
    return {
        "ts": ts,
        "ratings_source": ratings_source,
        "ratings_sha": ratings_sha,
        "n_anchors": n_anchors,
        "n_completed": len(completed),
        "upcoming": [list(m) for m in upcoming],
        "p_pass": p_pass,
        "e_ticks": e_ticks,
        "teams": stats,
        "rooting": [{"a": a, "b": b, "p_pass_a": pa, "p_pass_b": pb,
                     "swing": swing}
                    for swing, a, b, pa, pb in rooting],
        "manifest": manifest,
        "match_forecasts": match_forecasts,
    }


def per_match_forecasts(ratings, upcoming):
    """Immutable per-match model probs for the announced pairings — the
    strongest pre-registration of a forward forecast (replayable even if the
    input files later change). model_prob = what the model plays (the market
    override if the pair is priced, else the rating-implied prob); market_prob =
    the line used at lock, or None if unpriced."""
    rows = []
    for a, b in upcoming:
        override = simulate.PAIR_OVERRIDES.get((a, b))
        rows.append({"a": a, "b": b,
                     "model_prob": simulate.match_prob(ratings, a, b),
                     "market_prob": override})
    return rows


def forecast_manifest(ratings_source, ratings_sha, anchors_source, anchors_sha):
    """Replay manifest (DoR §2.3): names + hashes every input needed to
    reproduce the per-match probabilities, plus a code-dirty marker. The grader
    treats a row as adoption-eligible only if this is present and not dirty.
    event_config_sha is 'pending-w5' until the event-config refactor (W5) gives
    events a config hash."""
    return {
        "manifest_version": "v1",
        "code_sha": _git_sha(),
        "code_dirty": _src_dirty(),
        "ratings_source": ratings_source,
        "ratings_sha": ratings_sha,
        "anchors_source": anchors_source,
        "anchors_sha": anchors_sha,
        "pair_overrides_source": "market_anchors.json",
        "pair_overrides_sha": _sha(DATA / "market_anchors.json"),
        "event_config_sha": "pending-w5",
        "market_policy_version": "exact-override+ANCHOR_LAMBDA=0.5",
        "sim_policy_version": f"swiss-mc;priority_table={simulate.USE_PRIORITY_TABLE}",
        "n_sims": N_SIMS,
        "seed": SEED,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", action="store_true",
                    help="use frozen v3 lock ratings instead of the living fit")
    args = ap.parse_args()

    ratings_file = ("ratings_locked_v3.json" if args.locked
                    else "ratings_fitted.json")
    raw = (DATA / ratings_file).read_bytes()
    ratings_sha = hashlib.sha1(raw).hexdigest()[:12]
    ratings = json.loads(raw)
    print(f"Ratings: {ratings_file} ({ratings_sha})")
    live = json.load(open(DATA / "live_state.json"))
    completed = [tuple(m) for m in live.get("completed", [])]
    upcoming = [tuple(m) for m in live.get("upcoming", [])]

    # Optional mid-stage market lines for upcoming pairings.
    live_anchors_path = DATA / "live_anchors.json"
    n_loaded = 0
    if live_anchors_path.exists():
        for anc in json.load(open(live_anchors_path))["anchors"]:
            simulate.PAIR_OVERRIDES[(anc["a"], anc["b"])] = anc["p"]
            simulate.PAIR_OVERRIDES[(anc["b"], anc["a"])] = 1.0 - anc["p"]
            n_loaded += 1
        print(f"Loaded {n_loaded} live market anchors.\n")

    state = make_state(completed, upcoming)
    p5, ev, stats = run_from(ratings, state)

    done = {t for w, l in completed for t in (w, l)}
    print(f"After {len(completed)} completed matches "
          f"({len(upcoming)} announced upcoming):\n")
    print(f"{'Team':12s} {'rec':>5s} {'P(3-0)':>7s} {'P(adv)':>7s} {'P(0-3)':>7s}")
    for t in sorted(STAGE3_TEAMS, key=lambda t: -stats[t]["pany"]):
        w = state["wins"][t]
        l = state["losses"][t]
        s = stats[t]
        print(f"{t:12s} {w}-{l:>3d} {s['p30']:7.3f} {s['padv']:7.3f} {s['p03']:7.3f}")

    print(f"\nLocked v3 slate live:  P(>=5 correct) = {p5:.3f}   "
          f"E[ticks] = {ev:.2f}")

    rows = []
    if upcoming:
        print(f"\nRooting guide (locked slate, {N_SIMS} sims per branch):")
        print(f"{'Match':28s} {'P(pass|A)':>10s} {'P(pass|B)':>10s} {'root for':>12s} {'swing':>7s}")
        for i, (a, b) in enumerate(upcoming):
            rest = [m for j, m in enumerate(upcoming) if j != i]
            pa, _, _ = run_from(ratings, make_state(completed + [(a, b)], rest))
            pb, _, _ = run_from(ratings, make_state(completed + [(b, a)], rest))
            rows.append((abs(pa - pb), a, b, pa, pb))
        for swing, a, b, pa, pb in sorted(rows, reverse=True):
            fav = a if pa >= pb else b
            print(f"{a} vs {b:14s} {pa:10.3f} {pb:10.3f} {fav:>12s} {swing:7.3f}")

    anchors_source = "live_anchors.json" if live_anchors_path.exists() else "none"
    anchors_sha = _sha(live_anchors_path) if live_anchors_path.exists() else None
    entry = build_log_entry(
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ratings_source=ratings_file, ratings_sha=ratings_sha,
        n_anchors=n_loaded, completed=completed, upcoming=upcoming,
        p_pass=p5, e_ticks=ev, stats=stats, rooting=sorted(rows, reverse=True),
        manifest=forecast_manifest(ratings_file, ratings_sha, anchors_source,
                                   anchors_sha),
        match_forecasts=per_match_forecasts(ratings, upcoming))
    with open(DATA / "calibration_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nLogged -> data/calibration_log.jsonl")


if __name__ == "__main__":
    main()
