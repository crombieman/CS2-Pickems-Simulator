"""V1 validation harness - walk-forward core + self-validation (W6a).

The adoption gate's machinery (docs/plans/2026-07-05-w6-v1-harness-spec.md).
W6a ships: the consumer-view event universe + eligibility (spec 4.1), event
boundaries + dev/holdout split with loud straddler exclusion, per-event fit
windows + hop-bounded fit universes (4.4), engine fits through
model.fit_bradley_terry (explicit sigma, flat priors, recenter_on=universe),
the coverage rule (4.3), per-match paired grading rows via calibration's
scoring primitives (no second scoring dialect), the resolved-input fit cache
(4.5), event-blocked paired stats with t-quantile CIs (5.1), verdict
classification (5.2), and the DoR-6 synthetic self-test (6.1) whose known
answers gate every real run.

W6b adds: pre-replay gates, content-addressed run manifests + run dirs,
baselines, sweep families, holdout burn log, CLI. W6c adds: the 5(8)
objective check + the Cologne known-answer replay.

Determinism: no wall-clock in any output; every stochastic path takes an
explicit seed; fit inputs are built in sorted order so float accumulation is
reproducible; JSON float round-trips are exact, so a cache hit is
byte-equivalent to the fit it stored.
"""

import argparse
import hashlib
import json
import math
import random
import shutil
import sqlite3
from calendar import monthrange
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from calibration import (GRADE_VERSION, _brier, _git_sha, _logloss, _sha,
                         _src_dirty)
from event_config import EVENTS as EVENTS_DIR
from event_config import load_event
from integrity_audit import consumer_rows
from model import fit_bradley_terry, win_prob
from playoffs import paired_margin

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
HARNESS_DIR = DATA / "harness"
CACHE_DIR = HARNESS_DIR / "cache"          # derived, gitignored
CONFIGS_DIR = HARNESS_DIR / "configs"      # committed (spec 3.1)
RUNS_DIR = HARNESS_DIR / "runs"            # derived, gitignored (spec 3.3)
BURN_LOG = HARNESS_DIR / "holdout_burn_log.jsonl"   # committed (spec 5.3)

# Candidacy bookkeeping fields excluded from the config diff (they differ
# by construction and are not knob changes; spec 3.1/3.2.7).
IDENTITY_FIELDS = ("name", "knob_id", "expected_diff_paths", "sweep_family")

HARNESS_VERSION = "v0:event-scoped-walkforward+paired-events"
# Bump whenever fit-path code semantics change (the PARSE_VERSION idiom).
# Part of every fit-cache key, so stale fits are structurally unreachable.
# v0.1: convergence-stop added to the fit path (ship-gate oscillation fix).
FIT_CODE_VERSION = "fit-v0.1:bt-flat-uniform+converge-stop"


class HarnessError(RuntimeError):
    """Fail-loud: anything raised here must block a verdict, never degrade
    one (a better score from suspect data is not evidence, DoR 5(7))."""


# -- config identity (spec 3.1) --------------------------------------------------
def canonical_sha(obj):
    """sha256 hexdigest of the canonical JSON bytes (sorted keys, compact)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# Config identity = sha256 of canonical JSON (spec 3.1); one definition.
config_sha = canonical_sha


def load_config(path):
    with open(path) as f:
        return json.load(f)


# -- date arithmetic --------------------------------------------------------------
@lru_cache(maxsize=None)
def utc_key(iso):
    """Offset-proof temporal sort key: ISO-8601 (any offset; naive treated
    as UTC) -> normalized UTC 'YYYY-MM-DDTHH:MM:SS[.ffffff]'. EVERY temporal
    comparison in the harness goes through this - raw ISO strings order
    wrongly across offsets (Codex W6a review P1: '...T00:30:00+02:00' sorts
    after '...T23:00:00+00:00' lexicographically but is chronologically
    earlier). The archive is uniformly +00:00 today (verified 2026-07-05);
    this defends the contract against upstream serialization drift.
    Output compares correctly against date-only strings ('2025-07-01')."""
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.isoformat()


def months_before(day, months):
    """'YYYY-MM-DD' (or a full ISO timestamp) minus N calendar months, day
    clamped to the target month's length. Returns a date string that compares
    lexicographically against the substrate's full ISO timestamps (the
    integrity_audit CSV_SINCE idiom)."""
    y, m, d = int(day[0:4]), int(day[5:7]), int(day[8:10])
    total = (y * 12 + (m - 1)) - months
    y2, m2 = divmod(total, 12)
    m2 += 1
    d2 = min(d, monthrange(y2, m2)[1])
    return f"{y2:04d}-{m2:02d}-{d2:02d}"


# -- event universe + split (spec 4.1) ---------------------------------------------
def event_universe(rows, tiers=("s", "a"), min_matches=8):
    """Eligible events over consumer rows: group by tournament_id; event tier
    = MAX over non-NULL per-row tiers (the census + W4 cross-check rule);
    eligible iff tier in `tiers` AND consumer-match count >= min_matches.
    Boundary = MIN(start_date) over the event's consumer matches - derived
    from matches, never from the mutable tournaments.start_date (W3c);
    last_start = MAX(start_date) for the dual-end split rule. Exclusions are
    named + counted, never silent."""
    by_tid = {}
    null_tier = null_tid = 0
    for r in rows:
        if r["tournament_id"] is None:
            # never mint a synthetic None-event from orphan rows (Codex
            # W6a review P2); counted, not silent
            null_tid += 1
            continue
        by_tid.setdefault(r["tournament_id"], []).append(r)
        if r["tier"] is None:
            null_tier += 1
    events, excl_tier, excl_small = [], [], []
    for tid in sorted(by_tid):
        ev_rows = by_tid[tid]
        tier_vals = [r["tier"] for r in ev_rows if r["tier"] is not None]
        tier = max(tier_vals) if tier_vals else None
        if tier not in tiers:
            excl_tier.append(tid)
            continue
        if len(ev_rows) < min_matches:
            excl_small.append(tid)
            continue
        starts = [r["start_date"] for r in ev_rows]
        # min/max in UTC time, not string order; the RAW value is kept for
        # provenance (comparisons downstream re-normalize via utc_key)
        events.append({"tournament_id": tid, "tier": tier,
                       "n_matches": len(ev_rows),
                       "boundary": min(starts, key=utc_key),
                       "last_start": max(starts, key=utc_key)})
    report = {"null_tier_rows": null_tier, "null_tournament_rows": null_tid,
              "excluded_tier": excl_tier, "excluded_small": excl_small}
    return events, report


def classify_split(events, split_day):
    """Dual-end split classification (spec 4.1, review fix): dev iff the
    WHOLE event predates the split (MAX < split); holdout iff it starts
    at-or-after (MIN >= split); a straddler is excluded from BOTH, loudly -
    classifying by boundary alone would let its holdout-period outcomes
    inform dev-side selection."""
    out = {"dev": [], "holdout": [], "straddlers": []}
    for ev in events:
        if utc_key(ev["last_start"]) < split_day:
            out["dev"].append(ev)
        elif utc_key(ev["boundary"]) >= split_day:
            out["holdout"].append(ev)
        else:
            out["straddlers"].append(ev)
    return out


# -- fit universe (spec 4.4) ---------------------------------------------------------
def build_fit_universe(event_rows, window_rows, hops=1):
    """Fit universe = event participants + teams within `hops` match-graph
    steps of them inside the window (1-hop default; 2-hop exists for the W6a
    spot-check that validates the approximation). Fit matches = window rows
    with BOTH sides in the universe, in match_id order so the fit's float
    accumulation is deterministic. Keys are stringified team ids end-to-end
    (name-drift immunity, W3 spec 4)."""
    participants = ({str(r["team1_id"]) for r in event_rows}
                    | {str(r["team2_id"]) for r in event_rows})
    universe = set(participants)
    frontier = participants
    for _ in range(hops):
        nxt = set()
        for r in window_rows:
            a, b = str(r["team1_id"]), str(r["team2_id"])
            if a in frontier and b not in universe:
                nxt.add(b)
            if b in frontier and a not in universe:
                nxt.add(a)
        universe |= nxt
        frontier = nxt
    fit_matches, fit_ids, counts = [], [], {}
    team_tiers, team_last, fit_meta = {}, {}, []
    for r in sorted(window_rows, key=lambda x: x["match_id"]):
        a, b = str(r["team1_id"]), str(r["team2_id"])
        if a not in universe or b not in universe:
            continue
        w = str(r["winner_team_id"])
        if w not in (a, b):
            raise HarnessError(f"match {r['match_id']}: winner_team_id "
                               f"{r['winner_team_id']} is neither side")
        fit_matches.append((w, b if w == a else a, 1.0))
        fit_ids.append(r["match_id"])
        su = utc_key(r["start_date"])
        # per-match age/format inputs for W8's weighting knobs (F2/F3) and
        # per-team last activity for staleness sigma (F4) - all window-only
        fit_meta.append({"start_utc": su, "bo_type": r["bo_type"]})
        counts[a] = counts.get(a, 0) + 1
        counts[b] = counts.get(b, 0) + 1
        for t in (a, b):
            if t not in team_last or su > team_last[t]:
                team_last[t] = su
        if r["tier"] is not None:
            # per-team tier exposure (W7/F1 hierarchical priors), window-only
            for t in (a, b):
                team_tiers.setdefault(t, {})
                team_tiers[t][r["tier"]] = team_tiers[t].get(r["tier"], 0) + 1
    boundary_utc = min(utc_key(r["start_date"]) for r in event_rows)
    return {"universe": universe, "participants": participants,
            "fit_matches": fit_matches, "fit_match_ids": fit_ids,
            "fit_match_meta": fit_meta, "boundary_utc": boundary_utc,
            "window_counts": counts, "team_tiers": team_tiers,
            "team_last_start": team_last}


def assert_no_leakage(fit_rows, graded_rows, boundary):
    """Spec 3.2.4, asserted per event: nothing the fit sees may start
    at-or-after the event boundary; nothing graded may start before it.
    All comparisons in UTC (utc_key) - offset drift must not hide a leak."""
    b = utc_key(boundary)
    for r in fit_rows:
        if utc_key(r["start_date"]) >= b:
            raise HarnessError(
                f"temporal leakage: fit match {r['match_id']} starts "
                f"{r['start_date']} at-or-after boundary {boundary}")
    for r in graded_rows:
        if utc_key(r["start_date"]) < b:
            raise HarnessError(
                f"graded match {r['match_id']} starts {r['start_date']} "
                f"before boundary {boundary}")


# -- engine fit + cache (spec 4.4/4.5) --------------------------------------------------
# Modal-tier tie-break preference (higher tier wins a tie); unknown labels
# fall back to a deterministic lexicographic pick.
TIER_PREF = ("s", "a", "b", "c", "d")


def _modal_tier(counts):
    if not counts:
        return None
    best = max(counts.values())
    for t in TIER_PREF:
        if counts.get(t) == best:
            return t
    return sorted(k for k, v in counts.items() if v == best)[0]


def fit_engine(engine_cfg, universe_info):
    """One engine fit under a declared config. Supported priors schemes:
    'flat1000' (the v0 incumbent) and 'tier-empirical-2pass' (W7/F1
    candidate, default-off: pass-1 flat fit -> per-tier means of the fitted
    ratings, tier per team = modal window-tier exposure -> pass-2 refit
    with tier-mean priors; window-only, deterministic, no hand-set
    offsets). Any other declared scheme fails loud rather than silently
    approximating (candidate knobs add schemes here, THROUGH the gate).
    Sigma is passed explicitly for both buckets so the id-universe
    semantics are declared, not incidental (model.py's sigma bucketing
    references STAGE3_TEAMS)."""
    model_cfg = engine_cfg["model"]
    scheme = model_cfg["priors_scheme"]
    universe = sorted(universe_info["universe"])
    base = float(model_cfg["prior_mean"])
    sigma = float(model_cfg["sigma"])

    # -- data-prep weighting (W8/F2 half-life + F3 BO1 discount) --
    weighting = engine_cfg["data_prep"]["weighting"]
    if weighting == "uniform":
        matches = universe_info["fit_matches"]
    elif isinstance(weighting, dict) and weighting.get("scheme") == "weighted":
        hl = weighting.get("half_life_days")
        bo1 = float(weighting.get("bo1_discount", 1.0))
        b_dt = datetime.fromisoformat(universe_info["boundary_utc"])
        matches = []
        for (w, l, _wt), meta in zip(universe_info["fit_matches"],
                                     universe_info["fit_match_meta"]):
            wt = 1.0
            if hl is not None:
                age_days = ((b_dt - datetime.fromisoformat(meta["start_utc"]))
                            .total_seconds() / 86400.0)
                wt *= 0.5 ** (age_days / float(hl))
            if meta["bo_type"] == 1:
                wt *= bo1
            matches.append((w, l, wt))
    else:
        raise HarnessError(f"unknown weighting {weighting!r} "
                           f"(supported: uniform, "
                           f"{{scheme: weighted, ...}})")

    # -- staleness-inflated prior sigma (W8/F4, Glicko-flavored) --
    sigma_by_team = None
    staleness = model_cfg.get("staleness")
    if staleness is not None:
        rate = float(staleness["sigma_per_year"])
        b_dt = datetime.fromisoformat(universe_info["boundary_utc"])
        last = universe_info["team_last_start"]
        sigma_by_team = {}
        for t in universe:
            if t in last:   # no-window-match teams sit at their prior anyway
                years = ((b_dt - datetime.fromisoformat(last[t]))
                         .total_seconds() / (86400.0 * 365.0))
                sigma_by_team[t] = sigma + rate * years

    def fit(priors):
        return fit_bradley_terry(matches,
                                 priors=priors,
                                 sigma_s3=sigma, sigma_other=sigma,
                                 iters=model_cfg["iters"],
                                 lr=model_cfg["lr"],
                                 recenter_on=universe,
                                 converge_tol=model_cfg.get("converge_tol"),
                                 sigma_by_team=sigma_by_team)

    if scheme == "flat1000":
        return fit({t: base for t in universe})
    if scheme == "tier-empirical-2pass":
        flat = fit({t: base for t in universe})
        team_tiers = universe_info["team_tiers"]
        by_tier = {}
        for t in universe:
            mt = _modal_tier(team_tiers.get(t, {}))
            if mt is not None:
                by_tier.setdefault(mt, []).append(flat[t])
        tier_means = {k: sum(v) / len(v) for k, v in by_tier.items()}
        priors = {t: tier_means.get(_modal_tier(team_tiers.get(t, {})), base)
                  for t in universe}
        return fit(priors)
    raise HarnessError(f"unknown priors_scheme {scheme!r} "
                       f"(supported: flat1000, tier-empirical-2pass)")


def fit_cache_key(engine_config_sha, universe, fit_match_ids, window_spec,
                  substrate_id):
    """Resolved-input content addressing (spec 4.5, review fix): the key
    covers everything that changes a fit - engine config, fit-code version,
    the RESOLVED universe + match set (any upstream rule change alters these
    and misses the cache - stale reuse is structurally impossible), the
    window spec, and the substrate identity. `fit_match_ids` is hashed IN
    FIT ORDER (Codex W6a review P2): gradients accumulate floats in that
    order, so a reordering is a different fit and must miss the cache. The
    universe is a resolved SET and is canonicalized."""
    return canonical_sha({
        "fit_code_version": FIT_CODE_VERSION,
        "engine_config_sha": engine_config_sha,
        "universe": sorted(universe),
        "fit_match_ids": list(fit_match_ids),
        "window": list(window_spec),
        "substrate_id": substrate_id,
    })


def cached_fit(cache_dir, key, fit_fn):
    """Content-addressed fit cache in data/harness/cache/ (derived,
    gitignored). Incumbent fits across a knob sweep hit cache after the
    first run - sweeps are incremental by construction.

    Values are stored in a versioned envelope (Codex W6b review P1): a
    corrupt, legacy, or fit-code-stale file at the right key is a MISS
    (recomputed and overwritten), never served. The manifest additionally
    pins the sha of the ratings actually used, so a poisoned cache is
    diagnosable across runs even though it cannot be prevented here."""
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{key}.json"
    if path.exists():
        try:
            with open(path) as f:
                blob = json.load(f)
        except ValueError:
            blob = None
        if (isinstance(blob, dict)
                and blob.get("fit_code_version") == FIT_CODE_VERSION
                and isinstance(blob.get("ratings"), dict)):
            return blob["ratings"]
    ratings = fit_fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"fit_code_version": FIT_CODE_VERSION,
                   "ratings": ratings}, f)
    tmp.replace(path)
    return ratings


# -- substrate identity (spec 3.3) -----------------------------------------------------
META_IDENTITY_KEYS = ("parse_version", "archive_max_fetched_at",
                      "tournaments_snapshot_id", "audit_version")


def substrate_identity(db_path, con):
    """sha256 of the DB file bytes PLUS the parse_meta identity keys - the
    meta keys are what the sha MEANS; both recorded so drift is diagnosable.
    Missing keys stay None here; the W6b pre-replay gates enforce presence
    before any verdict exists."""
    h = hashlib.sha256()
    with open(db_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    meta = dict(con.execute("SELECT key, value FROM parse_meta"))
    ident = {"file_sha256": h.hexdigest()}
    for k in META_IDENTITY_KEYS:
        ident[k] = meta.get(k)
    return ident


# -- grading + coverage (spec 4.3 / 3.4) -------------------------------------------------
def grade_event_walkforward(event_rows, ratings_cand, ratings_inc,
                            window_counts, min_obs=3):
    """Per-match paired grading rows. A match is graded iff BOTH teams have
    >= min_obs window matches inside the fit universe; below the floor it is
    skipped-with-reason (a fallback probability would be a silent error, and
    the excluded-row count is itself a diagnostic). Probabilities are
    P(team1 wins); scoring reuses calibration's primitives."""
    graded, skipped = [], []
    for r in sorted(event_rows, key=lambda x: x["match_id"]):
        a, b = str(r["team1_id"]), str(r["team2_id"])
        thin = [t for t in (a, b) if window_counts.get(t, 0) < min_obs]
        if thin:
            skipped.append({"match_id": r["match_id"],
                            "reason": "below_min_obs:" + ",".join(thin)})
            continue
        w = str(r["winner_team_id"])
        if w not in (a, b):
            skipped.append({"match_id": r["match_id"],
                            "reason": f"winner_not_a_side:{w}"})
            continue
        y = 1.0 if w == a else 0.0
        p_c = win_prob(ratings_cand, a, b)
        p_i = win_prob(ratings_inc, a, b)
        graded.append({
            "kind": "walkforward",
            "tournament_id": r["tournament_id"], "match_id": r["match_id"],
            "team1_id": r["team1_id"], "team2_id": r["team2_id"],
            "bo_type": r["bo_type"], "result": y,
            "p_cand": p_c, "p_inc": p_i,
            "brier_cand": _brier(p_c, y), "brier_inc": _brier(p_i, y),
            "log_cand": _logloss(p_c, y), "log_inc": _logloss(p_i, y),
            "delta_brier": _brier(p_c, y) - _brier(p_i, y),
            "delta_log": _logloss(p_c, y) - _logloss(p_i, y),
        })
    return graded, skipped


def event_summary(tournament_id, graded, skipped, coverage_floor=0.5):
    """Per-event summary (spec 3.4). coverage < coverage_floor excludes the
    event from the paired stats - loudly (excluded=True, counted, named by
    the caller), never by silent absence."""
    n_g, n_s = len(graded), len(skipped)
    total = n_g + n_s
    coverage = (n_g / total) if total else 0.0
    reasons = {}
    for s in skipped:
        key = s["reason"].split(":", 1)[0]
        reasons[key] = reasons.get(key, 0) + 1
    out = {"tournament_id": tournament_id, "n_graded": n_g, "n_skipped": n_s,
           "coverage": coverage, "skip_reasons": reasons,
           "excluded": coverage < coverage_floor,
           "mean_delta_brier": None, "mean_delta_log": None}
    if n_g:
        out["mean_delta_brier"] = sum(g["delta_brier"] for g in graded) / n_g
        out["mean_delta_log"] = sum(g["delta_log"] for g in graded) / n_g
    return out


# -- t quantile (spec 5.1: stdlib, no scipy) ----------------------------------------------
def _betacf(a, b, x):
    """Continued fraction for the regularized incomplete beta (Lentz)."""
    maxit, eps, fpmin = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise HarnessError(f"betacf failed to converge (a={a}, b={b}, x={x})")


def _betai(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_cdf(t, df):
    x = df / (df + t * t)
    tail = 0.5 * _betai(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_quantile(p, df):
    """Student's t inverse CDF via bisection on the monotone CDF. Pure
    stdlib; deterministic; accurate to ~1e-9 - the CI multiplier for
    paired_event_stats at EVERY n (review fix: no normal-approx special
    case, no fake-significant z-CI at small n)."""
    if not (isinstance(df, int) and df >= 1):
        raise ValueError(f"df must be a positive int, got {df!r}")
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p!r}")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    hi = 1.0
    while _t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e12:
            raise HarnessError(f"t_quantile bracket blew up (p={p}, df={df})")
    lo = 0.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# -- paired stats + verdicts (spec 5) -------------------------------------------------------
def paired_event_stats(event_deltas, ci_level=0.95):
    """Event-blocked paired interval: the independent unit is the EVENT
    (matches within one share form/bracket/regime, DoR 9.1);
    playoffs.paired_margin supplies (mean, SE) verbatim; the CI uses the
    t quantile at df = n_events - 1."""
    n = len(event_deltas)
    if n < 2:
        raise HarnessError(f"paired stats need >= 2 events, got {n} "
                           f"(SE degenerates to 0.0 at n=1)")
    mean, se = paired_margin(list(event_deltas), [0.0] * n)
    t_crit = t_quantile(0.5 + ci_level / 2.0, n - 1)
    return {"n_events": n, "mean": mean, "se": se, "t_crit": t_crit,
            "ci": [mean - t_crit * se, mean + t_crit * se]}


def verdict(stats, *, mde, min_events, objective_ok=True, split="dev"):
    """Verdict classification (spec 5.2). Two-stage by design: on the dev
    split a pass is DEV-SCREENED (screening evidence only); the SAME test
    passed on the holdout split is ADOPTED - the only path to
    default-flipping. Negative delta = candidate better (Brier/log-loss).
    objective_ok=False marks a metric-screened knob proxy-only (DoR 5(8)):
    kept for research, barred from screening."""
    n, mean = stats["n_events"], stats["mean"]
    lo, hi = stats["ci"]
    if n < min_events:
        return {"verdict": "BLOCKED",
                "reason": f"insufficient-n: {n} events < min_events"
                          f" {min_events}", "stats": stats}
    if hi < 0 and abs(mean) >= mde:
        if not objective_ok:
            return {"verdict": "INCONCLUSIVE",
                    "reason": "metric-screened but objective check negative"
                              " -> proxy-only (DoR 5(8))", "stats": stats}
        name = "DEV-SCREENED" if split == "dev" else "ADOPTED"
        return {"verdict": name,
                "reason": f"CI [{lo:.6f}, {hi:.6f}] entirely < 0 and "
                          f"|mean| {abs(mean):.6f} >= MDE {mde}",
                "stats": stats}
    if lo > 0:
        return {"verdict": "REJECT",
                "reason": f"CI [{lo:.6f}, {hi:.6f}] entirely > 0: candidate"
                          " significantly worse", "stats": stats}
    return {"verdict": "INCONCLUSIVE",
            "reason": "CI crosses 0 or |mean| < MDE - incumbent stays;"
                      " explicitly NOT evidence of equivalence",
            "stats": stats}


def select_nominee(results):
    """Sweep-family nominee selection (spec 3.1/5.2): among DEV-SCREENED
    variants of ONE family, the best dev mean is the single nominee allowed
    to touch the holdout. Per-variant dev significance is screening
    evidence only - family size cannot inflate adoption false positives
    because adoption never happens on dev."""
    families = {r["sweep_family"] for r in results}
    if len(families) > 1:
        raise HarnessError(f"select_nominee: results span multiple sweep "
                           f"families {sorted(map(str, families))} - a "
                           f"nominee is per-family")
    screened = [r for r in results if r["verdict"] == "DEV-SCREENED"]
    if not screened:
        return None
    return min(screened, key=lambda r: r["stats"]["mean"])


# -- synthetic self-test (spec 6.1 / DoR 6) ----------------------------------------------------
def _synthetic_events(rng, n_events, n_matches):
    """Seeded synthetic truth: per match, a true win prob from a rating gap,
    an incumbent forecast = logit-noised truth, an outcome sampled from the
    truth."""
    events = []
    for _ in range(n_events):
        ev = []
        for _ in range(n_matches):
            gap = rng.uniform(-250.0, 250.0)
            p_true = 1.0 / (1.0 + 10 ** (-gap / 400.0))
            logit = math.log(p_true / (1.0 - p_true)) + rng.gauss(0.0, 0.5)
            p_inc = 1.0 / (1.0 + math.exp(-logit))
            y = 1.0 if rng.random() < p_true else 0.0
            ev.append((p_true, p_inc, y))
        events.append(ev)
    return events


def _case_deltas(events, weight):
    """Per-event mean Brier deltas for a candidate that mixes the incumbent
    toward the generating probability by `weight` (0 = exact clone;
    negative = anti-truth = strictly worse)."""
    deltas = []
    for ev in events:
        ds = []
        for p_true, p_inc, y in ev:
            p_cand = p_inc + weight * (p_true - p_inc)
            p_cand = min(max(p_cand, 1e-6), 1.0 - 1e-6)
            ds.append(_brier(p_cand, y) - _brier(p_inc, y))
        deltas.append(sum(ds) / len(ds))
    return deltas


SELF_TEST_EXPECTED = {"better": "DEV-SCREENED", "worse": "REJECT",
                      "clone": "INCONCLUSIVE"}


def run_self_test(seed, n_events=40, n_matches=20, mde=0.002, min_events=10,
                  ci_level=0.95, weights=(0.5, -0.5)):
    """Three constructed candidates with KNOWN answers: a truth-leaning
    mixture must DEV-SCREEN, an anti-truth mixture must REJECT, an exact
    clone must be INCONCLUSIVE with mean exactly 0. The objective arm is
    mocked green (W6c wires the real one). Any wrong verdict -> ok=False,
    and (W6b, spec 3.2.5) every real verdict that run is BLOCKED. `weights`
    is exposed so tests can prove ok=False is reachable - a self-test that
    cannot fail validates nothing."""
    rng = random.Random(seed)
    events = _synthetic_events(rng, n_events, n_matches)
    better_w, worse_w = weights
    cases = {"better": _case_deltas(events, better_w),
             "worse": _case_deltas(events, worse_w),
             "clone": _case_deltas(events, 0.0)}
    verdicts, stats = {}, {}
    for name, deltas in cases.items():
        s = paired_event_stats(deltas, ci_level=ci_level)
        verdicts[name] = verdict(s, mde=mde, min_events=min_events,
                                 objective_ok=True)["verdict"]
        stats[name] = s
    ok = all(verdicts[k] == SELF_TEST_EXPECTED[k] for k in SELF_TEST_EXPECTED)
    return {"ok": ok, "verdicts": verdicts, "expected": SELF_TEST_EXPECTED,
            "stats": stats, "seed": seed}


# -- pre-replay gates (spec 3.2, W6b) ------------------------------------------------------
def gate_parse_meta(meta):
    """Gates 3.2.1 + 3.2.2: the substrate must be reconciled (W3b), audited
    OK (W4), and the audit that blessed it must not have been vacuous
    (consumes the 74c50e6 guard). Raises with the offending stamp named."""
    if meta.get("reconciled") != "true":
        raise HarnessError(f"gate parse_meta.reconciled: "
                           f"{meta.get('reconciled')!r} != 'true'")
    if meta.get("audit_ok") != "true":
        raise HarnessError(f"gate parse_meta.audit_ok: "
                           f"{meta.get('audit_ok')!r} != 'true'")
    if not meta.get("audit_version"):
        raise HarnessError("gate parse_meta.audit_version: missing")
    raw = meta.get("audit_input_counts")
    if not raw:
        raise HarnessError("gate parse_meta.audit_input_counts: missing "
                           "(cannot prove the audit was non-vacuous)")
    try:
        counts = json.loads(raw)
    except ValueError as e:
        raise HarnessError(f"gate parse_meta.audit_input_counts: "
                           f"unparseable ({e})")
    if not (counts.get("reference_n", 0) > 0 and counts.get("csv_n", 0) > 0):
        raise HarnessError(f"gate parse_meta.audit_input_counts: vacuous "
                           f"audit inputs {counts} - audit_ok proves nothing")


def diff_paths(cand, inc, _prefix=""):
    """Dotted paths where two configs differ. Top-level candidacy
    bookkeeping (IDENTITY_FIELDS) is excluded - it differs by construction.
    Lists compare atomically (a changed grid IS the knob)."""
    paths = []
    for k in sorted(set(cand) | set(inc)):
        if _prefix == "" and k in IDENTITY_FIELDS:
            continue
        path = f"{_prefix}{k}"
        if k not in cand or k not in inc:
            paths.append(path)
        elif isinstance(cand[k], dict) and isinstance(inc[k], dict):
            paths.extend(diff_paths(cand[k], inc[k], _prefix=path + "."))
        elif cand[k] != inc[k]:
            paths.append(path)
    return paths


def gate_holdout_nominee(nominated_by, run_root, cand_sha, inc_sha,
                         harness_sha):
    """Gate for holdout runs (spec 5.3, Codex W6b review P1): the holdout
    CONFIRMS a nominee, it never screens. The run must reference a recorded
    dev run whose verdict is DEV-SCREENED for THIS candidate/incumbent pair
    under THIS harness config, and that dev run must itself have been
    adoption-eligible. Defends against workflow mistakes, not filesystem
    tampering (single-user local run store)."""
    if not nominated_by:
        raise HarnessError(
            "gate holdout-nominee: a holdout run must reference the "
            "DEV-SCREENED dev run that nominated this candidate "
            "(--nominee-from RUN_ID); the holdout confirms nominees, it "
            "does not screen")
    nom_dir = Path(run_root) / nominated_by
    try:
        nv = json.loads((nom_dir / "verdict.json").read_text())
        nm = json.loads((nom_dir / "manifest.json").read_text())
    except (OSError, ValueError) as e:
        raise HarnessError(f"gate holdout-nominee: cannot read nominee run "
                           f"{nominated_by}: {e}")
    for ok, why in (
            (nv.get("verdict") == "DEV-SCREENED",
             f"nominee verdict is {nv.get('verdict')!r}, not DEV-SCREENED"),
            (nm.get("split") == "dev", "nominee run was not a dev run"),
            (nm.get("candidate_config_sha") == cand_sha,
             "nominee candidate config differs from this candidate"),
            (nm.get("incumbent_config_sha") == inc_sha,
             "nominee incumbent config differs from this incumbent"),
            (nm.get("harness_config_sha") == harness_sha,
             "nominee ran under a different harness config (the holdout "
             "must apply the SAME pre-registered test)"),
            (nm.get("adoption_eligible") is True,
             "nominee dev run was exploratory-only, not adoption-eligible")):
        if not ok:
            raise HarnessError(
                f"gate holdout-nominee ({nominated_by}): {why}")


def gate_config_diff(cand_cfg, inc_cfg):
    """Gate 3.2.7 (review fix): the candidate must differ from its incumbent
    in the declared knob paths and NOTHING else. An accidental clone or an
    undeclared difference is a BLOCKED misconfiguration, never a quiet
    INCONCLUSIVE."""
    actual = diff_paths(cand_cfg, inc_cfg)
    if not actual:
        raise HarnessError(
            "gate config-diff: candidate is an incumbent clone (no "
            "non-identity differences) - nothing is under test")
    if not cand_cfg.get("knob_id"):
        raise HarnessError("gate config-diff: candidate declares no knob_id")
    declared = list(cand_cfg.get("expected_diff_paths") or [])
    if set(actual) != set(declared):
        raise HarnessError(
            f"gate config-diff: actual diff {sorted(actual)} != declared "
            f"expected_diff_paths {sorted(declared)} (knob "
            f"{cand_cfg.get('knob_id')!r})")


# -- replay stages (shared by walk_forward_replay and run_replay) ---------------------------
def _load_substrate(db_path):
    con = sqlite3.connect(db_path)
    try:
        rows = consumer_rows(con)
        substrate = substrate_identity(db_path, con)
        meta = dict(con.execute("SELECT key, value FROM parse_meta"))
        aliases = dict(con.execute(
            "SELECT canonical, team_id FROM canonical_alias"))
    finally:
        con.close()
    return rows, substrate, meta, aliases


def _run_self_test_gate(harness_cfg):
    v = harness_cfg["verdict"]
    self_test = run_self_test(harness_cfg["seeds"]["self_test"],
                              mde=v["mde_brier"],
                              min_events=v["min_events"],
                              ci_level=v["ci_level"])
    if not self_test["ok"]:
        raise HarnessError(
            f"self-test failed (spec 3.2.5): got {self_test['verdicts']}, "
            f"expected {self_test['expected']} - no real event may be "
            f"graded this run")
    return self_test


def _enumerate(rows, harness_cfg, split, limit):
    elig = harness_cfg["eligibility"]
    events, universe_report = event_universe(
        rows, tiers=tuple(elig["tiers"]),
        min_matches=elig["min_consumer_matches"])
    split_map = classify_split(events, harness_cfg["holdout_split"])
    chosen = split_map[split]
    if limit is not None:
        chosen = chosen[:limit]
    return chosen, split_map, universe_report, events


def _prep_events(rows, chosen, harness_cfg, universe_hops=None):
    """Window + leakage assertion + fit universe per chosen event. Pure
    enumeration - no fitting - so the manifest can pin fit-cache keys
    BEFORE any fit runs."""
    hops = (universe_hops if universe_hops is not None
            else {"1hop": 1, "2hop": 2}[harness_cfg["fit_universe"]["rule"]])
    rows_by_tid = {}
    for r in rows:
        rows_by_tid.setdefault(r["tournament_id"], []).append(r)
    rows_sorted = sorted(rows, key=lambda r: (utc_key(r["start_date"]),
                                              r["match_id"]))
    prepped = []
    for ev in chosen:
        ev_rows = rows_by_tid[ev["tournament_id"]]
        boundary = ev["boundary"]
        b_key = utc_key(boundary)
        w_start = months_before(b_key, harness_cfg["window_months"])
        win_rows = [r for r in rows_sorted
                    if w_start <= utc_key(r["start_date"]) < b_key]
        assert_no_leakage(win_rows, ev_rows, boundary)
        u = build_fit_universe(ev_rows, win_rows, hops=hops)
        prepped.append({"event": ev, "ev_rows": ev_rows, "universe": u,
                        "window_spec": (w_start, boundary)})
    return prepped


def _replay_prepped(prepped, cand_cfg, inc_cfg, harness_cfg, cache_dir,
                    substrate_sha):
    cand_sha, inc_sha = config_sha(cand_cfg), config_sha(inc_cfg)
    min_obs = harness_cfg["fit_universe"]["min_obs"]
    all_rows, summaries, cache_keys = [], [], {}
    for p in prepped:
        ev, u = p["event"], p["universe"]
        fits, keys = {}, {}
        for tag, cfg, sha in (("cand", cand_cfg, cand_sha),
                              ("inc", inc_cfg, inc_sha)):
            key = fit_cache_key(sha, u["universe"], u["fit_match_ids"],
                                p["window_spec"], substrate_sha)
            ratings = cached_fit(cache_dir, key,
                                 lambda c=cfg, ui=u: fit_engine(c, ui))
            fits[tag] = ratings
            # pin WHAT was served, not just the key (review P1: a poisoned
            # cache must be diagnosable from the manifest)
            keys[tag] = {"key": key, "ratings_sha": canonical_sha(ratings)}
        cache_keys[str(ev["tournament_id"])] = keys
        graded, skipped = grade_event_walkforward(
            p["ev_rows"], fits["cand"], fits["inc"], u["window_counts"],
            min_obs=min_obs)
        summary = event_summary(ev["tournament_id"], graded, skipped,
                                coverage_floor=harness_cfg["coverage_floor"])
        summary["boundary"] = ev["boundary"]
        summary["n_fit_matches"] = len(u["fit_match_ids"])
        summary["universe_size"] = len(u["universe"])
        summary["skipped"] = skipped
        summaries.append(summary)
        if not summary["excluded"]:
            all_rows.extend(graded)
    return summaries, all_rows, cache_keys


def _stats_from_summaries(summaries, harness_cfg):
    included = [s for s in summaries
                if not s["excluded"] and s["mean_delta_brier"] is not None]
    deltas = [s["mean_delta_brier"] for s in included]
    if len(deltas) < 2:
        return None, included
    return paired_event_stats(
        deltas, ci_level=harness_cfg["verdict"]["ci_level"]), included


# -- walk-forward orchestration (spec 4; in-memory/exploratory shape) -----------------------
def walk_forward_replay(db_path, harness_cfg, cand_cfg, inc_cfg, *,
                        split="dev", limit=None, cache_dir=CACHE_DIR,
                        universe_hops=None):
    """Replay candidate vs incumbent over the eligible events of one split.
    Returns in-memory results; run dirs, manifests, the full gate stack and
    verdict persistence are run_replay's layer on top. `limit` exists for
    the measured perf budget (spec 4.5) and the 1-vs-2-hop spot-check (4.4)
    - a limited run is exploratory only, never verdict-bearing.

    The 3.2.5 self-test gate is wired HERE too (Codex W6a review P1): a
    harness that misgrades synthetic known answers must refuse to grade
    anything real, even on exploratory runs."""
    self_test = _run_self_test_gate(harness_cfg)
    rows, substrate, _meta, _aliases = _load_substrate(db_path)
    chosen, split_map, universe_report, _events = _enumerate(
        rows, harness_cfg, split, limit)
    prepped = _prep_events(rows, chosen, harness_cfg,
                           universe_hops=universe_hops)
    summaries, all_rows, _keys = _replay_prepped(
        prepped, cand_cfg, inc_cfg, harness_cfg, cache_dir,
        canonical_sha(substrate))
    stats, _included = _stats_from_summaries(summaries, harness_cfg)
    return {"harness_version": HARNESS_VERSION, "split": split,
            "limited": limit is not None,
            "self_test_ok": True, "self_test_seed": self_test["seed"],
            "universe_report": universe_report,
            "n_events_chosen": len(chosen),
            "n_straddlers": len(split_map["straddlers"]),
            "straddlers": [e["tournament_id"]
                           for e in split_map["straddlers"]],
            "excluded_events": [s["tournament_id"] for s in summaries
                                if s["excluded"]],
            "events": summaries, "rows": all_rows, "stats": stats,
            "substrate": substrate}


# -- slate objective arm (spec 6.2 / DoR 5(8), W6c) -------------------------------------------
def slate_objective_check(cand_ratings, inc_ratings, eval_ratings, *,
                          n_sims, seed, ci_level=0.95, event_cfg=None,
                          k30=5, k03=5, kadv=9):
    """The proxy->objective check: does the candidate's belief change the
    SLATE, and does the changed slate hold up? Candidate and incumbent
    ratings each run sim -> optimizer; both chosen slates are then scored
    per-draw on ONE shared draw set from `eval_ratings` (the
    playoffs.paired_margin pattern at slate level). objective_ok iff the
    paired P(>=threshold) delta is non-negative or not significantly
    negative at ci_level.

    Eval-measure choice (v0, recorded in every verdict): real runs pass the
    INCUMBENT ratings - a conservative directional block against
    slate-harming knobs under the currently-trusted belief system; synthetic
    tests pass the generating TRUTH, which is what gives the known-answer
    cases their known answers. No realized outcomes are consumed, so the
    arm is split-safe. With n=1 slate event this blocks harm, it cannot
    prove slate improvement (honest limitation, spec 6.2).

    Market policy is 'none': PAIR_OVERRIDES is emptied for the duration
    (in-place, so every `from simulate import` binding sees it) and
    restored afterward. v0 supports only the currently-bound event config
    (Cologne) - the sim hardcodes the 16-team Swiss; multi-event rebind
    lands with the next EventConfig."""
    import optimize
    import simulate
    from event_config import COLOGNE
    cfg = event_cfg or COLOGNE
    if cfg.event_id != COLOGNE.event_id:
        raise HarnessError(
            f"slate arm v0 supports only the bound event "
            f"{COLOGNE.event_id!r}, got {cfg.event_id!r} (multi-event "
            f"rebind lands with the next EventConfig)")
    saved = dict(simulate.PAIR_OVERRIDES)
    simulate.PAIR_OVERRIDES.clear()
    try:
        sims_c, stats_c = simulate.run(cand_ratings, n_sims=n_sims, seed=seed)
        sims_i, stats_i = simulate.run(inc_ratings, n_sims=n_sims, seed=seed)
        best_c = optimize.optimize(sims_c, stats_c, k30=k30, k03=k03,
                                   kadv=kadv)
        best_i = optimize.optimize(sims_i, stats_i, k30=k30, k03=k03,
                                   kadv=kadv)
        if best_c is None or best_i is None:
            raise HarnessError(
                f"slate arm: optimizer found no feasible slate "
                f"(k30={k30}, k03={k03}, kadv={kadv}: advance pool "
                f"exhausted after 3-0/0-3 overlap - raise kadv)")
        p5c, _evc, c30c, c03c, advc = best_c
        p5i, _evi, c30i, c03i, advi = best_i
        if eval_ratings is inc_ratings:
            sims_e = sims_i
        elif eval_ratings is cand_ratings:
            sims_e = sims_c
        else:
            sims_e, _ = simulate.run(eval_ratings, n_sims=n_sims, seed=seed)
        thr = optimize.PASS_THRESHOLD
        hits_c = [1.0 if optimize.slate_correct(r, c30c, c03c, advc) >= thr
                  else 0.0 for r in sims_e]
        hits_i = [1.0 if optimize.slate_correct(r, c30i, c03i, advi) >= thr
                  else 0.0 for r in sims_e]
    finally:
        simulate.PAIR_OVERRIDES.update(saved)
    mean, se = paired_margin(hits_c, hits_i)
    t_crit = t_quantile(0.5 + ci_level / 2.0, len(sims_e) - 1)
    identical = (set(c30c) == set(c30i) and set(c03c) == set(c03i)
                 and set(advc) == set(advi))
    # Pass rule (recorded; spec 6.2 amended 2026-07-05 per W6c review P2):
    # the spec's "non-negative" is operationalized CI-aware because a strict
    # point-estimate rule false-flags a TRUE-zero slate delta ~50% of the
    # time under MC noise; identical slates still give exactly 0.
    pass_rule = ("mean>=0 or CI-upper>=0 at ci_level (operationalizes "
                 "'non-negative paired objective delta' under MC noise)")
    return {"objective_ok": mean >= 0 or (mean + t_crit * se) >= 0,
            "pass_rule": pass_rule,
            "mean": mean, "se": se,
            "ci": [mean - t_crit * se, mean + t_crit * se],
            "identical_slates": identical,
            "slate_cand": {"exact_3_0": sorted(c30c),
                           "exact_0_3": sorted(c03c),
                           "advance": sorted(advc), "p5_self": p5c},
            "slate_inc": {"exact_3_0": sorted(c30i),
                          "exact_0_3": sorted(c03i),
                          "advance": sorted(advi), "p5_self": p5i},
            "n_sims": n_sims, "seed": seed,
            "note": "n=1 slate event: directional block against "
                    "slate-harming knobs, not proof of slate improvement"}


def _guard_screening_needs_objective(verdict_name, objective_computed):
    """DoR 5(8): a screening-grade verdict without the objective arm is a
    hole in the gate, never a pass."""
    if verdict_name in ("DEV-SCREENED", "ADOPTED") and not objective_computed:
        raise HarnessError(
            f"{verdict_name} requires the slate objective check (DoR 5(8)) "
            f"but no slate-bearing event was computable this run")


def cologne_cross_check():
    """Real-data known-answer (spec 6.2): grade the locked Cologne snapshot
    through the harness's scoring path and require EXACT agreement with
    calibration.py's committed graded log (the already-verified grader).
    Read-only over committed artifacts; locked v3 ratings path."""
    from calibration import COLOGNE_SNAPSHOT, load_latest
    from model import load_pair_overrides
    snap = COLOGNE_SNAPSHOT
    matches = json.load(open(DATA / snap["matches_file"]))["matches"]
    ratings = json.load(open(DATA / snap["ratings_file"]))
    overrides = load_pair_overrides(DATA / snap["anchors_file"])
    committed = {(r["a"], r["b"]): r for r in load_latest()
                 if r["kind"] == "match" and r.get("event") == snap["event"]}
    mismatches = []
    for m in matches:
        a, b = m["a"], m["b"]
        y = 1.0 if m["winner"] == a else 0.0
        p = overrides.get((a, b))
        p = p if p is not None else win_prob(ratings, a, b)
        row = committed.get((a, b))
        if row is None:
            mismatches.append(f"{a} vs {b}: missing from graded log")
            continue
        for field, want in (("model_prob", p), ("result", y),
                            ("brier_model", _brier(p, y)),
                            ("log_model", _logloss(p, y))):
            if row[field] != want:
                mismatches.append(
                    f"{a} vs {b}: {field} {row[field]!r} != {want!r}")
    return {"n": len(matches), "n_committed": len(committed),
            "mismatches": mismatches, "ok": not mismatches}


# -- evidence table (spec 3.1/5.2: baselines are evidence, never gates) ----------------------
BASELINE_RULES = {"constant-0.5": lambda row: 0.5}


def evidence_table(graded_rows, baseline_cfgs):
    """Mean Brier/log for candidate, incumbent, and each declared simple
    baseline over the SAME graded rows (DoR 5(2)). Baselines appear in the
    verdict's evidence table; they never gate by themselves."""
    n = len(graded_rows)
    if n == 0:
        return {}

    def mean(vals):
        return sum(vals) / n

    out = {"candidate": {"brier": mean([r["brier_cand"] for r in graded_rows]),
                         "log": mean([r["log_cand"] for r in graded_rows])},
           "incumbent": {"brier": mean([r["brier_inc"] for r in graded_rows]),
                         "log": mean([r["log_inc"] for r in graded_rows])},
           "baselines": {}}
    for cfg in baseline_cfgs:
        rule = BASELINE_RULES.get(cfg.get("rule"))
        if rule is None:
            raise HarnessError(f"unknown baseline rule {cfg.get('rule')!r} "
                               f"(known: {sorted(BASELINE_RULES)})")
        briers, logs = [], []
        for r in graded_rows:
            p = rule(r)
            briers.append(_brier(p, r["result"]))
            logs.append(_logloss(p, r["result"]))
        out["baselines"][cfg["name"]] = {"brier": mean(briers),
                                         "log": mean(logs)}
    return out


# -- slate-arm wiring for gated runs (spec 6.2, W6c) -------------------------------------------
def _slate_event_configs():
    """Slate-bearing events = event configs that pin a substrate
    tournament_id (v0: Cologne). Discovered from data/events/*.json."""
    out = {}
    for p in sorted(EVENTS_DIR.glob("*.json")):
        cfg = load_event(p)
        if cfg.tournament_id is not None:
            out[cfg.tournament_id] = (cfg, p)
    return out


def _slate_arm(rows, all_events, aliases, harness_cfg, cand_cfg, inc_cfg,
               cache_dir, substrate_sha, ci_level):
    """Run the spec-6.2 objective arm on every slate-bearing event present
    in the substrate: fit both configs on the event's own walk-forward
    window (cached), map id-keyed ratings to model names via
    canonical_alias, replay the slate. Returns (objective_check,
    objective_replay_inputs, objective_ok); objective_ok is None when no
    slate event was computable - the screening guard then refuses any
    screening-grade verdict."""
    events_by_tid = {e["tournament_id"]: e for e in all_events}
    arm_events, inputs_events, computed_oks = {}, {}, []
    cand_sha, inc_sha = config_sha(cand_cfg), config_sha(inc_cfg)
    for tid, (ev_cfg, cfg_path) in _slate_event_configs().items():
        key = str(tid)
        ev = events_by_tid.get(tid)
        if ev is None:
            arm_events[key] = {"status": "not-in-substrate",
                               "event_id": ev_cfg.event_id}
            continue
        prep = _prep_events(rows, [ev], harness_cfg)[0]
        u = prep["universe"]
        fits, fit_keys = {}, {}
        for tag, cfg, sha in (("cand", cand_cfg, cand_sha),
                              ("inc", inc_cfg, inc_sha)):
            k = fit_cache_key(sha, u["universe"], u["fit_match_ids"],
                              prep["window_spec"], substrate_sha)
            ratings = cached_fit(cache_dir, k,
                                 lambda c=cfg, ui=u: fit_engine(c, ui))
            fits[tag] = ratings
            fit_keys[tag] = {"key": k, "ratings_sha": canonical_sha(ratings)}
        missing = [t for t in ev_cfg.teams
                   if t not in aliases or str(aliases[t]) not in fits["cand"]]
        if missing:
            raise HarnessError(
                f"slate arm: event {ev_cfg.event_id} teams missing from the "
                f"alias map or fit universe: {missing} - the arm must never "
                f"run on a partial slate")
        by_name = {tag: {t: fits[tag][str(aliases[t])]
                         for t in ev_cfg.teams} for tag in ("cand", "inc")}
        res = slate_objective_check(
            by_name["cand"], by_name["inc"], by_name["inc"],
            n_sims=harness_cfg["mc"]["slate_sims"],
            seed=harness_cfg["seeds"]["slate_sim"],
            ci_level=ci_level, event_cfg=ev_cfg)
        res["status"] = "computed"
        res["eval_measure"] = "incumbent"
        res["event_id"] = ev_cfg.event_id
        arm_events[key] = res
        computed_oks.append(res["objective_ok"])
        inputs_events[key] = {
            "event_config": {"path": Path(cfg_path).name,
                             "sha": _sha(cfg_path)},
            "fit_cache": fit_keys,
            "optimizer_objective": ev_cfg.optimizer_objective}
    if computed_oks:
        objective_ok = all(computed_oks)
        check_status = "ok" if objective_ok else "negative"
        src_dir = Path(__file__).resolve().parent
        # Spec 3.3 objective_replay_inputs, honestly populated (review P2):
        # everything the arm CONSUMED gets a sha; every spec-listed input
        # the v0 belief-based arm does NOT consume is named with the reason
        # - never silently partial.
        inputs = {"events": inputs_events,
                  "seed": harness_cfg["seeds"]["slate_sim"],
                  "n_sims": harness_cfg["mc"]["slate_sims"],
                  "eval_measure": "incumbent",
                  "market_policy": "none (pure-model arm)",
                  "sim_code_sha": _sha(src_dir / "simulate.py"),
                  "optimize_code_sha": _sha(src_dir / "optimize.py"),
                  "ratings_source": "walk-forward fits (fit_cache above; "
                                    "no locked/anchored ratings consumed)",
                  "anchors_files": "none-consumed (market policy 'none': "
                                   "PAIR_OVERRIDES emptied for the arm)",
                  "results_files": "none-consumed (belief-based arm: no "
                                   "realized outcomes touch the check)",
                  "graded_log": "none-consumed here (the Cologne "
                                "known-answer cross-check runs at test "
                                "level, not inside replay runs)"}
    else:
        objective_ok, check_status, inputs = None, "no-slate-events-computed", None
    check = {"status": check_status, "events": arm_events,
             "note": "n=1 slate event: directional block against "
                     "slate-harming knobs, not proof of slate improvement"}
    return check, inputs, objective_ok


# -- persisted gated runs (spec 3.2/3.3/3.4/5, W6b) ------------------------------------------
def _write_json(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def _has_placeholder(obj):
    """True if any string value anywhere is an unresolved provenance
    sentinel ('unknown' or 'pending-*') - the calibration adoption rule."""
    if isinstance(obj, dict):
        return any(_has_placeholder(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_placeholder(v) for v in obj)
    return isinstance(obj, str) and (obj == "unknown"
                                     or obj.startswith("pending"))


def run_replay(db_path, candidate_path, *, incumbent_path=None,
               harness_path=None, baseline_paths=None, split="dev",
               limit=None, nominated_by=None, run_root=None,
               cache_dir=CACHE_DIR, burn_log=BURN_LOG):
    """THE gated adoption-gate run (spec 3.2/3.3/3.4/5): all pre-replay
    gates, a content-addressed manifest + run dir with every config copied
    verbatim, per-match rows / per-event summaries / failure report, and
    the two-stage verdict. Any gate failure -> verdict BLOCKED with NO
    metrics (a better score from suspect data is not evidence, DoR 5(7)).
    A dirty code tree still runs but the whole run is exploratory-only
    (adoption_eligible=False), mirroring calibration's is_manifested rule.

    Every holdout invocation that actually graded events appends to the
    committed burn log (spec 5.3) - a burned holdout is logged as burned
    regardless of outcome."""
    if split == "holdout" and limit is not None:
        raise HarnessError(
            "a holdout run cannot be limited: a partial touch still burns "
            "the reserve while proving nothing (spec 5.3)")
    candidate_path = Path(candidate_path)
    incumbent_path = Path(incumbent_path or CONFIGS_DIR / "incumbent_v0.json")
    harness_path = Path(harness_path or CONFIGS_DIR / "harness_v0.json")
    baseline_paths = [Path(p) for p in
                      (baseline_paths if baseline_paths is not None
                       else [CONFIGS_DIR / "baseline_uniform.json"])]
    run_root = Path(run_root or RUNS_DIR)
    harness_cfg = load_config(harness_path)
    cand_cfg = load_config(candidate_path)
    inc_cfg = load_config(incumbent_path)
    baseline_cfgs = [load_config(p) for p in baseline_paths]
    v_cfg = harness_cfg["verdict"]

    blocked_gates = []
    if split == "holdout":
        try:
            gate_holdout_nominee(nominated_by, run_root,
                                 config_sha(cand_cfg), config_sha(inc_cfg),
                                 config_sha(harness_cfg))
        except HarnessError as e:
            blocked_gates.append(str(e))
    self_test_ok, self_test_seed = False, harness_cfg["seeds"]["self_test"]
    try:
        self_test = _run_self_test_gate(harness_cfg)
        self_test_ok, self_test_seed = True, self_test["seed"]
    except HarnessError as e:
        blocked_gates.append(str(e))
    try:
        gate_config_diff(cand_cfg, inc_cfg)
    except HarnessError as e:
        blocked_gates.append(str(e))
    rows, substrate, meta, aliases = _load_substrate(db_path)
    try:
        gate_parse_meta(meta)
    except HarnessError as e:
        blocked_gates.append(str(e))

    chosen, split_map, universe_report, all_events = _enumerate(
        rows, harness_cfg, split, limit)
    if len(chosen) < v_cfg["min_events"]:
        blocked_gates.append(
            f"gate min_events: insufficient-n: {len(chosen)} events "
            f"chosen < min_events {v_cfg['min_events']}")

    substrate_sha = canonical_sha(substrate)
    failures = [{"kind": "straddler_excluded", "tournament_id": tid}
                for tid in (e["tournament_id"]
                            for e in split_map["straddlers"])]
    failures.append({"kind": "universe_report", **universe_report})

    summaries, all_rows, cache_keys, stats = [], [], {}, None
    verdict_obj = None
    objective_check = {"status": "not-run", "events": {}}
    objective_inputs = None
    if not blocked_gates:
        prepped = _prep_events(rows, chosen, harness_cfg)
        summaries, all_rows, cache_keys = _replay_prepped(
            prepped, cand_cfg, inc_cfg, harness_cfg, cache_dir,
            substrate_sha)
        for s in summaries:
            for sk in s["skipped"]:
                failures.append({"kind": "skipped_row",
                                 "tournament_id": s["tournament_id"], **sk})
            if s["excluded"]:
                failures.append({"kind": "excluded_event_coverage",
                                 "tournament_id": s["tournament_id"],
                                 "coverage": s["coverage"]})
        stats, included = _stats_from_summaries(summaries, harness_cfg)
        if stats is None or stats["n_events"] < v_cfg["min_events"]:
            # blocked runs never fit/simulate the objective arm (review P2)
            n = 0 if stats is None else stats["n_events"]
            blocked_gates.append(
                f"gate min_events: insufficient-n: {n} included events "
                f"< min_events {v_cfg['min_events']} (post-coverage)")
        else:
            objective_check, objective_inputs, objective_ok = _slate_arm(
                rows, all_events, aliases, harness_cfg, cand_cfg, inc_cfg,
                cache_dir, substrate_sha, v_cfg["ci_level"])
            verdict_obj = verdict(stats, mde=v_cfg["mde_brier"],
                                  min_events=v_cfg["min_events"],
                                  objective_ok=(objective_ok
                                                if objective_ok is not None
                                                else True),
                                  split=split)
            try:
                # DoR 5(8): screening without the arm is a BLOCKED run with
                # full artifacts, never an unrecorded exception (review P1)
                _guard_screening_needs_objective(verdict_obj["verdict"],
                                                 objective_ok is not None)
            except HarnessError as e:
                blocked_gates.append(str(e))
                verdict_obj = None

    code_dirty = _src_dirty()
    manifest = {
        "harness_version": HARNESS_VERSION,
        "grade_version": GRADE_VERSION,
        "fit_code_version": FIT_CODE_VERSION,
        "git_sha": _git_sha(), "code_dirty": code_dirty,
        "substrate": substrate, "substrate_sha": substrate_sha,
        "harness_config_sha": config_sha(harness_cfg),
        "candidate_config_sha": config_sha(cand_cfg),
        "incumbent_config_sha": config_sha(inc_cfg),
        "baseline_config_shas": {c["name"]: config_sha(c)
                                 for c in baseline_cfgs},
        "sweep_family": cand_cfg.get("sweep_family"),
        "split": split, "holdout_split": harness_cfg["holdout_split"],
        "holdout_touched": split == "holdout",
        "event_set": [[e["tournament_id"], e["boundary"]] for e in chosen],
        "event_set_sha": canonical_sha(
            [[e["tournament_id"], e["boundary"]] for e in chosen]),
        "straddlers": [e["tournament_id"] for e in split_map["straddlers"]],
        "limited": limit is not None,
        "seeds": harness_cfg["seeds"], "mc": harness_cfg.get("mc", {}),
        "self_test_ok": self_test_ok, "self_test_seed": self_test_seed,
        "fit_cache_keys": cache_keys,
        "nominated_by_run_id": nominated_by,
        "blocked_gates": blocked_gates,
    }
    if objective_inputs is not None:
        # present iff the slate arm computed (spec 3.3: absent for runs
        # without a slate arm - never silently partial)
        manifest["objective_replay_inputs"] = objective_inputs
    # exploratory-only unless: clean tree, all gates green, UNLIMITED run
    # (review P1: a subset must never be adoption-grade), and no
    # placeholder hash anywhere - INCLUDING inside the configs themselves
    # (review P2: placeholders must not hide behind the config shas).
    manifest["adoption_eligible"] = (
        code_dirty is False and not blocked_gates and limit is None
        and not _has_placeholder(manifest)
        and not _has_placeholder({"candidate": cand_cfg,
                                  "incumbent": inc_cfg,
                                  "harness": harness_cfg,
                                  "baselines": baseline_cfgs}))
    run_id = canonical_sha(manifest)[:16]

    if blocked_gates:
        verdict_out = {"verdict": "BLOCKED",
                       "reason": "; ".join(blocked_gates),
                       "blocked_gates": blocked_gates}
    else:
        verdict_out = {"verdict": verdict_obj["verdict"],
                       "reason": verdict_obj["reason"],
                       "stats": stats, "blocked_gates": [],
                       "n_events": stats["n_events"],
                       "n_matches": len(all_rows),
                       "coverage_mean": (sum(s["coverage"] for s in included)
                                         / len(included)),
                       "mde": v_cfg["mde_brier"],
                       "ci_level": v_cfg["ci_level"],
                       "evidence": evidence_table(all_rows, baseline_cfgs)}
        if limit is not None:
            # review P1: a limited run is exploratory only - it keeps its
            # evidence but never bears a screening-grade verdict name
            verdict_out["verdict"] = "EXPLORATORY"
            verdict_out["reason"] = (
                f"limited run (--limit {limit}): would classify as "
                f"{verdict_obj['verdict']} - exploratory only, never "
                f"verdict-bearing (spec 4.5)")
    verdict_out.update({"run_id": run_id, "split": split,
                        "sweep_family": cand_cfg.get("sweep_family"),
                        "objective_check": objective_check,
                        "adoption_eligible": manifest["adoption_eligible"]})

    run_dir = run_root / run_id
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "verdict.json", verdict_out)
    events_out = [{k: v for k, v in s.items() if k != "skipped"}
                  for s in summaries]
    _write_jsonl(run_dir / "events.jsonl", events_out)
    _write_jsonl(run_dir / "rows.jsonl", all_rows)
    _write_jsonl(run_dir / "failures.jsonl", failures)
    for p in [harness_path, candidate_path, incumbent_path] + baseline_paths:
        shutil.copyfile(p, run_dir / "configs" / Path(p).name)

    if split == "holdout" and all_rows:
        # outcomes were actually GRADED -> the holdout is burned, whatever
        # the result (review P2: fits alone consume no holdout outcomes;
        # only graded rows do)
        entry = {"date": datetime.now(timezone.utc).date().isoformat(),
                 "run_id": run_id,
                 "candidate_config_sha": manifest["candidate_config_sha"],
                 "incumbent_config_sha": manifest["incumbent_config_sha"],
                 "sweep_family": cand_cfg.get("sweep_family"),
                 "nominated_by_run_id": nominated_by,
                 "event_set_sha": manifest["event_set_sha"],
                 "n_events": len([s for s in summaries if s["n_graded"]]),
                 "n_matches": len(all_rows),
                 "result": verdict_out["verdict"]}
        with open(burn_log, "a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    return {"run_id": run_id, "run_dir": str(run_dir),
            "verdict": verdict_out, "manifest": manifest}


# -- CLI (spec W6b: manual invocation, no cron) -----------------------------------------------
def main():
    from bo3gg_parse import DB_PATH
    ap = argparse.ArgumentParser(
        description="V1 validation harness - the adoption gate. Manual "
                    "runs only (prefers-manual-over-automation).")
    ap.add_argument("--replay", metavar="CANDIDATE_JSON",
                    help="candidate engine config to replay vs the incumbent")
    ap.add_argument("--holdout", action="store_true",
                    help="run the HOLDOUT split (burns it - logged); "
                         "requires --nominee-from")
    ap.add_argument("--nominee-from", metavar="RUN_ID", default=None,
                    help="run_id of the DEV-SCREENED dev run that "
                         "nominated this candidate (holdout runs only)")
    ap.add_argument("--incumbent", default=None,
                    help="incumbent config (default: incumbent_v0.json)")
    ap.add_argument("--harness-config", default=None,
                    help="harness config (default: harness_v0.json)")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--limit", type=int, default=None,
                    help="exploratory: first N events only")
    ap.add_argument("--self-test", action="store_true",
                    help="run only the synthetic self-test and exit")
    args = ap.parse_args()

    if args.self_test:
        cfg = load_config(CONFIGS_DIR / "harness_v0.json")
        st = run_self_test(cfg["seeds"]["self_test"])
        print(f"self-test ok={st['ok']} verdicts={st['verdicts']}")
        raise SystemExit(0 if st["ok"] else 1)
    if not args.replay:
        ap.error("--replay CANDIDATE_JSON required (or --self-test)")
    if args.holdout and args.limit is not None:
        ap.error("--holdout cannot be combined with --limit (a partial "
                 "touch burns the reserve while proving nothing)")

    out = run_replay(args.db, args.replay,
                     incumbent_path=args.incumbent,
                     harness_path=args.harness_config,
                     split="holdout" if args.holdout else "dev",
                     limit=args.limit,
                     nominated_by=args.nominee_from)
    v = out["verdict"]
    print(f"run {out['run_id']}  split={v['split']}  -> {out['run_dir']}")
    print(f"VERDICT: {v['verdict']}")
    print(f"  reason: {v['reason']}")
    if "stats" in v:
        s = v["stats"]
        print(f"  n_events={v['n_events']} n_matches={v['n_matches']} "
              f"coverage={v['coverage_mean']:.3f}")
        print(f"  paired delta-Brier mean {s['mean']:+.6f}  "
              f"CI [{s['ci'][0]:+.6f}, {s['ci'][1]:+.6f}]  "
              f"(MDE {v['mde']}, t={s['t_crit']:.3f})")
        for name, e in ([("candidate", v["evidence"]["candidate"]),
                         ("incumbent", v["evidence"]["incumbent"])]
                        + sorted(v["evidence"]["baselines"].items())):
            print(f"  {name:12s} brier {e['brier']:.4f}  log {e['log']:.4f}")
    oc = v["objective_check"]
    print(f"  sweep_family={v['sweep_family']}  "
          f"objective_check={oc['status']}  "
          f"adoption_eligible={v['adoption_eligible']}")
    for tid, e in sorted(oc["events"].items()):
        if e.get("status") == "computed":
            same = "identical slates" if e["identical_slates"] else \
                f"paired dP(>=thr) {e['mean']:+.4f} CI [{e['ci'][0]:+.4f}, " \
                f"{e['ci'][1]:+.4f}]"
            print(f"  slate arm {e['event_id']}: {same}  "
                  f"ok={e['objective_ok']}  ({oc['note']})")
    raise SystemExit(1 if v["verdict"] == "BLOCKED" else 0)


if __name__ == "__main__":
    main()
