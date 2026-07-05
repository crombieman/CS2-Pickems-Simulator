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

import hashlib
import json
import math
import random
import sqlite3
from calendar import monthrange
from pathlib import Path

from calibration import _brier, _logloss
from integrity_audit import consumer_rows
from model import fit_bradley_terry, win_prob
from playoffs import paired_margin

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
HARNESS_DIR = DATA / "harness"
CACHE_DIR = HARNESS_DIR / "cache"          # derived, gitignored
CONFIGS_DIR = HARNESS_DIR / "configs"      # committed (spec 3.1)

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
    null_tier = 0
    for r in rows:
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
        events.append({"tournament_id": tid, "tier": tier,
                       "n_matches": len(ev_rows),
                       "boundary": min(starts), "last_start": max(starts)})
    report = {"null_tier_rows": null_tier, "excluded_tier": excl_tier,
              "excluded_small": excl_small}
    return events, report


def classify_split(events, split_day):
    """Dual-end split classification (spec 4.1, review fix): dev iff the
    WHOLE event predates the split (MAX < split); holdout iff it starts
    at-or-after (MIN >= split); a straddler is excluded from BOTH, loudly -
    classifying by boundary alone would let its holdout-period outcomes
    inform dev-side selection."""
    out = {"dev": [], "holdout": [], "straddlers": []}
    for ev in events:
        if ev["last_start"] < split_day:
            out["dev"].append(ev)
        elif ev["boundary"] >= split_day:
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
        counts[a] = counts.get(a, 0) + 1
        counts[b] = counts.get(b, 0) + 1
    return {"universe": universe, "participants": participants,
            "fit_matches": fit_matches, "fit_match_ids": fit_ids,
            "window_counts": counts}


def assert_no_leakage(fit_rows, graded_rows, boundary):
    """Spec 3.2.4, asserted per event: nothing the fit sees may start
    at-or-after the event boundary; nothing graded may start before it."""
    for r in fit_rows:
        if r["start_date"] >= boundary:
            raise HarnessError(
                f"temporal leakage: fit match {r['match_id']} starts "
                f"{r['start_date']} at-or-after boundary {boundary}")
    for r in graded_rows:
        if r["start_date"] < boundary:
            raise HarnessError(
                f"graded match {r['match_id']} starts {r['start_date']} "
                f"before boundary {boundary}")


# -- engine fit + cache (spec 4.4/4.5) --------------------------------------------------
def fit_engine(engine_cfg, universe_info):
    """One engine fit under a declared config. v0 supports exactly
    flat-prior + uniform-weight Bradley-Terry; any other declared scheme
    fails loud rather than silently approximating (candidate knobs add
    schemes here, THROUGH the gate). Sigma is passed explicitly for both
    buckets so the id-universe semantics are declared, not incidental
    (model.py's sigma bucketing references STAGE3_TEAMS)."""
    model_cfg = engine_cfg["model"]
    if model_cfg["priors_scheme"] != "flat1000":
        raise HarnessError(f"unknown priors_scheme "
                           f"{model_cfg['priors_scheme']!r} (v0: flat1000)")
    if engine_cfg["data_prep"]["weighting"] != "uniform":
        raise HarnessError(f"unknown weighting "
                           f"{engine_cfg['data_prep']['weighting']!r} "
                           f"(v0: uniform)")
    universe = sorted(universe_info["universe"])
    priors = {t: float(model_cfg["prior_mean"]) for t in universe}
    sigma = float(model_cfg["sigma"])
    return fit_bradley_terry(universe_info["fit_matches"], priors=priors,
                             sigma_s3=sigma, sigma_other=sigma,
                             iters=model_cfg["iters"], lr=model_cfg["lr"],
                             recenter_on=universe,
                             converge_tol=model_cfg.get("converge_tol"))


def fit_cache_key(engine_config_sha, universe, fit_match_ids, window_spec,
                  substrate_id):
    """Resolved-input content addressing (spec 4.5, review fix): the key
    covers everything that changes a fit - engine config, fit-code version,
    the RESOLVED universe + match set (any upstream rule change alters these
    and misses the cache - stale reuse is structurally impossible), the
    window spec, and the substrate identity."""
    return canonical_sha({
        "fit_code_version": FIT_CODE_VERSION,
        "engine_config_sha": engine_config_sha,
        "universe": sorted(universe),
        "fit_match_ids": sorted(fit_match_ids),
        "window": list(window_spec),
        "substrate_id": substrate_id,
    })


def cached_fit(cache_dir, key, fit_fn):
    """Content-addressed fit cache in data/harness/cache/ (derived,
    gitignored). Incumbent fits across a knob sweep hit cache after the
    first run - sweeps are incremental by construction."""
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{key}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    ratings = fit_fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(ratings, f)
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


def verdict(stats, *, mde, min_events, objective_ok=True):
    """Verdict classification (spec 5.2, W6a slice - the full gate stack,
    baselines and sweep-family nominee selection land in W6b). Negative
    delta = candidate better (Brier/log-loss). objective_ok=False marks a
    metric-screened knob proxy-only (DoR 5(8)): kept for research, barred
    from screening."""
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
        return {"verdict": "DEV-SCREENED",
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


# -- walk-forward orchestration (spec 4; consumed by the W6b gate/manifest layer) ----------------
def walk_forward_replay(db_path, harness_cfg, cand_cfg, inc_cfg, *,
                        split="dev", limit=None, cache_dir=CACHE_DIR,
                        universe_hops=None):
    """Replay candidate vs incumbent over the eligible events of one split.
    Returns in-memory results; run dirs, manifests, pre-replay gates and
    verdict persistence are W6b's layer on top. `limit` exists for the
    measured perf budget (spec 4.5) and the 1-vs-2-hop spot-check (4.4) -
    a limited run is exploratory only, never verdict-bearing."""
    con = sqlite3.connect(db_path)
    try:
        rows = consumer_rows(con)
        substrate = substrate_identity(db_path, con)
    finally:
        con.close()
    elig = harness_cfg["eligibility"]
    events, universe_report = event_universe(
        rows, tiers=tuple(elig["tiers"]),
        min_matches=elig["min_consumer_matches"])
    split_map = classify_split(events, harness_cfg["holdout_split"])
    chosen = split_map[split]
    if limit is not None:
        chosen = chosen[:limit]
    hops = (universe_hops if universe_hops is not None
            else {"1hop": 1, "2hop": 2}[harness_cfg["fit_universe"]["rule"]])
    min_obs = harness_cfg["fit_universe"]["min_obs"]
    rows_by_tid = {}
    for r in rows:
        rows_by_tid.setdefault(r["tournament_id"], []).append(r)
    rows_sorted = sorted(rows, key=lambda r: (r["start_date"], r["match_id"]))
    substrate_sha = canonical_sha(substrate)
    cand_sha, inc_sha = config_sha(cand_cfg), config_sha(inc_cfg)

    all_rows, summaries = [], []
    for ev in chosen:
        ev_rows = rows_by_tid[ev["tournament_id"]]
        boundary = ev["boundary"]
        w_start = months_before(boundary, harness_cfg["window_months"])
        win_rows = [r for r in rows_sorted
                    if w_start <= r["start_date"] < boundary]
        assert_no_leakage(win_rows, ev_rows, boundary)
        u = build_fit_universe(ev_rows, win_rows, hops=hops)
        window_spec = (w_start, boundary)
        fits = {}
        for tag, cfg, sha in (("cand", cand_cfg, cand_sha),
                              ("inc", inc_cfg, inc_sha)):
            key = fit_cache_key(sha, u["universe"], u["fit_match_ids"],
                                window_spec, substrate_sha)
            fits[tag] = cached_fit(cache_dir, key,
                                   lambda c=cfg, ui=u: fit_engine(c, ui))
        graded, skipped = grade_event_walkforward(
            ev_rows, fits["cand"], fits["inc"], u["window_counts"],
            min_obs=min_obs)
        summary = event_summary(ev["tournament_id"], graded, skipped,
                                coverage_floor=harness_cfg["coverage_floor"])
        summary["boundary"] = boundary
        summary["n_fit_matches"] = len(u["fit_match_ids"])
        summary["universe_size"] = len(u["universe"])
        summaries.append(summary)
        if not summary["excluded"]:
            all_rows.extend(graded)

    included = [s for s in summaries
                if not s["excluded"] and s["mean_delta_brier"] is not None]
    deltas = [s["mean_delta_brier"] for s in included]
    stats = (paired_event_stats(deltas,
                                ci_level=harness_cfg["verdict"]["ci_level"])
             if len(deltas) >= 2 else None)
    return {"harness_version": HARNESS_VERSION, "split": split,
            "limited": limit is not None,
            "universe_report": universe_report,
            "n_events_chosen": len(chosen),
            "n_straddlers": len(split_map["straddlers"]),
            "straddlers": [e["tournament_id"]
                           for e in split_map["straddlers"]],
            "excluded_events": [s["tournament_id"] for s in summaries
                                if s["excluded"]],
            "events": summaries, "rows": all_rows, "stats": stats,
            "substrate": substrate}
