"""Graded calibration log + grader (W2 + W2-hardening of the Phase I build).

Turns the forecast log into a *graded* one: per-match model-vs-market-vs-outcome
rows and per-team stage-outcome rows, persisted append-only with superseding
semantics, re-gradeable from immutable inputs. The data V1 (the harness) and the
M-tier read.

Adoption gating (DoR §6 + §2.3) — a match row may be ADOPTION-GATE evidence only
if all hold:
  - it has a real market close (close_prob is not None),
  - that close is not flagged (flagged = in-play fallback, §6),
  - it is MANIFESTED: a recorded lock contract that makes the model prob exactly
    replayable (is_manifested) — backfill_reconstructed/forecast_manifest with the
    input hashes and a CLEAN code tree, or immutable_forecast. An unmanifested or
    dirty-tree row is logged + re-gradeable but never counted in an adoption gate.

This split (close-usability vs replayability) was added after an external review
caught that an unmanifested row was adoption-eligible and that the committed log
recorded a code_sha that did not contain this grader (generated from a dirty,
pre-commit tree). _git_sha now also reports code-dirtiness.

Two row kinds, discriminated by `kind`: "match" (market-relative) and "team"
(market-free p30/padv/p03 vs record). One append-only file; load_latest applies
superseding (last per key wins, nothing rewritten).

Usage (Cologne backfill / regrade):
  python src/calibration.py --grade-event cologne-stage3
  python src/calibration.py --grade-event cologne-stage3 --force   # superseding re-grade
"""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
GRADE_VERSION = "v1"
GRADED_LOG = DATA / "calibration_graded.jsonl"

_CATS = ("p30", "padv", "p03")

COLOGNE_SNAPSHOT = {
    "event": "cologne-stage3",
    "matches_file": "results_matches_stage3.json",
    "ratings_file": "ratings_locked_v3.json",
    "anchors_file": "market_anchors.json",
    "archive_file": "odds_archive.jsonl",
    "results_file": "results_stage3.json",
    "team_tables": [("v1", "stage3_probs_locked_v1.json"),
                    ("v2", "stage3_probs_locked_v2.json"),
                    ("v3", "stage3_probs.json")],
}

# Playoff bracket (Champions stage). ratings_fitted.json is the living file,
# but it is byte-identical to the lock-time fit: no fit-touching change landed
# between the R5 refit (the fit playoffs.py ran on at lock) and grading — held
# by the CI reproducibility gate and verified 2026-07-01. No per-team tables:
# the p30/padv/p03 stage-outcome object is Swiss-specific.
COLOGNE_PLAYOFFS_SNAPSHOT = {
    "event": "cologne-playoffs",
    "matches_file": "results_matches_playoffs.json",
    "ratings_file": "ratings_fitted.json",
    "anchors_file": "playoff_anchors.json",
    "archive_file": "odds_archive.jsonl",
}


# -- scoring primitives ------------------------------------------------------
def _brier(p, y):
    return (p - y) ** 2


def _logloss(p, y):
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


# -- manifest / adoption contract (DoR §2.3) ---------------------------------
# Replay-input hashes a row's model prob actually depends on, per reconstruction
# mode. A priced match's model prob IS the market anchor (grade_event_matches),
# so anchors_sha is load-bearing wherever the model prob can be an anchor; the
# forward manifest additionally pins pair_overrides separately. immutable_forecast
# logs the per-match prob itself, so the code sha alone makes it replayable.
_REQUIRED_HASHES = {
    "backfill_reconstructed": ("code_sha", "ratings_sha", "anchors_sha"),
    "forecast_manifest": ("code_sha", "ratings_sha", "anchors_sha",
                          "pair_overrides_sha"),
    "immutable_forecast": ("code_sha",),
}


def _resolved_hash(v):
    """A provenance hash that was actually pinned — not missing, empty, a known
    unresolved sentinel, or a 'pending-*' placeholder (e.g. event_config_sha=
    'pending-w5' on a forward row logged before the event config was pinned)."""
    return bool(v) and v != "unknown" and not str(v).startswith("pending")


def is_manifested(provenance):
    """True iff the row's model prob is exactly replayable from a recorded lock
    contract: a known reconstruction mode, a CLEAN code tree (the dirty marker
    must be present and False — a missing marker is unknown provenance), and
    every replay-input hash the mode depends on present and resolved. As defense
    in depth, a placeholder in ANY *_sha field the manifest carries also fails."""
    if not provenance:
        return False
    if provenance.get("code_dirty") is not False:   # absent (unknown) or True (dirty)
        return False
    required = _REQUIRED_HASHES.get(provenance.get("reconstruction_mode"))
    if not required:
        return False
    if not all(_resolved_hash(provenance.get(k)) for k in required):
        return False
    return all(_resolved_hash(v) for k, v in provenance.items()
               if k.endswith("_sha"))


def grade_match_row(a, b, winner, model_prob, close, *, market_prob=None,
                    provenance=None):
    """One kind:"match" graded row. Probabilities are P(a wins).

    close = a market_close.close_row(...) dict (p_a/flagged/close_rule/ts/slug/
    volume) or None. market_prob = the line the model USED at lock (anchor),
    distinct from close_prob (the closing line we grade against)."""
    y = 1.0 if winner == a else 0.0
    close_prob = close["p_a"] if close else None
    flagged = close["flagged"] if close else None
    manifested = is_manifested(provenance)
    row = {
        "kind": "match", "a": a, "b": b, "winner": winner, "result": y,
        "model_prob": model_prob, "market_prob": market_prob,
        "close_prob": close_prob,
        "close_rule": close["close_rule"] if close else None,
        "close_flagged": flagged,
        "close_ts": close["ts"] if close else None,
        "close_slug": close.get("slug") if close else None,
        "close_volume": close.get("volume") if close else None,
        "brier_model": _brier(model_prob, y),
        "log_model": _logloss(model_prob, y),
        "manifested": manifested,
    }
    if close_prob is not None:
        row["brier_market"] = _brier(close_prob, y)
        row["log_market"] = _logloss(close_prob, y)
        row["delta_brier"] = row["brier_market"] - row["brier_model"]
        row["delta_log"] = row["log_market"] - row["log_model"]
    else:
        row["brier_market"] = row["log_market"] = None
        row["delta_brier"] = row["delta_log"] = None
    # Adoption-eligible = close usable (§6) AND replayable (§2.3).
    row["adoption_eligible"] = (close_prob is not None and not flagged
                                and manifested)
    if provenance:
        row["provenance"] = provenance
    return row


def grade_team_row(team, probs, record, *, provenance=None):
    """One kind:"team" graded row: market-free p30/padv/p03 Brier vs the actual
    record (the postmortem.py object). record = (wins, losses)."""
    rec = (record[0], record[1])
    outcome = {"p30": rec == (3, 0),
               "padv": rec in ((3, 1), (3, 2)),
               "p03": rec == (0, 3)}
    row = {"kind": "team", "team": team, "record": [rec[0], rec[1]]}
    for c in _CATS:
        row[f"{c}_prob"] = probs[c]
        row[f"{c}_brier"] = _brier(probs[c], 1.0 if outcome[c] else 0.0)
    if provenance:
        row["provenance"] = provenance
    return row


# -- append-only log ---------------------------------------------------------
def _row_key(row):
    """Identity for superseding. Includes model_version so v1/v2/v3 rows for the
    same team/match coexist; a re-grade of the SAME (event, version, pair/team)
    supersedes."""
    if row["kind"] == "match":
        return ("match", row.get("event"), row.get("model_version"),
                row["a"], row["b"])
    return ("team", row.get("event"), row.get("model_version"), row.get("team"))


def append_log(rows, path=GRADED_LOG):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_log(path=GRADED_LOG):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_latest(path=GRADED_LOG):
    """Latest row per key (superseding semantics: last appended wins)."""
    latest = {}
    for r in load_log(path):
        latest[_row_key(r)] = r
    return list(latest.values())


# -- aggregate summary -------------------------------------------------------
def summarize(rows, include_flagged=False, require_manifest=True):
    """Aggregate model-vs-market Brier/log over match rows that HAVE a close.

    Default = the ADOPTION summary (DoR §2.3/§6): unflagged AND manifested rows
    only. require_manifest=False gives the manifest-agnostic scoring view (what
    postmortem_matches computes — used for the cross-check). include_flagged
    additionally folds in in-play-fallback closes. Per-team rows are ignored."""
    graded = [r for r in rows
              if r["kind"] == "match" and r["close_prob"] is not None]
    flagged_n = sum(1 for r in graded if r["close_flagged"])
    unflagged = [r for r in graded if not r["close_flagged"]]
    unmanifested_n = sum(1 for r in unflagged if not r.get("manifested"))
    pool = graded if include_flagged else unflagged
    used = [r for r in pool if r.get("manifested")] if require_manifest else pool
    summary = {"n": len(used),
               "eligible_n": sum(1 for r in graded if r.get("adoption_eligible")),
               "excluded_flagged_n": flagged_n,
               "excluded_unmanifested_n": unmanifested_n}
    if used:
        n = len(used)
        mb = sum(r["brier_model"] for r in used) / n
        kb = sum(r["brier_market"] for r in used) / n
        ml = sum(r["log_model"] for r in used) / n
        kl = sum(r["log_market"] for r in used) / n
        summary.update({"model": {"brier": mb, "log": ml},
                        "market": {"brier": kb, "log": kl},
                        "delta_brier": kb - mb, "delta_log": kl - ml})
    return summary


# -- provenance helpers ------------------------------------------------------
def _sha(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def _src_dirty():
    """True if tracked files under src/ have uncommitted changes — i.e. the code
    that produced these rows is not the committed code at code_sha. The output
    log's own dirtiness is irrelevant, so we scope the check to src/."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "src"], cwd=REPO, text=True)
        return bool(out.strip())
    except Exception:
        return True   # unknown provenance -> assume dirty (honest pessimism)


# -- event grading + regrade -------------------------------------------------
def grade_event_matches(matches, ratings, overrides, archive, event,
                        model_version, provenance):
    """Build kind:"match" rows for one event (pure given its inputs)."""
    from market_close import close_row
    from model import win_prob
    rows = []
    for m in matches:
        a, b, start = m["a"], m["b"], m["start"]
        market_prob = overrides.get((a, b))
        model_prob = market_prob if market_prob is not None else win_prob(ratings, a, b)
        close = close_row(archive, a, b, start)
        row = grade_match_row(a, b, m["winner"], model_prob, close,
                              market_prob=market_prob, provenance=provenance)
        row["event"] = event
        row["model_version"] = model_version
        row["round"] = m.get("round")
        rows.append(row)
    return rows


def grade_team_table(probs_table, results, event, model_version, provenance):
    """Build kind:"team" rows for one locked table vs final records. Stamps
    event + model_version so v1/v2/v3 rows for a team are keyed distinctly (else
    they collapse to one under the superseding semantics)."""
    rows = []
    for t in probs_table:
        row = grade_team_row(t, probs_table[t], tuple(results[t]),
                             provenance=provenance)
        row["event"] = event
        row["model_version"] = model_version
        rows.append(row)
    return rows


def grade_event(event, matches, ratings, overrides, archive, match_provenance,
                team_tables, results):
    """All graded rows (match + team) for one event from LOADED inputs. Pure
    given its inputs -> the unit of re-gradeability. team_tables = list of
    (model_version, probs_dict, provenance)."""
    rows = grade_event_matches(matches, ratings, overrides, archive, event,
                               "v3-lock", match_provenance)
    for version, probs, prov in team_tables:
        rows += grade_team_table(probs, results, event, version, prov)
    return rows


def grade_playoff_matches(matches, ratings, overrides, archive, event,
                          model_version, provenance):
    """kind:"match" rows for the playoff bracket, reconstructing the LOCK-TIME
    model probs via the same code path playoffs.py used at lock (make_prob_fn):
    anchored pairs take the market prob verbatim, unpriced BO3 = win_prob on
    the lock ratings, and the BO5 grand final converts via series_prob_bo5.
    Pure given its inputs -> the unit of re-gradeability."""
    from market_close import close_row
    from playoffs import make_prob_fn
    prob = make_prob_fn(ratings, overrides)
    rows = []
    for m in matches:
        a, b = m["a"], m["b"]
        model_prob = prob(a, b, bo5=(m.get("bo") == 5))
        row = grade_match_row(a, b, m["winner"], model_prob,
                              close_row(archive, a, b, m["start"]),
                              market_prob=overrides.get((a, b)),
                              provenance=provenance)
        row["event"] = event
        row["model_version"] = model_version
        row["round"] = m.get("round")
        rows.append(row)
    return rows


def regrade_playoffs_from_snapshot(snapshot=None):
    """Playoff analog of regrade_from_snapshot: load the immutable lock inputs
    (announced-bracket results, lock ratings, lock QF anchors, odds archive),
    stamp backfill_reconstructed provenance, and grade. Read-only."""
    snap = snapshot or COLOGNE_PLAYOFFS_SNAPSHOT
    from playoffs import load_playoff_overrides
    archive = load_log(DATA / snap["archive_file"])
    matches = json.load(open(DATA / snap["matches_file"]))["matches"]
    ratings = json.load(open(DATA / snap["ratings_file"]))
    overrides = load_playoff_overrides(DATA / snap["anchors_file"])
    prov = {
        "grade_version": GRADE_VERSION, "code_sha": _git_sha(),
        "code_dirty": _src_dirty(),
        "reconstruction_mode": "backfill_reconstructed",
        "ratings_source": snap["ratings_file"],
        "ratings_sha": _sha(DATA / snap["ratings_file"]),
        "anchors_source": snap["anchors_file"],
        "anchors_sha": _sha(DATA / snap["anchors_file"]),
        "pair_overrides_version": f"load_playoff_overrides@{snap['anchors_file']}",
    }
    return grade_playoff_matches(matches, ratings, overrides, archive,
                                 snap["event"], "playoff-lock", prov)


def regrade_from_snapshot(snapshot=None):
    """Load the immutable inputs named in the snapshot, stamp provenance with
    their current hashes + code state, and grade. Deterministic: identical inputs
    -> identical rows. Read-only (does not append)."""
    snap = snapshot or COLOGNE_SNAPSHOT
    from model import load_pair_overrides
    archive = load_log(DATA / snap["archive_file"])
    matches = json.load(open(DATA / snap["matches_file"]))["matches"]
    ratings = json.load(open(DATA / snap["ratings_file"]))
    overrides = load_pair_overrides(DATA / snap["anchors_file"])
    code_sha, code_dirty = _git_sha(), _src_dirty()
    match_prov = {
        "grade_version": GRADE_VERSION, "code_sha": code_sha,
        "code_dirty": code_dirty, "reconstruction_mode": "backfill_reconstructed",
        "ratings_source": snap["ratings_file"],
        "ratings_sha": _sha(DATA / snap["ratings_file"]),
        "anchors_source": snap["anchors_file"],
        "anchors_sha": _sha(DATA / snap["anchors_file"]),
        "pair_overrides_version": f"load_pair_overrides@{snap['anchors_file']}",
    }
    results = {t: tuple(r) for t, r in
               json.load(open(DATA / snap["results_file"])).items()}
    team_tables = []
    for version, fn in snap["team_tables"]:
        probs = json.load(open(DATA / fn))["probs"]
        prov = {"grade_version": GRADE_VERSION, "code_sha": code_sha,
                "code_dirty": code_dirty,
                "reconstruction_mode": "backfill_reconstructed",
                # team rows: ratings_sha stands in via the probs table hash
                "ratings_sha": _sha(DATA / fn),
                "probs_source": fn, "probs_sha": _sha(DATA / fn)}
        team_tables.append((version, probs, prov))
    return grade_event(snap["event"], matches, ratings, overrides, archive,
                       match_prov, team_tables, results)


EVENTS = {
    "cologne-stage3": regrade_from_snapshot,
    "cologne-playoffs": regrade_playoffs_from_snapshot,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grade-event", default="cologne-stage3",
                    help=f"event id to (re)grade; wired: {', '.join(EVENTS)}")
    ap.add_argument("--include-flagged", action="store_true",
                    help="include in-play-fallback closes in the summary")
    ap.add_argument("--force", action="store_true",
                    help="append a superseding re-grade even if already logged")
    args = ap.parse_args()
    if args.grade_event not in EVENTS:
        raise SystemExit(f"unknown event {args.grade_event!r}; "
                         f"wired: {', '.join(EVENTS)}")

    already = args.grade_event in {r.get("event") for r in load_log(GRADED_LOG)}
    if already and not args.force:
        raise SystemExit(
            f"{args.grade_event} already in {GRADED_LOG.name}; re-running would "
            f"append duplicates. Use --force for a superseding re-grade.")

    rows = EVENTS[args.grade_event]()
    append_log(rows, GRADED_LOG)
    match_rows = [r for r in rows if r["kind"] == "match"]
    s = summarize(rows, include_flagged=args.include_flagged)        # adoption view
    xc = summarize(rows, require_manifest=False)                     # cross-check view
    version = match_rows[0]["model_version"] if match_rows else "?"
    print(f"Graded {len(match_rows)} matches ({version}) + "
          f"{len(rows) - len(match_rows)} team rows -> {GRADED_LOG.name}")
    print(f"  adoption-eligible matches: {s['eligible_n']}  "
          f"(flagged {s['excluded_flagged_n']}, unmanifested {s['excluded_unmanifested_n']})")
    if xc["n"]:
        print(f"  model  Brier {xc['model']['brier']:.4f}  log {xc['model']['log']:.4f}")
        print(f"  market Brier {xc['market']['brier']:.4f}  log {xc['market']['log']:.4f}")
        print(f"  delta (market - model; + = model beat close): "
              f"Brier {xc['delta_brier']:+.4f}, log {xc['delta_log']:+.4f}  "
              f"[over {xc['n']} unflagged matches, manifest-agnostic]")
    print("E.7 humility: one event, correlated matches, tiny n. Evidence, not a verdict.")


if __name__ == "__main__":
    main()
