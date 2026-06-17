"""Graded calibration log + grader (W2 of the engine-correctness Phase I build).

Turns the forecast log into a *graded* one: per-match model-vs-market-vs-outcome
rows and per-team stage-outcome rows, persisted append-only with superseding
semantics, re-gradeable from immutable inputs. This is the data the validation
harness (V1) and the market layer (M-tier) will read.

Design choices (from docs/plans/2026-06-17-engine-correctness-implementation.md):
  - **Wraps the existing grader, no new scoring dialect.** Row-level Brier/log
    are the math that lived in postmortem_matches' print path; the market close
    join is market_close (W1). The aggregate summary is cross-checked against
    postmortem_matches.grade_matches.
  - **Two row kinds, discriminated by `kind`:** "match" (market-relative —
    model/close prob, result, brier, log_loss, delta) and "team" (market-free —
    p30/padv/p03 vs record). One file, never silently mixed.
  - **Adoption eligibility:** a match row is adoption-eligible only if it has a
    real close that is not flagged (flagged = in-play fallback, §6). Flagged rows
    are logged + re-gradeable but excluded from the default aggregate / adoption
    gates unless --include-flagged.
  - **Append-only + superseding:** corrections append a new row with the same
    key; load_latest() returns the last per key. Nothing is ever rewritten.
  - **Per-match model prob provenance:** reconstructed from a frozen ratings file
    + pair overrides (the Cologne backfill uses ratings_locked_v3). Recorded as
    reconstruction_mode="backfill_reconstructed" with the input hashes, so the
    row stays re-gradeable. Forward events get the live.py forecast manifest (W2b).

Usage (Cologne backfill):
  python src/calibration.py --grade-event cologne-stage3
  python src/calibration.py --grade-event cologne-stage3 --include-flagged
"""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
GRADE_VERSION = "v1"
GRADED_LOG = DATA / "calibration_graded.jsonl"

_CATS = ("p30", "padv", "p03")


# -- scoring primitives ------------------------------------------------------
def _brier(p, y):
    return (p - y) ** 2


def _logloss(p, y):
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def grade_match_row(a, b, winner, model_prob, close, *, market_prob=None,
                    provenance=None):
    """One kind:"match" graded row. Probabilities are P(a wins).

    close = a market_close.close_row(...) dict (with p_a/flagged/close_rule/ts/
    slug) or None. market_prob = the market line the model USED at lock (anchor),
    distinct from close_prob (the closing line we grade against); provenance for
    the anchored-pair degeneracy, not used in scoring."""
    y = 1.0 if winner == a else 0.0
    close_prob = close["p_a"] if close else None
    flagged = close["flagged"] if close else None
    row = {
        "kind": "match", "a": a, "b": b, "winner": winner, "result": y,
        "model_prob": model_prob, "market_prob": market_prob,
        "close_prob": close_prob,
        "close_rule": close["close_rule"] if close else None,
        "close_flagged": flagged,
        "close_ts": close["ts"] if close else None,
        "close_slug": close.get("slug") if close else None,
        "brier_model": _brier(model_prob, y),
        "log_model": _logloss(model_prob, y),
    }
    if close_prob is not None:
        row["brier_market"] = _brier(close_prob, y)
        row["log_market"] = _logloss(close_prob, y)
        row["delta_brier"] = row["brier_market"] - row["brier_model"]
        row["delta_log"] = row["log_market"] - row["log_model"]
    else:
        row["brier_market"] = row["log_market"] = None
        row["delta_brier"] = row["delta_log"] = None
    # Adoption-eligible only with a real, unflagged close (DoR §6).
    row["adoption_eligible"] = close_prob is not None and not flagged
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


# -- aggregate summary (cross-checks postmortem_matches.grade_matches) -------
def summarize(rows, include_flagged=False):
    """Aggregate model-vs-market Brier/log over match rows that HAVE a close.
    Adoption-eligible (unflagged) rows only by default; flagged rows counted but
    excluded unless include_flagged. Per-team rows are ignored here."""
    graded = [r for r in rows
              if r["kind"] == "match" and r["close_prob"] is not None]
    eligible = [r for r in graded if r["adoption_eligible"]]
    used = graded if include_flagged else eligible
    summary = {"eligible_n": len(eligible),
               "excluded_flagged_n": len(graded) - len(eligible),
               "n": len(used)}
    if not used:
        return summary
    n = len(used)
    mb = sum(r["brier_model"] for r in used) / n
    kb = sum(r["brier_market"] for r in used) / n
    ml = sum(r["log_model"] for r in used) / n
    kl = sum(r["log_market"] for r in used) / n
    summary.update({
        "model": {"brier": mb, "log": ml},
        "market": {"brier": kb, "log": kl},
        "delta_brier": kb - mb, "delta_log": kl - ml})
    return summary


# -- provenance + backfill orchestration -------------------------------------
def _sha(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, text=True).strip()
    except Exception:
        return "unknown"


def grade_event_matches(matches, ratings, overrides, archive, event,
                        model_version, provenance):
    """Build kind:"match" rows for one event (pure given its inputs).

    matches: [{a, b, winner, start}]; ratings: team->rating; overrides:
    {(a,b): p_a} market lines used at lock; archive: odds rows for close lookup."""
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


def backfill_cologne():
    """Reconstruct + persist the Cologne graded rows. Per-match = the v3 lock
    (model probs reconstructed from ratings_locked_v3 + market_anchors); per-team
    = v1/v2/v3 locked tables vs the final records."""
    event = "cologne-stage3"
    from model import load_pair_overrides
    archive = load_log(DATA / "odds_archive.jsonl")
    matches = json.load(open(DATA / "results_matches_stage3.json"))["matches"]
    ratings = json.load(open(DATA / "ratings_locked_v3.json"))
    overrides = load_pair_overrides(DATA / "market_anchors.json")
    match_prov = {
        "grade_version": GRADE_VERSION, "code_sha": _git_sha(),
        "reconstruction_mode": "backfill_reconstructed",
        "ratings_source": "ratings_locked_v3.json",
        "ratings_sha": _sha(DATA / "ratings_locked_v3.json"),
        "anchors_source": "market_anchors.json",
        "anchors_sha": _sha(DATA / "market_anchors.json"),
        "pair_overrides_version": "load_pair_overrides@market_anchors.json",
    }
    rows = grade_event_matches(matches, ratings, overrides, archive, event,
                               "v3-lock", match_prov)

    results = {t: tuple(r) for t, r in
               json.load(open(DATA / "results_stage3.json")).items()}
    tables = [("v1", "stage3_probs_locked_v1.json"),
              ("v2", "stage3_probs_locked_v2.json"),
              ("v3", "stage3_probs.json")]
    for ver, fn in tables:
        probs = json.load(open(DATA / fn))["probs"]
        prov = {"grade_version": GRADE_VERSION, "code_sha": _git_sha(),
                "reconstruction_mode": "backfill_reconstructed",
                "probs_source": fn, "probs_sha": _sha(DATA / fn)}
        rows += grade_team_table(probs, results, event, ver, prov)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grade-event", default="cologne-stage3",
                    help="event id to (re)grade; only cologne-stage3 wired today")
    ap.add_argument("--include-flagged", action="store_true",
                    help="include in-play-fallback closes in the summary")
    args = ap.parse_args()
    if args.grade_event != "cologne-stage3":
        raise SystemExit(f"only cologne-stage3 is wired (got {args.grade_event})")

    rows = backfill_cologne()
    append_log(rows, GRADED_LOG)
    match_rows = [r for r in rows if r["kind"] == "match"]
    s = summarize(rows, include_flagged=args.include_flagged)
    print(f"Graded {len(match_rows)} matches (v3 lock) + "
          f"{len(rows) - len(match_rows)} team rows -> {GRADED_LOG.name}")
    print(f"  eligible matches: {s['eligible_n']}  "
          f"(excluded flagged: {s['excluded_flagged_n']})")
    if s["n"]:
        print(f"  model  Brier {s['model']['brier']:.4f}  log {s['model']['log']:.4f}")
        print(f"  market Brier {s['market']['brier']:.4f}  log {s['market']['log']:.4f}")
        print(f"  delta (market - model; + = model beat close): "
              f"Brier {s['delta_brier']:+.4f}, log {s['delta_log']:+.4f}")
    print("E.7 humility: one event, correlated matches, tiny n. Evidence, not a verdict.")


if __name__ == "__main__":
    main()
