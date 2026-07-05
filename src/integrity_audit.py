"""D2 integrity audit (W4): zero silent errors on the parsed substrate before
any fit trusts it. Master spec Wave 4 + DoR 6 data gates + W3 child spec 5's
pinned promotion direction. Runs on data/bo3gg/parsed.sqlite AFTER a parse.

Checks:
- Promotion classifier: quarantined `score_bo_mismatch` rows whose score PROVES
  multi-map (winner_score >= 2 AND winner_score > loser_score) are flagged
  `inferred_multi_map` - usable by F3's BO1 discount despite format ambiguity.
  Forfeit signatures (1-0, 0-0 with a "winner") stay quarantined.
- Impossible-record scan: self-play, negative scores, end-before-start,
  duplicate pairings (same unordered pair, same scheduled start).
- Tier cross-check: MAX(matches.tier) per tournament vs tournaments.tier;
  divergences are informational (per-row tier stays authoritative, W3 spec 7).
  An empty tournaments slice flags tier_check_vacuous - surfaced, not silent.
- Reference orientation cross-check: the committed two-source Cologne results
  (results_matches_stage3.json + results_matches_playoffs.json) resolved via
  canonical_alias, joined by day + team pair - the parsed winner must agree.
  End-to-end orientation proof of the whole parse pipeline (DoR 6).
- CSV coverage cross-check: matches_2026.csv (the fit's hand-curated input,
  itself two-source verified at entry) must exist in the substrate with the
  same orientation. Rows naming teams without aliases are skipped-and-counted,
  never guessed (two-source alias rule, W3 spec 4).

The audit NEVER edits `matches` - findings land in the additive `audit_flags`
table (severity 'fail' = data integrity broken, nonzero exit; 'report' =
informational, surfaced not fatal), so build_db output stays byte-identical
and the audit re-runs idempotently. Empty reference/CSV inputs RAISE unless
explicitly opted into (vacuous cross-checks must never mint audit_ok=true);
the denominators used are stamped into parse_meta.audit_input_counts. E.1 Liquipedia source cross-validation is
a flagged follow-on (needs a fetcher - own small cycle, master spec 5).

Promoted-rows query for W6/W8 consumers (documented, NOT wired - the fit
still reads matches_2026.csv until W8 clears V1):
  SELECT m.* FROM matches m
  LEFT JOIN audit_flags f ON f.match_id = m.match_id
       AND f.flag = 'inferred_multi_map'
  WHERE m.quarantine_reason IS NULL OR f.flag IS NOT NULL

Run:  python src/integrity_audit.py [--db PATH]
"""

import argparse
import csv
import json
import sqlite3
from pathlib import Path

from bo3gg_parse import DATA, DB_PATH

AUDIT_VERSION = "v1.1:promotion+impossible+tier+reference+csv+vacuity-guards"

REFERENCE_FILES = (DATA / "results_matches_stage3.json",
                   DATA / "results_matches_playoffs.json")
FIT_CSV = DATA / "matches_2026.csv"

# matches_2026.csv carries no dates; the pair join is windowed to the file's
# domain (2026 matches) so historic rematches don't alias into it.
CSV_SINCE = "2026-01-01"

FLAGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_flags (
  match_id      INTEGER,           -- NULL when there is no row to point at
  flag          TEXT NOT NULL,
  severity      TEXT NOT NULL,     -- 'fail' | 'report'
  detail        TEXT NOT NULL,
  audit_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_flags_match ON audit_flags(match_id);
"""


def _ensure(con):
    con.executescript(FLAGS_SCHEMA)


def _flag(con, match_id, flag, severity, detail):
    con.execute("INSERT INTO audit_flags VALUES (?,?,?,?,?)",
                (match_id, flag, severity, detail, AUDIT_VERSION))


def _aliases(con):
    return dict(con.execute("SELECT canonical, team_id FROM canonical_alias"))


def _pair_rows(con, ta, tb, *, day=None, since=None):
    """Rows for an unordered team pair, deterministic order."""
    q = ("SELECT match_id, winner_team_id, quarantine_reason FROM matches "
         "WHERE min(team1_id, team2_id) = ? AND max(team1_id, team2_id) = ?")
    args = [min(ta, tb), max(ta, tb)]
    if day:
        q += " AND substr(start_date, 1, 10) = ?"
        args.append(day)
    if since:
        q += " AND start_date >= ?"
        args.append(since)
    return con.execute(q + " ORDER BY match_id", args).fetchall()


# -- promotion classifier (W3 spec 5: score-keyed, W4 owns it) -----------------
def classify_promotions(con):
    """Every score_bo_mismatch row gets exactly one classification flag.
    Promotion predicate is keyed on the SCORE, never the reason code: a
    winner score >= 2 (and > loser) proves multi-map regardless of the
    BO2-vs-BO3 ambiguity; 1-0/0-1/0-0 "winners" may be forfeit artifacts."""
    _ensure(con)
    counts = {"inferred_multi_map": 0, "forfeit_signature": 0}
    rows = con.execute(
        "SELECT match_id, bo_type, team1_id, team1_score, team2_score, "
        "winner_team_id FROM matches "
        "WHERE quarantine_reason = 'score_bo_mismatch' "
        "ORDER BY match_id").fetchall()
    for mid, bo, t1, s1, s2, w in rows:
        ws, ls = (s1, s2) if w == t1 else (s2, s1)
        if ws >= 2 and ws > ls:
            counts["inferred_multi_map"] += 1
            _flag(con, mid, "inferred_multi_map", "report",
                  f"bo{bo} scored {s1}-{s2}: winner took >=2 maps - "
                  f"multi-map proven despite format mislabel")
        else:
            counts["forfeit_signature"] += 1
            _flag(con, mid, "forfeit_signature", "report",
                  f"bo{bo} scored {s1}-{s2}: forfeit/walkover/abandon "
                  f"signature - winner may be an artifact, stays quarantined")
    return counts


# -- duplicate / impossible-record scan (DoR 6 state validation) ---------------
def scan_impossible(con):
    """Row-level analogues of simulate.make_state's validators (those stay
    the EVENT-scoped authority; this scans the whole substrate)."""
    _ensure(con)
    counts = {}

    def scan(flag, severity, sql, detail_fn):
        found = con.execute(sql).fetchall()
        counts[flag] = len(found)
        for row in found:
            _flag(con, row[0], flag, severity, detail_fn(row))

    scan("self_play", "fail",
         "SELECT match_id, team1_id FROM matches "
         "WHERE team1_id IS NOT NULL AND team1_id = team2_id "
         "ORDER BY match_id",
         lambda r: f"team {r[1]} plays itself")
    scan("negative_score", "fail",
         "SELECT match_id, team1_score, team2_score FROM matches "
         "WHERE team1_score < 0 OR team2_score < 0 ORDER BY match_id",
         lambda r: f"negative score {r[1]}-{r[2]}")
    scan("date_order", "report",
         "SELECT match_id, start_date, end_date FROM matches "
         "WHERE end_date IS NOT NULL AND end_date < start_date "
         "ORDER BY match_id",
         lambda r: f"ends {r[2]} before it starts {r[1]}")

    dup_groups = con.execute(
        "SELECT group_concat(match_id) FROM matches "
        "WHERE team1_id IS NOT NULL AND team2_id IS NOT NULL "
        "GROUP BY min(team1_id, team2_id), max(team1_id, team2_id), "
        "start_date HAVING COUNT(*) > 1 "
        "ORDER BY min(match_id)").fetchall()
    counts["duplicate_pairing"] = 0
    for (ids,) in dup_groups:
        mids = sorted(int(i) for i in ids.split(","))
        for mid in mids:
            counts["duplicate_pairing"] += 1
            _flag(con, mid, "duplicate_pairing", "report",
                  f"same pair, same scheduled start as matches {mids}")
    return counts


# -- tier cross-check (W3 spec 7 W3c: informational until proven load-bearing) -
def tier_cross_check(con):
    """MAX(matches.tier) per tournament vs tournaments.tier. Divergences are
    reported, never fatal - per-row tier stays authoritative for grouping.

    An EMPTY tournaments slice (e.g. --rebuild without --parse-tournaments)
    makes the join vacuous - that skip is surfaced as a report flag, never
    silent. Report-severity keeps W3c's independence: a matches-only DB still
    audits, but nobody mistakes a vacuous check for a passing one."""
    _ensure(con)
    if con.execute("SELECT COUNT(*) FROM tournaments").fetchone()[0] == 0:
        _flag(con, None, "tier_check_vacuous", "report",
              "tournaments table empty - tier cross-check ran against "
              "nothing (run bo3gg_parse.py --parse-tournaments after a "
              "rebuild)")
        return []
    divergences = con.execute(
        "SELECT m.tournament_id, MAX(m.tier) AS mt, t.tier "
        "FROM matches m JOIN tournaments t "
        "  ON t.tournament_id = m.tournament_id "
        "WHERE m.tier IS NOT NULL AND t.tier IS NOT NULL "
        "GROUP BY m.tournament_id, t.tier HAVING mt != t.tier "
        "ORDER BY m.tournament_id").fetchall()
    for tid, match_tier, tourn_tier in divergences:
        _flag(con, None, "tier_divergence", "report",
              f"tournament {tid}: matches say tier '{match_tier}', "
              f"tournaments slice says '{tourn_tier}'")
    return divergences


# -- reference orientation cross-check (DoR 6 orientation gate) -----------------
def cross_check_reference(con, reference):
    """Two-source verified results vs the parsed substrate: for each
    reference row (canonical a/b/winner + scheduled UTC start), the DB must
    hold exactly one same-day match of that pair, and its winner_team_id
    must agree. An unresolvable canonical name is contract drift in OUR
    committed files -> raise, never guess."""
    _ensure(con)
    aliases = _aliases(con)
    missing = sorted({n for ref in reference for n in
                      (ref["a"], ref["b"], ref["winner"])} - set(aliases))
    if missing:
        raise ValueError(f"reference names with no canonical_alias entry "
                         f"(two-source rule - add aliases, don't guess): "
                         f"{missing}")
    result = {"verified": 0, "failures": [], "coverage": {}}
    cov = result["coverage"]
    for ref in reference:
        for name in (ref["a"], ref["b"]):
            cov.setdefault(name, {"expected": 0, "found": 0})
            cov[name]["expected"] += 1

    def fail(match_id, flag, detail):
        _flag(con, match_id, flag, "fail", detail)
        result["failures"].append(detail)

    for ref in reference:
        ta, tb = aliases[ref["a"]], aliases[ref["b"]]
        day = ref["start"][:10]
        rows = _pair_rows(con, ta, tb, day=day)
        label = f"{ref['a']} vs {ref['b']} on {day}"
        if not rows:
            fail(None, "reference_missing", f"{label}: no parsed match")
            continue
        for name in (ref["a"], ref["b"]):
            cov[name]["found"] += 1
        if len(rows) > 1:
            fail(None, "reference_ambiguous",
                 f"{label}: {len(rows)} parsed matches "
                 f"{[m for m, _, _ in rows]} - cannot verify orientation")
            continue
        mid, winner_id, _reason = rows[0]
        if winner_id != aliases[ref["winner"]]:
            fail(mid, "orientation_mismatch",
                 f"{label}: reference winner {ref['winner']} "
                 f"(id {aliases[ref['winner']]}) but parsed winner_team_id "
                 f"is {winner_id}")
        else:
            result["verified"] += 1
    return result


# -- CSV coverage cross-check (E.2: the fit input exists in the substrate) -----
def cross_check_csv(con, csv_rows, since=CSV_SINCE):
    """Each hand-curated row whose teams BOTH have aliases must match exactly
    one clean substrate row (pair join, windowed - the CSV has no dates) with
    the same winner. Unresolvable teams are counted, never guessed; rematch
    ambiguity and quarantined-only hits are reported, not failed."""
    _ensure(con)
    aliases = _aliases(con)
    r = {"verified": 0, "unresolvable": 0, "ambiguous": 0, "unmatched": 0,
         "quarantined": 0, "mismatch": 0}
    for row in csv_rows:
        w_name, l_name = row["winner"], row["loser"]
        label = f"{w_name} over {l_name} ({row.get('event', '?')})"
        if w_name not in aliases or l_name not in aliases:
            r["unresolvable"] += 1
            continue
        w_id, l_id = aliases[w_name], aliases[l_name]
        rows = _pair_rows(con, w_id, l_id, since=since)
        clean = [x for x in rows if x[2] is None]
        if not rows:
            r["unmatched"] += 1
            _flag(con, None, "csv_row_unmatched", "fail",
                  f"{label}: no parsed match for the pair since {since} - "
                  f"phantom CSV row or archive gap")
        elif len(clean) > 1:
            r["ambiguous"] += 1
            _flag(con, None, "csv_ambiguous", "report",
                  f"{label}: {len(clean)} clean matches "
                  f"{[m for m, _, _ in clean]} in window - rematch, "
                  f"orientation not checkable without a date column")
        elif clean:
            mid, winner_id, _ = clean[0]
            if winner_id != w_id:
                r["mismatch"] += 1
                _flag(con, mid, "csv_orientation_mismatch", "fail",
                      f"{label}: CSV winner id {w_id} but parsed "
                      f"winner_team_id is {winner_id}")
            else:
                r["verified"] += 1
        else:
            for mid, _, reason in rows:
                r["quarantined"] += 1
                _flag(con, mid, "csv_match_quarantined", "report",
                      f"{label}: only quarantined parse rows for the pair "
                      f"(reason: {reason}) - CSV recorded a result the "
                      f"substrate holds as degenerate")
    return r


# -- orchestration --------------------------------------------------------------
def run_audit(db_path=DB_PATH, reference_rows=(), csv_rows=(),
              since=CSV_SINCE, allow_empty_inputs=False):
    """Full audit: clear + rewrite audit_flags, stamp parse_meta, return
    (ok, report). ok is False iff any severity='fail' flag exists.
    Deterministic given the same DB + inputs (no wall-clock).

    Empty reference/CSV inputs make those cross-checks vacuous - audit_ok
    would assert what was never verified - so they RAISE unless the caller
    explicitly opts into a partial audit (tests do; production must not)."""
    reference_rows = list(reference_rows)
    csv_rows = list(csv_rows)
    if not allow_empty_inputs and not (reference_rows and csv_rows):
        raise ValueError(
            "run_audit: empty reference_rows/csv_rows would make the "
            "orientation/coverage cross-checks vacuous (audit_ok=true while "
            "verifying nothing). Pass the real inputs (see main()) or set "
            "allow_empty_inputs=True for a deliberately partial audit.")
    con = sqlite3.connect(db_path)
    try:
        _ensure(con)
        con.execute("DELETE FROM audit_flags")
        report = {
            "promotions": classify_promotions(con),
            "impossible": scan_impossible(con),
            "tier_divergences": tier_cross_check(con),
            "reference": cross_check_reference(con, reference_rows),
            "csv": cross_check_csv(con, csv_rows, since=since),
            "input_counts": {"reference_n": len(reference_rows),
                             "csv_n": len(csv_rows)},
        }
        fails = con.execute("SELECT COUNT(*) FROM audit_flags "
                            "WHERE severity = 'fail'").fetchone()[0]
        ok = fails == 0
        report["flag_counts"] = dict(con.execute(
            "SELECT flag, COUNT(*) FROM audit_flags "
            "GROUP BY flag ORDER BY flag"))
        for k, v in (("audit_version", AUDIT_VERSION),
                     ("audit_ok", "true" if ok else "false"),
                     ("audit_flag_counts",
                      json.dumps(report["flag_counts"])),
                     ("audit_input_counts",
                      json.dumps(report["input_counts"]))):
            con.execute("INSERT OR REPLACE INTO parse_meta VALUES (?,?)",
                        (k, v))
        con.commit()
        return ok, report
    finally:
        con.close()


def load_reference(paths=REFERENCE_FILES):
    rows = []
    for p in paths:
        rows.extend(json.load(open(p))["matches"])
    return rows


def load_fit_csv(path=FIT_CSV):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH, help="parsed substrate path")
    args = ap.parse_args()
    if not Path(args.db).exists():
        raise SystemExit(f"{args.db} not found - run "
                         f"'python src/bo3gg_parse.py --rebuild' first")
    ok, report = run_audit(args.db, reference_rows=load_reference(),
                           csv_rows=load_fit_csv())
    print(f"audit {AUDIT_VERSION}")
    print(f"promotions: {report['promotions']}")
    print(f"impossible-record scan: {report['impossible']}")
    print(f"tier divergences: {len(report['tier_divergences'])}")
    ref = report["reference"]
    print(f"reference cross-check: {ref['verified']} verified, "
          f"{len(ref['failures'])} failures")
    deficits = {n: c for n, c in ref["coverage"].items()
                if c["found"] < c["expected"]}
    if deficits:
        print(f"  coverage deficits: {deficits}")
    print(f"fit-CSV cross-check: {report['csv']}")
    print(f"flags: {report['flag_counts']}")
    print(f"audit_ok: {ok}")
    if not ok:
        con = sqlite3.connect(args.db)
        for mid, flag, detail in con.execute(
                "SELECT match_id, flag, detail FROM audit_flags "
                "WHERE severity = 'fail' ORDER BY flag, match_id"):
            print(f"  FAIL [{flag}] match={mid}: {detail}")
        con.close()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
