"""D1 parse: bo3.gg raw archive -> validated per-series SQLite substrate (W3).

Turns data/bo3gg/matches_*.jsonl.gz (verbatim archived API pages) into
data/bo3gg/parsed.sqlite - the master-key substrate the F-tier knobs and the
V1 walk-forward harness validate against. Spec: docs/plans/2026-07-01-w3-
bo3gg-parse-spec.md (child of the engine-correctness master spec, Wave 3).

Contract vs quarantine (spec 5): a page/row that violates the archive CONTRACT
(malformed body, missing keys, null start_date) RAISES - fail loud, never write
a guess. A valid row describing a DEGENERATE match (draw, walkover, mislabeled
format) is kept with a quarantine_reason - never dropped, never silently fixed.
W4's classifier owns promotion out of quarantine (score-keyed inferred-multi-map).

Determinism: iteration is fetch-ordered (chunk files sort chronologically,
lines append in fetch order), last-fetch-wins per offset (same rule as
verify_archive, making the W3b reconciliation exact by construction), and
last-wins per match_id. Build identity in parse_meta carries no wall-clock -
two rebuilds from the same inputs produce byte-identical databases.

The DB is derived and gitignored; rebuild with:
  python src/bo3gg_parse.py --rebuild

W8's future training-set query (documented per spec 6; NOT wired - the fit
still reads matches_2026.csv until W8 clears V1):
  SELECT m.* FROM matches m WHERE m.quarantine_reason IS NULL
  -- joined through canonical_alias for modeled teams; weight computed from
  -- config half-life + BO1 discount over bo_type/start_date.
"""

import argparse
import gzip
import json
import sqlite3
from pathlib import Path

from bo3gg_archive import verify_archive

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
ARCHIVE_DIR = DATA / "bo3gg"
DB_PATH = ARCHIVE_DIR / "parsed.sqlite"
CANONICAL_TEAMS = DATA / "canonical_teams.json"

PARSE_VERSION = "v1:census-2026-07-01+quarantine-6rules"

WINS_NEEDED = {1: 1, 2: 2, 3: 2, 5: 3}
MAX_MAPS = {1: 1, 2: 2, 3: 3, 5: 5}
QUARANTINE_MAX_RATE = 0.02   # census-calibrated tripwire (spec 5): ~1.0% expected

# Load-bearing key subset (census 2026-07-01: all keys present on all 71,812
# rows). Absence = the API contract changed under us -> raise.
REQUIRED_KEYS = ("id", "slug", "start_date", "end_date", "bo_type",
                 "team1", "team2", "team1_id", "team2_id",
                 "team1_score", "team2_score", "winner_team_id", "tier",
                 "tournament_id", "stage_id", "round_id", "maps_score",
                 "status", "parsed_status")

SCHEMA = """
CREATE TABLE matches (
  match_id        INTEGER PRIMARY KEY,
  slug            TEXT,
  start_date      TEXT NOT NULL,
  end_date        TEXT,
  bo_type         INTEGER,
  team1_id        INTEGER, team2_id INTEGER,
  team1_name      TEXT,    team2_name TEXT,
  team1_score     INTEGER, team2_score INTEGER,
  winner_team_id  INTEGER,
  tier            TEXT,
  tournament_id   INTEGER, stage_id INTEGER, round_id INTEGER,
  maps_score      TEXT,
  parsed_status   TEXT,
  quarantine_reason TEXT,
  src_chunk       TEXT NOT NULL,
  src_offset      INTEGER NOT NULL,
  fetched_at      TEXT NOT NULL,
  parse_version   TEXT NOT NULL
);
CREATE INDEX idx_matches_start      ON matches(start_date);
CREATE INDEX idx_matches_tournament ON matches(tournament_id);
CREATE INDEX idx_matches_t1         ON matches(team1_id, start_date);
CREATE INDEX idx_matches_t2         ON matches(team2_id, start_date);

CREATE TABLE teams (
  team_id    INTEGER PRIMARY KEY,
  name       TEXT NOT NULL,
  slug       TEXT,
  country_id INTEGER,
  last_seen  TEXT NOT NULL
);

CREATE TABLE canonical_alias (
  canonical  TEXT PRIMARY KEY,
  team_id    INTEGER NOT NULL,
  source     TEXT NOT NULL
);

CREATE TABLE tournaments (
  tournament_id INTEGER PRIMARY KEY,
  name          TEXT,
  slug          TEXT,
  short_name    TEXT,
  start_date    TEXT, end_date TEXT,
  tier          TEXT, tier_rank INTEGER,
  region_id     INTEGER, country_id INTEGER,
  event_type    TEXT, event_scope TEXT, event_level TEXT,
  prize         INTEGER,
  status        TEXT,
  pickem_presence TEXT,
  snapshot_fetched_at TEXT NOT NULL,
  parse_version TEXT NOT NULL
);
CREATE INDEX idx_tournaments_start ON tournaments(start_date);

CREATE TABLE parse_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


# -- archive iteration --------------------------------------------------------
def _iter_lines(data_dir):
    """(chunk_name, line_no, rec) for every archive line, fetch order."""
    for chunk in sorted(Path(data_dir).glob("matches_*.jsonl.gz")):
        with gzip.open(chunk, "rt") as f:
            for i, line in enumerate(f):
                yield chunk.name, i, json.loads(line)


def iter_archive_rows(data_dir=ARCHIVE_DIR):
    """Yield (match, provenance) with last-fetch-wins PER OFFSET.

    Same dedup rule as verify_archive (a crash between the archiver's gzip
    append and state save can duplicate an offset), so the W3b reconciliation
    `rows_seen == verify rows` holds by construction. Two passes: locate the
    surviving line per offset, then stream and yield only those - no page
    bodies are held in memory."""
    last = {}   # offset -> (chunk_name, line_no)
    for chunk_name, i, rec in _iter_lines(data_dir):
        last[rec["offset"]] = (chunk_name, i)
    for chunk_name, i, rec in _iter_lines(data_dir):
        if last[rec["offset"]] != (chunk_name, i):
            continue
        page = json.loads(rec["body"])
        if "total" not in page or "results" not in page:
            raise ValueError(f"{chunk_name}:{i}: page missing total/results "
                             f"(contract drift)")
        prov = {"src_chunk": chunk_name, "src_offset": rec["offset"],
                "fetched_at": rec["fetched_at"]}
        for m in page["results"]:
            yield m, prov


# -- validation (spec 5: fixed dispatch order, first failure = reason) --------
def validate_row(m):
    """quarantine_reason for a degenerate match, None for a clean one.
    Assumes the contract keys exist (enforced by _require_contract first)."""
    bo = m["bo_type"]
    if bo not in WINS_NEEDED:                      # 1. must precede any lookup
        return "unknown_bo_type"
    if not m["team1"] or not m["team2"]:           # 2.
        return "null_team"
    s1, s2 = m["team1_score"], m["team2_score"]
    if s1 is None or s2 is None:                   # 3.
        return "missing_score"
    if bo == 2 and (s1, s2) == (1, 1):             # 4. explicit precedence
        return "bo2_draw"
    w, t1, t2 = m["winner_team_id"], m["team1_id"], m["team2_id"]
    if w is None or w not in (t1, t2):             # 5.
        return "null_winner"
    ws, ls = (s1, s2) if w == t1 else (s2, s1)     # 6.
    if not (ws == WINS_NEEDED[bo] and ls < WINS_NEEDED[bo]
            and s1 + s2 <= MAX_MAPS[bo]):
        return "score_bo_mismatch"
    return None


def _require_contract(m, prov):
    missing = [k for k in REQUIRED_KEYS if k not in m]
    if missing:
        raise ValueError(f"{prov['src_chunk']}@{prov['src_offset']}: match "
                         f"{m.get('id')} missing contract keys {missing}")
    if not m["start_date"]:
        raise ValueError(f"{prov['src_chunk']}@{prov['src_offset']}: match "
                         f"{m['id']} has no start_date (load-bearing for "
                         f"ordering - contract drift, not a degenerate match)")


# -- build --------------------------------------------------------------------
def build_db(db_path=DB_PATH, data_dir=ARCHIVE_DIR,
             canonical_path=CANONICAL_TEAMS):
    """Drop-and-rebuild the substrate in one transaction. Returns the report
    dict (also persisted into parse_meta). Deterministic given the same
    archive + canonical file + PARSE_VERSION."""
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    try:
        con.executescript(SCHEMA)
        rows_seen = 0
        teams = {}   # team_id -> (last_seen_start, name, slug, country_id)
        max_fetched = ""
        for m, prov in iter_archive_rows(data_dir):
            _require_contract(m, prov)
            rows_seen += 1
            max_fetched = max(max_fetched, prov["fetched_at"])
            reason = validate_row(m)
            con.execute(
                "INSERT OR REPLACE INTO matches VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (m["id"], m["slug"], m["start_date"], m["end_date"],
                 m["bo_type"], m["team1_id"], m["team2_id"],
                 (m["team1"] or {}).get("name"), (m["team2"] or {}).get("name"),
                 m["team1_score"], m["team2_score"], m["winner_team_id"],
                 m["tier"], m["tournament_id"], m["stage_id"], m["round_id"],
                 json.dumps(m["maps_score"]), m["parsed_status"], reason,
                 prov["src_chunk"], prov["src_offset"], prov["fetched_at"],
                 PARSE_VERSION))
            for t in (m["team1"], m["team2"]):
                if not t:
                    continue
                prev = teams.get(t["id"])
                # >= : on equal dates the later-fetched row wins - iteration
                # order is deterministic, so so is the tie-break.
                if prev is None or m["start_date"] >= prev[0]:
                    teams[t["id"]] = (m["start_date"], t["name"],
                                      t.get("slug"), t.get("country_id"))
        for tid in sorted(teams):
            seen, name, slug, country = teams[tid]
            con.execute("INSERT INTO teams VALUES (?,?,?,?,?)",
                        (tid, name, slug, country, seen))
        for a in sorted(json.load(open(canonical_path))["aliases"],
                        key=lambda a: a["canonical"]):
            con.execute("INSERT INTO canonical_alias VALUES (?,?,?)",
                        (a["canonical"], a["team_id"], a["source"]))

        unique_ids = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        # Reason counts from the TABLE, not the stream: a superseded duplicate
        # row's reason must not be counted (conservation counts DB rows).
        reasons = dict(con.execute(
            "SELECT quarantine_reason, COUNT(*) FROM matches "
            "WHERE quarantine_reason IS NOT NULL GROUP BY quarantine_reason"))
        state_path = Path(data_dir) / "state.json"
        next_offset = (json.load(open(state_path))["next_offset"]
                       if state_path.exists() else None)
        report = {"parse_version": PARSE_VERSION,
                  "archive_max_fetched_at": max_fetched,
                  "tournaments_snapshot_id": "none",   # W3c fills
                  "rows_seen": rows_seen, "unique_ids": unique_ids,
                  "clean_n": unique_ids - _quarantined_in_db(con),
                  "quarantined_n": _quarantined_in_db(con),
                  "archive_next_offset": next_offset,
                  "reason_counts": reasons, "reconciled": "pending"}
        for k, v in report.items():
            con.execute("INSERT OR REPLACE INTO parse_meta VALUES (?,?)",
                        (k, json.dumps(v) if isinstance(v, dict) else str(v)))
        con.commit()
        return report
    finally:
        con.close()


def _quarantined_in_db(con):
    """Count from the TABLE, not the stream: a superseded duplicate row's
    reason must not count twice (conservation identity counts DB rows)."""
    return con.execute("SELECT COUNT(*) FROM matches "
                       "WHERE quarantine_reason IS NOT NULL").fetchone()[0]


# -- reconciliation (W3b; spec 7) ----------------------------------------------
def reconcile(report, db_path=DB_PATH, data_dir=ARCHIVE_DIR):
    """The ship gate. Returns (ok, failures) and stamps parse_meta.reconciled.

    (a) archive-side: rows_seen == verify_archive rows (+ contiguity);
    (b) conservation: clean_n + quarantined_n == unique_ids == COUNT(*)
        - every unique id lands exactly once, clean or quarantined;
    (c) census tripwires: bo2_draw must occur; quarantine rate <= 2%."""
    failures = []
    v = verify_archive(data_dir)
    if not v["offsets_contiguous"]:
        failures.append("archive offsets not contiguous (fetch-side gap)")
    if report["rows_seen"] != v["rows"]:
        failures.append(f"rows_seen {report['rows_seen']} != "
                        f"verify_archive rows {v['rows']}")
    con = sqlite3.connect(db_path)
    try:
        n = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        q = _quarantined_in_db(con)
        if not (report["clean_n"] + report["quarantined_n"]
                == report["unique_ids"] == n):
            failures.append(
                f"conservation identity broken: clean {report['clean_n']} + "
                f"quarantined {report['quarantined_n']} != unique "
                f"{report['unique_ids']} != db {n}")
        if report["quarantined_n"] != q:
            failures.append(f"quarantine count drift: report "
                            f"{report['quarantined_n']} != db {q}")
        draws = con.execute(
            "SELECT COUNT(*) FROM matches WHERE quarantine_reason='bo2_draw'"
        ).fetchone()[0]
        if draws == 0:
            failures.append("tripwire: 0 bo2_draw rows (census says 223 - "
                            "validation drift?)")
        if n and q / n > QUARANTINE_MAX_RATE:
            failures.append(f"tripwire: quarantine rate {q/n:.2%} > "
                            f"{QUARANTINE_MAX_RATE:.0%} (upstream drift?)")
        ok = not failures
        con.execute("INSERT OR REPLACE INTO parse_meta VALUES (?,?)",
                    ("reconciled", "true" if ok else "false"))
        con.commit()
    finally:
        con.close()
    return ok, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="full deterministic rebuild from the raw archive")
    ap.add_argument("--db", default=DB_PATH, help="output sqlite path")
    args = ap.parse_args()
    if not args.rebuild:
        raise SystemExit("nothing to do (use --rebuild)")
    report = build_db(db_path=args.db)
    ok, failures = reconcile(report, db_path=args.db)
    print(f"parsed {report['rows_seen']} rows -> {report['unique_ids']} "
          f"matches ({report['clean_n']} clean, "
          f"{report['quarantined_n']} quarantined)")
    for reason, count in sorted(report["reason_counts"].items()):
        print(f"  {reason}: {count}")
    print(f"reconciled: {ok}")
    if not ok:
        for f in failures:
            print(f"  FAIL: {f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
