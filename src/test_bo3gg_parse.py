"""bo3gg_parse tests (W3a/W3b): validation dispatch, contract raises, offset +
match-id dedup, deterministic rebuild, conservation reconciliation, tripwires.

Fixture archives are built in tmp dirs - no live IO, no real-archive reads
(the census facts the rules encode are pinned in the spec, exercised here on
crafted rows)."""

import gzip
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from bo3gg_parse import (PARSE_VERSION, TOURNAMENT_KEYS, build_db,
                         fetch_tournaments, iter_archive_rows,
                         latest_complete_snapshot, parse_tournaments,
                         reconcile, validate_row)


def team(tid, name):
    return {"id": tid, "slug": name.lower(), "name": name, "rank": 1,
            "image_url": None, "tshirt_image_url": None, "icon_url": None,
            "country_id": 100 + tid, "discipline_id": 1}


def match(mid, *, bo=3, t1=1, t2=2, s1=2, s2=0, winner="t1",
          start="2026-01-01T10:00:00+00:00", **over):
    w = {"t1": t1, "t2": t2, None: None}[winner]
    loser = {"t1": t2, "t2": t1, None: None}[winner]
    m = {"id": mid, "slug": f"m{mid}", "start_date": start,
         "end_date": start, "bo_type": bo,
         "team1": team(t1, f"Team{t1}"), "team2": team(t2, f"Team{t2}"),
         "team1_id": t1, "team2_id": t2, "team1_score": s1, "team2_score": s2,
         "winner_team_id": w, "tier": "s", "tournament_id": 500,
         "stage_id": 9, "round_id": 7, "maps_score": [True, True],
         "status": "finished", "parsed_status": "done",
         # unconsumed archive-contract keys: the parser pins ALL observed
         # keys (spec 5 - a missing one is contract drift), so fixtures must
         # carry the full 43-key row shape the real archive has.
         "bet_updates": None, "comments": None, "comments_count": 0,
         "discipline_id": 1, "game_version": None, "live_coverage": False,
         "live_coverage_advantage": None, "live_coverage_source": None,
         "live_updates": None, "loser_team_id": loser, "points": None,
         "position": None, "prev_match1_id": None, "prev_match1_winner": None,
         "prev_match2_id": None, "prev_match2_winner": None, "rating": None,
         "stars": 0, "team1_last_game_score": None,
         "team1_new_participant": False, "team2_last_game_score": None,
         "team2_new_participant": False, "tier_rank": 1, "winner_team": None}
    m.update(over)
    return m


class FixtureArchive:
    """Writes crafted pages as a real gz archive layout in a tmp dir."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def write(self, pages, chunk="matches_0000.jsonl.gz", offsets=None):
        with gzip.open(self.dir / chunk, "at") as f:
            for i, rows in enumerate(pages):
                offset = offsets[i] if offsets else i * 100
                f.write(json.dumps({
                    "fetched_at": f"2026-06-0{(i % 8) + 1}T00:00:00+00:00",
                    "url": "fixture", "offset": offset,
                    "body": json.dumps({"total": {"count": 999},
                                        "results": rows}),
                }) + "\n")
        return self

    def state(self, next_offset):
        (self.dir / "state.json").write_text(
            json.dumps({"next_offset": next_offset}))
        return self

    def canonical(self, aliases=None):
        path = self.dir / "canonical_teams.json"
        path.write_text(json.dumps({"aliases": aliases if aliases is not None
                                    else [{"canonical": "T1", "team_id": 1,
                                           "source": "test"}]}))
        return path


# A fixture set whose census matches the tripwires: mostly clean, one bo2 draw,
# contiguous offsets (each page's rows start where the previous ended).
def standard_pages():
    clean = [match(i, s1=2, s2=1) for i in range(1, 90)]
    draw = match(90, bo=2, s1=1, s2=1, winner=None)
    return [clean[:50], clean[50:] + [draw]]


def build_fixture(fx=None):
    fx = fx or FixtureArchive()
    pages = standard_pages()
    fx.write(pages, offsets=[0, 50]).state(90)
    return fx


class TestValidateRow(unittest.TestCase):
    def test_clean_bo3(self):
        self.assertIsNone(validate_row(match(1, s1=2, s2=1)))

    def test_clean_bo1_bo5_bo2(self):
        self.assertIsNone(validate_row(match(1, bo=1, s1=1, s2=0)))
        self.assertIsNone(validate_row(match(2, bo=5, s1=3, s2=2)))
        self.assertIsNone(validate_row(match(3, bo=2, s1=2, s2=0)))

    def test_unknown_bo_type_precedes_lookup(self):
        # bo=7 must yield the reason, not KeyError inside wins_needed.
        self.assertEqual(validate_row(match(1, bo=7)), "unknown_bo_type")
        self.assertEqual(validate_row(match(2, bo=None)), "unknown_bo_type")

    def test_null_team(self):
        self.assertEqual(validate_row(match(1, team1=None)), "null_team")

    def test_missing_score(self):
        self.assertEqual(validate_row(match(1, team1_score=None)),
                         "missing_score")

    def test_bo2_draw_precedence(self):
        # (1,1) BO2 must be bo2_draw, never null_winner/score_bo_mismatch.
        self.assertEqual(validate_row(match(1, bo=2, s1=1, s2=1, winner=None)),
                         "bo2_draw")

    def test_null_winner(self):
        self.assertEqual(validate_row(match(1, winner=None)), "null_winner")
        # winner id not one of the two teams is equally unusable
        self.assertEqual(validate_row(match(2, winner_team_id=99)),
                         "null_winner")

    def test_score_bo_mismatch_classes(self):
        # mislabel: "BO1" scored 2-0; abandon: BO3 at 1-0; impossible BO2 2-1
        self.assertEqual(validate_row(match(1, bo=1, s1=2, s2=0)),
                         "score_bo_mismatch")
        self.assertEqual(validate_row(match(2, bo=3, s1=1, s2=0)),
                         "score_bo_mismatch")
        self.assertEqual(validate_row(match(3, bo=2, s1=2, s2=1)),
                         "score_bo_mismatch")

    def test_dispatch_first_failure_wins(self):
        # null team AND weird score: earlier rule (null_team) is the reason.
        self.assertEqual(validate_row(match(1, team1=None, s1=9, s2=9)),
                         "null_team")


class TestContractRaises(unittest.TestCase):
    def _build(self, fx):
        return build_db(db_path=fx.dir / "t.sqlite", data_dir=fx.dir,
                        canonical_path=fx.canonical())

    def test_missing_key_raises(self):
        fx = FixtureArchive()
        bad = match(1)
        del bad["winner_team_id"]
        fx.write([[bad]]).state(1)
        with self.assertRaises(ValueError):
            self._build(fx)

    def test_null_start_date_raises(self):
        fx = FixtureArchive().write([[match(1, start=None)]]).state(1)
        with self.assertRaises(ValueError):
            self._build(fx)

    def test_missing_unconsumed_contract_key_raises(self):
        # spec 1/5: the pinned contract is ALL observed keys, not just the
        # consumed subset - upstream dropping loser_team_id must stop the
        # build even though the parser never reads it.
        fx = FixtureArchive()
        bad = match(1)
        del bad["loser_team_id"]
        fx.write([[bad]]).state(1)
        with self.assertRaises(ValueError):
            self._build(fx)

    def test_unparseable_start_date_raises(self):
        # spec 5: NULL/unparseable start_date raises - a non-empty garbage
        # timestamp must not slip into ordering/joins/date windows.
        fx = FixtureArchive().write([[match(1, start="not-a-date")]]).state(1)
        with self.assertRaises(ValueError):
            self._build(fx)

    def test_malformed_page_raises(self):
        fx = FixtureArchive()
        with gzip.open(fx.dir / "matches_0000.jsonl.gz", "at") as f:
            f.write(json.dumps({"fetched_at": "x", "url": "x", "offset": 0,
                                "body": json.dumps({"nope": 1})}) + "\n")
        with self.assertRaises(ValueError):
            self._build(fx)


class TestDedup(unittest.TestCase):
    def test_duplicate_match_id_last_wins(self):
        fx = FixtureArchive()
        first = match(7, s1=2, s2=0)
        second = match(7, s1=2, s2=1)   # later fetch of the same match
        fx.write([[first], [second]], offsets=[0, 1]).state(2)
        db = fx.dir / "t.sqlite"
        build_db(db_path=db, data_dir=fx.dir, canonical_path=fx.canonical())
        con = sqlite3.connect(db)
        rows = con.execute(
            "SELECT team2_score FROM matches WHERE match_id=7").fetchall()
        con.close()
        self.assertEqual(rows, [(1,)])

    def test_duplicate_offset_last_fetch_wins(self):
        # Crash-window duplicate: same offset archived twice; only the later
        # line's rows may be yielded (verify_archive's rule, by construction).
        fx = FixtureArchive()
        fx.write([[match(1, s1=2, s2=0)]], offsets=[0])
        fx.write([[match(1, s1=2, s2=1)]],
                 chunk="matches_0001.jsonl.gz", offsets=[0])
        rows = [m for m, _ in iter_archive_rows(fx.dir)]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["team2_score"], 1)


class TestBuildAndReconcile(unittest.TestCase):
    def test_full_fixture_builds_clean_and_reconciles(self):
        fx = build_fixture()
        db = fx.dir / "t.sqlite"
        report = build_db(db_path=db, data_dir=fx.dir,
                          canonical_path=fx.canonical())
        self.assertEqual(report["rows_seen"], 90)
        self.assertEqual(report["unique_ids"], 90)
        self.assertEqual(report["quarantined_n"], 1)         # the bo2 draw
        self.assertEqual(report["reason_counts"], {"bo2_draw": 1})
        self.assertEqual(report["archive_next_offset"], 90)
        ok, failures = reconcile(report, db_path=db, data_dir=fx.dir)
        self.assertTrue(ok, failures)
        con = sqlite3.connect(db)
        self.assertEqual(con.execute(
            "SELECT value FROM parse_meta WHERE key='reconciled'"
        ).fetchone()[0], "true")
        self.assertEqual(con.execute(
            "SELECT team_id FROM canonical_alias WHERE canonical='T1'"
        ).fetchone()[0], 1)
        self.assertEqual(con.execute(
            "SELECT value FROM parse_meta WHERE key='parse_version'"
        ).fetchone()[0], PARSE_VERSION)
        con.close()

    def test_provenance_columns_populated(self):
        fx = build_fixture()
        db = fx.dir / "t.sqlite"
        build_db(db_path=db, data_dir=fx.dir, canonical_path=fx.canonical())
        con = sqlite3.connect(db)
        chunk, offset, fetched, ver = con.execute(
            "SELECT src_chunk, src_offset, fetched_at, parse_version "
            "FROM matches WHERE match_id=60").fetchone()
        con.close()
        self.assertEqual(chunk, "matches_0000.jsonl.gz")
        self.assertEqual(offset, 50)                  # second page
        self.assertTrue(fetched.startswith("2026-06-"))
        self.assertEqual(ver, PARSE_VERSION)

    def test_teams_latest_name_wins(self):
        fx = FixtureArchive()
        old = match(1, start="2025-01-01T00:00:00+00:00")
        old["team1"] = team(1, "OldName")
        new = match(2, start="2026-01-01T00:00:00+00:00")
        new["team1"] = team(1, "NewName")
        fx.write([[old, new]]).state(2)
        db = fx.dir / "t.sqlite"
        build_db(db_path=db, data_dir=fx.dir, canonical_path=fx.canonical())
        con = sqlite3.connect(db)
        name, seen = con.execute(
            "SELECT name, last_seen FROM teams WHERE team_id=1").fetchone()
        con.close()
        self.assertEqual(name, "NewName")
        self.assertEqual(seen, "2026-01-01T00:00:00+00:00")

    def test_rebuild_is_byte_identical(self):
        fx = build_fixture()
        canon = fx.canonical()
        dumps = []
        for name in ("a.sqlite", "b.sqlite"):
            db = fx.dir / name
            build_db(db_path=db, data_dir=fx.dir, canonical_path=canon)
            con = sqlite3.connect(db)
            dumps.append("\n".join(con.iterdump()))
            con.close()
        self.assertEqual(dumps[0], dumps[1])

    def test_conservation_catches_dropped_row(self):
        fx = build_fixture()
        db = fx.dir / "t.sqlite"
        report = build_db(db_path=db, data_dir=fx.dir,
                          canonical_path=fx.canonical())
        con = sqlite3.connect(db)
        con.execute("DELETE FROM matches WHERE match_id=5")
        con.commit()
        con.close()
        ok, failures = reconcile(report, db_path=db, data_dir=fx.dir)
        self.assertFalse(ok)
        self.assertTrue(any("conservation" in f for f in failures))

    def test_tripwire_no_bo2_draw_fails(self):
        fx = FixtureArchive()
        fx.write([[match(i, s1=2, s2=1) for i in range(1, 51)]],
                 offsets=[0]).state(50)
        db = fx.dir / "t.sqlite"
        report = build_db(db_path=db, data_dir=fx.dir,
                          canonical_path=fx.canonical())
        ok, failures = reconcile(report, db_path=db, data_dir=fx.dir)
        self.assertFalse(ok)
        self.assertTrue(any("bo2_draw" in f for f in failures))

    def test_tripwire_quarantine_rate_fails(self):
        fx = FixtureArchive()
        rows = ([match(i, s1=2, s2=1) for i in range(1, 41)]
                + [match(50 + i, bo=3, s1=1, s2=0) for i in range(10)]
                + [match(90, bo=2, s1=1, s2=1, winner=None)])
        fx.write([rows], offsets=[0]).state(len(rows))
        db = fx.dir / "t.sqlite"
        report = build_db(db_path=db, data_dir=fx.dir,
                          canonical_path=fx.canonical())
        ok, failures = reconcile(report, db_path=db, data_dir=fx.dir)
        self.assertFalse(ok)                      # 11/51 quarantined >> 2%
        self.assertTrue(any("rate" in f for f in failures))


def tournament(tid, name="Event", **over):
    t = {"id": tid, "name": name, "slug": f"e{tid}",
         "short_name": name[:4] if name else None,
         "start_date": "2026-01-01T00:00:00+00:00",
         "end_date": "2026-01-05T00:00:00+00:00", "tier": "s", "tier_rank": 1,
         "region_id": 1, "country_id": 2, "event_type": "lan",
         "event_scope": "international", "event_level": "major",
         "prize": 1000, "status": "finished", "pickem_presence": True}
    t.update(over)
    return t


def write_snapshot(path, sid, pages, totals=None):
    """Append one (possibly torn) snapshot: pages = list of row-lists."""
    n_before = 0
    with gzip.open(path, "at") as f:
        for i, rows in enumerate(pages):
            total = totals[i] if totals else sum(len(p) for p in pages)
            f.write(json.dumps({
                "snapshot_id": sid, "fetched_at": sid, "url": "fixture",
                "offset": n_before,
                "body": json.dumps({"total": {"count": total},
                                    "results": rows})}) + "\n")
            n_before += len(rows)


class TestTournamentsSnapshot(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.snaps = self.dir / "tournaments_snapshots.jsonl.gz"

    def tearDown(self):
        self._tmp.cleanup()

    def test_fetch_writes_complete_snapshot(self):
        rows = [tournament(i) for i in range(1, 6)]
        bodies = {0: json.dumps({"total": {"count": 5}, "results": rows[:3]}),
                  3: json.dumps({"total": {"count": 5}, "results": rows[3:]})}
        sid = fetch_tournaments(path=self.snaps,
                                fetch=lambda o: bodies[o],
                                sleep=lambda s: None, snapshot_id="S1")
        self.assertEqual(sid, "S1")
        got_sid, got = latest_complete_snapshot(self.snaps)
        self.assertEqual(got_sid, "S1")
        self.assertEqual([t["id"] for t in got], [1, 2, 3, 4, 5])

    def test_fetch_aborts_on_total_instability(self):
        bodies = {0: json.dumps({"total": {"count": 5},
                                 "results": [tournament(1)] * 3}),
                  3: json.dumps({"total": {"count": 6},   # shifted mid-fetch
                                 "results": [tournament(4)] * 2})}
        with self.assertRaises(ValueError):
            fetch_tournaments(path=self.snaps, fetch=lambda o: bodies[o],
                              sleep=lambda s: None, snapshot_id="S1")

    def test_latest_complete_wins_and_torn_tail_skipped(self):
        write_snapshot(self.snaps, "S1", [[tournament(1, name="OldRun")]])
        write_snapshot(self.snaps, "S2",
                       [[tournament(1, name="NewRun"), tournament(2)]])
        # S3 is torn: claims 5 rows, delivers 2 (crash after page 1)
        write_snapshot(self.snaps, "S3", [[tournament(1), tournament(2)]],
                       totals=[5])
        sid, rows = latest_complete_snapshot(self.snaps)
        self.assertEqual(sid, "S2")
        self.assertEqual(rows[0]["name"], "NewRun")

    def test_total_instability_within_snapshot_rejected_at_parse(self):
        write_snapshot(self.snaps, "S1",
                       [[tournament(1)], [tournament(2)]], totals=[2, 3])
        sid, rows = latest_complete_snapshot(self.snaps)
        self.assertIsNone(sid)


class TestParseTournaments(unittest.TestCase):
    def setUp(self):
        self._fx = build_fixture()          # matches fixture -> db
        self.dir = self._fx.dir
        self.db = self.dir / "t.sqlite"
        build_db(db_path=self.db, data_dir=self.dir,
                 canonical_path=self._fx.canonical())
        self.snaps = self.dir / "tournaments_snapshots.jsonl.gz"
        self.wl = self.dir / "whitelist.json"

    def _parse(self):
        return parse_tournaments(db_path=self.db, snapshots_path=self.snaps,
                                 whitelist_path=self.wl)

    def test_parse_loads_and_join_check_passes(self):
        # fixture matches all reference tournament_id=500
        write_snapshot(self.snaps, "S1", [[tournament(500)]])
        r = self._parse()
        self.assertTrue(r["ok"])
        self.assertEqual(r["tournaments_n"], 1)
        con = sqlite3.connect(self.db)
        self.assertEqual(con.execute(
            "SELECT name FROM tournaments WHERE tournament_id=500"
        ).fetchone()[0], "Event")
        self.assertEqual(con.execute(
            "SELECT value FROM parse_meta WHERE key='tournaments_snapshot_id'"
        ).fetchone()[0], "S1")
        con.close()

    def test_missing_id_fails_without_whitelist(self):
        # snapshot NEWER than all match fetches, without tid 500 -> deleted
        write_snapshot(self.snaps, "2026-06-30T00:00:00+00:00",
                       [[tournament(9)]])
        r = self._parse()
        self.assertFalse(r["ok"])
        self.assertEqual(r["failing_ids"], [500])

    def test_whitelist_skips_but_reports(self):
        write_snapshot(self.snaps, "2026-06-30T00:00:00+00:00",
                       [[tournament(9)]])
        self.wl.write_text(json.dumps({"whitelist": [
            {"id": 500, "evidence": "test", "date": "2026-07-02"}]}))
        r = self._parse()
        self.assertTrue(r["ok"])
        self.assertEqual(r["whitelisted_ids"], [500])

    def test_stale_snapshot_reports_not_fails(self):
        # snapshot OLDER than the matches' fetched_at -> stale, not deleted
        write_snapshot(self.snaps, "2026-01-01T00:00:00+00:00",
                       [[tournament(9)]])
        r = self._parse()
        self.assertTrue(r["ok"])
        self.assertEqual(r["stale_ids"], [500])
        self.assertEqual(r["failing_ids"], [])

    def test_null_name_reported_not_raised(self):
        write_snapshot(self.snaps, "S1",
                       [[tournament(500, name=None)]])
        r = self._parse()
        self.assertEqual(r["null_names"], 1)   # no IntegrityError

    def test_missing_contract_key_raises(self):
        bad = tournament(500)
        del bad["region_id"]
        write_snapshot(self.snaps, "S1", [[bad]])
        with self.assertRaises(ValueError):
            self._parse()

    def test_tournament_keys_match_probe(self):
        self.assertIn("pickem_presence", TOURNAMENT_KEYS)
        self.assertIn("region_id", TOURNAMENT_KEYS)


if __name__ == "__main__":
    unittest.main()
