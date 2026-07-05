"""integrity_audit tests (W4): promotion classifier, impossible-record scan,
tier cross-check, reference orientation cross-check, CSV coverage cross-check,
orchestration (flags table + parse_meta + idempotency + fail-loud exit).

Fixture DBs are built through bo3gg_parse.build_db on crafted archives (reusing
test_bo3gg_parse helpers) so the audit is exercised against the real substrate
schema, not a hand-rolled one."""

import json
import sqlite3
import unittest

from bo3gg_parse import build_db
from test_bo3gg_parse import FixtureArchive, match, team, tournament

from integrity_audit import (AUDIT_VERSION, classify_promotions,
                             cross_check_csv, cross_check_reference,
                             run_audit, scan_impossible, tier_cross_check)


def alias(canonical, team_id):
    return {"canonical": canonical, "team_id": team_id, "source": "test"}


class DbCase(unittest.TestCase):
    """Builds fixture substrates and closes every connection on teardown
    (an open sqlite handle blocks tmpdir removal on Windows)."""

    def build(self, rows, aliases=None):
        fx = FixtureArchive()
        fx.write([rows], offsets=[0]).state(len(rows))
        db = fx.dir / "t.sqlite"
        build_db(db_path=db, data_dir=fx.dir,
                 canonical_path=fx.canonical(aliases if aliases is not None
                                             else []))
        self._fx = fx                      # keep the tmp dir alive
        self.addCleanup(fx.tmp.cleanup)
        return db, self.connect(db)

    def connect(self, db):
        con = sqlite3.connect(db)
        self.addCleanup(con.close)         # LIFO: closes before tmp cleanup
        return con


def flags(con, flag=None):
    q = "SELECT match_id, flag, severity FROM audit_flags"
    if flag:
        return con.execute(q + " WHERE flag=?", (flag,)).fetchall()
    return con.execute(q).fetchall()


class TestPromotionClassifier(DbCase):
    """W3 spec 5 / master W4: score-keyed promotion out of score_bo_mismatch.
    winner_score >= 2 AND winner_score > loser_score PROVES multi-map
    (inferred_multi_map); forfeit signatures (1-0, 0-0...) stay quarantined."""

    def test_score_proves_multi_map(self):
        rows = [match(1, bo=1, s1=2, s2=0),           # mislabeled BO1
                match(2, bo=3, s1=3, s2=0),           # mislabeled BO3
                match(3, bo=1, s1=2, s2=1)]           # BO1 scored 2-1
        db, con = self.build(rows)
        classify_promotions(con)
        got = {mid for mid, _, _ in flags(con, "inferred_multi_map")}
        self.assertEqual(got, {1, 2, 3})

    def test_forfeit_signatures_stay_quarantined(self):
        rows = [match(1, bo=3, s1=1, s2=0),           # abandon with a winner
                match(2, bo=3, s1=0, s2=0)]           # 0-0 with a winner
        db, con = self.build(rows)
        classify_promotions(con)
        got = {mid for mid, _, _ in flags(con, "forfeit_signature")}
        self.assertEqual(got, {1, 2})
        self.assertEqual(flags(con, "inferred_multi_map"), [])

    def test_winner_orientation_respected(self):
        # team2 is the winner: 0-2 on a "BO1" must promote (ws=2), while
        # 0-1 on a BO3 must not (ws=1) - scores are winner-oriented.
        rows = [match(1, bo=1, s1=0, s2=2, winner="t2"),
                match(2, bo=3, s1=0, s2=1, winner="t2")]
        db, con = self.build(rows)
        classify_promotions(con)
        self.assertEqual(flags(con, "inferred_multi_map"), [(1, "inferred_multi_map", "report")])
        self.assertEqual([m for m, _, _ in flags(con, "forfeit_signature")], [2])

    def test_conservation_every_mismatch_row_flagged_once(self):
        rows = [match(1, bo=1, s1=2, s2=0), match(2, bo=3, s1=1, s2=0),
                match(3, s1=2, s2=1),                  # clean - untouched
                match(4, bo=2, s1=1, s2=1, winner=None)]  # bo2_draw - untouched
        db, con = self.build(rows)
        n = classify_promotions(con)
        mismatch_n = con.execute(
            "SELECT COUNT(*) FROM matches "
            "WHERE quarantine_reason='score_bo_mismatch'").fetchone()[0]
        self.assertEqual(n["inferred_multi_map"] + n["forfeit_signature"],
                         mismatch_n)
        flagged = [mid for mid, _, _ in flags(con)]
        self.assertEqual(sorted(flagged), [1, 2])      # 3 and 4 untouched


class TestScanImpossible(DbCase):
    def test_self_play_fails(self):
        rows = [match(1, t1=5, t2=5, winner="t1")]     # passes validate_row
        db, con = self.build(rows)
        scan_impossible(con)
        self.assertEqual(flags(con, "self_play"), [(1, "self_play", "fail")])

    def test_negative_score_fails(self):
        rows = [match(1, s1=2, s2=-1)]                 # clean per rule 6
        db, con = self.build(rows)
        scan_impossible(con)
        self.assertEqual([m for m, _, _ in flags(con, "negative_score")], [1])

    def test_duplicate_pairing_reported(self):
        same = "2026-03-01T10:00:00+00:00"
        rows = [match(1, t1=7, t2=8, start=same),
                match(2, t1=8, t2=7, start=same),      # unordered pair match
                match(3, t1=7, t2=8, start="2026-04-01T10:00:00+00:00")]
        db, con = self.build(rows)
        scan_impossible(con)
        got = flags(con, "duplicate_pairing")
        self.assertEqual(sorted(m for m, _, _ in got), [1, 2])
        self.assertTrue(all(sev == "report" for _, _, sev in got))

    def test_end_before_start_reported(self):
        rows = [match(1, end_date="2025-12-31T00:00:00+00:00")]
        db, con = self.build(rows)
        scan_impossible(con)
        self.assertEqual([m for m, _, _ in flags(con, "date_order")], [1])

    def test_clean_db_no_flags(self):
        rows = [match(1, t1=1, t2=2), match(2, t1=3, t2=4)]
        db, con = self.build(rows)
        scan_impossible(con)
        self.assertEqual(flags(con), [])


class TestTierCrossCheck(DbCase):
    def _with_tournament(self, rows, tier):
        db, con = self.build(rows)
        t = tournament(500, tier=tier)
        con.execute(
            "INSERT INTO tournaments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (t["id"], t["name"], t["slug"], t["short_name"], t["start_date"],
             t["end_date"], t["tier"], t["tier_rank"], t["region_id"],
             t["country_id"], t["event_type"], t["event_scope"],
             t["event_level"], t["prize"], t["status"],
             json.dumps(t["pickem_presence"]), "S1", "test"))
        con.commit()
        return con

    def test_divergence_reported(self):
        # fixture matches carry tier 's'; tournament row says 'a'
        con = self._with_tournament([match(1)], tier="a")
        div = tier_cross_check(con)
        self.assertEqual(div, [(500, "s", "a")])
        got = con.execute("SELECT flag, severity FROM audit_flags").fetchall()
        self.assertIn(("tier_divergence", "report"), got)

    def test_agreement_silent(self):
        con = self._with_tournament([match(1)], tier="s")
        self.assertEqual(tier_cross_check(con), [])

    def test_tournament_absent_skipped(self):
        db, con = self.build([match(1)])   # tournaments table empty
        self.assertEqual(tier_cross_check(con), [])


REF_ALIASES = [alias("Alpha", 21), alias("Bravo", 22), alias("Charlie", 23),
               alias("Delta", 24)]


def ref_rows():
    return [{"a": "Alpha", "b": "Bravo", "winner": "Alpha",
             "start": "2026-06-11T09:00:00+00:00"},
            {"a": "Charlie", "b": "Delta", "winner": "Delta",
             "start": "2026-06-11T12:00:00+00:00"}]


def ref_db_rows():
    return [match(1, t1=21, t2=22, s1=2, s2=0, winner="t1",
                  start="2026-06-11T10:30:00+00:00"),   # same day, hours off
            match(2, t1=23, t2=24, s1=1, s2=2, winner="t2",
                  start="2026-06-11T12:00:00+00:00")]


class TestReferenceCrossCheck(DbCase):
    def test_all_verified_with_coverage(self):
        db, con = self.build(ref_db_rows(), REF_ALIASES)
        r = cross_check_reference(con, ref_rows())
        self.assertEqual(r["verified"], 2)
        self.assertEqual(r["failures"], [])
        self.assertEqual(r["coverage"]["Alpha"], {"expected": 1, "found": 1})
        self.assertEqual(flags(con), [])

    def test_swapped_winner_caught(self):
        rows = ref_db_rows()
        rows[1] = match(2, t1=23, t2=24, s1=2, s2=1, winner="t1",
                        start="2026-06-11T12:00:00+00:00")  # ref says Delta
        db, con = self.build(rows, REF_ALIASES)
        r = cross_check_reference(con, ref_rows())
        self.assertEqual(r["verified"], 1)
        self.assertEqual([m for m, _, _ in flags(con, "orientation_mismatch")],
                         [2])

    def test_missing_reference_match_caught(self):
        db, con = self.build(ref_db_rows()[:1], REF_ALIASES)
        r = cross_check_reference(con, ref_rows())
        got = flags(con, "reference_missing")
        self.assertEqual(len(got), 1)
        self.assertIsNone(got[0][0])                   # no match_id to point at
        self.assertEqual(got[0][2], "fail")
        self.assertEqual(r["coverage"]["Charlie"], {"expected": 1, "found": 0})

    def test_unresolvable_canonical_raises(self):
        # a reference row naming a team with no alias is contract drift in OUR
        # committed files, not upstream data - fail loud, never guess.
        db, con = self.build(ref_db_rows(), REF_ALIASES[:2])
        with self.assertRaises(ValueError):
            cross_check_reference(con, ref_rows())


class TestCsvCrossCheck(DbCase):
    def _csv(self, winner="Alpha", loser="Bravo"):
        return [{"winner": winner, "loser": loser, "event": "test_ev",
                 "note": "bo3"}]

    def test_verified(self):
        db, con = self.build(ref_db_rows(), REF_ALIASES)
        r = cross_check_csv(con, self._csv())
        self.assertEqual(r["verified"], 1)
        self.assertEqual(flags(con), [])

    def test_flipped_winner_caught(self):
        db, con = self.build(ref_db_rows(), REF_ALIASES)
        r = cross_check_csv(con, self._csv(winner="Bravo", loser="Alpha"))
        got = flags(con, "csv_orientation_mismatch")
        self.assertEqual([m for m, _, _ in got], [1])
        self.assertEqual(got[0][2], "fail")

    def test_unmatched_row_fails(self):
        db, con = self.build(ref_db_rows(), REF_ALIASES)
        r = cross_check_csv(con, self._csv(winner="Alpha", loser="Delta"))
        self.assertEqual(len(flags(con, "csv_row_unmatched")), 1)

    def test_unresolvable_team_skipped_and_reported(self):
        db, con = self.build(ref_db_rows(), REF_ALIASES)
        r = cross_check_csv(con, self._csv(winner="Alpha", loser="NoAlias"))
        self.assertEqual(r["unresolvable"], 1)
        self.assertEqual(flags(con), [])               # skip, never guess

    def test_ambiguous_rematch_reported_not_checked(self):
        rows = ref_db_rows() + [match(3, t1=22, t2=21, s1=2, s2=1, winner="t1",
                                      start="2026-06-20T10:00:00+00:00")]
        db, con = self.build(rows, REF_ALIASES)
        r = cross_check_csv(con, self._csv())
        self.assertEqual(r["ambiguous"], 1)
        got = flags(con, "csv_ambiguous")
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0][2], "report")

    def test_only_quarantined_hit_reported(self):
        rows = [match(1, t1=21, t2=22, bo=3, s1=1, s2=0,
                      start="2026-06-11T10:30:00+00:00")]  # score_bo_mismatch
        db, con = self.build(rows, REF_ALIASES)
        r = cross_check_csv(con, self._csv())
        got = flags(con, "csv_match_quarantined")
        self.assertEqual([m for m, _, _ in got], [1])
        self.assertEqual(got[0][2], "report")


class TestRunAudit(DbCase):
    def _run(self, rows, aliases=None, reference=(), csv_rows=()):
        db, con = self.build(rows, aliases)
        con.close()
        ok, report = run_audit(db, reference_rows=list(reference),
                               csv_rows=list(csv_rows))
        return db, ok, report

    def test_clean_audit_ok_and_meta(self):
        db, ok, report = self._run(ref_db_rows(), REF_ALIASES,
                                   reference=ref_rows())
        self.assertTrue(ok)
        con = self.connect(db)
        self.assertEqual(con.execute(
            "SELECT value FROM parse_meta WHERE key='audit_version'"
        ).fetchone()[0], AUDIT_VERSION)
        self.assertEqual(con.execute(
            "SELECT value FROM parse_meta WHERE key='audit_ok'"
        ).fetchone()[0], "true")
        con.close()
        self.assertEqual(report["reference"]["verified"], 2)

    def test_fail_finding_fails_audit(self):
        rows = ref_db_rows() + [match(9, t1=5, t2=5, winner="t1",
                                      start="2026-05-05T10:00:00+00:00")]
        db, ok, report = self._run(rows, REF_ALIASES,
                                   reference=ref_rows())
        self.assertFalse(ok)
        con = self.connect(db)
        self.assertEqual(con.execute(
            "SELECT value FROM parse_meta WHERE key='audit_ok'"
        ).fetchone()[0], "false")
        con.close()

    def test_report_only_findings_stay_ok(self):
        rows = [match(1, bo=3, s1=1, s2=0)]            # forfeit_signature only
        db, ok, report = self._run(rows)
        self.assertTrue(ok)
        self.assertEqual(report["promotions"]["forfeit_signature"], 1)

    def test_idempotent_rerun_identical(self):
        db, con = self.build(ref_db_rows(), REF_ALIASES)
        con.close()
        dumps = []
        for _ in range(2):
            run_audit(db, reference_rows=ref_rows(), csv_rows=[])
            con = sqlite3.connect(db)
            dumps.append("\n".join(con.iterdump()))
            con.close()
        self.assertEqual(dumps[0], dumps[1])

    def test_flags_carry_audit_version(self):
        db, ok, report = self._run([match(1, bo=1, s1=2, s2=0)])
        con = self.connect(db)
        vers = con.execute(
            "SELECT DISTINCT audit_version FROM audit_flags").fetchall()
        con.close()
        self.assertEqual(vers, [(AUDIT_VERSION,)])


if __name__ == "__main__":
    unittest.main()
