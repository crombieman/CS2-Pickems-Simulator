"""calibration.py tests: row grading, manifest/adoption gate, append-only log,
snapshot regrade (W2 + W2-hardening).

Scoring is cross-checked against the hand values in test_postmortem_matches.py.
The manifest gate (P1a) and snapshot regrade (P2) were added after an external
review caught that unmanifested rows were adoptable and the named library
contracts were missing.
"""

import json
import os
import tempfile
import unittest

from calibration import (grade_match_row, grade_team_row, grade_team_table,
                         grade_event, grade_playoff_matches,
                         regrade_from_snapshot, regrade_playoffs_from_snapshot,
                         append_log, load_log, load_latest, summarize,
                         is_manifested)

CLOSE = {"p_a": 0.60, "flagged": False, "close_rule": "v1:test",
         "ts": "2026-06-11T08:00:00+00:00", "slug": "m1", "volume": 5000.0}
CLOSE_FLAGGED = {**CLOSE, "flagged": True}

# A valid lock contract (backfill mode, clean code, every load-bearing replay
# hash present). anchors_sha is required: a priced match's model prob IS the
# market anchor, so the anchors must be reproducible to replay the row.
MANIFEST = {"reconstruction_mode": "backfill_reconstructed",
            "code_sha": "abc1234", "ratings_sha": "def5678",
            "anchors_sha": "0011aabb", "code_dirty": False}
MANIFEST_DIRTY = {**MANIFEST, "code_dirty": True}

# A complete forward forecast lock contract (the W2b/live.py manifest shape).
FWD_MANIFEST = {"reconstruction_mode": "forecast_manifest",
                "code_sha": "abc1234", "ratings_sha": "def5678",
                "anchors_sha": "0011aabb", "pair_overrides_sha": "ccdd2233",
                "event_config_sha": "effeed009988", "code_dirty": False}


class TestGradeMatchRow(unittest.TestCase):
    def test_scoring_matches_hand_values(self):
        r = grade_match_row("FURIA", "B8", "FURIA", 0.8, {**CLOSE, "p_a": 0.6},
                            provenance=MANIFEST)
        self.assertEqual(r["kind"], "match")
        self.assertEqual(r["result"], 1.0)
        self.assertAlmostEqual(r["brier_model"], 0.04)    # (1-0.8)^2
        self.assertAlmostEqual(r["brier_market"], 0.16)   # (1-0.6)^2
        self.assertAlmostEqual(r["delta_brier"], 0.12)    # market - model

    def test_loser_side_is_complemented(self):
        r = grade_match_row("NAVI", "Spirit", "Spirit", 0.3, {**CLOSE, "p_a": 0.5},
                            provenance=MANIFEST)
        self.assertEqual(r["result"], 0.0)
        self.assertAlmostEqual(r["brier_model"], 0.09)    # (0.3-0)^2
        self.assertAlmostEqual(r["brier_market"], 0.25)   # (0.5-0)^2

    def test_missing_close_is_ungraded_market_side(self):
        r = grade_match_row("FURIA", "B8", "FURIA", 0.8, None, provenance=MANIFEST)
        self.assertIsNone(r["close_prob"])
        self.assertIsNone(r["brier_market"])
        self.assertIsNone(r["delta_brier"])
        self.assertFalse(r["adoption_eligible"])
        self.assertAlmostEqual(r["brier_model"], 0.04)   # model side still scored

    def test_close_volume_persisted(self):   # P3
        r = grade_match_row("FURIA", "B8", "FURIA", 0.8, {**CLOSE, "volume": 5000.0},
                            provenance=MANIFEST)
        self.assertEqual(r["close_volume"], 5000.0)

    def test_deterministic_regrade(self):
        args = ("FURIA", "B8", "FURIA", 0.8, {**CLOSE, "p_a": 0.6})
        self.assertEqual(grade_match_row(*args, provenance=MANIFEST),
                         grade_match_row(*args, provenance=MANIFEST))


class TestAdoptionEligibility(unittest.TestCase):   # P1a
    def test_manifested_unflagged_close_is_eligible(self):
        r = grade_match_row("A", "B", "A", 0.7, CLOSE, provenance=MANIFEST)
        self.assertTrue(r["manifested"])
        self.assertTrue(r["adoption_eligible"])

    def test_unmanifested_row_not_eligible(self):
        # real, unflagged close but NO lock contract -> not adoptable.
        r = grade_match_row("A", "B", "A", 0.7, CLOSE)   # no provenance
        self.assertFalse(r["manifested"])
        self.assertFalse(r["adoption_eligible"])

    def test_flagged_close_not_eligible(self):
        r = grade_match_row("A", "B", "A", 0.7, CLOSE_FLAGGED, provenance=MANIFEST)
        self.assertIsNotNone(r["brier_market"])          # still scored
        self.assertFalse(r["adoption_eligible"])

    def test_dirty_code_not_eligible(self):
        r = grade_match_row("A", "B", "A", 0.7, CLOSE, provenance=MANIFEST_DIRTY)
        self.assertFalse(r["manifested"])                # not replayable from a dirty tree
        self.assertFalse(r["adoption_eligible"])

    def test_is_manifested_contract(self):
        self.assertTrue(is_manifested(MANIFEST))
        self.assertTrue(is_manifested(FWD_MANIFEST))
        self.assertFalse(is_manifested(None))
        self.assertFalse(is_manifested({"reconstruction_mode": "backfill_reconstructed"}))  # no hashes
        self.assertFalse(is_manifested(MANIFEST_DIRTY))

    def test_backfill_missing_anchors_not_manifested(self):
        # The hole: a priced match's model prob IS the anchor, so a manifest
        # without anchors_sha cannot replay the row -> not adoptable.
        no_anchors = {k: v for k, v in MANIFEST.items() if k != "anchors_sha"}
        self.assertFalse(is_manifested(no_anchors))

    def test_forecast_manifest_missing_input_hash_not_manifested(self):
        # forward forecast missing one load-bearing input hash -> not replayable.
        no_pairov = {k: v for k, v in FWD_MANIFEST.items()
                     if k != "pair_overrides_sha"}
        self.assertFalse(is_manifested(no_pairov))

    def test_placeholder_hash_not_manifested(self):
        # A 'pending-*' placeholder in ANY hash field (e.g. a pre-W5 forward row
        # stamped event_config_sha='pending-w5') means it was never pinned.
        pending = {**FWD_MANIFEST, "event_config_sha": "pending-w5"}
        self.assertFalse(is_manifested(pending))

    def test_absent_dirty_marker_not_manifested(self):
        # No dirty marker = unknown provenance = honest pessimism -> not adoptable.
        no_marker = {k: v for k, v in MANIFEST.items() if k != "code_dirty"}
        self.assertFalse(is_manifested(no_marker))

    def test_unknown_reconstruction_mode_not_manifested(self):
        self.assertFalse(is_manifested({**MANIFEST, "reconstruction_mode": "guessed"}))

    def test_immutable_forecast_is_self_contained(self):
        # per-match prob logged at lock -> code_sha + clean tree is enough.
        self.assertTrue(is_manifested({"reconstruction_mode": "immutable_forecast",
                                       "code_sha": "abc1234", "code_dirty": False}))


class TestGradeTeamRow(unittest.TestCase):
    def test_market_free_brier(self):
        r = grade_team_row("Vitality", {"p30": 0.5, "padv": 0.4, "p03": 0.0}, (3, 0))
        self.assertEqual(r["kind"], "team")
        self.assertAlmostEqual(r["p30_brier"], 0.25)
        self.assertAlmostEqual(r["padv_brier"], 0.16)
        self.assertAlmostEqual(r["p03_brier"], 0.0)

    def test_table_stamps_event_and_version_so_versions_dont_collide(self):
        probs = {"Vitality": {"p30": 0.5, "padv": 0.4, "p03": 0.0}}
        rows = grade_team_table(probs, {"Vitality": [3, 0]}, "cologne", "v2", None)
        self.assertEqual(rows[0]["event"], "cologne")
        self.assertEqual(rows[0]["model_version"], "v2")


class TestAppendOnlyLog(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)

    def tearDown(self):
        os.remove(self.path)

    def test_superseding_row_wins_but_history_retained(self):
        v1 = {"kind": "match", "event": "c", "a": "FURIA", "b": "B8", "model_prob": 0.70}
        v2 = {"kind": "match", "event": "c", "a": "FURIA", "b": "B8", "model_prob": 0.74}
        append_log([v1], self.path)
        append_log([v2], self.path)
        self.assertEqual(len(load_log(self.path)), 2)        # nothing rewritten
        latest = load_latest(self.path)
        self.assertEqual(len(latest), 1)                     # superseded
        self.assertAlmostEqual(latest[0]["model_prob"], 0.74)

    def test_distinct_keys_coexist(self):
        rows = [
            {"kind": "match", "event": "c", "a": "FURIA", "b": "B8"},
            {"kind": "match", "event": "c", "a": "NAVI", "b": "Spirit"},
            {"kind": "team", "event": "c", "team": "Vitality"},
        ]
        append_log(rows, self.path)
        self.assertEqual(len(load_latest(self.path)), 3)


class TestSummarize(unittest.TestCase):
    def _rows(self):
        return [
            grade_match_row("A", "B", "A", 0.8, {**CLOSE, "p_a": 0.6}, provenance=MANIFEST),
            grade_match_row("C", "D", "C", 0.7, {**CLOSE_FLAGGED, "p_a": 0.55}, provenance=MANIFEST),  # flagged
            grade_match_row("E", "F", "E", 0.7, {**CLOSE, "p_a": 0.5}),  # unmanifested
            grade_team_row("Vitality", {"p30": 0.5, "padv": 0.4, "p03": 0.0}, (3, 0)),  # ignored
        ]

    def test_default_gate_excludes_flagged_and_unmanifested(self):   # P1a
        s = summarize(self._rows())
        self.assertEqual(s["n"], 1)                       # only the eligible one
        self.assertEqual(s["eligible_n"], 1)
        self.assertEqual(s["excluded_flagged_n"], 1)
        self.assertEqual(s["excluded_unmanifested_n"], 1)

    def test_require_manifest_false_for_crosscheck(self):
        # manifest-agnostic scoring view (what postmortem_matches computes).
        s = summarize(self._rows(), require_manifest=False)
        self.assertEqual(s["n"], 2)                       # both unflagged rows

    def test_include_flagged_opt_in(self):
        s = summarize(self._rows(), include_flagged=True)
        self.assertEqual(s["excluded_flagged_n"], 1)      # still reported


class TestRegradeFromSnapshot(unittest.TestCase):   # P2
    def test_regrade_reproduces(self):
        # Same immutable inputs -> identical rows (read-only; does not append).
        self.assertEqual(regrade_from_snapshot(), regrade_from_snapshot())

    def test_grade_event_is_pure(self):
        rows = grade_event("ev", [], {}, {}, [], MANIFEST, [], {})
        self.assertEqual(rows, [])


class TestGradePlayoffMatches(unittest.TestCase):
    """The playoff model prob must reconstruct the LOCK-TIME code path:
    anchor verbatim for priced pairs, win_prob for unpriced BO3, and the
    series_prob_bo5 conversion for the BO5 grand final."""
    RATINGS = {"A": 1600.0, "B": 1500.0, "C": 1400.0, "D": 1300.0}
    OVERRIDES = {("A", "B"): 0.7, ("B", "A"): 0.3}
    MATCHES = [
        {"a": "A", "b": "B", "winner": "A",
         "start": "2026-06-18T13:45:00+00:00", "round": "QF", "bo": 3},
        {"a": "A", "b": "C", "winner": "A",
         "start": "2026-06-20T13:45:00+00:00", "round": "SF", "bo": 3},
        {"a": "A", "b": "D", "winner": "D",
         "start": "2026-06-21T15:00:00+00:00", "round": "GF", "bo": 5},
    ]

    def _rows(self):
        return grade_playoff_matches(self.MATCHES, self.RATINGS, self.OVERRIDES,
                                     [], "ev-playoffs", "playoff-lock", MANIFEST)

    def test_anchored_qf_uses_market_prob_verbatim(self):
        qf = self._rows()[0]
        self.assertAlmostEqual(qf["model_prob"], 0.7)
        self.assertAlmostEqual(qf["market_prob"], 0.7)

    def test_unpriced_sf_uses_win_prob(self):
        from model import win_prob
        sf = self._rows()[1]
        self.assertAlmostEqual(sf["model_prob"], win_prob(self.RATINGS, "A", "C"))
        self.assertIsNone(sf["market_prob"])

    def test_bo5_final_converts_series_prob(self):
        from model import win_prob
        from playoffs import series_prob_bo5
        gf = self._rows()[2]
        p3 = win_prob(self.RATINGS, "A", "D")
        self.assertAlmostEqual(gf["model_prob"], series_prob_bo5(p3))
        self.assertGreater(gf["model_prob"], p3)   # favorite gains in BO5

    def test_rows_stamped_with_event_version_round(self):
        rows = self._rows()
        self.assertEqual([r["round"] for r in rows], ["QF", "SF", "GF"])
        self.assertTrue(all(r["event"] == "ev-playoffs" for r in rows))
        self.assertTrue(all(r["model_version"] == "playoff-lock" for r in rows))

    def test_no_close_is_not_adoption_eligible(self):
        # empty archive -> no close -> logged but never gate evidence.
        self.assertTrue(all(not r["adoption_eligible"] for r in self._rows()))

    def test_snapshot_regrade_covers_full_bracket_and_reproduces(self):
        rows = regrade_playoffs_from_snapshot()
        self.assertEqual(len(rows), 7)                        # 4 QF + 2 SF + GF
        self.assertEqual([r["round"] for r in rows],
                         ["QF", "QF", "QF", "QF", "SF", "SF", "GF"])
        self.assertEqual(rows, regrade_playoffs_from_snapshot())


if __name__ == "__main__":
    unittest.main()
