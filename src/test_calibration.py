"""calibration.py tests: row-level grading, append-only log, eligibility (W2).

Scoring is cross-checked against the hand values in test_postmortem_matches.py
so the extracted row-level math matches the existing print-path math.
"""

import json
import os
import tempfile
import unittest

from calibration import (grade_match_row, grade_team_row, grade_team_table,
                         append_log, load_log, load_latest, summarize)

CLOSE = {"p_a": 0.60, "flagged": False, "close_rule": "v1:test",
         "ts": "2026-06-11T08:00:00+00:00", "slug": "m1", "volume": 5000.0}
CLOSE_FLAGGED = {**CLOSE, "flagged": True}


class TestGradeMatchRow(unittest.TestCase):
    def test_scoring_matches_hand_values(self):
        # model 0.8 on the winner (a), close 0.6 on a -> same as the
        # test_postmortem_matches baseline.
        r = grade_match_row("FURIA", "B8", "FURIA", 0.8,
                            {**CLOSE, "p_a": 0.6})
        self.assertEqual(r["kind"], "match")
        self.assertEqual(r["result"], 1.0)
        self.assertAlmostEqual(r["brier_model"], 0.04)    # (1-0.8)^2
        self.assertAlmostEqual(r["brier_market"], 0.16)   # (1-0.6)^2
        self.assertAlmostEqual(r["delta_brier"], 0.12)    # market - model
        self.assertTrue(r["adoption_eligible"])

    def test_loser_side_is_complemented(self):
        # model put 0.3 on a (NAVI) but b (Spirit) won -> y=0.
        r = grade_match_row("NAVI", "Spirit", "Spirit", 0.3,
                            {**CLOSE, "p_a": 0.5})
        self.assertEqual(r["result"], 0.0)
        self.assertAlmostEqual(r["brier_model"], 0.09)    # (0.3-0)^2
        self.assertAlmostEqual(r["brier_market"], 0.25)   # (0.5-0)^2

    def test_missing_close_is_ungraded_market_side(self):
        r = grade_match_row("FURIA", "B8", "FURIA", 0.8, None)
        self.assertIsNone(r["close_prob"])
        self.assertIsNone(r["brier_market"])
        self.assertIsNone(r["delta_brier"])
        self.assertFalse(r["adoption_eligible"])
        # model side still scored
        self.assertAlmostEqual(r["brier_model"], 0.04)

    def test_flagged_close_excluded_from_adoption(self):
        r = grade_match_row("FURIA", "B8", "FURIA", 0.8,
                            {**CLOSE_FLAGGED, "p_a": 0.6})
        self.assertTrue(r["close_flagged"])
        self.assertIsNotNone(r["brier_market"])           # still scored
        self.assertFalse(r["adoption_eligible"])          # but not adoptable

    def test_deterministic_regrade(self):
        args = ("FURIA", "B8", "FURIA", 0.8, {**CLOSE, "p_a": 0.6})
        self.assertEqual(grade_match_row(*args), grade_match_row(*args))


class TestGradeTeamRow(unittest.TestCase):
    def test_market_free_brier(self):
        r = grade_team_row("Vitality", {"p30": 0.5, "padv": 0.4, "p03": 0.0},
                           (3, 0))
        self.assertEqual(r["kind"], "team")
        self.assertAlmostEqual(r["p30_brier"], 0.25)   # outcome 3-0 -> y=1
        self.assertAlmostEqual(r["padv_brier"], 0.16)  # not 3-1/3-2 -> y=0
        self.assertAlmostEqual(r["p03_brier"], 0.0)

    def test_table_stamps_event_and_version_so_versions_dont_collide(self):
        # Without event+model_version on team rows, v1/v2/v3 share a key and
        # silently supersede each other (the 48->16 collapse bug).
        probs = {"Vitality": {"p30": 0.5, "padv": 0.4, "p03": 0.0}}
        rows = grade_team_table(probs, {"Vitality": [3, 0]}, "cologne", "v2",
                                None)
        self.assertEqual(rows[0]["event"], "cologne")
        self.assertEqual(rows[0]["model_version"], "v2")


class TestAppendOnlyLog(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)

    def tearDown(self):
        os.remove(self.path)

    def test_superseding_row_wins_but_history_retained(self):
        v1 = {"kind": "match", "event": "cologne", "a": "FURIA", "b": "B8",
              "model_prob": 0.70}
        v2 = {"kind": "match", "event": "cologne", "a": "FURIA", "b": "B8",
              "model_prob": 0.74}   # correction: same key, new value
        append_log([v1], self.path)
        append_log([v2], self.path)
        self.assertEqual(len(load_log(self.path)), 2)        # nothing rewritten
        latest = load_latest(self.path)
        self.assertEqual(len(latest), 1)                     # superseded
        self.assertAlmostEqual(latest[0]["model_prob"], 0.74)

    def test_distinct_keys_coexist(self):
        rows = [
            {"kind": "match", "event": "cologne", "a": "FURIA", "b": "B8"},
            {"kind": "match", "event": "cologne", "a": "NAVI", "b": "Spirit"},
            {"kind": "team", "event": "cologne", "team": "Vitality"},
        ]
        append_log(rows, self.path)
        self.assertEqual(len(load_latest(self.path)), 3)


class TestSummarize(unittest.TestCase):
    def _rows(self):
        return [
            grade_match_row("FURIA", "B8", "FURIA", 0.8, {**CLOSE, "p_a": 0.6}),
            grade_match_row("MOUZ", "Legacy", "MOUZ", 0.7,
                            {**CLOSE_FLAGGED, "p_a": 0.55}),   # flagged
            grade_team_row("Vitality", {"p30": 0.5, "padv": 0.4, "p03": 0.0},
                           (3, 0)),                            # ignored by summary
        ]

    def test_default_excludes_flagged(self):
        s = summarize(self._rows())
        self.assertEqual(s["n"], 1)
        self.assertEqual(s["eligible_n"], 1)
        self.assertEqual(s["excluded_flagged_n"], 1)

    def test_include_flagged_opt_in(self):
        s = summarize(self._rows(), include_flagged=True)
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["excluded_flagged_n"], 1)   # still reported


if __name__ == "__main__":
    unittest.main()
