"""Per-match postmortem tests: closing-line selection and scoring, synthetic data."""

import unittest

from postmortem_matches import closing_prob, grade_matches

# Archive rows: (ts, outcomes, prices) — orientation varies on purpose.
ARCHIVE = [
    {"ts": "2026-06-11T06:00:00+00:00", "slug": "m1",
     "outcomes": ["FURIA", "B8"], "prices": [0.70, 0.30]},
    {"ts": "2026-06-11T08:00:00+00:00", "slug": "m1",
     "outcomes": ["FURIA", "B8"], "prices": [0.74, 0.26]},
    # Post-start row (in-play) — must NOT be used when a pre-start row exists.
    {"ts": "2026-06-11T10:00:00+00:00", "slug": "m1",
     "outcomes": ["FURIA", "B8"], "prices": [0.95, 0.05]},
    # Reversed orientation relative to the queried pair.
    {"ts": "2026-06-11T08:00:00+00:00", "slug": "m2",
     "outcomes": ["Natus Vincere", "Team Spirit"], "prices": [0.40, 0.60]},
    # Only an in-play row exists for this one.
    {"ts": "2026-06-11T12:30:00+00:00", "slug": "m3",
     "outcomes": ["MOUZ", "Legacy"], "prices": [0.80, 0.20]},
]


class TestClosingProb(unittest.TestCase):
    def test_last_pre_start_snapshot_wins(self):
        p, flagged = closing_prob(ARCHIVE, "FURIA", "B8",
                                  "2026-06-11T09:00:00+00:00")
        self.assertAlmostEqual(p, 0.74)
        self.assertFalse(flagged)

    def test_orientation_follows_queried_pair(self):
        p, flagged = closing_prob(ARCHIVE, "Spirit", "NAVI",
                                  "2026-06-11T09:00:00+00:00")
        self.assertAlmostEqual(p, 0.60)
        self.assertFalse(flagged)

    def test_no_pre_start_row_falls_back_flagged(self):
        p, flagged = closing_prob(ARCHIVE, "MOUZ", "Legacy",
                                  "2026-06-11T12:00:00+00:00")
        self.assertAlmostEqual(p, 0.80)
        self.assertTrue(flagged)

    def test_unknown_pair_returns_none(self):
        p, flagged = closing_prob(ARCHIVE, "G2", "FUT",
                                  "2026-06-11T09:00:00+00:00")
        self.assertIsNone(p)


class TestGradeMatches(unittest.TestCase):
    def test_brier_and_log_scores(self):
        # One match, model 0.8 on the winner, market 0.6 on the winner.
        rows = [{"a": "FURIA", "b": "B8", "winner": "FURIA",
                 "p_model": 0.8, "p_market": 0.6, "flagged": False}]
        g = grade_matches(rows)
        self.assertAlmostEqual(g["model"]["brier"], 0.04)   # (1-0.8)^2
        self.assertAlmostEqual(g["market"]["brier"], 0.16)  # (1-0.6)^2
        self.assertAlmostEqual(g["delta_brier"], 0.12)      # market - model
        self.assertEqual(g["n"], 1)

    def test_loser_side_probability_is_complemented(self):
        # Model put 0.3 on a, but b won: brier vs the b-win event = (0.7-1)^2.
        rows = [{"a": "NAVI", "b": "Spirit", "winner": "Spirit",
                 "p_model": 0.3, "p_market": 0.5, "flagged": False}]
        g = grade_matches(rows)
        self.assertAlmostEqual(g["model"]["brier"], 0.09)
        self.assertAlmostEqual(g["market"]["brier"], 0.25)


if __name__ == "__main__":
    unittest.main()
