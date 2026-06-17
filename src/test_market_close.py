"""market_close tests: liquidity-floored pre-start close selection + provenance.

W1 (engine-correctness impl spec). The close-selection cases are carried over
from test_postmortem_matches.py as the regression baseline — with a `volume`
field added to every fixture row (the liquidity floor reads it; without it the
floor would filter them all to None). New cases cover the floor itself and the
surfaced rule version.
"""

import unittest

from market_close import CLOSE_RULE, close_row, close_snapshot

LIQUID = 5000.0   # >= fetch_anchors.MIN_VOLUME (1000)
THIN = 500.0      # <  MIN_VOLUME

# Carried-over baseline: all rows liquid, orientation varies on purpose.
ARCHIVE = [
    {"ts": "2026-06-11T06:00:00+00:00", "slug": "m1",
     "outcomes": ["FURIA", "B8"], "prices": [0.70, 0.30], "volume": LIQUID},
    {"ts": "2026-06-11T08:00:00+00:00", "slug": "m1",
     "outcomes": ["FURIA", "B8"], "prices": [0.74, 0.26], "volume": LIQUID},
    # Post-start (in-play) — must NOT be used when a pre-start row exists.
    {"ts": "2026-06-11T10:00:00+00:00", "slug": "m1",
     "outcomes": ["FURIA", "B8"], "prices": [0.95, 0.05], "volume": LIQUID},
    # Reversed orientation relative to the queried pair.
    {"ts": "2026-06-11T08:00:00+00:00", "slug": "m2",
     "outcomes": ["Natus Vincere", "Team Spirit"], "prices": [0.40, 0.60],
     "volume": LIQUID},
    # Only an in-play row exists for this one.
    {"ts": "2026-06-11T12:30:00+00:00", "slug": "m3",
     "outcomes": ["MOUZ", "Legacy"], "prices": [0.80, 0.20], "volume": LIQUID},
]


class TestCloseSnapshot(unittest.TestCase):
    # --- carried over from test_postmortem_matches.py (regression baseline) ---
    def test_last_pre_start_snapshot_wins(self):
        p, flagged = close_snapshot(ARCHIVE, "FURIA", "B8",
                                    "2026-06-11T09:00:00+00:00")
        self.assertAlmostEqual(p, 0.74)
        self.assertFalse(flagged)

    def test_orientation_follows_queried_pair(self):
        p, flagged = close_snapshot(ARCHIVE, "Spirit", "NAVI",
                                    "2026-06-11T09:00:00+00:00")
        self.assertAlmostEqual(p, 0.60)
        self.assertFalse(flagged)

    def test_no_pre_start_row_falls_back_flagged(self):
        p, flagged = close_snapshot(ARCHIVE, "MOUZ", "Legacy",
                                    "2026-06-11T12:00:00+00:00")
        self.assertAlmostEqual(p, 0.80)
        self.assertTrue(flagged)

    def test_unknown_pair_returns_none_two_tuple(self):
        # 2-tuple contract: bare-None would break `p, flagged = ...` unpack.
        p, flagged = close_snapshot(ARCHIVE, "G2", "FUT",
                                    "2026-06-11T09:00:00+00:00")
        self.assertIsNone(p)
        self.assertFalse(flagged)

    # --- new: liquidity floor (the only behavior W1 adds over closing_prob) ---
    def test_only_thin_rows_returns_none(self):
        archive = [{"ts": "2026-06-11T06:00:00+00:00", "slug": "x",
                    "outcomes": ["Aurora", "Monte"], "prices": [0.60, 0.40],
                    "volume": THIN}]
        p, flagged = close_snapshot(archive, "Aurora", "Monte",
                                    "2026-06-11T09:00:00+00:00")
        self.assertIsNone(p)
        self.assertFalse(flagged)

    def test_thin_pre_start_falls_back_to_later_liquid_flagged(self):
        archive = [
            {"ts": "2026-06-11T06:00:00+00:00", "slug": "x",
             "outcomes": ["Aurora", "Monte"], "prices": [0.60, 0.40],
             "volume": THIN},   # thin pre-start -> filtered out
            {"ts": "2026-06-11T12:30:00+00:00", "slug": "x",
             "outcomes": ["Aurora", "Monte"], "prices": [0.85, 0.15],
             "volume": LIQUID},  # liquid in-play -> flagged fallback
        ]
        p, flagged = close_snapshot(archive, "Aurora", "Monte",
                                    "2026-06-11T09:00:00+00:00")
        self.assertAlmostEqual(p, 0.85)
        self.assertTrue(flagged)

    def test_floor_skips_thin_last_pre_start_for_liquid_earlier(self):
        archive = [
            {"ts": "2026-06-11T06:00:00+00:00", "slug": "x",
             "outcomes": ["Aurora", "Monte"], "prices": [0.55, 0.45],
             "volume": LIQUID},  # liquid pre-start -> should win
            {"ts": "2026-06-11T08:00:00+00:00", "slug": "x",
             "outcomes": ["Aurora", "Monte"], "prices": [0.60, 0.40],
             "volume": THIN},    # thin last-pre-start -> filtered, NOT chosen
        ]
        p, flagged = close_snapshot(archive, "Aurora", "Monte",
                                    "2026-06-11T09:00:00+00:00")
        self.assertAlmostEqual(p, 0.55)
        self.assertFalse(flagged)


class TestCloseRowAndRule(unittest.TestCase):
    def test_rule_version_surfaced(self):
        self.assertEqual(CLOSE_RULE, "v1:last-mid-before-start+minvol")

    def test_close_row_carries_provenance(self):
        row = close_row(ARCHIVE, "FURIA", "B8", "2026-06-11T09:00:00+00:00")
        self.assertAlmostEqual(row["p_a"], 0.74)
        self.assertFalse(row["flagged"])
        self.assertEqual(row["close_rule"], CLOSE_RULE)
        self.assertEqual(row["ts"], "2026-06-11T08:00:00+00:00")
        self.assertEqual(row["slug"], "m1")
        self.assertEqual(row["volume"], LIQUID)

    def test_close_row_none_for_unknown_pair(self):
        self.assertIsNone(
            close_row(ARCHIVE, "G2", "FUT", "2026-06-11T09:00:00+00:00"))


if __name__ == "__main__":
    unittest.main()
