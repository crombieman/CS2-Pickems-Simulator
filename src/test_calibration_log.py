"""Calibration-log entry builder tests: structure and determinism."""

import json
import unittest

from live import build_log_entry

STATS = {"Vitality": {"p30": 0.0, "padv": 0.88, "pany": 0.88, "p03": 0.0}}
ROOTING = [(0.378, "Aurora", "Spirit", 0.101, 0.479)]


class TestBuildLogEntry(unittest.TestCase):
    def setUp(self):
        self.entry = build_log_entry(
            ts="2026-06-12T21:00:00+00:00",
            ratings_source="ratings_fitted.json",
            ratings_sha="abc123",
            n_anchors=2,
            completed=[("Spirit", "NAVI")],
            upcoming=[("Aurora", "Spirit")],
            p_pass=0.343, e_ticks=4.01,
            stats=STATS, rooting=ROOTING)

    def test_required_keys_present(self):
        for key in ("ts", "ratings_source", "ratings_sha", "n_anchors",
                    "n_completed", "upcoming", "p_pass", "e_ticks",
                    "teams", "rooting"):
            self.assertIn(key, self.entry)

    def test_values_round_trip_json(self):
        rt = json.loads(json.dumps(self.entry))
        self.assertEqual(rt, self.entry)
        self.assertEqual(rt["n_completed"], 1)
        self.assertEqual(rt["upcoming"], [["Aurora", "Spirit"]])
        self.assertAlmostEqual(rt["p_pass"], 0.343)

    def test_rooting_rows_labeled(self):
        r = self.entry["rooting"][0]
        self.assertEqual((r["a"], r["b"]), ("Aurora", "Spirit"))
        self.assertAlmostEqual(r["p_pass_a"], 0.101)
        self.assertAlmostEqual(r["p_pass_b"], 0.479)
        self.assertAlmostEqual(r["swing"], 0.378)


if __name__ == "__main__":
    unittest.main()
