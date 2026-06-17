"""Calibration-log entry builder tests: structure, determinism, forward manifest."""

import json
import unittest

from live import build_log_entry, per_match_forecasts

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

    def test_manifest_and_forecasts_default_to_none(self):
        # Backward compatible: callers that don't pass them still round-trip.
        self.assertIsNone(self.entry["manifest"])
        self.assertIsNone(self.entry["match_forecasts"])


class TestForwardManifest(unittest.TestCase):   # W2b
    def test_entry_carries_manifest_and_forecasts(self):
        entry = build_log_entry(
            ts="t", ratings_source="ratings_locked_v3.json", ratings_sha="s",
            n_anchors=0, completed=[], upcoming=[("X", "Y")], p_pass=0.5,
            e_ticks=5.0, stats={}, rooting=[],
            manifest={"code_sha": "abc", "code_dirty": False,
                      "event_config_sha": "pending-w5", "n_sims": 40000, "seed": 11},
            match_forecasts=[{"a": "X", "b": "Y", "model_prob": 0.6,
                              "market_prob": None}])
        self.assertEqual(entry["manifest"]["code_sha"], "abc")
        self.assertEqual(entry["manifest"]["event_config_sha"], "pending-w5")
        self.assertEqual(entry["match_forecasts"][0]["model_prob"], 0.6)
        self.assertEqual(json.loads(json.dumps(entry)), entry)   # round-trips

    def test_per_match_forecasts_captures_immutable_model_prob(self):
        # Teams absent from PAIR_OVERRIDES -> model_prob is the rating-implied
        # prob, market_prob is None. 1000 vs 900 Elo -> ~0.64.
        rows = per_match_forecasts({"X": 1000.0, "Y": 900.0}, [("X", "Y")])
        self.assertEqual((rows[0]["a"], rows[0]["b"]), ("X", "Y"))
        self.assertAlmostEqual(rows[0]["model_prob"], 0.640065, places=4)
        self.assertIsNone(rows[0]["market_prob"])


class TestForecastManifestEventConfig(unittest.TestCase):   # W5
    def test_event_config_sha_is_real_hash_not_placeholder(self):
        # W5 replaced the 'pending-w5' placeholder with the actual hash of the
        # per-event config, so a forward forecast is replayable against the
        # teams/seeds/scoring it was made under.
        from live import forecast_manifest
        m = forecast_manifest("ratings_fitted.json", "deadbeef", "none", None)
        self.assertNotEqual(m["event_config_sha"], "pending-w5")
        self.assertEqual(len(m["event_config_sha"]), 12)   # _sha() -> 12-char hex


if __name__ == "__main__":
    unittest.main()
