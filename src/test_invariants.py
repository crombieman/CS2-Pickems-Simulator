"""Invariant + regression tests. Run: python -m unittest src.test_invariants -v
(or: cd src && python -m unittest test_invariants -v)

Guards the locked pipeline: any refactor of simulate.py must leave the
committed stage3_probs.json exactly reproducible (seed 11, 40k sims).
"""

import collections
import json
import random
import unittest
from pathlib import Path

from model import (STAGE3_TEAMS, load_pair_overrides, win_prob)
from simulate import PAIR_OVERRIDES, ROUND1, match_prob, run, simulate_stage

DATA = Path(__file__).resolve().parent.parent / "data"

# A 16-team Swiss always produces exactly this record multiset:
# R3 advances 2 from the 2-0 group, eliminates 2 from 0-2, etc.
SWISS_RECORD_COUNTS = {(3, 0): 2, (3, 1): 3, (3, 2): 3,
                       (2, 3): 3, (1, 3): 3, (0, 3): 2}


class TestSwissInvariants(unittest.TestCase):
    def test_record_distribution(self):
        ratings = json.load(open(DATA / "ratings_fitted.json"))
        rng = random.Random(99)
        for _ in range(500):
            result = simulate_stage(ratings, rng)
            counts = collections.Counter(result.values())
            self.assertEqual(dict(counts), SWISS_RECORD_COUNTS)
            self.assertEqual(set(result), set(STAGE3_TEAMS))

    def test_round1_pairings_are_disjoint_16(self):
        teams = [t for m in ROUND1 for t in m]
        self.assertEqual(sorted(teams), sorted(STAGE3_TEAMS))


class TestOverrides(unittest.TestCase):
    def test_anchored_pairs_play_market_prob_both_orientations(self):
        ratings = json.load(open(DATA / "ratings_fitted.json"))
        anchors = json.load(open(DATA / "market_anchors.json"))["anchors"]
        for anc in anchors:
            a, b, p = anc["a"], anc["b"], anc["p"]
            self.assertAlmostEqual(match_prob(ratings, a, b), p, places=12)
            self.assertAlmostEqual(match_prob(ratings, b, a), 1 - p, places=12)

    def test_non_anchored_pair_uses_ratings(self):
        ratings = json.load(open(DATA / "ratings_fitted.json"))
        # Vitality-Spirit has no anchor
        self.assertNotIn(("Vitality", "Spirit"), PAIR_OVERRIDES)
        self.assertAlmostEqual(match_prob(ratings, "Vitality", "Spirit"),
                               win_prob(ratings, "Vitality", "Spirit"), places=12)

    def test_anchor_pairs_disjoint(self):
        seen = set()
        for anc in json.load(open(DATA / "market_anchors.json"))["anchors"]:
            for t in (anc["a"], anc["b"]):
                self.assertNotIn(t, seen, f"{t} appears in two anchors")
                seen.add(t)


class TestLockedRegression(unittest.TestCase):
    """The committed stage3_probs.json (v3 lock) must reproduce exactly."""

    def test_probs_reproduce(self):
        ratings = json.load(open(DATA / "ratings_fitted.json"))
        locked = json.load(open(DATA / "stage3_probs.json"))
        _, stats = run(ratings, n_sims=locked["meta"]["n_sims"],
                       seed=locked["meta"]["seed"])
        for t, want in locked["probs"].items():
            for k, v in want.items():
                self.assertAlmostEqual(stats[t][k], v, places=9,
                                       msg=f"{t}.{k} drifted from locked value")


if __name__ == "__main__":
    unittest.main()
