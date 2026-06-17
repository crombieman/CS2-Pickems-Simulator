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
from simulate import (PAIR_OVERRIDES, ROUND1, _pair_group, make_state,
                      match_prob, run, simulate_stage)

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


class TestPriorityTable6(unittest.TestCase):
    """Valve's 15-row table for 6-team groups (R4/R5)."""

    GROUP = ["Vitality", "NAVI", "Falcons", "MongolZ", "Aurora", "FURIA"]
    BUCH = {t: 0 for t in GROUP}  # equal Buchholz -> order = initial seed

    def test_no_rematches_uses_priority_1(self):
        played = {t: set() for t in self.GROUP}
        pairs = _pair_group(self.GROUP, played, self.BUCH)
        # priority 1: 1v6, 2v5, 3v4 in seed order
        self.assertEqual(pairs, [("Vitality", "FURIA"), ("NAVI", "Aurora"),
                                 ("Falcons", "MongolZ")])

    def test_rematch_skips_to_next_row(self):
        played = {t: set() for t in self.GROUP}
        # block 1v6 (Vitality-FURIA): rows 1-6 all pair... rows starting
        # (1,6): rows 1,2 and 7. Row 3 is 1v5,2v6,3v4.
        played["Vitality"].add("FURIA")
        played["FURIA"].add("Vitality")
        pairs = _pair_group(self.GROUP, played, self.BUCH)
        self.assertEqual(pairs, [("Vitality", "Aurora"), ("NAVI", "FURIA"),
                                 ("Falcons", "MongolZ")])

    def test_diverges_from_greedy_on_fallback(self):
        """The case that motivated the table: greedy backtracking and the
        rulebook can disagree when constraints bind. Block 1v6 AND 1v5:
        table row 5 gives 1v4, 2v6, 3v5; greedy would also try 1v4 but
        then pairs 2v6? — assert the table's exact output regardless."""
        played = {t: set() for t in self.GROUP}
        for a, b in (("Vitality", "FURIA"), ("Vitality", "Aurora")):
            played[a].add(b)
            played[b].add(a)
        pairs = _pair_group(self.GROUP, played, self.BUCH)
        self.assertEqual(pairs, [("Vitality", "MongolZ"), ("NAVI", "FURIA"),
                                 ("Falcons", "Aurora")])


class TestMakeStateValidation(unittest.TestCase):
    def test_unknown_team_rejected(self):
        with self.assertRaises(ValueError) as cm:
            make_state([("Vitality", "Mongolz")])  # wrong capitalization
        self.assertIn("unknown team", str(cm.exception))

    def test_duplicate_pairing_rejected(self):
        with self.assertRaises(ValueError) as cm:
            make_state([("Vitality", "FUT"), ("FUT", "Vitality")])
        self.assertIn("twice", str(cm.exception))

    def test_impossible_record_rejected(self):
        bad = [("Vitality", "FUT"), ("Vitality", "G2"),
               ("Vitality", "B8"), ("Vitality", "Monte")]  # 4 wins
        with self.assertRaises(ValueError) as cm:
            make_state(bad, [("NAVI", "Spirit")])
        self.assertIn("impossible", str(cm.exception))

    def test_midround_without_upcoming_rejected(self):
        with self.assertRaises(AssertionError):
            make_state([("Vitality", "FUT")])  # 14 teams at 0 games


class TestLockedRegression(unittest.TestCase):
    """The committed stage3_probs.json (v3 lock) must reproduce exactly.

    The locked tables were generated under the pre-fix seed order
    (LEGACY_SEED); pin it for this test so the frozen artifacts stay
    reproducible after the 2026-06-12 seed correction.
    """

    def test_probs_reproduce(self):
        import simulate
        # ratings_locked_v3.json is the frozen INPUT that generated the
        # locked probs; ratings_fitted.json is the living pipeline output
        # and moves with the dataset (post-audit). Never conflate them.
        ratings = json.load(open(DATA / "ratings_locked_v3.json"))
        locked = json.load(open(DATA / "stage3_probs.json"))
        orig_seed, orig_table = simulate.SEED, simulate.USE_PRIORITY_TABLE
        simulate.SEED = simulate.LEGACY_SEED
        simulate.USE_PRIORITY_TABLE = False  # locked tables predate the table
        try:
            _, stats = run(ratings, n_sims=locked["meta"]["n_sims"],
                           seed=locked["meta"]["seed"])
        finally:
            simulate.SEED, simulate.USE_PRIORITY_TABLE = orig_seed, orig_table
        for t, want in locked["probs"].items():
            for k, v in want.items():
                self.assertAlmostEqual(stats[t][k], v, places=9,
                                       msg=f"{t}.{k} drifted from locked value")


class TestEventConfigWiring(unittest.TestCase):
    """W5: the reproducibility-critical event facts must be sourced from the
    Cologne event config (data/events/cologne_major.json), not re-hardcoded.
    If these drift from the config, the locked tables + ratings_fitted.json
    stop reproducing — so pin the wiring, not just the values."""

    def test_simulate_facts_come_from_cologne_config(self):
        import simulate
        from event_config import COLOGNE
        self.assertEqual(simulate.ROUND1, COLOGNE.round1)
        self.assertEqual(simulate.SEED, COLOGNE.seeds)
        self.assertEqual(STAGE3_TEAMS, COLOGNE.teams)

    def test_legacy_seed_still_derives_from_team_order(self):
        # TestLockedRegression replays under LEGACY_SEED; it must stay the
        # enumerate-of-team-order map the locked tables were generated with.
        import simulate
        self.assertEqual(simulate.LEGACY_SEED,
                         {t: i for i, t in enumerate(STAGE3_TEAMS)})

    def test_optimizer_scoring_comes_from_config(self):
        import optimize
        from event_config import COLOGNE
        s = COLOGNE.scoring
        self.assertEqual((optimize.N_30, optimize.N_03, optimize.N_ADV),
                         (s["exact_3_0"], s["exact_0_3"], s["advance"]))
        self.assertEqual(optimize.PASS_THRESHOLD, s["pass_threshold"])


if __name__ == "__main__":
    unittest.main()
