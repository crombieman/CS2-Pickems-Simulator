"""Playoff bracket engine tests. Run: python -m unittest src.test_playoffs -v

Wave 1: seeding derivation, QF layout, BO5 conversion, exact enumeration.
"""

import collections
import json
import random
import unittest
from pathlib import Path

from model import STAGE3_TEAMS
from playoffs import (RECORD_MULTISET, all_picks, bracket_distribution,
                      map_prob, optimize_picks, playoff_seeds, quarterfinals,
                      score_pick, series_prob_bo5, stage3_final_state)
from simulate import ROUND1, SEED, _pair_group, match_prob, simulate_stage

DATA = Path(__file__).resolve().parent.parent / "data"


def play_recorded_swiss(ratings, seed):
    """Mirror of simulate_stage's loop that also records (winner, loser)
    per match — consumes rng draws in the identical order, so the final
    records must match simulate_stage with the same seed."""
    rng = random.Random(seed)
    wins, losses = collections.Counter(), collections.Counter()
    played = {t: set() for t in STAGE3_TEAMS}
    opponents = {t: [] for t in STAGE3_TEAMS}
    matches = list(ROUND1)
    completed = []
    final = {}
    while True:
        if not matches:
            buch = {t: sum(wins[o] - losses[o] for o in opponents[t])
                    for t in STAGE3_TEAMS}
            groups = collections.defaultdict(list)
            for t in STAGE3_TEAMS:
                if t not in final:
                    groups[(wins[t], losses[t])].append(t)
            for grp in groups.values():
                matches += _pair_group(grp, played, buch)
            if not matches:
                return completed
        for a, b in matches:
            w, l = (a, b) if rng.random() < match_prob(ratings, a, b) else (b, a)
            completed.append((w, l))
            wins[w] += 1
            losses[l] += 1
            played[a].add(b)
            played[b].add(a)
            opponents[a].append(b)
            opponents[b].append(a)
        for t in STAGE3_TEAMS:
            if t not in final and (wins[t] == 3 or losses[t] == 3):
                final[t] = (wins[t], losses[t])
        matches = []


class TestBo5Conversion(unittest.TestCase):
    def test_even_series_is_fixed_point(self):
        self.assertAlmostEqual(series_prob_bo5(0.5), 0.5, places=12)

    def test_longer_series_amplifies_favorite(self):
        self.assertGreater(series_prob_bo5(0.7), 0.7)
        self.assertLess(series_prob_bo5(0.3), 0.3)

    def test_map_prob_round_trip(self):
        for q in (0.3, 0.55, 0.9):
            p3 = q * q * (3 - 2 * q)
            self.assertAlmostEqual(map_prob(p3), q, places=9)

    def test_extremes_stay_in_bounds(self):
        for p3 in (0.0, 1e-12, 1.0 - 1e-12, 1.0):
            p5 = series_prob_bo5(p3)
            self.assertGreaterEqual(p5, 0.0)
            self.assertLessEqual(p5, 1.0)
        self.assertLess(series_prob_bo5(0.001), 0.001)
        self.assertGreater(series_prob_bo5(0.999), 0.999)


class TestSeedingAndLayout(unittest.TestCase):
    def test_qf_layout_is_rulebook_bracket(self):
        seeds = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
        # Top half: 1v8 + 4v5 (SF1), bottom half: 2v7 + 3v6 (SF2).
        self.assertEqual(quarterfinals(seeds),
                         [("s1", "s8"), ("s4", "s5"),
                          ("s2", "s7"), ("s3", "s6")])

    def test_seed_order_record_then_buchholz_then_initial_seed(self):
        records = {t: (1, 3) for t in STAGE3_TEAMS}
        # 3-0s: NAVI (buch 7) over Vitality (buch 5) despite Vitality's
        # better initial seed; 3-1s next; 3-2s last. FURIA/MOUZ tie on
        # buchholz -> FURIA (seed 6) ahead of MOUZ (seed 7).
        for t, rec in (("Vitality", (3, 0)), ("NAVI", (3, 0)),
                       ("Falcons", (3, 1)), ("Spirit", (3, 1)),
                       ("MongolZ", (3, 1)), ("Aurora", (3, 2)),
                       ("FURIA", (3, 2)), ("MOUZ", (3, 2))):
            records[t] = rec
        buch = {t: 0 for t in STAGE3_TEAMS}
        buch.update({"Vitality": 5, "NAVI": 7, "Falcons": 2, "Spirit": 4,
                     "MongolZ": 3, "Aurora": 6, "FURIA": 1, "MOUZ": 1})
        self.assertEqual(
            playoff_seeds(records, buch),
            ["NAVI", "Vitality", "Spirit", "MongolZ", "Falcons",
             "Aurora", "FURIA", "MOUZ"])

    def test_final_state_matches_simulator(self):
        ratings = json.load(open(DATA / "ratings_fitted.json"))
        for seed in (7, 23):
            completed = play_recorded_swiss(ratings, seed)
            records, buchholz = stage3_final_state(completed)
            want = simulate_stage(ratings, random.Random(seed))
            self.assertEqual(records, want)
            self.assertEqual(collections.Counter(records.values()),
                             collections.Counter(RECORD_MULTISET))
            # spot-check buchholz definition on one team
            opp = [l if w == "Vitality" else w for w, l in completed
                   if "Vitality" in (w, l)]
            self.assertEqual(
                buchholz["Vitality"],
                sum(records[o][0] - records[o][1] for o in opp))

    def test_incomplete_stage_rejected(self):
        with self.assertRaises(ValueError) as cm:
            stage3_final_state([("Vitality", "FUT"), ("NAVI", "Spirit"),
                                ("MOUZ", "Legacy"), ("Falcons", "G2"),
                                ("MongolZ", "BetBoom"), ("Aurora", "Monte"),
                                ("FURIA", "B8"), ("PARIVISION", "9z")])
        self.assertIn("not complete", str(cm.exception))


class TestEnumeration(unittest.TestCase):
    SEEDS = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]

    def test_uniform_probs(self):
        branches = bracket_distribution(self.SEEDS,
                                        lambda a, b, bo5=False: 0.5)
        self.assertEqual(len(branches), 128)
        self.assertAlmostEqual(sum(p for *_, p in branches), 1.0, places=12)
        for *_, p in branches:
            self.assertAlmostEqual(p, 1 / 128, places=12)
        champ = collections.Counter()
        for qf_w, sf_w, c, p in branches:
            champ[c] += p
        for t in self.SEEDS:
            self.assertAlmostEqual(champ[t], 1 / 8, places=12)

    def test_dominant_team(self):
        def prob(a, b, bo5=False):
            if a == "s1":
                return 1.0
            if b == "s1":
                return 0.0
            return 0.5
        branches = bracket_distribution(self.SEEDS, prob)
        self.assertAlmostEqual(sum(p for *_, p in branches), 1.0, places=12)
        champ = collections.Counter()
        for qf_w, sf_w, c, p in branches:
            champ[c] += p
        self.assertAlmostEqual(champ["s1"], 1.0, places=12)

    def test_branches_are_consistent_brackets(self):
        branches = bracket_distribution(self.SEEDS,
                                        lambda a, b, bo5=False: 0.5)
        qfs = quarterfinals(self.SEEDS)
        for qf_w, sf_w, c, p in branches:
            for (a, b), w in zip(qfs, qf_w):
                self.assertIn(w, (a, b))
            self.assertIn(sf_w[0], qf_w[:2])
            self.assertIn(sf_w[1], qf_w[2:])
            self.assertIn(c, sf_w)

    def test_bo5_flag_on_grand_final_only(self):
        # QFs and SFs stay within a bracket half; only the grand final
        # crosses halves. So bo5=True must fire exactly on cross-half calls.
        top = {"s1", "s8", "s4", "s5"}
        calls = []

        def prob(a, b, bo5=False):
            calls.append((a, b, bo5))
            return 0.5
        bracket_distribution(self.SEEDS, prob)
        self.assertTrue(any(bo5 for *_, bo5 in calls))
        for a, b, bo5 in calls:
            self.assertEqual(bo5, (a in top) != (b in top))


def seed_chalk_prob(seeds):
    """Lower seed number always wins (deterministic bracket)."""
    rank = {t: i for i, t in enumerate(seeds)}

    def prob(a, b, bo5=False):
        return 1.0 if rank[a] < rank[b] else 0.0
    return prob


class TestPickScoring(unittest.TestCase):
    SEEDS = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]

    def test_all_picks_count_and_consistency(self):
        picks = all_picks(self.SEEDS)
        self.assertEqual(len(picks), 128)
        self.assertEqual(len(set(picks)), 128)
        qfs = quarterfinals(self.SEEDS)
        for qf_w, sf_w, champ in picks:
            for (a, b), w in zip(qfs, qf_w):
                self.assertIn(w, (a, b))
            self.assertIn(sf_w[0], qf_w[:2])
            self.assertIn(sf_w[1], qf_w[2:])
            self.assertIn(champ, sf_w)

    def test_deterministic_bracket_scoring(self):
        branches = bracket_distribution(self.SEEDS, seed_chalk_prob(self.SEEDS))
        truth = (("s1", "s4", "s2", "s3"), ("s1", "s2"), "s1")
        s = score_pick(branches, truth)
        self.assertAlmostEqual(s["challenges"], 1.0, places=12)
        self.assertAlmostEqual(s["champion"], 1.0, places=12)
        self.assertAlmostEqual(s["expected_correct"], 7.0, places=12)
        self.assertAlmostEqual(s["perfect"], 1.0, places=12)
        # same bracket but wrong champion: challenges joint fails
        wrong_champ = (truth[0], truth[1], "s2")
        s = score_pick(branches, wrong_champ)
        self.assertAlmostEqual(s["challenges"], 0.0, places=12)
        self.assertAlmostEqual(s["champion"], 0.0, places=12)
        self.assertAlmostEqual(s["expected_correct"], 6.0, places=12)

    def test_uniform_hand_computed_values(self):
        branches = bracket_distribution(self.SEEDS,
                                        lambda a, b, bo5=False: 0.5)
        for pick in (all_picks(self.SEEDS)[0], all_picks(self.SEEDS)[77]):
            s = score_pick(branches, pick)
            # E = 4*0.5 + 2*0.25 + 0.125
            self.assertAlmostEqual(s["expected_correct"], 2.625, places=12)
            self.assertAlmostEqual(s["perfect"], 1 / 128, places=12)
            self.assertAlmostEqual(s["champion"], 0.125, places=12)
            # champ correct (0.125) implies its QF+SF correct, so the
            # joint needs only >=1 of the other 3 QFs: 0.125 * 0.875
            self.assertAlmostEqual(s["challenges"], 0.125 * 0.875,
                                   places=12)

    def test_optimizer_picks_chalk_under_dominant_ratings(self):
        ratings = {t: 2000 - 250 * i for i, t in enumerate(self.SEEDS)}
        draws = [ratings, dict(ratings)]  # zero posterior spread
        results = optimize_picks(self.SEEDS, {}, draws)
        best = results[0]
        self.assertEqual(best["pick"],
                         (("s1", "s4", "s2", "s3"), ("s1", "s2"), "s1"))
        # identical draws -> per-draw values identical
        self.assertAlmostEqual(best["draw_values"][0],
                               best["draw_values"][1], places=12)
        self.assertAlmostEqual(
            best["means"]["challenges"], best["draw_values"][0], places=12)


    def test_challenges_ties_broken_by_expected_correct(self):
        # "challenges" ignores the non-champion-side SF pick (champion
        # correct already implies >=1 SF correct), so exact twins exist.
        # The winner among twins must be the one maximizing E[correct].
        ratings = {t: 2000 - 250 * i for i, t in enumerate(self.SEEDS)}
        results = optimize_picks(self.SEEDS, {}, [ratings])
        best = results[0]
        top = best["means"]["challenges"]
        for r in results[1:]:
            if abs(r["means"]["challenges"] - top) < 1e-12:
                self.assertLessEqual(r["means"]["expected_correct"],
                                     best["means"]["expected_correct"] + 1e-12)


class TestRatingDraws(unittest.TestCase):
    def test_draws_deterministic_and_spread(self):
        from posterior import laplace_factor, rating_draws
        factor = laplace_factor()
        a = rating_draws(k=3, seed=5, factor=factor)
        b = rating_draws(k=3, seed=5, factor=factor)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 3)
        for d in a:
            self.assertTrue(set(STAGE3_TEAMS) <= set(d))
        spread = max(abs(a[0][t] - a[1][t]) for t in STAGE3_TEAMS)
        self.assertGreater(spread, 1.0)  # draws actually vary


class TestEventConfigWiring(unittest.TestCase):
    """W5: playoff format comes from the event config, not a hard-coded flag."""

    def test_grand_final_bo5_from_config(self):
        import playoffs
        from event_config import COLOGNE
        self.assertEqual(playoffs.GRAND_FINAL_BO5,
                         COLOGNE.playoffs["grand_final_bo5"])


if __name__ == "__main__":
    unittest.main()
