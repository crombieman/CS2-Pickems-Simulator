"""posterior.py W15/U1 helper tests: structural weight perturbation + the E.3
published-interval envelope. The full sweep (structural_points) is a manual
analysis run, not unit-tested; these pin the pure combination logic, where a
bug would silently narrow or widen a published interval."""

import unittest

from posterior import perturb_weights, published_interval, fmt_interval


class TestPerturbWeights(unittest.TestCase):
    def test_scale_one_is_identity(self):
        m = [("A", "B", 1.0), ("C", "D", 0.5)]
        self.assertEqual(perturb_weights(m, 1.0), m)

    def test_scale_zero_flattens_to_one(self):
        m = [("A", "B", 0.5), ("C", "D", 0.65)]
        self.assertEqual([w for *_, w in perturb_weights(m, 0.0)], [1.0, 1.0])

    def test_scale_gt_one_sharpens(self):
        # w=0.5, scale=1.3 -> 1 - 1.3*0.5 = 0.35 (further from 1.0).
        self.assertAlmostEqual(perturb_weights([("A", "B", 0.5)], 1.3)[0][2], 0.35)

    def test_weight_clamped_positive(self):
        # scale large enough to drive 1 - scale*(1-w) below 0 -> clamped, not negative.
        self.assertGreater(perturb_weights([("A", "B", 0.5)], 5.0)[0][2], 0.0)

    def test_full_weight_unchanged_by_any_scale(self):
        # w=1.0 has no recency/format deviation, so no scale moves it.
        self.assertEqual(perturb_weights([("A", "B", 1.0)], 0.3)[0][2], 1.0)


class TestPublishedInterval(unittest.TestCase):
    def test_envelopes_structural_beyond_parameter(self):
        # structural corners reach wider than the parameter interval -> expand.
        self.assertEqual(
            published_interval(0.42, 0.40, 0.44, [0.37, 0.45], mc_floor=0.0),
            (0.37, 0.45))

    def test_mc_floor_widens_when_everything_is_tight(self):
        lo, hi = published_interval(0.42, 0.42, 0.42, [0.42], mc_floor=0.01)
        self.assertAlmostEqual(lo, 0.41)
        self.assertAlmostEqual(hi, 0.43)

    def test_parameter_interval_dominates_when_widest(self):
        self.assertEqual(
            published_interval(0.42, 0.30, 0.50, [0.41, 0.43], mc_floor=0.003),
            (0.30, 0.50))

    def test_clamped_to_unit_interval(self):
        self.assertEqual(
            published_interval(0.02, 0.01, 0.05, [-0.10, 1.20], mc_floor=0.0),
            (0.0, 1.0))


class TestFmtInterval(unittest.TestCase):
    def test_two_decimals_not_three(self):
        # three decimals overstates what we know by an order of magnitude (E.3).
        self.assertEqual(fmt_interval(0.418, 0.37, 0.452), "0.42 [0.37-0.45]")


if __name__ == "__main__":
    unittest.main()
