"""V1 harness tests (W6a): consumer view, recenter_on, walk-forward core,
fit cache, paired-event stats, and the synthetic self-test known answers.

Spec: docs/plans/2026-07-05-w6-v1-harness-spec.md 7 (W6a test list). Fixture
DBs go through bo3gg_parse.build_db + classify_promotions so consumer_rows is
exercised against the real substrate schema, never a hand-rolled one.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from bo3gg_parse import build_db
from test_bo3gg_parse import FixtureArchive, match

from calibration import _brier
from integrity_audit import classify_promotions, consumer_rows
from model import STAGE3_TEAMS, fit_bradley_terry

import harness
from harness import (FIT_CODE_VERSION, HARNESS_VERSION, HarnessError,
                     assert_no_leakage, build_fit_universe, cached_fit,
                     canonical_sha, classify_split, config_sha, event_summary,
                     event_universe, fit_cache_key, fit_engine,
                     grade_event_walkforward, load_config, months_before,
                     paired_event_stats, run_self_test, t_quantile, verdict)

DATA = Path(__file__).resolve().parent.parent / "data"
CONFIGS = DATA / "harness" / "configs"


# -- fixture helpers -----------------------------------------------------------
def row(mid, *, tid=500, t1=1, t2=2, winner=None, start="2024-01-01T10:00:00+00:00",
        tier="s", stage=9, rnd=7):
    """A consumer-view row dict shaped like integrity_audit.consumer_rows()."""
    return {"match_id": mid, "start_date": start, "bo_type": 3,
            "team1_id": t1, "team2_id": t2, "team1_score": 2, "team2_score": 0,
            "winner_team_id": winner if winner is not None else t1,
            "tier": tier, "tournament_id": tid, "stage_id": stage,
            "round_id": rnd, "quarantine_reason": None}


def event_rows(tid, n, *, tier="s", start="2024-06-01T10:00:00+00:00",
               base_mid=0):
    return [row(base_mid + i, tid=tid, t1=1, t2=2, start=start, tier=tier)
            for i in range(n)]


class DbCase(unittest.TestCase):
    """Fixture substrate through the real parser (same idiom as
    test_integrity_audit.DbCase): closes handles before tmp cleanup."""

    def build(self, rows, aliases=None):
        fx = FixtureArchive()
        fx.write([rows], offsets=[0]).state(len(rows))
        db = fx.dir / "t.sqlite"
        build_db(db_path=db, data_dir=fx.dir,
                 canonical_path=fx.canonical(aliases if aliases is not None
                                             else []))
        self._fx = fx
        self.addCleanup(fx.tmp.cleanup)
        con = sqlite3.connect(db)
        self.addCleanup(con.close)
        return db, con


# -- consumer view (spec 1: THE single tested query) ---------------------------
class TestConsumerRows(DbCase):
    def test_clean_plus_promoted_only(self):
        rows = [match(1, s1=2, s2=1),                         # clean
                match(2, bo=1, s1=2, s2=0),                   # -> promoted
                match(3, bo=3, s1=1, s2=0),                   # forfeit sig: out
                match(4, bo=2, s1=1, s2=1, winner=None)]      # bo2 draw: out
        db, con = self.build(rows)
        classify_promotions(con)
        got = consumer_rows(con)
        self.assertEqual([r["match_id"] for r in got], [1, 2])
        # promoted row keeps its quarantine_reason (visible, not laundered)
        self.assertEqual(got[1]["quarantine_reason"], "score_bo_mismatch")
        self.assertIsNone(got[0]["quarantine_reason"])

    def test_raises_before_audit_ran(self):
        # An unaudited DB has no audit_flags table; a silent empty view would
        # hide every promoted row - fail loud instead.
        db, con = self.build([match(1, s1=2, s2=1)])
        with self.assertRaises(ValueError):
            consumer_rows(con)

    def test_row_shape_pinned(self):
        db, con = self.build([match(1, s1=2, s2=1)])
        classify_promotions(con)
        (r,) = consumer_rows(con)
        self.assertEqual(r["team1_id"], 1)
        self.assertEqual(r["winner_team_id"], 1)
        self.assertEqual(r["tournament_id"], 500)
        self.assertEqual(r["tier"], "s")
        self.assertTrue(r["start_date"].startswith("2026-01-01"))


# -- model.recenter_on (spec 1: the STAGE3_TEAMS KeyError finding) --------------
class TestRecenterOn(unittest.TestCase):
    def test_id_universe_without_recenter_on_raises(self):
        priors = {"101": 1000.0, "102": 1000.0}
        with self.assertRaises(KeyError):
            fit_bradley_terry([("101", "102", 1.0)], priors=priors,
                              sigma_s3=50.0, sigma_other=50.0, iters=5)

    def test_id_universe_recenters_on_given_universe(self):
        priors = {"101": 1000.0, "102": 1000.0, "103": 1000.0}
        ratings = fit_bradley_terry(
            [("101", "102", 1.0), ("101", "103", 1.0)], priors=priors,
            sigma_s3=50.0, sigma_other=50.0, iters=50,
            recenter_on=sorted(priors))
        mean = sum(ratings.values()) / len(ratings)
        self.assertAlmostEqual(mean, 1000.0, places=9)
        self.assertGreater(ratings["101"], ratings["102"])

    def test_default_is_behavior_preserving(self):
        # recenter_on=None must be the exact old code path (STAGE3_TEAMS);
        # byte-level evidence is CI's fit-reproducibility gate on top of this.
        matches = [("Vitality", "Spirit", 1.0), ("NAVI", "FURIA", 0.5)]
        old = fit_bradley_terry(matches, iters=25)
        new = fit_bradley_terry(matches, iters=25, recenter_on=STAGE3_TEAMS)
        self.assertEqual(old, new)


class TestConvergence(unittest.TestCase):
    """W6a ship-gate finding (2026-07-05): lr=2000 is Cologne-tuned (~8
    matches/team) and OSCILLATES on dense historical fit graphs (probe:
    300-match pair flips between A=280/B=1720 and A=3158/B=-1158 per
    iteration; true MAP optimum A=1055/B=945). The harness incumbent
    declares a stable lr + converge_tol; these tests pin that the declared
    params actually reach the optimum and that non-convergence is LOUD."""

    DENSE = [("A", "B", 1.0)] * 200 + [("B", "A", 1.0)] * 100
    PRIORS = {"A": 1000.0, "B": 1000.0}

    def fit(self, **over):
        kw = dict(matches=self.DENSE, priors=self.PRIORS, sigma_s3=50.0,
                  sigma_other=50.0, iters=4000, lr=100.0, converge_tol=0.01,
                  recenter_on=["A", "B"])
        kw.update(over)
        return fit_bradley_terry(**kw)

    def test_iteration_cap_invariance_once_converged(self):
        # early stop => raising the cap must not change a single float
        self.assertEqual(self.fit(iters=4000), self.fit(iters=8000))

    def test_converges_to_the_true_map_optimum(self):
        # low-lr long-run reference (no early stop) = ground truth
        ref = fit_bradley_terry(self.DENSE, priors=self.PRIORS,
                                sigma_s3=50.0, sigma_other=50.0,
                                iters=60000, lr=50.0,
                                recenter_on=["A", "B"])
        got = self.fit()
        for t in ("A", "B"):
            self.assertAlmostEqual(got[t], ref[t], delta=0.1)

    def test_non_convergence_raises(self):
        with self.assertRaises(ValueError):
            self.fit(iters=3)

    def test_no_tol_is_behavior_preserving(self):
        old = fit_bradley_terry(self.DENSE, priors=self.PRIORS,
                                sigma_s3=50.0, sigma_other=50.0, iters=50,
                                lr=100.0, recenter_on=["A", "B"])
        new = self.fit(iters=50, converge_tol=None)
        self.assertEqual(old, new)


# -- date arithmetic ------------------------------------------------------------
class TestMonthsBefore(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(months_before("2024-06-15", 24), "2022-06-15")

    def test_accepts_full_timestamp(self):
        self.assertEqual(months_before("2024-06-15T10:30:00+00:00", 6),
                         "2023-12-15")

    def test_clamps_month_end(self):
        self.assertEqual(months_before("2024-03-31", 1), "2024-02-29")
        self.assertEqual(months_before("2023-03-31", 1), "2023-02-28")

    def test_year_rollover(self):
        self.assertEqual(months_before("2024-01-10", 3), "2023-10-10")


# -- event universe + eligibility (spec 4.1) ------------------------------------
class TestEventUniverse(unittest.TestCase):
    def test_eligibility_floor_and_tier(self):
        rows = (event_rows(1, 8, tier="s") +               # eligible
                event_rows(2, 7, tier="s", base_mid=100) + # too few
                event_rows(3, 8, tier="b", base_mid=200))  # wrong tier
        events, report = event_universe(rows, tiers=("s", "a"), min_matches=8)
        self.assertEqual([e["tournament_id"] for e in events], [1])
        self.assertEqual(report["excluded_small"], [2])
        self.assertEqual(report["excluded_tier"], [3])

    def test_boundary_is_min_start_and_last_is_max(self):
        rows = [row(1, tid=1, start="2024-06-02T10:00:00+00:00"),
                row(2, tid=1, start="2024-06-01T08:00:00+00:00"),
                row(3, tid=1, start="2024-06-05T12:00:00+00:00")]
        rows += event_rows(1, 5, base_mid=10)  # pad over the floor
        events, _ = event_universe(rows, min_matches=8)
        (ev,) = events
        # padding rows start 2024-06-01T10:00; row 2 is earlier still
        self.assertEqual(ev["boundary"], "2024-06-01T08:00:00+00:00")
        self.assertEqual(ev["last_start"], "2024-06-05T12:00:00+00:00")
        self.assertEqual(ev["n_matches"], 8)

    def test_null_tier_rows_counted_not_silent(self):
        rows = event_rows(1, 8, tier="s")
        rows[0]["tier"] = None
        events, report = event_universe(rows, min_matches=8)
        self.assertEqual(len(events), 1)          # max over non-null = s
        self.assertEqual(report["null_tier_rows"], 1)

    def test_all_null_tier_event_excluded(self):
        rows = event_rows(1, 8, tier="s")
        for r in rows:
            r["tier"] = None
        events, report = event_universe(rows, min_matches=8)
        self.assertEqual(events, [])
        self.assertEqual(report["excluded_tier"], [1])


# -- dev/holdout split (spec 4.1 dual-end rule) ----------------------------------
class TestClassifySplit(unittest.TestCase):
    def ev(self, tid, lo, hi):
        return {"tournament_id": tid, "boundary": lo, "last_start": hi}

    def test_dev_holdout_straddler(self):
        dev = self.ev(1, "2024-01-01T10:00:00+00:00", "2024-01-05T10:00:00+00:00")
        hold = self.ev(2, "2025-08-01T10:00:00+00:00", "2025-08-05T10:00:00+00:00")
        strad = self.ev(3, "2025-06-28T10:00:00+00:00", "2025-07-02T10:00:00+00:00")
        split = classify_split([dev, hold, strad], "2025-07-01")
        self.assertEqual([e["tournament_id"] for e in split["dev"]], [1])
        self.assertEqual([e["tournament_id"] for e in split["holdout"]], [2])
        self.assertEqual([e["tournament_id"] for e in split["straddlers"]], [3])

    def test_boundary_exactly_at_split_is_holdout(self):
        ev = self.ev(1, "2025-07-01T00:00:00+00:00", "2025-07-04T00:00:00+00:00")
        split = classify_split([ev], "2025-07-01")
        self.assertEqual(len(split["holdout"]), 1)


# -- fit universe (spec 4.4: participants + 1-hop) --------------------------------
class TestBuildFitUniverse(unittest.TestCase):
    def setUp(self):
        # event: teams 1 vs 2. window: 1-3 (connector 3), 3-4 (4 is 2-hop),
        # 5-6 (disconnected), 2-3 (connector match).
        self.ev_rows = [row(100, t1=1, t2=2)]
        self.win_rows = [row(1, t1=1, t2=3, start="2023-01-01T10:00:00+00:00"),
                         row(2, t1=3, t2=4, start="2023-02-01T10:00:00+00:00"),
                         row(3, t1=5, t2=6, start="2023-03-01T10:00:00+00:00"),
                         row(4, t1=2, t2=3, start="2023-04-01T10:00:00+00:00")]

    def test_one_hop(self):
        u = build_fit_universe(self.ev_rows, self.win_rows, hops=1)
        self.assertEqual(u["universe"], {"1", "2", "3"})
        # match 2 (3 vs 4) excluded: team 4 outside the 1-hop universe
        self.assertEqual(u["fit_match_ids"], [1, 4])
        self.assertEqual(u["fit_matches"],
                         [("1", "3", 1.0), ("2", "3", 1.0)])
        self.assertEqual(u["window_counts"],
                         {"1": 1, "2": 1, "3": 2})

    def test_two_hop_pulls_connector_opponents(self):
        u = build_fit_universe(self.ev_rows, self.win_rows, hops=2)
        self.assertEqual(u["universe"], {"1", "2", "3", "4"})
        self.assertEqual(u["fit_match_ids"], [1, 2, 4])
        self.assertNotIn("5", u["universe"])


# -- temporal leakage assertion (spec 3.2.4) --------------------------------------
class TestLeakage(unittest.TestCase):
    def test_fit_row_at_boundary_trips(self):
        boundary = "2024-06-01T10:00:00+00:00"
        fit = [row(1, start=boundary)]
        with self.assertRaises(HarnessError):
            assert_no_leakage(fit, [row(2, start=boundary)], boundary)

    def test_graded_row_before_boundary_trips(self):
        boundary = "2024-06-01T10:00:00+00:00"
        graded = [row(2, start="2024-05-31T10:00:00+00:00")]
        with self.assertRaises(HarnessError):
            assert_no_leakage([], graded, boundary)

    def test_clean_passes(self):
        boundary = "2024-06-01T10:00:00+00:00"
        fit = [row(1, start="2024-05-01T10:00:00+00:00")]
        graded = [row(2, start=boundary)]
        assert_no_leakage(fit, graded, boundary)   # no raise


# -- grading + coverage (spec 4.3) -------------------------------------------------
class TestGradeAndCoverage(unittest.TestCase):
    def setUp(self):
        self.cand = {"1": 1100.0, "2": 1000.0}
        self.inc = {"1": 1000.0, "2": 1000.0}
        self.counts = {"1": 5, "2": 5}

    def test_delta_math_matches_calibration_primitives(self):
        rows = [row(1, t1=1, t2=2, winner=1)]
        graded, skipped = grade_event_walkforward(
            rows, self.cand, self.inc, self.counts, min_obs=3)
        self.assertEqual(skipped, [])
        (g,) = graded
        p_c = 1.0 / (1.0 + 10 ** (-(1100 - 1000) / 400.0))
        self.assertAlmostEqual(g["p_cand"], p_c, places=12)
        self.assertAlmostEqual(g["p_inc"], 0.5, places=12)
        self.assertAlmostEqual(g["brier_cand"], _brier(p_c, 1.0), places=12)
        self.assertAlmostEqual(
            g["delta_brier"], _brier(p_c, 1.0) - _brier(0.5, 1.0), places=12)
        self.assertEqual(g["kind"], "walkforward")

    def test_below_min_obs_skipped_with_reason(self):
        counts = {"1": 5, "2": 2}
        rows = [row(1, t1=1, t2=2, winner=1)]
        graded, skipped = grade_event_walkforward(
            rows, self.cand, self.inc, counts, min_obs=3)
        self.assertEqual(graded, [])
        (s,) = skipped
        self.assertEqual(s["match_id"], 1)
        self.assertIn("below_min_obs", s["reason"])
        self.assertIn("2", s["reason"])            # names the offending team

    def test_event_summary_coverage_exclusion(self):
        graded = [{"delta_brier": -0.01, "delta_log": -0.02}] * 2
        skipped = [{"reason": "below_min_obs:9"}] * 3
        s = event_summary(7, graded, skipped, coverage_floor=0.5)
        self.assertEqual(s["tournament_id"], 7)
        self.assertEqual(s["n_graded"], 2)
        self.assertEqual(s["n_skipped"], 3)
        self.assertAlmostEqual(s["coverage"], 0.4)
        self.assertTrue(s["excluded"])
        self.assertAlmostEqual(s["mean_delta_brier"], -0.01)

    def test_event_summary_included_when_coverage_ok(self):
        graded = [{"delta_brier": -0.01, "delta_log": -0.02}] * 3
        s = event_summary(7, graded, [], coverage_floor=0.5)
        self.assertFalse(s["excluded"])
        self.assertEqual(s["coverage"], 1.0)


# -- fit cache (spec 4.5: resolved-input content addressing) -----------------------
class TestFitCache(unittest.TestCase):
    BASE = dict(engine_sha="e" * 64, universe={"1", "2", "3"},
                fit_match_ids=[1, 2, 3], window=("2022-06-01", "2024-06-01"),
                substrate_id="s" * 64)

    def key(self, **over):
        kw = dict(self.BASE)
        kw.update(over)
        return fit_cache_key(kw["engine_sha"], kw["universe"],
                             kw["fit_match_ids"], kw["window"],
                             kw["substrate_id"])

    def test_same_inputs_same_key_order_insensitive(self):
        a = self.key()
        b = self.key(universe={"3", "2", "1"}, fit_match_ids=[3, 1, 2])
        self.assertEqual(a, b)

    def test_every_resolved_input_changes_the_key(self):
        base = self.key()
        for over in (dict(engine_sha="f" * 64),
                     dict(universe={"1", "2"}),
                     dict(fit_match_ids=[1, 2]),
                     dict(window=("2022-07-01", "2024-06-01")),
                     dict(substrate_id="t" * 64)):
            self.assertNotEqual(base, self.key(**over), over)

    def test_fit_code_version_is_in_the_key(self):
        base = self.key()
        orig = harness.FIT_CODE_VERSION
        try:
            harness.FIT_CODE_VERSION = orig + "-bumped"
            self.assertNotEqual(base, self.key())
        finally:
            harness.FIT_CODE_VERSION = orig

    def test_cached_fit_computes_once(self):
        calls = []
        def fit_fn():
            calls.append(1)
            return {"1": 1010.5, "2": 989.5}
        with tempfile.TemporaryDirectory() as tmp:
            first = cached_fit(Path(tmp), "k1", fit_fn)
            second = cached_fit(Path(tmp), "k1", fit_fn)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first, second)


# -- t quantile + paired stats (spec 5.1) -------------------------------------------
class TestTQuantile(unittest.TestCase):
    KNOWN = {1: 12.7062, 2: 4.3027, 10: 2.2281, 30: 2.0423, 100: 1.9840}

    def test_known_975_values(self):
        for df, want in self.KNOWN.items():
            self.assertAlmostEqual(t_quantile(0.975, df), want, places=3)

    def test_symmetry_and_median(self):
        self.assertAlmostEqual(t_quantile(0.025, 10),
                               -t_quantile(0.975, 10), places=9)
        self.assertEqual(t_quantile(0.5, 7), 0.0)

    def test_rejects_bad_inputs(self):
        for bad in (0.0, 1.0, -0.1):
            with self.assertRaises(ValueError):
                t_quantile(bad, 10)
        with self.assertRaises(ValueError):
            t_quantile(0.975, 0)


class TestPairedEventStats(unittest.TestCase):
    def test_hand_computed_fixture(self):
        deltas = [-0.01, -0.02, -0.03]
        s = paired_event_stats(deltas, ci_level=0.95)
        self.assertEqual(s["n_events"], 3)
        self.assertAlmostEqual(s["mean"], -0.02, places=12)
        se = (0.0001 / 3) ** 0.5    # sample var 0.0001, SE = sd/sqrt(3)
        self.assertAlmostEqual(s["se"], se, places=12)
        t_crit = t_quantile(0.975, 2)
        self.assertAlmostEqual(s["ci"][0], -0.02 - t_crit * se, places=9)
        self.assertAlmostEqual(s["ci"][1], -0.02 + t_crit * se, places=9)

    def test_uses_t_at_every_n(self):
        # n=2: t(0.975, 1) = 12.7 - a normal-approx 1.96 here would be the
        # exact fake-significance the review flagged.
        s = paired_event_stats([-0.01, -0.011], ci_level=0.95)
        width = s["ci"][1] - s["ci"][0]
        self.assertGreater(width, 2 * 1.96 * s["se"])


# -- verdicts (spec 5.2 subset shipped in W6a; gates land in W6b) --------------------
class TestVerdict(unittest.TestCase):
    def stats(self, mean, lo, hi, n=20):
        return {"n_events": n, "mean": mean, "se": 0.001, "ci": [lo, hi]}

    def test_dev_screened(self):
        v = verdict(self.stats(-0.005, -0.007, -0.003), mde=0.002, min_events=10)
        self.assertEqual(v["verdict"], "DEV-SCREENED")

    def test_reject(self):
        v = verdict(self.stats(0.005, 0.003, 0.007), mde=0.002, min_events=10)
        self.assertEqual(v["verdict"], "REJECT")

    def test_inconclusive_ci_crosses_zero(self):
        v = verdict(self.stats(-0.005, -0.011, 0.001), mde=0.002, min_events=10)
        self.assertEqual(v["verdict"], "INCONCLUSIVE")

    def test_inconclusive_below_mde(self):
        v = verdict(self.stats(-0.001, -0.0015, -0.0005), mde=0.002,
                    min_events=10)
        self.assertEqual(v["verdict"], "INCONCLUSIVE")

    def test_blocked_below_min_events(self):
        v = verdict(self.stats(-0.005, -0.007, -0.003, n=9), mde=0.002,
                    min_events=10)
        self.assertEqual(v["verdict"], "BLOCKED")
        self.assertIn("insufficient-n", v["reason"])

    def test_objective_check_blocks_screening(self):
        v = verdict(self.stats(-0.005, -0.007, -0.003), mde=0.002,
                    min_events=10, objective_ok=False)
        self.assertEqual(v["verdict"], "INCONCLUSIVE")
        self.assertIn("proxy-only", v["reason"])


# -- synthetic self-test (spec 6.1: known-answer trio) --------------------------------
class TestSelfTest(unittest.TestCase):
    def test_known_answer_trio(self):
        r = run_self_test(seed=20260705)
        self.assertEqual(r["verdicts"]["better"], "DEV-SCREENED")
        self.assertEqual(r["verdicts"]["worse"], "REJECT")
        self.assertEqual(r["verdicts"]["clone"], "INCONCLUSIVE")
        self.assertTrue(r["ok"])
        self.assertAlmostEqual(r["stats"]["clone"]["mean"], 0.0, places=12)

    def test_deterministic(self):
        a = run_self_test(seed=20260705)
        b = run_self_test(seed=20260705)
        self.assertEqual(a, b)

    def test_seed_robustness(self):
        r = run_self_test(seed=99)
        self.assertTrue(r["ok"])

    def test_wrong_answer_flips_ok(self):
        # Zero mixture weights make "better"/"worse" actual clones, so their
        # DEV-SCREENED/REJECT expectations fail - ok=False must be reachable
        # (a self-test that cannot fail validates nothing, DoR 6).
        r = run_self_test(seed=20260705, weights=(0.0, 0.0))
        self.assertFalse(r["ok"])
        self.assertEqual(r["verdicts"]["better"], "INCONCLUSIVE")


# -- configs (spec 3.1: committed, declared before replay) ----------------------------
class TestConfigs(unittest.TestCase):
    def test_harness_v0_loads_with_required_fields(self):
        cfg = load_config(CONFIGS / "harness_v0.json")
        self.assertEqual(cfg["harness_version"], HARNESS_VERSION)
        self.assertEqual(cfg["eligibility"]["tiers"], ["s", "a"])
        self.assertEqual(cfg["eligibility"]["min_consumer_matches"], 8)
        self.assertEqual(cfg["window_months"], 24)
        # 2hop: the W6a spot-check MEASURED the rule (spec 4.4 pre-registered
        # tolerance): 1hop vs 2hop moved 48.4% of graded probs > 0.005;
        # 2hop vs FULL window universe moved 0/122 (max |dp| 0.0032).
        self.assertEqual(cfg["fit_universe"], {"rule": "2hop", "min_obs": 3})
        self.assertEqual(cfg["coverage_floor"], 0.5)
        self.assertEqual(cfg["holdout_split"], "2025-07-01")
        self.assertEqual(cfg["verdict"],
                         {"mde_brier": 0.002, "ci_level": 0.95,
                          "min_events": 10})
        self.assertIn("self_test", cfg["seeds"])

    def test_incumbent_v0_schema(self):
        # lr=100 + converge_tol=0.01 (NOT Cologne's lr=2000): the ship-gate
        # probe proved lr=2000 oscillates on dense historical graphs; iters
        # is a CAP, the stop rule is convergence (spec 4.5 remedy ladder).
        cfg = load_config(CONFIGS / "incumbent_v0.json")
        self.assertEqual(cfg["model"],
                         {"priors_scheme": "flat1000", "prior_mean": 1000,
                          "sigma": 50, "iters": 4000, "lr": 100.0,
                          "converge_tol": 0.01})
        self.assertEqual(cfg["data_prep"], {"weighting": "uniform"})
        self.assertEqual(cfg["market_policy"], "none")
        self.assertIsNone(cfg["knob_id"])
        self.assertIn("forward_prereg", cfg)     # W12 extends, never forks

    def test_config_sha_is_canonical(self):
        a = config_sha({"b": 1, "a": [2, 3]})
        b = config_sha({"a": [2, 3], "b": 1})
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)
        self.assertNotEqual(a, config_sha({"a": [3, 2], "b": 1}))


# -- engine fit wrapper (explicit sigma, flat priors, declared schemes only) ----------
class TestFitEngine(unittest.TestCase):
    def test_flat_priors_uniform_weights(self):
        # full config: the convergence stop makes the 4000-iter cap cheap
        eng = load_config(CONFIGS / "incumbent_v0.json")
        u = {"universe": {"1", "2"}, "fit_matches": [("1", "2", 1.0)],
             "fit_match_ids": [10]}
        ratings = fit_engine(eng, u)
        self.assertEqual(set(ratings), {"1", "2"})
        mean = sum(ratings.values()) / 2
        self.assertAlmostEqual(mean, 1000.0, places=9)
        self.assertGreater(ratings["1"], ratings["2"])

    def test_unknown_scheme_fails_loud(self):
        eng = load_config(CONFIGS / "incumbent_v0.json")
        bad = dict(eng, data_prep={"weighting": "recency"})
        u = {"universe": {"1", "2"}, "fit_matches": [("1", "2", 1.0)],
             "fit_match_ids": [10]}
        with self.assertRaises(HarnessError):
            fit_engine(bad, u)
        bad2 = dict(eng, model=dict(eng["model"], priors_scheme="tiered"))
        with self.assertRaises(HarnessError):
            fit_engine(bad2, u)


if __name__ == "__main__":
    unittest.main()
