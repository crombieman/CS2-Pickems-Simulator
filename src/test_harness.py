"""V1 harness tests (W6a): consumer view, recenter_on, walk-forward core,
fit cache, paired-event stats, and the synthetic self-test known answers.

Spec: docs/plans/2026-07-05-w6-v1-harness-spec.md 7 (W6a test list). Fixture
DBs go through bo3gg_parse.build_db + classify_promotions so consumer_rows is
exercised against the real substrate schema, never a hand-rolled one.
"""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bo3gg_parse import build_db
from test_bo3gg_parse import FixtureArchive, match

from calibration import _brier
from integrity_audit import classify_promotions, consumer_rows
from model import STAGE3_TEAMS, fit_bradley_terry

import harness
from harness import (FIT_CODE_VERSION, HARNESS_VERSION, HarnessError,
                     assert_no_leakage, build_fit_universe, cached_fit,
                     canonical_sha, classify_split, cologne_cross_check,
                     config_sha, diff_paths, event_summary, event_universe,
                     fit_cache_key, fit_engine, gate_config_diff,
                     gate_parse_meta, grade_event_walkforward, load_config,
                     months_before, paired_event_stats, run_replay,
                     run_self_test, select_nominee, slate_objective_check,
                     t_quantile, utc_key, verdict, walk_forward_replay)

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

    def test_default_path_matches_pre_w6_goldens(self):
        # Codex W6a review P3: comparing two NEW paths cannot catch both
        # drifting together. These values were generated at commit 6dd7466
        # (proven byte-identical to pre-W6 behavior by fit.py + CI gate) -
        # they ARE the old behavior, pinned.
        r = fit_bradley_terry([("Vitality", "Spirit", 1.0),
                               ("NAVI", "FURIA", 0.5)], iters=25)
        goldens = {"Vitality": 1189.288373985839, "Spirit": 1065.711626014161,
                   "NAVI": 1070.4391508097926, "FURIA": 989.5608491902075,
                   "G2": 920.0}
        for team, want in goldens.items():
            self.assertAlmostEqual(r[team], want, places=6)


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


# -- UTC-normalized temporal keys (Codex W6a review P1: offset-proof ordering) ---
class TestUtcKey(unittest.TestCase):
    """Raw ISO strings order wrongly across offsets. All harness temporal
    comparisons must go through utc_key. Current archive is uniformly
    +00:00 (verified 2026-07-05) - this defends the contract, not a live bug."""

    def test_offset_normalization_reverses_lexicographic_order(self):
        # Codex's exact scenario: +02:00 row is chronologically EARLIER
        a = "2024-06-01T00:30:00+02:00"    # = 2024-05-31T22:30 UTC
        b = "2024-05-31T23:00:00+00:00"    # = 2024-05-31T23:00 UTC
        self.assertGreater(a, b)                       # raw strings lie
        self.assertLess(utc_key(a), utc_key(b))        # utc_key does not

    def test_archive_format_normalizes_cleanly(self):
        self.assertEqual(utc_key("2024-06-01T10:00:00.000+00:00"),
                         "2024-06-01T10:00:00")

    def test_naive_timestamp_treated_as_utc(self):
        self.assertEqual(utc_key("2024-06-01T10:00:00"),
                         "2024-06-01T10:00:00")

    def test_comparable_against_date_only_strings(self):
        self.assertGreater(utc_key("2024-06-01T10:00:00.000+00:00"),
                           "2024-06-01")
        self.assertLess(utc_key("2024-05-31T23:59:59.000+00:00"),
                        "2024-06-01")

    def test_event_boundary_uses_utc_order(self):
        rows = event_rows(1, 7, start="2024-06-02T10:00:00+00:00")
        # lexicographically LATEST (T14:00) but chronologically EARLIEST
        # (= 2024-06-02T09:00 UTC) row:
        rows.append(row(99, tid=1, start="2024-06-02T14:00:00+05:00"))
        events, _ = event_universe(rows, min_matches=8)
        (ev,) = events
        self.assertEqual(ev["boundary"], "2024-06-02T14:00:00+05:00")
        self.assertEqual(ev["last_start"], "2024-06-02T10:00:00+00:00")

    def test_leakage_assertion_is_offset_proof(self):
        # boundary = 22:30 UTC; fit row 23:00 UTC is AFTER it even though
        # the raw string compares before - must trip.
        boundary = "2024-06-01T00:30:00+02:00"
        fit = [row(1, start="2024-05-31T23:00:00+00:00")]
        with self.assertRaises(HarnessError):
            assert_no_leakage(fit, [], boundary)

    def test_split_classification_is_offset_proof(self):
        # last_start 2025-07-01T01:00+03:00 = 2025-06-30T22:00 UTC -> dev
        ev = {"tournament_id": 1,
              "boundary": "2025-06-20T10:00:00+00:00",
              "last_start": "2025-07-01T01:00:00+03:00"}
        split = classify_split([ev], "2025-07-01")
        self.assertEqual([e["tournament_id"] for e in split["dev"]], [1])


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

    def test_null_tournament_id_rows_excluded_loudly(self):
        # Codex W6a review P2: a NULL tid must never mint a synthetic event
        # (0 such rows in the current DB - contract guard).
        rows = event_rows(1, 8, tier="s")
        rows.append(row(99, tid=None))
        events, report = event_universe(rows, min_matches=8)
        self.assertEqual([e["tournament_id"] for e in events], [1])
        self.assertEqual(report["null_tournament_rows"], 1)


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

    def test_team_tiers_counted_from_fit_matches(self):
        # W7/F1: per-team tier exposure, window-only, None rows excluded
        win = [row(1, t1=1, t2=3, tier="s",
                   start="2023-01-01T10:00:00+00:00"),
               row(2, t1=1, t2=3, tier="a",
                   start="2023-02-01T10:00:00+00:00"),
               row(3, t1=2, t2=3, tier=None,
                   start="2023-03-01T10:00:00+00:00")]
        u = build_fit_universe(self.ev_rows, win, hops=1)
        self.assertEqual(u["team_tiers"]["1"], {"s": 1, "a": 1})
        self.assertEqual(u["team_tiers"]["3"], {"s": 1, "a": 1})
        self.assertEqual(u["team_tiers"].get("2", {}), {})

    def test_fit_match_meta_and_boundary_for_weighting(self):
        # W8/F2-F3: per-fit-match age/bo_type inputs + the boundary they
        # age against, all UTC-normalized
        win = [row(1, t1=1, t2=3, start="2023-01-01T10:00:00+00:00"),
               row(2, t1=2, t2=3, start="2023-06-01T10:00:00+00:00")]
        win[1]["bo_type"] = 1
        u = build_fit_universe(self.ev_rows, win, hops=1)
        self.assertEqual(u["boundary_utc"], "2024-01-01T10:00:00")
        self.assertEqual(len(u["fit_match_meta"]), len(u["fit_matches"]))
        self.assertEqual(u["fit_match_meta"][0],
                         {"start_utc": "2023-01-01T10:00:00", "bo_type": 3})
        self.assertEqual(u["fit_match_meta"][1],
                         {"start_utc": "2023-06-01T10:00:00", "bo_type": 1})
        # W8/F4: last window activity per team
        self.assertEqual(u["team_last_start"]["3"], "2023-06-01T10:00:00")
        self.assertEqual(u["team_last_start"]["1"], "2023-01-01T10:00:00")


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

    def test_universe_order_insensitive_ids_order_sensitive(self):
        # universe is a resolved SET (canonicalized); fit_match_ids are the
        # FIT-ORDER sequence - float gradients accumulate in that order, so
        # a reordering is a different fit and must miss the cache (Codex
        # W6a review P2).
        a = self.key()
        self.assertEqual(a, self.key(universe={"3", "2", "1"}))
        self.assertNotEqual(a, self.key(fit_match_ids=[3, 1, 2]))

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

    def test_corrupt_or_legacy_cache_file_recomputed(self):
        # Codex W6b review P1: a well-keyed but stale/corrupt cache file
        # must be treated as a miss, not served
        calls = []
        def fit_fn():
            calls.append(1)
            return {"1": 1010.5}
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "k1.json"
            legacy.write_text('{"1": 555.0}')     # pre-envelope format
            got = cached_fit(Path(tmp), "k1", fit_fn)
            self.assertEqual(got, {"1": 1010.5})
            self.assertEqual(len(calls), 1)
            corrupt = Path(tmp) / "k2.json"
            corrupt.write_text("{not json")
            got2 = cached_fit(Path(tmp), "k2", fit_fn)
            self.assertEqual(got2, {"1": 1010.5})
            self.assertEqual(len(calls), 2)


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


# -- walk-forward replay end-to-end + self-test gate (Codex W6a review P1) -------------
class ReplayFixture(DbCase):
    """Shared fixture substrate for replay-path tests (no tests here)."""

    CFG = {"harness_version": HARNESS_VERSION,
           "eligibility": {"tiers": ["s"], "min_consumer_matches": 8},
           "window_months": 24,
           "fit_universe": {"rule": "2hop", "min_obs": 3},
           "coverage_floor": 0.5,
           "holdout_split": "2027-01-01",          # everything is dev
           "seeds": {"self_test": 20260705},
           "verdict": {"mde_brier": 0.002, "ci_level": 0.95,
                       "min_events": 10}}

    @staticmethod
    def m(mid, tid, winner, start):
        # scores must be winner-consistent or the parser quarantines the row
        s1, s2 = (2, 1) if winner == "t1" else (1, 2)
        return match(mid, tournament_id=tid, s1=s1, s2=s2, winner=winner,
                     start=start)

    def replay_db(self):
        rows = []
        # window-only tournament (6 matches: below the eligibility floor,
        # still feeds the fit windows of the real events)
        for i in range(6):
            rows.append(self.m(1 + i, 400, "t1" if i % 3 else "t2",
                               f"2026-01-{10 + i:02d}T10:00:00+00:00"))
        # two eligible events, 8 matches each, teams 1 vs 2
        for i in range(8):
            rows.append(self.m(100 + i, 501, "t2" if i % 2 else "t1",
                               f"2026-06-01T{10 + i}:00:00+00:00"))
            rows.append(self.m(200 + i, 502, "t1" if i % 2 else "t2",
                               f"2026-07-01T{10 + i}:00:00+00:00"))
        db, con = self.build(rows)
        classify_promotions(con)
        con.commit()
        return db


class TestWalkForwardReplay(ReplayFixture):
    """Fixture-DB integration: the replay path itself, and the 3.2.5 gate -
    a failed self-test must block the replay BEFORE it touches real events."""

    def test_end_to_end_fixture_replay(self):
        db = self.replay_db()
        inc = load_config(CONFIGS / "incumbent_v0.json")
        with tempfile.TemporaryDirectory() as tmp:
            out = walk_forward_replay(db, self.CFG, inc, inc,
                                      cache_dir=Path(tmp))
        self.assertTrue(out["self_test_ok"])
        self.assertEqual([s["tournament_id"] for s in out["events"]],
                         [501, 502])
        self.assertEqual(out["excluded_events"], [])
        self.assertEqual(len(out["rows"]), 16)
        self.assertEqual(out["stats"]["n_events"], 2)
        self.assertEqual(out["stats"]["mean"], 0.0)   # inc-vs-inc exactly
        self.assertEqual(out["stats"]["se"], 0.0)

    def test_failed_self_test_blocks_replay(self):
        db = self.replay_db()
        inc = load_config(CONFIGS / "incumbent_v0.json")
        broken = {"ok": False, "verdicts": {"better": "INCONCLUSIVE"},
                  "expected": {"better": "DEV-SCREENED"}, "stats": {},
                  "seed": 20260705}
        with mock.patch("harness.run_self_test", return_value=broken):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(HarnessError):
                    walk_forward_replay(db, self.CFG, inc, inc,
                                        cache_dir=Path(tmp))


# -- W6b: config diff + pre-replay gates (spec 3.2) -------------------------------------
class TestDiffPaths(unittest.TestCase):
    def test_nested_leaf_paths(self):
        a = {"model": {"sigma": 50, "lr": 100.0}, "market_policy": "none"}
        b = {"model": {"sigma": 60, "lr": 100.0}, "market_policy": "none"}
        self.assertEqual(diff_paths(a, b), ["model.sigma"])

    def test_added_and_removed_keys_are_diffs(self):
        self.assertEqual(diff_paths({"a": 1}, {"a": 1, "b": 2}), ["b"])
        self.assertEqual(diff_paths({"a": 1, "b": 2}, {"a": 1}), ["b"])

    def test_lists_compared_atomically(self):
        a = {"grid": [1, 2, 3]}
        b = {"grid": [1, 2, 4]}
        self.assertEqual(diff_paths(a, b), ["grid"])

    def test_identity_fields_excluded(self):
        a = {"name": "inc", "knob_id": None, "expected_diff_paths": [],
             "sweep_family": None, "model": {"sigma": 50}}
        b = {"name": "cand", "knob_id": "k", "expected_diff_paths": ["x"],
             "sweep_family": "f", "model": {"sigma": 50},
             "screens_against": {"config": "incumbent_v1.json", "sha": "x"}}
        self.assertEqual(diff_paths(a, b), [])


class TestGateScreensAgainst(unittest.TestCase):
    """Codex F-tier review P2: a family registered against one incumbent
    must never silently re-screen against a different one."""

    def test_matching_declaration_passes(self):
        from harness import gate_screens_against
        inc = load_config(CONFIGS / "incumbent_v1.json")
        cand = {"screens_against": {"config": "incumbent_v1.json",
                                    "sha": config_sha(inc)}}
        gate_screens_against(cand, inc)          # no raise

    def test_mismatched_incumbent_blocks(self):
        from harness import gate_screens_against
        inc_v0 = load_config(CONFIGS / "incumbent_v0.json")
        inc_v1 = load_config(CONFIGS / "incumbent_v1.json")
        cand = {"screens_against": {"config": "incumbent_v0.json",
                                    "sha": config_sha(inc_v0)}}
        with self.assertRaises(HarnessError):
            gate_screens_against(cand, inc_v1)

    def test_legacy_configs_without_declaration_pass(self):
        from harness import gate_screens_against
        inc = load_config(CONFIGS / "incumbent_v1.json")
        gate_screens_against({"name": "old-family-variant"}, inc)


class TestGateConfigDiff(unittest.TestCase):
    def cand(self, **over):
        cfg = load_config(CONFIGS / "incumbent_v0.json")
        cfg["name"] = "cand"
        cfg["knob_id"] = "smoke.sigma60"
        cfg["expected_diff_paths"] = ["model.sigma"]
        cfg["model"] = dict(cfg["model"], sigma=60)
        cfg.update(over)
        return cfg

    def test_declared_diff_passes(self):
        inc = load_config(CONFIGS / "incumbent_v0.json")
        gate_config_diff(self.cand(), inc)      # no raise

    def test_incumbent_clone_blocked(self):
        # identical except identity fields = accidental clone (spec 3.2.7)
        inc = load_config(CONFIGS / "incumbent_v0.json")
        clone = self.cand(model=dict(inc["model"]),
                          expected_diff_paths=[])
        with self.assertRaises(HarnessError):
            gate_config_diff(clone, inc)

    def test_undeclared_diff_blocked(self):
        inc = load_config(CONFIGS / "incumbent_v0.json")
        with self.assertRaises(HarnessError):
            gate_config_diff(self.cand(expected_diff_paths=[]), inc)

    def test_missing_declared_diff_blocked(self):
        inc = load_config(CONFIGS / "incumbent_v0.json")
        sneaky = self.cand(expected_diff_paths=["model.sigma",
                                                "data_prep.weighting"])
        with self.assertRaises(HarnessError):
            gate_config_diff(sneaky, inc)


class TestGateParseMeta(unittest.TestCase):
    GOOD = {"reconciled": "true", "audit_ok": "true",
            "audit_version": "v1.1-test",
            "audit_input_counts": '{"reference_n": 40, "csv_n": 16}'}

    def test_good_meta_passes(self):
        gate_parse_meta(dict(self.GOOD))        # no raise

    def test_each_missing_or_bad_stamp_blocks(self):
        for k, bad in (("reconciled", None), ("reconciled", "false"),
                       ("audit_ok", None), ("audit_ok", "false"),
                       ("audit_version", None),
                       ("audit_input_counts", None),
                       ("audit_input_counts", '{"reference_n": 0, "csv_n": 16}'),
                       ("audit_input_counts", '{"reference_n": 40, "csv_n": 0}')):
            meta = dict(self.GOOD)
            if bad is None:
                del meta[k]
            else:
                meta[k] = bad
            with self.assertRaises(HarnessError, msg=(k, bad)):
                gate_parse_meta(meta)


# -- W6b: nominee selection (spec 3.1/5.2 sweep families) --------------------------------
class TestSelectNominee(unittest.TestCase):
    def res(self, name, verdict_name, mean, family="f2.halflife"):
        return {"candidate_name": name, "verdict": verdict_name,
                "sweep_family": family, "stats": {"mean": mean}}

    def test_best_dev_mean_among_screened(self):
        results = [self.res("a", "DEV-SCREENED", -0.004),
                   self.res("b", "DEV-SCREENED", -0.006),
                   self.res("c", "INCONCLUSIVE", -0.009),
                   self.res("d", "REJECT", 0.004)]
        self.assertEqual(select_nominee(results)["candidate_name"], "b")

    def test_none_screened_returns_none(self):
        results = [self.res("a", "INCONCLUSIVE", -0.001),
                   self.res("b", "BLOCKED", 0.0)]
        self.assertIsNone(select_nominee(results))

    def test_mixed_families_fail_loud(self):
        results = [self.res("a", "DEV-SCREENED", -0.004),
                   self.res("b", "DEV-SCREENED", -0.006, family="other")]
        with self.assertRaises(HarnessError):
            select_nominee(results)


# -- W6b: gated persistent runs (spec 3.2/3.3/3.4/5) --------------------------------------
def bless(con):
    """Forge the parse_meta stamps a real audit writes - gate tests exercise
    the GATES (stamp consumers); the audit itself is tested in
    test_integrity_audit."""
    for k, v in (("reconciled", "true"), ("audit_ok", "true"),
                 ("audit_version", "v1.1-test"),
                 ("audit_input_counts", '{"reference_n": 40, "csv_n": 16}')):
        con.execute("INSERT OR REPLACE INTO parse_meta VALUES (?,?)", (k, v))
    con.commit()


class TestRunReplay(ReplayFixture):
    """run_replay = gates + manifest + persisted run dir + verdict on the
    fixture substrate."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # verdict-bearing config: the 2-event fixture must clear min_events
        cfg = dict(self.CFG)
        cfg["verdict"] = dict(cfg["verdict"], min_events=2)
        self.hpath = self.root / "harness_test.json"
        self.hpath.write_text(json.dumps(cfg))
        cand = load_config(CONFIGS / "incumbent_v0.json")
        cand.update(name="cand-sigma60", knob_id="smoke.sigma60",
                    expected_diff_paths=["model.sigma"])
        cand["model"] = dict(cand["model"], sigma=60)
        self.cpath = self.root / "cand.json"
        self.cpath.write_text(json.dumps(cand))

    def run_it(self, db, **kw):
        kw.setdefault("harness_path", self.hpath)
        kw.setdefault("run_root", self.root / "runs")
        kw.setdefault("cache_dir", self.root / "cache")
        kw.setdefault("burn_log", self.root / "burn.jsonl")
        return run_replay(db, self.cpath, **kw)

    def blessed_db(self):
        db = self.replay_db()
        con = sqlite3.connect(db)
        self.addCleanup(con.close)
        bless(con)
        return db

    def test_happy_path_writes_run_dir_and_verdict(self):
        with mock.patch("harness._src_dirty", return_value=False), \
             mock.patch("harness._git_sha", return_value="abc1234"):
            out = self.run_it(self.blessed_db())
        run_dir = Path(out["run_dir"])
        for f in ("manifest.json", "rows.jsonl", "events.jsonl",
                  "verdict.json", "failures.jsonl"):
            self.assertTrue((run_dir / f).exists(), f)
        self.assertTrue((run_dir / "configs" / "cand.json").exists())
        v = json.loads((run_dir / "verdict.json").read_text())
        # sigma 60 vs 50 on 2 events, t(0.975,1)=12.7 -> INCONCLUSIVE
        self.assertEqual(v["verdict"], "INCONCLUSIVE")
        # fixture substrate has no slate-bearing tournament -> recorded, and
        # screening is impossible without the arm (guard tested separately)
        self.assertEqual(v["objective_check"]["status"],
                         "no-slate-events-computed")
        self.assertTrue(v["adoption_eligible"])
        # baseline evidence present, never gating
        self.assertAlmostEqual(
            v["evidence"]["baselines"]["uniform-0.5"]["brier"], 0.25)
        m = json.loads((run_dir / "manifest.json").read_text())
        self.assertEqual(m["git_sha"], "abc1234")
        self.assertIs(m["code_dirty"], False)
        self.assertEqual(m["event_set"],
                         [[501, "2026-06-01T10:00:00+00:00"],
                          [502, "2026-07-01T10:00:00+00:00"]])
        self.assertIn("fit_cache_keys", m)
        self.assertEqual(out["run_id"], canonical_sha(m)[:16])

    FILES = ("manifest.json", "verdict.json", "rows.jsonl",
             "events.jsonl", "failures.jsonl")

    def test_manifest_determinism_two_runs_identical_bytes(self):
        # Codex W6b review P2: equal run_ids mean the same run_dir, so the
        # bytes must be CAPTURED between runs or the comparison is vacuous
        # (second run overwrites the first).
        db = self.blessed_db()
        with mock.patch("harness._src_dirty", return_value=False), \
             mock.patch("harness._git_sha", return_value="abc1234"):
            a = self.run_it(db)
            first = {f: (Path(a["run_dir"]) / f).read_bytes()
                     for f in self.FILES}
            b = self.run_it(db)
        self.assertEqual(a["run_id"], b["run_id"])
        for f in self.FILES:
            self.assertEqual(first[f],
                             (Path(b["run_dir"]) / f).read_bytes(), f)

    def test_bad_audit_stamp_blocks(self):
        db = self.replay_db()
        con = sqlite3.connect(db)
        self.addCleanup(con.close)
        bless(con)
        con.execute("INSERT OR REPLACE INTO parse_meta VALUES "
                    "('audit_ok', 'false')")
        con.commit()
        out = self.run_it(db)
        v = json.loads((Path(out["run_dir"]) / "verdict.json").read_text())
        self.assertEqual(v["verdict"], "BLOCKED")
        self.assertNotIn("stats", v)
        self.assertTrue(any("audit_ok" in g for g in v["blocked_gates"]))

    def test_vacuous_audit_inputs_block(self):
        db = self.replay_db()
        con = sqlite3.connect(db)
        self.addCleanup(con.close)
        bless(con)
        con.execute("INSERT OR REPLACE INTO parse_meta VALUES "
                    "('audit_input_counts', '{\"reference_n\": 0, \"csv_n\": 0}')")
        con.commit()
        out = self.run_it(db)
        v = json.loads((Path(out["run_dir"]) / "verdict.json").read_text())
        self.assertEqual(v["verdict"], "BLOCKED")

    def test_incumbent_clone_candidate_blocks(self):
        clone = load_config(CONFIGS / "incumbent_v0.json")
        clone["name"] = "sneaky-clone"
        cpath = self.root / "clone.json"
        cpath.write_text(json.dumps(clone))
        db = self.blessed_db()
        out = run_replay(db, cpath, harness_path=self.hpath,
                         run_root=self.root / "runs",
                         cache_dir=self.root / "cache",
                         burn_log=self.root / "burn.jsonl")
        v = json.loads((Path(out["run_dir"]) / "verdict.json").read_text())
        self.assertEqual(v["verdict"], "BLOCKED")

    def test_below_min_events_blocks(self):
        strict = json.loads(self.hpath.read_text())
        strict["verdict"]["min_events"] = 10       # fixture has only 2
        hpath = self.root / "harness_strict.json"
        hpath.write_text(json.dumps(strict))
        out = self.run_it(self.blessed_db(), harness_path=hpath)
        v = json.loads((Path(out["run_dir"]) / "verdict.json").read_text())
        self.assertEqual(v["verdict"], "BLOCKED")
        self.assertIn("insufficient-n", v["reason"])

    def test_dirty_tree_is_exploratory_only(self):
        with mock.patch("harness._src_dirty", return_value=True), \
             mock.patch("harness._git_sha", return_value="abc1234"):
            out = self.run_it(self.blessed_db())
        v = json.loads((Path(out["run_dir"]) / "verdict.json").read_text())
        self.assertEqual(v["verdict"], "INCONCLUSIVE")   # still computed
        self.assertFalse(v["adoption_eligible"])

    def holdout_cfg_path(self):
        # move the split so both fixture events land in HOLDOUT
        cfg = json.loads(self.hpath.read_text())
        cfg["holdout_split"] = "2026-01-01"
        hpath = self.root / "harness_holdout.json"
        hpath.write_text(json.dumps(cfg))
        return hpath

    def forged_dev_screening(self, hpath):
        """A real dev run whose recorded verdict is then forged to
        DEV-SCREENED (and its harness sha aligned to the holdout config):
        the holdout gate verifies nominee provenance from the run DIR - it
        defends against workflow mistakes, not filesystem tampering - and
        the screening statistics themselves are tested elsewhere. Returns
        the nominee run_id."""
        # dev split is empty under the holdout cfg; use the normal cfg run
        with mock.patch("harness._src_dirty", return_value=False), \
             mock.patch("harness._git_sha", return_value="abc1234"):
            dev = self.run_it(self.blessed_db())
        vpath = Path(dev["run_dir"]) / "verdict.json"
        v = json.loads(vpath.read_text())
        v["verdict"] = "DEV-SCREENED"
        vpath.write_text(json.dumps(v, sort_keys=True, indent=2) + "\n")
        mpath = Path(dev["run_dir"]) / "manifest.json"
        m = json.loads(mpath.read_text())
        m["harness_config_sha"] = config_sha(json.loads(
            Path(hpath).read_text()))
        mpath.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")
        return dev["run_id"]

    def test_holdout_requires_nominee_provenance(self):
        # Codex W6b review P1: bare --holdout must never mint ADOPTED
        out = self.run_it(self.blessed_db(),
                          harness_path=self.holdout_cfg_path(),
                          split="holdout")
        v = json.loads((Path(out["run_dir"]) / "verdict.json").read_text())
        self.assertEqual(v["verdict"], "BLOCKED")
        self.assertTrue(any("nominee" in g for g in v["blocked_gates"]))

    def test_holdout_rejects_unscreened_nominee(self):
        hpath = self.holdout_cfg_path()
        with mock.patch("harness._src_dirty", return_value=False), \
             mock.patch("harness._git_sha", return_value="abc1234"):
            dev = self.run_it(self.blessed_db())     # INCONCLUSIVE, not forged
        out = self.run_it(self.blessed_db(), harness_path=hpath,
                          split="holdout", nominated_by=dev["run_id"])
        v = json.loads((Path(out["run_dir"]) / "verdict.json").read_text())
        self.assertEqual(v["verdict"], "BLOCKED")

    def test_holdout_rejects_candidate_mismatch(self):
        hpath = self.holdout_cfg_path()
        nominee = self.forged_dev_screening(hpath)
        other = json.loads(self.cpath.read_text())
        other["model"] = dict(other["model"], sigma=70)
        opath = self.root / "other.json"
        opath.write_text(json.dumps(other))
        db = self.blessed_db()
        out = run_replay(db, opath, harness_path=hpath, split="holdout",
                         nominated_by=nominee, run_root=self.root / "runs",
                         cache_dir=self.root / "cache",
                         burn_log=self.root / "burn.jsonl")
        v = json.loads((Path(out["run_dir"]) / "verdict.json").read_text())
        self.assertEqual(v["verdict"], "BLOCKED")

    def test_holdout_with_valid_nominee_runs_and_burns(self):
        hpath = self.holdout_cfg_path()
        nominee = self.forged_dev_screening(hpath)
        burn = self.root / "burn.jsonl"
        with mock.patch("harness._src_dirty", return_value=False), \
             mock.patch("harness._git_sha", return_value="abc1234"):
            out = self.run_it(self.blessed_db(), harness_path=hpath,
                              split="holdout", nominated_by=nominee,
                              burn_log=burn)
        v = json.loads((Path(out["run_dir"]) / "verdict.json").read_text())
        # sigma-60-vs-50 will not confirm on 2 events; INCONCLUSIVE, and
        # the holdout is burned REGARDLESS of outcome
        self.assertIn(v["verdict"], ("INCONCLUSIVE", "ADOPTED", "REJECT"))
        (entry,) = [json.loads(l) for l in
                    burn.read_text().splitlines() if l.strip()]
        self.assertEqual(entry["candidate_config_sha"],
                         config_sha(json.loads(self.cpath.read_text())))
        self.assertEqual(entry["result"], v["verdict"])
        self.assertEqual(entry["nominated_by_run_id"], nominee)
        self.assertGreater(entry["n_matches"], 0)
        self.assertIn("event_set_sha", entry)
        self.assertIn("date", entry)
        m = json.loads((Path(out["run_dir"]) / "manifest.json").read_text())
        self.assertTrue(m["holdout_touched"])
        self.assertEqual(m["nominated_by_run_id"], nominee)

    def test_holdout_not_burned_when_nothing_graded(self):
        # Codex W6b review P2: all rows skipped -> no holdout outcome was
        # consumed -> no burn entry
        hpath = self.holdout_cfg_path()
        cfg = json.loads(hpath.read_text())
        cfg["fit_universe"] = dict(cfg["fit_universe"], min_obs=999)
        hpath2 = self.root / "harness_minobs.json"
        hpath2.write_text(json.dumps(cfg))
        nominee = self.forged_dev_screening(hpath2)
        burn = self.root / "burn_none.jsonl"
        out = self.run_it(self.blessed_db(), harness_path=hpath2,
                          split="holdout", nominated_by=nominee,
                          burn_log=burn)
        v = json.loads((Path(out["run_dir"]) / "verdict.json").read_text())
        self.assertEqual(v["verdict"], "BLOCKED")   # post-coverage min_events
        self.assertFalse(burn.exists())

    def test_holdout_cannot_be_limited(self):
        with self.assertRaises(HarnessError):
            self.run_it(self.blessed_db(),
                        harness_path=self.holdout_cfg_path(),
                        split="holdout", limit=1)

    def test_limited_run_is_exploratory_never_verdict_bearing(self):
        # Codex W6b review P1: a --limit subset must never emit a
        # screening-grade verdict nor be adoption-eligible
        with mock.patch("harness._src_dirty", return_value=False), \
             mock.patch("harness._git_sha", return_value="abc1234"):
            out = self.run_it(self.blessed_db(), limit=2)
        v = json.loads((Path(out["run_dir"]) / "verdict.json").read_text())
        self.assertEqual(v["verdict"], "EXPLORATORY")
        self.assertFalse(v["adoption_eligible"])
        self.assertIn("stats", v)                   # evidence still shown

    def test_placeholder_in_config_kills_adoption_eligibility(self):
        # Codex W6b review P2: placeholders hidden behind config shas
        cand = json.loads(self.cpath.read_text())
        cand["source_hash"] = "pending-fit-audit"
        cand["expected_diff_paths"] = ["model.sigma", "source_hash"]
        cpath = self.root / "cand_pending.json"
        cpath.write_text(json.dumps(cand))
        db = self.blessed_db()
        with mock.patch("harness._src_dirty", return_value=False), \
             mock.patch("harness._git_sha", return_value="abc1234"):
            out = run_replay(db, cpath, harness_path=self.hpath,
                             run_root=self.root / "runs",
                             cache_dir=self.root / "cache",
                             burn_log=self.root / "burn.jsonl")
        v = json.loads((Path(out["run_dir"]) / "verdict.json").read_text())
        self.assertNotEqual(v["verdict"], "BLOCKED")
        self.assertFalse(v["adoption_eligible"])

    def test_screening_without_arm_blocks_with_artifacts(self):
        # Codex W6c review P1: the DoR-5(8) guard must produce a BLOCKED
        # run WITH artifacts, never an exception that leaves no record
        real_verdict = harness.verdict

        def forced(stats, **kw):
            if "split" in kw:      # the final classification call only -
                return {"verdict": "DEV-SCREENED", "reason": "forced",
                        "stats": stats}
            return real_verdict(stats, **kw)   # self-test stays honest

        with mock.patch("harness.verdict", side_effect=forced), \
             mock.patch("harness._src_dirty", return_value=False), \
             mock.patch("harness._git_sha", return_value="abc1234"):
            out = self.run_it(self.blessed_db())
        run_dir = Path(out["run_dir"])
        self.assertTrue((run_dir / "manifest.json").exists())
        v = json.loads((run_dir / "verdict.json").read_text())
        self.assertEqual(v["verdict"], "BLOCKED")
        self.assertNotIn("stats", v)
        self.assertTrue(any("objective" in g for g in v["blocked_gates"]))

    def test_blocked_run_never_executes_slate_arm(self):
        # Codex W6c review P2: a post-coverage-blocked run must not fit or
        # simulate the objective arm
        strict = json.loads(self.hpath.read_text())
        strict["verdict"]["min_events"] = 10       # fixture has only 2
        hpath = self.root / "harness_strict2.json"
        hpath.write_text(json.dumps(strict))
        with mock.patch("harness._slate_arm",
                        side_effect=AssertionError("arm must not run")) :
            out = self.run_it(self.blessed_db(), harness_path=hpath)
        v = json.loads((Path(out["run_dir"]) / "verdict.json").read_text())
        self.assertEqual(v["verdict"], "BLOCKED")
        self.assertEqual(v["objective_check"]["status"], "not-run")

    def test_manifest_pins_cached_ratings_content(self):
        # Codex W6b review P1: the manifest must pin WHAT the cache served,
        # not just the key
        with mock.patch("harness._src_dirty", return_value=False), \
             mock.patch("harness._git_sha", return_value="abc1234"):
            out = self.run_it(self.blessed_db())
        m = json.loads((Path(out["run_dir"]) / "manifest.json").read_text())
        for tid, keys in m["fit_cache_keys"].items():
            for tag in ("cand", "inc"):
                self.assertIn("key", keys[tag])
                self.assertEqual(len(keys[tag]["ratings_sha"]), 64, tid)


# -- W6c: slate objective check (spec 6.2) ----------------------------------------------
class TestSlateObjectiveCheck(unittest.TestCase):
    """Synthetic known answers: in a synthetic world we HAVE the generating
    truth, so the arm is validated with truth-measure draws; real runs pass
    the incumbent measure (the conservative directional block)."""

    N_SIMS = 800
    # kadv must leave >= 6 advance picks after 3-0/0-3 overlap removal
    KS = {"k30": 3, "k03": 3, "kadv": 11}

    @classmethod
    def setUpClass(cls):
        rng = __import__("random").Random(20260705)
        cls.truth = {t: rng.uniform(850, 1150) for t in STAGE3_TEAMS}
        cls.noisy = {t: r + rng.gauss(0, 30) for t, r in cls.truth.items()}
        # rank inversion: decisively wrong beliefs
        cls.inverted = {t: 2000 - r for t, r in cls.truth.items()}

    def check(self, cand, inc, eval_r):
        return slate_objective_check(cand, inc, eval_r,
                                     n_sims=self.N_SIMS, seed=7,
                                     ci_level=0.95, **self.KS)

    def test_better_candidate_passes(self):
        r = self.check(self.truth, self.noisy, self.truth)
        self.assertTrue(r["objective_ok"])
        self.assertGreaterEqual(r["mean"], 0.0)

    def test_worse_candidate_fails(self):
        r = self.check(self.inverted, self.truth, self.truth)
        self.assertFalse(r["objective_ok"])
        self.assertLess(r["mean"], 0.0)

    def test_identical_ratings_zero_delta(self):
        r = self.check(self.truth, self.truth, self.truth)
        self.assertTrue(r["objective_ok"])
        self.assertEqual(r["mean"], 0.0)
        self.assertTrue(r["identical_slates"])

    def test_pass_rule_is_declared_in_the_result(self):
        # Codex W6c review P2: the CI-aware operationalization of the
        # spec's "non-negative" must be recorded, not implicit
        r = self.check(self.truth, self.truth, self.truth)
        self.assertIn("CI-upper", r["pass_rule"])

    def test_pair_overrides_neutralized_and_restored(self):
        import simulate
        before = dict(simulate.PAIR_OVERRIDES)
        self.assertTrue(before)          # anchors exist in this repo
        self.check(self.truth, self.truth, self.truth)
        self.assertEqual(simulate.PAIR_OVERRIDES, before)

    def test_screening_guard_requires_objective_arm(self):
        from harness import _guard_screening_needs_objective
        with self.assertRaises(HarnessError):
            _guard_screening_needs_objective("DEV-SCREENED", False)
        _guard_screening_needs_objective("INCONCLUSIVE", False)  # no raise
        _guard_screening_needs_objective("DEV-SCREENED", True)   # no raise


class TestSlateCorrectDialect(unittest.TestCase):
    def test_slate_correct_agrees_with_score_slate(self):
        # single-scoring-dialect pin: optimize.score_slate's inline k and
        # the canonical slate_correct must agree on every sim
        import optimize
        rng = __import__("random").Random(3)
        teams = list(STAGE3_TEAMS)
        recs = [(3, 0), (3, 1), (3, 2), (0, 3), (1, 3), (2, 3)]
        sims = [{t: recs[rng.randrange(len(recs))] for t in teams}
                for _ in range(50)]
        c30, c03 = teams[:2], teams[2:4]
        cadv = teams[4:10]
        p5, ev = optimize.score_slate(sims, c30, c03, cadv)
        ks = [optimize.slate_correct(r, c30, c03, cadv) for r in sims]
        self.assertEqual(p5, sum(k >= optimize.PASS_THRESHOLD
                                 for k in ks) / len(sims))
        self.assertEqual(ev, sum(ks) / len(sims))


# -- W6c: Cologne known-answer cross-check (spec 6.2) --------------------------------------
class TestCologneCrossCheck(unittest.TestCase):
    def test_exact_agreement_with_committed_graded_log(self):
        r = cologne_cross_check()
        self.assertEqual(r["mismatches"], [])
        self.assertEqual(r["n"], 33)
        self.assertTrue(r["ok"])


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

    def test_baseline_uniform_schema(self):
        cfg = load_config(CONFIGS / "baseline_uniform.json")
        self.assertEqual(cfg["kind"], "baseline")
        self.assertEqual(cfg["name"], "uniform-0.5")
        self.assertEqual(cfg["rule"], "constant-0.5")

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

    def test_adopted_incumbent_v1_and_reserve_rotation(self):
        # First gate adoption (f-sigma-v1, holdout run 5d14e8b9b8a5e486):
        # incumbent_v1 = sigma 85; harness_v1 rotates the burned reserve.
        inc = load_config(CONFIGS / "incumbent_v1.json")
        self.assertEqual(inc["name"], "incumbent_v1")
        self.assertEqual(inc["model"]["sigma"], 85)
        inc0 = load_config(CONFIGS / "incumbent_v0.json")
        self.assertEqual({k: v for k, v in inc["model"].items()
                          if k != "sigma"},
                         {k: v for k, v in inc0["model"].items()
                          if k != "sigma"})
        h1 = load_config(CONFIGS / "harness_v1.json")
        self.assertEqual(h1["holdout_split"], "2026-01-01")
        self.assertEqual(h1["reserve_split"], "2026-07-01")
        h0 = load_config(CONFIGS / "harness_v0.json")
        self.assertEqual(h0["holdout_split"], "2025-07-01")  # frozen history
        # the omitted-argument defaults themselves are pinned (review P3)
        self.assertEqual(harness.DEFAULT_INCUMBENT.name, "incumbent_v1.json")
        self.assertEqual(harness.DEFAULT_HARNESS.name, "harness_v1.json")

    def test_cologne_event_config_pins_substrate_tournament(self):
        # W6c: the slate arm links EventConfig -> substrate via tournament_id
        from event_config import COLOGNE
        self.assertEqual(COLOGNE.tournament_id, 4209)

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

    def test_tier_empirical_2pass_pulls_thin_teams_to_tier_mean(self):
        # W7/F1: two tier-s teams with strong window records cluster high; a
        # THIN tier-s team (few matches) must land nearer the tier-s mean
        # than the flat-1000 base, while a tier-b team stays near base.
        eng = load_config(CONFIGS / "incumbent_v0.json")
        eng = dict(eng, model=dict(eng["model"],
                                   priors_scheme="tier-empirical-2pass"))
        # s-tier: 1 and 2 beat b-tier 3 and 4 repeatedly; 5 is a thin s-tier
        # team with one win over 3
        fit_matches = ([("1", "3", 1.0)] * 6 + [("2", "4", 1.0)] * 6
                       + [("1", "4", 1.0)] * 3 + [("2", "3", 1.0)] * 3
                       + [("5", "3", 1.0)])
        tiers = {"1": {"s": 9}, "2": {"s": 9}, "3": {"b": 10},
                 "4": {"b": 9}, "5": {"s": 1}}
        u = {"universe": {"1", "2", "3", "4", "5"},
             "fit_matches": fit_matches, "fit_match_ids": list(range(19)),
             "team_tiers": tiers}
        hier = fit_engine(eng, u)
        flat = fit_engine(load_config(CONFIGS / "incumbent_v0.json"), u)
        # tier-s mean sits above base; the thin s-team must benefit
        self.assertGreater(hier["5"], flat["5"])
        # determinism
        self.assertEqual(hier, fit_engine(eng, u))

    def test_tier_loo_priors_exclude_self_and_singletons(self):
        # Codex F-tier review P2: a team's own pass-1 rating must not feed
        # its own prior (double-counting); singleton tiers fall back to base
        from harness import _tier_loo_priors
        flat = {"1": 1100.0, "2": 1060.0, "5": 990.0}
        tiers = {"1": {"s": 5}, "2": {"s": 5}, "5": {"a": 3}}
        priors = _tier_loo_priors(flat, tiers, ["1", "2", "5"], 1000.0)
        self.assertEqual(priors["1"], 1060.0)    # mean of OTHERS in tier s
        self.assertEqual(priors["2"], 1100.0)
        self.assertEqual(priors["5"], 1000.0)    # singleton tier -> base

    def test_weighted_meta_misalignment_fails_loud(self):
        # Codex F-tier review P2: zip truncation must never drop matches
        eng = load_config(CONFIGS / "incumbent_v0.json")
        eng = dict(eng, data_prep={"weighting": {
            "scheme": "weighted", "half_life_days": 180,
            "bo1_discount": 1.0}})
        u = self.weighted_universe()
        u["fit_match_meta"] = u["fit_match_meta"][:1]
        with self.assertRaises(HarnessError):
            fit_engine(eng, u)

    def test_tier_empirical_handles_unknown_tier_teams(self):
        eng = load_config(CONFIGS / "incumbent_v0.json")
        eng = dict(eng, model=dict(eng["model"],
                                   priors_scheme="tier-empirical-2pass"))
        u = {"universe": {"1", "2"}, "fit_matches": [("1", "2", 1.0)],
             "fit_match_ids": [10], "team_tiers": {}}
        ratings = fit_engine(eng, u)     # no tier info: falls back to base
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
        bad3 = dict(eng, data_prep={"weighting": {"scheme": "mystery"}})
        with self.assertRaises(HarnessError):
            fit_engine(bad3, u)

    def weighted_universe(self):
        # team 1 beat 2 long ago; team 2 beat 1 recently (equal counts)
        return {"universe": {"1", "2"},
                "fit_matches": [("1", "2", 1.0), ("2", "1", 1.0)],
                "fit_match_ids": [10, 11],
                "boundary_utc": "2024-01-01T00:00:00",
                "fit_match_meta": [
                    {"start_utc": "2022-01-01T00:00:00", "bo_type": 3},
                    {"start_utc": "2023-12-01T00:00:00", "bo_type": 3}],
                "team_tiers": {}, "team_last_start": {}}

    def test_halflife_weighting_favors_recent_results(self):
        # W8/F2: uniform weights tie 1-1 -> ratings equal; decay makes the
        # recent win (team 2) dominate
        eng = load_config(CONFIGS / "incumbent_v0.json")
        u = self.weighted_universe()
        flat = fit_engine(eng, u)
        self.assertAlmostEqual(flat["1"], flat["2"], places=6)
        decayed = dict(eng, data_prep={"weighting": {
            "scheme": "weighted", "half_life_days": 180,
            "bo1_discount": 1.0}})
        r = fit_engine(decayed, u)
        self.assertGreater(r["2"], r["1"])

    def test_bo1_discount_downweights_bo1s(self):
        # W8/F3: same two results, but the recent win is a BO1; discounting
        # it must flip the advantage back toward the older BO3 win
        eng = load_config(CONFIGS / "incumbent_v0.json")
        u = self.weighted_universe()
        u["fit_match_meta"][1]["bo_type"] = 1
        no_decay = dict(eng, data_prep={"weighting": {
            "scheme": "weighted", "half_life_days": None,
            "bo1_discount": 0.5}})
        r = fit_engine(no_decay, u)
        self.assertGreater(r["1"], r["2"])

    def test_staleness_sigma_loosens_stale_teams(self):
        # W8/F4: teams 1 (stale, won long ago) and 2 (fresh, won recently)
        # each beat shared opponent 3 once at uniform weight. Symmetric
        # without staleness; WITH it the stale team's weaker prior pin lets
        # the same evidence move it further.
        eng = load_config(CONFIGS / "incumbent_v0.json")
        u = {"universe": {"1", "2", "3"},
             "fit_matches": [("1", "3", 1.0), ("2", "3", 1.0)],
             "fit_match_ids": [10, 11],
             "boundary_utc": "2024-01-01T00:00:00",
             "fit_match_meta": [
                 {"start_utc": "2023-01-01T00:00:00", "bo_type": 3},
                 {"start_utc": "2023-12-25T00:00:00", "bo_type": 3}],
             "team_tiers": {},
             "team_last_start": {"1": "2023-01-01T00:00:00",
                                 "2": "2023-12-25T00:00:00",
                                 "3": "2023-12-25T00:00:00"}}
        base = fit_engine(eng, u)
        self.assertAlmostEqual(base["1"], base["2"], places=6)
        cfg = dict(eng, model=dict(eng["model"],
                                   staleness={"sigma_per_year": 100}))
        r = fit_engine(cfg, u)
        self.assertGreater(r["1"], r["2"])


if __name__ == "__main__":
    unittest.main()
