"""Event-config loader tests (W5).

Two jobs: (1) prove the Cologne config reproduces the EXACT hard-coded event
facts the locked fit + probability tables were generated under (byte-identity
guard), and (2) prove the loader validates an arbitrary second event so the
W6 harness can replay many events.
"""

import json
import unittest

from event_config import COLOGNE, EventConfig, load_event

# The locked literals (model.STAGE3_TEAMS, simulate.ROUND1/SEED) the v1-v3
# tables + ratings_fitted.json were generated under. The config MUST match
# these exactly or reproducibility breaks.
LOCKED_TEAMS = ["Vitality", "NAVI", "MOUZ", "Falcons", "MongolZ", "Aurora",
                "FURIA", "PARIVISION", "Spirit", "FUT", "G2", "9z", "BetBoom",
                "Legacy", "Monte", "B8"]
LOCKED_ROUND1 = [("Vitality", "FUT"), ("NAVI", "Spirit"), ("MOUZ", "Legacy"),
                 ("Falcons", "G2"), ("MongolZ", "BetBoom"), ("Aurora", "Monte"),
                 ("FURIA", "B8"), ("PARIVISION", "9z")]
LOCKED_SEED = {"Vitality": 1, "NAVI": 2, "Falcons": 3, "MongolZ": 4,
               "Aurora": 5, "FURIA": 6, "MOUZ": 7, "PARIVISION": 8,
               "FUT": 9, "Spirit": 10, "G2": 11, "BetBoom": 12,
               "Monte": 13, "B8": 14, "Legacy": 15, "9z": 16}


class TestCologneFidelity(unittest.TestCase):
    def test_teams_order_matches_locked_literal(self):
        self.assertEqual(COLOGNE.teams, LOCKED_TEAMS)

    def test_teams_is_a_list(self):
        # Must stay the same type the in-code literal was (list), so any
        # downstream behavior is byte-identical, not merely equivalent.
        self.assertIsInstance(COLOGNE.teams, list)

    def test_round1_matches_locked_literal(self):
        self.assertEqual(COLOGNE.round1, LOCKED_ROUND1)

    def test_round1_entries_are_tuples(self):
        for m in COLOGNE.round1:
            self.assertIsInstance(m, tuple)

    def test_seeds_match_locked_literal(self):
        self.assertEqual(COLOGNE.seeds, LOCKED_SEED)

    def test_scoring_matches_valve_stage3(self):
        s = COLOGNE.scoring
        self.assertEqual(s["exact_3_0"], 2)
        self.assertEqual(s["exact_0_3"], 2)
        self.assertEqual(s["advance"], 6)
        self.assertEqual(s["pass_threshold"], 5)
        self.assertEqual(s["slate_size"], 10)

    def test_playoffs_format(self):
        self.assertEqual(COLOGNE.playoffs["teams"], 8)
        self.assertTrue(COLOGNE.playoffs["grand_final_bo5"])


class TestValidation(unittest.TestCase):
    BASE = {
        "event_id": "synthetic_4",
        "name": "Synthetic 4-team",
        "teams": ["A", "B", "C", "D"],
        "round1": [["A", "B"], ["C", "D"]],
        "seeds": {"A": 1, "B": 2, "C": 3, "D": 4},
        "scoring": {"exact_3_0": 1, "exact_0_3": 1, "advance": 2,
                    "pass_threshold": 2, "slate_size": 4},
        "playoffs": {"teams": 4, "grand_final_bo5": False},
        "format": {"type": "swiss", "series": "bo3"},
        "ruleset_version": "test",
        "optimizer_objective": "p_ge_threshold",
        "lock_timestamp": None,
    }

    def _cfg(self, **override):
        d = json.loads(json.dumps(self.BASE))
        d.update(override)
        return EventConfig.from_dict(d)

    def test_valid_synthetic_loads_and_validates(self):
        cfg = self._cfg()
        self.assertEqual(cfg.teams, ["A", "B", "C", "D"])
        self.assertEqual(cfg.round1, [("A", "B"), ("C", "D")])
        self.assertEqual(cfg.seeds["C"], 3)

    def test_duplicate_team_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self._cfg(teams=["A", "A", "C", "D"])

    def test_round1_must_cover_every_team_exactly_once(self):
        with self.assertRaisesRegex(ValueError, "round1"):
            self._cfg(round1=[["A", "B"], ["A", "C"]])  # D missing, A twice

    def test_round1_unknown_team_rejected(self):
        with self.assertRaisesRegex(ValueError, "round1"):
            self._cfg(round1=[["A", "B"], ["C", "X"]])  # X not a team

    def test_seeds_missing_team_rejected(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            self._cfg(seeds={"A": 1, "B": 2, "C": 3})  # D missing

    def test_seeds_must_be_bijection_1_to_n(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            self._cfg(seeds={"A": 1, "B": 2, "C": 3, "D": 3})  # dup rank, no 4

    def test_scoring_missing_key_rejected(self):
        bad = dict(self.BASE["scoring"])
        del bad["advance"]
        with self.assertRaisesRegex(ValueError, "scoring"):
            self._cfg(scoring=bad)


class TestLoadEvent(unittest.TestCase):
    def test_load_event_round_trips_cologne(self):
        from event_config import EVENTS
        cfg = load_event(EVENTS / "cologne_major.json")
        self.assertEqual(cfg.teams, LOCKED_TEAMS)
        self.assertEqual(cfg.event_id, COLOGNE.event_id)


if __name__ == "__main__":
    unittest.main()
