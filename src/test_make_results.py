"""make_results tests: refuses incomplete stages, shapes records correctly."""

import json
import unittest
from pathlib import Path

from make_results import make_results

DATA = Path(__file__).resolve().parent.parent / "data"


class TestMakeResults(unittest.TestCase):
    def test_incomplete_stage_refused(self):
        # The real mid-stage live_state must be rejected, not half-graded.
        live = json.load(open(DATA / "live_state.json"))
        completed = [tuple(m) for m in live["completed"]]
        if len(completed) < 33:  # a full Swiss is 33 series
            with self.assertRaises(ValueError):
                make_results(completed)

    def test_complete_stage_shapes_records(self):
        from collections import Counter
        from model import STAGE3_TEAMS
        from test_playoffs import play_recorded_swiss
        ratings = {t: 1000.0 for t in STAGE3_TEAMS}
        completed = play_recorded_swiss(ratings, seed=7)
        results = make_results(completed)
        self.assertEqual(len(results), 16)
        self.assertTrue(all(isinstance(v, list) and len(v) == 2
                            for v in results.values()))
        multiset = Counter(tuple(v) for v in results.values())
        self.assertEqual(multiset[(3, 0)], 2)
        self.assertEqual(multiset[(0, 3)], 2)
        # JSON round-trip matches postmortem.py's expected shape
        rt = json.loads(json.dumps(results))
        self.assertEqual({t: tuple(r) for t, r in rt.items()},
                         {t: tuple(r) for t, r in results.items()})


if __name__ == "__main__":
    unittest.main()
