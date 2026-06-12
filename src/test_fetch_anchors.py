"""Anchor-writer tests: filtering and output format, no network."""

import json
import tempfile
import unittest
from pathlib import Path

from fetch_anchors import write_live_anchors, write_playoff_anchors

ROWS = [
    # Good QF market: both aliases map, healthy volume, live price.
    {"ts": "t", "slug": "qf1", "question": "q", "volume": 50000.0,
     "outcomes": ["FURIA", "Aurora Gaming"], "prices": [0.62, 0.38]},
    # Near-resolved market: must be skipped (finished match, not a forecast).
    {"ts": "t", "slug": "done", "question": "q", "volume": 900000.0,
     "outcomes": ["Natus Vincere", "Legacy"], "prices": [0.9995, 0.0005]},
    # Untraded ladder: below MIN_VOLUME, skipped.
    {"ts": "t", "slug": "thin", "question": "q", "volume": 10.0,
     "outcomes": ["Team Spirit", "MOUZ"], "prices": [0.7, 0.3]},
    # Unknown outcome label (non-Stage-3 team): skipped.
    {"ts": "t", "slug": "other", "question": "q", "volume": 50000.0,
     "outcomes": ["Astralis", "MOUZ"], "prices": [0.5, 0.5]},
]


class TestAnchorWriters(unittest.TestCase):
    def _check(self, writer):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.json"
            anchors = writer(ROWS, path=path)
            on_disk = json.load(open(path))["anchors"]
        self.assertEqual(anchors, on_disk)
        self.assertEqual(len(anchors), 1)
        a = anchors[0]
        self.assertEqual((a["a"], a["b"], a["p"]), ("FURIA", "Aurora", 0.62))
        return on_disk

    def test_live_writer_filters_and_maps(self):
        self._check(write_live_anchors)

    def test_playoff_writer_filters_and_maps(self):
        self._check(write_playoff_anchors)

    def test_playoff_writer_comment_names_consumer(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.json"
            write_playoff_anchors(ROWS, path=path)
            comment = json.load(open(path))["comment"]
        self.assertIn("playoffs.py", comment)


if __name__ == "__main__":
    unittest.main()
