"""bo3.gg archiver tests: chunking, state, fail-loud, verify — no network."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from bo3gg_archive import Archiver, verify_archive


def fake_response(offset, limit=100, total=250):
    n = max(0, min(limit, total - offset))
    return json.dumps({
        "total": {"count": total, "offset": offset, "limit": limit},
        "results": [{"id": offset + i, "status": "finished"}
                    for i in range(n)],
    })


def make_fetch(total=250, fail_offsets=(), bodies=None):
    calls = []

    def fetch(offset, limit):
        calls.append(offset)
        if bodies is not None and offset in bodies:
            return bodies[offset]
        if offset in fail_offsets:
            raise OSError(f"simulated 500 at {offset}")
        return fake_response(offset, limit, total)

    fetch.calls = calls
    return fetch


class TestArchiver(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def archiver(self, **kw):
        kw.setdefault("fetch", make_fetch())
        kw.setdefault("sleep", lambda s: None)
        kw.setdefault("pages_per_chunk", 2)
        return Archiver(self.dir, **kw)

    def test_full_run_archives_all_pages_and_advances_state(self):
        a = self.archiver()
        a.run()
        state = json.load(open(self.dir / "state.json"))
        self.assertEqual(state["next_offset"], 300)  # 3 pages of 100 fetched
        chunks = sorted(self.dir.glob("matches_*.jsonl.gz"))
        self.assertEqual(len(chunks), 2)  # 3 pages at 2 per chunk
        lines = []
        for c in chunks:
            with gzip.open(c, "rt") as f:
                lines += [json.loads(l) for l in f]
        self.assertEqual([l["offset"] for l in lines], [0, 100, 200])
        # body is the verbatim text, round-trippable
        self.assertEqual(json.loads(lines[0]["body"])["total"]["count"], 250)

    def test_resume_continues_from_cursor_without_refetch(self):
        fetch = make_fetch()
        a = self.archiver(fetch=fetch)
        a.run(max_pages=1)
        self.assertEqual(fetch.calls, [0])
        a2 = self.archiver(fetch=fetch)
        a2.run()
        # offset 300 is the tail probe (0 results, not archived) — required
        # so a top-up run can discover new matches. No offset fetched twice.
        self.assertEqual(fetch.calls, [0, 100, 200, 300])
        self.assertEqual(len(fetch.calls), len(set(fetch.calls)))

    def test_max_pages_caps_run(self):
        a = self.archiver()
        a.run(max_pages=1)
        state = json.load(open(self.dir / "state.json"))
        self.assertEqual(state["next_offset"], 100)

    def test_transient_failure_retries_then_succeeds(self):
        # First attempt at offset 100 raises; retry succeeds.
        flaky = {"raised": False}
        base = make_fetch()

        def fetch(offset, limit):
            if offset == 100 and not flaky["raised"]:
                flaky["raised"] = True
                raise OSError("simulated transient")
            return base(offset, limit)

        a = self.archiver(fetch=fetch, max_tries=3)
        a.run()
        state = json.load(open(self.dir / "state.json"))
        self.assertEqual(state["next_offset"], 300)

    def test_malformed_body_aborts_without_advancing_state(self):
        a = self.archiver(fetch=make_fetch(bodies={100: "<html>cf block</html>"}))
        with self.assertRaises(ValueError):
            a.run()
        state = json.load(open(self.dir / "state.json"))
        self.assertEqual(state["next_offset"], 100)  # page 0 kept, bad page not

    def test_shrinking_total_aborts(self):
        fetch = make_fetch(total=250)
        a = self.archiver(fetch=fetch)
        a.run(max_pages=1)
        a2 = self.archiver(fetch=make_fetch(total=90))  # count went down
        with self.assertRaises(ValueError):
            a2.run()

    def test_verify_passes_on_good_archive_and_catches_gaps(self):
        a = self.archiver()
        a.run()
        report = verify_archive(self.dir)
        self.assertEqual(report["pages"], 3)
        self.assertEqual(report["rows"], 250)
        self.assertTrue(report["offsets_contiguous"])
        # Corrupt: drop the middle line from the first chunk
        first = sorted(self.dir.glob("matches_*.jsonl.gz"))[0]
        with gzip.open(first, "rt") as f:
            lines = f.read().splitlines()
        with gzip.open(first, "wt") as f:
            f.write(lines[0] + "\n")
        report = verify_archive(self.dir)
        self.assertFalse(report["offsets_contiguous"])


if __name__ == "__main__":
    unittest.main()
