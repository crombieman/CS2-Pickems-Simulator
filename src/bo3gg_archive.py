"""bo3.gg archive-first fetcher (roadmap 2.2, slice 1). Pure stdlib.

Persists every raw API response VERBATIM before anything parses it —
the API is bo3.gg's undocumented internal frontend API (no contract;
see docs/plans/2026-06-12-bo3gg-archive-design.md). If it changes shape
or disappears, everything already fetched is kept forever.

Usage:
  python src/bo3gg_archive.py --max-pages 2      # smoke / incremental
  python src/bo3gg_archive.py                    # continue to the tail
  python src/bo3gg_archive.py --verify           # archive integrity check

Crawl design: finished matches sorted by start_date ASCENDING — history
never reorders, new matches append at the tail, so the single
next_offset cursor in data/bo3gg/state.json is exactly resumable and
drift-free. Re-running is always safe: nothing is ever re-fetched,
nothing is ever overwritten (append-only gzip chunks).

Politeness is mandatory design, not a nicety: >=1.5s + jitter between
requests, exponential backoff on failure, hard per-run page caps
available. Never give them a reason to add walls.

Fail-loud: a non-JSON body, missing keys, or a shrinking total count
aborts WITHOUT advancing state; the offending payload is printed.
Dedup/filtering are parse-time concerns (later slice), never fetch-time.
"""

import argparse
import gzip
import json
import random
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "bo3gg"

BASE = ("https://api.bo3.gg/api/v1/matches"
        "?filter%5Bmatches.status%5D%5Beq%5D=finished"
        "&sort=start_date&with=teams"
        "&page%5Blimit%5D={limit}&page%5Boffset%5D={offset}")
LIMIT = 100
PAGES_PER_CHUNK = 50
MIN_DELAY = 1.5


def http_fetch(offset, limit=LIMIT):
    url = BASE.format(limit=limit, offset=offset)
    req = urllib.request.Request(url, headers={
        "Accept": "application/json", "User-Agent": "curl/8.4.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


class Archiver:
    def __init__(self, data_dir=DATA, fetch=http_fetch, sleep=time.sleep,
                 pages_per_chunk=PAGES_PER_CHUNK, max_tries=5):
        self.dir = Path(data_dir)
        self.fetch = fetch
        self.sleep = sleep
        self.pages_per_chunk = pages_per_chunk
        self.max_tries = max_tries

    # -- state ---------------------------------------------------------
    def _state_path(self):
        return self.dir / "state.json"

    def _load_state(self):
        if self._state_path().exists():
            return json.load(open(self._state_path()))
        return {"next_offset": 0, "chunk_index": 0,
                "total_count_last_seen": None, "last_run": None}

    def _save_state(self, state):
        state["last_run"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        tmp = self._state_path().with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        tmp.replace(self._state_path())

    # -- fetch with retries --------------------------------------------
    def _fetch_with_retries(self, offset):
        for attempt in range(1, self.max_tries + 1):
            try:
                return self.fetch(offset, LIMIT)
            except OSError as e:
                if attempt == self.max_tries:
                    raise
                wait = min(60.0, 2.0 ** attempt)
                print(f"  fetch offset={offset} failed ({e}); "
                      f"retry {attempt}/{self.max_tries - 1} in {wait:.0f}s")
                self.sleep(wait)

    @staticmethod
    def _parse_page(body):
        """Validate the contract we depend on; raise ValueError loudly."""
        try:
            page = json.loads(body)
        except json.JSONDecodeError:
            raise ValueError(f"non-JSON response (API change or block?): "
                             f"{body[:300]!r}")
        if "total" not in page or "results" not in page:
            raise ValueError(f"missing total/results keys: "
                             f"{json.dumps(page)[:300]}")
        return page

    # -- main loop ------------------------------------------------------
    def run(self, max_pages=None):
        self.dir.mkdir(parents=True, exist_ok=True)
        state = self._load_state()
        pages_done = 0
        while max_pages is None or pages_done < max_pages:
            offset = state["next_offset"]
            body = self._fetch_with_retries(offset)
            page = self._parse_page(body)
            total = page["total"]["count"]
            prev_total = state.get("total_count_last_seen")
            if prev_total is not None and total < prev_total * 0.9:
                raise ValueError(
                    f"total count shrank {prev_total} -> {total}: dataset or "
                    f"filter semantics changed upstream; not advancing state")
            if offset >= total or not page["results"]:
                print(f"Reached the tail: offset {offset} >= total {total}.")
                break

            chunk = self.dir / f"matches_{state['chunk_index']:04d}.jsonl.gz"
            line = json.dumps({
                "fetched_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"),
                "url": BASE.format(limit=LIMIT, offset=offset),
                "offset": offset,
                "body": body,
            })
            with gzip.open(chunk, "at") as f:
                f.write(line + "\n")

            state["next_offset"] = offset + LIMIT
            state["total_count_last_seen"] = total
            pages_in_chunk = (offset // LIMIT) % self.pages_per_chunk
            if pages_in_chunk == self.pages_per_chunk - 1:
                state["chunk_index"] += 1
            self._save_state(state)

            pages_done += 1
            n = len(page["results"])
            print(f"  page offset={offset}: {n} matches -> {chunk.name} "
                  f"({state['next_offset']}/{total})")
            if state["next_offset"] < total:
                self.sleep(MIN_DELAY + random.random())
        return state


def verify_archive(data_dir=DATA):
    """Integrity report: every line parses, offsets contiguous, row total."""
    data_dir = Path(data_dir)
    offsets, rows = [], 0
    for chunk in sorted(data_dir.glob("matches_*.jsonl.gz")):
        with gzip.open(chunk, "rt") as f:
            for line in f:
                rec = json.loads(line)
                page = json.loads(rec["body"])
                offsets.append(rec["offset"])
                rows += len(page["results"])
    offsets.sort()
    contiguous = all(b - a == LIMIT for a, b in zip(offsets, offsets[1:]))
    return {"pages": len(offsets), "rows": rows,
            "offsets_contiguous": contiguous,
            "first_offset": offsets[0] if offsets else None,
            "last_offset": offsets[-1] if offsets else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=None,
                    help="cap pages this run (default: crawl to the tail)")
    ap.add_argument("--verify", action="store_true",
                    help="verify archive integrity instead of fetching")
    args = ap.parse_args()
    if args.verify:
        report = verify_archive()
        print(json.dumps(report, indent=2))
        if not report["offsets_contiguous"]:
            raise SystemExit("ARCHIVE GAP DETECTED")
        return
    state = Archiver().run(max_pages=args.max_pages)
    print(f"Done. next_offset={state['next_offset']}, "
          f"total={state['total_count_last_seen']}")


if __name__ == "__main__":
    main()
