"""Polymarket fetcher: odds archive + live anchors. Pure stdlib.

Usage:
  python src/fetch_anchors.py                # snapshot -> data/odds_archive.jsonl
  python src/fetch_anchors.py --live-anchors # ...also write data/live_anchors.json

Pages the Polymarket gamma API's open markets (by 24h volume), filters
Counter-Strike markets for the configured event, and:
  - APPENDS every matched market to data/odds_archive.jsonl (one JSON line
    per market per snapshot: timestamp, slug, outcomes, two-sided mid
    prices, volume, liquidity). The archive is append-only by design —
    it is the historical-odds dataset the backtest harness needs, and
    odds are unobtainable retroactively. Run often; duplicates are fine.
  - With --live-anchors: writes data/live_anchors.json for markets whose
    BOTH outcomes map to Stage 3 teams (consumed by live.py as match-prob
    overrides for upcoming pairings).

The gamma API is Polymarket's public read API (no auth). Mid prices from
deep books (>$80K) are treated as vig-free; pairs are normalized to sum
to 1 regardless.
"""

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
GAMMA = "https://gamma-api.polymarket.com/markets"
EVENT_FILTER = "IEM Cologne Major"   # substring of market question
PAGES = 8                            # x100 markets by 24h volume
MIN_VOLUME = 1000.0                  # skip untraded market-maker ladders

# Polymarket outcome label -> model team name (model.STAGE3_TEAMS)
ALIASES = {
    "vitality": "Vitality", "team vitality": "Vitality",
    "fut esports": "FUT", "fut": "FUT",
    "natus vincere": "NAVI", "navi": "NAVI",
    "spirit": "Spirit", "team spirit": "Spirit",
    "mouz": "MOUZ",
    "legacy": "Legacy",
    "team falcons": "Falcons", "falcons": "Falcons",
    "g2": "G2", "g2 esports": "G2",
    "themongolz": "MongolZ", "the mongolz": "MongolZ",
    "betboom team": "BetBoom", "betboom": "BetBoom",
    "aurora gaming": "Aurora", "aurora": "Aurora",
    "monte": "Monte",
    "furia": "FURIA",
    "b8": "B8",
    "parivision": "PARIVISION",
    "9z": "9z", "9z team": "9z",
}


def fetch_page(offset):
    url = (f"{GAMMA}?closed=false&order=volume24hr&ascending=false"
           f"&limit=100&offset={offset}")
    # Polymarket's CDN 403s urllib's default Python-urllib UA; curl-like is fine.
    req = urllib.request.Request(url, headers={
        "Accept": "application/json", "User-Agent": "curl/8.4.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def _maybe_json(v):
    return json.loads(v) if isinstance(v, str) else v


def snapshot(event_filter=EVENT_FILTER, pages=PAGES):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for page in range(pages):
        try:
            markets = fetch_page(page * 100)
        except OSError as e:
            print(f"page {page} fetch failed: {e}")
            break
        if not markets:
            break
        for m in markets:
            q = m.get("question") or ""
            if "Counter-Strike" not in q or event_filter not in q:
                continue
            outcomes = _maybe_json(m.get("outcomes"))
            prices = [float(p) for p in _maybe_json(m.get("outcomePrices"))]
            total = sum(prices)
            rows.append({
                "ts": ts,
                "slug": m.get("slug"),
                "question": q,
                "outcomes": outcomes,
                "prices": [p / total for p in prices] if total else prices,
                "raw_prices": prices,
                "volume": float(m.get("volumeNum") or 0),
                "liquidity": float(m.get("liquidityNum") or 0),
                "end_date": m.get("endDate"),
            })
    return rows


def append_archive(rows, path=DATA / "odds_archive.jsonl"):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_live_anchors(rows, path=DATA / "live_anchors.json"):
    anchors = []
    for r in rows:
        if r["volume"] < MIN_VOLUME or len(r["outcomes"]) != 2:
            continue
        a = ALIASES.get(r["outcomes"][0].strip().lower())
        b = ALIASES.get(r["outcomes"][1].strip().lower())
        if not a or not b:
            continue
        anchors.append({"a": a, "b": b, "p": round(r["prices"][0], 4),
                        "slug": r["slug"], "volume": round(r["volume"])})
    out = {"comment": f"Polymarket gamma mids, fetched {rows[0]['ts'] if rows else '?'}; "
                      "normalized two-sided; consumed by live.py as pair overrides.",
           "anchors": anchors}
    json.dump(out, open(path, "w"), indent=2)
    return anchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-anchors", action="store_true",
                    help="also write data/live_anchors.json for live.py")
    ap.add_argument("--event", default=EVENT_FILTER)
    args = ap.parse_args()

    rows = snapshot(event_filter=args.event)
    if not rows:
        print("No matching open markets found (event over, or markets "
              "below the top pages by 24h volume).")
        return
    append_archive(rows)
    print(f"Archived {len(rows)} markets -> data/odds_archive.jsonl")
    for r in sorted(rows, key=lambda r: -r["volume"]):
        o, p = r["outcomes"], r["prices"]
        print(f"  {o[0]} {p[0]:.3f} / {o[1]} {p[1]:.3f}   "
              f"(vol ${r['volume']:,.0f})  {r['slug']}")
    if args.live_anchors:
        anchors = write_live_anchors(rows)
        print(f"Wrote {len(anchors)} live anchors -> data/live_anchors.json")


if __name__ == "__main__":
    main()
