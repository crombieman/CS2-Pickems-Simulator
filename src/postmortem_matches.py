"""Per-match postmortem: the v3 LOCK model vs the market closing line.

Pre-registered 2026-06-12, mid-stage, BEFORE final results are known —
the grading rule is frozen now so June 15 can't shop for a flattering
metric (deep-dive 2026-06-12, recommendation 5; E.7-compatible: this
measures, it does not turn knobs).

Model under test: the frozen v3 lock — ratings_locked_v3.json plus the
frozen pre-stage pair overrides (market_anchors.json), exactly what
priced the entered slate. For pairs the lock never anchored (most R2+
matches) that is the rating-implied prob: we are grading the lock
model's FORESIGHT, with no in-stage updating. The market baseline gets
to move (closing line per match); the model does not. Beating closing
lines is the strong-form test; losing to them is expected and fine.

Market closing line: the LAST odds_archive.jsonl snapshot strictly
before match start (start times live in the results file). Snapshots
are 2-hourly and may be in-play, so a pre-start cut is the only honest
"closing" definition here; matches with no pre-start row fall back to
the earliest available row and are FLAGGED in the output.

Usage (fill data/results_matches_stage3.json incrementally as rounds
finish, then):
  python src/postmortem_matches.py

results_matches_stage3.json: {"matches": [{"a", "b", "winner",
"start" (ISO UTC), "round"}, ...]} — a/b in announced orientation,
winner one of a/b, start from Liquipedia (two-source the winners).
"""

import json
import math
from datetime import datetime
from pathlib import Path

from fetch_anchors import ALIASES

DATA = Path(__file__).resolve().parent.parent / "data"


def _team(label):
    return ALIASES.get(label.strip().lower())


def load_archive(path=DATA / "odds_archive.jsonl"):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def closing_prob(archive, a, b, start_iso):
    """(P(a wins) at close, fallback_flag) from the archive, or (None, False).

    Close = last snapshot with ts < start. No pre-start row -> earliest
    row instead, flagged True (likely in-play; treat with suspicion)."""
    start = datetime.fromisoformat(start_iso)
    rows = []
    for r in archive:
        o = r.get("outcomes") or []
        if len(o) != 2:
            continue
        mapped = (_team(o[0]), _team(o[1]))
        if set(mapped) != {a, b}:
            continue
        p_a = r["prices"][0] if mapped[0] == a else r["prices"][1]
        rows.append((datetime.fromisoformat(r["ts"]), p_a))
    if not rows:
        return None, False
    rows.sort()
    pre = [(ts, p) for ts, p in rows if ts < start]
    if pre:
        return pre[-1][1], False
    return rows[0][1], True


def grade_matches(rows):
    """Aggregate Brier + log score for model and market over per-match rows
    [{a, b, winner, p_model, p_market, flagged}]. Probabilities are P(a wins);
    scoring is vs the indicator of the actual winner."""
    agg = {"model": {"brier": 0.0, "log": 0.0},
           "market": {"brier": 0.0, "log": 0.0}}
    n = 0
    for r in rows:
        y = 1.0 if r["winner"] == r["a"] else 0.0
        for side, key in (("model", "p_model"), ("market", "p_market")):
            p = min(max(r[key], 1e-9), 1.0 - 1e-9)
            agg[side]["brier"] += (p - y) ** 2
            agg[side]["log"] += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        n += 1
    for side in agg:
        for k in agg[side]:
            agg[side][k] /= max(n, 1)
    return {"n": n, "model": agg["model"], "market": agg["market"],
            "delta_brier": agg["market"]["brier"] - agg["model"]["brier"],
            "delta_log": agg["market"]["log"] - agg["model"]["log"]}


def main():
    from model import win_prob
    from simulate import PAIR_OVERRIDES

    ratings = json.load(open(DATA / "ratings_locked_v3.json"))
    matches = json.load(open(DATA / "results_matches_stage3.json"))["matches"]
    archive = load_archive()

    rows, skipped = [], []
    for m in matches:
        a, b, start = m["a"], m["b"], m["start"]
        p_model = PAIR_OVERRIDES.get((a, b))
        if p_model is None:
            p_model = win_prob(ratings, a, b)
        p_market, flagged = closing_prob(archive, a, b, start)
        if p_market is None:
            skipped.append((a, b, "no market rows"))
            continue
        rows.append({"a": a, "b": b, "winner": m["winner"], "round": m.get("round", "?"),
                     "p_model": p_model, "p_market": p_market, "flagged": flagged})

    print(f"v3 lock model vs market closing line, {len(rows)} matches"
          f" ({len(skipped)} skipped: {skipped if skipped else 'none'})\n")
    print(f"{'Match':26s} {'rnd':>4s} {'win':12s} {'model':>7s} {'close':>7s}"
          f" {'mod-B':>7s} {'mkt-B':>7s}")
    for r in sorted(rows, key=lambda r: r["round"]):
        y = 1.0 if r["winner"] == r["a"] else 0.0
        mb = (r["p_model"] - y) ** 2
        kb = (r["p_market"] - y) ** 2
        flag = " [in-play close]" if r["flagged"] else ""
        print(f"{r['a'] + ' v ' + r['b']:26s} {r['round']:>4s} {r['winner']:12s}"
              f" {r['p_model']:7.3f} {r['p_market']:7.3f} {mb:7.3f} {kb:7.3f}{flag}")

    g = grade_matches(rows)
    print(f"\n{'':14s} {'Brier':>8s} {'log':>8s}")
    print(f"{'v3 lock model':14s} {g['model']['brier']:8.4f} {g['model']['log']:8.4f}")
    print(f"{'market close':14s} {g['market']['brier']:8.4f} {g['market']['log']:8.4f}")
    print(f"\ndelta (market - model; + = model beat the closing line): "
          f"Brier {g['delta_brier']:+.4f}, log {g['delta_log']:+.4f}")
    print("Read with E.7 humility: one stage, correlated matches, "
          "n is tiny. This logs evidence; the backtest harness rules.")


if __name__ == "__main__":
    main()
