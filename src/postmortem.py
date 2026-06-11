"""Post-stage calibration grading: Brier scores vs locked probabilities.

Usage (after the stage ends):
  1. Fill data/results_stage3.json with actual final records:
       {"Vitality": [3, 0], "FUT": [1, 3], ...}   (all 16 teams)
  2. python src/postmortem.py

Grades BOTH locked probability tables on three binary events per team —
exactly 3-0, advance (3-1/3-2), exactly 0-3 — against a uniform baseline
(p30=2/16, padv=6/16, p03=2/16). Lower Brier is better.

  v1: data/stage3_probs_locked_v1.json (2026-06-10 afternoon; 5 anchors,
      73 series). The original pre-event lock.
  v2: data/stage3_probs.json (2026-06-10 evening; 8 Polymarket anchors,
      87 series). The refresh the final picks were made from.

The v1-vs-v2 skill comparison answers the question that matters: did the
pre-event market/data refresh actually improve calibration? Also scores
both slates' ticks. Per the README: the Brier table is the postmortem
that matters, not whether a slate passed.
"""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# v1 locked slate (original lock, 2026-06-10 afternoon). Manual tie-break
# vs pipeline argmax at the time: Falcons in 3-0 over FURIA.
SLATE_V1 = {
    "30": ["Vitality", "Falcons"],
    "03": ["9z", "B8"],
    "adv": ["Spirit", "NAVI", "FURIA", "MOUZ", "PARIVISION", "Aurora"],
}
# v2 final slate (re-lock on refreshed model, 2026-06-10 evening).
# Pipeline argmax, stable across seeds 7/11/42/123.
SLATE_V2 = {
    "30": ["Vitality", "Spirit"],
    "03": ["B8", "Monte"],
    "adv": ["NAVI", "Falcons", "FURIA", "Aurora", "MOUZ", "G2"],
}

BASELINE = {"p30": 2 / 16, "padv": 6 / 16, "p03": 2 / 16}
CATS = ("p30", "padv", "p03")


def brier_table(name, probs, outcome):
    print(f"\n--- {name} ---")
    print(f"{'Team':12s} {'rec':>5s}" +
          "".join(f"  {c+' brier':>11s}" for c in CATS))
    model_b = {c: 0.0 for c in CATS}
    base_b = {c: 0.0 for c in CATS}
    for t in sorted(probs, key=lambda t: -probs[t]["pany"]):
        rec = outcome[t]["rec"]
        row = f"{t:12s} {rec[0]}-{rec[1]:1d}"
        for c in CATS:
            b = (probs[t][c] - outcome[t][c]) ** 2
            model_b[c] += b
            base_b[c] += (BASELINE[c] - outcome[t][c]) ** 2
            row += f"  {b:11.4f}"
        print(row)
    n = len(probs)
    print(f"{'Mean Brier':18s}" + "".join(f"  {c:>9s}" for c in CATS))
    print(f"{'  model':18s}" + "".join(f"  {model_b[c]/n:9.4f}" for c in CATS))
    print(f"{'  uniform base':18s}" + "".join(f"  {base_b[c]/n:9.4f}" for c in CATS))
    print(f"{'  skill (base-mod)':18s}" +
          "".join(f"  {(base_b[c]-model_b[c])/n:+9.4f}" for c in CATS))
    return {c: model_b[c] / n for c in CATS}


def score_ticks(name, slate, outcome):
    ticks = ([t for t in slate["30"] if outcome[t]["p30"]]
             + [t for t in slate["03"] if outcome[t]["p03"]]
             + [t for t in slate["adv"] if outcome[t]["padv"]])
    print(f"{name}: {len(ticks)}/10 ticks "
          f"({'PASS' if len(ticks) >= 5 else 'FAIL'}): {', '.join(ticks)}")


def main():
    v1 = json.load(open(DATA / "stage3_probs_locked_v1.json"))["probs"]
    v2 = json.load(open(DATA / "stage3_probs.json"))["probs"]
    results = {t: tuple(r) for t, r in
               json.load(open(DATA / "results_stage3.json")).items()}
    assert set(results) == set(v1) == set(v2), "results must cover all 16 teams"

    outcome = {t: {"rec": rec,
                   "p30": rec == (3, 0),
                   "padv": rec in ((3, 1), (3, 2)),
                   "p03": rec == (0, 3)}
               for t, rec in results.items()}

    m1 = brier_table("v1 model (original lock: 5 anchors, 73 series)", v1, outcome)
    m2 = brier_table("v2 model (refresh: 8 anchors, 87 series)", v2, outcome)

    print(f"\n{'Refresh delta (v1-v2, + = refresh helped)':18s}" +
          "".join(f"  {c}: {m1[c]-m2[c]:+.4f}" for c in CATS))

    print()
    score_ticks("v1 slate (superseded, not played)", SLATE_V1, outcome)
    score_ticks("v2 slate (final picks)           ", SLATE_V2, outcome)


if __name__ == "__main__":
    main()
