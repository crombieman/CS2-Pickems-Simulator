"""Post-stage calibration grading: Brier scores vs locked probabilities.

Usage (after the stage ends):
  1. Fill data/results_stage3.json with actual final records:
       {"Vitality": [3, 0], "FUT": [1, 3], ...}   (all 16 teams)
  2. python src/postmortem.py

Grades ALL locked probability tables on three binary events per team —
exactly 3-0, advance (3-1/3-2), exactly 0-3 — against a uniform baseline
(p30=2/16, padv=6/16, p03=2/16). Lower Brier is better.

  v1: stage3_probs_locked_v1.json (06-10 afternoon; 5 anchors, 73 series).
  v2: stage3_probs_locked_v2.json (06-10 evening; 8 Polymarket anchors,
      87 series, full anchor propagation).
  v3: stage3_probs.json (06-10 night; v2 data + pair overrides and
      lambda=0.5 partial propagation). The final picks.

The cross-version skill comparison answers the questions that matter:
did the market/data refresh (v1->v2) and the propagation model change
(v2->v3) each improve calibration? Also scores each slate's ticks. Per
the README: the Brier table is the postmortem that matters, not whether
a slate passed.
"""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

MODELS = [
    ("v1 (5 anchors, 73 series)", "stage3_probs_locked_v1.json"),
    ("v2 (8 anchors, 87 series, full propagation)", "stage3_probs_locked_v2.json"),
    ("v3 (pair overrides, lambda=0.5) [final]", "stage3_probs.json"),
]

SLATES = [
    # v1 original lock: manual tie-break, Falcons in 3-0 over FURIA argmax.
    ("v1 slate (superseded)", {
        "30": ["Vitality", "Falcons"], "03": ["9z", "B8"],
        "adv": ["Spirit", "NAVI", "FURIA", "MOUZ", "PARIVISION", "Aurora"]}),
    ("v2 slate (superseded)", {
        "30": ["Vitality", "Spirit"], "03": ["B8", "Monte"],
        "adv": ["NAVI", "Falcons", "FURIA", "Aurora", "MOUZ", "G2"]}),
    # v3 final: pipeline argmax, stable across seeds 7/11/42/123.
    ("v3 slate (FINAL PICKS)", {
        "30": ["Vitality", "Spirit"], "03": ["B8", "Monte"],
        "adv": ["NAVI", "Falcons", "FURIA", "Aurora", "MOUZ", "MongolZ"]}),
]

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
    tables = [(name, json.load(open(DATA / fn))["probs"]) for name, fn in MODELS]
    results = {t: tuple(r) for t, r in
               json.load(open(DATA / "results_stage3.json")).items()}
    for name, probs in tables:
        assert set(results) == set(probs), f"results must cover all 16 teams ({name})"

    outcome = {t: {"rec": rec,
                   "p30": rec == (3, 0),
                   "padv": rec in ((3, 1), (3, 2)),
                   "p03": rec == (0, 3)}
               for t, rec in results.items()}

    means = [(name, brier_table(name, probs, outcome)) for name, probs in tables]

    print("\nStep deltas (+ = the change helped):")
    for (n_prev, m_prev), (n_next, m_next) in zip(means, means[1:]):
        print(f"  {n_prev.split(' ')[0]} -> {n_next.split(' ')[0]}:" +
              "".join(f"  {c}: {m_prev[c]-m_next[c]:+.4f}" for c in CATS))

    print()
    for name, slate in SLATES:
        score_ticks(f"{name:28s}", slate, outcome)


if __name__ == "__main__":
    main()
