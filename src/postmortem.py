"""Post-stage calibration grading: Brier scores vs locked probabilities.

Usage (after the stage ends):
  1. Fill data/results_stage3.json with actual final records:
       {"Vitality": [3, 0], "FUT": [1, 3], ...}   (all 16 teams)
  2. python src/postmortem.py

Grades data/stage3_probs.json (locked 2026-06-10) on three binary events
per team — exactly 3-0, advance (3-1/3-2), exactly 0-3 — against a
uniform baseline (p30=2/16, padv=6/16, p03=2/16). Lower Brier is better.
Also scores the locked slate's ticks and pass/fail. Per the README: this
table is the postmortem that matters, not whether the slate passed.
"""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# Locked slate (README, 2026-06-10). Manual tie-break vs pipeline argmax:
# Falcons in 3-0 over FURIA (higher E[ticks], Delta p5 ~ 0.002).
SLATE_30 = ["Vitality", "Falcons"]
SLATE_03 = ["9z", "B8"]
SLATE_ADV = ["Spirit", "NAVI", "FURIA", "MOUZ", "PARIVISION", "Aurora"]

BASELINE = {"p30": 2 / 16, "padv": 6 / 16, "p03": 2 / 16}


def main():
    probs = json.load(open(DATA / "stage3_probs.json"))["probs"]
    results = {t: tuple(r) for t, r in
               json.load(open(DATA / "results_stage3.json")).items()}
    assert set(results) == set(probs), "results must cover all 16 teams"

    outcome = {t: {"p30": rec == (3, 0),
                   "padv": rec in ((3, 1), (3, 2)),
                   "p03": rec == (0, 3)}
               for t, rec in results.items()}

    cats = ("p30", "padv", "p03")
    print(f"{'Team':12s} {'rec':>5s}" +
          "".join(f"  {c+' brier':>11s}" for c in cats))
    model_b = {c: 0.0 for c in cats}
    base_b = {c: 0.0 for c in cats}
    for t in sorted(probs, key=lambda t: -probs[t]["pany"]):
        row = f"{t:12s} {results[t][0]}-{results[t][1]:1d}"
        for c in cats:
            b = (probs[t][c] - outcome[t][c]) ** 2
            model_b[c] += b
            base_b[c] += (BASELINE[c] - outcome[t][c]) ** 2
            row += f"  {b:11.4f}"
        print(row)

    n = len(probs)
    print(f"\n{'Mean Brier':18s}" + "".join(f"  {c:>9s}" for c in cats))
    print(f"{'  model':18s}" + "".join(f"  {model_b[c]/n:9.4f}" for c in cats))
    print(f"{'  uniform base':18s}" + "".join(f"  {base_b[c]/n:9.4f}" for c in cats))
    print(f"{'  skill (base-mod)':18s}" +
          "".join(f"  {(base_b[c]-model_b[c])/n:+9.4f}" for c in cats))

    ticks = ([t for t in SLATE_30 if outcome[t]["p30"]]
             + [t for t in SLATE_03 if outcome[t]["p03"]]
             + [t for t in SLATE_ADV if outcome[t]["padv"]])
    print(f"\nSlate: {len(ticks)}/10 ticks "
          f"({'PASS' if len(ticks) >= 5 else 'FAIL'}): {', '.join(ticks)}")


if __name__ == "__main__":
    main()
