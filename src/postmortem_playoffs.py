"""Playoff pick'em postmortem: entered slate vs actual bracket + model vs market.

Grades two distinct objects (never conflated, same discipline as the Stage 3
postmortems):
  1. The ENTERED 7-pick slate (4 QF winners + 2 SF winners + champion) against
     the actual bracket — the pick'em outcome.
  2. The lock-time per-match model probs vs the market closing line — read from
     the graded calibration log (kind:"match", event cologne-playoffs), which
     calibration.py reconstructs via the exact lock code path.

ENTERED_PICKS provenance: playoffs.py's recommendation at lock (pure chalk,
identical under all 4 objectives and the full lambda envelope — journal
2026-06-16); Will confirmed 2026-07-01 the slate was entered exactly as
recommended. playoffs.py had no forecast-log path at lock (frozen during lock
week), so this constant is the reconstructed record of what was entered, not a
pre-registered artifact.

LOCK_EXPECTATIONS: the lock-run posterior means reported by playoffs.py
(journal 2026-06-16), for expectation-vs-realization context.

Usage: python src/postmortem_playoffs.py
"""

import json
from pathlib import Path

from calibration import load_latest, summarize

DATA = Path(__file__).resolve().parent.parent / "data"
EVENT = "cologne-playoffs"

ENTERED_PICKS = {
    "qf": ("Spirit", "Vitality", "FURIA", "Aurora"),   # picked QF winners
    "sf": ("Vitality", "FURIA"),                       # picked finalists
    "champion": "Vitality",
}

LOCK_EXPECTATIONS = {"challenges": 0.341, "champion": 0.356,
                     "expected_correct": 3.96}


def actual_bracket(matches):
    """(qf_winners, sf_winners, champion) from the results file."""
    qf = tuple(m["winner"] for m in matches if m["round"] == "QF")
    sf = tuple(m["winner"] for m in matches if m["round"] == "SF")
    (champ,) = [m["winner"] for m in matches if m["round"] == "GF"]
    return qf, sf, champ


def grade_slate(picks, qf_w, sf_w, champ):
    """Per-slot correctness + the objective outcomes (same definitions as
    playoffs.score_pick: challenges = >=2 QF AND >=1 SF AND champion)."""
    qf_correct = len(set(picks["qf"]) & set(qf_w))
    sf_correct = len(set(picks["sf"]) & set(sf_w))
    champ_correct = picks["champion"] == champ
    return {
        "qf_correct": qf_correct, "sf_correct": sf_correct,
        "champ_correct": champ_correct,
        "total_correct": qf_correct + sf_correct + int(champ_correct),
        "challenges": qf_correct >= 2 and sf_correct >= 1 and champ_correct,
        "perfect": qf_correct == 4 and sf_correct == 2 and champ_correct,
    }


def main():
    matches = json.load(open(DATA / "results_matches_playoffs.json"))["matches"]
    qf_w, sf_w, champ = actual_bracket(matches)
    g = grade_slate(ENTERED_PICKS, qf_w, sf_w, champ)

    print("Playoff pick'em postmortem — Cologne Champions stage")
    print(f"  actual: QF {', '.join(qf_w)} | SF {', '.join(sf_w)} | champion {champ}\n")
    print("  Entered slate (chalk, objective-invariant at lock):")
    for team in ENTERED_PICKS["qf"]:
        print(f"    QF   {team:10s} {'HIT' if team in qf_w else 'MISS'}")
    for team in ENTERED_PICKS["sf"]:
        print(f"    SF   {team:10s} {'HIT' if team in sf_w else 'MISS'}")
    c = ENTERED_PICKS["champion"]
    print(f"    CHMP {c:10s} {'HIT' if g['champ_correct'] else 'MISS'}")
    print(f"\n  correct {g['total_correct']}/7 "
          f"(QF {g['qf_correct']}/4, SF {g['sf_correct']}/2, "
          f"champion {'1/1' if g['champ_correct'] else '0/1'})")
    print(f"  challenges objective (>=2 QF & >=1 SF & champ): "
          f"{'PASS' if g['challenges'] else 'FAIL'}")
    le = LOCK_EXPECTATIONS
    print(f"  lock expectation: P(challenges)={le['challenges']:.3f}, "
          f"P(champ)={le['champion']:.3f}, E[correct]={le['expected_correct']:.2f} "
          f"-> realized {g['total_correct']}")

    rows = [r for r in load_latest() if r.get("event") == EVENT
            and r["kind"] == "match"]
    if not rows:
        print(f"\n  (no {EVENT} rows in the graded log yet — run "
              f"`python src/calibration.py --grade-event {EVENT}`)")
        return
    print(f"\n  Model vs market close ({len(rows)} matches, graded log):")
    for r in sorted(rows, key=lambda r: (r["round"] != "QF", r["round"] != "SF")):
        cp = f"{r['close_prob']:.3f}" if r["close_prob"] is not None else "  —  "
        d = (f"{r['delta_brier']:+.4f}" if r["delta_brier"] is not None else "   —   ")
        flag = " [flagged]" if r["close_flagged"] else ""
        print(f"    {r['round']:3s} {r['a']:9s} v {r['b']:9s} -> {r['winner']:9s} "
              f"model {r['model_prob']:.3f} close {cp} dBrier {d}{flag}")
    s = summarize(rows, require_manifest=False)
    if s["n"]:
        print(f"    over {s['n']} unflagged closes: model Brier "
              f"{s['model']['brier']:.4f} vs market {s['market']['brier']:.4f} "
              f"(delta {s['delta_brier']:+.4f}; + = model beat close)")
    print("  E.7 humility: one 7-match bracket, correlated. Evidence, not a verdict.")


if __name__ == "__main__":
    main()
