"""Fit ratings: priors -> Bradley-Terry on match data -> market re-anchor.

Writes data/ratings_fitted.json (Stage 3 teams only) for simulate/optimize.
"""

import json
from pathlib import Path

from model import (PRIORS, STAGE3_TEAMS, apply_market_anchors,
                   fit_bradley_terry, load_matches, win_prob)

DATA = Path(__file__).resolve().parent.parent / "data"


def main():
    matches = load_matches()
    fitted = fit_bradley_terry(matches)
    anchored = apply_market_anchors(fitted)

    print(f"Fit on {len(matches)} weighted series.\n")
    print(f"{'Team':12s} {'prior':>6s} {'BT fit':>7s} {'anchored':>9s}")
    for t in sorted(STAGE3_TEAMS, key=lambda t: -anchored[t]):
        print(f"{t:12s} {PRIORS[t]:6.0f} {fitted[t]:7.0f} {anchored[t]:9.0f}")

    print("\nRound 1 model probabilities:")
    from simulate import ROUND1
    for a, b in ROUND1:
        print(f"  {a} over {b}: {win_prob(anchored, a, b):.3f}")

    out = {t: anchored[t] for t in STAGE3_TEAMS}
    json.dump(out, open(DATA / "ratings_fitted.json", "w"), indent=2)
    print(f"\nWrote {DATA / 'ratings_fitted.json'}")


if __name__ == "__main__":
    main()
