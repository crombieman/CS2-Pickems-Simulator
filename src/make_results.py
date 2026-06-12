"""Generate data/results_stage3.json from the completed live state.

Run after Stage 3 ends (June 15). Reuses playoffs.stage3_final_state,
which validates completeness (exact Swiss record multiset) and refuses
mid-stage states — so postmortem.py can never accidentally grade a
partial stage. Records in live_state.json are already two-source
verified at entry; this just reshapes them.

  python src/make_results.py        # -> data/results_stage3.json
  python src/postmortem.py          # then grade v1/v2/v3
"""

import json
from pathlib import Path

from playoffs import stage3_final_state

DATA = Path(__file__).resolve().parent.parent / "data"


def make_results(completed):
    """{team: [wins, losses]} from a COMPLETE stage's match list."""
    records, _ = stage3_final_state(completed)
    return {t: list(rec) for t, rec in sorted(records.items())}


def main():
    live = json.load(open(DATA / "live_state.json"))
    completed = [tuple(m) for m in live.get("completed", [])]
    results = make_results(completed)
    out = DATA / "results_stage3.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out} ({len(results)} teams).")


if __name__ == "__main__":
    main()
