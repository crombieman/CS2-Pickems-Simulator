"""Retrospective: lock-time inputs, lock vs current engine. READ-ONLY."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import simulate
from optimize import optimize, score_slate
from simulate import run

DATA = ROOT / "data"
ENTERED = {"30": ("Vitality", "Spirit"), "03": ("B8", "Monte"),
           "adv": ("NAVI", "Falcons", "FURIA", "Aurora", "MOUZ", "MongolZ")}

ratings = json.load(open(DATA / "ratings_locked_v3.json"))

# Engine B first: module state is already the current engine (corrected
# SEED, priority table on). Then mutate to the lock-night legacy engine.
configs = [
    ("B: current engine (corrected seeds, priority table)", None),
    ("A: lock-night engine (legacy seeds, greedy R4)", "legacy"),
]
for label, mode in configs:
    if mode == "legacy":
        simulate.SEED = simulate.LEGACY_SEED
        simulate.USE_PRIORITY_TABLE = False
    sims, stats = run(ratings, n_sims=40000, seed=11)
    p5e, eve = score_slate(sims, *[ENTERED[k] for k in ("30", "03", "adv")])
    p5, ev, c30, c03, cadv = optimize(sims, stats)
    print(f"\n=== {label} ===")
    print(f"entered slate: P(>=5) = {p5e:.4f}, E = {eve:.3f}")
    print(f"argmax slate : P(>=5) = {p5:.4f}, E = {ev:.3f}")
    print(f"  3-0: {c30}  0-3: {c03}")
    print(f"  adv: {cadv}")
    same = (set(c30) == set(ENTERED["30"]) and set(c03) == set(ENTERED["03"])
            and set(cadv) == set(ENTERED["adv"]))
    print(f"  argmax == entered slate: {same}", flush=True)
