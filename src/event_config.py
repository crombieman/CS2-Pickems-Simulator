"""Per-event configuration (W5): hard-coded event facts lifted into versioned
per-event JSON so the validation harness (W6) can replay many events.

The Cologne Stage-3 config (data/events/cologne_major.json) holds the EXACT
team order, Round-1 pairings, and seeds the locked fit + probability tables
were generated under, so sourcing them here is byte-identical to the former
in-code literals (model.STAGE3_TEAMS, simulate.ROUND1/SEED). The modules bind
their globals from COLOGNE; a future harness rebinds them per replayed event
(the same global-rebind pattern test_invariants already uses for LEGACY_SEED).

See docs/plans/2026-06-17-engine-correctness-implementation.md, W5.
"""

import json
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
EVENTS = DATA / "events"

_SCORING_KEYS = ("exact_3_0", "exact_0_3", "advance", "pass_threshold",
                 "slate_size")


@dataclass
class EventConfig:
    """One event's facts. `teams` (list), `round1` (list of (a, b) tuples) and
    `seeds` (dict) keep the exact types/orders of the former in-code literals
    so downstream fits and simulations are byte-identical."""

    event_id: str
    name: str
    teams: list
    round1: list
    seeds: dict
    scoring: dict
    playoffs: dict
    format: dict
    ruleset_version: str
    optimizer_objective: str
    lock_timestamp: object  # ISO-8601 str or None

    @classmethod
    def from_dict(cls, d: dict) -> "EventConfig":
        teams = list(d["teams"])
        round1 = [tuple(m) for m in d["round1"]]
        seeds = dict(d["seeds"])
        _validate(teams, round1, seeds, d.get("scoring", {}))
        return cls(
            event_id=d["event_id"],
            name=d["name"],
            teams=teams,
            round1=round1,
            seeds=seeds,
            scoring=dict(d["scoring"]),
            playoffs=dict(d["playoffs"]),
            format=dict(d.get("format", {})),
            ruleset_version=d.get("ruleset_version", "unknown"),
            optimizer_objective=d.get("optimizer_objective", "p_ge_threshold"),
            lock_timestamp=d.get("lock_timestamp"),
        )


def _validate(teams, round1, seeds, scoring):
    """Fail loud on a malformed event (the harness must never silently replay a
    broken config). Mirrors simulate.make_state's boundary-validation ethos."""
    n = len(teams)
    if n < 2:
        raise ValueError(f"event needs >=2 teams, got {n}")
    if len(set(teams)) != n:
        raise ValueError(f"team names must be unique: {teams}")
    known = set(teams)

    # round1 must partition the teams: every team in exactly one pairing.
    flat = [t for m in round1 for t in m]
    for m in round1:
        if len(m) != 2:
            raise ValueError(f"round1 pairing needs exactly 2 teams: {m!r}")
    unknown = [t for t in flat if t not in known]
    if unknown:
        raise ValueError(f"round1 references unknown team(s): {unknown}")
    if sorted(flat) != sorted(teams):
        raise ValueError(
            f"round1 must cover every team exactly once "
            f"({len(flat)} slots for {n} teams; check for missing/duplicate)")

    # seeds must be a bijection teams -> 1..n.
    if set(seeds) != known:
        raise ValueError(
            f"seeds must name exactly the {n} teams; "
            f"missing={known - set(seeds)} extra={set(seeds) - known}")
    if sorted(seeds.values()) != list(range(1, n + 1)):
        raise ValueError(
            f"seeds must be a bijection onto 1..{n}, got {sorted(seeds.values())}")

    missing = [k for k in _SCORING_KEYS if k not in scoring]
    if missing:
        raise ValueError(f"scoring missing required key(s): {missing}")


def load_event(path) -> EventConfig:
    with open(path) as f:
        return EventConfig.from_dict(json.load(f))


COLOGNE = load_event(EVENTS / "cologne_major.json")
