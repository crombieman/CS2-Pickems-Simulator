"""Operational market-"close" selection from the odds archive (W1).

Promotes + hardens postmortem_matches.closing_prob into a reusable, versioned
module. "Close" = the last LIQUID two-sided mid strictly before the match's
scheduled start (the only honest pre-match cut; archive snapshots are 2-hourly
and may otherwise be in-play). Matched by team-pair via fetch_anchors.ALIASES
(no slug join needed). Versioned so a future rule change is explicit and rows
stay re-gradeable.

Why a liquidity floor: a thin book (volume < MIN_VOLUME) should never set the
close — its mid is noise. Why NO `<0.95` exclusion: the pre-start timestamp cut
already guarantees the price is a pre-match forecast, and excluding `>0.95`
would wrongly drop legitimate lopsided-favorite closes.

Only the mid is available from the gamma archive (no bid/ask). Snapshots are
2-hourly, so a close can be up to ~2h stale relative to start; that is recorded,
not hidden. A match with no liquid pre-start row falls back to the earliest
liquid row, FLAGGED (likely in-play, treat with suspicion); the calibration
harness excludes flagged rows from adoption stats. No liquid row at all -> None.
"""

from datetime import datetime

from fetch_anchors import ALIASES, MIN_VOLUME

CLOSE_RULE = "v1:last-mid-before-start+minvol"


def _team(label):
    return ALIASES.get(label.strip().lower())


def _select(archive, a, b, match_start):
    """(row, p_a, flagged) for the close, or (None, None, False).

    Considers only liquid (volume >= MIN_VOLUME) two-sided rows for the queried
    pair. Returns the last such row strictly before match_start (flagged=False);
    if none is pre-start, the earliest liquid row (flagged=True)."""
    start = datetime.fromisoformat(match_start)
    rows = []
    for r in archive:
        outcomes = r.get("outcomes") or []
        if len(outcomes) != 2:
            continue
        if float(r.get("volume") or 0) < MIN_VOLUME:
            continue
        mapped = (_team(outcomes[0]), _team(outcomes[1]))
        if set(mapped) != {a, b}:
            continue
        p_a = r["prices"][0] if mapped[0] == a else r["prices"][1]
        rows.append((datetime.fromisoformat(r["ts"]), p_a, r))
    if not rows:
        return None, None, False
    rows.sort(key=lambda x: x[0])
    pre = [x for x in rows if x[0] < start]
    if pre:
        _, p_a, row = pre[-1]
        return row, p_a, False
    _, p_a, row = rows[0]
    return row, p_a, True


def close_snapshot(archive, a, b, match_start):
    """(P(a wins) at close, flagged) — always a 2-tuple. (None, False) if no
    liquid row exists for the pair. See module docstring / CLOSE_RULE."""
    _, p_a, flagged = _select(archive, a, b, match_start)
    return p_a, flagged


def close_row(archive, a, b, match_start):
    """Full close record with provenance for the calibration log, or None.

    {p_a, flagged, close_rule, ts, slug, volume} — everything W2 needs to log a
    re-gradeable close with its source and rule version."""
    row, p_a, flagged = _select(archive, a, b, match_start)
    if row is None:
        return None
    return {
        "p_a": p_a,
        "flagged": flagged,
        "close_rule": CLOSE_RULE,
        "ts": row["ts"],
        "slug": row.get("slug"),
        "volume": row.get("volume"),
    }
