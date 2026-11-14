"""Phase 4 scoreboard FSM: pure cricket-law scoring, ZERO media dependencies.

This module has ZERO video/pose/model imports — pure logic only (SCORE-03,
enforced by a source-level import guard test in tests/test_scoreboard.py).
It consumes ONLY cooled-down gated events: the stabilizer's fire gate
excludes NoAction upstream, so NoAction never reaches apply_event — an
unknown signal here is a fail-closed error naming the observed value, never
a silent no-op (law drift is impossible: LAW_TABLE is the ONLY place signal
effects are defined, and no analyzer/CLI code may re-derive them).
"""

from dataclasses import dataclass

# SCORE-01 law table — the ONLY place signal effects are defined.
# runs/extras/legal_balls/wickets deltas per signal (Laws of Cricket: Wide
# and NoBall are +1 penalty run and the ball is NOT legal; a boundary scores
# 4/6 runs off a legal ball; Out takes a wicket off a legal ball).
LAW_TABLE = {
    "Four": {"runs": 4, "extras": 0, "legal_balls": 1, "wickets": 0},
    "Six": {"runs": 6, "extras": 0, "legal_balls": 1, "wickets": 0},
    "Out": {"runs": 0, "extras": 0, "legal_balls": 1, "wickets": 1},
    "Wide": {"runs": 1, "extras": 1, "legal_balls": 0, "wickets": 0},
    "NoBall": {"runs": 1, "extras": 1, "legal_balls": 0, "wickets": 0},
}


@dataclass(frozen=True)
class ScoreState:
    """Frozen scoreboard state (functional updates via apply_event; Phase 5
    rendering reads these fields directly)."""

    runs: int = 0
    wickets: int = 0
    legal_balls: int = 0
    extras: int = 0

    @property
    def overs(self) -> str:
        # SCORE-02: f"{legal_balls // 6}.{legal_balls % 6}" — NO zero-pad
        # ("05" balls is not cricket notation: 61 legal balls render "10.1",
        # one legal ball renders "0.1", never "00.01"/"0.05").
        return f"{self.legal_balls // 6}.{self.legal_balls % 6}"


def apply_event(state: ScoreState, signal: str) -> tuple:
    """Pure law-table application: (new ScoreState, delta tuple) where
    delta = (this event's runs, this event's wickets, overs_after,
    this event's extras). The INPUT state is never mutated (frozen
    dataclass; a new ScoreState is built from the law row).

    Fail-closed: an unknown signal raises ValueError naming the observed
    signal — the scoreboard consumes only gated signal events (the
    stabilizer's fire gate excludes NoAction upstream), so an unrecognized
    signal reaching this function is a wiring bug, never a silent no-op
    (no-silent-no-op failure, RESEARCH Pattern 6 message shape)."""
    law = LAW_TABLE.get(signal)
    if law is None:
        raise ValueError(
            f"apply_event: unknown signal {signal!r} — scoreboard consumes "
            "only gated signal events (fail-closed, no silent no-op)"
        )
    new = ScoreState(
        runs=state.runs + law["runs"],
        wickets=state.wickets + law["wickets"],
        legal_balls=state.legal_balls + law["legal_balls"],
        extras=state.extras + law["extras"],
    )
    return new, (law["runs"], law["wickets"], new.overs, law["extras"])


@dataclass(frozen=True)
class EventLogEntry:
    """One detected-signal event — the Phase 5 Tab 2 rendering + CSV download
    contract. ``score`` is the (runs_after, wickets_after, overs_str,
    extras_after) tuple produced from apply_event's result for the event;
    to_row() renders it positionally under EVENT_CSV_COLUMNS."""

    timestamp: str
    signal: str
    confidence: float
    score: tuple

    def __post_init__(self):
        if len(self.score) != 4:
            raise ValueError(
                f"EventLogEntry: score must be the 4-tuple (runs_after, "
                f"wickets_after, overs_str, extras_after), got length "
                f"{len(self.score)} — fix the producer (fail-closed, no "
                "silent default)"
            )

    def to_row(self) -> tuple:
        """CSV row under EVENT_CSV_COLUMNS (column count matches the
        module constant exactly)."""
        return (
            self.timestamp,
            self.signal,
            self.confidence,
            self.score[0],
            self.score[1],
            self.score[2],
            self.score[3],
        )


# CSV download contract (Phase 5): exactly these seven columns, in this order.
EVENT_CSV_COLUMNS = ("timestamp", "signal", "confidence", "runs", "wickets", "overs", "extras")
