"""Scoreboard FSM pins (SCORE-01..03) — pure, weight-free, table-driven.

Pinned contracts, one test each:
1. Law-table coverage (SCORE-01): every transition comes from LAW_TABLE —
   Four/Six/Out are legal balls (+4/+6/+1 wicket); Wide/NoBall add 1 run +
   1 extra WITHOUT advancing legal balls; the 9-row LAW_CASES table is the
   RESEARCH-pinned shape (Out*3 also pins no innings-end termination —
   the locked CONTEXT boundary deferring it to Phase 5 display logic).
2. Overs format (SCORE-02): f"{legal_balls // 6}.{legal_balls % 6}" —
   no zero-pad ("0.1", "10.1"; never "10.01"/"010.1").
3. Purity + frozen dataclass + fail-closed unknown signal (SCORE-03):
   apply_event never mutates its input, is deterministic, and raises
   ValueError naming any signal outside the gated five.
4. Source-level import guard: src/scoreboard.py must import NOTHING beyond
   stdlib dataclasses/typing — zero video/pose/model dependencies.
5. CSV contract (Phase 5): EVENT_CSV_COLUMNS is the 7-column download
   contract; EventLogEntry.to_row() renders the score tuple positionally.
"""

import ast
import dataclasses
from pathlib import Path

import pytest

from src.scoreboard import (
    EVENT_CSV_COLUMNS,
    EventLogEntry,
    LAW_TABLE,
    UNTRAINED_SMOKE_MARKER,
    ScoreState,
    apply_event,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

LAW_CASES = [
    # (event_sequence, expected runs, wickets, overs_str, extras)
    ([], 0, 0, "0.0", 0),
    (["Four"], 4, 0, "0.1", 0),
    (["Six"], 6, 0, "0.1", 0),
    (["Out"], 0, 1, "0.1", 0),
    (["Wide"], 1, 0, "0.0", 1),  # ball NOT advanced
    (["NoBall"], 1, 0, "0.0", 1),  # ball NOT advanced
    (["Wide", "NoBall", "Four"], 6, 0, "0.1", 2),
    (["Four"] * 7, 28, 0, "1.1", 0),  # overs rollover: 7 legal balls
    (["Out"] * 3, 0, 3, "0.3", 0),  # no innings termination in v1
]


@pytest.mark.parametrize("seq,runs,wk,overs,extras", LAW_CASES)
def test_law_table_drives_every_transition(seq, runs, wk, overs, extras):
    st = ScoreState()
    for sig in seq:
        st, _ = apply_event(st, sig)
    assert (st.runs, st.wickets, st.overs, st.extras) == (runs, wk, overs, extras)


def test_law_table_covers_exactly_the_five_gated_signals():
    # The scoreboard consumes only cooled-down gated events: exactly these
    # five signals have law rows; everything else (NoAction included) fails
    # closed.
    assert set(LAW_TABLE) == {"Four", "Six", "Out", "Wide", "NoBall"}


def test_overs_format_never_zero_pads():
    # SCORE-02 pin: one legal ball renders "0.1"; 61 legal balls render
    # "10.1" — never a zero-padded string ("05" balls is not cricket
    # notation).
    st = ScoreState()
    st, _ = apply_event(st, "Four")
    assert st.overs == "0.1"
    for _ in range(60):
        st, _ = apply_event(st, "Four")
    assert st.legal_balls == 61
    assert st.overs == "10.1"
    assert st.overs not in ("10.01", "010.1")


def test_unknown_signal_fails_closed_naming_it():
    # NoAction never reaches the scoreboard (the gate excludes it upstream);
    # any unknown signal is a wiring bug and must raise naming the value.
    for bad in ("NoAction", "Bouncer"):
        with pytest.raises(ValueError, match=bad):
            apply_event(ScoreState(), bad)


def test_apply_event_is_pure_and_state_is_frozen():
    st = ScoreState()
    r1 = apply_event(st, "Wide")
    r2 = apply_event(st, "Wide")  # same (state, signal) -> equal results
    assert r1 == r2
    # the INPUT state is never mutated (a new ScoreState is built)
    assert (st.runs, st.wickets, st.legal_balls, st.extras) == (0, 0, 0, 0)
    try:
        st.runs = 5
        raise AssertionError("ScoreState must be a frozen dataclass")
    except dataclasses.FrozenInstanceError:
        pass


def test_scoreboard_import_guard_pure_module():
    # SCORE-03 truth, enforced at the source level: walk the AST of
    # src/scoreboard.py and allow ONLY stdlib dataclasses/typing — no cv2,
    # torch, ultralytics, joblib, sklearn, pandas, numpy, or any src/ module.
    tree = ast.parse(
        (REPO_ROOT / "src" / "scoreboard.py").read_text(encoding="utf-8")
    )
    forbidden = {
        "cv2",
        "torch",
        "ultralytics",
        "joblib",
        "sklearn",
        "pandas",
        "numpy",
        "src.pose_features",
        "src.models",
        "src.inference",
    }
    allowed = {"dataclasses", "typing"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert alias.name not in forbidden and root not in forbidden, (
                    f"src/scoreboard.py imports a forbidden module: {alias.name}"
                )
                assert root in allowed, (
                    f"src/scoreboard.py has a non-stdlib import: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                root = node.module.split(".")[0]
                assert node.module not in forbidden and root not in forbidden, (
                    f"src/scoreboard.py imports a forbidden module: {node.module}"
                )
                assert root in allowed, (
                    f"src/scoreboard.py has a non-stdlib import: {node.module}"
                )


def test_csv_contract_columns_and_positional_score():
    # Phase 5 Tab 2 + CSV download contract: 7 columns, score tuple fields
    # land positionally (runs, wickets, overs, extras).
    assert len(EVENT_CSV_COLUMNS) == 7
    entry = EventLogEntry("01:23", "Six", 0.9134, (6, 0, "0.2", 0))
    row = entry.to_row()
    assert len(row) == len(EVENT_CSV_COLUMNS)
    assert row == ("01:23", "Six", 0.9134, 6, 0, "0.2", 0)


def test_event_log_entry_rejects_wrong_score_length():
    # The score tuple is exactly (runs_after, wickets_after, overs_str,
    # extras_after) — a wrong-length tuple would corrupt the CSV contract.
    with pytest.raises(ValueError, match="4-tuple"):
        EventLogEntry("00:01", "Four", 0.9, (1, 2, 3))


def test_untrained_smoke_marker_field_contract():
    # T-04-07 transparency record (Plan 04-03): EventLogEntry gains the
    # smoke_marker field defaulting to "" WITHOUT disturbing the 7-column
    # CSV contract — positional construction still works, to_row() stays
    # exactly 7 fields under EVENT_CSV_COLUMNS (the marker is a
    # display/transparency field, NEVER a CSV column), and the module
    # constant equals the pinned UNTRAINED string. The source-level AST
    # import guard (test_scoreboard_import_guard_pure_module) pins that the
    # new constant adds no imports.
    entry = EventLogEntry("00:12", "Four", 0.8123, (4, 0, "0.1", 0))
    assert entry.smoke_marker == ""
    row = entry.to_row()
    assert len(row) == 7
    assert row == ("00:12", "Four", 0.8123, 4, 0, "0.1", 0)
    assert UNTRAINED_SMOKE_MARKER == "UNTRAINED — smoke only"
    # An explicitly stamped entry still renders a 7-field row.
    marked = EventLogEntry(
        "00:14",
        "Six",
        0.9,
        (10, 0, "0.2", 0),
        smoke_marker=UNTRAINED_SMOKE_MARKER,
    )
    assert marked.smoke_marker == "UNTRAINED — smoke only"
    assert len(marked.to_row()) == len(EVENT_CSV_COLUMNS)
    assert marked.to_row() == ("00:14", "Six", 0.9, 10, 0, "0.2", 0)
