"""Sequence-feature contract tests (pure, weight-free): velocity math on
synthetic deltas, the frozen (T, 118) assembly layout (columns [0:51] raw,
[51:110] engineered, [110:118] velocity, first frame zeros), and the
whole-sequence typed-failure semantics (a failure row propagates as a typed
whole-sequence failure — never a zero-feature row).
"""

import numpy as np
import pytest

from src.pose_features import (
    ExtractionFailure,
    assemble_sequence,
    failure_events,
    velocity_features,
)


def test_velocity_identical_coords_give_zeros():
    xy = np.zeros((17, 2))
    assert (velocity_features(xy, xy, 5.0) == 0.0).all()


def test_velocity_known_delta_divided_by_scale():
    prev = np.zeros((17, 2))
    cur = np.zeros((17, 2))
    prev[7] = (10.0, 10.0)
    cur[7] = (20.0, 15.0)
    vel = velocity_features(prev, cur, 5.0)
    # joint 7 (L elbow) is first in VELOCITY_JOINTS: dx = 10/5, dy = 5/5.
    assert vel[0] == pytest.approx(2.0)
    assert vel[1] == pytest.approx(1.0)
    assert (vel[2:] == 0.0).all()  # the other velocity joints did not move
    assert vel.dtype == np.float32
    assert vel.shape == (8,)


def _row(raw_fill, eng_fill, scale, xy_value):
    raw = np.full(51, raw_fill, dtype=np.float32)
    eng = np.full(59, eng_fill, dtype=np.float32)
    xy = np.full((17, 2), xy_value, dtype=np.float64)
    return (raw, eng, scale, xy)


def test_assemble_sequence_shape_and_frozen_layout():
    rows = [
        _row(0.1, 0.2, 2.0, 0.0),
        _row(0.3, 0.4, 2.0, 2.0),
        _row(0.5, 0.6, 2.0, 4.0),
    ]
    seq, failure = assemble_sequence(rows)
    assert failure is None
    assert seq.shape == (3, 118)
    assert seq.dtype == np.float32
    # columns [0:51] raw, [51:110] engineered — the verbatim inputs
    assert (seq[0, 0:51] == 0.1).all()
    assert (seq[2, 0:51] == 0.5).all()
    assert (seq[1, 51:110] == 0.4).all()
    # velocity block: first frame zeros; frames 1-2 get (2-0)/2 = 1.0 each
    assert (seq[0, 110:118] == 0.0).all()
    assert (seq[1, 110:118] == 1.0).all()
    assert (seq[2, 110:118] == 1.0).all()


def test_assemble_sequence_failure_row_propagates_typed_and_counts():
    rows = [
        _row(0.1, 0.2, 2.0, 0.0),
        (None, ExtractionFailure("no_person", "frame 1 had no person")),
        _row(0.5, 0.6, 2.0, 4.0),
    ]
    before = len(failure_events())
    seq, failure = assemble_sequence(rows)
    assert seq is None  # the sequence fails as a whole — never a zero row
    assert failure.reason == "no_person_in_sequence"
    assert "frame 1" in failure.detail  # the failing frame index is named
    assert len(failure_events()) == before + 1  # recorded, counted (ASVS V5)


def test_assemble_sequence_first_frame_failure_also_propagates():
    rows = [(None, ExtractionFailure("decode_error", "bad image bytes"))]
    seq, failure = assemble_sequence(rows)
    assert seq is None
    assert failure.reason == "no_person_in_sequence"
    assert "frame 0" in failure.detail
