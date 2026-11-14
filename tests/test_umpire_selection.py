"""Umpire-selection edge cases (weight-free, pure numpy): the locked
composite score (bbox area x horizontal centrality, exact float math), the
exact-equality tie-break by detection confidence (order-independent), the
typed no-person failure on empty boxes, and the no-size-filter decision
(a tiny detection that is the only person still wins).
"""

import numpy as np
import pytest

from src.pose_features import select_umpire


def test_larger_centered_box_beats_smaller_off_center_one():
    # 100x100 image. A: centered 50x50 -> 0.25 area x centrality 1 = 0.25.
    # B: 40x40 at top-left -> 0.16 x centrality 0.4 = 0.064.
    boxes = np.array([[25.0, 25.0, 75.0, 75.0], [0.0, 0.0, 40.0, 40.0]])
    confs = np.array([0.5, 0.9])
    best, failure = select_umpire(boxes, confs, 100, 100)
    assert failure is None
    assert best == 0  # the centered box wins despite lower confidence


def test_equal_areas_resolve_toward_horizontal_center():
    # Both boxes cover 0.25 of the image; the centered one scores
    # 0.25 x 1.0 = 0.25 vs the corner box's 0.25 x 0.5 = 0.125.
    boxes = np.array([[25.0, 25.0, 75.0, 75.0], [0.0, 0.0, 50.0, 50.0]])
    confs = np.array([0.3, 0.9])
    best, failure = select_umpire(boxes, confs, 100, 100)
    assert failure is None
    assert best == 0


def test_exact_tie_resolves_by_higher_conf_order_independently():
    # Exact float tie: b1 (centered 50x50) scores 0.25 x 1.0 = 0.25;
    # b2 (50x100 at the left edge) scores 0.5 x 0.5 = 0.25.
    b1 = (25.0, 25.0, 75.0, 75.0)
    b2 = (0.0, 0.0, 50.0, 100.0)
    boxes_a = np.array([b1, b2])
    boxes_b = np.array([b2, b1])
    confs_a = np.array([0.9, 0.3])
    confs_b = np.array([0.3, 0.9])
    best_a, failure_a = select_umpire(boxes_a, confs_a, 100, 100)
    best_b, failure_b = select_umpire(boxes_b, confs_b, 100, 100)
    assert failure_a is None and failure_b is None
    # The tie always resolves to the SAME box (higher confidence b1),
    # regardless of detection order — the index differs, the box does not.
    assert best_a == 0 and best_b == 1
    assert boxes_a[best_a][0] == boxes_b[best_b][0] == 25.0  # b1's x0
    assert boxes_a[best_a][2] == boxes_b[best_b][2] == 75.0  # b1's x1


def test_full_tie_resolves_lexicographically_order_independently():
    # WR-01 regression: when composite scores AND detection confidences
    # BOTH tie exactly, the old argmax-over-ties silently picked the FIRST
    # tied index — permuting detection order selected a different umpire.
    # The tertiary tie-break is lexicographic by box coords (x0, y0, x1,
    # y1): the SAME box wins under any permutation (only the index follows
    # the permutation). b_left (cx=25) and b_right (cx=75) on a 100x100
    # image have equal area frac (0.09) and equal centrality (0.5) — an
    # exact float score tie — and equal confidences.
    b_left = (10.0, 20.0, 40.0, 50.0)
    b_right = (60.0, 20.0, 90.0, 50.0)
    boxes_a = np.array([b_right, b_left])
    boxes_b = np.array([b_left, b_right])
    confs = np.array([0.7, 0.7])
    best_a, failure_a = select_umpire(boxes_a, confs, 100, 100)
    best_b, failure_b = select_umpire(boxes_b, confs, 100, 100)
    assert failure_a is None and failure_b is None
    assert boxes_a[best_a][0] == boxes_b[best_b][0] == 10.0  # same box wins
    assert best_a == 1 and best_b == 0  # index tracks the permutation


def test_higher_conf_still_beats_lexicographic_order():
    # The tertiary tie-break only fires on a full score AND conf tie: with
    # distinct confidences the higher-confidence box still wins even when
    # it is lexicographically larger.
    b_left = (10.0, 20.0, 40.0, 50.0)
    b_right = (60.0, 20.0, 90.0, 50.0)
    boxes = np.array([b_left, b_right])
    best, failure = select_umpire(boxes, np.array([0.2, 0.9]), 100, 100)
    assert failure is None
    assert best == 1  # higher confidence b_right, not the lex-smaller b_left


def test_empty_boxes_typed_no_person():
    best, failure = select_umpire(np.zeros((0, 4)), np.zeros((0,)), 100, 100)
    assert best is None
    assert failure is not None
    assert failure.reason == "no_person"


def test_no_size_filter_tiny_only_detection_still_wins():
    # CONTEXT locked decision: no extra person-size filter in v1 — a 2x2 px
    # detection that is the only person is still selected (the composite
    # handles size; no hidden filter exists).
    boxes = np.array([[49.0, 49.0, 51.0, 51.0]])
    best, failure = select_umpire(boxes, np.array([0.6]), 100, 100)
    assert failure is None
    assert best == 0
