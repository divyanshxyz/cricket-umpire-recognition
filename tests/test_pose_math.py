"""Pose feature math edge cases (weight-free, pure numpy): the masking
boundary at exactly 0.20, the torso-scale fallback chain including
zero-length fall-through and the unit-scale warning event, hand-computed
geometry values on a synthetic skeleton, and the zero-NaN mandate (invalid
geometry collapses to the 0.0 sentinel — the engineered vector is finite for
fully-masked and partially-masked inputs).
"""

import numpy as np
import pytest

from src.pose_features import (
    engineered_features,
    mask_keypoints,
    scale_warning_events,
    torso_scale,
)


# --- masking boundary (CONTEXT semantics: x/y zeroed, conf preserved) ---


def test_mask_boundary_exactly_at_threshold_stays_visible():
    kpts = np.zeros((17, 3))
    kpts[0] = (10.0, 20.0, 0.20)  # conf exactly AT the 0.20 boundary
    masked = mask_keypoints(kpts, 0.20)
    assert masked[0, 0] == 10.0  # visible: x untouched
    assert masked[0, 1] == 20.0  # visible: y untouched
    assert masked[0, 2] == 0.20


def test_mask_below_threshold_zeroes_xy_keeps_conf():
    kpts = np.zeros((17, 3))
    kpts[0] = (10.0, 20.0, 0.199)  # just below the boundary
    masked = mask_keypoints(kpts, 0.20)
    assert masked[0, 0] == 0.0  # x zeroed
    assert masked[0, 1] == 0.0  # y zeroed
    assert masked[0, 2] == 0.199  # conf PRESERVED — masking stays detectable


# --- torso scale fallback chain (CONTEXT locked) ---


def _torso_kpts(l_sh, r_sh, l_hip, r_hip, conf=0.9):
    kpts = np.zeros((17, 3))
    for idx, xy in ((5, l_sh), (6, r_sh), (11, l_hip), (12, r_hip)):
        kpts[idx, :2] = xy
        kpts[idx, 2] = conf
    return kpts


def test_torso_scale_all_four_joints_visible():
    # mid_shoulder (50,100) vs mid_hip (50,0) -> torso length exactly 100.
    kpts = _torso_kpts((0, 100), (100, 100), (0, 0), (100, 0))
    scale, source = torso_scale(kpts, 0.20)
    assert source == "torso"
    assert scale == pytest.approx(100.0)


def test_torso_scale_hip_below_threshold_falls_to_shoulder_width():
    # L hip conf 0.1 < 0.20 (EITHER hip fails the chain's first stage)
    # -> shoulder-width stage: |L_sh - R_sh| = 100.
    kpts = _torso_kpts((0, 100), (100, 100), (0, 0), (100, 0))
    kpts[11, 2] = 0.1
    scale, source = torso_scale(kpts, 0.20)
    assert source == "shoulder_width"
    assert scale == pytest.approx(100.0)


def test_torso_scale_all_torso_joints_below_threshold_unit_with_warning():
    kpts = _torso_kpts((0, 100), (100, 100), (0, 0), (100, 0), conf=0.1)
    before = len(scale_warning_events())
    scale, source = torso_scale(kpts, 0.20)
    assert (scale, source) == (1.0, "unit")
    assert len(scale_warning_events()) == before + 1  # the recorded warning


def test_torso_scale_zero_length_torso_falls_through_to_shoulder_width():
    # All four visible but mid_shoulder == mid_hip -> torso length 0
    # (<= 1e-6) falls through; shoulders are 100 apart, so stage 2 succeeds.
    kpts = _torso_kpts((0, 0), (100, 0), (25, 0), (75, 0))
    scale, source = torso_scale(kpts, 0.20)
    assert source == "shoulder_width"
    assert scale == pytest.approx(100.0)


def test_torso_scale_zero_length_everything_falls_to_unit_with_warning():
    # All four torso joints coincident: torso length 0 AND shoulder width 0
    # — every scale stage is degenerate (<= 1e-6) and falls through to unit.
    kpts = _torso_kpts((50, 50), (50, 50), (50, 50), (50, 50))
    before = len(scale_warning_events())
    scale, source = torso_scale(kpts, 0.20)
    assert (scale, source) == (1.0, "unit")
    assert len(scale_warning_events()) == before + 1


# --- geometry features (authored [ASSUMED] A1 formulas, hand-computed) ---


def _skeleton(conf=0.9):
    """Hand-computable synthetic skeleton (COCO order); torso scale target
    100 when fed to engineered_features(scale=100.0)."""
    coords = [
        (150, 30),   # 0 nose
        (140, 25),   # 1 L_eye
        (160, 25),   # 2 R_eye
        (135, 30),   # 3 L_ear
        (165, 30),   # 4 R_ear
        (100, 100),  # 5 L_shoulder
        (200, 100),  # 6 R_shoulder
        (100, 150),  # 7 L_elbow (upper arm hangs straight down)
        (200, 75),   # 8 R_elbow (arm half raised)
        (150, 150),  # 9 L_wrist (forearm points right -> perpendicular)
        (200, 50),   # 10 R_wrist (above the shoulder)
        (100, 200),  # 11 L_hip
        (200, 200),  # 12 R_hip
        (100, 250),  # 13 L_knee
        (200, 250),  # 14 R_knee
        (100, 300),  # 15 L_ankle
        (200, 300),  # 16 R_ankle
    ]
    kpts = np.zeros((17, 3))
    for idx, (x, y) in enumerate(coords):
        kpts[idx] = (x, y, conf)
    return kpts


def test_geometry_perpendicular_arm_gives_90_degree_elbow():
    kpts = _skeleton()
    eng = engineered_features(mask_keypoints(kpts, 0.20), 100.0, 0.20)
    # L arm: shoulder-elbow = (0,-50), wrist-elbow = (50,0) -> 90 degrees.
    assert eng[51 + 0] == pytest.approx(90.0)  # elbow_l
    # R arm is perfectly straight (collinear) -> 180 degrees.
    assert eng[51 + 1] == pytest.approx(180.0)  # elbow_r


def test_geometry_elevation_signs():
    kpts = _skeleton()
    eng = engineered_features(mask_keypoints(kpts, 0.20), 100.0, 0.20)
    assert eng[51 + 2] == pytest.approx(-0.5)  # elev_l: wrist BELOW shoulder
    assert eng[51 + 3] == pytest.approx(0.5)   # elev_r: wrist ABOVE shoulder
    assert eng[51 + 4] == pytest.approx(-1.2)  # wristnose_l: wrist below nose
    assert eng[51 + 5] == pytest.approx(-0.2)  # wristnose_r


def test_geometry_span_and_symmetry():
    kpts = _skeleton()
    eng = engineered_features(mask_keypoints(kpts, 0.20), 100.0, 0.20)
    assert eng[51 + 6] == pytest.approx(1.0)  # x spans [100, 200] / scale 100
    assert eng[51 + 7] == pytest.approx(1.0)  # |150 - 50| wrists / scale 100


def test_engineered_layout_17x3_normalized_then_8_geometry():
    kpts = _skeleton()
    eng = engineered_features(mask_keypoints(kpts, 0.20), 100.0, 0.20)
    assert eng.shape == (59,)
    assert eng.dtype == np.float32
    block = eng[:51].reshape(17, 3)
    assert block[0][0] == pytest.approx(1.5)  # nose x / 100
    assert block[0][1] == pytest.approx(0.3)  # nose y / 100
    assert block[0][2] == pytest.approx(0.9)  # conf preserved, unnormalized
    assert block[5][0] == pytest.approx(1.0)  # L shoulder x / 100


# --- zero-NaN mandate (invalid geometry -> 0.0 sentinel, never NaN) ---


def test_masked_wrist_yields_zero_sentinel_geometry():
    kpts = _skeleton()
    kpts[9, 2] = 0.1  # L wrist below the boundary
    eng = engineered_features(mask_keypoints(kpts, 0.20), 100.0, 0.20)
    assert eng[51 + 0] == 0.0  # elbow_l needs the (masked) wrist -> sentinel
    assert eng[51 + 2] == 0.0  # elev_l -> sentinel
    assert eng[51 + 4] == 0.0  # wristnose_l -> sentinel
    assert eng[51 + 7] == 0.0  # symmetry needs BOTH wrists -> sentinel
    assert np.isfinite(eng).all()


def test_engineered_vector_finite_for_fully_masked_input():
    kpts = _skeleton(conf=0.05)  # every keypoint below the boundary
    eng = engineered_features(mask_keypoints(kpts, 0.20), 1.0, 0.20)
    assert np.isfinite(eng).all()
    assert not np.isnan(eng).any()


def test_engineered_vector_finite_for_partially_masked_input():
    kpts = _skeleton()
    for idx in (0, 5, 9, 11, 12):  # nose, L shoulder, L wrist, both hips
        kpts[idx, 2] = 0.1
    eng = engineered_features(mask_keypoints(kpts, 0.20), 100.0, 0.20)
    assert np.isfinite(eng).all()
    assert not np.isnan(eng).any()
