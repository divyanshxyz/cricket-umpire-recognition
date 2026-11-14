"""TRAIN-02 flip/jitter augmentation contract tests (weight-free, synthetic
only).

Pins: (1) mirror_flip permutation EXACTNESS against the real
src.pose_features math — recomputing raw51/eng59 from keypoint-mirrored
synthetic input and comparing to mirror_flip output at allclose 1e-9 on
fully-visible AND masked keypoints (the research probe-3 logic, now a
permanent regression test); (2) the swap-only property — involution, pure
permutation (no value negated or transformed), invariant nose triple and
eng span/symmetry columns, PERM_RAW/PERM_ENG bijections; (3) the
empty-slice raise (TRAIN-02/empty); (4) the width guard naming the
observed width; (5) determinism + originals-then-mirrors row ordering +
bounded jitter (sigma 0.015). No torch, no weights, no frozen npz.
"""

import numpy as np
import pytest

from src.models.data import (
    KP_SWAP,
    PERM_ENG,
    PERM_RAW,
    augment,
    mirror_flip,
)
from src.pose_features import (
    engineered_features,
    mask_keypoints,
    raw_features,
    torso_scale,
)

MASK_CONF = 0.20


def _synthetic_keypoints(visible: bool = True, seed: int = 0) -> np.ndarray:
    """Synthetic (17, 3) COCO keypoints: x/y in a plausible image range,
    conf > 0.2 everywhere for the fully-visible case; several conf < 0.2
    for the masked case."""
    rng = np.random.default_rng(seed)
    kpts = np.empty((17, 3), dtype=np.float64)
    kpts[:, 0] = rng.uniform(100.0, 1000.0, 17)
    kpts[:, 1] = rng.uniform(100.0, 1000.0, 17)
    kpts[:, 2] = rng.uniform(0.5, 1.0, 17)
    if not visible:
        low = rng.choice(17, size=5, replace=False)
        kpts[low, 2] = rng.uniform(0.0, 0.15, 5)  # strictly below MASK_CONF
    return kpts


def _features_from_keypoints(kpts: np.ndarray) -> tuple:
    """(raw51, eng59) via the REAL src.pose_features math (the frozen Phase
    2 seam this permutation must match exactly)."""
    masked = mask_keypoints(kpts, MASK_CONF)
    raw = raw_features(kpts)  # DETECTED values, unmasked (raw_features contract)
    scale, _source = torso_scale(masked, MASK_CONF)
    eng = engineered_features(masked, scale, MASK_CONF)
    return raw, eng


def _mirrored_keypoints(kpts: np.ndarray) -> np.ndarray:
    """The ACTUALLY-mirrored keypoint set: each COCO L/R pair's coordinates
    AND confidence swap wholesale via KP_SWAP indexing (conf belongs to the
    keypoint, not the L/R slot — RESEARCH Pattern 1)."""
    return kpts[KP_SWAP]


def test_mirror_flip_matches_pose_features_visible():
    # Fully-visible keypoints: mirror_flip(raw51/eng59) == features
    # recomputed from the actually-mirrored keypoints, allclose 1e-9.
    kpts = _synthetic_keypoints(visible=True, seed=1)
    raw, eng = _features_from_keypoints(kpts)
    raw_m, eng_m = _features_from_keypoints(_mirrored_keypoints(kpts))
    assert np.allclose(mirror_flip(raw), raw_m, atol=1e-9)
    assert np.allclose(mirror_flip(eng), eng_m, atol=1e-9)


def test_mirror_flip_matches_pose_features_masked():
    # Masked case: conf < 0.2 keypoints are zeroed by mask_keypoints; the
    # swap still reproduces the recomputed math EXACTLY because conf
    # travels with its keypoint (masked entries stay 0.0, consistently on
    # both sides).
    kpts = _synthetic_keypoints(visible=False, seed=2)
    raw, eng = _features_from_keypoints(kpts)
    raw_m, eng_m = _features_from_keypoints(_mirrored_keypoints(kpts))
    assert np.allclose(mirror_flip(raw), raw_m, atol=1e-9)
    assert np.allclose(mirror_flip(eng), eng_m, atol=1e-9)


def test_mirror_flip_is_sign_preserving_involution():
    # Swap-only property (TRAIN-02/adjacency lift): flip is an involution
    # and a PURE permutation — every output value equals some input value
    # (no value negated or transformed), for both raw and eng widths.
    x = np.arange(51, dtype=np.float64)  # fully asymmetric vector
    flipped = mirror_flip(x)
    assert np.array_equal(mirror_flip(flipped), x)
    assert sorted(flipped.tolist()) == sorted(x.tolist())
    eng = np.arange(59, dtype=np.float64)
    flipped_eng = mirror_flip(eng)
    assert np.array_equal(mirror_flip(flipped_eng), eng)
    assert sorted(flipped_eng.tolist()) == sorted(eng.tolist())
    # Same-row multiset property on a 2-D matrix (per row, not per matrix).
    mat = np.arange(4 * 51, dtype=np.float64).reshape(4, 51)
    mat_flipped = mirror_flip(mat)
    for row_in, row_out in zip(mat, mat_flipped):
        assert sorted(row_out.tolist()) == sorted(row_in.tolist())


def test_mirror_flip_invariant_columns_and_perm_bijections():
    # Nose triple (columns 0-2) is invariant; eng span (57) and symmetry
    # (58) are invariant; PERM_RAW/PERM_ENG are bijections of 0..50/0..58.
    x = np.arange(51, dtype=np.float64)
    flipped = mirror_flip(x)
    assert np.array_equal(flipped[0:3], x[0:3])
    eng = np.arange(59, dtype=np.float64)
    flipped_eng = mirror_flip(eng)
    assert flipped_eng[57] == eng[57] and flipped_eng[58] == eng[58]
    assert np.array_equal(np.sort(PERM_RAW), np.arange(51))
    assert np.array_equal(np.sort(PERM_ENG), np.arange(59))
    # The geometry permutation tail swaps exactly the 6 L/R columns.
    assert PERM_ENG[51:].tolist() == [52, 51, 54, 53, 56, 55, 57, 58]


def test_augment_rejects_zero_row_slice():
    # TRAIN-02/empty lift: a zero-row train slice raises ValueError naming
    # emptiness — never a silent no-op.
    with pytest.raises(ValueError, match="zero-row"):
        augment(
            np.zeros((0, 51), dtype=np.float32),
            np.array([], dtype=np.int64),
            seed=42,
        )


def test_augment_rejects_unknown_width():
    # Width guard: a (N, 50) input has no flip permutation — ValueError
    # naming the observed width (inferred via mirror_flip).
    with pytest.raises(ValueError, match="50"):
        augment(
            np.zeros((4, 50), dtype=np.float32),
            np.zeros(4, dtype=np.int64),
            seed=42,
        )


def test_augment_rejects_misaligned_labels():
    # Defensive alignment guard: labels must zip 1:1 with rows.
    with pytest.raises(ValueError, match="misaligned"):
        augment(np.zeros((4, 51), dtype=np.float32), np.zeros(3, dtype=np.int64), seed=42)


def test_augment_deterministic_originals_then_mirrors():
    # TRAIN-02/ordering lift: identical seeds reproduce identical outputs;
    # on a (4, 51) input the output is 8 rows — rows 0-3 = originals +
    # jitter, rows 4-7 = mirrored rows + jitter (originals then mirrors in
    # row order); the jitter perturbation is bounded well under 0.1
    # (sigma 0.015, model-input space).
    X = np.arange(4 * 51, dtype=np.float32).reshape(4, 51)
    y = np.array([0, 1, 2, 3], dtype=np.int64)
    Xa1, ya1 = augment(X, y, seed=7)
    Xa2, ya2 = augment(X, y, seed=7)
    assert np.array_equal(Xa1, Xa2)
    assert np.array_equal(ya1, ya2)
    assert Xa1.shape == (8, 51)
    assert ya1.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert np.abs(Xa1[:4] - X).max() < 0.1          # originals + jitter
    assert np.abs(Xa1[4:] - mirror_flip(X)).max() < 0.1  # mirrors + jitter
    # A flip actually happened (mirrored rows differ from the originals).
    assert not np.array_equal(Xa1[4:], Xa1[:4])
    # The engineered width infers PERM_ENG and doubles the same way.
    eng = np.arange(3 * 59, dtype=np.float32).reshape(3, 59)
    eng_aug, eng_y = augment(eng, np.array([5, 4, 3], dtype=np.int64), seed=7)
    assert eng_aug.shape == (6, 59) and eng_y.tolist() == [5, 4, 3, 5, 4, 3]
