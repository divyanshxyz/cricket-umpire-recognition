"""Weight-gated end-to-end smoke for the pose seam (the suite's only skipif
site): one real manifest image flows predict -> select_umpire -> mask ->
(51,) raw + (59,) engineered float32, and a blank image returns a typed
no_person failure that the event recorder counts. Skipped only when the
yolov8n-pose weights are not bootstrapped (Task 1 bootstraps them in Phase 2,
so this module must RUN, not skip, at verify time).
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.pose_features import PoseFeatureExtractor, failure_events, engineer_extended_features

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = REPO_ROOT / "models" / "yolov8n-pose.pt"

pytestmark = pytest.mark.skipif(
    not WEIGHTS.exists(), reason="yolov8n-pose weights not bootstrapped"
)


def _first_manifest_image():
    """First manifest row (stored order) whose resolved image path exists.
    Resolution contract: crop_path when non-blank, else original_path —
    both repo-relative, joined under the repo root."""
    manifest = pd.read_csv(
        REPO_ROOT / "data" / "processed" / "manifest.csv", keep_default_na=False
    )
    for _, row in manifest.iterrows():
        rel = str(row["crop_path"]).strip() or str(row["original_path"]).strip()
        path = REPO_ROOT / rel
        if path.exists():
            return row, path
    raise AssertionError("no manifest row with an existing image path")


def test_real_manifest_image_end_to_end():
    # One real committed dataset image through the full seam: predict ->
    # select_umpire -> mask -> (51,) raw + (59,) engineered, float32.
    extractor = PoseFeatureExtractor()
    _, image_path = _first_manifest_image()
    image = cv2.imread(str(image_path))  # BGR uint8 (OpenCV convention)
    assert image is not None, f"cv2.imread failed on {image_path}"
    raw, engineered = extractor.extract(image)
    assert raw is not None and engineered is not None
    assert raw.shape == (51,)
    assert engineered.shape == (59,)
    assert raw.dtype == np.float32
    assert engineered.dtype == np.float32
    assert raw.any()  # raw vector from a real detection is not all-zero
    assert np.isfinite(raw).all()
    # Task 3 proof: the engineered slot is filled with the real math —
    # NOT the all-zero stub — and stays finite on a real detection.
    assert engineered.any()
    assert np.isfinite(engineered).all()


def test_real_image_three_frame_sequence_contract():
    # The same real image passed 3 times: identical frames -> identical
    # keypoints -> the velocity block [110:118] is exactly zero, and the
    # frozen (T, 118) column layout holds end-to-end with no typed failure.
    extractor = PoseFeatureExtractor()
    _, image_path = _first_manifest_image()
    image = cv2.imread(str(image_path))
    assert image is not None, f"cv2.imread failed on {image_path}"
    seq, failure = extractor.extract_sequence_features([image, image, image])
    assert failure is None  # typed-failure absence
    assert seq.shape == (3, 118)
    assert seq.dtype == np.float32
    assert np.isfinite(seq).all()
    assert (seq[:, 110:118] == 0.0).all()  # identical frames -> zero deltas
    assert (seq[0, 0:51] == seq[1, 0:51]).all()  # deterministic repeat rows
    assert (seq[0, 51:110] == seq[1, 51:110]).all()


def test_blank_image_returns_typed_no_person_failure():
    # A blank white 64x64 frame contains no person: the guard returns
    # (None, ExtractionFailure("no_person")) — no exception escapes — and
    # the module event recorder counts exactly one more failure.
    extractor = PoseFeatureExtractor()
    blank = np.full((64, 64, 3), 255, dtype=np.uint8)
    before = len(failure_events())
    result, failure = extractor.extract(blank)
    assert result is None
    assert failure.reason == "no_person"
    assert len(failure_events()) == before + 1


def test_engineer_extended_features_shape_and_finite():
    # Verify 1D and 2D inputs, (81,) and (N, 81) output shapes, and exact 59-D preservation
    np.random.seed(42)
    f_raw_1d = np.random.uniform(10, 200, size=(51,)).astype(np.float32)
    f_eng_1d = np.random.uniform(0.1, 2.0, size=(59,)).astype(np.float32)

    ext_1d = engineer_extended_features(f_raw_1d, f_eng_1d)
    assert ext_1d.shape == (81,)
    assert ext_1d.dtype == np.float32
    assert np.isfinite(ext_1d).all()
    # Preserves first 59 dimensions exactly
    np.testing.assert_allclose(ext_1d[:59], f_eng_1d)

    # 2D batch test
    f_raw_batch = np.random.uniform(10, 200, size=(10, 51)).astype(np.float32)
    f_eng_batch = np.random.uniform(0.1, 2.0, size=(10, 59)).astype(np.float32)
    ext_batch = engineer_extended_features(f_raw_batch, f_eng_batch)
    assert ext_batch.shape == (10, 81)
    assert ext_batch.dtype == np.float32
    assert np.isfinite(ext_batch).all()
    np.testing.assert_allclose(ext_batch[:, :59], f_eng_batch)


def test_engineer_extended_features_scale_invariance():
    # If keypoints are scaled by factor c, normalized features in f_eng are invariant,
    # and the computed extended features must be strictly invariant.
    from src.pose_features import engineered_features

    # Create synthetic keypoints (17, 3): (x, y, conf)
    kpts_base = np.zeros((17, 3), dtype=np.float64)
    kpts_base[:, 2] = 0.9  # all confident
    # Nose
    kpts_base[0, :2] = [100.0, 50.0]
    # L/R Shoulder
    kpts_base[5, :2] = [80.0, 100.0]
    kpts_base[6, :2] = [120.0, 100.0]
    # L/R Elbow
    kpts_base[7, :2] = [60.0, 140.0]
    kpts_base[8, :2] = [140.0, 140.0]
    # L/R Wrist
    kpts_base[9, :2] = [50.0, 180.0]
    kpts_base[10, :2] = [150.0, 180.0]
    # L/R Hip
    kpts_base[11, :2] = [85.0, 200.0]
    kpts_base[12, :2] = [115.0, 200.0]

    scale_base = 100.0
    eng_base = engineered_features(kpts_base, scale_base)
    ext_base = engineer_extended_features(None, eng_base)

    # Double scale
    scale_2x = 200.0
    kpts_2x = kpts_base.copy()
    kpts_2x[:, :2] *= 2.0
    eng_2x = engineered_features(kpts_2x, scale_2x)
    ext_2x = engineer_extended_features(None, eng_2x)

    np.testing.assert_allclose(ext_base, ext_2x, rtol=1e-5, atol=1e-5)


def test_engineer_extended_features_relational_semantics():
    from src.pose_features import engineered_features

    kpts = np.zeros((17, 3), dtype=np.float64)
    kpts[:, 2] = 0.85
    # Nose at (100, 50)
    kpts[0, :2] = [100.0, 50.0]
    # L/R Shoulder at (80, 100) and (120, 100)
    kpts[5, :2] = [80.0, 100.0]
    kpts[6, :2] = [120.0, 100.0]
    # L/R Elbow at (60, 100) and (140, 100)
    kpts[7, :2] = [60.0, 100.0]
    kpts[8, :2] = [140.0, 100.0]
    # L/R Wrist at (40, 100) and (160, 100)
    kpts[9, :2] = [40.0, 100.0]
    kpts[10, :2] = [160.0, 100.0]
    # L/R Hip at (85, 200) and (115, 200)
    kpts[11, :2] = [85.0, 200.0]
    kpts[12, :2] = [115.0, 200.0]

    scale = 100.0  # torso scale
    eng = engineered_features(kpts, scale)
    ext = engineer_extended_features(None, eng)

    # Wrist-to-wrist: dx = (160 - 40)/100 = 1.2, dy = 0, dist = 1.2
    assert pytest.approx(ext[59], rel=1e-4) == 1.2  # distance
    assert pytest.approx(ext[60], rel=1e-4) == 1.2  # dx
    assert pytest.approx(ext[61], abs=1e-4) == 0.0  # dy

    # Wrist-to-nose: nose.y=0.5, wrist.y=1.0 -> elevation dy = 0.5 - 1.0 = -0.5 (below nose)
    assert pytest.approx(ext[62], rel=1e-4) == -0.5
    assert pytest.approx(ext[63], rel=1e-4) == -0.5
    assert pytest.approx(ext[64], rel=1e-4) == -0.5  # max
    assert pytest.approx(ext[65], rel=1e-4) == -0.5  # min
    assert pytest.approx(ext[66], abs=1e-4) == 0.0   # abs diff

    # Arm conf: mean of 6 joints (5..10), each is 0.85
    assert pytest.approx(ext[80], rel=1e-4) == 0.85

