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

from src.pose_features import PoseFeatureExtractor, failure_events

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
