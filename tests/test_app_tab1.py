"""Phase 5 Tab 1 (Image Analyzer) engine-seam pins — 05-01-PLAN Task 1.

Contracts pinned here:

- annotate_image wrapper (UI-SPEC Decision 10, RED-first): the additive
  public seam on UmpireVideoAnalyzer returns an annotated ndarray of the
  input's shape for a real umpire image, and raises the typed [input]-class
  ValueError (naming the function and the extraction failure reason) on a
  no-person image — mirroring classify_image's fail-closed message shape.
- Engine-path e2e: fixture bytes -> validation seam -> engine ->
  LabelEncoder-ordered probabilities + 8 frozen-label arm-angle rows.
- Validation units (weight-free): size-before-magic order, exact oversize /
  wrong-content copies, decode-or-None garbage handling.

Gating posture (tests/test_inference_analyzer.py precedent): the wrapper
and engine-path tests skip on a fresh clone without bootstrapped weights
and when the UNTRACKED dataset image is absent; the validation units are
pure and always run — on THIS machine both gates pass, so the gated tests
must RUN, not skip.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import FIXTURE_IMAGE
from src.app_support import (
    ARM_ANGLE_LABELS,
    IMAGE_LIMIT_BYTES,
    IMAGE_OVERSIZE_COPY,
    IMAGE_WRONG_CONTENT_COPY,
    VIDEO_LIMIT_BYTES,
    arm_angle_rows,
    check_image_bytes,
    decode_image_bgr,
)
from src.inference.analyzer import UmpireVideoAnalyzer
from src.inference.stabilizer import CLASS_ORDER

REPO_ROOT = Path(__file__).resolve().parent.parent

_WEIGHTS = [
    REPO_ROOT / "models" / "yolov8n-pose.pt",
    REPO_ROOT / "models" / "baseline_ann.pt",
    REPO_ROOT / "models" / "label_encoder.joblib",
]

_requires_weights = pytest.mark.skipif(
    not all(w.exists() for w in _WEIGHTS),
    reason="yolov8n-pose / baseline_ann / label_encoder weights not bootstrapped "
    "(fresh clone posture — the analyzer's startup integrity sweep needs them)",
)

_requires_fixture_image = pytest.mark.skipif(
    not FIXTURE_IMAGE.exists(),
    reason="untracked dataset fixture image absent (datasets/ is not git-tracked)",
)


# ----------------------------------------------------------------------
# Wrapper pins (RED-first: these failed with AttributeError before the
# additive annotate_image method existed on UmpireVideoAnalyzer)
# ----------------------------------------------------------------------


@_requires_weights
@_requires_fixture_image
def test_annotate_image_returns_same_shape_ndarray():
    """UI-SPEC Decision 10: annotate_image mirrors the per-frame compose
    (extract -> _annotate) for a single image — the annotated render is an
    ndarray of the input's shape (full frame, skeleton + bbox drawn)."""
    analyzer = UmpireVideoAnalyzer()
    bgr = cv2.imread(str(FIXTURE_IMAGE))
    assert bgr is not None, f"cv2.imread failed on {FIXTURE_IMAGE}"
    annotated = analyzer.annotate_image(bgr)
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == bgr.shape


@_requires_weights
def test_annotate_image_black_image_raises_typed_no_person():
    """A zeros image detects no umpire — annotate_image raises the typed
    [input]-class ValueError naming the function and the extraction failure
    reason (fail-closed, no silent default), never a silent raw frame."""
    analyzer = UmpireVideoAnalyzer()
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError) as err:
        analyzer.annotate_image(black)
    message = str(err.value)
    assert "annotate_image" in message
    # The extractor's typed failure reason (no_person for an empty frame)
    # is embedded verbatim in the message — the shape classify_image raises.
    assert "no_person" in message


# ----------------------------------------------------------------------
# Engine-path e2e (weights + fixture): upload bytes -> engine -> renders
# ----------------------------------------------------------------------


@_requires_weights
@_requires_fixture_image
def test_fixture_bytes_end_to_end_classification_and_arm_angles():
    """The Tab 1 happy path's engine spine: the fixture image's raw bytes
    pass the content gate, decode to BGR, and classify_image returns the
    6-class probability dict keyed EXACTLY in the LabelEncoder CLASS_ORDER
    summing to ~1.0; arm_angle_rows maps raw51 through the frozen seam
    functions into exactly 8 rows carrying the frozen labels in order."""
    fixture_bytes = FIXTURE_IMAGE.read_bytes()
    assert check_image_bytes(FIXTURE_IMAGE.name, fixture_bytes) is None
    bgr = decode_image_bgr(fixture_bytes)
    assert bgr is not None

    analyzer = UmpireVideoAnalyzer()
    result = analyzer.classify_image(bgr)
    assert tuple(result.probabilities.keys()) == CLASS_ORDER
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-5)

    mask_conf = analyzer.extractor.config["mask_conf"]
    rows = arm_angle_rows(result.raw51, mask_conf)
    assert len(rows) == 8
    assert [row["Feature"] for row in rows] == list(ARM_ANGLE_LABELS)
    for row in rows:
        assert isinstance(row["Value"], float)
        assert np.isfinite(row["Value"])


# ----------------------------------------------------------------------
# Validation units (weight-free: pure app_support boundary behavior)
# ----------------------------------------------------------------------


def test_limit_constants_are_the_locked_ui_spec_limits():
    """The locked UI-SPEC Upload Validation limits (image <= 10 MB,
    video <= 200 MB) — the constants the app's gates enforce."""
    assert IMAGE_LIMIT_BYTES == 10 * 1024 * 1024
    assert VIDEO_LIMIT_BYTES == 200 * 1024 * 1024


def test_oversize_buffer_returns_exact_oversize_copy():
    """A 10 MiB + 1 byte buffer is over the limit — the EXACT UI-SPEC
    oversize copy with size_mb = round(len/1048576, 1) (10.0 here)."""
    name = "big.jpg"
    data = b"\xff\xd8\xff" + b"\x00" * (IMAGE_LIMIT_BYTES - 3 + 1)
    assert len(data) == IMAGE_LIMIT_BYTES + 1
    assert check_image_bytes(name, data) == (
        "Image too large — 'big.jpg' is 10.0 MB. The limit is 10 MB — "
        "please upload a smaller image."
    )


def test_limit_buffer_passes_size_then_fails_magic():
    """Exactly 10 MiB of zero bytes passes the size step (not > limit) and
    fails LATER at the magic sniff — proving size-before-content order:
    the returned copy is the wrong-content copy, never the oversize one."""
    data = b"\x00" * IMAGE_LIMIT_BYTES
    result = check_image_bytes("zeros.jpg", data)
    assert result == IMAGE_WRONG_CONTENT_COPY.format(name="zeros.jpg")
    assert result != IMAGE_OVERSIZE_COPY.format(
        name="zeros.jpg", size_mb=round(len(data) / 1048576, 1)
    )


def test_jpeg_magic_garbage_passes_gate_but_fails_decode():
    """A JPEG-magic-prefixed garbage payload passes the sniff (the magic
    check cannot prove decodability) and decode_image_bgr returns None —
    the caller then renders the exact wrong-content copy."""
    data = b"\xff\xd8\xff" + b"this is not really a jpeg payload"
    assert check_image_bytes("garbage.jpg", data) is None
    assert decode_image_bgr(data) is None
