"""Phase 5 Tab 1 (Image Analyzer) engine-seam pins — 05-01-PLAN Task 1.

Contracts pinned here:

- annotate_image wrapper (UI-SPEC Decision 10, RED-first): the additive
  public seam on UmpireVideoAnalyzer returns an annotated ndarray of the
  input's shape for a real umpire image, and raises the typed [input]-class
  ValueError (naming the function and the extraction failure reason) on a
  no-person image — mirroring classify_image's fail-closed message shape.
- Engine-path e2e (added GREEN): fixture bytes -> validation seam -> engine
  -> LabelEncoder-ordered probabilities + 8 frozen-label arm-angle rows.
- Validation units (added GREEN): size-before-magic order, exact oversize /
  wrong-content copies, decode-or-None garbage handling.

Weight-gated skipif posture (tests/test_inference_analyzer.py:43-47
precedent): the module skips on a fresh clone without bootstrapped weights,
and the fixture-image tests skip when the UNTRACKED dataset image is absent
— on THIS machine both gates pass, so every test here must RUN, not skip.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import FIXTURE_IMAGE
from src.inference.analyzer import UmpireVideoAnalyzer

REPO_ROOT = Path(__file__).resolve().parent.parent

_WEIGHTS = [
    REPO_ROOT / "models" / "yolov8n-pose.pt",
    REPO_ROOT / "models" / "baseline_ann.pt",
    REPO_ROOT / "models" / "label_encoder.joblib",
]

pytestmark = pytest.mark.skipif(
    not all(w.exists() for w in _WEIGHTS),
    reason="yolov8n-pose / baseline_ann / label_encoder weights not bootstrapped "
    "(fresh clone posture — the analyzer's startup integrity sweep needs them)",
)

_requires_fixture_image = pytest.mark.skipif(
    not FIXTURE_IMAGE.exists(),
    reason="untracked dataset fixture image absent (datasets/ is not git-tracked)",
)


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
    # The extractor's typed failure reason (e.g. no_person) is embedded
    # verbatim in the message — the same shape classify_image raises.
    assert "no_person" in message
