"""Extract-guard containment tests (weight-free, stubbed model): the
PoseFeatureExtractor.extract schema guards at the seam contract boundary —
a (17, 3) tensor passes, a wrong keypoint COUNT (e.g. (24, 3) from a
non-COCO checkpoint) is a typed no_keypoints failure, and the
write_features_npz fail-closed width guard refuses mis-shaped arrays.

CR-02 regression: before the fix, only shape[-1] (confidence channel) was
checked — a (24, 3) tensor flowed through as SUCCESS rows (raw (72,) +
eng (80,)) and np.stack wrote a silently mis-shaped frozen features.npz
while the build reported ok (T-02-04 silent-corruption class).

No model load: the lazy _ensure_model seam is monkeypatched with a stub
returning synthetic Results-like objects, so these run without the
yolov8n-pose weights.
"""

import numpy as np
import pytest

from src.pose_features import (
    ExtractionFailure,
    PoseFeatureExtractor,
    failure_events,
)


class _FakeBoxes:
    def __len__(self):
        return 1


class _FakeKeypoints:
    def __init__(self, data):
        self.data = data


class _FakeResult:
    """Minimal Results-like surface for what extract() touches: boxes
    (non-empty) + keypoints.data. The success-path case supplies torch
    tensors so the .cpu().numpy() accessors work; the guard-trip cases
    never reach them."""

    def __init__(self, keypoints_data, boxes=None, orig_img=None):
        self.boxes = boxes if boxes is not None else _FakeBoxes()
        self.keypoints = _FakeKeypoints(keypoints_data)
        self.orig_img = orig_img


class _FakeModel:
    def __init__(self, result):
        self._result = result

    def predict(self, **kwargs):
        return [self._result]


def _extractor_with(monkeypatch, result):
    extractor = PoseFeatureExtractor()
    monkeypatch.setattr(extractor, "_ensure_model", lambda: _FakeModel(result))
    return extractor


def test_wrong_keypoint_count_is_typed_no_keypoints_failure(monkeypatch):
    # CR-02: a (24, 3) tensor (non-COCO-17 pose checkpoint) must NOT pass
    # the seam as success rows — it is a typed no_keypoints failure (the
    # closed-set reason shared with the missing-pose-head trip), recorded
    # in the module event list, with the actual shape named in the detail.
    result = _FakeResult(np.zeros((1, 24, 3), dtype=np.float32))
    extractor = _extractor_with(monkeypatch, result)
    before = len(failure_events())
    raw, failure = extractor.extract(np.zeros((64, 64, 3), dtype=np.uint8))
    assert raw is None
    assert isinstance(failure, ExtractionFailure)
    assert failure.reason == "no_keypoints"
    assert "(1, 24, 3)" in failure.detail
    assert len(failure_events()) == before + 1


def test_missing_confidence_channel_keeps_its_typed_reason(monkeypatch):
    # The conf-channel guard keeps its original closed-set reason: a
    # (17, 2) tensor has no confidence channel -> keypoints_without_confidence.
    result = _FakeResult(np.zeros((1, 17, 2), dtype=np.float32))
    extractor = _extractor_with(monkeypatch, result)
    _raw, failure = extractor.extract(np.zeros((64, 64, 3), dtype=np.uint8))
    assert _raw is None
    assert failure.reason == "keypoints_without_confidence"


def test_valid_17_3_tensor_passes_the_schema_guards(monkeypatch):
    # Positive control: a (1, 17, 3) tensor flows through BOTH guards into
    # the locked math and returns the frozen (51,) + (59,) success tuple —
    # the new count guard does not false-positive on COCO-17 checkpoints.
    import torch

    class _FakeTorchBoxes:
        xyxy = torch.zeros(1, 4)
        conf = torch.zeros(1)

        def __len__(self):
            return 1

    result = _FakeResult(
        torch.zeros(1, 17, 3),
        boxes=_FakeTorchBoxes(),
        orig_img=np.zeros((64, 64, 3), dtype=np.uint8),
    )
    extractor = _extractor_with(monkeypatch, result)
    raw, eng = extractor.extract(np.zeros((64, 64, 3), dtype=np.uint8))
    assert raw is not None and eng is not None
    assert raw.shape == (51,) and eng.shape == (59,)
    assert raw.dtype == np.float32 and eng.dtype == np.float32


def test_model_load_failure_fails_fast_not_decode_error(monkeypatch):
    # WR-02 regression: model loading sits OUTSIDE the per-image
    # containment — a missing/corrupt weights path raises from extract()
    # (fail-fast config error, zero rows processed) instead of being
    # swallowed into a per-image decode_error row (which previously
    # produced 421 identical typed failures for one bad weights_path).
    def boom():
        raise FileNotFoundError("yolov8n-pose.pt missing (bad weights_path)")

    extractor = PoseFeatureExtractor()
    monkeypatch.setattr(extractor, "_ensure_model", boom)
    before = len(failure_events())
    with pytest.raises(FileNotFoundError):
        extractor.extract(np.zeros((64, 64, 3), dtype=np.uint8))
    assert len(failure_events()) == before  # NOT recorded as a decode_error


def test_warm_up_loads_model_eagerly(monkeypatch):
    # WR-02: batch callers warm the model up once before the loop so a
    # weights/config error fails before any row is processed.
    calls = []

    def fake_ensure(self):
        calls.append(1)
        return object()

    monkeypatch.setattr(PoseFeatureExtractor, "_ensure_model", fake_ensure)
    extractor = PoseFeatureExtractor()
    assert extractor.warm_up() is extractor
    assert len(calls) == 1
