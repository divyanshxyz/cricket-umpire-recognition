"""TwoHeadAttentionBiGRU forward-pass shape/param/pad tests (P2RDY-01).

Torch-only, weight-free, deterministic: torch.manual_seed(0) + torch.randn
inputs, no yolov8n-pose weights, no network, no training code. Tests 1-3
pin the stub's forward shapes and constructor params, Test 4 pins
pad_or_truncate semantics (zero-pad at the end / truncate from the end /
no-op at 15), Test 5 proves the Plan 01 (T,118) sequence contract feeds the
stub without rewrites, and Test 6 is the suite's second weight-gated site:
a REAL (3,118) extract_sequence_features output composed through
pad_or_truncate into forward — SC4's real-sequence-to-stub composition in
one test.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.temporal import TwoHeadAttentionBiGRU, pad_or_truncate

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = REPO_ROOT / "models" / "yolov8n-pose.pt"


def test_default_forward_shape_2_15_118_to_2_15_6():
    # Defaults: in_features=118, n_classes=6. Seeded randn -> exact shape.
    torch.manual_seed(0)
    model = TwoHeadAttentionBiGRU()
    out = model(torch.randn(2, 15, 118))
    assert out.shape == (2, 15, 6)


def test_n_classes_constructor_param_respected():
    # Constructor params are honored end-to-end: n_classes=8 -> (2,15,8).
    torch.manual_seed(0)
    model = TwoHeadAttentionBiGRU(n_classes=8)
    out = model(torch.randn(2, 15, 118))
    assert out.shape == (2, 15, 8)


def test_batch_1_forward_shape():
    # B=1 batch keeps the per-timestep contract: (1, 15, 118) -> (1, 15, 6).
    torch.manual_seed(0)
    model = TwoHeadAttentionBiGRU()
    out = model(torch.randn(1, 15, 118))
    assert out.shape == (1, 15, 6)


def test_pad_or_truncate_semantics():
    # (4,118) -> zero-pad AT THE END: rows 4..14 exactly zero, first 4 kept.
    seq4 = np.arange(4 * 118, dtype=np.float32).reshape(4, 118)
    padded = pad_or_truncate(seq4, 15)
    assert padded.shape == (15, 118)
    assert (padded[:4] == seq4).all()
    assert (padded[4:] == 0.0).all()
    # (20,118) -> truncate FROM THE END: keep exactly the FIRST 15 rows.
    seq20 = np.arange(20 * 118, dtype=np.float32).reshape(20, 118)
    trimmed = pad_or_truncate(seq20, 15)
    assert trimmed.shape == (15, 118)
    assert (trimmed == seq20[:15]).all()
    # (15,118) -> no-op: identical values.
    seq15 = np.arange(15 * 118, dtype=np.float32).reshape(15, 118)
    same = pad_or_truncate(seq15, 15)
    assert same.shape == (15, 118)
    assert (same == seq15).all()


def test_sequence_to_stub_wiring():
    # The Plan 01 seam contract ((T,118) rows) assembles into the stub's
    # (B,15,118) input with no rewrites: pad_or_truncate -> unsqueeze ->
    # forward -> (1,15,6).
    torch.manual_seed(0)
    seq = torch.randn(15, 118)
    batch = pad_or_truncate(seq.numpy(), 15)
    out = TwoHeadAttentionBiGRU()(torch.from_numpy(np.ascontiguousarray(batch)).unsqueeze(0))
    assert out.shape == (1, 15, 6)


@pytest.mark.skipif(
    not WEIGHTS.exists(), reason="yolov8n-pose weights not bootstrapped"
)
def test_real_sequence_composition_through_stub():
    # SC4 composition in ONE test: a real manifest image -> (3,118) sequence
    # features -> pad_or_truncate(15) -> unsqueeze -> forward -> (1,15,6)
    # with finite logits. Uses the same crop_path-else-original_path
    # resolution as tests/test_pose_features.py.
    manifest = pd.read_csv(
        REPO_ROOT / "data" / "processed" / "manifest.csv", keep_default_na=False
    )
    image_path = None
    for _, row in manifest.iterrows():
        rel = str(row["crop_path"]).strip() or str(row["original_path"]).strip()
        candidate = REPO_ROOT / rel
        if candidate.exists():
            image_path = candidate
            break
    assert image_path is not None, "no manifest row with an existing image path"

    from src.pose_features import PoseFeatureExtractor

    image = cv2.imread(str(image_path))  # BGR uint8 (OpenCV convention)
    assert image is not None, f"cv2.imread failed on {image_path}"
    extractor = PoseFeatureExtractor()
    seq, failure = extractor.extract_sequence_features([image, image, image])
    assert failure is None  # typed-failure absence on a real detection
    assert seq.shape == (3, 118)
    assert seq.dtype == np.float32

    torch.manual_seed(0)
    model = TwoHeadAttentionBiGRU()
    batch = pad_or_truncate(seq, 15)
    out = model(torch.from_numpy(np.ascontiguousarray(batch)).unsqueeze(0))
    assert out.shape == (1, 15, 6)
    assert torch.isfinite(out).all()
