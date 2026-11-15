"""Unit tests for MobileNetV3 VisualFeatureExtractor (FUSE-02)."""

import numpy as np
import pytest
import torch

from src.inference.vision_extractor import VisualFeatureExtractor


@pytest.fixture(scope="module")
def extractor():
    return VisualFeatureExtractor(device="cpu")


def test_extractor_output_dimension_and_dtype(extractor):
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    vec = extractor.extract_crop(img, bbox=(100, 100, 300, 400))
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (576,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()
    assert (vec != 0.0).any()


def test_fallback_on_none_and_empty_bbox(extractor):
    img = np.random.randint(0, 255, (400, 500, 3), dtype=np.uint8)

    # None bbox -> center crop
    vec_none = extractor.extract_crop(img, bbox=None)
    assert vec_none.shape == (576,)
    assert np.isfinite(vec_none).all()

    # Empty bbox -> center crop
    vec_empty = extractor.extract_crop(img, bbox=())
    assert vec_empty.shape == (576,)
    np.testing.assert_allclose(vec_none, vec_empty)


def test_fallback_on_degenerate_bbox(extractor):
    img = np.random.randint(0, 255, (400, 500, 3), dtype=np.uint8)

    # Inverted bbox (x0 >= x1, y0 >= y1) -> fallback to center crop
    vec_degen = extractor.extract_crop(img, bbox=(300, 300, 100, 100))
    vec_none = extractor.extract_crop(img, bbox=None)
    assert vec_degen.shape == (576,)
    np.testing.assert_allclose(vec_degen, vec_none)

    # Zero-area bbox
    vec_zero = extractor.extract_crop(img, bbox=(100, 100, 100, 100))
    np.testing.assert_allclose(vec_zero, vec_none)


def test_crop_padding_clamping(extractor):
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Bbox close to edge (5, 5, 25, 25) -> padding should clamp to [0, 200]
    crop = extractor.crop_image(img, bbox=(5, 5, 25, 25), pad_fraction=0.10)
    assert crop.shape[0] > 0 and crop.shape[1] > 0


def test_batch_extraction(extractor):
    img1 = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    batch_items = [
        (img1, (20, 20, 150, 150)),
        (img2, None),
    ]

    batch_vecs = extractor.extract_batch(batch_items)
    assert batch_vecs.shape == (2, 576)
    assert batch_vecs.dtype == np.float32
    assert np.isfinite(batch_vecs).all()

    # Verify batch results match individual extract_crop results
    single_1 = extractor.extract_crop(img1, bbox=(20, 20, 150, 150))
    single_2 = extractor.extract_crop(img2, bbox=None)

    np.testing.assert_allclose(batch_vecs[0], single_1, atol=1e-5)
    np.testing.assert_allclose(batch_vecs[1], single_2, atol=1e-5)


def test_empty_batch(extractor):
    empty_vecs = extractor.extract_batch([])
    assert empty_vecs.shape == (0, 576)
    assert empty_vecs.dtype == np.float32
