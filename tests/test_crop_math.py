"""Crop-math edge cases (RESEARCH.md Pitfall 5/6): 15% padding then clamp,
flush edge crops, no out-of-bounds boxes, degenerate rejection, largest-area
tiebreak, and a real-image crop size check. Pure math + one tiny Pillow image
via tmp_path.
"""

import pytest
from PIL import Image

from src.ingestion.adapters import Record
from src.ingestion.crops import clamped_box, crop_box, crop_row, largest_box


def test_centered_box_gets_exact_15pct_padding():
    # W=H=100; a 0.4x0.4 centered box spans (30,30)-(70,70); 15% of 40 = 6.
    assert clamped_box(100, 100, 0.5, 0.5, 0.4, 0.4, pad_frac=0.15) == (24, 24, 76, 76)


def test_left_edge_box_yields_flush_crop():
    # cx=0.1, w=0.2 -> x1=0, x2=20, px=3 -> padded x0=-3 clamps flush to 0.
    x0, y0, x1, y1 = clamped_box(100, 100, 0.1, 0.5, 0.2, 0.4, pad_frac=0.15)
    assert x0 == 0
    assert x1 == 23  # 20 + 3 (unclamped side keeps its padding)


def test_padded_extent_clamped_to_image_bounds():
    # cx=0.95, w=0.2 -> x1=85, x2=105, px=6 -> padded 111 clamps to W=100.
    x0, y0, x1, y1 = clamped_box(100, 100, 0.95, 0.5, 0.2, 0.4, pad_frac=0.15)
    assert x1 == 100
    assert 0 <= x0 <= x1 <= 100
    assert 0 <= y0 <= y1 <= 100


def test_degenerate_zero_area_box_rejected():
    with pytest.raises(ValueError):
        clamped_box(100, 100, 0.5, 0.5, 0.0, 0.4)  # w=0
    with pytest.raises(ValueError):
        clamped_box(100, 100, 0.5, 0.5, 0.4, 0.0)  # h=0


def test_largest_box_picks_max_area():
    small = (0.5, 0.5, 0.2, 0.2)  # area 0.04
    big = (0.5, 0.5, 0.6, 0.5)    # area 0.30
    assert largest_box([small, big]) == big


def test_largest_box_tie_is_deterministic():
    b1 = (0.5, 0.5, 0.4, 0.5)  # area 0.20
    b2 = (0.3, 0.3, 0.5, 0.4)  # area 0.20 — tie
    # Order-independent: the tie always resolves to the same box.
    assert largest_box([b1, b2]) == largest_box([b2, b1])
    # Deterministic tiebreak: lexicographically smallest tuple wins.
    assert largest_box([b1, b2]) == min(b1, b2)


def test_crop_box_size_matches_clamped_box(tmp_path):
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (120, 80), color=(10, 200, 30)).save(img_path)
    box = clamped_box(120, 80, 0.5, 0.5, 0.4, 0.4, pad_frac=0.15)
    cropped = crop_box(img_path, 0.5, 0.5, 0.4, 0.4, pad_frac=0.15)
    assert cropped.size[0] > 0 and cropped.size[1] > 0  # a real crop, not a stub
    assert cropped.size == (box[2] - box[0], box[3] - box[1])


def test_crop_row_writes_row_id_filename_under_crops_dir(tmp_path):
    # ASVS V14: crop filenames derive from the manifest row ID, never the
    # source filename (source relpath is deliberately odd and ignored).
    img_path = tmp_path / "weird source.file.jpg"
    Image.new("RGB", (100, 100), color=(1, 2, 3)).save(img_path)
    record = Record(
        source="rf", relpath="train/images/weird source.file.jpg",
        family="roboflow-yolo", priority=3, raw_label="four",
        bbox=(0.5, 0.5, 0.4, 0.4),
    )
    crops_dir = tmp_path / "crops"
    result = crop_row(record, "rf_00042", crops_dir, img_path=img_path)
    assert result["crop_path"] == "crop_rf_00042.jpg"
    assert (crops_dir / "crop_rf_00042.jpg").is_file()
    assert result["crop_x0"] < result["crop_x1"]
    assert result["crop_y0"] < result["crop_y1"]
    with Image.open(crops_dir / "crop_rf_00042.jpg") as saved:
        assert saved.size == (result["crop_x1"] - result["crop_x0"],
                              result["crop_y1"] - result["crop_y0"])
