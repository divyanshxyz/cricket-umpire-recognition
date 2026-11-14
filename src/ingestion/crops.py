"""Clamped-bbox crop math + crop writer for Roboflow survivors.

RESEARCH.md crop example (15% clamped padding, image-size-scaled — never a
constant). SEAM STUBS: Plan 02 fills the bodies behind these contracts
(clamped_box / largest_box / crop_box / crop_row); crop filenames derive
from manifest row IDs (crop_{row_id}.jpg), never source filenames (ASVS
V14), and writes go only under the configured crops_dir — ``datasets/``
stays read-only.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def clamped_box(W, H, cx, cy, w, h, pad_frac=0.15):
    """Normalized center/size -> pixel corners (x0, y0, x1, y1) with padding.

    SEAM STUB — Plan 02 implements: convert with the DECODED image's actual
    size, expand by pad_frac per side, clamp inclusively to [0, W]x[0, H];
    reject degenerate zero-area boxes.
    """
    return (0, 0, 0, 0)


def largest_box(boxes):
    """Max-area bbox (cx, cy, w, h) with a deterministic tiebreak.

    SEAM STUB — Plan 02 implements: max w*h among candidates; ties resolve
    to the lexicographically smallest tuple (order-independent).
    """
    return None


def crop_box(img_path, cx, cy, w, h, pad_frac=0.15):
    """Open an image read-only, crop the clamped padded box, return RGB image.

    SEAM STUB — Plan 02 implements behind this contract.
    """
    return Image.new("RGB", (0, 0))


def crop_row(record, row_id, crops_dir, img_path, pad_frac=0.15):
    """Crop one Roboflow survivor row -> manifest crop fields.

    SEAM STUB — Plan 02 implements: writes f"crop_{row_id}.jpg" (quality 95)
    under crops_dir and returns {"crop_x0", "crop_y0", "crop_x1", "crop_y1",
    "crop_path"}; degenerate boxes raise ValueError.
    """
    return {
        "crop_x0": 0,
        "crop_y0": 0,
        "crop_x1": 0,
        "crop_y1": 0,
        "crop_path": "",
    }
