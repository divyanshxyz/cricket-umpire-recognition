"""Clamped-bbox crop math + crop writer for Roboflow survivors.

RESEARCH.md crop example: normalized center/size convert with the DECODED
image's actual size (never a constant — Pitfall 5), expanded by 15% padding
per side, clamped inclusively to [0, W]x[0, H]. Degenerate zero-area boxes
are rejected (invalid — counted by the caller, the run continues per
T-01-05). Crop filenames derive from manifest row IDs (crop_{row_id}.jpg),
NEVER source filenames (ASVS V14 — source stems contain dots/space variants
across six sources and trusting them enables path traversal). The source
image is opened read-only; all writes stay under the configured crops_dir —
``datasets/`` is never touched.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def clamped_box(W, H, cx, cy, w, h, pad_frac=0.15):
    """Normalized center/size -> pixel corners (x0, y0, x1, y1), 15% padded.

    Converts with the image's actual (W, H), expands each side by
    ``pad_frac * side-length`` of the box, then clamps inclusively to
    [0, W]x[0, H]. Degenerate zero-area boxes (w<=0 or h<=0) raise
    ValueError — the caller counts them as invalid rows and continues.
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"degenerate zero-area box (w={w}, h={h}) — invalid, not cropped")
    x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
    x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
    px, py = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    return (
        max(0, int(x1 - px)),
        max(0, int(y1 - py)),
        min(W, int(x2 + px)),
        min(H, int(y2 + py)),
    )


def largest_box(boxes):
    """Max-area bbox (cx, cy, w, h) among candidates, deterministic on ties.

    Ties resolve to the lexicographically smallest tuple — order-independent
    and reproducible across runs (RESEARCH.md Pattern 4).

    NOTE (IN-04): this helper is exercised by tests/test_crop_math.py only.
    The pipeline's LIVE box selection — adapters._parse_label_boxes — uses a
    different, equally deterministic tie rule: max(..., key=area) keeps the
    FIRST row in file order on an area tie. The two rules must not be
    swapped silently: flipping the pipeline's tie rule changes which box
    wins on tied files and therefore the committed artifacts' bytes.
    """
    boxes = list(boxes)
    if not boxes:
        raise ValueError("largest_box: no candidate boxes")
    return sorted(boxes, key=lambda b: (-(b[2] * b[3]), b))[0]


def crop_box(img_path, cx, cy, w, h, pad_frac=0.15):
    """Open an image read-only, crop the clamped padded box, return RGB image."""
    with Image.open(img_path) as im:
        W, H = im.size
        box = clamped_box(W, H, cx, cy, w, h, pad_frac)
        return im.convert("RGB").crop(box)


def crop_row(record, row_id, crops_dir, img_path, pad_frac=0.15):
    """Crop one Roboflow survivor row -> manifest crop fields.

    Writes f"crop_{row_id}.jpg" (quality 95) under crops_dir and returns
    {"crop_x0", "crop_y0", "crop_x1", "crop_y1", "crop_path"} where crop_path
    is the bare filename — the caller prefixes its repo-relative directory.
    Degenerate zero-area bboxes raise ValueError (invalid-box: the caller
    counts it and the deterministic run continues, T-01-05); other failures
    propagate to the caller's per-file guard.
    """
    cx, cy, w, h = record.bbox
    with Image.open(img_path) as im:
        W, H = im.size
        box = clamped_box(W, H, cx, cy, w, h, pad_frac)
        cropped = im.convert("RGB").crop(box)
    dst = Path(crops_dir) / f"crop_{row_id}.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(dst, quality=95)
    return {
        "crop_x0": box[0],
        "crop_y0": box[1],
        "crop_x1": box[2],
        "crop_y1": box[3],
        "crop_path": f"crop_{row_id}.jpg",
    }
