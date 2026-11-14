"""Phase 4 annotation renderer: draws ONLY the selected umpire.

Ultralytics' ``Results.plot()`` is deliberately rejected: it annotates ALL
detections with no per-instance filter (docs.ultralytics.com/modes/predict),
which is misleading when the features come from ONE selected umpire — the
analyzer extracts exactly one person's keypoints (select_umpire) and the
skeleton must show that same person, nobody else. Drawing is manual cv2
(RESEARCH.md Pattern 3) from the raw51 pixel keypoints + the selected box.

Pure function of (frame, kpts, bbox): no model, no I/O, no imports beyond
cv2/numpy. kpts17x3 is the raw51 reshape (17, 3) in pixel coords (x, y,
conf) — the DETECTED values, unmasked (mask_conf only gates visibility).
"""

import cv2
import numpy as np

# VERIFIED verbatim from the installed
# .venv/Lib/site-packages/ultralytics/utils/plotting.py lines 425-445
# (Annotator.skeleton): 1-based limb indices over COCO-17 (0 = background).
# Converting to 0-based by subtracting 1 gives exactly 19 limb pairs — the
# ultralytics skeleton has 19 pairs, so LIMBS has 19 entries (RESEARCH
# Pattern 3's "20 limbs" aside is a typo, not copied).
ULTRA_SKELETON_1BASED = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]

# 0-based COCO-17 limbs (19 pairs).
LIMBS = [(a - 1, b - 1) for a, b in ULTRA_SKELETON_1BASED]

# BGR colors (OpenCV channel order): green box, blue limbs, red keypoints.
_BBOX_COLOR = (0, 255, 0)
_LIMB_COLOR = (255, 0, 0)
_KPT_COLOR = (0, 0, 255)


def draw_annotation(frame, kpts17x3, bbox_xyxy, mask_conf=0.20):
    """Draw the selected umpire's skeleton + bbox on a copy of the frame.

    frame: BGR HWC uint8 ndarray (the cv2.VideoCapture frame contract).
    kpts17x3: (17, 3) detected pixel coords + confidence — the raw51
        reshape; the input is never mutated (the frame is copied first).
    bbox_xyxy: (4,) selected umpire box (x0, y0, x1, y1) from the
        annotation predict's selected detection.
    mask_conf: visibility boundary — a limb line is drawn only when BOTH
        endpoint confidences >= mask_conf; a keypoint dot only when its
        confidence >= mask_conf (strict-< masking, pose_features.py
        mask_keypoints semantics: conf exactly at the threshold stays
        visible).

    Returns the annotated copy; the input frame is left untouched.
    """
    ann = frame.copy()
    x0, y0, x1, y1 = (int(round(float(v))) for v in bbox_xyxy)
    cv2.rectangle(ann, (x0, y0), (x1, y1), _BBOX_COLOR, 2)
    kpts = np.asarray(kpts17x3, dtype=np.float64).reshape(17, 3)
    for a, b in LIMBS:
        if kpts[a, 2] >= mask_conf and kpts[b, 2] >= mask_conf:
            pa = (int(kpts[a, 0]), int(kpts[a, 1]))
            pb = (int(kpts[b, 0]), int(kpts[b, 1]))
            cv2.line(ann, pa, pb, _LIMB_COLOR, 2)
    for x, y, c in kpts:
        if c >= mask_conf:
            cv2.circle(ann, (int(x), int(y)), 3, _KPT_COLOR, -1)
    return ann
