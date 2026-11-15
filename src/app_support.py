"""Phase 5 presentation-support helpers (streamlit-free pure functions).

Boundary (05-01-PLAN Step 3): ZERO inference/stabilizer/scoreboard decisions
live here — contract-object reuse and display mapping only. Importing this
module must NOT import streamlit (app.py is the only streamlit surface, and
the weight-free tests import these helpers directly).

Contents (image half; the video half + CSV helpers arrive with Plan 05-02):
- Upload limits (UI-SPEC Upload Validation table, locked): image <= 10 MB,
  video <= 200 MB — checked on the in-memory bytes BEFORE any sniff or
  engine call.
- check_image_bytes: size check FIRST, then the magic-byte sniff (JPEG
  FF D8 FF / PNG 89 50 4E 47) — the uploader's type= list is a convenience
  filter only, this content gate is the enforcement (T-05-01).
- decode_image_bgr: cv2.imdecode-or-None (caller renders the exact
  wrong-content copy on None).
- typed_error_copy: the three UI-SPEC Typed-Error templates keyed on stage,
  engine message embedded verbatim (no paraphrase, no truncation).
- arm_angle_rows: display mapping of result.raw51 through the frozen public
  seam functions ONLY (mask_keypoints -> torso_scale -> engineered_features
  [51:59]) — never a geometry re-derivation.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.pose_features import engineered_features, mask_keypoints, torso_scale

# Locked limits (UI-SPEC Upload Validation table) — bytes, enforced on the
# in-memory buffer length before any content sniff or engine call.
IMAGE_LIMIT_BYTES = 10 * 1024 * 1024
VIDEO_LIMIT_BYTES = 200 * 1024 * 1024

# Exact UI-SPEC copywriting-contract strings (toast copy; {name}/{size_mb}
# are the only slots). Rendered via st.error BEFORE any engine call.
IMAGE_OVERSIZE_COPY = (
    "Image too large — '{name}' is {size_mb} MB. The limit is 10 MB — "
    "please upload a smaller image."
)
IMAGE_WRONG_CONTENT_COPY = (
    "Unsupported image — '{name}' is not a readable JPG/PNG (content check "
    "failed). Please upload a .jpg or .png image; the file type is verified "
    "by content, not extension."
)

# Exact UI-SPEC Typed-Error Rendering templates (persistent st.error blocks;
# the engine's message renders verbatim inside {message} — the diagnostic
# record is never paraphrased, truncated, or swallowed).
_TYPED_ERROR_TEMPLATES = {
    "config": (
        "Configuration error — {message} Fix configs/inference.yaml and "
        "restart the app."
    ),
    "integrity": (
        "Model integrity failure — {message} The artifacts under models/ do "
        "not match configs/model_digests.yaml — re-verify or re-download "
        "the artifacts, then restart."
    ),
    "input": "Upload could not be processed — {message}",
}

# Frozen arm-angles display order (UI-SPEC Tab 1 step 6) — the 8 geometry
# features at engineered_features indices [51:59], never re-ordered.
ARM_ANGLE_LABELS = (
    "Left elbow angle (deg)",
    "Right elbow angle (deg)",
    "Left shoulder elevation",
    "Right shoulder elevation",
    "Left wrist vs nose",
    "Right wrist vs nose",
    "Horizontal span",
    "Vertical symmetry",
)


def check_image_bytes(name, data) -> str | None:
    """Upload content gate for images (T-05-01): size check FIRST on the
    buffer length, then the magic-byte sniff. Returns the exact UI-SPEC
    toast copy on the FIRST failing stage (proving size-before-content
    order), or None when both pass.

    ``name`` is the uploader's reported filename (display only); ``data``
    is the raw in-memory payload (UploadedFile.getvalue()).
    """
    if len(data) > IMAGE_LIMIT_BYTES:
        size_mb = round(len(data) / 1048576, 1)
        return IMAGE_OVERSIZE_COPY.format(name=name, size_mb=size_mb)
    is_jpeg = data[:3] == b"\xff\xd8\xff"
    is_png = data[:4] == b"\x89\x50\x4e\x47"
    if not (is_jpeg or is_png):
        return IMAGE_WRONG_CONTENT_COPY.format(name=name)
    return None


def decode_image_bgr(data) -> np.ndarray | None:
    """Decode an in-memory image payload to BGR (OpenCV convention), or
    return None when the bytes do not decode (caller renders the exact
    wrong-content copy — the magic check cannot prove decodability)."""
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def typed_error_copy(stage: str, message: str) -> str:
    """Render one UI-SPEC typed-error template keyed by engine-call stage
    ("config" / "integrity" / "input") with the engine message embedded
    verbatim. An unknown stage is a programming error — fail closed naming
    the observed stage (closed-set posture, never a silent fallback)."""
    try:
        template = _TYPED_ERROR_TEMPLATES[stage]
    except KeyError:
        raise ValueError(
            f"typed_error_copy: unknown stage {stage!r} — only "
            f"{sorted(_TYPED_ERROR_TEMPLATES)} are valid (fail-closed, no "
            "silent default)"
        ) from None
    return template.format(message=message)


def arm_angle_rows(raw51, mask_conf) -> list[dict]:
    """Tab 1 arm-angles table rows (8 x {Feature, Value}) derived from the
    classifier's raw51 through the frozen public seam functions ONLY:
    mask_keypoints -> torso_scale (scale element) -> engineered_features
    [51:59] — the display surface re-derives NOTHING with its own formulas
    (UI-SPEC Tab 1 step 6; CONTEXT locked zero-rewrite posture)."""
    kpts = np.asarray(raw51, dtype=np.float64).reshape(17, 3)
    kpts_masked = mask_keypoints(kpts, mask_conf)
    scale, _source = torso_scale(kpts_masked, mask_conf)
    geometry = engineered_features(kpts_masked, scale, mask_conf)[51:59]
    return [
        {"Feature": label, "Value": float(value)}
        for label, value in zip(ARM_ANGLE_LABELS, geometry)
    ]
