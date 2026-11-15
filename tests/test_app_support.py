"""Phase 5 app_support hardening battery — 05-01-PLAN Task 2 Step 1.

Pure, weight-free pins on the streamlit-free presentation-support module:
the upload-validation boundary (exact UI-SPEC copies, size-before-content
order), the three typed-error templates (byte-exact with the engine
message embedded verbatim), and the arm-angles display mapping through the
frozen seam functions. No model weights, no fixture image, no streamlit.
"""

import numpy as np
import pytest

from src.app_support import (
    ARM_ANGLE_LABELS,
    IMAGE_LIMIT_BYTES,
    IMAGE_OVERSIZE_COPY,
    IMAGE_WRONG_CONTENT_COPY,
    VIDEO_LIMIT_BYTES,
    arm_angle_rows,
    check_image_bytes,
    typed_error_copy,
)

# The exact UI-SPEC toast strings (copywriting contract), stated here as the
# pinned expectation — the app_support constants must match BYTE FOR BYTE.
_OVERSIZE_EXACT = (
    "Image too large — 'clip.gif' is 10.0 MB. The limit is 10 MB — please "
    "upload a smaller image."
)
_WRONG_CONTENT_EXACT = (
    "Unsupported image — 'clip.gif' is not a readable JPG/PNG (content check "
    "failed). Please upload a .jpg or .png image; the file type is verified "
    "by content, not extension."
)


# ----------------------------------------------------------------------
# Validation boundary battery
# ----------------------------------------------------------------------


def test_limit_constants_are_the_locked_ui_spec_limits():
    """Locked limits (UI-SPEC Upload Validation table): image <= 10 MB,
    video <= 200 MB, enforced on the in-memory bytes."""
    assert IMAGE_LIMIT_BYTES == 10 * 1024 * 1024
    assert VIDEO_LIMIT_BYTES == 200 * 1024 * 1024


def test_exact_limit_buffer_with_valid_magic_passes():
    """Exactly 10 MiB with a valid PNG magic passes the whole gate (size
    <= limit, content matches) — check_image_bytes returns None."""
    data = b"\x89\x50\x4e\x47" + b"\x00" * (IMAGE_LIMIT_BYTES - 4)
    assert len(data) == IMAGE_LIMIT_BYTES
    assert check_image_bytes("edge.png", data) is None


def test_one_byte_over_limit_returns_exact_oversize_copy():
    """10 MiB + 1 byte returns the EXACT UI-SPEC oversize copy with
    size_mb = round(len/1048576, 1) rendered into the slot."""
    data = b"\xff\xd8\xff" + b"\x00" * (IMAGE_LIMIT_BYTES - 2)  # 3 + (L-2) = L+1
    assert len(data) == IMAGE_LIMIT_BYTES + 1
    result = check_image_bytes("clip.gif", data)
    assert result == _OVERSIZE_EXACT
    assert result == IMAGE_OVERSIZE_COPY.format(name="clip.gif", size_mb=10.0)


def test_gif_header_returns_exact_wrong_content_copy():
    """A GIF89a header fails the magic sniff — the EXACT UI-SPEC
    wrong-content copy (content is verified, never the extension)."""
    data = b"GIF89a" + b"\x00" * 1024
    assert check_image_bytes("clip.gif", data) == _WRONG_CONTENT_EXACT


def test_oversize_combined_with_wrong_magic_returns_oversize_copy():
    """Oversize AND wrong magic: the OVERSIZE copy wins — size is checked
    BEFORE content (UI-SPEC sniff order), so the user fixes the size
    first and only then learns the content is wrong."""
    data = b"GIF89a" + b"\x00" * IMAGE_LIMIT_BYTES  # wrong magic + over limit
    assert len(data) == IMAGE_LIMIT_BYTES + 6 > IMAGE_LIMIT_BYTES
    result = check_image_bytes("clip.gif", data)
    assert result == IMAGE_OVERSIZE_COPY.format(name="clip.gif", size_mb=10.0)
    assert result != IMAGE_WRONG_CONTENT_COPY.format(name="clip.gif")


# ----------------------------------------------------------------------
# Typed-error template pins (byte-exact, engine message verbatim)
# ----------------------------------------------------------------------


def test_typed_error_templates_are_byte_exact():
    """The three UI-SPEC Typed-Error templates with a probe engine message
    embedded verbatim — byte-equal to the copy table (no paraphrase, no
    truncation, single-space joins exactly as the spec writes them)."""
    probe = "configs/inference.yaml: threshold must be in (0.0, 1.0], got -1"
    assert typed_error_copy("config", probe) == (
        f"Configuration error — {probe} Fix configs/inference.yaml and "
        "restart the app."
    )
    probe_integrity = (
        "startup integrity sweep: artifact models/pose_mlp.pt does not exist"
    )
    assert typed_error_copy("integrity", probe_integrity) == (
        f"Model integrity failure — {probe_integrity} The artifacts under "
        "models/ do not match configs/model_digests.yaml — re-verify or "
        "re-download the artifacts, then restart."
    )
    probe_input = (
        "classify_image: extraction failed (no_person: no person detected)"
    )
    assert typed_error_copy("input", probe_input) == (
        f"Upload could not be processed — {probe_input}"
    )


def test_typed_error_copy_preserves_multiline_engine_message_verbatim():
    """Long/multiline engine messages embed VERBATIM (the message is the
    user's diagnostic record — never truncated or ellipsized)."""
    probe = "line one\nline two with a sha256 42b9c5cc… and a long tail"
    rendered = typed_error_copy("input", probe)
    assert probe in rendered
    assert rendered == f"Upload could not be processed — {probe}"


def test_typed_error_copy_unknown_stage_fails_closed():
    """An unknown stage is a programming error — ValueError naming the
    observed stage (closed-set posture, never a silent fallback)."""
    with pytest.raises(ValueError, match="unknown stage 'video'"):
        typed_error_copy("video", "whatever")


# ----------------------------------------------------------------------
# arm_angle_rows label pin (frozen seam mapping, weight-free)
# ----------------------------------------------------------------------


def _synthetic_pose() -> np.ndarray:
    """A plausible standing umpire (17, 3) float64 in COCO order: torso,
    raised-ish arms, legs; all conf 0.9 except the right ankle (0.1) so the
    masking path (conf < mask_conf) is exercised without NaNs."""
    coords = {
        0: (150, 60),    # nose
        1: (140, 50),    # L eye
        2: (160, 50),    # R eye
        3: (135, 60),    # L ear
        4: (165, 60),    # R ear
        5: (110, 100),   # L shoulder
        6: (190, 100),   # R shoulder
        7: (90, 150),    # L elbow
        8: (210, 150),   # R elbow
        9: (100, 200),   # L wrist
        10: (200, 200),  # R wrist
        11: (125, 250),  # L hip
        12: (175, 250),  # R hip
        13: (120, 350),  # L knee
        14: (180, 350),  # R knee
        15: (115, 450),  # L ankle
        16: (185, 450),  # R ankle (masked below)
    }
    kpts = np.zeros((17, 3), dtype=np.float64)
    for idx, (x, y) in coords.items():
        kpts[idx] = (x, y, 0.9)
    kpts[16, 2] = 0.1  # below mask_conf 0.20 — exercises the masking path
    return kpts


def test_arm_angle_rows_frozen_labels_in_frozen_order():
    """A synthetic pose through the frozen seam functions yields exactly 8
    rows with the frozen labels in the frozen order, finite float values —
    arm_angle_rows re-derives NO geometry of its own."""
    rows = arm_angle_rows(_synthetic_pose().reshape(-1), 0.20)
    assert len(rows) == 8
    assert [row["Feature"] for row in rows] == list(ARM_ANGLE_LABELS)
    for row in rows:
        assert isinstance(row["Value"], float)
        assert np.isfinite(row["Value"])


def test_arm_angle_rows_masked_keypoints_stay_finite():
    """A pose with EVERY keypoint below mask_conf still yields 8 finite
    rows (the seam functions' 0.0 sentinels — never NaN) — the display
    surface cannot crash on a degenerate classification result."""
    kpts = np.zeros((17, 3), dtype=np.float64)
    kpts[:, 2] = 0.05  # everything masked
    rows = arm_angle_rows(kpts.reshape(-1), 0.20)
    assert len(rows) == 8
    assert all(np.isfinite(row["Value"]) for row in rows)
