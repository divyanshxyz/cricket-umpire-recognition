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

import csv
import io
import tempfile
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from src.pose_features import engineered_features, mask_keypoints, torso_scale
from src.scoreboard import EVENT_CSV_COLUMNS, EventLogEntry

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
VIDEO_OVERSIZE_COPY = (
    "Video too large — '{name}' is {size_mb} MB. The limit is 200 MB — "
    "please upload a shorter or more compressed clip."
)
VIDEO_WRONG_CONTENT_COPY = (
    "Unsupported video — '{name}' did not decode as a video OpenCV can read. "
    "Please upload an MP4 clip (content is verified by decode, not extension)."
)
PROGRESS_UNKNOWN_CAPTION = (
    "Video length unknown — progress measured against the max_frames guard."
)

# Tab 3 tracked artifact paths and missing-artifact copy (Plan 05-03, UI-04)
METRICS_REPORT_PATH = "reports/ablation_study.md"
METRICS_PNG_PATH = "reports/confusion_matrix.png"
MISSING_ARTIFACT_COPY = (
    "Report artifact not found — {path} is missing from disk. Restore the "
    "tracked Phase 3 artifact and reload."
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


# ----------------------------------------------------------------------
# Video-half helpers and CSV serializer (Plan 05-02)
# ----------------------------------------------------------------------


def check_video_bytes(name: str, data: bytes) -> str | None:
    """Upload size gate for videos (T-05-03): size check FIRST on the
    in-memory buffer length before any temporary file write or decode.
    Returns the exact UI-SPEC toast copy on oversize, or None when within limit.
    """
    if len(data) > VIDEO_LIMIT_BYTES:
        size_mb = round(len(data) / 1048576, 1)
        return VIDEO_OVERSIZE_COPY.format(name=name, size_mb=size_mb)
    return None


def write_video_tempfile(data: bytes) -> Path:
    """Write raw video bytes to an ephemeral temp file in the OS temp directory.
    delete=False is REQUIRED on Windows so cv2 can open it. The caller must
    ensure the temporary file is unlinked.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(data)
        tmp.flush()
        return Path(tmp.name)


def probe_video_decode(path: Path | str) -> tuple[bool, float, int | None]:
    """Decode probe via OpenCV: verifies file opens and first frame can be grabbed.
    Returns (ok, fps_raw, estimate_frame_count). estimate is None when unknown or <= 0.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False, 0.0, None
        grabbed = cap.grab()
        if not grabbed:
            return False, 0.0, None
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        raw_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        estimate = raw_count if raw_count > 0 else None
        return True, fps, estimate
    finally:
        cap.release()


def build_event_csv_bytes(events: Sequence[EventLogEntry]) -> bytes:
    """Build the single-source Phase 5 event log CSV bytes (UI-03 contract).
    Header is list(EVENT_CSV_COLUMNS), followed by entry.to_row() for each event.
    """
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(list(EVENT_CSV_COLUMNS))
    for entry in events:
        writer.writerow(entry.to_row())
    return out.getvalue().encode("utf-8")


def event_log_rows(events: Sequence[EventLogEntry]) -> list[list]:
    """Format events for Tab 2 UI display: the 7 to_row() values under
    EVENT_CSV_COLUMNS with confidence rendered to 4 decimals, plus
    smoke_marker as the 8th display column.
    """
    rows = []
    for entry in events:
        row = list(entry.to_row())
        row[2] = f"{entry.confidence:.4f}"
        row.append(entry.smoke_marker)
        rows.append(row)
    return rows


def format_failure_tally(tally: dict[str, int]) -> str:
    """Format failure tally as sorted 'label=count' pairs joined with two spaces,
    or 'none' when empty (CLI wording).
    """
    if not tally:
        return "none"
    return "  ".join(f"{label}={count}" for label, count in sorted(tally.items()))


def compute_progress(frames_read: int, estimate: int | None, max_frames: int) -> float:
    """Calculate progress fraction clamped to [0.0, 1.0]. Uses estimate if valid,
    else falls back to max_frames guard.
    """
    denom = estimate if (estimate is not None and estimate > 0) else max_frames
    if denom <= 0:
        return 1.0
    return min(max(frames_read / denom, 0.0), 1.0)


def execute_video_job(
    analyzer,
    video_bytes: bytes,
    on_progress=None,
    on_frame=None,
    on_event=None,
    name: str = "upload",
) -> dict | str:
    """Execute video analysis job with size check, decode probe, and guaranteed tempfile cleanup.
    Returns error toast string on validation failure, or dict with job results on success.
    """
    toast = check_video_bytes(name, video_bytes)
    if toast is not None:
        return toast

    tmp_path = write_video_tempfile(video_bytes)
    try:
        ok, fps, estimate = probe_video_decode(tmp_path)
        if not ok:
            return VIDEO_WRONG_CONTENT_COPY.format(name=name)

        from src.inference.analyzer import analyzer_events
        from src.inference.stabilizer import stabilizer_events

        events_baseline = len(analyzer_events())
        stabilizer_events_baseline = len(stabilizer_events())

        events = []
        last_strided_frame = None
        strided_count = 0
        frames_read = 0

        stride = analyzer.cfg.get("stride", 1)

        for record in analyzer.analyze_stream(tmp_path):
            frames_read = record.frame_idx + 1
            if on_progress is not None:
                try:
                    on_progress(frames_read, estimate)
                except TypeError:
                    on_progress(frames_read)

            if record.frame_idx % stride == 0:
                strided_count += 1
                last_strided_frame = record.annotated
                if on_frame is not None:
                    on_frame(record.annotated)

            if record.new_events:
                for ev in record.new_events:
                    events.append(ev)
                    if on_event is not None:
                        on_event(ev)

        tally: dict[str, int] = {}
        for line in analyzer_events()[events_baseline:]:
            label = line.split(": ", 1)[0]
            tally[label] = tally.get(label, 0) + 1
        for line in stabilizer_events()[stabilizer_events_baseline:]:
            label = line.split(": ", 1)[0]
            tally[label] = tally.get(label, 0) + 1

        return {
            "events": tuple(events),
            "last_frame": last_strided_frame,
            "frames_read": frames_read,
            "strided": strided_count,
            "fps_reported": fps,
            "estimate": estimate,
            "aborted": "max_frames_abort" in tally,
            "tally": tally,
        }
    finally:
        tmp_path.unlink(missing_ok=True)


# ----------------------------------------------------------------------
# Metrics helpers (Plan 05-03, UI-04)
# ----------------------------------------------------------------------


def load_metrics_artifacts(
    reports_root: Path | str,
) -> tuple[str, Path] | str:
    """Load Tab 3 research metrics artifacts from disk (UI-04 contract).
    Returns (report_text, png_path) on success, or the exact MISSING_ARTIFACT_COPY
    toast string naming the missing path if either artifact is missing.
    Fail-closed: Tab 3 never renders partially with a silent gap.
    """
    root = Path(reports_root)
    # Check if root is repo root or direct parent of report file
    if (root / METRICS_REPORT_PATH).is_file():
        report_path = root / METRICS_REPORT_PATH
    elif (root / Path(METRICS_REPORT_PATH).name).is_file():
        report_path = root / Path(METRICS_REPORT_PATH).name
    else:
        return MISSING_ARTIFACT_COPY.format(path=METRICS_REPORT_PATH)

    if (root / METRICS_PNG_PATH).is_file():
        png_path = root / METRICS_PNG_PATH
    elif (root / Path(METRICS_PNG_PATH).name).is_file():
        png_path = root / Path(METRICS_PNG_PATH).name
    else:
        return MISSING_ARTIFACT_COPY.format(path=METRICS_PNG_PATH)

    return report_path.read_text(encoding="utf-8"), png_path


def extract_lineage_lines(report_text: str) -> list[str]:
    """Extract sha256-bearing lineage lines from the '## Parameter block' section
    of reports/ablation_study.md at render time (never hardcoded)."""
    lines = report_text.splitlines()
    in_block = False
    extracted = []
    for line in lines:
        if line.startswith("## Parameter block"):
            in_block = True
            continue
        if in_block:
            if line.startswith("## "):
                break
            if "sha256" in line:
                extracted.append(line.strip())
    return extracted
