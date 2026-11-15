"""Phase 5 mechanical prohibition battery (05-03-PLAN Task 2).

Wires the four phase prohibitions to mechanical source/AST checks, each with
a planted-violation self-assertion proving detection works before scanning
the live codebase:
- P1: Thin-app boundary (no inference logic in app.py)
- P2: No un-sniffed upload reaches the engine
- P3: No raw-traceback rendering
- P4: No second CSV serializer
"""

from __future__ import annotations

import ast
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = REPO_ROOT / "app.py"
SUPPORT_PATH = REPO_ROOT / "src" / "app_support.py"


# ----------------------------------------------------------------------
# P1: Thin-app boundary (no inference logic in app.py)
# ----------------------------------------------------------------------

SANCTIONED_IMPORTS = {
    "src.inference.analyzer": {"UmpireVideoAnalyzer", "ImageClassification"},
    "src.inference.config": {"load_inference_config", "INFERENCE_DEFAULTS"},
    "src.scoreboard": {"EVENT_CSV_COLUMNS", "UNTRAINED_SMOKE_MARKER", "EventLogEntry"},
    "src.app_support": {
        "IMAGE_WRONG_CONTENT_COPY",
        "PROGRESS_UNKNOWN_CAPTION",
        "arm_angle_rows",
        "build_event_csv_bytes",
        "check_image_bytes",
        "compute_progress",
        "decode_image_bgr",
        "event_log_rows",
        "execute_video_job",
        "extract_lineage_lines",
        "format_failure_tally",
        "load_metrics_artifacts",
        "load_multimodal_metrics_report",
        "typed_error_copy",
    },
}

FORBIDDEN_IDENTIFIERS = {"apply_event", "LAW_TABLE"}


def check_thin_app_boundary(source: str) -> None:
    tree = ast.parse(source)

    for node in ast.walk(tree):
        # Direct imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src.") or alias.name == "src":
                    raise AssertionError(f"Forbidden direct src import: {alias.name}")

        # From imports
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith("src.") or mod == "src":
                if mod not in SANCTIONED_IMPORTS:
                    raise AssertionError(f"Forbidden module import from src: {mod}")
                allowed = SANCTIONED_IMPORTS[mod]
                for alias in node.names:
                    if alias.name not in allowed:
                        raise AssertionError(
                            f"Forbidden symbol imported from {mod}: {alias.name}"
                        )

    for name in FORBIDDEN_IDENTIFIERS:
        if name in source:
            raise AssertionError(f"Forbidden identifier referenced in source: {name}")


def test_p1_planted_violation_self_assertion():
    """P1 self-assertion: fails on planted violations."""
    # 1. Unsanctioned module import
    bad_mod = "from src.inference.stabilizer import TemporalStabilizer\nx = 1\n"
    with pytest.raises(AssertionError, match="Forbidden module import"):
        check_thin_app_boundary(bad_mod)

    # 2. Unsanctioned symbol import
    bad_sym = "from src.inference.analyzer import TemporalStabilizer\nx = 1\n"
    with pytest.raises(AssertionError, match="Forbidden symbol imported"):
        check_thin_app_boundary(bad_sym)

    # 3. Direct src import
    bad_dir = "import src.inference\nx = 1\n"
    with pytest.raises(AssertionError, match="Forbidden direct src import"):
        check_thin_app_boundary(bad_dir)

    # 4. Forbidden identifier reference
    bad_ident = "def f(): return apply_event(1, 2)\n"
    with pytest.raises(AssertionError, match="Forbidden identifier referenced"):
        check_thin_app_boundary(bad_ident)


def test_p1_thin_app_boundary_live():
    """P1 live check: app.py strictly obeys the thin-app boundary."""
    source = APP_PATH.read_text(encoding="utf-8")
    check_thin_app_boundary(source)


# ----------------------------------------------------------------------
# P2: No un-sniffed upload reaches the engine
# ----------------------------------------------------------------------


def check_upload_sniff_ordering(app_source: str, support_source: str) -> None:
    # Image path: check_image_bytes must precede classify_and_annotate in app.py
    idx_img_check = app_source.find("check_image_bytes")
    idx_img_engine = app_source.find("classify_and_annotate")
    if idx_img_check == -1 or idx_img_engine == -1 or idx_img_check > idx_img_engine:
        raise AssertionError("Image sniff does not precede engine classification")

    # Video path: check_video_bytes and probe_video_decode must precede analyze_stream in app_support.py
    idx_vid_check = support_source.find("check_video_bytes")
    idx_vid_probe = support_source.find("probe_video_decode")
    idx_vid_stream = support_source.find("analyze_stream")
    if idx_vid_stream == -1 or idx_vid_check == -1 or idx_vid_probe == -1:
        raise AssertionError("Video validation or streaming missing from support module")
    if not (idx_vid_check < idx_vid_probe < idx_vid_stream):
        raise AssertionError("Video validation/probe chain does not precede engine streaming")


def test_p2_planted_violation_self_assertion():
    """P2 self-assertion: fails if sniff order is reversed or missing."""
    reversed_img = """
def render_image_tab():
    classify_and_annotate(data)
    check_image_bytes(name, data)
"""
    valid_support = """
check_video_bytes()
probe_video_decode()
analyzer.analyze_stream()
"""
    with pytest.raises(AssertionError, match="Image sniff does not precede"):
        check_upload_sniff_ordering(reversed_img, valid_support)

    valid_img = """
def render_image_tab():
    check_image_bytes(name, data)
    classify_and_annotate(data)
"""
    reversed_support = """
analyzer.analyze_stream()
check_video_bytes()
probe_video_decode()
"""
    with pytest.raises(AssertionError, match="Video validation/probe chain does not precede"):
        check_upload_sniff_ordering(valid_img, reversed_support)


def test_p2_upload_sniff_ordering_live():
    """P2 live check: image and video upload paths enforce size/sniff before engine calls."""
    app_source = APP_PATH.read_text(encoding="utf-8")
    support_source = SUPPORT_PATH.read_text(encoding="utf-8")
    check_upload_sniff_ordering(app_source, support_source)


# ----------------------------------------------------------------------
# P3: No raw-traceback rendering
# ----------------------------------------------------------------------


def check_no_raw_tracebacks(source: str) -> None:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "traceback":
                    raise AssertionError("traceback module imported")
        if isinstance(node, ast.ImportFrom):
            if node.module == "traceback":
                raise AssertionError("traceback module imported")
        if isinstance(node, ast.Attribute):
            if node.attr == "exception":
                raise AssertionError("st.exception called")


def test_p3_planted_violation_self_assertion():
    """P3 self-assertion: catches traceback imports or st.exception calls."""
    bad_tb = "import traceback\nx = 1\n"
    with pytest.raises(AssertionError, match="traceback module imported"):
        check_no_raw_tracebacks(bad_tb)

    bad_exc = "import streamlit as st\nst.exception(ValueError())\n"
    with pytest.raises(AssertionError, match="st.exception called"):
        check_no_raw_tracebacks(bad_exc)


def test_p3_no_raw_tracebacks_live():
    """P3 live check: app.py renders no raw tracebacks or st.exception."""
    source = APP_PATH.read_text(encoding="utf-8")
    check_no_raw_tracebacks(source)


# ----------------------------------------------------------------------
# P4: No second CSV serializer
# ----------------------------------------------------------------------


def check_single_csv_serializer(app_src: str, supp_src: str) -> None:
    if "import csv" in app_src or "from csv" in app_src:
        raise AssertionError("app.py imports csv")
    if supp_src.count("csv.writer") != 1:
        raise AssertionError(
            f"src/app_support.py must have exactly 1 csv.writer, found {supp_src.count('csv.writer')}"
        )


def test_p4_planted_violation_self_assertion():
    """P4 self-assertion: catches csv import in app.py or duplicate writer in support."""
    with pytest.raises(AssertionError, match="app.py imports csv"):
        check_single_csv_serializer("import csv\n", "writer = csv.writer(f)\n")

    with pytest.raises(AssertionError, match="must have exactly 1 csv.writer"):
        check_single_csv_serializer("x = 1\n", "csv.writer(f)\ncsv.writer(f2)\n")


def test_p4_single_csv_serializer_live():
    """P4 live check: exactly one csv.writer in src/app_support.py, none in app.py."""
    app_src = APP_PATH.read_text(encoding="utf-8")
    supp_src = SUPPORT_PATH.read_text(encoding="utf-8")
    check_single_csv_serializer(app_src, supp_src)
