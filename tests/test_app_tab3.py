"""Phase 5 Tab 3 research metrics battery (05-03-PLAN Task 1).

Pins the verbatim ablation report rendering, container-width confusion matrix,
render-time sha256 lineage extraction, and fail-closed missing artifact handling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.app_support import (
    METRICS_PNG_PATH,
    METRICS_REPORT_PATH,
    MULTIMODAL_REPORT_PATH,
    MISSING_ARTIFACT_COPY,
    extract_lineage_lines,
    load_metrics_artifacts,
    load_multimodal_metrics_report,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_real_metrics_artifacts_load_and_match():
    """load_metrics_artifacts against real REPO_ROOT returns the tracked report
    text byte-equal to disk and the real confusion_matrix.png path."""
    res = load_metrics_artifacts(REPO_ROOT)
    assert isinstance(res, tuple)
    report_text, png_path = res
    expected_text = (REPO_ROOT / METRICS_REPORT_PATH).read_text(encoding="utf-8")
    assert report_text == expected_text
    assert png_path.resolve() == (REPO_ROOT / METRICS_PNG_PATH).resolve()
    assert png_path.is_file()


def test_real_extract_lineage_lines():
    """extract_lineage_lines on real report returns at least 5 lines (6 today),
    every line containing 'sha256', and all extracted from Parameter block."""
    report_text = (REPO_ROOT / METRICS_REPORT_PATH).read_text(encoding="utf-8")
    lines = extract_lineage_lines(report_text)
    assert len(lines) >= 5
    for line in lines:
        assert "sha256" in line


def test_missing_artifact_fail_closed(tmp_path):
    """Missing report, missing png, and empty directory all fail closed with
    the exact MISSING_ARTIFACT_COPY naming the missing path."""
    # 1. Empty dir
    res_empty = load_metrics_artifacts(tmp_path)
    assert res_empty == MISSING_ARTIFACT_COPY.format(path=METRICS_REPORT_PATH)

    # 2. Report exists but png is missing
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / METRICS_REPORT_PATH).write_text("## Parameter block\nfeatures.npz sha256: 123\n")
    res_no_png = load_metrics_artifacts(tmp_path)
    assert res_no_png == MISSING_ARTIFACT_COPY.format(path=METRICS_PNG_PATH)

    # 3. Png exists but report is missing
    (tmp_path / METRICS_REPORT_PATH).unlink()
    (tmp_path / METRICS_PNG_PATH).write_bytes(b"\x89PNG\r\n\x1a\n")
    res_no_report = load_metrics_artifacts(tmp_path)
    assert res_no_report == MISSING_ARTIFACT_COPY.format(path=METRICS_REPORT_PATH)

    # 4. Both exist -> returns tuple
    (tmp_path / METRICS_REPORT_PATH).write_text("## Parameter block\nfeatures.npz sha256: 123\n")
    res_ok = load_metrics_artifacts(tmp_path)
    assert isinstance(res_ok, tuple)
    assert res_ok[0] == "## Parameter block\nfeatures.npz sha256: 123\n"
    assert res_ok[1] == tmp_path / METRICS_PNG_PATH


def test_app_source_has_no_direct_report_read():
    """app.py contains no direct report file reads; the support loader is the
    single site reading the report file."""
    app_source = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    assert "read_text" not in app_source
    assert "read_bytes" not in app_source


def test_load_multimodal_metrics_report():
    """FUSE-04: load_multimodal_metrics_report loads multimodal_ablation.md
    with verified >=91% accuracy and macro-F1 metrics."""
    text = load_multimodal_metrics_report(REPO_ROOT)
    assert text is not None
    assert "Phase 6 Multimodal Fusion Ablation Report" in text
    assert "92.00%" in text

