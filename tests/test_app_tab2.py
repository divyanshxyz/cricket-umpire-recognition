"""Phase 5 Tab 2 video analyzer & hardening battery (05-02-PLAN Tasks 1-2).

Pins the video upload validation boundary, temporary file cleanup hygiene,
live stream loop execution, scoreboard continuation across runs, honest
max_frames abort handling, zero-event honesty, and the 7-column CSV serializer.
"""

from __future__ import annotations

import csv
import io
import tempfile
import uuid
from pathlib import Path

import pytest

from src.app_support import (
    PROGRESS_UNKNOWN_CAPTION,
    VIDEO_LIMIT_BYTES,
    VIDEO_OVERSIZE_COPY,
    VIDEO_WRONG_CONTENT_COPY,
    build_event_csv_bytes,
    check_video_bytes,
    compute_progress,
    event_log_rows,
    execute_video_job,
    format_failure_tally,
    probe_video_decode,
    write_video_tempfile,
)
from src.scoreboard import (
    EVENT_CSV_COLUMNS,
    EventLogEntry,
    ScoreState,
    apply_event,
)
from tests.conftest import FIXTURE_IMAGE


# ----------------------------------------------------------------------
# Weight-free validation and helper tests
# ----------------------------------------------------------------------


def test_check_video_bytes_boundary():
    """Exactly 200 MiB passes; 200 MiB + 1 byte returns exact oversize copy."""
    exact_limit_data = b"x" * VIDEO_LIMIT_BYTES
    assert check_video_bytes("clip.mp4", exact_limit_data) is None

    oversize_data = b"x" * (VIDEO_LIMIT_BYTES + 1)
    expected_msg = VIDEO_OVERSIZE_COPY.format(name="clip.mp4", size_mb=200.0)
    assert check_video_bytes("clip.mp4", oversize_data) == expected_msg


def test_build_event_csv_bytes_and_roundtrip():
    """CSV serializer emits list(EVENT_CSV_COLUMNS) as header, entry.to_row()
    values verbatim, and round-trips through csv.reader."""
    entry1 = EventLogEntry(
        timestamp="00:02",
        signal="Four",
        confidence=0.8765,
        score=(4, 0, "0.1", 0),
        smoke_marker="",
    )
    entry2 = EventLogEntry(
        timestamp="00:05",
        signal="Wide",
        confidence=0.9123,
        score=(5, 0, "0.1", 1),
        smoke_marker="UNTRAINED — smoke only",
    )
    csv_bytes = build_event_csv_bytes([entry1, entry2])
    text = csv_bytes.decode("utf-8")
    reader = list(csv.reader(io.StringIO(text)))

    assert len(reader) == 3
    assert reader[0] == list(EVENT_CSV_COLUMNS)
    assert reader[1] == ["00:02", "Four", "0.8765", "4", "0", "0.1", "0"]
    assert reader[2] == ["00:05", "Wide", "0.9123", "5", "0", "0.1", "1"]


def test_build_event_csv_bytes_zero_events_is_header_only():
    """Zero events emit an honest 7-column header-only CSV."""
    csv_bytes = build_event_csv_bytes([])
    text = csv_bytes.decode("utf-8")
    reader = list(csv.reader(io.StringIO(text)))
    assert len(reader) == 1
    assert reader[0] == list(EVENT_CSV_COLUMNS)


def test_event_log_rows_and_failure_tally():
    """event_log_rows renders confidence at 4 decimals and appends smoke_marker
    as 8th display column; format_failure_tally formats sorted or 'none'."""
    entry = EventLogEntry(
        timestamp="01:23",
        signal="Six",
        confidence=0.987654321,
        score=(6, 0, "0.1", 0),
        smoke_marker="smoke",
    )
    rows = event_log_rows([entry])
    assert len(rows) == 1
    assert rows[0] == ["01:23", "Six", "0.9877", 6, 0, "0.1", 0, "smoke"]

    assert format_failure_tally({}) == "none"
    assert (
        format_failure_tally({"no_person": 2, "max_frames_abort": 1})
        == "max_frames_abort=1  no_person=2"
    )


def test_progress_helpers_and_caption():
    """PROGRESS_UNKNOWN_CAPTION is exact UI-SPEC copy; compute_progress clamps <= 1.0
    and falls back to max_frames guard when estimate is None or <= 0."""
    assert (
        PROGRESS_UNKNOWN_CAPTION
        == "Video length unknown — progress measured against the max_frames guard."
    )
    # Valid estimate
    assert compute_progress(30, 60, 1000) == 0.5
    assert compute_progress(70, 60, 1000) == 1.0  # clamped

    # Unknown estimate -> fallback to max_frames
    assert compute_progress(50, None, 100) == 0.5
    assert compute_progress(50, 0, 100) == 0.5
    assert compute_progress(150, 0, 100) == 1.0  # clamped


# ----------------------------------------------------------------------
# Stub analyzer & tempfile hygiene tests
# ----------------------------------------------------------------------


class _RecordingStubAnalyzer:
    """Stub analyzer tracking method calls and config parameters."""

    def __init__(self, cfg=None):
        self.cfg = cfg or {"stride": 2, "max_frames": 100}
        self.analyze_stream_calls = []

    def analyze_stream(self, path):
        self.analyze_stream_calls.append(path)
        return iter([])


def test_execute_video_job_oversize_no_tempfile(monkeypatch):
    """An oversize upload returns the oversize toast before creating any tempfile,
    leaving the engine completely untouched."""
    stub = _RecordingStubAnalyzer()
    oversize = b"x" * (VIDEO_LIMIT_BYTES + 10)

    def _fail_on_tempfile(*args, **kwargs):
        raise AssertionError("write_video_tempfile was reached for oversize upload")

    monkeypatch.setattr("src.app_support.write_video_tempfile", _fail_on_tempfile)
    result = execute_video_job(stub, oversize, name="big_clip.mp4")
    assert result == VIDEO_OVERSIZE_COPY.format(name="big_clip.mp4", size_mb=200.0)
    assert stub.analyze_stream_calls == []


def test_execute_video_job_garbage_bytes_decode_fails_clean_tempfile():
    """Non-video bytes pass size check but fail decode probe: returns exact
    wrong-content copy, engine is untouched, and temp file is unlinked."""
    stub = _RecordingStubAnalyzer()
    garbage = b"THIS IS NOT A VALID VIDEO FILE " * 50

    temp_dir = Path(tempfile.gettempdir())
    mp4_before = set(temp_dir.glob("tmp*.mp4"))

    result = execute_video_job(stub, garbage, name="garbage.mp4")
    assert result == VIDEO_WRONG_CONTENT_COPY.format(name="garbage.mp4")
    assert stub.analyze_stream_calls == []

    mp4_after = set(temp_dir.glob("tmp*.mp4"))
    assert mp4_after == mp4_before, "Temporary mp4 file was leaked on decode failure"


def test_execute_video_job_engine_raise_cleans_tempfile():
    """Engine raising mid-iteration still guarantees tempfile unlink in finally."""
    temp_dir = Path(tempfile.gettempdir())

    class _ExplodingAnalyzer:
        cfg = {"stride": 1, "max_frames": 10}

        def analyze_stream(self, path):
            assert Path(path).exists(), "Temporary file must exist during analyze_stream"
            raise RuntimeError("Engine failed mid-stream")

    # Create valid video header bytes using write_video_tempfile + cv2 or dummy mp4
    import numpy as np
    import cv2

    dummy_path = temp_dir / f"probe_{uuid.uuid4().hex}.mp4"
    writer = cv2.VideoWriter(
        str(dummy_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 64)
    )
    writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    valid_video_bytes = dummy_path.read_bytes()
    dummy_path.unlink()

    mp4_before = set(temp_dir.glob("tmp*.mp4"))
    with pytest.raises(RuntimeError, match="Engine failed mid-stream"):
        execute_video_job(_ExplodingAnalyzer(), valid_video_bytes)

    mp4_after = set(temp_dir.glob("tmp*.mp4"))
    assert mp4_after == mp4_before, "Temporary mp4 file was leaked when engine raised"


# ----------------------------------------------------------------------
# Weight-gated / Fixture tests
# ----------------------------------------------------------------------


@pytest.mark.skipif(not FIXTURE_IMAGE.exists(), reason="fixture image absent")
def test_probe_video_decode_on_fixture_video(real_image_fixture_video):
    """probe_video_decode returns (True, fps > 0, estimate > 0) on the fixture video."""
    ok, fps, estimate = probe_video_decode(real_image_fixture_video)
    assert ok is True
    assert fps == 30.0
    assert estimate == 60


@pytest.mark.skipif(not FIXTURE_IMAGE.exists(), reason="fixture image absent")
def test_execute_video_job_fixture_happy_path_and_continuation(
    real_image_fixture_video,
):
    """execute_video_job on fixture video:
    - frames_read == 60, strided == 30 (stride 2), aborted == False
    - fresh-session idempotency: running with another fresh analyzer produces identical events
    - scoreboard continuation: running SAME analyzer again continues the scoreboard
    """
    from src.inference.analyzer import UmpireVideoAnalyzer

    video_bytes = real_image_fixture_video.read_bytes()

    analyzer1 = UmpireVideoAnalyzer()
    job1 = execute_video_job(analyzer1, video_bytes)
    assert isinstance(job1, dict)
    assert job1["frames_read"] == 60
    assert job1["strided"] == 30
    assert job1["aborted"] is False
    assert job1["fps_reported"] == 30.0
    assert job1["estimate"] == 60
    assert job1["last_frame"] is not None

    # Idempotency check with a fresh analyzer
    analyzer2 = UmpireVideoAnalyzer()
    job2 = execute_video_job(analyzer2, video_bytes)
    assert isinstance(job2, dict)
    assert job1["events"] == job2["events"]

    # Continuation check with the SAME analyzer
    first_score = analyzer1.score_state
    job1_cont = execute_video_job(analyzer1, video_bytes)
    second_score = analyzer1.score_state

    # Verify score progressed
    if job1_cont["events"]:
        expected_score = first_score
        for ev in job1_cont["events"]:
            expected_score, _ = apply_event(expected_score, ev.signal)
        assert second_score == expected_score


@pytest.mark.skipif(not FIXTURE_IMAGE.exists(), reason="fixture image absent")
def test_execute_video_job_max_frames_abort(real_image_fixture_video):
    """A config with max_frames=5 aborts at frame 5 with aborted=True and
    max_frames_abort in the failure tally."""
    from src.inference.analyzer import UmpireVideoAnalyzer

    video_bytes = real_image_fixture_video.read_bytes()
    analyzer = UmpireVideoAnalyzer()
    analyzer.cfg["max_frames"] = 5

    job = execute_video_job(analyzer, video_bytes)
    assert isinstance(job, dict)
    assert job["aborted"] is True
    assert job["frames_read"] == 5
    assert "max_frames_abort" in job["tally"]


@pytest.mark.skipif(not FIXTURE_IMAGE.exists(), reason="fixture image absent")
def test_execute_video_job_zero_events_honesty(real_image_fixture_video):
    """A config with threshold=1.0 on the still-frame fixture yields zero events:
    header-only CSV, empty tally, honest zero events."""
    from src.inference.analyzer import UmpireVideoAnalyzer

    video_bytes = real_image_fixture_video.read_bytes()
    analyzer = UmpireVideoAnalyzer()
    analyzer.cfg["threshold"] = 1.0

    job = execute_video_job(analyzer, video_bytes)
    assert isinstance(job, dict)
    assert job["events"] == ()
    csv_bytes = build_event_csv_bytes(job["events"])
    lines = csv_bytes.decode("utf-8").strip().splitlines()
    assert len(lines) == 1
    assert lines[0] == ",".join(EVENT_CSV_COLUMNS)
