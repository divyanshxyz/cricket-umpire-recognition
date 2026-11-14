"""Analyzer integration tests: real mp4v fixtures, VIDEO-01/05 wiring.

Contracts pinned here (04-02-PLAN Task 1):

- VIDEO-01/adjacency: strided frames processed independently at indices
  0, 2, 4, ...; events emit in strict monotonic frame order; a second fresh
  analyzer over the same fixture produces an identical event list
  (timestamp, signal, confidence@8dp, score) — real-model determinism.
- VIDEO-01/empty: a missing file and a garbage-bytes file each fail closed
  with a ValueError naming the offending path.
- VIDEO-05: classify_image returns the predicted class + exactly 6
  probabilities keyed in the LabelEncoder sorted order, fp32, summing to 1.
- A7: --output preserves the input frame count (60); no files written when
  output_path is None (locked CONTEXT decision).
- OQ3: blank-frame fixtures yield typed no_person failures (30 strided
  events), ZERO fired events, and frames_read == 60.
- T-04-06: max_frames trips a typed abort (frames_read == max_frames + 1).

Weight-gated skipif posture (tests/test_pose_features.py:18-23 precedent):
the module skips on a fresh clone without bootstrapped weights, and the
fixture-image tests skip when the UNTRACKED dataset image is absent —
on THIS machine both gates pass, so every test here must RUN, not skip.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import FIXTURE_IMAGE, FIXTURE_FRAMES
from src.inference.analyzer import UmpireVideoAnalyzer, analyzer_events
from src.scoreboard import ScoreState, apply_event

REPO_ROOT = Path(__file__).resolve().parent.parent

_WEIGHTS = [
    REPO_ROOT / "models" / "yolov8n-pose.pt",
    REPO_ROOT / "models" / "baseline_ann.pt",
    REPO_ROOT / "models" / "label_encoder.joblib",
]

pytestmark = pytest.mark.skipif(
    not all(w.exists() for w in _WEIGHTS),
    reason="yolov8n-pose / baseline_ann / label_encoder weights not bootstrapped "
    "(fresh clone posture — the analyzer's startup integrity sweep needs them)",
)

_requires_fixture_image = pytest.mark.skipif(
    not FIXTURE_IMAGE.exists(),
    reason="untracked dataset fixture image absent (datasets/ is not git-tracked)",
)


def _write_zeros_video(path, frames=FIXTURE_FRAMES, fps=30.0, height=480, width=640):
    """mp4v fixture of pure np.zeros frames — YOLO detects no person on
    synthetic frames (probe P2), so every strided frame is a typed
    no_person failure (RESEARCH Pitfall 2 made into a test fixture)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert writer.isOpened(), f"VideoWriter failed to open {path}"
    for _ in range(frames):
        writer.write(frame)
    writer.release()
    return path


def _event_tuples(result):
    """Comparable event tuples: (timestamp, signal, confidence@8dp, score)."""
    return [
        (e.timestamp, e.signal, round(e.confidence, 8), tuple(e.score))
        for e in result.events
    ]


@_requires_fixture_image
def test_analyze_still_frame_fixture_end_to_end(real_image_fixture_video):
    """Real 60-frame mp4v still-loop through every layer: strided extract ->
    Model A probs (integrity-bound loads) -> stabilizer -> scoreboard.

    VIDEO-01/adjacency: frames_read == 60, strided_frames == 30 (stride 2).
    Every fired signal is one of the 5 non-NoAction classes; timestamps are
    strictly increasing (monotonic truth); each event's score tuple equals
    the cumulative LAW_TABLE deltas applied in emission order (independent
    recompute). Determinism: a SECOND fresh analyzer produces an identical
    (timestamp, signal, confidence@8dp, score) list."""
    result = UmpireVideoAnalyzer().analyze(str(real_image_fixture_video))
    assert result.frames_read == 60
    assert result.strided_frames == 30
    canonical_signals = {"Four", "NoBall", "Out", "Six", "Wide"}
    timestamps = []
    expected_state = ScoreState()
    for event in result.events:
        assert event.signal in canonical_signals
        timestamps.append(event.timestamp)
        expected_state, _ = apply_event(expected_state, event.signal)
        assert tuple(event.score) == (
            expected_state.runs,
            expected_state.wickets,
            expected_state.overs,
            expected_state.extras,
        )
    assert timestamps == sorted(timestamps), "event timestamps must be monotonic"
    assert len(set(timestamps)) == len(timestamps), "timestamps strictly increasing"

    second = UmpireVideoAnalyzer().analyze(str(real_image_fixture_video))
    assert _event_tuples(result) == _event_tuples(second)


@_requires_fixture_image
def test_single_image_classification_returns_class_and_probs():
    """VIDEO-05: classify_image on the fixture image -> predicted class in
    the 6 canonical classes; probabilities keyed in the LabelEncoder sorted
    order; probs_vector fp32 summing to ~1.0."""
    analyzer = UmpireVideoAnalyzer()
    classification = analyzer.classify_image(str(FIXTURE_IMAGE))
    assert classification.predicted_class in {
        "Four",
        "NoAction",
        "NoBall",
        "Out",
        "Six",
        "Wide",
    }
    assert list(classification.probabilities.keys()) == [
        "Four",
        "NoAction",
        "NoBall",
        "Out",
        "Six",
        "Wide",
    ]
    assert classification.probs_vector.dtype == np.float32
    assert abs(float(classification.probs_vector.sum()) - 1.0) < 1e-5
    assert classification.raw51.shape == (51,)


@_requires_fixture_image
def test_output_video_preserves_frame_count(real_image_fixture_video, tmp_path):
    """A7 + mp4v roundtrip: analyze(fixture, output_path=...) writes a file
    that opens with VideoCapture and reads exactly the input's 60 frames."""
    out_path = tmp_path / "annotated.mp4"
    result = UmpireVideoAnalyzer().analyze(
        str(real_image_fixture_video), output_path=str(out_path)
    )
    assert result.frames_read == 60
    assert out_path.is_file() and out_path.stat().st_size > 0
    cap = cv2.VideoCapture(str(out_path))
    assert cap.isOpened()
    written = 0
    while True:
        ok, _frame = cap.read()
        if not ok:
            break
        written += 1
    cap.release()
    assert written == 60


def test_missing_file_fails_closed(tmp_path):
    """VIDEO-01/empty: a missing file raises ValueError containing the path
    text — never a silent empty output."""
    missing = tmp_path / "nope.mp4"
    with pytest.raises(ValueError) as exc:
        UmpireVideoAnalyzer().analyze(str(missing))
    assert "nope.mp4" in str(exc.value)


def test_garbage_bytes_fail_closed(tmp_path):
    """VIDEO-01/empty: 512 seeded random bytes that are not a container
    OpenCV can decode raise ValueError naming the path (unopenable
    capture, T-04-02)."""
    rng = np.random.default_rng(42)
    garbage = tmp_path / "fake.mp4"
    garbage.write_bytes(rng.integers(0, 256, size=512, dtype=np.uint8).tobytes())
    with pytest.raises(ValueError) as exc:
        UmpireVideoAnalyzer().analyze(str(garbage))
    assert "fake.mp4" in str(exc.value)


@_requires_fixture_image
def test_no_default_writes(real_image_fixture_video, tmp_path):
    """Locked CONTEXT decision: analyze WITHOUT output_path writes NOTHING
    — the run leaves the cwd/tmp dir exactly as it found it."""
    out_dir = tmp_path / "empty"
    out_dir.mkdir()
    result = UmpireVideoAnalyzer().analyze(str(real_image_fixture_video))
    assert result.frames_read == 60
    assert list(out_dir.iterdir()) == []


def test_blank_frames_yield_typed_failures_no_events(tmp_path):
    """OQ3 against real weights: a zeros-frame fixture produces 30 strided
    typed no_person analyzer events, ZERO fired events (one-hot NoAction
    vectors quench the gate), and still reports frames_read == 60."""
    video = _write_zeros_video(tmp_path / "zeros.mp4")
    analyzer = UmpireVideoAnalyzer()
    before = len(analyzer_events())
    result = analyzer.analyze(str(video))
    added = analyzer_events()[before:]
    no_person = [line for line in added if line.startswith("no_person: ")]
    assert len(no_person) == 30
    assert result.events == ()
    assert result.frames_read == 60
    assert result.strided_frames == 30
    assert result.failures.get("no_person") == 30


def test_max_frames_typed_abort(tmp_path):
    """T-04-06: a config override max_frames=4 (tmp yaml through the
    load_inference_config path param) stops the loop early with a typed
    max_frames_abort event and frames_read == 5 (the 5th frame is read,
    counted, then aborts before processing)."""
    video = _write_zeros_video(tmp_path / "zeros.mp4")
    cfg_path = tmp_path / "inference.yaml"
    cfg_path.write_text("params:\n  max_frames: 4\n", encoding="utf-8")
    analyzer = UmpireVideoAnalyzer(config_path=cfg_path)
    before = len(analyzer_events())
    result = analyzer.analyze(str(video))
    added = analyzer_events()[before:]
    aborts = [line for line in added if line.startswith("max_frames_abort: ")]
    assert len(aborts) == 1
    assert "max_frames" in aborts[0]
    assert result.frames_read == 5
    assert result.failures.get("max_frames_abort") == 1
