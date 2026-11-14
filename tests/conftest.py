"""Shared Phase 4 test fixtures: the real-image mp4v fixture-video builder.

tests/conftest.py is created FRESH for Plan 04-02 (the repo root conftest.py
is a 3-comment-line sys.path marker and stays untouched; its presence keeps
the checkout root importable, and this file extends the same convention to
the tests/ package). The session-scoped ``real_image_fixture_video`` fixture
builds a 60-frame mp4v still-loop of the probe-verified dataset umpire image
(RESEARCH Pattern 7) so test_inference_analyzer.py, test_run_inference.py's
weight-gated CPU smoke test, and Plan 04-03's temporal tests all reuse the
SAME builder and the SAME cached file.

mp4v ONLY: probe P1 verified exact roundtrip on this machine; avc1 fails
(missing OpenH264 DLL — RESEARCH Pitfall 1). Fixture binaries are generated
into the session tmp dir and never committed; the source dataset image is
UNTRACKED (datasets/ is not git-tracked), so consumers gate with skipif on
FIXTURE_IMAGE existence.
"""

from pathlib import Path

import cv2
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The probe-verified fixture image (probe P3: stable keypoints conf
# 0.448-0.980). UNTRACKED — tests must skip, not fail, when it is absent.
FIXTURE_IMAGE = (
    _REPO_ROOT
    / "datasets"
    / "roboflow_umpire_wide"
    / "test"
    / "images"
    / "frame_0000_jpg.rf.a83c753ae26d126b851344fd46e803aa.jpg"
)

FIXTURE_FRAMES = 60
FIXTURE_FPS = 30.0


@pytest.fixture(scope="session")
def real_image_fixture_video(tmp_path_factory):
    """Session-scoped 60-frame mp4v still-loop of the real dataset image.

    A constant frame per strided step means the Model A probability vector
    is constant, the EMA converges instantly (EMA of a constant is the
    constant), and the stabilizer/gate behavior is deterministic and
    predictable (RESEARCH Pattern 7). Built ONCE per session into the
    session tmp dir; binaries never committed."""
    image = cv2.imread(str(FIXTURE_IMAGE))  # BGR uint8 (OpenCV convention)
    assert image is not None, f"cv2.imread failed on {FIXTURE_IMAGE}"
    out_dir = tmp_path_factory.mktemp("fixture_videos")
    video_path = out_dir / "still_frame_fixture.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # mp4v ONLY — never avc1 on this machine
        FIXTURE_FPS,
        (image.shape[1], image.shape[0]),
    )
    assert writer.isOpened(), f"VideoWriter failed to open {video_path}"
    for _ in range(FIXTURE_FRAMES):
        writer.write(image)
    writer.release()
    return video_path
