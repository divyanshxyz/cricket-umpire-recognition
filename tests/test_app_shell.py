"""Phase 5 Tab 1 hardening battery — AppTest shell + memo pins (05-01-PLAN
Task 2 Steps 2-3).

Headless, NO model construction: the no-upload shell renders the 3-tab
layout, empty states, sidebar copy, and session-state initialization with
zero exceptions and provably zero analyzer builds. The memo pins prove the
bytes-keyed result cache calls the engine exactly once per distinct
payload and never caches exceptions. ScriptRunContext warnings from cache
calls outside a Streamlit runtime are expected and tolerated.
"""

import uuid
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.inference.config import load_inference_config

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = REPO_ROOT / "app.py"

# Exact UI-SPEC empty-state bodies (copywriting contract).
TAB1_EMPTY_BODY = (
    "Upload an umpire photo (JPG or PNG, up to 10 MB) — the original vs "
    "annotated view, predicted signal, class confidence chart, and "
    "arm-angles table appear here."
)
TAB2_EMPTY_BODY = (
    "Upload an MP4 clip (up to 200 MB) and press Analyze video — annotated "
    "frames, the live scoreboard, and the event log appear as frames are "
    "processed."
)


def _run_app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=60).run()


# ----------------------------------------------------------------------
# AppTest shell battery (no upload -> no analyzer construction)
# ----------------------------------------------------------------------


def test_shell_runs_clean_with_three_exact_tabs():
    """The shell completes with zero exceptions and EXACTLY 3 tabs carrying
    the exact labels in the exact order (UI-SPEC shared shell rule 2)."""
    at = _run_app()
    assert at.exception == []
    assert [tab.label for tab in at.tabs] == [
        "Image Analyzer",
        "Video Analyzer",
        "Research Metrics",
    ]


def test_shell_renders_tab1_and_tab2_empty_state_copy():
    """Before any upload, Tab 1 renders the 'No image yet' heading + body
    in place of the results columns; Tab 2 carries its exact pre-delivery
    copy; no engine call fires without an upload."""
    at = _run_app()
    assert at.exception == []
    assert [s.value for s in at.subheader] == ["No image yet", "No video yet"]
    markdown_values = [m.value for m in at.markdown]
    assert TAB1_EMPTY_BODY in markdown_values
    assert TAB2_EMPTY_BODY in markdown_values


def test_shell_sidebar_captions_are_config_derived():
    """The sidebar carries ONLY the status block: the 🏏 heading then the
    three exact captions, the first rendering the loaded config values."""
    cfg = load_inference_config()
    at = _run_app()
    assert at.exception == []
    assert [h.value for h in at.sidebar.header] == [
        "🏏 Cricket Umpire Signal Recognition"
    ]
    captions = [c.value for c in at.sidebar.caption]
    assert captions == [
        (
            f"Model mode: {cfg['model']} · device: {cfg['device']} · "
            f"max_frames guard: {cfg['max_frames']} — from configs/inference.yaml"
        ),
        (
            "Model artifacts are sha256-verified against "
            "configs/model_digests.yaml at analyzer startup — the app never "
            "loads model files itself."
        ),
        (
            "Switch model: edit model: in configs/inference.yaml and restart "
            "the app (no UI switch in v1)."
        ),
    ]


def test_shell_has_no_download_button_before_run():
    """Before any completed run, no download button exists in the app
    (the empty-state UI-Considerations row: the CSV download button renders
    only after a completed run)."""
    at = _run_app()
    assert at.exception == []
    assert len(at.download_button) == 0


def test_shell_session_state_initialized_from_config():
    """session_state['model_mode'] initializes from the loaded
    configs/inference.yaml; the run-summary/event keys exist for Tab 2."""
    cfg = load_inference_config()
    at = _run_app()
    assert at.exception == []
    assert at.session_state["model_mode"] == cfg["model"]
    assert at.session_state["last_run_summary"] is None
    assert at.session_state["last_run_events"] == ()


def test_shell_never_builds_analyzer(monkeypatch):
    """The no-upload shell path constructs NO analyzer (by design: the
    cache_resource factory and the engine calls are all behind the upload
    gate) — a patched constructor that raises proves the run never touches
    the model layer."""
    import src.inference.analyzer as analyzer_module

    def _boom(*args, **kwargs):
        raise AssertionError("the no-upload shell constructed the analyzer")

    monkeypatch.setattr(analyzer_module, "UmpireVideoAnalyzer", _boom)
    at = _run_app()
    assert at.exception == []


# ----------------------------------------------------------------------
# Memo pins (classify_and_annotate keyed on image bytes)
# ----------------------------------------------------------------------


class _CountingEngine:
    """Stub analyzer: counts engine invocations, returns picklable
    markers (st.cache_data stores the return value)."""

    def __init__(self):
        self.classify_calls = 0
        self.annotate_calls = 0

    def classify_image(self, bgr):
        self.classify_calls += 1
        return ("classification", self.classify_calls)

    def annotate_image(self, bgr):
        self.annotate_calls += 1
        return ("annotation", self.annotate_calls)


def test_memo_invokes_engine_once_per_distinct_payload(monkeypatch):
    """The same image bytes never re-call the engine (cache hit on the
    second call); distinct bytes re-call it — the memo is keyed on the
    payload bytes. Distinct synthetic payloads avoid cache
    cross-contamination across tests."""
    import app

    engine = _CountingEngine()
    monkeypatch.setattr(app, "get_analyzer", lambda sig: engine)
    sig = ("memo-sig", 1)

    payload_a = b"memo-payload-A-" + uuid.uuid4().bytes
    first = app.classify_and_annotate(payload_a, sig)
    second = app.classify_and_annotate(payload_a, sig)  # cache hit
    assert first == second
    assert engine.classify_calls == 1
    assert engine.annotate_calls == 1

    payload_b = b"memo-payload-B-" + uuid.uuid4().bytes
    app.classify_and_annotate(payload_b, sig)
    assert engine.classify_calls == 2
    assert engine.annotate_calls == 2


def test_memo_does_not_cache_exceptions(monkeypatch):
    """A typed engine failure re-raises on the second identical call AND
    re-executes the body (the exception is not cached — the user fixes the
    cause and re-runs, per UI-SPEC Typed-Error Rendering)."""
    import app

    attempts = {"n": 0}

    class ExplodingEngine:
        def classify_image(self, bgr):
            attempts["n"] += 1
            raise ValueError("extraction failed (no_person: typed failure)")

        def annotate_image(self, bgr):  # pragma: no cover — classify raises first
            raise AssertionError("annotate_image must not be reached")

    monkeypatch.setattr(app, "get_analyzer", lambda sig: ExplodingEngine())
    payload = b"exploding-payload-" + uuid.uuid4().bytes
    sig = ("exploding-sig", 1)

    with pytest.raises(ValueError, match="no_person"):
        app.classify_and_annotate(payload, sig)
    with pytest.raises(ValueError, match="no_person"):
        app.classify_and_annotate(payload, sig)
    assert attempts["n"] == 2, "the failed call must re-execute, not cache"
