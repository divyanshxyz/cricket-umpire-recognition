"""Phase 5 Streamlit app — the interactive path over the frozen Phase 4 seams.

Locked thin-app boundary (05-CONTEXT): this module owns ZERO inference,
stabilizer, or scoreboard logic. It composes ONLY the Phase 4 seams
(UmpireVideoAnalyzer.classify_image / annotate_image via get_analyzer) and
the streamlit-free helpers in src/app_support.py (upload gates, typed-error
templates, arm-angles display mapping). Every class name, probability dict,
and geometry value rendered here is engine-derived — nothing is retyped.

This plan (05-01) delivers the shell + Tab 1 (Image Analyzer) end-to-end;
Tab 2's processing loop arrives with Plan 05-02 and Tab 3 with Plan 05-03
(their bodies are the exact empty-state / signpost copies in the meantime).

Structure contract (05-01-PLAN Step 4): no st.* call happens at module
level — pytest imports of this module stay inert; main() runs under both
`streamlit run app.py` and AppTest (executed with __name__ == "__main__").
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.app_support import (
    IMAGE_WRONG_CONTENT_COPY,
    arm_angle_rows,
    check_image_bytes,
    decode_image_bgr,
    typed_error_copy,
)
from src.inference.analyzer import UmpireVideoAnalyzer
from src.inference.config import INFERENCE_DEFAULTS, load_inference_config

# All repo-relative paths resolve from HERE, never from cwd (configs/ and
# reports/ consumers: the engine seams resolve their own roots; Tab 3's
# reports/ reads in Plan 05-03 reuse this constant).
REPO_ROOT = Path(__file__).resolve().parent


@st.cache_resource
def get_analyzer(cfg_signature: tuple):
    """One UmpireVideoAnalyzer per config signature (UI-SPEC State contract):
    the analyzer builds ONCE and is reused across uploads in the session —
    the scoreboard FSM continues across uploads (Phase 4's documented reuse
    contract). ``cfg_signature`` is the tuple of ALL loaded
    configs/inference.yaml values in the fixed INFERENCE_DEFAULTS key order
    (sanctioned import, Plan 05-03 P1 allowlist choice (a)): ANY config edit
    changes the key, rebuilds, and re-runs the 5-digest integrity sweep. The
    app NEVER calls warm_up() itself — the engine warms up on first stream /
    classify (the one-time model load lands inside Tab 1's spinner)."""
    return UmpireVideoAnalyzer()


@st.cache_data
def classify_and_annotate(image_bytes: bytes, cfg_signature: tuple):
    """Tab 1 result memo keyed on the image bytes (UI-SPEC Decision 3): the
    same upload bytes never re-call the engine — a rerun (widget redraw,
    tab switch) is a cache hit. The analyzer resolves through the
    module-global get_analyzer name at CALL time so tests can inject a
    stub. Exceptions raised inside are NOT cached — a failed engine call
    retries on the next interaction (the user fixes the cause and re-runs)."""
    analyzer = get_analyzer(cfg_signature)
    bgr = decode_image_bgr(image_bytes)
    return analyzer.classify_image(bgr), analyzer.annotate_image(bgr)


def render_image_tab(cfg_signature: tuple) -> None:
    """Tab 1 — Image Analyzer (UI-01; UI-SPEC Per-Tab Layout Contract).

    Validation pipeline, in order, each failure rendering its exact toast
    and STOPPING before any engine call (T-05-01): size on the in-memory
    bytes -> magic-byte sniff -> cv2 decode. Then the analyzer ([integrity]
    boundary) and the engine run inside the spinner ([input] boundary)."""
    uploaded = st.file_uploader(
        "Upload an umpire image (JPG/PNG, up to 10 MB)",
        type=["jpg", "jpeg", "png"],  # convenience filter only — sniff is the gate
    )
    if uploaded is None:
        # Empty state (UI-SPEC copy table): rendered in place of the results
        # columns; no engine call fires without an upload.
        st.subheader("No image yet")
        st.markdown(
            "Upload an umpire photo (JPG or PNG, up to 10 MB) — the original "
            "vs annotated view, predicted signal, class confidence chart, "
            "and arm-angles table appear here."
        )
        return
    data = uploaded.getvalue()
    toast = check_image_bytes(uploaded.name, data)
    if toast is not None:
        st.error(toast)
        return
    bgr = decode_image_bgr(data)
    if bgr is None:
        # Magic bytes passed but the payload does not decode — the exact
        # wrong-content copy (UI-SPEC Tab 1 step 2), never an engine call.
        st.error(IMAGE_WRONG_CONTENT_COPY.format(name=uploaded.name))
        return
    try:
        analyzer = get_analyzer(cfg_signature)
    except ValueError as err:
        st.error(typed_error_copy("integrity", str(err)))
        return
    try:
        with st.spinner("Extracting pose and classifying…"):
            result, annotated = classify_and_annotate(data, cfg_signature)
    except ValueError as err:
        st.error(typed_error_copy("input", str(err)))
        return

    # Results (UI-SPEC Tab 1 steps 3-6): original vs annotated pair, the
    # predicted-signal badge, the 6-class confidence chart passed AS-IS
    # (keyed in LabelEncoder order — never re-sorted), and the 8-row
    # arm-angles table through the frozen seam functions.
    left, right = st.columns(2)
    with left:
        st.image(
            bgr[:, :, ::-1],
            caption="Original",
            use_container_width=True,
        )
    with right:
        st.image(
            annotated[:, :, ::-1],
            caption="Annotated — skeleton + bbox",
            use_container_width=True,
        )
    st.success(
        "Predicted signal: "
        f"{result.predicted_class} "
        f"(confidence {result.probabilities[result.predicted_class]:.2f})"
    )
    st.bar_chart(result.probabilities)
    st.dataframe(
        arm_angle_rows(result.raw51, analyzer.extractor.config["mask_conf"])
    )


def main() -> None:
    """App entry: page config -> [config] boundary -> sidebar status block ->
    exactly 3 tabs (Tab 1 wired end-to-end; Tab 2/3 carry the exact
    pre-delivery copy for Plans 05-02/05-03)."""
    # FIRST Streamlit call (UI-SPEC shared shell rule 1).
    st.set_page_config(
        page_title="Cricket Umpire Signal Recognition",
        page_icon="🏏",
        layout="wide",
    )

    # [config]-class boundary: no tabs render without a valid config
    # (fail-closed — the typed message names the offending key/fix).
    try:
        cfg = load_inference_config()
    except ValueError as err:
        st.error(typed_error_copy("config", str(err)))
        st.stop()

    # Cache signature: ALL 11 config values in the fixed INFERENCE_DEFAULTS
    # key order — any config edit changes the key and rebuilds the analyzer.
    cfg_signature = tuple(cfg[key] for key in INFERENCE_DEFAULTS)

    # Sidebar carries ONLY the status block (UI-SPEC shared shell rule 3).
    with st.sidebar:
        st.header("🏏 Cricket Umpire Signal Recognition")
        st.caption(
            f"Model mode: {cfg['model']} · device: {cfg['device']} · "
            f"max_frames guard: {cfg['max_frames']} — from configs/inference.yaml"
        )
        st.caption(
            "Model artifacts are sha256-verified against "
            "configs/model_digests.yaml at analyzer startup — the app never "
            "loads model files itself."
        )
        st.caption(
            "Switch model: edit model: in configs/inference.yaml and restart "
            "the app (no UI switch in v1)."
        )

    # Session state contract (UI-SPEC State/caching): model_mode mirrors the
    # loaded config (re-mirrored from analyzer.cfg after construction in
    # Plan 05-02); the run-summary/event keys are initialized for Tab 2.
    st.session_state["model_mode"] = cfg["model"]
    if "last_run_summary" not in st.session_state:
        st.session_state["last_run_summary"] = None
    if "last_run_events" not in st.session_state:
        st.session_state["last_run_events"] = ()

    tab1, tab2, tab3 = st.tabs(
        ["Image Analyzer", "Video Analyzer", "Research Metrics"]
    )
    with tab1:
        render_image_tab(cfg_signature)
    with tab2:
        # Empty state now; the uploader + Analyze-video loop arrive with
        # Plan 05-02 (final contract copy below).
        st.subheader("No video yet")
        st.markdown(
            "Upload an MP4 clip (up to 200 MB) and press Analyze video — "
            "annotated frames, the live scoreboard, and the event log appear "
            "as frames are processed."
        )
    with tab3:
        # Signpost only; replaced by the Plan 05-03 tracer.
        st.info(
            "Research Metrics — the tracked ablation report and confusion "
            "matrix render here (delivered by Plan 05-03)."
        )


if __name__ == "__main__":
    main()
