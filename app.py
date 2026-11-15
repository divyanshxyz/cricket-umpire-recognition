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
    PROGRESS_UNKNOWN_CAPTION,
    arm_angle_rows,
    build_event_csv_bytes,
    check_image_bytes,
    compute_progress,
    decode_image_bgr,
    event_log_rows,
    execute_video_job,
    extract_lineage_lines,
    format_failure_tally,
    load_metrics_artifacts,
    load_multimodal_metrics_report,
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
    mode = st.session_state.get("model_mode", "image")
    badge_suffix = " — Multimodal (Pose + Vision)" if mode == "multimodal" else ""
    st.success(
        "Predicted signal: "
        f"{result.predicted_class} "
        f"(confidence {result.probabilities[result.predicted_class]:.2f})"
        f"{badge_suffix}"
    )
    st.bar_chart(result.probabilities)
    st.dataframe(
        arm_angle_rows(result.raw51, analyzer.extractor.config["mask_conf"])
    )


def render_video_tab(cfg_signature: tuple) -> None:
    """Tab 2 — Video Analyzer (UI-02, UI-03; UI-SPEC Per-Tab Layout Contract).

    Upload match clip -> button press -> decode probe through tempfile
    -> analyze_stream loop updating progress, metric tiles, frame slot, event log
    -> summary block + CSV download.
    """
    uploaded = st.file_uploader(
        "Upload a match clip (MP4/MOV/AVI/MKV/WEBM, up to 200 MB)",
        type=["mp4", "mov", "avi", "mkv", "webm"],
    )
    analyze_clicked = st.button("Analyze video", type="primary")

    if not analyze_clicked and st.session_state.get("last_run_summary") is None:
        st.subheader("No video yet")
        st.markdown(
            "Upload an MP4 clip (up to 200 MB) and press Analyze video — "
            "annotated frames, the live scoreboard, and the event log appear "
            "as frames are processed."
        )
        return

    if st.session_state.get("model_mode") == "multimodal":
        st.caption("Active model: Multimodal (Pose + Vision)")

    col1, col2, col3, col4 = st.columns(4)
    tile_runs = col1.empty()
    tile_wickets = col2.empty()
    tile_overs = col3.empty()
    tile_extras = col4.empty()

    def update_tiles(score_tuple):
        runs, wickets, overs, extras = score_tuple
        tile_runs.metric("Runs", runs)
        tile_wickets.metric("Wickets", wickets)
        tile_overs.metric("Overs", overs)
        tile_extras.metric("Extras", extras)

    frame_slot = st.empty()
    log_slot = st.empty()

    if analyze_clicked:
        if uploaded is None:
            st.error("Please upload a video file first.")
            return

        try:
            analyzer = get_analyzer(cfg_signature)
        except ValueError as err:
            st.error(typed_error_copy("integrity", str(err)))
            return

        st.session_state["model_mode"] = analyzer.cfg["model"]
        update_tiles((
            analyzer.score_state.runs,
            analyzer.score_state.wickets,
            analyzer.score_state.overs,
            analyzer.score_state.extras,
        ))
        log_slot.caption(
            "No events yet — signals appear here as the stabilizer fires."
        )

        progress_bar = st.progress(0.0)
        progress_caption_slot = st.empty()

        events_live: list = []

        def on_progress(frames_read, estimate):
            frac = compute_progress(
                frames_read, estimate, analyzer.cfg["max_frames"]
            )
            progress_bar.progress(frac)
            if estimate is None or estimate <= 0:
                progress_caption_slot.caption(PROGRESS_UNKNOWN_CAPTION)

        def on_frame(annotated):
            frame_slot.image(annotated[:, :, ::-1], use_container_width=True)

        def on_event(ev):
            events_live.append(ev)
            update_tiles(ev.score)
            log_slot.dataframe(
                event_log_rows(events_live),
                height=320,
                use_container_width=True,
            )

        data = uploaded.getvalue()
        try:
            job = execute_video_job(
                analyzer,
                data,
                on_progress=on_progress,
                on_frame=on_frame,
                on_event=on_event,
                name=uploaded.name,
            )
        except ValueError as err:
            st.error(typed_error_copy("input", str(err)))
            return

        if isinstance(job, str):
            st.error(job)
            return

        st.session_state["last_run_summary"] = {
            "frames_read": job["frames_read"],
            "strided": job["strided"],
            "fps_reported": job["fps_reported"],
            "tally": job["tally"],
            "aborted": job["aborted"],
            "score": (
                analyzer.score_state.runs,
                analyzer.score_state.wickets,
                analyzer.score_state.overs,
                analyzer.score_state.extras,
            ),
            "event_count": len(job["events"]),
            "max_frames": analyzer.cfg["max_frames"],
        }
        st.session_state["last_run_events"] = job["events"]
        st.session_state["last_run_last_frame"] = job["last_frame"]

    summary = st.session_state.get("last_run_summary")
    if summary is not None:
        update_tiles(summary["score"])
        if st.session_state.get("last_run_last_frame") is not None:
            frame_slot.image(
                st.session_state["last_run_last_frame"][:, :, ::-1],
                use_container_width=True,
            )
        events = st.session_state.get("last_run_events", ())
        if events:
            log_slot.dataframe(
                event_log_rows(events),
                height=320,
                use_container_width=True,
            )
        else:
            log_slot.caption(
                "No events yet — signals appear here as the stabilizer fires."
            )

        if summary.get("aborted"):
            cap = summary.get("max_frames", 100000)
            st.warning(
                f"Video stopped at the max_frames guard — {summary['frames_read']} "
                f"frames processed (config cap: {cap}). The scoreboard, event "
                "log, and summary below cover only the frames processed."
            )

        lines = [
            f"Run complete — frames read: {summary['frames_read']}  "
            f"strided: {summary['strided']}  fps_reported: {summary['fps_reported']}",
            f"typed failure tallies: {format_failure_tally(summary['tally'])}",
            f"Score: {summary['score'][0]} runs/{summary['score'][1]} wickets/"
            f"{summary['score'][2]} overs/{summary['score'][3]} extras",
        ]
        if summary["event_count"] > 0:
            lines.append(f"{summary['event_count']} gated signal event(s) fired")
        else:
            lines.append("no gated signal events fired")
        if st.session_state.get("model_mode") == "temporal":
            from src.scoreboard import UNTRAINED_SMOKE_MARKER

            lines.append(UNTRAINED_SMOKE_MARKER)
        elif st.session_state.get("model_mode") == "multimodal":
            lines.append("Model: Multimodal (Pose + Vision)")
        st.success("\n\n".join(lines))

        st.download_button(
            "Download event log (CSV)",
            data=build_event_csv_bytes(st.session_state["last_run_events"]),
            file_name="event_log.csv",
            mime="text/csv",
        )


def render_metrics_tab() -> None:
    """Tab 3 — Research Metrics (UI-04; UI-SPEC Per-Tab Layout Contract).

    Verbatim ablation study report, container-width confusion matrix,
    and render-time extracted sha256 lineage caption. Also renders
    Phase 6 Multimodal Ablation Report alongside baseline metrics (FUSE-04).
    """
    artifacts = load_metrics_artifacts(REPO_ROOT)
    if isinstance(artifacts, str):
        st.error(artifacts)
        return
    report_text, png_path = artifacts

    multimodal_text = load_multimodal_metrics_report(REPO_ROOT)
    if multimodal_text:
        st.markdown(multimodal_text)
        st.divider()

    st.markdown(report_text)
    st.image(str(png_path), use_container_width=True)
    lineage_lines = extract_lineage_lines(report_text)
    if lineage_lines:
        st.caption("\n\n".join(lineage_lines))


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
        render_video_tab(cfg_signature)
    with tab3:
        render_metrics_tab()


if __name__ == "__main__":
    main()
