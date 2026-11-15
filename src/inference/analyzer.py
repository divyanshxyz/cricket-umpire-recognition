"""Phase 4 integration analyzer: the composition root of the inference engine.

Composition (per 04-02-PLAN): the FROZEN Phase 2 seam
(PoseFeatureExtractor.extract — zero rewrites) feeds the Phase 3 image-model
adapter (integrity-bound loads), which feeds the pure Phase 4 stabilizer
(window/EMA/gate/cooldown), whose fired outcomes are the ONLY thing the pure
scoreboard FSM consumes (SCORE-03: gated-event consumption). Nothing here
re-implements any of those layers.

Typed-failure posture (ASVS V5, pose_features.py lineage): per-frame
extraction failures, annotation-selection failures, fps fallbacks, and the
max_frames abort are TYPED and COUNTED via the module-level event recorder —
never silent, never a crash. A failed frame enters the stabilizer window as
a one-hot NoAction vector (OQ3) so the window update is never skipped.

Integrity posture (WR-04, Pattern 5): the constructor verifies ALL 5
digests from configs/model_digests.yaml BEFORE any deserialize (hashing 5
small files is milliseconds); every subsequent load re-verifies immediately
before its own deserialize. expected_sha256=None is never passed.

Reuse contract: a FRESH UmpireVideoAnalyzer per run keeps state isolated.
Calling analyze()/analyze_stream() on a REUSED instance CONTINUES from the
current stabilizer/score state (the window, EMA, cooldown latch, and
ScoreState persist across calls) — Phase 5 constructs one analyzer per
session instead of reusing one across uploads.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.inference.config import load_inference_config, load_model_digests
from src.inference.renderer import draw_annotation
from src.inference.stabilizer import (
    CLASS_ORDER,
    TemporalStabilizer,
    stabilizer_events,
)
from src.models.data import verify_artifact_sha256
from src.pose_features import (
    PoseFeatureExtractor,
    assemble_sequence,
    mask_keypoints,
    select_umpire,
    torso_scale,
)
from src.scoreboard import (
    UNTRAINED_SMOKE_MARKER,
    EventLogEntry,
    ScoreState,
    apply_event,
)

# src/inference/ is one level deeper than src/ — same depth as src/models/*
# (PATTERNS.md): three .parent hops resolve to the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# Module-level typed event recorder (pose_features.py:111-136 recorder
# shape): every per-frame extraction failure, annotation-selection failure,
# annotation-predict error, and max_frames abort is recorded here — counted,
# never silent (ASVS V5). Stabilizer-side events (fps_fallback / release /
# fire) land in the stabilizer's own recorder; analyze() tallies both.
_ANALYZER_EVENTS: list[str] = []


def record_analyzer_event(label, reason) -> None:
    """Record one typed analyzer event for the audit (ASVS V5)."""
    _ANALYZER_EVENTS.append(f"{label}: {reason}")


def analyzer_events() -> list:
    """Analyzer event records (one string per typed event)."""
    return list(_ANALYZER_EVENTS)


@dataclass(frozen=True)
class FrameRecord:
    """One yielded frame of the analyze_stream generator — the Phase 5 Tab 2
    per-frame contract. ``new_events`` carries the EventLogEntry values that
    FIRED on this frame (empty tuple when none): the app renders the
    annotated frame plus any scoreboard updates the frame produced."""

    frame_idx: int
    annotated: np.ndarray
    new_events: tuple


@dataclass(frozen=True)
class VideoAnalysisResult:
    """Aggregate result of one analyze() pass. ``events`` is in emission
    order (strictly monotonic in frame index by construction — the
    stabilizer fires at most once per frame and the cursor is monotonic);
    ``failures`` is the closed-set tally of THIS run's typed analyzer +
    stabilizer events (label -> count)."""

    events: tuple
    final_score: ScoreState
    frames_read: int
    strided_frames: int
    failures: dict
    fps_reported: float
    device: str
    model_mode: str


@dataclass(frozen=True)
class ImageClassification:
    """Single-image classification result (VIDEO-05). ``probabilities`` is
    keyed in the LabelEncoder sorted class order (asserted against the
    artifact at load time — never re-ordered); ``probs_vector`` is the fp32
    (6,) softmax row summing to 1."""

    predicted_class: str
    probabilities: dict
    probs_vector: np.ndarray
    raw51: np.ndarray


class UmpireVideoAnalyzer:
    """Video + single-image analyzer composing every frozen layer.

    Constructor contract: loads configs/inference.yaml through the
    fail-closed loader, guards device consistency against the pose seam's
    own config (Pitfall 7 device drift — the annotation predict reuses the
    extractor's device, so a mismatch is a config bug, never a silent
    fallback), builds the pure stabilizer from the config literals, and
    runs the startup integrity sweep over ALL 5 digests BEFORE any
    deserialize (WR-04 / Pattern 5).
    """

    def __init__(self, config_path=None, extractor=None):
        self.cfg = load_inference_config(config_path)
        if extractor is None:
            extractor = PoseFeatureExtractor()
        self.extractor = extractor
        if self.cfg["device"] != self.extractor.config["device"]:
            raise ValueError(
                f"UmpireVideoAnalyzer: device drift — inference config "
                f"device {self.cfg['device']!r} vs pose seam device "
                f"{self.extractor.config['device']!r} — the annotation "
                "predict must run on the SAME device as the frozen seam "
                "(RESEARCH Pitfall 7; fix configs/inference.yaml or "
                "configs/pose.yaml, fail-closed, no silent fallback)"
            )
        self.score_state = ScoreState()
        self.stabilizer = TemporalStabilizer(
            window=self.cfg["window"],
            alpha=self.cfg["alpha"],
            threshold=self.cfg["threshold"],
            fire_frac=self.cfg["fire_frac"],
            cooldown_seconds=self.cfg["cooldown_seconds"],
            fps_fallback=self.cfg["fps_fallback"],
            require_release=self.cfg["require_release"],
        )

        # Startup integrity sweep (Pattern 5): verify ALL 5 digests BEFORE
        # any inference/deserialize. A missing artifact refuses with a
        # message naming the path; a swapped/stale one refuses inside
        # verify_artifact_sha256. Both are fail-closed ValueErrors.
        self._digests = load_model_digests()
        for entry in self._digests.values():
            artifact = REPO_ROOT / entry["path"]
            if not artifact.is_file():
                raise ValueError(
                    f"startup integrity sweep: artifact {entry['path']} does "
                    f"not exist at {artifact} — refusing to run inference "
                    "against a missing model file (fail-closed, no silent "
                    "default; bootstrap the artifact or fix "
                    "configs/model_digests.yaml)"
                )
            verify_artifact_sha256(
                artifact, entry["sha256"], artifact_name=entry["path"]
            )

        # Model-adapter state (lazy: the image model loads on the first
        # strided frame / classify_image call, after the sweep).
        self._image_model = None
        self._image_scaler = None
        self._image_classes = None

        # Temporal arm (P2RDY-02): the seeded UNTRAINED stub is constructed
        # ONCE at init when model: temporal — torch.manual_seed(42)
        # IMMEDIATELY before TwoHeadAttentionBiGRU() (nothing between seed
        # and construction). No checkpoint exists for the stub (models/
        # carries none — verified), so the seed IS the determinism contract
        # (RESEARCH Pitfall 5); the stub is never loaded from disk.
        if self.cfg["model"] == "temporal":
            # Lazy heavy import (mirrors the image arm's discipline):
            # src.models.temporal imports torch at module level, so image-
            # mode users (and --dry-run) never pay the torch import.
            import torch

            from src.models.temporal import TwoHeadAttentionBiGRU

            torch.manual_seed(42)
            self._temporal_model = TwoHeadAttentionBiGRU()
            self._temporal_model.eval()
            self._temporal_model.to(self.cfg["device"])
            # Pattern 4 row buffer: assemble_sequence needs the per-frame
            # (raw51, eng59, scale, masked_xy) rows that extract() does not
            # return — reconstructed per strided frame in _temporal_probs.
            self._row_buffer = deque(maxlen=15)

        # Multimodal arm (FUSE-04): load VisualFeatureExtractor and
        # MultimodalFusionClassifier bundle.
        if self.cfg["model"] == "multimodal":
            from src.inference.vision_extractor import VisualFeatureExtractor
            from src.models.multimodal import MultimodalFusionClassifier

            self._vision_extractor = VisualFeatureExtractor(device=self.cfg["device"])
            fusion_rel = self._digests["multimodal_fusion"]["path"]
            self._multimodal_model = MultimodalFusionClassifier.load(
                REPO_ROOT / fusion_rel,
                expected_sha256=self._digests["multimodal_fusion"]["sha256"],
            )
            if list(self._multimodal_model.classes_) != list(CLASS_ORDER):
                raise ValueError(
                    f"UmpireVideoAnalyzer: multimodal model class order {list(self._multimodal_model.classes_)} "
                    f"does not match stabilizer CLASS_ORDER {list(CLASS_ORDER)} — fail closed"
                )

        # Fail-closed adapter dispatch (P2RDY-02 / FUSE-04): image, temporal,
        # and multimodal are the valid adapters — any other cfg["model"] value raises
        # naming the observed value.
        self._prob_adapters = {"image": self._image_probs}
        if self.cfg["model"] == "temporal":
            self._prob_adapters["temporal"] = self._temporal_probs
        elif self.cfg["model"] == "multimodal":
            self._prob_adapters["multimodal"] = self._multimodal_probs
        adapter = self._prob_adapters.get(self.cfg["model"])
        if adapter is None:
            raise ValueError(
                f"UmpireVideoAnalyzer: model mode {self.cfg['model']!r} has "
                f"no probability adapter wired (only image | temporal | multimodal are "
                "valid — P2RDY-02 / FUSE-04 fail-closed enum, no silent default)"
            )
        self._prob_adapter = adapter

        # Per-run stats (reset at each analyze_stream start; the stabilizer
        # and score state intentionally CONTINUE across reused instances).
        self._frames_read = 0
        self._strided_frames = 0
        self._fps_reported = 0.0
        self._events_baseline = len(analyzer_events())
        self._stabilizer_events_baseline = len(stabilizer_events())

    # ------------------------------------------------------------------
    # Image-model adapter (integrity-bound, lazy)
    # ------------------------------------------------------------------

    def _load_image_model(self) -> None:
        """Lazy one-shot image-model load (VIDEO-05 / A5 locked reading).

        Every deserialize here is integrity-bound (WR-04):
        - baseline_ann.pt gates internally via load_baseline_checkpoint
          with expected_sha256=<digest> — weights_only=False is REQUIRED
          because the train-fit MinMaxScaler rides inside the checkpoint
          (never pass expected_sha256=None, the sanctioned skip hatch).
        - label_encoder.joblib has no gated wrapper: verify_artifact_sha256
          runs manually BEFORE the bare joblib.load.
        The checkpoint's class list is cross-checked against the
        LabelEncoder's sorted order — a reordered probability vector would
        silently mislabel every prediction, so a mismatch refuses
        fail-closed (RESEARCH anti-pattern: reordering probabilities)."""
        # Lazy heavy imports: this module (and therefore --dry-run) stays
        # torch/sklearn-free at import time.
        import joblib

        from src.models.baseline_ann import (
            MiliBaseline,
            load_baseline_checkpoint,
            predict_proba,
        )

        baseline_rel = self._digests["baseline_ann"]["path"]
        ckpt = load_baseline_checkpoint(
            REPO_ROOT / baseline_rel,
            expected_sha256=self._digests["baseline_ann"]["sha256"],
        )
        model = MiliBaseline(n_classes=len(ckpt["classes"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        scaler = ckpt["scaler"]
        classes = list(ckpt["classes"])

        encoder_rel = self._digests["label_encoder"]["path"]
        encoder_path = REPO_ROOT / encoder_rel
        verify_artifact_sha256(
            encoder_path,
            self._digests["label_encoder"]["sha256"],
            artifact_name=encoder_rel,
        )
        label_encoder_payload = joblib.load(encoder_path)
        # The Phase 3 artifact stores {"encoder": LabelEncoder, "source":
        # "configs/classes.yaml"} (RESEARCH Pattern 4 shape — verified at
        # src/train_ablation.py:444 joblib.dump), not a bare encoder:
        # unwrap fail-closed, naming the observed payload shape.
        if (
            isinstance(label_encoder_payload, dict)
            and "encoder" in label_encoder_payload
        ):
            label_encoder = label_encoder_payload["encoder"]
        else:
            raise ValueError(
                f"_load_image_model: {encoder_rel} payload has unexpected "
                f"shape {type(label_encoder_payload).__name__} — expected a "
                "dict with an 'encoder' key holding the LabelEncoder "
                "(Phase 3 RESEARCH Pattern 4 shape; fail-closed, no silent "
                "default)"
            )
        if classes != list(label_encoder.classes_):
            raise ValueError(
                f"_load_image_model: checkpoint class order {classes} does "
                f"not match the LabelEncoder sorted order "
                f"{list(label_encoder.classes_)} — reordered probabilities "
                "would silently mislabel predictions (fail-closed, never "
                "re-order)"
            )
        self._image_model = model
        self._image_scaler = scaler
        self._image_classes = classes

    def _image_probs(self, raw51) -> np.ndarray:
        """raw51 -> fp32 (6,) probability vector via the bundled scaler +
        MiliBaseline (predict_proba: eval + no_grad + softmax, rows sum to
        1; column order = the asserted LabelEncoder sorted order)."""
        if self._image_model is None:
            self._load_image_model()
        # Lazy heavy import (torch rides in baseline_ann): keeps this
        # module importable without the heavy deps.
        from src.models.baseline_ann import predict_proba

        scaled = self._image_scaler.transform(
            np.asarray(raw51, dtype=np.float64).reshape(1, -1)
        )
        probs = predict_proba(self._image_model, scaled)
        return probs[0]

    def _noaction_vector(self) -> np.ndarray:
        """The OQ3 one-hot NoAction fallback vector fed to the stabilizer
        on a failed frame (zeros with 1.0 at the NoAction index of the
        stabilizer's own class order — derived, never a literal)."""
        names = self.stabilizer.class_names
        vec = np.zeros(len(names), dtype=np.float32)
        vec[names.index("NoAction")] = 1.0
        return vec

    # ------------------------------------------------------------------
    # Temporal-model adapter (P2RDY-02 — seeded untrained stub, T-04-07)
    # ------------------------------------------------------------------

    def _noaction_onehot(self) -> np.ndarray:
        """The OQ3 one-hot NoAction fallback vector for the temporal arm
        (zeros(6) float32 with 1.0 at the NoAction index of CLASS_ORDER —
        derived, never a literal)."""
        vec = np.zeros(len(CLASS_ORDER), dtype=np.float32)
        vec[CLASS_ORDER.index("NoAction")] = 1.0
        return vec

    def _temporal_probs(self, raw, payload, frame_idx) -> np.ndarray:
        """One strided frame's temporal-mode probability vector (P2RDY-02,
        RESEARCH Pattern 4 — one extract per frame in BOTH modes).

        The (raw51, eng59, scale, masked_xy) rows assemble_sequence needs
        are reconstructed from raw51 with the SAME pure functions
        extract_sequence_features uses internally (mask_keypoints,
        torso_scale — frozen seam functions, zero rewrites), buffered in
        deque(maxlen=15), assembled to (T, 118), pad_or_truncate'd at the
        END to 15, and the seeded untrained stub's per-timestep logits are
        mean-pooled over time then softmaxed (A6 — semantics moot for the
        untrained stub; fp32 in, fp32 out).

        Failure policy (OQ3, typed + counted, never skipped): raw is None
        (the frame's extraction failed) appends a None row — the frozen
        wholesale contract then fails EVERY assemble_sequence until the
        None evicts naturally after 15 more good appends — and each failing
        frame feeds a one-hot NoAction vector plus a typed "sequence_failure"
        event (T-04-08: the stabilizer keeps a single code path and
        processing continues; the window update is never skipped)."""
        # Lazy heavy imports (torch rides src.models.temporal at module
        # level): image-mode users never pay them; Python caches the module
        # so only the first strided frame imports.
        import torch

        from src.models.temporal import pad_or_truncate

        if raw is None:
            self._row_buffer.append(None)
            record_analyzer_event(
                "sequence_failure",
                f"frame {frame_idx}: extraction failed ({payload.reason}: "
                f"{payload.detail}) — None row appended to the sequence "
                "buffer (wholesale poisoning, OQ3 one-hot NoAction quench)",
            )
            return self._noaction_onehot()
        mask_conf = float(self.extractor.config["mask_conf"])
        kpts_masked = mask_keypoints(
            np.asarray(raw, dtype=np.float64).reshape(17, 3), mask_conf
        )
        scale, _source = torso_scale(kpts_masked, mask_conf, f"frame_{frame_idx}")
        self._row_buffer.append(
            (
                np.asarray(raw, dtype=np.float32),
                payload,
                scale,
                kpts_masked[:, :2],
            )
        )
        seq, failure = assemble_sequence(list(self._row_buffer))
        if failure is not None:
            record_analyzer_event(
                "sequence_failure",
                f"frame {frame_idx}: {failure.reason}: {failure.detail} — "
                "assemble_sequence wholesale failure (one-hot NoAction "
                "quench, OQ3; the None row evicts after 15 more good "
                "appends)",
            )
            return self._noaction_onehot()
        x = torch.from_numpy(pad_or_truncate(seq, 15)).unsqueeze(0)  # (1,15,118)
        with torch.no_grad():  # inference-only stub — never build a graph
            logits = self._temporal_model(x)  # (1, 15, 6) per-timestep logits
        return torch.softmax(logits.mean(dim=1), dim=1)[0].numpy()

    # ------------------------------------------------------------------
    # Multimodal-model adapter (FUSE-04)
    # ------------------------------------------------------------------

    def _get_umpire_bbox(self, raw51):
        """Derive bounding box envelope (x0, y0, x1, y1) from visible keypoints
        at mask_conf threshold for visual crop extraction."""
        kpts = np.asarray(raw51, dtype=np.float32).reshape(17, 3)
        mask_conf = float(self.extractor.config.get("mask_conf", 0.20))
        vis = kpts[:, 2] > mask_conf
        if vis.any():
            return (
                float(kpts[vis, 0].min()),
                float(kpts[vis, 1].min()),
                float(kpts[vis, 0].max()),
                float(kpts[vis, 1].max()),
            )
        return None

    def _multimodal_probs(self, raw, payload=None, frame=None) -> np.ndarray:
        """One strided frame's multimodal late-fusion probability vector (FUSE-04).
        Combines 81-D extended pose geometry with 576-D MobileNetV3 visual crop embeddings.
        """
        from src.pose_features import (
            engineered_features,
            engineer_extended_features,
            mask_keypoints,
            torso_scale,
        )

        mask_conf = float(self.extractor.config.get("mask_conf", 0.20))
        if payload is None:
            kpts = np.asarray(raw, dtype=np.float32).reshape(17, 3)
            kpts_masked = mask_keypoints(kpts, mask_conf)
            scale, _ = torso_scale(kpts_masked, mask_conf, "multimodal_adapter")
            payload = engineered_features(kpts_masked, scale, mask_conf)

        X_pose = engineer_extended_features(raw, payload, mask_conf=mask_conf)

        if frame is not None:
            bbox = self._get_umpire_bbox(raw)
            X_vision = self._vision_extractor.extract_crop(frame, bbox=bbox)
        else:
            X_vision = np.zeros(576, dtype=np.float32)

        probs = self._multimodal_model.predict_proba(X_pose, X_vision)
        return np.asarray(probs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Annotation (the ONE extra predict per strided frame, A8)
    # ------------------------------------------------------------------

    def _annotate(self, frame, raw51) -> np.ndarray:
        """One extra model.predict with EXACTLY the seam's predict kwargs
        (same det_conf/imgsz/device from extractor.config — same instance,
        same weights; the seam's own lazy loader is used deliberately so
        the params cannot drift, RESEARCH anti-pattern 'predict twice with
        different params'). Selects the umpire on the SECOND predict's
        boxes and draws skeleton+bbox. Defensive: a disagreeing second
        predict must never crash the loop — any failure here records a
        typed event and returns the frame UNANNOTATED."""
        model = self.extractor._ensure_model()
        try:
            results = model.predict(
                source=frame,
                conf=float(self.extractor.config["det_conf"]),
                imgsz=int(self.extractor.config["imgsz"]),
                device=self.extractor.config["device"],
                verbose=False,  # RESEARCH Pitfall 7: keep the batch log clean
            )
        except Exception as err:  # containment: annotation must not kill the run
            record_analyzer_event(
                "annotation_predict_error",
                f"frame annotation predict failed ({err}) — frame left "
                "unannotated (defensive containment)",
            )
            return frame
        r = results[0]
        if r.boxes is None:
            record_analyzer_event(
                "annotation_select_failure",
                "annotation predict returned no boxes object — frame left "
                "unannotated (defensive)",
            )
            return frame
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        box_conf = r.boxes.conf.cpu().numpy()
        img_h, img_w = frame.shape[0], frame.shape[1]
        idx, failure = select_umpire(boxes_xyxy, box_conf, img_w, img_h)
        if failure is not None:
            record_analyzer_event(
                "annotation_select_failure",
                f"{failure.reason}: {failure.detail} — frame left "
                "unannotated (defensive)",
            )
            return frame
        return draw_annotation(
            frame,
            np.asarray(raw51, dtype=np.float32).reshape(17, 3),
            boxes_xyxy[idx],
            mask_conf=float(self.extractor.config["mask_conf"]),
        )

    # ------------------------------------------------------------------
    # Video entry points
    # ------------------------------------------------------------------

    def analyze_stream(self, video_path, output_path=None):
        """Generator: the ONE entry point through every layer (Phase 5 Tab 2
        per-frame contract).

        Yields a FrameRecord for EVERY frame (strided + skipped) in input
        order; ``new_events`` carries the EventLogEntry values that fired on
        that frame. Skipped (stride-gap) frames pass through UNANNOTATED so
        the output video keeps the input's frame count and timing (A7).
        When ``output_path`` is given, EVERY frame (annotated processed
        frames + raw pass-through) is written with the mp4v fourcc — the
        ONLY reliable codec on this machine (avc1 fails: missing OpenH264
        DLL, RESEARCH Pitfall 1).

        Failure semantics: an unopenable container raises ValueError naming
        the input (VIDEO-01/empty — fail closed, never a silent empty
        output); the frame cursor iterates until read() fails (CAP_PROP_
        FRAME_COUNT is never trusted, T-04-02); per-frame extraction
        failures are typed + counted and feed the one-hot NoAction fallback
        (OQ3); ``max_frames`` trips a typed abort (T-04-06)."""
        self._frames_read = 0
        self._strided_frames = 0
        self._events_baseline = len(analyzer_events())
        self._stabilizer_events_baseline = len(stabilizer_events())

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(
                f"analyze_stream: unopenable video input {video_path} — the "
                "file is missing, unreadable, or not a container OpenCV can "
                "decode (fail-closed, never a silent empty output)"
            )
        fps = cap.get(cv2.CAP_PROP_FPS)  # RAW value; 0.0 -> typed stabilizer fallback
        self._fps_reported = float(fps)
        writer = None
        frames_written = 0
        if output_path is not None:
            writer_fps = fps if fps > 0 else float(self.cfg["fps_fallback"])
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                writer_fps,
                (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                ),
            )
            if not writer.isOpened():
                # WR-01: OpenCV fails to open an unwritable path, a missing
                # parent directory, or an unsupported extension — and for a
                # bad extension it leaves a ZERO-BYTE stub behind while
                # write() keeps "succeeding" silently. Fail closed symmetric
                # with the input guard above, and never leave the stub.
                writer.release()
                Path(output_path).unlink(missing_ok=True)
                raise ValueError(
                    f"analyze_stream: cannot open annotated-video output "
                    f"{output_path} (mp4v fourcc) — the path is unwritable "
                    "or has an unsupported extension (fail-closed, never a "
                    "silent missing output)"
                )
        try:
            # WR-02: warm up ONCE before the loop — a missing/corrupt pose
            # weights path fails fast as one config error, never as N
            # per-frame failures.
            self.extractor.warm_up()
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break  # iterate until read() fails — never trust FRAME_COUNT
                self._frames_read += 1
                if self._frames_read > self.cfg["max_frames"]:
                    record_analyzer_event(
                        "max_frames_abort",
                        f"frames_read {self._frames_read} exceeded config "
                        f"max_frames={self.cfg['max_frames']} — typed abort "
                        "(T-04-06 unbounded-processing guard)",
                    )
                    break
                ann = frame
                new_events: tuple = ()
                if frame_idx % self.cfg["stride"] == 0:
                    self._strided_frames += 1
                    raw, payload = self.extractor.extract(frame)  # frozen seam
                    if raw is None:
                        # Typed + counted; the window update is never
                        # skipped (OQ3 one-hot NoAction quench).
                        record_analyzer_event(
                            payload.reason,
                            f"frame {frame_idx}: {payload.detail}",
                        )
                        if self.cfg["model"] == "temporal":
                            # Temporal arm: the failed frame is a None row —
                            # the sequence buffer is poisoned (frozen
                            # wholesale contract) and the typed
                            # sequence_failure event is recorded inside
                            # _temporal_probs (OQ3/T-04-08: counted, never
                            # skipped).
                            probs = self._temporal_probs(
                                None, payload, frame_idx
                            )
                        else:
                            probs = self._noaction_vector()
                        outcome = self.stabilizer.update(probs, frame_idx, fps)
                    else:
                        if self.cfg["model"] == "temporal":
                            # Temporal arm (P2RDY-02): seeded stub probs
                            # through the SAME stabilizer/scoreboard path —
                            # no second pipeline.
                            probs = self._temporal_probs(raw, payload, frame_idx)
                        elif self.cfg["model"] == "multimodal":
                            probs = self._multimodal_probs(raw, payload, frame)
                        else:
                            probs = self._prob_adapter(raw)
                        ann = self._annotate(frame, raw)
                        outcome = self.stabilizer.update(probs, frame_idx, fps)
                    if outcome.fired:
                        # ONLY stabilizer-fired outcomes reach the scoreboard
                        # (SCORE-03 gated-event consumption).
                        new_state, _delta = apply_event(
                            self.score_state, outcome.signal
                        )
                        self.score_state = new_state
                        entry = EventLogEntry(
                            timestamp=outcome.timestamp_mmss,
                            signal=outcome.signal,
                            confidence=outcome.confidence,
                            score=(
                                new_state.runs,
                                new_state.wickets,
                                new_state.overs,
                                new_state.extras,
                            ),
                            smoke_marker=(
                                UNTRAINED_SMOKE_MARKER
                                if self.cfg["model"] == "temporal"
                                else ""
                            ),
                        )
                        new_events = (entry,)
                # else: skipped frame passes through UNANNOTATED (A7)
                if writer is not None:
                    writer.write(ann)
                    frames_written += 1
                yield FrameRecord(frame_idx, ann, new_events)
                frame_idx += 1
            if self._frames_read == 0:
                # VIDEO-01/empty: a container that opens but decodes ZERO
                # frames is a 0-frame stream — fails closed naming the
                # input, never a silent empty output.
                raise ValueError(
                    f"analyze_stream: zero frames decoded from {video_path} "
                    "— a 0-frame stream fails closed (VIDEO-01/empty, never "
                    "a silent empty output; verify the file is a readable "
                    "video container)"
                )
        finally:
            cap.release()
            if writer is not None:
                writer.release()
                if frames_written == 0:
                    # WR-01: a run that failed before writing ANY frame (the
                    # 0-frame ValueError, a warm_up() raise, or any abort
                    # before the first write) must not leave an empty output
                    # artifact — unlink the stub on the way out. A successful
                    # run always writes >= 1 frame (the 0-frame guard above),
                    # so this only ever fires on a failure path.
                    Path(output_path).unlink(missing_ok=True)

    def analyze(self, video_path, output_path=None) -> VideoAnalysisResult:
        """Consume analyze_stream and aggregate the frozen result (events in
        emission order — strictly monotonic in frame index by
        construction). A fresh UmpireVideoAnalyzer per run keeps state
        isolated; on a reused instance the stabilizer/score state CONTINUES
        (documented class contract — Phase 5 constructs per session)."""
        events: list = []
        for record in self.analyze_stream(video_path, output_path):
            events.extend(record.new_events)
        failures: dict = {}
        for line in analyzer_events()[self._events_baseline:]:
            label = line.split(": ", 1)[0]
            failures[label] = failures.get(label, 0) + 1
        for line in stabilizer_events()[self._stabilizer_events_baseline:]:
            label = line.split(": ", 1)[0]
            failures[label] = failures.get(label, 0) + 1
        return VideoAnalysisResult(
            events=tuple(events),
            final_score=self.score_state,
            frames_read=self._frames_read,
            strided_frames=self._strided_frames,
            failures=failures,
            fps_reported=self._fps_reported,
            device=self.cfg["device"],
            model_mode=self.cfg["model"],
        )

    # ------------------------------------------------------------------
    # Single-image path (VIDEO-05)
    # ------------------------------------------------------------------

    def classify_image(self, image) -> ImageClassification:
        """Single image (path or BGR ndarray) -> predicted class + the
        6-class probability vector. Fail-closed single-image posture: an
        extraction failure (e.g. no person detected) raises ValueError
        naming the input and the typed failure reason — never a silent
        default. The predicted class uses numpy's first-index argmax
        tie-break (Phase 3 backstop) over the asserted LabelEncoder-sorted
        probability vector."""
        if self.cfg["model"] not in ("image", "multimodal"):
            # Rule 2 guard (Plan 04-03 / FUSE-04): the temporal arm never loads the
            # image model/labels, so the LabelEncoder-sorted class names a
            # single-image query needs do not exist in temporal mode — fail
            # closed naming the observed mode BEFORE any model work (no
            # latent None-subscript crash, no wasted predict).
            raise ValueError(
                f"classify_image: single-image classification requires "
                f"model: image or multimodal — this analyzer runs model "
                f"{self.cfg['model']!r} (temporal mode classifies via the "
                "video path; no image model/labels were loaded) — rerun "
                "with model: image or multimodal in the inference config (fail-closed, "
                "no silent default)"
            )
        raw, payload = self.extractor.extract(image)
        if raw is None:
            raise ValueError(
                f"classify_image: extraction failed for {image} "
                f"({payload.reason}: {payload.detail}) — a single-image "
                "query without a detectable umpire is an error, not a "
                "silent default (fail-closed, no silent default)"
            )
        if self.cfg["model"] == "multimodal":
            if isinstance(image, (str, Path)):
                image_bgr = cv2.imread(str(image))
                if image_bgr is None:
                    raise ValueError(
                        f"classify_image: failed to decode image at {image}"
                    )
            else:
                image_bgr = image
            probs = self._multimodal_probs(raw, payload, image_bgr)
            classes = list(self._multimodal_model.classes_)
        else:
            probs = self._prob_adapter(raw)
            classes = self._image_classes
        best = int(np.argmax(probs))
        return ImageClassification(
            predicted_class=classes[best],
            probabilities={
                name: float(p) for name, p in zip(classes, probs)
            },
            probs_vector=probs,
            raw51=np.asarray(raw),
        )

    def annotate_image(self, image) -> np.ndarray:
        """Phase 5 Tab 1 annotated-render seam (UI-SPEC Decision 10, ADDITIVE
        — zero changes to existing logic): the public single-image surface
        that returns the ANNOTATED frame, mirroring the per-frame compose in
        analyze_stream (extract -> _annotate). Failure posture mirrors
        classify_image's [input]-class message shape: a no-person extraction
        raises ValueError naming the function, the input, and the typed
        failure reason — fail-closed, never a silent unannotated frame."""
        raw, payload = self.extractor.extract(image)
        if raw is None:
            raise ValueError(
                f"annotate_image: extraction failed for {image} "
                f"({payload.reason}: {payload.detail}) — a single-image "
                "annotation without a detectable umpire is an error, not a "
                "silent default (fail-closed, no silent default)"
            )
        return self._annotate(image, raw)
