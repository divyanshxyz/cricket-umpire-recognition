"""Phase 2 pose feature seam: PoseFeatureExtractor + pure feature math.

Turns an umpire image into 17 COCO keypoints (yolov8n-pose via ultralytics),
a 51-D raw keypoint vector, and a 59-D engineered vector under explicit
validity guards (CONTEXT 'Umpire Selection & Validity Guards' + 'Engineered
Feature Math'). All decisions and math live in module-level pure numpy
functions (RESEARCH.md Pattern 1) so tests run without weights/network; the
class owns model loading + tensor->numpy conversion only.

Failure contract (locked): every guard failure returns (None,
ExtractionFailure) with a typed reason recorded in the module event list —
nothing raises mid-batch, nothing is silent (ASVS V5 posture, Phase 01).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Single source of pose-config defaults (IN-02): configs/pose.yaml params are
# merged over this dict; both surfaces carry the same five locked keys.
POSE_DEFAULTS = {
    "det_conf": 0.25,       # person-detection threshold (ultralytics default; CONTEXT locked)
    "mask_conf": 0.20,      # keypoint masking boundary (x/y zeroed below it, conf preserved)
    "imgsz": 640,           # inference size
    "weights_path": "models/yolov8n-pose.pt",  # explicit path, never a bare-name load
    "device": "cpu",        # frozen-build device (device-dependent artifacts); code stays agnostic
}

# COCO-17 keypoint indices used by the engineered math
# (docs.ultralytics.com/tasks/pose): nose=0, L/R shoulder=5/6, elbow=7/8,
# wrist=9/10, hip=11/12.
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

# Velocity joints: L/R elbow, L/R wrist (4 joints x 2 coords = 8 values).
VELOCITY_JOINTS = (7, 8, 9, 10)

# Torso-scale zero-length floor: a computed scale at or below this is
# unusable and falls through to the next stage of the fallback chain.
_SCALE_MIN = 1e-6


def load_pose_config(path) -> dict:
    """Merge configs/pose.yaml params over POSE_DEFAULTS (IN-02 single
    source) and fail-close validate every key (manifest.py leak_check
    posture: a bad value raises ValueError naming the key and its value —
    never a silent default)."""
    cfg_path = Path(path)
    yaml_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg = {**POSE_DEFAULTS, **(yaml_cfg.get("params") or {})}

    for key in ("det_conf", "mask_conf"):
        value = cfg[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(
                f"configs/pose.yaml: {key} must be a number in [0, 1], got "
                f"{value!r} — fix the config or POSE_DEFAULTS (fail-closed, "
                "no silent default)"
            )
    imgsz = cfg["imgsz"]
    if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
        raise ValueError(
            f"configs/pose.yaml: imgsz must be a positive int, got {imgsz!r} "
            "— fix the config or POSE_DEFAULTS (fail-closed, no silent default)"
        )
    weights_path = cfg["weights_path"]
    if not isinstance(weights_path, str) or not weights_path.strip():
        raise ValueError(
            f"configs/pose.yaml: weights_path must be a non-empty explicit "
            f"path string, got {weights_path!r} — a bare model name would "
            "download weights to CWD (RESEARCH.md anti-pattern)"
        )
    if cfg["device"] not in ("cpu", "cuda"):
        raise ValueError(
            f"configs/pose.yaml: device must be 'cpu' or 'cuda', got "
            f"{cfg['device']!r} — fix the config or POSE_DEFAULTS (fail-closed)"
        )
    return cfg


@dataclass
class ExtractionFailure:
    """Typed extraction failure (never an exception escaping the call).

    ``reason`` is one of the closed set: "no_person" | "no_keypoints" |
    "keypoints_without_confidence" | "decode_error". ``detail`` names what
    was seen and the fix (canonical.py error-message style).
    """

    reason: str
    detail: str = ""


# Module-level typed event recorders (adapters.py pattern): every guard
# failure and unit-scale fallback is recorded here — counted, never silent
# (ASVS V5). Plan 02's build CLI counts them into the param block.
_FAILURE_EVENTS: list[str] = []
_SCALE_WARNINGS: list[str] = []


def record_failure(row_id, reason: str) -> None:
    """Record one typed extraction failure for the audit (ASVS V5)."""
    _FAILURE_EVENTS.append(f"{row_id}: {reason}")


def failure_events() -> list[str]:
    """Extraction failure records (one string per guard trip)."""
    return list(_FAILURE_EVENTS)


def record_scale_warning(row_id, detail: str) -> None:
    """Record one unit-scale torso fallback warning (the CONTEXT 'logged
    warning' for the scale=1.0 fallback lands here)."""
    _SCALE_WARNINGS.append(f"{row_id}: {detail}")


def scale_warning_events() -> list[str]:
    """Unit-scale fallback warnings (one string per event)."""
    return list(_SCALE_WARNINGS)


def select_umpire(boxes_xyxy, box_conf, img_w, img_h):
    """Locked composite umpire selection (pure numpy, weight-free testable).

    Score = (bbox_area / image_area) x (1 - 2*|cx_norm - 0.5|), both terms in
    [0, 1], computed in float64 with NO rounding anywhere; max wins. The
    exact-equality tie set is resolved by maximum detection confidence
    (order-independent). Returns (person_index, None) or (None,
    ExtractionFailure("no_person", ...)).

    Multi-person scenes are first-class: exactly one index comes back
    regardless of how many people are detected (assumption-delta
    disposition). No person-size filter exists in v1 — the composite handles
    tiny detections (CONTEXT locked decision).
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    confs = np.asarray(box_conf, dtype=np.float64).reshape(-1)
    if boxes.shape[0] == 0:
        return None, ExtractionFailure(
            "no_person",
            "no person detections (empty box list) — check det_conf in "
            "configs/pose.yaml or verify the image actually contains an umpire",
        )
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    img_area = float(img_w) * float(img_h)
    area_frac = ((x1 - x0) * (y1 - y0)) / img_area
    cx_norm = ((x0 + x1) / 2.0) / float(img_w)
    centrality = 1.0 - 2.0 * np.abs(cx_norm - 0.5)
    scores = area_frac * centrality
    best = int(np.argmax(scores))
    ties = np.flatnonzero(scores == scores[best])  # exact float comparison
    if len(ties) > 1:
        best = int(ties[int(np.argmax(confs[ties]))])
    return best, None


def mask_keypoints(kpts, mask_conf):
    """Masking semantics (CONTEXT locked): conf < mask_conf zeroes x and y
    while PRESERVING conf — masking stays detectable as x=y=0 with the
    confidence information surviving downstream. conf exactly at the
    threshold stays visible (strict < comparison)."""
    masked = np.array(kpts, dtype=np.float64, copy=True)
    low = masked[:, 2] < mask_conf
    masked[low, 0] = 0.0
    masked[low, 1] = 0.0
    return masked


def raw_features(kpts_detected):
    """51-D raw vector: (x, y, conf) per keypoint flattened in COCO order —
    the DETECTED values unmodified (no masking, no normalization). Layout is
    frozen for Phase 3 Model A's min-max consumption path."""
    return np.asarray(kpts_detected, dtype=np.float32).reshape(-1)


def engineered_features(kpts_masked):
    """59-D engineered vector slot: 17 x (torso-normalized x_hat, y_hat,
    conf) + 8 geometry features [elbow_l, elbow_r, elev_l, elev_r,
    wristnose_l, wristnose_r, span, symmetry]. Task 3 fills this slot with
    the real math (torso fallback chain + authored geometry formulas); the
    (59,) float32 slot contract is real now, so the fill is purely additive
    with no signature or call-site architectural change."""
    del kpts_masked  # unused until Task 3 fills the slot
    return np.zeros(59, dtype=np.float32)


class PoseFeatureExtractor:
    """Thin seam over ultralytics YOLO pose inference.

    Owns lazy model loading + tensor->numpy conversion ONLY; every decision
    or piece of math is a module-level pure function. ``extract`` accepts a
    path string (any image ultralytics/cv2 can decode) or a BGR HWC uint8
    ndarray — the cv2.VideoCapture frame contract (RESEARCH.md Pitfall 3:
    ultralytics assumes OpenCV BGR order for numpy inputs).
    """

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = REPO_ROOT / "configs" / "pose.yaml"
        self.config = load_pose_config(config_path)
        self._model = None

    def _ensure_model(self):
        """Lazily import torch/ultralytics (keeps this module importable
        without the heavy deps) and load the EXPLICIT weights_path from
        config — never a bare name, which would download to CWD per the
        settings-docs lookup order (RESEARCH.md anti-pattern)."""
        if self._model is None:
            from ultralytics import YOLO

            weights = Path(self.config["weights_path"])
            if not weights.is_absolute():
                weights = REPO_ROOT / weights
            self._model = YOLO(str(weights))
        return self._model

    def extract(self, image):
        """image -> (raw51 float32, engineered59 float32) on success, or
        (None, ExtractionFailure) on any guard trip. Every failure is typed,
        recorded in the module event list, and never raises (T-02-01
        containment: a corrupt/hostile file costs one typed row, not a
        crashed batch)."""
        source_label = str(image) if isinstance(image, (str, Path)) else "ndarray"
        try:
            results = self._ensure_model().predict(
                source=image,
                conf=float(self.config["det_conf"]),
                imgsz=int(self.config["imgsz"]),
                device=self.config["device"],
                verbose=False,  # RESEARCH.md Pitfall 7: keep the build log clean
            )
        except Exception as err:
            failure = ExtractionFailure(
                "decode_error",
                f"decode/inference failed for {source_label}: {err} — check "
                "the file is a readable image (T-02-01 containment)",
            )
            record_failure(source_label, failure.reason)
            return None, failure
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            failure = ExtractionFailure(
                "no_person",
                f"no person detected in {source_label} at det_conf="
                f"{self.config['det_conf']} — lower det_conf in "
                "configs/pose.yaml or verify the image contains an umpire",
            )
            record_failure(source_label, failure.reason)
            return None, failure
        if r.keypoints is None:
            failure = ExtractionFailure(
                "no_keypoints",
                f"detections for {source_label} came back without a "
                "keypoints object — the loaded weights lack a pose head "
                "(kpt_shape must be [17, 3])",
            )
            record_failure(source_label, failure.reason)
            return None, failure
        if r.keypoints.data.shape[-1] != 3:
            failure = ExtractionFailure(
                "keypoints_without_confidence",
                f"keypoint tensor for {source_label} has "
                f"{r.keypoints.data.shape[-1]} channels (no confidence) — "
                "the seam needs an x,y,conf pose checkpoint (RESEARCH.md "
                "Pitfall 4 guard)",
            )
            record_failure(source_label, failure.reason)
            return None, failure
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        box_conf = r.boxes.conf.cpu().numpy()
        kpts_all = r.keypoints.data.cpu().numpy()
        img_h, img_w = r.orig_img.shape[0], r.orig_img.shape[1]
        person_idx, failure = select_umpire(boxes_xyxy, box_conf, img_w, img_h)
        if failure is not None:
            record_failure(source_label, failure.reason)
            return None, failure
        kpts_detected = kpts_all[person_idx]
        kpts_masked = mask_keypoints(kpts_detected, float(self.config["mask_conf"]))
        raw = raw_features(kpts_detected)
        eng = engineered_features(kpts_masked)
        return raw, eng
