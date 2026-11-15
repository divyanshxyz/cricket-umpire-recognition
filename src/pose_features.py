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
The one sanctioned raise is a model-LOAD failure (WR-02): a missing or
corrupt weights path is a systemic config error that fails fast before any
row is processed — never N identical per-image decode_error rows.
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
    "keypoints_without_confidence" | "decode_error" |
    "no_person_in_sequence" (whole-sequence failure). ``detail`` names what
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

    Score = (bbox_area / image_area) x (1 - 2*|cx_norm - 0.5|), computed in
    float64 with NO rounding anywhere; max wins. Valid input domain (IN-05):
    for box centers within image bounds both terms are in [0, 1] —
    ultralytics boxes are image-bounded in the real call path — but
    centrality itself is NOT bounded as a pure function: it goes negative
    for a center outside the image (e.g. -0.5 for a box spanning x 50-200
    on a 100-px image). The formula is the locked CONTEXT composite — no
    clamping; this docstring is the contract. The exact-equality tie set is
    resolved by maximum detection confidence (order-independent); a full
    confidence tie INSIDE that set resolves lexicographically by box
    coordinates (x0, y0, x1, y1) — selection is fully order-independent
    under detection-order permutation (WR-01: a bare argmax over the tied
    subset would silently pick the first index and follow detection order).
    Returns (person_index, None) or (None, ExtractionFailure("no_person",
    ...)).

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
        # Tie-break 1: maximum detection confidence.
        tied = ties[confs[ties] == confs[ties].max()]
        if len(tied) > 1:
            # Tie-break 2 (WR-01): full score AND confidence tie —
            # lexicographic by box coords (x0 primary; np.lexsort's LAST
            # key is primary). Deterministic under any detection-order
            # permutation; boxes with identical coords are the same
            # detection, so the order is total over distinct boxes.
            order = np.lexsort(
                (boxes[tied, 3], boxes[tied, 2], boxes[tied, 1], boxes[tied, 0])
            )
            best = int(tied[order[0]])
        else:
            best = int(tied[0])
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


def torso_scale(kpts_masked, mask_conf: float = 0.20, row_id: str = "torso_scale"):
    """Torso scale fallback chain (CONTEXT locked; zero-length scales
    (<= 1e-6) fall through to the next stage):

    1. all four of L/R shoulder (COCO 5/6) and L/R hip (COCO 11/12) conf
       >= mask_conf -> norm(mid_shoulder - mid_hip) as ("torso", s);
    2. otherwise both shoulders visible (EITHER hip conf < mask_conf, or
       the torso length is degenerate) -> shoulder-width norm as
       ("shoulder_width", s);
    3. otherwise (1.0, "unit") plus a record_scale_warning entry — the
       CONTEXT 'logged warning' for the unit-scale fallback.

    ``row_id`` is the recorded warning's row/image identity (IN-04): the
    extract() seam threads ``source_label`` and the sequence path threads
    ``sequence_frame_{i}``, so unit-scale fallbacks are attributable per
    row, not just countable; the default keeps the legacy label for direct
    weight-free callers.

    Returns (scale, source). Pure math except the terminal-warning event
    record (the module-level recorder is the sanctioned side channel, so
    the weight-free test can assert the warning fires).
    """
    kpts = np.asarray(kpts_masked, dtype=np.float64)
    sh_ok = bool(
        kpts[L_SHOULDER, 2] >= mask_conf and kpts[R_SHOULDER, 2] >= mask_conf
    )
    hip_ok = bool(kpts[L_HIP, 2] >= mask_conf and kpts[R_HIP, 2] >= mask_conf)
    if hip_ok and sh_ok:
        mid_sh = (kpts[L_SHOULDER, :2] + kpts[R_SHOULDER, :2]) / 2.0
        mid_hip = (kpts[L_HIP, :2] + kpts[R_HIP, :2]) / 2.0
        s = float(np.linalg.norm(mid_sh - mid_hip))
        if s > _SCALE_MIN:
            return s, "torso"
    if sh_ok:
        s = float(np.linalg.norm(kpts[L_SHOULDER, :2] - kpts[R_SHOULDER, :2]))
        if s > _SCALE_MIN:
            return s, "shoulder_width"
    record_scale_warning(
        row_id,
        "torso and shoulder-width scales both unusable — falling back to "
        "scale=1.0 (unit); masked coords are 0 anyway",
    )
    return 1.0, "unit"


def _angle_deg(v1, v2):
    """Angle at a vertex between vectors v1 and v2 via arctan2(|cross|, dot)
    — degrees in [0, 180] by construction (authored [ASSUMED] A1 formula)."""
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    return float(np.degrees(np.arctan2(abs(cross), dot)))


def _geometry_features(kpts, scale, mask_conf):
    """The 8 geometry features in the frozen order [elbow_l, elbow_r,
    elev_l, elev_r, wristnose_l, wristnose_r, span, symmetry] using the
    authored [ASSUMED] A1 formulas: elbow angle at the elbow vertex between
    (shoulder - elbow) and (wrist - elbow) via arctan2(|cross|, dot) in
    degrees; elevation = (shoulder_y - wrist_y)/scale (positive when the
    wrist is above the shoulder); wrist-vs-nose = (nose_y - wrist_y)/scale
    (positive when the wrist is above the nose); span = (max x - min x over
    conf-visible keypoints)/scale; symmetry = |y_Lwrist - y_Rwrist|/scale.

    Any term computed from a keypoint pair with conf < mask_conf yields the
    0.0 sentinel — never NaN (researcher disposition 2). Visible-mask
    definition: conf >= mask_conf (the exact-0.20 boundary)."""
    vis = kpts[:, 2] >= mask_conf
    elbow_l = (
        _angle_deg(
            kpts[L_SHOULDER, :2] - kpts[L_ELBOW, :2],
            kpts[L_WRIST, :2] - kpts[L_ELBOW, :2],
        )
        if (vis[L_SHOULDER] and vis[L_ELBOW] and vis[L_WRIST])
        else 0.0
    )
    elbow_r = (
        _angle_deg(
            kpts[R_SHOULDER, :2] - kpts[R_ELBOW, :2],
            kpts[R_WRIST, :2] - kpts[R_ELBOW, :2],
        )
        if (vis[R_SHOULDER] and vis[R_ELBOW] and vis[R_WRIST])
        else 0.0
    )
    elev_l = (
        (kpts[L_SHOULDER, 1] - kpts[L_WRIST, 1]) / scale
        if (vis[L_SHOULDER] and vis[L_WRIST])
        else 0.0
    )  # positive when the wrist is above the shoulder
    elev_r = (
        (kpts[R_SHOULDER, 1] - kpts[R_WRIST, 1]) / scale
        if (vis[R_SHOULDER] and vis[R_WRIST])
        else 0.0
    )
    wristnose_l = (
        (kpts[NOSE, 1] - kpts[L_WRIST, 1]) / scale
        if (vis[NOSE] and vis[L_WRIST])
        else 0.0
    )  # positive when the wrist is above the nose
    wristnose_r = (
        (kpts[NOSE, 1] - kpts[R_WRIST, 1]) / scale
        if (vis[NOSE] and vis[R_WRIST])
        else 0.0
    )
    if vis.any():
        vis_x = kpts[vis, 0]
        span = float(vis_x.max() - vis_x.min()) / scale
    else:
        span = 0.0  # no visible keypoints — 0.0 sentinel, not NaN
    symmetry = (
        abs(kpts[L_WRIST, 1] - kpts[R_WRIST, 1]) / scale
        if (vis[L_WRIST] and vis[R_WRIST])
        else 0.0
    )
    return np.array(
        [elbow_l, elbow_r, elev_l, elev_r, wristnose_l, wristnose_r, span, symmetry],
        dtype=np.float64,
    )


def engineered_features(kpts_masked, scale, mask_conf: float = 0.20):
    """59-D engineered vector (float32): 17 x (torso-normalized x_hat,
    y_hat, conf) from the MASKED keypoints (masked entries stay 0.0 because
    their x/y are zeroed, conf is preserved unnormalized) + the 8 geometry
    features in the frozen order [elbow_l, elbow_r, elev_l, elev_r,
    wristnose_l, wristnose_r, span, symmetry] (must_haves truth 11)."""
    kpts = np.asarray(kpts_masked, dtype=np.float64)
    scale = float(scale)
    normalized = np.empty_like(kpts)
    normalized[:, :2] = kpts[:, :2] / scale
    normalized[:, 2] = kpts[:, 2]
    geometry = _geometry_features(kpts, scale, mask_conf)
    return np.concatenate([normalized.reshape(-1), geometry]).astype(np.float32)


def engineer_extended_features(
    features_raw: np.ndarray,
    features_eng: np.ndarray,
    mask_conf: float = 0.20,
) -> np.ndarray:
    """Computes 22 relational pose features proven in Spike 001a and concatenates
    them onto features_eng to yield an (N, 81) or (81,) float32 array (FUSE-01).

    Features (indices 59:81):
      59: Wrist-to-wrist distance (Euclidean norm)
      60: Wrist-to-wrist horizontal dx (r_wrist.x - l_wrist.x)
      61: Wrist-to-wrist vertical dy (r_wrist.y - l_wrist.y)
      62: Left wrist-to-nose vertical elevation (nose.y - l_wrist.y)
      63: Right wrist-to-nose vertical elevation (nose.y - r_wrist.y)
      64: Max wrist-to-nose elevation (max(dy_l, dy_r))
      65: Min wrist-to-nose elevation (min(dy_l, dy_r))
      66: Abs diff of wrist-to-nose elevation (|dy_l - dy_r|)
      67: Left wrist-to-hip normalized distance
      68: Right wrist-to-hip normalized distance
      69: Left wrist-to-left shoulder distance (same side)
      70: Right wrist-to-right shoulder distance (same side)
      71: Left wrist-to-right shoulder distance (cross)
      72: Right wrist-to-left shoulder distance (cross)
      73: Left elbow-to-shoulder dx (l_el.x - l_sh.x)
      74: Left elbow-to-shoulder dy elevation (l_sh.y - l_el.y)
      75: Right elbow-to-shoulder dx (r_el.x - r_sh.x)
      76: Right elbow-to-shoulder dy elevation (r_sh.y - r_el.y)
      77: Upper-body bounding box span_x (width)
      78: Upper-body bounding box span_y (height)
      79: Upper-body bounding box aspect ratio (span_x / (span_y + 1e-6))
      80: Mean arm keypoint confidence (COCO 5..10)
    """
    f_eng = np.asarray(features_eng, dtype=np.float32)
    is_1d = (f_eng.ndim == 1)
    if is_1d:
        f_eng = f_eng[None, :]

    N = f_eng.shape[0]
    kpts = f_eng[:, :51].reshape(N, 17, 3)

    # 1. Wrist-to-wrist (59:62)
    ww_dist = np.linalg.norm(kpts[:, R_WRIST, :2] - kpts[:, L_WRIST, :2], axis=1, keepdims=True)
    ww_dx = (kpts[:, R_WRIST, 0] - kpts[:, L_WRIST, 0])[:, None]
    ww_dy = (kpts[:, R_WRIST, 1] - kpts[:, L_WRIST, 1])[:, None]

    # 2. Wrist-to-nose vertical elevation dy (62:67)
    wn_dy_l = (kpts[:, NOSE, 1] - kpts[:, L_WRIST, 1])[:, None]
    wn_dy_r = (kpts[:, NOSE, 1] - kpts[:, R_WRIST, 1])[:, None]
    wn_dy_max = np.maximum(wn_dy_l, wn_dy_r)
    wn_dy_min = np.minimum(wn_dy_l, wn_dy_r)
    wn_dy_diff = np.abs(wn_dy_l - wn_dy_r)

    # 3. Wrist-to-hip normalized distances (67:69)
    wh_dist_l = np.linalg.norm(kpts[:, L_WRIST, :2] - kpts[:, L_HIP, :2], axis=1, keepdims=True)
    wh_dist_r = np.linalg.norm(kpts[:, R_WRIST, :2] - kpts[:, R_HIP, :2], axis=1, keepdims=True)

    # 4. Wrist-to-shoulder distances (same side and cross-shoulder) (69:73)
    ws_same_l = np.linalg.norm(kpts[:, L_WRIST, :2] - kpts[:, L_SHOULDER, :2], axis=1, keepdims=True)
    ws_same_r = np.linalg.norm(kpts[:, R_WRIST, :2] - kpts[:, R_SHOULDER, :2], axis=1, keepdims=True)
    ws_cross_l = np.linalg.norm(kpts[:, L_WRIST, :2] - kpts[:, R_SHOULDER, :2], axis=1, keepdims=True)
    ws_cross_r = np.linalg.norm(kpts[:, R_WRIST, :2] - kpts[:, L_SHOULDER, :2], axis=1, keepdims=True)

    # 5. Elbow-to-shoulder dx and dy elevations (73:77)
    es_dx_l = (kpts[:, L_ELBOW, 0] - kpts[:, L_SHOULDER, 0])[:, None]
    es_dy_l = (kpts[:, L_SHOULDER, 1] - kpts[:, L_ELBOW, 1])[:, None]
    es_dx_r = (kpts[:, R_ELBOW, 0] - kpts[:, R_SHOULDER, 0])[:, None]
    es_dy_r = (kpts[:, R_SHOULDER, 1] - kpts[:, R_ELBOW, 1])[:, None]

    # 6. Upper body bounding box span_x, span_y, aspect ratio (77:80)
    span_x = np.zeros((N, 1), dtype=np.float32)
    span_y = np.zeros((N, 1), dtype=np.float32)
    for i in range(N):
        vis = kpts[i, :13, 2] >= mask_conf
        if vis.any():
            xs = kpts[i, :13, 0][vis]
            ys = kpts[i, :13, 1][vis]
            span_x[i] = xs.max() - xs.min()
            span_y[i] = ys.max() - ys.min()
    aspect = span_x / (span_y + 1e-6)

    # 7. Mean upper-body arm keypoint confidence (index 80)
    arm_conf = np.mean(kpts[:, 5:11, 2], axis=1, keepdims=True)

    relational = np.hstack([
        ww_dist, ww_dx, ww_dy,
        wn_dy_l, wn_dy_r, wn_dy_max, wn_dy_min, wn_dy_diff,
        wh_dist_l, wh_dist_r,
        ws_same_l, ws_same_r, ws_cross_l, ws_cross_r,
        es_dx_l, es_dy_l, es_dx_r, es_dy_r,
        span_x, span_y, aspect,
        arm_conf,
    ]).astype(np.float32)

    extended = np.hstack([f_eng, relational])
    extended = np.nan_to_num(extended, nan=0.0, posinf=0.0, neginf=0.0)

    if is_1d:
        return extended[0]
    return extended


def velocity_features(prev_xy, cur_xy, scale):
    """Frame-to-frame velocity (8,) float32: (cur - prev) for VELOCITY_JOINTS
    (COCO 7,8,9,10 — L/R elbow, L/R wrist) divided by the CURRENT frame's
    torso scale, flattened (dx, dy) per joint order. Computed from the
    stored masked coords as-is [A4] — deltas crossing a masked frame are
    deterministic but semantically degraded; downstream conf columns let
    models downweight them. prev_xy / cur_xy: (17, 2) masked pixel coords."""
    prev = np.asarray(prev_xy, dtype=np.float64).reshape(17, -1)[:, :2]
    cur = np.asarray(cur_xy, dtype=np.float64).reshape(17, -1)[:, :2]
    joints = list(VELOCITY_JOINTS)
    delta = cur[joints] - prev[joints]  # (4, 2): (dx, dy) per joint order
    return (delta / float(scale)).astype(np.float32).reshape(-1)


def assemble_sequence(per_frame_rows):
    """Pure (T, 118) sequence assembly — weight-free testable.

    per_frame_rows: one entry per frame, either a success 4-tuple
    (raw51, eng59, scale, masked_xy) — masked_xy is that frame's masked
    (17, 2) pixel coords, which become the NEXT frame's prev — or a failure
    pair (None, ExtractionFailure) exactly as extract() returns it.

    Row layout (frozen, CONTEXT D14): columns [0:51] raw, [51:110]
    engineered, [110:118] velocity; the first frame's velocity is zeros(8),
    frame i > 0 uses velocity_features(prev, cur, frame i's scale). T is
    returned as-is — padding/truncating to 15 is Plan 02's pad_or_truncate,
    not this function.

    Whole-sequence failure (researcher disposition 1 / A10): ANY failure row
    fails the whole sequence — recorded, then (None,
    ExtractionFailure("no_person_in_sequence", ...)) naming the failing
    frame index. No zero-feature rows are ever emitted.
    """
    feature_rows = []
    prev_xy = None
    for frame_idx, row in enumerate(per_frame_rows):
        if row is None or (len(row) == 2 and row[0] is None):
            failure = (
                row[1]
                if row is not None
                else ExtractionFailure("no_person", "frame produced no result")
            )
            record_failure(f"sequence_frame_{frame_idx}", failure.reason)
            return None, ExtractionFailure(
                "no_person_in_sequence",
                f"frame {frame_idx} failed extraction ({failure.reason}): "
                f"{failure.detail} — the sequence fails as a whole, no "
                "zero-feature rows (researcher disposition 1)",
            )
        raw, eng, scale, masked_xy = row
        if prev_xy is None:
            velocity = np.zeros(8, dtype=np.float32)
        else:
            velocity = velocity_features(prev_xy, masked_xy, scale)
        feature_rows.append(
            np.concatenate(
                [
                    np.asarray(raw, dtype=np.float32),
                    np.asarray(eng, dtype=np.float32),
                    velocity,
                ]
            )
        )
        prev_xy = masked_xy
    return np.stack(feature_rows, axis=0), None


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

    def warm_up(self):
        """Eagerly load the model (WR-02): batch callers invoke this ONCE
        before the row loop so a missing/corrupt weights path fails fast as
        the config error it is — one clear traceback, zero rows processed —
        never as N identical per-image decode_error rows after minutes of
        CPU. Returns self for chaining."""
        self._ensure_model()
        return self

    def extract(self, image):
        """image -> (raw51 float32, engineered59 float32) on success, or
        (None, ExtractionFailure) on any guard trip. Every failure is typed
        and recorded in the module event list; per-image inference is
        contained (T-02-01: a corrupt/hostile file costs one typed row, not
        a crashed batch). Model loading is deliberately NOT contained
        (WR-02): a missing/corrupt weights path raises once, loudly, before
        any row is processed — batch callers should warm_up() first."""
        source_label = str(image) if isinstance(image, (str, Path)) else "ndarray"
        # WR-02: OUTSIDE the per-image containment — model-load failures
        # (missing/corrupt weights, bad device) are systemic config errors,
        # not per-image decode failures.
        model = self._ensure_model()
        try:
            results = model.predict(
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
        if r.keypoints.data.shape[-2] != 17:
            # CR-02: the keypoint-count dim is guarded too. A non-17 count
            # (any pose checkpoint other than COCO-17, e.g. a (24, 3)
            # tensor) would otherwise flow through as SUCCESS rows: raw
            # (72,) + eng (80,) np.stack'd into a silently mis-shaped
            # frozen features.npz while the build reports ok — exactly the
            # T-02-04 silent-corruption class. Same typed reason as the
            # missing-pose-head trip (already in the closed set).
            failure = ExtractionFailure(
                "no_keypoints",
                f"keypoint tensor for {source_label} has shape "
                f"{tuple(r.keypoints.data.shape)} — the seam needs a "
                "[17, 3] (x, y, conf) COCO-17 checkpoint; check kpt_shape "
                "in the weights",
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
        mask_conf = float(self.config["mask_conf"])
        kpts_masked = mask_keypoints(kpts_detected, mask_conf)
        raw = raw_features(kpts_detected)
        # IN-04: the real row/image identity rides on the unit-scale
        # fallback warning so build reports can attribute the event.
        scale, _source = torso_scale(kpts_masked, mask_conf, source_label)
        eng = engineered_features(kpts_masked, scale, mask_conf)
        return raw, eng

    def extract_sequence_features(self, frames):
        """frames (list of BGR HWC uint8 ndarrays — the cv2.VideoCapture
        frame contract) -> (T, 118) float32 via the pure assemble_sequence,
        or (None, ExtractionFailure("no_person_in_sequence", ...)) naming
        the first failing frame index. Reuses self.extract per frame (same
        extractor, RESEARCH.md contract) and derives each frame's masked
        coords + torso scale from the returned raw vector with the same
        pure math extract() used — exactly one model call per frame."""
        if not frames:
            failure = ExtractionFailure(
                "no_person_in_sequence",
                "empty frames list — nothing to extract (ASVS V5: rejected "
                "and counted, never silently defaulted)",
            )
            record_failure("empty_sequence", failure.reason)
            return None, failure
        mask_conf = float(self.config["mask_conf"])
        per_frame_rows = []
        for frame_idx, frame in enumerate(frames):
            raw, payload = self.extract(frame)
            if raw is None:
                per_frame_rows.append((None, payload))  # (None, ExtractionFailure)
                continue
            kpts_masked = mask_keypoints(
                np.asarray(raw, dtype=np.float64).reshape(17, 3), mask_conf
            )
            # IN-04: frame identity rides on the unit-scale fallback
            # warning, matching the failure events' sequence_frame_N labels.
            scale, _source = torso_scale(
                kpts_masked, mask_conf, f"sequence_frame_{frame_idx}"
            )
            per_frame_rows.append((raw, payload, scale, kpts_masked[:, :2]))
        return assemble_sequence(per_frame_rows)
