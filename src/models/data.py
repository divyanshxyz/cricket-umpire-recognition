"""Phase 3 data seam: frozen features.npz loading/validation, split masks,
the shared label encoder, the mirror-flip column permutation, flip+jitter
augmentation, the test-slice digest guard, and seed_everything.

Locked contracts (03-CONTEXT.md / 03-RESEARCH.md):
- data/processed/features.npz is the ONLY input; the frozen ``split``
  column is consumed as data, never recomputed (no re-splitting logic
  anywhere in Phase 3).
- Every contract violation raises ValueError naming the offending key and
  its observed value — never a silent default (ASVS V5 posture, the
  load_pose_config / write_features_npz guard lineage).
- The loader asserts ZERO byte-identical feature rows across the train and
  test slices of BOTH feature arrays (TRAIN-01/adjacency; upstream
  Hamming<=8 phash dedup guarantees this — the loader re-proves it
  fail-closed at every Phase 3 entry).
- The shared LabelEncoder is fit ONCE on configs/classes.yaml
  canonical_classes; ``le.classes_`` is ALWAYS sorted
  ([Four, NoAction, NoBall, Out, Six, Wide]) and is THE class order for
  every table row, confusion-matrix axis, proba column, and
  classification-report target_names in the phase (RESEARCH Pitfall 4 —
  never re-order).
- Mirror flip is swap-only with NO sign flips: PERM_RAW / PERM_ENG verified
  allclose-1e-9 against the real src.pose_features math on fully-visible
  AND masked keypoints (RESEARCH Pattern 1; probe-3 logic is now a
  permanent regression test in tests/test_augmentation.py).
- Augmentation runs on train-slice rows ONLY, in model-input space (the
  caller passes already-scaled or already-sliced inputs per RESEARCH A2 —
  sigma=0.015 in raw pixel space is a verified no-op): originals then
  mirrored copies in row order, seeded jitter N(0, 0.015) applied to the
  doubled set.
- test_slice_digest: sha256 over the canonical key order (features_raw,
  features_eng, labels, row_ids, split), test rows only, each array's
  test-slice tobytes in sequence (RESEARCH Pattern 5) — recorded at
  training start and asserted unchanged before any metric write
  (CONTEXT-locked frozen-test guard, T-03-01).

This module is torch-free at module level (torch may be imported only
inside function bodies, mirroring the build_features lazy-import
precedent) so --dry-run and the weight-free test suite import without
torch. The data seam never touches src/pose_features.py: Phase 3 consumes
the frozen npz and leaves the Phase 2 extraction seam untouched.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# The exact frozen npz key contract (Phase 2 write_features_npz writer):
# a missing OR extra key is a contract violation — never tolerated.
EXPECTED_KEYS = frozenset(
    {
        "features_raw",
        "features_eng",
        "labels",
        "split",
        "row_ids",
        "failed_row_ids",
        "failure_reasons",
    }
)

# Canonical key order for the test-slice digest (RESEARCH Pattern 5).
DIGEST_KEY_ORDER = ("features_raw", "features_eng", "labels", "row_ids", "split")

# Frozen feature widths + dtypes (the Phase 2 seam contract, CR-02 lineage).
RAW_WIDTH = 51
ENG_WIDTH = 59

# The split column may carry exactly these two values.
VALID_SPLITS = frozenset({"train", "test"})

# Augmentation constants (TRAIN-02 exact sizing, locked CONTEXT).
JITTER_SIGMA = 0.015  # N(0, 0.015) in model-input space (RESEARCH A2)


# ---------------------------------------------------------------------------
# Mirror-flip column permutation — copied EXACTLY from 03-RESEARCH.md
# Pattern 1 (source of truth; verified allclose-1e-9 against the real
# src.pose_features raw_features / mask_keypoints / torso_scale /
# engineered_features math on fully-visible AND masked synthetic keypoints).
# ---------------------------------------------------------------------------
# COCO-17 mirror pairs (matches ultralytics flip_idx;
# docs.ultralytics.com/datasets/pose): nose 0 fixed; eyes 1-2, ears 3-4,
# shoulders 5-6, elbows 7-8, wrists 9-10, hips 11-12, knees 13-14,
# ankles 15-16.
KP_SWAP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])


def _kp_block_perm():
    """Feature column for keypoint k = [3k, 3k+1, 3k+2]; whole triple swaps."""
    perm = np.arange(51)
    for k in range(17):
        perm[3 * k: 3 * k + 3] = [3 * KP_SWAP[k], 3 * KP_SWAP[k] + 1, 3 * KP_SWAP[k] + 2]
    return perm


PERM_RAW = _kp_block_perm()  # 51 columns (Model A + raw)
# eng geometry order (src/pose_features.py lines 335-347): [elbow_l 51,
# elbow_r 52, elev_l 53, elev_r 54, wristnose_l 55, wristnose_r 56,
# span 57, symmetry 58] — the 6 L/R geometry columns swap, span (57) and
# symmetry (58) are invariant.
PERM_ENG = np.concatenate([_kp_block_perm(), [52, 51, 54, 53, 56, 55, 57, 58]])  # 59 columns


def mirror_flip(x: np.ndarray) -> np.ndarray:
    """Swap-only mirror flip: pure column permutation, NO sign flips.

    Public flip entry — infers the permutation from ``x.shape[-1]``:
    51 -> PERM_RAW (raw keypoints), 59 -> PERM_ENG (engineered features),
    anything else -> ValueError naming the observed width. Swap-only (no
    x -> W-x transform) is the locked reading of the TRAIN-02 wording:
    image width W is NOT stored in features.npz (raw features are
    ultralytics pixel coords, RESEARCH A3) and every class-discriminative
    signal is y-derived / geometry-based, which a column swap preserves
    exactly while destroying the L/R x-ordering shortcut. The Model A
    with/without-augmentation probe (Plan 03-03) is the empirical canary.

    Works on any leading shape (last axis is permuted); nothing flips
    sign — elbow angles are arctan2 magnitudes in [0, 180], elevations and
    wrist-vs-nose are y-derived (y unchanged under horizontal mirror),
    span is max-min (invariant), symmetry is |dy| (invariant), and conf
    columns swap WITH their keypoint (they belong to the keypoint, not
    the L/R slot).
    """
    arr = np.asarray(x)
    width = arr.shape[-1]
    if width == RAW_WIDTH:
        return arr[..., PERM_RAW]
    if width == ENG_WIDTH:
        return arr[..., PERM_ENG]
    raise ValueError(
        f"mirror_flip: expected feature width {RAW_WIDTH} (raw keypoints) "
        f"or {ENG_WIDTH} (engineered), got {width} — no flip permutation is "
        "defined for this feature width (fail-closed, no silent default)"
    )


# ---------------------------------------------------------------------------
# Augmentation (TRAIN-02: mirror flip x2 + single seeded jitter pass)
# ---------------------------------------------------------------------------


def augment(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Flip + jitter augmentation of a train slice -> (X_aug, y_aug).

    Contract (locked):
    - ZERO-row X raises ValueError (TRAIN-02/empty: fail-closed, never a
      silent no-op) — augmentation is only ever fed a real train slice.
    - The flip permutation is inferred from the feature width (51 raw /
      59 engineered); unknown widths raise via mirror_flip.
    - X_aug = concatenate([X, mirror_flip(X)]) — ORIGINALS THEN MIRRORED
      COPIES in row order (TRAIN-02/ordering); y_aug = concatenate([y, y]).
    - Jitter N(0, 0.015) drawn from np.random.default_rng(seed) is added to
      the WHOLE doubled set — identical seeds reproduce identical outputs
      (the Generator is local to this call, independent of global RNG
      state, so per-fold/per-model call order cannot perturb it).
    - MODEL-INPUT-SPACE contract (RESEARCH A2 / anti-pattern 3): the caller
      passes already-scaled or already-sliced inputs — post-MinMaxScaler
      rows for Model A, raw engineered rows for B/C/D. sigma=0.015 in raw
      pixel space is a verified physical no-op (raw x spans 0..1679 px).
    - Never fed the test slice: callers run this on train-slice rows only,
      inside each CV fold's train portion (RESEARCH anti-pattern 1 —
      augmenting before the CV split leaks flipped val copies).
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    if X_arr.ndim != 2:
        raise ValueError(
            f"augment: expected a 2-D (rows, features) matrix, got shape "
            f"{X_arr.shape} — fail-closed, no silent default"
        )
    if X_arr.shape[0] == 0:
        raise ValueError(
            "augment: zero-row input — augmentation doubles a train slice "
            "and cannot run on an empty slice (TRAIN-02/empty: fail-closed)"
        )
    if len(y_arr) != X_arr.shape[0]:
        raise ValueError(
            f"augment: X has {X_arr.shape[0]} rows but y has {len(y_arr)} "
            "labels — misaligned inputs are rejected, never silently zipped"
        )
    flipped = mirror_flip(X_arr)  # width-validated permutation
    X_aug = np.concatenate([X_arr, flipped], axis=0)
    y_aug = np.concatenate([y_arr, y_arr], axis=0)
    jitter = np.random.default_rng(seed).normal(0.0, JITTER_SIGMA, X_aug.shape)
    return (X_aug + jitter).astype(np.float32, copy=False), y_aug


# ---------------------------------------------------------------------------
# Frozen npz loading / validation (fail-closed, ASVS V5 posture)
# ---------------------------------------------------------------------------


def canonical_classes(path=None) -> list:
    """configs/classes.yaml canonical_classes (the shared-encoder fit
    source of truth) — on disk [Four, Six, Out, Wide, NoBall, NoAction];
    LabelEncoder sorts this to [Four, NoAction, NoBall, Out, Six, Wide]
    (Pitfall 4). Fail-closed on a malformed config."""
    cfg_path = Path(path) if path is not None else REPO_ROOT / "configs" / "classes.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    classes = cfg.get("canonical_classes")
    if (
        not isinstance(classes, list)
        or not classes
        or not all(isinstance(c, str) and c.strip() for c in classes)
    ):
        raise ValueError(
            f"{cfg_path}: canonical_classes must be a non-empty list of "
            f"class-name strings, got {classes!r} — fix the config "
            "(fail-closed, no silent default)"
        )
    return list(classes)


def load_features_npz(path) -> dict:
    """Load + validate the frozen features.npz -> dict of arrays in stored
    row order (never sorted).

    Fail-closed contract (every violation raises ValueError naming the
    key and its observed value — never a silent default):
    - exact key set {features_raw, features_eng, labels, split, row_ids,
      failed_row_ids, failure_reasons} (missing OR extra key named);
    - features_raw (N, 51) and features_eng (N, 59), both float32
      (wrong width/dtype named) — features_eng carries EXACTLY N rows
      (features_raw's row count; a mismatched eng row count raises the
      contract ValueError naming BOTH observed counts instead of being
      silently zipped or tripping the adjacency guard as a raw
      IndexError);
    - labels / split / row_ids all length N;
    - split values subset of {train, test}; label vocabulary subset of
      configs/classes.yaml canonical_classes;
    - np.isfinite over both feature arrays (NaN/inf counted and named);
    - N == 0 raises naming the empty array; zero test rows raises
      (TRAIN-07/empty — evaluation would be impossible);
    - ZERO train/test byte-identical feature rows across BOTH arrays
      (TRAIN-01/adjacency — upstream phash dedup guarantees this; the
      loader re-proves it at every Phase 3 entry).
    """
    npz_path = Path(path)
    if not npz_path.is_file():
        raise ValueError(
            f"load_features_npz: frozen npz not found at {npz_path} — "
            "Phase 3 consumes the Phase 2 artifact and never rebuilds it "
            "(fail-closed)"
        )
    with np.load(npz_path, allow_pickle=False) as z:
        keys = set(z.files)
        missing = sorted(EXPECTED_KEYS - keys)
        if missing:
            raise ValueError(
                f"load_features_npz: missing required key(s) {missing} in "
                f"{npz_path} (frozen contract: {sorted(EXPECTED_KEYS)}) — "
                "fail-closed, no silent default"
            )
        extra = sorted(keys - EXPECTED_KEYS)
        if extra:
            raise ValueError(
                f"load_features_npz: unexpected extra key(s) {extra} in "
                f"{npz_path} — the frozen contract allows exactly "
                f"{sorted(EXPECTED_KEYS)}"
            )
        raw = np.asarray(z["features_raw"])
        eng = np.asarray(z["features_eng"])
        labels = np.asarray(z["labels"])
        split = np.asarray(z["split"])
        row_ids = np.asarray(z["row_ids"])
        failed_row_ids = np.asarray(z["failed_row_ids"])
        failure_reasons = np.asarray(z["failure_reasons"])

        n = raw.shape[0]
        if n == 0:
            raise ValueError(
                f"load_features_npz: features_raw is empty (0 rows) in "
                f"{npz_path} — the frozen artifact carries zero usable rows"
            )
        if raw.ndim != 2 or raw.shape[1] != RAW_WIDTH:
            raise ValueError(
                f"load_features_npz: features_raw must be (N, {RAW_WIDTH}) "
                f"float32, got shape {raw.shape} — refusing to train on a "
                "mis-shaped frozen artifact (17 x 3 keypoint seam contract)"
            )
        if raw.dtype != np.float32:
            raise ValueError(
                f"load_features_npz: features_raw must be float32, got "
                f"{raw.dtype} — refusing to train on a re-typed artifact"
            )
        if eng.ndim != 2 or eng.shape[1] != ENG_WIDTH:
            raise ValueError(
                f"load_features_npz: features_eng must be (N, {ENG_WIDTH}) "
                f"float32, got shape {eng.shape} — refusing to train on a "
                "mis-shaped frozen artifact (17 x 3 normalized + 8 geometry)"
            )
        if eng.dtype != np.float32:
            raise ValueError(
                f"load_features_npz: features_eng must be float32, got "
                f"{eng.dtype} — refusing to train on a re-typed artifact"
            )
        # Row-count parity (WR-01): features_eng must carry EXACTLY
        # features_raw's row count. Checked BEFORE the labels/split/row_ids
        # length block and BEFORE the adjacency guard — an eng matrix with
        # more rows was silently accepted (downstream slicing consumed the
        # first N rows of a misaligned artifact) and an eng matrix with
        # fewer rows tripped the adjacency guard as a raw IndexError
        # instead of the contract error.
        if eng.shape[0] != n:
            raise ValueError(
                f"load_features_npz: features_eng must have {n} rows to "
                f"match features_raw, got {eng.shape[0]} — misaligned "
                "frozen artifact is rejected, never silently zipped"
            )
        lengths = {"labels": len(labels), "split": len(split), "row_ids": len(row_ids)}
        bad = {k: v for k, v in lengths.items() if v != n}
        if bad:
            raise ValueError(
                f"load_features_npz: labels/split/row_ids must all be length "
                f"{n} (features_raw rows), got {bad} — misaligned frozen "
                "artifact is rejected, never silently zipped"
            )
        unknown_splits = sorted(set(split.tolist()) - VALID_SPLITS)
        if unknown_splits:
            raise ValueError(
                f"load_features_npz: split values outside {sorted(VALID_SPLITS)}: "
                f"{unknown_splits} — the frozen split column carries only "
                "train/test (fail-closed)"
            )
        canonical = set(canonical_classes())
        unknown_labels = sorted(set(labels.tolist()) - canonical)
        if unknown_labels:
            raise ValueError(
                f"load_features_npz: labels outside configs/classes.yaml "
                f"canonical_classes: {unknown_labels} — out-of-vocabulary "
                "labels are rejected, never silently dropped"
            )
        bad_raw = int(np.count_nonzero(~np.isfinite(raw)))
        bad_eng = int(np.count_nonzero(~np.isfinite(eng)))
        if bad_raw or bad_eng:
            raise ValueError(
                f"load_features_npz: non-finite values (NaN/inf) — "
                f"features_raw {bad_raw}, features_eng {bad_eng} — refusing "
                "to train on corrupted features (fail-closed)"
            )
        train_idx, test_idx = split_masks(split)
        if test_idx.size == 0:
            raise ValueError(
                f"load_features_npz: test split has 0 rows in {npz_path} "
                f"({train_idx.size} train rows) — evaluation would be "
                "impossible (TRAIN-07/empty: fail-closed before any metric)"
            )

        # TRAIN-01/adjacency guard: per-row byte views across BOTH feature
        # arrays; any train/test intersection is a leak and raises naming
        # the overlap count.
        train_raw_bytes = {raw[i].tobytes() for i in train_idx}
        train_eng_bytes = {eng[i].tobytes() for i in train_idx}
        overlap = 0
        for i in test_idx:
            if raw[i].tobytes() in train_raw_bytes or eng[i].tobytes() in train_eng_bytes:
                overlap += 1
        if overlap:
            raise ValueError(
                f"load_features_npz: {overlap} test row(s) byte-identical to "
                "train row(s) across features_raw/features_eng — "
                "cross-split feature-vector leak (TRAIN-01/adjacency; "
                "upstream Hamming<=8 phash dedup guarantees 0)"
            )

        return {
            "features_raw": raw,
            "features_eng": eng,
            "labels": labels,
            "split": split,
            "row_ids": row_ids,
            "failed_row_ids": failed_row_ids,
            "failure_reasons": failure_reasons,
        }


def split_masks(split_array) -> tuple:
    """(train_idx, test_idx) from the frozen split column — the split is
    consumed as data, never recomputed (no re-split logic anywhere in
    Phase 3). Indices are stored row order (flatnonzero never sorts)."""
    split = np.asarray(split_array)
    return np.flatnonzero(split == "train"), np.flatnonzero(split == "test")


def fit_label_encoder(canonical_classes_list) -> LabelEncoder:
    """Fit the ONE shared LabelEncoder on the canonical class list.

    Pitfall 4 (pinned): ``le.classes_`` is ALWAYS sorted — canonical
    [Four, Six, Out, Wide, NoBall, NoAction] becomes
    [Four, NoAction, NoBall, Out, Six, Wide]. That sorted order is THE
    class order for every table row, confusion-matrix axis, proba column,
    and classification-report target_names in the phase — accepted
    as-is everywhere, never re-ordered. One encoder for every model:
    separate encoders would silently misalign the Model D proba columns
    (RESEARCH anti-pattern).
    """
    return LabelEncoder().fit(list(canonical_classes_list))


# ---------------------------------------------------------------------------
# Frozen-test guard (RESEARCH Pattern 5, CONTEXT-locked, T-03-01)
# ---------------------------------------------------------------------------


def test_slice_digest(npz_path) -> str:
    """sha256 over the canonical key order (features_raw, features_eng,
    labels, row_ids, split), TEST ROWS ONLY, each array's test-slice
    tobytes in sequence (RESEARCH Pattern 5 — the frozen-test guard).
    Recorded at training start and asserted unchanged before any metric
    write: a mismatch refuses the run (fail-closed, never
    warn-and-continue) so no metric can silently drift to a different
    test population."""
    with np.load(Path(npz_path), allow_pickle=False) as z:
        te = np.asarray(z["split"]) == "test"
        digest = hashlib.sha256()
        for key in DIGEST_KEY_ORDER:
            digest.update(z[key][te].tobytes())
    return digest.hexdigest()


def seed_everything(seed: int = 42) -> None:
    """random + numpy + torch seeds and torch.set_num_threads(1)
    (RESEARCH Pattern 2: fixed reduction order for byte-identical CPU
    re-runs — thread count is a determinism-relevant setting and is
    recorded in every report param block). torch is imported inside the
    function body — this module stays torch-free at module level."""
    random.seed(seed)
    np.random.seed(seed)
    import torch  # lazy: keeps --dry-run and the weight-free tests torch-free

    torch.manual_seed(seed)
    torch.set_num_threads(1)
