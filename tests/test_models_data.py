"""Loader/encoder/digest contract tests for the Phase 3 data seam
(weight-free except the gated real-data test).

Pins: the fail-closed loader contract over synthetic npz files (missing /
extra key, wrong width, wrong dtype, features_eng row-count mismatch —
WR-01: 11-vs-9 was silently accepted, 5-vs-9 tripped a raw IndexError,
both now raise the contract ValueError naming both observed counts —
NaN, out-of-vocabulary label, out-of-vocabulary split value, zero total
rows, zero test rows), the cross-split adjacency trip (TRAIN-01), the
sorted shared-encoder order (Pitfall 4), test-slice digest stability +
tamper sensitivity (TEST rows only), the torch-free module source guard,
and the REAL frozen npz contract re-proven at Phase 3 entry (shapes,
334/84, all six classes in both slices, zero cross-slice row overlap).
No torch, no weights.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from src.models.data import (
    canonical_classes,
    fit_label_encoder,
    load_features_npz,
    sha256_of,
    split_masks,
    verify_artifact_sha256,
)
from src.models.data import test_slice_digest as compute_test_slice_digest

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_NPZ = REPO_ROOT / "data" / "processed" / "features.npz"

# The yaml canonical order (fit source of truth) — LabelEncoder sorts to a
# DIFFERENT order (Pitfall 4); both are pinned below.
YAML_ORDER = ["Four", "Six", "Out", "Wide", "NoBall", "NoAction"]
SORTED_ORDER = ["Four", "NoAction", "NoBall", "Out", "Six", "Wide"]


def _base_arrays(n: int = 9, n_test: int = 3) -> dict:
    """A clean synthetic features.npz payload: distinct float32 rows (no
    adjacency overlap), all six canonical labels present, train-first
    stored row order, one typed no_person failure row."""
    n_train = n - n_test
    return {
        "features_raw": (np.arange(n * 51, dtype=np.float32) * 1.5).reshape(n, 51),
        "features_eng": (np.arange(n * 59, dtype=np.float32) * 1.5).reshape(n, 59),
        "labels": np.array([YAML_ORDER[i % 6] for i in range(n)]),
        "split": np.array(["train"] * n_train + ["test"] * n_test),
        "row_ids": np.array([f"r_{i:05d}" for i in range(n)]),
        "failed_row_ids": np.array(["f_00000"]),
        "failure_reasons": np.array(["no_person"]),
    }


def _write_npz(tmp_path: Path, arrays: dict, name: str = "synthetic.npz") -> Path:
    path = tmp_path / name
    np.savez(path, **arrays)
    return path


# ---------------------------------------------------------------------------
# Loader fail-closed contract (each violation names key + observed value)
# ---------------------------------------------------------------------------


def test_loader_rejects_missing_key(tmp_path):
    arrays = _base_arrays()
    del arrays["row_ids"]
    with pytest.raises(ValueError, match="row_ids"):
        load_features_npz(_write_npz(tmp_path, arrays))


def test_loader_rejects_extra_key(tmp_path):
    arrays = _base_arrays()
    arrays["surprise"] = np.array([1.0])
    with pytest.raises(ValueError, match="surprise"):
        load_features_npz(_write_npz(tmp_path, arrays))


def test_loader_rejects_raw_width_50(tmp_path):
    arrays = _base_arrays()
    arrays["features_raw"] = arrays["features_raw"][:, :50]
    with pytest.raises(ValueError, match=r"\(9, 50\)"):
        load_features_npz(_write_npz(tmp_path, arrays))


def test_loader_rejects_float64_dtype(tmp_path):
    arrays = _base_arrays()
    arrays["features_raw"] = arrays["features_raw"].astype(np.float64)
    with pytest.raises(ValueError, match="float64"):
        load_features_npz(_write_npz(tmp_path, arrays))


def test_loader_rejects_eng_row_count_longer_than_raw(tmp_path):
    # WR-01: features_eng with MORE rows than features_raw (11 vs 9) was
    # silently accepted — the loader returned the mis-shaped eng matrix and
    # downstream slicing consumed the first N rows of a provably misaligned
    # artifact. The contract ValueError names BOTH observed counts and
    # fires before the adjacency guard.
    arrays = _base_arrays()
    arrays["features_eng"] = (np.arange(11 * 59, dtype=np.float32) * 1.5).reshape(11, 59)
    with pytest.raises(ValueError, match=r"features_eng must have 9 rows"):
        load_features_npz(_write_npz(tmp_path, arrays))


def test_loader_rejects_eng_row_count_shorter_than_raw(tmp_path):
    # WR-01 counterpart: features_eng with FEWER rows than features_raw
    # (5 vs 9) used to fail closed only incidentally — a raw
    # "IndexError: index 5 is out of bounds" from the adjacency guard's
    # eng[i].tobytes() instead of the contract ValueError naming the key
    # and the observed value. Both mismatch directions now raise the same
    # contract error.
    arrays = _base_arrays()
    arrays["features_eng"] = (np.arange(5 * 59, dtype=np.float32) * 1.5).reshape(5, 59)
    with pytest.raises(ValueError, match=r"got 5"):
        load_features_npz(_write_npz(tmp_path, arrays))


def test_loader_rejects_nan_in_features_eng(tmp_path):
    arrays = _base_arrays()
    eng = arrays["features_eng"].copy()
    eng[2, 7] = np.nan
    arrays["features_eng"] = eng
    with pytest.raises(ValueError, match="non-finite"):
        load_features_npz(_write_npz(tmp_path, arrays))


def test_loader_rejects_label_outside_canonical_six(tmp_path):
    arrays = _base_arrays()
    labels = arrays["labels"].copy()
    labels[0] = "LegBye"
    arrays["labels"] = labels
    with pytest.raises(ValueError, match="LegBye"):
        load_features_npz(_write_npz(tmp_path, arrays))


def test_loader_rejects_split_value_outside_train_test(tmp_path):
    arrays = _base_arrays()
    split = arrays["split"].copy()
    split[0] = "validate"
    arrays["split"] = split
    # NOTE: numpy's fixed-width U5 dtype truncates the assigned value to
    # 'valid' — the loader names the OBSERVED (stored) value, which is
    # exactly what a researcher needs to see.
    with pytest.raises(ValueError, match=r"\['valid'\]"):
        load_features_npz(_write_npz(tmp_path, arrays))


def test_loader_rejects_zero_total_rows(tmp_path):
    arrays = _base_arrays()
    for key in ("features_raw", "features_eng", "labels", "split", "row_ids"):
        arrays[key] = arrays[key][:0]
    with pytest.raises(ValueError, match="0 rows"):
        load_features_npz(_write_npz(tmp_path, arrays))


def test_loader_rejects_zero_test_rows(tmp_path):
    # TRAIN-07/empty lift: an all-train npz raises BEFORE any metric could
    # exist — evaluation on an empty test slice is impossible.
    arrays = _base_arrays(n=6, n_test=0)
    with pytest.raises(ValueError, match="test split has 0 rows"):
        load_features_npz(_write_npz(tmp_path, arrays))


# ---------------------------------------------------------------------------
# Adjacency guard (TRAIN-01) + stored row order
# ---------------------------------------------------------------------------


def test_loader_rejects_cross_split_row_overlap(tmp_path):
    # One train row byte-equals one test row -> ValueError naming the
    # overlap count (the leak-free claim asserted fail-closed).
    arrays = _base_arrays(n=9, n_test=3)
    arrays["features_raw"][8] = arrays["features_raw"][0]  # test row 8 == train row 0
    with pytest.raises(ValueError, match="byte-identical"):
        load_features_npz(_write_npz(tmp_path, arrays))


def test_loader_passes_clean_synthetic_and_preserves_row_order(tmp_path):
    # The clean counterpart: stored row order preserved end-to-end (nothing
    # sorts rows between the loader and the caller — TRAIN-01/ordering).
    arrays = _base_arrays(n=12, n_test=4)
    loaded = load_features_npz(_write_npz(tmp_path, arrays))
    assert loaded["features_raw"].shape == (12, 51)
    assert np.array_equal(loaded["features_raw"], arrays["features_raw"])
    assert np.array_equal(loaded["features_eng"], arrays["features_eng"])
    assert loaded["labels"].tolist() == arrays["labels"].tolist()
    assert loaded["row_ids"].tolist() == arrays["row_ids"].tolist()
    train_idx, test_idx = split_masks(loaded["split"])
    assert train_idx.tolist() == list(range(8))
    assert test_idx.tolist() == [8, 9, 10, 11]
    assert loaded["failed_row_ids"].tolist() == ["f_00000"]


# ---------------------------------------------------------------------------
# Shared encoder (Pitfall 4)
# ---------------------------------------------------------------------------


def test_canonical_classes_read_from_yaml():
    # The fit source of truth keeps the yaml order.
    assert canonical_classes() == YAML_ORDER


def test_encoder_sorted_order_and_roundtrip():
    # le.classes_ is ALWAYS sorted: [Four, NoAction, NoBall, Out, Six,
    # Wide] regardless of the yaml order — THE class order for every
    # table row, axis, and proba column in the phase.
    le = fit_label_encoder(canonical_classes())
    assert list(le.classes_) == SORTED_ORDER
    y = le.transform(["Four", "Wide", "NoAction"])
    assert y.tolist() == [0, 5, 1]
    assert le.inverse_transform(y).tolist() == ["Four", "Wide", "NoAction"]


# ---------------------------------------------------------------------------
# Test-slice digest (stability + tamper sensitivity)
# ---------------------------------------------------------------------------


def test_digest_stable_across_repeated_computations(tmp_path):
    arrays = _base_arrays()
    path = _write_npz(tmp_path, arrays, name="d1.npz")
    digest = compute_test_slice_digest(path)
    assert digest == compute_test_slice_digest(path)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_digest_flips_on_test_row_tamper_and_holds_on_train_row_tamper(tmp_path):
    arrays = _base_arrays()
    baseline = compute_test_slice_digest(_write_npz(tmp_path, arrays, name="d_base.npz"))
    # Tamper ONE test-row feature value (row 8 is a test row) -> digest flips.
    tampered = dict(arrays)
    eng = arrays["features_eng"].copy()
    eng[8, 3] += 1.0
    tampered["features_eng"] = eng
    digest_tampered = compute_test_slice_digest(
        _write_npz(tmp_path, tampered, name="d_test.npz")
    )
    assert digest_tampered != baseline
    # Tamper only a TRAIN row (row 1) -> digest unchanged (test rows only).
    train_tampered = dict(arrays)
    raw = arrays["features_raw"].copy()
    raw[1, 3] += 1.0
    train_tampered["features_raw"] = raw
    digest_train = compute_test_slice_digest(
        _write_npz(tmp_path, train_tampered, name="d_train.npz")
    )
    assert digest_train == baseline


# ---------------------------------------------------------------------------
# Source guard: the data seam stays torch-free at module level
# ---------------------------------------------------------------------------


def test_data_module_source_has_no_top_level_heavy_imports():
    # The test_build_features.py heavy-import guard pattern applied to the
    # data seam: no line-start torch/ultralytics/cv2 import in
    # src/models/data.py (the lazy seed_everything torch import is
    # function-body indented and exempt by the caret anchor).
    import inspect

    import src.models.data as data_module

    source = inspect.getsource(data_module)
    for dep in ("torch", "ultralytics", "cv2"):
        assert not re.search(rf"^import {dep}\b", source, re.M), dep
        assert not re.search(rf"^from {dep}\b", source, re.M), dep


# ---------------------------------------------------------------------------
# Real-data gated test (the leak-free claim re-proven at Phase 3 entry)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not REAL_NPZ.exists(), reason="frozen data/processed/features.npz not present"
)
def test_real_frozen_npz_contract_and_leak_free():
    # Full fail-closed load of the REAL frozen artifact — the adjacency
    # guard re-proves zero cross-split row overlap on every Phase 3 entry.
    loaded = load_features_npz(REAL_NPZ)
    assert loaded["features_raw"].shape == (418, 51)
    assert loaded["features_raw"].dtype == np.float32
    assert loaded["features_eng"].shape == (418, 59)
    assert loaded["features_eng"].dtype == np.float32
    train_idx, test_idx = split_masks(loaded["split"])
    assert train_idx.size == 334
    assert test_idx.size == 84
    six = set(loaded["labels"].tolist())
    assert six == set(SORTED_ORDER)
    for label in sorted(six):
        is_label = loaded["labels"] == label
        assert (is_label & (loaded["split"] == "train")).any(), label
        assert (is_label & (loaded["split"] == "test")).any(), label


# ---------------------------------------------------------------------------
# Artifact integrity helpers (WR-04: bind pickle loads to provenance)
# ---------------------------------------------------------------------------


def test_sha256_of_matches_hashlib_file_digest(tmp_path):
    import hashlib

    payload = b"artifact-bytes-for-the-integrity-helper"
    path = tmp_path / "artifact.bin"
    path.write_bytes(payload)
    assert sha256_of(path) == hashlib.sha256(payload).hexdigest()
    assert re.fullmatch(r"[0-9a-f]{64}", sha256_of(path))
    # The digest is a pure function of the bytes — same file, same digest.
    assert sha256_of(path) == sha256_of(path)


def test_verify_artifact_sha256_accepts_match_case_insensitive_and_none(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"stable-bytes")
    digest = sha256_of(path)
    # Matching expectation -> verified digest returned (load may proceed).
    assert verify_artifact_sha256(path, digest, artifact_name="x.bin") == digest
    # Uppercase hex from a caller is normalized (case-insensitive compare).
    assert verify_artifact_sha256(path, digest.upper()) == digest
    # None expectation -> check skipped, recomputed digest returned (the
    # bare interactive reload stays usable; the runner always passes the
    # sha256 it recorded in the same run).
    assert verify_artifact_sha256(path, None) == digest


def test_verify_artifact_sha256_refuses_mismatch_naming_both_values(tmp_path):
    # A corrupted, swapped, or stale artifact must never reach the
    # unpickler: the mismatch raises ValueError naming the expected and
    # observed digests (fail-closed, test-slice-digest posture).
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"corrupted-or-swapped-bytes")
    with pytest.raises(ValueError, match="integrity check FAILED"):
        verify_artifact_sha256(path, "0" * 64, artifact_name="models/x.bin")
