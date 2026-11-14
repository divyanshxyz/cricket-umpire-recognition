"""Digest-guard + fold-integrity tests for the Phase 3 eval discipline
(weight-free, synthetic npz + tmp_path outputs).

Pins: (1) the frozen-test guard SystemExits naming the drift when the
test slice is tampered between training and evaluation (fail-closed, never
warn-and-continue); (2) an empty/missing-test-slice npz raises BEFORE any
report file is written (TRAIN-07/empty — the output directory stays
empty); (3) fold integrity of the locked CV protocol — every validation
fold contains all six classes (Four's 19 train rows guarantee 3-4 per
fold) and every fold's train/val indices are disjoint (index-level leak
guard; augmentation derives only from fold-train indices). No torch, no
weights.
"""

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from src.models.data import (
    canonical_classes,
    fit_label_encoder,
)
from src.models.data import test_slice_digest as compute_test_slice_digest
from src import train_ablation

YAML_ORDER = ["Four", "Six", "Out", "Wide", "NoBall", "NoAction"]

# The REAL train label census (334 rows: Four 19 / NoAction 83 / NoBall 69
# / Out 39 / Six 43 / Wide 81) — fold integrity is proven on this shape.
TRAIN_CENSUS = {
    "Four": 19,
    "NoAction": 83,
    "NoBall": 69,
    "Out": 39,
    "Six": 43,
    "Wide": 81,
}


def _base_arrays(n: int = 9, n_test: int = 3) -> dict:
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


def _write_npz(tmp_path, arrays: dict, name: str = "synthetic.npz"):
    path = tmp_path / name
    np.savez(path, **arrays)
    return path


# ---------------------------------------------------------------------------
# Digest guard: trips on a tampered test slice (fail-closed proof)
# ---------------------------------------------------------------------------


def test_guard_trips_on_tampered(tmp_path):
    # Compute the digest, rewrite the npz with ONE test feature value
    # changed, then call the runner's digest-assert logic against the
    # stale digest -> SystemExit naming the drift, BEFORE anything loads
    # or writes.
    arrays = _base_arrays()
    path = _write_npz(tmp_path, arrays)
    stale_digest = compute_test_slice_digest(path)
    tampered = dict(arrays)
    raw = arrays["features_raw"].copy()
    raw[8, 10] += 1.0  # row 8 is a test row
    tampered["features_raw"] = raw
    tampered_path = _write_npz(tmp_path, tampered, name="tampered.npz")
    with pytest.raises(SystemExit, match="digest drift"):
        train_ablation.evaluate_and_report(
            tampered_path,
            stale_digest,
            ["A"],
            {},
            6,
            models_dir=tmp_path / "models",
            reports_dir=tmp_path / "reports",
        )
    # Nothing was written before the guard fired.
    assert not (tmp_path / "reports").exists()


def test_guard_accepts_untouched_slice(tmp_path):
    # The counterpart: an untouched slice passes the assert (no
    # SystemExit) — the guard trips only on genuine drift.
    arrays = _base_arrays()
    path = _write_npz(tmp_path, arrays)
    digest = compute_test_slice_digest(path)
    train_ablation.assert_test_slice_unchanged(path, digest)  # must not raise


# ---------------------------------------------------------------------------
# Empty test slice: fail-closed BEFORE any write (TRAIN-07/empty)
# ---------------------------------------------------------------------------


def test_empty_test_slice_fails_closed_before_any_write(tmp_path):
    # An all-train npz -> the loader's ValueError propagates out of the
    # eval entry BEFORE any report file is written; the output directory
    # stays empty.
    arrays = _base_arrays(n=6, n_test=0)
    path = _write_npz(tmp_path, arrays, name="no_test.npz")
    reports_dir = tmp_path / "reports"
    with pytest.raises(ValueError):
        train_ablation.evaluate_and_report(
            path,
            compute_test_slice_digest(path),
            ["A"],
            {},
            6,
            models_dir=tmp_path / "models",
            reports_dir=reports_dir,
        )
    assert not reports_dir.exists() or list(reports_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# Fold integrity (5-fold stratified train-CV, RESEARCH anti-pattern guard)
# ---------------------------------------------------------------------------


def test_every_val_fold_holds_all_six_classes_and_folds_are_disjoint():
    # StratifiedKFold(5, shuffle=True, random_state=42) over the real
    # train label census: every validation set contains at least one row
    # of all six classes (Four: 19 rows over 5 folds -> 3-4 per fold), and
    # every fold's train indices and val indices are disjoint (index-level
    # leak guard — augmentation is derived only from fold-train indices).
    le = fit_label_encoder(canonical_classes())
    labels = [name for name, count in TRAIN_CENSUS.items() for _ in range(count)]
    y_int = le.transform(labels)
    assert len(y_int) == 334
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds_seen = 0
    for tr_idx, va_idx in splitter.split(np.zeros(len(y_int)), y_int):
        folds_seen += 1
        assert set(np.unique(y_int[va_idx])) == set(range(6)), (
            "a validation fold lost a class — per-class F1 would be "
            "undefined there"
        )
        assert len(set(tr_idx.tolist()) & set(va_idx.tolist())) == 0
    assert folds_seen == 5
