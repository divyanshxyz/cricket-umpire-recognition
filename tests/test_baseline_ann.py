"""Model A (baseline_ann) checkpoint contract tests — WR-04 integrity gate.

Torch-using (the checkpoint is a torch.save payload with the bundled
MinMaxScaler). Pinned contracts:

- Checkpoint roundtrip (RESEARCH Pattern 4): the bundled scaler +
  state_dict survive the weights_only=False load — the scaler-bundled
  contract REQUIRES weights_only=False (verified Pitfall 1: the default
  safe loader rejects the pickled sklearn scaler with UnpicklingError).
- Load-time integrity gate (WR-04): load_baseline_checkpoint verifies the
  file's sha256 against the caller-provided expected value BEFORE
  deserialization — a wrong expectation raises ValueError naming the
  mismatch and the bytes are NEVER unpickled (proven with a non-torch
  garbage file: the integrity ValueError fires, not an UnpicklingError);
  a matching expectation loads; None skips the check (bare interactive
  reload stays usable — the runner always passes the sha256 it recorded
  in the same run).
- eval_model_a threads the meta-carried ``artifact_sha256`` into the gate
  end-to-end: correct sha -> probas; wrong sha -> the gate trips inside
  the eval entry; no meta -> bare reload.
"""

import numpy as np
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

from src.models.baseline_ann import (
    MiliBaseline,
    eval_model_a,
    load_baseline_checkpoint,
    save_baseline_checkpoint,
)
from src.models.data import canonical_classes, fit_label_encoder, sha256_of

LE = fit_label_encoder(canonical_classes())


def _save_checkpoint(tmp_path):
    """One real baseline_ann.pt payload: tiny seeded Model A + a scaler
    fit on deterministic synthetic rows (MinMaxScaler's exact-reproduction
    default, clip=False)."""
    torch.manual_seed(0)
    model = MiliBaseline(n_classes=6)
    scaler = MinMaxScaler().fit(np.random.default_rng(0).normal(size=(8, 51)))
    path = tmp_path / "baseline_ann.pt"
    save_baseline_checkpoint(
        path, model, scaler, LE, meta={"k": "v"}, code_version="wr04-test"
    )
    return path


def _synthetic_eval_inputs():
    """arrays + test_idx for eval_model_a: 6 distinct raw rows, last 3 are
    the test slice (features_raw is the only key eval_model_a reads)."""
    arrays = {
        "features_raw": (np.arange(6 * 51, dtype=np.float32) * 0.01).reshape(6, 51),
    }
    test_idx = np.array([3, 4, 5])
    return arrays, test_idx


# ---------------------------------------------------------------------------
# Checkpoint roundtrip (weights_only=False is the scaler-bundled contract)
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip_bundled_scaler_loads(tmp_path):
    path = _save_checkpoint(tmp_path)
    ckpt = load_baseline_checkpoint(path)  # None expectation: bare reload
    assert ckpt["arch"] == "mili_baseline"
    assert "state_dict" in ckpt and "scaler" in ckpt
    assert ckpt["code_version"] == "wr04-test"
    assert ckpt["classes"] == list(LE.classes_)


# ---------------------------------------------------------------------------
# WR-04: sha256-bound load
# ---------------------------------------------------------------------------


def test_load_with_matching_sha256_loads(tmp_path):
    # The documented Phase 4 recipe: verify against the recorded digest
    # before load — a matching expectation is a no-op gate.
    path = _save_checkpoint(tmp_path)
    ckpt = load_baseline_checkpoint(path, expected_sha256=sha256_of(path))
    assert ckpt["arch"] == "mili_baseline"


def test_load_with_wrong_sha256_refuses_before_deserialization(tmp_path):
    # The gate fires BEFORE torch.load: a non-torch garbage file with a
    # wrong expectation raises the integrity ValueError (never the
    # unpickler's own error) — unverified bytes never reach torch.load.
    garbage = tmp_path / "baseline_ann.pt"
    garbage.write_bytes(b"definitely-not-a-torch-checkpoint")
    with pytest.raises(ValueError, match="integrity check FAILED"):
        load_baseline_checkpoint(garbage, expected_sha256="0" * 64)


def test_load_with_none_expectation_skips_the_gate(tmp_path):
    path = _save_checkpoint(tmp_path)
    ckpt = load_baseline_checkpoint(path, expected_sha256=None)
    assert ckpt["arch"] == "mili_baseline"


# ---------------------------------------------------------------------------
# WR-04 end-to-end: eval_model_a threads the meta sha into the gate
# ---------------------------------------------------------------------------


def test_eval_model_a_threads_meta_sha_into_the_gate(tmp_path):
    path = _save_checkpoint(tmp_path)
    arrays, test_idx = _synthetic_eval_inputs()
    # Correct same-run sha in the meta -> probas for the 3 test rows,
    # rows summing to 1 (softmax contract).
    proba = eval_model_a(
        arrays, test_idx, LE, tmp_path,
        meta={"artifact_sha256": sha256_of(path)},
    )
    assert proba.shape == (3, 6)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    # Wrong sha -> the integrity gate trips inside the eval entry.
    with pytest.raises(ValueError, match="integrity check FAILED"):
        eval_model_a(
            arrays, test_idx, LE, tmp_path,
            meta={"artifact_sha256": "f" * 64},
        )
    # No meta -> bare reload stays usable (interactive path).
    proba2 = eval_model_a(arrays, test_idx, LE, tmp_path, meta=None)
    assert proba2.shape == (3, 6)
