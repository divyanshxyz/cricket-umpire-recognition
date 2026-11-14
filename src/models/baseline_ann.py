"""Phase 3 Model A: the exact Mili et al. baseline reproduction (TRAIN-03)
plus the shared 5-fold stratified train-CV training protocol that Models
B/C/D reuse in Plan 03-02.

Architecture (TRAIN-03 exact spec, RESEARCH Pattern 3): raw 51 ->
train-fit MinMaxScaler (default, NO clip — research-verified
exact-reproduction semantics) -> Linear(51,128) -> BatchNorm1d(128) ->
ReLU -> Dropout(0.3) -> Linear(128,64) -> BatchNorm1d(64) -> ReLU ->
Dropout(0.3) -> Linear(64, n_classes), Adam lr=1e-3, plain
CrossEntropyLoss. Model A is UNWEIGHTED — only Model C carries class
weights (locked CONTEXT decision: weighting more models would confound
the architecture ablation).

Training protocol (RESEARCH Pattern 2, verified byte-identical re-runs in
this venv): seed_everything(42) + torch.set_num_threads(1), manual
torch.randperm batch loops (no DataLoader, no workers), StratifiedKFold(5,
shuffle=True, random_state=42), augmentation INSIDE each fold's train
portion only (never before the split — RESEARCH anti-pattern 1), early
stop on val loss with patience 15 (restore the best-val-loss state_dict),
final model trained for round(mean(fold best epochs)) on the full
augmented train slice. Batch guard (TRAIN-04/empty, verified Pitfall 2):
train-mode BatchNorm1d crashes on batch_size=1 — the loop raises
ValueError on n < 2 and skips a trailing singleton batch.

Checkpoint contract (RESEARCH Pattern 4, verified roundtrips): the fitted
MinMaxScaler rides INSIDE the baseline_ann.pt dict (locked CONTEXT
decision — Phase 4 image inference is self-contained) and MUST be loaded
with weights_only=False: torch 2.15 defaults to the safe loader
(weights_only=True) which REJECTS the pickled sklearn scaler (verified
Pitfall 1 — UnpicklingError).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from src.models import data as data_mod

# Locks the registry-dispatch contract with src/train_ablation.py (Plans
# 03-02/03-03 add their own train_model_b/c/d + registry rows).
__all__ = [
    "MiliBaseline",
    "train_with_cv",
    "train_baseline",
    "predict_proba",
    "save_baseline_checkpoint",
    "load_baseline_checkpoint",
    "train_model_a",
    "eval_model_a",
]

# Spec optimizer (TRAIN-03): Adam lr=1e-3. Model B's AdamW + cosine stack
# and Model C's residual optimizer ship with Plan 03-02 (their plans own
# the train_with_cv signature extension + the CODE_VERSION bump).
LEARNING_RATE = 1e-3
N_FOLDS = 5


def _make_optimizer(model) -> torch.optim.Optimizer:
    """The spec optimizer for the shared protocol: Adam lr=1e-3 (TRAIN-03)."""
    return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def _assert_trainable_batch(n_rows: int) -> None:
    """TRAIN-04/empty guard: BatchNorm1d in train mode needs > 1 value per
    channel (verified Pitfall 2) — a training matrix of fewer than 2 rows
    is rejected fail-closed instead of crashing mid-epoch or silently
    training nothing."""
    if n_rows < 2:
        raise ValueError(
            f"train_with_cv: {n_rows} training row(s) — train-mode "
            "BatchNorm1d needs n >= 2 (TRAIN-04/empty: fail-closed, never "
            "a singleton batch)"
        )


def _run_batch_epochs(model, X, y, loss_fn, epochs: int, batch_size: int) -> None:
    """One fixed-epoch training pass — manual torch.randperm batching, no
    DataLoader, no workers (RESEARCH Pattern 2 determinism). A trailing
    batch of size 1 is SKIPPED (train-mode BatchNorm1d crashes on
    batch_size=1, verified Pitfall 2); eval-mode passes are unaffected."""
    n = X.shape[0]
    _assert_trainable_batch(n)
    optimizer = _make_optimizer(model)
    for _epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i: i + batch_size]
            if idx.numel() < 2:
                continue  # trailing singleton — BN1d train-mode crash guard
            optimizer.zero_grad()
            loss_fn(model(X[idx]), y[idx]).backward()
            optimizer.step()


def _run_early_stopped_epochs(model, X, y, X_val, y_val, loss_fn,
                              max_epochs: int, patience: int,
                              batch_size: int) -> int:
    """One fold's training pass with early stopping on val loss.

    Val loss is computed under model.eval() + torch.no_grad() on the CLEAN
    (never-augmented) fold-val slice. The best-val-loss state_dict is
    restored before returning, and the fold's best epoch (1-based) is
    returned for the final-epochs derivation (round(mean fold-best))."""
    n = X.shape[0]
    _assert_trainable_batch(n)
    optimizer = _make_optimizer(model)
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i: i + batch_size]
            if idx.numel() < 2:
                continue  # trailing singleton — BN1d train-mode crash guard
            optimizer.zero_grad()
            loss_fn(model(X[idx]), y[idx]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val), y_val).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch


def train_with_cv(X_train, y_train, model_factory, loss_fn, seed: int = 42,
                  max_epochs: int = 150, patience: int = 15,
                  batch_size: int = 32, augment_fn=None, aug_seed: int = 42):
    """The shared 5-fold stratified train-CV protocol (Model A here; B/C/D
    in Plan 03-02) — epoch/schedule selection on TRAIN ROWS ONLY; the test
    set is never loaded during selection (CONTEXT-locked discipline).

    Per fold: (optionally) augment the fold-train slice ONLY via
    augment_fn(X_fold, y_fold, aug_seed + fold_idx) — NEVER augment before
    the split (RESEARCH anti-pattern 1: flipped copies of val rows in
    fold-train inflate CV and mis-select epochs) — then train a fresh
    model_factory() model with early stopping on val loss (patience 15,
    best-val-loss state_dict restored) and record the fold's best epoch.
    After all folds: train the final model on the full (augmented) train
    slice for round(mean(fold best epochs)) fixed epochs.

    seed_everything(seed) runs at entry so every call is deterministic and
    order-independent (re-runs byte-identical per RESEARCH Pattern 2).
    augment_fn defaults to data.augment semantics — MODEL-INPUT-SPACE
    jitter (RESEARCH A2): callers pass already-scaled rows (Model A) or
    raw engineered rows (B/C/D).

    Returns (final_model, meta) where meta carries fold_best_epochs,
    final_epochs, max_epochs, patience, batch_size, n_folds, cv, augmented
    for the report param block."""
    X_all = np.asarray(X_train, dtype=np.float32)
    y_all = np.asarray(y_train)
    if X_all.ndim != 2:
        raise ValueError(
            f"train_with_cv: expected a 2-D (rows, features) matrix, got "
            f"shape {X_all.shape} — fail-closed, no silent default"
        )
    if len(y_all) != X_all.shape[0]:
        raise ValueError(
            f"train_with_cv: {X_all.shape[0]} feature rows vs "
            f"{len(y_all)} labels — misaligned inputs are rejected"
        )
    data_mod.seed_everything(seed)
    _assert_trainable_batch(X_all.shape[0])

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_best_epochs: list = []
    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X_all, y_all)):
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_va, y_va = X_all[va_idx], y_all[va_idx]
        if augment_fn is not None:
            # Augment the fold-train slice ONLY — the fold-val slice stays
            # clean originals (leak guard; RESEARCH anti-pattern 1).
            X_tr, y_tr = augment_fn(X_tr, y_tr, aug_seed + fold_idx)
        model = model_factory()
        best_epoch = _run_early_stopped_epochs(
            model,
            torch.from_numpy(np.ascontiguousarray(X_tr)),
            torch.from_numpy(np.ascontiguousarray(y_tr)),
            torch.from_numpy(np.ascontiguousarray(X_va)),
            torch.from_numpy(np.ascontiguousarray(y_va)),
            loss_fn,
            max_epochs,
            patience,
            batch_size,
        )
        fold_best_epochs.append(int(best_epoch))

    final_epochs = max(1, int(round(float(np.mean(fold_best_epochs)))))
    X_full, y_full = X_all, y_all
    if augment_fn is not None:
        X_full, y_full = augment_fn(X_all, y_all, aug_seed)
    final_model = model_factory()
    _run_batch_epochs(
        final_model,
        torch.from_numpy(np.ascontiguousarray(X_full)),
        torch.from_numpy(np.ascontiguousarray(y_full)),
        loss_fn,
        final_epochs,
        batch_size,
    )
    meta = {
        "fold_best_epochs": fold_best_epochs,
        "final_epochs": final_epochs,
        "max_epochs": max_epochs,
        "patience": patience,
        "batch_size": batch_size,
        "n_folds": N_FOLDS,
        "cv": f"StratifiedKFold({N_FOLDS}, shuffle=True, random_state=42)",
        "augmented": augment_fn is not None,
        "seed": seed,
    }
    return final_model, meta


class MiliBaseline(nn.Module):
    """Model A — the exact Mili et al. reproduction (TRAIN-03).

    Input: min-max-scaled raw 51-keypoint vector. Spec chain:
    Linear(51,128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.3) ->
    Linear(128,64) -> BatchNorm1d(64) -> ReLU -> Dropout(0.3) ->
    Linear(64, n_classes)."""

    def __init__(self, n_classes: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(51, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        """x: (B, 51) batch of scaled raw features -> logits (B, n_classes)."""
        return self.net(x)


def train_baseline(X_train_raw, y_int, le, seed: int = 42):
    """Model A training entry (TRAIN-03).

    MinMaxScaler() default (no clip — exact-reproduction semantics per the
    RESEARCH TRAIN-03 row) fit on the train rows ONLY (TRAIN-03/boundary:
    the test slice is transform-only downstream, never a scaler refit —
    the scaler is part of Model A's learned state); transform train;
    train through the shared 5-fold CV protocol with
    augment_fn=data.augment (scaled inputs — model-input-space jitter per
    RESEARCH A2) and plain CrossEntropyLoss (Model A UNWEIGHTED — only
    Model C carries class weights, locked CONTEXT decision).

    Returns (model, scaler, meta)."""
    X = np.asarray(X_train_raw, dtype=np.float64)
    scaler = MinMaxScaler()  # default clip=False — exact-reproduction semantics
    X_scaled = scaler.fit_transform(X)  # fit on train rows ONLY
    model, meta = train_with_cv(
        X_scaled,
        np.asarray(y_int),
        MiliBaseline,
        nn.CrossEntropyLoss(),
        seed=seed,
        augment_fn=data_mod.augment,
    )
    meta = dict(meta)
    meta["classes"] = list(le.classes_)
    meta["scaler"] = "MinMaxScaler() default clip=False (exact-reproduction semantics)"
    return model, scaler, meta


def predict_proba(model, X) -> np.ndarray:
    """fp32 softmax probabilities under torch.no_grad() + model.eval().

    Rows sum to 1 — required by roc_auc_score's multiclass validation
    (RESEARCH Pitfall 6: logits are rejected by sklearn). Column order is
    the model's output index order = shared-encoder int order
    (le.classes_) — the eval never reorders proba columns away from the
    shared encoder order (TRAIN-03/precision backstop: argmax tie-break is
    numpy's deterministic first-index rule)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.ascontiguousarray(np.asarray(X, dtype=np.float32))))
        probs = torch.softmax(logits, dim=1)
    return probs.numpy().astype(np.float32)


def save_baseline_checkpoint(path, model, scaler, le, meta, code_version: str) -> None:
    """Persist baseline_ann.pt (RESEARCH Pattern 4, verified contract).

    The fitted MinMaxScaler rides INSIDE the checkpoint dict (locked
    CONTEXT decision — Phase 4 image inference is self-contained without
    retraining code); ``meta`` (fold-best epochs, final epochs, CV
    hyperparameters) is embedded for the eval stage and Phase 4
    provenance. The artifact is a gitignored runtime file — its sha256
    goes into the tracked report, never the binary."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": "mili_baseline",
            "input_dim": 51,
            "hidden": [128, 64],
            "dropout": 0.3,
            "n_classes": 6,
            "classes": list(le.classes_),
            "scaler": scaler,
            "seed": 42,
            "torch_version": torch.__version__,
            "code_version": code_version,
            "meta": dict(meta),
        },
        Path(path),
    )


def load_baseline_checkpoint(path) -> dict:
    """Load baseline_ann.pt.

    MUST pass weights_only=False: torch 2.15 defaults to the safe loader
    (weights_only=True) which REJECTS the pickled sklearn MinMaxScaler
    bundled inside the checkpoint (verified Pitfall 1 — UnpicklingError).
    This is our own trusted, locally-produced artifact."""
    return torch.load(Path(path), weights_only=False)


def train_model_a(X_train_raw, y_int, le, models_dir, code_version: str) -> dict:
    """MODEL_REGISTRY["A"] train entry: train_baseline + persist
    baseline_ann.pt; returns the meta dict for the report param block."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model, scaler, meta = train_baseline(X_train_raw, y_int, le)
    save_baseline_checkpoint(
        models_dir / "baseline_ann.pt", model, scaler, le, meta, code_version
    )
    return meta


def eval_model_a(arrays, test_idx, le, models_dir) -> np.ndarray:
    """MODEL_REGISTRY["A"] eval entry: reload baseline_ann.pt from disk
    (proving the save/load contract end-to-end), transform the test slice
    with the TRAIN-FIT scaler (transform-only — never a refit,
    TRAIN-03/boundary), return fp32 softmax probas (columns in shared
    encoder int order). The test slice is loaded exactly once per run by
    the guarded eval stage and only sliced here."""
    ckpt = load_baseline_checkpoint(Path(models_dir) / "baseline_ann.pt")
    scaler = ckpt["scaler"]
    model = MiliBaseline(n_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    X_test_raw = np.asarray(arrays["features_raw"])[test_idx]
    return predict_proba(model, scaler.transform(X_test_raw))
