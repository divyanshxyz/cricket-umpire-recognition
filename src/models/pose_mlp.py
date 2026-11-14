"""Phase 3 Models B + C: PoseMLP (torso-normalized MLP, TRAIN-04) and
ResidualMLP (residual MLP on the engineered features, TRAIN-05), plus the
programmatic class-weight math, the B/C loss + optimizer factories, the
train/eval registry entry points, and the pose_mlp.pt checkpoint contract.

Locked contracts (03-CONTEXT.md / 03-RESEARCH.md):
- Model B (TRAIN-04): input = features_eng[:, 0:51] — 17 x (torso-
  normalized x, y, conf) by the Phase 2 engineered-feature contract.
  Architecture mirrors Model A's hidden shape (128/64, dropout 0.3) so
  the A-vs-B ablation isolates EXACTLY the optimizer/schedule/smoothing
  delta: Linear(51,128) -> BN -> ReLU -> Dropout(0.3) -> Linear(128,64)
  -> BN -> ReLU -> Dropout(0.3) -> Linear(64, n_classes). Optimizer
  stack: AdamW(lr=1e-3, weight_decay=1e-4) + CosineAnnealingLR(T_max =
  max_epochs, eta_min=1e-5) stepped ONCE PER EPOCH. Loss:
  CrossEntropyLoss(label_smoothing=0.05) with NO class-weight tensor.
- Model C (TRAIN-05): input = features_eng (all 59). Residual MLP: stem
  Linear(59, 128) + BN + GELU, two ResBlock(dim 128, dropout 0.25) units
  (x + dropout(gelu(bn(fc(x))))), head Linear(128, 6). Loss:
  class-weighted CrossEntropyLoss with inverse-frequency weights
  N/(C * n_c) computed PROGRAMMATICALLY from the train label counts
  (never hard-coded; expected encoder-order values near Four 2.930,
  NoAction 0.671, NoBall 0.807, Out 1.427, Six 1.295, Wide 0.687).
  Optimizer: AdamW WITHOUT the cosine schedule (discretion: the schedule
  is Model B's distinctive lever; C's levers are the residual block +
  class weighting).
- UNWEIGHTED ablation design (locked): A/B/D carry no class weights and
  no oversampling/SMOTE exists anywhere — weighting more models would
  confound the architecture ablation.
- Width contracts (T-03-06 mitigation): both train entries assert the
  observed input width at entry and raise ValueError naming it —
  fail-closed, never a silent reshape.
- Artifact disposition (RESEARCH A4 lock): pose_mlp.pt holds MODEL B's
  weights with a primitives-only payload (str/list/int/tensor only) so
  it loads under torch 2.15's DEFAULT weights_only=True safe loader; the
  load side must NOT pass weights_only=False. Model C's weights are NOT
  persisted standalone — they ride the train-stage meta here (the same
  model_c_state + model_c_hyperparams keys Plan 03-03 stores inside
  ensemble_model.joblib) and move inside that bundle in Plan 03-03.
- Training protocol: BOTH entries reuse baseline_ann.train_with_cv (the
  SAME 5-fold stratified train-CV protocol as Model A — augment inside
  each fold's train portion, early stop on val loss patience 15, batch
  guard) on raw engineered rows (eng features are already normalized-
  scale, so jitter lands in model-input space per RESEARCH A2; the eng
  first-51 columns carry the same 17-keypoint triple structure, so the
  width-inferred PERM_RAW flip permutation is exact for Model B's slice).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models import data as data_mod
from src.models.baseline_ann import predict_proba, train_with_cv

__all__ = [
    "PoseMLP",
    "ResBlock",
    "ResidualMLP",
    "compute_class_weights",
    "make_b_optimizers",
    "make_b_loss",
    "make_c_loss",
    "train_pose_mlp",
    "train_residual_mlp",
    "save_pose_mlp_checkpoint",
    "load_pose_mlp_checkpoint",
    "train_model_b",
    "eval_model_b",
    "train_model_c",
    "eval_model_c",
]

# Model B input width: 17 keypoints x (torso-normalized x, y, conf).
B_INPUT_DIM = 51
# Model C input width: all 59 engineered features.
C_INPUT_DIM = 59

# Model B optimizer stack (TRAIN-04).
B_LEARNING_RATE = 1e-3
B_WEIGHT_DECAY = 1e-4
B_ETA_MIN = 1e-5
B_LABEL_SMOOTHING = 0.05

# Model C architecture + optimizer (TRAIN-05; the no-scheduler choice is
# recorded discretion — see the report param block).
C_HIDDEN_DIM = 128
C_NUM_BLOCKS = 2
C_DROPOUT = 0.25
C_LEARNING_RATE = 1e-3
C_WEIGHT_DECAY = 1e-4


# ---------------------------------------------------------------------------
# Class weights (TRAIN-05: programmatic inverse frequency, never hard-coded)
# ---------------------------------------------------------------------------


def compute_class_weights(y_int, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights N/(C * n_c) computed programmatically
    from the train label counts (TRAIN-05; RESEARCH A7 — never hard-coded).

    A zero-support class raises ValueError naming the class: a zero-support
    class cannot be weighted honestly (fail-closed, V5 posture)."""
    y = np.asarray(y_int)
    counts = np.bincount(y, minlength=n_classes)
    total = int(counts.sum())
    for cls in range(n_classes):
        if int(counts[cls]) == 0:
            raise ValueError(
                f"compute_class_weights: class {cls} has zero train support "
                f"(0 of {total} rows across {n_classes} classes) — a "
                "zero-support class cannot be weighted honestly (TRAIN-05: "
                "fail-closed)"
            )
    weights = total / (n_classes * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Loss + optimizer factories (the objects the tests assert against)
# ---------------------------------------------------------------------------


def make_b_optimizers(model, max_epochs: int):
    """Model B optimizer stack (TRAIN-04): AdamW(lr=1e-3, weight_decay=
    1e-4) + CosineAnnealingLR(T_max=max_epochs, eta_min=1e-5). The
    scheduler is stepped ONCE PER EPOCH inside train_with_cv's loops."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=B_LEARNING_RATE, weight_decay=B_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=B_ETA_MIN
    )
    return optimizer, scheduler


def make_b_loss() -> nn.CrossEntropyLoss:
    """Model B loss (TRAIN-04): label smoothing 0.05, NO class-weight
    tensor (locked ablation design: only Model C is weighted)."""
    return nn.CrossEntropyLoss(label_smoothing=B_LABEL_SMOOTHING)


def make_c_loss(y_int, n_classes: int = 6) -> nn.CrossEntropyLoss:
    """Model C loss (TRAIN-05): CrossEntropyLoss weighted by the
    programmatic inverse-frequency weights (compute_class_weights)."""
    return nn.CrossEntropyLoss(weight=compute_class_weights(y_int, n_classes))


# ---------------------------------------------------------------------------
# Architectures (RESEARCH Pattern 3)
# ---------------------------------------------------------------------------


class PoseMLP(nn.Module):
    """Model B — torso-normalized MLP (TRAIN-04).

    Input: features_eng[:, 0:51] (17 x (torso-normalized x, y, conf)).
    Deliberately mirrors Model A's hidden shape (128/64, dropout 0.3) so
    the A-vs-B ablation isolates exactly the optimizer/schedule/smoothing
    delta (flagged assumption, 03-02 frontmatter)."""

    def __init__(self, input_dim: int = B_INPUT_DIM, n_classes: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
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
        """x: (B, 51) batch of engineered rows -> logits (B, n_classes)."""
        return self.net(x)


class ResBlock(nn.Module):
    """Residual unit (RESEARCH Pattern 3): x + dropout(gelu(bn(fc(x))))."""

    def __init__(self, dim: int, p: float = C_DROPOUT):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """x: (B, dim) -> (B, dim); the skip connection carries x."""
        return x + self.drop(self.act(self.bn(self.fc(x))))


class ResidualMLP(nn.Module):
    """Model C — residual MLP on the engineered features (TRAIN-05).

    Stem Linear(59, 128) + BN + GELU, two ResBlock(128, dropout 0.25)
    units, head Linear(128, n_classes)."""

    def __init__(self, input_dim: int = C_INPUT_DIM, hidden_dim: int = C_HIDDEN_DIM,
                 num_blocks: int = C_NUM_BLOCKS, p: float = C_DROPOUT,
                 n_classes: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResBlock(hidden_dim, p) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        """x: (B, 59) batch of engineered rows -> logits (B, n_classes)."""
        return self.head(self.blocks(self.stem(x)))


# ---------------------------------------------------------------------------
# Train entry points (width asserts fail closed; shared train_with_cv)
# ---------------------------------------------------------------------------


def _assert_model_input_width(X: np.ndarray, expected: int, entry: str) -> None:
    """T-03-06 mitigation: a wrong feature slice can never silently train
    on mis-shaped inputs — both train entries assert the observed width at
    entry and raise ValueError naming it (fail-closed, never a silent
    reshape)."""
    if X.ndim != 2:
        raise ValueError(
            f"{entry}: expected a 2-D (rows, {expected}) input matrix, got "
            f"shape {X.shape} — fail-closed, never a silent reshape"
        )
    if X.shape[1] != expected:
        raise ValueError(
            f"{entry}: input contract is exactly {expected} features, got "
            f"observed width {X.shape[1]} (shape {tuple(X.shape)}) — refusing "
            "to train on a mis-shaped input (fail-closed, never a silent "
            "reshape; T-03-06 mitigation)"
        )


def _b_optimization(max_epochs: int):
    """optimizer_factory for Model B: a fresh AdamW + CosineAnnealingLR per
    training run (each fold restarts the schedule; T_max = the passed
    max_epochs budget)."""
    def factory(model):
        return make_b_optimizers(model, max_epochs)

    return factory


def _make_c_optimizers(model):
    """optimizer_factory for Model C: AdamW WITHOUT the cosine schedule
    (discretion: the schedule is Model B's distinctive lever; C's levers
    are the residual block + class weighting). Returns (optimizer, None)
    so train_with_cv's scheduler handling stays uniform."""
    return (
        torch.optim.AdamW(
            model.parameters(), lr=C_LEARNING_RATE, weight_decay=C_WEIGHT_DECAY
        ),
        None,
    )


def train_pose_mlp(X_train_eng51, y_int, le, seed: int = 42,
                   max_epochs: int = 150, patience: int = 15,
                   batch_size: int = 32):
    """Model B training entry (TRAIN-04).

    Asserts the 51-feature input contract at entry (fail-closed, naming
    the observed width), then trains through the SAME train_with_cv
    5-fold protocol as Model A — augment_fn=data.augment on raw engineered
    rows (eng features are already normalized-scale, so the jitter lands
    in model-input space per RESEARCH A2), make_b_loss() (label smoothing
    0.05, no weights), and the AdamW + CosineAnnealingLR stack wired
    through train_with_cv's optimizer_factory injection (scheduler stepped
    once per epoch inside the loop).

    Returns (model, meta)."""
    X = np.asarray(X_train_eng51, dtype=np.float32)
    _assert_model_input_width(X, B_INPUT_DIM, "train_pose_mlp")
    model, meta = train_with_cv(
        X,
        np.asarray(y_int),
        PoseMLP,
        make_b_loss(),
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        augment_fn=data_mod.augment,
        optimizer_factory=_b_optimization(max_epochs),
    )
    meta = dict(meta)
    meta["classes"] = list(le.classes_)
    meta["scaler"] = "none (raw engineered rows — already normalized-scale, RESEARCH A2)"
    meta["optimizer"] = (
        "AdamW(lr=1e-3, weight_decay=1e-4) + CosineAnnealingLR"
        f"(T_max=max_epochs={max_epochs}, eta_min=1e-5), stepped per epoch"
    )
    meta["loss"] = (
        f"CrossEntropyLoss(label_smoothing={B_LABEL_SMOOTHING}) — NO class "
        "weights (locked: only Model C is weighted)"
    )
    meta["input"] = "features_eng[:, 0:51] (17 x (torso-normalized x, y, conf))"
    return model, meta


def train_residual_mlp(X_train_eng59, y_int, le, seed: int = 42,
                       max_epochs: int = 150, patience: int = 15,
                       batch_size: int = 32):
    """Model C training entry (TRAIN-05).

    Same assert pattern for the 59-feature input contract; class-weighted
    CE (programmatic inverse-frequency weights from the train counts);
    AdamW WITHOUT the cosine schedule; the same train_with_cv protocol +
    augment as Model B.

    Returns (model, meta)."""
    X = np.asarray(X_train_eng59, dtype=np.float32)
    _assert_model_input_width(X, C_INPUT_DIM, "train_residual_mlp")
    y_arr = np.asarray(y_int)
    model, meta = train_with_cv(
        X,
        y_arr,
        ResidualMLP,
        make_c_loss(y_arr),
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        augment_fn=data_mod.augment,
        optimizer_factory=_make_c_optimizers,
    )
    meta = dict(meta)
    meta["classes"] = list(le.classes_)
    meta["scaler"] = "none (raw engineered rows — already normalized-scale, RESEARCH A2)"
    meta["optimizer"] = (
        "AdamW(lr=1e-3, weight_decay=1e-4) — no scheduler (the cosine "
        "schedule is Model B's distinctive lever; C's levers are the "
        "residual block + class weighting)"
    )
    meta["loss"] = (
        "CrossEntropyLoss(weight = N/(6 * n_c) inverse-frequency class "
        "weights, computed programmatically from the train label counts)"
    )
    meta["input"] = "features_eng (all 59 engineered features)"
    return model, meta


# ---------------------------------------------------------------------------
# pose_mlp.pt checkpoint contract (RESEARCH Pattern 4 — Model B occupant)
# ---------------------------------------------------------------------------


def save_pose_mlp_checkpoint(path, model, le, code_version: str, seed: int = 42) -> None:
    """Persist pose_mlp.pt (RESEARCH Pattern 4; the locked A4 disposition
    makes this MODEL B's standalone artifact).

    The payload is PRIMITIVES-ONLY (str/list/int/tensor) so it loads under
    torch 2.15's DEFAULT weights_only=True safe loader — the Phase 4 image
    inference reload must not need the training module. The scaler is NOT
    bundled (Model B consumes raw engineered rows) and Model C's weights
    are NOT persisted standalone (they move inside ensemble_model.joblib
    in Plan 03-03). Gitignored runtime file — its sha256 goes into the
    tracked report, never the binary."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": "pose_mlp",
            "input_dim": int(B_INPUT_DIM),
            "n_classes": 6,
            # Plain Python strings — numpy.str_ items (LabelEncoder.classes_
            # elements) pickle through numpy._core.multiarray.scalar and are
            # REJECTED by the default weights_only safe loader (TDD-caught in
            # 03-02): the primitives-only contract means str/list/int/tensor,
            # not numpy scalar subclasses.
            "classes": [str(c) for c in le.classes_],
            "seed": int(seed),
            # str() — torch.__version__ is a torch.torch_version.TorchVersion
            # instance (a str subclass with its own pickle global), which the
            # default weights_only safe loader REJECTS (TDD-caught in 03-02).
            "torch_version": str(torch.__version__),
            "code_version": code_version,
        },
        Path(path),
    )


def load_pose_mlp_checkpoint(path) -> dict:
    """Load pose_mlp.pt under the DEFAULT weights_only mode — the whole
    point of the primitives-only payload. Do NOT pass weights_only=False
    here (that is baseline_ann.pt's scaler-bundled contract, not this
    one): the safe default loader must accept this artifact for the Phase
    4 reload contract to hold (RESEARCH Pitfall 1 inverted)."""
    return torch.load(Path(path))


# ---------------------------------------------------------------------------
# MODEL_REGISTRY entries (src/train_ablation.py dispatch)
# ---------------------------------------------------------------------------


def train_model_b(X_train_eng51, y_int, le, models_dir, code_version: str) -> dict:
    """MODEL_REGISTRY["B"] train entry: train_pose_mlp + persist
    pose_mlp.pt (primitives-only); returns the meta dict for the report
    param block."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model, meta = train_pose_mlp(X_train_eng51, y_int, le)
    save_pose_mlp_checkpoint(models_dir / "pose_mlp.pt", model, le, code_version)
    return meta


def eval_model_b(arrays, test_idx, le, models_dir, meta=None) -> np.ndarray:
    """MODEL_REGISTRY["B"] eval entry: reload pose_mlp.pt from disk under
    the DEFAULT weights_only loader (proving the primitives-only save/load
    contract end-to-end), slice the TEST features_eng[:, 0:51], return
    fp32 softmax probas (columns in shared-encoder int order). The test
    slice is loaded exactly once per run by the guarded eval stage and
    only sliced here. ``meta`` is accepted for dispatch uniformity and
    unused (B reloads from disk)."""
    ckpt = load_pose_mlp_checkpoint(Path(models_dir) / "pose_mlp.pt")
    model = PoseMLP(input_dim=int(ckpt["input_dim"]), n_classes=int(ckpt["n_classes"]))
    model.load_state_dict(ckpt["state_dict"])
    X_test = np.asarray(arrays["features_eng"])[test_idx][:, : int(ckpt["input_dim"])]
    return predict_proba(model, X_test)


def train_model_c(X_train_eng59, y_int, le, models_dir, code_version: str) -> dict:
    """MODEL_REGISTRY["C"] train entry: train_residual_mlp. Persists NO
    standalone artifact (locked disposition: Model C's weights live inside
    ensemble_model.joblib from Plan 03-03) — the trained state rides the
    returned meta (model_c_state + model_c_hyperparams, the exact keys the
    Plan 03-03 ensemble bundle stores) for the same-process guarded eval
    stage. ``models_dir``/``code_version`` are accepted for dispatch
    signature uniformity and unused."""
    model, meta = train_residual_mlp(X_train_eng59, y_int, le)
    meta = dict(meta)
    meta["model_c_state"] = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    meta["model_c_hyperparams"] = {
        "input_dim": C_INPUT_DIM,
        "hidden_dim": C_HIDDEN_DIM,
        "num_blocks": C_NUM_BLOCKS,
        "dropout": C_DROPOUT,
        "n_classes": 6,
    }
    return meta


def eval_model_c(arrays, test_idx, le, models_dir, meta=None) -> np.ndarray:
    """MODEL_REGISTRY["C"] eval entry: rebuild ResidualMLP from the
    train-stage meta handoff (model_c_state + model_c_hyperparams — the
    exact keys Plan 03-03 persists inside ensemble_model.joblib). Model
    C's weights are NOT persisted standalone, so a missing handoff is a
    fail-closed ValueError, never a silent retrain or a wrong-proba run."""
    meta = meta or {}
    state = meta.get("model_c_state")
    if state is None:
        raise ValueError(
            "eval_model_c: no model_c_state in the train-stage meta — Model "
            "C's weights are not persisted standalone (they live inside "
            "models/ensemble_model.joblib from Plan 03-03); run the train "
            "stage for Model C in the same process before evaluating it"
        )
    hyper = meta.get("model_c_hyperparams") or {}
    model = ResidualMLP(
        input_dim=int(hyper.get("input_dim", C_INPUT_DIM)),
        hidden_dim=int(hyper.get("hidden_dim", C_HIDDEN_DIM)),
        num_blocks=int(hyper.get("num_blocks", C_NUM_BLOCKS)),
        p=float(hyper.get("dropout", C_DROPOUT)),
        n_classes=int(hyper.get("n_classes", len(le.classes_))),
    )
    model.load_state_dict(state)
    X_test = np.asarray(arrays["features_eng"])[test_idx]
    return predict_proba(model, X_test)
