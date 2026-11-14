"""Phase 3 Model D: soft-voting ensemble of Model C probabilities +
grid-tuned XGBoost + RandomForest on the 59 engineered features
(TRAIN-06), plus the self-contained ensemble_model.joblib bundle contract.

Locked contracts (03-CONTEXT.md / 03-RESEARCH.md):
- Model D (TRAIN-06): p_ens = (p_C + p_XGB + p_RF) / 3.0 — the manual
  numpy mean is the ONE sanctioned hand-roll in this phase: sklearn's
  voting-ensemble meta-estimator cannot host the torch Model C member,
  the mean is one
  line, and it preserves the row-sum-1 property roc_auc_score validates
  (RESEARCH Don't-Hand-Roll exception + verified probe 4). Members vote
  ONLY after alignment is proven: xgb.classes_ and rf.classes_ must equal
  list(range(6)) — the shared LabelEncoder's int order — and every proba
  matrix must carry exactly 6 columns (T-03-07 mitigation: a silently
  misaligned class order would mislabel every probability column).
- XGBoost/RF members are tuned via GridSearchCV(scoring="f1_macro",
  cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=1) on the
  TRAIN slice only — NEVER a bare cv int (no seed leverage, research
  anti-pattern). Locked ~12-combo split: XGB 8 combos (max_depth {3,6} x
  learning_rate {0.05,0.1} x n_estimators {100,300}) + RF 4 combos
  (n_estimators {200,400} x max_depth {None,20}).
- Augmentation happens per-CV-fold INSIDE fit() via the Augmenting*
  wrappers below — never augment-before-split (RESEARCH anti-pattern 1).
  Subclassing keeps grid params native (max_depth / n_estimators /
  learning_rate are plain grid keys — no meta-estimator prefixing) and
  sklearn clone semantics intact. The augmentation seed is the member's
  fixed random_state=42, so every fold fit is deterministic (Pitfall 7).
- The bundle (models/ensemble_model.joblib, gitignored) is SELF-CONTAINED
  (A4 disposition): fitted xgb, fitted rf, Model C state_dict (stored as
  numpy mirrors — torch-storage pickling embeds the storage's
  process-varying data_ptr, which broke the byte-identity double-run
  gate; numpy buffers pickle deterministically) +
  model_c_hyperparams {input_dim 59, hidden_dim, num_blocks, dropout,
  n_classes}, classes, voting "soft", members ["model_c", "xgboost",
  "random_forest"]. joblib roundtrip reloads and rebuild_model_c
  reconstructs Model C WITHOUT any training code — Phase 4 loads by this
  contract. The bundle is a runtime artifact: its sha256 goes into the
  tracked report, never the binary.
- Determinism (Pitfall 7): n_jobs=1 everywhere, fixed random_state=42
  members, explicit seeded stratified splitter — grid selection and the
  refit members are byte-reproducible on this machine.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.models import data as data_mod
from src.models.baseline_ann import predict_proba
from src.models.pose_mlp import ResidualMLP

__all__ = [
    "AugmentingXGBClassifier",
    "AugmentingRandomForestClassifier",
    "XGB_GRID",
    "RF_GRID",
    "tune_grids",
    "soft_vote",
    "save_ensemble_bundle",
    "load_ensemble_bundle",
    "rebuild_model_c",
    "train_model_d",
    "eval_model_d",
]

# Locked ~12-combo grid split (03-CONTEXT decision, RESOLVED in research):
# XGB 8 combos + RF 4 combos, tuned on the train slice ONLY.
XGB_GRID = {
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [100, 300],
}
RF_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 20],
}

# Model D input width: all 59 engineered features (same slice as Model C).
D_INPUT_DIM = 59

# The bundle contract (RESEARCH Pattern 4; frozen for Phase 4 loading).
BUNDLE_MEMBERS = ["model_c", "xgboost", "random_forest"]
BUNDLE_KEYS = frozenset(
    {"xgb", "rf", "model_c_state", "model_c_hyperparams", "classes",
     "voting", "members"}
)
MODEL_C_HYPERPARAM_KEYS = ("input_dim", "hidden_dim", "num_blocks", "dropout",
                           "n_classes")


# ---------------------------------------------------------------------------
# Fit-time augmentation wrappers (per-CV-fold augment inside fit)
# ---------------------------------------------------------------------------


def _augmented_fit_inputs(estimator_name, X, y, random_state, augment_fn):
    """Resolve the augmented (X, y) a wrapper trains on.

    random_state=None cannot seed the augmentation deterministically
    (Pitfall 7) — fail-closed naming the estimator, never a silent
    unseeded jitter. augment_fn defaults to data.augment (width-inferred
    mirror flip + seeded jitter); X must be a 2-D matrix whose width the
    augment contract accepts (51 raw / 59 engineered)."""
    if random_state is None:
        raise ValueError(
            f"{estimator_name}: random_state is None — the fit-time "
            "augmentation seed is the estimator's random_state and an "
            "unset seed cannot be deterministic (RESEARCH Pitfall 7); "
            "pass a fixed random_state (the Model D members use 42) "
            "(fail-closed, never a silently unseeded augmentation)"
        )
    fn = augment_fn if augment_fn is not None else data_mod.augment
    X_aug, y_aug = fn(np.asarray(X), np.asarray(y), seed=random_state)
    X_aug = np.asarray(X_aug)
    if X_aug.shape[0] != 2 * np.asarray(X).shape[0]:
        raise ValueError(
            f"{estimator_name}: augmentation produced {X_aug.shape[0]} rows "
            f"from {np.asarray(X).shape[0]} — the TRAIN-02 contract doubles "
            "the train slice (mirror flip x2); refusing to fit on a "
            "non-doubled set (fail-closed)"
        )
    return X_aug, y_aug


class AugmentingXGBClassifier(xgb.XGBClassifier):
    """XGBClassifier whose fit() first augments the fold-train slice.

    fit(X, y) calls the augment fn (default data.augment — flip + jitter,
    seed = self.random_state) on the RAW fold-train rows it receives, then
    fits on the doubled set. GridSearchCV clones per fold and calls
    fit(fold_train) — so augmentation happens per-CV-fold INSIDE fit(),
    never augment-before-split. xgboost's get_params merges **kwargs, so
    clone carries every constructor kwarg (random_state / n_jobs /
    tree_method / ...) plus augment_fn — grid params stay native."""

    def __init__(self, augment_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.augment_fn = augment_fn

    def fit(self, X, y):
        X_aug, y_aug = _augmented_fit_inputs(
            "AugmentingXGBClassifier", X, y, self.random_state, self.augment_fn
        )
        return super().fit(X_aug, y_aug)


class AugmentingRandomForestClassifier(RandomForestClassifier):
    """RandomForestClassifier whose fit() first augments the fold-train
    slice (same contract as AugmentingXGBClassifier).

    sklearn inspects the __init__ signature for get_params/clone, so the
    mirrored params are EXPLICIT — a **kwargs subclass would silently drop
    n_estimators/random_state on clone (grid + determinism breaker)."""

    def __init__(self, n_estimators=100, max_depth=None, bootstrap=True,
                 random_state=None, n_jobs=None, augment_fn=None):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.augment_fn = augment_fn

    def fit(self, X, y):
        X_aug, y_aug = _augmented_fit_inputs(
            "AugmentingRandomForestClassifier", X, y, self.random_state,
            self.augment_fn,
        )
        return super().fit(X_aug, y_aug)


# ---------------------------------------------------------------------------
# Grid tuning (train slice ONLY — the test set never enters this function)
# ---------------------------------------------------------------------------


def _n_combos(grid: dict) -> int:
    combos = 1
    for values in grid.values():
        combos *= len(values)
    return combos


def tune_grids(X_train_eng, y_int):
    """Tune the XGB + RF ensemble members on the train slice ONLY.

    Both searches use GridSearchCV(scoring="f1_macro",
    cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=1,
    refit=True) — the explicit seeded splitter (NEVER a bare cv int) and
    single-process determinism (Pitfall 7). Each member is an
    Augmenting*Classifier with random_state=42, so every CV fold's
    fit() augments the fold-train slice deterministically.

    Returns (xgb_best, rf_best, grid_meta) — the refit-best members plus
    the meta dict the report param block renders (best params/scores per
    member, the locked cv/scoring/n_jobs protocol, and the combo counts)."""
    X = np.asarray(X_train_eng, dtype=np.float32)
    y = np.asarray(y_int)
    if X.ndim != 2 or X.shape[1] != D_INPUT_DIM:
        raise ValueError(
            f"tune_grids: Model D input contract is exactly {D_INPUT_DIM} "
            f"engineered features, got observed shape {X.shape} — refusing "
            "to tune on a mis-shaped input (fail-closed, never a silent "
            "reshape; T-03-06 posture)"
        )

    xgb_search = GridSearchCV(
        AugmentingXGBClassifier(
            random_state=42, n_jobs=1, tree_method="hist",
            eval_metric="mlogloss", verbosity=0,
        ),
        XGB_GRID,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=1,
        refit=True,
    )
    xgb_search.fit(X, y)

    rf_search = GridSearchCV(
        AugmentingRandomForestClassifier(random_state=42, n_jobs=1),
        RF_GRID,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=1,
        refit=True,
    )
    rf_search.fit(X, y)

    grid_meta = {
        "cv": "StratifiedKFold(5, shuffle=True, random_state=42)",
        "scoring": "f1_macro",
        "n_jobs": 1,
        "n_combos": {"xgb": _n_combos(XGB_GRID), "rf": _n_combos(RF_GRID)},
        "xgb_best_params": dict(xgb_search.best_params_),
        "xgb_best_score": float(xgb_search.best_score_),
        "rf_best_params": dict(rf_search.best_params_),
        "rf_best_score": float(rf_search.best_score_),
    }
    return xgb_search.best_estimator_, rf_search.best_estimator_, grid_meta


# ---------------------------------------------------------------------------
# Soft vote (the ONE sanctioned hand-roll — see module docstring)
# ---------------------------------------------------------------------------


def soft_vote(p_c, p_xgb, p_rf, member_classes=None, n_classes: int = 6):
    """Soft vote: (p_C + p_XGB + p_RF) / 3.0 with alignment asserts.

    Fail-closed BEFORE averaging (T-03-07 mitigation — a silently
    misaligned class order would mislabel every probability column):
    - every member matrix must be 2-D with the same column count AND
      exactly ``n_classes`` columns (the shared encoder's int order);
    - ``member_classes`` (when passed — the eval stage always passes the
      xgb/rf classes_ arrays) must be one classes_ sequence per member,
      each equal to list(range(n_classes));
    - the averaged rows must sum to 1 within 1e-5 (roc_auc_score's
      multiclass validation, RESEARCH Pitfall 6).

    Internal math is float64: (a + a + a) / 3.0 is exact for float32
    inputs, so identical members return those rows bit-for-bit. Returns
    the (rows, n_classes) float64 vote matrix — columns in the shared
    encoder int order, feeding the guarded eval path unchanged."""
    names = ("model_c", "xgboost", "random_forest")
    matrices = [np.asarray(p_c), np.asarray(p_xgb), np.asarray(p_rf)]
    for name, m in zip(names, matrices):
        if m.ndim != 2:
            raise ValueError(
                f"soft_vote: member {name!r} proba matrix must be 2-D "
                f"(rows, {n_classes}), got shape {m.shape} — fail-closed, "
                "never a silent vote over a mis-shaped matrix"
            )
    widths = {name: m.shape[1] for name, m in zip(names, matrices)}
    if len(set(widths.values())) != 1:
        raise ValueError(
            f"soft_vote: member proba column counts disagree {widths} — "
            "every member votes over the SAME class space (fail-closed)"
        )
    for name, m in zip(names, matrices):
        if m.shape[1] != n_classes:
            raise ValueError(
                f"soft_vote: member {name!r} proba matrix has {m.shape[1]} "
                f"columns instead of {n_classes} (the shared LabelEncoder's "
                "int order) — a misaligned column count would mislabel "
                "every probability in the vote (T-03-07: fail-closed)"
            )
    if member_classes is not None:
        entries = list(member_classes)
        if len(entries) != len(names):
            raise ValueError(
                f"soft_vote: member_classes must carry one classes_ entry "
                f"per member ({len(names)}), got {len(entries)} — "
                "fail-closed, never a partial alignment check"
            )
        for name, classes in zip(names, entries):
            if [int(c) for c in classes] != list(range(n_classes)):
                raise ValueError(
                    f"soft_vote: member {name!r} classes_ "
                    f"{[int(c) for c in classes]} != list(range({n_classes})) "
                    "(the shared LabelEncoder's int order) — voting over "
                    "misaligned class orders would silently mislabel every "
                    "probability column (T-03-07: fail-closed)"
                )
    p = (
        matrices[0].astype(np.float64)
        + matrices[1].astype(np.float64)
        + matrices[2].astype(np.float64)
    ) / 3.0
    row_sums = p.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError(
            f"soft_vote: voted rows sum to {row_sums.min():.6f}.."
            f"{row_sums.max():.6f}, not 1 (tolerance 1e-5) — roc_auc_score's "
            "multiclass validation rejects non-probabilities (RESEARCH "
            "Pitfall 6: fail-closed, never feed logits-like rows to AUC)"
        )
    return p


# ---------------------------------------------------------------------------
# Self-contained bundle (RESEARCH Pattern 4 — A4 disposition)
# ---------------------------------------------------------------------------


def save_ensemble_bundle(path, xgb_best, rf_best, model_c_state,
                         model_c_hyperparams, le) -> None:
    """Persist models/ensemble_model.joblib — the SELF-CONTAINED Model D
    bundle (A4: the ensemble cannot reproduce Model C's probabilities
    without C's weights inside).

    Payload: fitted xgb, fitted rf, Model C state_dict stored as NUMPY
    MIRRORS of the tensors (key-for-key, same dtypes/shapes),
    model_c_hyperparams (exactly {input_dim, hidden_dim, num_blocks,
    dropout, n_classes}), classes (plain str list — the shared encoder
    order), voting "soft", members ["model_c", "xgboost",
    "random_forest"]. rebuild_model_c converts the numpy mirrors back to
    tensors — the bundle stays self-contained without any training code.

    WHY NUMPY MIRRORS (determinism finding, Plan 03-03 Task 3 gate):
    pickling a torch tensor goes through UntypedStorage.__reduce__, which
    embeds a legacy torch.save of the storage — and that container
    carries the storage's data_ptr (a process-varying memory address).
    Two byte-identical-weight runs therefore produced different
    ensemble_model.joblib bytes, breaking the byte-identity double-run
    gate. Numpy arrays pickle as raw buffers — byte-stable for identical
    contents (the same reason pose_mlp.pt/baseline_ann.pt are stable:
    same-path torch.save writes no addresses). Gitignored runtime
    artifact — its sha256 goes into the tracked report, never the
    binary."""
    hyper = dict(model_c_hyperparams)
    missing = [k for k in MODEL_C_HYPERPARAM_KEYS if k not in hyper]
    if missing:
        raise ValueError(
            f"save_ensemble_bundle: model_c_hyperparams missing key(s) "
            f"{missing} (contract: {list(MODEL_C_HYPERPARAM_KEYS)}) — a "
            "bundle without the full Model C construction contract cannot "
            "be rebuilt self-contained (fail-closed)"
        )
    state_out = {}
    for k, v in dict(model_c_state).items():
        if isinstance(v, torch.Tensor):
            state_out[k] = np.array(v.detach().cpu().numpy(), copy=True)
        else:
            state_out[k] = np.array(v, copy=True)
    bundle = {
        "xgb": xgb_best,
        "rf": rf_best,
        "model_c_state": state_out,
        "model_c_hyperparams": {
            "input_dim": int(hyper["input_dim"]),
            "hidden_dim": int(hyper["hidden_dim"]),
            "num_blocks": int(hyper["num_blocks"]),
            "dropout": float(hyper["dropout"]),
            "n_classes": int(hyper["n_classes"]),
        },
        "classes": [str(c) for c in le.classes_],
        "voting": "soft",
        "members": list(BUNDLE_MEMBERS),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_ensemble_bundle(path, expected_sha256=None) -> dict:
    """joblib.load the ensemble bundle (the Phase 4 load recipe — no
    training code needed on the load side).

    joblib.load executes arbitrary embedded code at load time, so the
    bytes are INTEGRITY-BOUND first (WR-04): when ``expected_sha256`` is
    provided, the file's sha256 is verified against it BEFORE the load
    and any mismatch refuses fail-closed. The documented Phase 4 reload
    recipe is ``load_ensemble_bundle(path, expected_sha256=<the
    ensemble_model sha256 recorded in reports/ablation_study.md's param
    block>)``. A None expectation skips the check (bare interactive
    reload); the runner passes the sha256 it recorded in the same run."""
    path = Path(path)
    data_mod.verify_artifact_sha256(
        path, expected_sha256, artifact_name="models/ensemble_model.joblib"
    )
    return joblib.load(path)


def rebuild_model_c(state_dict, hyperparams) -> ResidualMLP:
    """Rebuild Model C from the bundle's state + hyperparams WITHOUT any
    training code (the self-contained bundle proof): construct
    ResidualMLP from model_c_hyperparams, load_state_dict, eval() —
    ready for predict_proba at inference time.

    Accepts the bundle's NUMPY MIRRORS (see save_ensemble_bundle — the
    determinism contract) and, for the in-memory train-stage handoff,
    plain torch tensors; numpy entries convert back to tensors of the
    same dtype/shape before load_state_dict."""
    hyper = dict(hyperparams)
    model = ResidualMLP(
        input_dim=int(hyper["input_dim"]),
        hidden_dim=int(hyper["hidden_dim"]),
        num_blocks=int(hyper["num_blocks"]),
        p=float(hyper["dropout"]),
        n_classes=int(hyper["n_classes"]),
    )
    tensors = {}
    for k, v in dict(state_dict).items():
        if isinstance(v, torch.Tensor):
            tensors[k] = v
        else:
            tensors[k] = torch.from_numpy(np.ascontiguousarray(v))
    model.load_state_dict(tensors)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# MODEL_REGISTRY entries (src/train_ablation.py dispatch)
# ---------------------------------------------------------------------------


def train_model_d(X_train_eng, y_int, le, models_dir, code_version: str,
                  prev_meta=None) -> dict:
    """MODEL_REGISTRY["D"] train entry: tune the XGB/RF members via the
    locked seeded grids (train rows only) and persist the self-contained
    bundle.

    Requires Model C trained in the SAME run (prev_meta["C"] carries the
    model_c_state + model_c_hyperparams handoff — the exact keys stored
    inside the bundle), so the vote members are the exact reported models.
    A missing handoff is a fail-closed SystemExit naming the fix — never a
    silent retrain or a wrong-proba ensemble. ``code_version`` is accepted
    for dispatch signature uniformity and unused (the bundle carries no
    torch checkpoint)."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    prev = prev_meta or {}
    c_meta = prev.get("C") or {}
    if "model_c_state" not in c_meta or "model_c_hyperparams" not in c_meta:
        raise SystemExit(
            "[ablation] Model D requires Model C trained in the same run — "
            "prev_meta carries no model_c_state/model_c_hyperparams handoff "
            "(the ensemble bundle stores Model C's weights inside; the vote "
            "members must be the exact reported models). Re-run with Model "
            "C selected (e.g. --models A,B,C,D — MODULE_ORDER trains C "
            "before D) (fail-closed, never a silent retrain)"
        )

    xgb_best, rf_best, grid_meta = tune_grids(X_train_eng, y_int)
    save_ensemble_bundle(
        models_dir / "ensemble_model.joblib",
        xgb_best,
        rf_best,
        c_meta["model_c_state"],
        c_meta["model_c_hyperparams"],
        le,
    )

    meta = {
        "classes": list(le.classes_),
        "input": "features_eng (all 59 engineered features)",
        "members": list(BUNDLE_MEMBERS),
        "voting": "soft",
        "soft_vote": "(p_C + p_XGB + p_RF) / 3.0 — manual numpy mean (the "
                     "ONE sanctioned hand-roll: sklearn's voting-ensemble "
                     "meta-estimator cannot host the torch Model C member)",
        "augmentation": "fit-time wrappers (AugmentingXGBClassifier / "
                        "AugmentingRandomForestClassifier) — per-CV-fold "
                        "flip+jitter inside fit(), seed = member "
                        "random_state 42; never augment-before-split",
        "grid_xgb_best_params": grid_meta["xgb_best_params"],
        "grid_xgb_best_score": grid_meta["xgb_best_score"],
        "grid_rf_best_params": grid_meta["rf_best_params"],
        "grid_rf_best_score": grid_meta["rf_best_score"],
        "grid_cv": grid_meta["cv"],
        "grid_scoring": grid_meta["scoring"],
        "grid_n_jobs": grid_meta["n_jobs"],
        "grid_n_combos": grid_meta["n_combos"],
        "model_c_source": "Model C trained in the same run (weights ride "
                          "the train-stage meta and are stored inside the "
                          "bundle)",
    }
    return meta


def eval_model_d(arrays, test_idx, le, models_dir, meta=None) -> np.ndarray:
    """MODEL_REGISTRY["D"] eval entry: reload the self-contained bundle
    from disk (proving the save/load contract end-to-end), rebuild Model
    C from model_c_state + model_c_hyperparams, compute the three member
    proba matrices on the IDENTICAL held-out test slice, and soft-vote
    them (alignment asserted, rows summing to 1). ``meta`` may carry
    ``artifact_sha256`` (the runner records sha256_of(ensemble_model.joblib)
    at train time, WR-04) — when present, the bundle reload is
    integrity-gated against it BEFORE joblib.load; absent, the bare
    reload stays usable (interactive path)."""
    meta = meta or {}
    bundle = load_ensemble_bundle(
        Path(models_dir) / "ensemble_model.joblib",
        expected_sha256=meta.get("artifact_sha256"),
    )
    missing = [k for k in BUNDLE_KEYS if k not in bundle]
    if missing:
        raise ValueError(
            f"eval_model_d: ensemble bundle missing contract key(s) "
            f"{missing} (contract: {sorted(BUNDLE_KEYS)}) — refusing to "
            "vote over an incomplete bundle (fail-closed)"
        )
    model_c = rebuild_model_c(bundle["model_c_state"],
                              bundle["model_c_hyperparams"])
    X_test = np.asarray(arrays["features_eng"])[test_idx]
    p_c = predict_proba(model_c, X_test)
    p_xgb = bundle["xgb"].predict_proba(X_test)
    p_rf = bundle["rf"].predict_proba(X_test)
    member_classes = [
        list(range(len(le.classes_))),
        [int(c) for c in bundle["xgb"].classes_],
        [int(c) for c in bundle["rf"].classes_],
    ]
    return soft_vote(p_c, p_xgb, p_rf, member_classes=member_classes,
                     n_classes=len(le.classes_))
