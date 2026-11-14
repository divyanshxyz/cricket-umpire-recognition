"""Model D (ensemble) contract tests (TRAIN-06, Plan 03-03).

Weight-free except where torch is the subject (bundle roundtrip +
rebuild_model_c), deterministic (torch.manual_seed(0) / default_rng seeds),
synthetic data only — no npz, no real training runs. Pinned contracts:

- soft_vote alignment trips (T-03-07 mitigation): a member proba matrix
  with 5 columns instead of 6 raises ValueError; a passed member classes_
  list differing from list(range(6)) (the shared LabelEncoder int order)
  raises ValueError — a silently misaligned column order would mislabel
  every probability in the vote.
- soft_vote mean: three identical rows return those rows; three different
  rows return their elementwise mean; every output row sums to 1 within
  1e-6 (roc_auc_score's multiclass validation, RESEARCH Pitfall 6). Rows
  that do NOT sum to 1 are rejected (proba-sum guard).
- Grid enumeration (locked ~12-combo split, 03-CONTEXT): XGB 8 combos
  (max_depth {3,6} x learning_rate {0.05,0.1} x n_estimators {100,300}) +
  RF 4 combos (n_estimators {200,400} x max_depth {None,20}).
- Fit-time augmentation wrappers: AugmentingRandomForestClassifier trains
  on the DOUBLED fold (40 rows from 20) with the augmentation seed fixed
  by random_state — verified via an overridden augment_fn capture AND an
  exact row-count probe (bootstrap=False makes the tree root carry every
  training row); a plain RandomForestClassifier sees 20. The default path
  (data.augment on width-59 rows) doubles identically.
- Bundle roundtrip: ensemble_model.joblib holds EXACTLY {xgb, rf,
  model_c_state, model_c_hyperparams, classes, voting, members} with
  members == ['model_c', 'xgboost', 'random_forest'], and
  rebuild_model_c(state, hyperparams) forward-passes (2, 59) -> (2, 6)
  in eval mode WITHOUT any training code (self-contained bundle, A4).
"""

import numpy as np
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier

from src.models.data import canonical_classes, fit_label_encoder

LE = fit_label_encoder(canonical_classes())

# Module under test (Plan 03-03 RED: src.models.ensemble does not exist
# yet — this import is the RED evidence for the whole file).
from src.models.ensemble import (  # noqa: E402
    AugmentingRandomForestClassifier,
    AugmentingXGBClassifier,
    RF_GRID,
    XGB_GRID,
    rebuild_model_c,
    save_ensemble_bundle,
    load_ensemble_bundle,
    soft_vote,
    tune_grids,
)


def _valid_proba(rows: int, seed: int) -> np.ndarray:
    """Deterministic (rows, 6) proba rows summing to 1.

    float32 — the dtype torch softmax / xgboost predict_proba actually
    produce. float64 rows would round (a + a + a) away from bit-exactness
    (a 53-bit mantissa cannot always hold 3a); float32-sourced values are
    exact in float64 (24-bit mantissas)."""
    rng = np.random.default_rng(seed)
    r = rng.random((rows, 6))
    return (r / r.sum(axis=1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------------------
# Test 1: alignment trips (a misaligned member is a ValueError, never a vote)
# ---------------------------------------------------------------------------


def test_soft_vote_trips_on_5_column_member_proba():
    p6 = _valid_proba(3, seed=0)
    p5 = _valid_proba(3, seed=1)[:, :5]  # 5 columns instead of 6
    with pytest.raises(ValueError, match=r"column"):
        soft_vote(p6, p5, p6)
    with pytest.raises(ValueError, match=r"column"):
        soft_vote(p6, p6, p5)


def test_soft_vote_trips_on_misaligned_member_classes():
    p6 = _valid_proba(3, seed=2)
    # A member whose classes_ lost a class — the shared encoder int order
    # is EXACTLY list(range(6)); anything else must trip before averaging.
    with pytest.raises(ValueError, match=r"classes"):
        soft_vote(
            p6, p6, p6,
            member_classes=[list(range(6)), list(range(5)), list(range(6))],
        )
    with pytest.raises(ValueError, match=r"classes"):
        soft_vote(
            p6, p6, p6,
            member_classes=[np.arange(6), np.arange(6), np.array([0, 1, 2, 3, 4, 5][::-1])],
        )
    # Aligned members (numpy int arrays — what xgb/rf classes_ carry) pass.
    p = soft_vote(p6, p6, p6, member_classes=[np.arange(6)] * 3)
    assert np.allclose(p, p6, atol=1e-12)


# ---------------------------------------------------------------------------
# Test 2: the mean is exact on identical rows, elementwise on different rows,
# and every output row sums to 1 (roc_auc_score's multiclass validation)
# ---------------------------------------------------------------------------


def test_soft_vote_identical_rows_return_exactly_those_rows():
    rows = _valid_proba(2, seed=3)
    out = soft_vote(rows, rows, rows)
    # float64 internal math: (a + a + a) / 3.0 is exact for float32-sourced
    # values — "returns exactly those rows" holds bit-for-bit.
    assert np.array_equal(out, rows)


def test_soft_vote_different_rows_return_elementwise_mean_and_rows_sum_to_1():
    a = _valid_proba(4, seed=4)
    b = _valid_proba(4, seed=5)
    c = _valid_proba(4, seed=6)
    out = soft_vote(a, b, c)
    assert np.allclose(out, (a + b + c) / 3.0, atol=1e-12)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_soft_vote_rejects_rows_that_do_not_sum_to_one():
    bad = np.full((2, 6), 0.15)  # each row sums to 0.9, not 1
    with pytest.raises(ValueError, match=r"sum"):
        soft_vote(bad, bad, bad)


# ---------------------------------------------------------------------------
# Test 3: locked grid enumeration — XGB 8 combos + RF 4 combos
# ---------------------------------------------------------------------------


def test_grid_enumeration_is_the_locked_8_plus_4_combos():
    assert XGB_GRID == {
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 300],
    }
    assert RF_GRID == {"n_estimators": [200, 400], "max_depth": [None, 20]}
    xgb_combos = 1
    for values in XGB_GRID.values():
        xgb_combos *= len(values)
    rf_combos = 1
    for values in RF_GRID.values():
        rf_combos *= len(values)
    assert xgb_combos == 8
    assert rf_combos == 4


# ---------------------------------------------------------------------------
# Test 4: fit-time augmentation — the wrapper trains on the DOUBLED fold
# ---------------------------------------------------------------------------


def test_augmenting_rf_trains_on_doubled_fold_via_captured_augment_fn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5)).astype(np.float32)
    y = np.arange(20) % 6
    captured = {}

    def capture_augment(X_in, y_in, seed=42):
        captured["in_rows"] = int(np.asarray(X_in).shape[0])
        captured["seed"] = seed
        return np.vstack([X_in, X_in]), np.concatenate([y_in, y_in])

    clf = AugmentingRandomForestClassifier(
        n_estimators=5, random_state=42, bootstrap=False, augment_fn=capture_augment
    )
    clf.fit(X, y)
    # fit() received the raw 20-row fold and augmented it to 40 with the
    # random_state-fixed seed (deterministic augmentation, RESEARCH Pitfall 7).
    assert captured["in_rows"] == 20
    assert captured["seed"] == 42
    # bootstrap=False -> the tree root carries EVERY training row exactly.
    assert clf.estimators_[0].tree_.n_node_samples[0] == 40

    plain = RandomForestClassifier(
        n_estimators=5, random_state=42, bootstrap=False
    ).fit(X, y)
    assert plain.estimators_[0].tree_.n_node_samples[0] == 20


def test_augmenting_rf_default_path_doubles_width_59_rows():
    # Default augment_fn=None -> data.augment — valid on width-59 rows
    # (mirror_flip's width validation), doubling 20 -> 40.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 59)).astype(np.float32)
    y = np.arange(20) % 6
    clf = AugmentingRandomForestClassifier(
        n_estimators=5, random_state=42, bootstrap=False
    )
    clf.fit(X, y)
    assert clf.estimators_[0].tree_.n_node_samples[0] == 40


def test_augmenting_wrapper_rejects_unset_random_state():
    # The augmentation seed is random_state — an unset (None) seed cannot
    # be deterministic (RESEARCH Pitfall 7); fail-closed, never silently.
    rng = np.random.default_rng(2)
    X = rng.normal(size=(12, 59)).astype(np.float32)
    y = np.arange(12) % 6
    clf = AugmentingRandomForestClassifier(n_estimators=2, bootstrap=False)
    with pytest.raises(ValueError, match=r"random_state"):
        clf.fit(X, y)


# ---------------------------------------------------------------------------
# Test 5: bundle roundtrip — self-contained, rebuild_model_c needs no
# training code
# ---------------------------------------------------------------------------


def test_bundle_roundtrip_and_rebuild_model_c(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(24, 59)).astype(np.float32)
    y = np.arange(24) % 6
    xgb_clf = AugmentingXGBClassifier(
        n_estimators=5, max_depth=2, random_state=42, n_jobs=1,
        tree_method="hist", eval_metric="mlogloss", verbosity=0,
    )
    xgb_clf.fit(X, y)
    rf_clf = AugmentingRandomForestClassifier(n_estimators=5, random_state=42, n_jobs=1)
    rf_clf.fit(X, y)

    torch.manual_seed(0)
    model_c = ResidualMLP_for_bundle()
    state = {k: v.detach().cpu().clone() for k, v in model_c.state_dict().items()}
    hyper = {"input_dim": 59, "hidden_dim": 16, "num_blocks": 2, "dropout": 0.1,
             "n_classes": 6}

    path = tmp_path / "ensemble_model.joblib"
    save_ensemble_bundle(path, xgb_clf, rf_clf, state, hyper, LE)

    bundle = load_ensemble_bundle(path)
    assert set(bundle.keys()) == {
        "xgb", "rf", "model_c_state", "model_c_hyperparams",
        "classes", "voting", "members",
    }
    assert bundle["members"] == ["model_c", "xgboost", "random_forest"]
    assert bundle["voting"] == "soft"
    assert bundle["classes"] == [str(c) for c in LE.classes_]
    assert bundle["model_c_hyperparams"] == hyper

    # The persisted members still predict after the joblib roundtrip.
    assert bundle["xgb"].predict_proba(X[:2]).shape == (2, 6)
    assert bundle["rf"].predict_proba(X[:2]).shape == (2, 6)

    rebuilt = rebuild_model_c(bundle["model_c_state"], bundle["model_c_hyperparams"])
    assert rebuilt.training is False  # eval mode — no training code needed
    out = rebuilt(torch.from_numpy(X[:2]))
    assert out.shape == (2, 6)


def ResidualMLP_for_bundle():
    """A small Model C for the roundtrip test (avoids importing the real
    ResidualMLP in the RED phase — the bundle contract is state +
    hyperparams, whatever module builds it)."""
    from src.models.pose_mlp import ResidualMLP

    return ResidualMLP(input_dim=59, hidden_dim=16, num_blocks=2, p=0.1,
                       n_classes=6)


# ---------------------------------------------------------------------------
# Test 6: tune_grids returns (xgb_best, rf_best, grid_meta) through the
# locked seeded-stratified GridSearchCV protocol
# ---------------------------------------------------------------------------


def test_tune_grids_returns_fitted_members_and_grid_meta():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 59)).astype(np.float32)
    y = np.arange(30) % 6  # 5 per class — stratifiable by 5 folds
    xgb_best, rf_best, grid_meta = tune_grids(X, y)
    assert isinstance(xgb_best, AugmentingXGBClassifier)
    assert isinstance(rf_best, AugmentingRandomForestClassifier)
    assert xgb_best.classes_.tolist() == list(range(6))
    assert rf_best.classes_.tolist() == list(range(6))
    assert set(grid_meta.keys()) == {
        "cv", "scoring", "n_jobs", "n_combos",
        "xgb_best_params", "xgb_best_score",
        "rf_best_params", "rf_best_score",
    }
    assert grid_meta["cv"] == "StratifiedKFold(5, shuffle=True, random_state=42)"
    assert grid_meta["scoring"] == "f1_macro"
    assert grid_meta["n_jobs"] == 1
    assert grid_meta["n_combos"] == {"xgb": 8, "rf": 4}
    assert set(grid_meta["xgb_best_params"].keys()) == set(XGB_GRID.keys())
    assert set(grid_meta["rf_best_params"].keys()) == set(RF_GRID.keys())
    # refit=True semantics: the returned members are the grid winners.
    assert xgb_best.get_params()["max_depth"] == grid_meta["xgb_best_params"]["max_depth"]
    assert rf_best.get_params()["n_estimators"] == grid_meta["rf_best_params"]["n_estimators"]
