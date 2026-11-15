"""Unit tests for MultimodalFusionClassifier (FUSE-03)."""

import numpy as np
import pytest

from src.models.multimodal import MultimodalFusionClassifier


def test_multimodal_fit_predict_and_proba(tmp_path):
    np.random.seed(42)
    n_train = 40
    n_test = 10
    classes = ["Four", "NoAction", "Out", "Six"]

    X_p_tr = np.random.uniform(0.1, 2.0, (n_train, 81)).astype(np.float32)
    X_v_tr = np.random.uniform(-1.0, 1.0, (n_train, 576)).astype(np.float32)
    y_tr = np.random.choice(classes, size=n_train)

    X_p_te = np.random.uniform(0.1, 2.0, (n_test, 81)).astype(np.float32)
    X_v_te = np.random.uniform(-1.0, 1.0, (n_test, 576)).astype(np.float32)

    model = MultimodalFusionClassifier()
    model.fit(X_p_tr, X_v_tr, y_tr)

    assert len(model.classes_) == len(classes)

    # Batch predict_proba
    probas = model.predict_proba(X_p_te, X_v_te)
    assert probas.shape == (n_test, len(classes))
    assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    # Batch predict
    preds = model.predict(X_p_te, X_v_te)
    assert preds.shape == (n_test,)
    assert all(p in classes for p in preds)

    # 1D single-instance handling
    proba_1d = model.predict_proba(X_p_te[0], X_v_te[0])
    assert proba_1d.shape == (len(classes),)
    assert np.isclose(proba_1d.sum(), 1.0, atol=1e-5)
    np.testing.assert_allclose(proba_1d, probas[0], atol=1e-5)

    pred_1d = model.predict(X_p_te[0], X_v_te[0])
    assert pred_1d == preds[0]


def test_multimodal_save_load_integrity(tmp_path):
    np.random.seed(42)
    n_samples = 30
    classes = ["NoBall", "Wide"]

    X_p = np.random.uniform(0.1, 2.0, (n_samples, 81)).astype(np.float32)
    X_v = np.random.uniform(-1.0, 1.0, (n_samples, 576)).astype(np.float32)
    y = np.random.choice(classes, size=n_samples)

    model = MultimodalFusionClassifier()
    model.fit(X_p, X_v, y)

    save_path = tmp_path / "test_fusion.joblib"
    digest = model.save(save_path)
    assert len(digest) == 64

    # Valid load
    reloaded = MultimodalFusionClassifier.load(save_path, expected_sha256=digest)
    orig_preds = model.predict(X_p, X_v)
    reloaded_preds = reloaded.predict(X_p, X_v)
    assert (orig_preds == reloaded_preds).all()

    # Tampered sha256 fails closed
    with pytest.raises(ValueError, match="integrity check FAILED"):
        MultimodalFusionClassifier.load(save_path, expected_sha256="0" * 64)
