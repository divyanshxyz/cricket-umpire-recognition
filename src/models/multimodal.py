"""Phase 6 Multimodal Fusion Classifier (FUSE-03).

Combines 81-D extended pose geometric features with 576-D MobileNetV3 visual embeddings
via late probability fusion (0.5 * P_pose + 0.5 * P_vis).
Provides fail-closed SHA-256 integrity checks on bundle serialization and loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

from src.models.data import sha256_of, verify_artifact_sha256


class MultimodalFusionClassifier(BaseEstimator, ClassifierMixin):
    """Late-fusion multimodal classifier combining pose geometry and visual appearance."""

    def __init__(
        self,
        pose_scaler: Optional[RobustScaler] = None,
        pose_model: Optional[HistGradientBoostingClassifier] = None,
        vision_scaler: Optional[StandardScaler] = None,
        vision_model: Optional[LogisticRegression] = None,
        label_encoder: Optional[LabelEncoder] = None,
        weights: Tuple[float, float] = (0.5, 0.5),
    ):
        self.pose_scaler = pose_scaler or RobustScaler()
        self.pose_model = pose_model or HistGradientBoostingClassifier(
            learning_rate=0.05,
            min_samples_leaf=20,
            l2_regularization=1.0,
            max_iter=100,
            random_state=42,
        )
        self.vision_scaler = vision_scaler or StandardScaler()
        self.vision_model = vision_model or LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=42,
        )
        self.label_encoder = label_encoder or LabelEncoder()
        self.weights = weights

    @property
    def classes_(self) -> np.ndarray:
        if hasattr(self.label_encoder, "classes_"):
            return self.label_encoder.classes_
        return np.array([])

    def fit(self, X_pose: np.ndarray, X_vision: np.ndarray, y: Sequence[str]) -> MultimodalFusionClassifier:
        """Fit pose and vision branches on training features and class label strings."""
        y_encoded = self.label_encoder.fit_transform(y)

        # Pose branch
        X_p_scaled = self.pose_scaler.fit_transform(X_pose)
        self.pose_model.fit(X_p_scaled, y_encoded)

        # Vision branch
        X_v_scaled = self.vision_scaler.fit_transform(X_vision)
        self.vision_model.fit(X_v_scaled, y_encoded)

        return self

    def predict_proba(self, X_pose: np.ndarray, X_vision: np.ndarray) -> np.ndarray:
        """Compute late fused probabilities: w_pose * P_pose + w_vis * P_vis."""
        xp = np.asarray(X_pose, dtype=np.float32)
        xv = np.asarray(X_vision, dtype=np.float32)

        is_1d = (xp.ndim == 1)
        if is_1d:
            xp = xp[None, :]
        if xv.ndim == 1:
            xv = xv[None, :]

        xp_scaled = self.pose_scaler.transform(xp)
        p_pose = self.pose_model.predict_proba(xp_scaled)

        xv_scaled = self.vision_scaler.transform(xv)
        p_vis = self.vision_model.predict_proba(xv_scaled)

        p_fused = self.weights[0] * p_pose + self.weights[1] * p_vis
        p_fused = p_fused / np.sum(p_fused, axis=1, keepdims=True)

        if is_1d:
            return p_fused[0]
        return p_fused

    def predict(self, X_pose: np.ndarray, X_vision: np.ndarray) -> np.ndarray:
        """Predict class string labels for inputs."""
        probas = self.predict_proba(X_pose, X_vision)
        is_1d = (probas.ndim == 1)
        if is_1d:
            best_idx = int(np.argmax(probas))
            return self.classes_[best_idx]
        best_indices = np.argmax(probas, axis=1)
        return self.classes_[best_indices]

    def save(self, path: Union[str, Path]) -> str:
        """Save the multimodal bundle and return its SHA-256 digest."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "pose_scaler": self.pose_scaler,
            "pose_model": self.pose_model,
            "vision_scaler": self.vision_scaler,
            "vision_model": self.vision_model,
            "label_encoder": self.label_encoder,
            "weights": self.weights,
            "classes": [str(c) for c in self.classes_],
        }
        joblib.dump(bundle, path)
        return sha256_of(path)

    @classmethod
    def load(cls, path: Union[str, Path], expected_sha256: Optional[str] = None) -> MultimodalFusionClassifier:
        """Load the multimodal bundle with fail-closed SHA-256 verification."""
        path = Path(path)
        verify_artifact_sha256(
            path, expected_sha256, artifact_name="models/multimodal_fusion.joblib"
        )
        bundle = joblib.load(path)
        return cls(
            pose_scaler=bundle["pose_scaler"],
            pose_model=bundle["pose_model"],
            vision_scaler=bundle["vision_scaler"],
            vision_model=bundle["vision_model"],
            label_encoder=bundle["label_encoder"],
            weights=bundle.get("weights", (0.5, 0.5)),
        )
