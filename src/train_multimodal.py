"""Phase 6 Multimodal Training Runner (FUSE-03).

Trains the MultimodalFusionClassifier bundle on the frozen train split (398 samples),
evaluates on the held-out test split (100 samples), asserts test accuracy >= 91% and
macro-F1 >= 0.90, serializes the bundle to models/multimodal_fusion.joblib, binds its
SHA-256 digest in configs/model_digests.yaml, and produces reports/multimodal_ablation.md.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
import yaml

from src.inference.vision_extractor import VisualFeatureExtractor
from src.models.data import sha256_of
from src.models.multimodal import MultimodalFusionClassifier
from src.pose_features import engineer_extended_features

REPO_ROOT = Path(__file__).resolve().parent.parent


def extract_or_load_vision_features(
    features_npz: Dict[str, np.ndarray],
    manifest_df: pd.DataFrame,
    cache_path: Path,
) -> np.ndarray:
    """Extract or load cached 576-D MobileNetV3 visual features for all dataset rows."""
    if cache_path.exists():
        print(f"Loading cached vision features from {cache_path}...")
        data = np.load(cache_path)
        return data["X_vision"]

    print("Extracting 576-D vision features with MobileNetV3-Small backbone...")
    t0 = time.time()
    extractor = VisualFeatureExtractor(device="cpu")
    row_ids = features_npz["row_ids"]
    f_raw = features_npz["features_raw"]
    n_samples = len(row_ids)
    X_vision = np.zeros((n_samples, 576), dtype=np.float32)

    for i, rid in enumerate(row_ids):
        row = manifest_df.loc[rid]
        crop_path_val = str(row["crop_path"]) if pd.notna(row["crop_path"]) else ""
        resolved_crop = REPO_ROOT / crop_path_val if crop_path_val else None

        if resolved_crop and resolved_crop.exists():
            im = cv2.imread(str(resolved_crop))
            vec = extractor.extract_crop(im, bbox=None)
        else:
            orig_path_val = str(row["original_path"])
            orig_path = REPO_ROOT / orig_path_val
            im = cv2.imread(str(orig_path))
            kpts = f_raw[i].reshape(17, 3)
            vis = kpts[:, 2] > 0.20
            if vis.any():
                bbox = (
                    float(kpts[vis, 0].min()),
                    float(kpts[vis, 1].min()),
                    float(kpts[vis, 0].max()),
                    float(kpts[vis, 1].max()),
                )
            else:
                bbox = None
            vec = extractor.extract_crop(im, bbox=bbox)
        X_vision[i] = vec

    print(f"Vision feature extraction complete in {time.time() - t0:.2f}s.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X_vision=X_vision)
    return X_vision


def update_model_digests(model_path: Path, digest: str) -> None:
    """Update configs/model_digests.yaml with the multimodal_fusion artifact entry."""
    digests_yaml_path = REPO_ROOT / "configs" / "model_digests.yaml"
    with open(digests_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    artifacts = data.setdefault("artifacts", {})
    rel_path = model_path.relative_to(REPO_ROOT).as_posix()
    artifacts["multimodal_fusion"] = {
        "path": rel_path,
        "sha256": digest,
    }

    with open(digests_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"Updated {digests_yaml_path} with multimodal_fusion sha256: {digest}")


def write_ablation_report(
    report_path: Path,
    model_sha256: str,
    acc_pose: float,
    f1_pose: float,
    acc_vis: float,
    f1_vis: float,
    acc_fused: float,
    f1_fused: float,
    clf_report: str,
    target_names: Sequence[str],
    f1_per_class: Dict[str, float],
) -> None:
    """Write comprehensive multimodal ablation report."""
    content = f"""# Phase 6 Multimodal Fusion Ablation Report

Provenance for the Phase 6 Multimodal Late-Fusion model (FUSE-01, FUSE-02, FUSE-03)
evaluated on the frozen held-out test split (100 samples) of data/processed/features.npz.

## Split Census

- Total rows: 498
- Split: train 398 / test 100
- Frozen split discipline: strictly zero hyperparameter tuning on test slice.

## Ablation Comparison Table

| Architecture | Input Features | Backbone / Estimator | Test Accuracy | Macro-F1 | Target Met (>=91% / >=0.90) |
| ------------ | -------------- | -------------------- | ------------- | -------- | --------------------------- |
| 59-D Pose Baseline | 59-D static skeleton | HistGradientBoosting | 0.8600 | 0.8613 | NO |
| 81-D Extended Pose (FUSE-01) | 81-D relational geometry | HistGradientBoosting (RobustScaler) | {acc_pose:.4f} | {f1_pose:.4f} | {'YES' if acc_pose >= 0.91 else 'NO'} |
| 576-D Vision Baseline (FUSE-02) | 576-D visual crop | MobileNetV3-Small + LogisticRegression | {acc_vis:.4f} | {f1_vis:.4f} | NO |
| **Multimodal Late Fusion (FUSE-03)** | **81-D Pose + 576-D Vision** | **0.5 * P_pose + 0.5 * P_vis** | **{acc_fused * 100:.2f}%** | **{f1_fused:.4f}** | **YES** |

## Per-Class Evaluation (Multimodal Late Fusion)

```
{clf_report}
```

### Class F1 Highlights
"""
    for name in target_names:
        score = f1_per_class.get(name, 0.0)
        content += f"- **{name}**: F1 = {score:.4f}\n"

    content += f"""
## Target Verification Gate

- Test Accuracy Bar: >= 91.0% -> **OBSERVED: {acc_fused * 100:.2f}%** (PASS)
- Macro-F1 Bar: >= 0.9000 -> **OBSERVED: {f1_fused:.4f}** (PASS)

## Model Artifact Integrity

- Artifact path: `models/multimodal_fusion.joblib`
- SHA-256 Digest: `{model_sha256}`
- Integrity verification: registered in `configs/model_digests.yaml` and verified fail-closed via `verify_artifact_sha256`.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    print(f"Wrote multimodal ablation report to {report_path}")


def train_and_evaluate() -> Tuple[MultimodalFusionClassifier, float, float]:
    """Train multimodal fusion bundle, evaluate, save, and update digests."""
    npz_path = REPO_ROOT / "data" / "processed" / "features.npz"
    manifest_path = REPO_ROOT / "data" / "processed" / "manifest.csv"
    vision_cache_path = REPO_ROOT / "data" / "processed" / "vision_features_cache.npz"
    model_path = REPO_ROOT / "models" / "multimodal_fusion.joblib"
    report_path = REPO_ROOT / "reports" / "multimodal_ablation.md"

    f = np.load(npz_path)
    manifest_df = pd.read_csv(manifest_path).set_index("row_id")

    f_raw = f["features_raw"]
    f_eng = f["features_eng"]
    labels = f["labels"]
    split = f["split"]

    train_mask = (split == "train")
    test_mask = (split == "test")

    # 1. Extended pose features (FUSE-01)
    X_pose = engineer_extended_features(f_raw, f_eng)
    print(f"Engineered 81-D pose features shape: {X_pose.shape}")

    # 2. Visual features (FUSE-02)
    X_vision = extract_or_load_vision_features(f, manifest_df, vision_cache_path)
    print(f"Visual features shape: {X_vision.shape}")

    # 3. Multimodal Classifier (FUSE-03)
    model = MultimodalFusionClassifier()
    model.fit(
        X_pose[train_mask],
        X_vision[train_mask],
        labels[train_mask],
    )

    # Branch evaluations for ablation report
    p_pose_te = model.pose_model.predict_proba(model.pose_scaler.transform(X_pose[test_mask]))
    pred_pose = model.label_encoder.inverse_transform(np.argmax(p_pose_te, axis=1))
    acc_pose = float(accuracy_score(labels[test_mask], pred_pose))
    f1_pose = float(f1_score(labels[test_mask], pred_pose, average="macro"))

    p_vis_te = model.vision_model.predict_proba(model.vision_scaler.transform(X_vision[test_mask]))
    pred_vis = model.label_encoder.inverse_transform(np.argmax(p_vis_te, axis=1))
    acc_vis = float(accuracy_score(labels[test_mask], pred_vis))
    f1_vis = float(f1_score(labels[test_mask], pred_vis, average="macro"))

    # Fused evaluation
    y_test = labels[test_mask]
    y_pred = model.predict(X_pose[test_mask], X_vision[test_mask])
    acc_fused = float(accuracy_score(y_test, y_pred))
    f1_fused = float(f1_score(y_test, y_pred, average="macro"))

    print("\n" + "=" * 50)
    print(f"MULTIMODAL LATE FUSION EVALUATION (100 frozen test samples)")
    print(f"Accuracy: {acc_fused * 100:.2f}% (Target: >= 91.0%)")
    print(f"Macro-F1: {f1_fused:.4f} (Target: >= 0.9000)")
    print("=" * 50)

    report_str = classification_report(y_test, y_pred, target_names=model.classes_)
    print(report_str)

    # Per-class F1 dict
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    f1_per_class = {cls_name: report_dict[cls_name]["f1-score"] for cls_name in model.classes_}

    # Strict target verification gate
    if acc_fused < 0.910:
        raise ValueError(
            f"Multimodal test accuracy {acc_fused * 100:.2f}% failed target bar >= 91.0%"
        )
    if f1_fused < 0.900:
        raise ValueError(
            f"Multimodal macro-F1 {f1_fused:.4f} failed target bar >= 0.9000"
        )

    # 4. Save bundle and verify digest
    digest = model.save(model_path)
    print(f"Saved multimodal fusion model to {model_path} (SHA-256: {digest})")

    # Verify reload integrity
    reloaded = MultimodalFusionClassifier.load(model_path, expected_sha256=digest)
    reloaded_pred = reloaded.predict(X_pose[test_mask], X_vision[test_mask])
    assert (reloaded_pred == y_pred).all(), "Reloaded model predictions mismatch"
    print("Verified reload integrity successfully.")

    # 5. Update configs/model_digests.yaml
    update_model_digests(model_path, digest)

    # 6. Write ablation report
    write_ablation_report(
        report_path=report_path,
        model_sha256=digest,
        acc_pose=acc_pose,
        f1_pose=f1_pose,
        acc_vis=acc_vis,
        f1_vis=f1_vis,
        acc_fused=acc_fused,
        f1_fused=f1_fused,
        clf_report=report_str,
        target_names=model.classes_,
        f1_per_class=f1_per_class,
    )

    return model, acc_fused, f1_fused


if __name__ == "__main__":
    train_and_evaluate()
