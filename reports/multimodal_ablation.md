# Phase 6 Multimodal Fusion Ablation Report

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
| 81-D Extended Pose (FUSE-01) | 81-D relational geometry | HistGradientBoosting (RobustScaler) | 0.8600 | 0.8615 | NO |
| 576-D Vision Baseline (FUSE-02) | 576-D visual crop | MobileNetV3-Small + LogisticRegression | 0.7800 | 0.7659 | NO |
| **Multimodal Late Fusion (FUSE-03)** | **81-D Pose + 576-D Vision** | **0.5 * P_pose + 0.5 * P_vis** | **92.00%** | **0.9072** | **YES** |

## Per-Class Evaluation (Multimodal Late Fusion)

```
              precision    recall  f1-score   support

        Four       0.78      0.78      0.78         9
    NoAction       0.92      0.96      0.94        25
      NoBall       0.86      0.95      0.90        19
         Out       0.91      0.83      0.87        12
         Six       1.00      1.00      1.00        12
        Wide       1.00      0.91      0.95        23

    accuracy                           0.92       100
   macro avg       0.91      0.91      0.91       100
weighted avg       0.92      0.92      0.92       100

```

### Class F1 Highlights
- **Four**: F1 = 0.7778
- **NoAction**: F1 = 0.9412
- **NoBall**: F1 = 0.9000
- **Out**: F1 = 0.8696
- **Six**: F1 = 1.0000
- **Wide**: F1 = 0.9545

## Target Verification Gate

- Test Accuracy Bar: >= 91.0% -> **OBSERVED: 92.00%** (PASS)
- Macro-F1 Bar: >= 0.9000 -> **OBSERVED: 0.9072** (PASS)

## Model Artifact Integrity

- Artifact path: `models/multimodal_fusion.joblib`
- SHA-256 Digest: `7198d159e08616aaa75ffeef71dff90ed5b9f5e0d593aa9e60dfd305b0396f90`
- Integrity verification: registered in `configs/model_digests.yaml` and verified fail-closed via `verify_artifact_sha256`.
