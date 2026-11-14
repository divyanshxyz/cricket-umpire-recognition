# Phase 3 ablation study report

Provenance for the Phase 3 ablation over the frozen data/processed/features.npz (code version lineage: reports/features_build.md phase2-plan01-r2, reports/dedup_audit.md phase1-plan03-r2). Deterministic: no wall-clock timestamps; single guarded evaluation entry point; the test-slice digest is recorded at training start and asserted unchanged before any metric write. The split column is consumed as data — never re-split.

## Counts

total ok rows: 418
split census (frozen split column): train 334 / test 84
Per-class train/test counts (LabelEncoder order):
  Four: 19/5
  NoAction: 83/21
  NoBall: 69/16
  Out: 39/10
  Six: 43/11
  Wide: 81/21
augmentation: mirror flip (x2) + jitter N(0, 0.015) in model-input space — 334 -> 668 effective rows (inside each CV fold's train portion; the test slice is never augmented and the scaler is never refit on it)
Caveat: Four per-class F1 is computed on n=5 test rows (Phase 1 audit flag: 24 usable < 30) — small-sample noisy by construction; never re-tune on test to move it (pre-registered discipline).

## Ablation table

Rows render in fixed A, B, C, D order; unregistered models are skipped (writers iterate MODULE_ORDER). Per-class F1 columns follow the LabelEncoder order.

| model | accuracy | macro-F1 | F1 Four | F1 NoAction | F1 NoBall | F1 Out | F1 Six | F1 Wide | ROC-AUC ovr-macro |
| ----- | -------- | -------- | ------- | ----------- | --------- | ------ | ------ | ------- | ----------------- |
| A | 0.7381 | 0.7227 | 0.6154 | 0.8333 | 0.7857 | 0.5926 | 0.9091 | 0.6000 | 0.9292 |

Trained and evaluated this run: Model A.

## Parameter block

seed: 42
torch num threads: 1
torch version: 2.15.0+cpu
torchvision version: 0.29.2+cpu
ultralytics version: 8.4.212
scikit-learn version: 1.9.1
numpy version: 2.2.6
xgboost version: 3.4.1
joblib version: 1.6.0
matplotlib version: 3.11.2
features.npz sha256: 42b9c5cc714e3c22005b9c612390a05075f9a7df801678269bb0453d5b6473cd
manifest sha256: eca8579b4adf794af2bf4d211a598f42f20a933b5049c26268112da2169ef67a
test-slice digest: c75427be36a02e59fe1d1a92abeded39107a7935cb6db851fea37618c735efa8
models/label_encoder.joblib sha256: b439eff2f6effdcdf056481243d1b9bd448d97b2a1eb9d16bedccce2b7fa8a0f
models/baseline_ann.pt sha256: 87a223be1454dbace634a792af1c175c8d72ddb4c6c63a6bf1ebf96a2f7cf26d
Model A meta: fold_best_epochs [78, 43, 48, 60, 36]; final_epochs 53; batch_size 32
Model A cv protocol: StratifiedKFold(5, shuffle=True, random_state=42) (n_folds 5); max_epochs 150; early stop on val loss (patience 15); final epochs = round(mean fold-best); augmentation inside each fold's train portion only
Model A scaler: MinMaxScaler() default clip=False (exact-reproduction semantics)
single-seed caveat: every number above is one seed-42 run (multi-seed mean±std deferred per CONTEXT)
code version: phase3-plan01-r1
