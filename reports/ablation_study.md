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

## Ablation table

Rows render in fixed A, B, C, D order; unregistered models are skipped (writers iterate MODULE_ORDER). Per-class F1 columns follow the LabelEncoder order.

| model | accuracy | macro-F1 | F1 Four | F1 NoAction | F1 NoBall | F1 Out | F1 Six | F1 Wide | ROC-AUC ovr-macro |
| ----- | -------- | -------- | ------- | ----------- | --------- | ------ | ------ | ------- | ----------------- |
| A | 0.7381 | 0.7227 | 0.6154 | 0.8333 | 0.7857 | 0.5926 | 0.9091 | 0.6000 | 0.9292 |
| B | 0.7738 | 0.7737 | 0.8000 | 0.8261 | 0.6667 | 0.7368 | 0.8571 | 0.7556 | 0.9603 |
| C | 0.7500 | 0.7412 | 0.6154 | 0.7500 | 0.7143 | 0.7059 | 0.9000 | 0.7619 | 0.9354 |
| D | 0.8452 | 0.8264 | 0.6667 | 0.8000 | 0.7857 | 0.8235 | 0.9524 | 0.9302 | 0.9543 |

Trained and evaluated this run: Model A, Model B, Model C, Model D.

## Augmentation probe (Model A, no augmentation)

unaugmented Model A: accuracy 0.7857 / macro-F1 0.7822
augmented Model A (main table): accuracy 0.7381 / macro-F1 0.7227
Interpretation: this probe is the empirical canary for the swap-only mirror flip and the model-input-space jitter (RESEARCH A2/A3) — if the unaugmented row matches or beats the augmented row beyond noise, that is a reported FINDING on the flip/jitter assumptions, never a trigger to re-tune on test (pre-registered discipline). The probe never enters the main table.
FINDING: the unaugmented probe MATCHES OR BEATS augmented Model A on this split (macro-F1 0.7822 vs 0.7227) — the canary flags the swap-only flip and/or model-input-space jitter assumptions (RESEARCH A2/A3); reported as measured, never re-tuned on test.

## Verdict

Best proposed model: D (macro-F1 0.8264)
Tie-break rule: ties resolve to the earliest of B, C, D in table order (explicit ordered scan over the fixed tuple — never np.argmax over a reordered collection).
Beats baseline (Model A): YES
macro-F1 >= 0.90 target: NOT MET

## Caveats

Four: n=5 test rows; the 24-usable < 30 flag is carried from the Phase 1 audit; per-class F1 for Four is small-sample noisy and never tuned on test.
single seed 42 everywhere (matches the locked split seed); no multi-seed mean+-std in this phase.

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
baseline_ann sha256: 5f1a595ede3b94e5ae04d4b8561aa67d7079b6b80d68ea8db05ce00d8114d5ae (models/baseline_ann.pt)
pose_mlp sha256: 6a1c4e7e35e2efbe83865bbe62a6ace89242275b4bb4c0a193fd2df990442962 (models/pose_mlp.pt)
ensemble_model sha256: 12c84a36bc574ce62da96bc6b09aa864c97bfdbc596169eaa1ece070b155562f (models/ensemble_model.joblib)
label_encoder sha256: b439eff2f6effdcdf056481243d1b9bd448d97b2a1eb9d16bedccce2b7fa8a0f (models/label_encoder.joblib)
Model A meta: fold_best_epochs [78, 43, 48, 60, 36]; final_epochs 53; batch_size 32
Model A cv protocol: StratifiedKFold(5, shuffle=True, random_state=42) (n_folds 5); max_epochs 150; early stop on val loss (patience 15); final epochs = round(mean fold-best); augmentation inside each fold's train portion only
Model A scaler: MinMaxScaler() default clip=False (exact-reproduction semantics)
Model B meta: fold_best_epochs [83, 75, 65, 51, 56]; final_epochs 66; batch_size 32
Model B cv protocol: StratifiedKFold(5, shuffle=True, random_state=42) (n_folds 5); max_epochs 150; early stop on val loss (patience 15); final epochs = round(mean fold-best); augmentation inside each fold's train portion only
Model B scaler: none (raw engineered rows — already normalized-scale, RESEARCH A2)
Model B optimizer: AdamW(lr=1e-3, weight_decay=1e-4) + CosineAnnealingLR(T_max=max_epochs=150, eta_min=1e-5), stepped per epoch
Model B loss: CrossEntropyLoss(label_smoothing=0.05) — NO class weights (locked: only Model C is weighted)
Model B input: features_eng[:, 0:51] (17 x (torso-normalized x, y, conf))
Model C meta: fold_best_epochs [22, 40, 37, 28, 25]; final_epochs 30; batch_size 32
Model C cv protocol: StratifiedKFold(5, shuffle=True, random_state=42) (n_folds 5); max_epochs 150; early stop on val loss (patience 15); final epochs = round(mean fold-best); augmentation inside each fold's train portion only
Model C scaler: none (raw engineered rows — already normalized-scale, RESEARCH A2)
Model C optimizer: AdamW(lr=1e-3, weight_decay=1e-4) — no scheduler (the cosine schedule is Model B's distinctive lever; C's levers are the residual block + class weighting)
Model C loss: CrossEntropyLoss(weight = N/(6 * n_c) inverse-frequency class weights, computed programmatically from the train label counts)
Model C input: features_eng (all 59 engineered features)
Model D input: features_eng (all 59 engineered features)
Model D members: ['model_c', 'xgboost', 'random_forest'] — soft vote = (p_C + p_XGB + p_RF) / 3.0 — manual numpy mean (the ONE sanctioned hand-roll: sklearn's voting-ensemble meta-estimator cannot host the torch Model C member)
Model D augmentation: fit-time wrappers (AugmentingXGBClassifier / AugmentingRandomForestClassifier) — per-CV-fold flip+jitter inside fit(), seed = member random_state 42; never augment-before-split
Model D grid protocol: GridSearchCV(scoring=f1_macro, cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=1) — {'xgb': 8, 'rf': 4} combos, tuned on the train slice only
Model D grid (XGBoost member): best params {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 300}; best cv f1_macro 0.8362
Model D grid (RandomForest member): best params {'max_depth': None, 'n_estimators': 200}; best cv f1_macro 0.7978
Model D model C source: Model C trained in the same run (weights ride the train-stage meta and are stored inside the bundle)
code version: phase3-plan01-r4
