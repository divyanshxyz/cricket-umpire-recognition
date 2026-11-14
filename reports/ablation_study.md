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
| B | 0.7738 | 0.7737 | 0.8000 | 0.8261 | 0.6667 | 0.7368 | 0.8571 | 0.7556 | 0.9603 |
| C | 0.7500 | 0.7412 | 0.6154 | 0.7500 | 0.7143 | 0.7059 | 0.9000 | 0.7619 | 0.9354 |
| D | 0.8452 | 0.8264 | 0.6667 | 0.8000 | 0.7857 | 0.8235 | 0.9524 | 0.9302 | 0.9543 |

Trained and evaluated this run: Model A, Model B, Model C, Model D.

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
models/baseline_ann.pt sha256: b4058adff1e22211935061d5dda1a03fb1ae6385e3d9d4c68cc3f4d4685a82a3
Model A meta: fold_best_epochs [78, 43, 48, 60, 36]; final_epochs 53; batch_size 32
Model A cv protocol: StratifiedKFold(5, shuffle=True, random_state=42) (n_folds 5); max_epochs 150; early stop on val loss (patience 15); final epochs = round(mean fold-best); augmentation inside each fold's train portion only
Model A scaler: MinMaxScaler() default clip=False (exact-reproduction semantics)
models/pose_mlp.pt sha256: 3ec38a0fbabdb4d018bd5f092fc334e5124321750f7caf72443571095685e80b
Model B meta: fold_best_epochs [83, 75, 65, 51, 56]; final_epochs 66; batch_size 32
Model B cv protocol: StratifiedKFold(5, shuffle=True, random_state=42) (n_folds 5); max_epochs 150; early stop on val loss (patience 15); final epochs = round(mean fold-best); augmentation inside each fold's train portion only
Model B scaler: none (raw engineered rows — already normalized-scale, RESEARCH A2)
Model B optimizer: AdamW(lr=1e-3, weight_decay=1e-4) + CosineAnnealingLR(T_max=max_epochs=150, eta_min=1e-5), stepped per epoch
Model B loss: CrossEntropyLoss(label_smoothing=0.05) — NO class weights (locked: only Model C is weighted)
Model B input: features_eng[:, 0:51] (17 x (torso-normalized x, y, conf))
models/ensemble_model.joblib sha256: 2650a0fc6e0adef28d457eb336d20c568c9a73c3b3add438671795e9201001ab
Model C meta: fold_best_epochs [22, 40, 37, 28, 25]; final_epochs 30; batch_size 32
Model C cv protocol: StratifiedKFold(5, shuffle=True, random_state=42) (n_folds 5); max_epochs 150; early stop on val loss (patience 15); final epochs = round(mean fold-best); augmentation inside each fold's train portion only
Model C scaler: none (raw engineered rows — already normalized-scale, RESEARCH A2)
Model C optimizer: AdamW(lr=1e-3, weight_decay=1e-4) — no scheduler (the cosine schedule is Model B's distinctive lever; C's levers are the residual block + class weighting)
Model C loss: CrossEntropyLoss(weight = N/(6 * n_c) inverse-frequency class weights, computed programmatically from the train label counts)
Model C input: features_eng (all 59 engineered features)
Model D input: features_eng (all 59 engineered features)
Model D members: ['model_c', 'xgboost', 'random_forest'] — soft vote = (p_C + p_XGB + p_RF) / 3.0 — manual numpy mean (the ONE sanctioned hand-roll: sklearn VotingClassifier cannot host the torch Model C member)
Model D augmentation: fit-time wrappers (AugmentingXGBClassifier / AugmentingRandomForestClassifier) — per-CV-fold flip+jitter inside fit(), seed = member random_state 42; never augment-before-split
Model D grid protocol: GridSearchCV(scoring=f1_macro, cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=1) — {'xgb': 8, 'rf': 4} combos, tuned on the train slice only
Model D grid (XGBoost member): best params {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 300}; best cv f1_macro 0.8362
Model D grid (RandomForest member): best params {'max_depth': None, 'n_estimators': 200}; best cv f1_macro 0.7978
Model D model C source: Model C trained in the same run (weights ride the train-stage meta and are stored inside the bundle)
single-seed caveat: every number above is one seed-42 run (multi-seed mean±std deferred per CONTEXT)
code version: phase3-plan01-r3
