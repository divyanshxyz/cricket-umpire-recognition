# Cricket Umpire Signal Recognition (Phase 1)

## What This Is

An end-to-end computer vision pipeline that recognizes cricket umpire hand signals from static images and video, and automatically updates a cricket scoreboard from detected signals. It is a research-grade Python project (Streamlit web app included) comparing a reproduction of the Mili et al. pose-keypoint ANN baseline against proposed torso-normalized pose-geometry models and a hybrid ensemble — built modularly so a Phase 2 temporal Attention-BiGRU video model can be plugged in later without rewriting the feature extractor or the web UI.

## Core Value

The ablation study must demonstrate the proposed pose-geometry models outperforming the reproduced Mili et al. baseline (target: macro-F1 ≥ 0.90 on the held-out test split), delivered through a working sliding-window video inference app with an automated cricket scoreboard.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Unified multi-format dataset ingestion (Roboflow YOLOv8 + folder/filename-labeled) with canonical class mapping and leak-free deduplication
- [ ] Pose + geometric feature engineering (51 raw keypoints, 59 engineered features) via YOLOv8n-pose
- [ ] Training + ablation study: Mili et al. baseline reproduction vs 3 proposed models
- [ ] Phase-2-ready video inference engine with EMA smoothing, confidence gating, cooldown, and automated scoreboard state
- [ ] Streamlit web app: image analyzer, video analyzer + live scoreboard, research metrics dashboard

### Out of Scope

- [Phase 2 temporal model training] — video-clip dataset does not exist yet; only the architecture stub and sequence-feature plumbing ship now
- [DRS and other non-core gesture classes] — filtered to 6 core classes per canonical mapping
- [Real-time webcam/stream inference] — uploaded .mp4 and static images only in Phase 1
- [Cloud deployment] — local laptop execution is the target environment

## Context

**Research foundation (read and anchored):**
- Mini9.pdf — Mili et al., "Pose Recognition in Cricket using Keypoints": SNOW dataset (5 classes, 390 images, 78/class), 18 MobileNet keypoints → 54 features → min-max normalization → 2-layer ANN (Adam, dropout 0.3), 80/20 split, 87% accuracy. This is the Model A baseline reproduction target (adapted to YOLOv8n-pose's 17 COCO keypoints / 51 features).
- Mini1.pdf — Dey et al. 2024, "Umpire's Signal Recognition in Cricket Using an Attention based DC-GRU Network": CUAVd video dataset, Two-Head Attention DC-GRU. Source of the Phase 2 `TwoHeadAttentionBiGRU` stub architecture.
- Mini8.pdf — earlier IJRESM paper: Caffe pose estimation + DNN + Random Forest, 8 classes, ~77–85% accuracy. Historical context only.

**Datasets on disk (workspace `datasets/` folder, audited 2026-11-14):**
- `datasets/archive` — filename-labeled (`no_action_N.jpg` etc.), train/test split, 312 train files (SNOW-derived)
- `datasets/archive1` — folder-labeled: noball/out/six/wide, ~54 train / 13 test / 11 validate per class
- `datasets/archive3` — filename-labeled, appears identical to `archive` (deduplication critical)
- `datasets/Inception_test_train_split` — filename-labeled: non_umpire_* and no_action_* files overlapping archive names (deduplication critical); `non_umpire` is not a signal class
- `datasets/roboflow_umpire_wide` — Roboflow YOLOv8 bbox export: four, no_action, out, six, wide + generic `umpire` (bbox usable for cropping)
- `datasets/Umpire_Gestures.v1i.yolov8` — Roboflow YOLOv8 bbox export: DRS, LegBye, NoBall, WideBall

**Known data risks:** "Four" exists only in `roboflow_umpire_wide` — expected to be the scarcest core class. DRS is filtered (not in core 6). LegBye/DeadBall included only if >30 samples. SNOW images recur across archive/archive3/Inception splits — perceptual-hash dedup must run before feature extraction so no image leaks between train and test.

**Domain classes (canonical 6):** Four, Six, Out, Wide, NoBall, NoAction. Scoreboard semantics: Four/Six/Out advance legal balls; Wide/NoBall add 1 run + 1 extra without advancing legal balls; overs formatted `legal_balls // 6.legal_balls % 6`.

## Constraints

- **Hardware**: Local laptop — AMD Ryzen 5, 16GB RAM, NVIDIA GTX 1650 Ti (4GB VRAM) or CPU fallback. All training/inference must run within this budget.
- **Tech stack**: Python; Ultralytics `yolov8n-pose.pt` for pose estimation; PyTorch + scikit-learn + XGBoost for models; Streamlit for UI.
- **Layout**: Project code lives at the workspace root (configs/, src/, models/, reports/, app.py) with `datasets/` as the sibling raw-data source — NOT a nested `umpire-signal-recognition/` folder.
- **Evaluation discipline**: Strict stratified 80/20 split (random_state=42); test set never touched by augmentation or tuning; pose augmentation (mirror flip + jitter) applied to train only.
- **Modularity**: `PoseFeatureExtractor` and the web UI must be reusable by Phase 2 temporal training without rewrites; `extract_sequence_features` + `TwoHeadAttentionBiGRU` stub ship now.
- **Success bar**: Best proposed model reaches macro-F1 ≥ 0.90 AND beats the reproduced baseline; if the bar is not reached, report honestly in the ablation table (no tuning on test).

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build at workspace root, not nested `umpire-signal-recognition/` | Datasets already live at root; simplest paths, one git repo | — Pending |
| Success bar: macro-F1 ≥ 0.90 + beats baseline | Research-grade result for a clean 6-class pose problem | — Pending |
| GSD roadmap covers Phase 1 only | Phase 2 needs a video-clip dataset that doesn't exist yet; stubs ship now | — Pending |
| Use Mini1/Mini8/Mini9 PDFs as research anchors | They define the baseline (Mili et al.), Phase 2 architecture (Dey et al.), and historical context | — Pending |
| 6 core classes; DRS filtered; LegBye/DeadBall conditional (>30 samples) | Canonical mapping from spec; avoids fragmenting scarce data | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-11-14 after initialization*
