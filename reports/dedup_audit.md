# Dataset ingestion + leak-free dedup audit (Phase 1)

Single source of truth for Phases 2-5: census, dedup, exclusions, the guarded post-dedup split, and the cross-split leak check. Deterministic: re-running src/data_ingestion.py reproduces this report and data/processed/manifest.csv byte-identically (no wall-clock timestamps).

## Parameter block

phash_threshold: 8
pad_frac: 0.15
test_size: 0.2
random_state: 42
min_class_flag: 30
min_class_split: 10
imagehash version: 4.3.2
Pillow version: 13.0.0
scipy version: 1.18.1
code version: phase1-plan03

## Per-source census

| source | root | family | raw count | expected (census) | sanity | raw classes |
|---|---|---|---|---|---|---|
| roboflow_umpire_wide | datasets/roboflow_umpire_wide | roboflow-yolo | 2497 | 2497 | ok | four, no_action, out, six, wide |
| umpire_gestures | datasets/Umpire_Gestures.v1i.yolov8 | roboflow-yolo | 145 | 145 | ok | DRS, LegBye, NoBall, WideBall |
| archive | datasets/archive | filename-label | 390 | 390 | ok | no_action, no_ball, out, sixes, wide |
| archive3 | datasets/archive3 | filename-label | 390 | 390 | ok | no_action, no_ball, out, sixes, wide |
| archive1 | datasets/archive1 | folder-label | 312 | 312 | ok | noball, out, six, wide |
| inception | datasets/Inception_test_train_split | filename-label | 843 | 843 | ok | no_action, no_ball, non_umpire, out, sixes, umpire, wide |
| umpire_pose_dataset.csv | datasets/umpire_pose_dataset.csv | precomputed-csv | 2909 rows | - | - | audit-only, excluded from the image pipeline |
roboflow_umpire_wide unique stems: 558 (expected 558) [ok]

## Totals

Raw image files: 4577
Kept after dedup: 434 (sanity band 1250-1450)
Sanity band verdict: OUT OF BAND - investigate dedup before Phase 2
Manifest class rows (post conditional evaluation): 421

## Duplicates removed

Duplicates removed - Roboflow stem-groups: 1939
  roboflow_umpire_wide: 1939 (file-level: 2497 files from 558 stems)
  umpire_gestures: 0 (file-level: 145 files from 145 stems)
  reconciliation: 1937 pooled copies removed by Layer A + 2 zero-box copies excluded as no_box before dedup (counted above) = 1939 file-level
Stem groups whose copies disagreed on raw_label: 298 (Layer A resolves by sort order — no conflict flag, unlike Layer B; not part of the conflict count below)
Duplicates removed - phash near-duplicates: 1719
  archive: 390
  archive1: 136
  archive3: 390
  inception: 317
  roboflow_umpire_wide: 486
  (Layer B is union-find over EVERY pair at Hamming <= threshold: near-identical sequences chain transitively — the export's video frames include distinct stems with identical 64-bit hashes, so long runs merge into single clusters. This is the locked CONTEXT policy (threshold 8) operating on real data; the sanity-band verdict above reports the result.)

## Exclusions (counted, never written as class rows)

DRS: 32 excluded
non_umpire: 213 excluded
umpire: 240 excluded
Zero-box images excluded as no_box: 2
Multi-box images flagged (largest-area box wins): 21
Invalid bbox rows rejected (out-of-range/malformed, ASVS V5): 0
Label decode replacements (utf-8, errors=replace): 0
unknown: 0 (unknown labels abort the run — this stays 0)

Conditional classes (INGEST-06: included only when the usable count is STRICTLY greater than 30):
DeadBall: 0 excluded (conditional class, no usable samples — 0 is not > 30)
LegBye: 13 excluded (conditional class, usable count 13 is not > 30)

## Crop rule (research-driven adjustment, logged explicitly)

bbox rows seen by class: DRS 32, LegBye 13, NoBall 48, WideBall 52, four 798, no_action 926, out 259, six 283, umpire 1, wide 250 (total 2662)
umpire-class bbox rows seen: 1 — logged explicitly; the umpire annotation class has 1 bbox row across the 2,497-image export because boxes carry the signal class (the umpire performing it)
Adjustment note: the CONTEXT decision 'when multiple umpire boxes exist, crop the largest-area umpire box' was unimplementable as written — the umpire class has 1 bbox row across 2,497 images. Operative rule: crop the largest-area annotated box of ANY class (every box denotes the umpire); the row's canonical class = that box's class. Intent preserved: every Roboflow image still yields the umpire crop with 15% clamped padding.

Label conflicts resolved (tier priority, majority at equal tier, lexicographic tiebreak): 106
  top conflict examples (discarded labels x count):
    sixes x44
    no_ball x39
    wide x7
    no_action x6
    no_ball; noball x4

## Per-class post-dedup counts (assigned split)

  Class         Train    Test   Total
  Four             17       4      21
  NoAction         85      21     106
  NoBall           69      18      87
  Out              39      10      49
  Six              44      11      55
  Wide             82      21     103
FLAG Four: 21 usable samples < 30 — collect more 'Four' images before Phase 2 (collect-more contingency)

Cross-split phash leaks (Hamming <= 8): 0
  zero near-duplicate pairs across the train/test boundary — leak-free split (dedup merged every pair within the threshold BEFORE the split existed)

Crops written (crop_<row_id>.jpg): 172, non-Roboflow full-image rows 249, invalid/degenerate 0, failed 0

Determinism contract: sorted walks, fixed random_state, static code version tag, no wall-clock timestamps — two consecutive full runs produce byte-identical manifest.csv and dedup_audit.md.