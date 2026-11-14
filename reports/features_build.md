# Phase 2 features build report

Provenance for the frozen data/processed/features.npz over the Phase 1 manifest (code version lineage: reports/dedup_audit.md, phase1-plan03-r1). Deterministic: no wall-clock timestamps; the npz zip bytes differ per run (numpy stamps local time), so determinism is claimed at parameter-text + array level only.

## Counts

total manifest rows: 421
ok: 415
failed: 6
  no_person: 6
  no_keypoints: 0
  path_escape: 0
  file_missing: 0
  decode_error: 0
  keypoints_without_confidence: 0
  other: 0
reconciliation: ok + failed == total (415 + 6 == 421)
scale fallback warnings: 0

Per-class OK counts:
  Four: 21
  NoAction: 104
  NoBall: 85
  Out: 49
  Six: 54
  Wide: 102

Per-split OK counts:
  test: 82
  train: 333

Per-class zero-OK red flags: none

## Parameter block

det_conf: 0.25
imgsz: 640
mask_conf: 0.2
device: cpu
torch num threads: 8
torch version: 2.15.0+cpu
torchvision version: 0.29.2+cpu
ultralytics version: 8.4.212
opencv-python version: 4.14.0.94
numpy version: 2.2.6
weights sha256: c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212
manifest sha256: 5101eeade2f9596032d6114ddf174aeed4c57897bf2680c0f559829afe56ba5d
code version: phase2-plan01-r1
