"""Phase 2/3 model package.

Phase 2: temporal readiness stubs (forward-pass only) — the P2RDY-01
TwoHeadAttentionBiGRU stub consumes the Plan 01 (T, 118) sequence-feature
contract as-is; Phase 4 plugs the trained temporal model in here without
rewrites.

Phase 3: the training package. data.py is the torch-free data seam (frozen
features.npz loader with fail-closed validation + the cross-split adjacency
guard, split masks, the shared label encoder, the mirror-flip permutation +
flip/jitter augmentation, and the test-slice digest guard); baseline_ann.py
is Model A (the exact Mili et al. reproduction, TRAIN-03) plus the shared
5-fold train-CV protocol reused by Models B/C/D; pose_mlp.py holds Models
B/C and ensemble.py holds Model D (Plans 03-02/03-03 extend this inventory).
No imports, no re-exports.
"""
