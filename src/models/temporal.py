"""Phase 2 readiness stub: TwoHeadAttentionBiGRU + pad_or_truncate (P2RDY-01).

Architecture source: Dey et al. (Mini1) two-head attention DC-GRU — Input
→ Dense(128, GELU) → MultiHeadAttention(2 heads) → BiGRU(128) → BiGRU(128)
→ Linear, with the frozen shape chain (RESEARCH.md Pitfall 5):
(B,15,118) → Linear(118,128)+GELU → (B,15,128) → MHA(128, heads=2) self-attn
→ (B,15,128) → GRU(128→128, bi) → (B,15,256) → GRU(256→128, bi) → (B,15,256)
→ Linear(256, n_classes) → (B,15,n_classes) per-timestep logits (no pooling,
researcher disposition 4 / A5).

FORWARD-PASS ONLY stub (P2RDY-01): no training loop, no optimizer, no
checkpoint save — training ships with Phase 3/4. The P2RDY-02 image/temporal
config switch is deferred to Phase 4 per the recorded roadmap decision (do
not pre-implement it here). ``pad_or_truncate`` assembles a (T,118)
extract_sequence_features output into the stub's (15,118) input (A6:
zero-pad / truncate at the END; a 15-frame sequence is returned untouched).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def pad_or_truncate(seq: np.ndarray, target_len: int = 15) -> np.ndarray:
    """Assemble a (T, F) sequence-feature array into (target_len, F).

    Zero-pads AT THE END when T < target_len, truncates FROM THE END
    (keeping the FIRST target_len rows) when T > target_len, and returns
    the input's values unchanged when T == target_len [A6 adopted: the
    padding side is revisitable in Phase 4 without contract churn].
    Pure numpy — no torch, no weights.
    """
    arr = np.asarray(seq)
    t = arr.shape[0]
    if t == target_len:
        return arr
    if t > target_len:
        return arr[:target_len]
    pad_width = [(0, target_len - t)] + [(0, 0)] * (arr.ndim - 1)
    return np.pad(arr, pad_width, mode="constant", constant_values=0.0)


class TwoHeadAttentionBiGRU(nn.Module):
    """Input(B,15,F) → Dense(128,GELU) → MHA(2 heads) → BiGRU(128) →
    BiGRU(128) → Linear (P2RDY-01, forward-pass only)."""

    def __init__(self, in_features: int = 118, n_classes: int = 6):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_features, 128), nn.GELU())
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=2, batch_first=True)
        self.gru1 = nn.GRU(128, 128, batch_first=True, bidirectional=True)  # → (B,15,256)
        self.gru2 = nn.GRU(256, 128, batch_first=True, bidirectional=True)  # → (B,15,256)
        self.head = nn.Linear(256, n_classes)

    def forward(self, x):
        """x: (B, 15, F) batch of sequence features → per-timestep logits
        (B, 15, n_classes). No pooling — Phase 4 adds it when the temporal
        head actually trains (researcher disposition 4)."""
        h = self.proj(x)                # (B,15,128)
        a, _ = self.attn(h, h, h)       # self-attention, (B,15,128)
        g1, _ = self.gru1(a)            # (B,15,256)
        g2, _ = self.gru2(g1)           # (B,15,256)
        return self.head(g2)            # (B,15,n_classes)
