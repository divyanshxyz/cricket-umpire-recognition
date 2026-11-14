"""Content-hash deduplication.

Layer A: Roboflow stem groups (:func:`stem_groups` — strips the export's own
``.rf.<32hex>`` source-frame ID within each Roboflow source; wired into the
run by Plan 02). Layer B: imagehash.phash clustered by brute-force pairwise
Hamming distance <= threshold (INCLUSIVE at 8; distance 9 is distinct) via
union-find — verified-fast at this dataset scale, no index. Cluster labels
resolve per the locked policy (:func:`resolve_cluster_labels`): higher
priority tier wins across tiers, majority label wins at equal tier.

Determinism (RESEARCH.md Pattern 4 + Pitfall 9): candidate pairs come from a
row-major scan, union roots resolve to the smaller index, and each cluster's
survivor is selected by (highest priority, lexicographic relpath, source) —
output ordering is fully deterministic across runs.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from imagehash import phash
from PIL import Image

RF_HASH = re.compile(r"\.rf\.[0-9a-f]{32}$")  # Roboflow suffix per export naming


def roboflow_stem(filename: str) -> str:
    """Strip the '.rf.<32 hex>' Roboflow suffix from a filename stem.

    frame_0028_jpg.rf.4ab9... -> frame_0028_jpg. This is Roboflow's own
    source-frame ID (Layer A grouping key), not a filename dedup key —
    cross-source identity stays content-hash based.
    """
    stem = Path(filename).stem
    return RF_HASH.sub("", stem)


def stem_groups(records):
    """Layer A: group Roboflow-family records by (source, stripped .rf stem).

    SEAM STUB — Plan 02 wires this layer (one survivor per stripped stem,
    lexicographically smallest relpath; removed records reported separately
    from phash removals per RESEARCH.md Pitfall 1) behind this contract.
    """
    return [], []


def resolve_cluster_labels(cluster_records):
    """Cluster-level label resolution (locked CONTEXT policy).

    SEAM STUB — Plan 02 implements: higher priority tier wins across tiers;
    majority label wins at equal tier (ties broken by lexicographic label
    for determinism); conflict=True + discarded labels in the detail.
    Returns (winner_raw_label, conflict, conflict_detail).
    """
    return "", False, ""


def phash_file(path) -> str:
    """64-bit perceptual hash of an image file, as 16 hex chars.

    imagehash.phash defaults ARE the spec (RESEARCH.md lines 393-403):
    hash_size=8, highfreq_factor=4 -> 32x32 grayscale thumbnail, 64-bit hash.
    """
    with Image.open(path) as im:
        return str(phash(im.convert("RGB")))


def pack64(hex_strs) -> np.ndarray:
    """Pack 16-hex-char strings into a uint64 array (RESEARCH.md lines 411-427)."""
    return np.array([int(s, 16) for s in hex_strs], dtype=np.uint64)


def popcount(x: np.ndarray) -> np.ndarray:
    """Bit population count for uint64 arrays (RESEARCH.md lines 411-427)."""
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def cluster_by_phash(records, threshold: int = 8):
    """Cluster records by phash Hamming distance <= threshold (INCLUSIVE).

    Brute-force pairwise XOR+popcount, union-find over candidate pairs
    (np.triu keeps i < j; transitive chaining is a known, accepted risk —
    RESEARCH.md Sources, tertiary). Survivor per cluster: highest priority,
    then lexicographically smallest relpath, then source (deterministic
    tiebreak on the (priority, relpath) contract).

    Returns (survivors, removed); a single-record union yields one survivor
    cluster and zero removals.
    """
    records = list(records)
    n = len(records)
    if n == 0:
        return [], []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n > 1:
        hashes = [rec.phash_hex for rec in records]
        for rec, h in zip(records, hashes):
            if h is None:
                raise ValueError(
                    f"record without phash_hex: {rec.source}/{rec.relpath} "
                    "(hash before clustering)"
                )
        H = pack64(hashes)
        D = popcount(np.bitwise_xor.outer(H, H))
        for i, j in np.argwhere(np.triu(D <= threshold, 1)):
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

    clusters: dict[int, list] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(records[i])

    survivors, removed = [], []
    for members in clusters.values():
        ordered = sorted(members, key=lambda r: (-r.priority, r.relpath, r.source))
        survivors.append(ordered[0])
        removed.extend(ordered[1:])
    survivors.sort(key=lambda r: (r.source, r.relpath))
    removed.sort(key=lambda r: (r.source, r.relpath))
    return survivors, removed
