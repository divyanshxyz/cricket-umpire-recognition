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
from collections import Counter
from pathlib import Path

import numpy as np
from imagehash import phash
from PIL import Image, ImageOps

RF_HASH = re.compile(r"\.rf\.[0-9a-f]{32}$")  # Roboflow suffix per export naming


def roboflow_stem(filename: str) -> str:
    """Strip the '.rf.<32 hex>' Roboflow suffix from a filename (or stem).

    frame_0028_jpg.rf.4ab9... -> frame_0028_jpg, for both the raw filename
    ('...rf.<32hex>.jpg' — extension dropped first) and the extensionless
    stem ('...rf.<32hex>' — the suffix ends the string). A 31-hex (or
    otherwise non-32-hex) suffix never matches and is NOT stripped. This is
    Roboflow's own source-frame ID (Layer A grouping key), not a filename
    dedup key — cross-source identity stays content-hash based.
    """
    text = str(filename)
    if not RF_HASH.search(text):
        text = Path(text).stem  # drop a trailing image extension (.jpg) first
    return RF_HASH.sub("", text, count=1)


def stem_groups(records):
    """Layer A: group Roboflow-family records by (source, stripped .rf stem).

    Roboflow exports bake N augmented copies of one source frame into the
    tree (measured median pairwise Hamming 30 — phash at threshold 8 cannot
    catch them, RESEARCH.md Pitfall 1), so this layer removes them first and
    reports its removals SEPARATELY from phash removals. Non-Roboflow-family
    records pass through untouched. Survivor within a group = lexicographic
    smallest relpath (deterministic). Returns (survivors, removed).
    """
    groups: dict[tuple[str, str], list] = {}
    passthrough: list = []
    for rec in records:
        if rec.family == "roboflow-yolo":
            groups.setdefault((rec.source, roboflow_stem(rec.relpath)), []).append(rec)
        else:
            passthrough.append(rec)
    survivors: list = []
    removed: list = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda r: r.relpath)
        survivors.append(members[0])
        removed.extend(members[1:])
    survivors.extend(passthrough)
    survivors.sort(key=lambda r: (r.source, r.relpath))
    removed.sort(key=lambda r: (r.source, r.relpath))
    return survivors, removed


def resolve_cluster_labels(cluster_records):
    """Cluster-level label resolution (locked CONTEXT policy).

    Higher priority tier wins across tiers; at equal tier the majority
    raw_label wins, with ties broken by lexicographically smallest label
    (deterministic). conflict=True when the cluster carried more than one
    distinct raw label; conflict_detail lists the discarded labels (sorted,
    semicolon-joined). Returns (winner_raw_label, conflict, conflict_detail).
    """
    members = list(cluster_records)
    labels = {r.raw_label for r in members}
    top_tier = max(r.priority for r in members)
    tier_members = [r for r in members if r.priority == top_tier]
    counts = Counter(r.raw_label for r in tier_members)
    winner = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]
    discarded = sorted(str(label) for label in labels if label != winner)
    conflict = len(labels) > 1
    detail = "; ".join(discarded)
    return winner, conflict, detail


def phash_file(path) -> str:
    """64-bit perceptual hash of an image file, as 16 hex chars.

    imagehash.phash defaults ARE the spec (RESEARCH.md lines 393-403):
    hash_size=8, highfreq_factor=4 -> 32x32 grayscale thumbnail, 64-bit hash.

    EXIF orientation is applied first (IN-07): two visual duplicates carried
    with different un-baked rotation tags would otherwise hash far apart and
    evade dedup. A no-op for the current dataset (every orientation tag in
    datasets/ is 1 — Roboflow re-exports bake rotation into the pixels).
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
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

    Cluster labels resolve via the locked policy
    (:func:`resolve_cluster_labels`): the survivor's raw_label becomes the
    winning label and the survivor carries conflict=True + the discarded
    labels in conflict_detail when the cluster disagreed. assemble_rows
    re-maps raw_label -> canonical afterwards, so the manifest row carries
    the resolved label.

    Returns (survivors, removed); a single-record union yields one survivor
    cluster and zero removals. Keyed by content hash ONLY — never by
    filename (CONTEXT locked decision).
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
        survivor = ordered[0]
        winner, conflict, detail = resolve_cluster_labels(members)
        survivor.raw_label = winner
        survivor.conflict = conflict
        survivor.conflict_detail = detail
        survivors.append(survivor)
        removed.extend(ordered[1:])
    survivors.sort(key=lambda r: (r.source, r.relpath))
    removed.sort(key=lambda r: (r.source, r.relpath))
    return survivors, removed
