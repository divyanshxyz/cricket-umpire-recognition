"""Deterministic manifest row assembly, guarded split assignment, leak check,
and CSV writer (data/processed/manifest.csv).

One row per kept image (RESEARCH.md architecture box 7). The split column is
assigned HERE, only after dedup (:func:`assign_split` — guarded stratified
80/20, seed 42); no source's own split (split0) ever determines it. All paths
are repo-root-relative posix strings (RESEARCH.md Pitfall 8).
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .canonical import UnknownLabelError, map_label
from .dedup import pack64, popcount

# Manifest column contract (frozen in Plan 01; Plans 02/03 fill crop/split/conflict).
COLUMNS = [
    "row_id", "source", "original_path", "family", "split0", "raw_label",
    "canonical_class", "crop_x0", "crop_y0", "crop_x1", "crop_y1",
    "crop_path", "phash_hex", "split", "conflict", "conflict_detail",
]


def assemble_rows(records, class_cfg: dict, roots=None) -> list[dict]:
    """Run map_label per record and build the post-dedup manifest rows.

    map_label is pure, so running it here on already-mapped records is
    idempotent (the CLI maps in the canonical stage first per the
    architecture order; unknown labels raise UnknownLabelError naming label
    + source — a hard error, never a silent skip).

    Only outcome 'keep'/'conditional' records become class rows; excluded and
    no_box records are counted by the audit and never written as class rows.
    Rows are sorted by (source, relpath); row_id = f"{source}_{i:05d}" with i
    restarting per source — fully deterministic.
    """
    mapped = []
    for rec in records:
        try:
            outcome, canonical, reason = map_label(rec.raw_label, class_cfg)
        except UnknownLabelError as err:
            raise UnknownLabelError(f"source '{rec.source}' ({rec.relpath}): {err}") from err
        rec.outcome, rec.canonical, rec.exclude_reason = outcome, canonical, reason
        if outcome in ("keep", "conditional"):
            mapped.append(rec)
    mapped.sort(key=lambda r: (r.source, r.relpath))

    roots = roots or {}
    rows: list[dict] = []
    per_source: dict[str, int] = {}
    for rec in mapped:
        i = per_source.get(rec.source, 0)
        per_source[rec.source] = i + 1
        root = str(roots.get(rec.source, "")).replace("\\", "/").rstrip("/")
        original_path = f"{root}/{rec.relpath}" if root else rec.relpath
        rows.append({
            "row_id": f"{rec.source}_{i:05d}",
            "source": rec.source,
            "original_path": original_path,
            "family": rec.family,
            "split0": rec.split0 or "",
            "raw_label": rec.raw_label or "",
            "canonical_class": rec.canonical or "",
            "crop_x0": "",
            "crop_y0": "",
            "crop_x1": "",
            "crop_y1": "",
            "crop_path": "",  # Plan 02's crop stage fills crop_{row_id}.jpg for Roboflow rows
            "phash_hex": rec.phash_hex or "",
            "split": "",  # filled by assign_split AFTER dedup (never from split0)
            "conflict": bool(rec.conflict),  # cluster disagreed on the raw label
            "conflict_detail": rec.conflict_detail or "",  # discarded labels
        })
    return rows


def evaluate_conditional(counts, min_class_flag: int) -> dict:
    """INGEST-06 strict >30 rule for conditional classes (LegBye/DeadBall).

    ``counts`` maps conditional class -> usable (post-dedup) sample count; a
    class joins the manifest ONLY when its count is STRICTLY greater than
    ``min_class_flag`` — exactly 30 is excluded. A class with zero usable
    samples is excluded with its reason recorded (never silently omitted).

    Returns ``{class: {"count": n, "included": bool, "reason": str}}`` — the
    reason strings render verbatim in the audit's decision lines.
    """
    decision: dict[str, dict] = {}
    for cls in sorted(counts):
        n = int(counts[cls])
        included = n > min_class_flag
        if n == 0:
            reason = f"no usable samples — 0 is not > {min_class_flag}"
        elif included:
            reason = f"usable count {n} > {min_class_flag}"
        else:
            reason = f"usable count {n} is not > {min_class_flag}"
        decision[cls] = {"count": n, "included": included, "reason": reason}
    return decision


def assign_split(rows, test_size, random_state, min_class_split) -> dict:
    """Guarded stratified 80/20 split assigned ONLY after dedup.

    Pre-split guard (RESEARCH.md Pattern 3 / Pitfall 3): sklearn's stratified
    splitter fails cryptically in two verified ways — any class with < 2
    members and a rounded test fold smaller than the class count. A minimum
    class count of ``min_class_split`` (10) pre-empts both with a readable
    SystemExit naming the offending class, the per-class counts, and the
    collect-more contingency.

    The split itself is one sklearn ``train_test_split`` call over positional
    indices with ``stratify`` on canonical_class and ``random_state`` (42)
    — deterministic across runs. Rows' ``split`` column is filled in place
    ("train"/"test"); ``split0`` is metadata only and never consulted.

    Returns ``{class: {"train": n, "test": n}}`` for the audit's per-class
    table.
    """
    counts = Counter(
        row["canonical_class"] for row in rows if row.get("canonical_class")
    )
    if not counts:
        raise SystemExit(
            "cannot stratify 80/20: no post-dedup class rows — nothing to split "
            "(collect-more contingency: every canonical class came out empty)"
        )
    min_count = min(counts.values())
    if min_count < min_class_split:
        offending = sorted(c for c, n in counts.items() if n == min_count)[0]
        raise SystemExit(
            f"cannot stratify 80/20: class '{offending}' has {min_count} usable "
            f"samples < min_class_split {min_class_split}; per-class counts: "
            f"{dict(sorted(counts.items()))} — collect more images for the "
            "class(es) below the guard and re-run (collect-more contingency)"
        )
    y = [row["canonical_class"] for row in rows]
    positions = np.arange(len(rows))
    train_pos, test_pos = train_test_split(
        positions, test_size=test_size, stratify=y, random_state=random_state
    )
    split_counts = {cls: {"train": 0, "test": 0} for cls in sorted(counts)}
    for i in train_pos:
        row = rows[int(i)]
        row["split"] = "train"
        split_counts[row["canonical_class"]]["train"] += 1
    for i in test_pos:
        row = rows[int(i)]
        row["split"] = "test"
        split_counts[row["canonical_class"]]["test"] += 1
    return split_counts


def leak_check(rows, threshold: int) -> int:
    """Cross-split near-duplicate count (phase success criterion).

    Packs both splits' phash hex into uint64 and brute-forces the train x
    test Hamming matrix (XOR + popcount — RESEARCH.md lines 461-467). Returns
    the number of cross-split pairs at Hamming distance <= threshold; the
    caller asserts zero and records the result in the audit. Zero is
    provable by construction (every pair within the threshold was merged by
    dedup BEFORE the split existed) — this check is the assertion.

    Raises ValueError when any row lacks a phash (the leak check would
    otherwise silently under-count).
    """
    train: list[str] = []
    test: list[str] = []
    for row in rows:
        h = row.get("phash_hex") or ""
        if not h:
            raise ValueError(
                f"row {row.get('row_id', '?')} has no phash_hex — the leak "
                "check would silently under-count; hash before splitting"
            )
        if row.get("split") == "train":
            train.append(h)
        elif row.get("split") == "test":
            test.append(h)
    if not train or not test:
        return 0
    h_train, h_test = pack64(train), pack64(test)
    distances = popcount(np.bitwise_xor.outer(h_train, h_test))
    return int((distances <= threshold).sum())


def write_manifest(rows, out_path) -> None:
    """Write manifest rows with the frozen column contract.

    mkdir(parents=True, exist_ok=True) for data/processed; the CSV is utf-8
    with the csv module's standard dialect, newline="" per the csv docs.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
