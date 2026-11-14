"""Deterministic manifest row assembly + CSV writer (data/processed/manifest.csv).

One row per kept image (RESEARCH.md architecture box 7). The split column is
written EMPTY — Plan 03 assigns the stratified split AFTER dedup; no source's
own split (split0) ever determines it. All paths are repo-root-relative posix
strings (RESEARCH.md Pitfall 8).
"""

from __future__ import annotations

import csv
from pathlib import Path

from .canonical import UnknownLabelError, map_label

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
            "crop_path": "",  # Plan 02 derives crop_{row_id}.jpg for Roboflow rows
            "phash_hex": rec.phash_hex or "",
            "split": "",  # written EMPTY — Plan 03 assigns the stratified split AFTER dedup
            "conflict": False,  # cluster-conflict flags arrive with Plan 02
            "conflict_detail": "",
        })
    return rows


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
