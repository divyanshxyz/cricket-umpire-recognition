"""Source discovery: walk dataset roots, count images, collect raw vocabularies.

A missing or empty source root warns and records a zero count (never crashes
the run); an absent datasets/ root — or every image source empty — aborts
with an actionable error (prior-script main-loop style). Every walk is sorted
(Pattern 4) and no dict-order-dependent logic exists: iteration follows the
paths.yaml file order deterministically.

All entry paths (roots, audit-only CSVs, the datasets/ abort check) resolve
against the caller-supplied ``base`` — the same config-derived repo root the
parse/hash/write stages use — never against the current working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .adapters import IMAGE_EXTS, filename_raw_label


def _scan_images(root: Path) -> tuple[int, list[str]]:
    """Sorted walk -> (image_count, sorted raw class vocabulary).

    The raw vocabulary is the filename stems' label tokens and is collected
    only for filename-label sources; other families' vocabularies arrive from
    their own parsers (Plan 02 fills them).
    """
    count = 0
    vocab: set[str] = set()
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            count += 1
            vocab.add(filename_raw_label(p.stem))
    return count, sorted(vocab)


def _scan_csv(path: Path) -> int:
    """Data-row count for an audit-only CSV (row count optional, presence required)."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return max(0, sum(1 for _ in f) - 1)  # minus header


def discover_sources(paths_cfg: dict, base: Path) -> list[dict]:
    """Per-source summary rows for the audit.

    ``base`` is the config-derived repo root (configs/paths.yaml's
    grandparent, as computed by the CLI coordinator) — every entry path is
    resolved as ``base / entry[...]`` so discovery reads the same tree the
    parse/hash/write stages read, regardless of the current working
    directory.

    Returns a list of dicts with keys: name, root, family, priority,
    image_count, raw_classes, status ('ok'|'warn'), note. Audit-only entries
    are emitted with family precomputed-csv — listed in the audit, excluded
    from the image pipeline.

    Warn-and-continue on a missing/empty root (zero count); abort via
    sys.exit when the datasets/ root is absent or every image source is empty.
    """
    rows: list[dict] = []
    roots = paths_cfg.get("dataset_roots") or {}

    expects_datasets_root = any(
        str(entry.get("root", "")).replace("\\", "/").startswith("datasets")
        for entry in roots.values()
    )
    if expects_datasets_root and not (base / "datasets").is_dir():
        sys.exit(
            "[discover] datasets/ root not found. The pipeline reads raw data from "
            "the sibling datasets/ folder (read-only) — check the 'root' paths in "
            "configs/paths.yaml."
        )

    for name, entry in roots.items():
        root = base / entry["root"]
        family = entry["family"]
        if not root.is_dir():
            rows.append({
                "name": name, "root": entry["root"], "family": family,
                "priority": entry["priority"], "image_count": 0,
                "raw_classes": [], "status": "warn", "note": "root not found",
            })
            continue
        if family == "filename-label":
            count, vocab = _scan_images(root)
        else:
            # Vocabulary for roboflow-yolo / folder-label comes from their own
            # parsers (Plan 02 fills them); discovery counts images only.
            count, vocab = _count_images(root), []
        rows.append({
            "name": name, "root": entry["root"], "family": family,
            "priority": entry["priority"], "image_count": count,
            "raw_classes": vocab,
            "status": "ok" if count else "warn",
            "note": "" if count else "no images found",
        })

    for name, entry in (paths_cfg.get("audit_only") or {}).items():
        path = base / entry["path"]
        present = path.is_file()
        rows.append({
            "name": name, "root": entry["path"],
            "family": entry.get("family", "precomputed-csv"),
            "priority": None,
            "image_count": _scan_csv(path) if present else 0,
            "raw_classes": [],
            "status": "ok" if present else "warn",
            "note": "audit-only, excluded from the image pipeline"
            if present else "audit-only file not found",
        })

    image_rows = [r for r in rows if r["name"] in roots]
    if image_rows and all(r["image_count"] == 0 for r in image_rows):
        sys.exit('No items found. Check the "root" paths in configs/paths.yaml.')
    return rows


def _count_images(root: Path) -> int:
    """Image count only (sorted walk; vocabulary left to the family parser)."""
    count = 0
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            count += 1
    return count
