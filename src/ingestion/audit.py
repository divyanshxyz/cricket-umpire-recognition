"""Console audit summary — no silent exclusion.

Every excluded/conditional/no_box record is counted explicitly (RESEARCH.md
anti-pattern: exclusions must never be dropped silently). Per-source
[ok]/[warn] lines keep the prior-script console convention. The file surface
(reports/dedup_audit.md) lands in Plan 03; this module builds one string that
the CLI prints in a single print call. No wall-clock timestamps anywhere
(determinism, Pitfall 9).
"""

from __future__ import annotations

import importlib.metadata
from collections import Counter

from .adapters import bbox_row_class_counts, decode_events, invalid_box_events


def build_console_summary(source_rows, records, survivors, removed,
                          stem_removed=None, crop_stats=None) -> str:
    """Build the console audit text (returned as one string; the CLI prints it).

    Sections: per-source [ok]/[warn] lines (parsed count vs on-disk census +
    raw class names), duplicates removed (Layer A stem groups and Layer B
    phash as SEPARATE lines — RESEARCH.md Pitfall 1), exclusions counted by
    raw label, per-class post-dedup counts, decode-replacement events, and
    the tool version (full parameter block lands in Plan 03's report file).
    ``stem_removed``/``crop_stats`` are optional kwargs added by Plan 02;
    the frozen 4-positional-arg contract is unchanged.
    """
    lines: list[str] = []
    lines.append("=== Phase 1 ingestion audit (walking skeleton) ===")

    # Per-source lines: what entered the pipeline vs the on-disk census.
    lines.append("")
    lines.append("Per-source (parsed / on-disk census):")
    for row in source_rows:
        name = row["name"]
        family = row["family"]
        census = row["image_count"]
        parsed = sum(1 for r in records if r.source == name)
        vocab = (
            sorted({r.raw_label for r in records if r.source == name and r.raw_label})
            or row["raw_classes"]
        )
        if family == "precomputed-csv":
            lines.append(
                f"[{row['status']}] {name:<24} {row['image_count']:>5} rows  "
                f"family={family}  ({row['note']})"
            )
            continue
        status = "ok" if parsed else "warn"
        detail = f"{parsed:>4} items"
        if parsed and census > parsed:
            detail += f" (of {census} on disk; capped)"
        elif not parsed and census:
            detail += f" (of {census} on disk; adapter stub — Plan 02 fills)"
        classes = ", ".join(vocab) if vocab else "-"
        lines.append(f"[{status}] {name:<24} {detail}  family={family}  classes: {classes}")

    # Duplicates removed — two layers, separate audit lines (Pitfall 1:
    # stem-dup and phash-dup removals must never be merged into one number).
    stem_removed = stem_removed or []
    removed_by_source = Counter(r.source for r in removed)
    stem_by_source = Counter(r.source for r in stem_removed)
    lines.append("")
    lines.append(
        f"Duplicates removed — Layer A Roboflow stem groups (.rf source-frame ID): {len(stem_removed)}"
    )
    for src in sorted(stem_by_source):
        lines.append(f"  {src}: {stem_by_source[src]}")
    lines.append(
        f"Duplicates removed — Layer B phash (Hamming <= threshold, inclusive): {len(removed)}"
    )
    for src in sorted(removed_by_source):
        lines.append(f"  {src}: {removed_by_source[src]}")

    # Exclusions — counted explicitly, never silent.
    excluded = [r for r in records if r.outcome == "excluded"]
    conditional = [r for r in records if r.outcome == "conditional"]
    no_box = [r for r in records if r.outcome == "no_box"]
    unknown = [r for r in records if r.outcome == "unknown"]
    lines.append("")
    lines.append("Exclusions (counted, never written as class rows):")
    if excluded:
        by_label = Counter(r.raw_label for r in excluded)
        detail = ", ".join(f"{label}: {n}" for label, n in sorted(by_label.items(), key=str))
        lines.append(f"  excluded as class: {len(excluded)} ({detail})")
    else:
        lines.append("  excluded as class: 0")
    lines.append(f"  conditional (LegBye/DeadBall >30 rule evaluated in Plan 03): {len(conditional)}")
    lines.append(f"  no_box: {len(no_box)}")
    multibox = [r for r in records if r.multibox]
    lines.append(
        f"  multibox images (largest-area annotated box of any class kept): {len(multibox)}"
    )
    row_counts = bbox_row_class_counts()
    if row_counts:
        detail = ", ".join(f"{cls}: {n}" for cls, n in row_counts.items())
        lines.append(
            "  bbox rows seen by class (boxes carry the signal class; the crop rule selects "
            f"the largest of any class): {detail}"
        )
    lines.append(
        f"  invalid bbox rows rejected (out-of-range/malformed, ASVS V5): {len(invalid_box_events())}"
    )
    lines.append(f"  unknown: {len(unknown)} (unknown labels abort the run — this stays 0)")

    # Per-class post-dedup counts.
    lines.append("")
    lines.append("Per-class post-dedup counts:")
    by_class = Counter(r.canonical for r in survivors if r.canonical)
    if by_class:
        for cls in sorted(by_class):
            lines.append(f"  {cls:<12} {by_class[cls]:>5}")
    else:
        lines.append("  (none — every parsed record was excluded)")

    # Crop stage (Plan 02): Roboflow survivors crop to crop_<row_id>.jpg.
    if crop_stats is not None:
        lines.append("")
        if crop_stats.get("dry_run"):
            lines.append(
                f"Crops (dry-run — nothing written): Roboflow rows would be cropped, "
                f"{crop_stats['non_roboflow']} non-Roboflow rows stay full-image"
            )
        else:
            lines.append(
                f"Crops (Roboflow survivors, crop_<row_id>.jpg): written {crop_stats['written']}, "
                f"non-Roboflow full-image rows {crop_stats['non_roboflow']}, "
                f"invalid/degenerate {crop_stats['invalid']}, failed {crop_stats['failed']}"
            )

    # Environment/parameter trace (Pitfall 9: parameters written into the audit).
    lines.append("")
    lines.append(f"Label decode replacements (utf-8, errors=replace): {decode_events()}")
    try:
        imagehash_version = importlib.metadata.version("imagehash")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - venv always has it
        imagehash_version = "unknown"
    lines.append(f"imagehash version: {imagehash_version}")
    lines.append(
        "Parameters: phash threshold + pad_frac + split params from configs/paths.yaml; "
        "full parameter block lands in reports/dedup_audit.md (Plan 03)"
    )
    return "\n".join(lines)
