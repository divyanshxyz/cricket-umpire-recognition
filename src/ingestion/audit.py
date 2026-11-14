"""Console audit summary + the full audit report writer — no silent exclusion.

Every excluded/conditional/no_box record is counted explicitly (RESEARCH.md
anti-pattern: exclusions must never be dropped silently). Per-source
[ok]/[warn] lines keep the prior-script console convention.
:func:`build_console_summary` builds the stage-level console text;
:func:`build_audit_report` builds the full report that the CLI surfaces
twice (INGEST-07) — printed to the console AND written to
reports/dedup_audit.md — from the SAME string, so the two can never diverge.
No wall-clock timestamps anywhere (determinism, Pitfall 9): two consecutive
full runs produce byte-identical output.
"""

from __future__ import annotations

import importlib.metadata
from collections import Counter

from .adapters import bbox_row_class_counts, decode_events, invalid_box_events
from .dedup import roboflow_stem

# Static code version tag for the determinism parameter block (Pitfall 9:
# the audit records thresholds + tool versions + a code version — never a
# wall-clock timestamp, so re-runs stay byte-identical). Bump when the
# pipeline's behavior-affecting code changes.
CODE_VERSION = "phase1-plan03"

# Expected per-source census (RESEARCH.md Verified Dataset Census, read from
# disk 2026-11-14) — the sanity column the report renders against the actual
# on-disk counts so re-runs catch data drift immediately.
EXPECTED_CENSUS = {
    "roboflow_umpire_wide": 2497,
    "umpire_gestures": 145,
    "archive": 390,
    "archive3": 390,
    "archive1": 312,
    "inception": 843,
}

# Roboflow exports' unique source-frame stems (.rf.<32hex> stripped) —
# 2,497 roboflow_umpire_wide files come from only 558 stems (1,939 baked-in
# augmented copies, RESEARCH.md Pitfall 1).
EXPECTED_STEMS = {
    "roboflow_umpire_wide": 558,
}

# The export whose umpire-class bbox-row count the crop-rule section logs
# (IN-01 interpolates its parsed-file count from the run's records instead
# of hardcoding "2,497") — the same source the stems-sanity row keys.
CROP_RULE_EXPORT_SOURCE = "roboflow_umpire_wide"

# Single source for the pipeline's parameter defaults (IN-02: previously
# re-declared independently in the CLI coordinator and this module's report
# fallbacks, and the report's parameter block could render a blank for a
# value the pipeline actually defaulted). The CLI merges paths.yaml's
# `params` section over this dict and passes the EFFECTIVE params into
# :func:`build_audit_report`, so the T-01-06 determinism trace always shows
# the values the run actually used. phash_threshold 8 and random_state 42
# are locked CONTEXT decisions — change only via CONTEXT.
DEFAULT_PARAMS = {
    "phash_threshold": 8,   # near-duplicate Hamming distance <= threshold (inclusive)
    "pad_frac": 0.15,       # bbox crop context padding, clamped to image bounds
    "test_size": 0.2,       # stratified 80/20 split
    "random_state": 42,     # split seed
    "min_class_flag": 30,   # audit collect-more flag boundary (strict > rule)
    "min_class_split": 10,  # stratified split guard
}

# Post-dedup kept-count sanity band (RESEARCH.md dedup expectations,
# lines 124-127) — named once (IN-02) instead of two inline 1250/1450 pairs.
SANITY_BAND = (1250, 1450)


def no_box_only_stems(roboflow_records) -> int:
    """Count stripped stems whose Roboflow files are ALL no_box (WR-03).

    Those stems' copies never enter the keep/conditional dedup pool, so the
    plain ``pooled + no_box files = file-level`` equation over-attributes
    ``(members - 1)`` file-level removals to each such stem. The correct
    identity is ``file_level = pooled + no_box_files - no_box_only_stems``;
    the reconciliation sites below verify it with this term BEFORE printing
    any equation — the section's own "nothing is double-counted silently"
    claim must not rely on an unverified arithmetic assumption.
    """
    no_box_keys = {
        (r.source, roboflow_stem(r.relpath))
        for r in roboflow_records if r.outcome == "no_box"
    }
    pooled_keys = {
        (r.source, roboflow_stem(r.relpath))
        for r in roboflow_records if r.outcome != "no_box"
    }
    return len(no_box_keys - pooled_keys)


def build_console_summary(source_rows, records, survivors, removed,
                          stem_removed=None, crop_stats=None,
                          conditional_decision=None) -> str:
    """Build the console audit text (returned as one string; the CLI prints it).

    Sections: per-source [ok]/[warn] lines (parsed count vs on-disk census +
    raw class names), duplicates removed (Layer A stem groups and Layer B
    phash as SEPARATE lines — RESEARCH.md Pitfall 1), exclusions counted by
    raw label, per-class post-dedup counts, decode-replacement events, and
    the tool version (full parameter block is written to the report file).
    ``stem_removed``/``crop_stats`` are optional kwargs added by Plan 02;
    ``conditional_decision`` (Plan 03's strict >30 evaluation result) renders
    the decision per conditional class when provided. The frozen
    4-positional-arg contract is unchanged.
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
    roboflow_parsed = [r for r in records if r.family == "roboflow-yolo"]
    if roboflow_parsed:
        roboflow_stems = len({
            (r.source, roboflow_stem(r.relpath)) for r in roboflow_parsed
        })
        file_level = len(roboflow_parsed) - roboflow_stems
        if file_level != len(stem_removed):
            # WR-03: verify the reconciliation identity before printing it —
            # file_level = pooled + no_box_files - no_box_only_stems. Printing
            # an assumed equation would state a false arithmetic fact the
            # moment two no_box files share a stripped stem.
            no_box_count = sum(1 for r in roboflow_parsed if r.outcome == "no_box")
            no_box_only = no_box_only_stems(roboflow_parsed)
            delta = file_level - len(stem_removed)
            if delta == no_box_count and no_box_only == 0:
                lines.append(
                    f"  (file-level stem-copy removals: {file_level} = {len(stem_removed)} "
                    f"pooled + {no_box_count} no_box copies excluded "
                    "before dedup — reconciled in reports/dedup_audit.md)"
                )
            elif delta == no_box_count - no_box_only:
                lines.append(
                    f"  (file-level stem-copy removals: {file_level} = {len(stem_removed)} "
                    f"pooled + {no_box_count} no_box copies - {no_box_only} all-no_box "
                    "stems — reconciled in reports/dedup_audit.md)"
                )
            else:
                lines.append(
                    f"  (file-level stem-copy removals {file_level} vs pooled "
                    f"{len(stem_removed)} + no_box {no_box_count}: UNRECONCILED "
                    f"(delta {delta - no_box_count}) — see reports/dedup_audit.md)"
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
    if conditional_decision is None:
        lines.append(f"  conditional (LegBye/DeadBall >30 rule): {len(conditional)}")
    else:
        for cls in sorted(conditional_decision):
            d = conditional_decision[cls]
            verdict = "included" if d["included"] else "excluded"
            lines.append(
                f"  conditional {cls}: {d['count']} usable -> {verdict} ({d['reason']})"
            )
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
        "full parameter block written to reports/dedup_audit.md"
    )
    return "\n".join(lines)


def build_audit_report(source_rows, records, survivors, removed, stem_removed,
                       rows, params, class_cfg, conditional_decision,
                       split_counts, leaks, crop_stats=None) -> str:
    """Build the full audit report (reports/dedup_audit.md content).

    INGEST-07 presentation resolution: the CLI surfaces this report TWICE —
    printed to the console on every run AND written to reports/dedup_audit.md
    — from this one string, so console and file can never diverge.

    Section contract (01-03-PLAN.md Task 1 + must_haves): parameter block
    (no wall-clock timestamps — byte-identical re-runs are the Phase 3
    frozen-test precondition, threat T-01-06); per-source census table with
    the RESEARCH.md expected-count sanity column and the audit-only
    precomputed-csv row; totals with the 1250-1450 sanity band; stem-group
    and phash removals as SEPARATE lines (Pitfall 1); every exclusion
    counted (DRS/umpire/non_umpire/no_box/multibox/invalid boxes/unknown);
    conditional-class decision lines from evaluate_conditional (INGEST-06
    strict >30); the explicit crop-rule log for the single umpire-class bbox
    row + the research-driven crop-rule adjustment note; conflicts with top
    examples; the per-class post-dedup train/test table with <30
    collect-more flags (the Four contingency surfaces automatically); and
    the cross-split leak line (must read 0).
    """
    threshold = int(params.get("phash_threshold", DEFAULT_PARAMS["phash_threshold"]))
    min_class_flag = int(params.get("min_class_flag", DEFAULT_PARAMS["min_class_flag"]))
    lines: list[str] = []
    lines.append("# Dataset ingestion + leak-free dedup audit (Phase 1)")
    lines.append("")
    lines.append(
        "Single source of truth for Phases 2-5: census, dedup, exclusions, the "
        "guarded post-dedup split, and the cross-split leak check. Deterministic: "
        "re-running src/data_ingestion.py reproduces this report and "
        "data/processed/manifest.csv byte-identically (no wall-clock timestamps)."
    )

    # Parameter block (Pitfall 9; threat T-01-06 determinism mitigation).
    lines.append("")
    lines.append("## Parameter block")
    lines.append("")
    try:
        imagehash_version = importlib.metadata.version("imagehash")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - venv always has it
        imagehash_version = "unknown"
    for key in ("phash_threshold", "pad_frac", "test_size", "random_state",
                "min_class_flag", "min_class_split"):
        lines.append(f"{key}: {params.get(key, '')}")
    lines.append(f"imagehash version: {imagehash_version}")
    # IN-08: Pillow and scipy are determinism-relevant (phash depends on
    # Pillow's resize and scipy's DCT behavior) — record their installed
    # versions alongside imagehash's so the T-01-06 trace identifies every
    # dependency that can move the hashes.
    for helper in ("Pillow", "scipy"):
        try:
            helper_version = importlib.metadata.version(helper)
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover - venv always has them
            helper_version = "unknown"
        lines.append(f"{helper} version: {helper_version}")
    lines.append(f"code version: {CODE_VERSION}")

    # Per-source census table (INGEST-01) with the expected-count sanity column.
    lines.append("")
    lines.append("## Per-source census")
    lines.append("")
    lines.append(
        "| source | root | family | raw count | expected (census) | sanity | raw classes |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for row in source_rows:
        name = row["name"]
        if row["family"] == "precomputed-csv":
            lines.append(
                f"| {name} | {row['root']} | {row['family']} | {row['image_count']} rows "
                "| - | - | audit-only, excluded from the image pipeline |"
            )
            continue
        vocab = (
            sorted({str(r.raw_label) for r in records if r.source == name and r.raw_label})
            or row["raw_classes"]
        )
        classes = ", ".join(vocab) if vocab else "-"
        expected = EXPECTED_CENSUS.get(name)
        if expected is None:
            expected_cell, sanity = "-", "-"
        elif row["image_count"] == expected:
            expected_cell, sanity = expected, "ok"
        else:
            expected_cell = expected
            sanity = f"MISMATCH (expected {expected})"
        lines.append(
            f"| {name} | {row['root']} | {row['family']} | {row['image_count']} "
            f"| {expected_cell} | {sanity} | {classes} |"
        )
    # Roboflow unique-stem sanity: 2,497 files from 558 source frames.
    for name in sorted(EXPECTED_STEMS):
        stems = {
            roboflow_stem(r.relpath)
            for r in records
            if r.source == name and r.family == "roboflow-yolo"
        }
        expected = EXPECTED_STEMS[name]
        verdict = "ok" if len(stems) == expected else f"MISMATCH (expected {expected})"
        lines.append(f"{name} unique stems: {len(stems)} (expected {expected}) [{verdict}]")

    # Totals with the sanity band (RESEARCH.md dedup expectations, lines 124-127).
    raw_total = sum(
        row["image_count"] for row in source_rows
        if row["family"] != "precomputed-csv"
    )
    kept = len(survivors)
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append(f"Raw image files: {raw_total}")
    lines.append(f"Kept after dedup: {kept} (sanity band {SANITY_BAND[0]}-{SANITY_BAND[1]})")
    lines.append(
        "Sanity band verdict: "
        + ("in band" if SANITY_BAND[0] <= kept <= SANITY_BAND[1]
           else "OUT OF BAND - investigate dedup before Phase 2")
    )
    lines.append(f"Manifest class rows (post conditional evaluation): {len(rows)}")

    # Duplicates removed — two layers, separate lines (Pitfall 1: stem-dup
    # and phash-dup removals must never be merged into one number). The
    # stem-groups line reports FILE-level copy removals (every parsed
    # Roboflow file beyond its stem group's surviving keeper) so it
    # reconciles with the RESEARCH.md census (2497 files / 558 stems =
    # 1939): the dedup pool is keep/conditional only (Plan 01 locked
    # decision), so the zero-box records exit at the canonical stage before
    # Layer A runs. The reconciliation is rendered explicitly — nothing is
    # double-counted silently.
    lines.append("")
    lines.append("## Duplicates removed")
    lines.append("")
    stem_removed = stem_removed or []
    roboflow_records = [r for r in records if r.family == "roboflow-yolo"]
    roboflow_stem_keys = {
        (r.source, roboflow_stem(r.relpath)) for r in roboflow_records
    }
    file_level_stem = len(roboflow_records) - len(roboflow_stem_keys)
    no_box_roboflow = sum(
        1 for r in roboflow_records if r.outcome == "no_box"
    )
    lines.append(f"Duplicates removed - Roboflow stem-groups: {file_level_stem}")
    for src in sorted({k[0] for k in roboflow_stem_keys}):
        src_files = sum(1 for r in roboflow_records if r.source == src)
        src_stems = sum(1 for k in roboflow_stem_keys if k[0] == src)
        lines.append(
            f"  {src}: {src_files - src_stems} (file-level: {src_files} files "
            f"from {src_stems} stems)"
        )
    pooled_stem = len(stem_removed)
    if file_level_stem != pooled_stem:
        # WR-03: verify before printing — the equation is only rendered when
        # the identity it states actually holds on this run's data. Under
        # data drift (two no_box files sharing a stripped stem) the plain
        # equation would be arithmetically false, so the corrected form or
        # an explicit UNRECONCILED verdict is printed instead.
        no_box_only = no_box_only_stems(roboflow_records)
        delta = file_level_stem - pooled_stem
        if delta == no_box_roboflow and no_box_only == 0:
            lines.append(
                f"  reconciliation: {pooled_stem} pooled copies removed by Layer A "
                f"+ {no_box_roboflow} zero-box copies excluded as no_box before "
                f"dedup (counted above) = {file_level_stem} file-level"
            )
        elif delta == no_box_roboflow - no_box_only:
            lines.append(
                f"  reconciliation: {pooled_stem} pooled copies removed by Layer A "
                f"+ {no_box_roboflow} zero-box copies - {no_box_only} all-no_box "
                f"stems excluded before dedup (counted above) = {file_level_stem} file-level"
            )
        else:
            lines.append(
                f"  UNRECONCILED: file-level stem-copy removals {file_level_stem} "
                f"vs pooled {pooled_stem} + no_box {no_box_roboflow} "
                f"(delta {delta - no_box_roboflow}) — investigate before Phase 2"
            )
    lines.append(f"Duplicates removed - phash near-duplicates: {len(removed)}")
    removed_by_source = Counter(r.source for r in removed)
    for src in sorted(removed_by_source):
        lines.append(f"  {src}: {removed_by_source[src]}")
    lines.append(
        "  (Layer B is union-find over EVERY pair at Hamming <= threshold: "
        "near-identical sequences chain transitively — the export's video frames "
        "include distinct stems with identical 64-bit hashes, so long runs merge "
        "into single clusters. This is the locked CONTEXT policy (threshold 8) "
        "operating on real data; the sanity-band verdict above reports the result.)"
    )

    # Exclusions — every exclusion counted, never silent (RESEARCH.md
    # anti-pattern; INGEST-06's exclusion behavior is itself an audit deliverable).
    excluded = [r for r in records if r.outcome == "excluded"]
    no_box = [r for r in records if r.outcome == "no_box"]
    multibox = [r for r in records if r.multibox]
    unknown = [r for r in records if r.outcome == "unknown"]
    lines.append("")
    lines.append("## Exclusions (counted, never written as class rows)")
    lines.append("")
    if excluded:
        by_label = Counter(str(r.raw_label) for r in excluded)
        for label in sorted(by_label):
            lines.append(f"{label}: {by_label[label]} excluded")
    else:
        lines.append("no excluded-as-class records")
    lines.append(f"Zero-box images excluded as no_box: {len(no_box)}")
    lines.append(f"Multi-box images flagged (largest-area box wins): {len(multibox)}")
    lines.append(
        f"Invalid bbox rows rejected (out-of-range/malformed, ASVS V5): "
        f"{len(invalid_box_events())}"
    )
    lines.append(f"Label decode replacements (utf-8, errors=replace): {decode_events()}")
    lines.append(f"unknown: {len(unknown)} (unknown labels abort the run — this stays 0)")

    # Conditional classes — decision lines from evaluate_conditional
    # (INGEST-06 strict >30 boundary; zero-usable classes are never omitted).
    lines.append("")
    lines.append(
        f"Conditional classes (INGEST-06: included only when the usable count "
        f"is STRICTLY greater than {min_class_flag}):"
    )
    for cls in sorted(conditional_decision or {}):
        d = conditional_decision[cls]
        verdict = "included" if d["included"] else "excluded"
        lines.append(f"{cls}: {d['count']} {verdict} (conditional class, {d['reason']})")

    # Crop rule — explicit log of the umpire-class bbox rows plus the
    # research-driven adjustment note (CONTEXT 'largest umpire box' was
    # unimplementable: the umpire annotation class carries a single bbox
    # row). IN-01: both the count and the export size are interpolated, not
    # hardcoded — a hardcoded "1 bbox row across the 2,497-image export"
    # would self-contradict the "seen: N" line above it under data drift.
    # The export is roboflow_umpire_wide (the source EXPECTED_STEMS keys);
    # its parsed-file count is computed from this run's records, not frozen.
    distribution = bbox_row_class_counts()
    umpire_rows = distribution.get("umpire", 0)
    export_images = sum(
        1 for r in records
        if r.family == "roboflow-yolo" and r.source == CROP_RULE_EXPORT_SOURCE
    )
    row_word = "row" if umpire_rows == 1 else "rows"
    lines.append("")
    lines.append("## Crop rule (research-driven adjustment, logged explicitly)")
    lines.append("")
    if distribution:
        dist = ", ".join(f"{cls} {n}" for cls, n in distribution.items())
        lines.append(f"bbox rows seen by class: {dist} (total {sum(distribution.values())})")
    lines.append(
        f"umpire-class bbox rows seen: {umpire_rows} — logged explicitly; the "
        f"umpire annotation class has {umpire_rows} bbox {row_word} across the "
        f"{export_images:,}-image export "
        "because boxes carry the signal class (the umpire performing it)"
    )
    lines.append(
        "Adjustment note: the CONTEXT decision 'when multiple umpire boxes exist, "
        "crop the largest-area umpire box' was unimplementable as written — the "
        f"umpire class has {umpire_rows} bbox {row_word} across "
        f"{export_images:,} images. Operative rule: crop the "
        "largest-area annotated box of ANY class (every box denotes the umpire); "
        "the row's canonical class = that box's class. Intent preserved: every "
        "Roboflow image still yields the umpire crop with 15% clamped padding."
    )

    # Label conflicts (Pitfall 7 — frequent, not rare).
    conflict_rows = [r for r in rows if r.get("conflict")]
    lines.append("")
    if conflict_rows:
        details = Counter(str(r.get("conflict_detail") or "") for r in conflict_rows)
        lines.append(
            f"Label conflicts resolved (tier priority, majority at equal tier, "
            f"lexicographic tiebreak): {len(conflict_rows)}"
        )
        lines.append("  top conflict examples (discarded labels x count):")
        for detail, n in sorted(details.items(), key=lambda kv: (-kv[1], kv[0]))[:5]:
            lines.append(f"    {detail} x{n}")
    else:
        lines.append("Label conflicts resolved: 0")

    # Per-class post-dedup table with the assigned split + <30 collect-more
    # flags (the Four contingency from STATE.md surfaces automatically).
    lines.append("")
    lines.append("## Per-class post-dedup counts (assigned split)")
    lines.append("")
    lines.append("  Class         Train    Test   Total")
    for cls in sorted(split_counts or {}):
        sides = split_counts[cls]
        total = sides["train"] + sides["test"]
        lines.append(
            f"  {cls:<12} {sides['train']:>6} {sides['test']:>7} {total:>7}"
        )
    for cls, sides in sorted((split_counts or {}).items()):
        total = sides["train"] + sides["test"]
        if total < min_class_flag:
            lines.append(
                f"FLAG {cls}: {total} usable samples < {min_class_flag} — collect "
                f"more '{cls}' images before Phase 2 (collect-more contingency)"
            )

    # Cross-split leak check — the phase's success criterion line.
    lines.append("")
    lines.append(f"Cross-split phash leaks (Hamming <= {threshold}): {leaks}")
    if leaks == 0:
        lines.append(
            "  zero near-duplicate pairs across the train/test boundary — "
            "leak-free split (dedup merged every pair within the threshold "
            "BEFORE the split existed)"
        )

    # Crop stage outcome.
    if crop_stats is not None:
        lines.append("")
        if crop_stats.get("dry_run"):
            lines.append(
                "Crops (dry-run — nothing written): Roboflow rows would be "
                f"cropped, {crop_stats.get('non_roboflow', 0)} non-Roboflow rows "
                "stay full-image"
            )
        else:
            lines.append(
                f"Crops written (crop_<row_id>.jpg): {crop_stats.get('written', 0)}, "
                f"non-Roboflow full-image rows {crop_stats.get('non_roboflow', 0)}, "
                f"invalid/degenerate {crop_stats.get('invalid', 0)}, "
                f"failed {crop_stats.get('failed', 0)}"
            )

    lines.append("")
    lines.append(
        "Determinism contract: sorted walks, fixed random_state, static code "
        "version tag, no wall-clock timestamps — two consecutive full runs "
        "produce byte-identical manifest.csv and dedup_audit.md."
    )
    return "\n".join(lines)
