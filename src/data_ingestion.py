#!/usr/bin/env python
"""Phase 1 ingestion CLI coordinator.

Stage order per the RESEARCH.md architecture diagram, with the crop stage
moved AFTER the leak check (WR-04 — the split guard and leak check can still
abort the run by design, and writing crops before them would strand stale
crop files on every abort):
discover -> parse -> canonical -> dedup (two layers) -> conditional >30
evaluation -> split (guarded, AFTER dedup) -> leak check -> crop survivors ->
write artifacts (manifest.csv + reports/dedup_audit.md) -> audit (console
summary + the full report printed AND written — INGEST-07). --dry-run runs
everything in memory and writes nothing (the crop stage counts would-be
crops; the report is printed but not written).

Flags:
  --config   paths.yaml location (default configs/paths.yaml; classes.yaml is
             loaded from the same directory)
  --dry-run  full pipeline in memory + console print; writes NOTHING
  --limit N  cap files parsed per source after sorting — development/sample
             runs only; the full run takes no flag
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import yaml
from tqdm import tqdm

from ingestion.adapters import (
    parse_filename_label,
    parse_folder_label,
    parse_roboflow,
    stamp_source,
)
from ingestion.audit import (
    DEFAULT_PARAMS,
    build_audit_report,
    build_console_summary,
)
from ingestion.canonical import UnknownLabelError, load_class_map, map_label
from ingestion.crops import crop_row
from ingestion.dedup import cluster_by_phash, phash_file, stem_groups
from ingestion.discover import discover_sources
from ingestion.manifest import (
    assemble_rows,
    assign_split,
    evaluate_conditional,
    leak_check,
    write_manifest,
)

# Family -> parser dispatch. roboflow-yolo and folder-label are seam stubs
# returning [] until Plan 02 fills them behind the identical contract.
PARSERS = {
    "roboflow-yolo": parse_roboflow,
    "folder-label": parse_folder_label,
    "filename-label": parse_filename_label,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="data_ingestion",
        description=(
            "Phase 1 dataset ingestion: discover sources, parse to uniform records, "
            "map to canonical classes, phash-dedup, write the manifest, print the audit."
        ),
    )
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="full pipeline in memory + console print; writes NOTHING",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap files parsed per source after sorting (development/sample runs only)",
    )
    args = ap.parse_args(argv)

    config_path = Path(args.config)
    base = config_path.resolve().parent.parent  # configs/paths.yaml -> repo root
    paths_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    class_cfg = load_class_map(config_path.parent / "classes.yaml")

    # 1. DISCOVER — per-source census rows (warn-and-continue; hard aborts
    #    inside). Resolves every entry path against `base` like every other
    #    stage, so discovery works from any working directory.
    source_rows = discover_sources(paths_cfg, base)

    # 2. PARSE — per family adapter, sorted walk, --limit cap after sorting.
    records = []
    for row in source_rows:
        parser = PARSERS.get(row["family"])
        if parser is None:
            continue  # audit-only (precomputed-csv) — never in the image pipeline
        recs = parser(base / row["root"], row["family"], row["priority"])
        if args.limit is not None:
            recs = recs[: args.limit]
        stamp_source(recs, row["name"])
        records.extend(recs)

    # 3. CANONICAL MAP — unknown label is a hard abort naming label + source.
    for rec in records:
        if rec.raw_label is None and rec.outcome != "no_box":
            sys.exit(
                f"[canonical] record with no raw label: {rec.source}/{rec.relpath} "
                f"(family {rec.family}) — adapter must set raw_label or outcome no_box"
            )
    try:
        for rec in records:
            if rec.outcome == "no_box":
                continue  # pre-set by the adapter (zero valid bbox rows); audit counts it
            rec.outcome, rec.canonical, rec.exclude_reason = map_label(rec.raw_label, class_cfg)
    except UnknownLabelError as err:
        sys.exit(f"[canonical] {err}")

    # 4. DEDUP — two layers, content-keyed only (CONTEXT locked decision).
    #    Layer A: Roboflow stem groups first (strips the export's own .rf
    #    source-frame ID — removes the augmented copies phash cannot catch,
    #    RESEARCH.md Pitfall 1). Layer B: phash union-find across the union
    #    of ALL remaining records (threshold inclusive at 8). Effective
    #    params = paths.yaml merged over DEFAULT_PARAMS (IN-02 single
    #    source); they flow to every stage AND to the audit's parameter
    #    block, so the determinism trace always records the values this run
    #    actually used instead of rendering blanks for defaulted keys.
    params = {**DEFAULT_PARAMS, **(paths_cfg.get("params") or {})}
    threshold = int(params["phash_threshold"])
    roots = {
        name: entry["root"]
        for name, entry in (paths_cfg.get("dataset_roots") or {}).items()
    }
    pool = [r for r in records if r.outcome in ("keep", "conditional")]
    stem_survivors, stem_removed = stem_groups(pool)
    for rec in tqdm(stem_survivors, desc="phash", unit="img", disable=None):
        img_path = base / roots[rec.source] / rec.relpath
        try:
            rec.phash_hex = phash_file(img_path)
        except Exception as err:  # surface the failing path — never a silent skip
            sys.exit(f"[phash] failed on {rec.source}/{rec.relpath}: {err}")
    survivors, removed = cluster_by_phash(stem_survivors, threshold=threshold)

    # 5. MANIFEST ROWS — rows from post-dedup survivors (row IDs are the
    #    stable crop-name source, consumed by the crop stage in 7b below).
    #    Only Roboflow rows carry a bbox; every other family stays a
    #    full-image row with blank crop fields, no copy made.
    rows = assemble_rows(survivors, class_cfg, roots=roots)
    ordered = sorted(survivors, key=lambda r: (r.source, r.relpath))

    # 5b. CONDITIONAL EVALUATION — INGEST-06 strict >30 rule on the usable
    #     (post-dedup) counts: a conditional class joins the manifest only
    #     when its count is STRICTLY greater than min_class_flag (exactly 30
    #     is excluded); zero-usable classes are excluded with their reason
    #     recorded in the audit, never silently omitted. Failed classes'
    #     rows leave the manifest here (both aligned lists filtered by the
    #     same indices) and the decision renders in the audit.
    min_class_flag = int(params["min_class_flag"])
    conditional_names = sorted((class_cfg.get("conditional_classes") or {}))
    cond_counter = Counter(row["canonical_class"] for row in rows)
    conditional_decision = evaluate_conditional(
        {name: cond_counter.get(name, 0) for name in conditional_names},
        min_class_flag,
    )
    dropped_conditional = {
        name for name, d in conditional_decision.items() if not d["included"]
    }
    if dropped_conditional:
        keep_idx = [
            i for i, row in enumerate(rows)
            if row["canonical_class"] not in dropped_conditional
        ]
        rows = [rows[i] for i in keep_idx]
        ordered = [ordered[i] for i in keep_idx]

    # 6. SPLIT — guarded stratified 80/20 assigned ONLY after dedup and the
    #    conditional evaluation (RESEARCH.md diagram box 6). The min-class
    #    guard pre-empts sklearn's two cryptic stratify failure modes with a
    #    readable per-class message + the collect-more contingency; no
    #    source's own split0 is ever consulted.
    test_size = float(params["test_size"])
    random_state = int(params["random_state"])
    min_class_split = int(params["min_class_split"])
    split_counts = assign_split(rows, test_size, random_state, min_class_split)

    # 7. LEAK CHECK — cross-split phash pairs must be zero (the phase's
    #    success criterion, provable by construction: dedup merged every
    #    pair within the threshold BEFORE the split existed). The check
    #    itself fails closed (WR-02): unassigned splits, a missing phash, or
    #    an empty side raise ValueError instead of a green 0.
    try:
        leaks = leak_check(rows, threshold)
    except ValueError as err:
        raise SystemExit(f"[leak-check] {err}")
    if leaks != 0:
        raise SystemExit(
            f"[leak-check] cross-split phash pairs at Hamming <= {threshold}: "
            f"{leaks} — dedup invariant broken; refusing to write the manifest"
        )

    # 7b. CROPS — Roboflow survivors crop to crop_<row_id>.jpg, written only
    #     AFTER the abortable split guard (step 6, SystemExit on the
    #     collect-more contingency) and the leak check (step 7) have passed
    #     (WR-04). Safe to reorder: a crop depends only on its row_id and the
    #     survivor's bbox — never on the split. Every non-dry-run starts the
    #     stage by CLEARING crops_dir, so the directory exactly mirrors this
    #     run's manifest: row_ids are positional ({source}_{i:05d}), so after
    #     data drift a run would otherwise overwrite some crops and strand
    #     the rest as ghost images for directory-scanning consumers.
    crops_rel = str(paths_cfg["outputs"]["crops_dir"]).replace("\\", "/").rstrip("/")
    pad_frac = float(params["pad_frac"])
    crop_stats = {
        "written": 0,
        "non_roboflow": 0,
        "invalid": 0,
        "failed": 0,
        "dry_run": bool(args.dry_run),
    }
    crops_abs = base / paths_cfg["outputs"]["crops_dir"]
    if not args.dry_run:
        try:
            if crops_abs.exists():
                shutil.rmtree(crops_abs)  # crops_dir is pipeline-owned output
            crops_abs.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            raise SystemExit(
                f"[crops] cannot reset {crops_abs}: {err} — refusing to write "
                "crops into a directory this run cannot fully regenerate"
            )
    for row, rec in zip(rows, ordered):
        if rec.family != "roboflow-yolo" or rec.bbox is None:
            crop_stats["non_roboflow"] += 1
            continue
        if args.dry_run:
            continue  # nothing written; the survivor still counts as would-crop
        try:
            res = crop_row(
                rec,
                row["row_id"],
                crops_abs,
                img_path=base / roots[rec.source] / rec.relpath,
                pad_frac=pad_frac,
            )
        except ValueError:
            crop_stats["invalid"] += 1  # degenerate zero-area box — invalid row
            continue
        except Exception as err:  # per-file guard: count and continue (T-01-05)
            print(f"[crop] failed on {rec.source}/{rec.relpath}: {err}")
            crop_stats["failed"] += 1
            continue
        row.update(res)
        row["crop_path"] = f"{crops_rel}/{res['crop_path']}" if crops_rel else res["crop_path"]
        crop_stats["written"] += 1

    # 8. WRITE ARTIFACTS (full run): manifest.csv; crops were written by the
    #    crop stage above. --dry-run writes NOTHING.
    if not args.dry_run:
        write_manifest(rows, base / paths_cfg["outputs"]["manifest"])

    # 9. AUDIT — console summary + the full report surfaced twice (INGEST-07):
    #    printed here AND written to reports/dedup_audit.md from the SAME
    #    string, so console and file can never diverge. No timestamps
    #    anywhere — byte-identical re-runs are the determinism contract.
    print(
        build_console_summary(
            source_rows, records, survivors, removed,
            stem_removed=stem_removed, crop_stats=crop_stats,
            conditional_decision=conditional_decision,
        )
    )
    report = build_audit_report(
        source_rows, records, survivors, removed, stem_removed,
        rows, params, class_cfg, conditional_decision,
        split_counts, leaks, crop_stats=crop_stats,
    )
    print(report)
    if not args.dry_run:
        report_path = base / paths_cfg["outputs"]["audit_report"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8", newline="\n")
    if args.dry_run:
        print("[dry-run] writes disabled — pipeline ran in memory only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
