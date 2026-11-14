#!/usr/bin/env python
"""Phase 1 ingestion CLI coordinator.

Stage order per the RESEARCH.md architecture diagram:
discover -> parse -> canonical -> dedup (two layers) -> crop survivors ->
manifest -> audit (console). --dry-run runs everything in memory and writes
nothing (the crop stage counts would-be crops instead).

Flags:
  --config   paths.yaml location (default configs/paths.yaml; classes.yaml is
             loaded from the same directory)
  --dry-run  full pipeline in memory + console print; writes NOTHING
  --limit N  cap files parsed per source after sorting — development/sample
             runs only; the full run takes no flag
"""

from __future__ import annotations

import argparse
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
from ingestion.audit import build_console_summary
from ingestion.canonical import UnknownLabelError, load_class_map, map_label
from ingestion.crops import crop_row
from ingestion.dedup import cluster_by_phash, phash_file, stem_groups
from ingestion.discover import discover_sources
from ingestion.manifest import assemble_rows, write_manifest

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

    # 1. DISCOVER — per-source census rows (warn-and-continue; hard aborts inside).
    source_rows = discover_sources(paths_cfg)

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
    #    of ALL remaining records (threshold inclusive at 8).
    params = paths_cfg.get("params") or {}
    threshold = int(params.get("phash_threshold", 8))
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

    # 5. MANIFEST + CROPS — rows from post-dedup survivors (row IDs are the
    #    stable crop-name source, so the crop stage runs after row assembly
    #    and before the write; research diagram order: dedup -> crop
    #    survivors -> manifest). Only Roboflow rows carry a bbox; every other
    #    family stays a full-image row with blank crop fields, no copy made.
    rows = assemble_rows(survivors, class_cfg, roots=roots)
    crops_rel = str(paths_cfg["outputs"]["crops_dir"]).replace("\\", "/").rstrip("/")
    pad_frac = float(params.get("pad_frac", 0.15))
    crop_stats = {
        "written": 0,
        "non_roboflow": 0,
        "invalid": 0,
        "failed": 0,
        "dry_run": bool(args.dry_run),
    }
    ordered = sorted(survivors, key=lambda r: (r.source, r.relpath))
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
                base / paths_cfg["outputs"]["crops_dir"],
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
    if not args.dry_run:
        write_manifest(rows, base / paths_cfg["outputs"]["manifest"])

    # 6. AUDIT — console summary in one print call.
    print(
        build_console_summary(
            source_rows, records, survivors, removed,
            stem_removed=stem_removed, crop_stats=crop_stats,
        )
    )
    if args.dry_run:
        print("[dry-run] writes disabled — pipeline ran in memory only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
