#!/usr/bin/env python
"""Phase 1 ingestion CLI coordinator (walking skeleton).

Stage order per the RESEARCH.md architecture diagram:
discover -> parse -> canonical -> dedup -> manifest -> audit (console).

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
from ingestion.dedup import cluster_by_phash, phash_file
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
        if rec.raw_label is None:
            sys.exit(
                f"[canonical] record with no raw label: {rec.source}/{rec.relpath} "
                f"(family {rec.family}) — adapter must set raw_label or outcome no_box"
            )
    try:
        for rec in records:
            rec.outcome, rec.canonical, rec.exclude_reason = map_label(rec.raw_label, class_cfg)
    except UnknownLabelError as err:
        sys.exit(f"[canonical] {err}")

    # 4. DEDUP — phash the mapped pool (Layer B; threshold inclusive at 8).
    params = paths_cfg.get("params") or {}
    threshold = int(params.get("phash_threshold", 8))
    roots = {
        name: entry["root"]
        for name, entry in (paths_cfg.get("dataset_roots") or {}).items()
    }
    pool = [r for r in records if r.outcome in ("keep", "conditional")]
    for rec in tqdm(pool, desc="phash", unit="img", disable=None):
        img_path = base / roots[rec.source] / rec.relpath
        try:
            rec.phash_hex = phash_file(img_path)
        except Exception as err:  # surface the failing path — never a silent skip
            sys.exit(f"[phash] failed on {rec.source}/{rec.relpath}: {err}")
    survivors, removed = cluster_by_phash(pool, threshold=threshold)

    # 5. MANIFEST — rows from post-dedup survivors; split column written empty.
    rows = assemble_rows(survivors, class_cfg, roots=roots)
    if not args.dry_run:
        write_manifest(rows, base / paths_cfg["outputs"]["manifest"])

    # 6. AUDIT — console summary in one print call.
    print(build_console_summary(source_rows, records, survivors, removed))
    if args.dry_run:
        print("[dry-run] writes disabled — pipeline ran in memory only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
