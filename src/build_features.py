#!/usr/bin/env python
"""Phase 2 features build CLI: frozen manifest -> per-row pose extraction ->
data/processed/features.npz + the tracked reports/features_build.md provenance
report.

Stage order (single roster, never a datasets/ re-walk):
read data/processed/manifest.csv in stored row order -> per-row V5 path guard
(path_escape / file_missing, counted + skipped, loop continues) ->
PoseFeatureExtractor.extract (typed (None, ExtractionFailure) guard returns
mapped 1:1 into the closed stats set; per-row try/except -> "other"; the
batch NEVER raises mid-batch) -> write features.npz (features_raw (N_ok, 51)
float32 + features_eng (N_ok, 59) float32 + labels + split + row_ids aligned
to manifest row order over OK rows, plus failed_row_ids / failure_reasons
parallel unicode arrays — failures are explicit, never NaN placeholders,
never silently dropped) -> reconcile ok + failed == total (fail-closed) ->
write reports/features_build.md (counts + per-class/per-split OK
reconciliation + full parameter block; NO wall-clock timestamps, NO npz-file
hash — numpy's npz writer stamps zip entries with local time, so determinism
is claimed at param-text + array level only, RESEARCH.md Pitfall 1) — the
report is surfaced twice (console + file) from the SAME string, INGEST-07
discipline.

--dry-run reads + guards the manifest, prints the row census (total,
per-class counts, ok-eligible after path guards) and exits WITHOUT loading
the model or writing anything — the weights-free self-check of the whole
input contract.

Flags:
  --config   pose.yaml location (default configs/pose.yaml); paths.yaml is
             read from the same directory for outputs.manifest
  --dry-run  manifest census + path guards only; no model load, no writes
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import sys
from collections import Counter
from pathlib import Path

# Import the feature seam through the SAME 'src.pose_features' spelling the
# test suite uses (conftest.py puts the checkout root on sys.path). The bare
# 'pose_features' spelling would create a second module object for the same
# file (IN-03), splitting the module-level failure/scale-warning event lists
# and silently under-counting them in this build's report.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yaml

from src.pose_features import (
    PoseFeatureExtractor,
    ExtractionFailure,
    failure_events,
    load_pose_config,
    scale_warning_events,
)

# Static code version tag for the determinism parameter block (Pitfall 9
# discipline, audit.py pattern: thresholds + tool versions + a code version —
# never a wall-clock timestamp, so re-runs stay byte-identical). Bump when
# the build's behavior-affecting code changes. r1: first Phase 2 build —
# manifest roster over the Plan 01 seam (extract -> raw51 + eng59), npz
# contract features_raw/features_eng/labels/split/row_ids +
# failed_row_ids/failure_reasons, counts + parameter block in
# reports/features_build.md.
CODE_VERSION = "phase2-plan01-r1"

# The closed failure-reason set (CONTEXT never-raises decision): every way a
# manifest row can fail this build, counted explicitly. The extractor's typed
# reasons (no_person / no_keypoints / decode_error /
# keypoints_without_confidence) map 1:1 into these keys — no seam reason ever
# degrades into the "other" catch-all; "other" only catches UNEXPECTED
# exceptions (and an unrecognized future seam reason, printed loudly).
STATS_KEYS = (
    "ok",
    "no_person",
    "no_keypoints",
    "path_escape",
    "file_missing",
    "decode_error",
    "keypoints_without_confidence",
    "other",
)

# Packages recorded in the parameter block (IN-08 discipline: these are the
# determinism-relevant dependencies of the frozen artifact — exact pins live
# in requirements.txt, installed versions live here).
VERSION_PACKAGES = ("torch", "torchvision", "ultralytics", "opencv-python", "numpy")

FEATURE_KEYS = ("features_raw", "features_eng", "labels", "split", "row_ids")


def sha256_of(path) -> str:
    """sha256 hex digest of a file (provenance recording, V6-adjacent
    integrity metadata — not crypto)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_versions() -> dict:
    """Installed versions for VERSION_PACKAGES via importlib.metadata —
    dist metadata only, so this never imports torch/ultralytics/cv2 (keeps
    the module importable without the heavy deps; audit.py IN-08 pattern).
    PackageNotFoundError -> "unknown"."""
    versions = {}
    for pkg in VERSION_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover
            versions[pkg] = "unknown"
    return versions


def classify_row_path(row, base: Path):
    """V5 path guard for one manifest row (T-02-06 mitigation).

    Resolves the row image path (crop_path when non-blank, else
    original_path — repo-relative posix strings) against ``base`` and
    returns (resolved_path, None) when the path stays inside the repo root
    AND is an existing regular file; otherwise (None, typed_reason) with
    reason "path_escape" (resolves outside the root — never followed) or
    "file_missing" (inside the root but not an existing file; a directory
    counts as missing — ultralytics would scan it). Pure path logic, no
    model, no I/O beyond exists checks."""
    crop = str(row.get("crop_path", "") or "").strip()
    rel = crop or str(row.get("original_path", "") or "").strip()
    candidate = Path(rel)
    resolved = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        return None, "path_escape"
    if not resolved.is_file():
        return None, "file_missing"
    return resolved, None


def _unicode_array(values) -> np.ndarray:
    """numpy unicode array from a list of strings; empty lists still yield a
    '<U' array (never float64) and non-empty lists pick the max width."""
    return np.array(list(values), dtype=np.str_)


def write_features_npz(path, arrays: dict) -> None:
    """Write data/processed/features.npz (np.savez).

    Contract: features_raw (N_ok, 51) float32, features_eng (N_ok, 59)
    float32, labels / split / row_ids unicode arrays aligned to manifest row
    order over OK rows, plus failed_row_ids + failure_reasons parallel
    unicode arrays over EVERY failed row in encounter order (researcher
    disposition 3: failures are explicit parallel arrays — never NaN
    placeholders, never silently dropped). The npz is a gitignored runtime
    artifact; provenance lives in the tracked report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        features_raw=arrays["features_raw"],
        features_eng=arrays["features_eng"],
        labels=_unicode_array(arrays["labels"]),
        split=_unicode_array(arrays["split"]),
        row_ids=_unicode_array(arrays["row_ids"]),
        failed_row_ids=_unicode_array(arrays["failed_row_ids"]),
        failure_reasons=_unicode_array(arrays["failure_reasons"]),
    )


def extract_rows(manifest: pd.DataFrame, base: Path, extract_fn) -> tuple:
    """The per-row guard loop (crop-stage template, data_ingestion.py).

    Iterates the manifest in stored row order; NEVER re-walks datasets/.
    Per row: path guard -> extract_fn(resolved_path) -> typed-failure
    mapping into the closed stats set -> per-row try/except Exception ->
    "other" with a printed [features] line. Nothing raises mid-batch
    (CONTEXT locked decision): every failure is counted, persisted into
    (failures, stats), and the loop continues.

    Returns (ok_rows, failures, stats) where ok_rows is a list of
    (raw51, eng59, label, split, row_id) tuples, failures a list of
    (row_id, reason) pairs in encounter order, and stats a dict keyed by
    exactly STATS_KEYS.
    """
    stats = {key: 0 for key in STATS_KEYS}
    ok_rows: list = []
    failures: list = []
    for _, row in manifest.iterrows():
        row_id = str(row.get("row_id", "") or "")
        path, guard_reason = classify_row_path(row, base)
        if guard_reason is not None:
            stats[guard_reason] += 1
            failures.append((row_id, guard_reason))
            print(f"[features] rejected {row_id}: {guard_reason}")
            continue
        try:
            raw, payload = extract_fn(path)
        except Exception as err:  # per-row containment: count and continue
            stats["other"] += 1
            failures.append((row_id, "other"))
            print(f"[features] failed on {row_id}: {err}")
            continue
        if raw is None:
            reason = getattr(payload, "reason", "other")
            if reason not in stats:
                # Unrecognized future seam reason — counted loudly under
                # "other", never silently folded into a wrong bucket.
                print(f"[features] failed on {row_id}: unrecognized typed "
                      f"reason {reason!r} — counted under 'other'")
                stats["other"] += 1
                failures.append((row_id, str(reason)))
                continue
            stats[reason] += 1
            failures.append((row_id, reason))
            print(f"[features] failed on {row_id}: {reason}")
            continue
        stats["ok"] += 1
        ok_rows.append(
            (
                np.asarray(raw, dtype=np.float32),
                np.asarray(payload, dtype=np.float32),
                str(row.get("canonical_class", "") or ""),
                str(row.get("split", "") or ""),
                row_id,
            )
        )
    return ok_rows, failures, stats


def build_counts_section(total, stats, per_class_ok, per_split_ok,
                         class_census) -> list:
    """Pure counts section for reports/features_build.md.

    Renders total/ok/failed-by-reason, the fail-closed ok + failed == total
    reconciliation line, the unit-scale fallback count (Plan 01's
    scale_warning_events observable where the build runs), per-class and
    per-split OK counts, and a loud per-class zero-OK red flag when a
    canonical class dropped to 0 OK rows (would starve Phase 3 — e.g. Four,
    the scarce class)."""
    ok = stats["ok"]
    failed = total - ok
    lines: list = []
    lines.append("## Counts")
    lines.append("")
    lines.append(f"total manifest rows: {total}")
    lines.append(f"ok: {ok}")
    lines.append(f"failed: {failed}")
    for key in STATS_KEYS:
        if key == "ok":
            continue
        lines.append(f"  {key}: {stats[key]}")
    lines.append(f"reconciliation: ok + failed == total ({ok} + {failed} == {total})")
    lines.append(f"scale fallback warnings: {len(scale_warning_events())}")
    lines.append("")
    lines.append("Per-class OK counts:")
    for label in sorted(class_census):
        lines.append(f"  {label}: {per_class_ok.get(label, 0)}")
    lines.append("")
    lines.append("Per-split OK counts:")
    for split in sorted(per_split_ok):
        lines.append(f"  {split}: {per_split_ok[split]}")
    zero_ok = sorted(
        label for label, count in class_census.items() if per_class_ok.get(label, 0) == 0
    )
    lines.append("")
    if zero_ok:
        lines.append(
            "RED FLAG — canonical classes with 0 OK rows (Phase 3 would be "
            f"starved for these): {', '.join(zero_ok)}"
        )
    else:
        lines.append("Per-class zero-OK red flags: none")
    return lines


def build_param_block(cfg, num_threads, versions, weights_sha256,
                      manifest_sha256) -> list:
    """Pure parameter block (audit.py pattern): the exact values this build
    used, plus weights + manifest sha256 provenance and the static code
    version. NO wall-clock timestamps, NO datetime calls, NO npz-file hash —
    numpy's npz writer stamps zip entries with local time, so determinism is
    claimed and verified at param-text + array level only (RESEARCH.md
    Pitfall 1)."""
    lines: list = []
    lines.append("## Parameter block")
    lines.append("")
    lines.append(f"det_conf: {cfg.get('det_conf', '')}")
    lines.append(f"imgsz: {cfg.get('imgsz', '')}")
    lines.append(f"mask_conf: {cfg.get('mask_conf', '')}")
    lines.append(f"device: {cfg.get('device', '')}")
    lines.append(f"torch num threads: {num_threads}")
    for pkg in VERSION_PACKAGES:
        lines.append(f"{pkg} version: {versions.get(pkg, 'unknown')}")
    lines.append(f"weights sha256: {weights_sha256}")
    lines.append(f"manifest sha256: {manifest_sha256}")
    lines.append(f"code version: {CODE_VERSION}")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="build_features",
        description=(
            "Phase 2 features build: read the frozen manifest, extract pose "
            "features per row, write data/processed/features.npz and the "
            "tracked reports/features_build.md provenance report."
        ),
    )
    ap.add_argument("--config", default="configs/pose.yaml")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="manifest census + path guards only; no model load, no writes",
    )
    args = ap.parse_args(argv)

    config_path = Path(args.config)
    base = config_path.resolve().parent.parent  # configs/pose.yaml -> repo root
    pose_cfg = load_pose_config(config_path)
    paths_cfg = (
        yaml.safe_load((config_path.parent / "paths.yaml").read_text(encoding="utf-8"))
        or {}
    )
    manifest_path = base / str(paths_cfg["outputs"]["manifest"])
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    total = len(manifest)
    class_census = Counter(manifest["canonical_class"])

    if args.dry_run:
        # Weights-free self-check of the whole input contract: census +
        # path guards, no model, no writes.
        eligible = 0
        guard_counts: Counter = Counter()
        for _, row in manifest.iterrows():
            _path, guard_reason = classify_row_path(row, base)
            if guard_reason is not None:
                guard_counts[guard_reason] += 1
            else:
                eligible += 1
        print("[dry-run] manifest row census (stored order, no model loaded):")
        print(f"  total rows: {total}")
        for label in sorted(class_census):
            print(f"  {label}: {class_census[label]}")
        print(f"  ok-eligible after path guards: {eligible}")
        for reason in ("path_escape", "file_missing"):
            print(f"  {reason}: {guard_counts.get(reason, 0)}")
        print("[dry-run] writes disabled — no model loaded, nothing written")
        return 0

    # Real run: ONE extractor instance before the loop (never per row); the
    # model itself loads lazily on first extract (heavy torch/ultralytics
    # imports stay inside the extraction path).
    extractor = PoseFeatureExtractor(config_path)
    ok_rows, failures, stats = extract_rows(
        manifest, base, lambda image: extractor.extract(image)
    )

    # Fail-closed reconciliation (T-02-04): every manifest row is either ok
    # or counted under exactly one typed reason — a mismatch refuses to
    # write the frozen artifact.
    if stats["ok"] + sum(v for k, v in stats.items() if k != "ok") != total:
        raise SystemExit(
            f"[features] accounting mismatch: ok {stats['ok']} + failed "
            f"{total - stats['ok']} != total {total} — refusing to write a "
            "silently-dropped or over-counted frozen artifact"
        )

    per_class_ok = Counter(label for _r, _e, label, _s, _i in ok_rows)
    per_split_ok = Counter(split for _r, _e, _label, split, _i in ok_rows)

    lines: list = []
    lines.append("# Phase 2 features build report")
    lines.append("")
    lines.append(
        "Provenance for the frozen data/processed/features.npz over the "
        "Phase 1 manifest (code version lineage: reports/dedup_audit.md, "
        "phase1-plan03-r1). Deterministic: no wall-clock timestamps; the npz "
        "zip bytes differ per run (numpy stamps local time), so determinism "
        "is claimed at parameter-text + array level only."
    )
    lines.append("")
    lines.extend(
        build_counts_section(total, stats, per_class_ok, per_split_ok, class_census)
    )
    lines.append("")
    weights_path = base / str(pose_cfg["weights_path"])
    lines.extend(
        build_param_block(
            pose_cfg,
            _torch_num_threads(),
            collect_versions(),
            sha256_of(weights_path),
            sha256_of(manifest_path),
        )
    )

    report = "\n".join(lines) + "\n"

    arrays = {
        "features_raw": (
            np.stack([r for r, _e, _l, _s, _i in ok_rows])
            if ok_rows
            else np.empty((0, 51), dtype=np.float32)
        ),
        "features_eng": (
            np.stack([e for _r, e, _l, _s, _i in ok_rows])
            if ok_rows
            else np.empty((0, 59), dtype=np.float32)
        ),
        "labels": [label for _r, _e, label, _s, _i in ok_rows],
        "split": [split for _r, _e, _l, split, _i in ok_rows],
        "row_ids": [row_id for _r, _e, _l, _s, row_id in ok_rows],
        "failed_row_ids": [row_id for row_id, _reason in failures],
        "failure_reasons": [reason for _row_id, reason in failures],
    }
    npz_path = base / "data" / "processed" / "features.npz"
    write_features_npz(npz_path, arrays)

    report_path = base / "reports" / "features_build.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")

    # INGEST-07 discipline: console + file from the SAME string.
    print(report)
    if len(failure_events()) != sum(v for k, v in stats.items() if k != "ok"):
        print(
            "[features] note: seam event list carries "
            f"{len(failure_events())} entries vs {total - stats['ok']} failed "
            "rows (sequence-path recorders may double-record; build rows are "
            "counted from the row loop, which is authoritative)"
        )
    return 0


def _torch_num_threads() -> int:
    """torch.get_num_threads() recorded for the parameter block (Pitfall 8:
    thread count is a determinism-relevant setting). Imported lazily so
    --dry-run and module import never need torch."""
    import torch

    return int(torch.get_num_threads())


if __name__ == "__main__":
    sys.exit(main())
