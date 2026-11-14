"""Format-family adapters: parse every source into the uniform Record seam.

RESEARCH.md Pattern 2 — nothing downstream sees format-specific code; each
family parses into the same Record shape.

Security (ASVS V5/V14, RESEARCH.md Security Domain): every label/text file is
read via :func:`read_label_text` (utf-8, ``errors="replace"``) and each
decode-replacement event is counted for the audit — never silently corrupted.
Bbox floats are validated into [0, 1] before any math; malformed or
out-of-range rows are counted via :func:`record_invalid_box` and dropped from
box selection. ``datasets/`` is read-only; adapters never write.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

# Image extensions collected by the filename-label walker (plan contract).
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Trailing digit counter token(s) on filename labels: no_action_1 -> no_action,
# out_23 -> out, non_umpire_101 -> non_umpire. This is the on-disk-verified
# naming convention (RESEARCH.md Verified Dataset Census; the prior iteration's
# resolve_label strips the same trailing counter). A plain "before the first
# underscore" prefix would yield "no" for no_action_1 — outside the verified
# vocabulary, so the trailing-counter strip is the correct rule here.
_TRAILING_COUNTER = re.compile(r"(?:[_\-\s]*\d+)+$")

# Decode-replacement events observed by read_label_text (utf-8, errors="replace").
_DECODE_EVENTS: list[int] = []

# Invalid bbox rows observed while parsing label files (ASVS V5): malformed or
# out-of-range rows are recorded here for the audit — never silently passed.
# The counter seam ships alongside the decode counter; Plan 02's parsers call
# record_invalid_box.
_INVALID_BOX: list[str] = []

# Valid bbox rows seen while parsing, by class name (RESEARCH.md census
# cross-check). The annotation boxes carry the signal class — including the
# single umpire-class row across 2,497 roboflow_umpire_wide images, which the
# largest-area crop rule never selects (it loses the area contest in its
# multibox file) — so the audit logs this distribution explicitly and the
# umpire row is never silently dropped.
_BOX_ROW_CLASSES: Counter = Counter()


def decode_events() -> int:
    """Number of decode-replacement events observed while reading label files."""
    return len(_DECODE_EVENTS)


def record_invalid_box(path, line_no: int, reason: str) -> None:
    """Record one malformed/out-of-range bbox row for the audit (ASVS V5)."""
    _INVALID_BOX.append(f"{Path(path)}:{line_no}: {reason}")


def invalid_box_events() -> list[str]:
    """Invalid bbox row records (one string per rejected row)."""
    return list(_INVALID_BOX)


def bbox_row_class_counts() -> dict[str, int]:
    """Valid bbox rows seen while parsing, by class name (sorted; audit census)."""
    return dict(sorted(_BOX_ROW_CLASSES.items()))


def read_label_text(path) -> str:
    """Read a label/text file with encoding utf-8, errors="replace" (ASVS V5/V14).

    Every replacement character is recorded and surfaced in the audit via
    :func:`decode_events`. No skeleton adapter reads label files yet; Plan 02's
    Roboflow/folder parsers call this helper so the counting contract holds.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    _DECODE_EVENTS.extend(range(text.count("\ufffd")))
    return text


@dataclass
class Record:
    """The uniform record seam every adapter emits (01-01-PLAN.md <interfaces>).

    One record per parsed image regardless of family. ``outcome``,
    ``canonical`` and ``exclude_reason`` are filled by canonical.map_label;
    ``phash_hex`` is filled by the dedup stage (RESEARCH.md records carry
    ``record["phash_hex"] = str(h)``). ``source`` is stamped by the
    coordinator from the paths.yaml config key (adapters never know it).
    """

    source: str  # config key from paths.yaml, e.g. "archive"
    relpath: str  # posix-relative path under the source root
    family: str  # roboflow-yolo | folder-label | filename-label
    priority: int  # 3 roboflow-yolo, 2 folder-label, 1 filename-label (survivor + label priority)
    raw_label: str | None  # family-native raw label
    split0: str | None = None  # source's own split dir name; metadata ONLY, never the split source
    bbox: tuple | None = None  # (cx, cy, w, h) normalized, largest-area box; None for non-Roboflow
    box_class: str | None = None  # raw class of the selected box (Roboflow only)
    multibox: bool = False  # image had >1 bbox row
    outcome: str = ""  # keep | excluded | conditional | no_box | unknown ("" until mapped)
    canonical: str | None = None  # filled by canonical.map_label when outcome == keep
    exclude_reason: str | None = None
    phash_hex: str | None = None  # 16-hex-char 64-bit phash (dedup stage)
    conflict: bool = False  # cluster had >1 distinct raw label (dedup stage)
    conflict_detail: str = ""  # discarded labels, semicolon-joined (dedup stage)


def stamp_source(records, source: str):
    """Stamp the paths.yaml config key onto records (adapters never know it)."""
    for rec in records:
        rec.source = source
    return records


def filename_raw_label(stem: str) -> str:
    """Family-native raw label from a filename stem.

    Strips trailing digit counter token(s): ``no_action_1`` -> ``no_action``.
    Falls back to the whole stem when stripping leaves nothing (e.g. an
    all-digit stem), which then surfaces as an unknown-label hard error.
    """
    label = _TRAILING_COUNTER.sub("", stem).strip("_- ")
    return label or stem


def parse_filename_label(root, family: str, priority: int) -> list[Record]:
    """Parse a filename-labeled source tree into Records (real implementation).

    Sorted walk over .jpg/.jpeg/.png; raw_label = stem minus the trailing
    digit counter (no_action_1.jpg -> no_action); split0 = nearest parent
    directory name inside the source tree (e.g. train, model1_train) —
    metadata only, never the split source.
    """
    root = Path(root)
    records: list[Record] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = p.relative_to(root)
        records.append(
            Record(
                source="",
                relpath=rel.as_posix(),
                family=family,
                priority=priority,
                raw_label=filename_raw_label(p.stem),
                split0=rel.parent.name or None,
            )
        )
    return records


def parse_roboflow(root, family: str, priority: int) -> list[Record]:
    """Parse a Roboflow YOLOv8 export into Records.

    ``yaml.safe_load`` reads the export's data.yaml names list verbatim (no
    regex over YAML). Each split dir (train/valid/test, fixed order) pairs
    ``images/<split>/<stem>.jpg`` with the sibling ``labels/<split>/<stem>.txt``
    by stem; the split dir name is recorded as ``split0`` metadata ONLY — a
    source's own split never determines the manifest split column (Plan 03
    assigns the stratified split after dedup).

    Label rows parse as whitespace-separated ``class cx cy w h`` floats
    validated into [0, 1] (NaN/inf fail the range check); malformed or
    out-of-range rows are counted via :func:`record_invalid_box` and dropped
    from box selection — never silently passed. Zero valid rows -> ``outcome
    no_box`` (the image is still counted, excluded from class rows).

    Otherwise ONE record per image: the largest-area annotated box of ANY
    class wins (w*h in normalized space) — the research-driven adjustment of
    the CONTEXT 'largest umpire box' decision, because the annotation boxes
    carry the signal class (the umpire class has exactly 1 bbox row across
    2,497 roboflow_umpire_wide images while four/no_action/six/out/wide carry
    798/926/283/259/250). ``raw_label``/``box_class`` = the selected box's
    class; ``multibox`` flags files with >1 row; an umpire-class selection
    lands as excluded in the canonical stage where the audit logs it
    explicitly.
    """
    root = Path(root)
    data_yaml = root / "data.yaml"
    if not data_yaml.is_file():
        sys.exit(
            f"[roboflow] {root}: data.yaml not found — expected a Roboflow "
            "YOLOv8 export layout (data.yaml + train/valid/test with images/ + labels/)"
        )
    meta = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = meta.get("names")
    if not isinstance(names, list) or not names:
        sys.exit(
            f"[roboflow] {root}: data.yaml has no 'names' list — cannot map "
            "class indices; check the export"
        )
    records: list[Record] = []
    for split in ("train", "valid", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = img.relative_to(root).as_posix()
            box, box_class, multibox = _parse_label_boxes(
                lbl_dir / f"{img.stem}.txt", names
            )
            if box is None:
                records.append(
                    Record(
                        source="",
                        relpath=rel,
                        family=family,
                        priority=priority,
                        raw_label=None,
                        split0=split,
                        multibox=multibox,
                        outcome="no_box",
                    )
                )
            else:
                records.append(
                    Record(
                        source="",
                        relpath=rel,
                        family=family,
                        priority=priority,
                        raw_label=box_class,
                        split0=split,
                        bbox=box,
                        box_class=box_class,
                        multibox=multibox,
                    )
                )
    return records


def _parse_label_boxes(lbl_path, names) -> tuple[tuple | None, str | None, bool]:
    """Parse one YOLO label file -> (largest_box, box_class, multibox).

    Returns (None, None, multibox) when the file has zero valid rows (missing
    file, or every row malformed/out-of-range — those are counted via
    :func:`record_invalid_box`). ``multibox`` is True when the file had >1
    (non-empty) row. Ties on area resolve to the first row in file order —
    deterministic.
    """
    total_rows = 0
    valid: list[tuple[tuple, str]] = []
    if Path(lbl_path).is_file():
        text = read_label_text(lbl_path)
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            total_rows += 1
            parts = line.split()
            if len(parts) != 5:
                record_invalid_box(
                    lbl_path, line_no, f"expected 5 tokens 'class cx cy w h', got {len(parts)}"
                )
                continue
            try:
                idx = int(parts[0])
                cx, cy, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                record_invalid_box(lbl_path, line_no, f"non-numeric token in {parts!r}")
                continue
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                record_invalid_box(lbl_path, line_no, f"values outside [0,1]: {parts[1:]!r}")
                continue
            if not (0 <= idx < len(names)):
                record_invalid_box(lbl_path, line_no, f"class index {idx} out of range")
                continue
            _BOX_ROW_CLASSES[str(names[idx])] += 1
            valid.append(((cx, cy, w, h), str(names[idx])))
    if not valid:
        return None, None, total_rows > 1
    # Tie rule (IN-04): max(key=area) keeps the FIRST row in file order on an
    # area tie — deterministic, and deliberately NOT crops.largest_box's
    # lexicographic tie rule (that helper is tests-only). Swapping the
    # implementations would flip winners on tied files and change the
    # committed artifacts' bytes; change only via CONTEXT.
    box, box_class = max(valid, key=lambda item: item[0][2] * item[0][3])
    return box, box_class, total_rows > 1


def parse_folder_label(root, family: str, priority: int) -> list[Record]:
    """Parse a folder-labeled source tree into Records.

    Sorted walk over .jpg/.jpeg/.png under the root (archive1 layout:
    train/test/validate/{class}/). ``raw_label`` = the image's immediate
    parent directory name (the class dir); ``split0`` = the grandparent split
    dir name when present (>= 3 path parts) — metadata only, never the split
    source. No bbox: folder sources carry no annotations, so crop coords stay
    blank and no crop is written (full-image rows).
    """
    root = Path(root)
    records: list[Record] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = p.relative_to(root)
        split0 = rel.parent.parent.name if len(rel.parts) >= 3 else None
        records.append(
            Record(
                source="",
                relpath=rel.as_posix(),
                family=family,
                priority=priority,
                raw_label=rel.parent.name,
                split0=split0,
            )
        )
    return records
