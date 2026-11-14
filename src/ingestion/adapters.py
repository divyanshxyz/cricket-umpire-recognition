"""Format-family adapters: parse every source into the uniform Record seam.

RESEARCH.md Pattern 2 — nothing downstream sees format-specific code; each
family parses into the same Record shape. ``parse_roboflow`` and
``parse_folder_label`` ship as seam stubs; Plan 02 fills them behind this
identical contract (additive, no renames).

Security (ASVS V5/V14, RESEARCH.md Security Domain): every label/text file is
read via :func:`read_label_text` (utf-8, ``errors="replace"``) and each
decode-replacement event is counted for the audit — never silently corrupted.
``datasets/`` is read-only; adapters never write.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

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


def decode_events() -> int:
    """Number of decode-replacement events observed while reading label files."""
    return len(_DECODE_EVENTS)


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

    SEAM STUB — Plan 02 fills this parser (yaml.safe_load of data.yaml names,
    labels/*.txt parsed to the largest-area box with floats validated into
    [0, 1], zero-box -> outcome no_box, split0 from the train/valid/test dir)
    behind this identical contract. The dispatch and Record shape are real,
    so filling it changes nothing downstream.
    """
    return []


def parse_folder_label(root, family: str, priority: int) -> list[Record]:
    """Parse a folder-labeled source tree into Records.

    SEAM STUB — Plan 02 fills this parser (nearest parent folder is the raw
    label, resolved nearest-to-farthest; split0 from train/test/validate dirs
    recorded as metadata only) behind this identical contract. The dispatch
    and Record shape are real, so filling it changes nothing downstream.
    """
    return []
