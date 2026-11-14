"""Adapter tests on synthetic tmp_path fixtures (zero-byte images; no decode).

Covers the 01-02 behavior contract: largest-area annotated box of ANY class
wins (multibox flagged), zero-box -> no_box, out-of-range floats rejected to
the audit counter, utf-8 errors=replace tolerance, images/ + labels/ stem
pairing with missing-label -> no_box, data.yaml names mapped verbatim, and
folder-label resolution (raw_label = parent class dir, split0 = grandparent).
Also extends the Plan 01 filename-label coverage (trailing counter strip).
"""

import pytest

from src.ingestion.adapters import (
    decode_events,
    filename_raw_label,
    invalid_box_events,
    parse_folder_label,
    parse_filename_label,
    parse_roboflow,
)

RF_NAMES = ("four", "no_action", "out", "six", "umpire", "wide")


def make_roboflow(tmp_path, names=RF_NAMES):
    """Minimal Roboflow export: data.yaml + train/{images,labels} siblings."""
    root = tmp_path / "export"
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    yaml_names = ", ".join(f"'{n}'" for n in names)
    (root / "data.yaml").write_text(
        f"train: ../train/images\nnc: {len(names)}\nnames: [{yaml_names}]\n",
        encoding="utf-8",
    )
    return root


def test_largest_area_box_wins(tmp_path):
    root = make_roboflow(tmp_path)
    (root / "train" / "images" / "img1.jpg").write_bytes(b"")
    # Three box rows of differing areas; the largest (six) must win.
    (root / "train" / "labels" / "img1.txt").write_text(
        "0 0.5 0.5 0.10 0.10\n3 0.5 0.5 0.60 0.50\n2 0.5 0.5 0.30 0.30\n",
        encoding="utf-8",
    )
    recs = parse_roboflow(root, "roboflow-yolo", 3)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.bbox == (0.5, 0.5, 0.60, 0.50)
    assert rec.multibox is True
    assert rec.box_class == "six"
    assert rec.raw_label == "six"
    assert rec.split0 == "train"


def test_empty_label_is_no_box(tmp_path):
    root = make_roboflow(tmp_path)
    (root / "train" / "images" / "empty.jpg").write_bytes(b"")
    (root / "train" / "labels" / "empty.txt").write_text("", encoding="utf-8")
    recs = parse_roboflow(root, "roboflow-yolo", 3)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.outcome == "no_box"
    assert rec.bbox is None
    assert rec.raw_label is None
    assert rec.split0 == "train"


def test_out_of_range_row_rejected_to_audit(tmp_path):
    root = make_roboflow(tmp_path)
    (root / "train" / "images" / "bad.jpg").write_bytes(b"")
    (root / "train" / "labels" / "bad.txt").write_text(
        "0 1.2 0.5 0.3 0.3\n", encoding="utf-8"
    )
    before = len(invalid_box_events())
    recs = parse_roboflow(root, "roboflow-yolo", 3)
    # The row is rejected (counted for the audit) — no crash, no bad bbox kept.
    assert len(invalid_box_events()) == before + 1
    assert recs[0].outcome == "no_box"
    assert recs[0].bbox is None


def test_decode_replacement_does_not_raise(tmp_path):
    root = make_roboflow(tmp_path)
    (root / "train" / "images" / "enc.jpg").write_bytes(b"")
    # Invalid utf-8 bytes in the class-index token decode to U+FFFD.
    (root / "train" / "labels" / "enc.txt").write_bytes(
        b"\xff\xfe 0.5 0.5 0.2 0.2\n"
    )
    before_dec = decode_events()
    before_inv = len(invalid_box_events())
    recs = parse_roboflow(root, "roboflow-yolo", 3)  # must not raise
    assert decode_events() > before_dec
    # The garbled class token is malformed -> counted, no bbox kept.
    assert len(invalid_box_events()) == before_inv + 1
    assert recs[0].outcome == "no_box"


def test_stem_pairing_and_missing_label_is_no_box(tmp_path):
    root = make_roboflow(tmp_path)
    (root / "train" / "images" / "img1.jpg").write_bytes(b"")
    (root / "train" / "labels" / "img1.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    (root / "train" / "images" / "img2.jpg").write_bytes(b"")  # no label file
    recs = parse_roboflow(root, "roboflow-yolo", 3)
    by_rel = {r.relpath: r for r in recs}
    assert by_rel["train/images/img1.jpg"].bbox == (0.5, 0.5, 0.4, 0.4)
    assert by_rel["train/images/img1.jpg"].multibox is False
    assert by_rel["train/images/img2.jpg"].outcome == "no_box"


def test_data_yaml_names_mapped_verbatim(tmp_path):
    root = make_roboflow(tmp_path, names=("alpha", "beta"))
    (root / "train" / "images" / "idx.jpg").write_bytes(b"")
    (root / "train" / "labels" / "idx.txt").write_text(
        "1 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    recs = parse_roboflow(root, "roboflow-yolo", 3)
    assert recs[0].raw_label == "beta"
    assert recs[0].box_class == "beta"


def test_folder_labels_resolve_parent_dir_and_split(tmp_path):
    root = tmp_path / "archive1"
    for split in ("train", "test", "validate"):
        for cls in ("noball", "out", "six", "wide"):
            (root / split / cls).mkdir(parents=True, exist_ok=True)
    (root / "train" / "noball" / "a.jpg").write_bytes(b"")
    (root / "test" / "six" / "b.jpg").write_bytes(b"")
    recs = parse_folder_label(root, "folder-label", 2)
    by_rel = {r.relpath: r for r in recs}
    assert len(recs) == 2
    a = by_rel["train/noball/a.jpg"]
    assert a.raw_label == "noball"
    assert a.split0 == "train"
    assert a.bbox is None
    assert a.family == "folder-label"
    assert a.priority == 2
    b = by_rel["test/six/b.jpg"]
    assert b.raw_label == "six"
    assert b.split0 == "test"


def test_filename_raw_label_strips_trailing_counter():
    assert filename_raw_label("no_action_1") == "no_action"
    assert filename_raw_label("out_23") == "out"
    assert filename_raw_label("sixes") == "sixes"
