"""Manifest-stage tests: guarded stratified split, cross-split leak check,
conditional >30 boundary, and the audit report contract — pure logic on
synthetic rows, no datasets needed (01-03-PLAN.md Task 1).

Proves (01-03 contract):
- assign_split trips with a clear SystemExit naming the offending class +
  per-class counts + the collect-more contingency when any canonical class
  has fewer than min_class_split (10) members; passes and returns a
  deterministic 80/20 assignment at >= 10 per class (same seed -> same
  assignment on re-run);
- leak_check returns 0 for two disjoint hash sets and counts > 0 when a test
  hash is within threshold 8 of a train hash (inclusive boundary: distance
  exactly 8 is a leak, 9 is not);
- evaluate_conditional includes a class at 31 and excludes one at exactly 30
  (strict greater-than boundary; zero usable samples excluded with reason);
- build_audit_report renders the full section contract — parameter block,
  census table, totals with the sanity band, separate stem/phash dedup lines,
  counted exclusions, conditional decision lines, conflicts, per-class
  train/test table with <30 collect-more flags, the leak-zero literal, and
  the crop-rule adjustment note — with NO wall-clock timestamps anywhere.
"""

import re
from types import SimpleNamespace

import pytest

from src.ingestion.audit import build_audit_report, no_box_only_stems
from src.ingestion.manifest import assign_split, evaluate_conditional, leak_check, write_manifest

# 9-bit windows stepped 5 bits apart: adjacent windows overlap 4 bits, so any
# two of these hashes sit >= 10 bits apart — a genuinely disjoint set for
# leak_check (nothing within Hamming 8 of anything else).
FAR_HASHES = [f"{0x1FF << (5 * i):016x}" for i in range(12)]


def make_rows(counts: dict[str, int]) -> list[dict]:
    """Synthetic post-dedup rows: counts[class] rows, distinct far-apart hashes."""
    rows = []
    for cls in sorted(counts):
        for i in range(counts[cls]):
            rows.append({
                "row_id": f"src_{cls}_{i:05d}",
                "source": "src",
                "original_path": f"datasets/src/{cls}_{i}.jpg",
                "canonical_class": cls,
                "phash_hex": FAR_HASHES[i % len(FAR_HASHES)],
                "split": "",
            })
    return rows


# ---------------------------------------------------------------------------
# assign_split — guarded stratified 80/20, AFTER dedup
# ---------------------------------------------------------------------------


def test_assign_split_guard_trips_below_10_with_counts_and_contingency():
    rows = make_rows({"NoAction": 20, "NoBall": 12, "Four": 9})
    with pytest.raises(SystemExit) as exc:
        assign_split(rows, test_size=0.2, random_state=42, min_class_split=10)
    msg = str(exc.value)
    assert "cannot stratify 80/20" in msg  # designed guard signature
    assert "Four" in msg  # offending class named
    assert "'Four': 9" in msg  # per-class counts present
    assert "NoAction" in msg and "'NoBall': 12" in msg
    assert "collect more" in msg and "collect-more contingency" in msg  # contingency


def test_assign_split_guard_names_offending_class_deterministically():
    counts = {"Six": 11, "Wide": 10, "Out": 9, "Four": 10,
              "NoBall": 10, "NoAction": 10}
    rows_a, rows_b = make_rows(counts), make_rows(counts)
    with pytest.raises(SystemExit) as exc_a:
        assign_split(rows_a, 0.2, 42, 10)
    with pytest.raises(SystemExit) as exc_b:
        assign_split(rows_b, 0.2, 42, 10)
    assert str(exc_a.value) == str(exc_b.value)  # same data -> same message
    assert "Out" in str(exc_a.value)  # the min-count class is the offender


def test_assign_split_passes_at_10_per_class_and_is_deterministic():
    counts = {c: 12 for c in ("Four", "Six", "Out", "Wide", "NoBall", "NoAction")}
    rows_a = make_rows(counts)
    rows_b = make_rows(counts)
    split_a = assign_split(rows_a, test_size=0.2, random_state=42, min_class_split=10)
    split_b = assign_split(rows_b, test_size=0.2, random_state=42, min_class_split=10)
    assert split_a == split_b  # same seed -> same per-class assignment
    by_id_a = {r["row_id"]: r["split"] for r in rows_a}
    by_id_b = {r["row_id"]: r["split"] for r in rows_b}
    assert by_id_a == by_id_b  # identical per-row assignment on re-run
    for row in rows_a:
        assert row["split"] in ("train", "test")  # split column filled
    for cls, sides in split_a.items():
        assert sides["train"] + sides["test"] == 12
        assert sides["train"] >= 8 and sides["test"] >= 2  # stratified 80/20 shape


def test_assign_split_returns_per_class_train_test_counts():
    rows = make_rows({"NoAction": 20, "NoBall": 10})
    split_counts = assign_split(rows, 0.2, 42, 10)
    assert set(split_counts) == {"NoAction", "NoBall"}
    for sides in split_counts.values():
        assert set(sides) == {"train", "test"}
        assert sides["train"] + sides["test"] in (10, 20)


def test_assign_split_guard_fires_on_empty_rows():
    with pytest.raises(SystemExit) as exc:
        assign_split([], 0.2, 42, 10)
    assert "cannot stratify 80/20" in str(exc.value)


# ---------------------------------------------------------------------------
# leak_check — cross-split phash pairs must be zero
# ---------------------------------------------------------------------------


def test_leak_check_zero_on_disjoint_hash_sets():
    rows = [
        {"row_id": "t0", "split": "train", "phash_hex": FAR_HASHES[0]},
        {"row_id": "t1", "split": "train", "phash_hex": FAR_HASHES[1]},
        {"row_id": "s0", "split": "test", "phash_hex": FAR_HASHES[2]},
        {"row_id": "s1", "split": "test", "phash_hex": FAR_HASHES[3]},
    ]
    assert leak_check(rows, threshold=8) == 0


def test_leak_check_detects_synthetic_leak_within_threshold():
    train_hash = "00000000000000ff"  # 8 low bits set
    leaked = "0000000000000f7f"  # 0xff ^ 0x7f -> Hamming distance 1
    rows = [
        {"row_id": "t0", "split": "train", "phash_hex": train_hash},
        {"row_id": "s0", "split": "test", "phash_hex": leaked},
    ]
    assert leak_check(rows, threshold=8) == 1


def test_leak_check_inclusive_at_8_distinct_at_9():
    # 0x1FF is 9 set bits (0..8); 0x100 differs in exactly 1 bit -> distance 8.
    rows = [{"row_id": "t0", "split": "train", "phash_hex": "00000000000001ff"},
            {"row_id": "s0", "split": "test", "phash_hex": "0000000000000100"}]
    assert leak_check(rows, threshold=8) == 1  # distance 8 counts (inclusive)
    rows[1]["phash_hex"] = "0000000000000000"  # distance 9
    assert leak_check(rows, threshold=8) == 0  # distance 9 is distinct


def test_leak_check_missing_hash_raises():
    rows = [{"row_id": "t0", "split": "train", "phash_hex": ""}]
    with pytest.raises(ValueError):
        leak_check(rows, threshold=8)


def test_leak_check_unassigned_split_raises_fail_closed():
    # WR-02: a row whose split is neither train nor test must raise — the
    # old behavior silently dropped it from both sides and under-counted.
    rows = [
        {"row_id": "t0", "split": "train", "phash_hex": FAR_HASHES[0]},
        {"row_id": "u0", "split": "", "phash_hex": FAR_HASHES[1]},
        {"row_id": "s0", "split": "test", "phash_hex": FAR_HASHES[2]},
    ]
    with pytest.raises(ValueError):
        leak_check(rows, threshold=8)


def test_leak_check_empty_side_raises_fail_closed():
    # WR-02: a one-sided comparison proves nothing — the old `return 0`
    # returned the success value for a degenerate call.
    rows = [{"row_id": "t0", "split": "train", "phash_hex": FAR_HASHES[0]}]
    with pytest.raises(ValueError):
        leak_check(rows, threshold=8)


# ---------------------------------------------------------------------------
# evaluate_conditional — INGEST-06 strict >30 boundary
# ---------------------------------------------------------------------------


def test_evaluate_conditional_includes_at_31():
    decision = evaluate_conditional({"LegBye": 31}, min_class_flag=30)
    assert decision["LegBye"]["included"] is True
    assert decision["LegBye"]["count"] == 31
    assert "31" in decision["LegBye"]["reason"] and "> 30" in decision["LegBye"]["reason"]


def test_evaluate_conditional_excludes_exactly_30():
    decision = evaluate_conditional({"LegBye": 30}, min_class_flag=30)
    assert decision["LegBye"]["included"] is False  # STRICTLY greater than 30
    assert "is not > 30" in decision["LegBye"]["reason"]


def test_evaluate_conditional_zero_usable_samples_excluded_with_reason():
    decision = evaluate_conditional({"DeadBall": 0, "LegBye": 13}, min_class_flag=30)
    assert decision["DeadBall"]["included"] is False
    assert decision["DeadBall"]["count"] == 0
    assert "no usable samples" in decision["DeadBall"]["reason"]  # never silent
    assert decision["LegBye"]["included"] is False
    assert decision["LegBye"]["reason"] == "usable count 13 is not > 30"


# ---------------------------------------------------------------------------
# build_audit_report — the full section contract (non-trip branch render)
# ---------------------------------------------------------------------------


def srow(name, count, family, classes=()):
    return {
        "name": name, "root": f"datasets/{name}", "family": family,
        "priority": 3 if family == "roboflow-yolo" else 1,
        "image_count": count, "raw_classes": list(classes),
        "status": "ok", "note": "" if family != "precomputed-csv"
        else "audit-only, excluded from the image pipeline",
    }


def rec(source, raw_label, outcome, multibox=False, relpath=None):
    return SimpleNamespace(
        source=source, relpath=relpath or f"{source}/{raw_label}_{source}.jpg",
        family="roboflow-yolo" if source == "roboflow_umpire_wide" else "filename-label",
        priority=3 if source == "roboflow_umpire_wide" else 1, raw_label=raw_label,
        outcome=outcome, canonical=None, multibox=multibox,
        conflict=False, conflict_detail="",
    )


def report_inputs():
    source_rows = [
        srow("roboflow_umpire_wide", 2497, "roboflow-yolo",
             ["four", "no_action", "out", "six", "umpire", "wide"]),
        srow("umpire_gestures", 145, "roboflow-yolo",
             ["DRS", "LegBye", "NoBall", "WideBall"]),
        srow("archive", 390, "filename-label",
             ["no_action", "no_ball", "out", "sixes", "wide"]),
        srow("archive3", 390, "filename-label",
             ["no_action", "no_ball", "out", "sixes", "wide"]),
        srow("archive1", 312, "folder-label", ["noball", "out", "six", "wide"]),
        srow("inception", 843, "filename-label",
             ["no_action", "no_ball", "non_umpire", "out", "sixes", "umpire", "wide"]),
        srow("umpire_pose_dataset.csv", 2909, "precomputed-csv"),
    ]
    # Roboflow records exercise the file-level stem accounting: 21 four
    # keepers where the first 6 form 3 stem pairs (18 unique keeper stems ->
    # 3 pooled Layer A removals) and both no_box copies share the first
    # keeper stem (excluded before dedup) -> 23 roboflow files over 18
    # stems = 5 file-level stem-copy removals = 3 pooled + 2 no_box, so the
    # report's reconciliation equation is arithmetically consistent.
    roboflow_records = []
    for i in range(21):
        stem = i // 2 if i < 6 else i - 3
        roboflow_records.append(rec(
            "roboflow_umpire_wide", "four", "keep", multibox=True,
            relpath=f"train/images/frame_{stem:04d}_jpg.rf.{'ab' * 16}.jpg",
        ))
    roboflow_records.append(rec(
        "roboflow_umpire_wide", None, "no_box",
        relpath=f"train/images/frame_0000_jpg.rf.{'cd' * 16}.jpg",
    ))
    roboflow_records.append(rec(
        "roboflow_umpire_wide", None, "no_box",
        relpath=f"train/images/frame_0000_jpg.rf.{'ef' * 16}.jpg",
    ))
    records = (
        [rec("umpire_gestures", "DRS", "excluded") for _ in range(32)]
        + [rec("inception", "umpire", "excluded") for _ in range(24)]
        + [rec("inception", "non_umpire", "excluded") for _ in range(21)]
        + [rec("umpire_gestures", "LegBye", "conditional") for _ in range(13)]
        + roboflow_records
        + [rec("archive", "no_action", "keep") for _ in range(20)]
    )
    survivors = [rec("archive", "no_action", "keep") for _ in range(40)]
    removed = [rec("archive3", "no_action", "keep") for _ in range(7)]
    stem_removed = [rec("roboflow_umpire_wide", "four", "keep") for _ in range(3)]
    rows = make_rows({c: 12 for c in ("Four", "Six", "Out", "Wide", "NoBall", "NoAction")})
    assign_split(rows, 0.2, 42, 10)
    rows[0]["conflict"] = True
    rows[0]["conflict_detail"] = "sixes; six"
    rows[1]["conflict"] = True
    rows[1]["conflict_detail"] = "noball; no_ball"
    params = {
        "phash_threshold": 8, "pad_frac": 0.15, "test_size": 0.2,
        "random_state": 42, "min_class_flag": 30, "min_class_split": 10,
    }
    class_cfg = {"conditional_classes": {"LegBye": ["legbye"], "DeadBall": ["deadball"]}}
    conditional_decision = evaluate_conditional(
        {"LegBye": 13, "DeadBall": 0}, min_class_flag=30
    )
    split_counts = {cls: {"train": 9, "test": 3} for cls in sorted({r["canonical_class"] for r in rows})}
    crop_stats = {"written": 40, "non_roboflow": 0, "invalid": 0, "failed": 0, "dry_run": False}
    return {
        "source_rows": source_rows, "records": records, "survivors": survivors,
        "removed": removed, "stem_removed": stem_removed, "rows": rows,
        "params": params, "class_cfg": class_cfg,
        "conditional_decision": conditional_decision,
        "split_counts": split_counts, "leaks": 0, "crop_stats": crop_stats,
    }


def test_build_audit_report_renders_full_contract():
    report = build_audit_report(**report_inputs())
    # Parameter block literals (determinism contract).
    assert "phash_threshold: 8" in report
    assert "random_state: 42" in report
    assert "min_class_flag: 30" in report and "min_class_split: 10" in report
    assert "imagehash version:" in report and "code version:" in report
    # Totals with the sanity band (RESEARCH.md census + dedup expectations).
    assert "Raw image files: 4577" in report
    assert "Kept after dedup: 40 (sanity band 1250-1450)" in report
    # Dedup lines — separate (Pitfall 1); stem-groups at file level with the
    # explicit pooled + no_box reconciliation.
    assert "Duplicates removed - Roboflow stem-groups: 5" in report
    assert "reconciliation: 3 pooled copies removed by Layer A" in report
    assert "= 5 file-level" in report
    assert "Duplicates removed - phash near-duplicates: 7" in report
    # Exclusions counted explicitly, never silent.
    assert "DRS: 32 excluded" in report
    assert "umpire: 24 excluded" in report
    assert "non_umpire: 21 excluded" in report
    assert "Zero-box images excluded as no_box: 2" in report
    assert "Multi-box images flagged (largest-area box wins): 21" in report
    assert "Invalid bbox rows rejected" in report
    # Conditional decision lines from evaluate_conditional (strict >30).
    assert "LegBye: 13 excluded (conditional class, usable count 13 is not > 30)" in report
    assert "DeadBall: 0 excluded" in report and "no usable samples" in report
    # Census table incl. the audit-only CSV and the expected-count column.
    assert "umpire_pose_dataset.csv" in report and "precomputed-csv" in report
    assert "excluded from the image pipeline" in report
    assert "expected (census)" in report and "2497" in report
    # Per-class post-dedup train/test table + <30 collect-more flags.
    assert "Per-class post-dedup counts" in report
    for cls in ("Four", "Six", "Out", "Wide", "NoBall", "NoAction"):
        assert cls in report
        assert f"FLAG {cls}: 12 usable samples < 30" in report
        assert "collect more" in report
    # Leak line literal.
    assert "Cross-split phash leaks (Hamming <= 8): 0" in report
    # Crop-rule log line + adjustment note (IN-01: counts are interpolated
    # from the data, never hardcoded — these inputs saw 0 umpire bbox rows
    # and 23 roboflow files, and the narrative must say exactly that).
    assert "umpire-class bbox rows" in report
    assert "largest-area" in report.lower()
    assert "umpire-class bbox rows seen: 0" in report
    assert "has 0 bbox rows across the 23-image export" in report
    # Conflicts count + top examples.
    assert "Label conflicts resolved" in report
    assert "sixes; six" in report


def test_build_audit_report_has_no_wall_clock_timestamps():
    report = build_audit_report(**report_inputs())
    assert not re.search(r"\d{4}-\d{2}-\d{2}", report), "date-like string found"
    assert not re.search(r"\d{2}:\d{2}:\d{2}", report), "time-like string found"
    assert not re.search(r"\d{4}/\d{2}/\d{2}", report), "date-like string found"


def test_build_audit_report_flags_no_class_when_all_above_30():
    inputs = report_inputs()
    inputs["rows"] = make_rows({c: 35 for c in ("Four", "Six", "Out", "Wide", "NoBall", "NoAction")})
    classes = sorted({r["canonical_class"] for r in inputs["rows"]})
    inputs["split_counts"] = {cls: {"train": 28, "test": 7} for cls in classes}
    report = build_audit_report(**inputs)
    assert "FLAG" not in report  # no class below 30 -> no collect-more flag
    assert "Kept after dedup: 40 (sanity band 1250-1450)" in report  # survivors len


# ---------------------------------------------------------------------------
# WR-03 reconciliation — verify-before-print on drifted data
# ---------------------------------------------------------------------------


def test_no_box_only_stems_counts_all_no_box_stems():
    # WR-03 term: a stripped stem whose files are ALL no_box over-attributes
    # (members - 1) file-level removals; mixed stems contribute 0.
    roboflow = [
        rec("roboflow_umpire_wide", "four", "keep",
            relpath=f"train/images/frame_0001_jpg.rf.{'aa' * 16}.jpg"),
        rec("roboflow_umpire_wide", None, "no_box",
            relpath=f"train/images/frame_0001_jpg.rf.{'bb' * 16}.jpg"),
        rec("roboflow_umpire_wide", None, "no_box",
            relpath=f"train/images/frame_0002_jpg.rf.{'cc' * 16}.jpg"),
        rec("roboflow_umpire_wide", None, "no_box",
            relpath=f"test/images/frame_0002_jpg.rf.{'dd' * 16}.jpg"),
    ]
    assert no_box_only_stems(roboflow) == 1  # only frame_0002 is all-no_box


def test_build_audit_report_reconciliation_corrected_for_all_no_box_stems():
    # Drift case: two no_box files share a NEW all-no_box stem, so the plain
    # pooled + no_box equation no longer holds — the corrected form with the
    # all-no_box-stems term must render, never the false equation.
    inputs = report_inputs()
    inputs["records"] = inputs["records"] + [
        rec("roboflow_umpire_wide", None, "no_box",
            relpath=f"valid/images/frame_9000_jpg.rf.{'11' * 16}.jpg"),
        rec("roboflow_umpire_wide", None, "no_box",
            relpath=f"valid/images/frame_9000_jpg.rf.{'22' * 16}.jpg"),
    ]
    report = build_audit_report(**inputs)
    assert "UNRECONCILED" not in report  # identity still holds in corrected form
    assert "+ 4 zero-box copies - 1 all-no_box stems excluded before dedup" in report
    assert "= 6 file-level" in report


def test_build_audit_report_reconciliation_flags_unreconciled_pool():
    # The identity is derived from the same records, so it can only break
    # when the pooled removals disagree with the pool Layer A actually saw
    # (a caller bug) — exactly the case the UNRECONCILED verdict exists for.
    inputs = report_inputs()
    inputs["stem_removed"] = inputs["stem_removed"][:2]  # 2 pooled, not 3
    report = build_audit_report(**inputs)
    assert "UNRECONCILED" in report and "investigate" in report
    assert "reconciliation: 2 pooled copies removed by Layer A" not in report


# ---------------------------------------------------------------------------
# IN-06 — CSV formula-injection guard at the manifest sink
# ---------------------------------------------------------------------------


def test_write_manifest_sanitizes_formula_prefix_cells(tmp_path):
    rows = [{
        "row_id": "s_00000", "source": "s", "original_path": "datasets/s/x.jpg",
        "family": "filename-label", "split0": "", "raw_label": '=HYPERLINK("http://x","y")',
        "canonical_class": "Four", "crop_x0": "", "crop_y0": "", "crop_x1": "",
        "crop_y1": "", "crop_path": "", "phash_hex": "0" * 16,
        "split": "train", "conflict": False, "conflict_detail": "",
    }]
    out = tmp_path / "manifest.csv"
    write_manifest(rows, out)
    text = out.read_text(encoding="utf-8")
    assert "'=HYPERLINK" in text  # formula-neutralized: renders as text
    assert "s_00000" in text  # clean cells pass through untouched
