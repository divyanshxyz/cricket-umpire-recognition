"""build_features contract tests (pure, weight-free): V5 path guards, typed
failure accounting over the closed reason set, the ok + failed == total
equality, timestamp-free param-block rendering, the npz key/shape/dtype
contract, and the counts-section red-flag logic. No torch, no weights, no
manifest I/O — synthetic DataFrames + tmp_path bases only; the module itself
must stay importable without the heavy deps."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.build_features import (
    STATS_KEYS,
    build_counts_section,
    build_param_block,
    classify_row_path,
    extract_rows,
    write_features_npz,
)
from src.pose_features import ExtractionFailure


def _manifest(rows):
    return pd.DataFrame(
        rows,
        columns=["row_id", "crop_path", "original_path", "canonical_class", "split"],
    )


def _ok_extract(raw=None):
    def extract_fn(path):
        return (
            np.zeros(51, dtype=np.float32) if raw is None else raw,
            np.zeros(59, dtype=np.float32),
        )

    return extract_fn


def test_path_escape_rejected_and_loop_continues(tmp_path):
    # A row whose path resolves outside the base root is a typed
    # path_escape REJECTION (never followed); the loop continues and the
    # next valid row is still extracted.
    outside = tmp_path.parent / "evil.jpg"
    ok_file = tmp_path / "img_ok.jpg"
    ok_file.write_bytes(b"x")  # guard only needs an existing regular file
    manifest = _manifest(
        [
            ("r1", "", "../evil.jpg", "Six", "train"),
            ("r2", "", "C:/Windows/nonexistent_elsewhere.jpg", "Six", "train"),
            ("r3", "", "img_ok.jpg", "Wide", "test"),
        ]
    )
    calls = []

    def spy_extract(path):
        calls.append(path)
        return np.ones(51, dtype=np.float32), np.ones(59, dtype=np.float32)

    ok_rows, failures, stats = extract_rows(manifest, tmp_path, spy_extract)
    assert len(calls) == 1  # only the in-root existing row reached extraction
    assert calls[0] == ok_file.resolve()
    assert stats["path_escape"] == 2
    assert stats["ok"] == 1
    assert stats["file_missing"] == 0
    assert ("r1", "path_escape") in failures and ("r2", "path_escape") in failures
    assert stats["ok"] + sum(v for k, v in stats.items() if k != "ok") == 3
    assert len(ok_rows) == 1 and ok_rows[0][4] == "r3"


def test_crafted_path_strings_are_typed_not_fatal(tmp_path):
    # CR-01 regression: a hostile manifest path string (embedded null byte)
    # makes resolve()/stat raise ValueError INSIDE the path guard itself —
    # the guard must contain it as a typed path_escape row (never followed,
    # never fatal) and the loop must continue to the next row. Before the
    # fix this crashed the whole build (guard call sat outside the per-row
    # try/except), violating the locked never-raises-mid-batch decision.
    ok_file = tmp_path / "img_ok.jpg"
    ok_file.write_bytes(b"x")
    manifest = _manifest(
        [
            ("r_null", "", "foo\x00bar.jpg", "Six", "train"),
            ("r_ok", "", "img_ok.jpg", "Wide", "test"),
        ]
    )
    ok_rows, failures, stats = extract_rows(manifest, tmp_path, _ok_extract())
    assert stats["path_escape"] == 1  # the null-byte row, typed and counted
    assert stats["ok"] == 1  # the loop continued past the hostile row
    assert stats["other"] == 0 and stats["file_missing"] == 0
    assert failures == [("r_null", "path_escape")]
    assert len(ok_rows) == 1 and ok_rows[0][4] == "r_ok"
    # The guard function itself (also used by the --dry-run loop) returns
    # the typed pair instead of raising.
    path, reason = classify_row_path(
        {"crop_path": "", "original_path": "foo\x00bar.jpg"}, tmp_path
    )
    assert path is None and reason == "path_escape"


def test_file_missing_rejected_not_extracted(tmp_path):
    # A path inside the root that is not an existing file is a typed
    # file_missing; extraction is never attempted on it.
    manifest = _manifest([("r1", "", "missing.jpg", "Out", "train")])
    ok_rows, failures, stats = extract_rows(manifest, tmp_path, _ok_extract())
    assert stats["file_missing"] == 1
    assert stats["ok"] == 0
    assert failures == [("r1", "file_missing")]
    assert ok_rows == []


def test_crop_path_preferred_over_original(tmp_path):
    # Resolution contract: crop_path when non-blank, else original_path.
    crop = tmp_path / "crop_r1.jpg"
    crop.write_bytes(b"x")
    manifest = _manifest([("r1", "crop_r1.jpg", "missing_original.jpg", "Four", "test")])
    ok_rows, failures, stats = extract_rows(manifest, tmp_path, _ok_extract())
    assert stats["ok"] == 1 and failures == []
    assert ok_rows[0][2] == "Four" and ok_rows[0][3] == "test"


def test_stats_accounting_closed_set_and_equality(tmp_path):
    # Fake per-row results drive every bucket: 1 ok, 1 typed no_person,
    # 1 unexpected exception -> "other", 1 file_missing; the stats keys are
    # EXACTLY the closed set and ok + failed == total holds.
    ok_file = tmp_path / "img.jpg"
    ok_file.write_bytes(b"x")
    manifest = _manifest(
        [
            ("r_ok", "", "img.jpg", "NoAction", "train"),
            ("r_np", "", "img.jpg", "NoBall", "train"),
            ("r_boom", "", "img.jpg", "Six", "test"),
            ("r_miss", "", "gone.jpg", "Wide", "test"),
        ]
    )

    def flaky_extract(path):
        # All three extractable rows share tmp/img.jpg, so the crash is
        # triggered by CALL ORDER (3rd extractable row = r_boom), not path.
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated unexpected crash")
        return np.zeros(51, dtype=np.float32), np.zeros(59, dtype=np.float32)

    calls = {"n": 0}
    ok_rows, failures, stats = extract_rows(manifest, tmp_path, flaky_extract)
    assert set(stats) == set(STATS_KEYS)
    assert stats["ok"] == 2  # r_ok + r_np extract fine (payload checked below)
    assert stats["other"] == 1  # r_boom: per-row containment, never raises
    assert stats["file_missing"] == 1
    assert stats["ok"] + sum(v for k, v in stats.items() if k != "ok") == 4
    assert len(failures) == 2
    assert ("r_boom", "other") in failures
    assert ("r_miss", "file_missing") in failures
    assert len(ok_rows) == 2

    # Typed-failure bucket: the extractor's (None, ExtractionFailure) guard
    # return maps 1:1 into its named stats key (never "other").
    def typed_fail_extract(path):
        return None, ExtractionFailure("no_person", "no box")

    ok_rows2, failures2, stats2 = extract_rows(
        _manifest([("r_np", "", "img.jpg", "NoBall", "train")]),
        tmp_path,
        typed_fail_extract,
    )
    assert stats2["no_person"] == 1 and stats2["other"] == 0
    assert failures2 == [("r_np", "no_person")]
    assert ok_rows2 == []


def test_param_block_required_keys_and_timestamp_free():
    cfg = {
        "det_conf": 0.25,
        "imgsz": 640,
        "mask_conf": 0.2,
        "device": "cpu",
        "weights_path": "models/yolov8n-pose.pt",
    }
    versions = {pkg: "0.0.0-test" for pkg in ("torch", "torchvision", "ultralytics", "opencv-python", "numpy")}
    lines = build_param_block(
        cfg,
        num_threads=8,
        versions=versions,
        weights_sha256="a" * 64,
        manifest_sha256="b" * 64,
    )
    text = "\n".join(lines)
    for expected in (
        "det_conf: 0.25",
        "imgsz: 640",
        "mask_conf: 0.2",
        "device: cpu",
        "torch num threads: 8",
        "torch version: 0.0.0-test",
        "torchvision version: 0.0.0-test",
        "ultralytics version: 0.0.0-test",
        "opencv-python version: 0.0.0-test",
        "numpy version: 0.0.0-test",
        "weights sha256: " + "a" * 64,
        "manifest sha256: " + "b" * 64,
        "code version: phase2-plan01-r1",
    ):
        assert expected in text, f"missing param-block line: {expected}"
    # Determinism contract: NO rendered line matches a wall-clock date.
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text)


def test_counts_section_reconciliation_and_zero_ok_red_flag():
    stats = {key: 0 for key in STATS_KEYS}
    stats.update({"ok": 3, "no_person": 1})
    class_census = {"Four": 1, "Six": 2, "Wide": 1}
    per_class_ok = {"Six": 2, "Wide": 1}
    per_split_ok = {"train": 2, "test": 1}
    lines = build_counts_section(4, stats, per_class_ok, per_split_ok, class_census)
    text = "\n".join(lines)
    assert "total manifest rows: 4" in text
    assert "ok: 3" in text and "failed: 1" in text
    assert "  no_person: 1" in text
    assert "reconciliation: ok + failed == total (3 + 1 == 4)" in text
    assert "  Four: 0" in text  # scarce class dropped to zero OK rows
    assert "RED FLAG" in text and "Four" in text

    # All classes OK -> the explicit 'none' line instead of a red flag.
    stats_all = {key: 0 for key in STATS_KEYS}
    stats_all["ok"] = 4
    lines_ok = build_counts_section(
        4, stats_all, {"Four": 1, "Six": 2, "Wide": 1}, per_split_ok, class_census
    )
    assert "Per-class zero-OK red flags: none" in "\n".join(lines_ok)
    assert not any("RED FLAG" in line for line in lines_ok)


def test_write_features_npz_round_trip(tmp_path):
    npz_path = tmp_path / "features.npz"
    raw = np.arange(3 * 51, dtype=np.float32).reshape(3, 51)
    eng = np.arange(3 * 59, dtype=np.float32).reshape(3, 59)
    arrays = {
        "features_raw": raw,
        "features_eng": eng,
        "labels": ["Six", "Wide", "Four"],
        "split": ["train", "test", "train"],
        "row_ids": ["a_00000", "a_00001", "b_00000"],
        "failed_row_ids": ["c_00000", "c_00001"],
        "failure_reasons": ["no_person", "path_escape"],
    }
    write_features_npz(npz_path, arrays)
    with np.load(npz_path) as z:
        assert set(z.files) == {
            "features_raw",
            "features_eng",
            "labels",
            "split",
            "row_ids",
            "failed_row_ids",
            "failure_reasons",
        }
        assert z["features_raw"].shape == (3, 51)
        assert z["features_raw"].dtype == np.float32
        assert (z["features_raw"] == raw).all()
        assert z["features_eng"].shape == (3, 59)
        assert z["features_eng"].dtype == np.float32
        assert (z["features_eng"] == eng).all()
        assert z["labels"].tolist() == ["Six", "Wide", "Four"]
        assert z["labels"].dtype.kind == "U"
        assert z["split"].dtype.kind == "U" and z["row_ids"].dtype.kind == "U"
        assert len(z["failed_row_ids"]) == len(z["failure_reasons"]) == 2
        assert z["failed_row_ids"].tolist() == ["c_00000", "c_00001"]
        assert z["failure_reasons"].tolist() == ["no_person", "path_escape"]


def test_module_source_has_no_top_level_heavy_imports():
    # The plan's weights-free contract: src.build_features stays importable
    # without torch/ultralytics/cv2 — the heavy imports live inside the
    # extraction path (the lazy _torch_num_threads helper), never at module
    # top level. Source-level assertion because torch IS installed in this
    # venv (an import-time failure would not reproduce here).
    import inspect

    import src.build_features as bf

    source = inspect.getsource(bf)
    for dep in ("torch", "ultralytics", "cv2"):
        assert not re.search(rf"^import {dep}\b", source, re.M), dep
        assert not re.search(rf"^from {dep}\b", source, re.M), dep
    # The single sanctioned torch usage is nested in the lazy helper.
    assert "    import torch" in source


def test_stats_keys_are_exactly_the_closed_set():
    assert STATS_KEYS == (
        "ok",
        "no_person",
        "no_keypoints",
        "path_escape",
        "file_missing",
        "decode_error",
        "keypoints_without_confidence",
        "other",
    )
