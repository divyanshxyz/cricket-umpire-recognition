"""Ablation report contract tests (TRAIN-07/ordering + TRAIN-08 lineage,
Plan 03-03).

Weight-free: synthetic metric rows + tmp_path writers, no npz, no torch
training, no real artifacts. Pinned contracts:

- Fixed row order (MODULE_ORDER enforcement): a writer fed metrics dicts
  in scrambled insertion order still renders table rows A, B, C, D in
  order — dict order can never reorder the table (TRAIN-07/ordering).
- Column order: per-class F1 columns render in the shared encoder order
  [Four, NoAction, NoBall, Out, Six, Wide] (Pitfall 4 — never re-order).
- Tie-break determinism (TRAIN-07/adjacency): the best-proposed verdict
  is an explicit ordered scan over the fixed B, C, D tuple — EQUAL
  macro-F1 ties resolve to the EARLIEST model in that order, and the
  tie-break sentence renders so a reader knows the rule.
- Verdict + caveat presence: "Best proposed model", "Beats baseline
  (Model A)", "macro-F1 >= 0.90 target", the Four n=5 caveat (with the
  24-usable < 30 flag) and the single-seed caveat all render; the
  augmentation probe renders in its own separated section and NEVER as a
  main-table row.
- Partial --models runs (WR-02): an UNMEASURED baseline comparison
  renders the explicit "n/a — not measured in this run (partial --models
  run)" line — never a fabricated NO / NOT MET for a model that was
  never evaluated this run.
- Determinism: no rendered line matches a wall-clock date
  (\\d{4}-\\d{2}-\\d{2}) — the features_build.md posture.
- Lineage (WR-03): the artifact sha256 lines render in the tracked
  report ONLY for artifacts the train stage actually produced this run —
  a stale leftover under models/ from an earlier run renders no line
  rather than a lie (artifacts are gitignored; hashes are the
  provenance).
- classification_report.txt: one section per evaluated model in fixed
  order, explicit not-trained markers for the rest; confusion_matrix.png
  is a non-empty 2x2-subplot grid (matplotlib Agg).
"""

import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.models.data import canonical_classes, fit_label_encoder, sha256_of
from src.train_ablation import (
    _ablation_report_text,
    _classification_report_text,
    _write_confusion_png,
    compute_verdict,
)
import src.train_ablation as ta

LE = fit_label_encoder(canonical_classes())


def _synth_metrics(seed: int) -> dict:
    """Deterministic synthetic metrics block for one model (same key
    contract the guarded eval produces)."""
    rng = np.random.default_rng(seed)
    proba = rng.random((12, 6))
    proba = proba / proba.sum(axis=1, keepdims=True)
    y_true = np.arange(12) % 6
    y_pred = proba.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_per_class": [
            float(v)
            for v in f1_score(y_true, y_pred, average=None, labels=list(range(6)))
        ],
        "roc_auc_ovr_macro": float(
            roc_auc_score(
                y_true, proba, multi_class="ovr", average="macro",
                labels=list(range(6)),
            )
        ),
        "cm": confusion_matrix(y_true, y_pred, labels=list(range(6))),
        "classification_report": classification_report(
            y_true, y_pred, labels=list(range(6)),
            target_names=list(LE.classes_), digits=4,
        ),
    }


def _synth_census() -> dict:
    per_class = {
        "Four": (19, 5), "NoAction": (83, 21), "NoBall": (69, 16),
        "Out": (39, 10), "Six": (43, 11), "Wide": (81, 21),
    }
    return {"total": 418, "train": 334, "test": 84, "per_class": per_class}


def _synth_lineage() -> dict:
    return {
        "features_sha256": "42" * 32,
        "manifest_sha256": "ab" * 32,
        "digest": "cd" * 32,
        "num_threads": 1,
    }


def _render(metrics, meta=None, probe=None, verdict=None, models_dir=None,
            produced_artifacts=None):
    return _ablation_report_text(
        _synth_census(), metrics, meta or {}, probe,
        verdict if verdict is not None else compute_verdict(metrics),
        334, LE, _synth_lineage(),
        models_dir if models_dir is not None else Path("."),
        produced_artifacts,
    )


# ---------------------------------------------------------------------------
# (a) Fixed row order regardless of dict insertion order
# ---------------------------------------------------------------------------


def test_report_rows_render_in_fixed_abcd_order():
    metrics = {
        "C": _synth_metrics(1),
        "A": _synth_metrics(0),
        "D": _synth_metrics(3),
        "B": _synth_metrics(2),
    }  # deliberately scrambled insertion order
    text = _render(metrics)
    rows = [ln for ln in text.splitlines() if re.match(r"^\| [ABCD] \|", ln)]
    ids = [r.split("|")[1].strip() for r in rows]
    assert ids == ["A", "B", "C", "D"], ids


def test_report_skips_untrained_models_but_keeps_row_order():
    text = _render({"D": _synth_metrics(0), "A": _synth_metrics(1)})
    rows = [ln for ln in text.splitlines() if re.match(r"^\| [ABCD] \|", ln)]
    ids = [r.split("|")[1].strip() for r in rows]
    assert ids == ["A", "D"], ids


# ---------------------------------------------------------------------------
# (b) Column order: per-class F1 columns in encoder order
# ---------------------------------------------------------------------------


def test_report_columns_follow_encoder_order():
    text = _render({"A": _synth_metrics(0)})
    header = next(ln for ln in text.splitlines() if ln.startswith("| model |"))
    expected = (
        "| model | accuracy | macro-F1 | F1 Four | F1 NoAction | F1 NoBall "
        "| F1 Out | F1 Six | F1 Wide | ROC-AUC ovr-macro |"
    )
    assert header == expected, header


# ---------------------------------------------------------------------------
# (c) Tie-break determinism on equal macro-F1 (explicit ordered scan)
# ---------------------------------------------------------------------------


def test_verdict_tie_breaks_to_earliest_proposed_model():
    b = _synth_metrics(0)
    c = _synth_metrics(1)
    c["f1_macro"] = b["f1_macro"]  # EXACT tie between B and C
    v = compute_verdict({"A": _synth_metrics(2), "B": b, "C": c})
    assert v["best_id"] == "B"  # earliest of B, C, D wins the tie

    c2 = _synth_metrics(1)
    d = _synth_metrics(3)
    d["f1_macro"] = c2["f1_macro"]  # tie between C and D, B lower
    b2 = _synth_metrics(0)
    b2["f1_macro"] = 0.1
    v2 = compute_verdict({"A": _synth_metrics(2), "B": b2, "C": c2, "D": d})
    assert v2["best_id"] == "C"  # earliest of the tied pair


def test_verdict_tie_break_sentence_renders():
    metrics = {"A": _synth_metrics(0), "B": _synth_metrics(1),
               "C": _synth_metrics(2), "D": _synth_metrics(3)}
    text = _render(metrics)
    assert "ties resolve to the earliest of B, C, D in table order" in text


# ---------------------------------------------------------------------------
# (d) Verdict + caveat + probe presence
# ---------------------------------------------------------------------------


def test_verdict_and_caveat_lines_render():
    metrics = {"A": _synth_metrics(0), "B": _synth_metrics(1),
               "C": _synth_metrics(2), "D": _synth_metrics(3)}
    text = _render(metrics)
    assert "Best proposed model:" in text
    assert "Beats baseline (Model A): YES" in text
    assert "macro-F1 >= 0.90 target:" in text
    assert "Four: n=5 test rows" in text
    assert "24-usable < 30" in text
    assert "single seed 42 everywhere" in text


def test_target_not_met_renders_honestly():
    # All proposed models below the bar -> NOT MET, plainly.
    metrics = {"A": _synth_metrics(0), "B": _synth_metrics(1)}
    v = compute_verdict(metrics)
    assert v["target_met"] is False
    text = _render(metrics, verdict=v)
    assert "macro-F1 >= 0.90 target: NOT MET" in text


def test_beats_baseline_false_when_equal_macro_f1():
    a = _synth_metrics(0)
    b = _synth_metrics(1)
    b["f1_macro"] = a["f1_macro"]  # tie with the baseline is NOT a beat
    v = compute_verdict({"A": a, "B": b})
    assert v["beats_baseline"] is False
    text = _render({"A": a, "B": b}, verdict=v)
    assert "Beats baseline (Model A): NO" in text


# ---------------------------------------------------------------------------
# (d2) Partial --models runs: unmeasured comparisons render honest n/a (WR-02)
# ---------------------------------------------------------------------------


def test_partial_run_without_model_a_renders_na_never_fabricated_negative():
    # WR-02: a partial --models run (e.g. --models B — Model A not
    # evaluated) must NOT render "NO"/"NOT MET" for a comparison that was
    # never measured: the verdict stays None and the report says so
    # explicitly, naming the partial run.
    metrics = {"B": _synth_metrics(1)}
    v = compute_verdict(metrics)
    assert v["best_id"] == "B"
    assert v["baseline_f1"] is None
    assert v["beats_baseline"] is None
    assert v["target_met"] is None
    text = _render(metrics, verdict=v)
    assert (
        "Beats baseline (Model A): n/a — not measured in this run "
        "(partial --models run: Model A was not evaluated)" in text
    )
    assert "Beats baseline (Model A): NO" not in text
    assert "macro-F1 >= 0.90 target: NOT MET" not in text
    assert "macro-F1 >= 0.90 target: MET" not in text
    assert (
        "macro-F1 >= 0.90 target: n/a — not measured in this run" in text
    )


def test_partial_run_without_any_model_renders_na_for_every_line():
    # WR-02 counterpart: neither the baseline nor any proposed model was
    # evaluated -> every verdict line is the explicit unmeasured n/a.
    v = compute_verdict({})
    assert v["best_id"] is None
    assert v["beats_baseline"] is None
    assert v["target_met"] is None
    text = _render({}, verdict=v)
    assert (
        "Best proposed model: n/a (no proposed model (B/C/D) was "
        "evaluated this run)" in text
    )
    assert (
        "Beats baseline (Model A): n/a — not measured in this run "
        "(partial --models run: no proposed model (B/C/D) was evaluated "
        "either)" in text
    )
    assert "Beats baseline (Model A): NO" not in text


def test_probe_renders_in_its_own_section_never_in_the_table():
    metrics = {"A": _synth_metrics(0), "B": _synth_metrics(1)}
    probe = {"accuracy": 0.7100, "f1_macro": 0.6900}
    text = _render(metrics, probe=probe)
    assert "## Augmentation probe (Model A, no augmentation)" in text
    assert "unaugmented Model A: accuracy 0.7100 / macro-F1 0.6900" in text
    assert "augmented Model A (main table)" in text  # side-by-side comparison
    assert "RESEARCH A2/A3" in text  # the canary interpretation instruction
    assert "never a trigger to re-tune on test" in text
    # The probe NEVER enters the main table: rows stay exactly A, B.
    rows = [ln for ln in text.splitlines() if re.match(r"^\| [ABCD] \|", ln)]
    ids = [r.split("|")[1].strip() for r in rows]
    assert ids == ["A", "B"], ids


def test_probe_absent_renders_explicit_marker():
    text = _render({"A": _synth_metrics(0)}, probe=None)
    assert "not run this run" in text


def test_probe_canary_finding_renders_when_unaugmented_beats_augmented():
    # The flagged assumption (plan frontmatter): if the unaugmented probe
    # matches or beats augmented Model A, the report SAYS SO as a finding
    # on the flip/jitter assumptions — never a re-tune trigger.
    a = _synth_metrics(0)
    a["f1_macro"] = 0.7227
    probe = {"accuracy": 0.7857, "f1_macro": 0.7822}
    text = _render({"A": a}, probe=probe)
    assert "FINDING: the unaugmented probe MATCHES OR BEATS augmented" in text
    assert "macro-F1 0.7822 vs 0.7227" in text

    # Probe strictly below augmented -> no finding line.
    a2 = _synth_metrics(1)
    a2["f1_macro"] = 0.9
    text2 = _render({"A": a2}, probe={"accuracy": 0.5, "f1_macro": 0.4})
    assert "FINDING:" not in text2


# ---------------------------------------------------------------------------
# (e) Timestamp-free rendering
# ---------------------------------------------------------------------------


def test_report_renders_no_wall_clock_timestamps():
    metrics = {"A": _synth_metrics(0), "B": _synth_metrics(1),
               "C": _synth_metrics(2), "D": _synth_metrics(3)}
    text = _render(metrics)
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text), "wall-clock date leaked"


# ---------------------------------------------------------------------------
# Lineage: four canonical artifact sha256 lines
# ---------------------------------------------------------------------------


def test_artifact_sha256_lines_render_for_produced_artifacts(tmp_path):
    # WR-03 contract: the param block's sha256 provenance renders ONLY the
    # artifacts the train stage actually produced this run (full-run shape:
    # all four canonical TRAIN-08 files were produced and render in
    # canonical order).
    produced = ["baseline_ann.pt", "pose_mlp.pt", "ensemble_model.joblib",
                "label_encoder.joblib"]
    for name in produced:
        (tmp_path / name).write_bytes(b"artifact-bytes-" + name.encode())
    metrics = {"A": _synth_metrics(0), "B": _synth_metrics(1),
               "D": _synth_metrics(2)}
    text = _render(metrics, models_dir=tmp_path, produced_artifacts=produced)
    for art in ("baseline_ann", "pose_mlp", "ensemble_model", "label_encoder"):
        assert re.search(rf"^{art} sha256: [0-9a-f]{{64}}", text, re.M), art


def test_stale_artifact_renders_no_sha256_line(tmp_path):
    # WR-03: a stale leftover under models/ from an EARLIER run (here:
    # ensemble_model.joblib on a run that trained A/B but never D —
    # MODULE_ORDER trains C before D, so the bundle can only be stale at
    # C's print time) renders NO sha256 line rather than a lie, while the
    # artifacts this run DID produce still render.
    for name in ("baseline_ann.pt", "pose_mlp.pt", "ensemble_model.joblib"):
        (tmp_path / name).write_bytes(b"stale-or-produced-" + name.encode())
    metrics = {"A": _synth_metrics(0), "B": _synth_metrics(1)}
    text = _render(
        metrics,
        models_dir=tmp_path,
        produced_artifacts=["baseline_ann.pt", "pose_mlp.pt"],
    )
    assert re.search(r"^baseline_ann sha256: [0-9a-f]{64}", text, re.M)
    assert re.search(r"^pose_mlp sha256: [0-9a-f]{64}", text, re.M)
    assert "ensemble_model sha256:" not in text
    # Nothing recorded as produced -> no artifact sha256 line at all
    # (honest unknown, never a possibly-stale claim).
    text_none = _render(metrics, models_dir=tmp_path, produced_artifacts=None)
    art_lines = [
        ln for ln in text_none.splitlines()
        if re.match(
            r"^(baseline_ann|pose_mlp|ensemble_model|label_encoder) sha256:", ln
        )
    ]
    assert art_lines == []


# ---------------------------------------------------------------------------
# (f) classification_report.txt sections + the 2x2 confusion PNG
# ---------------------------------------------------------------------------


def test_classification_report_sections_in_fixed_order_with_markers():
    metrics = {"B": _synth_metrics(0), "A": _synth_metrics(1)}
    text = _classification_report_text(metrics, LE)
    order = [text.index(f"=== Model {mid}") for mid in ("A", "B", "C", "D")]
    assert order == sorted(order)
    for mid in ("C", "D"):
        section = text.split(f"=== Model {mid}")[1]
        assert "not trained yet" in section


def test_confusion_png_is_a_non_empty_two_by_two_grid(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    captured = {}
    original = plt.subplots

    def recording_subplots(*args, **kwargs):
        captured["args"] = args
        return original(*args, **kwargs)

    monkeypatch.setattr(plt, "subplots", recording_subplots)
    metrics = {"B": _synth_metrics(0), "A": _synth_metrics(1)}
    _write_confusion_png(tmp_path, metrics, LE)
    assert captured["args"][:2] == (2, 2)
    path = tmp_path / "confusion_matrix.png"
    assert path.exists() and path.stat().st_size > 1000
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# (g) run_train_stage bookkeeping seam (WR-03/WR-04): produced_artifacts +
#     artifact_sha256 actually LEAVE the train stage. Stub trainers — no
#     heavy training here. (The live full runner caught this seam KeyError-
#     ing after minutes of training; pinned so it can never regress.)
# ---------------------------------------------------------------------------


def test_run_train_stage_returns_produced_artifacts_and_sha256(tmp_path, monkeypatch):
    # The train stage must RETURN the produced-artifact list it recorded
    # (the eval stage threads it into the report writer — a dropped key is
    # a live KeyError after training) and record each produced artifact's
    # sha256 in the same-run meta (the WR-04 eval-side integrity gate
    # verifies against exactly this digest).
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "classes.yaml").write_text(
        "canonical_classes:\n"
        "  - Four\n  - Six\n  - Out\n  - Wide\n  - NoBall\n  - NoAction\n",
        encoding="utf-8",
    )
    (tmp_path / "configs" / "paths.yaml").write_text(
        "outputs:\n  manifest: data/processed/manifest.csv\n", encoding="utf-8"
    )
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "manifest.csv").write_text(
        "row_id\n", encoding="utf-8"
    )

    real_npz = (
        Path(__file__).resolve().parent.parent / "data" / "processed" / "features.npz"
    )
    if not real_npz.exists():
        pytest.skip("frozen data/processed/features.npz not present")

    def stub_trainer(X, y, le, models_dir, code_version, **kwargs):
        (Path(models_dir) / "stub_artifact.bin").write_bytes(b"stub-artifact-bytes")
        return {"stub": True}

    stub = types.ModuleType("stub_models_fix")
    stub.train_model_stub = stub_trainer
    monkeypatch.setitem(sys.modules, "stub_models_fix", stub)
    monkeypatch.setattr(
        ta,
        "MODEL_REGISTRY",
        {
            "S": {
                "name": "stub model (WR-03/WR-04 seam test)",
                "module": "stub_models_fix",
                "train_fn": "train_model_stub",
                "eval_fn": "eval_model_stub",
                "artifact": "stub_artifact.bin",
                "input": "raw",
            }
        },
    )
    # Non-"A" id -> the augmentation probe (real torch training) is skipped;
    # MODULE_ORDER/REPO_ROOT stubs keep every write inside tmp_path.
    monkeypatch.setattr(ta, "MODULE_ORDER", ("S",))
    monkeypatch.setattr(ta, "REPO_ROOT", tmp_path)

    result = ta.run_train_stage(real_npz, ["S"])
    assert result["produced_artifacts"] == [
        "label_encoder.joblib",
        "stub_artifact.bin",
    ]
    models_dir = tmp_path / "models"
    assert (models_dir / "stub_artifact.bin").exists()
    assert result["meta_by_model"]["S"]["artifact_sha256"] == sha256_of(
        models_dir / "stub_artifact.bin"
    )
