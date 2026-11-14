#!/usr/bin/env python
"""Phase 3 ablation runner: frozen features.npz -> Model A-D training ->
ONE guarded evaluation entry point -> the three tracked report artifacts.

Stage contract (locked CONTEXT decisions — this docstring IS the contract):

- Train stage: seed_everything(42) FIRST; records lineage (features.npz
  file sha256, manifest sha256, test-slice digest) as printed lines; loads
  the train slice ONLY (no test array enters any training computation);
  fits the ONE shared LabelEncoder on configs/classes.yaml
  canonical_classes; trains each selected model through its
  MODEL_REGISTRY entry (epoch/schedule selection via 5-fold stratified
  train-CV on train rows — the test set is never loaded during selection;
  no evaluate-then-retune loop exists anywhere in this runner) and
  persists artifacts under models/ (gitignored runtime files; their
  sha256s go into the tracked report, never the binaries).

- Eval stage (evaluate_and_report) — the ONLY path to reported metrics:
  recomputes the test-slice digest and compares it to the recorded one;
  on drift raises SystemExit naming the drift BEFORE loading test rows or
  writing anything (fail-closed, never warn-and-continue — the
  build_features reconciliation-raise posture). Loads the test slice
  exactly ONCE per run; computes accuracy, macro-F1, per-class F1
  (LabelEncoder order), ROC-AUC ovr-macro, and the confusion matrix per
  evaluated model on the IDENTICAL held-out arrays; writes
  reports/ablation_study.md, reports/confusion_matrix.png (ONE 2x2-subplot
  grid — the final TRAIN-07 layout from day one), and
  reports/classification_report.txt — every report is built from ONE
  string, written with newline="\\n" after mkdir parents, and printed to
  console (INGEST-07 discipline). Deterministic: NO wall-clock timestamps,
  NO datetime calls anywhere.

Flags:
  --dry-run  full fail-closed npz contract validation (incl. the
             cross-split adjacency guard) + census print; exits WITHOUT
             importing torch or any model module and writes nothing.
  --models   comma-separated subset of MODEL_REGISTRY keys (e.g. "A,C")
             honored by BOTH the train and eval stages; default: all
             registered models, iterated in fixed MODULE_ORDER.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import sys
from collections import Counter
from pathlib import Path

# Import the data seam through the SAME 'src.models.data' spelling the test
# suite uses (conftest.py puts the checkout root on sys.path). data.py is
# torch-free at module level, so this runner --dry-run stays torch-free;
# the torch model modules are imported lazily INSIDE the train/eval stages
# via MODEL_REGISTRY module paths.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.models.data import (
    canonical_classes,
    fit_label_encoder,
    load_features_npz,
    seed_everything,
    split_masks,
    test_slice_digest,
)

# Static code version tag for the determinism parameter block (audit.py
# pattern: thresholds + tool versions + a code version — never a
# wall-clock timestamp, so re-runs stay byte-identical). Bump the
# r-suffix whenever runner behavior changes in later plans.
# r1: first Phase 3 runner — frozen-data seam, Model A (TRAIN-03 exact
# spec) through the shared 5-fold train-CV protocol, guarded eval stage,
# the three report artifacts (ablation table / 2x2 CM grid / per-model
# classification sections), MODEL_REGISTRY with only "A" registered.
# r2: Plan 03-02 behavior change — two more trained models: MODEL_REGISTRY
# gains B (features_eng[:, 0:51], AdamW + CosineAnnealingLR + label
# smoothing 0.05) and C (features_eng full 59, residual MLP with
# class-weighted CE); the runner selects each model's train input by its
# registry "input" kind and threads the train-stage meta to the eval
# entry points (Model C's weights are handed off in-memory — no standalone
# artifact until Plan 03-03's ensemble bundle).
CODE_VERSION = "phase3-plan01-r2"

SEED = 42

# Fixed model order for EVERY report surface (TRAIN-07/ordering: rows
# render in fixed A, B, C, D order; the writers iterate MODULE_ORDER and
# skip unregistered ids). Plan 03-01 registered only "A"; Plan 03-02 adds
# B/C; Plan 03-03 adds D.
MODULE_ORDER = ("A", "B", "C", "D")

MODEL_REGISTRY = {
    "A": {
        "name": "Model A — Mili et al. baseline reproduction (TRAIN-03)",
        "module": "src.models.baseline_ann",
        "train_fn": "train_model_a",
        "eval_fn": "eval_model_a",
        "artifact": "baseline_ann.pt",
        "input": "raw",
    },
    "B": {
        "name": "Model B — torso-normalized MLP, AdamW + CosineAnnealingLR + label smoothing 0.05 (TRAIN-04)",
        "module": "src.models.pose_mlp",
        "train_fn": "train_model_b",
        "eval_fn": "eval_model_b",
        "artifact": "pose_mlp.pt",
        "input": "eng51",
    },
    "C": {
        "name": "Model C — residual MLP on engineered features, class-weighted CE (TRAIN-05)",
        "module": "src.models.pose_mlp",
        "train_fn": "train_model_c",
        "eval_fn": "eval_model_c",
        "artifact": "ensemble_model.joblib",
        "input": "eng",
    },
}

# Packages recorded in the parameter block (IN-08 discipline: the
# determinism-relevant dependencies of the Phase 3 artifacts — exact pins
# live in requirements.txt, installed versions live here; a package not
# yet installed renders "unknown" rather than failing the run).
VERSION_PACKAGES = (
    "torch",
    "torchvision",
    "ultralytics",
    "scikit-learn",
    "numpy",
    "xgboost",
    "joblib",
    "matplotlib",
)


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
    dist metadata only, so this never imports torch/ultralytics/cv2
    (audit.py IN-08 pattern). PackageNotFoundError -> "unknown"."""
    versions = {}
    for pkg in VERSION_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover
            versions[pkg] = "unknown"
    return versions


def _torch_num_threads() -> int:
    """torch.get_num_threads() recorded for the parameter block (Pitfall 7:
    thread count is a determinism-relevant setting). Imported lazily so
    --dry-run and module import never need torch."""
    import torch

    return int(torch.get_num_threads())


def _paths_cfg() -> dict:
    """configs/paths.yaml (outputs.manifest is the Phase 2 lineage file
    hashed into the param block)."""
    return (
        yaml.safe_load((REPO_ROOT / "configs" / "paths.yaml").read_text(encoding="utf-8"))
        or {}
    )


def _parse_models_arg(raw) -> list:
    """--models parsing: comma-separated subset of MODEL_REGISTRY keys,
    honored by BOTH stages; default (empty/None) = all registered models.
    Unknown ids are a fail-closed SystemExit naming the offender."""
    if raw is None or not raw.strip():
        return [mid for mid in MODULE_ORDER if mid in MODEL_REGISTRY]
    selected = list(dict.fromkeys(tok.strip().upper() for tok in raw.split(",") if tok.strip()))
    if not selected:
        raise SystemExit(
            "[ablation] --models parsed to an empty selection — pass "
            "MODEL_REGISTRY keys like A or A,C"
        )
    unknown = [tok for tok in selected if tok not in MODEL_REGISTRY]
    if unknown:
        raise SystemExit(
            f"[ablation] unknown model id(s) {unknown} — MODEL_REGISTRY "
            f"holds {sorted(MODEL_REGISTRY)} (fixed MODULE_ORDER "
            f"{list(MODULE_ORDER)})"
        )
    return selected


def _census(arrays, le) -> dict:
    """Split + per-class census in LabelEncoder order (shared by --dry-run
    and the report counts section — deterministic from the frozen arrays)."""
    train_idx, test_idx = split_masks(arrays["split"])
    per_class = {}
    for label in le.classes_:
        is_label = arrays["labels"] == label
        per_class[label] = (
            int((is_label & (arrays["split"] == "train")).sum()),
            int((is_label & (arrays["split"] == "test")).sum()),
        )
    return {
        "total": int(len(arrays["labels"])),
        "train": int(train_idx.size),
        "test": int(test_idx.size),
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# --dry-run: validate the frozen contract + print the census, torch-free
# ---------------------------------------------------------------------------


def _dry_run(npz_path) -> int:
    """Full fail-closed npz validation (loader includes the cross-split
    adjacency guard) + census print; writes nothing and NEVER imports
    torch or any model module."""
    arrays = load_features_npz(npz_path)
    le = fit_label_encoder(canonical_classes())
    census = _census(arrays, le)
    lines = []
    lines.append(
        "[dry-run] frozen features.npz contract validation (fail-closed "
        "loader incl. cross-split adjacency guard):"
    )
    lines.append(f"  total rows: {census['total']}")
    lines.append(f"  train rows: {census['train']}")
    lines.append(f"  test rows: {census['test']}")
    lines.append(
        f"  features_raw shape: {tuple(arrays['features_raw'].shape)} "
        f"{arrays['features_raw'].dtype}"
    )
    lines.append(
        f"  features_eng shape: {tuple(arrays['features_eng'].shape)} "
        f"{arrays['features_eng'].dtype}"
    )
    lines.append("  per-class train/test counts (LabelEncoder order):")
    for label in le.classes_:
        train_n, test_n = census["per_class"][label]
        lines.append(f"    {label}: {train_n}/{test_n}")
    lines.append(f"  features.npz sha256: {sha256_of(npz_path)}")
    lines.append(f"  test-slice digest: {test_slice_digest(npz_path)}")
    lines.append(
        "[dry-run] writes disabled — no model loaded, nothing written "
        "(torch never imported)"
    )
    print("\n".join(lines))
    return 0


# ---------------------------------------------------------------------------
# Train stage: train slice ONLY, lineage recorded, artifacts persisted
# ---------------------------------------------------------------------------


def _model_train_input(spec, arrays, train_idx) -> np.ndarray:
    """Per-model train input by the registry "input" kind (Plan 03-02):
    "raw" -> features_raw (Model A, min-max-scaled inside its train fn),
    "eng51" -> features_eng[:, 0:51] (Model B — the 17 x (torso-normalized
    x, y, conf) block per the Phase 2 engineered-feature contract), "eng"
    -> features_eng full 59 (Model C/D). An unknown kind is a fail-closed
    SystemExit naming the offender (V5 posture)."""
    kind = spec.get("input", "raw")
    eng = np.asarray(arrays["features_eng"])[train_idx]
    if kind == "raw":
        return np.asarray(arrays["features_raw"])[train_idx]
    if kind == "eng51":
        return eng[:, :51]
    if kind == "eng":
        return eng
    raise SystemExit(
        f"[ablation] MODEL_REGISTRY entry with unknown input kind {kind!r} "
        "(expected raw | eng51 | eng) — refusing to train on an "
        "unresolved feature slice (fail-closed)"
    )


def run_train_stage(npz_path, selected) -> dict:
    """Train each selected registered model on the frozen train slice.

    seed_everything(42) runs first; lineage (features.npz sha256, manifest
    sha256, test-slice digest) is recorded as printed lines BEFORE any
    training; the shared LabelEncoder is fit once on
    configs/classes.yaml canonical_classes and persisted as
    models/label_encoder.joblib ({"encoder", "source"} per RESEARCH
    Pattern 4). Each model's train input is resolved by its registry
    "input" kind (Model A: features_raw; B: features_eng[:, 0:51]; C:
    features_eng). Returns {"digest", "meta_by_model", "n_train"}."""
    seed_everything(SEED)
    digest = test_slice_digest(npz_path)
    manifest_path = REPO_ROOT / str(_paths_cfg()["outputs"]["manifest"])
    print(f"[ablation] lineage — features.npz sha256: {sha256_of(npz_path)}")
    print(
        f"[ablation] lineage — manifest sha256: {sha256_of(manifest_path)} "
        "(reports/features_build.md phase2-plan01-r2)"
    )
    print(f"[ablation] lineage — test-slice digest: {digest}")
    print(f"[ablation] seed_everything({SEED}); torch num threads set to 1")

    arrays = load_features_npz(npz_path)
    train_idx, _test_idx = split_masks(arrays["split"])
    le = fit_label_encoder(canonical_classes())
    y_int = le.transform(arrays["labels"][train_idx])
    n_train = int(train_idx.size)

    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = models_dir / "label_encoder.joblib"
    joblib.dump({"encoder": le, "source": "configs/classes.yaml"}, encoder_path)
    print(
        f"[ablation] wrote models/label_encoder.joblib (shared encoder, "
        "source configs/classes.yaml)"
    )

    meta_by_model: dict = {}
    for model_id in MODULE_ORDER:
        if model_id not in selected:
            continue
        spec = MODEL_REGISTRY[model_id]
        module = importlib.import_module(spec["module"])
        trainer = getattr(module, spec["train_fn"])
        X_model = _model_train_input(spec, arrays, train_idx)
        meta_by_model[model_id] = trainer(X_model, y_int, le, models_dir, CODE_VERSION)
        artifact_path = models_dir / spec["artifact"]
        if artifact_path.exists():
            print(
                f"[ablation] wrote models/{spec['artifact']} "
                f"(Model {model_id} trained)"
            )
        else:
            print(
                f"[ablation] Model {model_id} trained — no standalone "
                f"artifact ({spec['artifact']} is built by a later plan; "
                "weights handed to the eval stage in-memory)"
            )
    return {"digest": digest, "meta_by_model": meta_by_model, "n_train": n_train}


# ---------------------------------------------------------------------------
# Guarded eval stage — the ONLY path to reported metrics
# ---------------------------------------------------------------------------


def assert_test_slice_unchanged(npz_path, expected_digest: str) -> None:
    """CONTEXT-locked frozen-test guard (T-03-01): recompute the test-slice
    digest and refuse (SystemExit naming the drift) BEFORE any metric is
    computed or written — never warn-and-continue (build_features
    reconciliation-raise posture)."""
    current = test_slice_digest(npz_path)
    if current != expected_digest:
        raise SystemExit(
            f"[ablation] test-slice digest drift: recorded {expected_digest} "
            f"vs current {current} — the test population changed between "
            "training and evaluation; refusing to write metrics against a "
            "drifted test slice (fail-closed, never warn-and-continue)"
        )


def evaluate_and_report(npz_path, expected_digest: str, selected, meta_by_model,
                        n_train: int, models_dir=None, reports_dir=None) -> dict:
    """The guarded eval stage: assert the digest, load the test slice
    exactly ONCE, evaluate every selected registered model on the
    IDENTICAL held-out arrays in fixed MODULE_ORDER, write the three
    report artifacts, print everything (INGEST-07).

    Predictions come from proba.argmax(axis=1) — numpy's deterministic
    first-index tie-break (TRAIN-03/precision backstop); the eval never
    reorders proba columns away from the shared encoder order. The
    train-stage meta is threaded to every evaluator (Plan 03-02: Model C
    rebuilds from the in-memory model_c_state handoff; A/B reload their
    artifacts from disk and ignore it). Models not selected/registered
    are skipped (writers iterate MODULE_ORDER), so report rows always
    render in fixed A, B, C, D order."""
    assert_test_slice_unchanged(npz_path, expected_digest)
    models_dir = Path(models_dir) if models_dir is not None else REPO_ROOT / "models"
    reports_dir = Path(reports_dir) if reports_dir is not None else REPO_ROOT / "reports"

    arrays = load_features_npz(npz_path)  # test slice loaded exactly ONCE per run
    _train_idx, test_idx = split_masks(arrays["split"])
    le = fit_label_encoder(canonical_classes())
    y_test = le.transform(arrays["labels"][test_idx])

    metrics_by_model: dict = {}
    for model_id in MODULE_ORDER:
        if model_id not in selected or model_id not in MODEL_REGISTRY:
            continue
        spec = MODEL_REGISTRY[model_id]
        module = importlib.import_module(spec["module"])
        evaluator = getattr(module, spec["eval_fn"])
        proba = evaluator(arrays, test_idx, le, models_dir, meta_by_model.get(model_id))
        y_pred = proba.argmax(axis=1)
        metrics_by_model[model_id] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "f1_per_class": [
                float(v)
                for v in f1_score(y_test, y_pred, average=None, labels=list(range(6)))
            ],
            "roc_auc_ovr_macro": float(
                roc_auc_score(
                    y_test, proba, multi_class="ovr", average="macro",
                    labels=list(range(6)),
                )
            ),
            "cm": confusion_matrix(y_test, y_pred, labels=list(range(6))),
            "classification_report": classification_report(
                y_test,
                y_pred,
                labels=list(range(6)),
                target_names=list(le.classes_),
                digits=4,
            ),
        }
        print(
            f"[ablation] Model {model_id} evaluated — accuracy "
            f"{metrics_by_model[model_id]['accuracy']:.4f}, macro-F1 "
            f"{metrics_by_model[model_id]['f1_macro']:.4f}, ROC-AUC "
            f"{metrics_by_model[model_id]['roc_auc_ovr_macro']:.4f}"
        )

    census = _census(arrays, le)
    report_text = _ablation_report_text(
        npz_path, census, metrics_by_model, meta_by_model, n_train, le, models_dir
    )
    report_path = reports_dir / "ablation_study.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8", newline="\n")
    print(report_text)
    print("[ablation] wrote reports/ablation_study.md")

    _write_confusion_png(reports_dir, metrics_by_model, le)
    print(
        f"[ablation] wrote reports/confusion_matrix.png (2x2 grid, "
        f"{len(metrics_by_model)}/{len(MODULE_ORDER)} subplots filled)"
    )
    cls_text = _classification_report_text(metrics_by_model, le)
    cls_path = reports_dir / "classification_report.txt"
    cls_path.parent.mkdir(parents=True, exist_ok=True)
    cls_path.write_text(cls_text, encoding="utf-8", newline="\n")
    print(cls_text)
    print("[ablation] wrote reports/classification_report.txt")
    return metrics_by_model


# ---------------------------------------------------------------------------
# Report writers (generated — never hand-edited; one string each)
# ---------------------------------------------------------------------------


def _ablation_report_text(npz_path, census, metrics_by_model, meta_by_model,
                          n_train, le, models_dir) -> str:
    """reports/ablation_study.md — provenance lineage, counts +
    augmentation census, the ablation table (fixed A-D order), and the
    param block (seed, threads, versions, sha256 lineage, CV
    hyperparameters, scaler semantics, code version). NO wall-clock
    timestamps, NO datetime calls — byte-identical re-runs at the
    param-text level (artifact sha256s reflect artifact bytes, the
    features_build.md posture)."""
    lines: list = []
    lines.append("# Phase 3 ablation study report")
    lines.append("")
    lines.append(
        "Provenance for the Phase 3 ablation over the frozen "
        "data/processed/features.npz (code version lineage: "
        "reports/features_build.md phase2-plan01-r2, "
        "reports/dedup_audit.md phase1-plan03-r2). Deterministic: no "
        "wall-clock timestamps; single guarded evaluation entry point; "
        "the test-slice digest is recorded at training start and asserted "
        "unchanged before any metric write. The split column is consumed "
        "as data — never re-split."
    )
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"total ok rows: {census['total']}")
    lines.append(
        f"split census (frozen split column): train {census['train']} / "
        f"test {census['test']}"
    )
    lines.append("Per-class train/test counts (LabelEncoder order):")
    for label in le.classes_:
        train_n, test_n = census["per_class"][label]
        lines.append(f"  {label}: {train_n}/{test_n}")
    lines.append(
        f"augmentation: mirror flip (x2) + jitter N(0, 0.015) in "
        f"model-input space — {n_train} -> {2 * n_train} effective rows "
        "(inside each CV fold's train portion; the test slice is never "
        "augmented and the scaler is never refit on it)"
    )
    lines.append(
        "Caveat: Four per-class F1 is computed on n=5 test rows (Phase 1 "
        "audit flag: 24 usable < 30) — small-sample noisy by "
        "construction; never re-tune on test to move it (pre-registered "
        "discipline)."
    )
    lines.append("")
    lines.append("## Ablation table")
    lines.append("")
    lines.append(
        "Rows render in fixed A, B, C, D order; unregistered models are "
        "skipped (writers iterate MODULE_ORDER). Per-class F1 columns "
        "follow the LabelEncoder order."
    )
    lines.append("")
    header = (
        "| model | accuracy | macro-F1 | "
        + " | ".join(f"F1 {label}" for label in le.classes_)
        + " | ROC-AUC ovr-macro |"
    )
    lines.append(header)
    lines.append(
        "| " + " | ".join(["-" * len(col) for col in
                           ["model", "accuracy", "macro-F1"]
                           + [f"F1 {label}" for label in le.classes_]
                           + ["ROC-AUC ovr-macro"]]) + " |"
    )
    for model_id in MODULE_ORDER:
        if model_id not in metrics_by_model:
            continue  # unregistered/untrained this run — row skipped
        m = metrics_by_model[model_id]
        cells = (
            [model_id, f"{m['accuracy']:.4f}", f"{m['f1_macro']:.4f}"]
            + [f"{v:.4f}" for v in m["f1_per_class"]]
            + [f"{m['roc_auc_ovr_macro']:.4f}"]
        )
        lines.append("| " + " | ".join(cells) + " |")
    trained = ", ".join(f"Model {mid}" for mid in MODULE_ORDER if mid in metrics_by_model)
    lines.append("")
    lines.append(f"Trained and evaluated this run: {trained}.")
    lines.append("")
    lines.append("## Parameter block")
    lines.append("")
    lines.append(f"seed: {SEED}")
    lines.append(f"torch num threads: {_torch_num_threads()}")
    for pkg in VERSION_PACKAGES:
        lines.append(f"{pkg} version: {collect_versions().get(pkg, 'unknown')}")
    lines.append(f"features.npz sha256: {sha256_of(npz_path)}")
    manifest_path = REPO_ROOT / str(_paths_cfg()["outputs"]["manifest"])
    lines.append(f"manifest sha256: {sha256_of(manifest_path)}")
    lines.append(f"test-slice digest: {test_slice_digest(npz_path)}")
    encoder_path = models_dir / "label_encoder.joblib"
    if encoder_path.exists():
        lines.append(f"models/label_encoder.joblib sha256: {sha256_of(encoder_path)}")
    for model_id in MODULE_ORDER:
        if model_id not in metrics_by_model:
            continue
        artifact = models_dir / MODEL_REGISTRY[model_id]["artifact"]
        if artifact.exists():
            lines.append(
                f"models/{MODEL_REGISTRY[model_id]['artifact']} sha256: "
                f"{sha256_of(artifact)}"
            )
        meta = meta_by_model.get(model_id, {})
        if meta:
            lines.append(
                f"Model {model_id} meta: fold_best_epochs "
                f"{meta.get('fold_best_epochs', 'n/a')}; final_epochs "
                f"{meta.get('final_epochs', 'n/a')}; batch_size "
                f"{meta.get('batch_size', 'n/a')}"
            )
            lines.append(
                f"Model {model_id} cv protocol: {meta.get('cv', 'n/a')} "
                f"(n_folds {meta.get('n_folds', 'n/a')}); max_epochs "
                f"{meta.get('max_epochs', 'n/a')}; early stop on val loss "
                f"(patience {meta.get('patience', 'n/a')}); final epochs = "
                "round(mean fold-best); augmentation inside each fold's "
                "train portion only"
            )
            lines.append(f"Model {model_id} scaler: {meta.get('scaler', 'n/a')}")
            if meta.get("optimizer"):
                lines.append(f"Model {model_id} optimizer: {meta['optimizer']}")
            if meta.get("loss"):
                lines.append(f"Model {model_id} loss: {meta['loss']}")
            if meta.get("input"):
                lines.append(f"Model {model_id} input: {meta['input']}")
    lines.append(
        "single-seed caveat: every number above is one seed-42 run "
        "(multi-seed mean±std deferred per CONTEXT)"
    )
    lines.append(f"code version: {CODE_VERSION}")
    return "\n".join(lines) + "\n"


def _write_confusion_png(reports_dir, metrics_by_model, le) -> None:
    """reports/confusion_matrix.png — ONE tracked PNG: a 2x2-subplot grid,
    one subplot per model in MODULE_ORDER (the final TRAIN-07 layout from
    day one); untrained subplots annotated 'not trained yet'. Axis ticks
    are labeled with le.classes_ (the shared-encoder order)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(
        "Phase 3 confusion matrices — rows/cols in LabelEncoder order "
        "(test slice, digest-guarded)"
    )
    for ax, model_id in zip(axes.ravel(), MODULE_ORDER):
        if model_id in metrics_by_model:
            disp = ConfusionMatrixDisplay(
                metrics_by_model[model_id]["cm"], display_labels=list(le.classes_)
            )
            disp.plot(ax=ax, colorbar=False, values_format="d")
            ax.set_title(f"Model {model_id} — {MODEL_REGISTRY[model_id]['name']}")
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.5, 0.5, "not trained yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
            )
            ax.set_title(f"Model {model_id}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = reports_dir / "confusion_matrix.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _classification_report_text(metrics_by_model, le) -> str:
    """reports/classification_report.txt — one sklearn
    classification_report section per evaluated model, target_names in
    le.classes_ order (fixed A-D iteration; untrained models carry an
    explicit not-trained marker so the file layout is stable across
    plans)."""
    lines: list = []
    lines.append(
        "Phase 3 classification reports — one section per evaluated model; "
        "target order (LabelEncoder): "
        + ", ".join(le.classes_)
    )
    lines.append("")
    for model_id in MODULE_ORDER:
        name = MODEL_REGISTRY.get(model_id, {}).get(
            "name", f"Model {model_id} (registered by a later plan)"
        )
        lines.append(f"=== {name} ===")
        lines.append("")
        if model_id in metrics_by_model:
            lines.append(metrics_by_model[model_id]["classification_report"].rstrip("\n"))
        else:
            lines.append("not trained yet (registered by a later plan)")
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="train_ablation",
        description=(
            "Phase 3 ablation runner: train the registered models on the "
            "frozen features.npz train slice and evaluate them through the "
            "single digest-guarded eval entry point, writing the three "
            "tracked report artifacts."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="frozen npz contract validation + census only; no torch, no "
        "model import, no writes",
    )
    ap.add_argument(
        "--models",
        default=None,
        help="comma-separated subset of MODEL_REGISTRY keys (e.g. A,C); "
        "default: all registered models",
    )
    args = ap.parse_args(argv)

    npz_path = REPO_ROOT / "data" / "processed" / "features.npz"
    if args.dry_run:
        return _dry_run(npz_path)

    selected = _parse_models_arg(args.models)
    trained = run_train_stage(npz_path, selected)
    evaluate_and_report(
        npz_path,
        trained["digest"],
        selected,
        trained["meta_by_model"],
        trained["n_train"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
