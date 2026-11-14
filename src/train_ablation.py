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
  evaluated model on the IDENTICAL held-out arrays; evaluates the
  same-run augmentation probe (Model A trained with augment_fn=None —
  its metrics render in the separated probe section, never in the main
  table); computes the best-proposed verdict over the fixed B, C, D scan
  (ties resolve to the earliest — the rule is stated in the report);
  writes
  reports/ablation_study.md (table + probe + verdict + caveats + param
  block), reports/confusion_matrix.png (ONE 2x2-subplot
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
# r3: Plan 03-03 behavior change — MODEL_REGISTRY gains D (features_eng
# full 59; the soft-voting ensemble of Model C probabilities +
# grid-tuned XGBoost + RandomForest): the train stage tunes both members
# via the locked seeded GridSearchCV protocol (train rows only) and
# persists the self-contained models/ensemble_model.joblib bundle (Model
# C's weights stored inside), so the guarded eval votes over the exact
# reported members; the param block gains the grid best-params/best-score
# lines and the ensemble bundle sha256.
# r4: review-fix behavior change (WR-02) — compute_verdict keeps an
# unmeasured baseline comparison None on partial --models runs (Model A or
# every proposed model not evaluated) so the verdict section renders the
# honest "n/a — not measured in this run (partial --models run)" line
# instead of a fabricated NO / NOT MET. Plus (WR-03) Model C's registry
# row carries "artifact": None (the bundle is Model D's write — no false
# "wrote" console line on partial runs) and the param block's artifact
# sha256 provenance renders only artifacts actually produced by THIS run.
CODE_VERSION = "phase3-plan01-r4"

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
        # WR-03: Model C has NO standalone artifact — its weights ride the
        # train-stage meta and are persisted INSIDE the ensemble bundle by
        # Model D's train stage. Because MODULE_ORDER trains C before D,
        # models/ensemble_model.joblib can only exist at C's print time as
        # a STALE leftover from a previous run — never a C write. The None
        # marker keeps the train-stage print honest (no false "wrote"
        # line) and keeps the bundle out of C's artifact provenance.
        "artifact": None,
        "input": "eng",
    },
    "D": {
        "name": "Model D — soft-voting ensemble: Model C + grid-tuned XGBoost + RandomForest (TRAIN-06)",
        "module": "src.models.ensemble",
        "train_fn": "train_model_d",
        "eval_fn": "eval_model_d",
        "artifact": "ensemble_model.joblib",
        "input": "eng",
        # The ensemble bundle stores Model C's weights, so D's train stage
        # consumes the same-run Model C handoff (train_model_d fails
        # closed without it — the vote members must be the exact reported
        # models). MODULE_ORDER trains C before D whenever both are
        # selected.
        "requires_prev_meta": "C",
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
# Verdict logic (TRAIN-07/adjacency: deterministic, stated tie-break)
# ---------------------------------------------------------------------------

# The fixed scan order for "best proposed model" — an explicit ordered
# loop, NEVER np.argmax over a reordered collection: equal macro-F1 ties
# resolve to the EARLIEST model in this tuple (strict > comparison keeps
# the earlier winner). The rule is stated verbatim in the report.
PROPOSED_ORDER = ("B", "C", "D")

# The honest pre-registered project success bar (PROJECT.md): MET requires
# beating the reproduced baseline AND reaching macro-F1 >= 0.90.
TARGET_MACRO_F1 = 0.90


def compute_verdict(metrics_by_model) -> dict:
    """Best-proposed verdict over the fixed PROPOSED_ORDER scan.

    Returns {"best_id", "best_f1", "baseline_f1", "beats_baseline",
    "target_met"}; None values mean the honest n/a renders (e.g. a partial
    --models run) — an UNMEASURED comparison stays None (beats_baseline /
    target_met are None whenever Model A, or every proposed model, was not
    evaluated this run), never a fabricated negative (WR-02).
    beats_baseline is a STRICT > (an exact tie with the baseline is not a
    beat); target_met = beats_baseline AND best_f1 >= TARGET_MACRO_F1 (the
    pre-registered bar combines both — reported as measured, never tuned
    toward)."""
    best_id = None
    best_f1 = None
    for model_id in PROPOSED_ORDER:
        m = metrics_by_model.get(model_id)
        if m is None:
            continue
        f1 = float(m["f1_macro"])
        if best_id is None or f1 > best_f1:
            best_id, best_f1 = model_id, f1
    baseline = metrics_by_model.get("A")
    baseline_f1 = float(baseline["f1_macro"]) if baseline is not None else None
    if best_id is None:
        beats_baseline = None
        target_met = None
    else:
        # WR-02: "Model A was not measured this run" is NOT "did not beat
        # the baseline" — an unmeasured comparison stays None so the
        # renderer prints the honest n/a line instead of a fabricated
        # NO / NOT MET (partial --models runs, e.g. --models B).
        beats_baseline = None if baseline_f1 is None else best_f1 > baseline_f1
        target_met = (
            None if beats_baseline is None
            else bool(beats_baseline) and best_f1 >= TARGET_MACRO_F1
        )
    return {
        "best_id": best_id,
        "best_f1": best_f1,
        "baseline_f1": baseline_f1,
        "beats_baseline": beats_baseline,
        "target_met": target_met,
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
    features_eng). Returns {"digest", "meta_by_model", "n_train",
    "produced_artifacts"} — the last carries the file names actually
    written by THIS run (shared encoder + each selected model's registry
    artifact), which the report's sha256 provenance is restricted to
    (WR-03: stale leftovers from earlier runs render no line). Each
    produced artifact's sha256 is also recorded in its model's meta as
    ``artifact_sha256`` (WR-04: the eval-stage reload integrity-gates
    against exactly the bytes this run wrote)."""
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

    # Artifacts actually produced by THIS run (WR-03): the shared encoder
    # is written unconditionally below; each selected model appends its
    # own artifact when its trainer just wrote it. The report's sha256
    # provenance renders ONLY these — a stale leftover under models/ from
    # an earlier run renders no line rather than a lie.
    produced_artifacts: list = []
    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = models_dir / "label_encoder.joblib"
    joblib.dump({"encoder": le, "source": "configs/classes.yaml"}, encoder_path)
    produced_artifacts.append("label_encoder.joblib")
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
        train_kwargs = {}
        if spec.get("requires_prev_meta"):
            # e.g. Model D consumes the same-run Model C handoff (the
            # ensemble bundle stores C's weights) — train_model_d fails
            # closed when the dependency is missing.
            train_kwargs["prev_meta"] = meta_by_model
        meta_by_model[model_id] = trainer(
            X_model, y_int, le, models_dir, CODE_VERSION, **train_kwargs
        )
        # WR-03 bookkeeping: only a registry row WITH an artifact name can
        # have written one just now — each train fn saves before returning,
        # so a file existing at this moment was produced by THIS run. A
        # stale leftover can never be claimed: Model C's row carries
        # "artifact": None, so the D-owned bundle is never reported as a
        # C write.
        artifact_name = spec.get("artifact")
        if artifact_name is not None and (models_dir / artifact_name).exists():
            produced_artifacts.append(artifact_name)
            # WR-04: record the artifact's sha256 in the same-run meta so
            # the eval-stage reload integrity-gates against exactly the
            # bytes this run wrote (the report param block records the
            # same digest — the Phase 4 reload recipe verifies against it).
            meta_by_model[model_id]["artifact_sha256"] = sha256_of(
                models_dir / artifact_name
            )
            print(
                f"[ablation] wrote models/{artifact_name} "
                f"(Model {model_id} trained)"
            )
        elif model_id == "C":
            deferred = (
                "written later in this run by the Model D train stage"
                if "D" in selected
                else "deferred to the Model D stage — NOT written by this "
                     "run (Model D was not selected)"
            )
            print(
                f"[ablation] Model C trained — no standalone artifact (its "
                "weights ride the train-stage meta and are handed to the "
                "eval stage in-memory; models/ensemble_model.joblib is "
                f"{deferred})"
            )
        else:
            print(
                f"[ablation] Model {model_id} trained — no standalone "
                "artifact (weights handed to the eval stage in-memory)"
            )

    # Augmentation probe (CONTEXT-locked ONE cheap probe): Model A through
    # the SAME train_baseline path with augmentation disabled — identical
    # seeded folds, early stopping, and final-epochs derivation, isolating
    # the augmentation effect WITHOUT touching the main-table Model A
    # (which already trained above). The probe trains after every main
    # model and its metrics never enter the main table.
    if "A" in selected:
        from src.models.baseline_ann import train_baseline

        X_raw = np.asarray(arrays["features_raw"])[train_idx]
        probe_model, probe_scaler, probe_meta = train_baseline(
            X_raw, y_int, le, augment=False
        )
        meta_by_model["_probe_A"] = {
            "model": probe_model,
            "scaler": probe_scaler,
            "meta": probe_meta,
        }
        print(
            "[ablation] augmentation probe trained — Model A WITHOUT "
            "augmentation (same train_baseline path, augment_fn=None; "
            "metrics render in the separated probe section, never in the "
            "main table)"
        )
    return {
        "digest": digest,
        "meta_by_model": meta_by_model,
        "n_train": n_train,
        "produced_artifacts": produced_artifacts,
    }


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
                        n_train: int, models_dir=None, reports_dir=None,
                        produced_artifacts=None) -> dict:
    """The guarded eval stage: assert the digest, load the test slice
    exactly ONCE, evaluate every selected registered model on the
    IDENTICAL held-out arrays in fixed MODULE_ORDER, write the three
    report artifacts, print everything (INGEST-07).

    ``produced_artifacts`` (WR-03) threads the train stage's list of file
    names actually written by THIS run into the report writer — the param
    block's artifact sha256 provenance renders ONLY those (a stale
    leftover under models/ from an earlier run renders no line rather
    than a lie). None (no recorded provenance) renders no artifact lines.

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

    # Augmentation probe eval — inside the SAME guarded stage (the only
    # path to reported metrics), on the identical test arrays the main
    # models were just measured on.
    probe_metrics = None
    if "_probe_A" in meta_by_model:
        from src.models.baseline_ann import predict_proba

        probe = meta_by_model["_probe_A"]
        X_test_raw = np.asarray(arrays["features_raw"])[test_idx]
        probe_proba = predict_proba(probe["model"], probe["scaler"].transform(X_test_raw))
        probe_y_pred = probe_proba.argmax(axis=1)
        probe_metrics = {
            "accuracy": float(accuracy_score(y_test, probe_y_pred)),
            "f1_macro": float(f1_score(y_test, probe_y_pred, average="macro")),
            "f1_per_class": [
                float(v)
                for v in f1_score(y_test, probe_y_pred, average=None, labels=list(range(6)))
            ],
            "roc_auc_ovr_macro": float(
                roc_auc_score(
                    y_test, probe_proba, multi_class="ovr", average="macro",
                    labels=list(range(6)),
                )
            ),
            "cm": confusion_matrix(y_test, probe_y_pred, labels=list(range(6))),
            "classification_report": classification_report(
                y_test,
                probe_y_pred,
                labels=list(range(6)),
                target_names=list(le.classes_),
                digits=4,
            ),
        }
        print(
            f"[ablation] Augmentation probe (Model A, no augmentation) "
            f"evaluated — accuracy {probe_metrics['accuracy']:.4f}, "
            f"macro-F1 {probe_metrics['f1_macro']:.4f} (renders in the "
            "separated probe section, never in the main table)"
        )

    verdict = compute_verdict(metrics_by_model)
    manifest_path = REPO_ROOT / str(_paths_cfg()["outputs"]["manifest"])
    lineage = {
        "features_sha256": sha256_of(npz_path),
        "manifest_sha256": sha256_of(manifest_path),
        "digest": expected_digest,
        "num_threads": _torch_num_threads(),
    }
    report_text = _ablation_report_text(
        census, metrics_by_model, meta_by_model, probe_metrics, verdict,
        n_train, le, lineage, models_dir, produced_artifacts,
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


def _ablation_report_text(census, metrics_by_model, meta_by_model,
                          probe_metrics, verdict, n_train, le, lineage,
                          models_dir, produced_artifacts=None) -> str:
    """reports/ablation_study.md — provenance lineage, counts +
    augmentation census, the ablation table (fixed A-D order), the
    separated augmentation-probe section, the plainly-stated verdict
    lines (best proposed + stated tie-break rule, beats-baseline, 0.90
    target), the caveat block (Four n=5 + single seed), and the completed
    param block (seed, threads, versions, sha256 lineage incl. the
    artifacts actually produced by THIS run, CV + grid hyperparameters,
    code version). NO wall-clock timestamps, NO datetime calls —
    byte-identical re-runs at the param-text level (artifact sha256s
    reflect artifact bytes, the features_build.md posture).
    ``lineage`` carries the precomputed features.npz/manifest sha256s,
    the test-slice digest, and the torch thread count so this writer
    stays pure (weight-free testable). ``produced_artifacts`` (WR-03)
    carries the train stage's file names written by THIS run — the
    artifact sha256 lines render ONLY for those (a stale leftover under
    models/ from an earlier run renders no line rather than a lie);
    None renders no artifact lines."""
    models_dir = Path(models_dir) if models_dir is not None else Path(".")
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
    lines.append("## Augmentation probe (Model A, no augmentation)")
    lines.append("")
    if probe_metrics is None:
        lines.append("not run this run (the probe requires Model A's selection)")
    else:
        lines.append(
            f"unaugmented Model A: accuracy {probe_metrics['accuracy']:.4f} / "
            f"macro-F1 {probe_metrics['f1_macro']:.4f}"
        )
        baseline_metrics = metrics_by_model.get("A")
        if baseline_metrics is not None:
            lines.append(
                f"augmented Model A (main table): accuracy "
                f"{baseline_metrics['accuracy']:.4f} / macro-F1 "
                f"{baseline_metrics['f1_macro']:.4f}"
            )
        lines.append(
            "Interpretation: this probe is the empirical canary for the "
            "swap-only mirror flip and the model-input-space jitter "
            "(RESEARCH A2/A3) — if the unaugmented row matches or beats "
            "the augmented row beyond noise, that is a reported FINDING "
            "on the flip/jitter assumptions, never a trigger to re-tune "
            "on test (pre-registered discipline). The probe never enters "
            "the main table."
        )
        if (
            baseline_metrics is not None
            and probe_metrics["f1_macro"] >= baseline_metrics["f1_macro"]
        ):
            lines.append(
                "FINDING: the unaugmented probe MATCHES OR BEATS augmented "
                "Model A on this split (macro-F1 "
                f"{probe_metrics['f1_macro']:.4f} vs "
                f"{baseline_metrics['f1_macro']:.4f}) — the canary flags "
                "the swap-only flip and/or model-input-space jitter "
                "assumptions (RESEARCH A2/A3); reported as measured, "
                "never re-tuned on test."
            )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if verdict["best_id"] is None:
        lines.append(
            "Best proposed model: n/a (no proposed model (B/C/D) was "
            "evaluated this run)"
        )
    else:
        lines.append(
            f"Best proposed model: {verdict['best_id']} "
            f"(macro-F1 {verdict['best_f1']:.4f})"
        )
    lines.append(
        "Tie-break rule: ties resolve to the earliest of B, C, D in table "
        "order (explicit ordered scan over the fixed tuple — never "
        "np.argmax over a reordered collection)."
    )
    # WR-02: an unmeasured comparison renders as an explicit "n/a — not
    # measured in this run" line naming the partial --models run — never a
    # fabricated NO / NOT MET for a model that was never evaluated.
    if verdict["beats_baseline"] is None:
        if verdict["best_id"] is None:
            lines.append(
                "Beats baseline (Model A): n/a — not measured in this run "
                "(partial --models run: no proposed model (B/C/D) was "
                "evaluated either)"
            )
        else:
            lines.append(
                "Beats baseline (Model A): n/a — not measured in this run "
                "(partial --models run: Model A was not evaluated)"
            )
    else:
        lines.append(
            "Beats baseline (Model A): "
            + ("YES" if verdict["beats_baseline"] else "NO")
        )
    if verdict["target_met"] is None:
        if verdict["best_id"] is None:
            lines.append(
                "macro-F1 >= 0.90 target: n/a — not measured in this run "
                "(partial --models run: no proposed model (B/C/D) was "
                "evaluated)"
            )
        else:
            lines.append(
                "macro-F1 >= 0.90 target: n/a — not measured in this run "
                "(partial --models run: Model A was not evaluated, so the "
                "beats-baseline conjunct is unmeasured)"
            )
    else:
        lines.append(
            "macro-F1 >= 0.90 target: "
            + ("MET" if verdict["target_met"] else "NOT MET")
        )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "Four: n=5 test rows; the 24-usable < 30 flag is carried from the "
        "Phase 1 audit; per-class F1 for Four is small-sample noisy and "
        "never tuned on test."
    )
    lines.append(
        "single seed 42 everywhere (matches the locked split seed); no "
        "multi-seed mean+-std in this phase."
    )
    lines.append("")
    lines.append("## Parameter block")
    lines.append("")
    lines.append(f"seed: {SEED}")
    lines.append(f"torch num threads: {lineage['num_threads']}")
    versions = collect_versions()
    for pkg in VERSION_PACKAGES:
        lines.append(f"{pkg} version: {versions.get(pkg, 'unknown')}")
    lines.append(f"features.npz sha256: {lineage['features_sha256']}")
    lines.append(f"manifest sha256: {lineage['manifest_sha256']}")
    lines.append(f"test-slice digest: {lineage['digest']}")
    # The TRAIN-08 artifact provenance — gitignored runtime files; their
    # sha256s ARE the tracked provenance. WR-03: ONLY artifacts actually
    # produced by THIS run render (the train stage records them in
    # ``produced_artifacts``) — a stale leftover under models/ from an
    # earlier run (e.g. ensemble_model.joblib on a run that never trained
    # D) renders no line rather than a lie. Canonical TRAIN-08 order kept.
    produced = set(produced_artifacts or ())
    for art_name, file_name in (
        ("baseline_ann", "baseline_ann.pt"),
        ("pose_mlp", "pose_mlp.pt"),
        ("ensemble_model", "ensemble_model.joblib"),
        ("label_encoder", "label_encoder.joblib"),
    ):
        if file_name in produced:
            artifact = models_dir / file_name
            if artifact.exists():
                lines.append(
                    f"{art_name} sha256: {sha256_of(artifact)} (models/{file_name})"
                )
    for model_id in MODULE_ORDER:
        if model_id not in metrics_by_model:
            continue
        meta = meta_by_model.get(model_id, {})
        if not meta:
            continue
        if "fold_best_epochs" in meta:
            lines.append(
                f"Model {model_id} meta: fold_best_epochs "
                f"{meta.get('fold_best_epochs')}; final_epochs "
                f"{meta.get('final_epochs')}; batch_size "
                f"{meta.get('batch_size')}"
            )
            lines.append(
                f"Model {model_id} cv protocol: {meta.get('cv')} "
                f"(n_folds {meta.get('n_folds')}); max_epochs "
                f"{meta.get('max_epochs')}; early stop on val loss "
                f"(patience {meta.get('patience')}); final epochs = "
                "round(mean fold-best); augmentation inside each fold's "
                "train portion only"
            )
        if meta.get("scaler"):
            lines.append(f"Model {model_id} scaler: {meta['scaler']}")
        if meta.get("optimizer"):
            lines.append(f"Model {model_id} optimizer: {meta['optimizer']}")
        if meta.get("loss"):
            lines.append(f"Model {model_id} loss: {meta['loss']}")
        if meta.get("input"):
            lines.append(f"Model {model_id} input: {meta['input']}")
        if meta.get("members"):
            lines.append(
                f"Model {model_id} members: {meta['members']} — "
                f"{meta.get('voting')} vote = {meta.get('soft_vote')}"
            )
        if meta.get("augmentation"):
            lines.append(f"Model {model_id} augmentation: {meta['augmentation']}")
        if meta.get("grid_xgb_best_params") is not None:
            lines.append(
                f"Model {model_id} grid protocol: GridSearchCV(scoring="
                f"{meta.get('grid_scoring')}, cv={meta.get('grid_cv')}, "
                f"n_jobs={meta.get('grid_n_jobs', 1)}) — "
                f"{meta.get('grid_n_combos')} combos, tuned on the train "
                "slice only"
            )
            lines.append(
                f"Model {model_id} grid (XGBoost member): best params "
                f"{meta['grid_xgb_best_params']}; best cv f1_macro "
                f"{meta['grid_xgb_best_score']:.4f}"
            )
            lines.append(
                f"Model {model_id} grid (RandomForest member): best params "
                f"{meta['grid_rf_best_params']}; best cv f1_macro "
                f"{meta['grid_rf_best_score']:.4f}"
            )
        if meta.get("model_c_source"):
            lines.append(
                f"Model {model_id} model C source: {meta['model_c_source']}"
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
        produced_artifacts=trained["produced_artifacts"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
