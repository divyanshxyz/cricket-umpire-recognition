"""Inference config pins — merged single source (IN-02), fail-closed every key.

Pinned contracts, one test each:
1. Merge posture: a minimal yaml missing most keys yields INFERENCE_DEFAULTS
   values for them; yaml overrides win where present.
2. P2RDY-02 boundary: an unknown ``model:`` value fails closed with a
   ValueError naming the OBSERVED value — only image | temporal are valid,
   and the enum is case-sensitive.
3. Fire-gate bypass prohibition (T-04-01): threshold/fire_frac must lie in
   (0.0, 1.0] — 0 or negative cannot disable the gate.
4. Domain rejects: alpha in (0.0, 1.0] (WR-02: 0 would freeze the EMA);
   cooldown_seconds/fps_fallback > 0;
   window/stride/max_frames ints >= 1 (bool is not an int); device
   non-empty — every reject names the offending key.
5. Missing config file -> ValueError naming the path (never a bare
   FileNotFoundError, never a silent default).
6. WR-04 digest shape: the REAL configs/model_digests.yaml loads to exactly
   5 entries (the four Phase 3 digests + the ADOPTED pose_weights digest)
   with models/ paths and 64-char lowercase hex sha256 values; malformed
   digests fail closed naming the offending entry.
"""

import re
from pathlib import Path

import pytest

from src.inference.config import (
    INFERENCE_DEFAULTS,
    load_inference_config,
    load_model_digests,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _write_yaml(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_defaults_merge_posture_minimal_yaml(tmp_path):
    p = _write_yaml(tmp_path, "inference.yaml", "params:\n  alpha: 0.5\n")
    cfg = load_inference_config(p)
    assert cfg["alpha"] == 0.5
    for key, value in INFERENCE_DEFAULTS.items():
        if key == "alpha":
            continue
        assert cfg[key] == value, key


def test_yaml_overrides_win(tmp_path):
    p = _write_yaml(tmp_path, "inference.yaml", "params:\n  threshold: 0.80\n")
    cfg = load_inference_config(p)
    assert cfg["threshold"] == 0.80


def test_unknown_model_value_fails_closed_naming_observed_value(tmp_path):
    # P2RDY-02/boundary truth: only image | temporal are valid; the error
    # names the observed value and the enum is case-sensitive.
    for bad in ("video", "IMAGE"):
        p = _write_yaml(tmp_path, f"model_{bad}.yaml", f"params:\n  model: {bad}\n")
        with pytest.raises(ValueError, match=bad):
            load_inference_config(p)


@pytest.mark.parametrize(
    "key,value",
    [
        ("threshold", 0.0),
        ("threshold", -0.1),
        ("fire_frac", 0.0),
    ],
)
def test_fire_gate_bypass_prohibited(tmp_path, key, value):
    # T-04-01: config cannot disable the fire gate — 0/negative threshold or
    # fire_frac is rejected, not clamped.
    p = _write_yaml(tmp_path, "bypass.yaml", f"params:\n  {key}: {value}\n")
    with pytest.raises(ValueError, match=key):
        load_inference_config(p)


@pytest.mark.parametrize(
    "key,value",
    [
        ("alpha", 1.5),
        ("alpha", 0.0),
        ("cooldown_seconds", 0),
        ("window", 0),
        ("stride", 0),
        ("max_frames", 0),
        ("fps_fallback", -1),
    ],
)
def test_domain_rejects_name_the_key(tmp_path, key, value):
    p = _write_yaml(tmp_path, "domain.yaml", f"params:\n  {key}: {value}\n")
    with pytest.raises(ValueError, match=key):
        load_inference_config(p)


@pytest.mark.parametrize("value", [0.0, 0])
def test_alpha_zero_frozen_ema_rejected_t0401_posture(tmp_path, value):
    # WR-02: alpha 0 makes the EMA recurrence s_t = s_{t-1} — the smoothed
    # vector freezes at the FIRST frame and the gate then fires a stale
    # frozen class (or 0.0 confidence) forever, defeating the locked gate
    # semantics via a config value (the exact hazard class T-04-01 closes).
    # Same posture as threshold/fire_frac: rejected, not clamped, naming
    # the key + observed value and the freeze rationale.
    p = _write_yaml(tmp_path, "alpha0.yaml", f"params:\n  alpha: {value}\n")
    with pytest.raises(ValueError) as exc:
        load_inference_config(p)
    assert "alpha" in str(exc.value)
    assert "(0.0, 1.0]" in str(exc.value)
    assert "freeze the EMA" in str(exc.value)
    assert repr(value) in str(exc.value)


def test_empty_device_rejected(tmp_path):
    p = _write_yaml(tmp_path, "device.yaml", 'params:\n  device: ""\n')
    with pytest.raises(ValueError, match="device"):
        load_inference_config(p)


def test_missing_config_file_raises_naming_the_path(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(ValueError, match="nope.yaml"):
        load_inference_config(missing)


def test_real_model_digests_loads_five_entries():
    # The REAL configs/model_digests.yaml: exactly 5 entries — the four
    # Phase 3 WR-04 digests (ablation_study.md:65-68) plus the ADOPTED
    # 5th pose_weights digest — each with its models/ path and a 64-char
    # lowercase hex sha256.
    digests = load_model_digests()
    assert len(digests) == 5
    expected = {
        "baseline_ann": (
            "models/baseline_ann.pt",
            "5f1a595ede3b94e5ae04d4b8561aa67d7079b6b80d68ea8db05ce00d8114d5ae",
        ),
        "pose_mlp": (
            "models/pose_mlp.pt",
            "6a1c4e7e35e2efbe83865bbe62a6ace89242275b4bb4c0a193fd2df990442962",
        ),
        "ensemble_model": (
            "models/ensemble_model.joblib",
            "12c84a36bc574ce62da96bc6b09aa864c97bfdbc596169eaa1ece070b155562f",
        ),
        "label_encoder": (
            "models/label_encoder.joblib",
            "b439eff2f6effdcdf056481243d1b9bd448d97b2a1eb9d16bedccce2b7fa8a0f",
        ),
        "pose_weights": (
            "models/yolov8n-pose.pt",
            "c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212",
        ),
    }
    assert set(digests) == set(expected)
    for name, (path, sha) in expected.items():
        assert digests[name]["path"] == path
        assert digests[name]["sha256"] == sha
        assert _SHA256_RE.fullmatch(digests[name]["sha256"])


def test_digest_shape_rejects_name_the_entry(tmp_path):
    base = (
        "artifacts:\n"
        "  baseline_ann:\n"
        "    path: models/baseline_ann.pt\n"
        '    sha256: "{digest}"\n'
    )
    # 63-char digest
    p = _write_yaml(tmp_path, "short.yaml", base.format(digest="a" * 63))
    with pytest.raises(ValueError, match="baseline_ann"):
        load_model_digests(p)
    # uppercase digest
    p = _write_yaml(tmp_path, "upper.yaml", base.format(digest="A" * 64))
    with pytest.raises(ValueError, match="baseline_ann"):
        load_model_digests(p)
    # missing sha256 key
    p = _write_yaml(
        tmp_path,
        "nosha.yaml",
        "artifacts:\n  baseline_ann:\n    path: models/baseline_ann.pt\n",
    )
    with pytest.raises(ValueError, match="baseline_ann"):
        load_model_digests(p)
    # empty artifacts mapping
    p = _write_yaml(tmp_path, "empty.yaml", "artifacts: {}\n")
    with pytest.raises(ValueError, match="artifacts"):
        load_model_digests(p)
