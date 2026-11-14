"""Phase 4 inference config seam: fail-closed loaders for configs/inference.yaml
and configs/model_digests.yaml.

Contract (load_pose_config lineage, src/pose_features.py:55-94): each loader
merges the YAML ``params`` over the module-level defaults dict (IN-02 merged
single source) and then validates EVERY key fail-closed — a bad value raises
ValueError naming the key, its observed value, and the fix (the
"(fail-closed, no silent default)" clause, data.py posture). A missing or
unparseable config file is a ValueError naming the path — never a silent
default and never a bare FileNotFoundError escaping the seam.

The fire-gate minimums are security-relevant (T-04-01): threshold and
fire_frac must lie in (0.0, 1.0] and window/stride/max_frames must be ints
>= 1, so config can never disable the fire gate or the cooldown by setting a
degenerate value.
"""

from pathlib import Path

import re

import yaml

# src/inference/ is one level deeper than src/ — same depth as src/models/*
# (PATTERNS.md): three .parent hops resolve to the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Single source of inference defaults (IN-02): configs/inference.yaml params
# are merged over this dict; both surfaces carry the same eleven locked keys
# (REQUIREMENTS-literal defaults — no validation-video tuning was performed,
# no video dataset exists per PROJECT.md Out-of-Scope).
INFERENCE_DEFAULTS = {
    "model": "image",            # P2RDY-02 switch — only image | temporal are valid
    "device": "cpu",             # locked CONTEXT decision; code stays device-agnostic
    "stride": 2,                 # VIDEO-01 literal
    "window": 12,                # VIDEO-02 deque maxlen
    "alpha": 0.3,                # VIDEO-02 EMA blend
    "threshold": 0.75,           # VIDEO-03 fire threshold (excluding NoAction)
    "fire_frac": 0.60,           # VIDEO-03 window fraction (len(window) denominator)
    "cooldown_seconds": 4.0,     # VIDEO-04 cooldown = 4 x fps frames
    "fps_fallback": 30.0,        # fallback when CAP_PROP_FPS is 0/missing
    "require_release": True,     # OQ1 adopted: release-based re-arm
    "max_frames": 100000,        # typed-abort DoS guard for the headless CLI (T-04-06)
}

# A digest is exactly 64 lowercase hex chars — the artifact entries feed
# verify_artifact_sha256 (src/models/data.py:451-480), so anything else is a
# corrupted config, not a digest.
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _read_yaml_mapping(cfg_path: Path, label: str):
    """Shared loader spine: read the file text (utf-8) and yaml.safe_load it,
    converting every I/O or parse failure into a ValueError naming the path
    (fail-closed, no silent default)."""
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except OSError as err:
        raise ValueError(
            f"{label}: config file could not be read at {cfg_path} ({err}) "
            "— fix the path (fail-closed, no silent default)"
        ) from err
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise ValueError(
            f"{label}: invalid YAML at {cfg_path} ({err}) — fix the file "
            "(fail-closed, no silent default)"
        ) from err
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(
            f"{label}: top-level YAML must be a mapping, got {cfg!r} — fix "
            "the file (fail-closed, no silent default)"
        )
    return cfg


def load_inference_config(path=None) -> dict:
    """Merge configs/inference.yaml params over INFERENCE_DEFAULTS (IN-02
    single source) and fail-close validate every key (load_pose_config
    posture: a bad value raises ValueError naming the key and its observed
    value — never a silent default)."""
    cfg_path = Path(path) if path is not None else REPO_ROOT / "configs" / "inference.yaml"
    yaml_cfg = _read_yaml_mapping(cfg_path, "configs/inference.yaml")
    params = yaml_cfg.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(
            f"configs/inference.yaml: params must be a mapping, got "
            f"{params!r} — fix the file (fail-closed, no silent default)"
        )
    cfg = {**INFERENCE_DEFAULTS, **params}

    model = cfg["model"]
    if model not in ("image", "temporal"):
        raise ValueError(
            f"configs/inference.yaml: model must be exactly 'image' or "
            f"'temporal' (P2RDY-02 switch), got {model!r} — fix the config "
            "or INFERENCE_DEFAULTS (fail-closed, no silent default)"
        )

    # T-04-01: the fire-gate minimums are (0.0, 1.0] — 0 or negative would
    # disable the gate, so they are rejected, not clamped.
    for key in ("threshold", "fire_frac"):
        value = cfg[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 < float(value) <= 1.0
        ):
            raise ValueError(
                f"configs/inference.yaml: {key} must be a number in (0.0, "
                f"1.0] — 0 or negative would disable the fire gate (T-04-01)"
                f", got {value!r} (fail-closed, no silent default)"
            )

    value = cfg["alpha"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(
            f"configs/inference.yaml: alpha must be a number in [0.0, 1.0], "
            f"got {value!r} — fix the config or INFERENCE_DEFAULTS "
            "(fail-closed, no silent default)"
        )

    for key in ("cooldown_seconds", "fps_fallback"):
        value = cfg[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not float(value) > 0.0
        ):
            raise ValueError(
                f"configs/inference.yaml: {key} must be a number > 0, got "
                f"{value!r} — fix the config or INFERENCE_DEFAULTS "
                "(fail-closed, no silent default)"
            )

    # Bool is not an int here: True would otherwise sneak through as 1.
    for key in ("window", "stride", "max_frames"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"configs/inference.yaml: {key} must be an int >= 1 (bool is "
                f"not an int here), got {value!r} — fix the config or "
                "INFERENCE_DEFAULTS (fail-closed, no silent default)"
            )

    device = cfg["device"]
    if not isinstance(device, str) or not device.strip():
        raise ValueError(
            f"configs/inference.yaml: device must be a non-empty string, got "
            f"{device!r} — fix the config or INFERENCE_DEFAULTS (fail-closed, "
            "no silent default)"
        )

    if not isinstance(cfg["require_release"], bool):
        raise ValueError(
            f"configs/inference.yaml: require_release must be a bool, got "
            f"{cfg['require_release']!r} — fix the config or "
            "INFERENCE_DEFAULTS (fail-closed, no silent default)"
        )
    return cfg


def load_model_digests(path=None) -> dict:
    """Load + validate configs/model_digests.yaml -> the ``artifacts`` mapping
    (name -> {path, sha256}) that feeds verify_artifact_sha256
    (src/models/data.py:451-480) before every deserialization in this phase
    (WR-04: refuse-before-deserialize; expected_sha256=None is never passed).

    Fail-closed: ``artifacts`` must be a non-empty mapping and every entry
    needs a non-empty string ``path`` plus a sha256 that is exactly 64
    lowercase hex chars — anything else is a ValueError naming the offending
    entry."""
    cfg_path = (
        Path(path) if path is not None else REPO_ROOT / "configs" / "model_digests.yaml"
    )
    yaml_cfg = _read_yaml_mapping(cfg_path, "configs/model_digests.yaml")
    artifacts = yaml_cfg.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(
            f"configs/model_digests.yaml: artifacts must be a non-empty "
            f"mapping of artifact name -> {{path, sha256}}, got "
            f"{artifacts!r} — fix the file (fail-closed, no silent default)"
        )
    for name, entry in artifacts.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"configs/model_digests.yaml: artifact {name!r} must be a "
                f"mapping with path + sha256 keys, got {entry!r} — fix the "
                "file (fail-closed, no silent default)"
            )
        rel_path = entry.get("path")
        if not isinstance(rel_path, str) or not rel_path.strip():
            raise ValueError(
                f"configs/model_digests.yaml: artifact {name!r} path must be "
                f"a non-empty string, got {rel_path!r} — fix the file "
                "(fail-closed, no silent default)"
            )
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ValueError(
                f"configs/model_digests.yaml: artifact {name!r} sha256 must "
                f"be a 64-char lowercase hex string, got {digest!r} — a "
                "corrupted or hand-edited digest is not a digest (fail-closed, "
                "no silent default)"
            )
    return artifacts
