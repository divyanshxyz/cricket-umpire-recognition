"""Phase 4 CLI: headless inference over an uploaded .mp4 or a single image.

Contract (04-02-PLAN Task 2): a thin controller over
src/inference/analyzer.py — fail-closed flags, a startup integrity sweep
over ALL 5 digests from configs/model_digests.yaml (verify_artifact_sha256
BEFORE anything else; a --dry-run census does config + digest validation +
artifact existence only: no torch import, no model import, no inference,
matching the train_ablation dry-run convention), and a deterministic
timestamp-free summary carrying the [MM:SS] event log + the final
law-correct scoreboard line.

No default file writes: the summary is console-only; an annotated mp4 is
written ONLY when --output names a path (mp4v fourcc enforced by the
analyzer). Byte-identical re-runs on the same input are the determinism
contract — no wall-clock timestamps anywhere (data_ingestion.py:295-298
posture; the summary is built as ONE string and printed once).

Fail-closed exit taxonomy (data_ingestion stage-exit convention):
- config ValueError          -> sys.exit(f"[config] {err}")
- integrity refusal          -> sys.exit(f"[integrity] {err}")
- analyzer input ValueError  -> sys.exit(f"[input] {err}")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Direct-run support (conftest.py only covers pytest): put the checkout
# root on sys.path BEFORE the src imports. The 'src.*' spelling is the ONLY
# module spelling (IN-03) — a second spelling would create a second module
# object and split the module-level event counters.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.analyzer import UmpireVideoAnalyzer
from src.inference.config import load_inference_config, load_model_digests
from src.models.data import verify_artifact_sha256

_PROVENANCE_LINE = (
    "stabilizer parameters: spec literals from configs/inference.yaml — "
    "no validation-video tuning performed"
)


def _sweep_digests(repo_root: Path, digests: dict) -> None:
    """Startup integrity sweep (WR-04 / Pattern 5): verify EVERY digest
    entry BEFORE any deserialize. A missing artifact refuses naming the
    path; a swapped/stale one refuses inside verify_artifact_sha256 — both
    fail-closed ValueErrors the caller renders as "[integrity]"."""
    for entry in digests.values():
        artifact = repo_root / entry["path"]
        if not artifact.is_file():
            raise ValueError(
                f"startup integrity sweep: artifact {entry['path']} does not "
                f"exist at {artifact} — refusing to run inference against a "
                "missing model file (fail-closed, no silent default; "
                "bootstrap the artifact or fix configs/model_digests.yaml)"
            )
        verify_artifact_sha256(
            artifact, entry["sha256"], artifact_name=entry["path"]
        )


def _run_dry_run(cfg_path) -> int:
    """--dry-run: config + digest validation + artifact census only — no
    torch/model import, no inference (train_ablation dry-run convention).
    The sweep refuses fail-closed ("[integrity] ...") before anything else
    prints."""
    try:
        cfg = load_inference_config(cfg_path)
        digests = load_model_digests()
    except ValueError as err:
        sys.exit(f"[config] {err}")
    try:
        _sweep_digests(REPO_ROOT, digests)
    except ValueError as err:
        sys.exit(f"[integrity] {err}")

    lines = []
    lines.append("=== run_inference dry-run ===")
    # WR-03: the reported count is the MEASURED len(digests), never a magic
    # number — the census line below already renders len(digests), so a
    # digests file with any entry count produces a self-consistent report.
    lines.append(
        f"integrity: {len(digests)}/{len(digests)} artifact digests verified "
        "against configs/model_digests.yaml (no deserialization performed)"
    )
    lines.append("resolved params:")
    for key in (
        "model",
        "device",
        "stride",
        "window",
        "alpha",
        "threshold",
        "fire_frac",
        "cooldown_seconds",
        "fps_fallback",
        "require_release",
        "max_frames",
    ):
        lines.append(f"  {key}: {cfg[key]}")
    lines.append(f"artifact census ({len(digests)} entries):")
    for name, entry in digests.items():
        lines.append(f"  {name}: {entry['path']} exists sha256 ok")
    print("\n".join(lines))
    return 0


def _render_video_summary(cfg, result) -> str:
    """ONE deterministic string for the whole run summary (no wall-clock
    timestamps anywhere — byte-identical re-runs are the contract)."""
    lines = []
    lines.append("=== run_inference video summary ===")
    lines.append(f"model_mode: {cfg['model']}  device: {cfg['device']}")
    lines.append(
        f"stride: {cfg['stride']}  window: {cfg['window']}  alpha: {cfg['alpha']}"
    )
    lines.append(
        f"threshold: {cfg['threshold']}  fire_frac: {cfg['fire_frac']}  "
        f"cooldown_seconds: {cfg['cooldown_seconds']}  "
        f"require_release: {cfg['require_release']}"
    )
    lines.append(
        f"frames read: {result.frames_read}  strided: {result.strided_frames}"
        f"  fps_reported: {result.fps_reported}"
    )
    if result.failures:
        tally = "  ".join(
            f"{label}={count}" for label, count in sorted(result.failures.items())
        )
        lines.append(f"typed failure tallies: {tally}")
    else:
        lines.append("typed failure tallies: none")
    lines.append("event log:")
    if result.events:
        for event in result.events:
            lines.append(
                f"[{event.timestamp}] {event.signal} "
                f"conf={event.confidence:.4f} "
                f"score=({event.score[0]}, {event.score[1]}, "
                f"{event.score[2]}, {event.score[3]})"
            )
    else:
        lines.append("  (no gated signal events fired)")
    state = result.final_score
    lines.append(
        f"Score: {state.runs} runs/{state.wickets} wickets/{state.overs} "
        f"overs/{state.extras} extras"
    )
    lines.append(_PROVENANCE_LINE)
    if cfg["model"] == "temporal":
        # Branch-local lazy import (Plan 04-03 wiring seam): the constant
        # does not exist until Plan 04-03 adds it to src/scoreboard.py —
        # at Wave 2 this branch is dead for every config this plan ships
        # (model: image), so the module must NEVER import the name at load
        # time. Plan 04-03 wires the real constant (and may edit this CLI
        # per its own files_modified).
        from src.scoreboard import UNTRAINED_SMOKE_MARKER

        lines.append(UNTRAINED_SMOKE_MARKER)
    return "\n".join(lines)


def _run_video(cfg_path, video_path, output_path) -> int:
    try:
        cfg = load_inference_config(cfg_path)
    except ValueError as err:
        sys.exit(f"[config] {err}")
    try:
        analyzer = UmpireVideoAnalyzer(config_path=cfg_path)
    except ValueError as err:
        # Construction-time refusals are integrity-flavored: the 5-digest
        # startup sweep (WR-04) refuses before ANY deserialize (WR-02
        # fail-fast).
        sys.exit(f"[integrity] {err}")
    try:
        result = analyzer.analyze(video_path, output_path=output_path)
    except ValueError as err:
        sys.exit(f"[input] {err}")
    print(_render_video_summary(cfg, result))
    return 0


def _render_image_summary(cfg, classification) -> str:
    lines = []
    lines.append("=== run_inference image summary ===")
    lines.append(f"model_mode: {cfg['model']}  device: {cfg['device']}")
    lines.append(f"predicted_class: {classification.predicted_class}")
    for name, prob in classification.probabilities.items():
        lines.append(f"  {name}: {prob:.4f}")
    return "\n".join(lines)


def _run_image(cfg_path, image_path) -> int:
    try:
        cfg = load_inference_config(cfg_path)
    except ValueError as err:
        sys.exit(f"[config] {err}")
    try:
        analyzer = UmpireVideoAnalyzer(config_path=cfg_path)
    except ValueError as err:
        sys.exit(f"[integrity] {err}")
    try:
        classification = analyzer.classify_image(image_path)
    except ValueError as err:
        sys.exit(f"[input] {err}")
    print(_render_image_summary(cfg, classification))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="run_inference",
        description=(
            "Phase 4 inference runner: analyze an uploaded .mp4 (stride-2 "
            "pose -> stabilizer -> law-correct scoreboard) or classify a "
            "single image, with a startup integrity sweep over all 5 model "
            "digests and a deterministic timestamp-free run summary."
        ),
    )
    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--video",
        default=None,
        help="path to the input .mp4 (analyzed every stride-th frame)",
    )
    group.add_argument(
        "--image",
        default=None,
        help="path to a single image (predicted class + per-class probabilities)",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="optional annotated .mp4 output path (mp4v fourcc; writes ONLY "
        "when named — no default file writes; video runs only)",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="inference config path (default: configs/inference.yaml)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="config + digest validation + artifact census only; no torch, "
        "no model import, no inference, no writes",
    )
    args = ap.parse_args(argv)

    if args.output is not None and args.image is not None:
        # Semantically invalid flag combination (data_ingestion ap.error
        # posture): --output annotates a VIDEO run's frames.
        ap.error(
            f"--output applies only to --video runs (got --image "
            f"{args.image!r}) — drop --output or use --video"
        )
    if not args.dry_run and args.video is None and args.image is None:
        # Exactly one of --video/--image is required for a real run (the
        # mutual-exclusion group rejects BOTH-given; this guard rejects
        # NEITHER-given). --dry-run alone is valid (census posture).
        ap.error(
            "one of --video PATH or --image PATH is required for a real run "
            "(or pass --dry-run for the config + digest census)"
        )

    # Resolved AT RUNTIME through REPO_ROOT so monkeypatched roots (test
    # scaffolds) re-resolve the default correctly.
    cfg_path = (
        Path(args.config)
        if args.config is not None
        else REPO_ROOT / "configs" / "inference.yaml"
    )

    if args.dry_run:
        return _run_dry_run(cfg_path)
    if args.video is not None:
        return _run_video(cfg_path, args.video, args.output)
    return _run_image(cfg_path, args.image)


if __name__ == "__main__":
    sys.exit(main())
