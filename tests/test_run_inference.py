"""CLI contract tests (Plan 04-02 Task 2): src/run_inference.py.

Contracts pinned here (test_ablation_report.py monkeypatch + tmp_path
analog; weight-free by default with monkeypatched module seams):

- --dry-run: config + digest-sweep verification + census naming every
  artifact path (the integrity count is the MEASURED len(digests) — WR-03),
  exit 0, no torch import (train_ablation dry-run convention).
- Integrity refusal BEFORE any deserialize (prohibition 1/2 mechanical
  check): a tampered digest or a missing artifact exits with "[integrity]"
  naming the artifact while torch.load/joblib.load recorders provably
  never fire (planner choice per checker MINOR 5: the former stubbed
  call-order test is DROPPED — stubbing both the loaders AND the analyzer
  class makes an order assertion tautological; (b)/(c) pin the refusal
  end-to-end over the real main() flow instead).
- Determinism: no rendered line matches a wall-clock date
  (\\d{4}-\\d{2}-\\d{2}) — the features_build.md posture.
- Flags fail closed: --video + --image together and neither-given are both
  argparse errors (exit 2); --dry-run alone is valid.
- Summary contract: "[MM:SS] <signal> conf=" event lines in emission order,
  the law-correct final "Score:" line, and the no-validation-video-tuning
  provenance line (rendered from a stubbed analyzer — no weights needed).
- Missing video input: a valid-digest scaffold with dummy artifacts fails
  at the video-open guard with "[input]" naming the path (weight-free by
  construction — the guard fires before any warm_up/model load).
- CPU smoke (STATE eval flag, weight+fixture gated at the bottom): the
  full CLI path on device=cpu with byte-identical summaries across two
  runs.
"""

import re
from pathlib import Path

import pytest

import src.inference.analyzer as analyzer_mod
import src.inference.config as inference_config
import src.run_inference as ri
from conftest import FIXTURE_IMAGE
from src.inference.config import load_model_digests
from src.models.data import sha256_of
from src.scoreboard import EventLogEntry, ScoreState, apply_event

REPO_ROOT = Path(__file__).resolve().parent.parent

# The 5 artifact names + repo-relative paths, in configs/model_digests.yaml
# order (baseline_ann first — the sweep order the tamper test relies on).
ARTIFACT_ORDER = (
    "baseline_ann",
    "pose_mlp",
    "ensemble_model",
    "label_encoder",
    "pose_weights",
)


def _write_scaffold_digests(scaffold, sha_by_name):
    """Write the scaffold's configs/model_digests.yaml in ARTIFACT_ORDER
    (baseline_ann first — the sweep refuses there first)."""
    lines = ["artifacts:"]
    for name in ARTIFACT_ORDER:
        lines.append(f"  {name}:")
        lines.append(f"    path: {sha_by_name[name]['path']}")
        lines.append(f"    sha256: \"{sha_by_name[name]['sha256']}\"")
    (scaffold / "configs" / "model_digests.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _patch_scaffold_roots(monkeypatch, scaffold):
    """Point every REPO_ROOT the CLI + loaders resolve paths through at the
    scaffold: the CLI's own default-config path, the config loaders'
    default digests path, and the analyzer's artifact paths."""
    monkeypatch.setattr(ri, "REPO_ROOT", scaffold)
    monkeypatch.setattr(inference_config, "REPO_ROOT", scaffold)
    monkeypatch.setattr(analyzer_mod, "REPO_ROOT", scaffold)


def _make_scaffold(tmp_path):
    """Minimal valid scaffold: an empty-params inference.yaml (the
    INFERENCE_DEFAULTS carry the values, so the config stage passes)."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "inference.yaml").write_text(
        "params: {}\n", encoding="utf-8"
    )
    return tmp_path


def _real_digests():
    """The real 5-entry digest mapping (read with an EXPLICIT path — no
    monkeypatch interplay)."""
    return load_model_digests(REPO_ROOT / "configs" / "model_digests.yaml")


def test_dry_run_validates_and_prints_census(capsys):
    """--dry-run against the REAL configs: exit 0, the census names all 5
    artifact paths, and the 5 digests verify (all artifacts exist here)."""
    assert ri.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    digests = _real_digests()
    assert len(digests) == 5
    for name, entry in digests.items():
        assert entry["path"] in out, f"census must name {entry['path']}"
    assert f"integrity: {len(digests)}/{len(digests)}" in out


def test_dry_run_integrity_count_is_measured_not_hardcoded(
    tmp_path, monkeypatch, capsys
):
    """WR-03: the integrity line renders len(digests), never a hardcoded
    "5/5" — a fully-verifying 3-entry digests scaffold renders
    "integrity: 3/3" AND "artifact census (3 entries)" consistently in the
    same report (dummy artifacts with matching sha256s; no weights)."""
    scaffold = _make_scaffold(tmp_path)
    lines = ["artifacts:"]
    for name in ARTIFACT_ORDER[:3]:  # baseline_ann, pose_mlp, ensemble_model
        rel = _real_digests()[name]["path"]
        artifact = scaffold / rel
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"dummy {name} bytes".encode("utf-8"))
        lines.append(f"  {name}:")
        lines.append(f"    path: {rel}")
        lines.append(f'    sha256: "{sha256_of(artifact)}"')
    (scaffold / "configs" / "model_digests.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    _patch_scaffold_roots(monkeypatch, scaffold)

    assert ri.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "integrity: 3/3 artifact digests verified" in out
    assert "artifact census (3 entries):" in out
    assert "5/5" not in out, "the hardcoded 5/5 literal must be gone"


def test_tampered_artifact_refused_before_deserialize(tmp_path, monkeypatch, capsys):
    """Prohibition 1 mechanical check: a real-format but WRONG baseline_ann
    digest (swept first) refuses fail-closed ("[integrity]" naming the
    artifact) BEFORE any deserialize — torch.load AND joblib.load
    recorders provably never called."""
    scaffold = _make_scaffold(tmp_path)
    (scaffold / "models").mkdir()
    (scaffold / "models" / "baseline_ann.pt").write_bytes(b"fake bytes")
    sha_by_name = {
        name: {"path": entry["path"], "sha256": entry["sha256"]}
        for name, entry in _real_digests().items()
    }
    sha_by_name["baseline_ann"]["sha256"] = "e" * 64  # real-format, wrong value
    _write_scaffold_digests(scaffold, sha_by_name)
    _patch_scaffold_roots(monkeypatch, scaffold)

    # Recorders on BOTH deserializers — the refusal must precede any load.
    torch_calls = []
    joblib_calls = []

    def _torch_load_recorder(*args, **kwargs):
        torch_calls.append(args)
        raise AssertionError("torch.load called before the integrity refusal")

    def _joblib_load_recorder(*args, **kwargs):
        joblib_calls.append(args)
        raise AssertionError("joblib.load called before the integrity refusal")

    import joblib

    import torch

    monkeypatch.setattr(torch, "load", _torch_load_recorder)
    monkeypatch.setattr(joblib, "load", _joblib_load_recorder)

    with pytest.raises(SystemExit) as exc:
        ri.main(["--dry-run"])
    message = str(exc.value)
    assert "[integrity]" in message
    assert "baseline_ann" in message
    assert torch_calls == []
    assert joblib_calls == []


def test_missing_artifact_fails_closed(tmp_path, monkeypatch):
    """Second refusal branch: a valid-format digest entry whose artifact
    file does not exist refuses on the absent bytes BEFORE any
    deserialize ("[integrity]" naming the missing artifact path)."""
    scaffold = _make_scaffold(tmp_path)
    sha_by_name = {
        name: {"path": entry["path"], "sha256": entry["sha256"]}
        for name, entry in _real_digests().items()
    }
    _write_scaffold_digests(scaffold, sha_by_name)
    _patch_scaffold_roots(monkeypatch, scaffold)
    with pytest.raises(SystemExit) as exc:
        ri.main(["--dry-run"])
    message = str(exc.value)
    assert "[integrity]" in message
    assert "models/baseline_ann.pt" in message


def test_summary_has_no_wall_clock_timestamp(capsys):
    """Determinism convention (test_ablation_report.py:321): no rendered
    dry-run line matches a wall-clock date."""
    assert ri.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert not re.search(r"\d{4}-\d{2}-\d{2}", out), "wall-clock date leaked"


def test_flags_fail_closed(capsys):
    """--video + --image together, and neither-given, are both argparse
    errors (exit 2); --dry-run alone is valid."""
    with pytest.raises(SystemExit) as exc:
        ri.main(["--video", "a.mp4", "--image", "b.jpg"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        ri.main([])
    assert exc.value.code == 2
    assert ri.main(["--dry-run"]) == 0
    capsys.readouterr()  # drain the dry-run census output


def test_event_log_line_format(tmp_path, monkeypatch, capsys):
    """Canned-result summary: two "[MM:SS] <signal> conf=" lines in emission
    order, the law-correct final Score line (Four then Wide -> 5 runs,
    0 wickets, 0.1 overs, 1 extra), and the no-video-tuning provenance
    line. Analyzer class monkeypatched — no weights needed."""
    state = ScoreState()
    state, _ = apply_event(state, "Four")
    four = EventLogEntry(
        timestamp="00:00",
        signal="Four",
        confidence=0.8123,
        score=(state.runs, state.wickets, state.overs, state.extras),
    )
    state, _ = apply_event(state, "Wide")
    wide = EventLogEntry(
        timestamp="00:04",
        signal="Wide",
        confidence=0.7901,
        score=(state.runs, state.wickets, state.overs, state.extras),
    )

    from src.inference.analyzer import VideoAnalysisResult

    class _StubAnalyzer:
        def __init__(self, config_path=None, extractor=None):
            self.config_path = config_path

        def analyze(self, video_path, output_path=None):
            return VideoAnalysisResult(
                events=(four, wide),
                final_score=state,
                frames_read=120,
                strided_frames=60,
                failures={},
                fps_reported=30.0,
                device="cpu",
                model_mode="image",
            )

        def classify_image(self, image):  # pragma: no cover - not exercised here
            raise AssertionError("classify_image must not run for --video")

    monkeypatch.setattr(ri, "UmpireVideoAnalyzer", _StubAnalyzer)

    assert ri.main(["--video", str(tmp_path / "whatever.mp4")]) == 0
    out = capsys.readouterr().out
    four_idx = out.index("[00:00] Four conf=")
    wide_idx = out.index("[00:04] Wide conf=")
    assert four_idx < wide_idx, "event lines must render in emission order"
    assert "Score: 5 runs/0 wickets/0.1 overs/1 extras" in out
    assert "no validation-video tuning performed" in out


def test_missing_video_input_fails_closed(tmp_path, monkeypatch):
    """A valid-digest scaffold of five dummy model files: the sweep passes,
    analyzer construction succeeds, and the video-open guard raises
    "[input]" naming the nonexistent path — weight-free by construction
    (the guard fires before warm_up loads anything)."""
    scaffold = _make_scaffold(tmp_path)
    sha_by_name = {}
    for name in ARTIFACT_ORDER:
        rel = _real_digests()[name]["path"]
        artifact = scaffold / rel
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"dummy {name} bytes".encode("utf-8"))
        sha_by_name[name] = {"path": rel, "sha256": sha256_of(artifact)}
    _write_scaffold_digests(scaffold, sha_by_name)
    _patch_scaffold_roots(monkeypatch, scaffold)

    missing_video = scaffold / "nope.mp4"
    with pytest.raises(SystemExit) as exc:
        ri.main(["--video", str(missing_video)])
    message = str(exc.value)
    assert "[input]" in message
    assert "nope.mp4" in message


# ---------------------------------------------------------------------------
# Weight + fixture gated smoke tests (must RUN on this machine — the gates
# exist for fresh clones, never to mask failures here).
# ---------------------------------------------------------------------------

_WEIGHTS_OK = all(
    (REPO_ROOT / "models" / n).exists()
    for n in ("yolov8n-pose.pt", "baseline_ann.pt", "label_encoder.joblib")
)


@pytest.mark.skipif(
    not (_WEIGHTS_OK and FIXTURE_IMAGE.exists()),
    reason="phase weights / untracked fixture image not present (fresh clone posture)",
)
def test_cpu_smoke_end_to_end(real_image_fixture_video, capsys):
    """STATE eval flag — smoke-test the CPU fallback once this phase: the
    full CLI path over the real-image fixture on device=cpu (extract ->
    probs -> stabilizer -> scoreboard -> summary). Asserts exit 0, the
    summary's device field == cpu, an event-log section present, and
    byte-identical summaries across two runs (determinism contract)."""
    fixture = str(real_image_fixture_video)
    assert ri.main(["--video", fixture]) == 0
    first = capsys.readouterr().out
    assert "device: cpu" in first
    assert "event log:" in first
    assert "Score:" in first

    assert ri.main(["--video", fixture]) == 0
    second = capsys.readouterr().out
    assert first == second, "summaries must be byte-identical across re-runs"


@pytest.mark.skipif(
    not (_WEIGHTS_OK and FIXTURE_IMAGE.exists()),
    reason="phase weights / untracked fixture image not present (fresh clone posture)",
)
def test_temporal_summary_renders_untrained_marker(
    real_image_fixture_video, tmp_path, capsys
):
    """Plan 04-03 (P2RDY-02): the Plan 04-02 `model == "temporal"` summary
    branch becomes reachable — main(["--video", fixture, "--config",
    temporal_yaml]) exits 0 and the rendered summary contains the UNTRAINED
    smoke-marker line (the marker constant imported from src.scoreboard —
    the single source — rendered verbatim) and the no-validation-video-tuning
    provenance line. Two runs produce byte-identical summaries (determinism
    contract: seeded stub, timestamp-free rendering)."""
    cfg_path = tmp_path / "inference_temporal.yaml"
    cfg_path.write_text("params:\n  model: temporal\n", encoding="utf-8")
    fixture = str(real_image_fixture_video)

    assert ri.main(["--video", fixture, "--config", str(cfg_path)]) == 0
    first = capsys.readouterr().out

    from src.scoreboard import UNTRAINED_SMOKE_MARKER

    assert UNTRAINED_SMOKE_MARKER == "UNTRAINED — smoke only"
    assert UNTRAINED_SMOKE_MARKER in first, "summary must render the UNTRAINED line"
    assert "model_mode: temporal" in first
    assert "no validation-video tuning performed" in first

    assert ri.main(["--video", fixture, "--config", str(cfg_path)]) == 0
    second = capsys.readouterr().out
    assert first == second, "temporal summaries must be byte-identical across re-runs"
