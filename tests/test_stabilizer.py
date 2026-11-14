"""Temporal stabilizer pins — pure, injected numpy vectors only.

No torch, no cv2, no fixtures, no RNG: every test feeds deterministic
one-hot probability vectors and pins the shipped contracts:
1. Window fill-in (OQ2): the len(window) denominator fires on the FIRST
   qualifying frame (needed = ceil(fire_frac * len(window))).
2. NoAction exclusion: never fires (argmax over non-NoAction only) AND
   never counts toward the qualifying fraction — window entries while being
   evicted still don't count (OQ3 policy).
3. EMA math (A10): s_0 = x_0 exactly; then s_t = alpha*x_t + (1-alpha)*s_{t-1}.
4. Cooldown frame-space: cooldown_frames = round(cooldown_seconds * fps) in
   ORIGINAL frame-index space; fps<=0/NaN falls back to fps_fallback with a
   typed fps_fallback event (counted, never silent).
5. Release-based re-arm (OQ1, the DEFAULT): one held gesture = one event;
   require_release=False restores the spec-literal cooldown-only re-arm.
6. Failure-vector policy (OQ3): one-hot NoAction quench; window bounded.
7. No-RNG determinism: identical streams -> identical outcomes; AST scan
   forbids random/numpy.random/torch imports in the module.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from src.inference.stabilizer import (
    CLASS_ORDER,
    TemporalStabilizer,
    mmss,
    stabilizer_events,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Derived from CLASS_ORDER (sorted LabelEncoder order) — never spelled as a
# bare literal in the tests either.
FOUR = CLASS_ORDER.index("Four")
NOACTION = CLASS_ORDER.index("NoAction")
SIX = CLASS_ORDER.index("Six")


def _one_hot(idx, value=1.0):
    v = np.zeros(len(CLASS_ORDER), dtype=np.float64)
    v[idx] = value
    return v


def test_oq2_fill_in_single_qualifying_vector_fires_immediately():
    # needed = ceil(0.6 * 1) = 1: the len(window) denominator means ONE
    # qualifying vector at frame 0 fires immediately (graceful fill-in
    # during the first window-1 frames — no warm-up requirement).
    st = TemporalStabilizer()
    out = st.update(_one_hot(FOUR, 0.9), 0, 30.0)
    assert out.fired and out.reason == "fired"
    assert out.signal == "Four" and out.signal_idx == FOUR
    assert out.frame_idx == 0 and out.confidence == pytest.approx(0.9)
    assert out.timestamp_mmss == "00:00"


def test_noaction_never_fires_over_50_updates():
    # A stream of pure one-hot NoAction vectors never fires: NoAction is
    # excluded from the argmax consideration (and the gate value).
    st = TemporalStabilizer()
    outs = [st.update(_one_hot(NOACTION, 1.0), i, 30.0) for i in range(50)]
    assert not any(o.fired for o in outs)
    assert all(o.reason == "gate_not_met" for o in outs)


def test_noaction_excluded_from_qualifying_count_even_while_evicted():
    # 12 NoAction one-hots prefill the window (zero fires); then 7 Four
    # vectors do NOT fire (7 < needed = ceil(0.6 * 12) = 8 at the now-full
    # window) — NoAction entries, even while being evicted, never count
    # toward qualifying. The 8th Four fires exactly there, not earlier.
    st = TemporalStabilizer()
    for i in range(12):
        assert not st.update(_one_hot(NOACTION, 1.0), i, 30.0).fired
    for i in range(12, 19):
        out = st.update(_one_hot(FOUR, 0.9), i, 30.0)
        assert not out.fired and out.reason == "gate_not_met"
    out = st.update(_one_hot(FOUR, 0.9), 19, 30.0)
    assert out.fired and out.signal == "Four" and out.frame_idx == 19


def test_ema_pin_s0_equals_x0_then_alpha_blend():
    # A10: s_0 = x_0 exactly; after a second distinct vector the smoothed
    # value is 0.3*x + 0.7*s_0 to float tolerance (alpha 0.3 default).
    st = TemporalStabilizer(alpha=0.3)
    x0 = _one_hot(FOUR, 0.9)
    st.update(x0, 0, 30.0)
    assert np.array_equal(st._smoothed, x0)
    x1 = _one_hot(FOUR, 0.5)
    st.update(x1, 1, 30.0)
    expected = 0.3 * 0.5 + 0.7 * 0.9  # 0.78 at the Four index
    assert st._smoothed[FOUR] == pytest.approx(expected, abs=1e-12)
    smoothed_max = max(st._smoothed[i] for i in range(len(CLASS_ORDER)) if i != NOACTION)
    assert smoothed_max == pytest.approx(expected, abs=1e-12)


def test_cooldown_only_mode_double_fire_pin_require_release_false():
    # cooldown_seconds 4.0 at fps 10 -> 40-frame cooldown in ORIGINAL frame
    # space: a constant qualifying stream over 80 frames fires exactly twice
    # (frames 0 and 40; a third would land at frame 80, one past the
    # stream). A held gesture RE-FIRES under the cooldown-only mode — that
    # mode is available but NOT the default (OQ1 flipped off).
    st = TemporalStabilizer(cooldown_seconds=4.0, require_release=False)
    v = _one_hot(FOUR, 0.9)
    outs = [st.update(v, i, 10.0) for i in range(80)]
    fires = [o for o in outs if o.fired]
    assert [f.frame_idx for f in fires] == [0, 40]


def test_release_based_rearm_default_oq1():
    # OQ1 default: require_release=True, fps 30, constant qualifying stream
    # over 600 frames (3x the 120-frame cooldown) produces exactly ONE fire
    # — one held gesture = one event (frames 120-599 are "held").
    st = TemporalStabilizer()
    v = _one_hot(FOUR, 0.9)
    na = _one_hot(NOACTION, 1.0)
    outs = [st.update(v, i, 30.0) for i in range(600)]
    fires = [o for o in outs if o.fired]
    assert [f.frame_idx for f in fires] == [0]
    # Dip: 5 NoAction frames drop the smoothed max below threshold -> the
    # typed "release" event re-arms the trigger (OQ1); the rise that follows
    # cannot re-fire while held, so the second fire lands on the first
    # post-release frame where the gate still qualifies. Pinned update order
    # (release -> gate -> cooldown -> held -> fire): on the dip's first
    # frame the window still holds 11 qualifying Fours, so the re-armed
    # trigger fires right there (frame 600); the 60-frame rise stays inside
    # that fire's 120-frame cooldown, so no third fire appears.
    ev_before = len(stabilizer_events())
    outs += [st.update(na, 600 + i, 30.0) for i in range(5)]
    outs += [st.update(v, 605 + i, 30.0) for i in range(60)]
    new_events = stabilizer_events()[ev_before:]
    assert any(e.startswith("release") for e in new_events)
    fires = [o for o in outs if o.fired]
    assert [f.frame_idx for f in fires] == [0, 600]
    assert all(f.signal == "Four" for f in fires)


def test_fps_fallback_typed_event_and_cooldown_math():
    # fps=0.0 and fps=NaN each record exactly one typed "fps_fallback"
    # event per fallback update (counted, never silent), and the cooldown
    # math runs off fps_fallback 30.0: fires at frames 0 and
    # round(4 * 30) = 120 over a 125-frame stream — exactly 2
    # (require_release=False mirrors the cooldown-only pin so a second fire
    # is reachable on a constant qualifying stream).
    ev_before = len(stabilizer_events())
    st = TemporalStabilizer(require_release=False, cooldown_seconds=4.0, fps_fallback=30.0)
    v = _one_hot(FOUR, 0.9)
    outs = [
        st.update(v, i, 0.0 if i % 2 == 0 else float("nan")) for i in range(125)
    ]
    new_events = stabilizer_events()[ev_before:]
    fallback_events = [e for e in new_events if "fps_fallback" in e]
    # exactly one typed fps_fallback event per fallback update; the two
    # typed fire events (frames 0 and 120) are also recorded, so the list
    # grows by 125 + 2
    assert len(fallback_events) == 125
    assert len(new_events) == 127
    fires = [o for o in outs if o.fired]
    assert [f.frame_idx for f in fires] == [0, 120]


def test_failure_vector_quench_oq3():
    # Failure frames enter as one-hot NoAction vectors (OQ3): injected
    # mid-stream they quench the gate — after a fire, 12 NoAction one-hots
    # yield no second fire, and the window stays bounded by maxlen (a
    # window update is never skipped, never grows unbounded).
    st = TemporalStabilizer()
    v = _one_hot(FOUR, 0.9)
    outs = [st.update(v, i, 30.0) for i in range(12)]
    assert sum(o.fired for o in outs) == 1  # fire at frame 0
    quenched = [st.update(_one_hot(NOACTION, 1.0), 12 + i, 30.0) for i in range(12)]
    assert not any(o.fired for o in quenched)
    assert len(st.window) == 12


def test_mmss_pins():
    assert mmss(150, 30.0) == "00:05"
    assert mmss(2490, 30.0) == "01:23"


def test_no_rng_determinism_identical_streams_identical_outcomes():
    stream = [
        _one_hot(FOUR, 0.9) if i % 3 != 2 else _one_hot(SIX, 0.8) for i in range(40)
    ]
    st1 = TemporalStabilizer()
    st2 = TemporalStabilizer()  # independently constructed, same params
    t1 = [
        (o.fired, o.reason, o.signal, o.confidence)
        for o in (st1.update(v, i, 30.0) for i, v in enumerate(stream))
    ]
    t2 = [
        (o.fired, o.reason, o.signal, o.confidence)
        for o in (st2.update(v, i, 30.0) for i, v in enumerate(stream))
    ]
    assert t1 == t2


def test_no_rng_import_scan():
    # AST scan: src/inference/stabilizer.py must not import random,
    # numpy.random, or torch — the stabilizer is deterministic by
    # construction (numpy arrays arrive as input, never created here).
    tree = ast.parse(
        (REPO_ROOT / "src" / "inference" / "stabilizer.py").read_text(encoding="utf-8")
    )
    forbidden = {"random", "numpy.random", "torch"}
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name)
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module)
                roots.add(node.module.split(".")[0])
    assert not (roots & forbidden), f"forbidden RNG/model imports: {roots & forbidden}"


def test_wrong_length_vector_raises_naming_observed_length():
    # Contract violation (not a model/video problem): a 5-length vector is
    # fail-closed rejected naming the observed length.
    st = TemporalStabilizer()
    with pytest.raises(ValueError, match="length 5"):
        st.update(np.zeros(5), 0, 30.0)
