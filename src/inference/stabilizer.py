"""Phase 4 temporal stabilizer: pure window/EMA/gate/cooldown logic.

Consumes one (prob_vector, frame_idx, fps) tuple per strided frame and
returns a typed StabilizerOutcome. No cv2, no torch, no model, NO RNG
anywhere — the module imports only math, collections.deque, and dataclasses;
numpy arrays arrive as input and are read, never created with randomness
(every REQUIREMENTS literal is reproducible from injected vectors).

Exclusion semantics (pinned, RESEARCH Pattern 1 / OQ3): "excluding NoAction"
means the per-frame gate value AND the fired argmax are computed over
NON-NoAction classes only. The window still holds EVERY strided frame — a
NoAction frame contributes a non-qualifying entry; failure frames enter as
one-hot NoAction vectors fed by the analyzer (OQ3), quenching the gate
instead of silently skipping a window update. NoAction is never fired and
never counted toward the qualifying fraction.

Update order (pinned): fps fallback -> window append -> EMA -> release check
(OQ1) -> fire gate (OQ2) -> cooldown -> held check -> fire. The cooldown is
checked BEFORE the held check and lives in ORIGINAL frame-index space
(VIDEO-04: cooldown_frames = 4 x fps).
"""

import math
from collections import deque
from dataclasses import dataclass

# THE class order is the shared LabelEncoder sorted order
# [Four, NoAction, NoBall, Out, Six, Wide] — pinned in src/models/data.py
# lines 16-21 (RESEARCH Pitfall 4: never re-order). The NoAction index is
# DERIVED from this tuple in TemporalStabilizer.__init__, never spelled as a
# literal, so a custom class order fails loudly instead of gating the wrong
# class.
CLASS_ORDER = ("Four", "NoAction", "NoBall", "Out", "Six", "Wide")


@dataclass(frozen=True)
class StabilizerOutcome:
    """Typed per-frame stabilizer result.

    ``reason`` is one of the closed set: "fired" | "gate_not_met" |
    "cooldown" | "held". Only fired outcomes carry signal_idx / signal /
    confidence / frame_idx / timestamp_mmss; non-fired outcomes carry the
    reason only. update() never raises on model/video problems (none exist
    in this pure module) — it raises ValueError only on contract violations
    (e.g. a wrong-length probability vector).
    """

    fired: bool
    reason: str
    signal_idx: int | None = None
    signal: str | None = None
    confidence: float | None = None
    frame_idx: int | None = None
    timestamp_mmss: str | None = None


# Module-level typed event recorder (pose_features.py:111-136 recorder
# shape): every fps fallback, release, and fire is recorded here — counted,
# never silent (ASVS V5). Plan 04-02's analyzer surfaces them in the run
# summary; nothing in the stabilizer drops an event without a record.
_STABILIZER_EVENTS: list[str] = []


def record_stabilizer_event(label, reason) -> None:
    """Record one typed stabilizer event for the audit (ASVS V5)."""
    _STABILIZER_EVENTS.append(f"{label}: {reason}")


def stabilizer_events() -> list:
    """Stabilizer event records (one string per fps fallback / release / fire)."""
    return list(_STABILIZER_EVENTS)


def mmss(frame_idx, fps) -> str:
    """[MM:SS] timestamp for a frame index at the given effective fps
    (RESEARCH Pattern: s = int(frame_idx / fps), zero-padded MM:SS)."""
    s = int(frame_idx / fps)
    return f"{s // 60:02d}:{s % 60:02d}"


class TemporalStabilizer:
    """Pure rolling-window + EMA + fire-gate + cooldown logic.

    Every REQUIREMENTS literal is a constructor param sourced from
    configs/inference.yaml via INFERENCE_DEFAULTS (window 12, alpha 0.3,
    threshold 0.75, fire_frac 0.60, cooldown 4 s, fps fallback 30,
    require_release True). No RNG anywhere.

    Re-arm semantics (OQ1 adopted): with ``require_release=True`` (the
    default) a gesture that is still above threshold when the cooldown
    expires does NOT re-fire — the trigger re-arms only when the smoothed
    max over non-NoAction classes drops below the threshold (a typed
    "release" event). ``require_release=False`` restores the spec-literal
    cooldown-only re-arm for comparison.
    """

    def __init__(
        self,
        window=12,
        alpha=0.3,
        threshold=0.75,
        fire_frac=0.60,
        cooldown_seconds=4.0,
        fps_fallback=30.0,
        require_release=True,
        class_names=CLASS_ORDER,
    ):
        self.window = deque(maxlen=window)
        self.alpha = alpha
        self.threshold = threshold
        self.fire_frac = fire_frac
        self.cooldown_seconds = cooldown_seconds
        self.fps_fallback = fps_fallback
        self.require_release = require_release
        self.class_names = tuple(class_names)
        if self.class_names.count("NoAction") != 1:
            raise ValueError(
                f"TemporalStabilizer: class_names must contain 'NoAction' "
                f"exactly once (sorted LabelEncoder order [Four, NoAction, "
                f"NoBall, Out, Six, Wide]), got {self.class_names!r} — fix "
                "the class order (fail-closed, no silent default)"
            )
        # Derive the index from class_names — never spell it as a literal.
        self.noaction_idx = self.class_names.index("NoAction")
        self.signal_indices = tuple(
            i for i in range(len(self.class_names)) if i != self.noaction_idx
        )
        self._smoothed = None
        self._last_fire_frame = None  # ORIGINAL frame index of the last fire
        self._holding = False

    def update(self, probs, frame_idx, fps) -> StabilizerOutcome:
        """One strided frame -> typed outcome (contract violations excepted).

        Raises ValueError ONLY on a contract violation (a probs vector whose
        length differs from len(class_names)). Mechanics in the pinned
        order: (a) fps fallback, (b) window append, (c) EMA, (d) release
        check, (e) fire gate, (f) cooldown, (g) held check, (h) fire.
        """
        observed_len = len(probs)
        if observed_len != len(self.class_names):
            raise ValueError(
                f"TemporalStabilizer.update: expected a probability vector "
                f"of length {len(self.class_names)} ({self.class_names}), got "
                f"length {observed_len} — fix the injected vector (fail-closed, "
                "no silent default)"
            )

        # (a) fps fallback: CAP_PROP_FPS can be 0.0/missing/NaN — fall back
        # to the configured fps_fallback and COUNT it (typed fps_fallback
        # event with the observed value), never silently.
        if isinstance(fps, bool) or not isinstance(fps, (int, float)):
            fps_num = 0.0
        else:
            fps_num = float(fps)
        if fps_num > 0.0 and math.isfinite(fps_num):
            fps_eff = fps_num
        else:
            fps_eff = float(self.fps_fallback)
            record_stabilizer_event(
                "fps_fallback",
                f"frame {frame_idx} observed fps {fps!r} -> fallback "
                f"{self.fps_fallback!r}",
            )

        # (b) Window append — the window holds EVERY strided frame (OQ3:
        # failure frames enter as one-hot NoAction vectors fed by the
        # analyzer; a window update is never skipped).
        self.window.append(probs)

        # (c) EMA element-wise: s_t = alpha*x_t + (1-alpha)*s_{t-1}, s_0 =
        # x_0 (RESEARCH A10). Pure alpha-blend on the class-count vector.
        if self._smoothed is None:
            s = probs
        else:
            s = self.alpha * probs + (1.0 - self.alpha) * self._smoothed
        self._smoothed = s
        smoothed_max = max(s[i] for i in self.signal_indices)

        # (d) Release check (OQ1): a held gesture whose smoothed max drops
        # below the threshold re-arms the trigger — counted, never silent.
        if self._holding and smoothed_max < self.threshold:
            self._holding = False
            record_stabilizer_event(
                "release",
                f"frame {frame_idx} smoothed max {float(smoothed_max):.4f} "
                f"below threshold {self.threshold}",
            )

        # (e) Fire gate (OQ2): len(window) is the denominator — graceful
        # fill-in during the first window-1 frames; NoAction never qualifies
        # (excluded from both the argmax consideration and the fire count).
        qualifying = sum(
            1
            for p in self.window
            if max(p[i] for i in self.signal_indices) >= self.threshold
        )
        needed = math.ceil(self.fire_frac * len(self.window))
        if qualifying < needed:
            return StabilizerOutcome(fired=False, reason="gate_not_met")

        # (f) Cooldown in ORIGINAL frame-index space (VIDEO-04): cooldown
        # frames = round(cooldown_seconds * fps_eff) from the last fire.
        # Checked BEFORE the held check (pinned update order).
        cooldown_frames = round(self.cooldown_seconds * fps_eff)
        if (
            self._last_fire_frame is not None
            and frame_idx - self._last_fire_frame < cooldown_frames
        ):
            return StabilizerOutcome(fired=False, reason="cooldown")

        # (g) Held check (OQ1): a gesture still above threshold when the
        # cooldown expires does NOT re-fire — one held gesture = one event.
        if self.require_release and self._holding:
            return StabilizerOutcome(fired=False, reason="held")

        # (h) Fire: argmax over NON-NoAction classes of the CURRENT smoothed
        # vector (Python max = first-index tie-break, matching the Phase 3
        # pinned tie-break), then latch the cooldown and the held state.
        best = max(self.signal_indices, key=lambda i: s[i])
        confidence = float(s[best])
        self._last_fire_frame = frame_idx
        self._holding = True
        record_stabilizer_event(
            "fire",
            f"frame {frame_idx} signal {self.class_names[best]} "
            f"confidence {confidence:.4f}",
        )
        return StabilizerOutcome(
            fired=True,
            reason="fired",
            signal_idx=best,
            signal=self.class_names[best],
            confidence=confidence,
            frame_idx=frame_idx,
            timestamp_mmss=mmss(frame_idx, fps_eff),
        )
