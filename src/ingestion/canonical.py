"""Raw label -> canonical class mapping (configs/classes.yaml).

Alias resolution is exact-match on normalized label text — no fuzzy matching.
An unknown label is a hard error, never a silent skip (RESEARCH.md Pitfall 4).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


class UnknownLabelError(RuntimeError):
    """A raw label that maps to nothing in configs/classes.yaml."""


def normalize_label(text: str) -> str:
    """Lowercase and strip every non-alphanumeric character.

    'hand out' and 'handout', 'no action' and 'noaction', 'no-ball' and
    'noball' collide identically.
    """
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def load_class_map(path) -> dict:
    """yaml.safe_load configs/classes.yaml and build the inverted alias dict.

    The returned config carries the original blocks plus ``_alias``:
    normalized label text -> (outcome, target) where outcome is 'keep' |
    'conditional' | 'excluded' and target is the canonical/conditional class
    name, or the exclusion reason for excluded labels. The target's own name
    is always included in the vocabulary. Build order follows the YAML file,
    so the mapping is deterministic.
    """
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    alias: dict[str, tuple[str, str]] = {}
    for target, aliases in (cfg.get("mapping") or {}).items():
        for a in [target, *aliases]:
            alias[normalize_label(a)] = ("keep", target)
    for target, aliases in (cfg.get("conditional_classes") or {}).items():
        for a in [target, *aliases]:
            alias[normalize_label(a)] = ("conditional", target)
    for target, reason in (cfg.get("exclude_as_class") or {}).items():
        alias[normalize_label(target)] = ("excluded", str(reason))
    cfg["_alias"] = alias
    return cfg


def map_label(raw_label, class_cfg: dict) -> tuple[str, str | None, str | None]:
    """Map one raw label -> (outcome, canonical, exclude_reason).

    outcome 'keep' sets canonical to the canonical class; 'conditional' sets
    canonical to the conditional class name (LegBye/DeadBall, included only
    when the post-dedup count is strictly greater than 30 — Plan 03 audit);
    'excluded' sets exclude_reason. Unknown labels raise UnknownLabelError
    naming the label — the caller adds the source context.
    """
    key = normalize_label(raw_label)
    hit = (class_cfg.get("_alias") or {}).get(key)
    if hit is None:
        raise UnknownLabelError(
            f"unknown raw label '{raw_label}' (normalized '{key}'): not in "
            "configs/classes.yaml mapping / conditional_classes / exclude_as_class "
            "(exact-match only, no fuzzy matching) — add it to the config or fix the adapter"
        )
    outcome, target = hit
    if outcome == "excluded":
        return "excluded", None, target
    return outcome, target, None
