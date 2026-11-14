"""Canonical mapper tests: the full verified on-disk vocabulary.

Pure functions against the real configs/classes.yaml — no files written.
Covers keep aliases (sixes/six/6, no_ball/noball/no-ball, no_action/no action,
wideball, four/4/boundary, out/hand out/wicket), the on-disk folder/filename
tokens (six, noball, WideBall, NoBall), excluded labels (DRS, umpire,
non_umpire), conditional tagging (LegBye/legbye), and the unknown-label hard
error — empty string and None are unknown, never a default class (Pitfall 4).
"""

from pathlib import Path

import pytest

from src.ingestion.canonical import UnknownLabelError, load_class_map, map_label

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "classes.yaml"


@pytest.fixture(scope="module")
def class_cfg():
    return load_class_map(CONFIG_PATH)


def test_six_aliases(class_cfg):
    assert map_label("sixes", class_cfg) == ("keep", "Six", None)
    assert map_label("six", class_cfg) == ("keep", "Six", None)
    assert map_label("6", class_cfg) == ("keep", "Six", None)


def test_noball_normalization_strips_hyphen_and_underscore(class_cfg):
    assert map_label("no_ball", class_cfg) == ("keep", "NoBall", None)
    assert map_label("noball", class_cfg) == ("keep", "NoBall", None)
    assert map_label("no-ball", class_cfg) == ("keep", "NoBall", None)
    assert map_label("NoBall", class_cfg) == ("keep", "NoBall", None)


def test_noaction_underscore_and_space(class_cfg):
    assert map_label("no_action", class_cfg) == ("keep", "NoAction", None)
    assert map_label("no action", class_cfg) == ("keep", "NoAction", None)


def test_wide_family(class_cfg):
    assert map_label("wideball", class_cfg) == ("keep", "Wide", None)
    assert map_label("wide", class_cfg) == ("keep", "Wide", None)


def test_four_aliases(class_cfg):
    assert map_label("four", class_cfg) == ("keep", "Four", None)
    assert map_label("4", class_cfg) == ("keep", "Four", None)
    assert map_label("boundary", class_cfg) == ("keep", "Four", None)


def test_out_aliases(class_cfg):
    assert map_label("out", class_cfg) == ("keep", "Out", None)
    assert map_label("hand out", class_cfg) == ("keep", "Out", None)
    assert map_label("wicket", class_cfg) == ("keep", "Out", None)


def test_excluded_labels(class_cfg):
    outcome, canonical, reason = map_label("DRS", class_cfg)
    assert outcome == "excluded"
    assert canonical is None
    assert reason
    for label in ("umpire", "non_umpire"):
        outcome, canonical, reason = map_label(label, class_cfg)
        assert outcome == "excluded"
        assert canonical is None
        assert reason


def test_conditional_legbye(class_cfg):
    assert map_label("LegBye", class_cfg) == ("conditional", "LegBye", None)
    assert map_label("legbye", class_cfg) == ("conditional", "LegBye", None)


def test_empty_and_none_are_unknown(class_cfg):
    with pytest.raises(UnknownLabelError):
        map_label("", class_cfg)
    with pytest.raises(UnknownLabelError):
        map_label(None, class_cfg)


def test_unknown_label_error_names_label(class_cfg):
    with pytest.raises(UnknownLabelError) as err:
        map_label("framedump", class_cfg)
    assert "framedump" in str(err.value)
