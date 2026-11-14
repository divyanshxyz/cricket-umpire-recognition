"""Two-layer dedup tests on synthetic 16-hex-char hashes — no image files.

Proves (01-02 contract): inclusive phash threshold (Hamming 8 clusters, 9 is
distinct), transitive union-find merging, deterministic survivor selection
(priority tier, then lexicographic relpath), RF stem regex precision
(.rf.<32hex> only — a 31-hex suffix is NOT stripped), stem-group removal
counting, cluster-level conflict resolution (majority label at equal tier,
tier priority across tiers), and content-only keying (same filename,
different content -> separate clusters).

Note on the chain test: the plan sketched a~b dist 6, b~c dist 7, a-c dist
12 — but Hamming distances obey a parity constraint (dist(a,c) ≡ dist(a,b) +
dist(b,c) mod 2), so 6+7=13 odd can never yield an even a-c distance. The
test uses a-c dist 13 (> 8, so a and c do not cluster directly) — the
semantic intent (b bridges a and c; union-find merges transitively) is
preserved exactly.
"""

from src.ingestion.adapters import Record
from src.ingestion.dedup import (
    cluster_by_phash,
    resolve_cluster_labels,
    roboflow_stem,
    stem_group_label_disagreements,
    stem_groups,
)

ZERO = "0" * 16  # 64 zero bits — the anchor hash for exact distances


def rec(relpath, phash_hex=ZERO, priority=1, raw_label="out",
        family="filename-label", source="s"):
    return Record(
        source=source, relpath=relpath, family=family, priority=priority,
        raw_label=raw_label, phash_hex=phash_hex,
    )


# --- Layer B: inclusive threshold ------------------------------------------------


def test_distance_exactly_8_clusters_together():
    eight_bits = "ff" + "0" * 14  # popcount 8 -> Hamming distance 8 from ZERO
    survivors, removed = cluster_by_phash([rec("a.jpg", ZERO), rec("b.jpg", eight_bits)])
    assert len(survivors) == 1
    assert len(removed) == 1


def test_distance_9_stays_separate():
    nine_bits = "1ff" + "0" * 13  # popcount 9 -> Hamming distance 9 from ZERO
    survivors, removed = cluster_by_phash([rec("a.jpg", ZERO), rec("b.jpg", nine_bits)])
    assert len(survivors) == 2
    assert removed == []


def test_transitive_chain_merges_via_union_find():
    h_a = ZERO                # anchor
    h_b = "3f" + "0" * 14     # 6 bits  -> dist(a, b) = 6  (cluster pair)
    h_c = "3f7f" + "0" * 12   # 13 bits -> dist(a, c) = 13, dist(b, c) = 7
    assert int(h_c, 16).bit_count() == 13
    survivors, removed = cluster_by_phash(
        [rec("a.jpg", h_a), rec("b.jpg", h_b), rec("c.jpg", h_c)]
    )
    # a-c never pairs directly (13 > 8) but b bridges them -> one cluster.
    assert len(survivors) == 1
    assert len(removed) == 2


# --- Survivor selection -----------------------------------------------------------


def test_survivor_is_highest_priority_record():
    hi = rec("z_last.jpg", priority=3, source="rf")
    lo = rec("a_first.jpg", priority=1, source="file")
    survivors, _ = cluster_by_phash([lo, hi])
    assert survivors[0] is hi


def test_equal_priority_tiebreak_is_lexicographic_relpath():
    first = rec("a.jpg")
    second = rec("b.jpg")
    survivors, _ = cluster_by_phash([second, first])
    assert survivors[0] is first


def test_survivor_selection_deterministic_across_runs():
    members = [
        rec("dup_b.jpg", ZERO, priority=2, raw_label="wide"),
        rec("dup_a.jpg", ZERO, priority=2, raw_label="wide"),
        rec("dup_c.jpg", ZERO, priority=1, raw_label="six"),
        rec("dup_d.jpg", ZERO, priority=1, raw_label="six"),
    ]
    survivors1, removed1 = cluster_by_phash(list(members))
    survivors2, removed2 = cluster_by_phash(list(members))
    assert survivors1 == survivors2
    assert removed1 == removed2
    assert survivors1[0].relpath == "dup_a.jpg"


# --- Layer A: Roboflow stem groups -------------------------------------------------


def test_roboflow_stem_strips_32hex_suffix():
    assert (
        roboflow_stem("frame_0028_jpg.rf.4ab9b8326f6c8dead70622d44b607035")
        == "frame_0028_jpg"
    )


def test_roboflow_stem_without_suffix_unchanged():
    assert roboflow_stem("frame_0028_jpg") == "frame_0028_jpg"


def test_roboflow_stem_31hex_suffix_not_stripped():
    h31 = "4ab9b8326f6c8dead70622d44b60703"
    assert len(h31) == 31
    assert roboflow_stem(f"frame_0028_jpg.rf.{h31}") != "frame_0028_jpg"


def test_roboflow_stem_real_filename_shape():
    assert (
        roboflow_stem("frame_0028_jpg.rf.4ab9b8326f6c8dead70622d44b607035.jpg")
        == "frame_0028_jpg"
    )


def test_stem_groups_keep_one_per_stem_and_report_removals():
    members = [
        rec(f"frame_0000_jpg.rf.{i:032x}.jpg", f"{i:016x}", priority=3,
            family="roboflow-yolo", source="rf")
        for i in range(22)
    ]
    survivors, removed = stem_groups(members)
    assert len(survivors) == 1
    assert len(removed) == 21  # 22 copies -> 1 survivor + 21 removals
    assert survivors[0].relpath == min(m.relpath for m in members)


def test_stem_groups_pass_non_roboflow_through():
    plain = rec("x.jpg", family="filename-label")
    a = rec("frame_0000_jpg.rf." + "a" * 32 + ".jpg", priority=3,
            family="roboflow-yolo", source="rf")
    b = rec("frame_0000_jpg.rf." + "b" * 32 + ".jpg", priority=3,
            family="roboflow-yolo", source="rf")
    survivors, removed = stem_groups([plain, a, b])
    assert len(survivors) == 2
    assert len(removed) == 1
    assert plain in survivors


def test_stem_group_label_disagreements_counts_mixed_label_groups():
    # IN-10: Layer A keeps the first copy per stripped stem and discards the
    # rest WITHOUT label resolution — the helper counts groups whose members
    # carried different raw labels so the audit surfaces the silence.
    def rf(relpath, label):
        return rec(relpath, priority=3, family="roboflow-yolo", source="rf",
                   raw_label=label)

    agreeing = [
        rf(f"train/images/frame_0001_jpg.rf.{'a' * 32}.jpg", "four"),
        rf(f"test/images/frame_0001_jpg.rf.{'b' * 32}.jpg", "four"),
    ]
    disagreeing = [
        rf(f"train/images/frame_0002_jpg.rf.{'c' * 32}.jpg", "four"),
        rf(f"test/images/frame_0002_jpg.rf.{'d' * 32}.jpg", "wide"),
    ]
    plain = rec("x.jpg", family="filename-label", raw_label="out")
    assert stem_group_label_disagreements(agreeing + [plain]) == 0
    assert stem_group_label_disagreements(agreeing + disagreeing + [plain]) == 1


# --- Cluster-level conflict resolution ---------------------------------------------


def test_majority_label_wins_at_equal_tier():
    members = [
        rec("a1.jpg", ZERO, raw_label="six"),
        rec("a2.jpg", "1" * 16, raw_label="six"),
        rec("a3.jpg", "2" * 16, raw_label="wide"),
    ]
    winner, conflict, detail = resolve_cluster_labels(members)
    assert winner == "six"
    assert conflict is True
    assert "wide" in detail


def test_label_tie_broken_lexicographically():
    members = [
        rec("a1.jpg", ZERO, raw_label="wide"),
        rec("a2.jpg", "1" * 16, raw_label="six"),
    ]
    winner, conflict, detail = resolve_cluster_labels(members)
    assert winner == "six"  # 1 vs 1 -> lexicographically smallest label
    assert conflict is True
    assert "wide" in detail


def test_priority_tier_outvotes_majority():
    members = [
        rec("a1.jpg", ZERO, priority=1, raw_label="noball"),
        rec("a2.jpg", "1" * 16, priority=1, raw_label="noball"),
        rec("a3.jpg", "2" * 16, priority=3, raw_label="six"),
    ]
    winner, conflict, detail = resolve_cluster_labels(members)
    assert winner == "six"  # tier 3 beats the two tier-1 votes
    assert conflict is True
    assert "noball" in detail


def test_single_label_cluster_has_no_conflict():
    members = [
        rec("a1.jpg", ZERO, raw_label="out"),
        rec("a2.jpg", "1" * 16, raw_label="out"),
    ]
    winner, conflict, detail = resolve_cluster_labels(members)
    assert winner == "out"
    assert conflict is False
    assert detail == ""


def test_cluster_by_phash_resolves_labels_onto_survivor():
    members = [
        rec("b.jpg", ZERO, raw_label="wide"),
        rec("a.jpg", ZERO, raw_label="wide"),
        rec("c.jpg", ZERO, raw_label="six"),
    ]
    survivors, removed = cluster_by_phash(members)
    assert len(survivors) == 1
    assert len(removed) == 2
    s = survivors[0]
    assert s.relpath == "a.jpg"  # lexicographic survivor
    assert s.raw_label == "wide"  # majority label resolved onto the survivor
    assert s.conflict is True
    assert "six" in s.conflict_detail


# --- Content-only keying ------------------------------------------------------------


def test_same_filename_different_content_stays_separate():
    a = rec("dup.jpg", ZERO, source="archive")
    b = rec("dup.jpg", "f" * 16, source="archive")  # Hamming distance 64
    survivors, removed = cluster_by_phash([a, b])
    assert len(survivors) == 2  # dedup never keys on filenames
    assert removed == []
