#!/usr/bin/env python3
r"""
3d_relation_map.py – Flat relational structure for 3D assemblies.

Provides:
    RelationPoint     – one object in the map (obj_id, template, categories,
                        world AABB, centroid)
    RelationLink      – one predicate between two (or three for between)
                        points, with magnitude, verification status, and
                        source (geometry / llm / both)
    RelationMap       – a flat map of points and links for one assembly
                        frame

    extract_relation_map(assembly, ...)     – build a map from an Assembly3D
    map_similarity(map_a, map_b)            – structural similarity score
    describe_map(relation_map)              – human-readable summary
    map_to_search_terms(relation_map, ...)  – searchable term list with
                                              category expansion

Design:
    - One map per keyframe. The caller extracts a map from the assembly at
      the state it wants to reason about. The pipeline builds per-frame
      assemblies by applying operations and extracts one map per frame.

    - Pairwise links, not per-object primary reference. The v0.1 per-object
      form (each object has one "position.relation" to another) is a strict
      subset of this structure, so any v0.1 record can be expressed here.

    - The 3D assembly is the truth. Every geometric predicate is computed
      from AABBs. When the LLM proposes relations, they are compared against
      the geometry: proposals that geometry can verify are marked
      source="both"; proposals geometry cannot verify are stored with
      verified=False and source="llm". This makes the distinction explicit
      for downstream consumers.

    - Symmetric links are stored once with a canonical source/target order
      (lexicographically smaller point_id as source). Consumers that need
      the reverse direction flip the fields based on is_symmetric().

    - Search terms for a link are generated at multiple category levels.
      on(apple_01, counter_01) produces terms like:
          link:on(apple_01,counter_01)             – specific
          link:on(apple,counter_01)                – template level
          link:on(fruit,counter_01)                – category level
          link:on(apple,furniture)                 – via target category
          link:on(fruit,furniture)                 – both category level
      This makes a query for on(pear,counter) match a memory containing
      on(apple,counter) when both are fruit.
"""

import importlib
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_assembly = importlib.import_module("3d_assembly")
_predicates = importlib.import_module("3d_relation_predicates")

Assembly3D = _assembly.Assembly3D
Object3D = _assembly.Object3D
AABB = _predicates.AABB
element_local_aabb = _predicates.element_local_aabb
object_world_aabb = _predicates.object_world_aabb
compute_all_predicates = _predicates.compute_all_predicates
near_threshold = _predicates.near_threshold
is_symmetric = _predicates.is_symmetric
operator_arity = _predicates.operator_arity
OPERATORS = _predicates.OPERATORS


# =============================================================================
# Category levels used for term expansion
# =============================================================================

# When generating search terms, we expand each point's identity through its
# category chain. Categories beyond this depth are considered too abstract.
MAX_CATEGORY_TERM_DEPTH = 4

# Categories that are too generic to be useful search terms. Generated terms
# using these are skipped.
TOO_GENERAL_CATEGORIES = {
    "object", "physical_object", "entity", "thing",
}


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class RelationPoint:
    """One object in the relation map."""
    point_id: str            # the object_id from the assembly
    template_name: str
    categories: List[str]    # category chain from specific to abstract
    aabb: AABB
    centroid: Tuple[float, float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "template_name": self.template_name,
            "categories": list(self.categories),
            "aabb": self.aabb.to_tuple(),
            "centroid": list(self.centroid),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RelationPoint":
        return cls(
            point_id=d["point_id"],
            template_name=d["template_name"],
            categories=list(d.get("categories", [])),
            aabb=AABB.from_tuple(d["aabb"]),
            centroid=tuple(d.get("centroid", [0.0, 0.0, 0.0])),
        )


@dataclass
class RelationLink:
    """One predicate between points in the relation map."""
    link_id: str
    predicate: str
    source_point_id: str
    target_point_id: Optional[str]                # None for unary if we ever need it
    third_point_id: Optional[str] = None          # for "between"
    magnitude: Optional[float] = None
    verified: bool = True                         # True if geometry agrees
    source: str = "geometry"                      # "geometry" | "llm" | "both"

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "link_id": self.link_id,
            "predicate": self.predicate,
            "source_point_id": self.source_point_id,
            "target_point_id": self.target_point_id,
            "verified": self.verified,
            "source": self.source,
        }
        if self.third_point_id is not None:
            d["third_point_id"] = self.third_point_id
        if self.magnitude is not None:
            d["magnitude"] = self.magnitude
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RelationLink":
        return cls(
            link_id=d["link_id"],
            predicate=d["predicate"],
            source_point_id=d["source_point_id"],
            target_point_id=d.get("target_point_id"),
            third_point_id=d.get("third_point_id"),
            magnitude=d.get("magnitude"),
            verified=bool(d.get("verified", True)),
            source=d.get("source", "geometry"),
        )


@dataclass
class RelationMap:
    """Flat relational structure for one assembly frame."""
    map_id: str
    memory_id: str
    points: List[RelationPoint] = field(default_factory=list)
    links: List[RelationLink] = field(default_factory=list)
    keyframe_index: Optional[int] = None
    keyframe_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def get_point(self, point_id: str) -> Optional[RelationPoint]:
        for p in self.points:
            if p.point_id == point_id:
                return p
        return None

    def has_point(self, point_id: str) -> bool:
        return any(p.point_id == point_id for p in self.points)

    def links_for(self, point_id: str) -> List[RelationLink]:
        """All links that involve point_id as source, target, or third."""
        return [
            l for l in self.links
            if point_id in (l.source_point_id, l.target_point_id, l.third_point_id)
        ]

    def links_between(self, a_id: str, b_id: str) -> List[RelationLink]:
        """All links between two specific points, in either direction."""
        result = []
        for l in self.links:
            pair = {l.source_point_id, l.target_point_id}
            if l.third_point_id is not None:
                continue
            if pair == {a_id, b_id}:
                result.append(l)
        return result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "map_id": self.map_id,
            "memory_id": self.memory_id,
            "points": [p.to_dict() for p in self.points],
            "links": [l.to_dict() for l in self.links],
        }
        if self.keyframe_index is not None:
            d["keyframe_index"] = self.keyframe_index
        if self.keyframe_time is not None:
            d["keyframe_time"] = self.keyframe_time
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RelationMap":
        return cls(
            map_id=d["map_id"],
            memory_id=d["memory_id"],
            points=[RelationPoint.from_dict(p) for p in d.get("points", [])],
            links=[RelationLink.from_dict(l) for l in d.get("links", [])],
            keyframe_index=d.get("keyframe_index"),
            keyframe_time=d.get("keyframe_time"),
        )


# =============================================================================
# Helper: make link IDs
# =============================================================================

def _make_link_id() -> str:
    return "lnk_" + uuid.uuid4().hex[:10]


# =============================================================================
# Extract a relation map from an assembly
# =============================================================================

def _hierarchy_info_from_assembly(assembly: Assembly3D) -> Dict[str, Any]:
    """Return {"parents": {obj_id: parent_obj_id or None}} for the assembly."""
    parents = {}
    for obj in assembly.objects:
        parents[obj.object_id] = obj.parent_object_id
    return {"parents": parents}


def _categories_for_object(
    obj: Object3D,
    category_lookup: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """
    Return the category chain for an object. If category_lookup is provided
    (keyed on template_name), use it. Otherwise fall back to a chain built
    from the template name alone.
    """
    if category_lookup and obj.template_name in category_lookup:
        return list(category_lookup[obj.template_name])
    return [obj.template_name]


def extract_relation_map(
    assembly: Assembly3D,
    category_lookup: Optional[Dict[str, List[str]]] = None,
    keyframe_index: Optional[int] = None,
    keyframe_time: Optional[float] = None,
    llm_proposals: Optional[List[Dict[str, Any]]] = None,
) -> RelationMap:
    """
    Build a RelationMap from the current state of the assembly.

    Args:
        assembly        : the 3D assembly to extract from.
        category_lookup : optional dict mapping template_name to a category
                          chain. When provided, points get the full chain;
                          otherwise each point gets [template_name] only.
        keyframe_index  : optional index for the frame this map represents.
        keyframe_time   : optional time for the frame.
        llm_proposals   : optional list of relations the LLM proposed. Each
                          should be a dict with at least:
                              predicate, source_point_id, target_point_id
                          optionally third_point_id for "between".
                          Proposals that geometry also detects are folded
                          into the geometric link (source="both"). Proposals
                          geometry cannot verify are stored with
                          verified=False, source="llm".

    Returns:
        A RelationMap with all points, all geometric links, and any LLM
        proposals reconciled as described above.
    """
    world_matrices = assembly.get_all_world_matrices()
    hierarchy_info = _hierarchy_info_from_assembly(assembly)

    # Build points
    points: List[RelationPoint] = []
    aabb_by_id: Dict[str, AABB] = {}
    for obj in assembly.objects:
        M = world_matrices.get(obj.object_id)
        if M is None:
            continue
        aabb = object_world_aabb(obj, M)
        aabb_by_id[obj.object_id] = aabb
        categories = _categories_for_object(obj, category_lookup)
        points.append(RelationPoint(
            point_id=obj.object_id,
            template_name=obj.template_name,
            categories=categories,
            aabb=aabb,
            centroid=tuple(float(x) for x in aabb.center()),
        ))

    # Build links pairwise
    links: List[RelationLink] = []
    seen_symmetric: Set[Tuple[str, str, str]] = set()

    point_ids = [p.point_id for p in points]
    for i in range(len(point_ids)):
        for j in range(i + 1, len(point_ids)):
            a_id = point_ids[i]
            b_id = point_ids[j]
            a_aabb = aabb_by_id[a_id]
            b_aabb = aabb_by_id[b_id]

            # Forward direction
            raw = compute_all_predicates(
                a_aabb, b_aabb,
                a_id=a_id, b_id=b_id,
                hierarchy_info=hierarchy_info,
            )
            # Reverse direction (for directional predicates like "on",
            # "above", "below", "in", "around", "through")
            raw_rev = compute_all_predicates(
                b_aabb, a_aabb,
                a_id=b_id, b_id=a_id,
                hierarchy_info=hierarchy_info,
            )

            for r in raw + raw_rev:
                pred = r["predicate"]
                src = r["source_id"]
                tgt = r["target_id"]

                # Deduplicate symmetric predicates
                if is_symmetric(pred):
                    key = (pred, min(src, tgt), max(src, tgt))
                    if key in seen_symmetric:
                        continue
                    seen_symmetric.add(key)
                    # Canonical order: smaller id is source
                    if src > tgt:
                        src, tgt = tgt, src

                links.append(RelationLink(
                    link_id=_make_link_id(),
                    predicate=pred,
                    source_point_id=src,
                    target_point_id=tgt,
                    magnitude=r.get("magnitude"),
                    verified=True,
                    source="geometry",
                ))

    # Reconcile LLM proposals
    if llm_proposals:
        links = _reconcile_llm_proposals(links, llm_proposals)

    return RelationMap(
        map_id=f"relmap_{assembly.memory_id}"
              + (f"_{keyframe_index}" if keyframe_index is not None else ""),
        memory_id=assembly.memory_id,
        points=points,
        links=links,
        keyframe_index=keyframe_index,
        keyframe_time=keyframe_time,
    )


def _link_key(pred: str, src: str, tgt: str, third: Optional[str]) -> Tuple:
    """Canonical key for a link. Symmetric predicates canonicalise order."""
    if is_symmetric(pred):
        if src > tgt:
            src, tgt = tgt, src
    return (pred, src, tgt, third)


def _reconcile_llm_proposals(
    geometric_links: List[RelationLink],
    proposals: List[Dict[str, Any]],
) -> List[RelationLink]:
    """
    Merge LLM proposals into the geometric link set.

    Rules:
      - If a proposal matches an existing geometric link (same predicate,
        same points), mark the existing link source="both" (it is now
        confirmed by both geometry and the LLM).
      - If a proposal does not match any geometric link, add a new link
        with verified=False, source="llm". This preserves the LLM's claim
        while flagging it as unverified.
      - If a proposal uses an operator not in OPERATORS, skip it.
    """
    by_key = {_link_key(l.predicate, l.source_point_id,
                        l.target_point_id or "", l.third_point_id): l
              for l in geometric_links}

    result = list(geometric_links)

    for prop in proposals:
        if not isinstance(prop, dict):
            continue
        pred = prop.get("predicate")
        src = prop.get("source_point_id")
        tgt = prop.get("target_point_id")
        third = prop.get("third_point_id")
        if not pred or not src or not tgt:
            continue
        if pred not in OPERATORS:
            continue

        key = _link_key(pred, src, tgt, third)
        if key in by_key:
            existing = by_key[key]
            existing.source = "both"
            existing.verified = True
        else:
            result.append(RelationLink(
                link_id=_make_link_id(),
                predicate=pred,
                source_point_id=src,
                target_point_id=tgt,
                third_point_id=third,
                magnitude=prop.get("magnitude"),
                verified=False,
                source="llm",
            ))

    return result


# =============================================================================
# Map similarity
# =============================================================================

def _category_overlap(cats_a: List[str], cats_b: List[str]) -> float:
    """Return a similarity in [0, 1] based on category chain overlap."""
    if not cats_a or not cats_b:
        return 0.0
    set_a = set(cats_a)
    set_b = set(cats_b)
    # Weighted: more specific categories count more than abstract ones
    score = 0.0
    weight_sum = 0.0
    for i, cat in enumerate(cats_a):
        w = 1.0 / (i + 1)
        weight_sum += w
        if cat in set_b:
            score += w
    if weight_sum == 0:
        return 0.0
    return score / weight_sum


def _align_points(
    map_a: RelationMap,
    map_b: RelationMap,
) -> List[Tuple[RelationPoint, RelationPoint, float]]:
    """
    Return a list of aligned (point_a, point_b, similarity) triples, using
    a greedy best-match strategy based on category overlap.

    Points in either map that cannot be aligned to a partner above a
    minimum threshold are omitted.
    """
    MIN_ALIGNMENT = 0.3

    candidates: List[Tuple[float, RelationPoint, RelationPoint]] = []
    for pa in map_a.points:
        for pb in map_b.points:
            sim = _category_overlap(pa.categories, pb.categories)
            if sim >= MIN_ALIGNMENT:
                candidates.append((sim, pa, pb))

    # Greedy: best pairs first, skip any point already used
    candidates.sort(key=lambda x: -x[0])
    used_a: Set[str] = set()
    used_b: Set[str] = set()
    aligned: List[Tuple[RelationPoint, RelationPoint, float]] = []
    for sim, pa, pb in candidates:
        if pa.point_id in used_a or pb.point_id in used_b:
            continue
        used_a.add(pa.point_id)
        used_b.add(pb.point_id)
        aligned.append((pa, pb, sim))

    return aligned


def map_similarity(map_a: RelationMap, map_b: RelationMap) -> float:
    """
    Structural similarity between two relation maps, in [0, 1].

    The score combines:
        - point alignment quality (how well the two point sets correspond)
        - link agreement (how many links in one map have a corresponding
          link in the other under the point alignment)

    Two maps score high when they have the same relational pattern, even
    when the specific objects filling the slots are different. For example,
    "apple next to spoon" aligns with "pear next to fork" via category
    overlap on fruit/utensil and matches on the "near" link.
    """
    if not map_a.points or not map_b.points:
        return 0.0

    aligned = _align_points(map_a, map_b)
    if not aligned:
        return 0.0

    # Map each point to its alignment partner
    a_to_b: Dict[str, str] = {pa.point_id: pb.point_id for pa, pb, _ in aligned}
    b_to_a: Dict[str, str] = {pb.point_id: pa.point_id for pa, pb, _ in aligned}

    # Point alignment score: mean of the alignment similarities, weighted by
    # how many of each map's points were aligned.
    mean_sim = sum(sim for _, _, sim in aligned) / len(aligned)
    coverage = len(aligned) / max(len(map_a.points), len(map_b.points))
    point_score = mean_sim * coverage

    # Link agreement: for each link in A, translate its endpoints through
    # the alignment and check if a matching link exists in B.
    def translate_point(pid: str, mapping: Dict[str, str]) -> Optional[str]:
        return mapping.get(pid)

    def has_matching_link(
        target_map: RelationMap,
        pred: str,
        src_b: str,
        tgt_b: Optional[str],
        third_b: Optional[str],
    ) -> bool:
        target_key = _link_key(pred, src_b, tgt_b or "", third_b)
        for l in target_map.links:
            key = _link_key(l.predicate, l.source_point_id,
                            l.target_point_id or "", l.third_point_id)
            if key == target_key:
                return True
        return False

    matched_links = 0
    total_links_a = 0
    for la in map_a.links:
        src_b = translate_point(la.source_point_id, a_to_b)
        tgt_b = translate_point(la.target_point_id, a_to_b) if la.target_point_id else None
        third_b = translate_point(la.third_point_id, a_to_b) if la.third_point_id else None
        if src_b is None or tgt_b is None:
            continue
        total_links_a += 1
        if has_matching_link(map_b, la.predicate, src_b, tgt_b, third_b):
            matched_links += 1

    # Also account for links in B that are not matched in A
    unmatched_b = 0
    total_links_b = 0
    for lb in map_b.links:
        src_a = translate_point(lb.source_point_id, b_to_a)
        tgt_a = translate_point(lb.target_point_id, b_to_a) if lb.target_point_id else None
        third_a = translate_point(lb.third_point_id, b_to_a) if lb.third_point_id else None
        if src_a is None or tgt_a is None:
            continue
        total_links_b += 1
        if not has_matching_link(map_a, lb.predicate, src_a, tgt_a, third_a):
            unmatched_b += 1

    total_links = total_links_a + total_links_b
    if total_links == 0:
        link_score = 1.0   # both maps have no links; full agreement on empty
    else:
        link_score = (matched_links + (total_links_b - unmatched_b)) / total_links

    # Combine
    return 0.5 * point_score + 0.5 * link_score


# =============================================================================
# Search terms
# =============================================================================

def _expand_point_identities(point: RelationPoint) -> List[str]:
    """
    Return a list of identity strings for a point, from most specific
    (point_id) to most general (top of category chain).

    Example for apple_01 with categories [apple, fruit, food, physical_object]:
        ["apple_01", "apple", "fruit", "food", "physical_object"]
    """
    ids: List[str] = [point.point_id]
    if point.template_name and point.template_name != point.point_id:
        ids.append(point.template_name)
    for i, cat in enumerate(point.categories):
        if i >= MAX_CATEGORY_TERM_DEPTH:
            break
        if cat in TOO_GENERAL_CATEGORIES:
            continue
        if cat in ids:
            continue
        ids.append(cat)
    return ids


def map_to_search_terms(
    relation_map: RelationMap,
    max_terms_per_link: int = 25,
) -> List[str]:
    """
    Return a list of search terms representing the links in the map.

    Each link generates terms at multiple category levels. For example, a
    link "on(apple_01, counter_01)" where apple_01 has categories
    [apple, fruit, food] and counter_01 has categories [counter, furniture]
    generates:
        link:on(apple_01,counter_01)
        link:on(apple,counter_01)
        link:on(fruit,counter_01)
        link:on(food,counter_01)
        link:on(apple_01,counter)
        link:on(apple,counter)
        link:on(fruit,counter)
        link:on(food,counter)
        link:on(apple_01,furniture)
        link:on(apple,furniture)
        ... etc

    The specific term (both point_ids) is the most informative. Each step
    of generalisation is slightly weaker but broader. The caller can weight
    them accordingly.

    Unverified links (source="llm") are marked with an "unverified:" prefix
    so downstream consumers can choose whether to include them.

    The list is deduplicated. max_terms_per_link bounds combinatorial
    explosion when points have long category chains.
    """
    seen: Set[str] = set()
    terms: List[str] = []

    for link in relation_map.links:
        src_point = relation_map.get_point(link.source_point_id)
        if src_point is None:
            continue

        src_ids = _expand_point_identities(src_point)
        tgt_ids: List[str] = []
        third_ids: List[str] = []

        if link.target_point_id:
            tgt_point = relation_map.get_point(link.target_point_id)
            if tgt_point is None:
                continue
            tgt_ids = _expand_point_identities(tgt_point)

        if link.third_point_id:
            third_point = relation_map.get_point(link.third_point_id)
            if third_point is None:
                continue
            third_ids = _expand_point_identities(third_point)

        prefix = "link:"
        if not link.verified:
            prefix = "unverified_link:"

        # Generate combinations
        count = 0
        if third_ids:
            # between: three-part term
            for s in src_ids:
                for t in tgt_ids:
                    for u in third_ids:
                        term = f"{prefix}{link.predicate}({s},{t},{u})"
                        if term not in seen:
                            seen.add(term)
                            terms.append(term)
                            count += 1
                            if count >= max_terms_per_link:
                                break
                    if count >= max_terms_per_link:
                        break
                if count >= max_terms_per_link:
                    break
        else:
            for s in src_ids:
                for t in tgt_ids:
                    term = f"{prefix}{link.predicate}({s},{t})"
                    if term not in seen:
                        seen.add(term)
                        terms.append(term)
                        count += 1
                        if count >= max_terms_per_link:
                            break
                if count >= max_terms_per_link:
                    break

    return terms


# =============================================================================
# Human-readable summary
# =============================================================================

def describe_map(relation_map: RelationMap) -> str:
    """Return a multi-line human-readable summary of a relation map."""
    lines: List[str] = []
    lines.append(f"RelationMap {relation_map.map_id}")
    lines.append(f"  memory_id      : {relation_map.memory_id}")
    if relation_map.keyframe_index is not None:
        lines.append(f"  keyframe_index : {relation_map.keyframe_index}")
    if relation_map.keyframe_time is not None:
        lines.append(f"  keyframe_time  : {relation_map.keyframe_time:.2f}s")
    lines.append(f"  points         : {len(relation_map.points)}")
    lines.append(f"  links          : {len(relation_map.links)}")

    lines.append("")
    lines.append("  Points:")
    for p in relation_map.points:
        cats = " / ".join(p.categories) if p.categories else "(no categories)"
        lines.append(
            f"    {p.point_id:24s} template={p.template_name:18s} "
            f"aabb=[{p.aabb.min[0]:.1f},{p.aabb.min[1]:.1f},{p.aabb.min[2]:.1f}]"
            f"→[{p.aabb.max[0]:.1f},{p.aabb.max[1]:.1f},{p.aabb.max[2]:.1f}] "
            f"cats={cats}"
        )

    lines.append("")
    lines.append("  Links:")
    for l in relation_map.links:
        tag = ""
        if not l.verified:
            tag = "  [UNVERIFIED]"
        elif l.source == "both":
            tag = "  [verified by both]"
        mag = f"  mag={l.magnitude:.2f}" if l.magnitude is not None else ""
        if l.third_point_id:
            lines.append(
                f"    {l.predicate}({l.source_point_id},{l.target_point_id},"
                f"{l.third_point_id}){mag}{tag}"
            )
        else:
            lines.append(
                f"    {l.predicate}({l.source_point_id},{l.target_point_id})"
                f"{mag}{tag}"
            )

    return "\n".join(lines)


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("3d_relation_map.py – smoke test")
    print("=" * 70)

    make_element = _assembly.make_element
    make_transform = _assembly.make_transform

    # ---- Build a simple scene: apple on counter, knife near apple ----
    counter = Object3D(
        object_id="counter_01",
        template_name="counter",
        elements=[
            make_element("box", {"size_x_cm": 120.0, "size_y_cm": 60.0, "size_z_cm": 5.0}),
        ],
        material={"color": "wood"},
        object_transform=make_transform(translate=[0.0, 0.0, 77.5]),
    )
    apple = Object3D(
        object_id="apple_01",
        template_name="apple",
        elements=[make_element("sphere", {"radius_cm": 4.0})],
        material={"color": "red_brushed"},
        object_transform=make_transform(translate=[10.0, 5.0, 84.0]),
    )
    knife = Object3D(
        object_id="knife_01",
        template_name="knife",
        elements=[
            make_element("box", {"size_x_cm": 20.0, "size_y_cm": 2.0, "size_z_cm": 0.4}),
        ],
        material={"color": "silver"},
        object_transform=make_transform(translate=[25.0, 5.0, 80.5]),
    )

    assembly = Assembly3D(
        assembly_id="3d_rmap_smoke",
        memory_id="rmap_smoke",
        objects=[counter, apple, knife],
    )

    category_lookup = {
        "counter":  ["counter", "furniture"],
        "apple":    ["apple", "fruit", "food"],
        "knife":    ["knife", "cutlery", "utensil", "tool"],
    }

    print("\nExtracting relation map...")
    rmap = extract_relation_map(
        assembly,
        category_lookup=category_lookup,
        keyframe_index=0,
        keyframe_time=0.0,
    )

    print()
    print(describe_map(rmap))

    # ---- Search terms ----
    print("\n" + "=" * 70)
    print("Search terms")
    print("=" * 70)
    terms = map_to_search_terms(rmap, max_terms_per_link=25)
    print(f"\nGenerated {len(terms)} terms. First 40:")
    for t in terms[:40]:
        print(f"  {t}")
    if len(terms) > 40:
        print(f"  ... and {len(terms) - 40} more")

    # ---- Similar map: same pattern, different objects ----
    print("\n" + "=" * 70)
    print("Map similarity: same pattern with different objects")
    print("=" * 70)

    counter2 = Object3D(
        object_id="counter_01",
        template_name="counter",
        elements=[
            make_element("box", {"size_x_cm": 120.0, "size_y_cm": 60.0, "size_z_cm": 5.0}),
        ],
        material={"color": "wood"},
        object_transform=make_transform(translate=[0.0, 0.0, 77.5]),
    )
    pear = Object3D(
        object_id="pear_01",
        template_name="pear",
        elements=[make_element("ellipsoid", {"radius_x_cm": 3.0, "radius_y_cm": 3.0, "radius_z_cm": 5.0})],
        material={"color": "green"},
        object_transform=make_transform(translate=[10.0, 5.0, 85.0]),
    )
    fork = Object3D(
        object_id="fork_01",
        template_name="fork",
        elements=[
            make_element("box", {"size_x_cm": 18.0, "size_y_cm": 1.5, "size_z_cm": 0.3}),
        ],
        material={"color": "silver"},
        object_transform=make_transform(translate=[25.0, 5.0, 80.5]),
    )

    assembly2 = Assembly3D(
        assembly_id="3d_rmap_smoke2",
        memory_id="rmap_smoke2",
        objects=[counter2, pear, fork],
    )

    category_lookup2 = {
        "counter": ["counter", "furniture"],
        "pear":    ["pear", "fruit", "food"],
        "fork":    ["fork", "cutlery", "utensil", "tool"],
    }

    rmap2 = extract_relation_map(
        assembly2,
        category_lookup=category_lookup2,
        keyframe_index=0,
        keyframe_time=0.0,
    )

    print()
    print(describe_map(rmap2))

    sim = map_similarity(rmap, rmap2)
    print(f"\nSimilarity between apple-map and pear-map: {sim:.4f}")
    print("(Should be high because the relational pattern is the same.)")

    # ---- Dissimilar map: different pattern ----
    print("\n" + "=" * 70)
    print("Map similarity: same objects, different arrangement")
    print("=" * 70)

    # Move apple off the counter (far away), knife on top of apple
    apple3 = Object3D(
        object_id="apple_01",
        template_name="apple",
        elements=[make_element("sphere", {"radius_cm": 4.0})],
        material={"color": "red_brushed"},
        object_transform=make_transform(translate=[200.0, 200.0, 84.0]),
    )
    knife3 = Object3D(
        object_id="knife_01",
        template_name="knife",
        elements=[
            make_element("box", {"size_x_cm": 20.0, "size_y_cm": 2.0, "size_z_cm": 0.4}),
        ],
        material={"color": "silver"},
        object_transform=make_transform(translate=[10.0, 5.0, 86.0]),
    )
    assembly3 = Assembly3D(
        assembly_id="3d_rmap_smoke3",
        memory_id="rmap_smoke3",
        objects=[counter, apple3, knife3],
    )
    rmap3 = extract_relation_map(
        assembly3,
        category_lookup=category_lookup,
        keyframe_index=0,
        keyframe_time=0.0,
    )
    sim3 = map_similarity(rmap, rmap3)
    print(f"Similarity between apple-map and rearranged-map: {sim3:.4f}")
    print("(Should be lower because the arrangement is different.)")

    # ---- LLM proposals ----
    print("\n" + "=" * 70)
    print("LLM proposal reconciliation")
    print("=" * 70)

    proposals = [
        # Geometry already detects this: should be marked source="both"
        {"predicate": "on", "source_point_id": "apple_01", "target_point_id": "counter_01"},
        # Geometry does NOT detect this: should be stored verified=False
        {"predicate": "around", "source_point_id": "knife_01", "target_point_id": "apple_01"},
    ]
    rmap4 = extract_relation_map(
        assembly,
        category_lookup=category_lookup,
        keyframe_index=0,
        keyframe_time=0.0,
        llm_proposals=proposals,
    )
    print()
    print(describe_map(rmap4))

    print("\nSmoke test complete.")