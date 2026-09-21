#!/usr/bin/env python3
r"""
3d_relation_predicates.py – Spatial relation predicates for the 3D module.

Defines the restricted operator set (carried over from v0.1) and provides
the geometric computations that decide each relation from an assembly's
world coordinates.

Design:
    - Predicates are computed from axis-aligned bounding boxes (AABBs),
      centroids, and the assembly hierarchy. The 3D assembly is the truth;
      the LLM can propose relations, but the geometric computations here
      decide what goes in the relation map.

    - The operator registry mirrors the v0.1 restricted set:
        on, in, below, above, around, through, contact, attached, between,
        near, apart, from, none

    - Symmetric operators are stored once and matched in either direction.
    - Transitive operators can be chained.
    - Strong operators imply near: on, in, above, below, contact,
      attached, around, through, between all imply near(a,b).

    - The near threshold is size-dependent: for objects A and B, the
      threshold is (max_dim(A) + max_dim(B)) / 2 * 3.

    - AABBs honour clips. When an object carries metadata["clips"], its
      world AABB is computed by sampling each element's surface, applying
      the element transform, applying the clips, and taking the AABB of
      the surviving points. This gives correct tight bounds for split
      objects. When the shape similarity module is not available or the
      object has no clips, the AABB falls back to the corner-based
      computation over the primitive element bounds.

Public API:
    OPERATORS                               – registry of operator metadata
    AABB                                    – axis-aligned bounding box
    element_local_aabb(element)             – local AABB of an element
    object_world_aabb(obj, world_matrix)    – world AABB of an object
    compute_all_predicates(aabb_a, aabb_b, hierarchy_info) -> List[Dict]
    predicate_holds(aabb_a, aabb_b, name)   – single-predicate check
    is_symmetric(name) / is_transitive(name)
    implies_near(name)
"""

import importlib
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_assembly = importlib.import_module("3d_assembly")


# Lazy peer-module loader. The shape similarity module provides surface
# samplers and clip filtering; loading it here lets object_world_aabb
# sample clipped surfaces instead of expanding unclipped primitive AABBs.
try:
    _shape_sim = importlib.import_module("3d_shape_similarity")
    _SURFACE_SAMPLERS = _shape_sim._SAMPLERS
    _apply_transform_shape = _shape_sim._apply_transform
    _apply_clips_shape = _shape_sim._apply_clips
    _SHAPE_SIM_AVAILABLE = True
except Exception as _e:
    _SURFACE_SAMPLERS = {}
    _apply_transform_shape = None
    _apply_clips_shape = None
    _SHAPE_SIM_AVAILABLE = False


# =============================================================================
# Operator registry
# =============================================================================

OPERATORS: Dict[str, Dict[str, Any]] = {
    "on":       {"symmetric": False, "transitive": False, "implies_near": True,  "arity": 2},
    "in":       {"symmetric": False, "transitive": True,  "implies_near": True,  "arity": 2},
    "below":    {"symmetric": False, "transitive": True,  "implies_near": True,  "arity": 2},
    "above":    {"symmetric": False, "transitive": True,  "implies_near": True,  "arity": 2},
    "around":   {"symmetric": False, "transitive": False, "implies_near": True,  "arity": 2},
    "through":  {"symmetric": False, "transitive": False, "implies_near": True,  "arity": 2},
    "contact":  {"symmetric": True,  "transitive": False, "implies_near": True,  "arity": 2},
    "attached": {"symmetric": True,  "transitive": False, "implies_near": True,  "arity": 2},
    "between":  {"symmetric": False, "transitive": False, "implies_near": True,  "arity": 3},
    "near":     {"symmetric": True,  "transitive": True,  "implies_near": True,  "arity": 2},
    "apart":    {"symmetric": True,  "transitive": False, "implies_near": False, "arity": 2},
    "from":     {"symmetric": False, "transitive": False, "implies_near": False, "arity": 2},
    "none":     {"symmetric": True,  "transitive": False, "implies_near": False, "arity": 2},
}


def is_symmetric(name: str) -> bool:
    return OPERATORS.get(name, {}).get("symmetric", False)


def is_transitive(name: str) -> bool:
    return OPERATORS.get(name, {}).get("transitive", False)


def implies_near(name: str) -> bool:
    return OPERATORS.get(name, {}).get("implies_near", False)


def operator_arity(name: str) -> int:
    return OPERATORS.get(name, {}).get("arity", 2)


# =============================================================================
# Constants for the predicate computations
# =============================================================================

# Vertical margin for above/below.
VERTICAL_MARGIN_CM = 2.0

# Horizontal margin for left/right/front/behind relations.
HORIZONTAL_MARGIN_CM = 2.0

# Contact tolerance: AABBs whose separation is below this are in contact.
CONTACT_TOLERANCE_CM = 0.5

# Near threshold multiplier. Threshold = mean_max_dim(a, b) * this.
NEAR_THRESHOLD_MULTIPLIER = 3.0

# Tolerance for "in": A's AABB may overflow B's AABB by this much.
IN_TOLERANCE_CM = 1.0

# Tolerance for "on": A's bottom face may be above B's top face by up to
# this much (or below by up to this much) and still count as resting on B.
ON_VERTICAL_TOLERANCE_CM = 1.0

# For "around": B's AABB must be surrounded by A's AABB on at least this
# many axes for A to be considered "around" B.
AROUND_MIN_AXES = 2

# For "through": the longest dimension of the overlap box must be at least
# this fraction of the smallest dimension of the pierced object.
THROUGH_MIN_OVERLAP_RATIO = 0.6

# For "between": A's centroid must lie between B and C along some axis,
# and A must be near both.
BETWEEN_MARGIN_CM = 2.0

# Default number of surface points sampled per element when computing a
# clipped AABB. 2048 gives sub-millimetre accuracy for 5–20 cm objects.
DEFAULT_AABB_SAMPLES_PER_ELEMENT = 2048


# =============================================================================
# AABB
# =============================================================================

class AABB:
    """
    Axis-aligned bounding box in world coordinates.

    min: np.ndarray (3,) – the (x, y, z) minimum corner
    max: np.ndarray (3,) – the (x, y, z) maximum corner
    """

    __slots__ = ("min", "max")

    def __init__(self, min_corner, max_corner):
        self.min = np.array(min_corner, dtype=float)
        self.max = np.array(max_corner, dtype=float)

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    def size(self) -> np.ndarray:
        return self.max - self.min

    def max_dim(self) -> float:
        s = self.size()
        return float(np.max(s))

    def min_dim(self) -> float:
        s = self.size()
        return float(np.min(s))

    def center(self) -> np.ndarray:
        return (self.min + self.max) / 2.0

    def is_valid(self) -> bool:
        return bool(np.all(self.max >= self.min))

    def to_tuple(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return (
            (float(self.min[0]), float(self.min[1]), float(self.min[2])),
            (float(self.max[0]), float(self.max[1]), float(self.max[2])),
        )

    @classmethod
    def from_tuple(cls, t) -> "AABB":
        return cls(t[0], t[1])

    # ------------------------------------------------------------------
    # Geometric relations between two AABBs
    # ------------------------------------------------------------------
    def distance_to(self, other: "AABB") -> float:
        """
        Euclidean distance between the two AABBs. Zero if they overlap or
        touch. Otherwise the minimum distance between any two points, one
        in each box.
        """
        gap = np.maximum(0.0, np.maximum(self.min - other.max, other.min - self.max))
        return float(np.linalg.norm(gap))

    def separation_along(self, axis: int, other: "AABB") -> float:
        """
        Signed separation along one axis. Positive means self is above
        other on that axis. Negative means below. Zero means the intervals
        overlap.
        """
        return float(max(self.min[axis] - other.max[axis],
                         other.min[axis] - self.max[axis]))

    def overlaps_along(self, axis: int, other: "AABB") -> bool:
        """Return True if the intervals overlap on the given axis."""
        return not (self.max[axis] < other.min[axis] or self.min[axis] > other.max[axis])

    def contains(self, other: "AABB", tolerance: float = 0.0) -> bool:
        """Return True if other is contained within self (with tolerance)."""
        return bool(
            np.all(other.min >= self.min - tolerance) and
            np.all(other.max <= self.max + tolerance)
        )

    def contains_point(self, point: np.ndarray, tolerance: float = 0.0) -> bool:
        return bool(
            np.all(point >= self.min - tolerance) and
            np.all(point <= self.max + tolerance)
        )


# =============================================================================
# Element local AABBs (primitive corner-based; used as a fallback)
# =============================================================================

def element_local_aabb(element) -> AABB:
    """
    Return the element's axis-aligned bounding box in the element's own
    local frame (before the element transform, before the object transform).

    This is the primitive corner-based AABB and does not account for clips.
    Clips are applied at the object level in object_world_aabb.
    """
    t = element.element_type
    p = element.parameters

    if t == "sphere":
        r = p["radius_cm"]
        return AABB([-r, -r, -r], [r, r, r])

    if t == "ellipsoid":
        rx, ry, rz = p["radius_x_cm"], p["radius_y_cm"], p["radius_z_cm"]
        return AABB([-rx, -ry, -rz], [rx, ry, rz])

    if t == "box":
        sx, sy, sz = p["size_x_cm"], p["size_y_cm"], p["size_z_cm"]
        return AABB([-sx/2, -sy/2, -sz/2], [sx/2, sy/2, sz/2])

    if t == "cylinder":
        r, h = p["radius_cm"], p["height_cm"]
        return AABB([-r, -r, -h/2], [r, r, h/2])

    if t == "cone":
        r, h = p["radius_cm"], p["height_cm"]
        return AABB([-r, -r, -h/2], [r, r, h/2])

    if t == "torus":
        R, r = p["major_radius_cm"], p["minor_radius_cm"]
        return AABB([-(R+r), -(R+r), -r], [R+r, R+r, r])

    if t == "capsule":
        r, h = p["radius_cm"], p["height_cm"]
        return AABB([-r, -r, -(h/2+r)], [r, r, h/2+r])

    if t == "plane":
        sx = p.get("size_x_cm", 100.0) or 100.0
        sy = p.get("size_y_cm", 100.0) or 100.0
        return AABB([-sx/2, -sy/2, 0.0], [sx/2, sy/2, 0.0])

    if t == "sheet":
        sx, sy, th = p["size_x_cm"], p["size_y_cm"], p["thickness_cm"]
        return AABB([-sx/2, -sy/2, -th/2], [sx/2, sy/2, th/2])

    if t == "wedge":
        sx, sy, sz = p["size_x_cm"], p["size_y_cm"], p["size_z_cm"]
        return AABB([-sx/2, -sy/2, -sz/2], [sx/2, sy/2, sz/2])

    if t == "prism":
        r = max(p["size_x_cm"], p["size_y_cm"]) / 2.0
        h = p["size_y_cm"]
        return AABB([-r, -r, -h/2], [r, r, h/2])

    if t == "pyramid":
        bx, by, h = p["base_size_x_cm"], p["base_size_y_cm"], p["height_cm"]
        return AABB([-bx/2, -by/2, -h/2], [bx/2, by/2, h/2])

    if t == "hemisphere":
        r = p["radius_cm"]
        return AABB([-r, -r, 0.0], [r, r, r])

    if t == "frustum":
        r = max(p["bottom_radius_cm"], p["top_radius_cm"])
        h = p["height_cm"]
        return AABB([-r, -r, -h/2], [r, r, h/2])

    # Unknown element type: return a zero-size AABB at the origin.
    return AABB([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


# =============================================================================
# Object world AABB (honours clips via surface sampling)
# =============================================================================

def _aabb_corners(aabb: AABB) -> np.ndarray:
    """Return the 8 corners of an AABB as an (8, 3) array."""
    lo = aabb.min
    hi = aabb.max
    return np.array([
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [hi[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], lo[1], hi[2]],
        [lo[0], hi[1], hi[2]],
        [hi[0], hi[1], hi[2]],
    ])


def _transform_points(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 4x4 matrix to an (n, 3) array of points."""
    n = points.shape[0]
    homog = np.hstack([points, np.ones((n, 1))])
    out = homog @ M.T
    return out[:, :3]


def _object_world_aabb_sampled(
    obj,
    world_matrix: np.ndarray,
    clips: List[Dict[str, Any]],
    points_per_element: int,
) -> AABB:
    """
    Sample each element's surface, apply the element transform, apply the
    object's clips, then apply the world matrix. Returns the AABB of the
    surviving points.

    For a closed element, the AABB of surface points equals the AABB of
    the solid volume. For a clipped element (a split half), the AABB of
    the visible surface is the tight bound of the half.
    """
    rng = np.random.default_rng(seed=1234)
    all_points_local = []

    for el in obj.elements:
        sampler = _SURFACE_SAMPLERS.get(el.element_type)
        if sampler is None:
            # Unknown element type: fall back to the primitive corner AABB
            # so the element still contributes to the object's bounds.
            local_aabb = element_local_aabb(el)
            local_corners = _aabb_corners(local_aabb)
            el_matrix = _assembly.trs_to_matrix(el.transform)
            corners_after_el = _transform_points(local_corners, el_matrix)
            all_points_local.append(corners_after_el)
            continue

        ok, _msg = el.validate()
        if not ok:
            continue

        pts_local = sampler(el.parameters, points_per_element, rng)
        pts_after_el = _apply_transform_shape(pts_local, el.transform)
        all_points_local.append(pts_after_el)

    if not all_points_local:
        c = world_matrix[:3, 3]
        return AABB(c, c)

    combined_local = np.vstack(all_points_local)

    # Apply clips in the object's local frame
    combined_local = _apply_clips_shape(combined_local, clips)
    if combined_local.size == 0:
        # Everything clipped away; return the object's world origin as a
        # degenerate point.
        c = world_matrix[:3, 3]
        return AABB(c, c)

    # Transform to world
    combined_world = _transform_points(combined_local, world_matrix)
    return AABB(combined_world.min(axis=0), combined_world.max(axis=0))


def object_world_aabb(
    obj,
    world_matrix: np.ndarray,
    points_per_element: int = DEFAULT_AABB_SAMPLES_PER_ELEMENT,
) -> AABB:
    """
    Compute the world AABB of an object.

    When the shape similarity module is available and the object carries
    clips in metadata["clips"], this samples each element's surface,
    applies the element transform, filters by the clips, and takes the
    AABB of the surviving points. This gives the correct tight bounds for
    split objects.

    When the object has no clips or the samplers are unavailable, this
    falls back to the corner-based computation over the primitive element
    bounds, which is fast and exact for unclipped primitives.
    """
    if not obj.elements:
        c = world_matrix[:3, 3]
        return AABB(c, c)

    clips = None
    if isinstance(obj.metadata, dict):
        clips = obj.metadata.get("clips")

    if _SHAPE_SIM_AVAILABLE and clips:
        return _object_world_aabb_sampled(
            obj, world_matrix, clips, points_per_element
        )

    # No clips or no samplers available: corner-based AABB of all elements.
    all_corners = []
    for el in obj.elements:
        local_aabb = element_local_aabb(el)
        local_corners = _aabb_corners(local_aabb)
        el_matrix = _assembly.trs_to_matrix(el.transform)
        corners_after_el = _transform_points(local_corners, el_matrix)
        corners_world = _transform_points(corners_after_el, world_matrix)
        all_corners.append(corners_world)

    all_corners = np.vstack(all_corners)
    return AABB(all_corners.min(axis=0), all_corners.max(axis=0))


# =============================================================================
# Near threshold
# =============================================================================

def near_threshold(a: AABB, b: AABB) -> float:
    """
    Return the near-distance threshold for a pair of AABBs.

    The threshold is the mean of the two objects' maximum dimensions,
    multiplied by NEAR_THRESHOLD_MULTIPLIER. This makes near a relative
    notion: small objects are near within a small distance, large objects
    within a larger one.
    """
    mean_max_dim = (a.max_dim() + b.max_dim()) / 2.0
    return mean_max_dim * NEAR_THRESHOLD_MULTIPLIER


# =============================================================================
# Individual predicate computations
# =============================================================================

def _holds_above(a: AABB, b: AABB, margin: float = VERTICAL_MARGIN_CM) -> bool:
    return a.min[2] > b.max[2] + margin


def _holds_below(a: AABB, b: AABB, margin: float = VERTICAL_MARGIN_CM) -> bool:
    return a.max[2] < b.min[2] - margin


def _holds_left_of(a: AABB, b: AABB, margin: float = HORIZONTAL_MARGIN_CM) -> bool:
    return a.min[0] > b.max[0] + margin


def _holds_right_of(a: AABB, b: AABB, margin: float = HORIZONTAL_MARGIN_CM) -> bool:
    return a.max[0] < b.min[0] - margin


def _holds_front_of(a: AABB, b: AABB, margin: float = HORIZONTAL_MARGIN_CM) -> bool:
    return a.min[1] > b.max[1] + margin


def _holds_behind_of(a: AABB, b: AABB, margin: float = HORIZONTAL_MARGIN_CM) -> bool:
    return a.max[1] < b.min[1] - margin


def _holds_contact(a: AABB, b: AABB, tolerance: float = CONTACT_TOLERANCE_CM) -> bool:
    """Two AABBs are in contact if their distance is below the tolerance."""
    return a.distance_to(b) <= tolerance


def _holds_on(a: AABB, b: AABB) -> bool:
    """
    A is on B if:
      - A's bottom is at or just above B's top (within tolerance)
      - A and B overlap on the XZ plane
      - A is not contained in B
    """
    vertical_gap = a.min[2] - b.max[2]
    if not (-ON_VERTICAL_TOLERANCE_CM <= vertical_gap <= ON_VERTICAL_TOLERANCE_CM):
        return False
    if not (a.overlaps_along(0, b) and a.overlaps_along(1, b)):
        return False
    if b.contains(a, tolerance=0.0):
        return False
    return True


def _holds_in(a: AABB, b: AABB, tolerance: float = IN_TOLERANCE_CM) -> bool:
    """
    A is in B if A's AABB is contained in B's AABB (with tolerance), or
    A's centroid is inside B's AABB and A is smaller than B.

    A being "on" B takes precedence: if A rests on top of B, it is not
    inside B, even if the containment tolerance would otherwise classify
    a shallow object as contained. This prevents a flat object (a knife
    lying on a counter) from being incorrectly reported as "in" the
    counter just because its full vertical extent falls within the
    containment tolerance of the counter's top face.
    """
    if _holds_on(a, b):
        return False
    if b.contains(a, tolerance=tolerance):
        return True
    if a.max_dim() < b.max_dim():
        if b.contains_point(a.center(), tolerance=tolerance):
            return True
    return False


def _holds_near(a: AABB, b: AABB, threshold: Optional[float] = None) -> bool:
    if threshold is None:
        threshold = near_threshold(a, b)
    return a.distance_to(b) <= threshold


def _holds_apart(a: AABB, b: AABB, threshold: Optional[float] = None) -> bool:
    if threshold is None:
        threshold = near_threshold(a, b)
    return a.distance_to(b) > threshold


def _holds_around(a: AABB, b: AABB) -> bool:
    """
    A is around B if:
      - A and B overlap on all 3 axes (their AABBs intersect).
      - A's AABB spans B's AABB on at least AROUND_MIN_AXES axes
        (a.min < b.min and a.max > b.max on that axis).

    Examples:
        Hand  : x [-9, 9],  y [-6, 6],  z [78, 90]
        Apple : x [-3, 3],  y [-3, 3],  z [80, 86]
        Overlap on all axes : yes
        Span on x, y, z     : yes (3 axes)
        Result              : True

        Counter : x [-60, 60], y [-30, 30], z [75, 80]
        Apple   : x [6, 14],   y [1, 9],    z [80, 88]
        Overlap on z        : no (touch only at z=80)
        Result              : False
    """
    if not a.is_valid() or not b.is_valid():
        return False

    # AABBs must intersect on all axes
    for axis in range(3):
        if not (a.max[axis] > b.min[axis] and a.min[axis] < b.max[axis]):
            return False

    # At least AROUND_MIN_AXES axes must have A strictly spanning B
    span_count = 0
    for axis in range(3):
        if a.min[axis] < b.min[axis] and a.max[axis] > b.max[axis]:
            span_count += 1

    return span_count >= AROUND_MIN_AXES


def _holds_through(a: AABB, b: AABB) -> bool:
    """
    A is through B if the longest dimension of the overlap box (the part
    of A that is inside B) is at least THROUGH_MIN_OVERLAP_RATIO times
    B's smallest dimension.

    The 0.6 ratio means the overlap must cover at least 60% of B's
    thinnest cross-section. It is deliberately below 1.0 so that a knife
    that pierces most of the way through but not fully still counts, while
    a shallow poke does not.
    """
    overlap_min = np.maximum(a.min, b.min)
    overlap_max = np.minimum(a.max, b.max)
    overlap_size = overlap_max - overlap_min

    # Require a real 3D overlap. Touching on a face (zero extent on one
    # axis) does not count as "through".
    if np.any(overlap_size <= 1e-6):
        return False

    longest_overlap = float(np.max(overlap_size))
    b_smallest = float(np.min(b.size()))
    if b_smallest <= 0:
        return False

    return longest_overlap >= THROUGH_MIN_OVERLAP_RATIO * b_smallest


def _holds_between(a: AABB, b: AABB, c: AABB) -> bool:
    """
    A is between B and C if A's centroid lies between B's and C's on some
    axis, and A is near both.
    """
    if not _holds_near(a, b):
        return False
    if not _holds_near(a, c):
        return False

    center_a = a.center()
    center_b = b.center()
    center_c = c.center()
    margin = BETWEEN_MARGIN_CM

    for axis in range(3):
        lo = min(center_b[axis], center_c[axis]) - margin
        hi = max(center_b[axis], center_c[axis]) + margin
        if lo <= center_a[axis] <= hi:
            if abs(center_b[axis] - center_c[axis]) > margin:
                return True
    return False


def _holds_attached(
    a_id: str,
    b_id: str,
    hierarchy_info: Optional[Dict[str, Any]],
) -> bool:
    """
    A is attached to B if one is the parent of the other in the assembly
    hierarchy, or if both are children of the same parent.

    hierarchy_info is expected to be a dict with:
        "parents": {obj_id: parent_obj_id or None}
    """
    if not hierarchy_info:
        return False
    parents = hierarchy_info.get("parents", {})
    p_a = parents.get(a_id)
    p_b = parents.get(b_id)
    if p_a is not None and p_a == b_id:
        return True
    if p_b is not None and p_b == a_id:
        return True
    if p_a is not None and p_a == p_b:
        return True
    return False


# =============================================================================
# Full predicate computation
# =============================================================================

def compute_all_predicates(
    a: AABB,
    b: AABB,
    a_id: Optional[str] = None,
    b_id: Optional[str] = None,
    hierarchy_info: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Return every predicate that holds between AABBs a and b, as a list of
    dicts:
        {
            "predicate": "<name>",
            "source_id": a_id,
            "target_id": b_id,
            "magnitude": float or None,
        }

    Symmetric predicates are returned in a canonical direction (source is
    a_id). Callers that need both directions can flip based on
    is_symmetric. Directional predicates are returned as computed (a to b);
    callers that want the reverse (b to a) should call this function again
    with the arguments swapped.
    """
    results: List[Dict[str, Any]] = []

    # Contact
    if _holds_contact(a, b):
        results.append({
            "predicate": "contact",
            "magnitude": a.distance_to(b),
            "source_id": a_id,
            "target_id": b_id,
        })

    # Near / apart
    if _holds_near(a, b):
        results.append({
            "predicate": "near",
            "magnitude": a.distance_to(b),
            "source_id": a_id,
            "target_id": b_id,
        })
    elif _holds_apart(a, b):
        results.append({
            "predicate": "apart",
            "magnitude": a.distance_to(b),
            "source_id": a_id,
            "target_id": b_id,
        })

    # Above / below
    if _holds_above(a, b):
        results.append({
            "predicate": "above",
            "magnitude": float(a.min[2] - b.max[2]),
            "source_id": a_id,
            "target_id": b_id,
        })
    if _holds_below(a, b):
        results.append({
            "predicate": "below",
            "magnitude": float(b.min[2] - a.max[2]),
            "source_id": a_id,
            "target_id": b_id,
        })

    # On (directional)
    if _holds_on(a, b):
        results.append({
            "predicate": "on",
            "magnitude": float(a.min[2] - b.max[2]),
            "source_id": a_id,
            "target_id": b_id,
        })

    # In (directional)
    if _holds_in(a, b):
        results.append({
            "predicate": "in",
            "magnitude": None,
            "source_id": a_id,
            "target_id": b_id,
        })

    # Around (directional)
    if _holds_around(a, b):
        results.append({
            "predicate": "around",
            "magnitude": None,
            "source_id": a_id,
            "target_id": b_id,
        })

    # Through (directional)
    if _holds_through(a, b):
        results.append({
            "predicate": "through",
            "magnitude": None,
            "source_id": a_id,
            "target_id": b_id,
        })

    # Attached (symmetric, hierarchy-based)
    if a_id and b_id and hierarchy_info:
        if _holds_attached(a_id, b_id, hierarchy_info):
            results.append({
                "predicate": "attached",
                "magnitude": None,
                "source_id": a_id,
                "target_id": b_id,
            })

    return results


def predicate_holds(a: AABB, b: AABB, name: str) -> bool:
    """Single-predicate check for callers that only need one."""
    if name == "on":
        return _holds_on(a, b)
    if name == "in":
        return _holds_in(a, b)
    if name == "above":
        return _holds_above(a, b)
    if name == "below":
        return _holds_below(a, b)
    if name == "left_of":
        return _holds_left_of(a, b)
    if name == "right_of":
        return _holds_right_of(a, b)
    if name == "front_of":
        return _holds_front_of(a, b)
    if name == "behind_of":
        return _holds_behind_of(a, b)
    if name == "contact":
        return _holds_contact(a, b)
    if name == "near":
        return _holds_near(a, b)
    if name == "apart":
        return _holds_apart(a, b)
    if name == "around":
        return _holds_around(a, b)
    if name == "through":
        return _holds_through(a, b)
    raise ValueError(f"unknown predicate '{name}'")


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("3d_relation_predicates.py – smoke test (with clipped AABBs)")
    print("=" * 70)
    print(f"shape similarity samplers available: {_SHAPE_SIM_AVAILABLE}")

    # ---- 1. Near threshold is size-dependent ----
    print("\nNear threshold scales with object size:")
    small = AABB([0.0, 0.0, 0.0], [5.0, 5.0, 5.0])
    small2 = AABB([20.0, 0.0, 0.0], [25.0, 5.0, 5.0])
    large = AABB([0.0, 0.0, 0.0], [50.0, 50.0, 50.0])
    large2 = AABB([120.0, 0.0, 0.0], [170.0, 50.0, 50.0])
    print(f"  small vs small: threshold = {near_threshold(small, small2):.2f} cm, "
          f"distance = {small.distance_to(small2):.2f} cm, "
          f"near = {_holds_near(small, small2)}")
    print(f"  large vs large: threshold = {near_threshold(large, large2):.2f} cm, "
          f"distance = {large.distance_to(large2):.2f} cm, "
          f"near = {_holds_near(large, large2)}")

    # ---- 2. On / above / contact ----
    print("\nVertical stack: apple on counter, apple above board")
    counter = AABB([-60.0, -30.0, 75.0], [60.0, 30.0, 80.0])
    apple = AABB([-4.0, -4.0, 80.0], [4.0, 4.0, 88.0])
    board = AABB([-20.0, -20.0, 70.0], [20.0, 20.0, 72.0])

    print(f"  apple on counter:       {_holds_on(apple, counter)}")
    print(f"  apple above counter:    {_holds_above(apple, counter)}  (should be False)")
    print(f"  apple above board:      {_holds_above(apple, board)}")
    print(f"  apple contact counter:  {_holds_contact(apple, counter)}")

    # ---- 3. In: knife inside box ----
    print("\nContainment:")
    box = AABB([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0])
    knife = AABB([-4.0, -1.0, -0.5], [4.0, 1.0, 0.5])
    print(f"  knife in box:    {_holds_in(knife, box)}")
    print(f"  box in knife:    {_holds_in(box, knife)}")

    # ---- 4. Around: hand surrounds apple, box contains ball, counter does not ----
    print("\nAround:")
    hand = AABB([-9.0, -6.0, 78.0], [9.0, 6.0, 90.0])
    apple4 = AABB([-3.0, -3.0, 80.0], [3.0, 3.0, 86.0])
    print(f"  hand around apple:      {_holds_around(hand, apple4)}  (should be True)")

    big_box = AABB([-100.0, -100.0, 0.0], [100.0, 100.0, 200.0])
    ball = AABB([-3.0, -3.0, 100.0], [3.0, 3.0, 106.0])
    print(f"  big box around ball:    {_holds_around(big_box, ball)}  (should be True)")

    counter2 = AABB([-60.0, -30.0, 75.0], [60.0, 30.0, 80.0])
    apple5 = AABB([6.0, 1.0, 80.0], [14.0, 9.0, 88.0])
    print(f"  counter around apple:   {_holds_around(counter2, apple5)}  (should be False)")
    print(f"  apple around counter:   {_holds_around(apple5, counter2)}  (should be False)")

    # ---- 5. Through: knife through apple, counter under apple ----
    print("\nThrough:")
    knife2 = AABB([-8.0, -0.5, -0.5], [8.0, 0.5, 0.5])
    apple6 = AABB([-3.0, -3.0, -3.0], [3.0, 3.0, 3.0])
    print(f"  knife through apple:    {_holds_through(knife2, apple6)}  (should be True)")

    counter3 = AABB([-60.0, -30.0, 75.0], [60.0, 30.0, 80.0])
    apple7 = AABB([6.0, 1.0, 80.0], [14.0, 9.0, 88.0])
    print(f"  counter through apple:  {_holds_through(counter3, apple7)}  (should be False)")
    print(f"  apple through counter:  {_holds_through(apple7, counter3)}  (should be False)")

    # ---- 6. Between: pencil between two books ----
    print("\nBetween:")
    book_a = AABB([-20.0, -10.0, 0.0], [-10.0, 10.0, 20.0])
    book_b = AABB([10.0, -10.0, 0.0], [20.0, 10.0, 20.0])
    pencil = AABB([-1.0, -0.5, 8.0], [1.0, 0.5, 10.0])
    print(f"  pencil between books:   {_holds_between(pencil, book_a, book_b)}")

    # ---- 7. Attached from hierarchy ----
    print("\nAttached (from hierarchy):")
    hierarchy = {"parents": {"apple_stem_01": "apple_01", "apple_skin_01": "apple_01"}}
    print(f"  stem attached apple:    {_holds_attached('apple_stem_01', 'apple_01', hierarchy)}")
    print(f"  stem attached skin:     {_holds_attached('apple_stem_01', 'apple_skin_01', hierarchy)}")
    print(f"  stem attached knife:    {_holds_attached('apple_stem_01', 'knife_01', hierarchy)}")

    # ---- 8. Full predicate computation ----
    print("\nFull predicate computation (apple on counter):")
    preds = compute_all_predicates(
        apple5, counter2, a_id="apple_01", b_id="counter_01",
    )
    for p in preds:
        mag = f" (magnitude={p['magnitude']:.2f})" if p.get('magnitude') is not None else ""
        print(f"  {p['source_id']} {p['predicate']} {p['target_id']}{mag}")

    # ---- 9. Clipped AABB test ----
    print("\nClipped AABB test (half-sphere vs whole sphere):")
    if _SHAPE_SIM_AVAILABLE:
        make_element = _assembly.make_element
        make_transform = _assembly.make_transform
        Object3D = _assembly.Object3D

        whole = Object3D(
            object_id="whole_sphere",
            template_name="sphere",
            elements=[make_element("sphere", {"radius_cm": 5.0})],
        )
        top_half = Object3D(
            object_id="top_half",
            template_name="sphere",
            elements=[make_element("sphere", {"radius_cm": 5.0})],
            metadata={"clips": [{"point": [0.0, 0.0, 0.0],
                                 "normal": [0.0, 0.0, -1.0]}]},
        )
        bottom_half = Object3D(
            object_id="bottom_half",
            template_name="sphere",
            elements=[make_element("sphere", {"radius_cm": 5.0})],
            metadata={"clips": [{"point": [0.0, 0.0, 0.0],
                                 "normal": [0.0, 0.0, 1.0]}]},
        )

        identity = np.eye(4)

        aabb_whole = object_world_aabb(whole, identity)
        aabb_top = object_world_aabb(top_half, identity)
        aabb_bottom = object_world_aabb(bottom_half, identity)

        def fmt(aabb):
            return (f"min=({aabb.min[0]:6.2f},{aabb.min[1]:6.2f},{aabb.min[2]:6.2f}) "
                    f"max=({aabb.max[0]:6.2f},{aabb.max[1]:6.2f},{aabb.max[2]:6.2f})")

        print(f"  whole sphere : {fmt(aabb_whole)}")
        print(f"  top half     : {fmt(aabb_top)}  (z should start at 0)")
        print(f"  bottom half  : {fmt(aabb_bottom)}  (z should end at 0)")

        # Verify: top half z-min should be near 0 (the clip plane)
        # and z-max should be near 5 (the top of the sphere).
        top_ok = (abs(aabb_top.min[2]) < 0.2) and (abs(aabb_top.max[2] - 5.0) < 0.2)
        bottom_ok = (abs(aabb_bottom.min[2] + 5.0) < 0.2) and (abs(aabb_bottom.max[2]) < 0.2)
        whole_ok = (abs(aabb_whole.min[2] + 5.0) < 0.2) and (abs(aabb_whole.max[2] - 5.0) < 0.2)

        print(f"  whole OK: {whole_ok}")
        print(f"  top OK:   {top_ok}")
        print(f"  bottom OK:{bottom_ok}")

        if whole_ok and top_ok and bottom_ok:
            print("  → clipped AABBs are correct")
        else:
            print("  → WARNING: clipped AABBs are not as expected")
    else:
        print("  skipped (shape similarity module not importable)")

    print("\nSmoke test complete.")