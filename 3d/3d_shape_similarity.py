#!/usr/bin/env python3
r"""
3d_shape_similarity.py – Point cloud sampling and shape similarity.

Provides:
    - Surface sampling from element decompositions (per element_type)
    - Element transform application (translate, rotate, scale)
    - Clip filtering: reject sampled points on the hidden side of clip planes
    - Canonical alignment (PCA) so shape similarity is rotation-invariant
    - Sign-flip minimization so mirrored halves match
    - Point cloud normalization (bounding-box centre, max extent to 1)
    - Chamfer distance between two point clouds
    - Similarity score in [0, 1] derived from Chamfer distance
    - Disk cache for sampled point clouds, keyed on template_name plus a
      hash of the geometry and sampling resolution

Canonical alignment:
    Before comparison, both point clouds are rotated so their principal
    axes (PCA eigenvectors) align to the world axes. This makes the
    similarity score invariant to the original orientation of each object.
    A knife lying on its side matches a knife standing upright.

    For symmetric shapes (spheres, hemispheres, cylinders) the eigenvectors
    are not uniquely determined, because the covariance matrix has equal
    eigenvalues along the symmetric axes. In those cases any alignment
    consistent with the eigenvectors is as good as any other, and the
    sign-flip minimization handles the remaining ambiguity.

Sign-flip minimization:
    After canonical alignment, one cloud may be related to the other by a
    180-degree rotation about a principal axis (for example, the top half
    of a sphere versus the bottom half). We try the identity and the three
    180-degree rotations, and take the minimum Chamfer distance. This
    makes congruent shapes match exactly even when they are reflections
    of each other through a principal plane.

Sampling of templates (from the element storage) does not involve clips:
templates are whole shapes. Sampling of instances (from an Assembly3D's
Object3D) honours the object's metadata["clips"], so a split half of an
apple samples only the visible half.

Used by:
    - The planner, to replace string-based shape and dimension signals
      with a continuous geometric shape similarity
    - The 3d assembly generation stage, to keep instances of the same
      template geometrically consistent
    - Search and validation, to compare a query shape against the visible
      geometry of a memory object

Sampling is deterministic: the same element list plus the same n_points
always produces the same point cloud, because the RNG seed is derived
from the geometry hash.
"""

import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_assembly = importlib.import_module("3d_assembly")

Element3D = _assembly.Element3D
Transform = _assembly.Transform
ELEMENT_SCHEMA = _assembly.ELEMENT_SCHEMA
DATA_DIR = _assembly.DATA_DIR
POINTCLOUD_DIR = _assembly.POINTCLOUD_DIR
Object3D = _assembly.Object3D
Assembly3D = _assembly.Assembly3D
make_element = _assembly.make_element
make_transform = _assembly.make_transform

try:
    _elements = importlib.import_module("3d_elements_storage")
    Elements3DStorage = _elements.Elements3DStorage
    ElementEntry = _elements.ElementEntry
except Exception:
    Elements3DStorage = None
    ElementEntry = None


# =============================================================================
# Configuration
# =============================================================================
DEFAULT_N_POINTS = 512
DEFAULT_CHAMFER_SCALE = 3.0
DEFAULT_CACHE_ENABLED = True
CACHE_VERSION = "v1"


# =============================================================================
# Element samplers
# Each sampler has signature (params: dict, n: int, rng) -> np.ndarray (n, 3)
# =============================================================================

def _sample_sphere(params: Dict[str, float], n: int, rng) -> np.ndarray:
    r = params["radius_cm"]
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * r


def _sample_ellipsoid(params: Dict[str, float], n: int, rng) -> np.ndarray:
    rx = params["radius_x_cm"]
    ry = params["radius_y_cm"]
    rz = params["radius_z_cm"]
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * np.array([rx, ry, rz])


def _sample_box(params: Dict[str, float], n: int, rng) -> np.ndarray:
    sx = params["size_x_cm"]
    sy = params["size_y_cm"]
    sz = params["size_z_cm"]

    areas = np.array([sy * sz, sy * sz, sx * sz, sx * sz, sx * sy, sx * sy])
    probs = areas / areas.sum()
    faces = rng.choice(6, size=n, p=probs)
    pts = np.zeros((n, 3))

    for f in range(6):
        mask = faces == f
        m = int(mask.sum())
        if m == 0:
            continue
        u = rng.uniform(-1.0, 1.0, m)
        v = rng.uniform(-1.0, 1.0, m)
        if f == 0:
            pts[mask] = np.stack([np.full(m, sx / 2), u * sy / 2, v * sz / 2], axis=1)
        elif f == 1:
            pts[mask] = np.stack([np.full(m, -sx / 2), u * sy / 2, v * sz / 2], axis=1)
        elif f == 2:
            pts[mask] = np.stack([u * sx / 2, np.full(m, sy / 2), v * sz / 2], axis=1)
        elif f == 3:
            pts[mask] = np.stack([u * sx / 2, np.full(m, -sy / 2), v * sz / 2], axis=1)
        elif f == 4:
            pts[mask] = np.stack([u * sx / 2, v * sy / 2, np.full(m, sz / 2)], axis=1)
        elif f == 5:
            pts[mask] = np.stack([u * sx / 2, v * sy / 2, np.full(m, -sz / 2)], axis=1)

    return pts


def _sample_cylinder(params: Dict[str, float], n: int, rng) -> np.ndarray:
    r = params["radius_cm"]
    h = params["height_cm"]

    lateral_area = 2 * np.pi * r * h
    cap_area = 2 * np.pi * r * r
    total = lateral_area + cap_area
    p_lateral = lateral_area / total
    p_cap = cap_area / total

    kinds = rng.choice(3, size=n, p=[p_lateral, p_cap / 2, p_cap / 2])
    pts = np.zeros((n, 3))

    mask = kinds == 0
    m = int(mask.sum())
    if m > 0:
        theta = rng.uniform(0, 2 * np.pi, m)
        z = rng.uniform(-h / 2, h / 2, m)
        pts[mask] = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)

    mask = kinds == 1
    m = int(mask.sum())
    if m > 0:
        theta = rng.uniform(0, 2 * np.pi, m)
        rr = r * np.sqrt(rng.uniform(0, 1, m))
        pts[mask] = np.stack(
            [rr * np.cos(theta), rr * np.sin(theta), np.full(m, h / 2)], axis=1
        )

    mask = kinds == 2
    m = int(mask.sum())
    if m > 0:
        theta = rng.uniform(0, 2 * np.pi, m)
        rr = r * np.sqrt(rng.uniform(0, 1, m))
        pts[mask] = np.stack(
            [rr * np.cos(theta), rr * np.sin(theta), np.full(m, -h / 2)], axis=1
        )

    return pts


def _sample_cone(params: Dict[str, float], n: int, rng) -> np.ndarray:
    r = params["radius_cm"]
    h = params["height_cm"]

    slant_height = np.sqrt(r * r + h * h)
    lateral_area = np.pi * r * slant_height
    base_area = np.pi * r * r
    total = lateral_area + base_area
    p_lateral = lateral_area / total

    kinds = rng.choice(2, size=n, p=[p_lateral, 1 - p_lateral])
    pts = np.zeros((n, 3))

    mask = kinds == 0
    m = int(mask.sum())
    if m > 0:
        rf = np.sqrt(rng.uniform(0, 1, m))
        theta = rng.uniform(0, 2 * np.pi, m)
        rr = rf * r
        z = h / 2 - rf * h
        pts[mask] = np.stack([rr * np.cos(theta), rr * np.sin(theta), z], axis=1)

    mask = kinds == 1
    m = int(mask.sum())
    if m > 0:
        theta = rng.uniform(0, 2 * np.pi, m)
        rr = r * np.sqrt(rng.uniform(0, 1, m))
        pts[mask] = np.stack(
            [rr * np.cos(theta), rr * np.sin(theta), np.full(m, -h / 2)], axis=1
        )

    return pts


def _sample_torus(params: Dict[str, float], n: int, rng) -> np.ndarray:
    R = params["major_radius_cm"]
    r = params["minor_radius_cm"]
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return np.stack([x, y, z], axis=1)


def _sample_capsule(params: Dict[str, float], n: int, rng) -> np.ndarray:
    r = params["radius_cm"]
    h = params["height_cm"]

    body_area = 2 * np.pi * r * h
    hemi_area = 2 * np.pi * r * r * 2
    total = body_area + hemi_area
    p_body = body_area / total

    kinds = rng.choice(3, size=n, p=[p_body, (1 - p_body) / 2, (1 - p_body) / 2])
    pts = np.zeros((n, 3))

    mask = kinds == 0
    m = int(mask.sum())
    if m > 0:
        theta = rng.uniform(0, 2 * np.pi, m)
        z = rng.uniform(-h / 2, h / 2, m)
        pts[mask] = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)

    mask = kinds == 1
    m = int(mask.sum())
    if m > 0:
        cos_phi = rng.uniform(0, 1, m)
        sin_phi = np.sqrt(np.maximum(0, 1 - cos_phi * cos_phi))
        theta = rng.uniform(0, 2 * np.pi, m)
        pts[mask] = np.stack(
            [
                r * sin_phi * np.cos(theta),
                r * sin_phi * np.sin(theta),
                h / 2 + r * cos_phi,
            ],
            axis=1,
        )

    mask = kinds == 2
    m = int(mask.sum())
    if m > 0:
        cos_phi = rng.uniform(0, 1, m)
        sin_phi = np.sqrt(np.maximum(0, 1 - cos_phi * cos_phi))
        theta = rng.uniform(0, 2 * np.pi, m)
        pts[mask] = np.stack(
            [
                r * sin_phi * np.cos(theta),
                r * sin_phi * np.sin(theta),
                -h / 2 - r * cos_phi,
            ],
            axis=1,
        )

    return pts


def _sample_plane(params: Dict[str, float], n: int, rng) -> np.ndarray:
    sx = params.get("size_x_cm", 100.0) or 100.0
    sy = params.get("size_y_cm", 100.0) or 100.0
    x = rng.uniform(-sx / 2, sx / 2, n)
    y = rng.uniform(-sy / 2, sy / 2, n)
    z = np.zeros(n)
    return np.stack([x, y, z], axis=1)


def _sample_sheet(params: Dict[str, float], n: int, rng) -> np.ndarray:
    return _sample_box(
        {
            "size_x_cm": params["size_x_cm"],
            "size_y_cm": params["size_y_cm"],
            "size_z_cm": params["thickness_cm"],
        },
        n,
        rng,
    )


def _sample_wedge(params: Dict[str, float], n: int, rng) -> np.ndarray:
    sx = params["size_x_cm"]
    sy = params["size_y_cm"]
    sz = params["size_z_cm"]

    face_areas = np.array([
        sx * sz,
        sx * sz,
        sy * sz,
        sx * sy,
        sy * np.sqrt(sx * sx + sz * sz),
    ])
    probs = face_areas / face_areas.sum()
    faces = rng.choice(5, size=n, p=probs)
    pts = np.zeros((n, 3))

    for f in range(5):
        mask = faces == f
        m = int(mask.sum())
        if m == 0:
            continue
        u = rng.uniform(0, 1, m)
        v = rng.uniform(0, 1, m)
        if f == 0:
            a, b = np.minimum(u, v), np.maximum(u, v)
            pts[mask] = np.stack(
                [a * sx, np.full(m, -sy / 2), -sz / 2 + (1 - b) * sz], axis=1
            )
        elif f == 1:
            a, b = np.minimum(u, v), np.maximum(u, v)
            pts[mask] = np.stack(
                [a * sx, np.full(m, sy / 2), -sz / 2 + (1 - b) * sz], axis=1
            )
        elif f == 2:
            pts[mask] = np.stack(
                [u * sx, np.full(m, -sy / 2 + v * sy), -sz / 2 + u * sz], axis=1
            )
        elif f == 3:
            pts[mask] = np.stack(
                [u * sx, -sy / 2 + v * sy, np.full(m, -sz / 2)], axis=1
            )
        elif f == 4:
            pts[mask] = np.stack(
                [u * sx, np.full(m, sy / 2), -sz / 2 + v * sz], axis=1
            )

    return pts


def _sample_prism(params: Dict[str, float], n: int, rng) -> np.ndarray:
    sx = params["size_x_cm"]
    sy = params["size_y_cm"]
    num_sides = int(params["num_sides"])
    r = max(sx, sy) / 2.0
    h = sy

    lateral_area = num_sides * 2 * r * np.sin(np.pi / num_sides) * h
    cap_area = 2 * num_sides * 0.5 * r * r * np.sin(2 * np.pi / num_sides)
    total = lateral_area + cap_area
    p_lateral = lateral_area / total

    kinds = rng.choice(3, size=n, p=[p_lateral, (1 - p_lateral) / 2, (1 - p_lateral) / 2])
    pts = np.zeros((n, 3))

    mask = kinds == 0
    m = int(mask.sum())
    if m > 0:
        side = rng.integers(0, num_sides, m)
        t = rng.uniform(0, 1, m)
        theta1 = side * 2 * np.pi / num_sides
        theta2 = (side + 1) * 2 * np.pi / num_sides
        x1, y1 = r * np.cos(theta1), r * np.sin(theta1)
        x2, y2 = r * np.cos(theta2), r * np.sin(theta2)
        xs = x1 + t * (x2 - x1)
        ys = y1 + t * (y2 - y1)
        zs = rng.uniform(-h / 2, h / 2, m)
        pts[mask] = np.stack([xs, ys, zs], axis=1)

    for cap_idx, z_val in [(1, h / 2), (2, -h / 2)]:
        mask = kinds == cap_idx
        m = int(mask.sum())
        if m == 0:
            continue
        pts_local = np.zeros((m, 2))
        count = 0
        while count < m:
            batch = rng.uniform(-r, r, (m, 2))
            inside = (batch[:, 0] ** 2 + batch[:, 1] ** 2) <= r * r
            accepted = batch[inside]
            take = min(len(accepted), m - count)
            pts_local[count:count + take] = accepted[:take]
            count += take
        pts[mask] = np.stack(
            [pts_local[:, 0], pts_local[:, 1], np.full(m, z_val)], axis=1
        )

    return pts


def _sample_pyramid(params: Dict[str, float], n: int, rng) -> np.ndarray:
    bx = params["base_size_x_cm"]
    by = params["base_size_y_cm"]
    h = params["height_cm"]

    slant_x = np.sqrt(h * h + (bx / 2) ** 2)
    slant_y = np.sqrt(h * h + (by / 2) ** 2)
    side_area_x = 0.5 * by * slant_x
    side_area_y = 0.5 * bx * slant_y
    base_area = bx * by
    total = 2 * side_area_x + 2 * side_area_y + base_area

    probs = np.array([
        side_area_x, side_area_x, side_area_y, side_area_y, base_area,
    ]) / total
    faces = rng.choice(5, size=n, p=probs)
    pts = np.zeros((n, 3))

    apex = np.array([0.0, 0.0, h / 2])

    def sample_triangle(verts, m):
        r1 = np.sqrt(rng.uniform(0, 1, m))
        r2 = rng.uniform(0, 1, m)
        a, b, c = verts
        return (
            (1 - r1)[:, None] * a
            + (r1 * (1 - r2))[:, None] * b
            + (r1 * r2)[:, None] * c
        )

    for f in range(5):
        mask = faces == f
        m = int(mask.sum())
        if m == 0:
            continue
        if f == 0:
            verts = [apex, np.array([-bx / 2, -by / 2, -h / 2]),
                     np.array([bx / 2, -by / 2, -h / 2])]
        elif f == 1:
            verts = [apex, np.array([-bx / 2, by / 2, -h / 2]),
                     np.array([bx / 2, by / 2, -h / 2])]
        elif f == 2:
            verts = [apex, np.array([-bx / 2, -by / 2, -h / 2]),
                     np.array([-bx / 2, by / 2, -h / 2])]
        elif f == 3:
            verts = [apex, np.array([bx / 2, -by / 2, -h / 2]),
                     np.array([bx / 2, by / 2, -h / 2])]
        else:
            u = rng.uniform(-1, 1, m)
            v = rng.uniform(-1, 1, m)
            pts[mask] = np.stack(
                [u * bx / 2, v * by / 2, np.full(m, -h / 2)], axis=1
            )
            continue
        pts[mask] = sample_triangle(verts, m)

    return pts


def _sample_hemisphere(params: Dict[str, float], n: int, rng) -> np.ndarray:
    r = params["radius_cm"]
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v[:, 2] = np.abs(v[:, 2])
    return v * r


def _sample_frustum(params: Dict[str, float], n: int, rng) -> np.ndarray:
    rb = params["bottom_radius_cm"]
    rt = params["top_radius_cm"]
    h = params["height_cm"]

    slant_height = np.sqrt((rb - rt) ** 2 + h * h)
    lateral_area = np.pi * (rb + rt) * slant_height
    top_area = np.pi * rt * rt
    bottom_area = np.pi * rb * rb
    total = lateral_area + top_area + bottom_area

    probs = np.array([lateral_area, top_area, bottom_area]) / total
    kinds = rng.choice(3, size=n, p=probs)
    pts = np.zeros((n, 3))

    mask = kinds == 0
    m = int(mask.sum())
    if m > 0:
        rho = rng.uniform(0, 1, m)
        theta = rng.uniform(0, 2 * np.pi, m)
        rr = rb + rho * (rt - rb)
        z = -h / 2 + rho * h
        pts[mask] = np.stack([rr * np.cos(theta), rr * np.sin(theta), z], axis=1)

    mask = kinds == 1
    m = int(mask.sum())
    if m > 0 and rt > 0:
        theta = rng.uniform(0, 2 * np.pi, m)
        rr = rt * np.sqrt(rng.uniform(0, 1, m))
        pts[mask] = np.stack(
            [rr * np.cos(theta), rr * np.sin(theta), np.full(m, h / 2)], axis=1
        )

    mask = kinds == 2
    m = int(mask.sum())
    if m > 0:
        theta = rng.uniform(0, 2 * np.pi, m)
        rr = rb * np.sqrt(rng.uniform(0, 1, m))
        pts[mask] = np.stack(
            [rr * np.cos(theta), rr * np.sin(theta), np.full(m, -h / 2)], axis=1
        )

    return pts


_SAMPLERS = {
    "sphere": _sample_sphere,
    "ellipsoid": _sample_ellipsoid,
    "box": _sample_box,
    "cylinder": _sample_cylinder,
    "cone": _sample_cone,
    "torus": _sample_torus,
    "capsule": _sample_capsule,
    "plane": _sample_plane,
    "sheet": _sample_sheet,
    "wedge": _sample_wedge,
    "prism": _sample_prism,
    "pyramid": _sample_pyramid,
    "hemisphere": _sample_hemisphere,
    "frustum": _sample_frustum,
}


# =============================================================================
# Transform application
# =============================================================================

def _quaternion_to_matrix(q: List[float]) -> np.ndarray:
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _apply_transform(points: np.ndarray, transform: Transform) -> np.ndarray:
    pts = points * np.array(transform.scale)
    R = _quaternion_to_matrix(transform.rotate)
    pts = pts @ R.T
    pts = pts + np.array(transform.translate)
    return pts


# =============================================================================
# Clip filtering
# =============================================================================

def _apply_clips(points: np.ndarray, clips: List[Dict[str, Any]]) -> np.ndarray:
    """
    Reject points on the hidden side of any clip plane.

    Each clip is a dict with 'point' (a point on the plane, in the same
    frame as the input points) and 'normal' (the plane's outward normal).
    Points whose signed distance along the normal is positive are on the
    hidden side and are removed.

    Clips are applied in sequence, so a point must survive every clip.
    """
    if not clips or points.size == 0:
        return points

    mask = np.ones(points.shape[0], dtype=bool)
    for clip in clips:
        if not isinstance(clip, dict):
            continue
        p = np.array(clip.get("point", [0.0, 0.0, 0.0]), dtype=float)
        n = np.array(clip.get("normal", [0.0, 0.0, 1.0]), dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm == 0:
            continue
        n = n / n_norm
        signed = (points - p) @ n
        mask &= (signed <= 0)
        if not mask.any():
            break

    return points[mask]


# =============================================================================
# Canonical alignment (PCA) and sign-flip minimization
# =============================================================================

def _canonical_axes(points: np.ndarray) -> np.ndarray:
    """
    Return a 3x3 proper rotation matrix whose columns are the principal
    axes of the point cloud, ordered by descending eigenvalue.

    The eigenvectors of the covariance matrix give the object's natural
    axes. Sorting them by eigenvalue puts the longest extent first, and
    so on. The third axis is computed as the cross of the first two, so
    the result is guaranteed to be a proper rotation (determinant +1)
    rather than a reflection.
    """
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered / max(len(centered), 1)
    # eigh returns ascending eigenvalues; reverse for descending
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, ::-1]

    v1 = eigvecs[:, 0]
    v2 = eigvecs[:, 1]
    v3 = np.cross(v1, v2)
    # Cross of two orthonormal vectors gives a unit vector. But if the
    # two chosen eigenvectors are nearly parallel (degenerate eigenvalue
    # case), v3 can be small. Guard against that.
    if np.linalg.norm(v3) < 1e-6:
        # Pick any axis orthogonal to v1
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(v1[0]) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        v2 = np.cross(v1, arbitrary)
        v2 = v2 / np.linalg.norm(v2)
        v3 = np.cross(v1, v2)
    else:
        v3 = v3 / np.linalg.norm(v3)

    R = np.column_stack([v1, v2, v3])
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    return R


def _align_canonical(points: np.ndarray) -> np.ndarray:
    """
    Rotate a point cloud so its principal axes align to the world axes,
    then recentre and rescale using _normalize_pointcloud.
    """
    if points.size == 0:
        return points
    centroid = points.mean(axis=0)
    centered = points - centroid
    R = _canonical_axes(centered)
    aligned = centered @ R
    return _normalize_pointcloud(aligned)


def _chamfer_min_over_signs(A: np.ndarray, B: np.ndarray) -> float:
    """
    Return the minimum Chamfer distance between A and B across the four
    proper sign flips of B's principal axes.

    The four flips correspond to the identity and the three 180-degree
    rotations about the principal axes. Trying all four ensures that
    congruent shapes match even when one is a 180-degree rotation of the
    other (for example, the two halves of a sphere after a split).
    """
    best = _chamfer_distance(A, B)
    for axis in range(3):
        flipped = B.copy()
        # 180-degree rotation about `axis` flips the other two axes.
        for j in range(3):
            if j != axis:
                flipped[:, j] = -flipped[:, j]
        d = _chamfer_distance(A, flipped)
        if d < best:
            best = d
    return best


# =============================================================================
# Point cloud utilities
# =============================================================================

def _normalize_pointcloud(points: np.ndarray) -> np.ndarray:
    """
    Recentre a point cloud using the bounding-box centre, then scale so
    the maximum radial extent from the centre is 1.

    Using the bounding-box centre rather than the centroid makes mirrored
    shapes normalize to the same region of space, so their Chamfer
    distance reflects only genuine shape differences.
    """
    if points.size == 0:
        return points
    bb_min = points.min(axis=0)
    bb_max = points.max(axis=0)
    bb_center = (bb_min + bb_max) / 2.0
    pts = points - bb_center
    max_extent = np.max(np.linalg.norm(pts, axis=1))
    if max_extent > 0:
        pts = pts / max_extent
    return pts


def _chamfer_distance(A: np.ndarray, B: np.ndarray) -> float:
    if A.size == 0 or B.size == 0:
        return float("inf")
    diff = A[:, None, :] - B[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    min_ab = np.sqrt(d2.min(axis=1)).mean()
    min_ba = np.sqrt(d2.min(axis=0)).mean()
    return float(min_ab + min_ba)


def _similarity_from_chamfer(chamfer: float, scale: float) -> float:
    if chamfer == float("inf"):
        return 0.0
    return 1.0 / (1.0 + scale * chamfer)


# =============================================================================
# Geometry hashing
# =============================================================================

def _canonical_element_string(el: Element3D) -> str:
    params = "|".join(f"{k}={el.parameters[k]}" for k in sorted(el.parameters))
    t = el.transform
    tr = ",".join(f"{x:.6f}" for x in t.translate)
    ro = ",".join(f"{x:.6f}" for x in t.rotate)
    sc = ",".join(f"{x:.6f}" for x in t.scale)
    return f"{el.element_type}({params})[t={tr};r={ro};s={sc}]"


def _clips_hash(clips: Optional[List[Dict[str, Any]]]) -> str:
    if not clips:
        return ""
    parts = []
    for clip in clips:
        if not isinstance(clip, dict):
            continue
        p = ",".join(f"{x:.6f}" for x in clip.get("point", []))
        n = ",".join(f"{x:.6f}" for x in clip.get("normal", []))
        parts.append(f"p={p};n={n}")
    if not parts:
        return ""
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:12]


def _elements_hash(
    elements: List[Element3D],
    n_points: int,
    clips: Optional[List[Dict[str, Any]]] = None,
) -> str:
    parts = [f"v={CACHE_VERSION}", f"n={n_points}"]
    for el in elements:
        parts.append(_canonical_element_string(el))
    ch = _clips_hash(clips)
    if ch:
        parts.append(f"clips={ch}")
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return h[:24]


# =============================================================================
# ShapeSimilarity
# =============================================================================

class ShapeSimilarity:
    """
    Point cloud sampling and shape similarity for 3D element decompositions.

    Similarity is invariant to translation, uniform scale, and rotation
    thanks to canonical PCA alignment plus sign-flip minimization.

    Template-level sampling (from the element storage) is cached on disk.
    Instance-level sampling (from an Assembly3D Object3D) honours the
    object's clips and is not cached, since instance geometry is typically
    short-lived and varies by memory.
    """

    def __init__(
        self,
        storage: Any,
        cache_dir: Optional[Path] = None,
        n_points: int = DEFAULT_N_POINTS,
        chamfer_scale: float = DEFAULT_CHAMFER_SCALE,
        cache_enabled: bool = DEFAULT_CACHE_ENABLED,
    ):
        self.storage = storage
        self.n_points = n_points
        self.chamfer_scale = chamfer_scale
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir) if cache_dir is not None else POINTCLOUD_DIR
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API - template level
    # ------------------------------------------------------------------

    def sample_object_pointcloud(self, template_name: str) -> np.ndarray:
        """
        Return the point cloud for a template, sampling from the storage if
        not already cached. Templates are whole shapes; clips do not apply.
        Returns an empty array if the template is not in the storage or has
        no elements.
        """
        if template_name in self._memory_cache:
            return self._memory_cache[template_name]

        entry = self.storage.get_template(template_name) if self.storage else None
        if entry is None or not entry.elements:
            self._memory_cache[template_name] = np.zeros((0, 3))
            return self._memory_cache[template_name]

        h = _elements_hash(entry.elements, self.n_points, clips=None)
        cached = self._load_from_disk_cache(template_name, h)
        if cached is not None:
            self._memory_cache[template_name] = cached
            return cached

        pc = self.sample_from_elements(entry.elements, self.n_points, clips=None)
        self._memory_cache[template_name] = pc
        if self.cache_enabled:
            self._save_to_disk_cache(template_name, h, pc)
        return pc

    # ------------------------------------------------------------------
    # Public API - instance level (with clips)
    # ------------------------------------------------------------------

    def sample_object_instance(
        self,
        obj: Object3D,
        n_points: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample a point cloud from an Object3D instance. Honours the object's
        metadata["clips"] if present, so a split half samples only its
        visible geometry. Not cached (instances are memory-specific).
        """
        clips = None
        if isinstance(obj.metadata, dict):
            clips = obj.metadata.get("clips")
        return self.sample_from_elements(obj.elements, n_points, clips=clips)

    def sample_from_elements(
        self,
        elements: List[Element3D],
        n_points: Optional[int] = None,
        clips: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """
        Sample a point cloud from an ad-hoc element list, with optional
        clip filtering. Sampling is deterministic given the same element
        list, n, and clips.
        """
        if n_points is None:
            n_points = self.n_points
        if not elements:
            return np.zeros((0, 3))

        seed_str = "||".join(_canonical_element_string(el) for el in elements)
        seed = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        n_el = len(elements)
        per_el = max(1, n_points // n_el)
        remainder = n_points - per_el * n_el

        parts: List[np.ndarray] = []
        for i, el in enumerate(elements):
            sampler = _SAMPLERS.get(el.element_type)
            if sampler is None:
                continue
            ok, msg = el.validate()
            if not ok:
                continue
            n_this = per_el + (1 if i < remainder else 0)
            pts_local = sampler(el.parameters, n_this, rng)
            pts_world = _apply_transform(pts_local, el.transform)
            parts.append(pts_world)

        if not parts:
            return np.zeros((0, 3))

        combined = np.concatenate(parts, axis=0)

        if clips:
            combined = _apply_clips(combined, clips)

        return combined

    # ------------------------------------------------------------------
    # Public API - similarity
    # ------------------------------------------------------------------

    def shape_similarity(self, template_a: str, template_b: str) -> float:
        """Similarity in [0, 1] between two templates."""
        pc_a = self.sample_object_pointcloud(template_a)
        pc_b = self.sample_object_pointcloud(template_b)
        return self._compare_pointclouds(pc_a, pc_b)

    def shape_similarity_instances(
        self,
        obj_a: Object3D,
        obj_b: Object3D,
    ) -> float:
        """
        Similarity in [0, 1] between two Object3D instances, honouring
        their clips. Two halves of the same apple match at 1.0 after
        canonical alignment and sign-flip minimization.
        """
        pc_a = self.sample_object_instance(obj_a)
        pc_b = self.sample_object_instance(obj_b)
        return self._compare_pointclouds(pc_a, pc_b)

    def shape_similarity_from_elements(
        self,
        elements_a: List[Element3D],
        elements_b: List[Element3D],
        clips_a: Optional[List[Dict[str, Any]]] = None,
        clips_b: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """Similarity between two ad-hoc element lists, with optional clips."""
        pc_a = self.sample_from_elements(elements_a, clips=clips_a)
        pc_b = self.sample_from_elements(elements_b, clips=clips_b)
        return self._compare_pointclouds(pc_a, pc_b)

    def chamfer(self, template_a: str, template_b: str) -> float:
        """
        Return the canonical-aligned, sign-flip-minimized Chamfer distance
        between two templates' clouds.
        """
        pc_a = self.sample_object_pointcloud(template_a)
        pc_b = self.sample_object_pointcloud(template_b)
        if pc_a.size == 0 or pc_b.size == 0:
            return float("inf")
        na = _align_canonical(pc_a)
        nb = _align_canonical(pc_b)
        return _chamfer_min_over_signs(na, nb)

    # ------------------------------------------------------------------
    # Internal comparison
    # ------------------------------------------------------------------

    def _compare_pointclouds(self, pc_a: np.ndarray, pc_b: np.ndarray) -> float:
        if pc_a.size == 0 or pc_b.size == 0:
            return 0.0
        na = _align_canonical(pc_a)
        nb = _align_canonical(pc_b)
        chamfer = _chamfer_min_over_signs(na, nb)
        return _similarity_from_chamfer(chamfer, self.chamfer_scale)

    # ------------------------------------------------------------------
    # Disk cache
    # ------------------------------------------------------------------

    def _cache_path(self, template_name: str) -> Path:
        safe = template_name.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe}.npz"

    def _load_from_disk_cache(
        self,
        template_name: str,
        expected_hash: str,
    ) -> Optional[np.ndarray]:
        path = self._cache_path(template_name)
        if not path.exists():
            return None
        try:
            data = np.load(path, allow_pickle=False)
            stored_hash = str(data["hash"])
            if stored_hash != expected_hash:
                return None
            return data["points"]
        except Exception:
            return None

    def _save_to_disk_cache(
        self,
        template_name: str,
        h: str,
        points: np.ndarray,
    ):
        path = self._cache_path(template_name)
        try:
            np.savez_compressed(path, hash=h, points=points)
        except Exception as e:
            print(f"Warning: failed to cache point cloud for '{template_name}': {e}")

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_memory_cache(self):
        self._memory_cache.clear()

    def clear_disk_cache(self):
        if not self.cache_dir.exists():
            return
        for p in self.cache_dir.glob("*.npz"):
            try:
                p.unlink()
            except Exception:
                pass

    def invalidate(self, template_name: str):
        self._memory_cache.pop(template_name, None)
        path = self._cache_path(template_name)
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass


# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("3d_shape_similarity.py – smoke test (canonical alignment + clip)")
    print("=" * 70)

    if Elements3DStorage is None:
        print("3d_elements_storage not importable; cannot run full smoke test.")
        raise SystemExit(1)

    storage = Elements3DStorage()

    storage.add_template(ElementEntry(
        template_name="apple",
        elements=[make_element("sphere", {"radius_cm": 4.0})],
    ))
    storage.add_template(ElementEntry(
        template_name="ball",
        elements=[make_element("sphere", {"radius_cm": 4.0})],
    ))
    storage.add_template(ElementEntry(
        template_name="box",
        elements=[make_element("box", {"size_x_cm": 10.0, "size_y_cm": 10.0, "size_z_cm": 10.0})],
    ))
    storage.add_template(ElementEntry(
        template_name="rod",
        elements=[make_element("cylinder", {"radius_cm": 0.5, "height_cm": 20.0})],
    ))
    storage.add_template(ElementEntry(
        template_name="rod_rotated",
        elements=[
            make_element(
                "cylinder",
                {"radius_cm": 0.5, "height_cm": 20.0},
                transform=make_transform(
                    rotate=[0.0, 0.7071068, 0.0, 0.7071068],  # 90 deg about Y
                ),
            ),
        ],
    ))

    print(f"\n{storage.summary()}")

    sim = ShapeSimilarity(
        storage,
        cache_dir=Path(__file__).parent / "data" / "pointclouds_test",
        n_points=512,
    )

    print("\nPoint cloud shapes (template level):")
    for name in ["apple", "ball", "box", "rod", "rod_rotated"]:
        pc = sim.sample_object_pointcloud(name)
        if pc.size == 0:
            print(f"  {name}: EMPTY")
            continue
        extent = np.max(np.linalg.norm(pc - pc.mean(axis=0), axis=1))
        print(f"  {name}: {pc.shape}, extent={extent:.2f}")

    print("\nSimilarities (template level):")
    pairs = [
        ("apple", "ball"),
        ("apple", "box"),
        ("apple", "rod"),
        ("box",   "rod"),
        ("apple", "apple"),
        ("rod",   "rod_rotated"),   # rotation invariance test
    ]
    for a, b in pairs:
        s = sim.shape_similarity(a, b)
        c = sim.chamfer(a, b)
        print(f"  {a:12s} vs {b:12s}: similarity={s:.4f}  chamfer={c:.4f}")

    print("\nDeterminism check:")
    pc1 = sim.sample_from_elements(storage.get_elements("apple"))
    pc2 = sim.sample_from_elements(storage.get_elements("apple"))
    print(f"  identical arrays: {np.array_equal(pc1, pc2)}")

    # ---- Clip test ----
    print("\nClip test: half apple vs whole apple vs half ball")
    apple_obj = Object3D(
        object_id="apple_01",
        template_name="apple",
        elements=[make_element("sphere", {"radius_cm": 4.0})],
    )
    apple_half_top = Object3D(
        object_id="apple_top",
        template_name="apple",
        elements=[make_element("sphere", {"radius_cm": 4.0})],
        metadata={"clips": [{"point": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0]}]},
    )
    apple_half_bottom = Object3D(
        object_id="apple_bottom",
        template_name="apple",
        elements=[make_element("sphere", {"radius_cm": 4.0})],
        metadata={"clips": [{"point": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, -1.0]}]},
    )
    ball_half_top = Object3D(
        object_id="ball_top",
        template_name="ball",
        elements=[make_element("sphere", {"radius_cm": 4.0})],
        metadata={"clips": [{"point": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0]}]},
    )

    for label, obj in [
        ("apple whole", apple_obj),
        ("apple top half", apple_half_top),
        ("apple bottom half", apple_half_bottom),
    ]:
        pc = sim.sample_object_instance(obj)
        print(f"  {label:20s}: {pc.shape[0]} points")

    s_whole_vs_top = sim.shape_similarity_instances(apple_obj, apple_half_top)
    s_top_vs_bottom = sim.shape_similarity_instances(apple_half_top, apple_half_bottom)
    s_top_vs_ball_half = sim.shape_similarity_instances(apple_half_top, ball_half_top)

    print(f"\n  whole apple vs top half:      {s_whole_vs_top:.4f}  (expected moderate)")
    print(f"  top half vs bottom half:      {s_top_vs_bottom:.4f}  (expected ~1.0)")
    print(f"  apple top half vs ball half:  {s_top_vs_ball_half:.4f}  (expected 1.0)")

    print("\nCache check:")
    sim.clear_memory_cache()
    pc_cached = sim.sample_object_pointcloud("apple")
    print(f"  reload from disk cache: {pc_cached.shape}")

    sim.clear_disk_cache()
    print("\nSmoke test complete.")