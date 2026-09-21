#!/usr/bin/env python3
r"""
3d_render_assembly.py – Render a 3D assembly to a PNG.

Three backends:

    software    – z-buffered rasteriser written here in numpy. Exact depth
                  testing, no sorting, no occlusion bugs. Requires only
                  numpy and matplotlib (for saving the PNG). The default
                  backend for the pipeline.

    matplotlib  – best-effort painter's algorithm. Fast but can mis-sort
                  when a large object overlaps a small one. Kept as a
                  fallback.

    trimesh     – optional. Uses trimesh for geometry and pyrender (or
                  pyglet) for output. Best quality but requires installation.

Public API:

    render_assembly(assembly, output_path=None, backend="software",
                    mode="solid", show=False, axis=True, **kwargs)

    render_assembly_file(path, output_path=None, **kwargs)

Software backend specifics:

    - Standard pinhole camera: position, target, up vector, field of view.
    - View matrix (look-at), perspective projection matrix.
    - Triangles transformed to screen space and rasterised with a
      per-pixel z-buffer.
    - Lambertian shading with a fixed directional light.
    - Backface culling is disabled so thin objects (sheets, planes) show
      both sides.
    - Output size default 1000x800. Adjustable via image_width/image_height.

Matplotlib backend specifics:

    - Every triangle is added to a single Poly3DCollection in solid mode,
      so matplotlib's internal depth sort runs on individual triangles.
      This is the least-bad option available in matplotlib's 3D backend.
    - Wireframe mode is drawn with one collection per color; edges do not
      occlude each other.

Clipping:

    Object metadata["clips"] is honoured by _collect_triangles. Every
    backend receives already-clipped triangles.
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

Assembly3D = _assembly.Assembly3D
Object3D = _assembly.Object3D
Element3D = _assembly.Element3D
Transform = _assembly.Transform

MODULE_DIR = Path(__file__).parent
ASSEMBLIES_DIR = MODULE_DIR / "assemblies"
RENDERS_DIR = ASSEMBLIES_DIR / "renders"


# =============================================================================
# Optional backends
# =============================================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


# =============================================================================
# Color lookup
# =============================================================================
_COLOR_TABLE = {
    "red": (0.85, 0.20, 0.20),
    "red_brushed": (0.80, 0.22, 0.18),
    "green": (0.30, 0.75, 0.30),
    "blue": (0.25, 0.45, 0.85),
    "yellow": (0.95, 0.85, 0.20),
    "orange": (0.95, 0.55, 0.20),
    "orange_yellow": (0.95, 0.65, 0.25),
    "silver": (0.75, 0.75, 0.78),
    "gray": (0.55, 0.55, 0.55),
    "grey": (0.55, 0.55, 0.55),
    "black": (0.12, 0.12, 0.12),
    "white": (0.95, 0.95, 0.95),
    "brown": (0.55, 0.35, 0.20),
    "wood": (0.72, 0.55, 0.35),
    "flesh": (0.92, 0.78, 0.70),
    "cream": (0.96, 0.92, 0.82),
    "pale_white": (0.95, 0.94, 0.90),
    "steel": (0.65, 0.68, 0.72),
    "clear_yellowish": (0.95, 0.92, 0.70),
    "gold": (0.85, 0.70, 0.25),
}

_DEFAULT_COLOR = (0.6, 0.6, 0.6)


def _material_to_rgb(material: Dict[str, Any]) -> Tuple[float, float, float]:
    if not isinstance(material, dict):
        return _DEFAULT_COLOR
    color = material.get("color") or material.get("colour")
    if not isinstance(color, str):
        return _DEFAULT_COLOR
    key = color.strip().lower().replace(" ", "_")
    if key in _COLOR_TABLE:
        return _COLOR_TABLE[key]
    for k, v in _COLOR_TABLE.items():
        if k in key or key in k:
            return v
    return _DEFAULT_COLOR


# =============================================================================
# Mesh generation per element type
# =============================================================================

def _mesh_sphere(params: Dict[str, float], segments: int = 24, rings: int = 12):
    r = params["radius_cm"]
    verts = []
    for i in range(rings + 1):
        phi = math.pi * i / rings
        for j in range(segments):
            theta = 2 * math.pi * j / segments
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta)
            z = r * math.cos(phi)
            verts.append((x, y, z))
    faces = []
    for i in range(rings):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            c = (i + 1) * segments + (j + 1) % segments
            d = (i + 1) * segments + j
            faces.append((a, b, c, d))
    return np.array(verts), faces


def _mesh_ellipsoid(params: Dict[str, float], segments: int = 24, rings: int = 12):
    v, f = _mesh_sphere({"radius_cm": 1.0}, segments, rings)
    v = v * np.array([params["radius_x_cm"], params["radius_y_cm"], params["radius_z_cm"]])
    return v, f


def _mesh_box(params: Dict[str, float]):
    sx = params["size_x_cm"] / 2
    sy = params["size_y_cm"] / 2
    sz = params["size_z_cm"] / 2
    verts = np.array([
        [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
        [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz],
    ])
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return verts, faces


def _mesh_cylinder(params: Dict[str, float], segments: int = 24):
    r = params["radius_cm"]
    h = params["height_cm"]
    verts = []
    for j in range(segments):
        theta = 2 * math.pi * j / segments
        verts.append((r * math.cos(theta), r * math.sin(theta), -h / 2))
    for j in range(segments):
        theta = 2 * math.pi * j / segments
        verts.append((r * math.cos(theta), r * math.sin(theta), h / 2))
    verts.append((0, 0, -h / 2))
    verts.append((0, 0, h / 2))
    bottom_center = 2 * segments
    top_center = 2 * segments + 1
    faces = []
    for j in range(segments):
        a = j
        b = (j + 1) % segments
        c = segments + (j + 1) % segments
        d = segments + j
        faces.append((a, b, c, d))
        faces.append((a, b, bottom_center))
        faces.append((d, c, top_center))
    return np.array(verts), faces


def _mesh_cone(params: Dict[str, float], segments: int = 24):
    r = params["radius_cm"]
    h = params["height_cm"]
    verts = []
    for j in range(segments):
        theta = 2 * math.pi * j / segments
        verts.append((r * math.cos(theta), r * math.sin(theta), -h / 2))
    verts.append((0, 0, h / 2))
    verts.append((0, 0, -h / 2))
    apex = segments
    base_center = segments + 1
    faces = []
    for j in range(segments):
        a = j
        b = (j + 1) % segments
        faces.append((a, b, apex))
        faces.append((a, b, base_center))
    return np.array(verts), faces


def _mesh_torus(params: Dict[str, float], u_seg: int = 32, v_seg: int = 16):
    R = params["major_radius_cm"]
    r = params["minor_radius_cm"]
    verts = []
    for i in range(u_seg):
        u = 2 * math.pi * i / u_seg
        for j in range(v_seg):
            v = 2 * math.pi * j / v_seg
            x = (R + r * math.cos(v)) * math.cos(u)
            y = (R + r * math.cos(v)) * math.sin(u)
            z = r * math.sin(v)
            verts.append((x, y, z))
    faces = []
    for i in range(u_seg):
        for j in range(v_seg):
            a = i * v_seg + j
            b = i * v_seg + (j + 1) % v_seg
            c = ((i + 1) % u_seg) * v_seg + (j + 1) % v_seg
            d = ((i + 1) % u_seg) * v_seg + j
            faces.append((a, b, c, d))
    return np.array(verts), faces


def _mesh_capsule(params: Dict[str, float], segments: int = 24, rings: int = 8):
    r = params["radius_cm"]
    h = params["height_cm"]
    verts = []

    # Top hemisphere: equator at z=+h/2, pole at z=+h/2+r
    for i in range(rings + 1):
        phi = math.pi / 2 * i / rings
        for j in range(segments):
            theta = 2 * math.pi * j / segments
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta)
            z = h / 2 + r * math.cos(phi)
            verts.append((x, y, z))
    top_hemi_verts = (rings + 1) * segments

    # Bottom hemisphere: equator at z=-h/2, pole at z=-h/2-r
    for i in range(rings + 1):
        phi = math.pi / 2 * i / rings
        for j in range(segments):
            theta = 2 * math.pi * j / segments
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta)
            z = -h / 2 - r * math.cos(phi)
            verts.append((x, y, z))

    faces = []

    # Top hemisphere faces
    for i in range(rings):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            c = (i + 1) * segments + (j + 1) % segments
            d = (i + 1) * segments + j
            faces.append((a, b, c, d))

    # Bottom hemisphere faces
    base = top_hemi_verts
    for i in range(rings):
        for j in range(segments):
            a = base + i * segments + j
            b = base + i * segments + (j + 1) % segments
            c = base + (i + 1) * segments + (j + 1) % segments
            d = base + (i + 1) * segments + j
            faces.append((a, b, c, d))

    # Cylinder side band, connecting the two equator rings
    top_equator = rings * segments
    bottom_equator = base + rings * segments
    for j in range(segments):
        j1 = (j + 1) % segments
        a = top_equator + j
        b = top_equator + j1
        c = bottom_equator + j1
        d = bottom_equator + j
        faces.append((a, b, c, d))

    return np.array(verts), faces

def _mesh_plane(params: Dict[str, float]):
    sx = params.get("size_x_cm", 100.0) or 100.0
    sy = params.get("size_y_cm", 100.0) or 100.0
    verts = np.array([
        [-sx / 2, -sy / 2, 0], [sx / 2, -sy / 2, 0],
        [sx / 2, sy / 2, 0], [-sx / 2, sy / 2, 0],
    ])
    faces = [(0, 1, 2, 3)]
    return verts, faces


def _mesh_sheet(params: Dict[str, float]):
    return _mesh_box({
        "size_x_cm": params["size_x_cm"],
        "size_y_cm": params["size_y_cm"],
        "size_z_cm": params["thickness_cm"],
    })


def _mesh_wedge(params: Dict[str, float]):
    sx = params["size_x_cm"]
    sy = params["size_y_cm"]
    sz = params["size_z_cm"]
    verts = np.array([
        [-sx / 2, -sy / 2, -sz / 2],
        [sx / 2, -sy / 2, -sz / 2],
        [sx / 2, sy / 2, -sz / 2],
        [-sx / 2, sy / 2, -sz / 2],
        [-sx / 2, 0, sz / 2],
        [sx / 2, 0, sz / 2],
    ])
    faces = [
        (0, 1, 2, 3),
        (0, 1, 5, 4),
        (3, 2, 5, 4),
        (0, 4, 3),
        (1, 2, 5),
    ]
    return verts, faces


def _mesh_prism(params: Dict[str, float], segments: int = 12):
    n = int(params["num_sides"])
    r = max(params["size_x_cm"], params["size_y_cm"]) / 2.0
    h = params["size_y_cm"]
    verts = []
    for j in range(n):
        theta = 2 * math.pi * j / n
        verts.append((r * math.cos(theta), r * math.sin(theta), -h / 2))
    for j in range(n):
        theta = 2 * math.pi * j / n
        verts.append((r * math.cos(theta), r * math.sin(theta), h / 2))
    verts.append((0, 0, -h / 2))
    verts.append((0, 0, h / 2))
    bottom_center = 2 * n
    top_center = 2 * n + 1
    faces = []
    for j in range(n):
        a = j
        b = (j + 1) % n
        c = n + (j + 1) % n
        d = n + j
        faces.append((a, b, c, d))
        faces.append((a, b, bottom_center))
        faces.append((d, c, top_center))
    return np.array(verts), faces


def _mesh_pyramid(params: Dict[str, float]):
    bx = params["base_size_x_cm"]
    by = params["base_size_y_cm"]
    h = params["height_cm"]
    verts = np.array([
        [-bx / 2, -by / 2, -h / 2],
        [bx / 2, -by / 2, -h / 2],
        [bx / 2, by / 2, -h / 2],
        [-bx / 2, by / 2, -h / 2],
        [0, 0, h / 2],
    ])
    faces = [
        (0, 1, 2, 3),
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 0, 4),
    ]
    return verts, faces


def _mesh_hemisphere(params: Dict[str, float], segments: int = 24, rings: int = 12):
    r = params["radius_cm"]
    verts = []
    for i in range(rings + 1):
        phi = math.pi / 2 * i / rings
        for j in range(segments):
            theta = 2 * math.pi * j / segments
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta)
            z = r * math.cos(phi)
            verts.append((x, y, z))
    verts.append((0, 0, 0))
    bottom_center = len(verts) - 1
    faces = []
    for i in range(rings):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            c = (i + 1) * segments + (j + 1) % segments
            d = (i + 1) * segments + j
            faces.append((a, b, c, d))
    base = rings * segments
    for j in range(segments):
        a = base + j
        b = base + (j + 1) % segments
        faces.append((a, b, bottom_center))
    return np.array(verts), faces


def _mesh_frustum(params: Dict[str, float], segments: int = 24):
    rb = params["bottom_radius_cm"]
    rt = params["top_radius_cm"]
    h = params["height_cm"]
    verts = []
    for j in range(segments):
        theta = 2 * math.pi * j / segments
        verts.append((rb * math.cos(theta), rb * math.sin(theta), -h / 2))
    for j in range(segments):
        theta = 2 * math.pi * j / segments
        verts.append((rt * math.cos(theta), rt * math.sin(theta), h / 2))
    verts.append((0, 0, -h / 2))
    verts.append((0, 0, h / 2))
    bottom_center = 2 * segments
    top_center = 2 * segments + 1
    faces = []
    for j in range(segments):
        a = j
        b = (j + 1) % segments
        c = segments + (j + 1) % segments
        d = segments + j
        faces.append((a, b, c, d))
        faces.append((a, b, bottom_center))
        faces.append((d, c, top_center))
    return np.array(verts), faces


_MESH_BUILDERS = {
    "sphere": _mesh_sphere,
    "ellipsoid": _mesh_ellipsoid,
    "box": _mesh_box,
    "cylinder": _mesh_cylinder,
    "cone": _mesh_cone,
    "torus": _mesh_torus,
    "capsule": _mesh_capsule,
    "plane": _mesh_plane,
    "sheet": _mesh_sheet,
    "wedge": _mesh_wedge,
    "prism": _mesh_prism,
    "pyramid": _mesh_pyramid,
    "hemisphere": _mesh_hemisphere,
    "frustum": _mesh_frustum,
}


# =============================================================================
# Transform application
# =============================================================================

def _apply_transform(verts: np.ndarray, t: Transform) -> np.ndarray:
    v = verts * np.array(t.scale)
    R = _assembly._quaternion_to_matrix(t.rotate)
    v = v @ R.T
    v = v + np.array(t.translate)
    return v


def _apply_matrix(verts: np.ndarray, M: np.ndarray) -> np.ndarray:
    n = verts.shape[0]
    homog = np.hstack([verts, np.ones((n, 1))])
    transformed = homog @ M.T
    return transformed[:, :3]


# =============================================================================
# Triangle clipping (Sutherland-Hodgman against a single half-space)
# =============================================================================

def _clip_triangle(
    tri: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> List[np.ndarray]:
    n_norm = np.linalg.norm(plane_normal)
    if n_norm == 0:
        return [tri]
    n = plane_normal / n_norm
    p = plane_point

    d = [(v - p) @ n for v in tri]
    inside = [di <= 0 for di in d]

    if all(inside):
        return [tri]
    if not any(inside):
        return []

    out_verts: List[np.ndarray] = []
    for i in range(3):
        j = (i + 1) % 3
        if inside[i]:
            out_verts.append(tri[i].copy())
        if inside[i] != inside[j]:
            denom = d[i] - d[j]
            if abs(denom) < 1e-12:
                continue
            t = d[i] / denom
            out_verts.append(tri[i] + t * (tri[j] - tri[i]))

    if len(out_verts) < 3:
        return []

    if len(out_verts) == 3:
        return [np.array(out_verts)]
    return [
        np.array([out_verts[0], out_verts[1], out_verts[2]]),
        np.array([out_verts[0], out_verts[2], out_verts[3]]),
    ]


def _clip_triangles(
    triangles: List[np.ndarray],
    clips: List[Dict[str, Any]],
) -> List[np.ndarray]:
    if not clips:
        return triangles

    result = list(triangles)
    for clip in clips:
        if not isinstance(clip, dict):
            continue
        p = np.array(clip.get("point", [0.0, 0.0, 0.0]), dtype=float)
        n = np.array(clip.get("normal", [0.0, 0.0, 1.0]), dtype=float)
        if np.linalg.norm(n) == 0:
            continue
        next_result: List[np.ndarray] = []
        for tri in result:
            next_result.extend(_clip_triangle(tri, p, n))
        result = next_result
        if not result:
            break
    return result


# =============================================================================
# Scene assembly – collect world-space triangles
# =============================================================================

def _collect_triangles(
    assembly: Assembly3D,
) -> List[Tuple[np.ndarray, Tuple[float, float, float], str]]:
    """
    Return a list of (triangle_vertices (3,3), rgb, object_id) for every
    visible face of every element, with element transform, clip filtering,
    and object world transform applied.
    """
    triangles: List[Tuple[np.ndarray, Tuple[float, float, float], str]] = []
    world_matrices = assembly.get_all_world_matrices()

    for obj in assembly.objects:
        rgb = _material_to_rgb(obj.material)
        world_M = world_matrices.get(obj.object_id)
        if world_M is None:
            continue

        clips = None
        if isinstance(obj.metadata, dict):
            clips = obj.metadata.get("clips")

        for el in obj.elements:
            builder = _MESH_BUILDERS.get(el.element_type)
            if builder is None:
                continue
            try:
                verts, faces = builder(el.parameters)
            except Exception:
                continue

            verts = _apply_transform(verts, el.transform)

            el_rgb = rgb
            if el.material_override:
                el_rgb = _material_to_rgb(el.material_override)

            local_triangles: List[np.ndarray] = []
            for face in faces:
                if len(face) == 3:
                    local_triangles.append(verts[list(face)])
                elif len(face) == 4:
                    a, b, c, d = face
                    local_triangles.append(verts[[a, b, c]])
                    local_triangles.append(verts[[a, c, d]])

            if clips:
                local_triangles = _clip_triangles(local_triangles, clips)

            for tri in local_triangles:
                tri_world = _apply_matrix(tri, world_M)
                triangles.append((tri_world, el_rgb, obj.object_id))

    return triangles


# =============================================================================
# Software renderer (z-buffered rasteriser)
# =============================================================================

def _look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    f_norm = np.linalg.norm(forward)
    if f_norm < 1e-9:
        forward = np.array([0.0, 0.0, -1.0])
    else:
        forward = forward / f_norm
    right = np.cross(forward, up)
    r_norm = np.linalg.norm(right)
    if r_norm < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / r_norm
    true_up = np.cross(right, forward)
    M = np.eye(4)
    M[0, :3] = right
    M[1, :3] = true_up
    M[2, :3] = -forward
    M[0, 3] = -np.dot(right, eye)
    M[1, 3] = -np.dot(true_up, eye)
    M[2, 3] = np.dot(forward, eye)
    return M


def _perspective_matrix(fov_deg: float, aspect: float,
                        near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    M = np.zeros((4, 4))
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M


def _auto_camera_from_triangles(
    triangles: List[Tuple[np.ndarray, Tuple[float, float, float], str]],
    elevation_deg: float = 22.0,
    azimuth_deg: float = -50.0,
    distance_factor: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray]:
    all_verts = np.concatenate([t[0] for t in triangles], axis=0)
    center = all_verts.mean(axis=0)
    extent = np.max(all_verts, axis=0) - np.min(all_verts, axis=0)
    max_extent = float(np.max(extent))
    if max_extent <= 0:
        max_extent = 10.0
    distance = max_extent * distance_factor

    elev_rad = math.radians(elevation_deg)
    azim_rad = math.radians(azimuth_deg)
    direction = np.array([
        math.cos(elev_rad) * math.cos(azim_rad),
        math.cos(elev_rad) * math.sin(azim_rad),
        math.sin(elev_rad),
    ])
    eye = center + direction * distance
    return eye, center


def _shade_triangle(
    tri_world: np.ndarray,
    base_rgb: Tuple[float, float, float],
    light_dir: np.ndarray,
    ambient: float = 0.35,
    diffuse: float = 0.65,
) -> Tuple[float, float, float]:
    v0, v1, v2 = tri_world[0], tri_world[1], tri_world[2]
    n = np.cross(v1 - v0, v2 - v0)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        brightness = ambient
    else:
        n = n / n_norm
        ndotl = abs(float(np.dot(n, light_dir)))
        brightness = ambient + diffuse * ndotl
    return (
        min(1.0, base_rgb[0] * brightness),
        min(1.0, base_rgb[1] * brightness),
        min(1.0, base_rgb[2] * brightness),
    )


def _rasterize_triangle(
    screen_verts: np.ndarray,
    depths: np.ndarray,
    color: Tuple[float, float, float],
    zbuffer: np.ndarray,
    framebuffer: np.ndarray,
    width: int,
    height: int,
) -> None:
    v0, v1, v2 = screen_verts[0], screen_verts[1], screen_verts[2]
    z0, z1, z2 = depths[0], depths[1], depths[2]

    x_min = max(0, int(np.floor(min(v0[0], v1[0], v2[0]))))
    x_max = min(width - 1, int(np.ceil(max(v0[0], v1[0], v2[0]))))
    y_min = max(0, int(np.floor(min(v0[1], v1[1], v2[1]))))
    y_max = min(height - 1, int(np.ceil(max(v0[1], v1[1], v2[1]))))
    if x_min > x_max or y_min > y_max:
        return

    denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    if abs(denom) < 1e-12:
        return

    xs = np.arange(x_min, x_max + 1, dtype=np.float64) + 0.5
    ys = np.arange(y_min, y_max + 1, dtype=np.float64) + 0.5
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    l0 = ((v1[1] - v2[1]) * (XX - v2[0]) + (v2[0] - v1[0]) * (YY - v2[1])) / denom
    l1 = ((v2[1] - v0[1]) * (XX - v2[0]) + (v0[0] - v2[0]) * (YY - v2[1])) / denom
    l2 = 1.0 - l0 - l1

    mask = (l0 >= 0.0) & (l1 >= 0.0) & (l2 >= 0.0)
    if not mask.any():
        return

    depth = l0 * z0 + l1 * z1 + l2 * z2

    sub_zb = zbuffer[y_min:y_max + 1, x_min:x_max + 1]
    sub_fb = framebuffer[y_min:y_max + 1, x_min:x_max + 1]

    better = mask & (depth < sub_zb)
    if not better.any():
        return

    sub_zb[better] = depth[better]
    sub_fb[better] = np.array(color, dtype=np.float32)


def _render_software(
    assembly: Assembly3D,
    output_path: Optional[Path],
    mode: str = "solid",
    show: bool = False,
    axis: bool = True,
    image_width: int = 1000,
    image_height: int = 800,
    fov_deg: float = 45.0,
    camera_eye: Optional[List[float]] = None,
    camera_target: Optional[List[float]] = None,
    camera_up: Optional[List[float]] = None,
    light_direction: Optional[List[float]] = None,
    background_color: Tuple[float, float, float] = (0.97, 0.97, 0.97),
) -> Optional[Path]:
    """
    Z-buffered software rasteriser. Renders every triangle with exact
    per-pixel depth testing. No sorting, no occlusion bugs.
    """
    triangles = _collect_triangles(assembly)
    if not triangles:
        raise RuntimeError("no renderable geometry in assembly")

    width = int(image_width)
    height = int(image_height)

    # Camera
    if camera_eye is None or camera_target is None:
        auto_eye, auto_target = _auto_camera_from_triangles(triangles)
        eye = np.array(camera_eye, dtype=float) if camera_eye is not None else auto_eye
        target = np.array(camera_target, dtype=float) if camera_target is not None else auto_target
    else:
        eye = np.array(camera_eye, dtype=float)
        target = np.array(camera_target, dtype=float)

    up = np.array(camera_up, dtype=float) if camera_up is not None else np.array([0.0, 0.0, 1.0])

    # Light (directional, points toward the scene)
    if light_direction is not None:
        light = np.array(light_direction, dtype=float)
    else:
        light = np.array([0.3, -0.5, 0.8])
    light = light / (np.linalg.norm(light) or 1.0)

    view = _look_at_matrix(eye, target, up)
    aspect = width / float(height)
    proj = _perspective_matrix(fov_deg, aspect, near=0.1, far=100000.0)
    mvp = proj @ view

    # Framebuffers
    zbuffer = np.full((height, width), np.inf, dtype=np.float32)
    framebuffer = np.zeros((height, width, 3), dtype=np.float32)
    framebuffer[:, :] = np.array(background_color, dtype=np.float32)

    for tri_world, rgb, _obj_id in triangles:
        # Shade using world-space normal
        shaded = _shade_triangle(tri_world, rgb, light)

        # Transform to clip space
        homog = np.hstack([tri_world, np.ones((3, 1))])
        clip = (mvp @ homog.T).T  # shape (3, 4)

        # Reject if any vertex is at or behind the near plane
        if np.any(clip[:, 3] <= 1e-6):
            continue

        ndc = clip[:, :3] / clip[:, 3:4]  # shape (3, 3)

        # Discard if the triangle is fully outside the frustum
        if (ndc[:, 0] < -1.2).all() or (ndc[:, 0] > 1.2).all():
            continue
        if (ndc[:, 1] < -1.2).all() or (ndc[:, 1] > 1.2).all():
            continue
        if (ndc[:, 2] < -1.0).all() or (ndc[:, 2] > 1.0).all():
            continue

        screen = np.zeros((3, 2), dtype=np.float64)
        screen[:, 0] = (ndc[:, 0] + 1.0) * 0.5 * width
        screen[:, 1] = (1.0 - ndc[:, 1]) * 0.5 * height
        depths = ndc[:, 2]

        _rasterize_triangle(screen, depths, shaded, zbuffer, framebuffer, width, height)


    # Save PNG
    result_path: Optional[Path] = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_uint8 = (np.clip(framebuffer, 0.0, 1.0) * 255.0).astype(np.uint8)
        try:
            with open(output_path, "wb") as fh:
                plt.imsave(fh, image_uint8, format="png")
        except Exception:
            try:
                from PIL import Image
                Image.fromarray(image_uint8).save(str(output_path))
            except Exception:
                plt.imsave(str(output_path), image_uint8)
        result_path = output_path

    if show and MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(width / 100.0, height / 100.0))
        plt.imshow((np.clip(framebuffer, 0.0, 1.0) * 255.0).astype(np.uint8))
        plt.axis("off")
        try:
            plt.show()
        except Exception:
            pass
        plt.close()

    return result_path


# =============================================================================
# Matplotlib renderer (fallback, best-effort depth sorting)
# =============================================================================

def _render_matplotlib(
    assembly: Assembly3D,
    output_path: Optional[Path],
    mode: str = "solid",
    show: bool = False,
    axis: bool = True,
) -> Optional[Path]:
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is not installed")

    triangles = _collect_triangles(assembly)
    if not triangles:
        raise RuntimeError("no renderable geometry in assembly")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    all_verts = np.concatenate([t[0] for t in triangles], axis=0)
    center = all_verts.mean(axis=0)
    extent = np.max(all_verts, axis=0) - np.min(all_verts, axis=0)
    max_extent = float(np.max(extent))
    if max_extent <= 0:
        max_extent = 1.0

    half = max_extent / 2 * 1.05
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    # Solid mode: one collection with every triangle so matplotlib's
    # internal per-triangle depth sort runs. This is the least-bad option
    # available in matplotlib's 3D backend.
    if mode == "solid":
        poly = Poly3DCollection(
            [t[0] for t in triangles],
            facecolor=[t[1] for t in triangles],
            edgecolor=(0.15, 0.15, 0.15),
            linewidths=0.3,
            alpha=1.0,
        )
        poly.set_zsort("average")
        ax.add_collection3d(poly)
    else:
        by_color: Dict[Tuple[float, float, float], List[np.ndarray]] = {}
        for tri_verts, rgb, _obj_id in triangles:
            by_color.setdefault(rgb, []).append(tri_verts)
        for rgb, tri_list in by_color.items():
            poly = Poly3DCollection(
                tri_list,
                facecolor=rgb,
                edgecolor=(0.1, 0.1, 0.1),
                linewidths=0.6,
                alpha=0.0,
            )
            poly.set_zsort("average")
            ax.add_collection3d(poly)

    ax.set_box_aspect((1, 1, 1))

    if not axis:
        ax.set_axis_off()
    else:
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_zlabel("Z (cm)")
        ax.set_title(f"Assembly: {assembly.memory_id}\n"
                     f"{len(assembly.objects)} objects, "
                     f"{len(assembly.timeline)} keyframes")

    ax.view_init(elev=22, azim=-50)

    result_path: Optional[Path] = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        result_path = output_path

    if show:
        try:
            plt.show()
        except Exception:
            pass

    plt.close(fig)
    return result_path


# =============================================================================
# Trimesh renderer (optional)
# =============================================================================

def _render_trimesh(
    assembly: Assembly3D,
    output_path: Optional[Path],
    mode: str = "solid",
    show: bool = False,
    axis: bool = True,
) -> Optional[Path]:
    if not TRIMESH_AVAILABLE:
        raise RuntimeError("trimesh is not installed")

    scene = trimesh.Scene()
    world_matrices = assembly.get_all_world_matrices()

    for obj in assembly.objects:
        base_color = _material_to_rgb(obj.material)
        world_M = world_matrices.get(obj.object_id)
        if world_M is None:
            continue

        clips = None
        if isinstance(obj.metadata, dict):
            clips = obj.metadata.get("clips")

        for el in obj.elements:
            builder = _MESH_BUILDERS.get(el.element_type)
            if builder is None:
                continue
            try:
                verts, faces = builder(el.parameters)
            except Exception:
                continue

            el_mat = trimesh.transformations.quaternion_matrix(el.transform.rotate)
            el_mat[:3, 3] = el.transform.translate
            el_scale = np.diag(list(el.transform.scale) + [1.0])
            el_full = el_mat @ el_scale

            local_triangles: List[np.ndarray] = []
            for f in faces:
                if len(f) == 3:
                    local_triangles.append(verts[list(f)])
                elif len(f) == 4:
                    a, b, c, d = f
                    local_triangles.append(verts[[a, b, c]])
                    local_triangles.append(verts[[a, c, d]])

            transformed_tris: List[np.ndarray] = []
            for tri in local_triangles:
                homog = np.hstack([tri, np.ones((3, 1))])
                transformed = (el_full @ homog.T).T[:, :3]
                transformed_tris.append(transformed)

            if clips:
                transformed_tris = _clip_triangles(transformed_tris, clips)

            if not transformed_tris:
                continue

            color = base_color
            if el.material_override:
                color = _material_to_rgb(el.material_override)
            rgba = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 255)

            combined_verts = np.vstack(transformed_tris)
            n_tris = len(transformed_tris)
            combined_faces = np.array(
                [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(n_tris)],
                dtype=np.int64,
            )
            mesh = trimesh.Trimesh(
                vertices=combined_verts,
                faces=combined_faces,
                process=False,
            )
            mesh.apply_transform(world_M)
            mesh.visual.face_colors = rgba
            scene.add_geometry(mesh, node_name=f"{obj.object_id}.{el.element_type}")

    if not scene.geometry:
        raise RuntimeError("no renderable geometry in assembly")

    result_path: Optional[Path] = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        png = scene.save_image(resolution=(1000, 800), visible=True)
        output_path.write_bytes(png)
        result_path = output_path

    if show:
        try:
            scene.show()
        except Exception:
            pass

    return result_path


# =============================================================================
# Public API
# =============================================================================

def render_assembly(
    assembly: Assembly3D,
    output_path: Optional[Path] = None,
    backend: str = "software",
    mode: str = "solid",
    show: bool = False,
    axis: bool = True,
    **kwargs,
) -> Optional[Path]:
    """
    Render an assembly to a PNG.

    backend:
        "software"   – z-buffered rasteriser (default, exact)
        "matplotlib" – best-effort painter's algorithm
        "trimesh"    – trimesh-based, needs installation
        "auto"       – software if matplotlib is available, else trimesh

    kwargs are forwarded to the software backend (image_width, image_height,
    fov_deg, camera_eye, camera_target, camera_up, light_direction,
    background_color).
    """
    if backend == "software":
        return _render_software(
            assembly, output_path, mode=mode, show=show, axis=axis, **kwargs
        )
    if backend == "matplotlib":
        return _render_matplotlib(assembly, output_path, mode=mode, show=show, axis=axis)
    if backend == "trimesh":
        return _render_trimesh(assembly, output_path, mode=mode, show=show, axis=axis)
    if backend == "auto":
        if MATPLOTLIB_AVAILABLE:
            return _render_software(
                assembly, output_path, mode=mode, show=show, axis=axis, **kwargs
            )
        if TRIMESH_AVAILABLE:
            return _render_trimesh(assembly, output_path, mode=mode, show=show, axis=axis)
        raise RuntimeError("no rendering backend available")

    raise ValueError(f"unknown backend '{backend}'")


def render_assembly_file(
    path: Path,
    output_path: Optional[Path] = None,
    **kwargs,
) -> Optional[Path]:
    assembly = Assembly3D.load(Path(path))
    if output_path is None:
        RENDERS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RENDERS_DIR / f"{assembly.memory_id}.png"
    return render_assembly(assembly, output_path, **kwargs)


# =============================================================================
# Smoke test
# =============================================================================

def _smoke_test():
    print("=" * 70)
    print("3d_render_assembly.py – smoke test (software + fallbacks)")
    print("=" * 70)
    print(f"matplotlib available: {MATPLOTLIB_AVAILABLE}")
    print(f"trimesh available:    {TRIMESH_AVAILABLE}")

    if not MATPLOTLIB_AVAILABLE and not TRIMESH_AVAILABLE:
        print("No rendering backend available. Install matplotlib or trimesh.")
        return

    RENDERS_DIR.mkdir(parents=True, exist_ok=True)

    make_element = _assembly.make_element
    make_transform = _assembly.make_transform

    # Compact scene. All objects on the same scale so the camera framing
    # does not sacrifice one for another.
    counter = Object3D(
        object_id="counter_01",
        template_name="counter",
        elements=[
            make_element("box", {"size_x_cm": 40.0,
                                  "size_y_cm": 30.0,
                                  "size_z_cm": 3.0}),
        ],
        material={"color": "wood"},
        object_transform=make_transform(translate=[0.0, 0.0, 8.5]),
    )

    whole = Object3D(
        object_id="apple_whole",
        template_name="apple",
        elements=[make_element("sphere", {"radius_cm": 4.0})],
        material={"color": "red_brushed"},
        object_transform=make_transform(translate=[-12.0, 0.0, 14.0]),
    )

    top_half = Object3D(
        object_id="apple_top_half",
        template_name="apple",
        elements=[make_element("sphere", {"radius_cm": 4.0})],
        material={"color": "cream"},
        object_transform=make_transform(translate=[-2.0, 0.0, 14.0]),
        metadata={"clips": [{"point": [0.0, 0.0, 0.0],
                             "normal": [0.0, 0.0, -1.0]}]},
    )

    bottom_half = Object3D(
        object_id="apple_bottom_half",
        template_name="apple",
        elements=[make_element("sphere", {"radius_cm": 4.0})],
        material={"color": "cream"},
        object_transform=make_transform(translate=[8.0, 0.0, 14.0]),
        metadata={"clips": [{"point": [0.0, 0.0, 0.0],
                             "normal": [0.0, 0.0, 1.0]}]},
    )

    knife = Object3D(
        object_id="knife_01",
        template_name="knife",
        elements=[
            make_element(
                "box",
                {"size_x_cm": 12.0, "size_y_cm": 2.0, "size_z_cm": 0.3},
                transform=make_transform(translate=[-4.0, 0.0, 0.0]),
            ),
            make_element(
                "box",
                {"size_x_cm": 5.0, "size_y_cm": 2.5, "size_z_cm": 1.5},
                transform=make_transform(translate=[4.5, 0.0, 0.0]),
            ),
        ],
        material={"color": "silver"},
        object_transform=make_transform(translate=[4.0, 9.0, 10.2]),
    )

    assembly = Assembly3D(
        assembly_id="3d_render_smoke",
        memory_id="render_smoke",
        objects=[counter, whole, top_half, bottom_half, knife],
    )

    ok, msg = assembly.validate()
    print(f"  validation: {'OK' if ok else 'FAILED — ' + msg}")
    print(f"  {assembly.summary()}")

    print("\nWorld positions:")
    ws = assembly.get_all_world_matrices()
    for obj_id, M in ws.items():
        print(f"  {obj_id:24s} position = "
              f"({M[0,3]:7.2f}, {M[1,3]:7.2f}, {M[2,3]:7.2f})")

    triangles = _collect_triangles(assembly)
    print(f"\nTotal triangles: {len(triangles)}")
    for obj in assembly.objects:
        sub_asm = Assembly3D(
            assembly_id="sub", memory_id="sub", objects=[obj],
        )
        count = len(_collect_triangles(sub_asm))
        marker = ""
        if obj.object_id == "apple_whole":
            marker = "  (reference: full sphere)"
        elif "half" in obj.object_id:
            marker = "  (expected ~half of reference)"
        print(f"  {obj.object_id:24s}: {count} triangles{marker}")

    # ---- Software renderer (primary) ----
    if MATPLOTLIB_AVAILABLE:
        out = RENDERS_DIR / "smoke_test_software.png"
        result = render_assembly(
            assembly,
            output_path=out,
            backend="software",
            mode="solid",
            show=False,
            axis=True,
        )
        print(f"\nSoftware render -> {result}")
        if result and result.exists():
            print(f"  file size: {result.stat().st_size / 1024:.1f} KB")

    # ---- Matplotlib renderer (fallback) ----
    if MATPLOTLIB_AVAILABLE:
        out = RENDERS_DIR / "smoke_test_matplotlib.png"
        result = render_assembly(
            assembly,
            output_path=out,
            backend="matplotlib",
            mode="solid",
            show=False,
            axis=True,
        )
        print(f"Matplotlib render -> {result}")
        if result and result.exists():
            print(f"  file size: {result.stat().st_size / 1024:.1f} KB")

    # ---- Trimesh renderer (optional) ----
    if TRIMESH_AVAILABLE:
        out = RENDERS_DIR / "smoke_test_trimesh.png"
        try:
            result = render_assembly(
                assembly, output_path=out, backend="trimesh"
            )
            print(f"Trimesh render -> {result}")
        except Exception as e:
            print(f"Trimesh render failed: {e}")

    print("\nSmoke test complete.")


if __name__ == "__main__":
    _smoke_test()