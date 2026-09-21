#!/usr/bin/env python3
r"""
3d_assembly.py – Data model for the 3D module.

Provides the foundational classes for the 3D representation of a memory:

    Assembly3D        – the full 3D representation of one memory
    Object3D          – one object in the assembly, corresponding to an
                        object in the memory encoding (obj_id, template_name)
    Element3D         – one primitive element (sphere, box, cylinder, …)
                        that composes an object's geometry
    Transform         – translation / rotation (quaternion) / scale
    TimelineKeyframe  – a single segment in the assembly's timeline, linked
                        to an action instance in the memory encoding

Transform semantics (Option A):
    Each Object3D stores its object_transform as a LOCAL transform relative
    to its parent (or relative to world if it has no parent). World transforms
    are computed by composing the chain of local transforms from root to leaf.

    The composition is done in homogeneous 4x4 matrix space, which can
    represent shear exactly. Each local transform remains a readable TRS
    triple. When a composed world matrix contains shear (which happens only
    when a non-uniform scale sits above a rotated descendant), the object's
    stored TRS is a best-fit approximation and the exact matrix is kept in
    `world_matrix_override` for rendering and physics.

    The helper `Assembly3D.set_world_transform(object_id, world_transform)`
    accepts a world TRS from the LLM, computes the corresponding local TRS
    relative to the parent, stores it, and saves the exact world matrix as
    an override when the decomposition is lossy. The pipeline uses this
    method to convert LLM-authored world transforms into the runtime's
    local-transform representation.

Hierarchy consistency:
    Every object's `parent_object_id` is the authoritative statement of
    hierarchy. The `children` list on each Object3D is a derived field.
    `Assembly3D.__post_init__` calls `rebuild_children_lists()` to make the
    two consistent on construction. `add_object` and `remove_object` keep
    them consistent thereafter. `get_all_world_matrices` additionally
    derives a local children map from `parent_object_id` so that it remains
    correct even if the two fall out of sync for any reason.

Contacts:
    Each Assembly3D carries a `contacts` list. Each entry is a dict:

        {
            "object_a": str,
            "object_b": str,
            "contact_type": str,           # see CONTACT_TYPES in 3d_scene_operations
            "force": {...},                # placeholder for physics
            "friction": {...},             # placeholder for physics
            "contact_points": [...],       # optional list of world-frame points
        }

    Contacts are persistent: once started by a `contact` operation, they
    remain active until a `release` operation ends them. They are distinct
    from parent-child attachment, which is stored on the Object3D hierarchy
    (parent_object_id / children). See 3d_scene_operations.py for the
    contact-type-to-attachment convention table.

Ordering note:
    Both "rotate then scale" and "scale then rotate" orderings are accepted
    at the API level. Internally, trs_to_matrix applies scale first, then
    rotation, then translation. If a composed world matrix contains shear
    (i.e. cannot be expressed exactly as a single TRS triple), the exact
    matrix is preserved via world_matrix_override and the TRS triple is
    treated as a readable approximation. Downstream consumers that need
    exact geometry should use get_world_matrix, not get_world_transform.

Deferred for future work:
    - Shear handling in the local TRS is approximate. The exact composition
      is preserved via the world_matrix_override. A future revision can
      replace the local TRS with a full 4x4 matrix if shear becomes common
      enough to matter. See the note in `matrix_to_trs`.
    - Joint / constraint vocabulary for physics. Not yet implemented; the
      hierarchy currently implies fixed attachment between parent and child.
    - Physics material sub-dict (friction, restitution, damping) alongside
      the visual material dict. Not yet implemented.
    - Timeline mode flag (scripted / dynamic / kinematic). Not yet
      implemented; the current timeline is purely scripted.

Element types and their parameters (all lengths in cm):
    sphere, ellipsoid, box, cylinder, cone, torus, capsule, plane, sheet,
    wedge, prism, pyramid, hemisphere, frustum.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import copy
import json
import math

import numpy as np


# =============================================================================
# Paths
# =============================================================================
MODULE_DIR = Path(__file__).parent
DATA_DIR = MODULE_DIR / "data"
ASSEMBLIES_DIR = MODULE_DIR / "assemblies"
POINTCLOUD_DIR = DATA_DIR / "pointclouds"

MEMORIES_DIR = Path(r"F:\New folder (4)\New folder\Memories")


# =============================================================================
# Element schema
# =============================================================================
ELEMENT_SCHEMA: Dict[str, Dict[str, List[str]]] = {
    "sphere":     {"required": ["radius_cm"], "optional": []},
    "ellipsoid":  {"required": ["radius_x_cm", "radius_y_cm", "radius_z_cm"], "optional": []},
    "box":        {"required": ["size_x_cm", "size_y_cm", "size_z_cm"], "optional": []},
    "cylinder":   {"required": ["radius_cm", "height_cm"], "optional": []},
    "cone":       {"required": ["radius_cm", "height_cm"], "optional": []},
    "torus":      {"required": ["major_radius_cm", "minor_radius_cm"], "optional": []},
    "capsule":    {"required": ["radius_cm", "height_cm"], "optional": []},
    "plane":      {"required": [], "optional": ["size_x_cm", "size_y_cm"]},
    "sheet":      {"required": ["size_x_cm", "size_y_cm", "thickness_cm"], "optional": []},
    "wedge":      {"required": ["size_x_cm", "size_y_cm", "size_z_cm"], "optional": []},
    "prism":      {"required": ["size_x_cm", "size_y_cm", "num_sides"], "optional": []},
    "pyramid":    {"required": ["base_size_x_cm", "base_size_y_cm", "height_cm"], "optional": []},
    "hemisphere": {"required": ["radius_cm"], "optional": []},
    "frustum":    {"required": ["bottom_radius_cm", "top_radius_cm", "height_cm"], "optional": []},
}


# Contact types recognised by the assembly's validate() method. This mirrors
# the CONTACT_TYPES set in 3d_scene_operations.py. Duplicated here to avoid a
# circular import.
CONTACT_TYPES = {
    "touch",
    "grip",
    "hold",
    "push",
    "pull",
    "lean",
    "support",
    "rest_on",
}


# =============================================================================
# Transform
# =============================================================================
@dataclass
class Transform:
    """Translate / rotate / scale of an object or element.

    translate   : [x, y, z] in cm
    rotate      : quaternion [x, y, z, w]
    scale       : [sx, sy, sz] (default 1.0 each)
    """
    translate: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotate: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "translate": list(self.translate),
            "rotate": list(self.rotate),
            "scale": list(self.scale),
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Transform":
        if not isinstance(d, dict):
            return cls()
        return cls(
            translate=list(d.get("translate", [0.0, 0.0, 0.0])),
            rotate=list(d.get("rotate", [0.0, 0.0, 0.0, 1.0])),
            scale=list(d.get("scale", [1.0, 1.0, 1.0])),
        )

    @classmethod
    def identity(cls) -> "Transform":
        return cls()

    def is_identity(self) -> bool:
        return (
            self.translate == [0.0, 0.0, 0.0]
            and self.rotate == [0.0, 0.0, 0.0, 1.0]
            and self.scale == [1.0, 1.0, 1.0]
        )


# =============================================================================
# Matrix helpers (internal)
# =============================================================================

def _quaternion_to_matrix(q: List[float]) -> np.ndarray:
    """Quaternion [x, y, z, w] -> 3x3 rotation matrix."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_to_quaternion(R: np.ndarray) -> List[float]:
    """3x3 rotation matrix -> quaternion [x, y, z, w]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def trs_to_matrix(t: Transform) -> np.ndarray:
    """
    4x4 homogeneous matrix from a TRS transform.

    Order: scale first, then rotate, then translate.
    Upper-left 3x3 = R @ S, translation column = t.translate.

    This fixed order is an internal convention. Callers express transforms as
    a single TRS triple regardless of the conceptual order in which the LLM
    produced the scale and rotation. When the fixed order produces a matrix
    that cannot be decomposed back to a single TRS (i.e. contains shear
    relative to the parent frame), matrix_to_trs reports a nonzero residual
    and the caller preserves the exact matrix separately.
    """
    R = _quaternion_to_matrix(t.rotate)
    S = np.diag(t.scale)
    M = np.eye(4)
    M[:3, :3] = R @ S
    M[:3, 3] = np.array(t.translate, dtype=float)
    return M


def matrix_to_trs(M: np.ndarray) -> Tuple[Transform, float]:
    """
    Best-fit TRS decomposition of a 4x4 matrix, plus a residual.

    The residual is the Frobenius distance between the input's upper 3x3
    block after dividing out the scale, and the closest orthogonal matrix
    (via SVD). Zero residual means the decomposition is exact.

    If the residual is nonzero, the matrix contains shear, which cannot be
    expressed exactly as a single TRS triple. The TRS returned in that case
    is the closest approximation, and the caller is expected to preserve the
    exact matrix (for example via Object3D.world_matrix_override). Shear is
    treated as a deferred concern: the pipeline accepts both rotation-scale
    orderings and preserves exact geometry through the matrix override.
    """
    translate = [float(M[0, 3]), float(M[1, 3]), float(M[2, 3])]
    upper = M[:3, :3]

    sx = float(np.linalg.norm(upper[:, 0]))
    sy = float(np.linalg.norm(upper[:, 1]))
    sz = float(np.linalg.norm(upper[:, 2]))

    if sx == 0 or sy == 0 or sz == 0:
        return (
            Transform(translate=translate, rotate=[0.0, 0.0, 0.0, 1.0],
                      scale=[sx, sy, sz]),
            0.0,
        )

    R_unnorm = upper / np.array([sx, sy, sz])

    U, _, Vt = np.linalg.svd(R_unnorm)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        Vt[-1, :] *= -1.0
        sz *= -1.0
        R_ortho = U @ Vt

    residual = float(np.linalg.norm(R_unnorm - R_ortho, ord="fro"))

    quat = _matrix_to_quaternion(R_ortho)
    return Transform(translate=translate, rotate=quat, scale=[sx, sy, sz]), residual


def compose_matrices(parent_M: np.ndarray, local_M: np.ndarray) -> np.ndarray:
    """Compose a parent world matrix with a child local matrix."""
    return parent_M @ local_M


# =============================================================================
# Element3D
# =============================================================================
@dataclass
class Element3D:
    """One primitive element composing an object's geometry."""
    element_type: str
    parameters: Dict[str, float] = field(default_factory=dict)
    transform: Transform = field(default_factory=Transform)
    material_override: Optional[Dict[str, Any]] = None

    def validate(self) -> Tuple[bool, str]:
        schema = ELEMENT_SCHEMA.get(self.element_type)
        if schema is None:
            return False, f"unknown element_type '{self.element_type}'"

        for key in schema["required"]:
            if key not in self.parameters:
                return False, (
                    f"element_type '{self.element_type}' "
                    f"missing required parameter '{key}'"
                )
            val = self.parameters[key]
            if not isinstance(val, (int, float)):
                return False, (
                    f"parameter '{key}' of '{self.element_type}' "
                    f"must be a number, got {type(val).__name__}"
                )
            if key == "num_sides":
                if val < 3:
                    return False, f"parameter 'num_sides' must be >= 3, got {val}"
            else:
                if val <= 0:
                    return False, (
                        f"parameter '{key}' of '{self.element_type}' "
                        f"must be positive, got {val}"
                    )

        for key in self.parameters:
            if key in schema["required"] or key in schema["optional"]:
                continue
            return False, (
                f"unexpected parameter '{key}' for element_type '{self.element_type}'"
            )

        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "element_type": self.element_type,
            "parameters": dict(self.parameters),
            "transform": self.transform.to_dict(),
        }
        if self.material_override:
            d["material_override"] = dict(self.material_override)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Element3D":
        return cls(
            element_type=d["element_type"],
            parameters=dict(d.get("parameters", {})),
            transform=Transform.from_dict(d.get("transform")),
            material_override=d.get("material_override"),
        )


# =============================================================================
# Object3D
# =============================================================================
@dataclass
class Object3D:
    """One object in a 3D assembly.

    Corresponds 1:1 to an object in the memory encoding via object_id
    (the same obj_id used in the memory JSON). Composite objects nest via
    parent_object_id / children, mirroring the memory encoding's composite
    structure.

    object_transform is LOCAL to the parent (or world if no parent).
    world_matrix_override stores the exact 4x4 matrix when the local TRS
    decomposition was lossy (shear present). When set, it takes precedence
    over the local TRS for world-space computation.
    """
    object_id: str
    template_name: str
    elements: List[Element3D] = field(default_factory=list)
    material: Dict[str, Any] = field(default_factory=dict)
    object_transform: Transform = field(default_factory=Transform)
    mass_grams: Optional[float] = None
    parent_object_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    world_matrix_override: Optional[List[List[float]]] = None

    def validate(self) -> Tuple[bool, str]:
        if not self.object_id:
            return False, "object_id is empty"
        if not self.template_name:
            return False, f"template_name is empty for object '{self.object_id}'"
        if not self.elements:
            return False, f"object '{self.object_id}' has no elements"
        for i, el in enumerate(self.elements):
            ok, msg = el.validate()
            if not ok:
                return False, f"object '{self.object_id}' element {i}: {msg}"
        if self.mass_grams is not None and self.mass_grams <= 0:
            return False, f"object '{self.object_id}' mass must be positive"
        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "object_id": self.object_id,
            "template_name": self.template_name,
            "elements": [e.to_dict() for e in self.elements],
            "material": dict(self.material),
            "object_transform": self.object_transform.to_dict(),
        }
        if self.mass_grams is not None:
            d["mass_grams"] = self.mass_grams
        if self.parent_object_id is not None:
            d["parent_object_id"] = self.parent_object_id
        if self.children:
            d["children"] = list(self.children)
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        if self.world_matrix_override is not None:
            d["world_matrix_override"] = [list(row) for row in self.world_matrix_override]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Object3D":
        return cls(
            object_id=d["object_id"],
            template_name=d["template_name"],
            elements=[Element3D.from_dict(e) for e in d.get("elements", [])],
            material=dict(d.get("material", {})),
            object_transform=Transform.from_dict(d.get("object_transform")),
            mass_grams=d.get("mass_grams"),
            parent_object_id=d.get("parent_object_id"),
            children=list(d.get("children", [])),
            metadata=dict(d.get("metadata", {})),
            world_matrix_override=d.get("world_matrix_override"),
        )


# =============================================================================
# TimelineKeyframe
# =============================================================================
@dataclass
class TimelineKeyframe:
    """One segment in the assembly's timeline.

    Each keyframe corresponds to an action instance in the memory encoding
    via action_instance_id. The object_transforms map carries the transform
    of each named object at the start of this segment. These transforms are
    LOCAL to the object's parent (Option A).
    """
    t: float
    duration: float
    action_instance_id: Optional[str] = None
    action_template: Optional[str] = None
    object_transforms: Dict[str, Transform] = field(default_factory=dict)
    geometry_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "t": self.t,
            "duration": self.duration,
            "object_transforms": {
                k: v.to_dict() for k, v in self.object_transforms.items()
            },
        }
        if self.action_instance_id is not None:
            d["action_instance_id"] = self.action_instance_id
        if self.action_template is not None:
            d["action_template"] = self.action_template
        if self.geometry_events:
            d["geometry_events"] = list(self.geometry_events)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TimelineKeyframe":
        return cls(
            t=float(d.get("t", 0.0)),
            duration=float(d.get("duration", 0.0)),
            action_instance_id=d.get("action_instance_id"),
            action_template=d.get("action_template"),
            object_transforms={
                k: Transform.from_dict(v)
                for k, v in (d.get("object_transforms") or {}).items()
            },
            geometry_events=list(d.get("geometry_events", [])),
        )


# =============================================================================
# Assembly3D
# =============================================================================
@dataclass
class Assembly3D:
    """Full 3D representation of one memory."""
    assembly_id: str
    memory_id: str
    unit_system: str = "cm"
    coordinate_frame: str = "world"
    objects: List[Object3D] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    timeline: List[TimelineKeyframe] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    contacts: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        # Reconcile children lists from parent_object_id so that the
        # hierarchy is internally consistent from the moment of construction.
        self.rebuild_children_lists()

    # ------------------------------------------------------------------
    # Hierarchy consistency
    # ------------------------------------------------------------------
    def rebuild_children_lists(self):
        """
        Rebuild each object's `children` list from `parent_object_id`,
        which is the authoritative source of hierarchy.

        Any entries in a `children` list that do not have a matching
        `parent_object_id` back-reference are dropped. Any parent-child
        pairs declared only by `parent_object_id` are added.
        """
        by_id = {obj.object_id: obj for obj in self.objects}
        # Clear
        for obj in self.objects:
            obj.children = []
        # Rebuild from parent_object_id
        for obj in self.objects:
            if obj.parent_object_id is None:
                continue
            parent = by_id.get(obj.parent_object_id)
            if parent is None:
                continue
            if obj.object_id not in parent.children:
                parent.children.append(obj.object_id)

    # ------------------------------------------------------------------
    # Object access
    # ------------------------------------------------------------------
    def get_object(self, object_id: str) -> Optional[Object3D]:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    def has_object(self, object_id: str) -> bool:
        return any(o.object_id == object_id for o in self.objects)

    def get_root_objects(self) -> List[Object3D]:
        return [o for o in self.objects if o.parent_object_id is None]

    def get_children(self, object_id: str) -> List[Object3D]:
        return [o for o in self.objects if o.parent_object_id == object_id]

    def get_descendants(self, object_id: str) -> List[Object3D]:
        result: List[Object3D] = []
        stack = [object_id]
        while stack:
            current = stack.pop()
            for child in self.get_children(current):
                result.append(child)
                stack.append(child.object_id)
        return result

    def add_object(self, obj: Object3D):
        if self.has_object(obj.object_id):
            raise ValueError(f"object_id '{obj.object_id}' already exists in assembly")
        self.objects.append(obj)
        # Keep parent's children list in sync
        if obj.parent_object_id is not None:
            parent = self.get_object(obj.parent_object_id)
            if parent is not None and obj.object_id not in parent.children:
                parent.children.append(obj.object_id)

    def remove_object(self, object_id: str):
        obj = self.get_object(object_id)
        if obj is None:
            return
        # Remove from parent's children list
        if obj.parent_object_id is not None:
            parent = self.get_object(obj.parent_object_id)
            if parent and object_id in parent.children:
                parent.children.remove(object_id)
        # Orphan any children so we don't leave dangling parent_object_id
        for child_id in list(obj.children):
            child = self.get_object(child_id)
            if child is not None:
                child.parent_object_id = None
        self.objects = [o for o in self.objects if o.object_id != object_id]

    # ------------------------------------------------------------------
    # Contact access
    # ------------------------------------------------------------------
    def get_contacts(self) -> List[Dict[str, Any]]:
        """Return the list of active contacts."""
        return list(self.contacts)

    def has_contact(
        self,
        object_a: str,
        object_b: str,
        contact_type: Optional[str] = None,
    ) -> bool:
        """
        Return True if there is an active contact between the two objects,
        optionally filtered to a specific contact type.
        """
        for c in self.contacts:
            if c.get("object_a") == object_a and c.get("object_b") == object_b:
                if contact_type is None or c.get("contact_type") == contact_type:
                    return True
        return False

    def contacts_for(self, object_id: str) -> List[Dict[str, Any]]:
        """Return all contacts where the object appears as object_a or object_b."""
        return [
            c for c in self.contacts
            if c.get("object_a") == object_id or c.get("object_b") == object_id
        ]

    # ------------------------------------------------------------------
    # World transform computation
    # ------------------------------------------------------------------
    def get_world_matrix(self, object_id: str) -> np.ndarray:
        """
        Return the 4x4 world matrix for an object, composed through the
        parent chain. If the object has a world_matrix_override, that is
        returned instead (it takes precedence for exact shear preservation).
        """
        obj = self.get_object(object_id)
        if obj is None:
            raise KeyError(object_id)

        if obj.world_matrix_override is not None:
            return np.array(obj.world_matrix_override, dtype=float)

        local = trs_to_matrix(obj.object_transform)
        if obj.parent_object_id is None:
            return local
        parent_world = self.get_world_matrix(obj.parent_object_id)
        return compose_matrices(parent_world, local)

    def get_all_world_matrices(self) -> Dict[str, np.ndarray]:
        """
        Return a dict of object_id -> 4x4 world matrix for every object.
        Single DFS, O(N).

        The DFS derives the children map from each object's
        `parent_object_id` rather than from the `children` list, so the
        result is correct even if the two lists are inconsistent. This is
        a defensive measure; the assembly normally keeps them consistent
        via `rebuild_children_lists`.
        """
        # Derive adjacency from parent_object_id (authoritative).
        children_map: Dict[str, List[str]] = {}
        roots: List[str] = []
        for obj in self.objects:
            if obj.parent_object_id is None:
                roots.append(obj.object_id)
            else:
                children_map.setdefault(obj.parent_object_id, []).append(obj.object_id)

        result: Dict[str, np.ndarray] = {}

        def visit(obj_id: str, parent_matrix: Optional[np.ndarray]):
            obj = self.get_object(obj_id)
            if obj is None:
                return
            if obj.world_matrix_override is not None:
                world = np.array(obj.world_matrix_override, dtype=float)
            else:
                local = trs_to_matrix(obj.object_transform)
                world = local if parent_matrix is None else parent_matrix @ local
            result[obj_id] = world
            for child_id in children_map.get(obj_id, []):
                visit(child_id, world)

        for root_id in roots:
            visit(root_id, None)

        return result

    def get_world_transform(self, object_id: str) -> Transform:
        """
        Return a best-fit TRS triple for an object's world transform.
        When shear is present (world_matrix_override set), this is an
        approximation. Use get_world_matrix for exact values.
        """
        M = self.get_world_matrix(object_id)
        t, _ = matrix_to_trs(M)
        return t

    def get_world_transform_with_residual(self, object_id: str) -> Tuple[Transform, float]:
        """Return (best-fit TRS, residual). Residual > 0 indicates shear."""
        M = self.get_world_matrix(object_id)
        return matrix_to_trs(M)

    def get_all_world_transforms(self) -> Dict[str, Transform]:
        """Best-fit TRS triple for every object."""
        matrices = self.get_all_world_matrices()
        result: Dict[str, Transform] = {}
        for obj_id, M in matrices.items():
            t, _ = matrix_to_trs(M)
            result[obj_id] = t
        return result

    def set_world_transform(
        self,
        object_id: str,
        world_transform: Transform,
        tolerance: float = 1e-4,
    ) -> float:
        """
        Set an object's transform from a world-space TRS triple. The
        transform is converted to local (relative to parent), stored, and
        the exact world matrix is preserved as an override when the
        decomposition is lossy (residual > tolerance).

        Returns the residual. Zero means exact.
        """
        obj = self.get_object(object_id)
        if obj is None:
            raise KeyError(object_id)

        world_matrix = trs_to_matrix(world_transform)

        if obj.parent_object_id is None:
            obj.object_transform = Transform(
                translate=list(world_transform.translate),
                rotate=list(world_transform.rotate),
                scale=list(world_transform.scale),
            )
            obj.world_matrix_override = None
            return 0.0

        parent_world = self.get_world_matrix(obj.parent_object_id)
        parent_inv = np.linalg.inv(parent_world)
        local_matrix = parent_inv @ world_matrix

        local_transform, residual = matrix_to_trs(local_matrix)
        obj.object_transform = local_transform

        if residual > tolerance:
            obj.world_matrix_override = world_matrix.tolist()
        else:
            obj.world_matrix_override = None

        return residual

    # ------------------------------------------------------------------
    # Timeline access
    # ------------------------------------------------------------------
    def get_keyframe_for_action(self, action_instance_id: str) -> Optional[TimelineKeyframe]:
        for kf in self.timeline:
            if kf.action_instance_id == action_instance_id:
                return kf
        return None

    def total_duration(self) -> float:
        if not self.timeline:
            return 0.0
        last = max(self.timeline, key=lambda k: k.t + k.duration)
        return last.t + last.duration

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> Tuple[bool, str]:
        if not self.assembly_id:
            return False, "assembly_id is empty"
        if not self.memory_id:
            return False, "memory_id is empty"

        seen_ids = set()
        for obj in self.objects:
            if obj.object_id in seen_ids:
                return False, f"duplicate object_id '{obj.object_id}'"
            seen_ids.add(obj.object_id)
            ok, msg = obj.validate()
            if not ok:
                return False, msg

        for obj in self.objects:
            if obj.parent_object_id is not None:
                if obj.parent_object_id not in seen_ids:
                    return False, (
                        f"object '{obj.object_id}' has parent "
                        f"'{obj.parent_object_id}' which does not exist"
                    )
            for child_id in obj.children:
                if child_id not in seen_ids:
                    return False, (
                        f"object '{obj.object_id}' lists child "
                        f"'{child_id}' which does not exist"
                    )
                child = self.get_object(child_id)
                if child and child.parent_object_id != obj.object_id:
                    return False, (
                        f"object '{obj.object_id}' lists child '{child_id}' "
                        f"but that child's parent is '{child.parent_object_id}'"
                    )

        for kf in self.timeline:
            for obj_id in kf.object_transforms.keys():
                if obj_id not in seen_ids:
                    return False, (
                        f"timeline at t={kf.t} references object "
                        f"'{obj_id}' which does not exist"
                    )

        for i, c in enumerate(self.contacts):
            if not isinstance(c, dict):
                return False, f"contacts[{i}] is not a dict"
            obj_a = c.get("object_a")
            obj_b = c.get("object_b")
            ct = c.get("contact_type")
            if not obj_a or not obj_b:
                return False, (
                    f"contacts[{i}] is missing object_a or object_b"
                )
            if obj_a not in seen_ids:
                return False, (
                    f"contacts[{i}] references object_a '{obj_a}' "
                    f"which does not exist"
                )
            if obj_b not in seen_ids:
                return False, (
                    f"contacts[{i}] references object_b '{obj_b}' "
                    f"which does not exist"
                )
            if obj_a == obj_b:
                return False, (
                    f"contacts[{i}] has object_a == object_b ('{obj_a}')"
                )
            if ct not in CONTACT_TYPES:
                return False, (
                    f"contacts[{i}] has unknown contact_type '{ct}'; "
                    f"allowed: {sorted(CONTACT_TYPES)}"
                )

        return True, ""

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "assembly_id": self.assembly_id,
            "memory_id": self.memory_id,
            "unit_system": self.unit_system,
            "coordinate_frame": self.coordinate_frame,
            "objects": [o.to_dict() for o in self.objects],
            "timeline": [k.to_dict() for k in self.timeline],
        }
        if self.environment:
            d["environment"] = dict(self.environment)
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        if self.contacts:
            d["contacts"] = [dict(c) for c in self.contacts]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Assembly3D":
        return cls(
            assembly_id=d["assembly_id"],
            memory_id=d["memory_id"],
            unit_system=d.get("unit_system", "cm"),
            coordinate_frame=d.get("coordinate_frame", "world"),
            objects=[Object3D.from_dict(o) for o in d.get("objects", [])],
            environment=dict(d.get("environment", {})),
            timeline=[TimelineKeyframe.from_dict(k) for k in d.get("timeline", [])],
            metadata=dict(d.get("metadata", {})),
            contacts=[dict(c) for c in d.get("contacts", [])],
        )

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> Path:
        # Ensure hierarchy consistency before writing.
        self.rebuild_children_lists()
        if path is None:
            ASSEMBLIES_DIR.mkdir(parents=True, exist_ok=True)
            path = ASSEMBLIES_DIR / f"{self.memory_id}.3dassembly.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: Path) -> "Assembly3D":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "3d_assembly" in data:
            data = data["3d_assembly"]
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def copy(self) -> "Assembly3D":
        return Assembly3D.from_dict(copy.deepcopy(self.to_dict()))

    def summary(self) -> str:
        root_count = len(self.get_root_objects())
        total_count = len(self.objects)
        element_count = sum(len(o.elements) for o in self.objects)
        override_count = sum(
            1 for o in self.objects if o.world_matrix_override is not None
        )
        contact_count = len(self.contacts)
        extra = ""
        if override_count:
            extra += f", overrides={override_count}"
        if contact_count:
            extra += f", contacts={contact_count}"
        return (
            f"Assembly3D(memory_id={self.memory_id}, "
            f"objects={total_count} (roots={root_count}), "
            f"elements={element_count}, "
            f"timeline={len(self.timeline)} keyframes, "
            f"duration={self.total_duration():.2f}s{extra})"
        )


# =============================================================================
# Helpers
# =============================================================================
def make_assembly_id(memory_id: str) -> str:
    return f"3d_{memory_id}"


def make_transform(
    translate: Optional[List[float]] = None,
    rotate: Optional[List[float]] = None,
    scale: Optional[List[float]] = None,
) -> Transform:
    return Transform(
        translate=list(translate) if translate else [0.0, 0.0, 0.0],
        rotate=list(rotate) if rotate else [0.0, 0.0, 0.0, 1.0],
        scale=list(scale) if scale else [1.0, 1.0, 1.0],
    )


def make_element(
    element_type: str,
    parameters: Dict[str, float],
    transform: Optional[Transform] = None,
    material_override: Optional[Dict[str, Any]] = None,
) -> Element3D:
    return Element3D(
        element_type=element_type,
        parameters=dict(parameters),
        transform=transform if transform is not None else Transform.identity(),
        material_override=material_override,
    )


# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("3d_assembly.py – smoke test (with world transforms and contacts)")
    print("=" * 70)

    apple = Object3D(
        object_id="apple_01",
        template_name="apple",
        elements=[
            make_element("sphere", {"radius_cm": 4.0}),
        ],
        material={"color": "red_brushed"},
        object_transform=make_transform(translate=[10.0, 5.0, 84.0]),
        mass_grams=200.0,
    )
    stem = Object3D(
        object_id="apple_stem_01",
        template_name="apple_stem",
        elements=[
            make_element(
                "cylinder",
                {"radius_cm": 0.2, "height_cm": 2.0},
            ),
        ],
        object_transform=make_transform(translate=[0.0, 0.0, 4.5]),
        parent_object_id="apple_01",
    )
    skin = Object3D(
        object_id="apple_skin_01",
        template_name="apple_skin",
        elements=[
            make_element("sphere", {"radius_cm": 4.05}),
        ],
        object_transform=make_transform(),
        parent_object_id="apple_01",
    )
    hand = Object3D(
        object_id="left_hand_01",
        template_name="hand",
        elements=[
            make_element("box", {"size_x_cm": 8.0, "size_y_cm": 2.0, "size_z_cm": 18.0}),
        ],
        material={"color": "flesh"},
        object_transform=make_transform(translate=[-10.0, 5.0, 90.0]),
    )

    assembly = Assembly3D(
        assembly_id=make_assembly_id("mem_wt_test"),
        memory_id="mem_wt_test",
        objects=[apple, stem, skin, hand],
    )

    # __post_init__ should have rebuilt children lists
    print("\nTest 0a: children lists rebuilt from parent_object_id")
    print(f"  apple_01.children = {apple.children}")
    assert set(apple.children) == {"apple_stem_01", "apple_skin_01"}
    print(f"  apple_stem_01.children = {stem.children}")
    assert stem.children == []
    print("  → correct")

    # ---- Contact entries ----
    print("\nTest 0b: contacts field")
    assembly.contacts.append({
        "object_a": "left_hand_01",
        "object_b": "apple_01",
        "contact_type": "grip",
        "force": {"magnitude_n": 15.0, "direction_world": [0.0, 0.0, -1.0]},
        "friction": {"coefficient": 0.8, "regime": "static"},
        "contact_points": [],
    })
    assembly.contacts.append({
        "object_a": "left_hand_01",
        "object_b": "apple_01",
        "contact_type": "push",
        "force": {"magnitude_n": 2.0, "direction_world": [0.0, 1.0, 0.0]},
        "friction": {"coefficient": 0.6, "regime": "kinetic"},
        "contact_points": [],
    })
    print(f"  contacts count: {len(assembly.contacts)}")
    print(f"  has grip: {assembly.has_contact('left_hand_01', 'apple_01', 'grip')}")
    print(f"  has pull: {assembly.has_contact('left_hand_01', 'apple_01', 'pull')}")
    print(f"  contacts_for apple: {len(assembly.contacts_for('apple_01'))}")

    ok, msg = assembly.validate()
    print(f"\nValidation with contacts: {'OK' if ok else 'FAILED — ' + msg}")

    print("\nTest 1: world transforms at initial layout")
    ws = assembly.get_all_world_matrices()
    for obj_id in ["apple_01", "apple_stem_01", "apple_skin_01"]:
        M = ws[obj_id]
        tx, ty, tz = M[0, 3], M[1, 3], M[2, 3]
        print(f"  {obj_id:18s} world position = ({tx:.2f}, {ty:.2f}, {tz:.2f})")
    assert abs(ws["apple_01"][2, 3] - 84.0) < 1e-6
    assert abs(ws["apple_stem_01"][2, 3] - 88.5) < 1e-6
    assert abs(ws["apple_skin_01"][2, 3] - 84.0) < 1e-6
    print("  → correct")

    print("\nTest 2: move apple up by 20cm; stem and skin should follow")
    apple.object_transform = make_transform(translate=[10.0, 5.0, 104.0])
    ws = assembly.get_all_world_matrices()
    for obj_id in ["apple_01", "apple_stem_01", "apple_skin_01"]:
        M = ws[obj_id]
        tx, ty, tz = M[0, 3], M[1, 3], M[2, 3]
        print(f"  {obj_id:18s} world position = ({tx:.2f}, {ty:.2f}, {tz:.2f})")
    assert abs(ws["apple_01"][2, 3] - 104.0) < 1e-6
    assert abs(ws["apple_stem_01"][2, 3] - 108.5) < 1e-6
    assert abs(ws["apple_skin_01"][2, 3] - 104.0) < 1e-6
    print("  → correct")

    print("\nTest 3: rotate apple 90° around X; stem should tip to -Y")
    apple.object_transform = make_transform(
        translate=[10.0, 5.0, 84.0],
        rotate=[0.7071, 0.0, 0.0, 0.7071],
    )
    ws = assembly.get_all_world_matrices()
    for obj_id in ["apple_01", "apple_stem_01", "apple_skin_01"]:
        M = ws[obj_id]
        tx, ty, tz = M[0, 3], M[1, 3], M[2, 3]
        print(f"  {obj_id:18s} world position = ({tx:.2f}, {ty:.2f}, {tz:.2f})")
    assert abs(ws["apple_stem_01"][1, 3] - 0.5) < 1e-3
    print("  → correct")

    print("\nTest 4: set_world_transform on apple (no shear, exact)")
    apple.object_transform = make_transform(translate=[10.0, 5.0, 84.0])
    residual = assembly.set_world_transform(
        "apple_01",
        make_transform(translate=[30.0, 15.0, 100.0]),
    )
    print(f"  residual: {residual:.6f}")
    assert residual < 1e-6
    M = assembly.get_world_matrix("apple_01")
    assert abs(M[0, 3] - 30.0) < 1e-6
    assert abs(M[1, 3] - 15.0) < 1e-6
    assert abs(M[2, 3] - 100.0) < 1e-6
    print("  → correct")

    print("\nTest 5: squash apple in Z (no rotation, no shear expected)")
    residual = assembly.set_world_transform(
        "apple_01",
        make_transform(translate=[10.0, 5.0, 84.0], scale=[1.0, 1.0, 0.5]),
    )
    print(f"  residual: {residual:.6f}")
    assert residual < 1e-6
    ws = assembly.get_all_world_matrices()
    print(f"  apple world z: {ws['apple_01'][2, 3]:.4f}")
    print(f"  stem world z:  {ws['apple_stem_01'][2, 3]:.4f}")
    assert abs(ws["apple_stem_01"][2, 3] - 86.25) < 1e-4
    print("  → correct (stem inherits parent squash)")

    print("\nTest 6: squash apple in Z, then rotate stem around X (shear)")
    assembly.set_world_transform(
        "apple_01",
        make_transform(translate=[10.0, 5.0, 84.0], scale=[1.0, 1.0, 0.5]),
    )
    stem.object_transform = make_transform(
        translate=[0.0, 0.0, 4.5],
        rotate=[0.3827, 0.0, 0.0, 0.9239],
    )
    t, residual = assembly.get_world_transform_with_residual("apple_stem_01")
    print(f"  residual for stem: {residual:.6f}")
    print(f"  (residual > 0 confirms shear; best-fit TRS is approximate)")

    print("\nTest 7: hierarchy inconsistency is healed by rebuild")
    # Introduce an inconsistency: declare a parent_object_id without
    # updating the parent's children list.
    orphan = Object3D(
        object_id="apple_leaf_01",
        template_name="apple_leaf",
        elements=[make_element("sheet", {"size_x_cm": 2.0, "size_y_cm": 1.0, "thickness_cm": 0.02})],
        object_transform=make_transform(translate=[0.0, 0.0, 5.0]),
        parent_object_id="apple_01",
    )
    assembly.objects.append(orphan)
    # Children list of apple_01 does not yet include apple_leaf_01.
    print(f"  apple_01.children before rebuild: {apple.children}")
    assert "apple_leaf_01" not in apple.children
    # get_all_world_matrices derives locally from parent_object_id, so it
    # already sees the leaf.
    ws = assembly.get_all_world_matrices()
    assert "apple_leaf_01" in ws
    print(f"  get_all_world_matrices found apple_leaf_01 at "
          f"({ws['apple_leaf_01'][0,3]:.2f}, {ws['apple_leaf_01'][1,3]:.2f}, {ws['apple_leaf_01'][2,3]:.2f})")
    # Explicit rebuild now fixes the children list.
    assembly.rebuild_children_lists()
    print(f"  apple_01.children after rebuild: {apple.children}")
    assert "apple_leaf_01" in apple.children
    # Remove the leaf to keep the rest of the tests clean.
    assembly.remove_object("apple_leaf_01")
    print("  → correct")

    print("\nTest 8: JSON round trip (including contacts)")
    d = assembly.to_dict()
    assert "contacts" in d, "contacts should be serialized"
    print(f"  serialized contacts count: {len(d['contacts'])}")
    assembly2 = Assembly3D.from_dict(d)
    ok2, msg2 = assembly2.validate()
    print(f"  validation: {'OK' if ok2 else 'FAILED — ' + msg2}")
    if json.dumps(d, sort_keys=True) == json.dumps(assembly2.to_dict(), sort_keys=True):
        print("  round-trip serialization identical")
    else:
        print("  WARNING: round-trip differs")

    print("\nTest 9: copy carries contacts")
    assembly3 = assembly.copy()
    print(f"  original contacts: {len(assembly.contacts)}")
    print(f"  copied contacts:   {len(assembly3.contacts)}")
    assert len(assembly3.contacts) == len(assembly.contacts)
    # Mutation isolation
    assembly3.contacts.append({
        "object_a": "left_hand_01",
        "object_b": "apple_01",
        "contact_type": "lean",
        "force": {"magnitude_n": 1.0, "direction_world": [0.0, 0.0, -1.0]},
        "friction": {"coefficient": 0.5, "regime": "static"},
        "contact_points": [],
    })
    print(f"  original contacts after mutation: {len(assembly.contacts)}")
    print(f"  copied contacts after mutation:   {len(assembly3.contacts)}")
    assert len(assembly.contacts) == 2
    assert len(assembly3.contacts) == 3
    print("  → copy is isolated")

    print("\nTest 10: invalid contact type fails validation")
    assembly_bad = assembly.copy()
    assembly_bad.contacts.append({
        "object_a": "left_hand_01",
        "object_b": "apple_01",
        "contact_type": "squeeze",
        "force": {},
        "friction": {},
        "contact_points": [],
    })
    ok_bad, msg_bad = assembly_bad.validate()
    print(f"  validation: {'OK' if ok_bad else 'FAILED — ' + msg_bad}")
    assert not ok_bad

    print("\nTest 11: contact referencing missing object fails validation")
    assembly_bad2 = assembly.copy()
    assembly_bad2.contacts.append({
        "object_a": "left_hand_01",
        "object_b": "ghost_object",
        "contact_type": "touch",
        "force": {},
        "friction": {},
        "contact_points": [],
    })
    ok_bad2, msg_bad2 = assembly_bad2.validate()
    print(f"  validation: {'OK' if ok_bad2 else 'FAILED — ' + msg_bad2}")
    assert not ok_bad2

    path = assembly.save()
    print(f"\nSaved to: {path}")
    reloaded = Assembly3D.load(path)
    ok3, msg3 = reloaded.validate()
    print(f"Reloaded from disk: {'OK' if ok3 else 'FAILED — ' + msg3}")
    print(f"  {reloaded.summary()}")
    print(f"  reloaded contacts count: {len(reloaded.contacts)}")

    print("\nSmoke test complete.")