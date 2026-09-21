#!/usr/bin/env python3
r"""
3d_scene_operations.py – Scene operations for the 3D module.

Operations transition an Assembly3D from one state to the next. They are the
temporal mechanism of the v0.2 memory system: a memory's actions map to one
or more operations, and each operation is applied in sequence to produce
per-keyframe assemblies.

Every change in a memory – human action, machine action, or natural process –
is modelled as one or more operations. Geometric changes use the geometric
operations; non-geometric property changes (colour, temperature, moisture,
oxidation, cleanliness, surface state, etc.) use ChangePropertiesOp;
perceptual changes (light, sound, smell, taste, touch) use EmitEventOp;
physical interactions (grip, hold, push, pull, lean, support, rest_on,
touch) use ContactOp and ReleaseOp.

Design:
    - Each operation is a dataclass with a `to_dict` / `from_dict`.
    - `apply_operation(assembly, op)` returns a NEW Assembly3D (immutable
      style). The original is unchanged. The function also returns a list
      of event dicts describing what happened (used for tracing).
    - `apply_operation_sequence(assembly, ops)` returns a list of
      (op, assembly_after, events) tuples.
    - `validate_operation(assembly, op)` returns (bool, message) and does
      not modify anything.

Operation vocabulary (21 types):

    Lifecycle
        AddObjectOp        – introduce a new object
        RemoveObjectOp     – remove an object entirely

    Element composition (inside one object)
        AddElementOp       – add a primitive element to an object
        RemoveElementOp    – remove a primitive element

    Spatial transform
        MoveOp             – additive translation
        RotateOp           – multiplicative rotation (quaternion)
        ScaleOp            – multiplicative scale
        SetTransformOp     – set full world TRS in one shot
        ReachArmRotateOp   – rotate an articulated limb about its joint so
                             it points at a target world position

    Hierarchy
        AttachOp           – make A a child of B (keeps world position by
                             default; optional snap_local)
        DetachOp           – make A independent of its current parent
                             (keeps world position)

    Geometry subdivision
        SplitOp            – binary split: divide one object into exactly
                             two along a cut plane. Use chain_split_along_axis
                             for finer subdivision (quarters, eighths, etc.).
        MergeOp            – combine N objects into one

    Non-geometric properties
        ChangePropertiesOp – merge or replace an object's property dict
                             (colour, temperature, moisture, oxidation,
                             cleanliness, surface_state, physical_state,
                             texture, visibility, state). This covers
                             both natural processes (flesh oxidation,
                             juice seepage, cooling) and human actions
                             (painting, cleaning, wetting).

    Physical interaction
        ContactOp          – record a physical contact interaction. Contact
                             is a persistent relation: once started, it
                             remains active until a `release` ends it.
                             Force and friction are placeholder fields for
                             a future physics addon. They are recorded but
                             not consumed by the current pipeline.
        ReleaseOp          – terminate a prior contact. Inverse of
                             `ContactOp`, with mirrored object_a / object_b
                             argument order.

    Annotation
        LabelOp            – assign a semantic name to an object/element
        UnlabelOp          – remove the semantic name

    Temporal
        WaitOp             – advance time without geometry change
        EmitEventOp        – record a perceptual or semantic marker
                             (sound, smell, light, taste, touch)

    Compound
        CompoundOp         – group several operations under one action

Contact types and the contact/attachment convention:

    `ContactOp` supports these contact types:

        touch     – momentary surface contact, no binding
        grip      – a hand/finger hold; implies attachment
        hold      – a sustained hold; implies attachment
        push      – directed force away from the agent
        pull      – directed force toward the agent
        lean      – the object rests against another
        support   – the object is held up by another
        rest_on   – the object rests on another under gravity

    `ContactOp` and `AttachOp` are separate operations. `AttachOp` controls
    the scene-graph hierarchy (which objects follow which when the parent
    moves). `ContactOp` records the physical interaction (who is touching
    whom, how, with what force). They are complementary, not redundant.

    In practice, most grips and holds imply both: the object follows the
    hand's frame (attach) AND is bound by friction (contact). The
    convention is:

        contact type     implies attachment?
        ------------     -------------------
        touch            no
        grip             yes
        hold             yes
        push             optional
        pull             optional
        lean             no
        support          no
        rest_on          no

    When the LLM emits a ContactOp with type `grip` or `hold`, it should
    also emit the matching `attach` in the same OperationGroup, unless the
    object is already attached. Conversely, a `release` that ends a grip or
    hold should be accompanied by a matching `detach`.

    The pipeline does NOT auto-pair these operations. If the LLM forgets,
    a soft warning is emitted to the trace. The pairing is a documented
    convention, not a hard rule, because some contacts legitimately occur
    without attachment (a hand touching a countertop) and some attachments
    legitimately occur without contact (a composite sub-object).

Inverse pairs:
    add_object       <-> remove_object
    add_element      <-> remove_element
    attach           <-> detach
    split            <-> merge
    label            <-> unlabel
    contact          <-> release
    move / rotate / scale are self-inverse with negated/inverse arguments
    change_properties is self-inverse with the previous values
    reach_arm_rotate is self-inverse with a target at the limb's joint

Split semantics:
    A SplitOp is binary: one source object becomes exactly two new objects.
    Each new object is a copy of the source's elements plus one clip plane
    (in the object's local frame). Points on the positive side of the clip
    normal are hidden.

    For N > 2 pieces, use chain_split_along_axis (below), which returns a
    sequence of binary SplitOps that decompose source into N equal slabs
    along a given axis. Each final piece accumulates the clips of every
    split along its branch, so a quarter has two clips, an eighth has three.

Reach semantics:
    ReachArmRotateOp is used for articulated limbs (arms, legs, necks, and
    any other object whose local origin is a joint). It reads the limb's
    joint world position, computes the rotation that points the limb's
    local -Z direction at the given world target, and writes that rotation
    into the limb's object_transform. The limb's translate is left
    untouched, so the joint stays welded to its parent. The limb's length
    is fixed; a target beyond the limb's reach will still produce a
    rotation that points the limb toward the target, but the tip will stop
    short.

Attachment parent rule:
    The parent is the reference frame — the object the child is "in", "on",
    "held by", or "part of". Not the larger or heavier object. The LLM
    chooses the parent explicitly in each AttachOp.
"""

import copy
import importlib
import json
import math
import sys
import warnings
from dataclasses import dataclass, field
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
TimelineKeyframe = _assembly.TimelineKeyframe
make_element = _assembly.make_element
make_transform = _assembly.make_transform
trs_to_matrix = _assembly.trs_to_matrix
matrix_to_trs = _assembly.matrix_to_trs


# =============================================================================
# Contact type registry and default physics placeholders
# =============================================================================

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

# Contact types that conventionally imply a matching attach/detach pair.
CONTACT_TYPES_IMPLYING_ATTACHMENT = {"grip", "hold"}


def _default_force() -> Dict[str, Any]:
    """
    Default force placeholder. The physics addon is not yet implemented;
    these values are recorded but not consumed.
    """
    return {
        "magnitude_n": 0.0,
        "direction_world": [0.0, 0.0, -1.0],
    }


def _default_friction() -> Dict[str, Any]:
    """
    Default friction placeholder. Single effective coefficient, per design.
    """
    return {
        "coefficient": 0.5,
        "regime": "static",
    }


# =============================================================================
# Helpers
# =============================================================================

def _quaternion_multiply(q1: List[float], q2: List[float]) -> List[float]:
    """Hamilton product q1 * q2, both [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def _quaternion_from_to(a: np.ndarray, b: np.ndarray) -> List[float]:
    """
    Return the quaternion [x, y, z, w] that rotates unit vector a onto
    unit vector b. Handles the antiparallel case (a and b opposite) by
    selecting an arbitrary perpendicular axis and returning a 180-degree
    rotation about it.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    a, b = a / na, b / nb
    w = 1.0 + float(np.dot(a, b))
    if w < 1e-9:
        # a and b are antiparallel; pick any perpendicular axis.
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, arbitrary)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            axis = np.array([0.0, 0.0, 1.0])
            axis_norm = 1.0
        axis = axis / axis_norm
        return [float(axis[0]), float(axis[1]), float(axis[2]), 0.0]
    cross = np.cross(a, b)
    q = np.array([cross[0], cross[1], cross[2], w], dtype=float)
    q = q / np.linalg.norm(q)
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def _is_descendant(assembly: Assembly3D, ancestor_id: str, candidate_id: str) -> bool:
    """Return True if candidate_id is a descendant of ancestor_id."""
    current = candidate_id
    seen = set()
    while current is not None:
        if current in seen:
            return False
        seen.add(current)
        if current == ancestor_id:
            return True
        obj = assembly.get_object(current)
        if obj is None:
            return False
        current = obj.parent_object_id
    return False


def _is_attached_to(assembly: Assembly3D, child_id: str, parent_id: str) -> bool:
    """
    Return True if child_id is attached to parent_id at any depth in the
    hierarchy (child's parent chain includes parent_id).
    """
    if child_id == parent_id:
        return False
    return _is_descendant(assembly, parent_id, child_id)


def _world_plane_to_local(
    plane_point_world: List[float],
    plane_normal_world: List[float],
    world_matrix: np.ndarray,
) -> Tuple[List[float], List[float]]:
    """
    Convert a plane given in world coordinates to the object's local frame.
    """
    M_inv = np.linalg.inv(world_matrix)
    p_world = np.array(list(plane_point_world) + [1.0])
    p_local = (M_inv @ p_world)[:3]

    upper = world_matrix[:3, :3]
    try:
        normal_matrix = np.linalg.inv(upper).T
    except np.linalg.LinAlgError:
        normal_matrix = np.eye(3)
    n_world = np.array(plane_normal_world, dtype=float)
    n_local = normal_matrix @ n_world
    n_norm = np.linalg.norm(n_local)
    if n_norm > 0:
        n_local = n_local / n_norm

    return list(p_local), list(n_local)


def _get_assembly_contacts(assembly: Assembly3D) -> List[Dict[str, Any]]:
    """
    Return the assembly's contact list, creating it if the Assembly3D class
    does not yet have the field. This makes the module forward-compatible
    with a later 3d_assembly.py update that adds `contacts` explicitly.
    """
    contacts = getattr(assembly, "contacts", None)
    if contacts is None:
        contacts = []
        try:
            assembly.contacts = contacts
        except Exception:
            pass
    return contacts


def _contact_key(obj_a: str, obj_b: str) -> Tuple[str, str]:
    """Canonical key for a contact between two objects (ordered)."""
    return (obj_a, obj_b)


def _find_contact_index(
    contacts: List[Dict[str, Any]],
    obj_a: str,
    obj_b: str,
) -> Optional[int]:
    """Return the index of a matching contact entry, or None."""
    for i, c in enumerate(contacts):
        if c.get("object_a") == obj_a and c.get("object_b") == obj_b:
            return i
    return None


# =============================================================================
# Operation dataclasses
# =============================================================================

@dataclass
class AddObjectOp:
    op: str = "add_object"
    object_id: str = ""
    template_name: str = ""
    elements: List[Element3D] = field(default_factory=list)
    material: Dict[str, Any] = field(default_factory=dict)
    object_transform: Transform = field(default_factory=Transform)
    mass_grams: Optional[float] = None
    parent_object_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "op": "add_object",
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
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AddObjectOp":
        return cls(
            object_id=d["object_id"],
            template_name=d.get("template_name", ""),
            elements=[Element3D.from_dict(e) for e in d.get("elements", [])],
            material=dict(d.get("material", {})),
            object_transform=Transform.from_dict(d.get("object_transform")),
            mass_grams=d.get("mass_grams"),
            parent_object_id=d.get("parent_object_id"),
        )


@dataclass
class RemoveObjectOp:
    op: str = "remove_object"
    object_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"op": "remove_object", "object_id": self.object_id}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RemoveObjectOp":
        return cls(object_id=d["object_id"])


@dataclass
class AddElementOp:
    op: str = "add_element"
    object_id: str = ""
    element: Element3D = field(default_factory=lambda: Element3D(element_type="box"))

    def to_dict(self) -> Dict[str, Any]:
        return {"op": "add_element", "object_id": self.object_id, "element": self.element.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AddElementOp":
        return cls(object_id=d["object_id"], element=Element3D.from_dict(d["element"]))


@dataclass
class RemoveElementOp:
    op: str = "remove_element"
    object_id: str = ""
    element_index: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {"op": "remove_element", "object_id": self.object_id, "element_index": self.element_index}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RemoveElementOp":
        return cls(object_id=d["object_id"], element_index=int(d["element_index"]))


@dataclass
class MoveOp:
    op: str = "move"
    object_id: str = ""
    delta: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def to_dict(self) -> Dict[str, Any]:
        return {"op": "move", "object_id": self.object_id, "delta": list(self.delta)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MoveOp":
        return cls(object_id=d["object_id"], delta=list(d.get("delta", [0.0, 0.0, 0.0])))


@dataclass
class RotateOp:
    op: str = "rotate"
    object_id: str = ""
    quaternion: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])

    def to_dict(self) -> Dict[str, Any]:
        return {"op": "rotate", "object_id": self.object_id, "quaternion": list(self.quaternion)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RotateOp":
        return cls(object_id=d["object_id"], quaternion=list(d.get("quaternion", [0.0, 0.0, 0.0, 1.0])))


@dataclass
class ScaleOp:
    op: str = "scale"
    object_id: str = ""
    factor: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    def to_dict(self) -> Dict[str, Any]:
        return {"op": "scale", "object_id": self.object_id, "factor": list(self.factor)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScaleOp":
        return cls(object_id=d["object_id"], factor=list(d.get("factor", [1.0, 1.0, 1.0])))


@dataclass
class SetTransformOp:
    op: str = "set_transform"
    object_id: str = ""
    world_transform: Transform = field(default_factory=Transform)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": "set_transform",
            "object_id": self.object_id,
            "world_transform": self.world_transform.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SetTransformOp":
        return cls(
            object_id=d["object_id"],
            world_transform=Transform.from_dict(d.get("world_transform")),
        )


@dataclass
class ReachArmRotateOp:
    """
    Rotate an articulated limb about its joint so the limb points at a
    target world position. The joint stays at its world position; only the
    rotation changes. Use for any reach, lift, withdraw, or retract motion
    of an arm or any other limb whose local origin is its joint.

    Fields:
        object_id    – the limb being rotated (e.g. left_arm_01)
        target_world – the world-space point the limb should point at,
                       expressed as [x, y, z] in cm
        participants – semantic roles, recorded for trace
    """
    op: str = "reach_arm_rotate"
    object_id: str = ""
    target_world: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    participants: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "op": "reach_arm_rotate",
            "object_id": self.object_id,
            "target_world": list(self.target_world),
        }
        if self.participants:
            d["participants"] = dict(self.participants)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReachArmRotateOp":
        return cls(
            object_id=d["object_id"],
            target_world=list(d.get("target_world", [0.0, 0.0, 0.0])),
            participants=dict(d.get("participants", {})),
        )


@dataclass
class AttachOp:
    """
    Make one object a child of another.

    Default behaviour: preserve the child's current world position. The
    child's local transform is computed as the inverse of the parent's
    world matrix times the child's previous world matrix.

    Optional explicit placement: if snap_local is provided, the child's
    local transform is set to that value instead.

    Ambiguous case: preserve_world=False with snap_local=None is not
    meaningful. When this happens, __post_init__ falls back to
    preserve_world=True and emits a warning.
    """
    op: str = "attach"
    child_id: str = ""
    parent_id: str = ""
    preserve_world: bool = True
    snap_local: Optional[Transform] = None

    def __post_init__(self):
        if self.snap_local is not None:
            self.preserve_world = False
        elif not self.preserve_world:
            warnings.warn(
                f"attach({self.child_id} -> {self.parent_id}): "
                f"preserve_world=False but snap_local is None; "
                f"falling back to preserve_world=True"
            )
            self.preserve_world = True

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "op": "attach",
            "child_id": self.child_id,
            "parent_id": self.parent_id,
            "preserve_world": self.preserve_world,
        }
        if self.snap_local is not None:
            d["snap_local"] = self.snap_local.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AttachOp":
        snap_local = d.get("snap_local")
        return cls(
            child_id=d["child_id"],
            parent_id=d["parent_id"],
            preserve_world=bool(d.get("preserve_world", True)),
            snap_local=Transform.from_dict(snap_local) if snap_local else None,
        )


@dataclass
class DetachOp:
    op: str = "detach"
    child_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"op": "detach", "child_id": self.child_id}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DetachOp":
        return cls(child_id=d["child_id"])


@dataclass
class SplitOp:
    """
    Binary split: one source object becomes exactly two new objects along a
    cut plane. The plane is given in world coordinates; it is converted to
    the source's local frame before being stored as a clip.

    The first new object keeps the negative side of the local normal; the
    second keeps the positive side. For N > 2 pieces, use
    chain_split_along_axis (below).

    The source's children are re-parented to the first new object by
    default, unless children_assignment maps a child to one of the two new
    object IDs explicitly.
    """
    op: str = "split"
    source_id: str = ""
    cut_point_world: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    cut_normal_world: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    new_object_ids: List[str] = field(default_factory=list)
    children_assignment: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "op": "split",
            "source_id": self.source_id,
            "cut_point_world": list(self.cut_point_world),
            "cut_normal_world": list(self.cut_normal_world),
            "new_object_ids": list(self.new_object_ids),
        }
        if self.children_assignment:
            d["children_assignment"] = dict(self.children_assignment)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SplitOp":
        return cls(
            source_id=d["source_id"],
            cut_point_world=list(d.get("cut_point_world", [0.0, 0.0, 0.0])),
            cut_normal_world=list(d.get("cut_normal_world", [0.0, 0.0, 1.0])),
            new_object_ids=list(d.get("new_object_ids", [])),
            children_assignment=d.get("children_assignment"),
        )


@dataclass
class MergeOp:
    op: str = "merge"
    object_ids: List[str] = field(default_factory=list)
    new_object_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"op": "merge", "object_ids": list(self.object_ids), "new_object_id": self.new_object_id}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MergeOp":
        return cls(
            object_ids=list(d.get("object_ids", [])),
            new_object_id=d["new_object_id"],
        )


@dataclass
class ChangePropertiesOp:
    """
    Merge or replace an object's non-geometric property dict.

    The property dict can contain any of the controlled vocabulary
    properties (temperature, moisture, cleanliness, oxidation,
    surface_state, physical_state, state, texture, visibility) plus
    free-form values for colour and other continuous or adjectival
    properties.

    Example uses:
        - Flesh oxidation (natural process, colour change):
            ChangePropertiesOp(object_id="apple_flesh_01",
                               properties={"color": "light_brown",
                                           "oxidation": "light"})
        - Juice seepage (natural process, moisture change):
            ChangePropertiesOp(object_id="apple_01",
                               properties={"surface_state": "damp",
                                           "moisture": "damp"})
        - Cooling (natural process, temperature change):
            ChangePropertiesOp(object_id="soup_01",
                               properties={"temperature": "warm"})
        - Cleaning (human action, cleanliness change):
            ChangePropertiesOp(object_id="plate_01",
                               properties={"cleanliness": "clean"})
        - Painting (human action, colour change):
            ChangePropertiesOp(object_id="wall_01",
                               properties={"color": "blue"})

    The `merge` flag controls the semantics:
        merge=True  – update the properties listed, leave the rest intact
        merge=False – replace the entire property dict
    """
    op: str = "change_properties"
    object_id: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    merge: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": "change_properties",
            "object_id": self.object_id,
            "properties": dict(self.properties),
            "merge": self.merge,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChangePropertiesOp":
        return cls(
            object_id=d["object_id"],
            properties=dict(d.get("properties", {})),
            merge=bool(d.get("merge", True)),
        )


@dataclass
class ContactOp:
    """
    Record a physical contact interaction.

    Contact is a persistent relation: once started, it remains active in
    the assembly's contacts list until a `release` operation ends it.
    Contact is orthogonal to attachment. See the module docstring for the
    contact-type-to-attachment convention table.

    Fields:
        object_a         – the actor or subject of the contact
                           (typically the hand, tool, or body part).
        object_b         – the object being contacted.
        contact_type     – one of CONTACT_TYPES.
        contact_points   – optional list of
                           {point_world, normal_world} dicts. World coords
                           at the moment of the operation.
        force            – placeholder dict for future physics. Defaults to
                           magnitude_n=0.0 and direction_world=[0,0,-1].
                           Force is directed at object_b.
        friction         – placeholder dict for future physics. Defaults to
                           coefficient=0.5 and regime="static". A single
                           effective coefficient per design.
        starts_contact   – True if this operation begins a new contact
                           relation. Default True.
        ends_contact     – True if this operation ends a prior contact
                           relation. Default False. Mutually exclusive
                           with starts_contact.
        participants     – semantic roles for trace and planning.
        notes            – optional free-form string.
    """
    op: str = "contact"
    object_a: str = ""
    object_b: str = ""
    contact_type: str = "touch"
    contact_points: List[Dict[str, Any]] = field(default_factory=list)
    force: Dict[str, Any] = field(default_factory=_default_force)
    friction: Dict[str, Any] = field(default_factory=_default_friction)
    starts_contact: bool = True
    ends_contact: bool = False
    participants: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "op": "contact",
            "object_a": self.object_a,
            "object_b": self.object_b,
            "contact_type": self.contact_type,
            "force": dict(self.force),
            "friction": dict(self.friction),
            "starts_contact": self.starts_contact,
            "ends_contact": self.ends_contact,
        }
        if self.contact_points:
            d["contact_points"] = [dict(p) for p in self.contact_points]
        if self.participants:
            d["participants"] = dict(self.participants)
        if self.notes:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContactOp":
        force = dict(d.get("force", {})) or _default_force()
        friction = dict(d.get("friction", {})) or _default_friction()
        # Ensure required keys exist
        force.setdefault("magnitude_n", 0.0)
        force.setdefault("direction_world", [0.0, 0.0, -1.0])
        friction.setdefault("coefficient", 0.5)
        friction.setdefault("regime", "static")
        return cls(
            object_a=d.get("object_a", ""),
            object_b=d.get("object_b", ""),
            contact_type=d.get("contact_type", "touch"),
            contact_points=list(d.get("contact_points", [])),
            force=force,
            friction=friction,
            starts_contact=bool(d.get("starts_contact", True)),
            ends_contact=bool(d.get("ends_contact", False)),
            participants=dict(d.get("participants", {})),
            notes=d.get("notes", ""),
        )


@dataclass
class ReleaseOp:
    """
    Terminate a prior contact.

    Inverse of ContactOp. The object_a / object_b order mirrors the
    ContactOp that created the contact, so the two operations read
    symmetrically:

        ContactOp(object_a="left_hand_01", object_b="apple_01", type="grip")
        ...
        ReleaseOp(object_a="left_hand_01", object_b="apple_01")

    When the released contact implied attachment (grip or hold), the
    convention requires a matching DetachOp in the same OperationGroup.
    The pipeline emits a soft warning if the detach is missing.
    """
    op: str = "release"
    object_a: str = ""
    object_b: str = ""
    participants: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "op": "release",
            "object_a": self.object_a,
            "object_b": self.object_b,
        }
        if self.participants:
            d["participants"] = dict(self.participants)
        if self.notes:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReleaseOp":
        return cls(
            object_a=d.get("object_a", ""),
            object_b=d.get("object_b", ""),
            participants=dict(d.get("participants", {})),
            notes=d.get("notes", ""),
        )


@dataclass
class LabelOp:
    op: str = "label"
    object_id: str = ""
    label: str = ""
    element_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"op": "label", "object_id": self.object_id, "label": self.label}
        if self.element_index is not None:
            d["element_index"] = self.element_index
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LabelOp":
        return cls(
            object_id=d["object_id"],
            label=d["label"],
            element_index=d.get("element_index"),
        )


@dataclass
class UnlabelOp:
    op: str = "unlabel"
    object_id: str = ""
    element_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"op": "unlabel", "object_id": self.object_id}
        if self.element_index is not None:
            d["element_index"] = self.element_index
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UnlabelOp":
        return cls(object_id=d["object_id"], element_index=d.get("element_index"))


@dataclass
class WaitOp:
    op: str = "wait"
    duration: float = 0.0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"op": "wait", "duration": self.duration, "note": self.note}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WaitOp":
        return cls(duration=float(d.get("duration", 0.0)), note=d.get("note", ""))


@dataclass
class EmitEventOp:
    """
    Record a perceptual or semantic marker that does not change the scene
    geometry or properties. Used for sounds, smells, light effects, taste,
    and touch, and for marking a moment of interest in the timeline.

    Example uses:
        - Scraping sound:
            EmitEventOp(event_type="scraping_sound",
                        payload={"source": "knife_01", "target": "apple_01",
                                 "quality": "soft_dry"})
        - Apple flesh smell:
            EmitEventOp(event_type="apple_flesh_smell",
                        payload={"source": "apple_01"})
        - Light through peel:
            EmitEventOp(event_type="light_through_peel",
                        payload={"source": "peel_strip_01"})
    """
    op: str = "emit_event"
    event_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"op": "emit_event", "event_type": self.event_type, "payload": dict(self.payload)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EmitEventOp":
        return cls(event_type=d["event_type"], payload=dict(d.get("payload", {})))


@dataclass
class CompoundOp:
    op: str = "compound"
    operations: List[Any] = field(default_factory=list)
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": "compound",
            "operations": [op_to_dict(o) for o in self.operations],
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CompoundOp":
        return cls(
            operations=[op_from_dict(o) for o in d.get("operations", [])],
            label=d.get("label", ""),
        )


# =============================================================================
# Operation registry and dispatch
# =============================================================================

_OP_REGISTRY = {
    "add_object":         AddObjectOp,
    "remove_object":      RemoveObjectOp,
    "add_element":        AddElementOp,
    "remove_element":     RemoveElementOp,
    "move":               MoveOp,
    "rotate":             RotateOp,
    "scale":              ScaleOp,
    "set_transform":      SetTransformOp,
    "reach_arm_rotate":   ReachArmRotateOp,
    "attach":             AttachOp,
    "detach":             DetachOp,
    "split":              SplitOp,
    "merge":              MergeOp,
    "change_properties":  ChangePropertiesOp,
    "contact":            ContactOp,
    "release":            ReleaseOp,
    "label":              LabelOp,
    "unlabel":            UnlabelOp,
    "wait":               WaitOp,
    "emit_event":         EmitEventOp,
    "compound":           CompoundOp,
}


def op_from_dict(d: Dict[str, Any]):
    """Build the correct operation dataclass from a dict produced by the LLM."""
    if not isinstance(d, dict):
        raise ValueError("operation must be a dict")
    op_type = d.get("op")
    if op_type not in _OP_REGISTRY:
        raise ValueError(f"unknown operation type '{op_type}'")
    return _OP_REGISTRY[op_type].from_dict(d)


def op_to_dict(op) -> Dict[str, Any]:
    """Return the JSON form of an operation dataclass."""
    if not hasattr(op, "to_dict"):
        raise ValueError("operation has no to_dict method")
    return op.to_dict()


# =============================================================================
# Chain split helper
# =============================================================================

def chain_split_along_axis(
    source_id: str,
    n_pieces: int,
    axis_world: List[float],
    new_ids: List[str],
    base_point_world: Optional[List[float]] = None,
    extent_along_axis: float = 1.0,
) -> List[SplitOp]:
    """
    Produce a sequence of binary SplitOps that cut source_id into n_pieces
    equal slabs along the given world axis.

    For n_pieces = 2, returns a single SplitOp.
    For n_pieces = 4, returns three SplitOps: one cut at the centre of the
    source, then one cut at the centre of each half.
    For n_pieces = 8, three levels of binary cuts, and so on.

    Only powers of two are supported for the balanced tree form.

    Args:
        source_id        : the object to split.
        n_pieces         : number of final pieces (power of two, >= 2).
        axis_world       : the cutting axis, in world coordinates.
        new_ids          : exactly n_pieces final object IDs, in the order
                           from the negative end of the axis to the
                           positive end.
        base_point_world : the centre of the slab along the axis, in world
                           coordinates. Cuts are placed symmetrically
                           around this point. Defaults to the world origin.
        extent_along_axis: the total length of the slab along the axis, in
                           cm. Cuts are placed within [-extent/2, +extent/2]
                           relative to base_point_world. Defaults to 1.0.

    Returns a list of SplitOp objects that, applied in order, produce the
    n_pieces final objects. Intermediate splits use auto-generated temporary
    IDs of the form "__split_tmp_<n>"; these do not appear in the final
    assembly because they are themselves split before the sequence ends.

    The function does not modify the assembly. It only constructs the
    operation list. Callers are expected to wrap the list in a CompoundOp
    if they want it applied as one atomic action.
    """
    if n_pieces < 2:
        raise ValueError(f"n_pieces must be >= 2, got {n_pieces}")
    if (n_pieces & (n_pieces - 1)) != 0:
        raise ValueError(
            f"n_pieces must be a power of two for chain_split_along_axis; "
            f"got {n_pieces}. Split into a supported count first, then "
            f"re-split one piece."
        )
    if len(new_ids) != n_pieces:
        raise ValueError(
            f"new_ids must have exactly n_pieces entries "
            f"({n_pieces}), got {len(new_ids)}"
        )
    if len(axis_world) != 3 or np.linalg.norm(axis_world) == 0:
        raise ValueError("axis_world must be a nonzero 3-vector")
    if extent_along_axis <= 0:
        raise ValueError(
            f"extent_along_axis must be positive, got {extent_along_axis}"
        )

    axis = np.array(axis_world, dtype=float)
    axis = axis / np.linalg.norm(axis)
    base = (
        np.array(base_point_world, dtype=float)
        if base_point_world is not None
        else np.zeros(3, dtype=float)
    )
    E = float(extent_along_axis)

    temp_counter = [0]

    def new_temp_id() -> str:
        temp_counter[0] += 1
        return f"__split_tmp_{temp_counter[0]}"

    ops: List[SplitOp] = []

    def cut(source: str, ids: List[str], lo: float, hi: float):
        n = len(ids)
        if n == 1:
            return
        if n == 2:
            mid = (lo + hi) / 2.0
            cut_point = base + axis * mid
            ops.append(SplitOp(
                source_id=source,
                cut_point_world=list(cut_point),
                cut_normal_world=list(axis),
                new_object_ids=[ids[0], ids[1]],
            ))
            return
        half = n // 2
        mid = (lo + hi) / 2.0
        cut_point = base + axis * mid
        left_temp = new_temp_id()
        right_temp = new_temp_id()
        ops.append(SplitOp(
            source_id=source,
            cut_point_world=list(cut_point),
            cut_normal_world=list(axis),
            new_object_ids=[left_temp, right_temp],
        ))
        cut(left_temp, ids[:half], lo, mid)
        cut(right_temp, ids[half:], mid, hi)

    cut(source_id, new_ids, -E / 2.0, +E / 2.0)
    return ops


# =============================================================================
# Validation
# =============================================================================

def _validate_contact_op(assembly: Assembly3D, op: ContactOp) -> Tuple[bool, str]:
    if not op.object_a:
        return False, "contact: object_a is empty"
    if not op.object_b:
        return False, "contact: object_b is empty"
    if op.object_a == op.object_b:
        return False, "contact: object_a and object_b are the same"
    if not assembly.has_object(op.object_a):
        return False, f"contact: object_a '{op.object_a}' does not exist"
    if not assembly.has_object(op.object_b):
        return False, f"contact: object_b '{op.object_b}' does not exist"
    if op.contact_type not in CONTACT_TYPES:
        return False, (
            f"contact: unknown contact_type '{op.contact_type}'; "
            f"allowed: {sorted(CONTACT_TYPES)}"
        )
    if op.starts_contact and op.ends_contact:
        return False, "contact: starts_contact and ends_contact are mutually exclusive"
    if not op.starts_contact and not op.ends_contact:
        return False, (
            "contact: one of starts_contact or ends_contact must be True"
        )
    return True, ""


def _validate_release_op(assembly: Assembly3D, op: ReleaseOp) -> Tuple[bool, str]:
    if not op.object_a:
        return False, "release: object_a is empty"
    if not op.object_b:
        return False, "release: object_b is empty"
    if op.object_a == op.object_b:
        return False, "release: object_a and object_b are the same"
    if not assembly.has_object(op.object_a):
        return False, f"release: object_a '{op.object_a}' does not exist"
    if not assembly.has_object(op.object_b):
        return False, f"release: object_b '{op.object_b}' does not exist"
    contacts = _get_assembly_contacts(assembly)
    if _find_contact_index(contacts, op.object_a, op.object_b) is None:
        return False, (
            f"release: no active contact between "
            f"'{op.object_a}' and '{op.object_b}'"
        )
    return True, ""


def validate_operation(assembly: Assembly3D, op) -> Tuple[bool, str]:
    """
    Check that an operation is well-formed against the current assembly.
    Returns (True, "") on success, (False, message) on failure.

    The contact / release operations additionally emit a soft warning (not
    a failure) if a grip or hold contact is missing its paired attach or
    detach. See the module docstring for the convention.
    """
    if isinstance(op, AddObjectOp):
        if not op.object_id:
            return False, "add_object: object_id is empty"
        if assembly.has_object(op.object_id):
            return False, f"add_object: '{op.object_id}' already exists"
        if op.parent_object_id is not None and not assembly.has_object(op.parent_object_id):
            return False, f"add_object: parent '{op.parent_object_id}' does not exist"
        if not op.elements:
            return False, "add_object: at least one element is required"
        for i, el in enumerate(op.elements):
            ok, msg = el.validate()
            if not ok:
                return False, f"add_object element {i}: {msg}"
        return True, ""

    if isinstance(op, RemoveObjectOp):
        if not op.object_id:
            return False, "remove_object: object_id is empty"
        if not assembly.has_object(op.object_id):
            return False, f"remove_object: '{op.object_id}' does not exist"
        return True, ""

    if isinstance(op, AddElementOp):
        if not assembly.has_object(op.object_id):
            return False, f"add_element: '{op.object_id}' does not exist"
        ok, msg = op.element.validate()
        if not ok:
            return False, f"add_element: {msg}"
        return True, ""

    if isinstance(op, RemoveElementOp):
        obj = assembly.get_object(op.object_id)
        if obj is None:
            return False, f"remove_element: '{op.object_id}' does not exist"
        if op.element_index < 0 or op.element_index >= len(obj.elements):
            return False, (
                f"remove_element: index {op.element_index} out of range "
                f"for '{op.object_id}' (has {len(obj.elements)} elements)"
            )
        return True, ""

    if isinstance(op, (MoveOp, RotateOp, ScaleOp, SetTransformOp)):
        if not assembly.has_object(op.object_id):
            return False, f"{op.op}: '{op.object_id}' does not exist"
        return True, ""

    if isinstance(op, ReachArmRotateOp):
        if not assembly.has_object(op.object_id):
            return False, f"reach_arm_rotate: '{op.object_id}' does not exist"
        if not isinstance(op.target_world, list) or len(op.target_world) != 3:
            return False, "reach_arm_rotate: target_world must be a 3-vector [x, y, z]"
        try:
            float(op.target_world[0])
            float(op.target_world[1])
            float(op.target_world[2])
        except (TypeError, ValueError):
            return False, "reach_arm_rotate: target_world values must be numeric"
        return True, ""

    if isinstance(op, AttachOp):
        if not assembly.has_object(op.child_id):
            return False, f"attach: child '{op.child_id}' does not exist"
        if not assembly.has_object(op.parent_id):
            return False, f"attach: parent '{op.parent_id}' does not exist"
        if op.child_id == op.parent_id:
            return False, f"attach: child and parent are the same '{op.child_id}'"
        if _is_descendant(assembly, op.child_id, op.parent_id):
            return False, (
                f"attach: '{op.parent_id}' is a descendant of '{op.child_id}'; "
                f"would create a cycle"
            )
        return True, ""

    if isinstance(op, DetachOp):
        if not assembly.has_object(op.child_id):
            return False, f"detach: '{op.child_id}' does not exist"
        child = assembly.get_object(op.child_id)
        if child.parent_object_id is None:
            return False, f"detach: '{op.child_id}' has no parent"
        return True, ""

    if isinstance(op, SplitOp):
        src = assembly.get_object(op.source_id)
        if src is None:
            return False, f"split: source '{op.source_id}' does not exist"
        if len(op.new_object_ids) != 2:
            return False, (
                f"split: exactly two new_object_ids required, got "
                f"{len(op.new_object_ids)}. Use chain_split_along_axis for "
                f"finer subdivision."
            )
        for nid in op.new_object_ids:
            if assembly.has_object(nid):
                return False, f"split: '{nid}' already exists"
            if nid == op.source_id:
                return False, f"split: new id '{nid}' equals source id"
        if op.new_object_ids[0] == op.new_object_ids[1]:
            return False, "split: the two new_object_ids must be distinct"
        n = np.array(op.cut_normal_world, dtype=float)
        if np.linalg.norm(n) == 0:
            return False, "split: cut_normal_world is zero"
        return True, ""

    if isinstance(op, MergeOp):
        if len(op.object_ids) < 2:
            return False, "merge: at least two object_ids required"
        for oid in op.object_ids:
            if not assembly.has_object(oid):
                return False, f"merge: '{oid}' does not exist"
        if assembly.has_object(op.new_object_id):
            return False, f"merge: '{op.new_object_id}' already exists"
        return True, ""

    if isinstance(op, ChangePropertiesOp):
        if not assembly.has_object(op.object_id):
            return False, f"change_properties: '{op.object_id}' does not exist"
        if not op.properties:
            return False, "change_properties: properties dict is empty"
        return True, ""

    if isinstance(op, ContactOp):
        ok, msg = _validate_contact_op(assembly, op)
        if not ok:
            return False, msg
        # Soft warning: a grip or hold that starts a contact should have a
        # matching attach, unless the two objects are already in a
        # parent-child relationship (either direction).
        if op.starts_contact and op.contact_type in CONTACT_TYPES_IMPLYING_ATTACHMENT:
            attached_a_to_b = _is_attached_to(assembly, op.object_a, op.object_b)
            attached_b_to_a = _is_attached_to(assembly, op.object_b, op.object_a)
            if not attached_a_to_b and not attached_b_to_a:
                warnings.warn(
                    f"contact: '{op.contact_type}' between "
                    f"'{op.object_a}' and '{op.object_b}' does not have a "
                    f"matching attach in the current hierarchy. The "
                    f"convention is that grip and hold imply attachment. "
                    f"Emit an AttachOp in the same OperationGroup."
                )
        return True, ""

    if isinstance(op, ReleaseOp):
        ok, msg = _validate_release_op(assembly, op)
        if not ok:
            return False, msg
        # Soft warning: if the ending contact implied attachment, there
        # should be a matching detach already applied or coming soon.
        contacts = _get_assembly_contacts(assembly)
        idx = _find_contact_index(contacts, op.object_a, op.object_b)
        if idx is not None:
            contact_type = contacts[idx].get("contact_type", "")
            if contact_type in CONTACT_TYPES_IMPLYING_ATTACHMENT:
                attached_a_to_b = _is_attached_to(assembly, op.object_a, op.object_b)
                attached_b_to_a = _is_attached_to(assembly, op.object_b, op.object_a)
                if attached_a_to_b or attached_b_to_a:
                    warnings.warn(
                        f"release: ending a '{contact_type}' between "
                        f"'{op.object_a}' and '{op.object_b}' which is "
                        f"still attached in the hierarchy. The convention "
                        f"is that releasing a grip or hold implies a "
                        f"matching DetachOp."
                    )
        return True, ""

    if isinstance(op, (LabelOp, UnlabelOp)):
        obj = assembly.get_object(op.object_id)
        if obj is None:
            return False, f"{op.op}: '{op.object_id}' does not exist"
        if isinstance(op, LabelOp):
            if op.element_index is not None:
                if op.element_index < 0 or op.element_index >= len(obj.elements):
                    return False, (
                        f"label: element_index {op.element_index} out of "
                        f"range for '{op.object_id}'"
                    )
        return True, ""

    if isinstance(op, (WaitOp, EmitEventOp)):
        return True, ""

    if isinstance(op, CompoundOp):
        current = assembly
        for i, sub in enumerate(op.operations):
            ok, msg = validate_operation(current, sub)
            if not ok:
                return False, f"compound sub-op {i}: {msg}"
            break
        return True, ""

    return False, f"unknown operation class: {type(op).__name__}"


# =============================================================================
# Apply operations
# =============================================================================

def apply_operation(assembly: Assembly3D, op) -> Tuple[Assembly3D, List[Dict[str, Any]]]:
    """
    Apply an operation to the assembly, returning a NEW Assembly3D and a
    list of event dicts describing what happened. The input assembly is
    unchanged.
    """
    ok, msg = validate_operation(assembly, op)
    if not ok:
        raise ValueError(f"invalid operation: {msg}")

    new_assembly = assembly.copy()
    # Defensive: carry contacts across the copy. Once Assembly3D
    # serializes `contacts` this becomes a no-op, but it prevents loss
    # if the assembly class has not yet been updated.
    if hasattr(assembly, "contacts") and not hasattr(new_assembly, "contacts"):
        try:
            new_assembly.contacts = list(assembly.contacts)
        except Exception:
            pass
    events: List[Dict[str, Any]] = []

    # ---- Lifecycle ----
    if isinstance(op, AddObjectOp):
        new_obj = Object3D(
            object_id=op.object_id,
            template_name=op.template_name,
            elements=list(op.elements),
            material=dict(op.material),
            object_transform=Transform.identity(),
            mass_grams=op.mass_grams,
            parent_object_id=op.parent_object_id,
        )
        new_assembly.add_object(new_obj)
        if op.parent_object_id is not None:
            parent = new_assembly.get_object(op.parent_object_id)
            if parent is not None and op.object_id not in parent.children:
                parent.children.append(op.object_id)
        # Treat the op's object_transform as a WORLD transform, matching
        # the convention used elsewhere in the pipeline (and by the LLM).
        # set_world_transform computes the local transform relative to
        # the parent's current world matrix.
        new_assembly.set_world_transform(op.object_id, op.object_transform)
        events.append({
            "event": "object_added",
            "object_id": op.object_id,
            "template_name": op.template_name,
        })

    elif isinstance(op, RemoveObjectOp):
        new_assembly.remove_object(op.object_id)
        for kf in new_assembly.timeline:
            kf.object_transforms.pop(op.object_id, None)
        # Drop any contacts involving the removed object.
        contacts = _get_assembly_contacts(new_assembly)
        new_assembly.contacts = [
            c for c in contacts
            if c.get("object_a") != op.object_id and c.get("object_b") != op.object_id
        ]
        events.append({"event": "object_removed", "object_id": op.object_id})

    elif isinstance(op, AddElementOp):
        obj = new_assembly.get_object(op.object_id)
        obj.elements.append(op.element)
        events.append({
            "event": "element_added",
            "object_id": op.object_id,
            "element_type": op.element.element_type,
        })

    elif isinstance(op, RemoveElementOp):
        obj = new_assembly.get_object(op.object_id)
        removed = obj.elements.pop(op.element_index)
        events.append({
            "event": "element_removed",
            "object_id": op.object_id,
            "element_type": removed.element_type,
        })

    # ---- Spatial transform ----
    elif isinstance(op, MoveOp):
        obj = new_assembly.get_object(op.object_id)
        obj.object_transform = make_transform(
            translate=[
                obj.object_transform.translate[i] + op.delta[i]
                for i in range(3)
            ],
            rotate=list(obj.object_transform.rotate),
            scale=list(obj.object_transform.scale),
        )
        obj.world_matrix_override = None
        events.append({
            "event": "moved",
            "object_id": op.object_id,
            "delta": list(op.delta),
        })

    elif isinstance(op, RotateOp):
        obj = new_assembly.get_object(op.object_id)
        new_rotate = _quaternion_multiply(list(obj.object_transform.rotate), list(op.quaternion))
        obj.object_transform = make_transform(
            translate=list(obj.object_transform.translate),
            rotate=new_rotate,
            scale=list(obj.object_transform.scale),
        )
        obj.world_matrix_override = None
        events.append({
            "event": "rotated",
            "object_id": op.object_id,
            "quaternion": list(op.quaternion),
        })

    elif isinstance(op, ScaleOp):
        obj = new_assembly.get_object(op.object_id)
        new_scale = [
            obj.object_transform.scale[i] * op.factor[i]
            for i in range(3)
        ]
        obj.object_transform = make_transform(
            translate=list(obj.object_transform.translate),
            rotate=list(obj.object_transform.rotate),
            scale=new_scale,
        )
        obj.world_matrix_override = None
        events.append({
            "event": "scaled",
            "object_id": op.object_id,
            "factor": list(op.factor),
        })

    elif isinstance(op, SetTransformOp):
        residual = new_assembly.set_world_transform(op.object_id, op.world_transform)
        events.append({
            "event": "transform_set",
            "object_id": op.object_id,
            "world_transform": op.world_transform.to_dict(),
            "residual": residual,
        })

    elif isinstance(op, ReachArmRotateOp):
        arm = new_assembly.get_object(op.object_id)
        arm_world = new_assembly.get_world_matrix(op.object_id)
        joint_world = arm_world[:3, 3]
        target_world = np.array(op.target_world, dtype=float)
        d_world = target_world - joint_world
        d_norm = float(np.linalg.norm(d_world))
        if d_norm < 1e-9:
            events.append({
                "event": "arm_reach_noop",
                "object_id": op.object_id,
                "reason": "target coincides with joint",
            })
        else:
            d_world = d_world / d_norm
            if arm.parent_object_id is not None:
                parent_world = new_assembly.get_world_matrix(arm.parent_object_id)
                parent_rot = parent_world[:3, :3]
                try:
                    parent_rot_inv = np.linalg.inv(parent_rot)
                except np.linalg.LinAlgError:
                    parent_rot_inv = np.eye(3)
                d_parent = parent_rot_inv @ d_world
            else:
                d_parent = d_world
            d_parent = d_parent / (np.linalg.norm(d_parent) or 1.0)
            rest_dir_local = np.array([0.0, 0.0, -1.0])
            q = _quaternion_from_to(rest_dir_local, d_parent)
            arm.object_transform = make_transform(
                translate=list(arm.object_transform.translate),
                rotate=q,
                scale=list(arm.object_transform.scale),
            )
            arm.world_matrix_override = None
            events.append({
                "event": "arm_reached",
                "object_id": op.object_id,
                "target_world": list(op.target_world),
                "quaternion": q,
            })

    # ---- Hierarchy ----
    elif isinstance(op, AttachOp):
        child = new_assembly.get_object(op.child_id)
        parent = new_assembly.get_object(op.parent_id)

        if child.parent_object_id is not None:
            old_parent = new_assembly.get_object(child.parent_object_id)
            if old_parent and op.child_id in old_parent.children:
                old_parent.children.remove(op.child_id)

        world_before = new_assembly.get_world_matrix(op.child_id)

        child.parent_object_id = op.parent_id
        if op.child_id not in parent.children:
            parent.children.append(op.child_id)

        if op.snap_local is not None:
            child.object_transform = op.snap_local
            child.world_matrix_override = None
        elif op.preserve_world:
            parent_world = new_assembly.get_world_matrix(op.parent_id)
            parent_inv = np.linalg.inv(parent_world)
            local_matrix = parent_inv @ world_before
            local_transform, residual = matrix_to_trs(local_matrix)
            child.object_transform = local_transform
            if residual > 1e-4:
                child.world_matrix_override = world_before.tolist()
            else:
                child.world_matrix_override = None

        events.append({
            "event": "attached",
            "child_id": op.child_id,
            "parent_id": op.parent_id,
            "preserve_world": op.preserve_world,
            "snap_local": op.snap_local.to_dict() if op.snap_local else None,
        })

    elif isinstance(op, DetachOp):
        child = new_assembly.get_object(op.child_id)
        old_parent_id = child.parent_object_id
        if old_parent_id is not None:
            old_parent = new_assembly.get_object(old_parent_id)
            if old_parent and op.child_id in old_parent.children:
                old_parent.children.remove(op.child_id)

        world_before = new_assembly.get_world_matrix(op.child_id)
        child.parent_object_id = None

        local_transform, residual = matrix_to_trs(world_before)
        child.object_transform = local_transform
        if residual > 1e-4:
            child.world_matrix_override = world_before.tolist()
        else:
            child.world_matrix_override = None

        events.append({
            "event": "detached",
            "child_id": op.child_id,
            "old_parent_id": old_parent_id,
        })

    # ---- Subdivision ----
    elif isinstance(op, SplitOp):
        source = new_assembly.get_object(op.source_id)
        world_matrix = new_assembly.get_world_matrix(op.source_id)

        local_point, local_normal = _world_plane_to_local(
            op.cut_point_world, op.cut_normal_world, world_matrix
        )

        base_material = dict(source.material)
        base_mass = source.mass_grams
        per_half_mass = (base_mass / 2.0) if base_mass is not None else None

        parent_clips: List[Dict[str, Any]] = []
        if isinstance(source.metadata, dict):
            existing = source.metadata.get("clips")
            if isinstance(existing, list):
                parent_clips = [dict(c) for c in existing]

        new_objects = []
        for i, new_id in enumerate(op.new_object_ids):
            elements = [copy.deepcopy(e) for e in source.elements]

            if i == 0:
                this_clip_normal = list(local_normal)
            else:
                this_clip_normal = [-x for x in local_normal]

            this_clip = {"point": list(local_point), "normal": this_clip_normal}
            clips = parent_clips + [this_clip]

            new_obj = Object3D(
                object_id=new_id,
                template_name=source.template_name,
                elements=elements,
                material=dict(base_material),
                object_transform=Transform.from_dict(source.object_transform.to_dict()),
                mass_grams=per_half_mass,
                parent_object_id=source.parent_object_id,
                metadata={
                    "clips": clips,
                    "split_from": op.source_id,
                    "split_index": i,
                },
            )
            if source.world_matrix_override is not None:
                new_obj.world_matrix_override = [list(r) for r in source.world_matrix_override]
            new_objects.append(new_obj)

        children_assignment = op.children_assignment or {}
        default_new_parent = op.new_object_ids[0]
        for child_id in list(source.children):
            target = children_assignment.get(child_id, default_new_parent)
            if target in op.new_object_ids:
                for no in new_objects:
                    if no.object_id == target:
                        if child_id not in no.children:
                            no.children.append(child_id)
                        break
                child_obj = new_assembly.get_object(child_id)
                if child_obj is not None:
                    child_obj.parent_object_id = target

        parent_id = source.parent_object_id
        if parent_id is not None:
            parent = new_assembly.get_object(parent_id)
            if parent and op.source_id in parent.children:
                parent.children.remove(op.source_id)
        new_assembly.remove_object(op.source_id)
        for no in new_objects:
            new_assembly.add_object(no)

        # Contacts involving the source are re-pointed to the first new
        # object. This is a reasonable default; the LLM can emit explicit
        # contacts on the new objects if a different mapping is intended.
        contacts = _get_assembly_contacts(new_assembly)
        first_new_id = op.new_object_ids[0]
        for c in contacts:
            if c.get("object_a") == op.source_id:
                c["object_a"] = first_new_id
            if c.get("object_b") == op.source_id:
                c["object_b"] = first_new_id

        events.append({
            "event": "split",
            "source_id": op.source_id,
            "new_object_ids": list(op.new_object_ids),
            "cut_point_world": list(op.cut_point_world),
            "cut_normal_world": list(op.cut_normal_world),
        })

    elif isinstance(op, MergeOp):
        objects = [new_assembly.get_object(oid) for oid in op.object_ids]
        keep_world = new_assembly.get_world_matrix(op.object_ids[0])

        combined_elements: List[Element3D] = []
        for src in objects:
            src_world = new_assembly.get_world_matrix(src.object_id)
            to_keep = np.linalg.inv(keep_world) @ src_world
            for el in src.elements:
                new_el = copy.deepcopy(el)
                el_matrix = trs_to_matrix(el.transform)
                new_el_matrix = to_keep @ el_matrix
                new_el_t, _ = matrix_to_trs(new_el_matrix)
                new_el.transform = new_el_t
                combined_elements.append(new_el)

        masses = [o.mass_grams for o in objects if o.mass_grams is not None]
        total_mass = sum(masses) if masses else None

        new_children: List[str] = []
        for src in objects:
            for cid in src.children:
                if cid not in new_children:
                    new_children.append(cid)
                c = new_assembly.get_object(cid)
                if c is not None:
                    c.parent_object_id = op.new_object_id

        new_parent = objects[0].parent_object_id

        for src in objects:
            if src.parent_object_id is not None:
                p = new_assembly.get_object(src.parent_object_id)
                if p and src.object_id in p.children:
                    p.children.remove(src.object_id)
            new_assembly.remove_object(src.object_id)

        merged = Object3D(
            object_id=op.new_object_id,
            template_name=objects[0].template_name,
            elements=combined_elements,
            material=dict(objects[0].material),
            object_transform=Transform.from_dict(objects[0].object_transform.to_dict()),
            mass_grams=total_mass,
            parent_object_id=new_parent,
            children=new_children,
            metadata={"merged_from": list(op.object_ids)},
        )
        new_assembly.add_object(merged)
        if new_parent is not None:
            p = new_assembly.get_object(new_parent)
            if p is not None and op.new_object_id not in p.children:
                p.children.append(op.new_object_id)

        # Contacts involving any source are re-pointed to the merged object.
        contacts = _get_assembly_contacts(new_assembly)
        for c in contacts:
            if c.get("object_a") in op.object_ids:
                c["object_a"] = op.new_object_id
            if c.get("object_b") in op.object_ids:
                c["object_b"] = op.new_object_id

        events.append({
            "event": "merged",
            "object_ids": list(op.object_ids),
            "new_object_id": op.new_object_id,
        })

    # ---- Non-geometric properties ----
    elif isinstance(op, ChangePropertiesOp):
        obj = new_assembly.get_object(op.object_id)
        if op.merge:
            obj.material = {**obj.material, **op.properties}
        else:
            obj.material = dict(op.properties)
        events.append({
            "event": "properties_changed",
            "object_id": op.object_id,
            "merge": op.merge,
            "properties": dict(op.properties),
        })

    # ---- Physical interaction ----
    elif isinstance(op, ContactOp):
        contacts = _get_assembly_contacts(new_assembly)
        existing_index = _find_contact_index(contacts, op.object_a, op.object_b)

        if op.starts_contact:
            entry = {
                "object_a": op.object_a,
                "object_b": op.object_b,
                "contact_type": op.contact_type,
                "force": dict(op.force),
                "friction": dict(op.friction),
                "contact_points": [dict(p) for p in op.contact_points],
            }
            if existing_index is not None:
                # Replace the existing entry with the new one. This is the
                # "adjust grip" case, where a new ContactOp supersedes the
                # prior contact between the same pair.
                contacts[existing_index] = entry
            else:
                contacts.append(entry)
            events.append({
                "event": "contact_started",
                "object_a": op.object_a,
                "object_b": op.object_b,
                "contact_type": op.contact_type,
            })

        elif op.ends_contact:
            if existing_index is not None:
                contacts.pop(existing_index)
            events.append({
                "event": "contact_ended",
                "object_a": op.object_a,
                "object_b": op.object_b,
            })

        new_assembly.contacts = contacts

    elif isinstance(op, ReleaseOp):
        contacts = _get_assembly_contacts(new_assembly)
        idx = _find_contact_index(contacts, op.object_a, op.object_b)
        if idx is not None:
            contacts.pop(idx)
        new_assembly.contacts = contacts
        events.append({
            "event": "released",
            "object_a": op.object_a,
            "object_b": op.object_b,
        })

    # ---- Annotation ----
    elif isinstance(op, LabelOp):
        obj = new_assembly.get_object(op.object_id)
        if op.element_index is not None:
            el = obj.elements[op.element_index]
            if el.material_override is None:
                el.material_override = {}
            el.material_override["label"] = op.label
            events.append({
                "event": "element_labeled",
                "object_id": op.object_id,
                "element_index": op.element_index,
                "label": op.label,
            })
        else:
            obj.metadata["label"] = op.label
            events.append({
                "event": "object_labeled",
                "object_id": op.object_id,
                "label": op.label,
            })

    elif isinstance(op, UnlabelOp):
        obj = new_assembly.get_object(op.object_id)
        if op.element_index is not None:
            el = obj.elements[op.element_index]
            if el.material_override and "label" in el.material_override:
                del el.material_override["label"]
            events.append({
                "event": "element_unlabeled",
                "object_id": op.object_id,
                "element_index": op.element_index,
            })
        else:
            obj.metadata.pop("label", None)
            events.append({
                "event": "object_unlabeled",
                "object_id": op.object_id,
            })

    # ---- Temporal ----
    elif isinstance(op, WaitOp):
        events.append({"event": "wait", "duration": op.duration, "note": op.note})

    elif isinstance(op, EmitEventOp):
        events.append({
            "event": "emit_event",
            "event_type": op.event_type,
            "payload": dict(op.payload),
        })

    # ---- Compound ----
    elif isinstance(op, CompoundOp):
        current = new_assembly
        sub_events = []
        for i, sub in enumerate(op.operations):
            try:
                current, sub_evs = apply_operation(current, sub)
                sub_events.append({
                    "compound_index": i,
                    "op": op_to_dict(sub),
                    "events": sub_evs,
                })
            except ValueError as e:
                raise ValueError(
                    f"compound '{op.label}' failed at sub-op {i}: {e}"
                )
        new_assembly = current
        events.append({
            "event": "compound_applied",
            "label": op.label,
            "sub_events": sub_events,
        })

    else:
        raise ValueError(f"unknown operation class: {type(op).__name__}")

    return new_assembly, events


def apply_operation_sequence(
    assembly: Assembly3D,
    ops: List[Any],
) -> List[Tuple[Any, Assembly3D, List[Dict[str, Any]]]]:
    """
    Apply a sequence of operations in order. Returns a list of
    (op, assembly_after, events) for each operation. The original assembly
    is not modified.
    """
    results: List[Tuple[Any, Assembly3D, List[Dict[str, Any]]]] = []
    current = assembly
    for op in ops:
        current, events = apply_operation(current, op)
        results.append((op, current, events))
    return results


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("3d_scene_operations.py – smoke test (contact, release, reach)")
    print("=" * 70)

    apple = Object3D(
        object_id="apple_01",
        template_name="apple",
        elements=[
            make_element("sphere", {"radius_cm": 4.0}),
        ],
        material={"color": "red_brushed"},
        object_transform=make_transform(translate=[10.0, 5.0, 84.0]),
    )
    left_arm = Object3D(
        object_id="left_arm_01",
        template_name="arm",
        elements=[
            make_element(
                "cylinder",
                {"radius_cm": 4.0, "height_cm": 70.0},
                transform=make_transform(translate=[0.0, 0.0, -35.0]),
            ),
        ],
        material={"color": "flesh"},
        object_transform=make_transform(translate=[-25.0, 5.0, 155.0]),
    )
    left_hand = Object3D(
        object_id="left_hand_01",
        template_name="hand",
        elements=[
            make_element(
                "box",
                {"size_x_cm": 8.0, "size_y_cm": 2.0, "size_z_cm": 20.0},
                transform=make_transform(translate=[0.0, 0.0, -10.0]),
            ),
        ],
        material={"color": "flesh"},
        object_transform=make_transform(translate=[-25.0, 5.0, 85.0]),
        parent_object_id="left_arm_01",
    )
    left_arm.children = ["left_hand_01"]
    knife = Object3D(
        object_id="knife_01",
        template_name="knife",
        elements=[
            make_element("box", {"size_x_cm": 10.0, "size_y_cm": 2.0, "size_z_cm": 0.2},
                         transform=make_transform(translate=[-5.0, 0.0, 0.0])),
            make_element("box", {"size_x_cm": 10.0, "size_y_cm": 3.0, "size_z_cm": 1.0},
                         transform=make_transform(translate=[5.0, 0.0, 0.0])),
        ],
        material={"color": "silver"},
        object_transform=make_transform(translate=[20.0, 5.0, 80.5]),
    )
    board = Object3D(
        object_id="board_01",
        template_name="board",
        elements=[make_element("sheet", {"size_x_cm": 40.0, "size_y_cm": 30.0, "thickness_cm": 2.0})],
        material={"color": "wood"},
        object_transform=make_transform(translate=[10.0, 5.0, 79.0]),
    )

    assembly = Assembly3D(
        assembly_id="3d_ops_smoke",
        memory_id="ops_smoke",
        objects=[apple, left_arm, left_hand, knife, board],
    )
    if not hasattr(assembly, "contacts"):
        assembly.contacts = []

    ok, msg = assembly.validate()
    print(f"\nInitial assembly validation: {'OK' if ok else 'FAILED — ' + msg}")
    print(f"  {assembly.summary()}")

    # ---- 0. reach_arm_rotate: shoulder stays fixed ----
    print("\n[0] reach_arm_rotate: swing left arm to point at apple")
    shoulder_before = assembly.get_world_matrix("left_arm_01")[:3, 3].tolist()
    print(f"    shoulder before: {shoulder_before}")
    op = ReachArmRotateOp(
        object_id="left_arm_01",
        target_world=[10.0, 5.0, 84.0],
    )
    assembly, events = apply_operation(assembly, op)
    shoulder_after = assembly.get_world_matrix("left_arm_01")[:3, 3].tolist()
    print(f"    shoulder after:  {shoulder_after}")
    hand_world = assembly.get_world_matrix("left_hand_01")[:3, 3].tolist()
    print(f"    hand world after reach: {hand_world}")
    for e in events:
        print(f"    event: {e['event']} object={e['object_id']}")
    assert abs(shoulder_before[0] - shoulder_after[0]) < 1e-6
    assert abs(shoulder_before[1] - shoulder_after[1]) < 1e-6
    assert abs(shoulder_before[2] - shoulder_after[2]) < 1e-6
    print("    → shoulder did not move")

    # ---- 1. change_properties ----
    print("\n[1] change_properties: flesh oxidation")
    op = ChangePropertiesOp(
        object_id="apple_01",
        properties={"color": "light_brown", "oxidation": "light"},
    )
    assembly, events = apply_operation(assembly, op)
    for e in events:
        print(f"    event: {e['event']} object={e['object_id']} props={e.get('properties')}")

    # ---- 2. touch contact ----
    print("\n[2] contact (touch): left hand touches apple")
    op = ContactOp(
        object_a="left_hand_01",
        object_b="apple_01",
        contact_type="touch",
        starts_contact=True,
        contact_points=[
            {"point_world": [-6.0, 5.0, 88.0], "normal_world": [1.0, 0.0, 0.0]},
        ],
    )
    assembly, events = apply_operation(assembly, op)
    for e in events:
        print(f"    event: {e['event']} a={e['object_a']} b={e['object_b']} type={e.get('contact_type')}")
    print(f"    contacts now: {assembly.contacts}")

    # ---- 3. grip contact with attach ----
    print("\n[3] grip: left hand grips apple (with attach)")
    attach = AttachOp(child_id="apple_01", parent_id="left_hand_01", preserve_world=True)
    assembly, _ = apply_operation(assembly, attach)
    grip = ContactOp(
        object_a="left_hand_01",
        object_b="apple_01",
        contact_type="grip",
        starts_contact=True,
        force={"magnitude_n": 15.0, "direction_world": [0.0, 0.0, -1.0]},
        friction={"coefficient": 0.8, "regime": "static"},
        participants={"agent": "left_hand_01", "patient": "apple_01"},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assembly, events = apply_operation(assembly, grip)
        if caught:
            print(f"    soft warning: {caught[0].message}")
        else:
            print("    no soft warning (attach was in place)")
    for e in events:
        print(f"    event: {e['event']} a={e['object_a']} b={e['object_b']} type={e.get('contact_type')}")
    print(f"    contacts now: {[c['contact_type'] for c in assembly.contacts]}")

    # ---- 4. push ----
    print("\n[4] push: left hand pushes apple (no attach, no warning)")
    push = ContactOp(
        object_a="left_hand_01",
        object_b="apple_01",
        contact_type="push",
        starts_contact=True,
        force={"magnitude_n": 2.0, "direction_world": [0.0, 1.0, 0.0]},
        friction={"coefficient": 0.6, "regime": "kinetic"},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assembly, events = apply_operation(assembly, push)
        if caught:
            print(f"    soft warning: {caught[0].message}")
        else:
            print("    no soft warning (push does not imply attachment)")
    print(f"    contacts now: {[(c['object_a'], c['object_b'], c['contact_type']) for c in assembly.contacts]}")

    # ---- 5. grip without attach ----
    print("\n[5] grip without attach: soft warning")
    detach = DetachOp(child_id="apple_01")
    assembly, _ = apply_operation(assembly, detach)
    bad_grip = ContactOp(
        object_a="left_hand_01",
        object_b="knife_01",
        contact_type="grip",
        starts_contact=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assembly, events = apply_operation(assembly, bad_grip)
        if caught:
            print(f"    soft warning: {caught[0].message}")
        else:
            print("    ERROR: expected a soft warning")

    # ---- 6. release ----
    print("\n[6] release: left hand releases apple")
    release = ReleaseOp(object_a="left_hand_01", object_b="apple_01")
    assembly, events = apply_operation(assembly, release)
    for e in events:
        print(f"    event: {e['event']} a={e['object_a']} b={e['object_b']}")
    print(f"    contacts now: {[(c['object_a'], c['object_b'], c['contact_type']) for c in assembly.contacts]}")

    # ---- 7. release of nonexistent contact ----
    print("\n[7] invalid: release of nonexistent contact")
    try:
        bad_release = ReleaseOp(object_a="left_hand_01", object_b="apple_01")
        apply_operation(assembly, bad_release)
        print("    ERROR: should have raised")
    except ValueError as e:
        print(f"    correctly rejected: {e}")

    # ---- 8. invalid contact type ----
    print("\n[8] invalid: unknown contact_type")
    try:
        bad = ContactOp(
            object_a="left_hand_01",
            object_b="apple_01",
            contact_type="squeeze",
        )
        apply_operation(assembly, bad)
        print("    ERROR: should have raised")
    except ValueError as e:
        print(f"    correctly rejected: {e}")

    # ---- 9. both flags on ContactOp ----
    print("\n[9] invalid: starts_contact and ends_contact both true")
    try:
        bad = ContactOp(
            object_a="left_hand_01",
            object_b="apple_01",
            contact_type="touch",
            starts_contact=True,
            ends_contact=True,
        )
        apply_operation(assembly, bad)
        print("    ERROR: should have raised")
    except ValueError as e:
        print(f"    correctly rejected: {e}")

    # ---- 10. invalid reach target ----
    print("\n[10] invalid: reach target not 3-vector")
    try:
        bad = ReachArmRotateOp(object_id="left_arm_01", target_world=[1.0, 2.0])
        apply_operation(assembly, bad)
        print("    ERROR: should have raised")
    except ValueError as e:
        print(f"    correctly rejected: {e}")

    # ---- 11. registry round-trip ----
    print("\n[11] Registry round-trip: contact, release, reach")
    original_contact = ContactOp(
        object_a="left_hand_01",
        object_b="apple_01",
        contact_type="grip",
        force={"magnitude_n": 12.5, "direction_world": [0.0, -1.0, 0.0]},
        friction={"coefficient": 0.7, "regime": "static"},
        notes="test grip",
    )
    d = op_to_dict(original_contact)
    print(f"    serialized contact: {json.dumps(d)}")
    restored_contact = op_from_dict(d)
    print(f"    restored type: {type(restored_contact).__name__}")
    print(f"    restored contact_type: {restored_contact.contact_type}")
    print(f"    restored force: {restored_contact.force}")

    original_release = ReleaseOp(object_a="left_hand_01", object_b="apple_01")
    d2 = op_to_dict(original_release)
    print(f"    serialized release: {json.dumps(d2)}")
    restored_release = op_from_dict(d2)
    print(f"    restored type: {type(restored_release).__name__}")
    print(f"    restored object_a: {restored_release.object_a}")

    original_reach = ReachArmRotateOp(
        object_id="left_arm_01",
        target_world=[10.0, 5.0, 84.0],
        participants={"agent": "person_01", "destination": "apple_01"},
    )
    d3 = op_to_dict(original_reach)
    print(f"    serialized reach: {json.dumps(d3)}")
    restored_reach = op_from_dict(d3)
    print(f"    restored type: {type(restored_reach).__name__}")
    print(f"    restored target: {restored_reach.target_world}")

    print("\nSmoke test complete.")