#!/usr/bin/env python3
r"""
memory_model.py – Runtime classes for pickle memory search system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


class Object:
    """Runtime object instance."""

    def __init__(
        self,
        template_name: str,
        obj_id: str,
        number: str = "single",
        position: Optional[Dict[str, Any]] = None,
        position_relative_body: Optional[Dict[str, Any]] = None,
        dimensions: Optional[Dict[str, Any]] = None,
        weight: Optional[Dict[str, Any]] = None,
        mass: Optional[Dict[str, Any]] = None,
        shape: str = "",
        motion: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
        composite_of: Optional[List[str]] = None,
        functions: Optional[List[str]] = None,
        template_defaults: Optional[Dict[str, Any]] = None,
        categories: Optional[List[str]] = None,
        materials: Optional[List[str]] = None,
    ):
        self.template_name = template_name
        self.obj_id = obj_id
        self.number = number
        self.position = position or {}
        self.position_relative_body = position_relative_body or {}
        self.dimensions = dimensions or {}
        self.weight = weight or {}
        self.mass = mass or weight or {}
        self.shape = shape
        self.motion = motion or {}
        self.overrides = overrides or {}
        self.aliases = aliases or []
        self.composite_of = composite_of or []
        self.functions = functions or []
        self.template_defaults = template_defaults or {}
        self.categories = categories or []
        self.materials = materials or []

    @property
    def attributes(self) -> Dict[str, Any]:
        merged = dict(self.template_defaults)
        merged.update(self.overrides)
        return merged


class Protagonist(Object):
    def __init__(self, obj_id: str = "person_01", template_defaults: Optional[Dict[str, Any]] = None):
        super().__init__("person", obj_id, template_defaults=template_defaults)
        self.overrides.update(
            {
                "intention": None,
                "cheerfulness": 0.5,
                "focus": 0.8,
                "fatigue": 0.2,
                "satisfaction": 0.3,
                "sweat_level": 0.0,
                "breathing_rate": "normal",
                "energy_level": 0.8,
                "mental_state": {},
                "sensory_perception_touch": [],
                "sensory_perception_sight": [],
                "sensory_perception_smell": [],
                "sensory_perception_taste": [],
                "sensory_perception_sound": [],
            }
        )
        self.anatomical_features: List[str] = []


class GoalState:
    def __init__(
        self,
        target_object: Object,
        attribute: str,
        value: Any,
        aspect: Optional[str] = None,
        op: str = "eq",
    ):
        self.target_object = target_object
        self.attribute = attribute
        self.value = value
        self.aspect = aspect
        self.op = op


class VerbTemplate:
    def __init__(self, name: str):
        self.name = name
        self.action_templates: List["ActionGeneralTemplate"] = []

    def add_template(self, template: "ActionGeneralTemplate"):
        if template not in self.action_templates:
            self.action_templates.append(template)


class ActionGeneralTemplate:
    def __init__(
        self,
        name: str,
        verb: Optional[VerbTemplate] = None,
        default_duration: float = 1.0,
        action_category: str = "",
        kinematic_trajectory: str = "",
        general_sub_actions: Optional[List[str]] = None,
    ):
        self.name = name
        self.verb = verb
        self.default_duration = default_duration
        self.action_category = action_category
        self.kinematic_trajectory = kinematic_trajectory
        self.general_sub_actions = general_sub_actions or []
        self.alts: List[ActionAlt] = []

    def add_alt(self, alt: "ActionAlt"):
        self.alts.append(alt)
        alt.parent_general = self


class ActionAlt:
    def __init__(
        self,
        name: str,
        parent_general: Optional[ActionGeneralTemplate] = None,
        verb: Optional[VerbTemplate] = None,
        default_duration: float = 1.0,
        action_category: str = "",
        kinematic_trajectory: str = "",
        sub_actions: Optional[List[str]] = None,
    ):
        self.name = name
        self.parent_general = parent_general
        self.verb = verb
        self.default_duration = default_duration
        self.action_category = action_category
        self.kinematic_trajectory = kinematic_trajectory
        self.sub_actions = sub_actions or []


class ActionInstance:
    def __init__(
        self,
        instance_id: str,
        template_name: str,
        alt: Optional[ActionAlt] = None,
        general_template: Optional[ActionGeneralTemplate] = None,
        duration: float = 0.0,
        start_time: float = 0.0,
        end_time: float = 0.0,
        effective_duration: float = 0.0,
        participants: Optional[List[Object]] = None,
        changes: Optional[List[Dict[str, Any]]] = None,
        preconditions: Optional[List[Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None,
        action_category: str = "",
        kinematic_trajectory: str = "",
        temporal_type: str = "",
        repetitions: int = 1,
        cycle_duration: float = 0.0,
        sub_actions: Optional[List["ActionInstance"]] = None,
        path_parts: Optional[List[str]] = None,
    ):
        self.instance_id = instance_id
        self.template_name = template_name
        self.alt = alt
        self.general_template = general_template
        self.duration = duration
        self.start_time = start_time
        self.end_time = end_time
        self.effective_duration = effective_duration
        self.participants = participants or []
        self.changes = changes or []
        self.preconditions = preconditions or []
        self.tags = tags or []
        self.action_category = action_category
        self.kinematic_trajectory = kinematic_trajectory
        self.temporal_type = temporal_type
        self.repetitions = repetitions
        self.cycle_duration = cycle_duration
        self.sub_actions = sub_actions or []
        self.path_parts = path_parts or []


class Memory:
    def __init__(self, memory_id: str, protagonist: Protagonist):
        self.id = memory_id
        self.activity = ""
        self.protagonist = protagonist
        self.objects: Dict[str, Object] = {protagonist.obj_id: protagonist}
        self.actions: List[ActionInstance] = []
        self.goal_states: List[GoalState] = []
        self.alias_map: Dict[str, Object] = {}

    def add_object(self, obj: Object):
        self.objects[obj.obj_id] = obj

    def rebuild_alias_map(self):
        self.alias_map = {}
        for obj in self.objects.values():
            for alias in obj.aliases:
                self.alias_map[alias] = obj