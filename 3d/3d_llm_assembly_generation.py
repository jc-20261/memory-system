#!/usr/bin/env python3
r"""
3d_llm_assembly_generation.py – LLM-driven 3D assembly generation.

Two entry points:

    generate_assembly(memory_json) -> Assembly3D
        For one memory, produce the full 3D assembly. For each object:
            - look up its template in the element storage
            - if missing, ask the LLM for its element decomposition and
              store it for reuse
        Then call the LLM for scene layout (initial transforms) and the
        timeline (per-action keyframes), and assemble the final Assembly3D.

    batch_populate_storage(template_names)
        Scan a list of template names, skip the ones already in the storage,
        and ask the LLM for element decompositions of the rest. Useful for
        a one-time pass over the existing object templates.

World transforms vs local transforms:

    The LLM authors and emits WORLD transforms. It is much more reliable
    at placing objects in a global coordinate frame than at computing
    offsets relative to a parent.

    The runtime model stores LOCAL transforms on children, so that moving
    a parent moves the children and so that querying a child's world
    position composes the chain of local transforms.

    The conversion from world to local happens in `_build_assembly`, using
    `Assembly3D.set_world_transform`. Objects are added to the assembly
    first (with placeholder local transforms), then their world transforms
    are applied in topological order — parents before children — because
    the conversion needs the parent's world matrix to be already set.

Hierarchy consistency:

    The memory JSON declares `parent_object_id` on each object. Before
    building the assembly, `_populate_hierarchy_from_parents` populates
    each parent's `children` list from its children's `parent_object_id`,
    so that both directions of the relationship are consistent. This is a
    redundant safety net: `Assembly3D.__post_init__` and its `add_object`
    method also keep the two consistent, and `get_all_world_matrices`
    derives children locally from `parent_object_id`.

Both modes share the same LLM prompts and validation. The LLM client is
injectable so the smoke test can run offline with a mock.
"""

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_assembly = importlib.import_module("3d_assembly")
_elements = importlib.import_module("3d_elements_storage")

Assembly3D = _assembly.Assembly3D
Object3D = _assembly.Object3D
Element3D = _assembly.Element3D
Transform = _assembly.Transform
TimelineKeyframe = _assembly.TimelineKeyframe
make_element = _assembly.make_element
make_transform = _assembly.make_transform
make_assembly_id = _assembly.make_assembly_id

Elements3DStorage = _elements.Elements3DStorage
ElementEntry = _elements.ElementEntry

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================
MODULE_DIR = Path(__file__).parent
DATA_DIR = MODULE_DIR / "data"
ASSEMBLIES_DIR = MODULE_DIR / "assemblies"
MEMORIES_DIR = Path(r"F:\New folder (4)\New folder\Memories")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"

MAX_TOKENS = 8000
TEMPERATURE = 0.3
REASONING_EFFORT = "high"

COORDINATE_CONVENTION = (
    "Coordinate system: Z is up (right-handed). The ground is at z=0. "
    "Lengths are in centimeters. Common reference heights: floor=0cm, "
    "table=75cm, counter=80cm, eye level=160cm. Objects placed on a "
    "surface have their bottom face at the surface's z. Common object "
    "sizes: apple ~8cm diameter, knife ~20cm long, hand ~18cm long. "
    "Rotations are quaternions [x, y, z, w]. "
    "ALL TRANSFORMS YOU EMIT ARE WORLD TRANSFORMS. Do not compute them "
    "relative to any parent object. The pipeline converts them to local "
    "coordinates internally."
)

# =============================================================================
# Prompts
# =============================================================================

ELEMENT_DECOMPOSITION_PROMPT = f"""
You are a 3D geometry designer. Given an object's semantic description, produce
a decomposition of the object into primitive elements that approximates its
shape. {COORDINATE_CONVENTION}

Each element has:
    - element_type: one of sphere, ellipsoid, box, cylinder, cone, torus,
                    capsule, sheet, wedge, prism, pyramid, hemisphere, frustum
    - parameters: numbers for the type. All lengths in cm. Required parameters:
        sphere:      radius_cm
        ellipsoid:   radius_x_cm, radius_y_cm, radius_z_cm
        box:         size_x_cm, size_y_cm, size_z_cm
        cylinder:    radius_cm, height_cm
        cone:        radius_cm, height_cm
        torus:       major_radius_cm, minor_radius_cm
        capsule:     radius_cm, height_cm
        sheet:       size_x_cm, size_y_cm, thickness_cm
        wedge:       size_x_cm, size_y_cm, size_z_cm
        prism:       size_x_cm, size_y_cm, num_sides
        pyramid:     base_size_x_cm, base_size_y_cm, height_cm
        hemisphere:  radius_cm
        frustum:     bottom_radius_cm, top_radius_cm, height_cm
    - transform: {{translate: [x,y,z], rotate: [x,y,z,w], scale: [sx,sy,sz]}}
      All optional; identity if omitted. Use transform to position elements
      relative to the object's local origin (centroid).

Decompositions should be compact (1-6 elements). Use one element when the
object's overall shape is well captured by a single primitive (a ball is
just a sphere). Compose when needed (an apple is a sphere, a small cylinder
for the stem, and a torus for the calyx). Do not model surface details.

Return JSON with this shape:
{{
  "elements": [ {{...}}, ... ],
  "default_material": {{"color": "...", "roughness": 0.0-1.0, ...}},
  "default_mass_grams": <number>,
  "notes": "..."
}}

Only return the JSON. No commentary.
"""


SCENE_LAYOUT_PROMPT = f"""
You are a 3D scene arranger. Given a set of objects described semantically
and a set of initial conditions, produce WORLD-space transforms for each
object at the start of the memory. {COORDINATE_CONVENTION}

Use the semantic positions in the input (on counter, in left hand, etc.)
to infer plausible world coordinates. The counter top is at z=80. When
two objects are in contact ("on"), place them so their bounding volumes
just touch. When something is "in" another object, place it inside.
When a hand is "beside the body", place it at roughly x=±25, z=100.

CRITICAL: All transforms you produce are in WORLD coordinates. Do not
subtract the parent's transform. The pipeline handles parent-child
conversion internally.

Return JSON:
{{
  "object_transforms": {{
    "<obj_id>": {{"translate": [x,y,z], "rotate": [x,y,z,w], "scale": [sx,sy,sz]}}
  }},
  "environment": {{
    "ground": {{"present": true, "material": "..."}},
    "counter": {{"present": true, "top_z_cm": 80.0, "size_x_cm": ..., "size_y_cm": ...}},
    ...
  }}
}}

Only return the JSON. No commentary.
"""


TIMELINE_PROMPT = f"""
You are a 3D animator. Given a memory's action list with state changes, and
the initial transforms of all objects, produce a timeline of keyframes.
{COORDINATE_CONVENTION}

For each top-level action in the memory, produce one keyframe:
    - t: start time (seconds, cumulative from the previous keyframe's t+duration)
    - duration: the action's duration
    - action_instance_id: the action's instance_id (path in the memory)
    - action_template: the action's template name
    - object_transforms: the WORLD transforms of every moved object at the
      start of this keyframe (unchanged objects can be omitted; the renderer
      falls back to the previous keyframe)
    - geometry_events: optional list of {{type, ...}}. Common types:
        separation: {{"type": "separation", "source": "<obj_id>", "produces": "<new_obj_id>"}}
        contact:    {{"type": "contact", "a": "<obj_id>", "b": "<obj_id>"}}
        material_removal: {{"type": "material_removal", "target": "<obj_id>", "coverage_reduction": 0.0-1.0}}

Actions that do not move anything (perception, thought) can be omitted from
the timeline, but their duration still advances the clock.

All transforms in the timeline are WORLD transforms.

Return JSON:
{{
  "timeline": [
    {{"t": 0.0, "duration": 5.0, "action_instance_id": "...",
      "action_template": "...", "object_transforms": {{...}},
      "geometry_events": [...]}},
    ...
  ]
}}

Only return the JSON. No commentary.
"""


# =============================================================================
# Generator
# =============================================================================

class LLMAssemblyGenerator:
    """
    LLM-driven generator of Assembly3D objects from memory encodings.
    """

    def __init__(
        self,
        storage: Elements3DStorage,
        client: Any = None,
        model: str = MODEL_NAME,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        reasoning_effort: str = REASONING_EFFORT,
    ):
        self.storage = storage
        if client is None and OPENAI_AVAILABLE:
            client = AsyncOpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
            )
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_assembly(self, memory_json: Dict[str, Any]) -> Assembly3D:
        """
        Produce the 3D assembly for one memory. Uses the element storage
        where possible and asks the LLM only for what is missing.
        """
        memory_id = self._get_memory_id(memory_json)
        inner = memory_json.get("memory", memory_json)

        objects_data = inner.get("objects", [])
        actions_data = inner.get("actions", [])
        protagonist_data = inner.get("protagonist", {})

        # 1. Element decompositions (from storage, or generate + store)
        object_specs = await self._prepare_object_specs(
            objects_data, protagonist_data
        )

        # 2. Scene layout (LLM) — WORLD transforms
        layout = await self._infer_scene_layout(
            memory_json, object_specs
        )

        # 3. Timeline (LLM) — WORLD transforms per keyframe
        timeline_data = await self._infer_timeline(
            memory_json, actions_data, object_specs, layout
        )

        # 4. Build Assembly3D, converting world transforms to local
        #    via set_world_transform in topological order.
        assembly = self._build_assembly(
            memory_id=memory_id,
            object_specs=object_specs,
            layout=layout,
            timeline_data=timeline_data,
        )

        return assembly

    async def batch_populate_storage(
        self,
        template_names: List[str],
        context_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, bool]:
        """
        Populate the element storage for a list of templates.

        context_lookup maps template_name -> template defaults from
        object_storage.json, used to give the LLM useful context.

        Returns a dict mapping template_name -> True (added) / False (failed).
        """
        results: Dict[str, bool] = {}
        for template_name in template_names:
            if self.storage.has_geometry(template_name):
                results[template_name] = True
                continue
            context = (context_lookup or {}).get(template_name, {})
            try:
                entry = await self._generate_element_entry(template_name, context)
                if entry is None:
                    results[template_name] = False
                    continue
                ok, msg = self.storage.add_template(entry)
                results[template_name] = ok
                if not ok:
                    print(f"  Failed to add '{template_name}': {msg}")
            except Exception as e:
                print(f"  Exception generating '{template_name}': {e}")
                results[template_name] = False
        return results

    # ------------------------------------------------------------------
    # Memory parsing helpers
    # ------------------------------------------------------------------

    def _get_memory_id(self, memory_json: Dict[str, Any]) -> str:
        mid = memory_json.get("memory_id")
        if mid:
            return str(mid)
        inner = memory_json.get("memory", {})
        mid = inner.get("memory_id")
        if mid:
            return str(mid)
        return "unknown"

    def _extract_template_name(self, obj_data: Dict[str, Any]) -> str:
        return obj_data.get("template") or obj_data.get("template_name") or ""

    def _extract_context_for_template(
        self, obj_data: Dict[str, Any], templates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build a context dict for the LLM from the memory's object data."""
        template_name = self._extract_template_name(obj_data)
        template_info = templates.get(template_name, {}) if templates else {}
        context: Dict[str, Any] = {
            "template_name": template_name,
            "dimensions": obj_data.get("dimensions") or template_info.get("dimensions", {}),
            "shape": obj_data.get("shape") or template_info.get("shape", ""),
            "materials": obj_data.get("materials") or template_info.get("materials", []),
            "functions": obj_data.get("functions") or template_info.get("functions", []),
            "categories": obj_data.get("categories") or template_info.get("categories", []),
        }
        return context

    # ------------------------------------------------------------------
    # Element preparation
    # ------------------------------------------------------------------

    async def _prepare_object_specs(
        self,
        objects_data: List[Dict[str, Any]],
        protagonist_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Return a list of object spec dicts, each containing:
            obj_id, template_name, elements (list of Element3D),
            material, mass_grams, parent_object_id, children,
            raw_overrides, position_hint
        """
        specs: List[Dict[str, Any]] = []

        # Protagonist first, if present
        if protagonist_data:
            protag_id = protagonist_data.get("obj_id", "person_01")
            protag_template = protagonist_data.get("template", "person")
            elements = await self._get_or_generate_elements(
                protag_template,
                {
                    "template_name": protag_template,
                    "dimensions": protagonist_data.get("dimensions", {}),
                    "shape": protagonist_data.get("shape", "humanoid"),
                    "materials": ["flesh"],
                    "functions": ["agent", "observer"],
                    "categories": ["person", "physical_object", "object"],
                },
            )
            material = self.storage.get_default_material(protag_template) or {}
            mass = self.storage.get_default_mass(protag_template) or 70000.0
            specs.append({
                "obj_id": protag_id,
                "template_name": protag_template,
                "elements": elements,
                "material": material,
                "mass_grams": mass,
                "parent_object_id": None,
                "children": [],
                "raw_overrides": protagonist_data.get("overrides", {}),
                "position_hint": None,
            })

        # Regular objects
        for obj_data in objects_data:
            if not isinstance(obj_data, dict):
                continue
            template_name = self._extract_template_name(obj_data)
            obj_id = obj_data.get("obj_id")
            if not template_name or not obj_id:
                continue

            context = self._extract_context_for_template(obj_data, {})
            elements = await self._get_or_generate_elements(template_name, context)
            material = self.storage.get_default_material(template_name) or {}
            # Merge instance material overrides if present
            instance_material = obj_data.get("material")
            if isinstance(instance_material, dict):
                material = {**material, **instance_material}
            mass = self.storage.get_default_mass(template_name)
            if mass is None:
                mass = obj_data.get("mass_grams") or obj_data.get("mass")
                if isinstance(mass, dict):
                    mass = mass.get("value")

            # Composite hierarchy: the memory's composite_of field lists
            # sub-object template names; those sub-objects appear as
            # separate obj_data entries and are linked by parent_object_id.
            parent_object_id = obj_data.get("parent_object_id")
            children = [
                c for c in (obj_data.get("children") or [])
                if isinstance(c, str)
            ]

            specs.append({
                "obj_id": obj_id,
                "template_name": template_name,
                "elements": elements,
                "material": material,
                "mass_grams": mass,
                "parent_object_id": parent_object_id,
                "children": children,
                "raw_overrides": obj_data.get("overrides", {}),
                "position_hint": obj_data.get("position"),
            })

        # Link parent <-> children automatically if not explicitly set
        self._link_composite_parents(specs)
        # Populate every parent's children list from its children's
        # parent_object_id, so the specs are internally consistent.
        self._populate_hierarchy_from_parents(specs)

        return specs

    def _link_composite_parents(self, specs: List[Dict[str, Any]]):
        """
        For any spec whose template has composite_children_templates in the
        storage, look for matching child specs and link them. Also fill in
        missing parent links.
        """
        by_template: Dict[str, List[Dict[str, Any]]] = {}
        for spec in specs:
            by_template.setdefault(spec["template_name"], []).append(spec)

        for parent_spec in specs:
            child_templates = self.storage.get_composite_children(
                parent_spec["template_name"]
            )
            for ct in child_templates:
                for child_spec in by_template.get(ct, []):
                    if child_spec["parent_object_id"] is None:
                        child_spec["parent_object_id"] = parent_spec["obj_id"]
                    if child_spec["obj_id"] not in parent_spec["children"]:
                        parent_spec["children"].append(child_spec["obj_id"])

    def _populate_hierarchy_from_parents(self, specs: List[Dict[str, Any]]):
        """
        Populate each spec's `children` list from the parent_object_id of
        every other spec.

        The memory JSON declares `parent_object_id` on each object. The
        `children` list is a derived field. This method makes the two
        consistent before the assembly is built, so that both directions
        of the hierarchy are correct.
        """
        by_id = {spec["obj_id"]: spec for spec in specs}
        for spec in specs:
            parent_id = spec.get("parent_object_id")
            if parent_id is None:
                continue
            parent = by_id.get(parent_id)
            if parent is None:
                continue
            if spec["obj_id"] not in parent["children"]:
                parent["children"].append(spec["obj_id"])

    async def _get_or_generate_elements(
        self,
        template_name: str,
        context: Dict[str, Any],
    ) -> List[Element3D]:
        """
        Return elements for a template. Pull from storage if present.
        Otherwise call the LLM and add the result to the storage.
        """
        if self.storage.has_geometry(template_name):
            return self.storage.get_elements(template_name)

        entry = await self._generate_element_entry(template_name, context)
        if entry is None:
            # Fallback: a minimal placeholder so the assembly can still be built
            return [make_element("box", {"size_x_cm": 5.0, "size_y_cm": 5.0, "size_z_cm": 5.0})]

        ok, msg = self.storage.add_template(entry)
        if not ok:
            print(f"Warning: could not store elements for '{template_name}': {msg}")
        return entry.elements

    async def _generate_element_entry(
        self,
        template_name: str,
        context: Dict[str, Any],
    ) -> Optional[ElementEntry]:
        """Call the LLM for an element decomposition and parse the result."""
        user_msg = json.dumps(
            {
                "template_name": template_name,
                "dimensions": context.get("dimensions", {}),
                "shape": context.get("shape", ""),
                "materials": context.get("materials", []),
                "functions": context.get("functions", []),
                "categories": context.get("categories", []),
            },
            indent=2,
            ensure_ascii=False,
        )
        data = await self._call_llm_json(
            ELEMENT_DECOMPOSITION_PROMPT, user_msg, "element_decomposition"
        )
        if not isinstance(data, dict):
            return None

        raw_elements = data.get("elements", [])
        elements: List[Element3D] = []
        for el in raw_elements:
            if not isinstance(el, dict):
                continue
            try:
                elements.append(Element3D.from_dict(el))
            except Exception:
                continue
        if not elements:
            return None

        entry = ElementEntry(
            template_name=template_name,
            elements=elements,
            default_material=dict(data.get("default_material", {})),
            default_mass_grams=data.get("default_mass_grams"),
            notes=data.get("notes", ""),
        )
        ok, msg = entry.validate()
        if not ok:
            print(f"  LLM-generated entry for '{template_name}' failed validation: {msg}")
            return None
        return entry

    # ------------------------------------------------------------------
    # Scene layout
    # ------------------------------------------------------------------

    async def _infer_scene_layout(
        self,
        memory_json: Dict[str, Any],
        object_specs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        inner = memory_json.get("memory", memory_json)
        initial_conditions = memory_json.get("initial_conditions") or inner.get("initial_conditions") or []

        objects_summary = []
        for spec in object_specs:
            objects_summary.append({
                "obj_id": spec["obj_id"],
                "template_name": spec["template_name"],
                "position_hint": spec.get("position_hint"),
                "dimensions": self._dimensions_for_template(spec["template_name"]),
            })

        user_msg = json.dumps({
            "memory_id": self._get_memory_id(memory_json),
            "objects": objects_summary,
            "initial_conditions": initial_conditions,
        }, indent=2, ensure_ascii=False)

        data = await self._call_llm_json(SCENE_LAYOUT_PROMPT, user_msg, "scene_layout")
        if not isinstance(data, dict):
            return {"object_transforms": {}, "environment": {}}

        # Coerce transforms to Transform objects via from_dict
        obj_transforms = data.get("object_transforms", {})
        if not isinstance(obj_transforms, dict):
            obj_transforms = {}
        return {
            "object_transforms": obj_transforms,
            "environment": data.get("environment", {}),
        }

    def _dimensions_for_template(self, template_name: str) -> Dict[str, Any]:
        entry = self.storage.get_template(template_name)
        if entry is None:
            return {}
        return entry.default_dimensions_cm or {}

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    async def _infer_timeline(
        self,
        memory_json: Dict[str, Any],
        actions_data: List[Dict[str, Any]],
        object_specs: List[Dict[str, Any]],
        layout: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not actions_data:
            return []

        actions_summary = []
        for act in actions_data:
            if not isinstance(act, dict):
                continue
            actions_summary.append({
                "template": act.get("template"),
                "duration": act.get("duration"),
                "participants": act.get("participants", []),
                "changes": act.get("changes", []),
                "changes_total": act.get("changes_total", []),
                "kinematic_trajectory": act.get("kinematic_trajectory"),
                "temporal_type": act.get("temporal_type"),
            })

        user_msg = json.dumps({
            "memory_id": self._get_memory_id(memory_json),
            "initial_object_transforms": layout.get("object_transforms", {}),
            "objects": [{"obj_id": s["obj_id"], "template_name": s["template_name"]} for s in object_specs],
            "actions": actions_summary,
        }, indent=2, ensure_ascii=False)

        data = await self._call_llm_json(TIMELINE_PROMPT, user_msg, "timeline")
        if not isinstance(data, dict):
            return []
        timeline = data.get("timeline", [])
        return timeline if isinstance(timeline, list) else []

    # ------------------------------------------------------------------
    # Assembly construction
    # ------------------------------------------------------------------

    def _topological_order(
        self, object_specs: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Return the object ids in top-down order: roots first, then their
        children, then grandchildren, and so on. Used so that when a world
        transform is applied to a child via set_world_transform, its
        parent's world matrix has already been set.

        Cycles (which should not occur) are broken by visiting each id
        once, in input order.
        """
        by_id = {spec["obj_id"]: spec for spec in object_specs}
        order: List[str] = []
        visited = set()

        def visit(obj_id: str):
            if obj_id in visited:
                return
            visited.add(obj_id)
            spec = by_id.get(obj_id)
            if spec is None:
                return
            parent_id = spec.get("parent_object_id")
            if parent_id is not None and parent_id in by_id:
                visit(parent_id)
            order.append(obj_id)

        for spec in object_specs:
            visit(spec["obj_id"])

        return order

    def _build_assembly(
        self,
        memory_id: str,
        object_specs: List[Dict[str, Any]],
        layout: Dict[str, Any],
        timeline_data: List[Dict[str, Any]],
    ) -> Assembly3D:
        """
        Build the Assembly3D from object specs and layout.

        Steps:
            1. Create every Object3D with a placeholder identity local
               transform, carrying hierarchy, elements, material, mass.
            2. Add all objects to the assembly so the parent-child graph
               is complete.
            3. Apply the LLM-provided world transforms in topological
               order via `set_world_transform`, which converts each world
               transform to a local transform relative to the parent.
            4. Attach the timeline.

        The LLM-authored world transforms are respected exactly; the
        runtime model stores local transforms so that moving a parent
        moves its children.
        """
        layout_transforms = layout.get("object_transforms", {}) or {}

        # ---- Step 1: create all objects with identity local transforms
        objects: List[Object3D] = []
        for spec in object_specs:
            objects.append(Object3D(
                object_id=spec["obj_id"],
                template_name=spec["template_name"],
                elements=list(spec["elements"]),
                material=dict(spec.get("material") or {}),
                object_transform=Transform.identity(),
                mass_grams=spec.get("mass_grams"),
                parent_object_id=spec.get("parent_object_id"),
                children=list(spec.get("children") or []),
            ))

        # ---- Step 2: assemble hierarchy. Assembly3D.__post_init__ will
        #      reconcile children lists from parent_object_id, so the
        #      hierarchy is consistent regardless of what the specs
        #      declared.
        assembly = Assembly3D(
            assembly_id=make_assembly_id(memory_id),
            memory_id=memory_id,
            objects=objects,
            environment=layout.get("environment", {}),
            timeline=[],
        )

        # ---- Step 3: apply world transforms in topological order
        order = self._topological_order(object_specs)
        for obj_id in order:
            world_transform_dict = layout_transforms.get(obj_id)
            if world_transform_dict is None:
                # No world transform provided. Leave the placeholder
                # identity local transform in place. For a root, this
                # means the object sits at the origin. For a child, this
                # means the object sits at the parent's world position.
                continue
            try:
                world_transform = Transform.from_dict(world_transform_dict)
            except Exception as e:
                print(f"  Warning: could not parse world transform for '{obj_id}': {e}")
                continue
            try:
                assembly.set_world_transform(obj_id, world_transform)
            except Exception as e:
                print(f"  Warning: could not apply world transform for '{obj_id}': {e}")

        # ---- Step 4: attach timeline
        timeline: List[TimelineKeyframe] = []
        for kf_data in timeline_data:
            if not isinstance(kf_data, dict):
                continue
            try:
                kf = TimelineKeyframe.from_dict(kf_data)
                timeline.append(kf)
            except Exception:
                continue
        assembly.timeline = timeline

        return assembly

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    async def _call_llm_json(
        self,
        system_msg: str,
        user_msg: str,
        purpose: str,
    ) -> Optional[Dict[str, Any]]:
        if self.client is None:
            print(f"[{purpose}] no LLM client configured")
            return None
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body={"reasoning_effort": self.reasoning_effort},
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content)
        except Exception as e:
            print(f"[{purpose}] LLM call failed: {e}")
            return None


# =============================================================================
# Mock LLM client for offline smoke testing
# =============================================================================

class MockLLMChoice:
    def __init__(self, content: str):
        self.message = type("Msg", (), {"content": content})()


class MockLLMResponse:
    def __init__(self, content: str):
        self.choices = [MockLLMChoice(content)]


class MockLLMChat:
    def __init__(self):
        self.completions = self

    async def create(self, *, model, messages, **kwargs):
        # Decide response based on the system message content
        system = messages[0]["content"] if messages else ""
        if "geometry designer" in system:
            return MockLLMResponse(json.dumps({
                "elements": [
                    {"element_type": "sphere", "parameters": {"radius_cm": 4.0},
                     "transform": {"translate": [0,0,0], "rotate": [0,0,0,1], "scale": [1,1,1]}},
                    {"element_type": "cylinder", "parameters": {"radius_cm": 0.2, "height_cm": 2.0},
                     "transform": {"translate": [0,0,4.5], "rotate": [0,0,0,1], "scale": [1,1,1]}},
                ],
                "default_material": {"color": "red_brushed", "roughness": 0.4},
                "default_mass_grams": 200.0,
                "notes": "Mock apple"
            }))
        if "scene arranger" in system:
            # All values are WORLD transforms.
            # Hierarchy per the smoke test's memory JSON:
            #   person_01 (root)
            #     └── left_hand_01
            #           └── apple_01
            #   knife_01 (root)
            return MockLLMResponse(json.dumps({
                "object_transforms": {
                    "person_01":    {"translate": [-30, 0, 0],     "rotate": [0,0,0,1], "scale": [1,1,1]},
                    "left_hand_01": {"translate": [-25, 0, 100],   "rotate": [0,0,0,1], "scale": [1,1,1]},
                    "apple_01":     {"translate": [-25, 0, 104],   "rotate": [0,0,0,1], "scale": [1,1,1]},
                    "knife_01":     {"translate": [15, 5, 80.5],   "rotate": [0,0,0,1], "scale": [1,1,1]},
                },
                "environment": {
                    "ground": {"present": True, "material": "wood"},
                    "counter": {"present": True, "top_z_cm": 80.0, "size_x_cm": 120.0, "size_y_cm": 60.0},
                }
            }))
        if "3D animator" in system:
            return MockLLMResponse(json.dumps({
                "timeline": [
                    {"t": 0.0, "duration": 5.0, "action_template": "prepare_objects",
                     "object_transforms": {"apple_01": {"translate": [-25,0,104], "rotate": [0,0,0,1], "scale": [1,1,1]}}},
                    {"t": 5.0, "duration": 40.0, "action_template": "continuous_peeling",
                     "object_transforms": {"apple_01": {"translate": [-25,0,104], "rotate": [0,0,0.383,0.924], "scale": [1,1,1]}},
                     "geometry_events": [{"type": "separation", "source": "apple_01", "produces": "peel_strip_01"}]},
                ]
            }))
        # Fallback
        return MockLLMResponse("{}")


class MockLLMClient:
    def __init__(self):
        self.chat = MockLLMChat()


# =============================================================================
# Smoke test
# =============================================================================

async def _smoke_test():
    print("=" * 70)
    print("3d_llm_assembly_generation.py – smoke test (offline, mock LLM)")
    print("=" * 70)

    storage = Elements3DStorage()

    # Pre-seed the storage with knife, person, and left_hand so only apple
    # is LLM-generated.
    storage.add_template(ElementEntry(
        template_name="knife",
        elements=[
            make_element("box", {"size_x_cm": 10.0, "size_y_cm": 2.0, "size_z_cm": 0.2},
                         transform=make_transform(translate=[-5.0, 0.0, 0.0])),
            make_element("box", {"size_x_cm": 10.0, "size_y_cm": 3.0, "size_z_cm": 1.0},
                         transform=make_transform(translate=[5.0, 0.0, 0.0])),
        ],
        default_material={"color": "silver", "metalness": 1.0},
        default_mass_grams=150.0,
        default_dimensions_cm={"length_cm": 20.0, "width_cm": 3.0, "height_thickness_cm": 1.2},
    ))
    storage.add_template(ElementEntry(
        template_name="person",
        elements=[
            make_element("capsule", {"radius_cm": 15.0, "height_cm": 120.0},
                         transform=make_transform(translate=[0,0,80])),
            make_element("sphere", {"radius_cm": 10.0},
                         transform=make_transform(translate=[0,0,165])),
        ],
        default_material={"color": "flesh"},
        default_mass_grams=70000.0,
    ))
    storage.add_template(ElementEntry(
        template_name="left_hand",
        elements=[make_element("box", {"size_x_cm": 8.0, "size_y_cm": 2.0, "size_z_cm": 18.0})],
        default_material={"color": "flesh"},
        default_mass_grams=400.0,
    ))

    mock_client = MockLLMClient()
    gen = LLMAssemblyGenerator(storage=storage, client=mock_client)

    # Build a mock memory JSON.
    # Hierarchy: person_01 (root) -> left_hand_01 -> apple_01
    #            knife_01 (root)
    memory_json = {
        "memory_id": "mem_test_002",
        "activity": "peeling an apple",
        "memory": {
            "memory_id": "mem_test_002",
            "protagonist": {
                "template": "person",
                "obj_id": "person_01",
                "dimensions": {"height_thickness": {"value": 170, "unit": "cm"}},
                "shape": "humanoid",
            },
            "objects": [
                {
                    "template": "left_hand",
                    "obj_id": "left_hand_01",
                    "parent_object_id": "person_01",
                },
                {
                    "template": "apple",
                    "obj_id": "apple_01",
                    "parent_object_id": "left_hand_01",
                    "position": {"relation": "held_by", "relative_to": "left_hand_01"},
                    "dimensions": {
                        "length": {"value": 8, "unit": "cm"},
                        "width": {"value": 8, "unit": "cm"},
                        "height_thickness": {"value": 8, "unit": "cm"},
                    },
                    "shape": "spherical",
                    "materials": ["apple_skin", "apple_flesh"],
                },
                {
                    "template": "knife",
                    "obj_id": "knife_01",
                    "position": {"relation": "on", "relative_to": "counter"},
                },
            ],
            "actions": [
                {
                    "template": "prepare_objects",
                    "duration": 5,
                    "participants": ["left_hand_01", "apple_01"],
                },
                {
                    "template": "continuous_peeling",
                    "duration": 40,
                    "participants": ["left_hand_01", "knife_01", "apple_01"],
                    "changes": [
                        {"object": "apple_01", "attribute": "skin.present",
                         "old": True, "new": False},
                    ],
                },
            ],
        },
    }

    print("\nGenerating assembly...")
    assembly = await gen.generate_assembly(memory_json)

    print(f"\n{assembly.summary()}")

    ok, msg = assembly.validate()
    print(f"Validation: {'OK' if ok else 'FAILED — ' + msg}")

    # ----------------------------------------------------------------
    # Fix verification: hierarchy consistency
    # ----------------------------------------------------------------
    print("\nHierarchy consistency:")
    person = assembly.get_object("person_01")
    left_hand = assembly.get_object("left_hand_01")
    apple = assembly.get_object("apple_01")
    knife = assembly.get_object("knife_01")
    print(f"  person_01.parent = {person.parent_object_id}, children = {person.children}")
    print(f"  left_hand_01.parent = {left_hand.parent_object_id}, children = {left_hand.children}")
    print(f"  apple_01.parent = {apple.parent_object_id}, children = {apple.children}")
    print(f"  knife_01.parent = {knife.parent_object_id}, children = {knife.children}")
    assert left_hand.object_id in person.children, "person_01.children must contain left_hand_01"
    assert apple.object_id in left_hand.children, "left_hand_01.children must contain apple_01"
    assert knife.parent_object_id is None and knife.children == []
    print("  → hierarchy is bidirectionally consistent")

    # ----------------------------------------------------------------
    # Fix verification: world positions of all objects
    # ----------------------------------------------------------------
    print("\nObjects (with world positions):")
    world = assembly.get_all_world_matrices()
    for obj in assembly.objects:
        M = world.get(obj.object_id)
        pos = "?"
        if M is not None:
            pos = f"({M[0,3]:.1f}, {M[1,3]:.1f}, {M[2,3]:.1f})"
        print(f"  {obj.object_id} ({obj.template_name}) — "
              f"parent={obj.parent_object_id}, children={obj.children}, world_pos={pos}")
    assert "left_hand_01" in world, "left_hand_01 must have a world matrix"
    assert "apple_01" in world, "apple_01 must have a world matrix"
    # Expected world positions from the mock
    assert abs(world["person_01"][0,3]   - (-30.0)) < 1e-4
    assert abs(world["left_hand_01"][0,3] - (-25.0)) < 1e-4
    assert abs(world["left_hand_01"][2,3] -  100.0)  < 1e-4
    assert abs(world["apple_01"][0,3]     - (-25.0)) < 1e-4
    assert abs(world["apple_01"][2,3]     -  104.0)  < 1e-4
    print("  → all four objects have the expected world positions")

    # ----------------------------------------------------------------
    # Fix verification: propagation through the hierarchy
    # ----------------------------------------------------------------
    print("\nPropagation test:")
    # Move left_hand_01 to a new world position. apple_01's local transform
    # is unchanged, so its world position should move by the same delta.
    old_apple_pos = world["apple_01"][:3, 3].copy()
    new_hand_world = Transform(
        translate=[-20.0, 0.0, 100.0],
        rotate=[0.0, 0.0, 0.0, 1.0],
        scale=[1.0, 1.0, 1.0],
    )
    residual = assembly.set_world_transform("left_hand_01", new_hand_world)
    print(f"  set_world_transform(left_hand_01, [-20, 0, 100]) residual = {residual:.6f}")
    world_after = assembly.get_all_world_matrices()
    new_apple_pos = world_after["apple_01"][:3, 3]
    delta_hand = 5.0  # -25 -> -20 in x
    delta_apple = new_apple_pos[0] - old_apple_pos[0]
    print(f"  apple_01 was at ({old_apple_pos[0]:.2f}, {old_apple_pos[1]:.2f}, {old_apple_pos[2]:.2f})")
    print(f"  apple_01 now at ({new_apple_pos[0]:.2f}, {new_apple_pos[1]:.2f}, {new_apple_pos[2]:.2f})")
    print(f"  hand moved +{delta_hand:.2f} in x; apple moved +{delta_apple:.2f} in x")
    assert abs(delta_apple - delta_hand) < 1e-4, "apple should follow the hand's move"
    assert abs(new_apple_pos[2] - 104.0) < 1e-4, "apple z should be unchanged"
    print("  → apple_01 follows left_hand_01 as expected")

    # ----------------------------------------------------------------
    # Timeline
    # ----------------------------------------------------------------
    print("\nTimeline:")
    for kf in assembly.timeline:
        print(f"  t={kf.t:.1f}s dur={kf.duration:.1f}s template={kf.action_template} "
              f"transforms={list(kf.object_transforms.keys())} "
              f"geometry_events={len(kf.geometry_events)}")

    # Storage was mutated by the generator (apple added)
    print(f"\nStorage after generation: {storage.summary()}")
    print(f"  'apple' now in storage: {storage.has_template('apple')}")

    # Save
    path = assembly.save()
    print(f"\nSaved to: {path}")

    # Reload
    reloaded = Assembly3D.load(path)
    ok2, msg2 = reloaded.validate()
    print(f"Reloaded from disk: {'OK' if ok2 else 'FAILED — ' + msg2}")
    print(f"  {reloaded.summary()}")

    # Round-trip equality
    if json.dumps(assembly.to_dict(), sort_keys=True) == json.dumps(
        reloaded.to_dict(), sort_keys=True
    ):
        print("Round-trip serialization is identical.")
    else:
        print("WARNING: round-trip serialization differs.")

    print("\nSmoke test complete.")


if __name__ == "__main__":
    asyncio.run(_smoke_test())