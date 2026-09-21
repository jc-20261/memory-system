#!/usr/bin/env python3
r"""
3d_assembly_builder.py – Build Assembly3D objects from resolved data.

This is a pure helper. It makes no LLM calls. It holds no prompts. It
does not know what an API key is. All prompts and LLM calls live in
3d_mgen.py, which calls this module only after everything has been
resolved.

The builder takes:

    memory_id        – the memory identifier
    protagonist_spec – {template, obj_id, overrides, anatomical_features}
    object_specs     – list of object specs (from Stage 3a entity extraction)
    element_map      – {template_name: [Element3D, ...]}
    scene_layout     – {obj_id: world_transform_dict} from Stage 3c
    environment      – dict from Stage 3c

and returns a fully built Assembly3D.

Internally the builder:

    1. Instantiates the protagonist as a Protagonist (a special Object3D).
    2. Instantiates every other object as an Object3D, copying its
       elements from the element_map and merging storage defaults with
       the spec's overrides.
    3. Adds all objects to the assembly so the parent-child graph is
       complete.
    4. Applies the scene layout's world transforms in topological order
       (parents before children) via Assembly3D.set_world_transform,
       which converts each world transform to a local transform relative
       to the parent.
    5. Returns the assembly.

World vs local transforms:

    The LLM authors and emits WORLD transforms. The runtime model stores
    LOCAL transforms on children, so that moving a parent moves its
    children. The conversion happens here, in step 4, using
    set_world_transform. This mirrors the design decision recorded in
    the module docstring of 3d_assembly.py.

Hierarchy consistency:

    The memory JSON declares parent_object_id on each object. Before
    building, _populate_hierarchy_from_parents populates each parent's
    children list from its children's parent_object_id. Assembly3D's
    own __post_init__ also rebuilds children from parent_object_id,
    so the two are always consistent.

Contact and timeline:

    The builder does not produce contacts or a timeline. Contacts are
    produced by Stage 4 operations (contact/release). The timeline is
    produced by Stage 4 operations as well, applied frame by frame.
    The initial assembly has an empty timeline and an empty contacts
    list.
"""

import copy
import importlib
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
make_element = _assembly.make_element
make_transform = _assembly.make_transform
make_assembly_id = _assembly.make_assembly_id

Elements3DStorage = _elements.Elements3DStorage


# =============================================================================
# Protagonist
# =============================================================================
class Protagonist(Object3D):
    """
    A special Object3D representing the person whose memory this is.

    Adds two fields on top of Object3D:
        - anatomical_features: a list of obj_ids of body parts that are
          children of the protagonist (hands, eyes, thumbs, etc.)
        - The mental_state and sensory_perception_* overrides are
          populated by the caller from the entity extraction.

    The protagonist is still an Object3D; it participates in the same
    hierarchy and world-transform machinery.
    """

    def __init__(self, object_id: str, template_name: str = "person"):
        super().__init__(
            object_id=object_id,
            template_name=template_name,
            elements=[],
            material={},
            object_transform=Transform.identity(),
            mass_grams=None,
            parent_object_id=None,
            children=[],
        )
        self.anatomical_features: List[str] = []


# =============================================================================
# Builder
# =============================================================================
class AssemblyBuilder:
    """
    Builds Assembly3D objects from fully resolved inputs. No LLM calls.
    """

    def __init__(self, storage: Elements3DStorage):
        self.storage = storage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_assembly(
        self,
        memory_id: str,
        protagonist_spec: Dict[str, Any],
        object_specs: List[Dict[str, Any]],
        element_map: Dict[str, List[Element3D]],
        scene_layout: Dict[str, Dict[str, Any]],
        environment: Optional[Dict[str, Any]] = None,
    ) -> Assembly3D:
        """
        Build the initial Assembly3D for a memory.

        Parameters
        ----------
        memory_id : str
            The memory identifier.

        protagonist_spec : dict
            {
                "template": "person",
                "obj_id": "person_01",
                "overrides": {...},                # optional
                "anatomical_features": [...],     # optional
            }

        object_specs : list of dict
            Each spec has at least:
                obj_id, template, parent_object_id (or None), number
            And optionally:
                overrides, dimensions, shape, materials,
                position, position_relative_body, motion, aliases,
                composite_of, functions, children

        element_map : dict
            {template_name: [Element3D, ...]}. Every template used by
            the protagonist or by any object spec must have an entry.
            Missing templates get a fallback placeholder box and a
            warning is printed.

        scene_layout : dict
            {obj_id: world_transform_dict}. Every obj_id present in the
            assembly may have an entry. Objects without an entry keep
            an identity local transform (roots sit at origin; children
            sit at their parent's world position).

        environment : dict, optional
            Environment metadata, e.g. ground, counter, room. Stored
            verbatim on the assembly.

        Returns
        -------
        Assembly3D
        """
        scene_layout = scene_layout or {}
        environment = environment or {}

        # ---- Step 1: build the protagonist
        protagonist = self._build_protagonist(protagonist_spec, element_map)

        # ---- Step 2: build every other object
        objects: List[Object3D] = [protagonist]
        for spec in object_specs:
            obj = self._build_object(spec, element_map)
            objects.append(obj)

        # ---- Step 3: assemble hierarchy
        assembly = Assembly3D(
            assembly_id=make_assembly_id(memory_id),
            memory_id=memory_id,
            objects=objects,
            environment=dict(environment),
            timeline=[],
            contacts=[],
        )

        # ---- Step 4: apply world transforms in topological order
        # Build a combined spec list for topological ordering.
        all_specs: List[Dict[str, Any]] = [
            {
                "obj_id": protagonist.object_id,
                "parent_object_id": protagonist.parent_object_id,
            }
        ]
        for spec in object_specs:
            all_specs.append({
                "obj_id": spec["obj_id"],
                "parent_object_id": spec.get("parent_object_id"),
            })

        order = self._topological_order(all_specs)
        for obj_id in order:
            world_transform_dict = scene_layout.get(obj_id)
            if world_transform_dict is None:
                # No world transform provided. Leave the identity local
                # transform in place. For a root this means the object
                # sits at the origin. For a child it means the object
                # sits at the parent's world position.
                continue
            try:
                world_transform = Transform.from_dict(world_transform_dict)
            except Exception as e:
                print(f"  [assembly_builder] could not parse world transform "
                      f"for '{obj_id}': {e}")
                continue
            try:
                assembly.set_world_transform(obj_id, world_transform)
            except Exception as e:
                print(f"  [assembly_builder] could not apply world transform "
                      f"for '{obj_id}': {e}")

        return assembly

    # ------------------------------------------------------------------
    # Protagonist construction
    # ------------------------------------------------------------------
    def _build_protagonist(
        self,
        spec: Dict[str, Any],
        element_map: Dict[str, List[Element3D]],
    ) -> Protagonist:
        template = spec.get("template", "person")
        obj_id = spec.get("obj_id", "person_01")

        protagonist = Protagonist(object_id=obj_id, template_name=template)

        # Elements: prefer the element_map, fall back to storage, then
        # fall back to a placeholder box.
        protagonist.elements = self._resolve_elements(template, element_map)

        # Material and mass: storage defaults.
        protagonist.material = dict(self.storage.get_default_material(template) or {})
        protagonist.mass_grams = self.storage.get_default_mass(template)

        # Overrides from the spec.
        overrides = spec.get("overrides")
        if isinstance(overrides, dict):
            # Protagonist-specific fields go to the top-level attribute
            # set, not the material dict.
            for key, value in overrides.items():
                if key == "anatomical_features":
                    protagonist.anatomical_features = list(value or [])
                else:
                    setattr(protagonist, key, value) if hasattr(protagonist, key) \
                        else protagonist.metadata.__setitem__(key, value)
            # Material-level overrides (e.g. color)
            material_overrides = {
                k: v for k, v in overrides.items()
                if k in ("color", "material", "temperature", "posture")
            }
            if material_overrides:
                protagonist.material.update(material_overrides)

        # Anatomical features may also be given as a top-level field.
        anatomical = spec.get("anatomical_features")
        if isinstance(anatomical, list):
            protagonist.anatomical_features = list(anatomical)

        # Any extra fields that are not object-model fields become metadata.
        model_fields = {
            "obj_id", "template", "overrides", "anatomical_features",
        }
        for key, value in spec.items():
            if key in model_fields:
                continue
            protagonist.metadata[key] = value

        return protagonist

    # ------------------------------------------------------------------
    # Regular object construction
    # ------------------------------------------------------------------
    def _build_object(
        self,
        spec: Dict[str, Any],
        element_map: Dict[str, List[Element3D]],
    ) -> Object3D:
        template = spec.get("template") or spec.get("template_name")
        obj_id = spec["obj_id"]
        if not template:
            raise ValueError(f"Object spec '{obj_id}' has no template")

        # Resolve elements.
        elements = self._resolve_elements(template, element_map)

        # Merge storage defaults for material and mass with instance
        # overrides from the spec.
        material = dict(self.storage.get_default_material(template) or {})
        instance_material = spec.get("material")
        if isinstance(instance_material, dict):
            material = {**material, **instance_material}

        mass = self.storage.get_default_mass(template)
        if mass is None:
            raw_mass = spec.get("mass_grams") or spec.get("mass")
            if isinstance(raw_mass, dict):
                mass = raw_mass.get("value")
            else:
                mass = raw_mass

        # Children: will be rebuilt by Assembly3D.__post_init__, but we
        # accept whatever the spec declares to keep the spec self-
        # consistent.
        children = [
            c for c in (spec.get("children") or [])
            if isinstance(c, str)
        ]

        # Metadata: everything the model does not have first-class
        # attributes for.
        metadata: Dict[str, Any] = {}
        for key in (
            "number", "shape", "position", "position_relative_body",
            "motion", "aliases", "composite_of", "functions",
            "dimensions",
        ):
            if key in spec and spec[key] is not None:
                metadata[key] = spec[key]
        # Any explicit metadata dict on the spec is also merged.
        explicit_metadata = spec.get("metadata")
        if isinstance(explicit_metadata, dict):
            metadata = {**metadata, **explicit_metadata}

        obj = Object3D(
            object_id=obj_id,
            template_name=template,
            elements=elements,
            material=material,
            object_transform=Transform.identity(),
            mass_grams=mass if (mass is None or mass > 0) else None,
            parent_object_id=spec.get("parent_object_id"),
            children=children,
            metadata=metadata,
        )

        # Instance overrides from the spec are merged into the material
        # dict (they represent variable attributes of this instance).
        overrides = spec.get("overrides")
        if isinstance(overrides, dict):
            obj.material = {**obj.material, **overrides}

        return obj

    # ------------------------------------------------------------------
    # Element resolution
    # ------------------------------------------------------------------
    def _resolve_elements(
        self,
        template: str,
        element_map: Dict[str, List[Element3D]],
    ) -> List[Element3D]:
        """
        Return a deep copy of the elements for a template. The copy is
        important: mutating an instance's elements (e.g. via Stage 5
        labeling or Stage 4 add_element) must not affect the elements
        used by other instances of the same template.

        Falls back to storage if the element_map does not have the
        template, and to a placeholder box if neither does.
        """
        raw = element_map.get(template)
        if raw is None:
            raw = self.storage.get_elements(template)
        if raw is None or not raw:
            print(f"  [assembly_builder] no elements for template "
                  f"'{template}'; using placeholder box")
            return [make_element(
                "box",
                {"size_x_cm": 5.0, "size_y_cm": 5.0, "size_z_cm": 5.0},
            )]
        return [copy.deepcopy(el) for el in raw]

    # ------------------------------------------------------------------
    # Hierarchy helpers
    # ------------------------------------------------------------------
    def _topological_order(
        self,
        specs: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Return the object ids in top-down order: roots first, then their
        children, then grandchildren, and so on. Used so that when a
        world transform is applied to a child via set_world_transform,
        the parent's world matrix has already been set.

        Cycles (which should not occur) are broken by visiting each id
        once, in input order.
        """
        by_id = {spec["obj_id"]: spec for spec in specs}
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

        for spec in specs:
            visit(spec["obj_id"])

        return order


# =============================================================================
# Convenience function
# =============================================================================
def build_assembly(
    memory_id: str,
    protagonist_spec: Dict[str, Any],
    object_specs: List[Dict[str, Any]],
    element_map: Dict[str, List[Element3D]],
    scene_layout: Dict[str, Dict[str, Any]],
    environment: Optional[Dict[str, Any]],
    storage: Elements3DStorage,
) -> Assembly3D:
    """
    Convenience wrapper that instantiates a builder and calls
    build_assembly in one step.
    """
    builder = AssemblyBuilder(storage)
    return builder.build_assembly(
        memory_id=memory_id,
        protagonist_spec=protagonist_spec,
        object_specs=object_specs,
        element_map=element_map,
        scene_layout=scene_layout,
        environment=environment,
    )


# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("3d_assembly_builder.py – smoke test")
    print("=" * 70)

    # Build a tiny element storage with two templates.
    storage = Elements3DStorage()

    person_elements = [
        make_element(
            "capsule",
            {"radius_cm": 15.0, "height_cm": 120.0},
            transform=make_transform(translate=[0, 0, 80]),
        ),
        make_element(
            "sphere",
            {"radius_cm": 10.0},
            transform=make_transform(translate=[0, 0, 165]),
        ),
    ]
    storage.add_template(
        "person",
        defaults={},
        functions=["agent"],
        aliases=[],
        categories=["person", "physical_object", "object"],
        materials=["biological"],
    )
    # Note: Elements3DStorage stores ElementEntry objects; the get_elements
    # accessor returns deep copies. For this smoke test we bypass the
    # storage and pass the elements directly through the element_map.

    hand_elements = [
        make_element("box", {"size_x_cm": 8.0, "size_y_cm": 2.5, "size_z_cm": 18.0}),
    ]
    apple_elements = [
        make_element("sphere", {"radius_cm": 4.0}),
        make_element(
            "cylinder",
            {"radius_cm": 0.15, "height_cm": 1.6},
            transform=make_transform(translate=[0, 0, 4.2]),
        ),
    ]
    knife_elements = [
        make_element(
            "box",
            {"size_x_cm": 10.0, "size_y_cm": 2.0, "size_z_cm": 0.2},
            transform=make_transform(translate=[-5.0, 0, 0]),
        ),
        make_element(
            "box",
            {"size_x_cm": 9.0, "size_y_cm": 2.5, "size_z_cm": 1.5},
            transform=make_transform(translate=[5.5, 0, 0]),
        ),
    ]

    element_map = {
        "person": person_elements,
        "hand": hand_elements,
        "apple": apple_elements,
        "knife": knife_elements,
    }

    # Protagonist spec.
    protagonist_spec = {
        "template": "person",
        "obj_id": "person_01",
        "overrides": {
            "posture": "standing",
            "sensory_perception_sight": [],
            "sensory_perception_touch": [],
        },
        "anatomical_features": ["left_hand_01", "apple_01"],
    }

    # Object specs. Hierarchy: person_01 -> left_hand_01 -> apple_01.
    object_specs = [
        {
            "obj_id": "left_hand_01",
            "template": "hand",
            "parent_object_id": "person_01",
            "number": "single",
            "overrides": {"grip_type": "open", "finger_posture": "relaxed"},
        },
        {
            "obj_id": "apple_01",
            "template": "apple",
            "parent_object_id": "left_hand_01",
            "number": "single",
            "shape": "spherical",
            "overrides": {
                "color": "red_brushed",
                "skin_present": True,
                "temperature": "cold",
            },
        },
        {
            "obj_id": "knife_01",
            "template": "knife",
            "parent_object_id": None,
            "number": "single",
            "shape": "elongated",
            "overrides": {"color": "silver", "blade_edge": "sharp"},
        },
    ]

    # Scene layout: WORLD transforms.
    scene_layout = {
        "person_01":    {"translate": [-30.0, 0.0, 0.0],   "rotate": [0, 0, 0, 1], "scale": [1, 1, 1]},
        "left_hand_01": {"translate": [-25.0, 0.0, 100.0], "rotate": [0, 0, 0, 1], "scale": [1, 1, 1]},
        "apple_01":     {"translate": [-25.0, 0.0, 104.0], "rotate": [0, 0, 0, 1], "scale": [1, 1, 1]},
        "knife_01":     {"translate": [15.0, 5.0, 80.5],   "rotate": [0, 0, 0, 1], "scale": [1, 1, 1]},
    }

    environment = {
        "ground": {"present": True, "material": "wood"},
        "counter": {"present": True, "top_z_cm": 80.0},
    }

    # Build.
    builder = AssemblyBuilder(storage)
    assembly = builder.build_assembly(
        memory_id="mem_smoke_001",
        protagonist_spec=protagonist_spec,
        object_specs=object_specs,
        element_map=element_map,
        scene_layout=scene_layout,
        environment=environment,
    )

    print(f"\n{assembly.summary()}")
    ok, msg = assembly.validate()
    print(f"Validation: {'OK' if ok else 'FAILED — ' + msg}")

    # Check hierarchy is bidirectionally consistent.
    person = assembly.get_object("person_01")
    left_hand = assembly.get_object("left_hand_01")
    apple = assembly.get_object("apple_01")
    knife = assembly.get_object("knife_01")
    print("\nHierarchy:")
    for obj in (person, left_hand, apple, knife):
        print(f"  {obj.object_id}.parent = {obj.parent_object_id}, "
              f"children = {obj.children}")
    assert left_hand.object_id in person.children
    assert apple.object_id in left_hand.children
    assert knife.parent_object_id is None and knife.children == []

    # Check world positions match the layout exactly.
    world = assembly.get_all_world_matrices()
    print("\nWorld positions:")
    for obj_id, expected in scene_layout.items():
        M = world[obj_id]
        pos = (M[0, 3], M[1, 3], M[2, 3])
        exp = expected["translate"]
        print(f"  {obj_id:15s} pos = ({pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f})  "
              f"expected ({exp[0]:7.2f}, {exp[1]:7.2f}, {exp[2]:7.2f})")
        assert abs(pos[0] - exp[0]) < 1e-4
        assert abs(pos[1] - exp[1]) < 1e-4
        assert abs(pos[2] - exp[2]) < 1e-4

    # Check the protagonist carries its anatomical features.
    print(f"\nProtagonist anatomical features: {person.anatomical_features}")
    assert "left_hand_01" in person.anatomical_features
    assert "apple_01" in person.anatomical_features

    # Check elements were deep-copied (mutating the assembly's copy must
    # not affect the element_map).
    original_radius = element_map["apple"][0].parameters["radius_cm"]
    apple.elements[0].parameters["radius_cm"] = 99.0
    assert element_map["apple"][0].parameters["radius_cm"] == original_radius
    print("\nElement deep-copy verified (mutating an instance does not "
          "affect the element_map).")

    # Round-trip through JSON.
    d = assembly.to_dict()
    reloaded = Assembly3D.from_dict(d)
    ok2, msg2 = reloaded.validate()
    print(f"\nReloaded from dict: {'OK' if ok2 else 'FAILED — ' + msg2}")
    if d == reloaded.to_dict():
        print("Round-trip serialization is identical.")
    else:
        print("WARNING: round-trip differs.")

    print("\nSmoke test complete.")