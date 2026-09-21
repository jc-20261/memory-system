#!/usr/bin/env python3
r"""
3d_elements_storage.py – Storage of element decompositions for object templates.

Matches the pattern of object_storage, action_storage, verb_storage:
each entry is keyed on a template_name and holds the canonical geometry
decomposition for that template.

An entry holds:

    elements                   – list of Element3D decomposing the geometry
    default_material           – free-form dict (color, roughness, metalness,
                                 transparency, …)
    default_mass_grams         – reference mass in grams
    default_dimensions_cm      – {length_cm, width_cm, height_thickness_cm}
                                 reference bounding-box dimensions
    pointcloud_id              – identifier for the cached point cloud
    composite_children_templates – list of template names that are the
                                 composite sub-objects of this template
                                 (apple → apple_skin, apple_flesh, apple_core)
    notes                      – optional free-form notes

Storage JSON layout:

    {
      "version": "1.0",
      "templates": {
        "<template_name>": { ...entry... },
        ...
      }
    }

The storage is a regular class, not a singleton, so multiple instances can
be created for testing. A module-level default instance is provided for
convenience in scripts.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import copy

import importlib
import sys
from pathlib import Path

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_assembly = importlib.import_module("3d_assembly")

Element3D = _assembly.Element3D
ELEMENT_SCHEMA = _assembly.ELEMENT_SCHEMA
DATA_DIR = _assembly.DATA_DIR
make_element = _assembly.make_element


# =============================================================================
# Paths
# =============================================================================
MODULE_DIR = Path(__file__).parent
DEFAULT_STORAGE_PATH = DATA_DIR / "3d_elements_storage.json"


# =============================================================================
# Entry dataclass
# =============================================================================
@dataclass
class ElementEntry:
    """One template's element decomposition and default properties."""
    template_name: str
    elements: List[Element3D] = field(default_factory=list)
    default_material: Dict[str, Any] = field(default_factory=dict)
    default_mass_grams: Optional[float] = None
    default_dimensions_cm: Dict[str, float] = field(default_factory=dict)
    pointcloud_id: Optional[str] = None
    composite_children_templates: List[str] = field(default_factory=list)
    notes: str = ""

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> Tuple[bool, str]:
        if not self.template_name:
            return False, "template_name is empty"

        if not self.elements:
            return False, f"template '{self.template_name}' has no elements"

        for i, el in enumerate(self.elements):
            ok, msg = el.validate()
            if not ok:
                return False, (
                    f"template '{self.template_name}' element {i}: {msg}"
                )

        if self.default_mass_grams is not None:
            if not isinstance(self.default_mass_grams, (int, float)):
                return False, (
                    f"template '{self.template_name}' default_mass_grams "
                    f"must be a number"
                )
            if self.default_mass_grams <= 0:
                return False, (
                    f"template '{self.template_name}' default_mass_grams "
                    f"must be positive, got {self.default_mass_grams}"
                )

        for key in ("length_cm", "width_cm", "height_thickness_cm"):
            if key in self.default_dimensions_cm:
                val = self.default_dimensions_cm[key]
                if not isinstance(val, (int, float)):
                    return False, (
                        f"template '{self.template_name}' dimensions.{key} "
                        f"must be a number"
                    )
                if val <= 0:
                    return False, (
                        f"template '{self.template_name}' dimensions.{key} "
                        f"must be positive, got {val}"
                    )

        if not isinstance(self.composite_children_templates, list):
            return False, (
                f"template '{self.template_name}' composite_children_templates "
                f"must be a list"
            )

        return True, ""

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "elements": [e.to_dict() for e in self.elements],
        }
        if self.default_material:
            d["default_material"] = dict(self.default_material)
        if self.default_mass_grams is not None:
            d["default_mass_grams"] = self.default_mass_grams
        if self.default_dimensions_cm:
            d["default_dimensions_cm"] = dict(self.default_dimensions_cm)
        if self.pointcloud_id:
            d["pointcloud_id"] = self.pointcloud_id
        if self.composite_children_templates:
            d["composite_children_templates"] = list(
                self.composite_children_templates
            )
        if self.notes:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, template_name: str, d: Dict[str, Any]) -> "ElementEntry":
        return cls(
            template_name=template_name,
            elements=[Element3D.from_dict(e) for e in d.get("elements", [])],
            default_material=dict(d.get("default_material", {})),
            default_mass_grams=d.get("default_mass_grams"),
            default_dimensions_cm=dict(d.get("default_dimensions_cm", {})),
            pointcloud_id=d.get("pointcloud_id"),
            composite_children_templates=list(
                d.get("composite_children_templates", [])
            ),
            notes=d.get("notes", ""),
        )


# =============================================================================
# Storage class
# =============================================================================
class Elements3DStorage:
    """
    Storage of element decompositions, keyed on template_name.

    Typical usage:

        storage = Elements3DStorage()
        storage.load()

        entry = storage.get_template("apple")
        if entry is not None:
            for element in entry.elements:
                ...

        new_entry = ElementEntry(template_name="banana", elements=[...])
        storage.add_template(new_entry)
        storage.save()
    """

    STORAGE_VERSION = "1.0"

    def __init__(self):
        self._templates: Dict[str, ElementEntry] = {}
        self._path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Template management
    # ------------------------------------------------------------------
    def add_template(
        self,
        entry: ElementEntry,
        overwrite: bool = True,
    ) -> Tuple[bool, str]:
        """
        Add an element entry. Returns (success, message).

        If overwrite is False and the template already exists, the add fails.
        """
        if not isinstance(entry, ElementEntry):
            return False, "entry must be an ElementEntry instance"
        if not entry.template_name:
            return False, "entry.template_name is empty"

        if entry.template_name in self._templates and not overwrite:
            return False, (
                f"template '{entry.template_name}' already exists "
                f"and overwrite=False"
            )

        ok, msg = entry.validate()
        if not ok:
            return False, msg

        self._templates[entry.template_name] = entry
        return True, ""

    def remove_template(self, template_name: str) -> bool:
        """Remove a template. Returns True if it existed."""
        if template_name in self._templates:
            del self._templates[template_name]
            return True
        return False

    def has_template(self, template_name: str) -> bool:
        return template_name in self._templates

    def get_template(self, template_name: str) -> Optional[ElementEntry]:
        return self._templates.get(template_name)

    def list_templates(self) -> List[str]:
        return sorted(self._templates.keys())

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def get_elements(self, template_name: str) -> List[Element3D]:
        """Return a copy of the element list, or an empty list."""
        entry = self._templates.get(template_name)
        if entry is None:
            return []
        return [copy.deepcopy(el) for el in entry.elements]

    def get_default_material(self, template_name: str) -> Dict[str, Any]:
        entry = self._templates.get(template_name)
        if entry is None:
            return {}
        return dict(entry.default_material)

    def get_default_mass(self, template_name: str) -> Optional[float]:
        entry = self._templates.get(template_name)
        if entry is None:
            return None
        return entry.default_mass_grams

    def get_default_dimensions(self, template_name: str) -> Dict[str, float]:
        entry = self._templates.get(template_name)
        if entry is None:
            return {}
        return dict(entry.default_dimensions_cm)

    def get_composite_children(self, template_name: str) -> List[str]:
        entry = self._templates.get(template_name)
        if entry is None:
            return []
        return list(entry.composite_children_templates)

    def get_pointcloud_id(self, template_name: str) -> Optional[str]:
        entry = self._templates.get(template_name)
        if entry is None:
            return None
        return entry.pointcloud_id

    def has_geometry(self, template_name: str) -> bool:
        """Return True if a template has at least one element."""
        entry = self._templates.get(template_name)
        return entry is not None and len(entry.elements) > 0

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------
    def add_from_dict(self, template_name: str, d: Dict[str, Any]) -> Tuple[bool, str]:
        """Construct an entry from a dict and add it."""
        try:
            entry = ElementEntry.from_dict(template_name, d)
        except Exception as e:
            return False, f"failed to parse entry: {e}"
        return self.add_template(entry)

    def merge(self, other: "Elements3DStorage", overwrite: bool = False):
        """
        Merge another storage into this one. By default, existing
        templates are kept and only new ones are added.
        """
        for name, entry in other._templates.items():
            if name in self._templates and not overwrite:
                continue
            self._templates[name] = entry

    def clear(self):
        self._templates.clear()

    def __len__(self) -> int:
        return len(self._templates)

    def __contains__(self, template_name: str) -> bool:
        return template_name in self._templates

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_entry(self, template_name: str) -> Tuple[bool, str]:
        entry = self._templates.get(template_name)
        if entry is None:
            return False, f"template '{template_name}' not found"
        return entry.validate()

    def validate_all(self) -> List[Tuple[str, str]]:
        """
        Validate every template. Returns a list of (template_name, message)
        for entries that failed. An empty list means everything is valid.
        """
        failures = []
        for name, entry in self._templates.items():
            ok, msg = entry.validate()
            if not ok:
                failures.append((name, msg))
        return failures

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.STORAGE_VERSION,
            "templates": {
                name: entry.to_dict()
                for name, entry in self._templates.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Elements3DStorage":
        storage = cls()
        templates = d.get("templates", {})
        if not isinstance(templates, dict):
            raise ValueError("storage 'templates' must be a dict")
        for name, entry_dict in templates.items():
            if not isinstance(entry_dict, dict):
                continue
            try:
                entry = ElementEntry.from_dict(name, entry_dict)
                storage._templates[name] = entry
            except Exception:
                # Skip malformed entries rather than failing the whole load
                continue
        return storage

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save to JSON. If path is None, saves to DEFAULT_STORAGE_PATH.
        """
        if path is None:
            path = DEFAULT_STORAGE_PATH
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        self._path = path
        return path

    def load(self, path: Optional[Path] = None) -> bool:
        """
        Load from JSON. Returns True if a file was loaded, False if
        no file exists at the given path (empty storage remains).
        """
        if path is None:
            path = self._path if self._path is not None else DEFAULT_STORAGE_PATH
        path = Path(path)
        if not path.exists():
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded = Elements3DStorage.from_dict(data)
        self._templates = loaded._templates
        self._path = path
        return True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def summary(self) -> str:
        total_elements = sum(len(e.elements) for e in self._templates.values())
        with_children = sum(
            1 for e in self._templates.values()
            if e.composite_children_templates
        )
        return (
            f"Elements3DStorage(templates={len(self._templates)}, "
            f"total_elements={total_elements}, "
            f"with_children={with_children})"
        )


# =============================================================================
# Module-level default instance
# =============================================================================
_default_storage: Optional[Elements3DStorage] = None


def get_default_storage() -> Elements3DStorage:
    """
    Return the module-level default storage instance.
    Loads from DEFAULT_STORAGE_PATH on first access if the file exists.
    """
    global _default_storage
    if _default_storage is None:
        _default_storage = Elements3DStorage()
        _default_storage.load()
    return _default_storage


def reset_default_storage():
    """Reset the module-level storage. Useful for testing."""
    global _default_storage
    _default_storage = None


# =============================================================================
# Main – smoke test
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("3d_elements_storage.py – smoke test")
    print("=" * 70)

    storage = Elements3DStorage()

    # Build an apple entry with composite children
    apple = ElementEntry(
        template_name="apple",
        elements=[
            make_element("sphere", {"radius_cm": 4.0}),
            make_element(
                "cylinder",
                {"radius_cm": 0.2, "height_cm": 2.0},
                transform=None,
            ),
            make_element(
                "torus",
                {"major_radius_cm": 1.0, "minor_radius_cm": 0.3},
                transform=None,
            ),
        ],
        default_material={"color": "red_brushed", "roughness": 0.4},
        default_mass_grams=200.0,
        default_dimensions_cm={
            "length_cm": 8.0,
            "width_cm": 8.0,
            "height_thickness_cm": 8.0,
        },
        pointcloud_id="apple_pc_v1",
        composite_children_templates=["apple_skin", "apple_flesh", "apple_core"],
        notes="Standard apple. Skin, flesh, and core are separate templates.",
    )

    knife = ElementEntry(
        template_name="knife",
        elements=[
            make_element(
                "box",
                {"size_x_cm": 10.0, "size_y_cm": 2.0, "size_z_cm": 0.2},
            ),
            make_element(
                "box",
                {"size_x_cm": 10.0, "size_y_cm": 3.0, "size_z_cm": 1.0},
            ),
        ],
        default_material={"color": "silver", "metalness": 1.0, "roughness": 0.2},
        default_mass_grams=150.0,
        default_dimensions_cm={
            "length_cm": 20.0,
            "width_cm": 3.0,
            "height_thickness_cm": 1.2,
        },
        pointcloud_id="knife_pc_v1",
        notes="Chef's knife.",
    )

    # Add both entries
    ok, msg = storage.add_template(apple)
    print(f"Add apple:   {'OK' if ok else 'FAILED — ' + msg}")

    ok, msg = storage.add_template(knife)
    print(f"Add knife:   {'OK' if ok else 'FAILED — ' + msg}")

    print(f"\n{storage.summary()}")

    # Accessors
    print("\nAccessors:")
    print(f"  get_default_mass('apple'):    {storage.get_default_mass('apple')}")
    print(f"  get_default_material('knife'):{storage.get_default_material('knife')}")
    print(f"  get_composite_children('apple'): {storage.get_composite_children('apple')}")
    print(f"  has_geometry('apple'):        {storage.has_geometry('apple')}")
    print(f"  has_geometry('banana'):       {storage.has_geometry('banana')}")
    print(f"  list_templates():             {storage.list_templates()}")

    # Validation
    failures = storage.validate_all()
    print(f"\nvalidate_all() failures: {len(failures)}")
    for name, msg in failures:
        print(f"  {name}: {msg}")

    # Save and reload
    save_path = storage.save()
    print(f"\nSaved to: {save_path}")

    storage2 = Elements3DStorage()
    loaded = storage2.load(save_path)
    print(f"Loaded from disk: {loaded}")
    print(f"  {storage2.summary()}")

    # Confirm identical serialization after round-trip
    if json.dumps(storage.to_dict(), sort_keys=True) == json.dumps(
        storage2.to_dict(), sort_keys=True
    ):
        print("Round-trip serialization is identical.")
    else:
        print("WARNING: round-trip serialization differs.")

    # Negative test: missing required parameter
    bad = ElementEntry(
        template_name="bad_sphere",
        elements=[
            Element3D(element_type="sphere", parameters={}),
        ],
    )
    ok, msg = storage.add_template(bad)
    print(f"\nNegative test (missing radius): "
          f"{'correctly rejected' if not ok else 'WRONGLY ACCEPTED'}")
    print(f"  message: {msg}")

    # Negative test: unknown element_type
    bad2 = ElementEntry(
        template_name="bad_unknown",
        elements=[
            Element3D(element_type="dodecahedron", parameters={"radius_cm": 1.0}),
        ],
    )
    ok, msg = storage.add_template(bad2)
    print(f"Negative test (unknown type):  "
          f"{'correctly rejected' if not ok else 'WRONGLY ACCEPTED'}")
    print(f"  message: {msg}")

    print("\nSmoke test complete.")