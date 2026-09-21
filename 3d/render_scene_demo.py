#!/usr/bin/env python3
r"""
render_scene_demo.py

Render the 3d_scene_demo.txt file correctly.

The demo stores object_transform values that were authored as WORLD
positions, but the runtime treats object_transform as LOCAL relative to
the parent. Rendering the demo directly places every child of the
counter at ~z=178 instead of on the counter top.

This script:
    1. Loads the demo as an Assembly3D.
    2. Snapshots each object's current transform as a world transform.
    3. Resets all local transforms to identity.
    4. Re-applies the world transforms in topological order via
       Assembly3D.set_world_transform, which correctly computes local
       transforms relative to each parent.
    5. Renders both the raw (as-is) and corrected assemblies, and
       prints a comparison of world positions.

Usage:
    python render_scene_demo.py
    python render_scene_demo.py --demo path/to/3d_scene_demo.txt
    python render_scene_demo.py --backend matplotlib
    python render_scene_demo.py --skip-raw
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_assembly = importlib.import_module("3d_assembly")
_render = importlib.import_module("3d_render_assembly")

Assembly3D = _assembly.Assembly3D
Object3D = _assembly.Object3D
Transform = _assembly.Transform


DEFAULT_DEMO = _HERE / "data" / "demos" / "3d_scene_demo.txt"
DEFAULT_RAW_OUTPUT = _HERE / "assemblies" / "renders" / "scene_demo_raw.png"
DEFAULT_CORRECTED_OUTPUT = _HERE / "assemblies" / "renders" / "scene_demo_corrected.png"


def topological_order(assembly):
    """Return object IDs in parent-before-child order."""
    by_id = {o.object_id: o for o in assembly.objects}
    order = []
    visited = set()

    def visit(obj_id):
        if obj_id in visited:
            return
        visited.add(obj_id)
        obj = by_id.get(obj_id)
        if obj is None:
            return
        if obj.parent_object_id is not None:
            visit(obj.parent_object_id)
        order.append(obj_id)

    for obj in assembly.objects:
        visit(obj.object_id)
    return order


def load_demo(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "3d_assembly" in data:
        data = data["3d_assembly"]
    return Assembly3D.from_dict(data)


def world_positions(assembly):
    ws = assembly.get_all_world_matrices()
    return {oid: tuple(float(x) for x in M[:3, 3]) for oid, M in ws.items()}


def correct_world_transforms(assembly):
    """
    Treat each object's current object_transform as a WORLD transform,
    reset all local transforms to identity, then reapply the world
    transforms in topological order so that local transforms are
    computed correctly relative to each parent.

    Returns a dict of object_id -> residual from set_world_transform.
    """
    # 1. Snapshot current transforms as world transforms.
    world_snapshot = {}
    for obj in assembly.objects:
        world_snapshot[obj.object_id] = Transform(
            translate=list(obj.object_transform.translate),
            rotate=list(obj.object_transform.rotate),
            scale=list(obj.object_transform.scale),
        )

    # 2. Reset all local transforms to identity, drop any matrix overrides.
    for obj in assembly.objects:
        obj.object_transform = Transform.identity()
        obj.world_matrix_override = None

    # 3. Reapply in topological order.
    order = topological_order(assembly)
    residuals = {}
    for obj_id in order:
        residual = assembly.set_world_transform(obj_id, world_snapshot[obj_id])
        residuals[obj_id] = residual

    return residuals


def print_positions(title, positions):
    print()
    print(title)
    print("-" * 72)
    for oid in sorted(positions):
        x, y, z = positions[oid]
        print(f"  {oid:28s} ({x:8.2f}, {y:8.2f}, {z:8.2f})")


def render_one(assembly, output_path, backend):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _render.render_assembly(
            assembly,
            output_path=output_path,
            backend=backend,
            mode="solid",
        )
        return True
    except Exception as e:
        print(f"  render failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, default=DEFAULT_DEMO)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_CORRECTED_OUTPUT)
    parser.add_argument("--backend", type=str, default="software",
                        choices=["software", "matplotlib", "trimesh"])
    parser.add_argument("--skip-raw", action="store_true",
                        help="Skip the raw render, do only the corrected one.")
    args = parser.parse_args()

    if not args.demo.exists():
        print(f"Demo file not found: {args.demo}")
        return 1

    print(f"Loading demo: {args.demo}")
    assembly = load_demo(args.demo)
    print(f"  {assembly.summary()}")
    ok, msg = assembly.validate()
    print(f"  validation: {'OK' if ok else 'FAILED -- ' + msg}")

    if not args.skip_raw:
        print()
        print(f"Rendering RAW (as-is) to {args.raw_output}")
        render_one(assembly, args.raw_output, args.backend)
        print_positions("RAW world positions (object_transform read as LOCAL):",
                        world_positions(assembly))

    print()
    print("Reinterpreting object_transform as WORLD and converting to LOCAL ...")
    residuals = correct_world_transforms(assembly)
    max_res = max(residuals.values()) if residuals else 0.0
    print(f"  max residual: {max_res:.6f} "
          f"({'exact' if max_res < 1e-4 else 'shear present'})")

    ok2, msg2 = assembly.validate()
    print(f"  post-correction validation: {'OK' if ok2 else 'FAILED -- ' + msg2}")

    print_positions("CORRECTED world positions:",
                    world_positions(assembly))

    print()
    print(f"Rendering CORRECTED to {args.output}")
    if not render_one(assembly, args.output, args.backend):
        return 1

    # Also save the corrected assembly JSON, so you can load it later
    # without going through the correction step again.
    corrected_json = args.output.with_suffix(".3dassembly.json")
    assembly.save(corrected_json)
    print(f"  corrected assembly JSON: {corrected_json}")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())