#!/usr/bin/env python3
r"""
render_frames.py — render the scene at each top-level action.

Parses the operations demo (text format, one header + JSON block per
top-level action), applies each block's operations to the corrected
starting scene, saves a runtime-format assembly per frame, and renders
one PNG per frame with a fixed camera.

Behaviour:
- Skip-and-continue per operation: any operation that fails to parse or
  apply is logged and skipped; the rest of the block continues.
- Each frame is also saved as a runtime-format Assembly3D JSON so it can
  be reloaded later.
- A summary log is written next to the PNGs.

Usage:
    python render_frames.py
    python render_frames.py --scene path/to/scene.txt --operations path/to/ops.txt
"""

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_assembly = importlib.import_module("3d_assembly")
_ops = importlib.import_module("3d_scene_operations")
_render = importlib.import_module("3d_render_assembly")

Assembly3D = _assembly.Assembly3D
Transform = _assembly.Transform

apply_operation = _ops.apply_operation
op_from_dict = _ops.op_from_dict


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SCENE = _HERE / "data" / "demos" / "3d_scene_demo.txt"
DEFAULT_OPERATIONS = _HERE / "data" / "demos" / "3d_operations_demo.txt"
DEFAULT_OUTPUT_DIR = _HERE / "assemblies" / "renders" / "frames"
DEFAULT_JSON_DIR = _HERE / "assemblies" / "frames"

# Fixed camera for all frames. Positioned at ~22 degrees elevation,
# azimuth ~ -54 degrees, distance ~ 520 cm from the scene centre.
CAMERA_EYE    = [240.0, -20.0, 360.0]
CAMERA_TARGET = [  0.0,  -10.0,  95.0]
CAMERA_UP     = [  0.0,    0.0,   1.0]
IMAGE_WIDTH = 1200
IMAGE_HEIGHT = 900


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
TITLE_RE = re.compile(r"^\s*H(\d+)\s*[—–-]")
SEPARATOR_RE = re.compile(r"^\s*[═=]{5,}\s*$")


def parse_operations_demo(path: Path):
    """
    Return a list of (number, title, body) tuples.

    The demo format is: a separator line, a title line matching
    'Hn — ...', a separator line, then the JSON body, repeated.

    Parsing is tolerant: separator lines inside the body are ignored,
    and the body is everything up to the next title line.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    blocks = []
    current_number = None
    current_title = None
    current_lines = []

    def flush():
        if current_title is None:
            return
        body = "\n".join(current_lines).strip()
        if body:
            blocks.append((current_number, current_title, body))

    for line in lines:
        m = TITLE_RE.match(line)
        if m:
            flush()
            current_number = int(m.group(1))
            current_title = line.strip()
            current_lines = []
            continue
        if current_title is not None:
            if SEPARATOR_RE.match(line):
                continue
            current_lines.append(line)
    flush()
    return blocks


# ---------------------------------------------------------------------------
# Scene loading and world-to-local correction
# ---------------------------------------------------------------------------
def load_scene(path: Path) -> Assembly3D:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "3d_assembly" in data:
        data = data["3d_assembly"]
    return Assembly3D.from_dict(data)


def topological_order(assembly: Assembly3D):
    by_id = {o.object_id: o for o in assembly.objects}
    order, visited = [], set()

    def visit(oid):
        if oid in visited:
            return
        visited.add(oid)
        obj = by_id.get(oid)
        if obj is None:
            return
        if obj.parent_object_id is not None:
            visit(obj.parent_object_id)
        order.append(oid)

    for obj in assembly.objects:
        visit(obj.object_id)
    return order


def correct_world_transforms(assembly: Assembly3D):
    """
    Reinterpret each object_transform as a world transform, reset all
    local transforms to identity, and reapply in topological order.
    """
    snapshot = {
        o.object_id: Transform(
            translate=list(o.object_transform.translate),
            rotate=list(o.object_transform.rotate),
            scale=list(o.object_transform.scale),
        )
        for o in assembly.objects
    }
    for obj in assembly.objects:
        obj.object_transform = Transform.identity()
        obj.world_matrix_override = None
    for oid in topological_order(assembly):
        assembly.set_world_transform(oid, snapshot[oid])


# ---------------------------------------------------------------------------
# Applying one block
# ---------------------------------------------------------------------------
def apply_block(assembly: Assembly3D, operations):
    """
    Apply a list of raw operation dicts. Return (new_assembly, failures).

    Failures is a list of dicts with keys: op, index, reason, detail.
    Skip-and-continue: a failed operation does not abort the block.
    """
    failures = []
    current = assembly

    for i, raw in enumerate(operations):
        if not isinstance(raw, dict):
            failures.append({
                "op": None, "index": i,
                "reason": "not_a_dict",
                "detail": str(raw)[:160],
            })
            continue

        op_name = raw.get("op", "?")

        try:
            op = op_from_dict(raw)
        except Exception as e:
            failures.append({
                "op": op_name, "index": i,
                "reason": "parse_error",
                "detail": str(e)[:200],
            })
            continue

        try:
            current, _events = apply_operation(current, op)
        except Exception as e:
            failures.append({
                "op": op_name, "index": i,
                "reason": "apply_error",
                "detail": str(e)[:200],
            })
            continue

    return current, failures


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render_frame(assembly: Assembly3D, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _render.render_assembly(
        assembly,
        output_path=output_path,
        backend="software",
        mode="solid",
        camera_eye=CAMERA_EYE,
        camera_target=CAMERA_TARGET,
        camera_up=CAMERA_UP,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--operations", type=Path, default=DEFAULT_OPERATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    args = parser.parse_args()

    if not args.scene.exists():
        print(f"Scene demo not found: {args.scene}")
        return 1
    if not args.operations.exists():
        print(f"Operations demo not found: {args.operations}")
        return 1

    # ---- Load scene and correct transforms
    print(f"Loading scene: {args.scene}")
    assembly = load_scene(args.scene)
    print(f"  {assembly.summary()}")

    print("Correcting world -> local transforms ...")
    correct_world_transforms(assembly)

    ok, msg = assembly.validate()
    if not ok:
        print(f"  validation FAILED: {msg}")
        return 1
    print("  validation OK")

    # ---- Parse operations demo
    print(f"Parsing operations demo: {args.operations}")
    blocks = parse_operations_demo(args.operations)
    print(f"  {len(blocks)} blocks found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.json_dir.mkdir(parents=True, exist_ok=True)

    log_lines = []
    rendered = 0
    failed_ops_total = 0

    current = assembly

    for number, title, body in blocks:
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            line = f"H{number}: JSON parse failed: {e}"
            print(f"  {line}")
            log_lines.append(line)
            continue

        operations = data.get("operations", [])
        if not isinstance(operations, list):
            line = f"H{number}: 'operations' is not a list"
            print(f"  {line}")
            log_lines.append(line)
            continue

        print(f"  Applying H{number} ({len(operations)} operations) ...")
        current, failures = apply_block(current, operations)

        for f in failures:
            line = (f"H{number}: op={f['op']} index={f['index']} "
                    f"reason={f['reason']} detail={f['detail']}")
            log_lines.append(line)
            print(f"    FAIL {line}")
        failed_ops_total += len(failures)

        # Save runtime-format assembly JSON.
        json_path = args.json_dir / f"frame_H{number:02d}.3dassembly.json"
        try:
            current.save(json_path)
        except Exception as e:
            line = f"H{number}: assembly save failed: {e}"
            log_lines.append(line)
            print(f"    {line}")

        # Render.
        png_path = args.output_dir / f"frame_H{number:02d}.png"
        try:
            render_frame(current, png_path)
            rendered += 1
            print(f"    -> {png_path.name}")
        except Exception as e:
            line = f"H{number}: render failed: {e}"
            log_lines.append(line)
            print(f"    {line}")

    # Summary
    log_lines.append("")
    log_lines.append(f"Blocks parsed: {len(blocks)}")
    log_lines.append(f"Frames rendered: {rendered}")
    log_lines.append(f"Operations failed (skipped): {failed_ops_total}")

    log_path = args.output_dir / "render_frames_log.txt"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print()
    print(f"Blocks parsed         : {len(blocks)}")
    print(f"Frames rendered       : {rendered}")
    print(f"Operations skipped    : {failed_ops_total}")
    print(f"Log                   : {log_path}")
    print(f"Frames (PNG)          : {args.output_dir}")
    print(f"Frames (JSON)         : {args.json_dir}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())