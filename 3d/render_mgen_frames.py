#!/usr/bin/env python3
r"""
render_mgen_frames.py — render per-frame assemblies saved by mgen.

Reads the per-frame assemblies from:

    assemblies/frames/{memory_id}/frame_NNNN.3dassembly.json

and renders each one to PNG under:

    assemblies/renders/{memory_id}_frames/frame_NNNN.png

These frames are the exact assemblies the pipeline built during generation,
so no operation re-application or re-parsing is needed. The script loads each
assembly file and rasterises it directly.

Camera behaviour:
    By default, the camera is derived from the first frame's geometry and
    reused for every frame, so objects visibly move within a stable view.
    Override with --eye / --target / --up.

Usage:
    python render_mgen_frames.py --list
    python render_mgen_frames.py --memory-id mem3d_0123456789_00
    python render_mgen_frames.py --memory-id mem3d_0123456789_00 --start-frame 5
    python render_mgen_frames.py --memory-id mem3d_0123456789_00 --end-frame 10
    python render_mgen_frames.py --memory-id mem3d_0123456789_00 --backend matplotlib
    python render_mgen_frames.py --memory-id mem3d_0123456789_00 \
        --eye 240 -20 360 --target 0 -10 95
"""

import argparse
import importlib
import re
import sys
from pathlib import Path

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_assembly = importlib.import_module("3d_assembly")
_render = importlib.import_module("3d_render_assembly")

Assembly3D = _assembly.Assembly3D

ASSEMBLIES_DIR = _HERE / "assemblies"
FRAMES_DIR = ASSEMBLIES_DIR / "frames"
RENDERS_DIR = ASSEMBLIES_DIR / "renders"

FRAME_NAME_RE = re.compile(r"^frame_(\d+)\.3dassembly\.json$")


def list_available_memories():
    if not FRAMES_DIR.exists():
        return []
    return sorted(p.name for p in FRAMES_DIR.iterdir() if p.is_dir())


def discover_frames(memory_id):
    mem_dir = FRAMES_DIR / memory_id
    if not mem_dir.exists():
        return []
    frames = []
    for p in mem_dir.iterdir():
        m = FRAME_NAME_RE.match(p.name)
        if m:
            frames.append((int(m.group(1)), p))
    frames.sort(key=lambda x: x[0])
    return frames


def auto_camera_from_assembly(assembly):
    triangles = _render._collect_triangles(assembly)
    if not triangles:
        return None, None
    eye, target = _render._auto_camera_from_triangles(triangles)
    return list(eye), list(target)


def main():
    parser = argparse.ArgumentParser(description="Render mgen per-frame assemblies.")
    parser.add_argument("--memory-id", type=str, default=None,
                        help="Memory ID whose frames should be rendered.")
    parser.add_argument("--list", action="store_true",
                        help="List available memory IDs and exit.")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="Render only frames with index >= this value.")
    parser.add_argument("--end-frame", type=int, default=None,
                        help="Render only frames with index <= this value.")
    parser.add_argument("--backend", type=str, default="software",
                        choices=["software", "matplotlib", "trimesh"],
                        help="Rendering backend. Software is the default.")
    parser.add_argument("--image-width", type=int, default=1200,
                        help="Image width in pixels (software backend only).")
    parser.add_argument("--image-height", type=int, default=900,
                        help="Image height in pixels (software backend only).")
    parser.add_argument("--eye", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Explicit camera position. Overrides auto camera.")
    parser.add_argument("--target", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Explicit camera target. Overrides auto camera.")
    parser.add_argument("--up", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Explicit camera up vector. Defaults to (0, 0, 1).")
    args = parser.parse_args()

    # ---- --list mode ----
    if args.list:
        memories = list_available_memories()
        if not memories:
            print(f"No memory frame directories found under {FRAMES_DIR}")
            return 0
        print(f"Memory IDs with frames available ({len(memories)}):")
        for m in memories:
            frames = discover_frames(m)
            print(f"  {m}  ({len(frames)} frames)")
        return 0

    # ---- Validate arguments ----
    if not args.memory_id:
        print("--memory-id is required unless --list is used.")
        print("Run with --list to see available memory IDs.")
        return 1

    if (args.eye is None) != (args.target is None):
        print("Both --eye and --target must be provided together.")
        return 1

    frames = discover_frames(args.memory_id)
    if not frames:
        print(f"No frames found for memory '{args.memory_id}'.")
        print(f"Looked in: {FRAMES_DIR / args.memory_id}")
        available = list_available_memories()
        if available:
            print(f"Available memory IDs: {', '.join(available)}")
        return 1

    # ---- Apply frame range filters ----
    if args.start_frame is not None:
        frames = [f for f in frames if f[0] >= args.start_frame]
    if args.end_frame is not None:
        frames = [f for f in frames if f[0] <= args.end_frame]
    if not frames:
        print("No frames remain after applying --start-frame / --end-frame filters.")
        return 1

    output_dir = RENDERS_DIR / f"{args.memory_id}_frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Memory ID    : {args.memory_id}")
    print(f"Frames found : {len(frames)}")
    print(f"Output dir   : {output_dir}")
    print(f"Backend      : {args.backend}")

    # ---- Determine camera ----
    camera_eye = None
    camera_target = None
    if args.eye is not None and args.target is not None:
        camera_eye = list(args.eye)
        camera_target = list(args.target)
        print(f"Camera       : explicit "
              f"eye=[{camera_eye[0]:.1f}, {camera_eye[1]:.1f}, {camera_eye[2]:.1f}] "
              f"target=[{camera_target[0]:.1f}, {camera_target[1]:.1f}, {camera_target[2]:.1f}]")
    else:
        first_frame_path = frames[0][1]
        try:
            first_assembly = Assembly3D.load(first_frame_path)
            eye, target = auto_camera_from_assembly(first_assembly)
            if eye is None or target is None:
                print(f"Could not auto-derive camera from "
                      f"{first_frame_path.name}; renderer default will be used.")
            else:
                camera_eye = eye
                camera_target = target
                print(f"Camera       : auto from frame {frames[0][0]} "
                      f"eye=[{camera_eye[0]:.1f}, {camera_eye[1]:.1f}, {camera_eye[2]:.1f}] "
                      f"target=[{camera_target[0]:.1f}, {camera_target[1]:.1f}, {camera_target[2]:.1f}]")
        except Exception as e:
            print(f"Failed to load first frame {first_frame_path.name}: {e}")

    camera_up = list(args.up) if args.up is not None else [0.0, 0.0, 1.0]

    # ---- Build render kwargs ----
    render_kwargs = {
        "backend": args.backend,
        "mode": "solid",
        "show": False,
        "axis": True,
    }
    if camera_eye is not None and camera_target is not None:
        render_kwargs["camera_eye"] = camera_eye
        render_kwargs["camera_target"] = camera_target
        render_kwargs["camera_up"] = camera_up
    # Image size only applies to the software backend.
    if args.backend == "software":
        render_kwargs["image_width"] = args.image_width
        render_kwargs["image_height"] = args.image_height

    # ---- Render each frame ----
    rendered = 0
    failed = 0
    failures = []

    for idx, path in frames:
        try:
            assembly = Assembly3D.load(path)
        except Exception as e:
            failed += 1
            failures.append((idx, f"load failed: {e}"))
            print(f"  frame_{idx:04d}: load failed: {e}")
            continue

        output_path = output_dir / f"frame_{idx:04d}.png"
        try:
            _render.render_assembly(assembly, output_path=output_path, **render_kwargs)
            rendered += 1
            print(f"  frame_{idx:04d} -> {output_path.name}")
        except Exception as e:
            failed += 1
            failures.append((idx, f"render failed: {e}"))
            print(f"  frame_{idx:04d}: render failed: {e}")

    # ---- Write log ----
    log_path = output_dir / "render_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Memory ID   : {args.memory_id}\n")
        f.write(f"Frames      : {len(frames)}\n")
        f.write(f"Rendered    : {rendered}\n")
        f.write(f"Failed      : {failed}\n")
        f.write(f"Backend     : {args.backend}\n")
        f.write(f"Camera eye  : {camera_eye}\n")
        f.write(f"Camera tgt  : {camera_target}\n")
        f.write(f"Camera up   : {camera_up}\n")
        if failures:
            f.write("\nFailures:\n")
            for idx, reason in failures:
                f.write(f"  frame_{idx:04d}: {reason}\n")

    print()
    print(f"Rendered {rendered} frame(s), {failed} failed.")
    print(f"PNGs      : {output_dir}")
    print(f"Log       : {log_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())