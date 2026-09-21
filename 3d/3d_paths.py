#!/usr/bin/env python3
r"""
3d_paths.py – Centralised path constants for the 3D module (memory system v0.2).

Every path resolves under the 3d folder:
    F:\New folder (4)\New folder\3d\

Nothing in the 3D module writes outside this folder. The only paths that
point elsewhere are the read-only v0.1 references at the bottom, used when
the v0.2 pipeline wants to read v0.1 sources for comparison. Nothing writes
to them.

Usage in other 3d_*.py modules:

    import importlib, sys
    from pathlib import Path
    _HERE = Path(__file__).parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    _paths = importlib.import_module("3d_paths")
    DATA_DIR = _paths.DATA_DIR
    ...

Or import specific names directly:

    _paths = importlib.import_module("3d_paths")
    STORAGES_DIR = _paths.STORAGES_DIR
"""

from pathlib import Path


# =============================================================================
# Root
# =============================================================================
MODULE_DIR = Path(__file__).parent


# =============================================================================
# Data (storages, caches, point clouds)
# =============================================================================
DATA_DIR        = MODULE_DIR / "data"
STORAGES_DIR    = DATA_DIR / "storages"
POINTCLOUD_DIR  = DATA_DIR / "pointclouds"
CACHES_DIR      = DATA_DIR / "caches"


# =============================================================================
# 3D assemblies
# =============================================================================
ASSEMBLIES_DIR  = MODULE_DIR / "assemblies"
RENDERS_DIR     = ASSEMBLIES_DIR / "renders"


# =============================================================================
# v0.2 memories and records
# =============================================================================
MEMORIES_DIR    = MODULE_DIR / "memories"
SNAPSHOTS_DIR   = MEMORIES_DIR / "snapshots"


# =============================================================================
# v0.1 references (read-only; nothing writes here)
# =============================================================================
V01_ROOT_DIR        = Path(r"F:\New folder (4)\New folder")
V01_MEMORIES_DIR    = V01_ROOT_DIR / "Memories"
V01_EXPLORE_DIR     = V01_ROOT_DIR / "Explore"
V01_EXPLORE_DATA    = V01_EXPLORE_DIR / "data"


# =============================================================================
# Directory creation
# =============================================================================
def ensure_all_dirs() -> None:
    """
    Create every writable directory under the 3d folder if it does not
    exist. Safe to call repeatedly. Does not touch the v0.1 references.
    """
    for d in (
        DATA_DIR,
        STORAGES_DIR,
        POINTCLOUD_DIR,
        CACHES_DIR,
        ASSEMBLIES_DIR,
        RENDERS_DIR,
        MEMORIES_DIR,
        SNAPSHOTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def ensure_snapshot_dir(generation: int) -> Path:
    """
    Create and return a snapshot directory under MEMORIES_DIR/snapshots/
    for the given generation number (e.g. snapshots/gen_100/).
    """
    snap_dir = SNAPSHOTS_DIR / f"gen_{generation}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    return snap_dir


# =============================================================================
# Convenience accessors
# =============================================================================
def storage_file(name: str) -> Path:
    """
    Return a path under STORAGES_DIR. Appends .json if the name has no
    extension. Example: storage_file("3d_elements_storage") returns
    STORAGES_DIR / "3d_elements_storage.json".
    """
    p = Path(name)
    if p.suffix == "":
        p = p.with_suffix(".json")
    return STORAGES_DIR / p


def assembly_file(memory_id: str) -> Path:
    """Return the canonical 3D assembly path for a memory ID."""
    return ASSEMBLIES_DIR / f"{memory_id}.3dassembly.json"


def render_file(memory_id: str, variant: str = "solid") -> Path:
    """
    Return the canonical render path for a memory ID.
    Variant is a short suffix like "solid" or "wireframe".
    """
    return RENDERS_DIR / f"{memory_id}_{variant}.png"


def memory_file(name: str) -> Path:
    """
    Return a path under MEMORIES_DIR. Appends .jsonl if the name has no
    extension. Example: memory_file("memories") returns
    MEMORIES_DIR / "memories.jsonl".
    """
    p = Path(name)
    if p.suffix == "":
        p = p.with_suffix(".jsonl")
    return MEMORIES_DIR / p


# =============================================================================
# Diagnostics
# =============================================================================
def summary() -> str:
    """Human-readable summary of every path this module manages."""
    lines = [
        "3d_paths summary",
        "=" * 60,
        f"MODULE_DIR       {MODULE_DIR}",
        f"DATA_DIR         {DATA_DIR}",
        f"STORAGES_DIR     {STORAGES_DIR}",
        f"POINTCLOUD_DIR   {POINTCLOUD_DIR}",
        f"CACHES_DIR       {CACHES_DIR}",
        f"ASSEMBLIES_DIR   {ASSEMBLIES_DIR}",
        f"RENDERS_DIR      {RENDERS_DIR}",
        f"MEMORIES_DIR     {MEMORIES_DIR}",
        f"SNAPSHOTS_DIR    {SNAPSHOTS_DIR}",
        "",
        "v0.1 references (read-only)",
        "-" * 60,
        f"V01_ROOT_DIR     {V01_ROOT_DIR}",
        f"V01_MEMORIES_DIR {V01_MEMORIES_DIR}",
        f"V01_EXPLORE_DIR  {V01_EXPLORE_DIR}",
        f"V01_EXPLORE_DATA {V01_EXPLORE_DATA}",
    ]
    return "\n".join(lines)


# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("3d_paths.py – smoke test")
    print("=" * 70)

    print()
    print(summary())

    print()
    print("Creating all writable directories...")
    ensure_all_dirs()
    for d in (
        DATA_DIR, STORAGES_DIR, POINTCLOUD_DIR, CACHES_DIR,
        ASSEMBLIES_DIR, RENDERS_DIR, MEMORIES_DIR, SNAPSHOTS_DIR,
    ):
        status = "exists" if d.exists() else "MISSING"
        print(f"  {d}  [{status}]")

    print()
    print("Snapshot helper:")
    snap = ensure_snapshot_dir(100)
    print(f"  gen_100 snapshot dir: {snap}  [{'exists' if snap.exists() else 'MISSING'}]")

    print()
    print("Convenience accessors:")
    print(f"  storage_file('3d_elements_storage')  -> {storage_file('3d_elements_storage')}")
    print(f"  assembly_file('mem_000011')          -> {assembly_file('mem_000011')}")
    print(f"  render_file('mem_000011', 'solid')   -> {render_file('mem_000011', 'solid')}")
    print(f"  memory_file('memories')              -> {memory_file('memories')}")

    print()
    print("Smoke test complete.")