#!/usr/bin/env python3
r"""
list_selected_spatial_relations.py

Reads memory files from:
    F:\New folder (4)\New folder\Memories

Extracts position.relation values for only:
    attached
    between
    front_of

For each occurrence, outputs:
    Memory ID
    relation
    relative_to

Outputs:
    Explore\selected_spatial_relations.txt
    Explore\data\selected_spatial_relations.json

No memory files are modified.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
READABLE_OUTPUT = EXPLORE_DIR / "selected_spatial_relations.txt"
JSON_OUTPUT = EXPLORE_DIR / "data" / "selected_spatial_relations.json"

TARGET_RELATIONS = {"attached", "between", "front_of"}


# =============================================================================
# Memory loading
# =============================================================================
def get_memory_id(memory: Dict[str, Any]) -> str:
    memory_id = memory.get("memory_id")
    if memory_id:
        return str(memory_id)

    inner = memory.get("memory") or {}
    memory_id = inner.get("memory_id")
    return str(memory_id) if memory_id else "unknown"


def load_memory_records() -> List[Dict[str, Any]]:
    memories = []

    for jsonl_file in sorted(MEMORIES_DIR.glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line == "---":
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        memories.append(data)
                except json.JSONDecodeError:
                    continue

    for json_file in sorted(MEMORIES_DIR.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        if isinstance(data, dict):
            memories.append(data)
        elif isinstance(data, list):
            memories.extend(data)

    return memories


def get_memory_objects(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    inner = memory.get("memory") or {}
    objects = inner.get("objects") or []

    if objects:
        return [obj for obj in objects if isinstance(obj, dict)]

    objects = memory.get("objects") or []
    return [obj for obj in objects if isinstance(obj, dict)]


# =============================================================================
# Extraction
# =============================================================================
def extract_selected_relations(
    memories: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    records = []

    for memory in memories:
        memory_id = get_memory_id(memory)

        for obj in get_memory_objects(memory):
            position = obj.get("position")
            if not isinstance(position, dict):
                continue

            relation = position.get("relation")
            if relation is None:
                continue

            relation_str = str(relation).strip().lower()

            if relation_str not in TARGET_RELATIONS:
                continue

            relative_to = position.get("relative_to")
            relative_to = str(relative_to) if relative_to is not None else ""

            records.append(
                {
                    "memory_id": memory_id,
                    "relation": str(relation).strip(),
                    "relative_to": relative_to,
                }
            )

    return records


# =============================================================================
# Output
# =============================================================================
def write_readable(records: List[Dict[str, Any]], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("SELECTED SPATIAL RELATIONS\n")
        f.write("=" * 80 + "\n")
        f.write("Relations shown:\n")
        for rel in sorted(TARGET_RELATIONS):
            f.write(f"  - {rel}\n")
        f.write("=" * 80 + "\n\n")

        if not records:
            f.write("No matching spatial relations found.\n")
            return

        for rec in records:
            f.write(f"{rec['memory_id']}\n")
            f.write(f"{rec['relation']}\n")
            f.write(f"{rec['relative_to']}\n")
            f.write("\n")

    print(f"📄 Readable output saved to:\n   {output_path}")


def write_json(records: List[Dict[str, Any]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"📄 JSON output saved to:\n   {output_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Loading memories...")
    memories = load_memory_records()
    print(f"✅ Loaded {len(memories)} memory records.")

    print("🔍 Extracting selected spatial relations...")
    records = extract_selected_relations(memories)
    print(f"   Matching occurrences: {len(records)}")

    write_readable(records, READABLE_OUTPUT)
    write_json(records, JSON_OUTPUT)


if __name__ == "__main__":
    main()