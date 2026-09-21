#!/usr/bin/env python3
r"""
list_spatial_relations.py

Reads memory files from:
    F:\New folder (4)\New folder\Memories

Counts unique spatial relation values, e.g.:
    on
    in
    above
    below
    near
    etc.

Writes:
    Explore\spatial_relations_readable.txt
    Explore\data\spatial_relations_summary.json

No memory files are modified.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
READABLE_OUTPUT = EXPLORE_DIR / "spatial_relations_readable.txt"
JSON_OUTPUT = EXPLORE_DIR / "data" / "spatial_relations_summary.json"


# =============================================================================
# Memory loading
# =============================================================================
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
    """Return object dicts from a memory encoding."""
    inner = memory.get("memory") or {}
    objects = inner.get("objects") or []

    if objects:
        return [obj for obj in objects if isinstance(obj, dict)]

    # Fallback if memory is stored without an inner memory block
    objects = memory.get("objects") or []
    return [obj for obj in objects if isinstance(obj, dict)]


# =============================================================================
# Relation extraction
# =============================================================================
def extract_relation_counts(memories: List[Dict[str, Any]]):
    position_counter = Counter()
    body_counter = Counter()

    for memory in memories:
        for obj in get_memory_objects(memory):
            # Standard position relation
            position = obj.get("position")
            if isinstance(position, dict):
                relation = position.get("relation")
                if relation is not None:
                    position_counter[str(relation).strip()] += 1

            # Body-relative position relation, if present
            pos_rel_body = obj.get("position_relative_body")
            if isinstance(pos_rel_body, dict):
                relation = pos_rel_body.get("relation")
                if relation is not None:
                    body_counter[str(relation).strip()] += 1

    return position_counter, body_counter


# =============================================================================
# Output
# =============================================================================
def write_readable(
    position_counter: Counter,
    body_counter: Counter,
    output_path: Path,
):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("SPATIAL RELATION COUNTS\n")
        f.write("=" * 80 + "\n\n")

        f.write("OBJECT POSITION RELATIONS\n")
        f.write("(from position.relation)\n")
        f.write("-" * 80 + "\n")

        if position_counter:
            for relation, count in position_counter.most_common():
                f.write(f"{relation:30s} {count}\n")
        else:
            f.write("No position.relation values found.\n")

        f.write("\n")

        f.write("BODY-RELATIVE POSITION RELATIONS\n")
        f.write("(from position_relative_body.relation)\n")
        f.write("-" * 80 + "\n")

        if body_counter:
            for relation, count in body_counter.most_common():
                f.write(f"{relation:30s} {count}\n")
        else:
            f.write("No position_relative_body.relation values found.\n")

        f.write("\n")

    print(f"📄 Readable spatial relation counts saved to:\n   {output_path}")


def write_json(
    position_counter: Counter,
    body_counter: Counter,
    output_path: Path,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "object_position_relations": dict(position_counter.most_common()),
        "body_relative_position_relations": dict(body_counter.most_common()),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"📄 JSON summary saved to:\n   {output_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Loading memories...")
    memories = load_memory_records()
    print(f"✅ Loaded {len(memories)} memory records.")

    print("🔍 Extracting spatial relations...")
    position_counter, body_counter = extract_relation_counts(memories)

    write_readable(position_counter, body_counter, READABLE_OUTPUT)
    write_json(position_counter, body_counter, JSON_OUTPUT)

    if position_counter:
        print("\nTop object position relations:")
        for relation, count in position_counter.most_common():
            print(f"  {relation:30s} {count}")


if __name__ == "__main__":
    main()