#!/usr/bin/env python3
r"""
list_low_frequency_spatial_relations.py

Reads memory files from:
    F:\New folder (4)\New folder\Memories

Finds position.relation values whose total occurrence count is less than 10.
For each occurrence, prints:

    Memory ID
    <relation>
    <relative_to>

Outputs:
    Explore\low_frequency_spatial_relations.txt
    Explore\data\low_frequency_spatial_relations.json

No memory files are modified.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
READABLE_OUTPUT = EXPLORE_DIR / "low_frequency_spatial_relations.txt"
JSON_OUTPUT = EXPLORE_DIR / "data" / "low_frequency_spatial_relations.json"

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
def extract_spatial_relations(
    memories: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Return a list of records:
    {
        "memory_id": str,
        "relation": str,
        "relative_to": str
    }
    """
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


def filter_low_frequency_records(
    records: List[Dict[str, Any]],
    threshold: int = 10,
) -> List[Dict[str, Any]]:
    relation_counts = Counter(r["relation"] for r in records)

    low_frequency_records = [
        r for r in records
        if relation_counts[r["relation"]] < threshold
    ]

    return low_frequency_records


# =============================================================================
# Output
# =============================================================================
def write_readable(records: List[Dict[str, Any]], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("LOW-FREQUENCY SPATIAL RELATIONS\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"Showing position.relation values with total occurrence count < 10.\n"
        )
        f.write("=" * 80 + "\n\n")

        if not records:
            f.write("No low-frequency spatial relations found.\n")
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

    print("🔍 Extracting spatial relations...")
    records = extract_spatial_relations(memories)
    print(f"   Total spatial relation occurrences: {len(records)}")

    print(f"🔎 Filtering relations with count < 10...")
    low_freq_records = filter_low_frequency_records(records, threshold=10)
    print(f"   Low-frequency occurrence count: {len(low_freq_records)}")

    write_readable(low_freq_records, READABLE_OUTPUT)
    write_json(low_freq_records, JSON_OUTPUT)


if __name__ == "__main__":
    main()