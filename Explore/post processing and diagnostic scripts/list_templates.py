#!/usr/bin/env python3
r"""
list_memory_object_templates.py

Reads memory files from:
    F:\New folder (4)\New folder\Memories

For each memory, extracts:
    - memory ID
    - activity name
    - object template names used inside that memory

Outputs:
    Explore\memory_object_templates.txt
    Explore\data\memory_object_templates.json

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
READABLE_OUTPUT = EXPLORE_DIR / "memory_object_templates.txt"
JSON_OUTPUT = EXPLORE_DIR / "data" / "memory_object_templates.json"


# =============================================================================
# Helpers
# =============================================================================
def get_memory_id(memory: Dict[str, Any]) -> str:
    memory_id = memory.get("memory_id")
    if memory_id:
        return str(memory_id)

    inner = memory.get("memory") or {}
    memory_id = inner.get("memory_id")
    return str(memory_id) if memory_id else "unknown"


def get_activity(memory: Dict[str, Any]) -> str:
    activity = memory.get("activity")
    if activity:
        return str(activity)

    inner = memory.get("memory") or {}
    activity = inner.get("activity")
    return str(activity) if activity else "unknown"


def get_object_templates(memory: Dict[str, Any]) -> List[str]:
    """Return distinct object template names from this memory."""
    inner = memory.get("memory") or {}
    objects = inner.get("objects") or []

    templates = []
    for obj in objects:
        if isinstance(obj, dict):
            template = obj.get("template")
            if template:
                templates.append(str(template))

    # preserve first-seen order, but remove duplicates
    seen = set()
    unique_templates = []
    for template in templates:
        if template not in seen:
            seen.add(template)
            unique_templates.append(template)

    return unique_templates


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


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Loading memory records...")
    memories = load_memory_records()
    print(f"✅ Loaded {len(memories)} memory records.")

    records = []

    for memory in memories:
        memory_id = get_memory_id(memory)
        activity = get_activity(memory)
        templates = get_object_templates(memory)

        records.append(
            {
                "memory_id": memory_id,
                "activity": activity,
                "object_templates": templates,
            }
        )

    # Readable output
    with open(READABLE_OUTPUT, "w", encoding="utf-8") as f:
        f.write("MEMORY ACTIVITIES AND OBJECT TEMPLATES\n")
        f.write("=" * 80 + "\n\n")

        for record in records:
            f.write(f"ACTIVITY: {record['activity']}\n")
            f.write(f"Memory ID: {record['memory_id']}\n")
            f.write("-" * 80 + "\n")

            templates = record["object_templates"]
            if templates:
                for template in templates:
                    f.write(f"  - {template}\n")
            else:
                f.write("  (no object templates found)\n")

            f.write("\n")

    # JSON output
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Readable output saved to:")
    print(f"   {READABLE_OUTPUT}")

    print(f"📄 JSON output saved to:")
    print(f"   {JSON_OUTPUT}")

    print(f"\nSummary:")
    print(f"   Memories listed: {len(records)}")


if __name__ == "__main__":
    main()