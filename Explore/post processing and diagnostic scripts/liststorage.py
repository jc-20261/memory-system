#!/usr/bin/env python3
r"""
list_templates_instances_by_memory.py

Reads memory files from:
    F:\New folder (4)\New folder\Memories

Produces two lists:

1. Action template usage:
    memory_id, action_template

2. Object instance usage:
    memory_id, object_template, object_instance_id, aliases

Outputs:
    Explore\action_templates_by_memory.txt
    Explore\data\action_templates_by_memory.json

    Explore\object_instances_by_memory.txt
    Explore\data\object_instances_by_memory.json

No memory files are modified.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
DATA_DIR = EXPLORE_DIR / "data"

ACTION_TEXT_OUTPUT = EXPLORE_DIR / "action_templates_by_memory.txt"
ACTION_JSON_OUTPUT = DATA_DIR / "action_templates_by_memory.json"

OBJECT_TEXT_OUTPUT = EXPLORE_DIR / "object_instances_by_memory.txt"
OBJECT_JSON_OUTPUT = DATA_DIR / "object_instances_by_memory.json"


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


# =============================================================================
# Extraction
# =============================================================================
def get_memory_objects(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    inner = memory.get("memory") or {}
    objects = inner.get("objects") or []
    return [obj for obj in objects if isinstance(obj, dict)]


def get_memory_actions(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    inner = memory.get("memory") or {}
    actions = inner.get("actions") or []
    return [act for act in actions if isinstance(act, dict)]


def collect_action_template_usages(memory: Dict[str, Any]) -> Set[Tuple[str, str]]:
    """
    Collect unique (memory_id, action_template) pairs, including sub-actions.
    """
    memory_id = get_memory_id(memory)
    actions = get_memory_actions(memory)

    pairs = set()

    def walk(action: Dict[str, Any]):
        if not isinstance(action, dict):
            return

        template = action.get("template")
        if template:
            pairs.add((memory_id, str(template)))

        for sub in action.get("sub_actions") or []:
            if isinstance(sub, dict):
                walk(sub)

    for action in actions:
        walk(action)

    return pairs


def collect_object_instance_records(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Collect records for each object instance:
    memory_id, object_template, object_instance_id, aliases
    """
    memory_id = get_memory_id(memory)
    objects = get_memory_objects(memory)

    records = []

    for obj in objects:
        obj_id = obj.get("obj_id")
        template = obj.get("template")

        if not obj_id or not template:
            continue

        aliases = obj.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]

        records.append(
            {
                "memory_id": memory_id,
                "object_template": str(template),
                "object_instance_id": str(obj_id),
                "aliases": [str(a) for a in aliases if a],
            }
        )

    return records


# =============================================================================
# Output helpers
# =============================================================================
def write_action_output(action_pairs: Set[Tuple[str, str]], text_path: Path, json_path: Path):
    action_pairs = sorted(action_pairs, key=lambda x: (x[0], x[1]))

    with open(text_path, "w", encoding="utf-8") as f:
        f.write("ACTION TEMPLATES BY MEMORY\n")
        f.write("=" * 80 + "\n\n")

        current_memory_id = None
        for memory_id, action_template in action_pairs:
            if memory_id != current_memory_id:
                if current_memory_id is not None:
                    f.write("\n")
                f.write(f"Memory ID: {memory_id}\n")
                current_memory_id = memory_id
            f.write(f"  - {action_template}\n")

    summary = [
        {
            "memory_id": memory_id,
            "action_template": action_template,
        }
        for memory_id, action_template in action_pairs
    ]

    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"📄 Action template list saved to:\n   {text_path}")
    print(f"📄 Action template JSON saved to:\n   {json_path}")


def write_object_output(object_records: List[Dict[str, Any]], text_path: Path, json_path: Path):
    object_records = sorted(
        object_records,
        key=lambda x: (x["memory_id"], x["object_template"], x["object_instance_id"]),
    )

    with open(text_path, "w", encoding="utf-8") as f:
        f.write("OBJECT INSTANCES BY MEMORY\n")
        f.write("=" * 80 + "\n\n")

        current_memory_id = None
        for rec in object_records:
            if rec["memory_id"] != current_memory_id:
                if current_memory_id is not None:
                    f.write("\n")
                f.write(f"Memory ID: {rec['memory_id']}\n")
                current_memory_id = rec["memory_id"]

            alias_str = ", ".join(rec["aliases"]) if rec["aliases"] else ""
            if alias_str:
                f.write(
                    f"  - {rec['object_instance_id']} ({rec['object_template']}) "
                    f"aliases: {alias_str}\n"
                )
            else:
                f.write(
                    f"  - {rec['object_instance_id']} ({rec['object_template']})\n"
                )

    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(object_records, f, indent=2, ensure_ascii=False)

    print(f"📄 Object instance list saved to:\n   {text_path}")
    print(f"📄 Object instance JSON saved to:\n   {json_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Loading memories...")
    memories = load_memory_records()
    print(f"✅ Loaded {len(memories)} memory records.")

    print("🔍 Extracting action templates...")
    action_pairs = set()
    object_records = []

    for memory in memories:
        action_pairs.update(collect_action_template_usages(memory))
        object_records.extend(collect_object_instance_records(memory))

    print(f"   Unique action template pairs: {len(action_pairs)}")
    print(f"   Object instance records: {len(object_records)}")

    write_action_output(action_pairs, ACTION_TEXT_OUTPUT, ACTION_JSON_OUTPUT)
    write_object_output(object_records, OBJECT_TEXT_OUTPUT, OBJECT_JSON_OUTPUT)

    print("✅ Done.")


if __name__ == "__main__":
    main()