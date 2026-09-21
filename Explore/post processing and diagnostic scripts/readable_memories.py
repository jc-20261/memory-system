#!/usr/bin/env python3
r"""
export_memories_readable.py

Reads all memory files from:
    F:\New folder (4)\New folder\Memories

Writes a human-readable vertical text file to:
    Explore\memories_readable.txt

Each memory is displayed with:
- Memory ID
- Activity
- Object instances
- Actions (recursively, with sub-actions)
- Goal states

No memory files are modified.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
READABLE_OUTPUT = EXPLORE_DIR / "memories_readable.txt"


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


def get_memory_objects(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    inner = memory.get("memory") or {}
    objects = inner.get("objects") or []
    return [obj for obj in objects if isinstance(obj, dict)]


def get_memory_actions(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    inner = memory.get("memory") or {}
    actions = inner.get("actions") or []
    return [act for act in actions if isinstance(act, dict)]


def get_goal_states(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    goal_state = memory.get("goal_state")
    if not isinstance(goal_state, list):
        inner = memory.get("memory") or {}
        goal_state = inner.get("goal_state")

    if not isinstance(goal_state, list):
        return []

    return [goal for goal in goal_state if isinstance(goal, dict)]


# =============================================================================
# Formatting helpers
# =============================================================================
def format_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def write_object_info(f, obj: Dict[str, Any], indent: str = "  "):
    f.write(f"{indent}Object ID: {obj.get('obj_id', '?')}\n")
    f.write(f"{indent}Template: {obj.get('template', '?')}\n")

    number = obj.get("number")
    if number:
        f.write(f"{indent}Number: {number}\n")

    position = obj.get("position")
    if isinstance(position, dict):
        relation = position.get("relation")
        relative_to = position.get("relative_to")
        if relation:
            f.write(f"{indent}Position relation: {relation}\n")
        if relative_to:
            f.write(f"{indent}Position relative_to: {relative_to}\n")

    shape = obj.get("shape")
    if shape:
        f.write(f"{indent}Shape: {shape}\n")

    dimensions = obj.get("dimensions")
    if isinstance(dimensions, dict):
        f.write(f"{indent}Dimensions: {format_value(dimensions)}\n")

    overrides = obj.get("overrides")
    if isinstance(overrides, dict) and overrides:
        f.write(f"{indent}Overrides:\n")
        for key, value in overrides.items():
            f.write(f"{indent}  {key}: {format_value(value)}\n")


def write_action_info(f, action: Dict[str, Any], depth: int = 0):
    indent = "  " * depth

    f.write(f"{indent}Action Template: {action.get('template', '?')}\n")
    f.write(f"{indent}Duration: {action.get('duration', '?')}\n")

    temporal_type = action.get("temporal_type")
    if temporal_type:
        f.write(f"{indent}Temporal Type: {temporal_type}\n")

    participants = action.get("participants")
    if isinstance(participants, list) and participants:
        f.write(f"{indent}Participants: {', '.join(str(p) for p in participants)}\n")

    preconditions = action.get("preconditions")
    if isinstance(preconditions, list) and preconditions:
        f.write(f"{indent}Preconditions: {format_value(preconditions)}\n")

    changes = action.get("changes")
    if isinstance(changes, list) and changes:
        f.write(f"{indent}Changes: {format_value(changes)}\n")

    changes_total = action.get("changes_total")
    if isinstance(changes_total, list) and changes_total:
        f.write(f"{indent}Changes Total: {format_value(changes_total)}\n")

    # Recursively write sub-actions
    sub_actions = action.get("sub_actions")
    if isinstance(sub_actions, list) and sub_actions:
        f.write(f"{indent}Sub-actions:\n")
        for sub in sub_actions:
            if isinstance(sub, dict):
                write_action_info(f, sub, depth + 2)
                f.write("\n")


def write_goal_state_info(f, goal: Dict[str, Any], indent: str = "  "):
    f.write(f"{indent}Goal Object: {goal.get('object', '?')}\n")
    f.write(f"{indent}Goal Attribute: {goal.get('attribute', '?')}\n")
    f.write(f"{indent}Goal Value: {format_value(goal.get('value'))}\n")
    op = goal.get("op")
    if op:
        f.write(f"{indent}Goal Operator: {op}\n")


# =============================================================================
# Main export
# =============================================================================
def main():
    print("📂 Loading memories...")
    memories = load_memory_records()
    print(f"✅ Loaded {len(memories)} memory records.")

    with open(READABLE_OUTPUT, "w", encoding="utf-8") as f:
        f.write("MEMORIES READABLE EXPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total memories: {len(memories)}\n")
        f.write("=" * 80 + "\n\n")

        for memory in memories:
            memory_id = get_memory_id(memory)
            activity = get_activity(memory)

            f.write(f"MEMORY ID: {memory_id}\n")
            f.write(f"ACTIVITY : {activity}\n")
            f.write("-" * 80 + "\n")

            objects = get_memory_objects(memory)
            if objects:
                f.write("\nOBJECTS\n")
                for obj in objects:
                    write_object_info(f, obj)
                    f.write("\n")

            actions = get_memory_actions(memory)
            if actions:
                f.write("\nACTIONS\n")
                for action in actions:
                    write_action_info(f, action, depth=0)
                    f.write("\n")

            goals = get_goal_states(memory)
            if goals:
                f.write("\nGOAL STATES\n")
                for goal in goals:
                    write_goal_state_info(f, goal)
                    f.write("\n")

            f.write("=" * 80 + "\n\n")

    print(f"📄 Readable memories saved to:\n   {READABLE_OUTPUT}")


if __name__ == "__main__":
    main()