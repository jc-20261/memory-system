#!/usr/bin/env python3
r"""
export_goal_states_readable.py

Reads memory files from:
    F:\New folder (4)\New folder\Memories

Extracts each memory's activity and goal_state list, then writes a readable
summary to:
    Explore\goal_states_readable.txt

Also writes a compact JSON summary to:
    Explore\data\goal_states_summary.json

No memory files are modified.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
READABLE_OUTPUT = EXPLORE_DIR / "goal_states_readable.txt"
JSON_OUTPUT = EXPLORE_DIR / "data" / "goal_states_summary.json"


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


def get_goal_states(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return goal_state list from top-level or inner memory block."""
    goal_state = memory.get("goal_state")
    if not isinstance(goal_state, list):
        inner = memory.get("memory") or {}
        goal_state = inner.get("goal_state")

    if not isinstance(goal_state, list):
        return []

    result = []
    for goal in goal_state:
        if isinstance(goal, dict):
            result.append(goal)

    return result


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
# Formatting
# =============================================================================
def format_goal_state(goal: Dict[str, Any]) -> str:
    obj = goal.get("object", "?")
    attribute = goal.get("attribute", "?")
    value = goal.get("value")
    aspect = goal.get("aspect")
    op = goal.get("op", "eq")

    if aspect:
        target = f"{obj}.{aspect}.{attribute}"
    else:
        target = f"{obj}.{attribute}"

    op_symbol = {
        "eq": "=",
        "ne": "!=",
        "gt": ">",
        "gte": ">=",
        "lt": "<",
        "lte": "<=",
    }.get(op, op)

    return f"{target} {op_symbol} {value}"


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Loading memory records...")
    memories = load_memory_records()
    print(f"✅ Loaded {len(memories)} memory records.")

    activity_goal_states: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for memory in memories:
        activity = get_activity(memory)
        memory_id = get_memory_id(memory)
        goals = get_goal_states(memory)

        activity_goal_states[activity].append(
            {
                "memory_id": memory_id,
                "goal_state": goals,
            }
        )

    if not activity_goal_states:
        print("⚠️ No goal states found.")
        return

    # Readable file
    with open(READABLE_OUTPUT, "w", encoding="utf-8") as f:
        f.write("GOAL STATES BY ACTIVITY\n")
        f.write("=" * 80 + "\n\n")

        total_goals = 0
        total_memories_with_goals = 0

        for activity, records in sorted(activity_goal_states.items()):
            for record in records:
                goals = record["goal_state"]
                if not goals:
                    continue

                total_memories_with_goals += 1
                total_goals += len(goals)

                f.write(f"ACTIVITY: {activity}\n")
                f.write(f"Memory ID: {record['memory_id']}\n")
                f.write("-" * 80 + "\n")

                for goal in goals:
                    f.write(f"  - {format_goal_state(goal)}\n")

                f.write("\n")

        f.write("=" * 80 + "\n")
        f.write(f"Total memories with goal states: {total_memories_with_goals}\n")
        f.write(f"Total goal states: {total_goals}\n")

    # JSON summary
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    json_summary = {}
    for activity, records in activity_goal_states.items():
        json_summary[activity] = []
        for record in records:
            goals = record["goal_state"]
            if goals:
                json_summary[activity].append(
                    {
                        "memory_id": record["memory_id"],
                        "goal_state": goals,
                    }
                )

    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Readable goal states saved to:")
    print(f"   {READABLE_OUTPUT}")

    print(f"📄 JSON summary saved to:")
    print(f"   {JSON_OUTPUT}")


if __name__ == "__main__":
    main()