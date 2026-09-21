#!/usr/bin/env python3
r"""
add_temporal_timelines.py

Computes action start_time, end_time, and effective_duration for every
action and sub-action in the memory files, then writes the updated JSON
back to:

    F:\New folder (4)\New folder\Memories

Temporal computation is deterministic and based on:
- duration
- temporal_type (sequential, parallel, cyclical)
- start_offset
- cycle_duration
- repetitions
- sub_action structure

No LLM is used.

Also writes a timeline summary to:

    Explore\timeline_summary.txt
    Explore\data\timeline_summary.json

Back up the Memories folder before running.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
READABLE_SUMMARY = EXPLORE_DIR / "timeline_summary.txt"
JSON_SUMMARY = EXPLORE_DIR / "data" / "timeline_summary.json"


# =============================================================================
# Memory loading/writing
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


def load_memory_records() -> List[Dict[str, Any]]:
    records = []

    for jsonl_file in sorted(MEMORIES_DIR.glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and line.strip() != "---"]

        memories = []
        for line in lines:
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    memories.append(data)
            except json.JSONDecodeError:
                continue

        for idx, memory in enumerate(memories):
            records.append(
                {
                    "file_path": jsonl_file,
                    "memory": memory,
                    "index": idx,
                    "is_jsonl": True,
                }
            )

    for json_file in sorted(MEMORIES_DIR.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        if isinstance(data, dict):
            memories = [data]
        elif isinstance(data, list):
            memories = data
        else:
            continue

        for idx, memory in enumerate(memories):
            if isinstance(memory, dict):
                records.append(
                    {
                        "file_path": json_file,
                        "memory": memory,
                        "index": idx,
                        "is_jsonl": False,
                    }
                )

    return records


def write_memory_records(records: List[Dict[str, Any]]):
    file_groups = defaultdict(list)

    for rec in records:
        file_groups[rec["file_path"]].append(rec)

    for file_path, recs in file_groups.items():
        recs.sort(key=lambda x: x["index"])
        is_jsonl = recs[0]["is_jsonl"] if recs else False
        memories = [rec["memory"] for rec in recs]

        if is_jsonl:
            with open(file_path, "w", encoding="utf-8") as f:
                for mem in memories:
                    f.write(json.dumps(mem, ensure_ascii=False) + "\n\n---\n")
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                if len(memories) == 1:
                    json.dump(memories[0], f, indent=2, ensure_ascii=False)
                else:
                    json.dump(memories, f, indent=2, ensure_ascii=False)


# =============================================================================
# Timeline calculation
# =============================================================================
def compute_effective_duration(action: Dict[str, Any]) -> float:
    """Return the effective duration of an action, without considering children."""
    temporal_type = action.get("temporal_type", "sequential")
    duration = action.get("duration", 0)

    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 0.0

    if temporal_type == "cyclical":
        cycle_duration = action.get("cycle_duration")
        repetitions = action.get("repetitions")

        try:
            cycle_duration = float(cycle_duration) if cycle_duration is not None else None
            repetitions = float(repetitions) if repetitions is not None else None
        except (TypeError, ValueError):
            cycle_duration = None
            repetitions = None

        if (
            cycle_duration is not None
            and repetitions is not None
            and cycle_duration > 0
            and repetitions > 0
        ):
            return cycle_duration * repetitions

    return duration


def process_action(
    action: Dict[str, Any],
    parent_start: float,
    parent_offset: float,
) -> float:
    """
    Compute start_time, end_time, effective_duration for one action.

    Returns the action's end_time.
    """
    temporal_type = action.get("temporal_type", "sequential")

    start_offset = action.get("start_offset", 0.0)
    try:
        start_offset = float(start_offset)
    except (TypeError, ValueError):
        start_offset = 0.0

    action_start = parent_start + start_offset
    effective_duration = compute_effective_duration(action)

    sub_actions = action.get("sub_actions") or []

    if sub_actions and effective_duration == 0.0:
        # Derive duration from children
        if temporal_type == "parallel":
            max_end = action_start
            for child in sub_actions:
                if isinstance(child, dict):
                    child_end = process_action(child, action_start, action_start)
                    max_end = max(max_end, child_end)
            total_duration = max_end - action_start
        elif temporal_type == "sequential":
            current = action_start
            for child in sub_actions:
                if isinstance(child, dict):
                    current = process_action(child, current, current)
            total_duration = current - action_start
        else:
            total_duration = effective_duration
    else:
        total_duration = effective_duration

    action_end = action_start + total_duration

    action["start_time"] = action_start
    action["end_time"] = action_end
    action["effective_duration"] = total_duration

    # Process children if not already processed in zero-duration case
    if sub_actions and (effective_duration != 0.0 or not sub_actions):
        if temporal_type == "parallel":
            for child in sub_actions:
                if isinstance(child, dict):
                    process_action(child, action_start, action_start)
        elif temporal_type == "sequential":
            current = action_start
            for child in sub_actions:
                if isinstance(child, dict):
                    current = process_action(child, current, current)

    return action_end


def add_timelines_to_memory(memory: Dict[str, Any]) -> int:
    """Add timelines to all actions in a memory. Returns root action count."""
    inner = memory.get("memory") or memory
    actions = inner.get("actions") or []

    count = 0

    for action in actions:
        if isinstance(action, dict):
            process_action(action, 0.0, 0.0)
            count += 1

    return count


# =============================================================================
# Summary collection
# =============================================================================
def summarize_memory_timeline(memory: Dict[str, Any]) -> Dict[str, Any]:
    inner = memory.get("memory") or memory
    actions = inner.get("actions") or []

    total_actions = 0
    max_end_time = 0.0
    temporal_counts = defaultdict(int)
    zero_duration_count = 0
    root_action_count = len(actions)

    def walk(action: Dict[str, Any]):
        nonlocal total_actions, max_end_time, zero_duration_count

        total_actions += 1

        temporal_type = action.get("temporal_type", "sequential")
        temporal_counts[temporal_type] += 1

        effective_duration = action.get("effective_duration", 0.0)
        try:
            effective_duration = float(effective_duration)
        except (TypeError, ValueError):
            effective_duration = 0.0

        if effective_duration == 0.0:
            zero_duration_count += 1

        end_time = action.get("end_time", 0.0)
        try:
            end_time = float(end_time)
        except (TypeError, ValueError):
            end_time = 0.0

        max_end_time = max(max_end_time, end_time)

        for child in action.get("sub_actions") or []:
            if isinstance(child, dict):
                walk(child)

    for action in actions:
        if isinstance(action, dict):
            walk(action)

    return {
        "memory_id": get_memory_id(memory),
        "activity": get_activity(memory),
        "total_actions": total_actions,
        "root_actions": root_action_count,
        "max_end_time": max_end_time,
        "temporal_counts": dict(temporal_counts),
        "zero_duration_count": zero_duration_count,
    }


# =============================================================================
# Summary output
# =============================================================================
def write_summary(summaries: List[Dict[str, Any]], readable_path: Path, json_path: Path):
    with open(readable_path, "w", encoding="utf-8") as f:
        f.write("TEMPORAL TIMELINE SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total memories processed: {len(summaries)}\n")
        f.write("-" * 80 + "\n\n")

        for summary in summaries:
            f.write(f"Memory ID: {summary['memory_id']}\n")
            f.write(f"Activity : {summary['activity']}\n")
            f.write(f"Total actions: {summary['total_actions']}\n")
            f.write(f"Root actions : {summary['root_actions']}\n")
            f.write(f"Max end time : {summary['max_end_time']:.2f}s\n")
            f.write(f"Zero-duration actions: {summary['zero_duration_count']}\n")
            f.write("Temporal type counts:\n")
            for ttype, count in sorted(summary["temporal_counts"].items()):
                f.write(f"  {ttype}: {count}\n")
            f.write("-" * 80 + "\n")

    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(f"📄 Timeline summary saved to:\n   {readable_path}")
    print(f"📄 JSON summary saved to:\n   {json_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Loading memory files...")
    records = load_memory_records()
    print(f"✅ Loaded {len(records)} memory records.")

    total_actions_processed = 0

    for rec in records:
        memory = rec["memory"]
        count = add_timelines_to_memory(memory)
        total_actions_processed += count

    print(f"🔧 Processed {total_actions_processed} root actions.")

    print("💾 Writing updated memory files...")
    write_memory_records(records)

    print("📊 Collecting timeline summaries...")
    summaries = []

    for rec in records:
        summary = summarize_memory_timeline(rec["memory"])
        summaries.append(summary)

    write_summary(summaries, READABLE_SUMMARY, JSON_SUMMARY)

    print(f"📂 Updated memory files in:\n   {MEMORIES_DIR}")


if __name__ == "__main__":
    main()