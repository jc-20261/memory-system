#!/usr/bin/env python3
r"""
action_duration_audit.py

Audits all actions and sub-actions in the current memory files.

Inputs:
    F:\New folder (4)\New folder\Memories

Outputs:
    Explore\action_duration_audit.txt
    Explore\data\action_duration_audit.json

Reports:
- Total action instances
- Actions missing duration
- Actions with duration == 0
- Cyclical actions
- Duration distribution summary

No memory files are modified.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
READABLE_OUTPUT = EXPLORE_DIR / "action_duration_audit.txt"
JSON_OUTPUT = EXPLORE_DIR / "data" / "action_duration_audit.json"


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
# Action extraction
# =============================================================================
def collect_action_records(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Walk all action dicts and sub-actions in one memory.

    Returns list of action records.
    """
    memory_id = get_memory_id(memory)
    inner = memory.get("memory") or {}
    actions = inner.get("actions") or []

    records = []

    def walk(action: Dict[str, Any], path_parts: List[str]):
        if not isinstance(action, dict):
            return

        record = {
            "memory_id": memory_id,
            "template": action.get("template"),
            "duration": action.get("duration"),
            "temporal_type": action.get("temporal_type", "sequential"),
            "repetitions": action.get("repetitions"),
            "cycle_duration": action.get("cycle_duration"),
            "start_offset": action.get("start_offset"),
            "has_sub_actions": bool(action.get("sub_actions")),
            "path_parts": list(path_parts),
            "path": ".".join(path_parts),
        }

        records.append(record)

        sub_actions = action.get("sub_actions") or []
        if isinstance(sub_actions, list):
            for i, sub in enumerate(sub_actions):
                if isinstance(sub, dict):
                    walk(
                        sub,
                        path_parts + ["sub_actions", str(i)],
                    )

    for i, action in enumerate(actions):
        if isinstance(action, dict):
            walk(action, ["memory", "actions", str(i)])

    return records


# =============================================================================
# Audit
# =============================================================================
def analyze_duration(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)

    missing_duration = []
    zero_duration = []
    cyclical = []

    duration_counts = defaultdict(int)

    for rec in records:
        duration = rec.get("duration")
        temporal_type = rec.get("temporal_type")

        if duration is None:
            missing_duration.append(rec)
        else:
            try:
                duration_float = float(duration)
                if duration_float == 0.0:
                    zero_duration.append(rec)
            except (TypeError, ValueError):
                missing_duration.append(rec)

        if temporal_type == "cyclical":
            cyclical.append(rec)

        # Simple duration distribution
        if duration is not None:
            try:
                duration_int = int(round(float(duration)))
                duration_counts[duration_int] += 1
            except (TypeError, ValueError):
                pass

    return {
        "total_actions": total,
        "missing_duration_count": len(missing_duration),
        "zero_duration_count": len(zero_duration),
        "cyclical_count": len(cyclical),
        "missing_duration": missing_duration,
        "zero_duration": zero_duration,
        "cyclical": cyclical,
        "duration_distribution": dict(sorted(duration_counts.items())),
    }


# =============================================================================
# Output
# =============================================================================
def write_outputs(summary: Dict[str, Any], readable_path: Path, json_path: Path):
    # Readable text
    with open(readable_path, "w", encoding="utf-8") as f:
        f.write("ACTION DURATION AUDIT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total action instances: {summary['total_actions']}\n")
        f.write(f"Actions missing duration: {summary['missing_duration_count']}\n")
        f.write(f"Actions with duration == 0: {summary['zero_duration_count']}\n")
        f.write(f"Cyclical actions: {summary['cyclical_count']}\n")
        f.write("=" * 80 + "\n\n")

        if summary["missing_duration"]:
            f.write("ACTIONS MISSING DURATION\n")
            f.write("-" * 80 + "\n")
            for rec in summary["missing_duration"]:
                f.write(f"Memory ID: {rec['memory_id']}\n")
                f.write(f"Template: {rec['template']}\n")
                f.write(f"Path: {rec['path']}\n")
                f.write(f"Temporal type: {rec['temporal_type']}\n")
                f.write("-" * 60 + "\n")

        if summary["zero_duration"]:
            f.write("\nACTIONS WITH DURATION == 0\n")
            f.write("-" * 80 + "\n")
            for rec in summary["zero_duration"]:
                f.write(f"Memory ID: {rec['memory_id']}\n")
                f.write(f"Template: {rec['template']}\n")
                f.write(f"Path: {rec['path']}\n")
                f.write(f"Temporal type: {rec['temporal_type']}\n")
                f.write("-" * 60 + "\n")

        if summary["cyclical"]:
            f.write("\nCYCLICAL ACTIONS\n")
            f.write("-" * 80 + "\n")
            for rec in summary["cyclical"]:
                f.write(f"Memory ID: {rec['memory_id']}\n")
                f.write(f"Template: {rec['template']}\n")
                f.write(f"Duration: {rec['duration']}\n")
                f.write(f"Cycle duration: {rec['cycle_duration']}\n")
                f.write(f"Repetitions: {rec['repetitions']}\n")
                f.write(f"Path: {rec['path']}\n")
                f.write("-" * 60 + "\n")

        f.write("\nDURATION DISTRIBUTION\n")
        f.write("(rounded to nearest second)\n")
        f.write("-" * 80 + "\n")
        for duration, count in summary["duration_distribution"].items():
            f.write(f"{duration:>6d}s: {count}\n")

    # JSON output
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"📄 Readable audit saved to:\n   {readable_path}")
    print(f"📄 JSON audit saved to:\n   {json_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Loading memories...")
    memories = load_memory_records()
    print(f"✅ Loaded {len(memories)} memory records.")

    print("🔍 Extracting action records...")
    all_action_records = []

    for memory in memories:
        records = collect_action_records(memory)
        all_action_records.extend(records)

    print(f"   Total action instances: {len(all_action_records)}")

    summary = analyze_duration(all_action_records)

    print(f"   Missing duration: {summary['missing_duration_count']}")
    print(f"   Zero duration:    {summary['zero_duration_count']}")
    print(f"   Cyclical actions: {summary['cyclical_count']}")

    write_outputs(summary, READABLE_OUTPUT, JSON_OUTPUT)


if __name__ == "__main__":
    main()