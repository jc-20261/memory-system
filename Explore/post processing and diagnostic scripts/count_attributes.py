#!/usr/bin/env python3
r"""
attribute_component_frequencies.py

Counts individual attribute components across all memory encodings.

For every attribute path (from object overrides, preconditions, state changes,
conditional changes, goal states), each path component is counted separately.
Object identifiers (obj_id/template) are NOT included.

For example, path:
    left_hand_01 > configuration > grip.type
is counted as:
    configuration      (1 occurrence)
    grip.type          (1 occurrence)

For each component, four metrics are computed:
  - occurrence_count: total raw occurrences across all memories
  - occurrence_percentage: occurrence_count / total_component_occurrences * 100
  - memory_count: number of unique memories containing the component
  - memory_percentage: memory_count / total_memories * 100

Numeric-only components (e.g., list indices "0", "1") are skipped.

Outputs:
    Explore\attribute_component_frequencies.txt
    Explore\data\attribute_component_frequencies.json
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
OUTPUT_READABLE = EXPLORE_DIR / "attribute_component_frequencies.txt"
OUTPUT_JSON = EXPLORE_DIR / "data" / "attribute_component_frequencies.json"

# ----------------------------------------------------------------------
# Helpers for path extraction (robust aspect/attribute handling)
# ----------------------------------------------------------------------
def combine_value_unit(value: Any, unit: Any) -> str:
    if unit is None:
        return str(value)
    return f"{value}{unit}"

def flatten_to_path_components(obj: Any, prefix: Tuple[str, ...] = None) -> List[Tuple[Tuple[str, ...], Any]]:
    if prefix is None:
        prefix = ()
    result = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = prefix + (str(k),)
            if isinstance(v, dict):
                if "value" in v and set(v.keys()) <= {"value", "unit"}:
                    result.append((new_prefix, combine_value_unit(v.get("value"), v.get("unit"))))
                else:
                    result.extend(flatten_to_path_components(v, new_prefix))
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        result.extend(flatten_to_path_components(item, new_prefix))
                    else:
                        result.append((new_prefix, item))
            else:
                result.append((new_prefix, v))
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                result.extend(flatten_to_path_components(item, prefix))
            else:
                result.append((prefix, item))
    else:
        result.append((prefix, obj))
    return result

def collect_path_keys(node: Any, comps: List[str]):
    if isinstance(node, str):
        comps.append(node)
    elif isinstance(node, dict):
        for k, v in node.items():
            comps.append(str(k))
            if isinstance(v, dict):
                collect_path_keys(v, comps)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        collect_path_keys(item, comps)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, str):
                comps.append(item)
            elif isinstance(item, dict):
                collect_path_keys(item, comps)

def path_from_aspect_attribute(aspect: Any, attribute: Any) -> List[str]:
    comps: List[str] = []
    if aspect is not None:
        if isinstance(aspect, str):
            comps.append(aspect)
        elif isinstance(aspect, (dict, list)):
            collect_path_keys(aspect, comps)
    if attribute is not None:
        if isinstance(attribute, str):
            comps.append(attribute)
        elif isinstance(attribute, list):
            for item in attribute:
                if isinstance(item, str):
                    comps.append(item)
                elif isinstance(item, dict):
                    collect_path_keys(item, comps)
        elif isinstance(attribute, dict):
            collect_path_keys(attribute, comps)
    return comps

def extract_path_components_without_object(entry: Dict[str, Any]) -> List[Tuple[str, ...]]:
    """
    Extract attribute path components from a precondition/change/goal entry.
    Does NOT include the object identifier.
    """
    aspect = entry.get("aspect")
    attribute = entry.get("attribute")
    comps = path_from_aspect_attribute(aspect, attribute)
    if not comps:
        return []
    return [tuple(comps)]

# ----------------------------------------------------------------------
# Memory loading
# ----------------------------------------------------------------------
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
    mid = memory.get("memory_id")
    if mid:
        return str(mid)
    inner = memory.get("memory") or {}
    mid = inner.get("memory_id")
    return str(mid) if mid else "unknown"

def is_numeric_component(comp: str) -> bool:
    return bool(re.fullmatch(r"\d+", comp))

def collect_all_path_components(memory: Dict[str, Any]) -> List[Tuple[str, ...]]:
    """
    Return list of all attribute path tuples (without object identifiers)
    for all relevant fields in the memory.
    Each tuple represents one path; components are not flattened here.
    """
    paths: List[Tuple[str, ...]] = []
    inner = memory.get("memory") or memory

    # Objects: overrides and explicit fields
    objects = inner.get("objects") or []
    for obj in objects:
        if not isinstance(obj, dict):
            continue

        overrides = obj.get("overrides") or {}
        for path, _ in flatten_to_path_components(overrides):
            paths.append(path)  # path already does not include object

        for field in ("dimensions", "weight", "mass", "position",
                      "position_relative_body", "motion", "shape"):
            if field in obj:
                field_val = obj[field]
                if isinstance(field_val, dict):
                    for path, _ in flatten_to_path_components(field_val):
                        paths.append((field,) + path)
                else:
                    paths.append((field,))

    # Actions: preconditions, changes, conditional changes (recursive)
    actions = inner.get("actions") or []
    for action in actions:
        if isinstance(action, dict):
            _collect_action_path_components(action, paths)

    # Goal states
    goals = memory.get("goal_state") or inner.get("goal_state") or []
    if isinstance(goals, list):
        for goal in goals:
            if isinstance(goal, dict):
                for p in extract_path_components_without_object(goal):
                    paths.append(p)

    return paths

def _collect_action_path_components(action: Dict[str, Any], paths: List[Tuple[str, ...]]):
    if not isinstance(action, dict):
        return

    for prec in action.get("preconditions") or []:
        if isinstance(prec, dict):
            for p in extract_path_components_without_object(prec):
                paths.append(p)

    for list_key in ("changes", "changes_per_cycle", "changes_total"):
        for ch in action.get(list_key) or []:
            if isinstance(ch, dict):
                for p in extract_path_components_without_object(ch):
                    paths.append(p)

    for cond in action.get("conditional_changes") or []:
        if isinstance(cond, dict):
            for ch in cond.get("changes") or []:
                if isinstance(ch, dict):
                    for p in extract_path_components_without_object(ch):
                        paths.append(p)

    for sub in action.get("sub_actions") or []:
        if isinstance(sub, dict):
            _collect_action_path_components(sub, paths)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("Loading memories...")
    memories = load_memory_records()

    unique_memories = {}
    for mem in memories:
        mid = get_memory_id(mem)
        if mid not in unique_memories:
            unique_memories[mid] = mem
    memories = list(unique_memories.values())
    total_memories = len(memories)
    print(f"Processing {total_memories} unique memories.")

    occurrence_counts = {}
    memory_sets = {}

    for memory in memories:
        memory_id = get_memory_id(memory)
        all_paths = collect_all_path_components(memory)

        for path_tuple in all_paths:
            for comp in path_tuple:
                if not comp or is_numeric_component(comp):
                    continue
                occurrence_counts[comp] = occurrence_counts.get(comp, 0) + 1
                memory_sets.setdefault(comp, set()).add(memory_id)

    total_component_occurrences = sum(occurrence_counts.values())

    rows = []
    for comp, occ_count in occurrence_counts.items():
        mem_count = len(memory_sets[comp])
        occ_pct = (occ_count / total_component_occurrences * 100) if total_component_occurrences else 0.0
        mem_pct = (mem_count / total_memories * 100) if total_memories else 0.0
        rows.append({
            "attribute": comp,
            "occurrence_count": occ_count,
            "occurrence_percentage": round(occ_pct, 6),
            "memory_count": mem_count,
            "memory_percentage": round(mem_pct, 6),
        })

    rows.sort(key=lambda x: (-x["occurrence_count"], x["attribute"]))

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "total_memories": total_memories,
            "total_component_occurrences": total_component_occurrences,
            "attributes": rows,
        }, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_READABLE, "w", encoding="utf-8") as f:
        f.write("ATTRIBUTE COMPONENT FREQUENCIES\n")
        f.write("=" * 90 + "\n")
        f.write(f"Total memories: {total_memories}\n")
        f.write(f"Total component occurrences: {total_component_occurrences}\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Attribute':40s} {'Occurrences':>12s} {'Occ%':>10s} {'Memories':>10s} {'Mem%':>10s}\n")
        f.write("-" * 90 + "\n")
        for row in rows:
            f.write(f"{row['attribute']:40s} {row['occurrence_count']:>12d} {row['occurrence_percentage']:>9.6f}% {row['memory_count']:>10d} {row['memory_percentage']:>9.6f}%\n")

    print(f"✅ Readable frequencies saved to:\n   {OUTPUT_READABLE}")
    print(f"✅ JSON frequencies saved to:\n   {OUTPUT_JSON}")

if __name__ == "__main__":
    main()