#!/usr/bin/env python3
r"""
count_terms_for_alias_generation.py

Counts the number of unique terms per memory that would be considered for alias generation.

For each memory in:
    F:\New folder (4)\New folder\Memories

The script extracts:
  - isolated terms (object template names, object instance IDs, action templates,
    action categories, trajectories, tags)
  - attribute–value–op triples (path_tuple, op, value)
  - total terms (isolated + triples)

Attribute paths are kept as tuples of components so that dots inside a single
key (e.g., "surface.wetness") are preserved as one component.

Preconditions and state changes are included as triples.
For state changes, "old" and "new" are treated as separate op values ("old" and "new").

Outputs a readable file:
    Explore\term_counts_for_alias_generation.txt
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
OUTPUT_FILE = EXPLORE_DIR / "term_counts_for_alias_generation.txt"

# ----------------------------------------------------------------------
# Stop/unit words and filters
# ----------------------------------------------------------------------
STOP_WORDS = {
    "true", "false", "on", "off", "none", "open", "closed",
    "current", "value", "to", "from", "in", "the", "a", "an",
    "low", "medium", "high", "yes", "no", "0", "1", "and", "or",
    "with", "for", "of", "at", "by", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "shall", "should", "may", "might",
    "must", "can", "could", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "mine", "yours", "hers", "ours", "theirs",
    "category", "action", "template", "object", "trajectory",
    "function", "precondition", "state_change", "goal_state",
    "dimension", "dimensions", "position", "relation",
    "relative_to", "unit", "memory", "objects", "actions",
    "templates", "aliases", "categories", "functions", "materials"
}

UNIT_WORDS = {
    "cm", "m", "mm", "km", "g", "kg", "mg", "ml", "l", "s",
    "sec", "min", "hr", "hour", "hours", "degree", "degrees",
    "percent", "%", "gram", "grams", "meter", "meters",
    "centimeter", "centimeters", "millimeter", "millimeters",
    "liter", "liters", "milliliter", "milliliters", "kilogram",
    "kilograms"
}

def is_simple_value(value: Any) -> bool:
    """Return True if the value is a simple string that could be aliased."""
    if not isinstance(value, str):
        return False
    s = value.strip()
    if not s:
        return False
    if s.lower() in STOP_WORDS or s.lower() in UNIT_WORDS:
        return False
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return False
    return True

def make_hashable(value: Any) -> Any:
    """Return a hashable representation of a value."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value

def combine_value_unit(value: Any, unit: Any) -> str:
    """Combine a numeric value and unit into a single string."""
    if unit is None:
        return str(value)
    return f"{value}{unit}"

# ----------------------------------------------------------------------
# Path component helpers
# ----------------------------------------------------------------------
def collect_path_keys(node: Any, comps: List[str]):
    """
    Recursively collect keys from node (dict or list) into comps.
    Used for aspect/attribute when they are containers.
    """
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
                    # ignore simple list items
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, str):
                comps.append(item)
            elif isinstance(item, dict):
                collect_path_keys(item, comps)

def path_from_aspect_attribute(aspect: Any, attribute: Any) -> List[str]:
    """
    Build a list of path components from aspect and attribute fields.
    Handles string, list, dict, and None.
    """
    comps: List[str] = []
    # Aspect
    if aspect is not None:
        if isinstance(aspect, str):
            comps.append(aspect)
        elif isinstance(aspect, (dict, list)):
            collect_path_keys(aspect, comps)
    # Attribute
    if attribute is not None:
        if isinstance(attribute, str):
            comps.append(attribute)
        elif isinstance(attribute, list):
            # list of strings => sequential components
            for item in attribute:
                if isinstance(item, str):
                    comps.append(item)
                elif isinstance(item, dict):
                    collect_path_keys(item, comps)
        elif isinstance(attribute, dict):
            collect_path_keys(attribute, comps)
    return comps

# ----------------------------------------------------------------------
# Triplet extraction
# ----------------------------------------------------------------------
def expand_value(base_path: Tuple[str, ...], op: str, value: Any):
    """
    Recursively expand a value into one or more (path_tuple, op, leaf_value).
    Handles dicts, lists, and leaves. Dotted keys are preserved as single components.
    """
    if value is None:
        return

    # Check for measurement dict {value, unit}
    if isinstance(value, dict) and "value" in value and set(value.keys()) <= {"value", "unit"}:
        leaf = combine_value_unit(value.get("value"), value.get("unit"))
        yield (base_path, op, leaf)
        return

    if isinstance(value, dict):
        for k, v in value.items():
            new_path = base_path + (str(k),)
            yield from expand_value(new_path, op, v)
    elif isinstance(value, list):
        for item in value:
            # No index in path; each item yields separate triples with same base path
            yield from expand_value(base_path, op, item)
    else:
        # Leaf
        yield (base_path, op, make_hashable(value))

def extract_entry_triples(entry: Dict[str, Any], value_key: str, op_override: str = None) -> List[Tuple[Tuple[str, ...], str, Any]]:
    """
    Extract triples from a precondition, change, or goal state entry.
    value_key: "value", "old", or "new"
    op_override: if provided, use it as op; otherwise use entry['op'] or default
    """
    obj = entry.get("object")
    aspect = entry.get("aspect")
    attribute = entry.get("attribute")
    value = entry.get(value_key)

    if value is None:
        return []

    # Determine op
    if op_override:
        op = op_override
    elif "op" in entry:
        op = str(entry["op"])
    else:
        op = "eq"

    # Build full path components
    base_comps: List[str] = []
    if obj is not None and str(obj).strip():
        base_comps.append(str(obj))

    path_comps = base_comps + path_from_aspect_attribute(aspect, attribute)
    base_path = tuple(path_comps)

    triples = []
    for path_tuple, _, val in expand_value(base_path, op, value):
        triples.append((path_tuple, op, val))
    return triples

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

def get_activity(memory: Dict[str, Any]) -> str:
    act = memory.get("activity")
    if act:
        return str(act)
    inner = memory.get("memory") or {}
    act = inner.get("activity")
    return str(act) if act else "unknown"

# ----------------------------------------------------------------------
# Extraction for one memory
# ----------------------------------------------------------------------
def extract_terms(memory: Dict[str, Any]) -> Tuple[Set[Any], Set[Tuple[Tuple[str, ...], str, Any]]]:
    isolated: Set[Any] = set()
    triples: Set[Tuple[Tuple[str, ...], str, Any]] = set()

    inner = memory.get("memory") or memory

    # ----- Objects -----
    objects = inner.get("objects") or []
    for obj in objects:
        if not isinstance(obj, dict):
            continue

        template = obj.get("template")
        if template:
            isolated.add(f"template:{template}")

        obj_id = obj.get("obj_id")
        if obj_id:
            isolated.add(f"obj_id:{obj_id}")

        # Overrides
        overrides = obj.get("overrides") or {}
        for path, val in flatten_dict_to_components(overrides):
            base_path = tuple(path)
            for t in expand_value(base_path, "eq", val):
                triples.add(t)

        # Other explicit fields
        for field in ("dimensions", "weight", "mass", "position",
                      "position_relative_body", "motion", "shape"):
            if field in obj:
                field_val = obj[field]
                if isinstance(field_val, dict):
                    for path, val in flatten_dict_to_components(field_val):
                        full_path = (field,) + tuple(path)
                        for t in expand_value(full_path, "eq", val):
                            triples.add(t)
                else:
                    triples.add(((field,), "eq", make_hashable(field_val)))

    # ----- Actions -----
    actions = inner.get("actions") or []
    for action in actions:
        if isinstance(action, dict):
            _extract_action_terms(action, isolated, triples)

    # ----- Goal states -----
    goals = memory.get("goal_state") or inner.get("goal_state") or []
    if isinstance(goals, list):
        for goal in goals:
            if isinstance(goal, dict):
                # Treat as precondition-like
                obj = goal.get("object")
                attr = goal.get("attribute")
                val = goal.get("value")
                op = goal.get("op", "eq")
                if obj and attr and val is not None:
                    path = (str(obj), str(attr))
                    for t in expand_value(path, str(op), val):
                        triples.add(t)

    return isolated, triples

def flatten_dict_to_components(d: Dict[str, Any], prefix: Tuple[str, ...] = None) -> List[Tuple[Tuple[str, ...], Any]]:
    """Flatten a dict into (path_tuple, leaf_value) preserving dots in keys."""
    if prefix is None:
        prefix = ()
    result = []
    for k, v in d.items():
        new_prefix = prefix + (str(k),)
        if isinstance(v, dict):
            # Check for measurement dict
            if "value" in v and set(v.keys()) <= {"value", "unit"}:
                result.append((new_prefix, combine_value_unit(v.get("value"), v.get("unit"))))
            else:
                result.extend(flatten_dict_to_components(v, new_prefix))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    result.extend(flatten_dict_to_components(item, new_prefix))
                else:
                    result.append((new_prefix, item))
        else:
            result.append((new_prefix, v))
    return result

def _extract_action_terms(action: Dict[str, Any], isolated: Set[Any], triples: Set[Tuple[Tuple[str, ...], str, Any]]):
    if not isinstance(action, dict):
        return

    template = action.get("template")
    if template:
        isolated.add(f"action:{template}")

    category = action.get("action_category")
    if category:
        isolated.add(f"category:{category}")

    trajectory = action.get("kinematic_trajectory")
    if trajectory:
        isolated.add(f"trajectory:{trajectory}")

    for tag in action.get("tags") or []:
        if isinstance(tag, str):
            isolated.add(f"tag:{tag}")

    # Preconditions
    for prec in action.get("preconditions") or []:
        if isinstance(prec, dict):
            for t in extract_entry_triples(prec, "value"):
                triples.add(t)

    # Changes: old and new separately
    for change_list_key in ("changes", "changes_per_cycle", "changes_total"):
        changes = action.get(change_list_key) or []
        for ch in changes:
            if isinstance(ch, dict):
                for t in extract_entry_triples(ch, "new", op_override="new"):
                    triples.add(t)
                for t in extract_entry_triples(ch, "old", op_override="old"):
                    triples.add(t)

    # Conditional changes (similar)
    for cond in action.get("conditional_changes") or []:
        if isinstance(cond, dict):
            for ch in cond.get("changes") or []:
                if isinstance(ch, dict):
                    for t in extract_entry_triples(ch, "new", op_override="new"):
                        triples.add(t)
                    for t in extract_entry_triples(ch, "old", op_override="old"):
                        triples.add(t)

    # Sub-actions
    for sub in action.get("sub_actions") or []:
        if isinstance(sub, dict):
            _extract_action_terms(sub, isolated, triples)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("Loading memory records...")
    memories = load_memory_records()
    print(f"Loaded {len(memories)} raw memory records.")

    unique_memories = {}
    for mem in memories:
        mid = get_memory_id(mem)
        if mid not in unique_memories:
            unique_memories[mid] = mem
    memories = list(unique_memories.values())
    print(f"Deduplicated to {len(memories)} unique memories.")

    rows = []
    for memory in memories:
        mid = get_memory_id(memory)
        act = get_activity(memory)
        isolated, triples = extract_terms(memory)
        iso_count = len(isolated)
        triple_count = len(triples)
        total = iso_count + triple_count
        rows.append((mid, act, iso_count, triple_count, total))

    rows.sort(key=lambda x: x[0])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("TERM COUNTS FOR ALIAS GENERATION\n")
        f.write("=" * 90 + "\n")
        f.write(f"Total memories: {len(rows)}\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'memory_id':20s} {'activity':40s} {'isolated':>8s} {'triples':>8s} {'total':>8s}\n")
        f.write("-" * 90 + "\n")
        for mid, act, iso, triple, total in rows:
            f.write(f"{mid:20s} {act:40s} {iso:>8d} {triple:>8d} {total:>8d}\n")

    print(f"📄 Term counts saved to:\n   {OUTPUT_FILE}")
    print("\nSample (first 20):")
    for row in rows[:20]:
        print(f"  {row[0]} | {row[1]} | isolated={row[2]} triples={row[3]} total={row[4]}")

if __name__ == "__main__":
    main()