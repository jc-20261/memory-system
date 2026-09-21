#!/usr/bin/env python3
r"""
find_encoding_issues.py

Scans memory encodings and finds examples of the structural issues that
caused errors in earlier count scripts:

1. `aspect` field in a precondition/change is a dict or list (instead of string).
2. `attribute` field in a precondition/change is a dict or list.
3. `value`, `old`, or `new` fields are lists or dicts (causing unhashable
   value errors when added to a set).

For each issue type, the script prints a count and up to N examples
(memory_id, action template, and the relevant entry content).

Inputs:
    F:\New folder (4)\New folder\Memories

Outputs:
    Explore\encoding_issues_report.txt
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
OUTPUT_FILE = EXPLORE_DIR / "encoding_issues_report.txt"

MAX_EXAMPLES_PER_TYPE = 10  # how many examples to show per issue type

# ----------------------------------------------------------------------
# Memory loading (deduplicated)
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
# Issue collection
# ----------------------------------------------------------------------
def collect_issues(memories: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns a dict with keys:
      'aspect_dict_or_list'
      'attribute_dict_or_list'
      'value_list_or_dict'
    Each value is a list of dicts with keys:
      memory_id, activity, action_template, entry, field_name
    """
    issues = {
        "aspect_dict_or_list": [],
        "attribute_dict_or_list": [],
        "value_list_or_dict": [],
    }

    def process_entry(entry: Dict[str, Any], action_template: str, memory_id: str, activity: str):
        aspect = entry.get("aspect")
        attribute = entry.get("attribute")
        # Check aspect
        if isinstance(aspect, (dict, list)):
            issues["aspect_dict_or_list"].append({
                "memory_id": memory_id,
                "activity": activity,
                "action_template": action_template,
                "entry": entry,
                "field_name": "aspect",
            })
        # Check attribute
        if isinstance(attribute, (dict, list)):
            issues["attribute_dict_or_list"].append({
                "memory_id": memory_id,
                "activity": activity,
                "action_template": action_template,
                "entry": entry,
                "field_name": "attribute",
            })
        # Check value/old/new for list or dict
        for value_key in ("value", "old", "new"):
            val = entry.get(value_key)
            if isinstance(val, (list, dict)):
                issues["value_list_or_dict"].append({
                    "memory_id": memory_id,
                    "activity": activity,
                    "action_template": action_template,
                    "entry": entry,
                    "field_name": value_key,
                })

    def walk_action(action: Dict[str, Any], memory_id: str, activity: str):
        if not isinstance(action, dict):
            return
        template = action.get("template", "")
        # Preconditions
        for prec in action.get("preconditions") or []:
            if isinstance(prec, dict):
                process_entry(prec, template, memory_id, activity)
        # Changes lists
        for list_key in ("changes", "changes_per_cycle", "changes_total"):
            for ch in action.get(list_key) or []:
                if isinstance(ch, dict):
                    process_entry(ch, template, memory_id, activity)
        # Conditional changes
        for cond in action.get("conditional_changes") or []:
            if isinstance(cond, dict):
                for ch in cond.get("changes") or []:
                    if isinstance(ch, dict):
                        process_entry(ch, template, memory_id, activity)
        # Sub-actions
        for sub in action.get("sub_actions") or []:
            if isinstance(sub, dict):
                walk_action(sub, memory_id, activity)

    for memory in memories:
        memory_id = get_memory_id(memory)
        activity = get_activity(memory)
        inner = memory.get("memory") or memory
        actions = inner.get("actions") or []
        for action in actions:
            if isinstance(action, dict):
                walk_action(action, memory_id, activity)

    return issues

# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def write_report(issues: Dict[str, List[Dict[str, Any]]], output_path: Path, total_memories: int):
    lines = []
    lines.append("ENCODING ISSUES REPORT\n")
    lines.append("=" * 90)
    lines.append(f"Total memories scanned: {total_memories}")

    for issue_type, issue_list in issues.items():
        lines.append("\n" + "-" * 90)
        lines.append(f"ISSUE TYPE: {issue_type}")
        lines.append(f"Occurrences: {len(issue_list)}")
        lines.append("-" * 90)

        # Show up to MAX_EXAMPLES_PER_TYPE examples
        for i, item in enumerate(issue_list[:MAX_EXAMPLES_PER_TYPE], 1):
            lines.append(f"\nExample {i}:")
            lines.append(f"  Memory ID    : {item['memory_id']}")
            lines.append(f"  Activity     : {item['activity']}")
            lines.append(f"  Action       : {item['action_template']}")
            lines.append(f"  Field        : {item['field_name']}")
            # Show a compact JSON of the entry (truncate if too long)
            entry_json = json.dumps(item['entry'], indent=2, ensure_ascii=False)
            if len(entry_json) > 600:
                entry_json = entry_json[:600] + "... [truncated]"
            lines.append(f"  Entry snippet:")
            for line in entry_json.splitlines():
                lines.append(f"      {line}")

        if len(issue_list) > MAX_EXAMPLES_PER_TYPE:
            lines.append(f"\n  ... and {len(issue_list) - MAX_EXAMPLES_PER_TYPE} more occurrences.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"📄 Report saved to:\n   {output_path}")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("Loading memory records...")
    memories = load_memory_records()
    print(f"Loaded {len(memories)} raw records.")

    # Deduplicate by memory_id
    unique_memories = {}
    for mem in memories:
        mid = get_memory_id(mem)
        if mid not in unique_memories:
            unique_memories[mid] = mem
    memories = list(unique_memories.values())
    print(f"Deduplicated to {len(memories)} unique memories.")

    print("Scanning for encoding issues...")
    issues = collect_issues(memories)
    for issue_type, issue_list in issues.items():
        print(f"  {issue_type}: {len(issue_list)} occurrences")

    write_report(issues, OUTPUT_FILE, len(memories))

if __name__ == "__main__":
    main()