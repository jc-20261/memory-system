#!/usr/bin/env python3
r"""
add_missing_action_templates_v5.py

Scans memory files for action template names that are not present in
action_storage_linked.json and adds them based on the confirmed logic.

For each missing template:
- If it appears in exactly one memory instance:
    - If the memory instance has an "alt" field, use that alt name.
    - Otherwise, create an implicit alt: <general_name>_01
    - The alt copies general fields and receives the instance's sub_actions.
    - general_sub_actions stays empty.
- If it appears in more than one memory instance:
    - Do NOT add it automatically.
    - Report it to a text file with memory IDs.

Inputs:
    F:\New folder (4)\New folder\Memories
    Explore\data\action_storage_linked.json

Outputs:
    Explore\data\action_storage_linked.json             (updated)
    Explore\missing_templates_multiple_instances.txt
    Explore\data\missing_templates_multiple_instances.json

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

ACTION_STORAGE_LINKED_PATH = DATA_DIR / "action_storage_linked.json"

MULTIPLE_INSTANCES_TEXT_OUTPUT = EXPLORE_DIR / "missing_templates_multiple_instances.txt"
MULTIPLE_INSTANCES_JSON_OUTPUT = DATA_DIR / "missing_templates_multiple_instances.json"


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


def get_memory_actions(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    inner = memory.get("memory") or {}
    actions = inner.get("actions") or []
    return [act for act in actions if isinstance(act, dict)]


def collect_template_instances(
    memories: List[Dict[str, Any]],
) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    template_instances = defaultdict(list)

    def walk(action: Dict[str, Any], memory_id: str):
        if not isinstance(action, dict):
            return

        template = action.get("template")
        if template:
            template_instances[str(template)].append((memory_id, action))

        for sub in action.get("sub_actions") or []:
            if isinstance(sub, dict):
                walk(sub, memory_id)

    for memory in memories:
        memory_id = get_memory_id(memory)
        for action in get_memory_actions(memory):
            walk(action, memory_id)

    return dict(template_instances)


# =============================================================================
# Load / save linked action storage
# =============================================================================
def load_linked_action_storage() -> Dict[str, Any]:
    if not ACTION_STORAGE_LINKED_PATH.exists():
        return {}
    with open(ACTION_STORAGE_LINKED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_linked_action_storage(data: Dict[str, Any]):
    with open(ACTION_STORAGE_LINKED_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved {ACTION_STORAGE_LINKED_PATH}")


def get_existing_template_names(action_storage: Dict[str, Any]) -> Set[str]:
    general_names = set(action_storage.keys())
    alt_names = set()

    for entry in action_storage.values():
        if isinstance(entry, dict):
            alts = entry.get("alts", {})
            if isinstance(alts, dict):
                alt_names.update(alts.keys())

    return general_names | alt_names


# =============================================================================
# Helpers
# =============================================================================
def get_direct_sub_action_template_names(action: Dict[str, Any]) -> List[str]:
    sub_actions = action.get("sub_actions") or []
    if not isinstance(sub_actions, list):
        return []

    names = []
    for sub in sub_actions:
        if isinstance(sub, dict):
            template = sub.get("template")
            if template:
                names.append(str(template))

    return names


# =============================================================================
# Add missing templates
# =============================================================================
def add_missing_templates(
    linked_storage: Dict[str, Any],
    template_instances: Dict[str, List[Tuple[str, Dict[str, Any]]]],
) -> Tuple[int, List[Dict[str, Any]]]:
    existing_names = get_existing_template_names(linked_storage)
    missing_templates = set(template_instances.keys()) - existing_names

    added_count = 0
    reports = []

    for template_name in sorted(missing_templates):
        instances = template_instances.get(template_name, [])

        if len(instances) != 1:
            reports.append(
                {
                    "action_template": template_name,
                    "memory_ids": sorted({mid for mid, _ in instances}),
                    "instance_count": len(instances),
                }
            )
            continue

        memory_id, action = instances[0]

        default_duration = action.get("duration", 1)
        default_category = action.get("action_category", "HUMAN_INTERACTION")
        default_trajectory = action.get("kinematic_trajectory", "straight")

        try:
            default_duration = float(default_duration)
        except (TypeError, ValueError):
            default_duration = 1.0

        sub_actions = get_direct_sub_action_template_names(action)

        # Check for explicit alt in the memory instance
        explicit_alt = action.get("alt")

        if explicit_alt and str(explicit_alt).strip():
            alt_name = str(explicit_alt).strip()
        else:
            alt_name = f"{template_name}_01"
            counter = 1
            while alt_name in existing_names:
                counter += 1
                alt_name = f"{template_name}_{counter:02d}"

        alt_entry = {
            "verb": "",
            "default_duration": default_duration,
            "action_category": default_category,
            "kinematic_trajectory": default_trajectory,
            "sub_actions": sub_actions,
        }

        general_entry = {
            "verb": "",
            "default_duration": default_duration,
            "action_category": default_category,
            "kinematic_trajectory": default_trajectory,
            "general_sub_actions": [],
            "alts": {
                alt_name: alt_entry,
            },
        }

        linked_storage[template_name] = general_entry
        existing_names.add(template_name)
        existing_names.add(alt_name)

        added_count += 1

    return added_count, reports


# =============================================================================
# Report output
# =============================================================================
def write_reports(reports: List[Dict[str, Any]]):
    with open(MULTIPLE_INSTANCES_TEXT_OUTPUT, "w", encoding="utf-8") as f:
        f.write("MISSING ACTION TEMPLATES WITH MULTIPLE INSTANCES\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total templates reported: {len(reports)}\n")
        f.write("=" * 80 + "\n\n")

        for rec in reports:
            f.write(f"Action Template: {rec['action_template']}\n")
            f.write(f"Instance count: {rec['instance_count']}\n")
            f.write("Memory IDs:\n")
            for mid in rec["memory_ids"]:
                f.write(f"  - {mid}\n")
            f.write("-" * 60 + "\n")

    with open(MULTIPLE_INSTANCES_JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    print(f"📄 Multiple-instance report saved to:\n   {MULTIPLE_INSTANCES_TEXT_OUTPUT}")
    print(f"📄 JSON report saved to:\n   {MULTIPLE_INSTANCES_JSON_OUTPUT}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Loading memory records...")
    memories = load_memory_records()
    print(f"   Loaded {len(memories)} memories.")

    print("🔍 Collecting action template instances...")
    template_instances = collect_template_instances(memories)
    print(f"   Distinct action templates in memories: {len(template_instances)}")

    print("📂 Loading linked action storage...")
    linked_storage = load_linked_action_storage()
    existing_names = get_existing_template_names(linked_storage)
    print(f"   Existing template names: {len(existing_names)}")

    print("➕ Adding missing templates...")
    added_count, reports = add_missing_templates(linked_storage, template_instances)
    print(f"   Added {added_count} missing template(s).")
    print(f"   Reported {len(reports)} template(s) with multiple instances.")

    if added_count:
        save_linked_action_storage(linked_storage)

    if reports:
        write_reports(reports)

    print("✅ Done.")


if __name__ == "__main__":
    main()