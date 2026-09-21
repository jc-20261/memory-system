#!/usr/bin/env python3
r"""
add_missing_atomic_templates_with_multiple_instances.py

Special-case script for missing action templates that appear in multiple
memory instances but are atomic and identical across all instances.

For each missing template:
- If it appears exactly once, skip it. The previous script handles those.
- If it appears multiple times:
    - Check if every instance is atomic (no sub_actions) and has identical
      duration, action_category, and kinematic_trajectory.
    - If yes:
        - Treat it as an atomic general template.
        - Use the first instance's fields as general fields.
        - Create an implicit alt: <template_name>_01
        - The alt copies general fields and has "sub_actions": [].
        - general_sub_actions stays empty.
    - Otherwise:
        - Do NOT add the template automatically.
        - Report it to a text file with memory IDs and instance count.

Inputs:
    F:\New folder (4)\New folder\Memories
    Explore\data\action_storage_linked.json

Outputs:
    Explore\data\action_storage_linked.json
    Explore\atomic_multiple_instances_report.txt
    Explore\data\atomic_multiple_instances_report.json

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

REPORT_TEXT_OUTPUT = EXPLORE_DIR / "atomic_multiple_instances_report.txt"
REPORT_JSON_OUTPUT = DATA_DIR / "atomic_multiple_instances_report.json"


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
# Atomic check
# =============================================================================
def is_atomic_action(action: Dict[str, Any]) -> bool:
    sub_actions = action.get("sub_actions") or []
    if not isinstance(sub_actions, list):
        return False
    return len(sub_actions) == 0


def instances_are_identical_atomic(instances: List[Tuple[str, Dict[str, Any]]]) -> bool:
    if not instances:
        return False

    first = instances[0][1]

    if not is_atomic_action(first):
        return False

    first_duration = first.get("duration")
    first_category = first.get("action_category", "HUMAN_INTERACTION")
    first_trajectory = first.get("kinematic_trajectory", "straight")

    for _, action in instances[1:]:
        if not is_atomic_action(action):
            return False

        if action.get("duration") != first_duration:
            return False
        if action.get("action_category", "HUMAN_INTERACTION") != first_category:
            return False
        if action.get("kinematic_trajectory", "straight") != first_trajectory:
            return False

    return True


# =============================================================================
# Add missing atomic templates with multiple instances
# =============================================================================
def add_missing_atomic_multi_instance_templates(
    linked_storage: Dict[str, Any],
    template_instances: Dict[str, List[Tuple[str, Dict[str, Any]]]],
) -> Tuple[int, List[Dict[str, Any]]]:
    existing_names = get_existing_template_names(linked_storage)
    missing_templates = set(template_instances.keys()) - existing_names

    added_count = 0
    reports = []

    for template_name in sorted(missing_templates):
        instances = template_instances.get(template_name, [])

        # Skip single-instance templates; previous script handles them
        if len(instances) == 1:
            continue

        if not instances_are_identical_atomic(instances):
            reports.append(
                {
                    "action_template": template_name,
                    "memory_ids": sorted({mid for mid, _ in instances}),
                    "instance_count": len(instances),
                }
            )
            continue

        # Use first instance fields
        first_memory_id, first_action = instances[0]

        default_duration = first_action.get("duration", 1)
        default_category = first_action.get("action_category", "HUMAN_INTERACTION")
        default_trajectory = first_action.get("kinematic_trajectory", "straight")

        try:
            default_duration = float(default_duration)
        except (TypeError, ValueError):
            default_duration = 1.0

        implicit_alt_name = f"{template_name}_01"
        counter = 1
        while implicit_alt_name in existing_names:
            counter += 1
            implicit_alt_name = f"{template_name}_{counter:02d}"

        alt_entry = {
            "verb": "",
            "default_duration": default_duration,
            "action_category": default_category,
            "kinematic_trajectory": default_trajectory,
            "sub_actions": [],
        }

        general_entry = {
            "verb": "",
            "default_duration": default_duration,
            "action_category": default_category,
            "kinematic_trajectory": default_trajectory,
            "general_sub_actions": [],
            "alts": {
                implicit_alt_name: alt_entry,
            },
        }

        linked_storage[template_name] = general_entry
        existing_names.add(template_name)
        existing_names.add(implicit_alt_name)

        added_count += 1

    return added_count, reports


# =============================================================================
# Report output
# =============================================================================
def write_reports(reports: List[Dict[str, Any]]):
    with open(REPORT_TEXT_OUTPUT, "w", encoding="utf-8") as f:
        f.write("MISSING ATOMIC MULTIPLE-INSTANCE TEMPLATES NOT ADDED\n")
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

    with open(REPORT_JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    print(f"📄 Report saved to:\n   {REPORT_TEXT_OUTPUT}")
    print(f"📄 JSON report saved to:\n   {REPORT_JSON_OUTPUT}")


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

    print("➕ Adding missing atomic multi-instance templates...")
    added_count, reports = add_missing_atomic_multi_instance_templates(
        linked_storage,
        template_instances,
    )
    print(f"   Added {added_count} atomic template(s).")
    print(f"   Reported {len(reports)} non-atomic/non-identical template(s).")

    if added_count:
        save_linked_action_storage(linked_storage)

    if reports:
        write_reports(reports)

    print("✅ Done.")


if __name__ == "__main__":
    main()