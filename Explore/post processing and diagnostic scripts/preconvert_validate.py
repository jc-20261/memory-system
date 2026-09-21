#!/usr/bin/env python3
r"""
pre_conversion_check.py

Checks the current memory files and storage files for missing or broken links
before JSON-to-pickle conversion.

Checks:
- Every action template name used in memories exists in action storage.
- Every object template name used in memories exists in object storage.
- Every goal state object resolves to an object instance, template,
  or the memory protagonist.
- Object aliases are checked for:
    - duplicate aliases across different object instances
    - duplicate aliases within one object's alias list
    - empty/whitespace aliases
    - non-string aliases
    - alias conflicts with another object's obj_id
    - alias matching an object template name (informational)

Inputs:
    F:\New folder (4)\New folder\Memories
    Explore\data\object_storage.json
    Explore\data\action_storage_linked.json   (fallback: action_storage.json)

Outputs:
    Explore\pre_conversion_check.txt
    Explore\data\pre_conversion_check.json

No files are modified.
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

OBJECT_STORAGE_PATH = DATA_DIR / "object_storage.json"
ACTION_STORAGE_LINKED_PATH = DATA_DIR / "action_storage_linked.json"
ACTION_STORAGE_PATH = DATA_DIR / "action_storage.json"

READABLE_OUTPUT = EXPLORE_DIR / "pre_conversion_check.txt"
JSON_OUTPUT = DATA_DIR / "pre_conversion_check.json"


# =============================================================================
# Load memories
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
# Load storages
# =============================================================================
def load_object_storage(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def load_action_storage_for_checking() -> Dict[str, Any]:
    if ACTION_STORAGE_LINKED_PATH.exists():
        with open(ACTION_STORAGE_LINKED_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif ACTION_STORAGE_PATH.exists():
        with open(ACTION_STORAGE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    if not isinstance(data, dict):
        data = {}

    general_templates = {}
    alt_names = set()

    if "templates" in data and isinstance(data["templates"], dict):
        for tname, info in data["templates"].items():
            general_templates[tname] = info
            if isinstance(info, dict):
                alternatives = info.get("alternatives")
                if isinstance(alternatives, list):
                    for a in alternatives:
                        alt_names.add(str(a))
    else:
        for general_name, entry in data.items():
            if not isinstance(entry, dict):
                continue
            general_templates[general_name] = entry
            alts = entry.get("alts", {})
            if isinstance(alts, dict):
                for alt_name in alts.keys():
                    alt_names.add(alt_name)

    all_template_names = set(general_templates.keys()) | alt_names

    return {
        "general_templates": general_templates,
        "alt_names": alt_names,
        "all_template_names": all_template_names,
    }


# =============================================================================
# Memory extraction helpers
# =============================================================================
def get_protagonist(memory: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    inner = memory.get("memory") or {}
    protag = inner.get("protagonist") or memory.get("protagonist")
    return protag if isinstance(protag, dict) else None


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


def collect_action_template_names(actions: List[Dict[str, Any]]) -> Set[str]:
    names = set()

    def walk(action: Dict[str, Any]):
        if not isinstance(action, dict):
            return
        template = action.get("template")
        if template:
            names.add(str(template))

        for sub in action.get("sub_actions") or []:
            if isinstance(sub, dict):
                walk(sub)

    for action in actions:
        walk(action)

    return names


# =============================================================================
# Alias check
# =============================================================================
def check_aliases(
    objects: List[Dict[str, Any]],
    object_template_names: Set[str],
) -> List[Dict[str, Any]]:
    """
    Check instance aliases for problems.

    Returns list of issue dicts.
    """
    issues = []

    alias_to_obj_ids = defaultdict(list)
    all_obj_ids = set()

    for obj in objects:
        obj_id = obj.get("obj_id")
        if not obj_id:
            continue

        obj_id_str = str(obj_id)
        all_obj_ids.add(obj_id_str)

        aliases = obj.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]

        seen_in_this_obj = set()

        for alias in aliases:
            # Non-string alias
            if not isinstance(alias, str):
                issues.append(
                    {
                        "type": "non_string_alias",
                        "object_id": obj_id_str,
                        "alias": str(alias),
                    }
                )
                continue

            alias_str = alias.strip()

            # Empty/whitespace alias
            if not alias_str:
                issues.append(
                    {
                        "type": "empty_alias",
                        "object_id": obj_id_str,
                        "alias": alias,
                    }
                )
                continue

            # Duplicate within same object
            if alias_str in seen_in_this_obj:
                issues.append(
                    {
                        "type": "duplicate_alias_in_object",
                        "object_id": obj_id_str,
                        "alias": alias_str,
                    }
                )
                continue

            seen_in_this_obj.add(alias_str)
            alias_to_obj_ids[alias_str].append(obj_id_str)

    # Duplicate alias across different objects
    for alias_str, obj_ids in alias_to_obj_ids.items():
        unique_obj_ids = set(obj_ids)
        if len(unique_obj_ids) > 1:
            issues.append(
                {
                    "type": "duplicate_alias_across_objects",
                    "alias": alias_str,
                    "object_ids": sorted(unique_obj_ids),
                }
            )

    # Alias equal to another object's obj_id
    for alias_str, obj_ids in alias_to_obj_ids.items():
        if alias_str in all_obj_ids and alias_str not in obj_ids:
            issues.append(
                {
                    "type": "alias_conflicts_with_obj_id",
                    "alias": alias_str,
                    "object_ids": sorted(obj_ids),
                }
            )

    # Alias equal to object template name (informational)
    for alias_str, obj_ids in alias_to_obj_ids.items():
        if alias_str in object_template_names:
            issues.append(
                {
                    "type": "alias_matches_object_template",
                    "alias": alias_str,
                    "object_ids": sorted(obj_ids),
                }
            )

    return issues


# =============================================================================
# Main check
# =============================================================================
def check_memory(
    memory: Dict[str, Any],
    object_storage: Dict[str, Any],
    action_storage: Dict[str, Any],
) -> Dict[str, Any]:
    memory_id = get_memory_id(memory)

    protagonist = get_protagonist(memory)
    objects = get_memory_objects(memory)
    actions = get_memory_actions(memory)
    goals = get_goal_states(memory)

    # Valid object identifiers
    obj_ids = {obj.get("obj_id") for obj in objects if obj.get("obj_id")}
    obj_templates = {obj.get("template") for obj in objects if obj.get("template")}

    if protagonist:
        protag_id = protagonist.get("obj_id")
        if protag_id:
            obj_ids.add(protag_id)

        protag_template = protagonist.get("template")
        if protag_template:
            obj_templates.add(protag_template)

    object_template_names = set(object_storage.keys())

    # Action template check
    missing_actions = []
    all_template_names = action_storage.get("all_template_names", set())
    for name in sorted(collect_action_template_names(actions)):
        if name not in all_template_names:
            missing_actions.append(name)

    # Object template check
    missing_objects = []
    for template in sorted(obj_templates):
        if template not in object_template_names:
            missing_objects.append(template)

    # Goal state object check
    unresolved_goals = []
    for goal in goals:
        goal_object = goal.get("object")
        if not goal_object:
            unresolved_goals.append(
                {
                    "object": None,
                    "attribute": goal.get("attribute"),
                    "value": goal.get("value"),
                }
            )
            continue

        goal_object_str = str(goal_object)
        if (
            goal_object_str in obj_ids
            or goal_object_str in obj_templates
            or goal_object_str in object_template_names
        ):
            continue

        unresolved_goals.append(
            {
                "object": goal_object_str,
                "attribute": goal.get("attribute"),
                "value": goal.get("value"),
            }
        )

    # Alias check
    alias_issues = check_aliases(objects, object_template_names)

    return {
        "memory_id": memory_id,
        "missing_action_templates": missing_actions,
        "missing_object_templates": missing_objects,
        "unresolved_goal_objects": unresolved_goals,
        "alias_issues": alias_issues,
    }


# =============================================================================
# Output
# =============================================================================
def write_outputs(all_reports: List[Dict[str, Any]], readable_path: Path, json_path: Path):
    total_missing_actions = sum(len(r["missing_action_templates"]) for r in all_reports)
    total_missing_objects = sum(len(r["missing_object_templates"]) for r in all_reports)
    total_unresolved_goals = sum(len(r["unresolved_goal_objects"]) for r in all_reports)
    total_alias_issues = sum(len(r["alias_issues"]) for r in all_reports)

    with open(readable_path, "w", encoding="utf-8") as f:
        f.write("PRE-CONVERSION CHECK REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Memories checked: {len(all_reports)}\n")
        f.write(f"Missing action templates: {total_missing_actions}\n")
        f.write(f"Missing object templates: {total_missing_objects}\n")
        f.write(f"Unresolved goal objects: {total_unresolved_goals}\n")
        f.write(f"Alias issues: {total_alias_issues}\n")
        f.write("=" * 80 + "\n\n")

        for report in all_reports:
            if (
                not report["missing_action_templates"]
                and not report["missing_object_templates"]
                and not report["unresolved_goal_objects"]
                and not report["alias_issues"]
            ):
                continue

            f.write(f"Memory ID: {report['memory_id']}\n")
            f.write("-" * 60 + "\n")

            if report["missing_action_templates"]:
                f.write("  Missing action templates:\n")
                for name in report["missing_action_templates"]:
                    f.write(f"    - {name}\n")

            if report["missing_object_templates"]:
                f.write("  Missing object templates:\n")
                for name in report["missing_object_templates"]:
                    f.write(f"    - {name}\n")

            if report["unresolved_goal_objects"]:
                f.write("  Unresolved goal objects:\n")
                for g in report["unresolved_goal_objects"]:
                    f.write(
                        f"    - object={g.get('object')}, "
                        f"attribute={g.get('attribute')}, "
                        f"value={g.get('value')}\n"
                    )

            if report["alias_issues"]:
                f.write("  Alias issues:\n")
                for a in report["alias_issues"]:
                    f.write(
                        f"    - type={a.get('type')}, "
                        f"alias={a.get('alias')}, "
                        f"object_ids={a.get('object_ids')}\n"
                    )

            f.write("\n")

    json_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_memories": len(all_reports),
        "total_missing_action_templates": total_missing_actions,
        "total_missing_object_templates": total_missing_objects,
        "total_unresolved_goal_objects": total_unresolved_goals,
        "total_alias_issues": total_alias_issues,
        "reports": all_reports,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"📄 Check report saved to:\n   {readable_path}")
    print(f"📄 JSON report saved to:\n   {json_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Loading object storage...")
    object_storage = load_object_storage(OBJECT_STORAGE_PATH)
    print(f"   Loaded {len(object_storage)} object templates")

    print("📂 Loading action storage...")
    action_storage = load_action_storage_for_checking()
    print(
        f"   Loaded {len(action_storage['general_templates'])} general templates "
        f"and {len(action_storage['alt_names'])} alt names"
    )

    print("📂 Loading memories...")
    memories = load_memory_records()
    print(f"   Loaded {len(memories)} memories")

    print("🔍 Running pre-conversion checks...")
    reports = []

    for memory in memories:
        report = check_memory(memory, object_storage, action_storage)
        reports.append(report)

    write_outputs(reports, READABLE_OUTPUT, JSON_OUTPUT)

    print("✅ Pre-conversion check completed.")


if __name__ == "__main__":
    main()