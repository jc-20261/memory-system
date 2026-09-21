#!/usr/bin/env python3
r"""
apply_goal_states_from_text.py

Reads manually reviewed goal states from:
    Explore\goal_states_input.txt

Parses single-line goal states such as:

    browser_01.filter_mode=allowlist_active
    person_01.sensory_perception_smell.sensation=salty_earthy_fried_potato_aroma
    bird_feeder_01.contents.volume=2.5 cup
    some_object.attribute != some_value

Validates them against the actual memory JSON and adds:

    "goal_state": [
        {
            "object": "browser_01",
            "attribute": "filter_mode",
            "value": "allowlist_active",
            "op": "eq"
        }
    ]

This script modifies memory files in:
    F:\New folder (4)\New folder\Memories

Back up that folder before running.
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
GOAL_STATES_INPUT = EXPLORE_DIR / "goal_states_input.txt"


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
# Goal-state text parsing
# =============================================================================
def parse_goal_states_text(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Goal states input file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    result = defaultdict(list)
    current_memory_id = None

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line or set(line) <= {"-", "="}:
            continue

        if line.upper().startswith("ACTIVITY:"):
            current_memory_id = None
            continue

        if line.startswith("Memory ID:"):
            current_memory_id = line.split("Memory ID:")[1].strip()
            continue

        if current_memory_id is None:
            continue

        clean_line = line.lstrip("-• \t")
        if not clean_line:
            continue

        if "!=" in clean_line:
            separator = "!="
        elif "=" in clean_line:
            separator = "="
        else:
            continue

        left, right = clean_line.split(separator, 1)
        goal_string = f"{left.strip()}{separator}{right.strip()}"

        if goal_string:
            result[current_memory_id].append(goal_string)

    return dict(result)


def parse_goal_state_string(goal_string: str) -> Optional[Dict[str, Any]]:
    if "!=" in goal_string:
        separator = "!="
        op = "ne"
    elif "=" in goal_string:
        separator = "="
        op = "eq"
    else:
        return None

    left, right = goal_string.split(separator, 1)
    object_and_attr = left.strip()
    value = right.strip()

    if "." not in object_and_attr:
        return None

    object_id, attribute_path = object_and_attr.split(".", 1)

    if not object_id or not attribute_path or not value:
        return None

    return {
        "object": object_id,
        "attribute_path": attribute_path,
        "value": value,
        "op": op,
    }


# =============================================================================
# Value normalization
# =============================================================================
def normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        s = value.strip()

        if len(s) >= 2 and s[0] in {'"', "'"} and s[-1] == s[0]:
            s = s[1:-1].strip()

        low = s.lower()

        if low in {"true", "false"}:
            return low == "true"

        if low in {"null", "none"}:
            return None

        try:
            return int(s)
        except ValueError:
            pass

        try:
            return float(s)
        except ValueError:
            pass

        return s

    if isinstance(value, list):
        return [normalize_value(x) for x in value]

    if isinstance(value, dict):
        return {
            str(k): normalize_value(v)
            for k, v in value.items()
        }

    return value


def canonical_value(value: Any) -> str:
    return json.dumps(normalize_value(value), sort_keys=True, ensure_ascii=False)


def clean_path_component(value: Any) -> str:
    s = str(value).strip()
    if len(s) >= 2 and s[0] in {"'", '"'} and s[-1] == s[0]:
        s = s[1:-1]
    return s


# =============================================================================
# Memory path/value extraction
# =============================================================================
def get_memory_objects(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    inner = memory.get("memory") or {}
    objects = inner.get("objects") or []
    return [obj for obj in objects if isinstance(obj, dict)]


def get_memory_actions(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    inner = memory.get("memory") or {}
    actions = inner.get("actions") or []
    return [act for act in actions if isinstance(act, dict)]


def add_component_variants(variants: List[List[str]], component: Any):
    if component is None:
        return

    clean = clean_path_component(component)
    if not clean:
        return

    new_variants = []
    for variant in variants:
        new_variants.append(variant + [clean])
        if "." in clean:
            new_variants.append(variant + clean.split("."))

    variants.clear()
    variants.extend(new_variants)


def change_path_variants(obj_id: str, ch: Dict[str, Any]) -> List[List[str]]:
    variants = [[obj_id]]

    if "attribute_path" in ch and ch.get("attribute_path") is not None:
        attr_path = ch["attribute_path"]

        if isinstance(attr_path, list):
            for comp in attr_path:
                add_component_variants(variants, comp)
        else:
            add_component_variants(variants, attr_path)
    else:
        aspect = ch.get("aspect")
        if isinstance(aspect, list):
            for a in aspect:
                add_component_variants(variants, a)
        elif aspect is not None:
            add_component_variants(variants, aspect)

        attribute = ch.get("attribute")
        if isinstance(attribute, list):
            for a in attribute:
                add_component_variants(variants, a)
        elif attribute is not None:
            add_component_variants(variants, attribute)

        sub_attribute = ch.get("sub_attribute")
        if isinstance(sub_attribute, list):
            for a in sub_attribute:
                add_component_variants(variants, a)
        elif sub_attribute is not None:
            add_component_variants(variants, sub_attribute)

    return variants


def path_variant_to_string(parts: List[str]) -> str:
    return ".".join(clean_path_component(p) for p in parts if clean_path_component(p))


def value_variants(actual_value: Any) -> List[Any]:
    variants = [actual_value]

    if isinstance(actual_value, dict):
        if "value" in actual_value and "unit" in actual_value:
            val = actual_value.get("value")
            unit = actual_value.get("unit")
            combined = f"{val} {unit}"
            variants.append(combined)
            variants.append(f"{val}{unit}")

    return variants


def expand_sensory_list_entries(
    path: str,
    value: Any,
    source: str,
    path_to_entries: Dict[str, List[Dict[str, Any]]],
):
    if not isinstance(value, list):
        return

    for item in value:
        if not isinstance(item, dict):
            continue

        for key, item_value in item.items():
            child_path = f"{path}.{clean_path_component(key)}"
            add_single_entry(child_path, item_value, source, path_to_entries)


def add_single_entry(
    path: str,
    actual_value: Any,
    source: str,
    path_to_entries: Dict[str, List[Dict[str, Any]]],
):
    for v in value_variants(actual_value):
        path_to_entries[path].append(
            {
                "actual_value": actual_value,
                "canonical": canonical_value(v),
                "source": source,
            }
        )

    expand_sensory_list_entries(path, actual_value, source, path_to_entries)


def collect_memory_paths_and_values(
    memory: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    path_to_entries = defaultdict(list)

    # ----- Object instance attributes -----
    for obj in get_memory_objects(memory):
        obj_id = obj.get("obj_id")
        if not obj_id:
            continue
        obj_id = str(obj_id)

        def flatten_object(node: Dict[str, Any], parent_key: str):
            for key, value in node.items():
                if parent_key:
                    full_key = f"{parent_key}.{key}"
                else:
                    full_key = key

                if isinstance(value, dict):
                    # Tolerant position-dict check
                    if "relation" in value and "relative_to" in value:
                        path = f"{obj_id}.{full_key}"
                        add_single_entry(path, value, "object", path_to_entries)

                        for sub_key, sub_val in value.items():
                            add_single_entry(
                                f"{path}.{sub_key}",
                                sub_val,
                                "object",
                                path_to_entries,
                            )
                    else:
                        flatten_object(value, full_key)
                elif isinstance(value, list):
                    path = f"{obj_id}.{full_key}"
                    add_single_entry(path, value, "object", path_to_entries)
                else:
                    path = f"{obj_id}.{full_key}"
                    add_single_entry(path, value, "object", path_to_entries)

        for key, value in obj.items():
            if key == "obj_id":
                continue

            if isinstance(value, dict):
                flatten_object(value, key)
            else:
                add_single_entry(f"{obj_id}.{key}", value, "object", path_to_entries)

    # ----- Actions, preconditions, changes, conditional changes -----
    def process_condition_or_change(ch: Dict[str, Any], source: str):
        obj_id = ch.get("object")
        if obj_id is None:
            return

        obj_id = str(obj_id)

        for variant_parts in change_path_variants(obj_id, ch):
            path = path_variant_to_string(variant_parts)

            if "value" in ch:
                add_single_entry(path, ch.get("value"), source, path_to_entries)

            for value_key in ("old", "new", "final"):
                if value_key not in ch:
                    continue

                actual_value = ch.get(value_key)
                add_single_entry(path, actual_value, value_key, path_to_entries)

                # Tolerant position-dict check
                if (
                    isinstance(actual_value, dict)
                    and "relation" in actual_value
                    and "relative_to" in actual_value
                ):
                    for sub_key, sub_val in actual_value.items():
                        add_single_entry(
                            f"{path}.{sub_key}",
                            sub_val,
                            value_key,
                            path_to_entries,
                        )

                if (
                    isinstance(actual_value, str)
                    and (path.endswith(".position") or path == f"{obj_id}.position")
                ):
                    add_single_entry(
                        f"{path}.relative_to",
                        actual_value,
                        value_key,
                        path_to_entries,
                    )

            # Direct relative_to in the change itself
            if "relative_to" in ch and ch.get("relative_to") is not None:
                add_single_entry(
                    f"{path}.relative_to",
                    ch.get("relative_to"),
                    source,
                    path_to_entries,
                )

    def walk_action(action: Dict[str, Any]):
        for prec in action.get("preconditions") or []:
            if isinstance(prec, dict):
                process_condition_or_change(prec, "condition")

        change_lists = [
            action.get("changes"),
            action.get("changes_per_cycle"),
            action.get("changes_total"),
        ]

        for change_list in change_lists:
            for ch in change_list or []:
                if isinstance(ch, dict):
                    process_condition_or_change(ch, "change")

        for cond in action.get("conditional_changes") or []:
            if isinstance(cond, dict):
                for ch in cond.get("changes") or []:
                    if isinstance(ch, dict):
                        process_condition_or_change(ch, "change")

        for sub in action.get("sub_actions") or []:
            if isinstance(sub, dict):
                walk_action(sub)

    for action in get_memory_actions(memory):
        walk_action(action)

    return path_to_entries


def validate_goal_state(
    goal: Dict[str, Any],
    path_to_entries: Dict[str, List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    obj_id = goal.get("object")
    attr_path = goal.get("attribute_path")
    input_value = goal.get("value")
    op = goal.get("op", "eq")

    if not obj_id or not attr_path or input_value is None:
        return None

    obj_id = str(obj_id)
    attr_path = clean_path_component(attr_path)

    target_paths = [
        f"{obj_id}.{attr_path}",
        f"{obj_id}.{attr_path.replace(chr(34), '').replace(chr(39), '')}",
    ]

    input_canonical = canonical_value(input_value)

    for target_path in target_paths:
        entries = path_to_entries.get(target_path)
        if not entries:
            continue

        if op == "eq":
            matching_entries = [
                entry for entry in entries if entry["canonical"] == input_canonical
            ]
        else:
            matching_entries = [
                entry for entry in entries if entry["canonical"] != input_canonical
            ]

        if matching_entries:
            source_rank = {"new": 0, "final": 1, "old": 2, "object": 3, "condition": 4}
            best = min(matching_entries, key=lambda e: source_rank.get(e["source"], 10))

            return {
                "object": obj_id,
                "attribute": attr_path,
                "value": best["actual_value"],
                "op": op,
            }

    # Flexible suffix matching
    for path, entries in path_to_entries.items():
        path_parts = path.split(".")
        if len(path_parts) < 2:
            continue

        suffix = ".".join(path_parts[1:])
        if suffix == attr_path or suffix == attr_path.replace('"', '').replace("'", ''):
            if op == "eq":
                matching_entries = [
                    entry for entry in entries if entry["canonical"] == input_canonical
                ]
            else:
                matching_entries = [
                    entry for entry in entries if entry["canonical"] != input_canonical
                ]

            if matching_entries:
                source_rank = {"new": 0, "final": 1, "old": 2, "object": 3, "condition": 4}
                best = min(matching_entries, key=lambda e: source_rank.get(e["source"], 10))

                return {
                    "object": path_parts[0],
                    "attribute": suffix,
                    "value": best["actual_value"],
                    "op": op,
                }

    return None


# =============================================================================
# Goal-state duplicate check
# =============================================================================
def goal_state_exists(existing_goals: List[Dict[str, Any]], new_goal: Dict[str, Any]) -> bool:
    if not isinstance(existing_goals, list):
        return False

    new_key = (
        str(new_goal.get("object")),
        str(new_goal.get("attribute")),
        canonical_value(new_goal.get("value")),
        str(new_goal.get("op", "eq")),
    )

    for existing in existing_goals:
        if not isinstance(existing, dict):
            continue

        old_key = (
            str(existing.get("object")),
            str(existing.get("attribute")),
            canonical_value(existing.get("value")),
            str(existing.get("op", "eq")),
        )

        if new_key == old_key:
            return True

    return False


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Parsing goal states input...")
    goal_states_by_memory = parse_goal_states_text(GOAL_STATES_INPUT)
    print(f"✅ Parsed goal states for {len(goal_states_by_memory)} memories.")

    total_goals = sum(len(v) for v in goal_states_by_memory.values())
    print(f"   Total goal state entries: {total_goals}")

    print("📂 Loading memory files...")
    memory_records = load_memory_records()
    print(f"✅ Loaded {len(memory_records)} memory records.")

    memory_by_id = {}
    for rec in memory_records:
        memory_id = get_memory_id(rec["memory"])
        memory_by_id[memory_id] = rec

    added_count = 0
    skipped_existing_count = 0
    missing_count = 0

    for memory_id, goal_strings in goal_states_by_memory.items():
        target_record = memory_by_id.get(memory_id)

        if target_record is None:
            print(f"❌ Memory ID not found: {memory_id}")
            missing_count += len(goal_strings)
            continue

        memory = target_record["memory"]

        print(f"\n🧠 Processing {memory_id}...")

        path_to_entries = collect_memory_paths_and_values(memory)

        existing_goals = memory.get("goal_state") or []
        if not isinstance(existing_goals, list):
            existing_goals = []

        validated_goals = []
        current_goal_keys = set()

        for goal_string in goal_strings:
            parsed_goal = parse_goal_state_string(goal_string)

            if parsed_goal is None:
                print(f"   ⚠️ Skipping unparseable goal line: {goal_string}")
                missing_count += 1
                continue

            validated_goal = validate_goal_state(parsed_goal, path_to_entries)

            if validated_goal is None:
                print(f"   ❌ Could not validate: {goal_string}")
                missing_count += 1
                continue

            if goal_state_exists(existing_goals, validated_goal):
                print(f"   ⏭️ Already exists, skipping: {goal_string}")
                skipped_existing_count += 1
                continue

            goal_key = (
                validated_goal["object"],
                validated_goal["attribute"],
                canonical_value(validated_goal["value"]),
                validated_goal["op"],
            )
            if goal_key in current_goal_keys:
                print(f"   ⏭️ Duplicate in current run, skipping: {goal_string}")
                skipped_existing_count += 1
                continue

            current_goal_keys.add(goal_key)
            validated_goals.append(validated_goal)

        combined_goals = list(existing_goals)
        combined_goals.extend(validated_goals)

        memory["goal_state"] = combined_goals

        added_count += len(validated_goals)

        print(
            f"   ✅ Added {len(validated_goals)} new goal state(s). "
            f"Existing preserved: {len(existing_goals)}"
        )

    print("\n💾 Writing updated memory files...")
    write_memory_records(memory_records)

    print(f"\n✅ Total new goal states added: {added_count}")
    print(f"⏭️ Total goal states skipped because already present: {skipped_existing_count}")
    print(f"⚠️ Total goal states that could not be validated: {missing_count}")

    print(f"📂 Updated memory files in:\n   {MEMORIES_DIR}")


if __name__ == "__main__":
    main()