#!/usr/bin/env python3
r"""
convert_to_pickle.py – Convert memory JSON files to pickle runtime objects.

Reads memory files, object storage, linked action storage, and verb storage,
then builds runtime class instances and saves each memory as pickle.

Outputs:
    Explore\pickle_memories\<memory_id>.pkl
    Explore\pickle_storage\object_storage.pkl
    Explore\pickle_storage\action_storage_linked.pkl
    Explore\pickle_storage\verb_storage.pkl
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memory_model import (
    Memory,
    Object,
    Protagonist,
    GoalState,
    VerbTemplate,
    ActionGeneralTemplate,
    ActionAlt,
    ActionInstance,
)

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
DATA_DIR = EXPLORE_DIR / "data"

OBJECT_STORAGE_PATH = DATA_DIR / "object_storage.json"
ACTION_STORAGE_LINKED_PATH = DATA_DIR / "action_storage_linked.json"
VERB_STORAGE_PATH = DATA_DIR / "verb_storage.json"

PICKLE_MEMORIES_DIR = EXPLORE_DIR / "pickle_memories"
PICKLE_STORAGE_DIR = EXPLORE_DIR / "pickle_storage"

# =============================================================================
# Load helpers
# =============================================================================
def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


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


# =============================================================================
# Build storage runtime objects
# =============================================================================
def build_object_storage_runtime(object_storage_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    # For now, we keep object templates as plain dicts; later can wrap in classes if needed.
    return object_storage_json


def build_action_and_verb_runtime(
    action_storage_linked: Dict[str, Any],
    verb_storage_json: Dict[str, Any],
) -> Tuple[Dict[str, ActionGeneralTemplate], Dict[str, ActionAlt], Dict[str, VerbTemplate]]:
    """
    Build runtime action storage structures.

    Returns:
        general_templates: dict name -> ActionGeneralTemplate
        alts: dict name -> ActionAlt
        verbs: dict name -> VerbTemplate
    """
    # Create VerbTemplate objects
    verbs: Dict[str, VerbTemplate] = {}
    verb_templates_data = verb_storage_json.get("verb_templates", {})
    for verb_name in verb_templates_data.keys():
        verbs[verb_name] = VerbTemplate(verb_name)

    general_templates: Dict[str, ActionGeneralTemplate] = {}
    alts: Dict[str, ActionAlt] = {}

    for general_name, entry in action_storage_linked.items():
        if not isinstance(entry, dict):
            continue

        verb_name = entry.get("verb", "")
        verb_obj = verbs.get(verb_name, VerbTemplate(verb_name)) if verb_name else None
        if verb_name and verb_name not in verbs:
            verbs[verb_name] = verb_obj

        general = ActionGeneralTemplate(
            name=general_name,
            verb=verb_obj,
            default_duration=entry.get("default_duration", 1.0),
            action_category=entry.get("action_category", ""),
            kinematic_trajectory=entry.get("kinematic_trajectory", ""),
            general_sub_actions=entry.get("general_sub_actions", []),
        )
        general_templates[general_name] = general

        alts_data = entry.get("alts", {})
        for alt_name, alt_entry in alts_data.items():
            if not isinstance(alt_entry, dict):
                continue
            alt_verb_name = alt_entry.get("verb", "")
            alt_verb_obj = verbs.get(alt_verb_name, VerbTemplate(alt_verb_name)) if alt_verb_name else None
            if alt_verb_name and alt_verb_name not in verbs:
                verbs[alt_verb_name] = alt_verb_obj

            alt = ActionAlt(
                name=alt_name,
                parent_general=general,
                verb=alt_verb_obj or verb_obj,
                default_duration=alt_entry.get("default_duration", general.default_duration),
                action_category=alt_entry.get("action_category", general.action_category),
                kinematic_trajectory=alt_entry.get("kinematic_trajectory", general.kinematic_trajectory),
                sub_actions=alt_entry.get("sub_actions", []),
            )
            alts[alt_name] = alt
            general.add_alt(alt)

        if verb_obj:
            verb_obj.add_template(general)

    return general_templates, alts, verbs


# =============================================================================
# Conversion helpers
# =============================================================================
def resolve_alt_for_action(
    template_name: str,
    action_instance_json: Dict[str, Any],
    general_templates: Dict[str, ActionGeneralTemplate],
    alts: Dict[str, ActionAlt],
) -> Tuple[ActionAlt, ActionGeneralTemplate]:
    """
    Determine the correct alt and general template for an action instance.
    """
    # If template_name is an alt name
    if template_name in alts:
        alt = alts[template_name]
        return alt, alt.parent_general

    # If template_name is a general template name
    if template_name in general_templates:
        general = general_templates[template_name]
        if len(general.alts) == 1:
            return general.alts[0], general

        # Try to match sub-action sequence
        instance_sub_templates = [
            str(sub.get("template", ""))
            for sub in action_instance_json.get("sub_actions", [])
        ]
        for alt in general.alts:
            if alt.sub_actions == instance_sub_templates:
                return alt, general
        # Fallback: first alt
        return general.alts[0], general

    # Unknown template: create placeholder
    # In practice this shouldn't happen after pre-checks.
    general = ActionGeneralTemplate(name=template_name)
    alt = ActionAlt(name=f"{template_name}_01", parent_general=general)
    general.add_alt(alt)
    return alt, general


def build_object_from_json(
    obj_data: Dict[str, Any],
    object_storage_runtime: Dict[str, Any],
) -> Object:
    """Create an Object from memory JSON object dict."""
    template = obj_data.get("template")
    obj_id = obj_data.get("obj_id")
    if not template or not obj_id:
        raise ValueError("Object missing template or obj_id")

    template_defaults = object_storage_runtime.get(template, {}).copy()
    # Remove metadata keys from defaults
    for key in ("categories", "functions", "materials", "aliases"):
        template_defaults.pop(key, None)

    categories = object_storage_runtime.get(template, {}).get("categories", [])
    materials = object_storage_runtime.get(template, {}).get("materials", [])
    functions = obj_data.get("functions", object_storage_runtime.get(template, {}).get("functions", []))

    aliases = obj_data.get("aliases", object_storage_runtime.get(template, {}).get("aliases", []))

    return Object(
        template_name=template,
        obj_id=obj_id,
        number=obj_data.get("number", "single"),
        position=obj_data.get("position"),
        position_relative_body=obj_data.get("position_relative_body"),
        dimensions=obj_data.get("dimensions"),
        weight=obj_data.get("weight") or obj_data.get("mass"),
        mass=obj_data.get("mass") or obj_data.get("weight"),
        shape=obj_data.get("shape", ""),
        motion=obj_data.get("motion"),
        overrides=obj_data.get("overrides", {}),
        aliases=aliases,
        composite_of=obj_data.get("composite_of", []),
        functions=functions,
        template_defaults=template_defaults,
        categories=categories,
        materials=materials,
    )


def build_action_instance_tree(
    action_json: Dict[str, Any],
    memory: Memory,
    general_templates: Dict[str, ActionGeneralTemplate],
    alts: Dict[str, ActionAlt],
    path_parts: List[str],
) -> ActionInstance:
    """
    Recursively build ActionInstance from action JSON.
    """
    template_name = str(action_json.get("template", ""))
    if not template_name:
        raise ValueError("Action missing template")

    alt, general = resolve_alt_for_action(template_name, action_json, general_templates, alts)

    instance_id = f"{memory.id}.{'.'.join(path_parts)}"

    # Resolve participants
    participants = []
    for pid in action_json.get("participants", []):
        if pid in memory.objects:
            participants.append(memory.objects[pid])
        else:
            # Create placeholder
            placeholder = Object(template_name=str(pid), obj_id=str(pid), overrides={"unresolved": True})
            memory.add_object(placeholder)
            participants.append(placeholder)

    # Recursively build sub-actions
    sub_actions = []
    sub_json_list = action_json.get("sub_actions", [])
    for idx, sub_action in enumerate(sub_json_list):
        if isinstance(sub_action, dict):
            sub_instance = build_action_instance_tree(
                sub_action,
                memory,
                general_templates,
                alts,
                path_parts + ["sub_actions", str(idx)],
            )
            sub_actions.append(sub_instance)

    action_instance = ActionInstance(
        instance_id=instance_id,
        template_name=template_name,
        alt=alt,
        general_template=general,
        duration=action_json.get("duration", alt.default_duration),
        start_time=action_json.get("start_time", 0.0),
        end_time=action_json.get("end_time", 0.0),
        effective_duration=action_json.get("effective_duration", action_json.get("duration", alt.default_duration)),
        participants=participants,
        changes=action_json.get("changes", []),
        preconditions=action_json.get("preconditions", []),
        tags=action_json.get("tags", []),
        action_category=action_json.get("action_category", alt.action_category),
        kinematic_trajectory=action_json.get("kinematic_trajectory", alt.kinematic_trajectory),
        temporal_type=action_json.get("temporal_type", ""),
        repetitions=action_json.get("repetitions", 1),
        cycle_duration=action_json.get("cycle_duration", 0.0),
        sub_actions=sub_actions,
        path_parts=path_parts,
    )
    return action_instance


def resolve_goal_state_object(
    goal_object_name: str,
    memory: Memory,
    object_storage_runtime: Dict[str, Any],
) -> Object:
    """Resolve goal state object name to an Object instance."""
    # Direct obj_id match
    if goal_object_name in memory.objects:
        return memory.objects[goal_object_name]

    # Template match in memory
    for obj in memory.objects.values():
        if obj.template_name == goal_object_name:
            return obj

    # Object storage template
    if goal_object_name in object_storage_runtime:
        # Create new placeholder object from template
        template_defaults = object_storage_runtime[goal_object_name].copy()
        for key in ("categories", "functions", "materials", "aliases"):
            template_defaults.pop(key, None)
        obj = Object(
            template_name=goal_object_name,
            obj_id=f"{goal_object_name}_goal_01",
            template_defaults=template_defaults,
            overrides={"unresolved": True},
        )
        memory.add_object(obj)
        return obj

    # Fallback placeholder
    obj = Object(template_name=goal_object_name, obj_id=f"{goal_object_name}_goal_01", overrides={"unresolved": True})
    memory.add_object(obj)
    return obj


# =============================================================================
# Main conversion
# =============================================================================
def main():
    print("📂 Loading storage files...")
    object_storage_json = load_json(OBJECT_STORAGE_PATH)
    action_storage_linked = load_json(ACTION_STORAGE_LINKED_PATH)
    verb_storage_json = load_json(VERB_STORAGE_PATH)

    print("🔧 Building runtime storage objects...")
    object_storage_runtime = build_object_storage_runtime(object_storage_json)
    general_templates, alts, verbs = build_action_and_verb_runtime(action_storage_linked, verb_storage_json)

    print("📂 Loading memory records...")
    memory_records = load_memory_records()
    print(f"   Loaded {len(memory_records)} memories.")

    PICKLE_MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
    PICKLE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    print("🔄 Converting memories to pickle...")
    converted_count = 0
    for memory_json in memory_records:
        memory_id = get_memory_id(memory_json)
        activity = get_activity(memory_json)
        inner = memory_json.get("memory") or memory_json

        # Protagonist
        protag_data = inner.get("protagonist", {})
        protag_obj_id = protag_data.get("obj_id", "person_01")
        person_defaults = object_storage_runtime.get("person", {}).copy()
        for key in ("categories", "functions", "materials", "aliases"):
            person_defaults.pop(key, None)
        protagonist = Protagonist(obj_id=protag_obj_id, template_defaults=person_defaults)
        if "overrides" in protag_data:
            protagonist.overrides.update(protag_data["overrides"])
        if "anatomical_features" in protag_data:
            protagonist.anatomical_features = protag_data["anatomical_features"]

        memory = Memory(memory_id=memory_id, protagonist=protagonist)
        memory.activity = activity

        # Objects
        for obj_data in inner.get("objects", []):
            if isinstance(obj_data, dict):
                try:
                    obj = build_object_from_json(obj_data, object_storage_runtime)
                    memory.add_object(obj)
                except Exception as e:
                    print(f"   ⚠️ Could not build object in {memory_id}: {e}")

        # Actions
        for idx, action_data in enumerate(inner.get("actions", [])):
            if isinstance(action_data, dict):
                try:
                    action_instance = build_action_instance_tree(
                        action_data,
                        memory,
                        general_templates,
                        alts,
                        ["memory", "actions", str(idx)],
                    )
                    memory.actions.append(action_instance)
                except Exception as e:
                    print(f"   ⚠️ Could not build action in {memory_id}: {e}")

        # Goal states
        goal_list = memory_json.get("goal_state") or inner.get("goal_state") or []
        for goal_data in goal_list:
            if isinstance(goal_data, dict):
                goal_obj_name = str(goal_data.get("object", ""))
                if goal_obj_name:
                    target_obj = resolve_goal_state_object(goal_obj_name, memory, object_storage_runtime)
                    memory.goal_states.append(
                        GoalState(
                            target_object=target_obj,
                            attribute=goal_data.get("attribute", ""),
                            value=goal_data.get("value"),
                            aspect=goal_data.get("aspect"),
                            op=goal_data.get("op", "eq"),
                        )
                    )

        # Rebuild alias map
        memory.rebuild_alias_map()

        # Save memory pickle
        out_path = PICKLE_MEMORIES_DIR / f"{memory.id}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(memory, f, protocol=pickle.HIGHEST_PROTOCOL)
        converted_count += 1

    print(f"✅ Converted {converted_count} memories to pickle.")

    # Save storage pickles
    with open(PICKLE_STORAGE_DIR / "object_storage.pkl", "wb") as f:
        pickle.dump(object_storage_runtime, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PICKLE_STORAGE_DIR / "action_storage_linked.pkl", "wb") as f:
        pickle.dump({"general_templates": general_templates, "alts": alts}, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PICKLE_STORAGE_DIR / "verb_storage.pkl", "wb") as f:
        pickle.dump(verbs, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"💾 Saved storage pickles to {PICKLE_STORAGE_DIR}")

if __name__ == "__main__":
    main()