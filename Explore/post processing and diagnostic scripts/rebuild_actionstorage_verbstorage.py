#!/usr/bin/env python3
r"""
build_linked_action_storage_v5.py

Builds verb storage and linked action storage from existing files.

Rules:
- General template keeps its own fields.
- general_sub_actions is empty unless the general template has more than one
  explicit alt.
- The original sub_action_sequences only refer to general template names.
- If a general template has exactly one alt, that alt receives the general
  sub-action sequence; general_sub_actions remains empty.
- If a general template has multiple alts, each alt receives the general
  sub-action sequence, and general_sub_actions also receives that sequence.
- If a general template has no alt and no sub-action sequence, an implicit
  alt is created as <general_name>_01 with empty sub_actions.
- If a general template has no alt but has a sub-action sequence, an implicit
  alt is created with that sequence, and general_sub_actions remains empty.

Inputs:
    Explore\data\action_storage.json
    Explore\data\action_verb_generalization.json

Outputs:
    Explore\data\verb_storage.json
    Explore\data\action_storage_linked.json
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPLORE_DIR / "data"

ACTION_STORAGE_INPUT = DATA_DIR / "action_storage.json"
VERB_MAPPING_INPUT = DATA_DIR / "action_verb_generalization.json"

VERB_STORAGE_OUTPUT = DATA_DIR / "verb_storage.json"
ACTION_STORAGE_LINKED_OUTPUT = DATA_DIR / "action_storage_linked.json"


# =============================================================================
# Loaders
# =============================================================================
def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return data


def load_action_storage(path: Path) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    data = load_json(path)

    templates = data.get("templates", {})
    if not isinstance(templates, dict):
        templates = {}

    sub_action_sequences = data.get("sub_action_sequences", {})
    if not isinstance(sub_action_sequences, dict):
        sub_action_sequences = {}

    normalized_seqs = {}
    for key, seq in sub_action_sequences.items():
        if isinstance(seq, list):
            normalized_seqs[str(key)] = [str(s) for s in seq if s]
        else:
            normalized_seqs[str(key)] = []

    return templates, normalized_seqs


def load_verb_mapping(path: Path) -> Dict[str, str]:
    data = load_json(path)

    action_templates = data.get("action_templates", {})
    if not isinstance(action_templates, dict):
        return {}

    verb_map = {}
    for template_name, info in action_templates.items():
        if isinstance(info, dict):
            verb = info.get("verb")
            if verb:
                verb_map[str(template_name)] = str(verb)

    return verb_map


# =============================================================================
# Relationship building
# =============================================================================
def build_relationships(
    templates: Dict[str, Any],
) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """
    Determine general -> alts and alt -> general from template fields.
    """
    general_to_alts: Dict[str, Set[str]] = defaultdict(set)
    alt_to_general: Dict[str, str] = {}

    for name, info in templates.items():
        if not isinstance(info, dict):
            continue

        alternatives = info.get("alternatives")
        if isinstance(alternatives, list):
            for alt in alternatives:
                alt_str = str(alt)
                general_to_alts[name].add(alt_str)
                alt_to_general[alt_str] = name

        general_template = info.get("general_template")
        if general_template:
            general_str = str(general_template)
            general_to_alts[general_str].add(name)
            alt_to_general[name] = general_str

    # Templates with no relationship become their own general
    for name in templates.keys():
        if name not in alt_to_general and name not in general_to_alts:
            general_to_alts[name] = set()

    return general_to_alts, alt_to_general


# =============================================================================
# Build linked structures
# =============================================================================
def build_linked_structures(
    templates: Dict[str, Any],
    sub_action_sequences: Dict[str, List[str]],
    verb_map: Dict[str, str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    general_to_alts, alt_to_general = build_relationships(templates)

    action_storage_linked: Dict[str, Any] = {}
    verb_to_templates: Dict[str, Set[str]] = defaultdict(set)

    for template_name, verb in verb_map.items():
        if verb:
            verb_to_templates[verb].add(template_name)

    for general_name in sorted(general_to_alts.keys()):
        explicit_alts = set(general_to_alts.get(general_name, set()))
        general_info = templates.get(general_name, {})
        if not isinstance(general_info, dict):
            general_info = {}

        general_verb = verb_map.get(general_name, "")
        general_default_duration = general_info.get("default_duration", 1)
        general_action_category = general_info.get("action_category", "HUMAN_INTERACTION")
        general_kinematic_trajectory = general_info.get("kinematic_trajectory", "straight")

        # Sub-action sequence from original storage, keyed by general name
        general_seq = sub_action_sequences.get(general_name, [])

        alt_count = len(explicit_alts)

        # Determine general_sub_actions
        if alt_count > 1 and general_seq:
            general_sub_actions = list(general_seq)
        else:
            general_sub_actions = []

        alts_output: Dict[str, Any] = {}

        if explicit_alts:
            # For each alt, use the general sequence
            for alt_name in sorted(explicit_alts):
                alt_info = templates.get(alt_name, {})
                if not isinstance(alt_info, dict):
                    alt_info = {}

                alt_verb = verb_map.get(alt_name, general_verb)

                alt_entry = {
                    "verb": alt_verb,
                    "default_duration": alt_info.get(
                        "default_duration", general_default_duration
                    ),
                    "action_category": alt_info.get(
                        "action_category", general_action_category
                    ),
                    "kinematic_trajectory": alt_info.get(
                        "kinematic_trajectory", general_kinematic_trajectory
                    ),
                    "sub_actions": list(general_seq),
                }

                # Copy additional alt-specific fields
                for key, value in alt_info.items():
                    if key in (
                        "alternatives",
                        "general_template",
                        "sub_action_templates",
                        "default_duration",
                        "action_category",
                        "kinematic_trajectory",
                    ):
                        continue
                    alt_entry[key] = value

                alts_output[alt_name] = alt_entry

        else:
            # No explicit alts
            if general_seq:
                # Complex action with no alt: create implicit alt with sequence
                implicit_alt = f"{general_name}_01"
                counter = 1
                while implicit_alt in explicit_alts or implicit_alt in alts_output:
                    counter += 1
                    implicit_alt = f"{general_name}_{counter:02d}"

                implicit_entry = {
                    "verb": general_verb,
                    "default_duration": general_default_duration,
                    "action_category": general_action_category,
                    "kinematic_trajectory": general_kinematic_trajectory,
                    "sub_actions": list(general_seq),
                }

                for key, value in general_info.items():
                    if key in (
                        "alternatives",
                        "general_template",
                        "sub_action_templates",
                        "default_duration",
                        "action_category",
                        "kinematic_trajectory",
                    ):
                        continue
                    implicit_entry[key] = value

                alts_output[implicit_alt] = implicit_entry

            else:
                # Atomic action: create implicit alt with empty sub_actions
                implicit_alt = f"{general_name}_01"
                counter = 1
                while implicit_alt in explicit_alts or implicit_alt in alts_output:
                    counter += 1
                    implicit_alt = f"{general_name}_{counter:02d}"

                implicit_entry = {
                    "verb": general_verb,
                    "default_duration": general_default_duration,
                    "action_category": general_action_category,
                    "kinematic_trajectory": general_kinematic_trajectory,
                    "sub_actions": [],
                }

                for key, value in general_info.items():
                    if key in (
                        "alternatives",
                        "general_template",
                        "sub_action_templates",
                        "default_duration",
                        "action_category",
                        "kinematic_trajectory",
                    ):
                        continue
                    implicit_entry[key] = value

                alts_output[implicit_alt] = implicit_entry

        # Build general entry
        general_entry = {
            "verb": general_verb,
            "default_duration": general_default_duration,
            "action_category": general_action_category,
            "kinematic_trajectory": general_kinematic_trajectory,
            "general_sub_actions": general_sub_actions,
            "alts": alts_output,
        }

        for key, value in general_info.items():
            if key in (
                "alternatives",
                "general_template",
                "sub_action_templates",
                "default_duration",
                "action_category",
                "kinematic_trajectory",
            ):
                continue
            general_entry[key] = value

        action_storage_linked[general_name] = general_entry

    # Build verb storage
    verb_storage: Dict[str, Any] = {"verb_templates": {}}

    for verb, template_names in verb_to_templates.items():
        verb_storage["verb_templates"][verb] = {
            "action_templates": sorted(template_names),
        }

    return verb_storage, action_storage_linked


# =============================================================================
# Save
# =============================================================================
def save_json(data: Dict[str, Any], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved {path}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Loading action storage...")
    templates, sub_action_sequences = load_action_storage(ACTION_STORAGE_INPUT)
    print(f"   Loaded {len(templates)} action templates")
    print(f"   Loaded {len(sub_action_sequences)} sub-action sequences")

    print("📂 Loading verb mapping...")
    verb_map = load_verb_mapping(VERB_MAPPING_INPUT)
    print(f"   Loaded {len(verb_map)} verb mappings")

    print("🔧 Building linked structures...")
    verb_storage, action_storage_linked = build_linked_structures(
        templates,
        sub_action_sequences,
        verb_map,
    )

    save_json(verb_storage, VERB_STORAGE_OUTPUT)
    save_json(action_storage_linked, ACTION_STORAGE_LINKED_OUTPUT)

    print("✅ Done.")


if __name__ == "__main__":
    main()