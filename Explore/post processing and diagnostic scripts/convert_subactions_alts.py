#!/usr/bin/env python3
r"""
convert_subactions_to_alt_names_v2.py

Converts sub_actions entries in action_storage_linked.json so that:

- If a sub-action name refers to a general template with exactly 1 alt,
  replace it with that alt name.
- If a general template has more than 1 alt, keep the general template name.
- If the entry is already an alt name, keep it.
- If the entry is not found as either a general or alt name, keep it and
  report it.

Also reports any general template with no alts.

Input:
    Explore\data\action_storage_linked.json

Outputs:
    Explore\data\action_storage_linked.json   (updated)
    Explore\subaction_to_alt_conversion_report.txt
    Explore\data\subaction_to_alt_conversion_report.json

No memory files are modified.
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

ACTION_STORAGE_LINKED_PATH = DATA_DIR / "action_storage_linked.json"

REPORT_TEXT_OUTPUT = EXPLORE_DIR / "subaction_to_alt_conversion_report.txt"
REPORT_JSON_OUTPUT = DATA_DIR / "subaction_to_alt_conversion_report.json"


# =============================================================================
# Load / save
# =============================================================================
def load_action_storage_linked(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Action storage linked file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("action_storage_linked.json must be a JSON object")

    return data


def save_action_storage_linked(data: Dict[str, Any], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved {path}")


# =============================================================================
# Build mappings
# =============================================================================
def build_mappings(action_storage: Dict[str, Any]) -> Tuple[Dict[str, List[str]], Set[str], Set[str]]:
    """
    Build:
    - general_to_alts: general template name -> list of alt names
    - all_alt_names: set of all alt names
    - all_general_names: set of all general template names
    """
    general_to_alts = {}
    all_alt_names = set()
    all_general_names = set(action_storage.keys())

    for general_name, entry in action_storage.items():
        if not isinstance(entry, dict):
            continue

        alts = entry.get("alts", {})
        if isinstance(alts, dict):
            alt_names = list(alts.keys())
            general_to_alts[general_name] = alt_names
            all_alt_names.update(alt_names)
        else:
            general_to_alts[general_name] = []

    return general_to_alts, all_alt_names, all_general_names


# =============================================================================
# Conversion
# =============================================================================
def convert_subactions_to_alt_names(
    action_storage: Dict[str, Any],
    general_to_alts: Dict[str, List[str]],
    all_alt_names: Set[str],
    all_general_names: Set[str],
) -> Tuple[int, List[Dict[str, Any]], List[str]]:
    """
    Convert sub_actions entries.

    Returns:
        converted_count: number of sub_action entries replaced with alt name
        unresolved_entries: list of unresolved sub_action references
        generals_without_alts: list of general templates with no alts
    """
    converted_count = 0
    unresolved_entries = []
    generals_without_alts = []

    for general_name, entry in action_storage.items():
        if not isinstance(entry, dict):
            continue

        alts = entry.get("alts", {})
        if not isinstance(alts, dict):
            generals_without_alts.append(general_name)
            continue

        if not alts:
            generals_without_alts.append(general_name)

        for alt_name, alt_entry in alts.items():
            if not isinstance(alt_entry, dict):
                continue

            sub_actions = alt_entry.get("sub_actions")
            if not isinstance(sub_actions, list):
                continue

            new_sub_actions = []

            for sub_action in sub_actions:
                sub_action_str = str(sub_action).strip()

                if sub_action_str in all_alt_names:
                    # Already an alt name
                    new_sub_actions.append(sub_action_str)
                elif sub_action_str in all_general_names:
                    alts_of_sub = general_to_alts.get(sub_action_str, [])

                    if len(alts_of_sub) == 1:
                        new_sub_actions.append(alts_of_sub[0])
                        converted_count += 1
                    else:
                        # More than 1 alt, keep general template name
                        new_sub_actions.append(sub_action_str)
                else:
                    # Not found, keep and report
                    new_sub_actions.append(sub_action_str)
                    unresolved_entries.append(
                        {
                            "sub_action": sub_action_str,
                            "parent_general": general_name,
                            "parent_alt": alt_name,
                        }
                    )

            alt_entry["sub_actions"] = new_sub_actions

    return converted_count, unresolved_entries, generals_without_alts


# =============================================================================
# Output
# =============================================================================
def write_report(
    converted_count: int,
    unresolved_entries: List[Dict[str, Any]],
    generals_without_alts: List[str],
    readable_path: Path,
    json_path: Path,
):
    report_data = {
        "converted_count": converted_count,
        "unresolved_sub_actions": unresolved_entries,
        "generals_without_alts": generals_without_alts,
    }

    with open(readable_path, "w", encoding="utf-8") as f:
        f.write("SUB-ACTION TO ALT CONVERSION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Converted sub-action references: {converted_count}\n")
        f.write(f"Unresolved sub-action references: {len(unresolved_entries)}\n")
        f.write(f"General templates without alts: {len(generals_without_alts)}\n")
        f.write("=" * 80 + "\n\n")

        if unresolved_entries:
            f.write("UNRESOLVED SUB-ACTIONS\n")
            f.write("-" * 80 + "\n")
            for item in unresolved_entries:
                f.write(f"  sub_action: {item['sub_action']}\n")
                f.write(f"  parent_general: {item['parent_general']}\n")
                f.write(f"  parent_alt: {item['parent_alt']}\n")
                f.write("-" * 60 + "\n")

        if generals_without_alts:
            f.write("\nGENERAL TEMPLATES WITHOUT ALTS\n")
            f.write("-" * 80 + "\n")
            for general_name in generals_without_alts:
                f.write(f"  - {general_name}\n")

    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"📄 Conversion report saved to:\n   {readable_path}")
    print(f"📄 JSON report saved to:\n   {json_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Loading linked action storage...")
    action_storage = load_action_storage_linked(ACTION_STORAGE_LINKED_PATH)

    print("🔍 Building mappings...")
    general_to_alts, all_alt_names, all_general_names = build_mappings(action_storage)

    print(f"   General templates: {len(all_general_names)}")
    print(f"   Alt names: {len(all_alt_names)}")

    print("🔄 Converting sub-action references...")
    converted_count, unresolved_entries, generals_without_alts = convert_subactions_to_alt_names(
        action_storage,
        general_to_alts,
        all_alt_names,
        all_general_names,
    )

    print(f"   Converted: {converted_count}")
    print(f"   Unresolved: {len(unresolved_entries)}")
    print(f"   Generals without alts: {len(generals_without_alts)}")

    save_action_storage_linked(action_storage, ACTION_STORAGE_LINKED_PATH)

    write_report(
        converted_count,
        unresolved_entries,
        generals_without_alts,
        REPORT_TEXT_OUTPUT,
        REPORT_JSON_OUTPUT,
    )

    print("✅ Done.")


if __name__ == "__main__":
    main()