#!/usr/bin/env python3
r"""
add_unresolved_subactions_as_atomic.py

Adds unresolved sub-action names from the conversion report into
action_storage_linked.json as new atomic general templates.

For each unresolved sub-action name:
- Create a new general template with empty/default fields.
- Create one implicit alt <sub_action_name>_01.
- Both general and alt have empty verb, default_duration=null,
  action_category="", kinematic_trajectory="", sub_actions=[].

The new alt name is unique; if _01 exists, increment to _02, etc.

Inputs:
    Explore\data\subaction_to_alt_conversion_report.json
    Explore\data\action_storage_linked.json

Outputs:
    Explore\data\action_storage_linked.json   (updated)
    Explore\added_atomic_subactions_report.txt
    Explore\data\added_atomic_subactions_report.json

No memory files are modified.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPLORE_DIR / "data"

CONVERSION_REPORT_PATH = DATA_DIR / "subaction_to_alt_conversion_report.json"
ACTION_STORAGE_LINKED_PATH = DATA_DIR / "action_storage_linked.json"

READABLE_REPORT = EXPLORE_DIR / "added_atomic_subactions_report.txt"
JSON_REPORT = DATA_DIR / "added_atomic_subactions_report.json"


# =============================================================================
# Load / save
# =============================================================================
def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_json(data: Dict[str, Any], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved {path}")


def get_existing_names(action_storage: Dict[str, Any]) -> Set[str]:
    general_names = set(action_storage.keys())
    alt_names = set()

    for entry in action_storage.values():
        if isinstance(entry, dict):
            alts = entry.get("alts", {})
            if isinstance(alts, dict):
                alt_names.update(alts.keys())

    return general_names | alt_names


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Loading conversion report...")
    report_data = load_json(CONVERSION_REPORT_PATH)

    unresolved_entries = report_data.get("unresolved_sub_actions", [])
    if not isinstance(unresolved_entries, list):
        unresolved_entries = []

    unique_sub_actions = []
    seen = set()

    for item in unresolved_entries:
        sub_action = item.get("sub_action")
        if sub_action and sub_action not in seen:
            seen.add(sub_action)
            unique_sub_actions.append(str(sub_action))

    print(f"   Unique unresolved sub-actions: {len(unique_sub_actions)}")

    if not unique_sub_actions:
        print("   No unresolved sub-actions to add.")
        return

    print("📂 Loading linked action storage...")
    action_storage = load_json(ACTION_STORAGE_LINKED_PATH)

    existing_names = get_existing_names(action_storage)
    print(f"   Existing names before addition: {len(existing_names)}")

    added_records = []

    for sub_action_name in sorted(unique_sub_actions):
        if sub_action_name in existing_names:
            continue

        # Create implicit alt name with suffix
        alt_name = f"{sub_action_name}_01"
        counter = 1
        while alt_name in existing_names:
            counter += 1
            alt_name = f"{sub_action_name}_{counter:02d}"

        alt_entry = {
            "verb": "",
            "default_duration": None,
            "action_category": "",
            "kinematic_trajectory": "",
            "sub_actions": [],
        }

        general_entry = {
            "verb": "",
            "default_duration": None,
            "action_category": "",
            "kinematic_trajectory": "",
            "general_sub_actions": [],
            "alts": {
                alt_name: alt_entry,
            },
        }

        action_storage[sub_action_name] = general_entry
        existing_names.add(sub_action_name)
        existing_names.add(alt_name)

        added_records.append(
            {
                "sub_action_name": sub_action_name,
                "new_general_template": sub_action_name,
                "new_alt_name": alt_name,
            }
        )

    print(f"   Added {len(added_records)} atomic general templates.")

    if added_records:
        save_json(action_storage, ACTION_STORAGE_LINKED_PATH)

        with open(READABLE_REPORT, "w", encoding="utf-8") as f:
            f.write("ADDED ATOMIC SUB-ACTIONS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total added: {len(added_records)}\n")
            f.write("=" * 80 + "\n\n")

            for rec in added_records:
                f.write(f"Sub-action name: {rec['sub_action_name']}\n")
                f.write(f"New general template: {rec['new_general_template']}\n")
                f.write(f"New alt name: {rec['new_alt_name']}\n")
                f.write("-" * 60 + "\n")

        with open(JSON_REPORT, "w", encoding="utf-8") as f:
            json.dump(added_records, f, indent=2, ensure_ascii=False)

        print(f"📄 Report saved to:\n   {READABLE_REPORT}")
        print(f"📄 JSON report saved to:\n   {JSON_REPORT}")

    print("✅ Done.")


if __name__ == "__main__":
    main()