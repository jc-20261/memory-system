#!/usr/bin/env python3
r"""
check_alt_template_duplication.py

Checks action_storage_linked.json for duplicate/ambiguous alt and general
template names.

Checks:
1. Alt name equals any general template name.
2. Alt name duplicated across multiple general templates.
3. General template name also used as an alt elsewhere.
4. General templates with no alts.

Outputs:
    Explore\alt_template_duplication_check.txt
    Explore\data\alt_template_duplication_check.json

No files are modified.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPLORE_DIR / "data"

ACTION_STORAGE_LINKED_PATH = DATA_DIR / "action_storage_linked.json"

READABLE_OUTPUT = EXPLORE_DIR / "alt_template_duplication_check.txt"
JSON_OUTPUT = DATA_DIR / "alt_template_duplication_check.json"


# =============================================================================
# Load
# =============================================================================
def load_action_storage_linked(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Action storage linked file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("action_storage_linked.json must be a JSON object")

    return data


# =============================================================================
# Checks
# =============================================================================
def check_alt_template_duplication(action_storage: Dict[str, Any]) -> Dict[str, Any]:
    general_names = set(action_storage.keys())

    alt_to_generals = defaultdict(list)
    alt_name_equals_general = []
    general_name_also_used_as_alt = []
    generals_without_alts = []

    # Build alt mapping
    for general_name, entry in action_storage.items():
        if not isinstance(entry, dict):
            continue

        alts = entry.get("alts", {})
        if not isinstance(alts, dict) or not alts:
            generals_without_alts.append(general_name)
            continue

        for alt_name in alts.keys():
            alt_to_generals[alt_name].append(general_name)

            # Check if alt name equals any general template name
            if alt_name in general_names:
                alt_name_equals_general.append(
                    {
                        "alt_name": alt_name,
                        "parent_general": general_name,
                        "matched_general": alt_name,
                    }
                )

    # Check for alt duplicated across generals
    alt_name_duplicated_across_generals = []
    for alt_name, generals in alt_to_generals.items():
        if len(generals) > 1:
            alt_name_duplicated_across_generals.append(
                {
                    "alt_name": alt_name,
                    "general_templates": generals,
                }
            )

    # Check if general name appears as an alt anywhere
    for general_name in general_names:
        if general_name in alt_to_generals:
            general_name_also_used_as_alt.append(
                {
                    "name": general_name,
                    "used_as_general": True,
                    "used_as_alt_under": alt_to_generals[general_name],
                }
            )

    return {
        "alt_name_equals_general": alt_name_equals_general,
        "alt_name_duplicated_across_generals": alt_name_duplicated_across_generals,
        "general_name_also_used_as_alt": general_name_also_used_as_alt,
        "generals_without_alts": generals_without_alts,
    }


# =============================================================================
# Output
# =============================================================================
def write_output(results: Dict[str, Any], readable_path: Path, json_path: Path):
    with open(readable_path, "w", encoding="utf-8") as f:
        f.write("ALT / GENERAL TEMPLATE DUPLICATION CHECK\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. ALT NAME EQUALS GENERAL NAME\n")
        f.write("-" * 80 + "\n")
        if results["alt_name_equals_general"]:
            for item in results["alt_name_equals_general"]:
                f.write(
                    f"  alt={item['alt_name']}, parent_general={item['parent_general']}, "
                    f"matched_general={item['matched_general']}\n"
                )
        else:
            f.write("  None found.\n")

        f.write("\n2. ALT NAME DUPLICATED ACROSS GENERAL TEMPLATES\n")
        f.write("-" * 80 + "\n")
        if results["alt_name_duplicated_across_generals"]:
            for item in results["alt_name_duplicated_across_generals"]:
                f.write(f"  alt={item['alt_name']}, generals={item['general_templates']}\n")
        else:
            f.write("  None found.\n")

        f.write("\n3. GENERAL NAME ALSO USED AS ALT\n")
        f.write("-" * 80 + "\n")
        if results["general_name_also_used_as_alt"]:
            for item in results["general_name_also_used_as_alt"]:
                f.write(
                    f"  name={item['name']}, used_as_alt_under={item['used_as_alt_under']}\n"
                )
        else:
            f.write("  None found.\n")

        f.write("\n4. GENERAL TEMPLATES WITHOUT ALTS\n")
        f.write("-" * 80 + "\n")
        if results["generals_without_alts"]:
            for general_name in results["generals_without_alts"]:
                f.write(f"  - {general_name}\n")
        else:
            f.write("  None found.\n")

    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"📄 Duplication check saved to:\n   {readable_path}")
    print(f"📄 JSON check saved to:\n   {json_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Loading linked action storage...")
    action_storage = load_action_storage_linked(ACTION_STORAGE_LINKED_PATH)

    print("🔍 Checking alt/general template duplication...")
    results = check_alt_template_duplication(action_storage)

    print(f"   alt_name_equals_general: {len(results['alt_name_equals_general'])}")
    print(
        f"   alt_name_duplicated_across_generals: "
        f"{len(results['alt_name_duplicated_across_generals'])}"
    )
    print(
        f"   general_name_also_used_as_alt: "
        f"{len(results['general_name_also_used_as_alt'])}"
    )
    print(f"   generals_without_alts: {len(results['generals_without_alts'])}")

    write_output(results, READABLE_OUTPUT, JSON_OUTPUT)

    print("✅ Done.")


if __name__ == "__main__":
    main()