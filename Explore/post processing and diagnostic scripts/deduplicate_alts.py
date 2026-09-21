#!/usr/bin/env python3
r"""
rename_duplicate_alts.py

Renames duplicate or conflicting alt names in action_storage_linked.json.

Rules:
- If an alt name equals any general template name, rename it by appending
  __01, __02, etc.
- If an alt name is duplicated across multiple general templates, assign
  __01, __02, etc. in the order encountered.
- New alt name format: <original_alt_name>__NN
- If that name already exists, increment NN until available.

This version correctly updates the original dictionary and saves it.

Input:
    Explore\data\action_storage_linked.json

Outputs:
    Explore\data\action_storage_linked.json   (updated)
    Explore\alt_name_rename_report.txt
    Explore\data\alt_name_rename_report.json

No memory files are modified.
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

READABLE_REPORT = EXPLORE_DIR / "alt_name_rename_report.txt"
JSON_REPORT = DATA_DIR / "alt_name_rename_report.json"


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
# Rename logic
# =============================================================================
def make_available_name(
    base_name: str,
    existing_names: set,
    used_names: set,
) -> str:
    counter = 1
    while True:
        candidate = f"{base_name}__{counter:02d}"
        if candidate not in existing_names and candidate not in used_names:
            return candidate
        counter += 1


def rename_duplicate_alts(action_storage: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]]]:
    report = []

    # Working names
    general_names = set(action_storage.keys())

    all_alt_names = set()
    for entry in action_storage.values():
        if isinstance(entry, dict):
            alts = entry.get("alts", {})
            if isinstance(alts, dict):
                all_alt_names.update(alts.keys())

    all_existing_names = general_names | all_alt_names
    used_names = set()

    # ---------- Pass 1: alt equals any general template name ----------
    for general_name in sorted(action_storage.keys()):
        entry = action_storage[general_name]
        if not isinstance(entry, dict):
            continue

        alts = entry.get("alts", {})
        if not isinstance(alts, dict):
            continue

        new_alts = {}

        for alt_name in sorted(alts.keys()):
            alt_entry = alts[alt_name]

            if alt_name in general_names:
                new_alt_name = make_available_name(
                    alt_name,
                    all_existing_names,
                    used_names,
                )
                used_names.add(new_alt_name)
                new_alts[new_alt_name] = alt_entry

                report.append(
                    {
                        "original_alt_name": alt_name,
                        "new_alt_name": new_alt_name,
                        "parent_general": general_name,
                        "reason": "alt_equals_general",
                    }
                )
            else:
                new_alts[alt_name] = alt_entry

        entry["alts"] = new_alts

    # ---------- Pass 2: alt duplicated across multiple general templates ----------
    alt_to_generals = defaultdict(list)
    for general_name, entry in action_storage.items():
        if not isinstance(entry, dict):
            continue
        alts = entry.get("alts", {})
        if isinstance(alts, dict):
            for alt_name in alts.keys():
                alt_to_generals[alt_name].append(general_name)

    # Process only alt names that appear under more than one general
    for alt_name, gen_list in alt_to_generals.items():
        if len(gen_list) <= 1:
            continue

        counter = 1
        for gen_name in gen_list:
            entry = action_storage.get(gen_name)
            if not isinstance(entry, dict):
                continue

            alts = entry.get("alts", {})
            if alt_name not in alts:
                continue

            base_name = alt_name
            new_alt_name = f"{base_name}__{counter:02d}"

            while new_alt_name in all_existing_names or new_alt_name in used_names:
                counter += 1
                new_alt_name = f"{base_name}__{counter:02d}"

            used_names.add(new_alt_name)

            alt_entry = alts[alt_name]
            new_alts = dict(alts)
            del new_alts[alt_name]
            new_alts[new_alt_name] = alt_entry
            entry["alts"] = new_alts

            report.append(
                {
                    "original_alt_name": alt_name,
                    "new_alt_name": new_alt_name,
                    "parent_general": gen_name,
                    "reason": "duplicated_across_generals",
                }
            )
            counter += 1

    return len(report), report


# =============================================================================
# Output
# =============================================================================
def write_report(report: List[Dict[str, Any]], readable_path: Path, json_path: Path):
    with open(readable_path, "w", encoding="utf-8") as f:
        f.write("ALT NAME RENAME REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total alts renamed: {len(report)}\n")
        f.write("=" * 80 + "\n\n")

        for rec in report:
            f.write(f"Original alt name: {rec['original_alt_name']}\n")
            f.write(f"New alt name: {rec['new_alt_name']}\n")
            f.write(f"Parent general: {rec['parent_general']}\n")
            f.write(f"Reason: {rec['reason']}\n")
            f.write("-" * 60 + "\n")

    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"📄 Rename report saved to:\n   {readable_path}")
    print(f"📄 JSON report saved to:\n   {json_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Loading linked action storage...")
    action_storage = load_action_storage_linked(ACTION_STORAGE_LINKED_PATH)

    print("🔧 Renaming duplicate alts...")
    renamed_count, report = rename_duplicate_alts(action_storage)

    print(f"   Renamed {renamed_count} alt(s).")

    if renamed_count:
        save_action_storage_linked(action_storage, ACTION_STORAGE_LINKED_PATH)
        write_report(report, READABLE_REPORT, JSON_REPORT)
    else:
        print("   No duplicate alts found.")

    print("✅ Done.")


if __name__ == "__main__":
    main()