#!/usr/bin/env python3
r"""
count_alts_per_general_template.py

Counts how many alts each general action template has in
action_storage_linked.json.

Outputs:
    Explore\alt_count_summary.txt
    Explore\data\alt_count_summary.json

No files are modified.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPLORE_DIR / "data"

ACTION_STORAGE_LINKED_PATH = DATA_DIR / "action_storage_linked.json"

READABLE_OUTPUT = EXPLORE_DIR / "alt_count_summary.txt"
JSON_OUTPUT = DATA_DIR / "alt_count_summary.json"


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
# Count
# =============================================================================
def count_alts_per_general(action_storage: Dict[str, Any]) -> Dict[str, Any]:
    alt_count_to_general = {}

    for general_name, entry in action_storage.items():
        if not isinstance(entry, dict):
            continue

        alts = entry.get("alts", {})
        if not isinstance(alts, dict):
            alts = {}

        alt_count = len(alts)

        alt_count_to_general.setdefault(alt_count, []).append(general_name)

    summary = {}

    for alt_count in sorted(alt_count_to_general.keys()):
        general_names = sorted(alt_count_to_general[alt_count])
        summary[str(alt_count)] = {
            "general_template_count": len(general_names),
            "general_templates": general_names,
        }

    return summary


# =============================================================================
# Output
# =============================================================================
def write_outputs(summary: Dict[str, Any], readable_path: Path, json_path: Path):
    with open(readable_path, "w", encoding="utf-8") as f:
        f.write("ALT COUNT PER GENERAL TEMPLATE SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Alt Count':>10s} {'General Templates':>20s}\n")
        f.write("-" * 80 + "\n")

        for alt_count_str, info in summary.items():
            count = info["general_template_count"]
            f.write(f"{alt_count_str:>10s} {count:>20d}\n")

        f.write("-" * 80 + "\n")
        f.write(f"Total alt-count groups: {len(summary)}\n")

        total_general_templates = sum(
            info["general_template_count"] for info in summary.values()
        )
        f.write(f"Total general templates: {total_general_templates}\n")

        # Detailed listing for alt counts > 1
        f.write("\n\nDETAILED LISTING FOR GENERAL TEMPLATES WITH MULTIPLE ALTS\n")
        f.write("=" * 80 + "\n\n")

        for alt_count_str in sorted(summary.keys(), key=lambda x: int(x)):
            alt_count = int(alt_count_str)
            if alt_count <= 1:
                continue

            info = summary[alt_count_str]
            f.write(f"\nALT COUNT: {alt_count}\n")
            f.write("-" * 60 + "\n")
            for general_name in info["general_templates"]:
                f.write(f"  - {general_name}\n")

    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"📄 Alt count summary saved to:\n   {readable_path}")
    print(f"📄 JSON summary saved to:\n   {json_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Loading linked action storage...")
    action_storage = load_action_storage_linked(ACTION_STORAGE_LINKED_PATH)

    print("🔍 Counting alts per general template...")
    summary = count_alts_per_general(action_storage)

    total_general_templates = sum(
        info["general_template_count"] for info in summary.values()
    )
    print(f"   Total general templates: {total_general_templates}")

    for alt_count in sorted(summary.keys(), key=lambda x: int(x)):
        info = summary[alt_count]
        print(f"   {alt_count} alt(s): {info['general_template_count']} templates")

    write_outputs(summary, READABLE_OUTPUT, JSON_OUTPUT)

    print("✅ Done.")


if __name__ == "__main__":
    main()