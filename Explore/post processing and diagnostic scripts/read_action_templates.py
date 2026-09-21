#!/usr/bin/env python3
r"""
list_action_templates.py

Reads:
    Explore\data\action_storage.json

Writes:
    Explore\action_templates_readable.txt
    Explore\data\action_templates_summary.json

No action storage files are modified.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
ACTION_STORAGE_PATH = EXPLORE_DIR / "data" / "action_storage.json"

READABLE_OUTPUT = EXPLORE_DIR / "action_templates_readable.txt"
JSON_OUTPUT = EXPLORE_DIR / "data" / "action_templates_summary.json"

# =============================================================================
# Loading
# =============================================================================
def load_action_storage() -> Dict[str, Any]:
    if not ACTION_STORAGE_PATH.exists():
        raise FileNotFoundError(f"Action storage not found: {ACTION_STORAGE_PATH}")

    with open(ACTION_STORAGE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Action storage JSON is not a dict")

    return data


def get_templates(data: Dict[str, Any]) -> Dict[str, Any]:
    templates = data.get("templates")

    if not isinstance(templates, dict):
        # Some versions might store templates directly at top level
        if all(isinstance(v, dict) for v in data.values()):
            return data
        return {}

    return templates


def get_sub_action_sequences(data: Dict[str, Any]) -> Dict[str, List[str]]:
    seqs = data.get("sub_action_sequences")

    if not isinstance(seqs, dict):
        return {}

    return {
        str(k): list(v)
        for k, v in seqs.items()
        if isinstance(v, list)
    }


# =============================================================================
# Formatting
# =============================================================================
def format_field_value(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return "[]"
        return ", ".join(str(x) for x in value)

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)

    if value is None:
        return ""

    return str(value)


def write_readable_action_templates(
    templates: Dict[str, Any],
    sub_action_sequences: Dict[str, List[str]],
    output_path: Path,
):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ACTION TEMPLATES\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total action templates: {len(templates)}\n")
        f.write("=" * 80 + "\n\n")

        for name in sorted(templates.keys()):
            template = templates[name]
            if not isinstance(template, dict):
                template = {}

            f.write(f"TEMPLATE: {name}\n")
            f.write("-" * 80 + "\n")

            # Common fields first for readability
            common_fields = [
                "default_duration",
                "action_category",
                "kinematic_trajectory",
                "temporal_type",
                "tags",
                "alternatives",
                "general_template",
                "preconditions",
                "is_cyclical",
                "default_cycle_duration",
                "max_cycles",
                "repetitions",
            ]

            for field in common_fields:
                if field in template:
                    f.write(f"  {field}: {format_field_value(template[field])}\n")

            # Sub-action sequence if present
            if name in sub_action_sequences:
                seq = sub_action_sequences[name]
                f.write("  sub_action_templates:\n")
                for sub in seq:
                    f.write(f"    - {sub}\n")
            elif "sub_action_templates" in template:
                sub = template["sub_action_templates"]
                f.write("  sub_action_templates:\n")
                if isinstance(sub, list):
                    for s in sub:
                        f.write(f"    - {s}\n")
                else:
                    f.write(f"    {format_field_value(sub)}\n")

            # Remaining fields
            remaining_fields = [
                k for k in template.keys()
                if k not in common_fields
                and k != "sub_action_templates"
            ]

            if remaining_fields:
                f.write("  additional_fields:\n")
                for field in sorted(remaining_fields):
                    f.write(f"    {field}: {format_field_value(template[field])}\n")

            f.write("\n")

    print(f"📄 Readable action templates saved to:\n   {output_path}")


def write_json_summary(
    templates: Dict[str, Any],
    sub_action_sequences: Dict[str, List[str]],
    output_path: Path,
):
    summary = {
        "total_action_templates": len(templates),
        "templates": {},
        "sub_action_sequences": sub_action_sequences,
    }

    for name, template in templates.items():
        if not isinstance(template, dict):
            template = {}
        summary["templates"][name] = template

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"📄 JSON summary saved to:\n   {output_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Loading action storage...")
    data = load_action_storage()

    templates = get_templates(data)
    sub_action_sequences = get_sub_action_sequences(data)

    if not templates:
        print("⚠️ No action templates found.")
        return

    print(f"✅ Loaded {len(templates)} action templates.")

    write_readable_action_templates(
        templates,
        sub_action_sequences,
        READABLE_OUTPUT,
    )

    write_json_summary(
        templates,
        sub_action_sequences,
        JSON_OUTPUT,
    )


if __name__ == "__main__":
    main()