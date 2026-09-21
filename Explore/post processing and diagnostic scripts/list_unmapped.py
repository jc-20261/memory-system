#!/usr/bin/env python3
r"""
list_unmapped_spatial_relations.py

Reads:
    Explore\data\spatial_reenconding_output.json

Finds spatial relation records that have no final re-encoding.

A record is considered unmapped when:
- no manual_reencoded value
- and no valid reencoding_by_index value for its llm_index

Outputs:
    Explore\unmapped_spatial_relations.txt
    Explore\data\unmapped_spatial_relations.json

No memory files are modified.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
INPUT_JSON = EXPLORE_DIR / "data" / "spatial_reenconding_output.json"

READABLE_OUTPUT = EXPLORE_DIR / "unmapped_spatial_relations.txt"
JSON_OUTPUT = EXPLORE_DIR / "data" / "unmapped_spatial_relations.json"


# =============================================================================
# Load
# =============================================================================
def load_output(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Re-encoding output must be a JSON object")

    return data


def get_final_reencoded_value(
    record: Dict[str, Any],
    reencoding_by_index: Dict[str, str],
) -> Optional[str]:
    """
    Return the final re-encoded value for a record.

    Returns None if the record has no valid final re-encoding.
    """
    if "manual_reencoded" in record:
        manual = record.get("manual_reencoded")
        if manual is not None:
            return str(manual)

    llm_index = record.get("llm_index")
    if llm_index is not None:
        value = reencoding_by_index.get(str(llm_index))
        if value is not None:
            return str(value)

    return None


# =============================================================================
# Main
# =============================================================================
def main():
    if not INPUT_JSON.exists():
        print(f"❌ Input file not found: {INPUT_JSON}")
        print("Run llm_spatial_reenconding_extract.py first.")
        return

    print("📂 Loading re-encoding output...")
    data = load_output(INPUT_JSON)

    records = data.get("records")
    reencoding_by_index = data.get("reencoding_by_index")

    if not isinstance(records, list) or not isinstance(reencoding_by_index, dict):
        print("❌ Unexpected JSON structure.")
        return

    print(f"✅ Loaded {len(records)} spatial relation records.")
    print(f"✅ Loaded {len(reencoding_by_index)} LLM re-encoding rules.")

    unmapped = []

    for rec in records:
        final_value = get_final_reencoded_value(rec, reencoding_by_index)

        if final_value is None:
            unmapped.append(
                {
                    "memory_id": rec.get("memory_id", "?"),
                    "object": rec.get("object", ""),
                    "relation": rec.get("relation", ""),
                    "relative_to": rec.get("relative_to", ""),
                    "path": rec.get("path", ""),
                }
            )

    print(f"\n🔎 Found {len(unmapped)} unmapped spatial relation record(s).")

    # Readable output
    with open(READABLE_OUTPUT, "w", encoding="utf-8") as f:
        f.write("UNMAPPED SPATIAL RELATIONS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total unmapped occurrences: {len(unmapped)}\n")
        f.write("=" * 80 + "\n\n")

        if not unmapped:
            f.write("No unmapped spatial relations found.\n")
        else:
            for rec in unmapped:
                f.write(f"Memory ID: {rec['memory_id']}\n")
                f.write(f"Object    : {rec['object']}\n")
                f.write(f"Relation  : {rec['relation']}\n")
                f.write(f"Relative To: {rec['relative_to']}\n")
                f.write(f"Path      : {rec['path']}\n")
                f.write("\n")

    # JSON output
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(unmapped, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Readable output saved to:\n   {READABLE_OUTPUT}")
    print(f"📄 JSON output saved to:\n   {JSON_OUTPUT}")


if __name__ == "__main__":
    main()