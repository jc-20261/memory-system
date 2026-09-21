#!/usr/bin/env python3
r"""
validate_spatial_reenconding_output.py

Reads:
    Explore\data\spatial_reenconding_output.json

Checks every spatial relation record and verifies whether it has a final
re-encoding.

A final re-encoding exists if either:
- manual_reencoded is not null
- llm_index exists and reencoding_by_index has a matching string key

Prints all records that are still missing a final re-encoding.

Does not modify any files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
INPUT_JSON = EXPLORE_DIR / "data" / "spatial_reenconding_output.json"


# =============================================================================
# Loading and validation
# =============================================================================
def load_output(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected top-level JSON object")

    return data


def get_final_reencoded_value(
    record: Dict[str, Any],
    reencoding_by_index: Dict[str, str],
) -> Optional[str]:
    """Return final re-encoded value or None if missing."""
    manual = record.get("manual_reencoded")
    if manual is not None:
        return str(manual)

    llm_index = record.get("llm_index")
    if llm_index is not None:
        return reencoding_by_index.get(str(llm_index))

    return None


# =============================================================================
# Main
# =============================================================================
def main():
    if not INPUT_JSON.exists():
        print(f"❌ Input file not found: {INPUT_JSON}")
        return

    print("📂 Loading re-encoding output...")
    data = load_output(INPUT_JSON)

    records = data.get("records")
    reencoding_by_index = data.get("reencoding_by_index")

    if not isinstance(records, list) or not isinstance(reencoding_by_index, dict):
        print("❌ Unexpected JSON structure.")
        print("Expected:")
        print('  "records": [ ... ]')
        print('  "reencoding_by_index": { ... }')
        return

    print(f"✅ Loaded {len(records)} records.")
    print(f"✅ Loaded {len(reencoding_by_index)} LLM mapping entries.\n")

    missing_records = []
    type_mismatch_records = []

    for rec in records:
        final_value = get_final_reencoded_value(rec, reencoding_by_index)

        if final_value is None:
            missing_records.append(rec)

            # Check for common type-mismatch error
            llm_index = rec.get("llm_index")
            if llm_index is not None:
                if str(llm_index) not in reencoding_by_index:
                    type_mismatch_records.append(
                        {
                            "memory_id": rec.get("memory_id", "?"),
                            "llm_index": llm_index,
                            "relation": rec.get("relation", ""),
                            "available_keys_like": [
                                k
                                for k in reencoding_by_index.keys()
                                if str(k) == str(llm_index)
                            ],
                        }
                    )

    if not missing_records:
        print("✅ All spatial relation records have a final re-encoding.")
        return

    print(f"❌ Records still missing final re-encoding: {len(missing_records)}\n")

    for rec in missing_records:
        print("Missing record:")
        print(f"  memory_id:   {rec.get('memory_id', '?')}")
        print(f"  object:      {rec.get('object', '')}")
        print(f"  relation:    {rec.get('relation', '')}")
        print(f"  relative_to: {rec.get('relative_to', '')}")
        print(f"  path:        {rec.get('path', '')}")
        print(f"  llm_index:   {rec.get('llm_index')}")
        print(f"  manual_reencoded: {rec.get('manual_reencoded')}")
        print("-" * 70)

    if type_mismatch_records:
        print("\nPossible type/key mismatches:")
        for item in type_mismatch_records:
            print(
                f"  memory_id={item['memory_id']}, "
                f"relation={item['relation']}, "
                f"llm_index={item['llm_index']}, "
                f"available_keys={item['available_keys_like']}"
            )


if __name__ == "__main__":
    main()