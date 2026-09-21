#!/usr/bin/env python3
r"""
summarize_spatial_reencondings.py

Reads:
    Explore\data\spatial_reenconding_output.json

This version works with the latest extraction output format:

{
  "records": [
    {
      "memory_id": "...",
      "object": "...",
      "relation": "...",
      "relative_to": "...",
      "path": "...",
      "manual_reencoded": null,
      "llm_index": 0
    }
  ],
  "reencoding_by_index": {
    "0": "near",
    "1": "on"
  }
}

Counts unique original relation → reencoded relation pairs.

Writes:
    Explore\spatial_reenconding_summary.txt
    Explore\data\spatial_reenconding_summary.json

No memory files are modified.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
INPUT_JSON = EXPLORE_DIR / "data" / "spatial_reenconding_output.json"

READABLE_OUTPUT = EXPLORE_DIR / "spatial_reenconding_summary.txt"
JSON_OUTPUT = EXPLORE_DIR / "data" / "spatial_reenconding_summary.json"


# =============================================================================
# Load
# =============================================================================
def load_output(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Re-encoding output must be a JSON object")

    return data


def get_reencoded_value(record: Dict[str, Any], reencoding_by_index: Dict[str, str]) -> Optional[str]:
    """Return the final re-encoded relation for one extracted record."""
    if "manual_reencoded" in record:
        manual = record.get("manual_reencoded")
        if manual is not None:
            return str(manual)

    llm_index = record.get("llm_index")
    if llm_index is not None:
        return reencoding_by_index.get(str(llm_index))

    return None


# =============================================================================
# Summary building
# =============================================================================
def build_pair_counts(
    records: List[Dict[str, Any]],
    reencoding_by_index: Dict[str, str],
) -> Counter:
    pair_counts = Counter()

    for rec in records:
        original = rec.get("relation")
        if original is None:
            continue

        original_str = str(original).strip()
        reencoded = get_reencoded_value(rec, reencoding_by_index)

        if reencoded is None:
            reencoded = "(no mapping)"

        pair = (original_str, reencoded)
        pair_counts[pair] += 1

    return pair_counts


# =============================================================================
# Output
# =============================================================================
def write_summary(pair_counts: Counter, readable_path: Path, json_path: Path):
    sorted_pairs = sorted(
        pair_counts.items(),
        key=lambda x: (-x[1], x[0][0]),
    )

    with open(readable_path, "w", encoding="utf-8") as f:
        f.write("SPATIAL RELATION RE-ENCODING SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Original':30s} {'Reencoded':20s} {'Count':>6s}\n")
        f.write("-" * 80 + "\n")

        for (original, reencoded), count in sorted_pairs:
            f.write(f"{original:30s} {reencoded:20s} {count:>6d}\n")

        f.write("-" * 80 + "\n")
        f.write(f"Total unique pairs: {len(pair_counts)}\n")
        f.write(f"Total occurrences:  {sum(pair_counts.values())}\n")

    summary_data = {
        "total_unique_pairs": len(pair_counts),
        "total_occurrences": sum(pair_counts.values()),
        "pairs": [
            {
                "original_relation": original,
                "reencoded_relation": reencoded,
                "count": count,
            }
            for (original, reencoded), count in sorted_pairs
        ],
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"📄 Summary saved to:\n   {readable_path}")
    print(f"📄 JSON summary saved to:\n   {json_path}")


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

    pair_counts = build_pair_counts(records, reencoding_by_index)

    write_summary(pair_counts, READABLE_OUTPUT, JSON_OUTPUT)


if __name__ == "__main__":
    main()