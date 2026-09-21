#!/usr/bin/env python3
r"""
apply_spatial_reencondings.py

Reads the latest spatial re-encoding output:
    Explore\data\spatial_reenconding_output.json

For every spatial relation record that has a final re-encoding, it finds the
exact location in the original memory JSON using memory_id + path, and adds:

    "relationr": "<reencoded_relation>"

next to the original "relation".

This script modifies the memory files in:
    F:\New folder (4)\New folder\Memories

Please back up that folder before running.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
REENCONDING_OUTPUT_PATH = EXPLORE_DIR / "data" / "spatial_reenconding_output.json"


# =============================================================================
# Helpers
# =============================================================================
def get_memory_id(memory: Dict[str, Any]) -> str:
    """Return the best available memory ID from a memory JSON dict."""
    memory_id = memory.get("memory_id")
    if memory_id:
        return str(memory_id)

    inner = memory.get("memory") or {}
    memory_id = inner.get("memory_id")
    return str(memory_id) if memory_id else "unknown"


def compute_path_parts(node: Any, target_path: str) -> List[str]:
    """
    Helper used only to compare a generated path string with the target.

    We traverse and build the path parts list for every dict that has both
    "relation" and "relative_to". The path string is built exactly as the
    extraction script did: by joining path parts with ".".
    """
    # Not used directly; traversal happens in find_and_apply below.
    return []


def apply_relationr_at_path(
    memory: Dict[str, Any],
    target_path: str,
    relationr: str,
) -> bool:
    """
    Recursively traverse memory and add relationr to the dict whose generated
    path string equals target_path.

    Returns True if applied, False otherwise.
    """
    def walk(node: Any, path_parts: List[str]) -> bool:
        if isinstance(node, dict):
            # Check if this dict is a spatial relation node
            if "relation" in node and "relative_to" in node:
                current_path = ".".join(path_parts)

                if current_path == target_path:
                    node["relationr"] = relationr
                    return True

            # Recurse into values
            for key, value in node.items():
                if walk(value, path_parts + [str(key)]):
                    return True

        elif isinstance(node, list):
            for i, item in enumerate(node):
                if walk(item, path_parts + [str(i)]):
                    return True

        return False

    return walk(memory, [])


# =============================================================================
# Memory file loading/writing
# =============================================================================
def load_memory_records() -> List[Dict[str, Any]]:
    """
    Load all memory files from the Memories folder.

    Returns a list of records:
    {
        "file_path": Path,
        "memory": dict,
        "index": int,
        "is_jsonl": bool
    }
    """
    records = []

    for jsonl_file in sorted(MEMORIES_DIR.glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            lines = [
                line.strip()
                for line in f
                if line.strip() and line.strip() != "---"
            ]

        memories = []
        for line in lines:
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    memories.append(data)
            except json.JSONDecodeError:
                continue

        for idx, memory in enumerate(memories):
            records.append(
                {
                    "file_path": jsonl_file,
                    "memory": memory,
                    "index": idx,
                    "is_jsonl": True,
                }
            )

    for json_file in sorted(MEMORIES_DIR.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        if isinstance(data, dict):
            memories = [data]
        elif isinstance(data, list):
            memories = data
        else:
            continue

        for idx, memory in enumerate(memories):
            if isinstance(memory, dict):
                records.append(
                    {
                        "file_path": json_file,
                        "memory": memory,
                        "index": idx,
                        "is_jsonl": False,
                    }
                )

    return records


def write_memory_records(records: List[Dict[str, Any]]):
    """Write updated memories back to their original files."""
    file_groups = defaultdict(list)

    for rec in records:
        file_groups[rec["file_path"]].append(rec)

    for file_path, recs in file_groups.items():
        recs.sort(key=lambda x: x["index"])
        is_jsonl = recs[0]["is_jsonl"] if recs else False
        memories = [rec["memory"] for rec in recs]

        if is_jsonl:
            with open(file_path, "w", encoding="utf-8") as f:
                for mem in memories:
                    f.write(json.dumps(mem, ensure_ascii=False) + "\n\n---\n")
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                if len(memories) == 1:
                    json.dump(memories[0], f, indent=2, ensure_ascii=False)
                else:
                    json.dump(memories, f, indent=2, ensure_ascii=False)


# =============================================================================
# Re-encoding output loading
# =============================================================================
def load_reenconding_output(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Re-encoding output must be a JSON object")

    return data


def get_final_reencoded_value(
    record: Dict[str, Any],
    reencoding_by_index: Dict[str, str],
) -> Optional[str]:
    """Return final re-encoded value for a record, or None."""
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
    if not REENCONDING_OUTPUT_PATH.exists():
        print(f"❌ Re-encoding output not found: {REENCONDING_OUTPUT_PATH}")
        print("Run llm_spatial_reenconding_extract.py first.")
        return

    print("📂 Loading re-encoding output...")
    output_data = load_reenconding_output(REENCONDING_OUTPUT_PATH)

    records = output_data.get("records")
    reencoding_by_index = output_data.get("reencoding_by_index")

    if not isinstance(records, list) or not isinstance(reencoding_by_index, dict):
        print("❌ Unexpected JSON structure.")
        return

    print(f"✅ Loaded {len(records)} re-encoding records.")
    print(f"✅ Loaded {len(reencoding_by_index)} LLM mapping entries.")

    # Build list of records that actually have a final re-encoding
    applicable_records = []

    for rec in records:
        final_value = get_final_reencoded_value(rec, reencoding_by_index)

        if final_value is None:
            continue

        applicable_records.append(
            {
                "memory_id": rec.get("memory_id"),
                "path": rec.get("path"),
                "relationr": final_value,
            }
        )

    print(f"🔎 Records to apply: {len(applicable_records)}")

    if not applicable_records:
        print("⚠️ No applicable records found.")
        return

    print("📂 Loading memory files...")
    memory_records = load_memory_records()
    print(f"✅ Loaded {len(memory_records)} memory records.")

    # Build memory_id -> memory record mapping
    memory_by_id = {}
    for mem_rec in memory_records:
        mem_id = get_memory_id(mem_rec["memory"])
        memory_by_id[mem_id] = mem_rec

    applied_count = 0
    missing_memory_ids = set()
    missing_paths = set()

    for item in applicable_records:
        memory_id = item["memory_id"]
        path = item["path"]
        relationr = item["relationr"]

        if memory_id is None or path is None:
            continue

        target_memory_rec = memory_by_id.get(memory_id)

        if target_memory_rec is None:
            missing_memory_ids.add(memory_id)
            continue

        success = apply_relationr_at_path(
            target_memory_rec["memory"],
            path,
            relationr,
        )

        if success:
            applied_count += 1
        else:
            missing_paths.add(f"{memory_id}:{path}")

    print(f"✅ Applied {applied_count} relationr additions.")

    if missing_memory_ids:
        print(f"\n⚠️ Missing memory IDs ({len(missing_memory_ids)}):")
        for mid in sorted(missing_memory_ids):
            print(f"   - {mid}")

    if missing_paths:
        print(f"\n⚠️ Paths not found ({len(missing_paths)}):")
        for p in sorted(missing_paths):
            print(f"   - {p}")

    # Write back
    print("\n💾 Writing updated memory files...")
    write_memory_records(memory_records)

    print(f"📂 Updated memory files in:\n   {MEMORIES_DIR}")


if __name__ == "__main__":
    main()