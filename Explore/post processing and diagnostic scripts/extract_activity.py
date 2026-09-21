#!/usr/bin/env python3
"""
extract_activity_names.py
Reads data/memory_by_id.json and writes all activity names vertically to
activity_names.txt
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
MEMORY_BY_ID_PATH = DATA_DIR / "memory_by_id.json"
OUTPUT_PATH = Path(__file__).parent / "activity_names.txt"

def main():
    if not MEMORY_BY_ID_PATH.exists():
        print(f"❌ File not found: {MEMORY_BY_ID_PATH}")
        return

    with open(MEMORY_BY_ID_PATH, "r", encoding="utf-8") as f:
        memory_by_id = json.load(f)

    names = []
    for memory_id, memory in memory_by_id.items():
        if isinstance(memory, dict):
            activity = memory.get("activity")
            if not activity:
                inner = memory.get("memory") or {}
                activity = inner.get("activity", "Unknown")
        else:
            activity = str(memory)

        names.append(activity)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for name in names:
            f.write(f"{name}\n")

    print(f"✅ Wrote {len(names)} activity names to:")
    print(f"   {OUTPUT_PATH}")

if __name__ == "__main__":
    main()