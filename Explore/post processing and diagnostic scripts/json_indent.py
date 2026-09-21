#!/usr/bin/env python3
r"""
export_memories_full.py

Reads all memory files from:
    F:\New folder (4)\New folder\Memories

Writes a single file containing each memory as fully indented JSON.

Output:
    Explore\memories_full.txt

No memory files are modified.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
FULL_OUTPUT = EXPLORE_DIR / "memories_full.txt"


# =============================================================================
# Memory loading
# =============================================================================
def load_memory_records() -> List[Dict[str, Any]]:
    memories = []

    for jsonl_file in sorted(MEMORIES_DIR.glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line == "---":
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        memories.append(data)
                except json.JSONDecodeError:
                    continue

    for json_file in sorted(MEMORIES_DIR.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        if isinstance(data, dict):
            memories.append(data)
        elif isinstance(data, list):
            memories.extend(data)

    return memories


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Loading memories...")
    memories = load_memory_records()
    print(f"✅ Loaded {len(memories)} memory records.")

    with open(FULL_OUTPUT, "w", encoding="utf-8") as f:
        f.write("FULL MEMORIES EXPORT (PRETTY JSON)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total memories: {len(memories)}\n")
        f.write("=" * 80 + "\n\n")

        for memory in memories:
            # Pretty-print the full memory JSON
            f.write(json.dumps(memory, indent=2, ensure_ascii=False))
            f.write("\n\n" + "=" * 80 + "\n\n")

    print(f"📄 Full memories saved to:\n   {FULL_OUTPUT}")


if __name__ == "__main__":
    main()