#!/usr/bin/env python3
r"""
diagnose_verb_index.py

Checks whether a given verb term in the search index has postings for a
specific memory.

Usage:
    python diagnose_verb_index.py bake mem_000051
"""

import sys
from pathlib import Path

from pickle_loader import PickleMemoryLoader
from pickle_search_index import PickleSearchIndex

EXPLORE_DIR = Path(__file__).resolve().parent.parent
def main():
    if len(sys.argv) < 3:
        print("Usage: python diagnose_verb_index.py <verb> <memory_id>")
        return

    verb = sys.argv[1].strip().lower()
    target_memory_id = sys.argv[2].strip()

    print("Loading pickle memories and building search index...")
    loader = PickleMemoryLoader()
    index = PickleSearchIndex(
        loader.memories,
        loader.object_storage,
        loader.alias_expansion,
    )

    term = f"verb:{verb}"

    if term not in index.index:
        print(f"❌ Index term '{term}' not found.")
        return

    postings = index.index[term]
    print(f"Total postings for '{term}': {len(postings)}")

    memory_ids = sorted({mem.id for mem, src, term_type in postings})
    print(f"Memory IDs containing '{term}':")
    for mid in memory_ids:
        marker = "  <-- target" if mid == target_memory_id else ""
        print(f"  - {mid}{marker}")

    if target_memory_id not in memory_ids:
        print(f"\n❌ '{target_memory_id}' is not in the postings for '{term}'.")
    else:
        print(f"\n✅ '{target_memory_id}' is present for '{term}'.")


if __name__ == "__main__":
    main()