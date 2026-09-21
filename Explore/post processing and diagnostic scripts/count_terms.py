#!/usr/bin/env python3
r"""
searchable_term_frequencies_occurrences.py

Calculates frequencies of every searchable term in the pickle search index.

Primary metric: occurrence count.
Also includes distinct memory count.

Columns:
    term
    occurrences
    occurrence percentage
    memory count
    memory percentage

Terms are ranked by occurrence count descending.

Outputs:
    Explore\searchable_term_frequencies_occurrences.txt
    Explore\data\searchable_term_frequencies_occurrences.json
"""

import json
from collections import defaultdict
from pathlib import Path

from pickle_loader import PickleMemoryLoader
from pickle_search_index import PickleSearchIndex

EXPLORE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPLORE_DIR / "data"
READABLE_OUTPUT = EXPLORE_DIR / "searchable_term_frequencies_occurrences.txt"
JSON_OUTPUT = DATA_DIR / "searchable_term_frequencies_occurrences.json"

MAX_TERM_WIDTH = 80


def truncate_term(term: str, max_width: int = MAX_TERM_WIDTH) -> str:
    """Truncate a term for table display."""
    if len(term) <= max_width:
        return term
    return term[: max_width - 3] + "..."


def main():
    print("Loading pickle memories...")
    loader = PickleMemoryLoader()
    memories = loader.memories
    total_memories = len(memories)
    print(f"  Loaded {total_memories} memories.")

    print("Building search index...")
    index = PickleSearchIndex(memories, loader.object_storage)
    print(f"  Index exact terms: {len(index.index)}")
    print(f"  Index word terms:  {len(index.word_terms)}")

    occurrence_counts = defaultdict(int)
    memory_sets = defaultdict(set)

    # Collect exact index terms
    for term, postings in index.index.items():
        occurrence_counts[term] += len(postings)
        for mem, weight, source, term_type in postings:
            memory_sets[term].add(mem.id)

    # Collect word-level fuzzy terms
    for word, entries in index._word_to_memories.items():
        occurrence_counts[word] += len(entries)
        for mem, source, term_type in entries:
            memory_sets[word].add(mem.id)

    if not occurrence_counts:
        print("No searchable terms found.")
        return

    total_occurrences = sum(occurrence_counts.values())

    rows = []
    for term, occ_count in occurrence_counts.items():
        mem_count = len(memory_sets.get(term, set()))
        occ_pct = (occ_count / total_occurrences * 100) if total_occurrences else 0.0
        mem_pct = (mem_count / total_memories * 100) if total_memories else 0.0
        rows.append({
            "term": term,
            "occurrences": occ_count,
            "memory_count": mem_count,
            "occurrence_percentage": occ_pct,
            "memory_percentage": mem_pct,
        })

    # Sort by occurrences descending
    rows.sort(key=lambda x: (-x["occurrences"], x["term"]))

    # Determine display width
    display_rows = [
        (truncate_term(row["term"]), row["occurrences"], row["occurrence_percentage"], row["memory_count"], row["memory_percentage"])
        for row in rows
    ]
    max_display_len = max(len(row[0]) for row in display_rows) if display_rows else 1
    term_width = max(max_display_len, 20)

    with open(READABLE_OUTPUT, "w", encoding="utf-8") as f:
        header = (
            f"{'term':<{term_width}} "
            f"{'occurrences':>12} "
            f"{'occ%':>10} "
            f"{'mem_count':>10} "
            f"{'mem%':>10}"
        )
        f.write(header + "\n")

        separator = (
            "-" * term_width + " "
            + "-" * 12 + " "
            + "-" * 10 + " "
            + "-" * 10 + " "
            + "-" * 10
        )
        f.write(separator + "\n")

        for term, occ, occ_pct, mem_count, mem_pct in display_rows:
            line = (
                f"{term:<{term_width}} "
                f"{occ:>12} "
                f"{occ_pct:>9.6f}% "
                f"{mem_count:>10} "
                f"{mem_pct:>9.6f}%"
            )
            f.write(line + "\n")

    json_data = {
        "total_memories": total_memories,
        "total_occurrences": total_occurrences,
        "terms": rows,
    }

    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"📄 Occurrence-based frequencies saved to:\n   {READABLE_OUTPUT}")
    print(f"📄 JSON saved to:\n   {JSON_OUTPUT}")


if __name__ == "__main__":
    main()