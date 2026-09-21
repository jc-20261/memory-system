#!/usr/bin/env python3
r"""
merge_alias_files.py

Merges the previously generated alias files into a single global alias expansion file.

Inputs:
    Explore\data\single_term_aliases.json
    Explore\data\alias_expansion_component_value.json

Outputs:
    Explore\data\alias_expansion.json
    Explore\alias_merge_report.txt

The merged file contains global alias maps for:
    action_aliases
    object_aliases
    category_aliases
    trajectory_aliases
    tag_aliases
    attribute_component_aliases
    value_aliases

Aliases are de-duplicated and capped at 10 per term.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Set

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
EXPLORE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPLORE_DIR / "data"

SINGLE_TERM_ALIASES = DATA_DIR / "single_term_aliases.json"
COMPONENT_VALUE_ALIASES = DATA_DIR / "alias_expansion_component_value.json"

OUTPUT_JSON = DATA_DIR / "alias_expansion.json"
OUTPUT_READABLE = EXPLORE_DIR / "alias_merge_report.txt"

MAX_ALIASES_PER_TERM = 10

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def add_aliases(container: Dict[str, List[str]], term: str, aliases: List[str]):
    """Add aliases for a term, de-duplicating and capping at MAX_ALIASES_PER_TERM."""
    if not isinstance(aliases, list):
        return
    existing = container.setdefault(term, [])
    for alias in aliases:
        if not isinstance(alias, str):
            continue
        alias = alias.strip()
        if alias and alias not in existing:
            existing.append(alias)
        if len(existing) >= MAX_ALIASES_PER_TERM:
            break
    container[term] = existing[:MAX_ALIASES_PER_TERM]

def merge_single_term_aliases(data: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    """
    Parse per-memory single-term aliases into global maps.

    Structure of data: {memory_id: {term_string: [alias, ...]}, ...}
    """
    containers: Dict[str, Dict[str, List[str]]] = {
        "action_aliases": {},
        "object_aliases": {},
        "category_aliases": {},
        "trajectory_aliases": {},
        "tag_aliases": {},
    }

    prefix_to_container = {
        "action:": "action_aliases",
        "template:": "object_aliases",
        "obj_id:": "object_aliases",
        "category:": "category_aliases",
        "trajectory:": "trajectory_aliases",
        "tag:": "tag_aliases",
    }

    for memory_id, term_map in data.items():
        if not isinstance(term_map, dict):
            continue
        for term, aliases in term_map.items():
            term = str(term)
            for prefix, container_name in prefix_to_container.items():
                if term.startswith(prefix):
                    clean_term = term[len(prefix):]
                    if clean_term:
                        add_aliases(containers[container_name], clean_term, aliases)
                    break
            else:
                # Unknown prefix: place under a generic category if desired
                # For now, preserve under tag_aliases as catch-all
                add_aliases(containers["tag_aliases"], term, aliases)

    return containers

def merge_component_value_aliases(data: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    """Extract attribute_component_aliases and value_aliases from component/value file."""
    containers: Dict[str, Dict[str, List[str]]] = {
        "attribute_component_aliases": {},
        "value_aliases": {},
    }
    for key, term_map in data.items():
        if key in containers and isinstance(term_map, dict):
            for term, aliases in term_map.items():
                add_aliases(containers[key], str(term), aliases)
    return containers

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("📂 Loading single-term aliases...")
    if not SINGLE_TERM_ALIASES.exists():
        print(f"❌ Not found: {SINGLE_TERM_ALIASES}")
        return
    with open(SINGLE_TERM_ALIASES, "r", encoding="utf-8") as f:
        single_data = json.load(f)

    print("📂 Loading component/value aliases...")
    if not COMPONENT_VALUE_ALIASES.exists():
        print(f"❌ Not found: {COMPONENT_VALUE_ALIASES}")
        return
    with open(COMPONENT_VALUE_ALIASES, "r", encoding="utf-8") as f:
        component_data = json.load(f)

    print("🔧 Merging single-term aliases...")
    merged = merge_single_term_aliases(single_data)

    print("🔧 Merging component/value aliases...")
    component_containers = merge_component_value_aliases(component_data)
    for key, value in component_containers.items():
        if key not in merged:
            merged[key] = {}
        for term, aliases in value.items():
            add_aliases(merged[key], term, aliases)

    # Order keys for consistency
    ordered_keys = [
        "action_aliases",
        "object_aliases",
        "category_aliases",
        "trajectory_aliases",
        "tag_aliases",
        "attribute_component_aliases",
        "value_aliases",
    ]
    final_merged = {}
    for key in ordered_keys:
        if key in merged:
            final_merged[key] = merged[key]

    # Save JSON
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_merged, f, indent=2, ensure_ascii=False)

    # Save readable report
    with open(OUTPUT_READABLE, "w", encoding="utf-8") as f:
        f.write("ALIAS MERGE REPORT\n")
        f.write("=" * 90 + "\n")
        for key, aliases in final_merged.items():
            f.write(f"\n{key.upper()} ({len(aliases)} terms)\n")
            f.write("-" * 60 + "\n")
            for term, alias_list in sorted(aliases.items()):
                f.write(f"\nTerm: {term}\n")
                for alias in alias_list:
                    f.write(f"  - {alias}\n")
            f.write("\n")

    total_aliases = sum(len(aliases) for aliases in final_merged.values())
    total_terms = sum(len(aliases) for aliases in final_merged.values())
    print(f"✅ Merged aliases saved to:\n   {OUTPUT_JSON}")
    print(f"✅ Readable report saved to:\n   {OUTPUT_READABLE}")
    print(f"   Total alias maps: {len(final_merged)}")
    print(f"   Total terms with aliases: {sum(len(v) for v in final_merged.values())}")

if __name__ == "__main__":
    main()