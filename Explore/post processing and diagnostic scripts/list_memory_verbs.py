#!/usr/bin/env python3
r"""
list_memory_verbs.py

Lists all verbs that the search matching system sees for a specific pickle memory.

The verbs are read from:
    action.general_template.verb.name
    action.alt.verb.name

for every action and sub-action in the runtime pickle object.

Usage:
    python list_memory_verbs.py <memory_id>

Example:
    python list_memory_verbs.py mem_000051
"""

import pickle
import sys
from pathlib import Path

EXPLORE_DIR = Path(__file__).resolve().parent.parent
PICKLE_MEMORIES_DIR = EXPLORE_DIR / "pickle_memories"


def collect_verbs_from_action(action, results, path=""):
    """Recursively collect verb info from an ActionInstance."""
    if action is None:
        return

    action_name = getattr(action, "template_name", "?")
    general = getattr(action, "general_template", None)
    alt = getattr(action, "alt", None)

    general_name = getattr(general, "name", None) if general else None
    general_verb = getattr(getattr(general, "verb", None), "name", None) if general else None

    alt_name = getattr(alt, "name", None) if alt else None
    alt_verb = getattr(getattr(alt, "verb", None), "name", None) if alt else None

    results.append({
        "path": path or "root",
        "action_template": action_name,
        "general_template": general_name,
        "general_verb": general_verb,
        "alt_template": alt_name,
        "alt_verb": alt_verb,
    })

    sub_actions = getattr(action, "sub_actions", []) or []
    for i, sub in enumerate(sub_actions, 1):
        collect_verbs_from_action(sub, results, f"{path}.sub_{i}" if path else f"sub_{i}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python list_memory_verbs.py <memory_id>")
        return

    memory_id = sys.argv[1].strip()
    pkl_path = PICKLE_MEMORIES_DIR / f"{memory_id}.pkl"

    if not pkl_path.exists():
        print(f"❌ Pickle file not found: {pkl_path}")
        return

    with open(pkl_path, "rb") as f:
        memory = pickle.load(f)

    print(f"Memory ID: {memory.id}")
    print(f"Activity : {memory.activity}\n")

    results = []
    for i, action in enumerate(memory.actions, 1):
        collect_verbs_from_action(action, results, f"action_{i}")

    if not results:
        print("No actions found.")
        return

    # Print all verb-related information
    print("Action verbs visible to search index:\n")
    for r in results:
        print(f"Path             : {r['path']}")
        print(f"Action template  : {r['action_template']}")
        print(f"General template : {r['general_template']}")
        print(f"General verb     : {r['general_verb']}")
        print(f"Alt template     : {r['alt_template']}")
        print(f"Alt verb         : {r['alt_verb']}")
        print("-" * 60)

    # Print unique verb names found
    verb_set = set()
    for r in results:
        if r["general_verb"]:
            verb_set.add(r["general_verb"])
        if r["alt_verb"]:
            verb_set.add(r["alt_verb"])

    print("\nUnique verbs that would be indexed as 'verb:<name>':")
    if verb_set:
        for v in sorted(verb_set):
            print(f"  - {v}")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()