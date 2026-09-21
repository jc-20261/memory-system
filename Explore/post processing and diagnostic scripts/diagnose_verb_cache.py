#!/usr/bin/env python3
r"""
diagnose_verb_cache.py

Checks whether a specific verb term exists in the precomputed verb embedding cache.

Usage:
    python diagnose_verb_cache.py bake
"""

import json
import sys
from pathlib import Path

import numpy as np

EXPLORE_DIR = Path(__file__).resolve().parent.parent
VERB_CACHE = EXPLORE_DIR / "data" / "embeddings" / "verb.npz"


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_verb_cache.py <verb>")
        return

    target_verb = sys.argv[1].strip().lower()

    if not VERB_CACHE.exists():
        print(f"❌ Cache file not found: {VERB_CACHE}")
        return

    data = np.load(VERB_CACHE, allow_pickle=False)

    if "terms_json" not in data:
        print("❌ 'terms_json' not found in verb.npz")
        return

    terms = json.loads(str(data["terms_json"]))
    print(f"Total verb terms in cache: {len(terms)}")

    # Check exact term presence
    exact_terms = [t for t in terms if t == f"verb:{target_verb}"]
    print(f"\nExact term 'verb:{target_verb}' found: {len(exact_terms)}")

    # List all terms containing the target verb
    similar = [t for t in terms if target_verb in t.lower()]
    if similar:
        print(f"\nTerms containing '{target_verb}':")
        for t in similar[:50]:
            print(f"  - {t}")
    else:
        print(f"\nNo cached terms contain '{target_verb}'.")


if __name__ == "__main__":
    main()