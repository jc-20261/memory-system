#!/usr/bin/env python3
r"""
check_subaction_sequence_mismatches.py

Finds general action templates where:
    number of alts != total sub-action sequence keys associated with that general

Outputs:
    Explore\subaction_sequence_mismatches.txt
    Explore\data\subaction_sequence_mismatches.json

No files are modified.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPLORE_DIR / "data"

ACTION_STORAGE_PATH = DATA_DIR / "action_storage.json"

MISMATCH_TEXT_OUTPUT = EXPLORE_DIR / "subaction_sequence_mismatches.txt"
MISMATCH_JSON_OUTPUT = DATA_DIR / "subaction_sequence_mismatches.json"


# =============================================================================
# Load
# =============================================================================
def load_action_storage(path: Path) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Action storage JSON must be a dict")

    templates = data.get("templates", {})
    if not isinstance(templates, dict):
        templates = {}

    sub_action_sequences = data.get("sub_action_sequences", {})
    if not isinstance(sub_action_sequences, dict):
        sub_action_sequences = {}

    normalized_seqs = {}
    for key, seq in sub_action_sequences.items():
        if isinstance(seq, list):
            normalized_seqs[str(key)] = [str(s) for s in seq if s]
        else:
            normalized_seqs[str(key)] = []

    return templates, normalized_seqs


# =============================================================================
# Relationship building
# =============================================================================
def build_relationships(
    templates: Dict[str, Any],
) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    general_to_alts: Dict[str, Set[str]] = defaultdict(set)
    alt_to_general: Dict[str, str] = {}

    for name, info in templates.items():
        if not isinstance(info, dict):
            continue

        alternatives = info.get("alternatives")
        if isinstance(alternatives, list):
            for alt in alternatives:
                alt_str = str(alt)
                general_to_alts[name].add(alt_str)
                alt_to_general[alt_str] = name

        general_template = info.get("general_template")
        if general_template:
            general_str = str(general_template)
            general_to_alts[general_str].add(name)
            alt_to_general[name] = general_str

    # Templates with no relationship become their own general
    for name in templates.keys():
        if name not in alt_to_general and name not in general_to_alts:
            general_to_alts[name] = set()

    return general_to_alts, alt_to_general


# =============================================================================
# Mismatch detection
# =============================================================================
def find_mismatches(
    templates: Dict[str, Any],
    sub_action_sequences: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    general_to_alts, _ = build_relationships(templates)

    mismatches = []

    for general_name in sorted(general_to_alts.keys()):
        alts = general_to_alts.get(general_name, set())
        alt_count = len(alts)

        general_sequence_present = general_name in sub_action_sequences
        alt_sequence_matches = [alt for alt in sorted(alts) if alt in sub_action_sequences]

        alt_sequence_count = len(alt_sequence_matches)
        total_sequence_keys = (1 if general_sequence_present else 0) + alt_sequence_count

        if total_sequence_keys != alt_count:
            mismatches.append(
                {
                    "general_template": general_name,
                    "alt_count": alt_count,
                    "alts": sorted(alts),
                    "general_sequence_present": general_sequence_present,
                    "general_sequence_key": general_name if general_sequence_present else None,
                    "alt_sequence_matches": alt_sequence_matches,
                    "alt_sequence_count": alt_sequence_count,
                    "total_sequence_keys": total_sequence_keys,
                    "difference": total_sequence_keys - alt_count,
                }
            )

    return mismatches


# =============================================================================
# Output
# =============================================================================
def write_outputs(mismatches: List[Dict[str, Any]], text_path: Path, json_path: Path):
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("SUB-ACTION SEQUENCE MISMATCHES\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total mismatched templates: {len(mismatches)}\n")
        f.write("-" * 80 + "\n\n")

        for rec in mismatches:
            f.write(f"GENERAL TEMPLATE: {rec['general_template']}\n")
            f.write(f"Alt count: {rec['alt_count']}\n")
            f.write(f"Alts: {', '.join(rec['alts']) if rec['alts'] else '(none)'}\n")
            f.write(f"General sequence present: {rec['general_sequence_present']}\n")
            if rec["general_sequence_key"]:
                f.write(f"General sequence key: {rec['general_sequence_key']}\n")
            f.write(f"Alt sequence matches: {rec['alt_sequence_matches']}\n")
            f.write(f"Alt sequence count: {rec['alt_sequence_count']}\n")
            f.write(f"Total sequence keys: {rec['total_sequence_keys']}\n")
            f.write(f"Difference: {rec['difference']:+d}\n")
            f.write("-" * 60 + "\n")

    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mismatches, f, indent=2, ensure_ascii=False)

    print(f"📄 Mismatches text saved to:\n   {text_path}")
    print(f"📄 Mismatches JSON saved to:\n   {json_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Loading action storage...")
    templates, sub_action_sequences = load_action_storage(ACTION_STORAGE_PATH)
    print(f"   Loaded {len(templates)} templates")
    print(f"   Loaded {len(sub_action_sequences)} sub-action sequences")

    print("🔍 Finding mismatches...")
    mismatches = find_mismatches(templates, sub_action_sequences)
    print(f"   Mismatched general templates: {len(mismatches)}")

    write_outputs(mismatches, MISMATCH_TEXT_OUTPUT, MISMATCH_JSON_OUTPUT)


if __name__ == "__main__":
    main()