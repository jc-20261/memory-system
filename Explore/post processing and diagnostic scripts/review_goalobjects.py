#!/usr/bin/env python3
r"""
review_goal_state_templates.py

Reads:
    F:\New folder (4)\New folder\Memories
    Explore\data\object_storage.json

Extracts all goal-state object names, grouped by activity, and matches them
against object template names from ObjectStorage.

When no template match is found for a goal-state object, the output also lists
all object templates present in that memory/activity.

Outputs:
    Explore\goal_state_template_review.txt
    Explore\data\goal_state_template_review.json

No memory files are modified.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from model2vec import StaticModel
    MODEL2VEC_AVAILABLE = True
except ImportError:
    MODEL2VEC_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
OBJECT_STORAGE_PATH = EXPLORE_DIR / "data" / "object_storage.json"

REVIEW_TXT_PATH = EXPLORE_DIR / "goal_state_template_review.txt"
REVIEW_JSON_PATH = EXPLORE_DIR / "data" / "goal_state_template_review.json"

MATCH_THRESHOLD = 0.6


# =============================================================================
# Normalization
# =============================================================================
def normalize_name(value: Any) -> str:
    """Normalize an object name or template name for matching."""
    if not isinstance(value, str):
        value = str(value)

    s = value.lower().strip()
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# =============================================================================
# Memory/activity extraction
# =============================================================================
def get_activity(memory: Dict[str, Any]) -> str:
    activity = memory.get("activity")
    if activity:
        return str(activity)

    inner = memory.get("memory") or {}
    activity = inner.get("activity")
    return str(activity) if activity else "unknown"


def get_goal_state_objects(memory: Dict[str, Any]) -> List[str]:
    """Return the distinct goal-state object names for a memory."""
    goal_state = memory.get("goal_state")
    if not isinstance(goal_state, list):
        inner = memory.get("memory") or {}
        goal_state = inner.get("goal_state")

    if not isinstance(goal_state, list):
        return []

    objects = []
    for goal in goal_state:
        if isinstance(goal, dict):
            obj = goal.get("object")
            if obj:
                objects.append(str(obj))

    return list(dict.fromkeys(objects))


def get_memory_object_templates(memory: Dict[str, Any]) -> List[str]:
    """Return distinct object template names present in this memory."""
    inner = memory.get("memory") or {}
    objects = inner.get("objects") or []

    templates = []
    for obj in objects:
        if isinstance(obj, dict):
            template = obj.get("template")
            if template:
                templates.append(str(template))

    return list(dict.fromkeys(templates))


def load_memories() -> List[Dict[str, Any]]:
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
# Object storage loading
# =============================================================================
def load_object_storage() -> Dict[str, Any]:
    if not OBJECT_STORAGE_PATH.exists():
        print(f"❌ Object storage not found: {OBJECT_STORAGE_PATH}")
        return {}

    with open(OBJECT_STORAGE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return {}

    return data


def extract_template_entries(object_storage: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return a list of template entries:
    {
        "name": template_name,
        "name_norm": normalized_template_name,
        "aliases": [alias, alias_norm, ...]
    }
    """
    entries = []

    for template_name, template_data in object_storage.items():
        if not isinstance(template_data, dict):
            continue

        aliases = template_data.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        aliases = [str(a) for a in aliases if a]

        entries.append(
            {
                "name": template_name,
                "name_norm": normalize_name(template_name),
                "aliases": aliases,
                "aliases_norm": [normalize_name(a) for a in aliases],
            }
        )

    return entries


# =============================================================================
# Matching
# =============================================================================
class TemplateMatcher:
    def __init__(self, entries: List[Dict[str, Any]]):
        self.entries = entries
        self._model = None
        self._is_model2vec = False
        self._name_embeddings = None

        if MODEL2VEC_AVAILABLE:
            try:
                self._model = StaticModel.from_pretrained("minishlab/potion-base-8M")
                self._is_model2vec = True
                print("✅ Using Model2Vec for goal-state template matching.")
            except Exception as e:
                print(f"⚠️ Model2Vec failed: {e}")

        if self._model is None and ST_AVAILABLE:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Using SentenceTransformer for goal-state template matching.")

        if self._model is not None:
            self._precompute_embeddings()

    def _precompute_embeddings(self):
        names = [entry["name_norm"] for entry in self.entries]

        if self._is_model2vec:
            vectors = self._model.encode(names)
        else:
            vectors = self._model.encode(names, normalize_embeddings=True)

        vectors = np.asarray(vectors).astype("float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._name_embeddings = vectors / norms

    def _encode_query(self, text: str) -> np.ndarray:
        if self._is_model2vec:
            vec = self._model.encode([text])[0]
        else:
            vec = self._model.encode(text, normalize_embeddings=True)

        vec = np.asarray(vec).astype("float32")
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def find_candidates(self, goal_object_name: str) -> List[Dict[str, Any]]:
        """Return candidate template matches above threshold."""
        query_norm = normalize_name(goal_object_name)

        # 1. Exact template name match
        for entry in self.entries:
            if entry["name_norm"] == query_norm:
                return [
                    {
                        "template": entry["name"],
                        "score": 1.0,
                        "match_type": "exact_template",
                    }
                ]

        # 2. Exact alias match
        for entry in self.entries:
            if query_norm in entry["aliases_norm"]:
                return [
                    {
                        "template": entry["name"],
                        "score": 1.0,
                        "match_type": "exact_alias",
                    }
                ]

        if self._name_embeddings is None:
            return []

        query_vec = self._encode_query(query_norm)
        sims = self._name_embeddings @ query_vec

        candidates = []
        for idx, score in enumerate(sims):
            score = float(score)
            if score >= MATCH_THRESHOLD:
                candidates.append(
                    {
                        "template": self.entries[idx]["name"],
                        "score": score,
                        "match_type": "fuzzy",
                    }
                )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates


# =============================================================================
# Main
# =============================================================================
def main():
    print("📂 Loading object storage...")
    object_storage = load_object_storage()
    template_entries = extract_template_entries(object_storage)

    if not template_entries:
        print("❌ No object templates found.")
        return

    print(f"✅ Loaded {len(template_entries)} object templates.")

    print("📂 Loading memories...")
    memories = load_memories()
    print(f"✅ Loaded {len(memories)} memories.")

    # Group goal objects and memory object templates by activity
    activity_goal_objects: Dict[str, List[str]] = defaultdict(list)
    activity_object_templates: Dict[str, List[str]] = defaultdict(list)

    for memory in memories:
        activity = get_activity(memory)

        # Goal-state objects
        for obj in get_goal_state_objects(memory):
            if obj not in activity_goal_objects[activity]:
                activity_goal_objects[activity].append(obj)

        # Object templates present in this memory
        for template in get_memory_object_templates(memory):
            if template not in activity_object_templates[activity]:
                activity_object_templates[activity].append(template)

    if not activity_goal_objects:
        print("⚠️ No goal-state objects found in memories.")
        return

    print("🔍 Matching goal-state objects against templates...")
    matcher = TemplateMatcher(template_entries)

    results = {}
    total_goal_objects = 0
    exact_matches = 0
    fuzzy_matches = 0
    no_matches = 0

    for activity, goal_objects in activity_goal_objects.items():
        activity_results = []
        memory_templates = activity_object_templates.get(activity, [])

        for goal_object in goal_objects:
            total_goal_objects += 1
            candidates = matcher.find_candidates(goal_object)

            item = {
                "goal_object": goal_object,
                "candidates": candidates,
            }

            if not candidates:
                no_matches += 1
                item["memory_object_templates"] = memory_templates
            else:
                if candidates[0]["score"] >= 1.0:
                    exact_matches += 1
                else:
                    fuzzy_matches += 1

            activity_results.append(item)

        results[activity] = activity_results

    # Save JSON
    REVIEW_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save readable text
    with open(REVIEW_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("GOAL STATE OBJECT TEMPLATE MATCHING REVIEW\n")
        f.write("=" * 80 + "\n")
        f.write(f"Threshold: {MATCH_THRESHOLD}\n")
        f.write(f"Activities reviewed: {len(results)}\n")
        f.write(f"Total goal objects reviewed: {total_goal_objects}\n")
        f.write(f"Exact matches: {exact_matches}\n")
        f.write(f"Fuzzy matches: {fuzzy_matches}\n")
        f.write(f"No matches above threshold: {no_matches}\n")
        f.write("=" * 80 + "\n\n")

        for activity, activity_results in results.items():
            f.write(f"\nACTIVITY: {activity}\n")
            f.write("-" * 80 + "\n")

            for item in activity_results:
                goal_object = item["goal_object"]
                candidates = item["candidates"]

                f.write(f"\n  Goal object: {goal_object}\n")

                if not candidates:
                    f.write("    ❌ No candidates above threshold\n")

                    memory_templates = item.get("memory_object_templates", [])
                    if memory_templates:
                        f.write("    Memory object templates:\n")
                        for mt in memory_templates:
                            f.write(f"      - {mt}\n")
                    continue

                top = candidates[0]
                f.write(f"    ✅ Top match: {top['template']} ({top['score']:.4f})\n")

                if len(candidates) > 1:
                    f.write("    Other candidates:\n")
                    for cand in candidates[1:]:
                        f.write(f"      - {cand['template']} ({cand['score']:.4f})\n")

            f.write("\n")

    print(f"\n📄 Readable review saved to:")
    print(f"   {REVIEW_TXT_PATH}")
    print(f"📄 JSON review saved to:")
    print(f"   {REVIEW_JSON_PATH}")
    print(f"\nSummary:")
    print(f"   Activities reviewed: {len(results)}")
    print(f"   Goal objects reviewed: {total_goal_objects}")
    print(f"   Exact matches: {exact_matches}")
    print(f"   Fuzzy matches: {fuzzy_matches}")
    print(f"   No matches: {no_matches}")


if __name__ == "__main__":
    main()