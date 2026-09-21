#!/usr/bin/env python3
"""
search_index.py – Build a reverse index from memory dicts and perform search.

Updated features:
- Two-stage indexing: specified extraction first, filtered recursive catch-all second.
- All fields are indexed.
- True phrase embeddings only for action/object/template/tag/trajectory/category/alias names.
- Word-level fuzzy matching with precomputed distinct vocabulary.
- Stop/control/unit/numeric filtering.
- Attribute-value pair parsing with fuzzy value AND fuzzy attribute-path matching.
- Runtime progress reporting support.
- Fixed handling of list-type attribute paths in preconditions/changes.
- Category/function/material indexing from object_templates and object_storage.
- Search result detail generation and context location.
- Loads object_storage.json and indexes categories/functions/materials from it.
- Structural JSON keys are excluded from word-level indexing.
- Terms with structural prefixes do not fall through to word-level fuzzy matching.
- Memory score entries include activity name.
- Special operator += handled for additions.
- Position matching uses combined relation:relative_to, not standalone relation.
- Numeric range matching for measurements.
- Query-side phrase decomposition for phrase-like terms.
- Independent component matching with type-aware scoring.
- Goal-state indexing and matching.
- Unit-aware volume derivation and matching.
- Term/component frequency export.
"""

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional, Callable

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
# Similarity threshold default
# =============================================================================
DEFAULT_SIMILARITY_THRESHOLD = 0.8

# =============================================================================
# Stop/control/unit words
# =============================================================================
STOP_WORDS = {
    "true", "false", "on", "off", "none", "open", "closed",
    "current", "value", "to", "from", "in", "the", "a", "an",
    "low", "medium", "high", "yes", "no", "0", "1", "and", "or",
    "with", "for", "of", "at", "by", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "shall", "should", "may", "might",
    "must", "can", "could", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "mine", "yours", "hers", "ours", "theirs",
    "category", "action", "template", "object", "trajectory",
    "function", "precondition", "state_change", "goal_state",
    "dimension", "dimensions", "position", "relation",
    "relative_to", "unit", "memory", "objects", "actions",
    "templates", "aliases", "categories", "functions", "materials"
}

UNIT_WORDS = {
    "cm", "m", "mm", "km", "g", "kg", "mg", "ml", "l", "s",
    "sec", "min", "hr", "hour", "hours", "degree", "degrees",
    "percent", "%", "gram", "grams", "meter", "meters",
    "centimeter", "centimeters", "millimeter", "millimeters",
    "liter", "liters", "milliliter", "milliliters", "kilogram",
    "kilograms"
}

NON_DECOMPOSABLE_ATTRIBUTES = {
    "length", "width", "height_thickness", "height", "mass", "weight",
    "shape", "dimensions", "volume", "duration", "repetitions",
    "cycle_duration", "start_offset", "temporal_type", "number",
    "position", "position_relative_body", "material", "materials"
}

TOO_GENERAL_CATEGORIES = {
    "object", "physical_object", "person", "agent", "entity"
}

STRUCTURAL_KEYS = {
    "preconditions", "changes", "sub_actions", "objects", "actions",
    "object_templates", "action_templates", "memory", "changes_per_cycle",
    "changes_total", "conditional_changes"
}

STRUCTURAL_QUERY_PREFIXES = (
    "precondition:",
    "state_change:",
    "goal_state:",
)

MEASUREMENT_ATTRIBUTES = {
    "dimensions.length",
    "dimensions.width",
    "dimensions.height_thickness",
    "mass",
    "weight",
    "volume",
}

PHRASE_QUERY_PREFIXES = (
    "action:",
    "template:",
    "object:",
    "trajectory:",
    "category:",
    "function:",
    "material:",
    "tag:",
    "alias:",
)


# =============================================================================
# Tokenization helpers
# =============================================================================
def tokenize_into_words(text: str) -> List[str]:
    if not isinstance(text, str):
        text = str(text)
    separators = "_:=->./\\()[]{} "
    for sep in separators:
        text = text.replace(sep, " ")
    return [w.lower().strip() for w in text.split() if w.strip()]


def parse_measurement(value: str) -> Optional[Tuple[float, str]]:
    """Parse a measurement value like '30cm' into (30, 'cm')."""
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*([a-zA-Z%]*)\s*$", str(value))
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2).lower()
    return num, unit


def is_within_range(query_val: str, indexed_val: str) -> bool:
    """Return True if query measurement is within 50%-150% of indexed measurement."""
    q = parse_measurement(query_val)
    idx = parse_measurement(indexed_val)
    if q is None or idx is None:
        return False
    q_num, q_unit = q
    idx_num, idx_unit = idx
    if q_unit and idx_unit and q_unit != idx_unit:
        return False
    low = 0.5 * idx_num
    high = 1.5 * idx_num
    return low <= q_num <= high


# =============================================================================
# Volume conversion
# =============================================================================
VOLUME_TO_LITERS = {
    "cm^3": 0.001,
    "cm3": 0.001,
    "cc": 0.001,
    "ml": 0.001,
    "milliliter": 0.001,
    "milliliters": 0.001,
    "l": 1.0,
    "liter": 1.0,
    "liters": 1.0,
    "m^3": 1000.0,
    "m3": 1000.0,
    "mm^3": 0.000001,
    "mm3": 0.000001,
}


def parse_volume_to_liters(value: str) -> Optional[float]:
    """Convert a volume string like '1000cm^3' to liters."""
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*([a-zA-Z^0-9]*)\s*$", str(value))
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2).strip().lower()
    if unit not in VOLUME_TO_LITERS:
        return None
    return num * VOLUME_TO_LITERS[unit]


def is_within_volume_range(query_val: str, indexed_val: str) -> bool:
    q = parse_volume_to_liters(query_val)
    idx = parse_volume_to_liters(indexed_val)
    if q is None or idx is None or idx == 0:
        return False
    low = 0.5 * idx
    high = 1.5 * idx
    return low <= q <= high


# =============================================================================
# ReverseIndex
# =============================================================================
class ReverseIndex:
    def __init__(self, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.index: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.memory_by_id: Dict[str, Dict[str, Any]] = {}
        self.threshold = similarity_threshold

        self.object_storage = self._load_object_storage()

        self._embedding_model = None
        self._is_model2vec = False

        if MODEL2VEC_AVAILABLE:
            try:
                self._embedding_model = StaticModel.from_pretrained("minishlab/potion-base-8M")
                self._is_model2vec = True
                print("✅ Using Model2Vec for fuzzy matching.")
            except Exception as e:
                print(f"⚠️ Failed to load Model2Vec: {e}. Falling back to SentenceTransformer.")
        if self._embedding_model is None and ST_AVAILABLE:
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Using SentenceTransformer for fuzzy matching (Model2Vec not available).")
        elif self._embedding_model is None:
            print("⚠️ No embedding model available. Fuzzy matching disabled.")

        self._phrase_terms: List[str] = []
        self._phrase_embeddings: Optional[np.ndarray] = None
        self._phrase_to_idx: Dict[str, int] = {}

        self._word_terms: List[str] = []
        self._word_embeddings: Optional[np.ndarray] = None
        self._word_to_idx: Dict[str, int] = {}
        self._word_to_memories: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        self._attribute_pair_terms: List[str] = []
        self._attribute_component_to_terms: Dict[str, List[str]] = defaultdict(list)

    def _load_object_storage(self) -> Dict[str, Any]:
        path = Path(__file__).parent / "data" / "object_storage.json"
        if not path.exists():
            print("⚠️ object_storage.json not found in data folder; object categories/functions/materials will not be indexed from storage.")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load object_storage.json: {e}")
            return {}

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def add_memory(self, memory: Dict[str, Any]):
        memory_id = memory.get("memory_id")
        if not memory_id:
            inner_memory = memory.get("memory", {})
            memory_id = inner_memory.get("memory_id", str(id(memory)))

        self.memory_by_id[memory_id] = memory

        obj_id_to_template = self._build_obj_id_mapping(memory)

        terms, phrase_terms, word_terms = self._extract_terms_specified(memory, obj_id_to_template)

        for term, weight in terms.items():
            self.index[term].append((memory_id, weight))

        for term in phrase_terms:
            if term not in self._phrase_to_idx:
                self._phrase_terms.append(term)
                self._phrase_to_idx[term] = len(self._phrase_terms) - 1

        for word in word_terms:
            if word not in self._word_to_idx:
                self._word_terms.append(word)
                self._word_to_idx[word] = len(self._word_terms) - 1
            self._word_to_memories[word].append((memory_id, 1.0))

        generic_terms, generic_words = self._extract_terms_catchall(memory)
        for term, weight in generic_terms.items():
            self.index[term].append((memory_id, weight))
        for word in generic_words:
            if word not in self._word_to_idx:
                self._word_terms.append(word)
                self._word_to_idx[word] = len(self._word_terms) - 1
            self._word_to_memories[word].append((memory_id, 1.0))

    def build_index(self, memories: List[Dict[str, Any]]):
        for mem in memories:
            self.add_memory(mem)
        self._precompute_embeddings()

    def _build_obj_id_mapping(self, memory: Dict[str, Any]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        inner = memory.get("memory", {})
        objects = inner.get("objects") or []
        for obj in objects:
            if isinstance(obj, dict):
                obj_id = obj.get("obj_id")
                template = obj.get("template")
                if obj_id and template:
                    mapping[obj_id] = template
        return mapping

    def _precompute_embeddings(self):
        if self._embedding_model is None:
            return

        if self._phrase_terms:
            print(f"🔢 Precomputing phrase embeddings for {len(self._phrase_terms)} terms...")
            self._phrase_embeddings = self._embed_terms(self._phrase_terms)

        if self._word_terms:
            print(f"🔤 Precomputing word embeddings for {len(self._word_terms)} words...")
            self._word_embeddings = self._embed_terms(self._word_terms)

        print("✅ Embeddings ready.")

    def _embed_terms(self, terms: List[str]) -> np.ndarray:
        if self._is_model2vec:
            vectors = self._embedding_model.encode(terms)
        else:
            vectors = self._embedding_model.encode(terms, normalize_embeddings=True)

        vectors = np.asarray(vectors).astype("float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    # ------------------------------------------------------------------
    # Specified extraction
    # ------------------------------------------------------------------
    def _extract_terms_specified(self, memory: Dict[str, Any], obj_id_to_template: Dict[str, str]):
        terms: Dict[str, float] = {}
        phrase_terms: Set[str] = set()
        word_terms: Set[str] = set()

        def add_term(term: str, weight: float):
            terms[term] = terms.get(term, 0.0) + weight
            if not term.startswith(("action:", "template:", "object:", "trajectory:",
                                    "category:", "function:", "precondition:",
                                    "state_change:", "goal_state:")):
                for word in tokenize_into_words(term):
                    if word and word not in STOP_WORDS and word not in UNIT_WORDS and len(word) > 1:
                        word_terms.add(word)

        def add_phrase_words(phrase: str):
            clean = phrase.lower()
            words = [
                w for w in tokenize_into_words(clean)
                if w not in STOP_WORDS and w not in UNIT_WORDS and len(w) > 1
            ]
            for w in words:
                word_terms.add(w)

        def add_attribute_pair(path: Any, value: Any, weight: float = 1.0):
            if isinstance(path, (list, tuple)):
                path = ".".join(str(x) for x in path)
            else:
                path = str(path)

            if path == "position.relation" or path.startswith("position.relation."):
                return

            if value is None:
                return
            value_str = self._value_to_string(value)
            if value_str == "":
                return
            term = f"{path}={value_str}"
            add_term(term, weight)
            if term not in self._attribute_pair_terms:
                self._attribute_pair_terms.append(term)
            for comp in re.split(r"[.:]", path):
                comp = comp.strip()
                if comp and comp not in STOP_WORDS and comp not in UNIT_WORDS:
                    self._attribute_component_to_terms[comp].append(term)

            if isinstance(value, list):
                for item in value:
                    item_str = self._value_to_string(item)
                    if item_str:
                        add_attribute_pair(path, item_str, weight * 0.8)
            elif isinstance(value, str) and "," in value:
                for item in value.split(","):
                    item = item.strip()
                    if item:
                        add_attribute_pair(path, item, weight * 0.8)

        def add_phrase(term: str):
            if term.lower() not in STOP_WORDS and len(term) > 1:
                phrase_terms.add(term)
                add_phrase_words(term)

        activity = memory.get("activity")
        if activity:
            add_term(f"activity:{activity}", 1.0)
            add_phrase(activity)

        memory_id = memory.get("memory_id")
        if memory_id:
            add_term(f"memory_id:{memory_id}", 1.0)

        inner = memory.get("memory", {})
        if not isinstance(inner, dict):
            inner = {}

        # ----- Goal states -----
        goals = memory.get("goal_state") or inner.get("goal_state") or []
        if isinstance(goals, list):
            for goal in goals:
                if isinstance(goal, dict):
                    self._index_goal_state(goal, terms, obj_id_to_template, add_term, add_attribute_pair)

        # ----- Object templates -----
        object_templates = memory.get("object_templates") or {}
        if isinstance(object_templates, dict):
            for tname, tmpl in object_templates.items():
                if not isinstance(tmpl, dict):
                    continue
                cats = tmpl.get("categories") or []
                for cat in cats:
                    if str(cat).lower() in TOO_GENERAL_CATEGORIES:
                        continue
                    add_term(f"category:{cat}", 1.0)
                    add_phrase(str(cat))
                funcs = tmpl.get("functions") or []
                for func in funcs:
                    add_term(f"function:{func}", 0.8)
                    add_phrase(str(func))
                materials = tmpl.get("materials") or tmpl.get("material") or []
                if isinstance(materials, str):
                    materials = [materials]
                for mat in materials:
                    add_term(f"material:{mat}", 1.0)
                    add_phrase(str(mat))

        # ----- Object instances -----
        objects = inner.get("objects") or []
        for obj in objects:
            if not isinstance(obj, dict):
                continue

            template = obj.get("template")
            obj_id = obj.get("obj_id")

            if template:
                add_term(f"template:{template}", 1.0)
                add_term(f"object:{template}", 1.0)
                add_phrase(template)

                template_info = self.object_storage.get(template) or {}
                if isinstance(template_info, dict):
                    for cat in template_info.get("categories") or []:
                        if str(cat).lower() in TOO_GENERAL_CATEGORIES:
                            continue
                        add_term(f"category:{cat}", 1.0)
                        add_phrase(str(cat))
                    for func in template_info.get("functions") or []:
                        add_term(f"function:{func}", 0.8)
                        add_phrase(str(func))
                    materials = template_info.get("materials") or template_info.get("material") or []
                    if isinstance(materials, str):
                        materials = [materials]
                    for mat in materials:
                        add_term(f"material:{mat}", 1.0)
                        add_phrase(str(mat))

            if obj_id:
                add_term(f"object_id:{obj_id}", 1.0)

            number = obj.get("number")
            if number:
                add_term(f"number={number}", 0.5)

            cats = obj.get("categories") or []
            for cat in cats:
                if str(cat).lower() in TOO_GENERAL_CATEGORIES:
                    continue
                add_term(f"category:{cat}", 1.0)
                add_phrase(str(cat))

            overrides = obj.get("overrides") or {}
            for path, val in self._flatten_dict(overrides).items():
                add_attribute_pair(path, val, 1.0)

            dims = obj.get("dimensions") or {}
            dim_values = {}
            for dim, d in dims.items():
                if isinstance(d, dict):
                    combined = self._combine_value_unit(d.get("value"), d.get("unit"))
                    add_attribute_pair(f"dimensions.{dim}", combined, 1.0)
                    dim_values[dim] = combined

            # Derive volume from length x width x height_thickness
            volume_str = self._derive_volume(dim_values)
            if volume_str:
                add_attribute_pair("volume", volume_str, 1.0)

            weight = obj.get("weight") or obj.get("mass")
            if isinstance(weight, dict):
                combined = self._combine_value_unit(weight.get("value"), weight.get("unit"))
                add_attribute_pair("mass", combined, 1.0)
                add_attribute_pair("weight", combined, 0.8)

            position = obj.get("position")
            if isinstance(position, dict):
                rel = position.get("relation")
                rel_to = position.get("relative_to")
                if rel and rel_to:
                    add_term(f"position={rel}:{rel_to}", 0.5)
                    add_attribute_pair("position.relative_to", rel_to, 0.25)
                elif rel_to:
                    add_attribute_pair("position.relative_to", rel_to, 0.25)

            pos_rel_body = obj.get("position_relative_body")
            if isinstance(pos_rel_body, dict):
                rel = pos_rel_body.get("relation")
                if rel:
                    add_attribute_pair("position_relative_body.relation", rel, 0.5)

            motion = obj.get("motion")
            if isinstance(motion, dict):
                for k, v in motion.items():
                    add_attribute_pair(f"motion.{k}", v, 0.8)

            shape = obj.get("shape")
            if shape:
                add_attribute_pair("shape", shape, 1.0)

            for alias in obj.get("aliases") or []:
                add_term(f"alias:{alias}", 1.0)
                add_phrase(alias)

            for comp in obj.get("composite_of") or []:
                add_term(f"composite_of:{comp}", 0.5)
            for func in obj.get("functions") or []:
                add_term(f"function:{func}", 0.8)
                add_phrase(str(func))

            materials = obj.get("materials") or obj.get("material") or []
            if isinstance(materials, str):
                materials = [materials]
            for mat in materials:
                add_term(f"material:{mat}", 1.0)
                add_phrase(str(mat))

        # ----- Actions -----
        actions = inner.get("actions") or []
        for act in actions:
            self._index_action_specified(
                act, terms, phrase_terms, word_terms,
                obj_id_to_template, add_term, add_attribute_pair, add_phrase,
                scale=1.0,
            )

        return terms, phrase_terms, word_terms

    def _index_goal_state(self, goal, terms, obj_id_to_template, add_term, add_attribute_pair):
        obj_id = goal.get("object", "?")
        aspect = goal.get("aspect", "")
        attr = goal.get("attribute", "?")
        val = goal.get("value", "?")
        template = obj_id_to_template.get(obj_id, obj_id)

        full = f"{obj_id}.{aspect}.{attr}" if aspect else f"{obj_id}.{attr}"

        # High-weight goal-state terms
        add_term(f"goal_state:{full}={val}", 10.0)
        add_term(f"goal_state:{attr}={val}", 10.0)
        add_attribute_pair(full, val, 5.0)
        add_attribute_pair(attr, val, 5.0)

    def _index_action_specified(
        self,
        action,
        terms,
        phrase_terms,
        word_terms,
        obj_id_to_template,
        add_term,
        add_attribute_pair,
        add_phrase,
        scale=1.0,
    ):
        if not isinstance(action, dict):
            return

        template = action.get("template")
        if template:
            term = f"action:{template}"
            terms[term] = terms.get(term, 0.0) + 2.0 * scale
            phrase_terms.add(template)
            add_phrase(template)

        trajectory = action.get("kinematic_trajectory")
        if trajectory:
            term = f"trajectory:{trajectory}"
            terms[term] = terms.get(term, 0.0) + 2.0 * scale
            phrase_terms.add(trajectory)
            add_phrase(trajectory)

        category = action.get("action_category")
        if category:
            term = f"category:{category}"
            terms[term] = terms.get(term, 0.0) + 2.0 * scale
            phrase_terms.add(category)
            add_phrase(category)

        tags = action.get("tags") or []
        for tag in tags:
            terms[tag] = terms.get(tag, 0.0) + 2.0 * scale
            phrase_terms.add(tag)
            add_phrase(tag)

        duration = action.get("duration")
        if duration is not None:
            add_attribute_pair("duration", duration, 0.5 * scale)

        temporal_type = action.get("temporal_type")
        if temporal_type:
            add_attribute_pair("temporal_type", temporal_type, 0.5 * scale)

        for p in action.get("participants") or []:
            terms[f"action_participant:{p}"] = (
                terms.get(f"action_participant:{p}", 0.0) + 1.0 * scale
            )

        overrides = action.get("overrides") or {}
        if isinstance(overrides, dict):
            for path, val in self._flatten_dict(overrides).items():
                add_attribute_pair(f"{template}.{path}", val, 1.0 * scale)

        for prec in action.get("preconditions") or []:
            if isinstance(prec, dict):
                self._index_condition_specified(
                    prec, terms, phrase_terms, word_terms,
                    obj_id_to_template, add_term, add_attribute_pair,
                    prefix="precondition",
                )

        for ch in (
            action.get("changes", [])
            + action.get("changes_per_cycle", [])
            + action.get("changes_total", [])
        ):
            if isinstance(ch, dict):
                self._index_change_specified(
                    ch, terms, phrase_terms, word_terms,
                    obj_id_to_template, add_term, add_attribute_pair,
                )

        for cond in action.get("conditional_changes") or []:
            if isinstance(cond, dict):
                for ch in cond.get("changes") or []:
                    if isinstance(ch, dict):
                        self._index_change_specified(
                            ch, terms, phrase_terms, word_terms,
                            obj_id_to_template, add_term, add_attribute_pair,
                        )

        for sub in action.get("sub_actions") or []:
            self._index_action_specified(
                sub, terms, phrase_terms, word_terms,
                obj_id_to_template, add_term, add_attribute_pair, add_phrase,
                scale * 0.8,
            )

    def _index_condition_specified(
        self, cond, terms, phrase_terms, word_terms,
        obj_id_to_template, add_term, add_attribute_pair, prefix,
    ):
        obj_id = cond.get("object", "?")
        aspect = cond.get("aspect", "")
        attr = cond.get("attribute", "?")
        val = cond.get("value", "?")
        op = cond.get("op", "eq")
        template = obj_id_to_template.get(obj_id, obj_id)

        full = f"{obj_id}.{aspect}.{attr}" if aspect else f"{obj_id}.{attr}"

        if op == "add" or op == "+=":
            items = val if isinstance(val, list) else [val]
            for item in items:
                item_str = self._value_to_string(item)
                add_term(f"{prefix}:{full}+={item_str}", 1.0)
                add_attribute_pair(full, item_str, 0.8)
            return

        add_term(f"{prefix}:{full}={val}", 1.0)
        add_term(f"{prefix}:{attr}={val}", 1.0)
        add_attribute_pair(full, val, 0.5)
        add_attribute_pair(attr, val, 0.5)

    def _index_change_specified(
        self, ch, terms, phrase_terms, word_terms,
        obj_id_to_template, add_term, add_attribute_pair,
    ):
        obj_id = ch.get("object", "?")
        aspect = ch.get("aspect", "")
        attr = ch.get("attribute", "?")
        old = ch.get("old", "?")
        new = ch.get("new", "?")
        op = ch.get("op", "eq")
        template = obj_id_to_template.get(obj_id, obj_id)

        full = f"{obj_id}.{aspect}.{attr}" if aspect else f"{obj_id}.{attr}"

        if op == "add" or op == "+=":
            items = new if isinstance(new, list) else [new]
            for item in items:
                item_str = self._value_to_string(item)
                add_term(f"state_change:{full}+={item_str}", 1.0)
                add_attribute_pair(full, item_str, 0.8)
            return

        add_term(f"state_change:{full}:{old}->{new}", 1.0)
        add_term(f"state_change:{attr}:{new}", 1.0)
        add_attribute_pair(full, new, 0.5)
        add_attribute_pair(attr, new, 0.5)

    # ------------------------------------------------------------------
    # Generic recursive catch-all
    # ------------------------------------------------------------------
    def _extract_terms_catchall(self, memory: Dict[str, Any]):
        terms: Dict[str, float] = {}
        word_terms: Set[str] = set()

        def add_catchall_term(path: str, value: Any):
            if value is None:
                return
            value_str = self._value_to_string(value)
            if value_str == "":
                return

            final_key = path.rsplit(".", 1)[-1]
            if final_key == "unit":
                return
            if final_key in STRUCTURAL_KEYS:
                return
            if path.endswith(".position.relation") or path == "position.relation":
                return

            if value_str.lower() in STOP_WORDS or value_str.lower() in UNIT_WORDS:
                return
            if re.fullmatch(r"-?\d+(\.\d+)?", value_str):
                return

            term = f"{path}={value_str}"
            terms[term] = terms.get(term, 0.0) + 0.3

            if not path.startswith(("action:", "template:", "object:", "trajectory:",
                                    "category:", "function:", "precondition:",
                                    "state_change:", "goal_state:")):
                for word in tokenize_into_words(term):
                    if word and word not in STOP_WORDS and word not in UNIT_WORDS and len(word) > 1:
                        word_terms.add(word)

        def walk(obj, path_parts):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(v, path_parts + [str(k)])
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    walk(v, path_parts + [str(i)])
            else:
                add_catchall_term(".".join(path_parts), obj)

        walk(memory, [])
        return terms, word_terms

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        items = {}
        if not isinstance(d, dict):
            return items
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key, sep))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.update(self._flatten_dict(item, f"{new_key}.{i}", sep))
                    else:
                        items[f"{new_key}.{i}"] = item
            else:
                items[new_key] = v
        return items

    def _value_to_string(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return value.strip()
        return str(value)

    def _combine_value_unit(self, value: Any, unit: Any) -> str:
        if value is None:
            return ""
        value_str = self._value_to_string(value)
        if unit is not None and str(unit).strip():
            unit_str = str(unit).strip()
            if unit_str.lower() in UNIT_WORDS:
                return f"{value_str}{unit_str}"
        return value_str

    def _derive_volume(self, dim_values: Dict[str, str]) -> Optional[str]:
        """
        Derive a volume string from three dimension strings like:
        length, width, height_thickness.
        All must have the same compatible unit.
        Returns None if volume cannot be derived.
        """
        needed = ["length", "width", "height_thickness"]
        if not all(dim in dim_values for dim in needed):
            return None

        parsed = []
        unit = None
        for dim in needed:
            m = parse_measurement(dim_values[dim])
            if m is None:
                return None
            num, u = m
            if unit is None:
                unit = u if u else "unknown"
            elif u and unit and u != unit:
                return None
            parsed.append(num)

        if len(parsed) != 3:
            return None

        volume = parsed[0] * parsed[1] * parsed[2]

        if unit == "cm":
            return f"{volume}cm^3"
        elif unit == "m":
            return f"{volume}m^3"
        elif unit == "mm":
            return f"{volume}mm^3"
        elif unit == "unknown" or not unit:
            return str(volume)
        else:
            # unsupported unit
            return None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query_terms: List[Dict[str, float]],
        threshold: Optional[float] = None,
        progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Tuple[str, str, float, List[str]]]:
        detailed = self.search_detailed(query_terms, threshold=threshold, progress=progress)
        return [
            (m["memory_id"], m["activity"], m["total_score"], m["matched_terms"])
            for m in detailed["ranked_memories"]
        ]

    def search_detailed(
        self,
        query_terms: List[Dict[str, float]],
        threshold: Optional[float] = None,
        progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        if threshold is None:
            threshold = self.threshold

        scores = defaultdict(float)
        matched_terms_by_memory = defaultdict(set)
        memory_term_contributions = defaultdict(list)

        total_scanned = 0
        term_matches = []

        for q_idx, q in enumerate(query_terms, 1):
            term = q.get("term", "")
            weight = q.get("weight", 1.0)

            matched_entry_count = 0
            mode = "none"
            best_score = None
            matched_index_term = None
            scanned_this_term = 0
            contributions = defaultdict(float)

            # 1. Exact match
            if term in self.index:
                postings = self.index[term]
                matched_entry_count = len(postings)
                scanned_this_term = len(postings)
                total_scanned += scanned_this_term
                mode = "exact"
                matched_index_term = term
                for memory_id, mem_weight in postings:
                    contribution = weight * mem_weight
                    scores[memory_id] += contribution
                    matched_terms_by_memory[memory_id].add(term)
                    contributions[memory_id] += contribution
                    memory_term_contributions[memory_id].append(
                        {
                            "query_term": term,
                            "matched_index_term": term,
                            "mode": mode,
                            "score_contribution": contribution,
                        }
                    )

            # 2. Attribute-value parsed matching
            elif "=" in term:
                best_attr = self._fuzzy_match_attribute_value(term, threshold)
                if best_attr is not None:
                    best_index_term, best_sim = best_attr
                    postings = self.index.get(best_index_term, [])
                    matched_entry_count = len(postings)
                    scanned_this_term = len(self._attribute_pair_terms) + len(postings)
                    total_scanned += scanned_this_term
                    mode = "attribute"
                    best_score = best_sim
                    matched_index_term = best_index_term
                    for memory_id, mem_weight in postings:
                        contribution = weight * best_sim * mem_weight
                        scores[memory_id] += contribution
                        matched_terms_by_memory[memory_id].add(best_index_term)
                        contributions[memory_id] += contribution
                        memory_term_contributions[memory_id].append(
                            {
                                "query_term": term,
                                "matched_index_term": best_index_term,
                                "mode": mode,
                                "similarity": best_sim,
                                "score_contribution": contribution,
                            }
                        )

            # 3. Phrase-level fuzzy match
            if mode == "none" and self._phrase_embeddings is not None and self._phrase_terms:
                best_phrase = self._fuzzy_match_vector(
                    term, self._phrase_terms, self._phrase_embeddings, threshold
                )
                if best_phrase is not None:
                    best_term, sim = best_phrase
                    postings = self.index.get(best_term, [])
                    matched_entry_count = len(postings)
                    scanned_this_term = len(self._phrase_terms) + len(postings)
                    total_scanned += scanned_this_term
                    mode = "phrase"
                    best_score = sim
                    matched_index_term = best_term
                    for memory_id, mem_weight in postings:
                        contribution = weight * sim * mem_weight
                        scores[memory_id] += contribution
                        matched_terms_by_memory[memory_id].add(best_term)
                        contributions[memory_id] += contribution
                        memory_term_contributions[memory_id].append(
                            {
                                "query_term": term,
                                "matched_index_term": best_term,
                                "mode": mode,
                                "similarity": sim,
                                "score_contribution": contribution,
                            }
                        )

            # 4. Query-side phrase decomposition for phrase-like terms
            if mode == "none" and "=" not in term and term.startswith(PHRASE_QUERY_PREFIXES):
                best_decomp = self._fuzzy_match_phrase_words(term, threshold)
                if best_decomp is not None:
                    best_term, sim, postings = best_decomp
                    matched_entry_count = len(postings)
                    scanned_this_term = len(self._word_terms) + len(postings)
                    total_scanned += scanned_this_term
                    mode = "phrase_words"
                    best_score = sim
                    matched_index_term = best_term
                    for memory_id, mem_weight in postings:
                        contribution = weight * sim * mem_weight
                        scores[memory_id] += contribution
                        matched_terms_by_memory[memory_id].add(best_term)
                        contributions[memory_id] += contribution
                        memory_term_contributions[memory_id].append(
                            {
                                "query_term": term,
                                "matched_index_term": best_term,
                                "mode": mode,
                                "similarity": sim,
                                "score_contribution": contribution,
                            }
                        )

            # 5. Independent component matching for attribute=value terms
            if mode == "none" and "=" in term and not term.startswith(STRUCTURAL_QUERY_PREFIXES):
                attr_part, val_part = term.split("=", 1)
                attr_leaf = self._leaf_attribute(attr_part)
                value_clean = val_part.strip().lower()

                if not self._is_common_attribute(attr_leaf):
                    match_res = self._match_component_anywhere(attr_leaf, threshold, expected_type="attribute")
                    if match_res is not None:
                        matched_term, sim, postings, mtype = match_res
                        multiplier = 1.0 if mtype == "attribute" else 0.5
                        matched_entry_count += len(postings)
                        scanned_this_term += len(self._word_terms) + len(postings)
                        total_scanned += len(self._word_terms) + len(postings)
                        mode = "component_attribute"
                        best_score = sim if best_score is None else max(best_score, sim)
                        matched_index_term = matched_term
                        for memory_id, mem_weight in postings:
                            contribution = weight * multiplier * sim * mem_weight
                            scores[memory_id] += contribution
                            matched_terms_by_memory[memory_id].add(matched_term)
                            contributions[memory_id] += contribution
                            memory_term_contributions[memory_id].append(
                                {
                                    "query_term": term,
                                    "matched_index_term": matched_term,
                                    "mode": mode,
                                    "similarity": sim,
                                    "score_contribution": contribution,
                                }
                            )

                if not self._is_common_value(value_clean):
                    match_res = self._match_component_anywhere(value_clean, threshold, expected_type="value")
                    if match_res is not None:
                        matched_term, sim, postings, mtype = match_res
                        multiplier = 1.0 if mtype == "value" else 0.5
                        matched_entry_count += len(postings)
                        scanned_this_term += len(self._word_terms) + len(postings)
                        total_scanned += len(self._word_terms) + len(postings)
                        mode = "component_value"
                        best_score = sim if best_score is None else max(best_score, sim)
                        matched_index_term = matched_term
                        for memory_id, mem_weight in postings:
                            contribution = weight * multiplier * sim * mem_weight
                            scores[memory_id] += contribution
                            matched_terms_by_memory[memory_id].add(matched_term)
                            contributions[memory_id] += contribution
                            memory_term_contributions[memory_id].append(
                                {
                                    "query_term": term,
                                    "matched_index_term": matched_term,
                                    "mode": mode,
                                    "similarity": sim,
                                    "score_contribution": contribution,
                                }
                            )

            # 6. Word-level fuzzy match
            if mode == "none" and self._word_embeddings is not None and self._word_terms:
                if not term.startswith(STRUCTURAL_QUERY_PREFIXES):
                    best_word = self._fuzzy_match_vector(
                        term, self._word_terms, self._word_embeddings, threshold
                    )
                    if best_word is not None:
                        best_word_str, sim = best_word
                        postings = self._word_to_memories.get(best_word_str, [])
                        matched_entry_count = len(postings)
                        scanned_this_term = len(self._word_terms) + len(postings)
                        total_scanned += scanned_this_term
                        mode = "word"
                        best_score = sim
                        matched_index_term = best_word_str
                        for memory_id, mem_weight in postings:
                            contribution = weight * sim * mem_weight
                            scores[memory_id] += contribution
                            matched_terms_by_memory[memory_id].add(best_word_str)
                            contributions[memory_id] += contribution
                            memory_term_contributions[memory_id].append(
                                {
                                    "query_term": term,
                                    "matched_index_term": best_word_str,
                                    "mode": mode,
                                    "similarity": sim,
                                    "score_contribution": contribution,
                                }
                            )

            if mode == "none":
                scanned_this_term = 0

            if progress is not None:
                progress(
                    {
                        "index": q_idx,
                        "term": term,
                        "weight": weight,
                        "mode": mode,
                        "matched_entries": matched_entry_count,
                        "scanned_entries": scanned_this_term,
                        "cumulative_scanned_entries": total_scanned,
                        "best_score": best_score,
                    }
                )

            memory_scores = []
            for memory_id, contribution in contributions.items():
                activity = self.memory_by_id.get(memory_id, {}).get("activity", "Unknown")
                memory_scores.append(
                    {
                        "memory_id": memory_id,
                        "activity": activity,
                        "score": contribution,
                    }
                )
            memory_scores.sort(key=lambda x: x["score"], reverse=True)

            term_matches.append(
                {
                    "query_term": term,
                    "mode": mode,
                    "matched_index_term": matched_index_term,
                    "similarity": best_score,
                    "memory_scores": memory_scores,
                }
            )

        ranked_memories = []
        for memory_id, total_score in scores.items():
            memory = self.memory_by_id.get(memory_id, {})
            activity = memory.get("activity", "Unknown activity")
            matched = list(matched_terms_by_memory.get(memory_id, set()))
            term_contributions = memory_term_contributions.get(memory_id, [])

            for contrib in term_contributions:
                contrib["source_context"] = self.locate_term_in_memory(
                    memory_id, contrib.get("matched_index_term") or contrib.get("query_term")
                )

            ranked_memories.append(
                {
                    "memory_id": memory_id,
                    "activity": activity,
                    "total_score": total_score,
                    "matched_terms": matched,
                    "term_contributions": term_contributions,
                }
            )

        ranked_memories.sort(key=lambda x: x["total_score"], reverse=True)

        return {"term_matches": term_matches, "ranked_memories": ranked_memories}

    def _fuzzy_match_attribute_value(
        self, query_term: str, threshold: float
    ) -> Optional[Tuple[str, float]]:
        if "=" not in query_term:
            return None

        attr_part, val_part = query_term.split("=", 1)
        if not attr_part or val_part is None:
            return None

        query_path_parts = [p for p in re.split(r"[.:]", attr_part) if p]
        query_value = val_part.strip().lower()
        if not query_path_parts or query_value == "":
            return None

        best_term = None
        best_score = 0.0

        for cand in self._attribute_pair_terms:
            if "=" not in cand:
                continue
            cand_attr, cand_val = cand.split("=", 1)
            cand_path_parts = [p for p in re.split(r"[.:]", cand_attr) if p]

            path_sim = self._path_similarity(query_path_parts, cand_path_parts)
            if path_sim is None or path_sim < 0.8:
                continue

            cand_value = cand_val.strip().lower()

            if attr_part in MEASUREMENT_ATTRIBUTES and cand_attr in MEASUREMENT_ATTRIBUTES:
                if attr_part == "volume" and cand_attr == "volume":
                    if is_within_volume_range(query_value, cand_value):
                        return cand, 0.7
                elif is_within_range(query_value, cand_value):
                    return cand, 0.7

            if cand_value == query_value:
                return cand, 1.0

            sim = self._value_similarity(query_value, cand_value)
            if sim > best_score:
                best_score = sim
                best_term = cand

        if best_score >= threshold:
            return best_term, best_score
        return None

    def _path_similarity(self, query_parts: List[str], cand_parts: List[str]) -> Optional[float]:
        if self._is_contiguous_subsequence(query_parts, cand_parts):
            return 1.0

        if not query_parts or not cand_parts:
            return None

        bests = []
        for q in query_parts:
            best = 0.0
            for c in cand_parts:
                if q == c:
                    best = 1.0
                    break
                sim = self._value_similarity(q, c)
                if sim > best:
                    best = sim
            if best < 0.8:
                return None
            bests.append(best)

        if not bests:
            return None
        return sum(bests) / len(bests)

    def _is_contiguous_subsequence(self, small: List[str], large: List[str]) -> bool:
        if not small:
            return False
        m, n = len(small), len(large)
        for i in range(0, n - m + 1):
            if large[i : i + m] == small:
                return True
        return False

    def _value_similarity(self, val1: str, val2: str) -> float:
        if self._embedding_model is None:
            return 0.0
        try:
            if self._is_model2vec:
                e1 = self._embedding_model.encode([val1])[0]
                e2 = self._embedding_model.encode([val2])[0]
            else:
                e1 = self._embedding_model.encode(val1, normalize_embeddings=True)
                e2 = self._embedding_model.encode(val2, normalize_embeddings=True)
            e1 = np.asarray(e1, dtype="float32")
            e2 = np.asarray(e2, dtype="float32")
            n1 = np.linalg.norm(e1)
            n2 = np.linalg.norm(e2)
            if n1 == 0 or n2 == 0:
                return 0.0
            return float(np.dot(e1, e2) / (n1 * n2))
        except Exception:
            return 0.0

    def _fuzzy_match_vector(
        self,
        query_term: str,
        term_list: List[str],
        embedding_matrix: np.ndarray,
        threshold: float,
    ) -> Optional[Tuple[str, float]]:
        if embedding_matrix is None or not term_list:
            return None

        if self._is_model2vec:
            q_vec = self._embedding_model.encode([query_term])[0]
        else:
            q_vec = self._embedding_model.encode(query_term, normalize_embeddings=True)
        q_vec = np.asarray(q_vec).astype("float32")
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return None
        q_vec = q_vec / q_norm

        sims = embedding_matrix @ q_vec
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= threshold:
            return term_list[best_idx], best_score
        return None

    def _strip_phrase_prefix(self, term: str) -> Tuple[Optional[str], str]:
        for prefix in PHRASE_QUERY_PREFIXES:
            if term.startswith(prefix):
                return prefix, term[len(prefix):]
        return None, term

    def _fuzzy_match_phrase_words(self, term: str, threshold: float) -> Optional[Tuple[str, float, List[Tuple[str, float]]]]:
        prefix, phrase = self._strip_phrase_prefix(term)
        if not prefix and not phrase:
            return None

        words = [
            w for w in tokenize_into_words(phrase)
            if w not in STOP_WORDS and w not in UNIT_WORDS and len(w) > 1
        ]
        if not words:
            return None

        best_match = None
        for w in words:
            match_res = self._match_component_anywhere(w, threshold, expected_type=None)
            if match_res is not None:
                matched_term, sim, postings, mtype = match_res
                if best_match is None or sim > best_match[1]:
                    best_match = (matched_term, sim, postings)

        return best_match

    def _leaf_attribute(self, path: str) -> str:
        parts = [p for p in re.split(r"[.:]", path) if p]
        return parts[-1] if parts else path.strip().lower()

    def _is_common_attribute(self, attr: str) -> bool:
        return attr.lower() in NON_DECOMPOSABLE_ATTRIBUTES

    def _is_common_value(self, value: str) -> bool:
        if value in STOP_WORDS or value in UNIT_WORDS:
            return True
        if value in {"true", "false", "none", "current", "unknown"}:
            return True
        if re.fullmatch(r"-?\d+(\.\d+)?", value):
            return True
        return False

    def _match_component_anywhere(
        self,
        component: str,
        threshold: float,
        expected_type: Optional[str] = None,
    ) -> Optional[Tuple[str, float, List[Tuple[str, float]], str]]:
        """Find a component in word terms, phrase terms, or full index terms.

        Returns:
            matched_term, similarity, postings, matched_type
        """

        # 1. Exact word match
        if component in self._word_to_idx:
            postings = self._word_to_memories.get(component, [])
            return component, 1.0, postings, "word"

        # 2. Fuzzy word match
        if self._word_embeddings is not None and self._word_terms:
            best_word = self._fuzzy_match_vector(
                component, self._word_terms, self._word_embeddings, threshold
            )
            if best_word is not None:
                matched_word, sim = best_word
                postings = self._word_to_memories.get(matched_word, [])
                return matched_word, sim, postings, "word"

        # 3. Exact full term match in index
        if component in self.index:
            postings = self.index[component]
            term_type = self._classify_term(component)
            return component, 1.0, postings, term_type

        # 4. Fuzzy phrase match
        if self._phrase_embeddings is not None and self._phrase_terms:
            best_phrase = self._fuzzy_match_vector(
                component, self._phrase_terms, self._phrase_embeddings, threshold
            )
            if best_phrase is not None:
                matched_phrase, sim = best_phrase
                postings = self.index.get(matched_phrase, [])
                term_type = self._classify_term(matched_phrase)
                return matched_phrase, sim, postings, term_type

        return None

    def _classify_term(self, term: str) -> str:
        if term.startswith("action:"):
            return "action"
        if term.startswith("template:") or term.startswith("object:"):
            return "object"
        if term.startswith("trajectory:"):
            return "trajectory"
        if term.startswith("category:"):
            return "category"
        if term.startswith("function:"):
            return "function"
        if term.startswith("material:"):
            return "material"
        if term.startswith("precondition:"):
            return "attribute"
        if term.startswith("state_change:"):
            return "attribute"
        if term.startswith("goal_state:"):
            return "attribute"
        if "=" in term:
            return "attribute"
        return "word"

    # ------------------------------------------------------------------
    # Context location
    # ------------------------------------------------------------------
    def locate_term_in_memory(self, memory_id: str, term: str) -> Dict[str, Any]:
        memory = self.memory_by_id.get(memory_id, {})
        inner = memory.get("memory", {})
        objects = inner.get("objects") or []
        actions = inner.get("actions") or []

        if term.startswith("template:") or term.startswith("object:"):
            template = term.split(":", 1)[1]
            for obj in objects:
                if obj.get("template") == template:
                    return {
                        "type": "object",
                        "object_id": obj.get("obj_id"),
                        "template": template,
                    }
            return {"type": "object_template", "template": template}

        if term.startswith("action:"):
            action_name = term.split(":", 1)[1]
            result = self._find_action_context(actions, action_name)
            return result or {"type": "action_template", "template": action_name}

        if term.startswith("trajectory:"):
            traj = term.split(":", 1)[1]
            result = self._find_action_by_field(actions, "kinematic_trajectory", traj)
            return result or {"type": "trajectory", "trajectory": traj}

        if term.startswith("category:"):
            cat = term.split(":", 1)[1]
            return {"type": "category", "category": cat}

        if term.startswith("function:"):
            func = term.split(":", 1)[1]
            return {"type": "function", "function": func}

        if term.startswith("material:"):
            mat = term.split(":", 1)[1]
            return {"type": "material", "material": mat}

        if term.startswith("state_change:"):
            return self._locate_state_change(actions, term)

        if term.startswith("precondition:"):
            return self._locate_precondition(actions, term)

        if "=" in term:
            attr_part, val = term.split("=", 1)
            for obj in objects:
                if self._object_has_attribute(obj, attr_part, val):
                    return {
                        "type": "object",
                        "object_id": obj.get("obj_id"),
                        "template": obj.get("template"),
                        "attribute_path": attr_part,
                    }
            for act in actions:
                if self._action_has_attribute(act, attr_part, val):
                    return {
                        "type": "action",
                        "action_template": act.get("template"),
                        "attribute_path": attr_part,
                    }

        return {"type": "unknown", "term": term}

    # ... context location helper methods remain the same ...
    def _find_action_context(self, actions: List[Dict], action_name: str) -> Optional[Dict]:
        for act in actions:
            if act.get("template") == action_name:
                return {"type": "action", "action_template": action_name}
            found = self._find_action_context(act.get("sub_actions") or [], action_name)
            if found:
                return found
        return None

    def _find_action_by_field(self, actions: List[Dict], field: str, value: str) -> Optional[Dict]:
        for act in actions:
            if act.get(field) == value:
                return {"type": "action", "action_template": act.get("template"), field: value}
            found = self._find_action_by_field(act.get("sub_actions") or [], field, value)
            if found:
                return found
        return None

    def _object_has_attribute(self, obj: Dict, attr_part: str, val: str) -> bool:
        if str(obj.get(attr_part)) == val:
            return True
        for path, value in self._flatten_dict(obj).items():
            if path == attr_part and self._value_to_string(value) == val:
                return True
        if attr_part.startswith("dimensions."):
            dim_name = attr_part.rsplit(".", 1)[-1]
            dims = obj.get("dimensions") or {}
            if dim_name in dims:
                combined = self._combine_value_unit(
                    dims[dim_name].get("value"), dims[dim_name].get("unit")
                )
                if combined == val:
                    return True
        if attr_part in {"mass", "weight"}:
            for key in ("mass", "weight"):
                w = obj.get(key)
                if isinstance(w, dict):
                    combined = self._combine_value_unit(w.get("value"), w.get("unit"))
                    if combined == val:
                        return True
        return False

    def _action_has_attribute(self, action: Dict, attr_part: str, val: str) -> bool:
        for path, value in self._flatten_dict(action).items():
            if path == attr_part and self._value_to_string(value) == val:
                return True
        for ch in (
            action.get("changes", [])
            + action.get("changes_per_cycle", [])
            + action.get("changes_total", [])
        ):
            if isinstance(ch, dict):
                if str(ch.get("attribute")) == attr_part and self._value_to_string(ch.get("new")) == val:
                    return True
        return False

    def _locate_state_change(self, actions: List[Dict], term: str) -> Dict:
        rest = term.split(":", 1)[1] if ":" in term else ""
        if ":" not in rest:
            return {"type": "unknown", "term": term}
        path_part, change_part = rest.rsplit(":", 1)
        if "->" not in change_part:
            return {"type": "unknown", "term": term}
        old, new = change_part.split("->", 1)
        for act in actions:
            result = self._find_change_in_action(act, path_part, old, new)
            if result:
                return result
        return {"type": "unknown", "term": term}

    def _locate_precondition(self, actions: List[Dict], term: str) -> Dict:
        rest = term.split(":", 1)[1] if ":" in term else ""
        if "=" not in rest:
            return {"type": "unknown", "term": term}
        path_part, val = rest.split("=", 1)
        for act in actions:
            result = self._find_precondition_in_action(act, path_part, val)
            if result:
                return result
        return {"type": "unknown", "term": term}

    def _find_change_in_action(self, action: Dict, path_part: str, old: str, new: str) -> Optional[Dict]:
        for ch in (
            action.get("changes", [])
            + action.get("changes_per_cycle", [])
            + action.get("changes_total", [])
        ):
            if isinstance(ch, dict):
                full = f"{ch.get('object', '')}.{ch.get('aspect', '')}.{ch.get('attribute', '')}".strip(".")
                if full == path_part:
                    return {
                        "type": "action",
                        "action_template": action.get("template"),
                        "change": ch,
                    }
        for sub in action.get("sub_actions") or []:
            result = self._find_change_in_action(sub, path_part, old, new)
            if result:
                return result
        return None

    def _find_precondition_in_action(self, action: Dict, path_part: str, val: str) -> Optional[Dict]:
        for prec in action.get("preconditions") or []:
            if isinstance(prec, dict):
                full = f"{prec.get('object', '')}.{prec.get('aspect', '')}.{prec.get('attribute', '')}".strip(".")
                if full == path_part and self._value_to_string(prec.get("value")) == val:
                    return {
                        "type": "action",
                        "action_template": action.get("template"),
                        "precondition": prec,
                    }
        for sub in action.get("sub_actions") or []:
            result = self._find_precondition_in_action(sub, path_part, val)
            if result:
                return result
        return None

    # ------------------------------------------------------------------
    # Frequency export
    # ------------------------------------------------------------------
    def export_term_frequencies(self) -> Dict[str, Any]:
        """Return frequency counts for full index terms."""
        freq = {}
        for term, postings in self.index.items():
            freq[term] = {
                "count": len(postings),
                "type": self._classify_term(term),
            }
        return freq

    def save_term_frequencies(self, output_path: Optional[Path] = None):
        freq = self.export_term_frequencies()
        output_path = output_path or (Path(__file__).parent / "data" / "term_frequencies.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(freq, f, indent=2, ensure_ascii=False)
        print(f"💾 Term frequencies saved to {output_path}")
        return output_path

    def export_component_frequencies(self) -> Dict[str, Any]:
        """Return frequency counts for individual words/components."""
        freq = {}
        for word, postings in self._word_to_memories.items():
            freq[word] = {
                "count": len(postings),
                "type": "word",
            }

        # Add attribute component terms from attribute pair paths
        attr_freq = defaultdict(int)
        for term in self._attribute_pair_terms:
            if "=" in term:
                path = term.split("=", 0)[0]
                for comp in re.split(r"[.:]", path):
                    comp = comp.strip().lower()
                    if comp and comp not in STOP_WORDS and comp not in UNIT_WORDS:
                        attr_freq[comp] += 1

        for comp, count in attr_freq.items():
            if comp in freq:
                freq[comp]["attribute_count"] = count
            else:
                freq[comp] = {"count": count, "type": "attribute_component"}

        return freq

    def save_component_frequencies(self, output_path: Optional[Path] = None):
        freq = self.export_component_frequencies()
        output_path = output_path or (Path(__file__).parent / "data" / "component_frequencies.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(freq, f, indent=2, ensure_ascii=False)

        readable = output_path.with_suffix(".txt")
        with open(readable, "w", encoding="utf-8") as f:
            f.write("COMPONENT FREQUENCIES\n")
            f.write("=" * 80 + "\n\n")
            for comp, info in sorted(freq.items(), key=lambda x: x[1].get("count", 0), reverse=True):
                typ = info.get("type", "word")
                count = info.get("count", 0)
                attr_count = info.get("attribute_count")
                if attr_count is not None:
                    f.write(f"{comp:30s} count={count:>6d} attribute_count={attr_count:>6d}\n")
                else:
                    f.write(f"{comp:30s} count={count:>6d} type={typ}\n")

        print(f"💾 Component frequencies saved to {output_path} and {readable}")
        return output_path