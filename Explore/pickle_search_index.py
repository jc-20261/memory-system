#!/usr/bin/env python3
r"""
pickle_search_index.py – Complete updated reverse index with precomputed embeddings
and alias-to-base mapping.

This version implements:
- Full path structured entries: (action)(object)(attribute_path op value)
- Measurements list for dimensions+volume as one unit
- Goal state qualifier via `goalstate_terms` set
- Alias expansion with `_expanded` prefixes and tokens
- Object/action token indexing
- Verb indexing
- Participant as full path entry: (action)(attribute_path=("participants",), op="in", value=list)
- Conditional changes with statechangesC prefixes
- No index term weights (all postings are (memory, source_context, term_type))
- Frequency tracking
- Precomputed embeddings saved via EmbeddingCache
- Alias-to-base mapping so alias matches can be folded under original term
- Embedding groups include expanded object/action names for standalone fuzzy matching
- Banned attribute components include content/contents
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

from memory_model import Memory, Object, ActionInstance, GoalState
from embedding_cache import EmbeddingCache

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "fuzzy_threshold": 0.6,
    "max_core_score": 2.0,
    "object_exact_mult": 2.0,
    "object_fuzzy_mult": 1.6,
    "action_exact_mult": 2.0,
    "action_fuzzy_mult": 1.6,
    "alias_penalty": 0.9,
    "goalstate_multiplier": 4.0,
    "co_match_base": 1.2,
    "co_match_cap": 2.0,
    "freq_attr_comp_unmatchable": 0.003,
    "freq_attr_comp_one_match_min": 0.0003,
    "freq_attr_comp_one_match_max": 0.0007,
    "freq_token_unmatchable": 0.003,
    "freq_token_one_match_min": 0.0003,
    "freq_token_one_match_max": 0.0007,
    "diminishing_attr_comp": [1.0, 0.5, 0.1, 0.0],
    "diminishing_tokens": [1.0, 0.5, 0.1, 0.0],
    "diminishing_values": [1.0, 0.2, 0.0],
    "scalar_within_30pct": 0.7,
    "measurement_range": (0.5, 1.5),
    "dimension_mismatch_mult": 0.5,
    "volume_mismatch_mult": 0.5,
    "value_no_match_mult": 0.2,
    "verb_exact_score": 1.0,
    "verb_fuzzy_score": 0.9,
    "verb_target_cap": 4,
    "verb_context_cap": 2,
}

# =============================================================================
# UNIT CONVERSION TABLE
# =============================================================================
UNIT_CONVERSIONS = {
    "length": {
        "mm": 0.1, "millimeter": 0.1, "millimeters": 0.1,
        "cm": 1.0, "centimeter": 1.0, "centimeters": 1.0,
        "m": 100.0, "meter": 100.0, "meters": 100.0,
        "km": 100000.0,
    },
    "mass": {
        "mg": 0.001, "milligram": 0.001, "milligrams": 0.001,
        "g": 1.0, "gram": 1.0, "grams": 1.0,
        "kg": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0,
    },
    "volume": {
        "mm^3": 0.001, "mm3": 0.001,
        "cm^3": 1.0, "cm3": 1.0, "cc": 1.0,
        "ml": 1.0, "milliliter": 1.0, "milliliters": 1.0,
        "l": 1000.0, "liter": 1000.0, "liters": 1000.0,
        "m^3": 1000000.0, "m3": 1000000.0,
    },
    "temperature": {
        "c": 1.0, "celsius": 1.0,
        "f": None,  # special
        "k": None,
    },
    "time": {
        "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
        "min": 60.0, "minute": 60.0, "minutes": 60.0,
        "hr": 3600.0, "hour": 3600.0, "hours": 3600.0,
    },
    "angle": {
        "degree": 1.0, "degrees": 1.0,
        "radian": 57.2958, "radians": 57.2958,
    },
}

def convert_temperature(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit in ("c", "celsius"):
        return value
    elif unit in ("f", "fahrenheit"):
        return (value - 32) * 5.0 / 9.0
    elif unit in ("k", "kelvin"):
        return value - 273.15
    return value

def normalize_measurement(value: float, unit: str, quantity: str) -> float:
    unit = unit.lower()
    if quantity == "temperature":
        return convert_temperature(value, unit)
    table = UNIT_CONVERSIONS.get(quantity, {})
    factor = table.get(unit)
    if factor is None:
        return value
    return value * factor

def parse_measurement_string(s: str) -> Optional[Tuple[float, str]]:
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*([a-zA-Z^0-9%]*)\s*$", str(s))
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2).lower()
    return num, unit

def derive_quantity_from_unit(unit: str) -> str:
    if unit in UNIT_CONVERSIONS["length"]:
        return "length"
    if unit in UNIT_CONVERSIONS["mass"]:
        return "mass"
    if unit in UNIT_CONVERSIONS["volume"]:
        return "volume"
    if unit in UNIT_CONVERSIONS["temperature"]:
        return "temperature"
    if unit in UNIT_CONVERSIONS["time"]:
        return "time"
    if unit in UNIT_CONVERSIONS["angle"]:
        return "angle"
    return "unknown"

# =============================================================================
# PREFIXES & TERM TYPES
# =============================================================================
PREFIX_ATTRIBUTE_VALUE = "attribute_value"
PREFIX_ATTRIBUTE = "attribute"
PREFIX_VALUE = "value"
PREFIX_ATTRIBUTE_TOKEN = "attribute_token"
PREFIX_VALUE_TOKEN = "value_token"
PREFIX_OBJECT_TOKEN = "object_token"
PREFIX_ACTION_TOKEN = "action_token"
PREFIX_GOAL_TOKEN = "goal_token"
PREFIX_OTHER_TOKEN = "other_token"
PREFIX_FULL_PATH = "fullpath"
PREFIX_VERB = "verb"

PREFIX_ATTRIBUTE_EXPANDED = "attribute_expanded"
PREFIX_VALUE_EXPANDED = "value_expanded"
PREFIX_OBJECT_EXPANDED = "object_expanded"
PREFIX_ACTION_EXPANDED = "action_expanded"
PREFIX_ATTRIBUTE_EXPANDED_TOKEN = "attribute_expanded_token"
PREFIX_VALUE_EXPANDED_TOKEN = "value_expanded_token"
PREFIX_OBJECT_EXPANDED_TOKEN = "object_expanded_token"
PREFIX_ACTION_EXPANDED_TOKEN = "action_expanded_token"

# Conditional change prefixes
PREFIX_STATECHANGESC_ATTRIBUTE = "statechangesC_attribute"
PREFIX_STATECHANGESC_VALUE = "statechangesC_value"
PREFIX_STATECHANGESC_ATTRIBUTE_TOKEN = "statechangesC_attribute_token"
PREFIX_STATECHANGESC_VALUE_TOKEN = "statechangesC_value_token"
PREFIX_STATECHANGESC_ATTRIBUTE_EXPANDED = "statechangesC_attribute_expanded"
PREFIX_STATECHANGESC_VALUE_EXPANDED = "statechangesC_value_expanded"
PREFIX_STATECHANGESC_ATTRIBUTE_EXPANDED_TOKEN = "statechangesC_attribute_expanded_token"
PREFIX_STATECHANGESC_VALUE_EXPANDED_TOKEN = "statechangesC_value_expanded_token"

ALLOWED_PREFIXES = {
    PREFIX_ATTRIBUTE_VALUE, PREFIX_ATTRIBUTE, PREFIX_VALUE,
    PREFIX_ATTRIBUTE_TOKEN, PREFIX_VALUE_TOKEN, PREFIX_OBJECT_TOKEN,
    PREFIX_ACTION_TOKEN, PREFIX_GOAL_TOKEN, PREFIX_OTHER_TOKEN,
    PREFIX_FULL_PATH,
    PREFIX_ATTRIBUTE_EXPANDED, PREFIX_VALUE_EXPANDED,
    PREFIX_OBJECT_EXPANDED, PREFIX_ACTION_EXPANDED,
    PREFIX_ATTRIBUTE_EXPANDED_TOKEN, PREFIX_VALUE_EXPANDED_TOKEN,
    PREFIX_OBJECT_EXPANDED_TOKEN, PREFIX_ACTION_EXPANDED_TOKEN,
    PREFIX_VERB,
    PREFIX_STATECHANGESC_ATTRIBUTE, PREFIX_STATECHANGESC_VALUE,
    PREFIX_STATECHANGESC_ATTRIBUTE_TOKEN, PREFIX_STATECHANGESC_VALUE_TOKEN,
    PREFIX_STATECHANGESC_ATTRIBUTE_EXPANDED, PREFIX_STATECHANGESC_VALUE_EXPANDED,
    PREFIX_STATECHANGESC_ATTRIBUTE_EXPANDED_TOKEN, PREFIX_STATECHANGESC_VALUE_EXPANDED_TOKEN,
}

def get_term_prefix(term: str) -> str:
    if ":" not in term:
        return "other"
    return term.split(":", 1)[0]

# =============================================================================
# STOP WORDS & UNIT WORDS
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
    "templates", "aliases", "categories", "functions", "materials",
    "relationr", "relative_to", "content", "contents", "state", "surface"
}

UNIT_WORDS = set()
for _table in UNIT_CONVERSIONS.values():
    for _unit in _table.keys():
        UNIT_WORDS.add(_unit)
UNIT_WORDS.update({"percent", "%", "cm", "m", "mm", "km", "g", "kg", "mg",
                   "ml", "l", "s", "sec", "min", "hr", "hour", "hours",
                   "degree", "degrees"})

def tokenize_into_words(text: str) -> List[str]:
    if not isinstance(text, str):
        text = str(text)
    separators = "_:=->./\\()[]{} '\""
    for sep in separators:
        text = text.replace(sep, " ")
    return [w.lower().strip() for w in text.split() if w.strip()]

def is_valid_token(word: str) -> bool:
    if not word:
        return False
    if word in STOP_WORDS or word in UNIT_WORDS:
        return False
    if re.fullmatch(r"\d+", word):
        return False
    if len(word) <= 1:
        return False
    return True

# =============================================================================
# BANNED ATTRIBUTE COMPONENTS
# =============================================================================
BANNED_ATTRIBUTE_COMPONENTS = {
    "length", "width", "height_thickness", "height", "mass", "weight",
    "shape", "dimensions", "volume", "duration", "repetitions",
    "cycle_duration", "start_offset", "temporal_type", "number",
    "position", "position_relative_body", "material", "materials",
    "category", "categories", "function", "functions",
    "relation", "relative_to", "relationr",
    "content", "contents", "state", "participants", "participant", "color", "surface"
}

SPATIAL_OPERATORS = {
    "on", "in", "below", "above", "around", "through",
    "contact", "attached", "between", "near", "apart"
}

# =============================================================================
# INDEX CLASS
# =============================================================================
class PickleSearchIndex:
    def __init__(self, memories: List[Memory], object_storage: Dict[str, Any],
                 alias_expansion: Dict[str, Any]):
        self.memories = memories
        self.object_storage = object_storage
        self.alias_expansion = alias_expansion

        # Reverse index: term -> list of (memory, source_context, term_type)
        self.index: Dict[str, List[Tuple[Memory, Dict[str, Any], str]]] = defaultdict(list)

        # Structured full-path entries
        self.full_path_entries: List[Dict[str, Any]] = []

        # Measurements for dimensions+volume unit matching
        self.measurements: List[Dict[str, Any]] = []

        # Goalstate term set for multiplier detection
        self.goalstate_terms: Set[str] = set()

        # Alias-to-base mapping: alias_term -> base_term
        self.alias_to_base: Dict[str, str] = {}

        # Frequency tracking
        self.term_occurrences: Dict[str, int] = defaultdict(int)
        self.term_memory_sets: Dict[str, Set[str]] = defaultdict(set)

        # Embedding model
        self._embedding_model = None
        self._is_model2vec = False
        self.phrase_terms: Set[str] = set()
        self.word_terms: Set[str] = set()
        self._phrase_embeddings = None
        self._word_embeddings = None
        self._word_to_memories: Dict[str, List[Tuple[Memory, Dict, str]]] = defaultdict(list)

        if MODEL2VEC_AVAILABLE:
            try:
                self._embedding_model = StaticModel.from_pretrained("minishlab/potion-base-8M")
                self._is_model2vec = True
                print("Using Model2Vec for fuzzy matching.")
            except Exception as e:
                print(f"Model2Vec failed: {e}")
        if self._embedding_model is None and ST_AVAILABLE:
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("Using SentenceTransformer for fuzzy matching.")
        elif self._embedding_model is None:
            print("No embedding model available; fuzzy matching disabled.")

        self._build_index()
        self._precompute_embeddings()
        self._save_frequency_data()

    def _add_term(self, term: str, memory: Memory,
                  source_context: Dict[str, Any], term_type: str,
                  is_goalstate: bool = False):
        prefix = get_term_prefix(term)
        if prefix not in ALLOWED_PREFIXES:
            return
        self.index[term].append((memory, source_context, term_type))
        if is_goalstate:
            self.goalstate_terms.add(term)
        self.term_occurrences[term] += 1
        self.term_memory_sets[term].add(memory.id)
        phrase_part = term.split(":",1)[-1] if ":" in term else term
        self.phrase_terms.add(phrase_part)
        for word in tokenize_into_words(phrase_part):
            if is_valid_token(word):
                self.word_terms.add(word)
                self._word_to_memories[word].append((memory, source_context, term_type))

    def _add_full_path_entry(self, entry: Dict[str, Any]):
        self.full_path_entries.append(entry)

    def _add_measurement(self, meas: Dict[str, Any]):
        self.measurements.append(meas)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def _build_index(self):
        for mem in self.memories:
            self._index_memory(mem)

    def _index_memory(self, memory: Memory):
        for obj in memory.objects.values():
            self._index_object(memory, obj)
        for action in memory.actions:
            self._index_action(memory, action)
        for goal in memory.goal_states:
            self._index_goal_state(memory, goal)

    # ------------------------------------------------------------------
    # Object indexing
    # ------------------------------------------------------------------
    def _index_object(self, memory: Memory, obj: Object):
        src = {"type": "object", "object_id": obj.obj_id, "template": obj.template_name}

        # Overrides: full path entries and component/value terms
        overrides = obj.overrides or {}
        for path_tuple, val in self._flatten_with_path(overrides):
            self._add_full_path_from_object(memory, obj, path_tuple, "eq", val, src)

        # Dimensions
        dims = obj.dimensions or {}
        dim_values = {}
        for dim in ("length", "width", "height_thickness"):
            if dim in dims and isinstance(dims[dim], dict):
                val = dims[dim].get("value")
                unit = dims[dim].get("unit", "")
                value_str = f"{val}{unit}" if unit else str(val)
                path = ("dimensions", dim)
                self._add_full_path_from_object(memory, obj, path, "eq", value_str, src)
                parsed = parse_measurement_string(value_str)
                if parsed:
                    self._add_measurement({
                        "memory": memory,
                        "object": obj.obj_id,
                        "path": list(path),
                        "numeric": parsed[0],
                        "unit": parsed[1],
                        "source_context": src,
                    })
                dim_values[dim] = value_str

        # Derived volume
        if all(k in dim_values for k in ("length", "width", "height_thickness")):
            try:
                l = parse_measurement_string(dim_values["length"])
                w = parse_measurement_string(dim_values["width"])
                h = parse_measurement_string(dim_values["height_thickness"])
                if l and w and h:
                    l_norm = normalize_measurement(l[0], l[1], "length")
                    w_norm = normalize_measurement(w[0], w[1], "length")
                    h_norm = normalize_measurement(h[0], h[1], "length")
                    vol = l_norm * w_norm * h_norm
                    vol_str = f"{vol}cm^3"
                    path = ("volume",)
                    self._add_full_path_from_object(memory, obj, path, "eq", vol_str, src)
                    self._add_measurement({
                        "memory": memory,
                        "object": obj.obj_id,
                        "path": ["volume"],
                        "numeric": vol,
                        "unit": "cm^3",
                        "source_context": src,
                    })
            except:
                pass

        # Weight/mass
        weight = obj.weight or obj.mass
        if weight and isinstance(weight, dict):
            val = weight.get("value")
            unit = weight.get("unit", "")
            value_str = f"{val}{unit}" if unit else str(val)
            self._add_full_path_from_object(memory, obj, ("weight",), "eq", value_str, src)

        # Position unit
        if obj.position:
            rel = obj.position.get("relation", "")
            rel_r = obj.position.get("relationr", "")
            rel_to = obj.position.get("relative_to", "")
            pos_term = f"position_unit:{obj.obj_id}:{rel}:{rel_r}:{rel_to}"
            self._add_term(pos_term, memory, src, "position")

        # Object terms
        base_object_term = f"object:{obj.template_name}"
        self._add_term(base_object_term, memory, src, "object")
        self._add_term(f"object:{obj.obj_id}", memory, src, "object")
        for token in tokenize_into_words(obj.template_name):
            if is_valid_token(token):
                self._add_term(f"{PREFIX_OBJECT_TOKEN}:{token}", memory, src, "object")
        for token in tokenize_into_words(obj.obj_id):
            if is_valid_token(token):
                self._add_term(f"{PREFIX_OBJECT_TOKEN}:{token}", memory, src, "object")
        for alias in obj.aliases:
            self._add_term(f"alias:{alias}", memory, src, "object")
            self._add_term(f"object:{alias}", memory, src, "object")
            for token in tokenize_into_words(alias):
                if is_valid_token(token):
                    self._add_term(f"{PREFIX_OBJECT_TOKEN}:{token}", memory, src, "object")
        for cat in obj.categories:
            self._add_term(f"category:{cat}", memory, src, "object")
        for func in obj.functions:
            self._add_term(f"function:{func}", memory, src, "object")
        for mat in obj.materials:
            self._add_term(f"material:{mat}", memory, src, "object")

        # Object aliases from expansion
        obj_aliases = self.alias_expansion.get("object_aliases", {})
        for alias in obj_aliases.get(obj.template_name, []):
            alias_term = f"object_expanded:{alias}"
            self._add_term(alias_term, memory, src, "object")
            self.alias_to_base[alias_term] = base_object_term
            for token in tokenize_into_words(alias):
                if is_valid_token(token):
                    token_term = f"object_expanded_token:{token}"
                    self._add_term(token_term, memory, src, "object")
                    self.alias_to_base[token_term] = base_object_term

    def _flatten_with_path(self, d: Dict, prefix: Tuple[str,...] = None):
        if prefix is None:
            prefix = ()
        result = []
        for k, v in d.items():
            new_prefix = prefix + (str(k),)
            if isinstance(v, dict):
                if "value" in v and set(v.keys()) <= {"value","unit"}:
                    val = v.get("value")
                    unit = v.get("unit","")
                    result.append((new_prefix, f"{val}{unit}" if unit else str(val)))
                else:
                    result.extend(self._flatten_with_path(v, new_prefix))
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        result.extend(self._flatten_with_path(item, new_prefix))
                    else:
                        result.append((new_prefix, item))
            else:
                result.append((new_prefix, v))
        return result

    def _add_full_path_from_object(self, memory, obj, path_tuple, op, value, src):
        entry = {
            "memory": memory,
            "action": None,
            "object": obj.obj_id,
            "path": path_tuple,
            "op": op,
            "value": value,
            "state_change_type": None,
            "source_context": src,
            "source_type": "regular",
        }
        self._add_full_path_entry(entry)
        self._index_attribute_path_components(path_tuple, memory, src)
        if isinstance(value, str) and not self._is_numeric(value):
            self._index_value_component(value, memory, src)

    def _is_numeric(self, s: str) -> bool:
        return bool(re.fullmatch(r"-?\d+(\.\d+)?", s))

    def _index_attribute_path_components(self, path_tuple, memory, src,
                                         is_goalstate=False, conditional=False):
        for comp in path_tuple:
            comp = str(comp)
            if not is_valid_token(comp) or comp in BANNED_ATTRIBUTE_COMPONENTS:
                continue
            # Determine prefixes based on conditional and alias
            if conditional:
                attr_prefix = PREFIX_STATECHANGESC_ATTRIBUTE
                attr_token_prefix = PREFIX_STATECHANGESC_ATTRIBUTE_TOKEN
                attr_expanded_prefix = PREFIX_STATECHANGESC_ATTRIBUTE_EXPANDED
                attr_expanded_token_prefix = PREFIX_STATECHANGESC_ATTRIBUTE_EXPANDED_TOKEN
            else:
                attr_prefix = PREFIX_ATTRIBUTE
                attr_token_prefix = PREFIX_ATTRIBUTE_TOKEN
                attr_expanded_prefix = PREFIX_ATTRIBUTE_EXPANDED
                attr_expanded_token_prefix = PREFIX_ATTRIBUTE_EXPANDED_TOKEN

            base_attr_term = f"{attr_prefix}:{comp}"
            self._add_term(base_attr_term, memory, src, "attribute", is_goalstate=is_goalstate)
            for token in tokenize_into_words(comp):
                if is_valid_token(token):
                    base_token = f"{attr_token_prefix}:{token}"
                    self._add_term(base_token, memory, src, "attribute", is_goalstate=is_goalstate)
            attr_aliases = self.alias_expansion.get("attribute_component_aliases", {})
            for alias in attr_aliases.get(comp, []):
                alias_term = f"{attr_expanded_prefix}:{alias}"
                self._add_term(alias_term, memory, src, "attribute", is_goalstate=is_goalstate)
                self.alias_to_base[alias_term] = base_attr_term
                for token in tokenize_into_words(alias):
                    if is_valid_token(token):
                        alias_token_term = f"{attr_expanded_token_prefix}:{token}"
                        self._add_term(alias_token_term, memory, src, "attribute",
                                       is_goalstate=is_goalstate)
                        self.alias_to_base[alias_token_term] = base_attr_term

    def _index_value_component(self, val: str, memory, src,
                               is_goalstate=False, conditional=False):
        if not is_valid_token(val):
            return
        if conditional:
            val_prefix = PREFIX_STATECHANGESC_VALUE
            val_token_prefix = PREFIX_STATECHANGESC_VALUE_TOKEN
            val_expanded_prefix = PREFIX_STATECHANGESC_VALUE_EXPANDED
            val_expanded_token_prefix = PREFIX_STATECHANGESC_VALUE_EXPANDED_TOKEN
        else:
            val_prefix = PREFIX_VALUE
            val_token_prefix = PREFIX_VALUE_TOKEN
            val_expanded_prefix = PREFIX_VALUE_EXPANDED
            val_expanded_token_prefix = PREFIX_VALUE_EXPANDED_TOKEN

        base_value_term = f"{val_prefix}:{val}"
        self._add_term(base_value_term, memory, src, "value", is_goalstate=is_goalstate)
        for token in tokenize_into_words(val):
            if is_valid_token(token):
                self._add_term(f"{val_token_prefix}:{token}", memory, src, "value",
                               is_goalstate=is_goalstate)
        value_aliases = self.alias_expansion.get("value_aliases", {})
        for alias in value_aliases.get(val, []):
            alias_term = f"{val_expanded_prefix}:{alias}"
            self._add_term(alias_term, memory, src, "value", is_goalstate=is_goalstate)
            self.alias_to_base[alias_term] = base_value_term
            for token in tokenize_into_words(alias):
                if is_valid_token(token):
                    alias_token_term = f"{val_expanded_token_prefix}:{token}"
                    self._add_term(alias_token_term, memory, src, "value",
                                   is_goalstate=is_goalstate)
                    self.alias_to_base[alias_token_term] = base_value_term

    # ------------------------------------------------------------------
    # Action indexing
    # ------------------------------------------------------------------
    def _index_action(self, memory, action: ActionInstance):
        src = {
            "type": "action",
            "action_instance_id": action.instance_id,
            "template": action.template_name,
            "general_template": action.general_template.name if action.general_template else None,
            "alt": action.alt.name if action.alt else None,
        }

        # Action names and verbs
        base_action_term = f"action:{action.template_name}"
        self._add_term(base_action_term, memory, src, "action")
        if action.general_template:
            self._add_term(f"action:{action.general_template.name}", memory, src, "action")
            if action.general_template.verb:
                self._add_term(f"verb:{action.general_template.verb.name}", memory, src, "verb")
                for token in tokenize_into_words(action.general_template.verb.name):
                    if is_valid_token(token):
                        self._add_term(f"{PREFIX_ACTION_TOKEN}:{token}", memory, src, "action")
        if action.alt:
            self._add_term(f"action:{action.alt.name}", memory, src, "action")
            if action.alt.verb:
                self._add_term(f"verb:{action.alt.verb.name}", memory, src, "verb")
                for token in tokenize_into_words(action.alt.verb.name):
                    if is_valid_token(token):
                        self._add_term(f"{PREFIX_ACTION_TOKEN}:{token}", memory, src, "action")

        # Token terms for action names
        for name in (action.template_name,
                     action.general_template.name if action.general_template else None,
                     action.alt.name if action.alt else None):
            if name:
                for token in tokenize_into_words(name):
                    if is_valid_token(token):
                        self._add_term(f"{PREFIX_ACTION_TOKEN}:{token}", memory, src, "action")

        # Action aliases
        action_aliases = self.alias_expansion.get("action_aliases", {})
        for alias in action_aliases.get(action.template_name, []):
            alias_term = f"action_expanded:{alias}"
            self._add_term(alias_term, memory, src, "action")
            self.alias_to_base[alias_term] = base_action_term
            for token in tokenize_into_words(alias):
                if is_valid_token(token):
                    token_term = f"action_expanded_token:{token}"
                    self._add_term(token_term, memory, src, "action")
                    self.alias_to_base[token_term] = base_action_term

        # Participants as full path entry
        if action.participants:
            participant_ids = [p.obj_id for p in action.participants if p and p.obj_id]
            if participant_ids:
                entry = {
                    "memory": memory,
                    "action": action.template_name,
                    "object": None,
                    "path": ("participants",),
                    "op": "in",
                    "value": participant_ids,
                    "state_change_type": None,
                    "source_context": src,
                    "source_type": "regular",
                }
                self._add_full_path_entry(entry)

        # Preconditions
        for prec in action.preconditions or []:
            if isinstance(prec, dict):
                self._index_condition_entry(memory, action, prec, "precondition", src)

        # Changes (old and new)
        for change_list in (action.changes, getattr(action, 'changes_per_cycle', []),
                            getattr(action, 'changes_total', [])):
            for ch in change_list or []:
                if isinstance(ch, dict):
                    self._index_change_entry(memory, action, ch, "new", src)
                    self._index_change_entry(memory, action, ch, "old", src)

        # Conditional changes
        for cond in getattr(action, 'conditional_changes', []) or []:
            if isinstance(cond, dict):
                for ch in cond.get("changes", []) or []:
                    if isinstance(ch, dict):
                        self._index_change_entry(memory, action, ch, "conditional_new", src)
                        self._index_change_entry(memory, action, ch, "conditional_old", src)

        # Sub-actions
        for sub in action.sub_actions or []:
            self._index_action(memory, sub)

    def _index_condition_entry(self, memory, action, entry, state_type, src):
        obj = entry.get("object")
        aspect = entry.get("aspect")
        attribute = entry.get("attribute")
        value = entry.get("value")
        op = entry.get("op", "eq")
        if obj and attribute and value is not None:
            path_tuple = self._build_path_from_aspect_attribute(aspect, attribute)
            full_entry = {
                "memory": memory,
                "action": action.template_name,
                "object": str(obj),
                "path": path_tuple,
                "op": str(op),
                "value": value,
                "state_change_type": state_type,
                "source_context": src,
                "source_type": "regular",
            }
            self._add_full_path_entry(full_entry)
            conditional = state_type.startswith("conditional_")
            self._index_attribute_path_components(path_tuple, memory, src,
                                                  conditional=conditional)
            if isinstance(value, str) and not self._is_numeric(value):
                self._index_value_component(value, memory, src,
                                            conditional=conditional)

    def _index_change_entry(self, memory, action, entry, state_type, src):
        obj = entry.get("object")
        aspect = entry.get("aspect")
        attribute = entry.get("attribute")
        value = entry.get("new") if state_type in ("new","conditional_new") else entry.get("old")
        op = entry.get("op", "eq")
        if obj and attribute and value is not None:
            path_tuple = self._build_path_from_aspect_attribute(aspect, attribute)
            full_entry = {
                "memory": memory,
                "action": action.template_name,
                "object": str(obj),
                "path": path_tuple,
                "op": str(op),
                "value": value,
                "state_change_type": state_type,
                "source_context": src,
                "source_type": "regular",
            }
            self._add_full_path_entry(full_entry)
            conditional = state_type.startswith("conditional_")
            self._index_attribute_path_components(path_tuple, memory, src,
                                                  conditional=conditional)
            if isinstance(value, str) and not self._is_numeric(value):
                self._index_value_component(value, memory, src,
                                            conditional=conditional)

    def _build_path_from_aspect_attribute(self, aspect, attribute):
        comps = []
        if aspect:
            if isinstance(aspect, str):
                comps.append(aspect)
            elif isinstance(aspect, (dict,list)):
                self._collect_keys(aspect, comps)
        if attribute:
            if isinstance(attribute, str):
                comps.append(attribute)
            elif isinstance(attribute, list):
                for item in attribute:
                    if isinstance(item, str):
                        comps.append(item)
                    elif isinstance(item, dict):
                        self._collect_keys(item, comps)
            elif isinstance(attribute, dict):
                self._collect_keys(attribute, comps)
        return tuple(comps)

    def _collect_keys(self, node, comps):
        if isinstance(node, str):
            comps.append(node)
        elif isinstance(node, dict):
            for k, v in node.items():
                comps.append(str(k))
                if isinstance(v, dict):
                    self._collect_keys(v, comps)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            self._collect_keys(item, comps)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, str):
                    comps.append(item)
                elif isinstance(item, dict):
                    self._collect_keys(item, comps)

    # ------------------------------------------------------------------
    # Goal state indexing
    # ------------------------------------------------------------------
    def _index_goal_state(self, memory, goal: GoalState):
        obj = goal.target_object
        src = {"type": "goal_state", "object_id": obj.obj_id, "template": obj.template_name}
        attr = goal.attribute
        if isinstance(attr, str):
            path_tuple = (attr,)
        else:
            path_tuple = tuple(attr) if isinstance(attr, (list,tuple)) else (str(attr),)
        value = goal.value
        op = goal.op

        full_entry = {
            "memory": memory,
            "action": None,
            "object": obj.obj_id,
            "path": path_tuple,
            "op": op,
            "value": value,
            "state_change_type": "goal",
            "source_context": src,
            "source_type": "goalstate",
        }
        self._add_full_path_entry(full_entry)

        self._index_attribute_path_components(path_tuple, memory, src, is_goalstate=True)
        if isinstance(value, str) and not self._is_numeric(value):
            self._index_value_component(value, memory, src, is_goalstate=True)

    # ------------------------------------------------------------------
    # Precomputed embeddings
    # ------------------------------------------------------------------
    def _precompute_embeddings(self):
        if self._embedding_model is None:
            return

        cache = EmbeddingCache()
        groups = self._collect_embedding_groups()

        for group_name, terms in groups.items():
            if not terms:
                continue
            terms_sorted = sorted(terms)
            cached = cache.load_group(group_name)
            if cached is not None and cached[0] == terms_sorted:
                print(f"✅ Embedding cache valid for {group_name}")
                continue

            phrases = [self._term_to_phrase(t) for t in terms_sorted]
            embeddings = self._embed_terms(phrases)
            cache.save_group(group_name, terms_sorted, embeddings)
            print(f"💾 Precomputed embeddings saved for {group_name} ({len(terms_sorted)} terms)")

    def _collect_embedding_groups(self) -> Dict[str, List[str]]:
        groups = {
            "component_attribute": [],
            "component_value": [],
            "token_attribute": [],
            "token_value": [],
            "token_object": [],
            "token_action": [],
            "verb": [],
            "object_names": [],
            "action_names": [],
        }
        for term in self.index.keys():
            prefix = get_term_prefix(term)
            if prefix in (PREFIX_ATTRIBUTE, PREFIX_ATTRIBUTE_EXPANDED,
                          PREFIX_STATECHANGESC_ATTRIBUTE,
                          PREFIX_STATECHANGESC_ATTRIBUTE_EXPANDED):
                groups["component_attribute"].append(term)
            elif prefix in (PREFIX_VALUE, PREFIX_VALUE_EXPANDED,
                            PREFIX_STATECHANGESC_VALUE,
                            PREFIX_STATECHANGESC_VALUE_EXPANDED):
                groups["component_value"].append(term)
            elif prefix in (PREFIX_ATTRIBUTE_TOKEN, PREFIX_ATTRIBUTE_EXPANDED_TOKEN,
                            PREFIX_STATECHANGESC_ATTRIBUTE_TOKEN,
                            PREFIX_STATECHANGESC_ATTRIBUTE_EXPANDED_TOKEN):
                groups["token_attribute"].append(term)
            elif prefix in (PREFIX_VALUE_TOKEN, PREFIX_VALUE_EXPANDED_TOKEN,
                            PREFIX_STATECHANGESC_VALUE_TOKEN,
                            PREFIX_STATECHANGESC_VALUE_EXPANDED_TOKEN):
                groups["token_value"].append(term)
            elif prefix in (PREFIX_OBJECT_TOKEN, PREFIX_OBJECT_EXPANDED_TOKEN):
                groups["token_object"].append(term)
            elif prefix in (PREFIX_ACTION_TOKEN, PREFIX_ACTION_EXPANDED_TOKEN):
                groups["token_action"].append(term)
            elif prefix == PREFIX_VERB:
                groups["verb"].append(term)
            elif prefix in ("object", "object_expanded"):
                groups["object_names"].append(term)
            elif prefix in ("action", "action_expanded"):
                groups["action_names"].append(term)
        return groups

    def _term_to_phrase(self, term: str) -> str:
        if ":" in term:
            phrase = term.split(":", 1)[1]
        else:
            phrase = term
        return phrase

    def _embed_terms(self, terms: List[str]) -> np.ndarray:
        if self._is_model2vec:
            vecs = self._embedding_model.encode(terms)
        else:
            vecs = self._embedding_model.encode(terms, normalize_embeddings=True)
        vecs = np.asarray(vecs).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    # ------------------------------------------------------------------
    # Frequency data
    # ------------------------------------------------------------------
    def _save_frequency_data(self):
        freq_dir = Path(__file__).parent / "data"
        freq_dir.mkdir(parents=True, exist_ok=True)
        freq_path = freq_dir / "searchable_term_frequencies_occurrences.json"
        rows = []
        total_occ = sum(self.term_occurrences.values())
        total_mem = len(self.memories)
        for term, occ in self.term_occurrences.items():
            mem_count = len(self.term_memory_sets[term])
            occ_pct = (occ / total_occ * 100) if total_occ else 0.0
            mem_pct = (mem_count / total_mem * 100) if total_mem else 0.0
            rows.append({
                "term": term,
                "prefix": get_term_prefix(term),
                "occurrences": occ,
                "memory_count": mem_count,
                "occurrence_percentage": round(occ_pct, 6),
                "memory_percentage": round(mem_pct, 6),
            })
        rows.sort(key=lambda x: -x["occurrences"])
        with open(freq_path, "w", encoding="utf-8") as f:
            json.dump({"total_memories": total_mem, "total_occurrences": total_occ, "terms": rows},
                      f, indent=2, ensure_ascii=False)