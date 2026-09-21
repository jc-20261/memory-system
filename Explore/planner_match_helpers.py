#!/usr/bin/env python3
r"""
planner_match_helpers.py – Matching primitives for the planning module.

Copies the relevant matching logic from pickle_search_engine.py so that the
planning module does not depend on the search engine at runtime. The two
modules share only the on-disk pickle files.

Provides:
    PlannerMatchHelpers
        - embedding group loading (same cache as the search engine)
        - fuzzy matching helpers (similarity_to_specific_term, similarities_to_group)
        - value / path / operator match scoring
        - object_compatibility for object substitution
        - find_best_object_substitute
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pickle_search_index import (
    CONFIG,
    BANNED_ATTRIBUTE_COMPONENTS,
    STOP_WORDS,
    UNIT_WORDS,
    normalize_measurement,
    parse_measurement_string,
    derive_quantity_from_unit,
    tokenize_into_words,
    is_valid_token,
    PREFIX_ATTRIBUTE,
    PREFIX_ATTRIBUTE_EXPANDED,
    PREFIX_ATTRIBUTE_TOKEN,
    PREFIX_ATTRIBUTE_EXPANDED_TOKEN,
    PREFIX_VALUE,
    PREFIX_VALUE_EXPANDED,
    PREFIX_VALUE_TOKEN,
    PREFIX_VALUE_EXPANDED_TOKEN,
)

from embedding_cache import EmbeddingCache


# =============================================================================
# Configuration for object substitution
# =============================================================================
OBJECT_COMPAT_CONFIG = {
    # Signal weights (defining attributes). All are equal for now.
    "weight_category":    1.0,
    "weight_dimensions":  1.0,
    "weight_materials":   1.0,
    "weight_shape":       1.0,
    "weight_position":    1.0,
    "weight_functions":   1.0,

    # Instance attribute contribution per matched pair.
    "instance_attribute_weight": 0.3,

    # Attribute tier ceiling (matches alias and fuzzy-name tiers).
    "attribute_tier_ceiling": 0.9,

    # Score formula denominator (number of defining signals).
    "defining_signal_count": 6,

    # Multi-signal gate: at least this many signals required, unless category
    # or material fires alone.
    "gate_min_signals": 2,

    # Dimensions unit: at least this many of {length, width, height, volume}
    # must be within range for dimensions to count as matched.
    "dimensions_match_min": 2,
    "dimensions_match_range": (0.5, 1.5),

    # Instance attributes: 2 instances count as 1 gate signal.
    "gate_instance_ratio": 2,

    # Tier-2/3 fuzzy-name acceptance threshold (shared with search engine).
    "fuzzy_threshold": CONFIG["fuzzy_threshold"],

    # Generic categories to ignore in category overlap.
    "too_general_categories": {
        "object", "physical_object", "person", "agent", "entity",
        "human", "living_thing",
    },
}


class PlannerMatchHelpers:
    """Self-contained matching primitives for the planning module."""

    def __init__(self, alias_expansion: Dict[str, Any]):
        self.alias_expansion = alias_expansion or {}
        self.threshold = OBJECT_COMPAT_CONFIG["fuzzy_threshold"]

        # Unmatchable attribute components (banned + highly frequent)
        self._unmatchable_attr_components = set(BANNED_ATTRIBUTE_COMPONENTS)

        # Embedding groups (component/value/token/object/action/verb)
        self.embedding_cache = EmbeddingCache()
        self.group_terms: Dict[str, List[str]] = {}
        self.group_embeddings: Dict[str, np.ndarray] = {}
        self.group_term_index: Dict[str, Dict[str, int]] = {}
        self._query_emb_cache: Dict[str, np.ndarray] = {}
        self._embedding_model = None
        self._is_model2vec = False
        self._load_embedding_model()
        self._load_embedding_groups()

    # ------------------------------------------------------------------
    # Embedding model
    # ------------------------------------------------------------------
    def _load_embedding_model(self):
        try:
            from model2vec import StaticModel
            self._embedding_model = StaticModel.from_pretrained("minishlab/potion-base-8M")
            self._is_model2vec = True
            return
        except Exception:
            pass
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._is_model2vec = False
            return
        except Exception:
            self._embedding_model = None

    def _load_embedding_groups(self):
        groups = [
            "component_attribute",
            "component_value",
            "token_attribute",
            "token_value",
            "token_object",
            "token_action",
            "verb",
            "object_names",
            "action_names",
        ]
        for group in groups:
            result = self.embedding_cache.load_group(group)
            if result is not None:
                terms, embeddings = result
                self.group_terms[group] = terms
                self.group_embeddings[group] = embeddings
                self.group_term_index[group] = {term: i for i, term in enumerate(terms)}

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _query_embedding(self, phrase: str) -> Optional[np.ndarray]:
        if self._embedding_model is None:
            return None
        if phrase in self._query_emb_cache:
            return self._query_emb_cache[phrase]
        if self._is_model2vec:
            vec = self._embedding_model.encode([phrase])[0]
        else:
            vec = self._embedding_model.encode(phrase, normalize_embeddings=True)
        vec = np.asarray(vec).astype("float32")
        norm = np.linalg.norm(vec)
        if norm == 0:
            norm = 1.0
        vec = vec / norm
        self._query_emb_cache[phrase] = vec
        return vec

    def similarities_to_group(self, query_phrase: str, group_name: str) -> List[Tuple[str, float]]:
        terms = self.group_terms.get(group_name)
        embeddings = self.group_embeddings.get(group_name)
        if terms is None or embeddings is None or len(terms) == 0:
            return []
        q_vec = self._query_embedding(query_phrase)
        if q_vec is None:
            return []
        sims = embeddings @ q_vec
        results = []
        for idx, sim in enumerate(sims):
            if sim >= self.threshold:
                results.append((terms[idx], float(sim)))
        return results

    def similarity_to_specific_term(self, query_phrase: str, group_name: str, full_term: str) -> float:
        term_index = self.group_term_index.get(group_name, {}).get(full_term)
        if term_index is None:
            return 0.0
        embeddings = self.group_embeddings.get(group_name)
        if embeddings is None:
            return 0.0
        q_vec = self._query_embedding(query_phrase)
        if q_vec is None:
            return 0.0
        return float(embeddings[term_index] @ q_vec)

    # ------------------------------------------------------------------
    # Value / path / op matching (copied from search engine)
    # ------------------------------------------------------------------
    def normalize_value(self, val: Any) -> Any:
        if isinstance(val, str):
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
                val = val[1:-1]
            low = val.lower()
            if low in ("true", "false"):
                return low == "true"
            if low in ("null", "none"):
                return None
            m = re.match(r"^(-?\d+(?:\.\d+)?)\s*([a-zA-Z^0-9%]*)$", val)
            if m:
                num = float(m.group(1))
                unit = m.group(2).lower()
                quantity = derive_quantity_from_unit(unit)
                if quantity != "unknown":
                    return normalize_measurement(num, unit, quantity)
                return num
            return val
        return val

    def value_match_score(self, q_val: Any, e_val: Any) -> float:
        # Numeric / non-string branch
        if not isinstance(q_val, str) or not isinstance(e_val, str):
            q_norm = self.normalize_value(q_val)
            e_norm = self.normalize_value(e_val)
            if q_norm == e_norm:
                return 1.0
            if isinstance(q_norm, (int, float)) and isinstance(e_norm, (int, float)):
                if e_norm != 0 and abs(q_norm - e_norm) / abs(e_norm) <= 0.3:
                    return CONFIG["scalar_within_30pct"]
            if isinstance(e_norm, list) and not isinstance(q_norm, list):
                if q_norm in e_norm:
                    return 0.7
                if isinstance(q_norm, str):
                    aliases_q = self.alias_expansion.get("value_aliases", {}).get(q_val, [])
                    for item in e_norm:
                        if isinstance(item, str):
                            aliases_e = self.alias_expansion.get("value_aliases", {}).get(item, [])
                            if item in aliases_q or q_val in aliases_e:
                                return 0.6 * CONFIG["alias_penalty"]
                            sim = self.similarity_to_specific_term(q_norm, "component_value", f"value:{item}")
                            if sim >= self.threshold:
                                return 0.6
            return 0.0

        q_norm = self.normalize_value(q_val)
        e_norm = self.normalize_value(e_val)
        if q_norm == e_norm:
            return 1.0

        if isinstance(q_norm, (int, float)) and isinstance(e_norm, (int, float)):
            if e_norm != 0 and abs(q_norm - e_norm) / abs(e_norm) <= 0.3:
                return CONFIG["scalar_within_30pct"]

        if isinstance(q_norm, list) and isinstance(e_norm, list):
            matched = sum(1 for e_item in e_norm if e_item in q_norm)
            if matched == len(e_norm):
                return 1.0
            elif matched > 0:
                return 0.7
            return 0.0

        if isinstance(e_norm, list) and not isinstance(q_norm, list):
            if q_norm in e_norm:
                return 0.7
            aliases_q = self.alias_expansion.get("value_aliases", {}).get(q_val, [])
            for item in e_norm:
                if isinstance(item, str):
                    aliases_e = self.alias_expansion.get("value_aliases", {}).get(item, [])
                    if item in aliases_q or q_val in aliases_e:
                        return 0.6 * CONFIG["alias_penalty"]
                    sim = self.similarity_to_specific_term(q_norm, "component_value", f"value:{item}")
                    if sim >= self.threshold:
                        return 0.6
            return 0.0

        aliases_q = self.alias_expansion.get("value_aliases", {}).get(q_val, [])
        aliases_e = self.alias_expansion.get("value_aliases", {}).get(e_val, [])
        if q_val in aliases_e or e_val in aliases_q:
            return 1.0 * CONFIG["alias_penalty"]

        base_term = f"value:{e_val}"
        sim = self.similarity_to_specific_term(q_val, "component_value", base_term)
        if sim >= self.threshold:
            return 0.9

        for alias in aliases_q:
            sim_alias = self.similarity_to_specific_term(alias, "component_value", base_term)
            if sim_alias >= self.threshold:
                return 0.9 * CONFIG["alias_penalty"]

        for alias in aliases_e:
            sim_alias2 = self.similarity_to_specific_term(
                q_val, "component_value", f"value:{alias}"
            )
            if sim_alias2 >= self.threshold:
                return 0.9 * CONFIG["alias_penalty"]

        return 0.0

    def op_match_score(self, q_op: Any, e_op: Any) -> float:
        q_op = str(q_op)
        e_op = str(e_op)
        if q_op == e_op:
            return 1.0
        if q_op == "!=" or e_op == "!=":
            return 0.5
        return 0.8

    def is_contiguous_subsequence(self, small, large) -> bool:
        if not small:
            return False
        m, n = len(small), len(large)
        for i in range(0, n - m + 1):
            if tuple(large[i:i + m]) == tuple(small):
                return True
        return False

    def path_match_score(self, q_path, e_path) -> Tuple[float, bool]:
        q_path = tuple(q_path)
        e_path = tuple(e_path)
        q_len = len(q_path)
        e_len = len(e_path)
        if q_len == 0 or e_len == 0:
            return 0.0, False

        if self.is_contiguous_subsequence(q_path, e_path):
            return 1.0, False
        if all(p in e_path for p in q_path):
            return 0.9, False

        matches = []
        for q_comp in q_path:
            best_score = 0.0
            best_idx = -1
            best_alias = False
            for i, e_comp in enumerate(e_path):
                if q_comp == e_comp:
                    if 1.0 > best_score:
                        best_score = 1.0
                        best_idx = i
                        best_alias = False
                    continue
                aliases_q = self.alias_expansion.get("attribute_component_aliases", {}).get(q_comp, [])
                aliases_e = self.alias_expansion.get("attribute_component_aliases", {}).get(e_comp, [])
                if e_comp in aliases_q or q_comp in aliases_e:
                    if 1.0 > best_score:
                        best_score = 1.0
                        best_idx = i
                        best_alias = True
                    continue
                sim = self.similarity_to_specific_term(q_comp, "component_attribute", f"attribute:{e_comp}")
                if sim >= self.threshold and sim > best_score:
                    best_score = sim
                    best_idx = i
                    best_alias = False
            if best_score > 0:
                matches.append((best_score, best_idx, best_alias))
            else:
                matches.append(None)

        matched = [m for m in matches if m is not None]
        matched_count = len(matched)
        if matched_count == 0:
            return 0.0, False

        alias_used = any(m[2] for m in matched)
        any_fuzzy = any(not m[2] and m[0] < 1.0 for m in matched)

        if matched_count < q_len:
            ratio = matched_count / e_len
            score = ratio * 0.8 if any_fuzzy else ratio
            if alias_used:
                score *= CONFIG["alias_penalty"]
            return score, alias_used

        positions = [m[1] for m in matched]
        pos_sorted = sorted(positions)
        contiguous = (
            len(pos_sorted) == q_len and
            max(pos_sorted) - min(pos_sorted) == q_len - 1
        )
        if contiguous:
            score = 0.9 if any_fuzzy else 1.0
        else:
            score = 0.8 if any_fuzzy else 0.9
        if alias_used:
            score *= CONFIG["alias_penalty"]
        return score, alias_used

    # ------------------------------------------------------------------
    # Position relative_to matching (scalar or list)
    # ------------------------------------------------------------------
    def _relative_to_match_score(self, q_rel_to, c_rel_to) -> float:
        """
        Best matching score between two relative_to values.
        Each side may be a string or a list of strings.
        Returns a score in [0, 1]:
            1.0 for any exact string match
            otherwise the best fuzzy similarity, or 0.0
        """
        q_list = q_rel_to if isinstance(q_rel_to, (list, tuple)) else [q_rel_to]
        c_list = c_rel_to if isinstance(c_rel_to, (list, tuple)) else [c_rel_to]

        best = 0.0
        for q in q_list:
            if not q:
                continue
            q_str = str(q)
            for c in c_list:
                if not c:
                    continue
                c_str = str(c)
                if q_str == c_str:
                    return 1.0
                sim = self.similarity_to_specific_term(
                    q_str, "component_value", f"value:{c_str}"
                )
                if sim > best:
                    best = sim
        return best

    # ------------------------------------------------------------------
    # Object compatibility (planning-specific)
    # ------------------------------------------------------------------
    def object_compatibility(
        self,
        query_obj: Dict[str, Any],
        candidate_obj: Dict[str, Any],
    ) -> Tuple[float, List[str], str]:
        """
        Return (score, signals, tier) for a query object vs a candidate object.

        Both objects are dicts with:
            template_name, obj_id (optional)
            categories (list), functions (list), materials (list)
            shape (str)
            dimensions (dict: {length, width, height_thickness} each with value+unit)
            position (dict: {relation, relationr, relative_to})
            attributes (dict: flat {path: value})

        score is in [0, 1].
        signals is the list of positive defining-attribute signals.
        tier is one of "exact", "alias", "fuzzy_name", "attribute", "rejected".
        """
        # Tier 1: exact object ID or template
        q_id = query_obj.get("obj_id") or query_obj.get("template_name")
        c_id = candidate_obj.get("obj_id") or candidate_obj.get("template_name")
        q_tmpl = query_obj.get("template_name", "")
        c_tmpl = candidate_obj.get("template_name", "")

        if q_id and q_id == c_id:
            return 1.0, ["exact_id"], "exact"
        if q_tmpl and q_tmpl == c_tmpl:
            return 1.0, ["exact_template"], "exact"

        # Tier 2: alias relation
        if q_tmpl and c_tmpl:
            q_aliases = self.alias_expansion.get("object_aliases", {}).get(q_tmpl, [])
            c_aliases = self.alias_expansion.get("object_aliases", {}).get(c_tmpl, [])
            if c_tmpl in q_aliases or q_tmpl in c_aliases:
                return 0.9, ["alias"], "alias"

        # Tier 3: fuzzy name match
        if q_tmpl and c_tmpl:
            sim = self.similarity_to_specific_term(
                q_tmpl, "object_names", f"object:{c_tmpl}"
            )
            if sim >= self.threshold:
                return 0.9, [f"fuzzy_name:{sim:.3f}"], "fuzzy_name"

        # Tier 4: attribute comparison with multi-signal gate
        signals: List[str] = []

        # Category (non-generic only)
        q_cats = set(
            c for c in query_obj.get("categories", [])
            if c not in OBJECT_COMPAT_CONFIG["too_general_categories"]
        )
        c_cats = set(
            c for c in candidate_obj.get("categories", [])
            if c not in OBJECT_COMPAT_CONFIG["too_general_categories"]
        )
        if q_cats and c_cats and (q_cats & c_cats):
            signals.append("category")

        # Materials
        q_mats = set(query_obj.get("materials", []) or [])
        c_mats = set(candidate_obj.get("materials", []) or [])
        if q_mats and c_mats and (q_mats & c_mats):
            signals.append("materials")

        # Shape
        q_shape = query_obj.get("shape", "") or ""
        c_shape = candidate_obj.get("shape", "") or ""
        if q_shape and c_shape:
            if q_shape == c_shape:
                signals.append("shape")
            else:
                sim = self.similarity_to_specific_term(
                    q_shape, "component_value", f"value:{c_shape}"
                )
                if sim >= self.threshold:
                    signals.append("shape")

        # Dimensions (unit: 2 of 4 must match)
        if self._dimensions_unit_match(
            query_obj.get("dimensions", {}) or {},
            candidate_obj.get("dimensions", {}) or {},
        ):
            signals.append("dimensions")

        # Position (unit)
        if self._position_unit_match(
            query_obj.get("position", {}) or {},
            candidate_obj.get("position", {}) or {},
        ):
            signals.append("position")

        # Functions (any fuzzy-matched function pair counts)
        q_funcs = query_obj.get("functions", []) or []
        c_funcs = candidate_obj.get("functions", []) or []
        if q_funcs and c_funcs:
            functions_matched = False
            for qf in q_funcs:
                for cf in c_funcs:
                    if qf == cf:
                        functions_matched = True
                        break
                    sim = self.similarity_to_specific_term(
                        qf, "component_value", f"value:{cf}"
                    )
                    if sim >= self.threshold:
                        functions_matched = True
                        break
                if functions_matched:
                    break
            if functions_matched:
                signals.append("functions")

        # Instance attribute matches
        q_attrs = query_obj.get("attributes", {}) or {}
        c_attrs = candidate_obj.get("attributes", {}) or {}
        q_flat = self._flatten_attributes(q_attrs)
        c_flat = self._flatten_attributes(c_attrs)
        instance_matches = 0
        for path, q_val in q_flat.items():
            if path in c_flat:
                if self.value_match_score(q_val, c_flat[path]) > 0:
                    instance_matches += 1

        # Multi-signal gate
        n_defining = len(signals)
        gate_signals = n_defining + (instance_matches // OBJECT_COMPAT_CONFIG["gate_instance_ratio"])
        has_category = "category" in signals
        has_materials = "materials" in signals
        gate_pass = (
            gate_signals >= OBJECT_COMPAT_CONFIG["gate_min_signals"]
            or has_category
            or has_materials
        )
        if not gate_pass:
            return 0.0, signals, "rejected"

        # Score formula
        units = n_defining * 1.0 + instance_matches * OBJECT_COMPAT_CONFIG["instance_attribute_weight"]
        raw_score = (units / OBJECT_COMPAT_CONFIG["defining_signal_count"]) * OBJECT_COMPAT_CONFIG["attribute_tier_ceiling"]
        score = min(OBJECT_COMPAT_CONFIG["attribute_tier_ceiling"], raw_score)

        return score, signals, "attribute"

    # ------------------------------------------------------------------
    # Object substitution lookup
    # ------------------------------------------------------------------
    def find_best_object_substitute(
        self,
        query_obj: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        min_threshold: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Return the best candidate above threshold, or None.

        The result includes the candidate dict, its score, signals, and tier.
        """
        best = None
        best_score = min_threshold
        for cand in candidates:
            score, signals, tier = self.object_compatibility(query_obj, cand)
            if score > best_score:
                best_score = score
                best = {
                    "candidate": cand,
                    "score": score,
                    "signals": signals,
                    "tier": tier,
                }
        return best

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _flatten_attributes(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten a nested attribute dict into {dotted_path: value}."""
        result: Dict[str, Any] = {}
        if not isinstance(attrs, dict):
            return result

        def walk(node, prefix):
            if isinstance(node, dict):
                if "value" in node and set(node.keys()) <= {"value", "unit"}:
                    val = node.get("value")
                    unit = node.get("unit", "")
                    combined = f"{val}{unit}" if unit else str(val)
                    result[".".join(prefix)] = combined
                    return
                for k, v in node.items():
                    walk(v, prefix + [str(k)])
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    if isinstance(item, (dict, list)):
                        walk(item, prefix + [str(i)])
                    else:
                        result[".".join(prefix + [str(i)])] = item
            else:
                result[".".join(prefix)] = node

        walk(attrs, [])
        return result

    def _dimensions_unit_match(self, q_dims: Dict, c_dims: Dict) -> bool:
        """
        Return True if at least `dimensions_match_min` of the four signals
        (length, width, height_thickness, derived volume) are within range.
        """
        if not isinstance(q_dims, dict) or not isinstance(c_dims, dict):
            return False

        low, high = OBJECT_COMPAT_CONFIG["dimensions_match_range"]
        matched = 0

        for dim in ("length", "width", "height_thickness"):
            q_d = q_dims.get(dim)
            c_d = c_dims.get(dim)
            if not (isinstance(q_d, dict) and isinstance(c_d, dict)):
                continue
            q_m = parse_measurement_string(f"{q_d.get('value')}{q_d.get('unit','')}")
            c_m = parse_measurement_string(f"{c_d.get('value')}{c_d.get('unit','')}")
            if q_m is None or c_m is None:
                continue
            q_norm = normalize_measurement(q_m[0], q_m[1], "length")
            c_norm = normalize_measurement(c_m[0], c_m[1], "length")
            if q_norm is None or c_norm is None or c_norm == 0:
                continue
            if low * c_norm <= q_norm <= high * c_norm:
                matched += 1

        # Derived volume
        q_vol = self._derive_volume_cm3(q_dims)
        c_vol = self._derive_volume_cm3(c_dims)
        if q_vol is not None and c_vol is not None and c_vol != 0:
            if low * c_vol <= q_vol <= high * c_vol:
                matched += 1

        return matched >= OBJECT_COMPAT_CONFIG["dimensions_match_min"]

    def _derive_volume_cm3(self, dims: Dict) -> Optional[float]:
        values = {}
        for dim in ("length", "width", "height_thickness"):
            d = dims.get(dim)
            if not isinstance(d, dict):
                return None
            m = parse_measurement_string(f"{d.get('value')}{d.get('unit','')}")
            if m is None:
                return None
            norm = normalize_measurement(m[0], m[1], "length")
            if norm is None:
                return None
            values[dim] = norm
        return values["length"] * values["width"] * values["height_thickness"]

    def _position_unit_match(self, q_pos: Dict, c_pos: Dict) -> bool:
        """
        Position matches if relative_to matches, and either relationr or
        relation matches.

        relative_to may be a string or a list of strings on either side.
        The match score is the best pair across the two sides; any exact
        string match scores 1.0, and otherwise the best fuzzy similarity
        above the threshold is used.
        """
        if not isinstance(q_pos, dict) or not isinstance(c_pos, dict):
            return False

        q_rel = q_pos.get("relation", "")
        q_relr = q_pos.get("relationr", q_rel) or q_rel
        q_rel_to = q_pos.get("relative_to", "")

        c_rel = c_pos.get("relation", "")
        c_relr = c_pos.get("relationr", c_rel) or c_rel
        c_rel_to = c_pos.get("relative_to", "")

        if not q_rel_to or not c_rel_to:
            return False

        rel_to_score = self._relative_to_match_score(q_rel_to, c_rel_to)
        if rel_to_score < self.threshold:
            return False

        return (q_relr and c_relr and q_relr == c_relr) or (q_rel and c_rel and q_rel == c_rel)