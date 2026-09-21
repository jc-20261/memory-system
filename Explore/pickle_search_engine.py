#!/usr/bin/env python3
r"""
pickle_search_engine.py – Fully corrected search engine with query-side alias grouping.

Incorporates:
- List-to-list value matching (full/partial/no match)
- Path normalization to tuple
- Banned/unmatchable attribute components in full-path value fallback
- Null object/action neutral multiplier (×1)
- Symmetric object/action alias matching
- Query-side alias grouping using alias_expansion mapping (including value aliases)
- Postings deduplication for alias folding
- Proper verb matching with caps
- Frequency controls, global diminishing, co‑match, goalstate multiplier
- Progress callback
- Precomputed embeddings from EmbeddingCache
- PER-POSTING goalstate detection via source_context["type"] == "goal_state"
  (replaces the previous global-set string check that caused false positives)
- goalstate_query flag threaded through _process_term and all dispatchers
"""

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

from pickle_search_index import (
    ALLOWED_PREFIXES,
    CONFIG,
    PickleSearchIndex,
    PREFIX_FULL_PATH,
    PREFIX_ATTRIBUTE_VALUE,
    PREFIX_ATTRIBUTE,
    PREFIX_VALUE,
    PREFIX_ATTRIBUTE_TOKEN,
    PREFIX_VALUE_TOKEN,
    PREFIX_OBJECT_TOKEN,
    PREFIX_ACTION_TOKEN,
    PREFIX_GOAL_TOKEN,
    PREFIX_OTHER_TOKEN,
    PREFIX_ATTRIBUTE_EXPANDED,
    PREFIX_VALUE_EXPANDED,
    PREFIX_OBJECT_EXPANDED,
    PREFIX_ACTION_EXPANDED,
    PREFIX_ATTRIBUTE_EXPANDED_TOKEN,
    PREFIX_VALUE_EXPANDED_TOKEN,
    PREFIX_OBJECT_EXPANDED_TOKEN,
    PREFIX_ACTION_EXPANDED_TOKEN,
    PREFIX_VERB,
    SPATIAL_OPERATORS,
    BANNED_ATTRIBUTE_COMPONENTS,
    get_term_prefix,
    tokenize_into_words,
    is_valid_token,
    normalize_measurement,
    parse_measurement_string,
    derive_quantity_from_unit,
    STOP_WORDS,
    UNIT_WORDS,
)

from memory_model import Memory
from embedding_cache import EmbeddingCache


TERM_TYPE_ATTRIBUTE = "attribute"
TERM_TYPE_VALUE = "value"
TERM_TYPE_OBJECT = "object"
TERM_TYPE_ACTION = "action"
TERM_TYPE_GOAL = "goal"
TERM_TYPE_OTHER = "other"


TOKEN_MULTIPLIERS = {
    (True, True): 0.3,
    (True, False): 0.2,
    (False, True): 0.2,
    (False, False): 0.1,
}

COMPONENT_MULTIPLIERS = {
    (True, True): 0.5,
    (True, False): 0.4,
    (False, True): 0.4,
    (False, False): 0.3,
}


def _is_goalstate_posting(src: Any) -> bool:
    """Return True if a posting's source_context marks it as a goalstate entry."""
    return isinstance(src, dict) and src.get("type") == "goal_state"


class PickleSearchEngine:
    def __init__(self, index: PickleSearchIndex, diagnostic: bool = False):
        self.index = index
        self.threshold = CONFIG["fuzzy_threshold"]
        self.diagnostic = diagnostic
        self.freq_map = {}
        self._load_frequency_data()
        self.embedding_cache = EmbeddingCache()
        self.group_terms = {}
        self.group_embeddings = {}
        self.group_term_index = {}
        self._load_embedding_groups()
        self._query_emb_cache = {}

        # Combined unmatchable attribute components
        self._unmatchable_attr_components = set(BANNED_ATTRIBUTE_COMPONENTS)
        for term, info in self.freq_map.items():
            if term.startswith(PREFIX_ATTRIBUTE + ":"):
                comp = term.split(":", 1)[1]
                if info["occurrence_percentage"] / 100.0 >= CONFIG["freq_attr_comp_unmatchable"]:
                    self._unmatchable_attr_components.add(comp)

        # Build query-side alias mapping
        self._query_alias_to_base = {}
        # Attribute component aliases
        for base_comp, aliases in self.index.alias_expansion.get("attribute_component_aliases", {}).items():
            for alias in aliases:
                self._query_alias_to_base[f"{PREFIX_ATTRIBUTE_EXPANDED}:{alias}"] = f"{PREFIX_ATTRIBUTE}:{base_comp}"
                self._query_alias_to_base[f"{PREFIX_ATTRIBUTE_EXPANDED_TOKEN}:{alias}"] = f"{PREFIX_ATTRIBUTE_TOKEN}:{base_comp}"
        # Object aliases
        for base_obj, aliases in self.index.alias_expansion.get("object_aliases", {}).items():
            for alias in aliases:
                self._query_alias_to_base[f"object_expanded:{alias}"] = f"object:{base_obj}"
                self._query_alias_to_base[f"object_expanded_token:{alias}"] = f"object_token:{base_obj}"
        # Action aliases
        for base_act, aliases in self.index.alias_expansion.get("action_aliases", {}).items():
            for alias in aliases:
                self._query_alias_to_base[f"action_expanded:{alias}"] = f"action:{base_act}"
                self._query_alias_to_base[f"action_expanded_token:{alias}"] = f"action_token:{base_act}"
        # Value aliases
        for base_val, aliases in self.index.alias_expansion.get("value_aliases", {}).items():
            for alias in aliases:
                self._query_alias_to_base[f"{PREFIX_VALUE_EXPANDED}:{alias}"] = f"{PREFIX_VALUE}:{base_val}"
                self._query_alias_to_base[f"{PREFIX_VALUE_EXPANDED_TOKEN}:{alias}"] = f"{PREFIX_VALUE_TOKEN}:{base_val}"

        # Progress tracking state
        self._index_examined_count = 0
        self._last_progress_time = 0.0
        self._progress_callback = None
        self._current_query_term = ""
        self._query_idx = 0
        self._total_query_terms = 0
        self._start_time = 0.0

    def _load_frequency_data(self):
        freq_path = Path(__file__).parent / "data" / "searchable_term_frequencies_occurrences.json"
        if not freq_path.exists():
            return
        with open(freq_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.get("terms", []):
            self.freq_map[entry["term"]] = {
                "occurrence_percentage": entry["occurrence_percentage"],
                "memory_percentage": entry["memory_percentage"],
                "prefix": entry["prefix"],
            }

    def _get_freq(self, term: str) -> float:
        info = self.freq_map.get(term)
        if info:
            return info["occurrence_percentage"] / 100.0
        return 0.0

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
                print(f"✅ Loaded embedding group {group} ({len(terms)} terms)")
            else:
                print(f"⚠️ No embedding cache for {group}; fuzzy matching for that group disabled.")

    def _query_embedding(self, query_phrase: str) -> np.ndarray:
        if query_phrase in self._query_emb_cache:
            return self._query_emb_cache[query_phrase]

        if self.index._is_model2vec:
            vec = self.index._embedding_model.encode([query_phrase])[0]
        else:
            vec = self.index._embedding_model.encode(query_phrase, normalize_embeddings=True)
        vec = np.asarray(vec).astype("float32")
        norm = np.linalg.norm(vec)
        if norm == 0:
            norm = 1.0
        vec = vec / norm
        self._query_emb_cache[query_phrase] = vec
        return vec

    def _similarities_to_group(self, query_phrase: str, group_name: str) -> List[Tuple[str, float]]:
        terms = self.group_terms.get(group_name)
        embeddings = self.group_embeddings.get(group_name)
        if terms is None or embeddings is None or len(terms) == 0:
            return []

        q_vec = self._query_embedding(query_phrase)
        sims = embeddings @ q_vec
        results = []
        for idx, sim in enumerate(sims):
            if sim >= self.threshold:
                results.append((terms[idx], float(sim)))
        return results

    def _similarity_to_specific_term(self, query_phrase: str, group_name: str, full_term: str) -> float:
        term_index = self.group_term_index.get(group_name, {}).get(full_term)
        if term_index is None:
            return 0.0
        embeddings = self.group_embeddings.get(group_name)
        if embeddings is None:
            return 0.0
        q_vec = self._query_embedding(query_phrase)
        return float(embeddings[term_index] @ q_vec)

    def _deduplicate_postings(self, postings):
        seen = {}
        for mem, src, term_type in postings:
            try:
                src_key = json.dumps(src, sort_keys=True, ensure_ascii=False)
            except Exception:
                src_key = repr(src)
            key = (mem.id, src_key, term_type)
            if key not in seen:
                seen[key] = (mem, src, term_type)
        return list(seen.values())

    # ------------------------------------------------------------------
    # Progress helper
    # ------------------------------------------------------------------
    def _set_progress_context(self, callback, current_query_term, q_idx, total_q, start_time):
        self._progress_callback = callback
        self._current_query_term = current_query_term
        self._query_idx = q_idx
        self._total_query_terms = total_q
        self._start_time = start_time
        self._index_examined_count = 0
        self._last_progress_time = 0.0

    def _increment_index_examined(self):
        self._index_examined_count += 1
        now = time.perf_counter()
        if self._progress_callback is not None:
            if self._index_examined_count % 50 == 0 or (now - self._last_progress_time) >= 0.2:
                self._last_progress_time = now
                elapsed = now - self._start_time
                self._progress_callback(
                    self._current_query_term,
                    self._query_idx,
                    self._total_query_terms,
                    self._index_examined_count,
                    elapsed
                )

    # ------------------------------------------------------------------
    # Main search with query-side alias grouping
    # ------------------------------------------------------------------
    def search(self, query_terms: List[Tuple[str, float]],
               progress: Optional[Callable[[str, int, int, int, float], None]] = None,
               query_alias_map: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        # Merge query-side alias map for the duration of this search.
        # The engine's built-in _query_alias_to_base is restored afterwards.
        original_alias_map = self._query_alias_to_base
        if query_alias_map:
            merged_alias_map = dict(original_alias_map)
            merged_alias_map.update(query_alias_map)
            self._query_alias_to_base = merged_alias_map

        try:
            grouped_query_terms = self._group_query_terms(query_terms)

            contributions: List[Dict[str, Any]] = []

            measurement_groups = []
            regular_groups = []

            for group in grouped_query_terms:
                representative = group[0][0]
                if self._is_measurement_query(representative):
                    measurement_groups.append(group)
                else:
                    regular_groups.append(group)

            total_terms = sum(len(group) for group in grouped_query_terms)
            start_time = time.perf_counter()

            if measurement_groups:
                measurement_terms = [term for group in measurement_groups for term in group]
                self._process_measurement_group(measurement_terms, contributions)

            for q_idx, group in enumerate(regular_groups, 1):
                group_contribs = []
                for term, _weight in group:
                    self._set_progress_context(progress, term, q_idx, total_terms, start_time)
                    self._process_term(term, group_contribs)
                group_contribs = self._merge_alias_group_contributions(group_contribs)
                group_contribs = self._apply_query_term_cap(group_contribs, cap=6)
                contributions.extend(group_contribs)

                if progress is not None:
                    elapsed = time.perf_counter() - start_time
                    progress(group[0][0], q_idx, total_terms, self._index_examined_count, elapsed)

            # Final progress update after all query terms have been processed
            if progress is not None:
                elapsed = time.perf_counter() - start_time
                progress("DONE", total_terms, total_terms, self._index_examined_count, elapsed)

            # Apply global diminishing rule
            contributions = self._apply_global_diminishing(contributions)

            mem_obj_counts = defaultdict(lambda: defaultdict(int))
            for c in contributions:
                if c.get("object_id") is not None:
                    mem_obj_counts[c["memory"]][c["object_id"]] += 1

            for c in contributions:
                if c.get("object_id") is not None:
                    n = mem_obj_counts[c["memory"]][c["object_id"]]
                    if n > 1:
                        mult = min(CONFIG["co_match_base"] ** (n - 1), CONFIG["co_match_cap"])
                        c["score_contribution"] *= mult

            # After co-match multiplier
            contributions = self._apply_index_term_cap(contributions, cap=6)

            score_map = defaultdict(float)
            details_map = defaultdict(list)
            for c in contributions:
                mem = c["memory"]
                score_map[mem] += c["score_contribution"]
                details_map[mem].append({
                    "query_term": c["query_term"],
                    "matched_index_term": c["matched_index_term"],
                    "score_contribution": c["score_contribution"],
                    "mode": c["mode"],
                    "source_context": c["source_context"],
                    "term_type": c["term_type"],
                    "similarity": c.get("similarity"),
                    "alias_used": c.get("alias_used", False),
                })

            results = []
            for mem, score in score_map.items():
                results.append({
                    "memory": mem,
                    "score": score,
                    "term_details": details_map[mem],
                })
            results.sort(key=lambda x: -x["score"])
            return results

        finally:
            # Restore the engine's built-in alias map
            self._query_alias_to_base = original_alias_map

    def _group_query_terms(self, query_terms):
        groups = []
        seen = {}
        for term, weight in query_terms:
            base = self._resolve_query_term_base(term)
            if base in seen:
                groups[seen[base]].append((term, weight))
            else:
                seen[base] = len(groups)
                groups.append([(term, weight)])
        return groups

    def _resolve_query_term_base(self, term: str) -> str:
        # 1. Query alias mapping
        base = self._query_alias_to_base.get(term)
        if base:
            return base

        # 2. Index alias mapping
        base = self.index.alias_to_base.get(term)
        if base:
            return base

        # 3. No alias relation
        return term

    def _merge_alias_group_contributions(self, contributions):
        grouped = defaultdict(list)
        for c in contributions:
            key = (
                self._resolve_query_term_base(c["query_term"]),
                c["matched_index_term"],
                c["memory"]
            )
            grouped[key].append(c)

        merged = []
        for key, items in grouped.items():
            if len(items) == 1:
                merged.extend(items)
                continue
            by_memory = defaultdict(list)
            for item in items:
                by_memory[item["memory"]].append(item)

            for mem, mem_items in by_memory.items():
                seen_src = {}
                for item in mem_items:
                    try:
                        src_key = json.dumps(item["source_context"], sort_keys=True, ensure_ascii=False)
                    except Exception:
                        src_key = repr(item["source_context"])
                    if src_key not in seen_src or item["score_contribution"] > seen_src[src_key]["score_contribution"]:
                        seen_src[src_key] = item
                merged.extend(seen_src.values())

        return merged

    def _is_measurement_query(self, term: str) -> bool:
        if term.startswith(PREFIX_ATTRIBUTE_VALUE + ":"):
            rest = term.split(":", 1)[1]
            if "=" in rest:
                path = rest.split("=", 1)[0]
                return path in (
                    "dimensions.length",
                    "dimensions.width",
                    "dimensions.height_thickness",
                    "weight",
                    "mass",
                    "volume",
                )
        return False

    # ------------------------------------------------------------------
    # Global diminishing rule
    # ------------------------------------------------------------------
    def _apply_global_diminishing(self, contributions: List[Dict]) -> List[Dict]:
        excluded_modes = {
            "component_exact", "component_fuzzy",
            "token_exact", "token_fuzzy",
            "verb_exact", "verb_fuzzy",
        }
        subject = [c for c in contributions if c.get("mode") not in excluded_modes]
        exempt = [c for c in contributions if c.get("mode") in excluded_modes]

        grouped = defaultdict(list)
        for c in subject:
            key = (c["query_term"], c["matched_index_term"], c["memory"])
            grouped[key].append(c)

        processed = []
        for key, items in grouped.items():
            items.sort(key=lambda x: -x["score_contribution"])
            multipliers = [1.0, 0.5, 0.1] + [0.0] * max(0, len(items) - 3)
            for item, mult in zip(items, multipliers):
                if mult == 0:
                    continue
                item["score_contribution"] *= mult
                processed.append(item)

        return exempt + processed

    # ------------------------------------------------------------------
    # Query term cap
    # ------------------------------------------------------------------
    def _apply_query_term_cap(self, contributions, cap=6):
        """
        Keep only the top `cap` contributions per (memory, query_term_base).

        The query_term_base is the resolved base of the query term with all
        aliases folded under. Two query terms that resolve to the same base
        share this cap, so alias variants cannot flood a single memory.
        """
        grouped = defaultdict(list)
        for c in contributions:
            base = self._resolve_query_term_base(c["query_term"])
            key = (c["memory"], base)
            grouped[key].append(c)

        result = []
        for key, items in grouped.items():
            items.sort(key=lambda x: -x["score_contribution"])
            result.extend(items[:cap])

        return result

    # ------------------------------------------------------------------
    # Global cap rule
    # ------------------------------------------------------------------
    def _apply_index_term_cap(self, contributions, cap=6):
        grouped = defaultdict(list)

        for c in contributions:
            key = (c["memory"], c["matched_index_term"])
            grouped[key].append(c)

        result = []
        for key, items in grouped.items():
            items.sort(key=lambda x: -x["score_contribution"])
            result.extend(items[:cap])

        return result

    # ------------------------------------------------------------------
    # Measurement group matching
    # ------------------------------------------------------------------
    def _process_measurement_group(self, measurement_terms, contributions):
        required = [
            "dimensions.length",
            "dimensions.width",
            "dimensions.height_thickness",
            "volume",
        ]
        query_dims = {}
        for term, _ in measurement_terms:
            path, val = self._parse_attribute_value_term(term)
            if path in required and val is not None:
                query_dims[path] = (term, val)

        if len(query_dims) < 4:
            return

        mem_results = defaultdict(list)
        mem_obj = {}
        for path, (term, val) in query_dims.items():
            for meas in self.index.measurements:
                if meas["path"] != path:
                    continue
                mem = meas["memory"]
                mem_obj[mem] = meas.get("object")
                q_num, q_unit = parse_measurement_string(val)
                if q_num is None:
                    continue
                idx_num = meas["numeric"]
                idx_unit = meas["unit"]
                quantity = derive_quantity_from_unit(idx_unit)
                if quantity == "unknown":
                    quantity = derive_quantity_from_unit(q_unit)
                q_norm = normalize_measurement(q_num, q_unit, quantity)
                i_norm = normalize_measurement(idx_num, idx_unit, quantity)
                if q_norm is None or i_norm is None:
                    continue
                low, high = CONFIG["measurement_range"]
                mult = 1.0 if (low * i_norm <= q_norm <= high * i_norm) else CONFIG["dimension_mismatch_mult"]
                mem_results[mem].append(mult)
                self._increment_index_examined()

        for mem, mults in mem_results.items():
            if len(mults) != 4:
                continue
            product = 1.0
            for m in mults:
                product *= m
            if abs(product - (0.5 ** 4)) < 1e-9:
                score = 0.0
            else:
                score = min(product, 1.0)

            if score > 0:
                contributions.append({
                    "memory": mem,
                    "object_id": mem_obj.get(mem),
                    "query_term": "measurement_unit",
                    "matched_index_term": "dimensions+volume",
                    "score_contribution": score,
                    "mode": "measurement",
                    "source_context": {},
                    "term_type": TERM_TYPE_ATTRIBUTE,
                    "alias_used": False,
                })

    def _parse_attribute_value_term(self, term: str) -> Tuple[Optional[str], Optional[str]]:
        if ":" not in term:
            return None, None
        rest = term.split(":", 1)[1]
        if "=" not in rest:
            return None, None
        path, val = rest.split("=", 1)
        return path, val

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def _process_term(self, term: str, contributions: List[Dict[str, Any]],
                      is_goalstate_query: bool = False):
        prefix = get_term_prefix(term)
        if term.startswith(PREFIX_FULL_PATH + ":"):
            self._process_full_path_term(term, contributions)
        elif term.startswith("position_unit:"):
            self._process_position_unit_term(term, contributions)
        elif term.startswith("spatial:"):
            self._process_spatial_term(term, contributions)
        elif prefix in (PREFIX_VERB, "verb_target", "verb_context"):
            self._process_verb_term(term, contributions, is_goalstate_query)
        elif prefix in (PREFIX_ATTRIBUTE, PREFIX_VALUE,
                        PREFIX_ATTRIBUTE_EXPANDED, PREFIX_VALUE_EXPANDED,
                        "statechangesC_attribute", "statechangesC_value",
                        "statechangesC_attribute_expanded", "statechangesC_value_expanded"):
            self._process_component_term(term, contributions, is_goalstate_query)
        elif prefix in (PREFIX_ATTRIBUTE_TOKEN, PREFIX_VALUE_TOKEN,
                        PREFIX_OBJECT_TOKEN, PREFIX_ACTION_TOKEN,
                        PREFIX_ATTRIBUTE_EXPANDED_TOKEN, PREFIX_VALUE_EXPANDED_TOKEN,
                        PREFIX_OBJECT_EXPANDED_TOKEN, PREFIX_ACTION_EXPANDED_TOKEN,
                        "statechangesC_attribute_token", "statechangesC_value_token",
                        "statechangesC_attribute_expanded_token", "statechangesC_value_expanded_token"):
            self._process_token_term(term, contributions, is_goalstate_query)
        elif prefix in ("object", "object_expanded"):
            self._process_object_term(term, contributions, is_goalstate_query)
        elif prefix in ("action", "action_expanded"):
            self._process_action_term(term, contributions, is_goalstate_query)
        elif prefix == "goalstate":
            self._process_goalstate_term(term, contributions)
        else:
            self._process_regular_term(term, contributions, is_goalstate_query)

    # ------------------------------------------------------------------
    # Full path matching
    # ------------------------------------------------------------------
    def _process_full_path_term(self, term: str, contributions):
        json_str = term.split(":", 1)[1]
        try:
            query = json.loads(json_str)
        except json.JSONDecodeError:
            return

        for entry in self.index.full_path_entries:
            score, detail = self._match_full_path(query, entry)
            if score > 0:
                safe_entry = {k: v for k, v in entry.items() if k != "memory"}
                contributions.append({
                    "memory": entry["memory"],
                    "object_id": entry.get("object"),
                    "query_term": term,
                    "matched_index_term": f"fullpath:{json.dumps(safe_entry)}",
                    "score_contribution": score,
                    "mode": "fullpath",
                    "source_context": entry.get("source_context"),
                    "term_type": "attribute",
                    "alias_used": detail.get("alias_used", False),
                })
            self._increment_index_examined()

    def _match_full_path(self, query: Dict, entry: Dict) -> Tuple[float, Dict]:
        core_mult, alias_core = self._match_core_triple(
            query.get("path"), query.get("op", "eq"), query.get("value"),
            entry.get("path"), entry.get("op", "eq"), entry.get("value")
        )
        if core_mult == 0:
            return 0.0, {}

        obj_mult, alias_obj = self._object_match_mult(query.get("object"), entry.get("object"))
        act_mult, alias_act = self._action_match_mult(query.get("action"), entry.get("action"))

        total_mult = core_mult * obj_mult * act_mult

        if query.get("source_type") == "goalstate":
            total_mult *= CONFIG["goalstate_multiplier"]
        if entry.get("source_type") == "goalstate":
            total_mult *= CONFIG["goalstate_multiplier"]

        score = CONFIG["max_core_score"] * total_mult
        alias_used = alias_core or alias_obj or alias_act
        return score, {"alias_used": alias_used}

    def _match_core_triple(self, q_path, q_op, q_val, e_path, e_op, e_val):
        q_path = tuple(q_path)
        e_path = tuple(e_path)

        # Compute value match first
        value_mult = self._value_match_score(q_val, e_val)

        if value_mult == 0:
            # Filter out banned/unmatchable attribute components
            filtered_q = tuple(
                comp for comp in q_path
                if comp not in self._unmatchable_attr_components
            )
            filtered_e = tuple(
                comp for comp in e_path
                if comp not in self._unmatchable_attr_components
            )

            if not filtered_q or not filtered_e:
                return 0.0, False

            path_mult, alias_path = self._path_match_score(filtered_q, filtered_e)
            if path_mult == 0:
                return 0.0, False

            value_mult = CONFIG["value_no_match_mult"]
        else:
            path_mult, alias_path = self._path_match_score(q_path, e_path)
            if path_mult == 0:
                return 0.0, False

        op_mult = self._op_match_score(q_op, e_op)
        alias_used = alias_path
        return path_mult * value_mult * op_mult, alias_used

    def _path_match_score(self, q_path, e_path):
        q_path = tuple(q_path)
        e_path = tuple(e_path)
        q_len = len(q_path)
        e_len = len(e_path)
        if q_len == 0 or e_len == 0:
            return 0.0, False

        if self._is_contiguous_subsequence(q_path, e_path):
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

                aliases_q = self.index.alias_expansion.get("attribute_component_aliases", {}).get(q_comp, [])
                aliases_e = self.index.alias_expansion.get("attribute_component_aliases", {}).get(e_comp, [])
                if e_comp in aliases_q or q_comp in aliases_e:
                    if 1.0 > best_score:
                        best_score = 1.0
                        best_idx = i
                        best_alias = True
                    continue

                sim = self._similarity_to_specific_term(q_comp, "component_attribute", f"attribute:{e_comp}")
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
        contiguous = (len(pos_sorted) == q_len and max(pos_sorted) - min(pos_sorted) == q_len - 1)

        if contiguous:
            score = 0.9 if any_fuzzy else 1.0
        else:
            score = 0.8 if any_fuzzy else 0.9

        if alias_used:
            score *= CONFIG["alias_penalty"]
        return score, alias_used

    def _value_match_score(self, q_val, e_val):
        if not isinstance(q_val, str) or not isinstance(e_val, str):
            q_norm = self._normalize_value(q_val)
            e_norm = self._normalize_value(e_val)
            if q_norm == e_norm:
                return 1.0
            if isinstance(q_norm, (int, float)) and isinstance(e_norm, (int, float)):
                if e_norm != 0 and abs(q_norm - e_norm) / abs(e_norm) <= 0.3:
                    return CONFIG["scalar_within_30pct"]
            if isinstance(e_norm, list) and not isinstance(q_norm, list):
                if q_norm in e_norm:
                    return 0.7
                if isinstance(q_norm, str):
                    aliases_q = self.index.alias_expansion.get("value_aliases", {}).get(q_val, [])
                    for item in e_norm:
                        if isinstance(item, str):
                            aliases_e = self.index.alias_expansion.get("value_aliases", {}).get(item, [])
                            if item in aliases_q or q_val in aliases_e:
                                return 0.6 * CONFIG["alias_penalty"]
                            sim = self._similarity_to_specific_term(q_norm, "component_value", f"value:{item}")
                            if sim >= self.threshold:
                                return 0.6
            return 0.0

        q_norm = self._normalize_value(q_val)
        e_norm = self._normalize_value(e_val)
        if q_norm == e_norm:
            return 1.0

        if isinstance(q_norm, (int, float)) and isinstance(e_norm, (int, float)):
            if e_norm != 0 and abs(q_norm - e_norm) / abs(e_norm) <= 0.3:
                return CONFIG["scalar_within_30pct"]

        if isinstance(q_norm, list) and isinstance(e_norm, list):
            matched = 0
            for e_item in e_norm:
                if e_item in q_norm:
                    matched += 1
            if matched == len(e_norm):
                return 1.0
            elif matched > 0:
                return 0.7
            else:
                return 0.0

        if isinstance(e_norm, list) and not isinstance(q_norm, list):
            if q_norm in e_norm:
                return 0.7
            aliases_q = self.index.alias_expansion.get("value_aliases", {}).get(q_val, [])
            for item in e_norm:
                if isinstance(item, str):
                    aliases_e = self.index.alias_expansion.get("value_aliases", {}).get(item, [])
                    if item in aliases_q or q_val in aliases_e:
                        return 0.6 * CONFIG["alias_penalty"]
                    sim = self._similarity_to_specific_term(q_norm, "component_value", f"value:{item}")
                    if sim >= self.threshold:
                        return 0.6
            return 0.0

        aliases_q = self.index.alias_expansion.get("value_aliases", {}).get(q_val, [])
        aliases_e = self.index.alias_expansion.get("value_aliases", {}).get(e_val, [])
        if q_val in aliases_e or e_val in aliases_q:
            return 1.0 * CONFIG["alias_penalty"]

        base_term = f"value:{e_val}"
        sim = self._similarity_to_specific_term(q_val, "component_value", base_term)
        if sim >= self.threshold:
            return 0.9

        for alias in aliases_q:
            sim_alias = self._similarity_to_specific_term(alias, "component_value", base_term)
            if sim_alias >= self.threshold:
                return 0.9 * CONFIG["alias_penalty"]

        for alias in aliases_e:
            base_term_alias = f"value:{alias}"
            sim_alias2 = self._similarity_to_specific_term(q_val, "component_value", base_term_alias)
            if sim_alias2 >= self.threshold:
                return 0.9 * CONFIG["alias_penalty"]

        return 0.0

    def _op_match_score(self, q_op, e_op):
        q_op = str(q_op)
        e_op = str(e_op)
        if q_op == e_op:
            return 1.0
        if q_op == "!=" or e_op == "!=":
            return 0.5
        return 0.8

    def _object_match_mult(self, q_obj, e_obj):
        if q_obj is None or e_obj is None:
            return 1.0, False
        if q_obj == e_obj:
            return CONFIG["object_exact_mult"], False

        aliases_q = self.index.alias_expansion.get("object_aliases", {}).get(q_obj, [])
        aliases_e = self.index.alias_expansion.get("object_aliases", {}).get(e_obj, [])
        if e_obj in aliases_q or q_obj in aliases_e:
            return CONFIG["object_exact_mult"] * CONFIG["alias_penalty"], True

        sim_base = self._similarity_to_specific_term(q_obj, "object_names", f"object:{e_obj}")
        if sim_base >= self.threshold:
            return CONFIG["object_fuzzy_mult"], False

        for alias in aliases_e:
            sim_alias = self._similarity_to_specific_term(q_obj, "object_names", f"object_expanded:{alias}")
            if sim_alias >= self.threshold:
                return CONFIG["object_fuzzy_mult"] * CONFIG["alias_penalty"], True

        for alias in aliases_q:
            sim_alias_q = self._similarity_to_specific_term(alias, "object_names", f"object:{e_obj}")
            if sim_alias_q >= self.threshold:
                return CONFIG["object_fuzzy_mult"] * CONFIG["alias_penalty"], True

        return 1.0, False

    def _action_match_mult(self, q_action, e_action):
        if q_action is None or e_action is None:
            return 1.0, False
        if q_action == e_action:
            return CONFIG["action_exact_mult"], False

        aliases_q = self.index.alias_expansion.get("action_aliases", {}).get(q_action, [])
        aliases_e = self.index.alias_expansion.get("action_aliases", {}).get(e_action, [])
        if e_action in aliases_q or q_action in aliases_e:
            return CONFIG["action_exact_mult"] * CONFIG["alias_penalty"], True

        sim_base = self._similarity_to_specific_term(q_action, "action_names", f"action:{e_action}")
        if sim_base >= self.threshold:
            return CONFIG["action_fuzzy_mult"], False

        for alias in aliases_e:
            sim_alias = self._similarity_to_specific_term(q_action, "action_names", f"action_expanded:{alias}")
            if sim_alias >= self.threshold:
                return CONFIG["action_fuzzy_mult"] * CONFIG["alias_penalty"], True

        for alias in aliases_q:
            sim_alias_q = self._similarity_to_specific_term(alias, "action_names", f"action:{e_action}")
            if sim_alias_q >= self.threshold:
                return CONFIG["action_fuzzy_mult"] * CONFIG["alias_penalty"], True

        return 1.0, False

    def _normalize_value(self, val):
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

    def _is_contiguous_subsequence(self, small, large):
        if not small:
            return False
        m, n = len(small), len(large)
        for i in range(0, n - m + 1):
            if large[i:i + m] == small:
                return True
        return False

    # ------------------------------------------------------------------
    # Component matching
    # ------------------------------------------------------------------
    def _process_component_term(self, term: str, contributions,
                                is_goalstate_query: bool = False):
        prefix = get_term_prefix(term)
        query_type = self._component_type_from_prefix(prefix)
        if query_type is None:
            return
        query_phrase = term.split(":", 1)[1]

        group_name = "component_value" if query_type == TERM_TYPE_VALUE else "component_attribute"

        if term in self.index.index:
            base_term = self.index.alias_to_base.get(term, term)
            alias_used = term in self.index.alias_to_base
            postings = self.index.index[term]
            postings = self._deduplicate_postings(postings)
            self._add_frequency_controlled_contributions(
                term, base_term, postings, True, 1.0, query_type, alias_used,
                contributions, "component_exact", is_goalstate_query)
            self._increment_index_examined()
            return

        if group_name in self.group_embeddings:
            similar_terms = self._similarities_to_group(query_phrase, group_name)
            base_postings = defaultdict(list)
            alias_used_map = {}
            for full_term, sim in similar_terms:
                base_term = self.index.alias_to_base.get(full_term, full_term)
                alias_used = full_term in self.index.alias_to_base
                alias_used_map[base_term] = alias_used
                postings = self.index.index.get(full_term, [])
                base_postings[base_term].extend(postings)

            for base_term, postings in base_postings.items():
                postings = self._deduplicate_postings(postings)
                alias_used = alias_used_map[base_term]
                self._add_frequency_controlled_contributions(
                    term, base_term, postings, False, 0.9, query_type, alias_used,
                    contributions, "component_fuzzy", is_goalstate_query)
            self._increment_index_examined()

    def _component_type_from_prefix(self, prefix: str) -> Optional[str]:
        if prefix in (PREFIX_ATTRIBUTE, PREFIX_ATTRIBUTE_EXPANDED,
                      "statechangesC_attribute", "statechangesC_attribute_expanded"):
            return TERM_TYPE_ATTRIBUTE
        elif prefix in (PREFIX_VALUE, PREFIX_VALUE_EXPANDED,
                        "statechangesC_value", "statechangesC_value_expanded"):
            return TERM_TYPE_VALUE
        return None

    def _token_type_from_prefix(self, prefix: str) -> Optional[str]:
        mapping = {
            PREFIX_ATTRIBUTE_TOKEN: TERM_TYPE_ATTRIBUTE,
            PREFIX_ATTRIBUTE_EXPANDED_TOKEN: TERM_TYPE_ATTRIBUTE,
            PREFIX_VALUE_TOKEN: TERM_TYPE_VALUE,
            PREFIX_VALUE_EXPANDED_TOKEN: TERM_TYPE_VALUE,
            PREFIX_OBJECT_TOKEN: TERM_TYPE_OBJECT,
            PREFIX_OBJECT_EXPANDED_TOKEN: TERM_TYPE_OBJECT,
            PREFIX_ACTION_TOKEN: TERM_TYPE_ACTION,
            PREFIX_ACTION_EXPANDED_TOKEN: TERM_TYPE_ACTION,
            "statechangesC_attribute_token": TERM_TYPE_ATTRIBUTE,
            "statechangesC_attribute_expanded_token": TERM_TYPE_ATTRIBUTE,
            "statechangesC_value_token": TERM_TYPE_VALUE,
            "statechangesC_value_expanded_token": TERM_TYPE_VALUE,
        }
        return mapping.get(prefix)

    def _process_token_term(self, term: str, contributions,
                            is_goalstate_query: bool = False):
        prefix = get_term_prefix(term)
        query_type = self._token_type_from_prefix(prefix)
        if query_type is None:
            return
        query_phrase = term.split(":", 1)[1]

        group_name = {
            TERM_TYPE_ATTRIBUTE: "token_attribute",
            TERM_TYPE_VALUE: "token_value",
            TERM_TYPE_OBJECT: "token_object",
            TERM_TYPE_ACTION: "token_action",
        }.get(query_type)
        if group_name is None:
            return

        if term in self.index.index:
            base_term = self.index.alias_to_base.get(term, term)
            alias_used = term in self.index.alias_to_base
            postings = self.index.index[term]
            postings = self._deduplicate_postings(postings)
            self._add_frequency_controlled_contributions(
                term, base_term, postings, True, 1.0, query_type, alias_used,
                contributions, "token_exact", is_goalstate_query)
            self._increment_index_examined()
            return

        if group_name in self.group_embeddings:
            similar_terms = self._similarities_to_group(query_phrase, group_name)
            base_postings = defaultdict(list)
            alias_used_map = {}
            for full_term, sim in similar_terms:
                base_term = self.index.alias_to_base.get(full_term, full_term)
                alias_used = full_term in self.index.alias_to_base
                alias_used_map[base_term] = alias_used
                postings = self.index.index.get(full_term, [])
                base_postings[base_term].extend(postings)

            for base_term, postings in base_postings.items():
                postings = self._deduplicate_postings(postings)
                alias_used = alias_used_map[base_term]
                self._add_frequency_controlled_contributions(
                    term, base_term, postings, False, 0.9, query_type, alias_used,
                    contributions, "token_fuzzy", is_goalstate_query)
            self._increment_index_examined()

    def _add_frequency_controlled_contributions(self, query_term, matched_term,
                                                postings, exact, sim,
                                                query_type, alias_used,
                                                contributions, mode,
                                                is_goalstate_query: bool = False):
        freq = self._get_freq(matched_term)

        if mode.startswith("component"):
            if query_type == TERM_TYPE_ATTRIBUTE:
                if freq >= CONFIG["freq_attr_comp_unmatchable"]:
                    return
                one_match = (CONFIG["freq_attr_comp_one_match_min"] < freq <= CONFIG["freq_attr_comp_one_match_max"])
                diminishing = CONFIG["diminishing_attr_comp"]
            else:
                one_match = False
                diminishing = CONFIG["diminishing_values"]
        else:
            if query_type in (TERM_TYPE_ATTRIBUTE, TERM_TYPE_OBJECT, TERM_TYPE_ACTION):
                if freq >= CONFIG["freq_token_unmatchable"]:
                    return
                one_match = (CONFIG["freq_token_one_match_min"] < freq <= CONFIG["freq_token_one_match_max"])
                diminishing = CONFIG["diminishing_tokens"]
            else:
                one_match = False
                diminishing = CONFIG["diminishing_values"]

        mem_to_entries = defaultdict(list)
        for mem, src, term_type in postings:
            mem_to_entries[mem].append((src, term_type))

        for mem, entries in mem_to_entries.items():
            if one_match:
                entries = entries[:1]

            multipliers = []
            for idx in range(len(entries)):
                if idx < len(diminishing):
                    m = diminishing[idx]
                else:
                    m = 0.0
                multipliers.append(m)

            for idx, (src, term_type) in enumerate(entries):
                if multipliers[idx] == 0:
                    continue

                if mode.startswith("component"):
                    type_mult = COMPONENT_MULTIPLIERS[(exact, query_type == term_type)]
                else:
                    type_mult = TOKEN_MULTIPLIERS[(exact, query_type == term_type)]

                contribution = type_mult * multipliers[idx]

                if alias_used:
                    contribution *= CONFIG["alias_penalty"]

                # Per-posting goalstate detection: only amplifies when the
                # matched posting is actually a goalstate entry (its source
                # context says so), not merely because the term string happens
                # to exist in the global goalstate_terms set.
                is_goal_matched = _is_goalstate_posting(src)
                if is_goalstate_query and is_goal_matched:
                    contribution *= CONFIG["goalstate_multiplier"] ** 2
                elif is_goalstate_query or is_goal_matched:
                    contribution *= CONFIG["goalstate_multiplier"]

                contributions.append({
                    "memory": mem,
                    "object_id": None,
                    "query_term": query_term,
                    "matched_index_term": matched_term,
                    "score_contribution": contribution,
                    "mode": mode,
                    "source_context": src,
                    "term_type": term_type,
                    "similarity": sim if not exact else None,
                    "alias_used": alias_used,
                })

    # ------------------------------------------------------------------
    # Standalone object/action matching
    # ------------------------------------------------------------------
    def _process_object_term(self, term: str, contributions,
                             is_goalstate_query: bool = False):
        query_phrase = term.split(":", 1)[1]
        group_name = "object_names"
        if group_name not in self.group_embeddings:
            return

        if term in self.index.index:
            base_term = self.index.alias_to_base.get(term, term)
            alias_used = term in self.index.alias_to_base
            postings = self.index.index[term]
            postings = self._deduplicate_postings(postings)
            for mem, src, term_type in postings:
                contribution = 1.0
                if alias_used:
                    contribution *= CONFIG["alias_penalty"]
                is_goal_matched = _is_goalstate_posting(src)
                if is_goalstate_query and is_goal_matched:
                    contribution *= CONFIG["goalstate_multiplier"] ** 2
                elif is_goalstate_query or is_goal_matched:
                    contribution *= CONFIG["goalstate_multiplier"]
                contributions.append({
                    "memory": mem,
                    "object_id": None,
                    "query_term": term,
                    "matched_index_term": base_term,
                    "score_contribution": contribution,
                    "mode": "object_exact",
                    "source_context": src,
                    "term_type": term_type,
                    "similarity": None,
                    "alias_used": alias_used,
                })
            self._increment_index_examined()
            return

        similar_terms = self._similarities_to_group(query_phrase, group_name)
        base_postings = defaultdict(list)
        alias_used_map = {}
        for full_term, sim in similar_terms:
            base_term = self.index.alias_to_base.get(full_term, full_term)
            alias_used = full_term in self.index.alias_to_base
            alias_used_map[base_term] = alias_used
            postings = self.index.index.get(full_term, [])
            base_postings[base_term].extend(postings)

        for base_term, postings in base_postings.items():
            postings = self._deduplicate_postings(postings)
            alias_used = alias_used_map[base_term]
            for mem, src, term_type in postings:
                contribution = 0.9
                if alias_used:
                    contribution *= CONFIG["alias_penalty"]
                is_goal_matched = _is_goalstate_posting(src)
                if is_goalstate_query and is_goal_matched:
                    contribution *= CONFIG["goalstate_multiplier"] ** 2
                elif is_goalstate_query or is_goal_matched:
                    contribution *= CONFIG["goalstate_multiplier"]
                contributions.append({
                    "memory": mem,
                    "object_id": None,
                    "query_term": term,
                    "matched_index_term": base_term,
                    "score_contribution": contribution,
                    "mode": "object_fuzzy",
                    "source_context": src,
                    "term_type": term_type,
                    "similarity": 0.9,
                    "alias_used": alias_used,
                })
        self._increment_index_examined()

    def _process_action_term(self, term: str, contributions,
                             is_goalstate_query: bool = False):
        query_phrase = term.split(":", 1)[1]
        group_name = "action_names"
        if group_name not in self.group_embeddings:
            return

        if term in self.index.index:
            base_term = self.index.alias_to_base.get(term, term)
            alias_used = term in self.index.alias_to_base
            postings = self.index.index[term]
            postings = self._deduplicate_postings(postings)
            for mem, src, term_type in postings:
                contribution = 1.0
                if alias_used:
                    contribution *= CONFIG["alias_penalty"]
                is_goal_matched = _is_goalstate_posting(src)
                if is_goalstate_query and is_goal_matched:
                    contribution *= CONFIG["goalstate_multiplier"] ** 2
                elif is_goalstate_query or is_goal_matched:
                    contribution *= CONFIG["goalstate_multiplier"]
                contributions.append({
                    "memory": mem,
                    "object_id": None,
                    "query_term": term,
                    "matched_index_term": base_term,
                    "score_contribution": contribution,
                    "mode": "action_exact",
                    "source_context": src,
                    "term_type": term_type,
                    "similarity": None,
                    "alias_used": alias_used,
                })
            self._increment_index_examined()
            return

        similar_terms = self._similarities_to_group(query_phrase, group_name)
        base_postings = defaultdict(list)
        alias_used_map = {}
        for full_term, sim in similar_terms:
            base_term = self.index.alias_to_base.get(full_term, full_term)
            alias_used = full_term in self.index.alias_to_base
            alias_used_map[base_term] = alias_used
            postings = self.index.index.get(full_term, [])
            base_postings[base_term].extend(postings)

        for base_term, postings in base_postings.items():
            postings = self._deduplicate_postings(postings)
            alias_used = alias_used_map[base_term]
            for mem, src, term_type in postings:
                contribution = 0.9
                if alias_used:
                    contribution *= CONFIG["alias_penalty"]
                is_goal_matched = _is_goalstate_posting(src)
                if is_goalstate_query and is_goal_matched:
                    contribution *= CONFIG["goalstate_multiplier"] ** 2
                elif is_goalstate_query or is_goal_matched:
                    contribution *= CONFIG["goalstate_multiplier"]
                contributions.append({
                    "memory": mem,
                    "object_id": None,
                    "query_term": term,
                    "matched_index_term": base_term,
                    "score_contribution": contribution,
                    "mode": "action_fuzzy",
                    "source_context": src,
                    "term_type": term_type,
                    "similarity": 0.9,
                    "alias_used": alias_used,
                })
        self._increment_index_examined()

    # ------------------------------------------------------------------
    # Verb matching
    # ------------------------------------------------------------------
    def _process_verb_term(self, term: str, contributions,
                           is_goalstate_query: bool = False):
        is_target = term.startswith("verb_target:")
        verb = term.split(":", 1)[1]
        cap = CONFIG["verb_target_cap"] if is_target else CONFIG["verb_context_cap"]

        if "verb" not in self.group_embeddings:
            return

        q_vec = self._query_embedding(verb)
        sims = self.group_embeddings["verb"] @ q_vec

        # Group matches by memory, collect all candidate postings for each memory
        mem_matches = defaultdict(list)

        for idx, sim in enumerate(sims):
            if sim >= self.threshold:
                full_term = self.group_terms["verb"][idx]
                postings = self.index.index.get(full_term, [])
                exact = (sim == 1.0)

                base_score = CONFIG["verb_exact_score"] if exact else CONFIG["verb_fuzzy_score"]

                for mem, src, term_type in postings:
                    # Per-posting goalstate detection for the 2x bonus
                    is_goal_matched = _is_goalstate_posting(src)
                    actual_score = base_score
                    if not (is_goalstate_query or is_goal_matched):
                        actual_score *= 2.0
                    mem_matches[mem].append(
                        (actual_score, full_term, src, term_type, sim, exact)
                    )

        # Apply per-memory cap: keep top `cap` matches within each memory
        for mem, matches in mem_matches.items():
            matches.sort(key=lambda x: -x[0])

            for score, idx_term, src, term_type, sim, exact in matches[:cap]:
                contributions.append({
                    "memory": mem,
                    "object_id": None,
                    "query_term": term,
                    "matched_index_term": idx_term,
                    "score_contribution": score,
                    "mode": "verb_exact" if exact else "verb_fuzzy",
                    "source_context": src,
                    "term_type": term_type,
                    "similarity": sim,
                    "alias_used": False,
                })

    # ------------------------------------------------------------------
    # Goalstate term handling (query side)
    # ------------------------------------------------------------------
    def _process_goalstate_term(self, term, contributions):
        inner = term.split(":", 1)[1]
        self._process_term(inner, contributions, is_goalstate_query=True)

    # ------------------------------------------------------------------
    # Regular term handling (exact fallback)
    # ------------------------------------------------------------------
    def _process_regular_term(self, term, contributions,
                              is_goalstate_query: bool = False):
        if term in self.index.index:
            for mem, src, term_type in self.index.index[term]:
                contribution = 1.0
                is_goal_matched = _is_goalstate_posting(src)
                if is_goalstate_query and is_goal_matched:
                    contribution *= CONFIG["goalstate_multiplier"] ** 2
                elif is_goalstate_query or is_goal_matched:
                    contribution *= CONFIG["goalstate_multiplier"]
                contributions.append({
                    "memory": mem,
                    "object_id": None,
                    "query_term": term,
                    "matched_index_term": term,
                    "score_contribution": contribution,
                    "mode": "exact",
                    "source_context": src,
                    "term_type": term_type,
                    "alias_used": False,
                })
            self._increment_index_examined()

    # ------------------------------------------------------------------
    # Position unit matching
    # ------------------------------------------------------------------
    def _process_position_unit_term(self, term, contributions):
        parts = term.split(":", 1)[1].split(":")
        if len(parts) != 4:
            return
        q_obj, q_rel, q_relr, q_rel_to = parts
        for post_term, postings in self.index.index.items():
            if not post_term.startswith("position_unit:"):
                continue
            post_parts = post_term.split(":", 1)[1].split(":")
            if len(post_parts) != 4:
                continue
            e_obj, e_rel, e_relr, e_rel_to = post_parts
            rel_to_mult = 1.0 if q_rel_to == e_rel_to or self._similarity_to_specific_term(q_rel_to, "component_value", f"value:{e_rel_to}") >= self.threshold else 0.0
            if rel_to_mult == 0:
                continue
            relr_mult = 1.0 if q_relr == e_relr else 0.5
            rel_mult = 1.0 if q_rel == e_rel else 0.8
            total_mult = rel_to_mult * relr_mult * rel_mult
            for mem, src, term_type in postings:
                contributions.append({
                    "memory": mem,
                    "object_id": q_obj,
                    "query_term": term,
                    "matched_index_term": post_term,
                    "score_contribution": total_mult,
                    "mode": "position_unit",
                    "source_context": src,
                    "term_type": term_type,
                    "alias_used": False,
                })
            self._increment_index_examined()

    # ------------------------------------------------------------------
    # Spatial one-hop
    # ------------------------------------------------------------------
    def _process_spatial_term(self, term, contributions):
        m = re.match(r"^spatial:(\w+)\(([^,]+),([^)]+)\)$", term)
        if not m:
            return
        op, subj_str, target_str = m.groups()
        if op not in {"near", "on", "in", "below", "above", "around",
                      "through", "contact", "attached", "between"}:
            return

        for mem in self.index.memories:
            obj_map = {obj.obj_id: obj for obj in mem.objects.values()}
            subj = obj_map.get(subj_str) or next(
                (o for o in mem.objects.values() if o.template_name == subj_str), None)
            target = obj_map.get(target_str) or next(
                (o for o in mem.objects.values() if o.template_name == target_str), None)
            if not subj or not target:
                continue

            if subj.position:
                rel_r = subj.position.get("relationr")
                rel_to = subj.position.get("relative_to")
                if rel_r and rel_to in (target.obj_id, target.template_name):
                    if rel_r == op:
                        contributions.append({
                            "memory": mem,
                            "object_id": subj.obj_id,
                            "query_term": term,
                            "matched_index_term": f"positionr:{rel_r}:{rel_to}",
                            "score_contribution": 1.0,
                            "mode": "spatial_direct",
                            "source_context": {"type": "object", "object_id": subj.obj_id,
                                               "template": subj.template_name},
                            "term_type": TERM_TYPE_ATTRIBUTE,
                            "alias_used": False,
                        })
                        self._increment_index_examined()
                        continue

            for mid in mem.objects.values():
                if mid.obj_id in (subj.obj_id, target.obj_id):
                    continue
                if not subj.position or not mid.position:
                    continue
                if subj.position.get("relative_to") == mid.obj_id:
                    rel_subj = subj.position.get("relationr")
                    if rel_subj and rel_subj in SPATIAL_OPERATORS:
                        if mid.position.get("relative_to") == target.obj_id:
                            rel_mid = mid.position.get("relationr")
                            if rel_mid and rel_mid in SPATIAL_OPERATORS:
                                contributions.append({
                                    "memory": mem,
                                    "object_id": subj.obj_id,
                                    "query_term": term,
                                    "matched_index_term": f"inferred via {mid.obj_id}",
                                    "score_contribution": 0.5,
                                    "mode": "spatial_one_hop",
                                    "source_context": {"type": "object", "object_id": subj.obj_id,
                                                       "template": subj.template_name},
                                    "term_type": TERM_TYPE_ATTRIBUTE,
                                    "alias_used": False,
                                })
                                self._increment_index_examined()
                                break