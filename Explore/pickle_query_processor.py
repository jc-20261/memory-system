#!/usr/bin/env python3
r"""
pickle_query_processor.py – Complete updated query processor.

This version:

- STAGE1 generates COMPLETE object templates from scratch. Every object returned
  by STAGE1 must include all six DEFINING ATTRIBUTES plus a template name.
  Instance attributes are also generated, with `inferred: true` when the value
  is not directly supported by the query.

- A targeted retry fills in any missing defining attributes on a per-object
  basis (OBJECT_DEFINING_FILL_PROMPT).

- STAGE2 (object template generation from storage) is retired; the query
  processor no longer touches object_storage. The prompt is retained in this
  file for reference only.

- A separate STAGE_ALIASES call generates query-side aliases for objects,
  actions, attribute components, and values. These are merged with the
  offline alias_expansion.json aliases during search term generation.

- A query-side alias map (get_last_query_aliases()) is exposed for the search
  engine to fold alias terms into their base terms during search.

Planning support:
- captures resolved structured data after search term generation
- provides get_last_planning_data() and get_or_create_planning_data()
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import AsyncOpenAI

from pickle_search_index import (
    CONFIG,
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
    BANNED_ATTRIBUTE_COMPONENTS,
    STOP_WORDS,
    UNIT_WORDS,
    get_term_prefix,
    tokenize_into_words,
    is_valid_token,
    normalize_measurement,
    parse_measurement_string,
    derive_quantity_from_unit,
)

# =============================================================================
# Paths and configuration
# =============================================================================
EXPLORE_DIR = Path(__file__).parent
DATA_DIR = EXPLORE_DIR / "data"
MANUAL_TERMS_FILE = EXPLORE_DIR / "query_terms.txt"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"
REASONING_EFFORT = "high"

# Retry configuration
OBJECT_TEMPLATE_MAX_ATTEMPTS = 2   # 1 initial + 1 retry (for missing defining fields)
ALIAS_MAX_ATTEMPTS = 3             # 1 initial + 2 retries (query-side alias generation)
TEMPLATE_GEN_MAX_RETRIES = 2       # used by _call_llm_json_with_retry

# Defining attribute names that must be present for every object
DEFINING_ATTRIBUTE_KEYS = [
    "template_name",
    "category_hierarchy",
    "functions",
    "materials",
    "shape",
    "dimensions",
    "position",
]

# Alias cap per term (consistent with offline generation)
ALIAS_CAP_PER_TERM = 10

# =============================================================================
# STAGE1 prompt – updated with complete object templates
# =============================================================================
STAGE1_PROMPT = """
You are a query analyzer for a memory system.

Convert the user's question into a structured JSON representation of the query.
The JSON will be used only to search the memory system. Do NOT answer the user's question directly.

Important principles:

1. Distinguish between:
   - target actions: the actions the user is asking about; the answer should come from the memory system.
   - context actions: actions already stated or presupposed in the query.
   - initial conditions: current or presumed states.
   - goal state: the desired result.

2. Expand context actions as much as possible.
   - For every non-target action, provide hierarchical sub-actions, preconditions, changes, tags, trajectory, category, and verb.
   - Use the same style as the memory encoder's annotated action plot and final JSON encoding.
   - This gives the memory system more context.

3. Do NOT expand the target actions. 
   - For each target action, provide only:
       - name
       - participants
       - is_target: true
       - verb: if a clear verb exists (e.g., "open", "clean"). The verb should capture the essence of the target action. 
   - Do NOT provide sub_actions, preconditions, changes, changes_total, or tags for the target actions.
   - Otherwise, the query processor would be solving the problem instead of searching memory.

4. If the query implies observation, inspection, or visual checking:
   - Include a "person" object as "observer".
   - Include a perception action such as "look_at", "inspect_object", etc.
   - Include sensory perception attributes on the person, with "source" referring to the relevant object.

5. For query objects:

   Generate a COMPLETE object template for every object mentioned or implied in
   the query. Start fresh. Do NOT assume any pre-existing object storage.
   Produce the description entirely from your own knowledge of what this kind
   of object is like. Use a canonical, semantic template name (e.g.
   "microwave_oven", "canned_fruit") rather than a raw physical noun where a
   more precise term exists.

   The template must include the following DEFINING ATTRIBUTES. These must
   ALWAYS be present, even if the query does not mention them and they are not
   relevant to the task. Provide your best estimate. Do not omit any field.

     - template_name        : canonical semantic name
     - category_hierarchy   : list from specific to abstract, ending with
                              "physical_object" and "object"
     - functions            : list of 1-5 plausible functions
     - materials            : LIST of materials (not a single string)
     - shape                : single word or short phrase
     - dimensions           : {length, width, height_thickness}, each with
                              value and unit
     - position             : {relation, relative_to}, best-guess default for
                              this object type

   Additionally, include COMMON INSTANCE ATTRIBUTES that are typical for this
   kind of object. Examples:
     - color, texture, surface_state, cleanliness, temperature, mass, weight
     - any other attributes commonly associated with this class of object

   Also include every attribute explicitly stated in the query.

   Mark any attribute whose value is not directly supported by the query with
   "inferred": true. Attributes explicitly stated in the query must NOT be
   marked inferred.

   For "person" or protagonist objects, do NOT include physical attributes
   (dimensions, mass, shape, posture) or mental states (fatigue, focus), unless
   the query explicitly mentions them. Only include sensory perceptions or
   explicitly mentioned attributes.

   Return each object as:

   {
     "name": "microwave_oven",
     "template_name": "microwave_oven",
     "category_hierarchy": ["microwave_oven", "kitchen_appliance", "appliance",
                            "physical_object", "object"],
     "functions": ["heating food", "cooking", "defrosting"],
     "materials": ["metal", "plastic", "glass"],
     "shape": "rectangular",
     "dimensions": {
       "length": {"value": 50, "unit": "cm", "inferred": true},
       "width":  {"value": 40, "unit": "cm", "inferred": true},
       "height_thickness": {"value": 30, "unit": "cm", "inferred": true}
     },
     "position": {"relation": "on", "relative_to": "counter", "inferred": true},
     "attributes": [
       {"path": "color", "value": "silver", "inferred": true},
       {"path": "surface_state", "value": "clean", "inferred": true}
     ],
     "role": "target_object"
   }

   Defining attributes are top-level keys. Instance attributes go in the
   "attributes" list. Do not mix them.

6. For actions:
   - Always include the target actions, even if they are barebones.
   - each target action must have "is_target": true and no sub_actions, preconditions, or changes.
   - Context actions can be expanded.
   - For every action (target or context), if a clear verb exists, include a "verb" field.
   - Participants should be basic natural descriptions like person or box, not drawn out phrases.

7. For preconditions and changes (including changes_per_cycle, changes_total, conditional_changes):
   - Each condition or change must include an "op" field. Common operators: "eq", "ne", "gt", "gte", "lt", "lte", "!=", "+=", "in".
   - If a change has both an old and a new value, include both "old" and "new".
   - For conditional changes, include "conditional_changes" as a list of objects, each with "condition" and "changes".

8. If the query involves spatial relationships (e.g., "above", "below", "inside", "near", "on"):
   - Include a condition object with:
       "object": <subject object>,
       "relation": <original relation string if available>,
       "relationr": <normalized relation operator from the allowed set: on, in, below, above, around, through, contact, attached, between, near, apart>,
       "relative_to": <reference object>

9. For compound or multi-stage/multigoal queries with multiple target actions, such as "how do I turn on the radio and listen to music?", the JSON output should be formatted so that target actions appear in the same order as in the query. SO for the example given, there would be a field for "turn_on", followed by a field for "listen". Both would be target actions, with respective verbs of "turn_on" and "listen". Target actions should appear before context actions, under query actions. In addition, if the query is multi-stage, it could involve multiple goals, and the goal states should reflect this. The goal states should be concise but carefully crafted. There can be more than one condition for each goal, in order to capture the essence of the goal state.  

10. Return JSON with this structure:

{
  "query_objects": [ {...}, ... ],
  "query_actions": [ {...}, ... ],
  "initial_conditions": [ {...}, ... ],
  "goal_state": [ {...}, ... ]
}

For non-target actions, you may include as many sub-action levels as useful.
For target action, do not include sub_actions, preconditions, changes, or other expansion fields.

DEMONSTRATION:

Example query:
How do I clean a microwave?

Expected output:
{
  "query_objects": [
    {
      "name": "person",
      "template_name": "person",
      "role": "actor"
    },
    {
      "name": "microwave_oven",
      "template_name": "microwave_oven",
      "category_hierarchy": ["microwave_oven", "kitchen_appliance", "appliance", "physical_object", "object"],
      "functions": ["heating food", "cooking", "defrosting"],
      "materials": ["metal", "plastic", "glass"],
      "shape": "rectangular",
      "dimensions": {
        "length": {"value": 50, "unit": "cm", "inferred": true},
        "width":  {"value": 40, "unit": "cm", "inferred": true},
        "height_thickness": {"value": 30, "unit": "cm", "inferred": true}
      },
      "position": {"relation": "on", "relative_to": "counter", "inferred": true},
      "attributes": [
        {"path": "color", "value": "silver", "inferred": true},
        {"path": "surface_state", "value": "dirty", "inferred": true}
      ],
      "role": "target_object"
    }
  ],
  "query_actions": [
    {
      "name": "clean",
      "participants": ["person", "microwave_oven"],
      "is_target": true,
      "verb": "clean"
    }
  ],
  "initial_conditions": [
    {
      "object": "microwave_oven",
      "attribute": "surface_state",
      "op": "eq",
      "value": "dirty"
    }
  ],
  "goal_state": [
    {
      "object": "microwave_oven",
      "attribute": "surface_state",
      "op": "eq",
      "value": "clean"
    }
  ]
}
"""

# =============================================================================
# STAGE2 prompt – RETIRED, kept for reference only.
# =============================================================================
STAGE2_OBJECT_TEMPLATE_PROMPT = """
[RETIRED] This prompt is no longer invoked by the query processor.
The functionality has been merged into STAGE1_PROMPT section 5.
Retained here for reference in case the storage-based template lookup is
reintroduced later.
"""

# =============================================================================
# STAGE3 prompt – context action template generation (unchanged)
# =============================================================================
STAGE3_ACTION_TEMPLATE_PROMPT = """
You are an action decomposition engine for a memory search system.

The query parser encountered a context action that is not in the action storage.
Expand this context action into a structured action tree. This is used only to improve search terms.
Do NOT answer any how-to question, and do NOT expand the target action.

Return JSON with this structure:

{
  "name": "open_can",
  "duration": 10,
  "action_category": "TOOL_USAGE",
  "kinematic_trajectory": "circular",
  "temporal_type": "sequential",
  "participants": ["canned_fruit", "can_opener"],
  "verb": "open",
  "tags": ["can", "opening", "lid removal"],
  "preconditions": [
    {
      "object": "canned_fruit",
      "attribute": "lid.state",
      "op": "eq",
      "value": "closed"
    }
  ],
  "changes_total": [
    {
      "object": "canned_fruit",
      "attribute": "lid.state",
      "old": "closed",
      "new": "open",
      "op": "eq"
    }
  ],
  "sub_actions": [
    {
      "name": "position_can_opener",
      "duration": 2,
      "action_category": "MANIPULATION",
      "kinematic_trajectory": "straight",
      "temporal_type": "sequential",
      "participants": ["canned_fruit", "can_opener"],
      "verb": "position",
      "preconditions": [],
      "changes": [],
      "sub_actions": []
    }
  ]
}

Important:
- Use the same style as the memory encoder's annotated action plot and final JSON encoding.
- Include preconditions, changes, changes_per_cycle, changes_total, conditional_changes, and sub_actions as appropriate.
- For every condition or change, include an "op" field. Common operators: "eq", "ne", "gt", "gte", "lt", "lte", "!=", "+=", "in".
- If a change has both old and new values, include both "old" and "new".
- Every action (including sub-actions) should have a "verb" field if a clear verb exists.
- Participants must be object IDs or template names, not natural language descriptions.
- Do not include any commentary.
"""

# =============================================================================
# Defining-attribute fill prompt – targeted retry for incomplete objects
# =============================================================================
OBJECT_DEFINING_FILL_PROMPT = """
You are a knowledge completion engine for object descriptions.

An object has been described by a user but is missing some DEFINING ATTRIBUTES.
Provide best-effort values for the missing fields only.

The DEFINING ATTRIBUTES are:
  - category_hierarchy   : list from specific to abstract, ending with
                           "physical_object" and "object"
  - functions            : list of 1-5 plausible functions
  - materials            : LIST of materials (not a single string)
  - shape                : single word or short phrase
  - dimensions           : {length, width, height_thickness}, each with value
                           and unit
  - position             : {relation, relative_to}

Mark any value that is not directly supported by the query with "inferred": true.

Return a JSON object containing ONLY the missing fields.
Do not include fields that are already present.
Do not include commentary.
"""

# =============================================================================
# STAGE_ALIASES prompt – query-side alias generation
# =============================================================================
STAGE_ALIASES_PROMPT = """
You are a synonym generator for a memory search system.

For each term below, produce up to 10 concise alternative names (aliases).

Guidelines:
- Join words with underscores instead of spaces.
- Do not include the original term itself.
- Aliases should be plausible synonyms, near-synonyms, or alternative common names.
- Do not include explanations.
- If a term has no good aliases, return an empty list for it.

Return a single JSON object with exactly these four keys:

{
  "object_aliases":             {"<template_name>": ["alias1", ...], ...},
  "action_aliases":             {"<action_name>":   ["alias1", ...], ...},
  "attribute_component_aliases": {"<component>":    ["alias1", ...], ...},
  "value_aliases":              {"<value>":         ["alias1", ...], ...}
}

Every input term must appear as a key in its corresponding dictionary, even if
its alias list is empty.
"""

# =============================================================================
# Query processor
# =============================================================================
class PickleQueryProcessor:
    def __init__(self, loader):
        self.loader = loader
        self.object_storage = loader.object_storage
        self.action_storage_general = loader.action_storage_general
        self.action_storage_alts = loader.action_storage_alts
        self.verb_storage = loader.verb_storage
        self.alias_expansion = loader.alias_expansion
        self.client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.mode = "manual"

        # Stores the last resolved structured query data for planning reuse
        self.last_planning_data: Optional[Dict[str, Any]] = None

        # Stores the last resolved objects and actions (for alias extraction and
        # for downstream consumers such as the planner)
        self._last_resolved_objects: List[Dict[str, Any]] = []
        self._last_resolved_actions: List[Dict[str, Any]] = []

        # Query-side aliases generated for the last query
        self._last_query_aliases: Dict[str, Dict[str, List[str]]] = {
            "object_aliases": {},
            "action_aliases": {},
            "attribute_component_aliases": {},
            "value_aliases": {},
        }

        # Alias-term -> base-term map for the last query
        self._last_query_alias_map: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------
    def set_mode(self, mode: str):
        if mode in {"manual", "llm"}:
            self.mode = mode
        else:
            raise ValueError("Mode must be 'manual' or 'llm'")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def generate_terms(self, query: str) -> List[Tuple[str, float]]:
        if self.mode == "manual":
            terms = self._load_manual_terms()
        else:
            terms = await self._generate_terms_llm(query)

        # Final filter: remove explicit unmatchable terms and their goal-state variants
        explicit_unmatchable = {
            "attribute:state",
            "value:true",
            "value:false",
            "goalstate:attribute:state",
            "goalstate:value:true",
            "goalstate:value:false",
        }
        return [(t, w) for t, w in terms if t not in explicit_unmatchable]

    def get_last_query_aliases(self) -> Dict[str, str]:
        """
        Return {alias_term -> base_term} for the most recent generate_terms call.
        The search engine merges this into its _query_alias_to_base map.
        """
        return self._last_query_alias_map

    def get_last_planning_data(self) -> Optional[Dict[str, Any]]:
        return self.last_planning_data

    def get_or_create_planning_data(self) -> Dict[str, Any]:
        if self.last_planning_data is not None:
            return self.last_planning_data
        return {
            "target_verbs": [],
            "objects": [],
            "initial_conditions": [],
            "goal_conditions": [],
            "action_category": None,
        }

    # ------------------------------------------------------------------
    # Manual terms
    # ------------------------------------------------------------------
    def _load_manual_terms(self) -> List[Tuple[str, float]]:
        if not MANUAL_TERMS_FILE.exists():
            return []
        terms = []
        with open(MANUAL_TERMS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|", 1)
                term = parts[0].strip()
                weight = 1.0
                if len(parts) == 2:
                    try:
                        weight = float(parts[1].strip())
                    except ValueError:
                        weight = 1.0
                if term:
                    terms.append((term, weight))
        return terms

    # ------------------------------------------------------------------
    # Banned / filter helpers
    # ------------------------------------------------------------------
    def _is_banned_component(self, comp: str) -> bool:
        return comp in BANNED_ATTRIBUTE_COMPONENTS

    def _is_stop_or_unit(self, token: str) -> bool:
        return token in STOP_WORDS or token in UNIT_WORDS

    # ------------------------------------------------------------------
    # LLM orchestration
    # ------------------------------------------------------------------
    async def _generate_terms_llm(self, query: str) -> List[Tuple[str, float]]:
        # Reset per-query alias state
        self._last_query_aliases = {
            "object_aliases": {},
            "action_aliases": {},
            "attribute_component_aliases": {},
            "value_aliases": {},
        }
        self._last_query_alias_map = {}

        # Stage 1: parse
        parsed = await self._call_llm_json(STAGE1_PROMPT, query, "stage1_parse")
        if parsed is None:
            return self._fallback_terms(query)

        # Ensure all defining attributes present (retry if not)
        parsed = await self._ensure_defining_attributes(parsed, query)

        # Resolve objects and actions
        resolved_objects = await self._resolve_objects(parsed.get("query_objects", []))
        resolved_actions = await self._resolve_actions(parsed.get("query_actions", []))

        self._last_resolved_objects = resolved_objects
        self._last_resolved_actions = resolved_actions

        # Build planning-ready structured data
        planning_objects = []
        for obj in resolved_objects:
            planning_objects.append({
                "template_name": obj.get("template_name"),
                "obj_id": obj.get("obj_id") or obj.get("template_name"),
                "categories": obj.get("categories", []),
                "attributes": obj.get("attributes", {}),
                "functions": obj.get("functions", []),
                "materials": obj.get("materials", []),
                "shape": obj.get("shape", ""),
                "dimensions": obj.get("dimensions", {}),
                "position": obj.get("position", {}),
            })

        target_verbs = []
        action_category = None
        for act in resolved_actions:
            if act.get("is_target", False):
                verb = act.get("verb") or act.get("name")
                if verb:
                    target_verbs.append(verb)
                if not action_category and act.get("action_category"):
                    action_category = act["action_category"]

        target_actions = []
        for act in resolved_actions:
            if act.get("is_target", False):
                target_actions.append({
                    "name": act.get("name"),
                    "verb": act.get("verb") or act.get("name"),
                    "participants": act.get("participants", []),
                })

        self.last_planning_data = {
            "target_verbs": target_verbs,
            "target_actions": target_actions,
            "objects": planning_objects,
            "initial_conditions": parsed.get("initial_conditions", []),
            "goal_conditions": parsed.get("goal_state", []),
            "action_category": action_category,
        }

        # Extract base terms for alias generation
        base_terms = self._extract_base_terms(resolved_objects, resolved_actions, parsed)

        # Generate query-side aliases (with retries)
        await self._generate_aliases(base_terms)

        # Build search terms
        terms: List[Tuple[str, float]] = []

        for obj in resolved_objects:
            self._add_object_terms(obj, terms)

        for act in resolved_actions:
            self._add_action_terms(act, terms)

        initial_conditions = parsed.get("initial_conditions", [])
        goal_conditions = parsed.get("goal_state", [])
        for cond in initial_conditions + goal_conditions:
            if isinstance(cond, dict):
                is_goal = cond in goal_conditions
                self._add_condition_terms(cond, terms, is_goal=is_goal)

        # Phrase decomposition
        terms = self._add_phrase_decomposition(terms)

        return terms

    # ------------------------------------------------------------------
    # Defining-attribute validation and retry
    # ------------------------------------------------------------------
    async def _ensure_defining_attributes(self, parsed: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Check every query object for the six defining attributes plus
        template_name. If any are missing, run a targeted fill retry.
        """
        query_objects = parsed.get("query_objects", [])
        if not isinstance(query_objects, list):
            return parsed

        for attempt in range(OBJECT_TEMPLATE_MAX_ATTEMPTS):
            missing_by_obj = self._find_missing_defining(query_objects)
            if not missing_by_obj:
                return parsed

            if attempt == OBJECT_TEMPLATE_MAX_ATTEMPTS - 1:
                # Out of retries; accept what we have
                print(f"⚠️ Objects still missing defining attributes after {OBJECT_TEMPLATE_MAX_ATTEMPTS} attempts: "
                      f"{[m['name'] for m in missing_by_obj]}")
                return parsed

            # Fill missing fields for each incomplete object
            for entry in missing_by_obj:
                obj = entry["obj"]
                missing_fields = entry["missing"]
                fill_result = await self._fill_missing_defining(obj, missing_fields, original_query)
                if isinstance(fill_result, dict):
                    for key, value in fill_result.items():
                        if key in missing_fields:
                            obj[key] = value

        return parsed

    def _find_missing_defining(self, query_objects: List[Dict]) -> List[Dict[str, Any]]:
        """
        Return a list of entries {name, obj, missing} for objects that are
        missing any defining attribute.
        """
        result = []
        for obj in query_objects:
            if not isinstance(obj, dict):
                continue
            name = obj.get("template_name") or obj.get("name") or "unknown"
            # Person objects are exempt from dimensions, shape, position
            is_person = (
                str(obj.get("template_name", "")).lower() == "person"
                or obj.get("role") in {"actor", "observer", "protagonist"}
            )
            missing = []
            for key in DEFINING_ATTRIBUTE_KEYS:
                if key == "template_name":
                    if not obj.get("template_name"):
                        missing.append(key)
                    continue
                if is_person and key in {"category_hierarchy", "functions", "materials",
                                         "shape", "dimensions", "position"}:
                    # Person objects only need a template_name
                    continue
                if key not in obj or obj.get(key) in (None, "", [], {}):
                    missing.append(key)
            if missing:
                result.append({"name": name, "obj": obj, "missing": missing})
        return result

    async def _fill_missing_defining(
        self,
        obj: Dict[str, Any],
        missing_fields: List[str],
        original_query: str,
    ) -> Optional[Dict[str, Any]]:
        user_msg = (
            f"Original query: {original_query}\n\n"
            f"Partial object (JSON):\n{json.dumps(obj, indent=2, ensure_ascii=False)}\n\n"
            f"Missing defining fields: {missing_fields}\n\n"
            "Provide best-effort values for the missing fields only."
        )
        return await self._call_llm_json(
            OBJECT_DEFINING_FILL_PROMPT, user_msg, "defining_fill"
        )

    # ------------------------------------------------------------------
    # Base term extraction
    # ------------------------------------------------------------------
    def _extract_base_terms(
        self,
        objects: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
        parsed: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        objects_set: Set[str] = set()
        actions_set: Set[str] = set()
        attr_components_set: Set[str] = set()
        values_set: Set[str] = set()

        # ---- Objects ----
        for obj in objects:
            tname = obj.get("template_name")
            if tname:
                objects_set.add(str(tname))

        # ---- Actions (recursive) ----
        def _collect_action(act):
            if not isinstance(act, dict):
                return
            name = act.get("name")
            if name:
                actions_set.add(str(name))
            verb = act.get("verb")
            if verb:
                actions_set.add(str(verb))
            for sub in act.get("sub_actions", []) or []:
                _collect_action(sub)

        for act in actions:
            _collect_action(act)

        # ---- Attribute components ----
        def _collect_path(path):
            if path is None:
                return
            if isinstance(path, str):
                parts = path.split(".")
            elif isinstance(path, (list, tuple)):
                parts = [str(p) for p in path]
            else:
                parts = [str(path)]
            for comp in parts:
                comp = comp.strip()
                if not comp:
                    continue
                if self._is_banned_component(comp):
                    continue
                if self._is_stop_or_unit(comp):
                    continue
                if comp.isdigit():
                    continue
                if len(comp) <= 1:
                    continue
                attr_components_set.add(comp)

        # ---- Values ----
        def _collect_value(val):
            if val is None:
                return
            if isinstance(val, str):
                v = val.strip()
                if not v:
                    return
                if v.lower() in {"true", "false", "none", "null"}:
                    return
                if v.isdigit():
                    return
                if v in STOP_WORDS or v in UNIT_WORDS:
                    return
                if len(v) <= 1:
                    return
                values_set.add(v)
            elif isinstance(val, list):
                for item in val:
                    _collect_value(item)

        def _collect_condition(cond):
            if not isinstance(cond, dict):
                return
            _collect_path(cond.get("attribute"))
            _collect_path(cond.get("aspect"))
            for key in ("value", "old", "new"):
                if key in cond:
                    _collect_value(cond[key])

        # From objects: instance attributes
        for obj in objects:
            attrs = obj.get("attributes", {})
            if isinstance(attrs, dict):
                for path_str, val in attrs.items():
                    _collect_path(path_str)
                    _collect_value(val)

        # From actions: preconditions, changes, conditional changes
        def _collect_action_conditions(act):
            if not isinstance(act, dict):
                return
            for prec in act.get("preconditions", []) or []:
                _collect_condition(prec)
            for key in ("changes", "changes_per_cycle", "changes_total"):
                for ch in act.get(key, []) or []:
                    _collect_condition(ch)
            for cond in act.get("conditional_changes", []) or []:
                if isinstance(cond, dict):
                    for ch in cond.get("changes", []) or []:
                        _collect_condition(ch)
            for sub in act.get("sub_actions", []) or []:
                _collect_action_conditions(sub)

        for act in actions:
            _collect_action_conditions(act)

        # From parsed initial conditions and goal state
        for cond in parsed.get("initial_conditions", []) + parsed.get("goal_state", []):
            _collect_condition(cond)

        return {
            "objects": sorted(objects_set),
            "actions": sorted(actions_set),
            "attribute_components": sorted(attr_components_set),
            "values": sorted(values_set),
        }

    # ------------------------------------------------------------------
    # Alias generation
    # ------------------------------------------------------------------
    async def _generate_aliases(self, base_terms: Dict[str, List[str]]) -> None:
        """
        Call the LLM to generate query-side aliases for the extracted base
        terms. Populates self._last_query_aliases and self._last_query_alias_map.
        Retries up to ALIAS_MAX_ATTEMPTS - 1 times on failure.
        """
        if not any(base_terms.get(k) for k in ("objects", "actions",
                                               "attribute_components", "values")):
            return

        user_msg = json.dumps(base_terms, indent=2, ensure_ascii=False)

        result = None
        for attempt in range(1, ALIAS_MAX_ATTEMPTS + 1):
            result = await self._call_llm_json(
                STAGE_ALIASES_PROMPT, user_msg, f"stage_aliases_attempt{attempt}"
            )
            if result is not None:
                break
            print(f"⚠️ Alias generation attempt {attempt}/{ALIAS_MAX_ATTEMPTS} returned None")

        if result is None:
            print("⚠️ Alias generation failed after all attempts; falling back to offline aliases only.")
            return

        # Validate and normalize the four alias dictionaries
        for key in ("object_aliases", "action_aliases",
                    "attribute_component_aliases", "value_aliases"):
            raw = result.get(key)
            if not isinstance(raw, dict):
                continue
            cleaned: Dict[str, List[str]] = {}
            for term, aliases in raw.items():
                if not isinstance(term, str):
                    continue
                if not isinstance(aliases, list):
                    continue
                new_aliases: List[str] = []
                for alias in aliases:
                    if not isinstance(alias, str):
                        continue
                    a = "_".join(alias.strip().split())
                    if not a:
                        continue
                    if a == term:
                        continue
                    if a in new_aliases:
                        continue
                    new_aliases.append(a)
                    if len(new_aliases) >= ALIAS_CAP_PER_TERM:
                        break
                cleaned[term] = new_aliases
            self._last_query_aliases[key] = cleaned

        # Build alias -> base map
        alias_map: Dict[str, str] = {}
        for base, aliases in self._last_query_aliases["object_aliases"].items():
            for alias in aliases:
                alias_map[f"object_expanded:{alias}"] = f"object:{base}"
                alias_map[f"{PREFIX_OBJECT_EXPANDED_TOKEN}:{alias}"] = f"{PREFIX_OBJECT_TOKEN}:{base}"
        for base, aliases in self._last_query_aliases["action_aliases"].items():
            for alias in aliases:
                alias_map[f"action_expanded:{alias}"] = f"action:{base}"
                alias_map[f"{PREFIX_ACTION_EXPANDED_TOKEN}:{alias}"] = f"{PREFIX_ACTION_TOKEN}:{base}"
        for base, aliases in self._last_query_aliases["attribute_component_aliases"].items():
            for alias in aliases:
                alias_map[f"{PREFIX_ATTRIBUTE_EXPANDED}:{alias}"] = f"{PREFIX_ATTRIBUTE}:{base}"
                alias_map[f"{PREFIX_ATTRIBUTE_EXPANDED_TOKEN}:{alias}"] = f"{PREFIX_ATTRIBUTE_TOKEN}:{base}"
        for base, aliases in self._last_query_aliases["value_aliases"].items():
            for alias in aliases:
                alias_map[f"{PREFIX_VALUE_EXPANDED}:{alias}"] = f"{PREFIX_VALUE}:{base}"
                alias_map[f"{PREFIX_VALUE_EXPANDED_TOKEN}:{alias}"] = f"{PREFIX_VALUE_TOKEN}:{base}"

        self._last_query_alias_map = alias_map

    # ------------------------------------------------------------------
    # Object resolution (simplified – no storage lookup, no STAGE2)
    # ------------------------------------------------------------------
    async def _resolve_objects(self, query_objects: List[Dict]) -> List[Dict]:
        resolved = []
        for q_obj in query_objects:
            if not isinstance(q_obj, dict):
                continue

            name = q_obj.get("name", "")
            template_name = q_obj.get("template_name") or name
            is_person = (
                str(template_name).lower() == "person"
                or q_obj.get("role") in {"actor", "observer", "protagonist"}
            )

            attrs = self._attributes_from_list(q_obj.get("attributes", []))

            resolved.append({
                "name": name,
                "template_name": template_name,
                "obj_id": q_obj.get("obj_id"),
                "categories": q_obj.get("category_hierarchy", []) or q_obj.get("categories", []),
                "functions": q_obj.get("functions", []),
                "materials": q_obj.get("materials", []),
                "shape": q_obj.get("shape", ""),
                "dimensions": q_obj.get("dimensions", {}) or {},
                "position": q_obj.get("position", {}) or {},
                "attributes": attrs,
                "role": q_obj.get("role"),
                "is_person": is_person,
            })
        return resolved

    def _attributes_from_list(self, attrs: List[Dict]) -> Dict:
        result: Dict[str, Any] = {}
        for attr in attrs:
            if not isinstance(attr, dict):
                continue
            path = attr.get("path")
            value = attr.get("value")
            if path:
                result[path] = value
        return result

    # ------------------------------------------------------------------
    # Action resolution (unchanged)
    # ------------------------------------------------------------------
    async def _resolve_actions(self, query_actions: List[Dict]) -> List[Dict]:
        resolved = []
        for q_act in query_actions:
            if not isinstance(q_act, dict):
                continue

            name = q_act.get("name", "")
            is_target = q_act.get("is_target", False)

            if is_target:
                resolved.append({
                    "name": name,
                    "participants": q_act.get("participants", []),
                    "is_target": True,
                    "verb": q_act.get("verb"),
                })
                continue

            existing = self._find_action_template(name)
            if existing is not None:
                resolved.append(self._expand_action_from_storage(existing, q_act))
            else:
                temp = await self._call_llm_json_with_retry(
                    STAGE3_ACTION_TEMPLATE_PROMPT,
                    f"Context action: {name}\nParticipants: {q_act.get('participants', [])}",
                    "stage3_action_template",
                )
                if temp is not None:
                    temp["is_target"] = False
                    resolved.append(temp)
                else:
                    resolved.append({
                        "name": name,
                        "participants": q_act.get("participants", []),
                        "is_target": False,
                        "verb": q_act.get("verb"),
                    })
        return resolved

    def _find_action_template(self, name: str) -> Optional[Dict]:
        if name in self.action_storage_general:
            general = self.action_storage_general[name]
            return {"kind": "general", "name": name, "object": general}
        if name in self.action_storage_alts:
            alt = self.action_storage_alts[name]
            return {
                "kind": "alt",
                "name": name,
                "object": alt,
                "general_name": alt.parent_general.name if alt.parent_general else None,
            }
        return None

    def _expand_action_from_storage(self, existing: Dict, q_act: Dict) -> Dict:
        name = q_act.get("name", "")
        participants = q_act.get("participants", [])
        obj = existing["object"]

        if existing["kind"] == "general":
            general = obj
            verb_name = general.verb.name if general.verb else ""
            alt_names = [alt.name for alt in general.alts]
            sub_actions = getattr(general, "general_sub_actions", []) or []
            if not sub_actions and general.alts:
                sub_actions = general.alts[0].sub_actions or []
            return {
                "name": general.name,
                "verb": verb_name,
                "participants": participants,
                "is_target": False,
                "action_category": getattr(general, "action_category", ""),
                "kinematic_trajectory": getattr(general, "kinematic_trajectory", ""),
                "temporal_type": "",
                "tags": [],
                "preconditions": [],
                "changes": [],
                "sub_actions": sub_actions,
                "alt_names": alt_names,
            }
        else:
            alt = obj
            verb_name = alt.verb.name if alt.verb else ""
            general_name = existing.get("general_name", "")
            return {
                "name": alt.name,
                "general_name": general_name,
                "verb": verb_name,
                "participants": participants,
                "is_target": False,
                "action_category": getattr(alt, "action_category", ""),
                "kinematic_trajectory": getattr(alt, "kinematic_trajectory", ""),
                "temporal_type": "",
                "tags": [],
                "preconditions": [],
                "changes": [],
                "sub_actions": getattr(alt, "sub_actions", []),
                "alt_names": [alt.name],
            }

    # ------------------------------------------------------------------
    # Search term generation – objects
    # ------------------------------------------------------------------
    def _add_object_terms(self, obj: Dict[str, Any], terms: List[Tuple[str, float]]):
        tname = obj.get("template_name")

        # ---- Core object identity + aliases ----
        if tname:
            terms.append((f"object:{tname}", 1.0))
            for token in tokenize_into_words(tname):
                if is_valid_token(token):
                    terms.append((f"{PREFIX_OBJECT_TOKEN}:{token}", 0.5))

            # Offline aliases
            offline_aliases = self.alias_expansion.get("object_aliases", {}).get(tname, [])
            # Query-side aliases
            query_aliases = self._last_query_aliases.get("object_aliases", {}).get(tname, [])
            for alias in set(offline_aliases) | set(query_aliases):
                terms.append((f"object_expanded:{alias}", 1.0))
                for token in tokenize_into_words(alias):
                    if is_valid_token(token):
                        terms.append((f"{PREFIX_OBJECT_EXPANDED_TOKEN}:{token}", 0.5))

        # ---- Categories ----
        for cat in obj.get("categories", []):
            if isinstance(cat, str) and cat:
                terms.append((f"category:{cat}", 1.0))

        # ---- Functions ----
        for func in obj.get("functions", []):
            if isinstance(func, str) and func:
                terms.append((f"function:{func}", 0.8))

        # ---- Materials ----
        for mat in obj.get("materials", []):
            if isinstance(mat, str) and mat:
                terms.append((f"material:{mat}", 1.0))

        # ---- Shape (top-level) ----
        shape = obj.get("shape")
        if shape:
            terms.append((f"{PREFIX_ATTRIBUTE_VALUE}:shape={shape}", 1.0))
            terms.append((f"{PREFIX_ATTRIBUTE}:shape", 0.8))
            terms.append((f"{PREFIX_VALUE}:{shape}", 0.8))
            for token in tokenize_into_words(shape):
                if is_valid_token(token):
                    terms.append((f"{PREFIX_VALUE_TOKEN}:{token}", 0.5))

        # ---- Top-level dimensions ----
        top_dims = obj.get("dimensions", {}) or {}
        for dim in ("length", "width", "height_thickness"):
            d = top_dims.get(dim)
            if isinstance(d, dict):
                val = d.get("value")
                unit = d.get("unit", "")
                if val is not None:
                    val_str = f"{val}{unit}" if unit else str(val)
                    terms.append((f"{PREFIX_ATTRIBUTE_VALUE}:dimensions.{dim}={val_str}", 1.0))

        # ---- Top-level position ----
        pos = obj.get("position", {}) or {}
        if isinstance(pos, dict):
            rel = pos.get("relation")
            rel_r = pos.get("relationr")
            rel_to = pos.get("relative_to")
            if rel:
                terms.append((f"{PREFIX_ATTRIBUTE_VALUE}:position.relation={rel}", 0.8))
            if rel_r:
                terms.append((f"{PREFIX_ATTRIBUTE_VALUE}:position.relationr={rel_r}", 0.8))
            if rel_to:
                terms.append((f"{PREFIX_ATTRIBUTE_VALUE}:position.relative_to={rel_to}", 0.8))
            if rel and rel_to:
                terms.append((f"position={rel}:{rel_to}", 0.5))

        # ---- Volume derivation (top-level first, then attributes) ----
        dims_for_volume: Dict[str, Tuple[Any, str]] = {}
        for dim in ("length", "width", "height_thickness"):
            d = top_dims.get(dim)
            if isinstance(d, dict) and d.get("value") is not None:
                dims_for_volume[dim] = (d.get("value"), d.get("unit", ""))
                continue
            # Fallback: look inside attributes
            attrs = obj.get("attributes", {}) or {}
            for key, val in attrs.items():
                if key == f"dimensions.{dim}" or key == dim or key.endswith(f".{dim}"):
                    m = parse_measurement_string(str(val))
                    if m:
                        dims_for_volume[dim] = m
                    break

        if len(dims_for_volume) == 3:
            try:
                l_norm = normalize_measurement(dims_for_volume["length"][0],
                                               dims_for_volume["length"][1], "length")
                w_norm = normalize_measurement(dims_for_volume["width"][0],
                                               dims_for_volume["width"][1], "length")
                h_norm = normalize_measurement(dims_for_volume["height_thickness"][0],
                                               dims_for_volume["height_thickness"][1], "length")
                if l_norm is not None and w_norm is not None and h_norm is not None:
                    vol = l_norm * w_norm * h_norm
                    terms.append((f"{PREFIX_ATTRIBUTE_VALUE}:volume={vol}cm^3", 1.0))
            except Exception:
                pass

        # ---- Measurement attribute_value terms for weight/mass ----
        attrs = obj.get("attributes", {}) or {}
        for path in ("weight", "mass"):
            if path in attrs:
                val = attrs[path]
                if isinstance(val, dict) and "value" in val:
                    unit = val.get("unit", "")
                    val_str = f"{val['value']}{unit}" if unit else str(val["value"])
                else:
                    val_str = str(val)
                terms.append((f"{PREFIX_ATTRIBUTE_VALUE}:{path}={val_str}", 1.0))

        # ---- Instance attributes ----
        is_protagonist = obj.get("is_person", False) or (
            obj.get("role") in {"actor", "observer", "protagonist"}
        ) or tname == "person"

        for path_str, val in attrs.items():
            if path_str in ("categories", "functions", "materials"):
                continue
            if is_protagonist and not str(path_str).startswith("sensory_perception"):
                continue

            path_parts = str(path_str).split(".")
            fp = {
                "action": None,
                "object": obj.get("obj_id") or tname or None,
                "path": list(path_parts),
                "op": "eq",
                "value": val,
                "source_type": "regular",
            }
            terms.append((f"{PREFIX_FULL_PATH}:{json.dumps(fp)}", 1.0))

            for comp in path_parts:
                self._add_attribute_component_terms(comp, terms)

            if isinstance(val, str) and not val.isdigit():
                terms.append((f"{PREFIX_VALUE}:{val}", 0.8))
                for token in tokenize_into_words(val):
                    if is_valid_token(token):
                        terms.append((f"{PREFIX_VALUE_TOKEN}:{token}", 0.5))

                # Value aliases (offline + query-side)
                offline_val_aliases = self.alias_expansion.get("value_aliases", {}).get(val, [])
                query_val_aliases = self._last_query_aliases.get("value_aliases", {}).get(val, [])
                for alias in set(offline_val_aliases) | set(query_val_aliases):
                    terms.append((f"{PREFIX_VALUE_EXPANDED}:{alias}", 0.8))
                    for token in tokenize_into_words(alias):
                        if is_valid_token(token):
                            terms.append((f"{PREFIX_VALUE_EXPANDED_TOKEN}:{token}", 0.5))

    # ------------------------------------------------------------------
    # Search term generation – attribute component helper
    # ------------------------------------------------------------------
    def _add_attribute_component_terms(
        self,
        comp: str,
        terms: List[Tuple[str, float]],
        is_goal: bool = False,
    ):
        if not comp or not isinstance(comp, str):
            return
        if self._is_banned_component(comp):
            return
        if self._is_stop_or_unit(comp):
            return
        if comp.isdigit():
            return
        if len(comp) <= 1:
            return

        prefix_attr = PREFIX_ATTRIBUTE
        prefix_token = PREFIX_ATTRIBUTE_TOKEN
        prefix_expanded = PREFIX_ATTRIBUTE_EXPANDED
        prefix_expanded_token = PREFIX_ATTRIBUTE_EXPANDED_TOKEN

        if is_goal:
            terms.append((f"goalstate:{prefix_attr}:{comp}", 0.8))
            for token in tokenize_into_words(comp):
                if is_valid_token(token):
                    terms.append((f"goalstate:{prefix_token}:{token}", 0.5))
        else:
            terms.append((f"{prefix_attr}:{comp}", 0.8))
            for token in tokenize_into_words(comp):
                if is_valid_token(token):
                    terms.append((f"{prefix_token}:{token}", 0.5))

        # Aliases (offline + query-side)
        offline_aliases = self.alias_expansion.get("attribute_component_aliases", {}).get(comp, [])
        query_aliases = self._last_query_aliases.get("attribute_component_aliases", {}).get(comp, [])
        for alias in set(offline_aliases) | set(query_aliases):
            if is_goal:
                terms.append((f"goalstate:{prefix_expanded}:{alias}", 0.8))
                for token in tokenize_into_words(alias):
                    if is_valid_token(token):
                        terms.append((f"goalstate:{prefix_expanded_token}:{token}", 0.5))
            else:
                terms.append((f"{prefix_expanded}:{alias}", 0.8))
                for token in tokenize_into_words(alias):
                    if is_valid_token(token):
                        terms.append((f"{prefix_expanded_token}:{token}", 0.5))

    # ------------------------------------------------------------------
    # Search term generation – actions
    # ------------------------------------------------------------------
    def _add_action_terms(self, act: Dict[str, Any], terms: List[Tuple[str, float]]):
        name = act.get("name")
        if name:
            terms.append((f"action:{name}", 2.0))
            for token in tokenize_into_words(name):
                if is_valid_token(token):
                    terms.append((f"{PREFIX_ACTION_TOKEN}:{token}", 0.5))

            offline_aliases = self.alias_expansion.get("action_aliases", {}).get(name, [])
            query_aliases = self._last_query_aliases.get("action_aliases", {}).get(name, [])
            for alias in set(offline_aliases) | set(query_aliases):
                terms.append((f"action_expanded:{alias}", 1.0))
                for token in tokenize_into_words(alias):
                    if is_valid_token(token):
                        terms.append((f"{PREFIX_ACTION_EXPANDED_TOKEN}:{token}", 0.5))

        verb = act.get("verb")
        if verb:
            if act.get("is_target", False):
                terms.append((f"verb_target:{verb}", 1.0))
            else:
                terms.append((f"verb_context:{verb}", 1.0))

        if act.get("action_category"):
            terms.append((f"category:{act['action_category']}", 1.5))
        if act.get("kinematic_trajectory"):
            terms.append((f"trajectory:{act['kinematic_trajectory']}", 1.5))

        participants = act.get("participants") or []
        if participants:
            fp = {
                "action": name,
                "object": None,
                "path": ["participants"],
                "op": "in",
                "value": participants,
                "source_type": "regular",
            }
            terms.append((f"{PREFIX_FULL_PATH}:{json.dumps(fp)}", 1.0))

        if not act.get("is_target", False):
            for prec in act.get("preconditions", []):
                if isinstance(prec, dict):
                    fp = self._condition_to_fullpath(prec)
                    if fp:
                        terms.append((f"{PREFIX_FULL_PATH}:{json.dumps(fp)}", 1.0))
                        self._add_condition_component_terms(prec, terms, is_goal=False)

            for ch in act.get("changes", []):
                if isinstance(ch, dict):
                    for state in ("old", "new"):
                        fp = self._condition_to_fullpath(ch, state=state)
                        if fp:
                            terms.append((f"{PREFIX_FULL_PATH}:{json.dumps(fp)}", 1.0))
                            self._add_condition_component_terms(ch, terms, is_goal=False, state=state)

            for sub in act.get("sub_actions", []) or []:
                if isinstance(sub, str):
                    terms.append((f"action:{sub}", 1.0))
                elif isinstance(sub, dict):
                    if sub.get("name"):
                        terms.append((f"action:{sub['name']}", 1.0))
                    if sub.get("verb"):
                        terms.append((f"verb_context:{sub['verb']}", 1.0))

    # ------------------------------------------------------------------
    # Condition term helpers
    # ------------------------------------------------------------------
    def _add_condition_component_terms(
        self,
        cond: Dict[str, Any],
        terms: List[Tuple[str, float]],
        is_goal: bool = False,
        state: Optional[str] = None,
    ):
        attr = cond.get("attribute")
        aspect = cond.get("aspect")

        path_parts: List[str] = []
        if isinstance(aspect, str):
            path_parts.extend(aspect.split("."))
        elif isinstance(aspect, (list, tuple)):
            path_parts.extend([str(x) for x in aspect])
        if isinstance(attr, str):
            path_parts.extend(attr.split("."))
        elif isinstance(attr, (list, tuple)):
            path_parts.extend([str(x) for x in attr])
        elif attr:
            path_parts.append(str(attr))

        for comp in path_parts:
            self._add_attribute_component_terms(comp, terms, is_goal=is_goal)

        if state == "old":
            val = cond.get("old")
        elif state == "new":
            val = cond.get("new")
        else:
            val = cond.get("value")

        if isinstance(val, str) and not val.isdigit():
            if is_goal:
                terms.append((f"goalstate:{PREFIX_VALUE}:{val}", 0.8))
                for token in tokenize_into_words(val):
                    if is_valid_token(token):
                        terms.append((f"goalstate:{PREFIX_VALUE_TOKEN}:{token}", 0.5))
            else:
                terms.append((f"{PREFIX_VALUE}:{val}", 0.8))
                for token in tokenize_into_words(val):
                    if is_valid_token(token):
                        terms.append((f"{PREFIX_VALUE_TOKEN}:{token}", 0.5))

            offline_aliases = self.alias_expansion.get("value_aliases", {}).get(val, [])
            query_aliases = self._last_query_aliases.get("value_aliases", {}).get(val, [])
            for alias in set(offline_aliases) | set(query_aliases):
                if is_goal:
                    terms.append((f"goalstate:{PREFIX_VALUE_EXPANDED}:{alias}", 0.8))
                else:
                    terms.append((f"{PREFIX_VALUE_EXPANDED}:{alias}", 0.8))

    def _condition_to_fullpath(
        self,
        cond: Dict[str, Any],
        state: Optional[str] = None,
    ) -> Optional[Dict]:
        obj = cond.get("object")
        attr = cond.get("attribute")
        if state == "old":
            val = cond.get("old")
        elif state == "new":
            val = cond.get("new")
        else:
            val = cond.get("value")
        op = cond.get("op", "eq")
        if not obj or not attr or val is None:
            return None
        path_parts = tuple(attr.split(".")) if isinstance(attr, str) else tuple(attr)
        return {
            "action": cond.get("action"),
            "object": str(obj),
            "path": list(path_parts),
            "op": str(op),
            "value": val,
            "source_type": "regular",
            "state_change_type": state,
        }

    def _add_condition_terms(
        self,
        cond: Dict[str, Any],
        terms: List[Tuple[str, float]],
        is_goal: bool = False,
    ):
        fp = self._condition_to_fullpath(cond)
        if fp:
            fp["source_type"] = "goalstate" if is_goal else "regular"
            terms.append((f"{PREFIX_FULL_PATH}:{json.dumps(fp)}", 1.0))

            for comp in fp["path"]:
                self._add_attribute_component_terms(comp, terms, is_goal=is_goal)

            val = fp["value"]
            if isinstance(val, str) and not val.isdigit():
                if is_goal:
                    terms.append((f"goalstate:{PREFIX_VALUE}:{val}", 0.8))
                else:
                    terms.append((f"{PREFIX_VALUE}:{val}", 0.8))
                for token in tokenize_into_words(val):
                    if is_valid_token(token):
                        if is_goal:
                            terms.append((f"goalstate:{PREFIX_VALUE_TOKEN}:{token}", 0.5))
                        else:
                            terms.append((f"{PREFIX_VALUE_TOKEN}:{token}", 0.5))

                offline_aliases = self.alias_expansion.get("value_aliases", {}).get(val, [])
                query_aliases = self._last_query_aliases.get("value_aliases", {}).get(val, [])
                for alias in set(offline_aliases) | set(query_aliases):
                    if is_goal:
                        terms.append((f"goalstate:{PREFIX_VALUE_EXPANDED}:{alias}", 0.8))
                    else:
                        terms.append((f"{PREFIX_VALUE_EXPANDED}:{alias}", 0.8))

        spatial_term = self._build_spatial_term(cond)
        if spatial_term:
            terms.append((spatial_term, 1.0))

    def _build_spatial_term(self, cond: Dict[str, Any]) -> Optional[str]:
        relation = cond.get("relation") or cond.get("relationr")
        obj = cond.get("object")
        rel_to = cond.get("relative_to")
        if relation and obj and rel_to:
            return f"spatial:{relation}({obj},{rel_to})"
        return None

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------
    async def _call_llm_json(
        self,
        system_msg: str,
        user_msg: str,
        purpose: str,
    ) -> Optional[Dict]:
        try:
            resp = await self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=40000,
                extra_body={"reasoning_effort": REASONING_EFFORT},
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content)
        except Exception as e:
            print(f"LLM call failed for {purpose}: {e}")
            return None

    async def _call_llm_json_with_retry(
        self,
        system_msg: str,
        user_msg: str,
        purpose: str,
    ) -> Optional[Dict]:
        for attempt in range(1, TEMPLATE_GEN_MAX_RETRIES + 1):
            result = await self._call_llm_json(system_msg, user_msg, purpose)
            if result is not None:
                return result
            print(f"⚠️ {purpose} attempt {attempt} returned None; retrying...")
        print(f"❌ {purpose} failed after {TEMPLATE_GEN_MAX_RETRIES} attempts")
        return None

    # ------------------------------------------------------------------
    # Fallback and phrase decomposition
    # ------------------------------------------------------------------
    def _fallback_terms(self, query: str) -> List[Tuple[str, float]]:
        words = [w.lower() for w in query.split() if len(w) > 2]
        return [(f"other_token:{w}", 1.0) for w in words]

    def _add_phrase_decomposition(
        self,
        terms: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        new_terms = list(terms)

        for term, weight in terms:
            if term.startswith("verb"):
                continue
            if any(ch in term for ch in ("=", ":", "[", "]", "{", "}", "'", '"')):
                continue
            if "_" in term or " " in term:
                words = re.split(r"[_\s]+", term)
                for w in words:
                    w = w.strip()
                    if w and w not in STOP_WORDS and w not in UNIT_WORDS and len(w) > 1:
                        new_terms.append((f"other_token:{w}", weight * 0.5))

        return new_terms