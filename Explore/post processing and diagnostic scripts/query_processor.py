#!/usr/bin/env python3
"""
query_processor.py – Expanded query processor for memory search.

Now includes:
- Public generate_search_terms()
- Perception/observer detection in Stage 1 prompt
- Filtered search-term extraction
- Measurement value+unit combining
- No noisy standalone component terms
- Protagonist defaults not merged unless query explicitly specifies them
- Overly general category terms filtered out
- Lower weights for dimensions, mass, position, etc.
- Category weights increased
- Target action included
- Material list handling
- Position relation terms suppressed
- Goal-state terms weighted at 10.0
- Only object: term generated for object templates, not duplicate template:
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from openai import AsyncOpenAI

# =============================================================================
# Configuration
# =============================================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT_FILE = DATA_DIR / "query_processor_prompt.txt"
LLM_RESPONSES_FILE = DATA_DIR / "llm_responses.jsonl"

OBJECT_STORAGE_PATH = DATA_DIR / "object_storage.json"
ACTION_STORAGE_PATH = DATA_DIR / "action_storage.json"

# =============================================================================
# Unit words
# =============================================================================
UNIT_WORDS = {
    "cm", "m", "mm", "km", "g", "kg", "mg", "ml", "l", "s",
    "sec", "min", "hr", "hour", "hours", "degree", "degrees",
    "percent", "%", "gram", "grams", "meter", "meters",
    "centimeter", "centimeters", "millimeter", "millimeters",
    "liter", "liters", "milliliter", "milliliters", "kilogram",
    "kilograms"
}

# =============================================================================
# Categories that are too generic to use as search terms
# =============================================================================
TOO_GENERAL_CATEGORIES = {
    "object", "physical_object", "person", "agent", "entity"
}

# =============================================================================
# Attribute weights for generic/measurement fields
# =============================================================================
ATTRIBUTE_WEIGHTS = {
    "dimensions": 0.5,
    "mass": 0.5,
    "weight": 0.5,
    "shape": 1.5,
    "position": 1.0,
    "position_relative_body": 1.0,
    "material": 2.0,
    "materials": 2.0,
    "color": 2.0,
    "state": 2.0,
}

# =============================================================================
# Prompts
# =============================================================================

STAGE1_PROMPT = """
You are a query analyzer for a memory system.

Convert the user's question into a structured JSON representation of the query.
The JSON will be used only to search the memory system. Do NOT answer the user's question directly.

Important principles:

1. Distinguish between:
   - target action: the action the user is asking about; the answer should come from the memory system.
   - context actions: actions already stated or presupposed in the query.
   - initial conditions: current or presumed states.
   - goal state: the desired result.

2. Expand context actions as much as possible.
   - For every non-target action, provide hierarchical sub-actions, preconditions, changes, tags, trajectory, and category.
   - Use the same style as the memory encoder's annotated action plot and final JSON encoding.
   - This gives the memory system more context.

3. Do NOT expand the target action.
   - For the target action, provide only:
       - name
       - participants
       - is_target: true
   - Do NOT provide sub_actions, preconditions, changes, changes_total, or tags for the target action.
   - Otherwise, the query processor would be solving the problem instead of searching memory.

4. If the query implies observation, inspection, or visual checking:
   - Include a "person" object as "observer".
   - Include a perception action such as "look_at", "inspect_object", etc.
   - Include sensory perception attributes on the person, with "source" referring to the relevant object.
   - Example:
     {
       "name": "person",
       "template_name": "person",
       "role": "observer",
       "attributes": [
         {
           "path": "sensory_perception_sight",
           "value": [
             {"sensation": "yellow", "source": "leaves"}
           ]
         }
       ]
     }

5. For query objects:
   - Use canonical object template names and categories when possible.
   - Prefer semantic template names over raw physical names.
   - Example: instead of "can", use "canned_fruit" or "canned_food".
   - Include composition/content where appropriate.
   - Include only attributes explicitly stated or strongly implied by the query.
   - Do not invent a full object encoding; the system can later fill defaults from ObjectStorage.
   - For "person" or protagonist objects, do NOT include generic physical attributes like dimensions, mass, shape, posture, fatigue, focus, etc., unless the query specifically mentions them. Only include sensory perceptions or explicitly mentioned attributes.

6. For actions:
   - Always include the target action, even if it is barebones.
   - The target action must have "is_target": true and no sub_actions, preconditions, or changes.
   - Context actions can be expanded.

7. Return JSON with this structure:

{
  "query_objects": [
    {
      "name": "fruit",
      "template_name": "canned_fruit",
      "category_hierarchy": ["canned_foods", "preserved_food", "food", "physical_object", "object"],
      "attributes": [
        {"path": "color", "value": "yellow"}
      ],
      "composition": {
        "content": "fruit",
        "container": "can"
      },
      "role": "target_object"
    }
  ],
  "query_actions": [
    {
      "name": "look_at_leaves",
      "participants": ["person", "leaves"],
      "is_target": false,
      "action_category": "PERCEPTION",
      "kinematic_trajectory": "angular",
      "temporal_type": "sequential",
      "tags": ["looking", "inspection"],
      "preconditions": [],
      "changes": [
        {
          "object": "person",
          "attribute": "sensory_perception_sight",
          "old": [],
          "new": [
            {"sensation": "yellow", "source": "leaves"}
          ]
        }
      ],
      "sub_actions": [
        {
          "name": "saccade_scene",
          "participants": ["person"],
          "duration": 0.3,
          "action_category": "PERCEPTION",
          "kinematic_trajectory": "angular",
          "temporal_type": "sequential",
          "preconditions": [],
          "changes": [],
          "sub_actions": []
        },
        {
          "name": "fixate_on_surface",
          "participants": ["person", "leaves"],
          "duration": 0.7,
          "action_category": "PERCEPTION",
          "kinematic_trajectory": "straight",
          "temporal_type": "sequential",
          "preconditions": [],
          "changes": [],
          "sub_actions": []
        }
      ]
    },
    {
      "name": "restore_leaf_color",
      "participants": ["leaves", "plant"],
      "is_target": true
    }
  ],
  "initial_conditions": [
    {
      "object": "leaves",
      "attribute": "color",
      "value": "yellow"
    }
  ],
  "goal_state": [
    {
      "object": "leaves",
      "attribute": "color",
      "value": "green"
    }
  ]
}

For non-target actions, you may include as many sub-action levels as useful.
For target action, do not include sub_actions, preconditions, changes, or other expansion fields.

Use the demonstration below as a guide.

DEMONSTRATION:

Example query:
I opened a can of fruit. How do I get the fruit out?

Expected output:
{
  "query_objects": [
    {
      "name": "fruit",
      "template_name": "canned_fruit",
      "category_hierarchy": ["canned_foods", "preserved_food", "food", "physical_object", "object"],
      "attributes": [
        {"path": "color", "value": "yellow"}
      ],
      "composition": {
        "content": "fruit",
        "container": "can"
      },
      "role": "target_object"
    }
  ],
  "query_actions": [
    {
      "name": "open_can",
      "participants": ["canned_fruit"],
      "is_target": false,
      "action_category": "TOOL_USAGE",
      "kinematic_trajectory": "circular",
      "temporal_type": "sequential",
      "tags": ["can", "opening", "lid removal"],
      "preconditions": [
        {
          "object": "canned_fruit",
          "attribute": "lid.state",
          "value": "closed"
        }
      ],
      "changes_total": [
        {
          "object": "canned_fruit",
          "attribute": "lid.state",
          "old": "closed",
          "new": "open"
        }
      ],
      "sub_actions": [
        {
          "name": "position_can_opener",
          "participants": ["canned_fruit", "can_opener"],
          "duration": 2,
          "action_category": "MANIPULATION",
          "kinematic_trajectory": "straight",
          "temporal_type": "sequential",
          "preconditions": [],
          "changes": [],
          "sub_actions": []
        }
      ]
    },
    {
      "name": "retrieve_item_from_can",
      "participants": ["canned_fruit", "fruit"],
      "is_target": true
    }
  ],
  "initial_conditions": [
    {
      "object": "fruit",
      "attribute": "position",
      "value": "inside_can"
    }
  ],
  "goal_state": [
    {
      "object": "fruit",
      "attribute": "position",
      "value": "outside_can"
    }
  ]
}
"""

STAGE2_OBJECT_TEMPLATE_PROMPT = """
You are an object template generator for a memory search system.

The query parser encountered an object that is not in the object storage.
Create a temporary object template for this object. This template is used only to improve search terms.
Do not answer any how-to question.

Return JSON with this structure:

{
  "template_name": "canned_fruit",
  "defaults": {
    "dimensions": {
      "length": {"value": 7.5, "unit": "cm"},
      "width": {"value": 7.5, "unit": "cm"},
      "height_thickness": {"value": 11.0, "unit": "cm"}
    },
    "mass": {"value": 400, "unit": "g"},
    "shape": "cylindrical",
    "position": {"relation": "on", "relative_to": "counter"},
    "material": ["aluminum"],
    "lid": {"state": "closed"},
    "contents": "canned_fruit"
  },
  "functions": ["food container", "preserved food"],
  "aliases": ["tin", "can"],
  "categories": ["canned_foods", "preserved_food", "food", "physical_object", "object"],
  "composition_templates": ["can_lid", "can_body"]
}

Important:
- `template_name` should be a semantic name, not a raw physical noun when possible.
- `defaults` must include dimensions, mass, shape, position, and any attributes explicitly mentioned in the query.
- `material` must be a LIST of materials, not a composite string.
- Include composition_templates if the object is composite.
- Do not include any commentary.

Use the demonstration below as a guide.

DEMONSTRATION:

Object name: can
Query-specified attributes: color=red

Expected temporary object template:
{
  "template_name": "canned_food",
  "defaults": {
    "dimensions": {
      "length": {"value": 7.5, "unit": "cm"},
      "width": {"value": 7.5, "unit": "cm"},
      "height_thickness": {"value": 11.0, "unit": "cm"}
    },
    "mass": {"value": 400, "unit": "g"},
    "shape": "cylindrical",
    "position": {"relation": "on", "relative_to": "counter"},
    "material": ["aluminum"],
    "color": "red",
    "lid": {"state": "closed"},
    "contents": "canned_food"
  },
  "functions": ["food container", "preserved food"],
  "aliases": ["tin", "can"],
  "categories": ["canned_food", "preserved_food", "food", "physical_object", "object"],
  "composition_templates": ["can_lid", "can_body"]
}
"""

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
  "tags": ["can", "opening", "lid removal"],
  "preconditions": [
    {
      "object": "canned_fruit",
      "attribute": "lid.state",
      "value": "closed"
    }
  ],
  "changes_total": [
    {
      "object": "canned_fruit",
      "attribute": "lid.state",
      "old": "closed",
      "new": "open"
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
      "preconditions": [],
      "changes": [],
      "sub_actions": []
    }
  ]
}

Important:
- Use the same style as the memory encoder's annotated action plot and final JSON encoding.
- Include preconditions, changes, changes_per_cycle, changes_total, conditional_changes, and sub_actions as appropriate.
- Use `aspect` only when the attribute is part of a specific aspect.
- For sub-actions, use the same structure recursively until atomic actions are reached.
- Do not include any commentary.

Use the demonstration below as a guide.

DEMONSTRATION:

Context action: open_can
Participants: canned_fruit, can_opener

Expected temporary action decomposition:
{
  "name": "open_can",
  "duration": 10,
  "action_category": "TOOL_USAGE",
  "kinematic_trajectory": "circular",
  "temporal_type": "sequential",
  "participants": ["canned_fruit", "can_opener"],
  "tags": ["can", "opening", "lid removal"],
  "preconditions": [
    {
      "object": "canned_fruit",
      "attribute": "lid.state",
      "value": "closed"
    }
  ],
  "changes_total": [
    {
      "object": "canned_fruit",
      "attribute": "lid.state",
      "old": "closed",
      "new": "open"
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
      "preconditions": [],
      "changes": [],
      "sub_actions": []
    }
  ]
}
"""


class QueryProcessor:
    def __init__(self, api_key: str = None):
        self.client = AsyncOpenAI(api_key=api_key or DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.object_storage = self._load_object_storage()
        self.action_storage = self._load_action_storage()

        with open(SYSTEM_PROMPT_FILE, "w", encoding="utf-8") as f:
            f.write(STAGE1_PROMPT)

    # ------------------------------------------------------------------
    # Storage loading
    # ------------------------------------------------------------------
    def _load_object_storage(self) -> Dict[str, Any]:
        if not OBJECT_STORAGE_PATH.exists():
            print(f"⚠️ Object storage not found at {OBJECT_STORAGE_PATH}")
            return {}
        with open(OBJECT_STORAGE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_action_storage(self) -> Dict[str, Any]:
        if not ACTION_STORAGE_PATH.exists():
            print(f"⚠️ Action storage not found at {ACTION_STORAGE_PATH}")
            return {}
        with open(ACTION_STORAGE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------
    def _append_llm_response(self, query: str, response_text: str, error: Optional[str] = None):
        record = {
            "query": query,
            "response": response_text,
            "error": error,
            "timestamp": str(__import__("datetime").datetime.now()),
        }
        with open(LLM_RESPONSES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _call_llm_json(self, system_msg: str, user_msg: str, purpose: str) -> Optional[Dict]:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        try:
            response = await self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=4000,
                extra_body={"reasoning_effort": "high"},
            )
            content = response.choices[0].message.content.strip()
            self._append_llm_response(purpose, content)
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content)
        except Exception as e:
            self._append_llm_response(purpose, "", error=str(e))
            print(f"⚠️ LLM call failed for {purpose}: {e}")
            return None

    # ------------------------------------------------------------------
    # Main public methods
    # ------------------------------------------------------------------
    async def process(self, query: str) -> List[Dict[str, float]]:
        full_result = await self.process_full(query)
        if full_result is None:
            return self._fallback_terms(query)
        return self.generate_search_terms(full_result)

    async def process_full(self, query: str) -> Optional[Dict[str, Any]]:
        parsed = await self._call_llm_json(STAGE1_PROMPT, query, "stage1_parse")
        if parsed is None:
            return None

        query_objects = parsed.get("query_objects") or []
        resolved_objects = await self._resolve_objects(query_objects)

        query_actions = parsed.get("query_actions") or []
        resolved_actions = await self._resolve_actions(query_actions)

        return {
            "query_objects": resolved_objects,
            "query_actions": resolved_actions,
            "initial_conditions": parsed.get("initial_conditions") or [],
            "goal_state": parsed.get("goal_state") or [],
        }

    # ------------------------------------------------------------------
    # Object resolution
    # ------------------------------------------------------------------
    async def _resolve_objects(self, query_objects: List[Dict]) -> List[Dict]:
        resolved = []
        for q_obj in query_objects:
            name = q_obj.get("name", "").lower()
            template_name = q_obj.get("template_name") or q_obj.get("name", "")

            is_person = (
                template_name.lower() == "person"
                or q_obj.get("role") in {"actor", "observer", "protagonist"}
            )

            existing = self._find_object_template(template_name)
            if existing is not None:
                if is_person:
                    attributes = self._attributes_from_list(q_obj.get("attributes", []))
                else:
                    attributes = self._merge_object_attributes(
                        existing.get("defaults", {}), q_obj.get("attributes", [])
                    )

                resolved_obj = {
                    "template_name": existing["template_name"],
                    "categories": q_obj.get("category_hierarchy")
                    or self._infer_categories(existing["template_name"]),
                    "attributes": attributes,
                    "role": q_obj.get("role"),
                }
            else:
                temp = await self._call_llm_json(
                    STAGE2_OBJECT_TEMPLATE_PROMPT,
                    f"Object name: {name}\nQuery-specified attributes: {q_obj.get('attributes', [])}",
                    "stage2_object_template",
                )
                if temp is None:
                    resolved_obj = {
                        "template_name": template_name,
                        "categories": q_obj.get("category_hierarchy") or ["object"],
                        "attributes": self._attributes_from_list(q_obj.get("attributes", [])),
                        "role": q_obj.get("role"),
                    }
                else:
                    if is_person:
                        attributes = self._attributes_from_list(q_obj.get("attributes", []))
                    else:
                        attributes = self._merge_object_attributes(
                            temp.get("defaults", {}), q_obj.get("attributes", [])
                        )
                    resolved_obj = {
                        "template_name": temp.get("template_name", template_name),
                        "categories": q_obj.get("category_hierarchy")
                        or self._infer_categories(temp.get("template_name", template_name)),
                        "attributes": attributes,
                        "role": q_obj.get("role"),
                    }

            resolved.append(resolved_obj)
        return resolved

    def _find_object_template(self, name: str) -> Optional[Dict]:
        if not self.object_storage:
            return None
        if name in self.object_storage:
            return {"template_name": name, "defaults": self.object_storage[name]}
        for key, val in self.object_storage.items():
            if key.lower() == name.lower():
                return {"template_name": key, "defaults": val}
        return None

    def _merge_object_attributes(self, defaults: Dict, query_attrs: List[Dict]) -> Dict:
        merged = defaults.copy() if defaults else {}
        for attr in query_attrs:
            path = attr.get("path", "")
            value = attr.get("value")
            if path:
                merged[path] = value
        return merged

    def _attributes_from_list(self, attrs: List[Dict]) -> Dict:
        result = {}
        for attr in attrs:
            path = attr.get("path")
            value = attr.get("value")
            if path:
                result[path] = value
        return result

    def _infer_categories(self, template_name: str) -> List[str]:
        return [template_name, "physical_object", "object"]

    # ------------------------------------------------------------------
    # Action resolution
    # ------------------------------------------------------------------
    async def _resolve_actions(self, query_actions: List[Dict]) -> List[Dict]:
        resolved = []
        for q_act in query_actions:
            name = q_act.get("name", "")
            is_target = q_act.get("is_target", False)

            if is_target:
                resolved.append(
                    {
                        "name": name,
                        "participants": q_act.get("participants", []),
                        "is_target": True,
                    }
                )
                continue

            existing = self._find_action_template(name)
            if existing is not None:
                action_tree = self._expand_action_from_storage(existing, q_act)
                resolved.append(action_tree)
            else:
                temp = await self._call_llm_json(
                    STAGE3_ACTION_TEMPLATE_PROMPT,
                    f"Context action: {name}\nParticipants: {q_act.get('participants', [])}",
                    "stage3_action_template",
                )
                if temp is not None:
                    temp["is_target"] = False
                    resolved.append(temp)
                else:
                    resolved.append(
                        {
                            "name": name,
                            "participants": q_act.get("participants", []),
                            "is_target": False,
                        }
                    )

        return resolved

    def _find_action_template(self, name: str) -> Optional[Dict]:
        templates = self.action_storage.get("templates", {})
        if not templates:
            return None
        if name in templates:
            return templates[name]
        for key, val in templates.items():
            if key.lower() == name.lower():
                return val
        return None

    def _expand_action_from_storage(self, template: Dict, q_act: Dict) -> Dict:
        name = q_act.get("name")
        participants = q_act.get("participants", [])
        sub_action_templates = template.get("sub_action_templates") or []

        sub_actions = []
        for sub_name in sub_action_templates:
            sub_template = self.action_storage.get("templates", {}).get(sub_name)
            if sub_template:
                sub_actions.append(
                    self._expand_action_from_storage(sub_template, {"name": sub_name})
                )
            else:
                sub_actions.append(
                    {"name": sub_name, "participants": [], "sub_actions": []}
                )

        return {
            "name": name,
            "participants": participants,
            "is_target": False,
            "action_category": template.get("action_category", "HUMAN_INTERACTION"),
            "kinematic_trajectory": template.get("kinematic_trajectory", "straight"),
            "temporal_type": template.get("temporal_type", "sequential"),
            "tags": template.get("tags", []),
            "preconditions": template.get("preconditions", []),
            "changes": template.get("changes", []),
            "changes_per_cycle": template.get("changes_per_cycle", []),
            "changes_total": template.get("changes_total", []),
            "conditional_changes": template.get("conditional_changes", []),
            "sub_actions": sub_actions,
        }

    # ------------------------------------------------------------------
    # Search term generation
    # ------------------------------------------------------------------
    def generate_search_terms(self, structured_query: Dict) -> List[Dict[str, float]]:
        terms: Dict[str, float] = defaultdict(float)

        for obj in structured_query.get("query_objects", []):
            template_name = obj.get("template_name")
            if template_name:
                # Use only object: term to avoid duplicate template/object terms.
                terms[f"object:{template_name}"] += 2.0

            for cat in obj.get("categories", []):
                if str(cat).lower() in TOO_GENERAL_CATEGORIES:
                    continue
                terms[f"category:{cat}"] += 1.0

            attrs = obj.get("attributes", {})
            if isinstance(attrs, dict):
                # Special handling for position to avoid position.relation alone.
                position = attrs.get("position")
                if isinstance(position, dict):
                    relation = position.get("relation")
                    relative_to = position.get("relative_to")
                    if relation and relative_to:
                        terms[f"position={relation}:{relative_to}"] += 0.5
                    if relative_to:
                        terms[f"position.relative_to={relative_to}"] += 0.25

            for path, value in self._flatten_attribute_items(attrs):
                if path == "position.relation" or path.startswith("position.relation."):
                    continue
                weight = self._attribute_weight(path)
                self._add_attribute_term(terms, path, value, weight)

        for act in structured_query.get("query_actions", []):
            name = act.get("name")
            is_target = act.get("is_target", False)

            if name:
                weight = 3.0 if is_target else 2.0
                terms[f"action:{name}"] += weight

            if is_target:
                continue

            category = act.get("action_category")
            if category:
                terms[f"category:{category}"] += 1.5
            trajectory = act.get("kinematic_trajectory")
            if trajectory:
                terms[f"trajectory:{trajectory}"] += 1.5
            for tag in act.get("tags", []):
                terms[f"{tag}"] += 1.5
                terms[f"tag:{tag}"] += 1.0

            for prec in act.get("preconditions", []):
                self._add_condition_terms(terms, prec, prefix="precondition")
            for ch in act.get("changes", []):
                self._add_change_terms(terms, ch, prefix="state_change")
            for ch in act.get("changes_per_cycle", []):
                self._add_change_terms(terms, ch, prefix="state_change")
            for ch in act.get("changes_total", []):
                self._add_change_terms(terms, ch, prefix="state_change")

            for sub in act.get("sub_actions", []):
                self._add_action_terms_recursive(terms, sub, scale=0.8)

        for cond in structured_query.get("initial_conditions", []):
            self._add_condition_terms(terms, cond, prefix="precondition")
        for goal in structured_query.get("goal_state", []):
            self._add_condition_terms(terms, goal, prefix="goal_state")

        result = [{"term": term, "weight": weight} for term, weight in terms.items()]
        result.sort(key=lambda x: x["weight"], reverse=True)
        return result

    def _add_action_terms_recursive(self, terms, action, scale=1.0):
        name = action.get("name")
        if name:
            terms[f"action:{name}"] += 1.5 * scale
        category = action.get("action_category")
        if category:
            terms[f"category:{category}"] += 1.5 * scale
        trajectory = action.get("kinematic_trajectory")
        if trajectory:
            terms[f"trajectory:{trajectory}"] += 1.5 * scale
        for tag in action.get("tags", []):
            terms[f"tag:{tag}"] += 1.0 * scale
        for prec in action.get("preconditions", []):
            self._add_condition_terms(terms, prec, prefix="precondition", scale=scale)
        for ch in action.get("changes", []):
            self._add_change_terms(terms, ch, prefix="state_change", scale=scale)
        for ch in action.get("changes_per_cycle", []):
            self._add_change_terms(terms, ch, prefix="state_change", scale=scale)
        for ch in action.get("changes_total", []):
            self._add_change_terms(terms, ch, prefix="state_change", scale=scale)
        for sub in action.get("sub_actions", []):
            self._add_action_terms_recursive(terms, sub, scale * 0.8)

    def _add_condition_terms(self, terms, cond, prefix, scale=1.0):
        obj = cond.get("object", "?")
        attr = cond.get("attribute", "?")
        val = cond.get("value", "?")
        full = f"{obj}.{attr}"

        if prefix == "goal_state":
            # Goal-state matches should be very high weight.
            terms[f"goal_state:{full}={val}"] += 10.0 * scale
            terms[f"goal_state:{attr}={val}"] += 10.0 * scale
        else:
            terms[f"{prefix}:{full}={val}"] += 1.5 * scale
            terms[f"{prefix}:{attr}={val}"] += 1.2 * scale

    def _add_change_terms(self, terms, ch, prefix, scale=1.0):
        obj = ch.get("object", "?")
        aspect = ch.get("aspect", "")
        attr = ch.get("attribute", "?")
        old = ch.get("old", "?")
        new = ch.get("new", "?")
        full = f"{obj}.{aspect}.{attr}" if aspect else f"{obj}.{attr}"
        terms[f"{prefix}:{full}:{old}->{new}"] += 1.5 * scale
        terms[f"{prefix}:{attr}:{new}"] += 1.2 * scale

    def _add_attribute_term(self, terms, path, value, weight):
        if path == "position.relation" or path.startswith("position.relation."):
            return
        value_str = self._value_to_string(value)
        if value_str == "":
            return
        terms[f"{path}={value_str}"] += weight

        if isinstance(value, list):
            for item in value:
                item_str = self._value_to_string(item)
                if item_str:
                    terms[f"{path}={item_str}"] += weight * 0.8
        elif isinstance(value, str) and "," in value:
            for item in value.split(","):
                item = item.strip()
                if item:
                    terms[f"{path}={item}"] += weight * 0.8

    def _attribute_weight(self, path):
        top = path.split(".")[0] if path else ""
        return ATTRIBUTE_WEIGHTS.get(top, 2.0)

    def _flatten_attribute_items(self, attrs: Dict) -> List[Tuple[str, Any]]:
        items = []
        if not isinstance(attrs, dict):
            return items

        def walk(obj, path_parts):
            if isinstance(obj, dict):
                if set(obj.keys()) == {"value", "unit"}:
                    val = self._combine_value_unit(obj.get("value"), obj.get("unit"))
                    items.append((".".join(path_parts), val))
                    return
                for k, v in obj.items():
                    walk(v, path_parts + [str(k)])
            elif isinstance(obj, list):
                if all(not isinstance(x, (dict, list)) for x in obj):
                    items.append((".".join(path_parts), obj))
                else:
                    for i, v in enumerate(obj):
                        walk(v, path_parts + [str(i)])
            else:
                items.append((".".join(path_parts), obj))

        walk(attrs, [])
        return items

    def _combine_value_unit(self, value, unit):
        if value is None:
            return ""
        value_str = self._value_to_string(value)
        if unit is not None and str(unit).strip():
            unit_str = str(unit).strip()
            if unit_str.lower() in UNIT_WORDS:
                return f"{value_str}{unit_str}"
        return value_str

    def _value_to_string(self, value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return ",".join(self._value_to_string(v) for v in value)
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        return str(value)

    def _fallback_terms(self, query):
        words = [w.lower() for w in query.split() if len(w) > 2]
        return [{"term": w, "weight": 0.5} for w in words]