#!/usr/bin/env python3
r"""
populate_missing_verbs.py

Populates missing verb fields in action_storage_linked.json using DeepSeek.

Steps:
1. Find general templates with missing/empty "verb".
2. Ask DeepSeek to generate a basic verb-level generalization for each.
3. Update the general template's "verb".
4. For each alt with missing/empty "verb", copy the general verb.
5. Rebuild verb_storage.json from the updated action storage.
6. Save both action_storage_linked.json and verb_storage.json.

No memory files are modified.
"""

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from openai import AsyncOpenAI

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPLORE_DIR / "data"

ACTION_STORAGE_LINKED_PATH = DATA_DIR / "action_storage_linked.json"
VERB_STORAGE_PATH = DATA_DIR / "verb_storage.json"

# =============================================================================
# LLM config
# =============================================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"

MAX_TOKENS = 50000
BATCH_SIZE = 50
MAX_RETRIES = 1


# =============================================================================
# Example mappings for the LLM prompt
# =============================================================================
VERB_EXAMPLES = {
    "inspect_object": "inspect",
    "grasp_object_hand": "grasp",
    "draw_knife_along_surface": "pull",
    "cut_along_seam": "cut",
    "rotate_apple_by_hand": "rotate",
    "apply_gentle_pressure": "apply",
    "position_object": "position",
    "insert_blade": "insert",
    "lift_surface_object_pry": "lift",
    "place_object": "place",
    "release_grip": "release",
    "peel_object": "peel",
    "detach_object": "detach",
    "monitor_visually": "monitor",
    "saccade_scene": "saccade",
    "fixate_on_surface": "fixate",
    "reach_toward_object": "reach",
    "open_palm_and_spread_fingers": "open",
    "curl_fingers_around": "curl",
    "tilt_tool_upward": "tilt",
    "rotate_wrist": "rotate",
    "flex_forearm": "flex",
    "press_tool_into_surface": "press",
    "liquid_flow": "flow",
    "flesh_oxidation": "oxidize",
    "gravity_pull": "pull",
}


# =============================================================================
# Load / save helpers
# =============================================================================
def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_json(data: Dict[str, Any], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved {path}")


# =============================================================================
# Find missing verbs
# =============================================================================
def find_general_templates_missing_verbs(action_storage: Dict[str, Any]) -> List[str]:
    """Return sorted list of general template names with missing/empty verb."""
    missing = []

    for general_name, entry in action_storage.items():
        if not isinstance(entry, dict):
            continue

        verb = entry.get("verb", "")
        if not verb:
            missing.append(general_name)

    return sorted(missing)


# =============================================================================
# LLM calls
# =============================================================================
def build_system_prompt() -> str:
    examples = json.dumps(VERB_EXAMPLES, indent=2, ensure_ascii=False)

    return f"""
You are an action template verb generalizer.

Given an action template name, produce its most basic verb-level generalization. It should not be just identifying any verbs in the action template name. The verb produced must capture the essence of the action, taking into account what takes place during the action. Try to avoid ambiguity.
For example, for the action template draw_knife_along_surface, the action is about pulling a knife along a surface. The verb could conceivably be "draw", but that is fairly ambiguous with draw as in sketching a drawing. The verb that describes the essence of the action and that is general and unambiguous is pull, which should be the verb produced for this action template. 

Rules:
- Return a single verb or very short verb phrase.
- Remove object/tool/surface nouns.
- Prefer simple verbs: inspect, grasp, draw, cut, rotate, apply, position, insert, lift, place, release, peel, detach, monitor, flow, oxidize, pull, press, etc.
- Do not include underscores unless the verb itself is multiword and clear. Use underscores very sparingly. 
- Do not return "action", "do", or "perform".

Return only a JSON object mapping action_template to verb.

Example mappings:
{examples}
"""


async def call_llm_for_batch(
    client: AsyncOpenAI,
    templates: List[str],
    batch_index: int,
    total_batches: int,
) -> Dict[str, str]:
    """Call LLM for one batch of action template names."""
    system_msg = build_system_prompt()
    user_msg = (
        f"Batch {batch_index}/{total_batches}\n"
        "Assign a verb-level generalization to each action template.\n"
        "Return JSON: {\"action_template\": \"verb\"}\n\n"
        + json.dumps(templates, indent=2, ensure_ascii=False)
    )

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=MAX_TOKENS,
                extra_body={"reasoning_effort": "high"},
            )

            content = response.choices[0].message.content

            if not content:
                print(f"   ⚠️ Batch {batch_index}: empty content, attempt {attempt + 1}")
                continue

            content = content.strip()

            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            data = json.loads(content)

            if not isinstance(data, dict):
                print(f"   ⚠️ Batch {batch_index}: response is not a JSON object")
                continue

            valid_mapping = {}
            for template, verb in data.items():
                template_str = str(template).strip()
                verb_str = str(verb).strip()
                if template_str and verb_str:
                    valid_mapping[template_str] = verb_str

            if valid_mapping:
                return valid_mapping
            else:
                print(f"   ⚠️ Batch {batch_index}: no valid mappings")

        except Exception as e:
            print(f"   ❌ Batch {batch_index} failed: {e}")

    return {}


async def generate_verbs_for_templates(
    client: AsyncOpenAI,
    templates: List[str],
) -> Dict[str, str]:
    """Generate verbs for a list of action template names, using batches."""
    if not templates:
        return {}

    templates = sorted(templates)
    batches = [
        templates[i : i + BATCH_SIZE]
        for i in range(0, len(templates), BATCH_SIZE)
    ]

    total_batches = len(batches)
    print(f"🤖 Generating verbs for {len(templates)} templates in {total_batches} batches...")

    result = {}

    for idx, batch in enumerate(batches, 1):
        mapping = await call_llm_for_batch(client, batch, idx, total_batches)
        result.update(mapping)
        print(f"   Batch {idx}/{total_batches}: {len(mapping)} verbs")

    return result


# =============================================================================
# Update action storage
# =============================================================================
def update_action_storage_with_verbs(
    action_storage: Dict[str, Any],
    verb_map: Dict[str, str],
):
    """
    Update general and alt verb fields.

    - General template verb is set directly.
    - Alt verb is set to alt's own verb if present, otherwise copied from general.
    """
    for general_name, entry in action_storage.items():
        if not isinstance(entry, dict):
            continue

        general_verb = verb_map.get(general_name, entry.get("verb", ""))
        if general_verb:
            entry["verb"] = general_verb

        alts = entry.get("alts", {})
        if isinstance(alts, dict):
            for alt_name, alt_entry in alts.items():
                if isinstance(alt_entry, dict):
                    alt_verb = alt_entry.get("verb", "")
                    if not alt_verb:
                        alt_entry["verb"] = general_verb


# =============================================================================
# Rebuild verb storage
# =============================================================================
def build_verb_storage(action_storage: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild verb_storage.json from action_storage_linked.json."""
    verb_to_templates = defaultdict(set)

    for general_name, entry in action_storage.items():
        if not isinstance(entry, dict):
            continue

        general_verb = entry.get("verb", "")
        if general_verb:
            verb_to_templates[general_verb].add(general_name)

        alts = entry.get("alts", {})
        if isinstance(alts, dict):
            for alt_name, alt_entry in alts.items():
                if isinstance(alt_entry, dict):
                    alt_verb = alt_entry.get("verb", "")
                    if alt_verb:
                        verb_to_templates[alt_verb].add(alt_name)

    verb_storage = {"verb_templates": {}}

    for verb, template_names in verb_to_templates.items():
        verb_storage["verb_templates"][verb] = {
            "action_templates": sorted(template_names),
        }

    return verb_storage


# =============================================================================
# Main
# =============================================================================
async def main():
    print("📂 Loading linked action storage...")
    action_storage = load_json(ACTION_STORAGE_LINKED_PATH)
    print(f"   Loaded {len(action_storage)} general templates.")

    print("🔍 Finding general templates with missing verbs...")
    missing_verb_templates = find_general_templates_missing_verbs(action_storage)
    print(f"   Templates needing verbs: {len(missing_verb_templates)}")

    if not missing_verb_templates:
        print("   No missing verbs found.")
    else:
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

        verb_map = await generate_verbs_for_templates(client, missing_verb_templates)

        if verb_map:
            update_action_storage_with_verbs(action_storage, verb_map)
            save_json(action_storage, ACTION_STORAGE_LINKED_PATH)
        else:
            print("   ⚠️ No verbs generated by LLM.")

    print("🔧 Rebuilding verb storage...")
    verb_storage = build_verb_storage(action_storage)
    save_json(verb_storage, VERB_STORAGE_PATH)

    print("✅ Done.")


if __name__ == "__main__":
    asyncio.run(main())