#!/usr/bin/env python3
r"""
llm_action_verb_generalization.py

Reads action templates and existing memories, then asks an LLM to assign a
verb-level generalization to every action template.

Inputs:
    Explore\data\action_storage.json
    F:\New folder (4)\New folder\Memories

Outputs:
    Explore\action_verb_generalization.txt
    Explore\data\action_verb_generalization.json

    Explore\verb_frequencies.txt
    Explore\data\verb_frequencies.json

The generated JSON supports tracing:

    verb → action_template → memory_ids → action_instances

Each action instance stores its location using an array path:

    "path_parts": ["memory", "actions", "0", "sub_actions", "1"]

No memory files are modified.
"""

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import AsyncOpenAI

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
ACTION_STORAGE_PATH = EXPLORE_DIR / "data" / "action_storage.json"

READABLE_ACTION_VERB_OUTPUT = EXPLORE_DIR / "action_verb_generalization.txt"
JSON_ACTION_VERB_OUTPUT = EXPLORE_DIR / "data" / "action_verb_generalization.json"

READABLE_VERB_FREQ_OUTPUT = EXPLORE_DIR / "verb_frequencies.txt"
JSON_VERB_FREQ_OUTPUT = EXPLORE_DIR / "data" / "verb_frequencies.json"

# =============================================================================
# LLM config
# =============================================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"

MAX_TOKENS = 50000
BATCH_SIZE = 100
MAX_RETRIES = 1


# =============================================================================
# Manual examples for the LLM
# =============================================================================
VERB_EXAMPLES = {
    "inspect_object": "inspect",
    "grasp_object_hand": "grasp",
    "draw_knife_along_surface": "pull",
    "cut_along_seam": "cut",
    "rotate_apple_by_hand": "rotate",
    "apply_gentle_pressure": "pressure",
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
    "open_palm_and_spread_fingers": "spread_open",
    "curl_fingers_around": "curl",
    "tilt_tool_upward": "tilt",
    "rotate_wrist": "rotate",
    "flex_forearm": "flex",
    "press_tool_into_surface": "press",
    "liquid_flow": "flow",
    "flesh_oxidation": "oxidize",
    "gravity_pull": "gravitate",
}


# =============================================================================
# Memory/action extraction
# =============================================================================
def get_memory_id(memory: Dict[str, Any]) -> str:
    memory_id = memory.get("memory_id")
    if memory_id:
        return str(memory_id)

    inner = memory.get("memory") or {}
    memory_id = inner.get("memory_id")
    return str(memory_id) if memory_id else "unknown"


def load_memory_records() -> List[Dict[str, Any]]:
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


def collect_action_template_usage(
    memories: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Build mapping:

    {
      "action_template": {
        "memory_ids": set(),
        "instances": [
          {"memory_id": "...", "path_parts": [...]}
        ]
      }
    }
    """
    mapping = defaultdict(lambda: {"memory_ids": set(), "instances": []})

    for memory in memories:
        memory_id = get_memory_id(memory)
        inner = memory.get("memory") or {}
        actions = inner.get("actions") or []

        def walk(action: Dict[str, Any], path_parts: List[str]):
            if not isinstance(action, dict):
                return

            template = action.get("template")
            if template:
                template_str = str(template).strip()
                entry = mapping[template_str]
                entry["memory_ids"].add(memory_id)
                entry["instances"].append(
                    {
                        "memory_id": memory_id,
                        "path_parts": list(path_parts),
                    }
                )

            sub_actions = action.get("sub_actions") or []
            if isinstance(sub_actions, list):
                for i, sub in enumerate(sub_actions):
                    walk(
                        sub,
                        path_parts + ["sub_actions", str(i)],
                    )

        for i, action in enumerate(actions):
            if isinstance(action, dict):
                walk(action, ["memory", "actions", str(i)])

    # Convert sets to sorted lists later when writing
    normalized = {}
    for template, data in mapping.items():
        normalized[template] = {
            "memory_ids": sorted(data["memory_ids"]),
            "instances": data["instances"],
        }

    return normalized


# =============================================================================
# Action storage loading
# =============================================================================
def load_action_template_names() -> Set[str]:
    if not ACTION_STORAGE_PATH.exists():
        return set()

    with open(ACTION_STORAGE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return set()

    templates = data.get("templates")
    if isinstance(templates, dict):
        return set(templates.keys())

    return set()


# =============================================================================
# LLM batching
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
    batch_templates: List[str],
    batch_index: int,
    total_batches: int,
) -> Dict[str, str]:
    system_msg = build_system_prompt()
    user_msg = (
        f"Batch {batch_index}/{total_batches}\n"
        "Assign a verb-level generalization to each action template.\n"
        "Return JSON: {\"action_template\": \"verb\"}\n\n"
        + json.dumps(batch_templates, indent=2, ensure_ascii=False)
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


async def call_llm_for_all_templates(
    client: AsyncOpenAI,
    templates: List[str],
) -> Dict[str, str]:
    if not templates:
        return {}

    templates = sorted(templates)

    batches = [
        templates[i : i + BATCH_SIZE]
        for i in range(0, len(templates), BATCH_SIZE)
    ]

    total_batches = len(batches)
    print(f"🤖 Verb-generalizing {len(templates)} action templates in {total_batches} batches...")

    result = {}

    for idx, batch in enumerate(batches, 1):
        mapping = await call_llm_for_batch(client, batch, idx, total_batches)
        result.update(mapping)
        print(f"   Batch {idx}/{total_batches}: {len(mapping)} mappings")

    return result


# =============================================================================
# Output
# =============================================================================
def write_outputs(
    template_usage: Dict[str, Dict[str, Any]],
    verb_by_template: Dict[str, str],
):
    # Prepare rich structure
    action_templates_output = {}
    verbs_to_templates = defaultdict(list)

    for template, usage in template_usage.items():
        verb = verb_by_template.get(template, "(no verb)")

        action_templates_output[template] = {
            "verb": verb,
            "memory_ids": usage["memory_ids"],
            "instances": usage["instances"],
        }

        verbs_to_templates[verb].append(template)

    # 1. Readable action-template + verb + memory ids
    with open(READABLE_ACTION_VERB_OUTPUT, "w", encoding="utf-8") as f:
        f.write("ACTION TEMPLATE VERB GENERALIZATION\n")
        f.write("=" * 80 + "\n\n")

        for template in sorted(action_templates_output.keys()):
            entry = action_templates_output[template]
            f.write(f"ACTION TEMPLATE: {template}\n")
            f.write(f"Verb: {entry['verb']}\n")
            f.write(f"Memory IDs: {', '.join(entry['memory_ids']) if entry['memory_ids'] else '(none)'}\n")
            f.write(f"Instance count: {len(entry['instances'])}\n")
            f.write("-" * 80 + "\n")

    # 2. JSON output
    JSON_ACTION_VERB_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    json_output = {
        "action_templates": action_templates_output,
        "verbs": {
            verb: sorted(action_templates)
            for verb, action_templates in verbs_to_templates.items()
        },
    }

    with open(JSON_ACTION_VERB_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    # 3. Verb frequency files
    verb_freq_data = {}

    for verb, action_templates in verbs_to_templates.items():
        memory_ids = set()
        instance_count = 0

        for template in action_templates:
            usage = template_usage.get(template, {})
            memory_ids.update(usage.get("memory_ids", []))
            instance_count += len(usage.get("instances", []))

        verb_freq_data[verb] = {
            "action_template_count": len(action_templates),
            "memory_count": len(memory_ids),
            "instance_count": instance_count,
        }

    with open(READABLE_VERB_FREQ_OUTPUT, "w", encoding="utf-8") as f:
        f.write("VERB FREQUENCIES\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Verb':30s} {'Action Templates':>18s} {'Memories':>10s} {'Instances':>10s}\n")
        f.write("-" * 80 + "\n")

        for verb, stats in sorted(
            verb_freq_data.items(),
            key=lambda x: (
                -x[1]["instance_count"],
                -x[1]["memory_count"],
                x[0],
            ),
        ):
            f.write(
                f"{verb:30s} "
                f"{stats['action_template_count']:>18d} "
                f"{stats['memory_count']:>10d} "
                f"{stats['instance_count']:>10d}\n"
            )

        f.write("-" * 80 + "\n")
        f.write(f"Total unique verbs: {len(verb_freq_data)}\n")
        f.write(f"Total action templates: {len(action_templates_output)}\n")

    JSON_VERB_FREQ_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(JSON_VERB_FREQ_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(verb_freq_data, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Action-template verb output saved to:")
    print(f"   {READABLE_ACTION_VERB_OUTPUT}")
    print(f"📄 JSON action-template output saved to:")
    print(f"   {JSON_ACTION_VERB_OUTPUT}")
    print(f"📄 Verb frequencies saved to:")
    print(f"   {READABLE_VERB_FREQ_OUTPUT}")
    print(f"📄 JSON verb frequencies saved to:")
    print(f"   {JSON_VERB_FREQ_OUTPUT}")


# =============================================================================
# Main
# =============================================================================
async def main():
    print("📂 Loading memory records...")
    memories = load_memory_records()
    print(f"✅ Loaded {len(memories)} memory records.")

    print("🔍 Extracting action template usage...")
    template_usage = collect_action_template_usage(memories)
    print(f"   Action templates found in memories: {len(template_usage)}")

    print("📂 Loading action storage templates...")
    storage_templates = load_action_template_names()
    print(f"   Action templates in storage: {len(storage_templates)}")

    all_templates = set(template_usage.keys()) | storage_templates

    # Ensure every action template in usage has a usage entry
    for template in all_templates:
        if template not in template_usage:
            template_usage[template] = {
                "memory_ids": [],
                "instances": [],
            }

    print(f"   Total unique action templates: {len(all_templates)}")

    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    verb_by_template = await call_llm_for_all_templates(
        client,
        sorted(all_templates),
    )

    # Fill manual examples first
    for template, verb in VERB_EXAMPLES.items():
        if template not in verb_by_template:
            verb_by_template[template] = verb

    write_outputs(template_usage, verb_by_template)


if __name__ == "__main__":
    asyncio.run(main())