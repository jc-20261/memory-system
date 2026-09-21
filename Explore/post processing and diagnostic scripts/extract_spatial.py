#!/usr/bin/env python3
r"""
llm_spatial_reenconding_extract.py

Reads memory files from:
    F:\New folder (4)\New folder\Memories

Extracts existing spatial relation occurrences:
    object, relation, relative_to

Uses an LLM to re-encode the relation field using all three fields:
    object
    relation
    relative_to

This version:

- Batches occurrences for the LLM.
- Uses max_tokens=50000 per call.
- Retries failed batches once.
- Validates returned re-encoded values.
- Still respects manually approved relation mappings.
- Writes per-occurrence re-encoding results.

Outputs:
    Explore\spatial_reenconding_output.txt
    Explore\data\spatial_reenconding_output.json

No memory files are modified.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

# =============================================================================
# Paths
# =============================================================================
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
READABLE_OUTPUT = EXPLORE_DIR / "spatial_reenconding_output.txt"
JSON_OUTPUT = EXPLORE_DIR / "data" / "spatial_reenconding_output.json"

# =============================================================================
# LLM config
# =============================================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"

MAX_TOKENS = 50000
BATCH_SIZE = 25
MAX_RETRIES = 1

# =============================================================================
# Manual re-encodings already approved
# =============================================================================
MANUAL_REENCODINGS = {
    "on": "on",
    "part_of": "in",
    "in": "in",
    "inside": "in",
    "surrounding": "around",
    "attached_to": "attached",
    "at": "near",
    "displayed_on": "on",
    "near": "near",
    "above": "above",
    "below": "below",
    "within": "in",
    "under": "below",
    "covering": "on",
    "front_of": "near",
    "attached": "attached",
    "beside": "near",
    "around": "around",
    "against": "contact",
    "between": "between",
    "behind": "near",
    "top_of": "above",
    "overlay": "on",
    "propagating": "in",
    "none": "none",
    "from": "from",
    "running_on": "on",
    "through": "through",
    "installed_in_ground": "on",
    "airborne": "above",
    "travelling": "in",
    "off": "apart",
    "under_mulch": "below",
    "on_mantel": "in",
    "from_source_to_surface": "on",
    "distant_from": "apart",
    "surrounding_interface": "on",
    "around_lens": "in",
    "between_light_source_and_aperture": "near",
    "connected_to_motor": "attached",
}

SPECIAL_KEEP = {"none", "from"}

RESTRICTED_SPATIAL_OPERATORS = [
    "on",
    "in",
    "below",
    "above",
    "around",
    "through",
    "contact",
    "attached",
    "between",
    "near",
    "apart",
]

VALID_OUTPUT_VALUES = set(RESTRICTED_SPATIAL_OPERATORS) | SPECIAL_KEEP


# =============================================================================
# Helpers
# =============================================================================
def get_memory_id(memory: Dict[str, Any]) -> str:
    memory_id = memory.get("memory_id")
    if memory_id:
        return str(memory_id)

    inner = memory.get("memory") or {}
    memory_id = inner.get("memory_id")
    return str(memory_id) if memory_id else "unknown"


def safe_string(value: Any) -> str:
    """Convert a value to a stable string for LLM input/output."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


# =============================================================================
# Memory loading
# =============================================================================
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


# =============================================================================
# Spatial extraction
# =============================================================================
def extract_spatial_records(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Recursively find dicts with relation and relative_to keys.
    Preserve the nearest object name/id from ancestor context.
    """
    memory_id = get_memory_id(memory)
    records = []

    def walk(node, path_parts, current_object):
        if isinstance(node, dict):
            if "obj_id" in node:
                current_object = node.get("obj_id")
            elif "object" in node:
                current_object = node.get("object")
            elif "template" in node:
                current_object = node.get("template")

            if "relation" in node and "relative_to" in node:
                relation = node.get("relation")
                relative_to = node.get("relative_to")

                if relation is not None:
                    records.append(
                        {
                            "memory_id": memory_id,
                            "object": current_object or "",
                            "relation": str(relation).strip(),
                            "relative_to": relative_to,
                            "path": ".".join(path_parts),
                        }
                    )

            for key, value in node.items():
                walk(value, path_parts + [str(key)], current_object)

        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, path_parts + [str(i)], current_object)

    walk(memory, [], "")
    return records


# =============================================================================
# LLM re-encoding
# =============================================================================
def build_system_prompt() -> str:
    manual = json.dumps(MANUAL_REENCODINGS, indent=2, ensure_ascii=False)

    return f"""
You are a spatial relation re-encoder.

You will be given a numbered list of spatial relation occurrences.
Each occurrence has:
- object
- relation
- relative_to

Use all three fields to select the correct re-encoded relation. Consider the object and the subject that object's position is relative to (i.e. the value in the relative_to field), then compare the relation value to the spatial operators in the list below. Which operator from the list best fits or can best replace the current relation value, given the object and the subject it is positioned relative to.
For example, if the object is person, the relation is "center_of_room", and the position is relative to "room_01", the most appropriate replacement spatial operator would be "in", because if the person is center of room with respect to room, then he's in the room. 

Restricted spatial operators:
{", ".join(RESTRICTED_SPATIAL_OPERATORS)}

Special operators:
- "none" stays "none"
- "from" stays "from"

Do not invent new operators.
Return only a JSON object mapping each occurrence number to its re-encoded relation.

Example:
{{
  "0": "near",
  "1": "on",
  "2": "in"
}}

Manual re-encodings already approved:
{manual}
"""


async def call_llm_for_batch(
    client: AsyncOpenAI,
    batch_items: List[Dict[str, Any]],
    batch_index: int,
    total_batches: int,
) -> Dict[str, str]:
    """
    Call the LLM for one batch of spatial relation occurrences.

    Returns mapping: occurrence_index -> reencoded relation
    """
    system_msg = build_system_prompt()

    user_lines = []
    for item in batch_items:
        user_lines.append(
            f"{item['index']}: object={safe_string(item['object'])} | "
            f"relation={safe_string(item['relation'])} | "
            f"relative_to={safe_string(item['relative_to'])}"
        )

    user_msg = (
        f"Batch {batch_index}/{total_batches}\n"
        "Re-encode the following spatial relation occurrences.\n"
        "Return JSON: {\"<occurrence_number>\": \"reencoded_relation\"}\n\n"
        + "\n".join(user_lines)
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

            for key, value in data.items():
                try:
                    occurrence_index = str(key)
                except Exception:
                    continue

                reencoded_str = str(value).strip()

                if reencoded_str in VALID_OUTPUT_VALUES:
                    valid_mapping[occurrence_index] = reencoded_str

            if valid_mapping:
                return valid_mapping
            else:
                print(f"   ⚠️ Batch {batch_index}: no valid mappings")

        except Exception as e:
            print(f"   ❌ Batch {batch_index} failed: {e}")

    return {}


async def call_llm_for_occurrences(
    client: AsyncOpenAI,
    items: List[Dict[str, Any]],
) -> Dict[str, str]:
    if not items:
        return {}

    batches = [
        items[i : i + BATCH_SIZE]
        for i in range(0, len(items), BATCH_SIZE)
    ]

    total_batches = len(batches)
    print(f"🤖 Re-encoding {len(items)} occurrences in {total_batches} batches...")

    result = {}

    for idx, batch in enumerate(batches, 1):
        mapping = await call_llm_for_batch(client, batch, idx, total_batches)
        result.update(mapping)
        print(f"   Batch {idx}/{total_batches}: {len(mapping)} mappings")

    return result


# =============================================================================
# Output
# =============================================================================
def write_output(
    records: List[Dict[str, Any]],
    reencoding_by_index: Dict[str, str],
    readable_path: Path,
    json_path: Path,
):
    with open(readable_path, "w", encoding="utf-8") as f:
        f.write("SPATIAL RELATION RE-ENCODING OUTPUT\n")
        f.write("=" * 80 + "\n\n")

        for rec in records:
            original = rec["relation"]
            llm_index = rec.get("llm_index")

            if llm_index is not None:
                reencoded = reencoding_by_index.get(str(llm_index))
            else:
                reencoded = rec.get("manual_reencoded")

            if reencoded is None:
                continue

            f.write(f"Memory ID: {rec['memory_id']}\n")
            f.write("Original encoding:\n")
            f.write(f"  object: {rec['object']}\n")
            f.write(f"  relation: {original}\n")
            f.write(f"  relative_to: {rec['relative_to']}\n")
            f.write("Recoded encoding:\n")
            f.write(f"  object: {rec['object']}\n")
            f.write(f"  relationr: {reencoded}\n")
            f.write(f"  relative_to: {rec['relative_to']}\n")
            f.write("\n")

    json_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "records": records,
        "reencoding_by_index": reencoding_by_index,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"📄 Readable output saved to:\n   {readable_path}")
    print(f"📄 JSON output saved to:\n   {json_path}")


# =============================================================================
# Main
# =============================================================================
async def main():
    print("📂 Loading memories...")
    memories = load_memory_records()
    print(f"✅ Loaded {len(memories)} memory records.")

    print("🔍 Extracting spatial relation records...")
    all_records = []

    for memory in memories:
        records = extract_spatial_records(memory)
        all_records.extend(records)

    print(f"   Total spatial relation occurrences: {len(all_records)}")

    # Separate manual and LLM-needed records
    llm_items = []
    llm_index_counter = 0

    for rec in all_records:
        relation = rec["relation"]

        if relation in MANUAL_REENCODINGS:
            rec["manual_reencoded"] = MANUAL_REENCODINGS[relation]
            rec["llm_index"] = None
        elif relation in SPECIAL_KEEP:
            rec["manual_reencoded"] = relation
            rec["llm_index"] = None
        else:
            rec["manual_reencoded"] = None
            rec["llm_index"] = llm_index_counter
            llm_items.append(
                {
                    "index": llm_index_counter,
                    "object": rec["object"],
                    "relation": rec["relation"],
                    "relative_to": rec["relative_to"],
                }
            )
            llm_index_counter += 1

    print(f"   Occurrences handled by manual rules: "
          f"{sum(1 for r in all_records if r.get('llm_index') is None)}")
    print(f"   Occurrences needing LLM: {len(llm_items)}")

    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    reencoding_by_index = {}

    if llm_items:
        reencoding_by_index = await call_llm_for_occurrences(client, llm_items)
        print(f"   LLM returned {len(reencoding_by_index)} valid mappings.")
    else:
        print("   No LLM re-encoding needed.")

    write_output(
        all_records,
        reencoding_by_index,
        READABLE_OUTPUT,
        JSON_OUTPUT,
    )


if __name__ == "__main__":
    asyncio.run(main())