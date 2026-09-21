#!/usr/bin/env python3
r"""
populate_goals.py – Add goal states to each existing memory JSON.

Loads memories from:
    F:\New folder (4)\New folder\Memories

For each unique activity, asks DeepSeek:
    "What is/are the goal state(s) of this activity?"

Then adds a top-level "goal_state" key to every memory with that activity.

This version:
- Keeps reasoning_effort high.
- Uses a larger token limit to avoid empty final content.
- Retries empty or invalid LLM responses.
- Processes unique activities, not duplicate records.
- Applies the same goal state to all duplicate memories with the same activity.
- Saves back to the original memory files.
"""

import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

# =============================================================================
# Configuration
# =============================================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"

MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
MAX_CONCURRENT = int(os.getenv("POPULATE_GOAL_MAX_CONCURRENT", "1"))
MAX_RETRIES = int(os.getenv("POPULATE_GOAL_MAX_RETRIES", "3"))
MAX_TOKENS = 4000

LLM_RESPONSES_FILE = Path(__file__).parent / "data" / "goal_state_llm_responses.jsonl"

# =============================================================================
# LLM prompt
# =============================================================================
SYSTEM_PROMPT = r"""
You are an expert task goal analyzer for a memory system.

You will be given an activity description from a memory encoding.
You must output the goal state(s) of that activity.

The goal state describes the successful end state of the activity, not the process.

Return JSON with a single key "goal_state", which is a list of objects.

Each goal-state object must have:
{
  "object": "object_template_name_or_generic_name",
  "attribute": "attribute_name",
  "value": "desired_value",
  "aspect": "optional_aspect"
}

Rules:
- Use object template names or generic object names, not specific obj_ids.
- Use attributes and values consistent with memory encodings.
- If multiple goals exist, list them all.
- Do not include preconditions, changes, or actions.
- Do not include any commentary.

DEMONSTRATION:

Activity: peeling an apple

Expected output:
{
  "goal_state": [
    {
      "object": "apple",
      "attribute": "skin.present",
      "value": false,
      "aspect": null
    },
    {
      "object": "peel_pile",
      "attribute": "composition",
      "value": "apple_peel + peel_strip_01",
      "aspect": null
    }
  ]
}
"""


# =============================================================================
# Memory file loading
# =============================================================================
def get_memory_id(memory: Dict[str, Any]) -> str:
    """Return the best available memory ID from a memory JSON dict."""
    memory_id = memory.get("memory_id")
    if memory_id:
        return str(memory_id)

    inner = memory.get("memory") or {}
    memory_id = inner.get("memory_id")
    return str(memory_id) if memory_id else "unknown"


def get_activity(memory: Dict[str, Any]) -> str:
    """Return the activity string for a memory JSON dict."""
    activity = memory.get("activity")
    if activity:
        return str(activity)

    inner = memory.get("memory") or {}
    activity = inner.get("activity")
    return str(activity) if activity else "unknown"


def has_goal_state(memory: Dict[str, Any]) -> bool:
    """Return True if this memory already has a non-empty goal_state list."""
    goal = memory.get("goal_state")
    if isinstance(goal, list) and goal:
        return True

    inner = memory.get("memory") or {}
    goal = inner.get("goal_state")
    return isinstance(goal, list) and bool(goal)


def load_memory_records() -> List[Dict[str, Any]]:
    """
    Scan the Memories folder for .jsonl and .json files.

    Returns a list of records:
    {
        "file_path": Path,
        "memory": dict,
        "index": int,
        "is_jsonl": bool
    }
    """
    records: List[Dict[str, Any]] = []

    # JSONL files first
    for jsonl_file in sorted(MEMORIES_DIR.glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and line.strip() != "---"]

        file_records = []
        for line in lines:
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    file_records.append(data)
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping invalid JSON line in {jsonl_file.name}: {e}")

        for idx, memory in enumerate(file_records):
            records.append(
                {
                    "file_path": jsonl_file,
                    "memory": memory,
                    "index": idx,
                    "is_jsonl": True,
                }
            )

    # JSON files
    for json_file in sorted(MEMORIES_DIR.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping invalid JSON file {json_file.name}: {e}")
                continue

        if isinstance(data, dict):
            file_memories = [data]
        elif isinstance(data, list):
            file_memories = data
        else:
            file_memories = []

        for idx, memory in enumerate(file_memories):
            if isinstance(memory, dict):
                records.append(
                    {
                        "file_path": json_file,
                        "memory": memory,
                        "index": idx,
                        "is_jsonl": False,
                    }
                )

    return records


def write_memory_records(records: List[Dict[str, Any]]):
    """
    Write updated memories back to their original files.

    Records are grouped by file path and written in their original order.
    """
    file_groups = defaultdict(list)

    for rec in records:
        file_groups[rec["file_path"]].append(rec)

    for file_path, recs in file_groups.items():
        recs.sort(key=lambda x: x["index"])
        is_jsonl = recs[0]["is_jsonl"] if recs else False
        memories = [rec["memory"] for rec in recs]

        if is_jsonl:
            with open(file_path, "w", encoding="utf-8") as f:
                for mem in memories:
                    f.write(json.dumps(mem, ensure_ascii=False) + "\n\n---\n")
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                if len(memories) == 1:
                    json.dump(memories[0], f, indent=2, ensure_ascii=False)
                else:
                    json.dump(memories, f, indent=2, ensure_ascii=False)


# =============================================================================
# LLM calls
# =============================================================================
def log_llm_response(activity: str, content: str, error: Optional[str] = None):
    """Append an LLM response record for inspection."""
    LLM_RESPONSES_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "activity": activity,
        "content": content,
        "error": error,
        "timestamp": str(__import__("datetime").datetime.now()),
    }

    with open(LLM_RESPONSES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_goal_state_response(content: str) -> Optional[List[Dict[str, Any]]]:
    """Parse LLM JSON content and return the goal_state list, or None."""
    if not content:
        return None

    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    try:
        data = json.loads(cleaned)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    goal_state = data.get("goal_state")
    if isinstance(goal_state, list):
        return goal_state

    return None


async def call_goal_state_llm(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    activity: str,
) -> Optional[List[Dict[str, Any]]]:
    """Call the LLM once for an activity. Returns goal_state list or None."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Activity: {activity}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=MAX_TOKENS,
                extra_body={"reasoning_effort": "high"},
            )

            content = response.choices[0].message.content
            if not content:
                log_llm_response(activity, "", error="empty_response")
                return None

            content = content.strip()
            goal_state = parse_goal_state_response(content)

            if goal_state is None:
                log_llm_response(activity, content, error="parse_failed")
            else:
                log_llm_response(activity, content)

            return goal_state

        except Exception as e:
            log_llm_response(activity, "", error=str(e))
            return None


async def generate_goal_state_for_activity(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    activity: str,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """Generate goal state for one activity, with retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        goal_state = await call_goal_state_llm(client, semaphore, activity)

        if goal_state:
            return activity, goal_state

        print(f"   ⚠️ Attempt {attempt}/{MAX_RETRIES} failed for: {activity}")

        if attempt < MAX_RETRIES:
            await asyncio.sleep(2 * attempt)

    return activity, None


# =============================================================================
# Main
# =============================================================================
async def main():
    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    if not MEMORIES_DIR.exists():
        print(f"❌ Memories folder not found: {MEMORIES_DIR}")
        return

    records = load_memory_records()
    if not records:
        print("❌ No memory records found.")
        return

    print(f"Loaded {len(records)} memory records from {MEMORIES_DIR}")

    # Identify unique activities that still need goal states
    activity_to_records = defaultdict(list)

    for rec in records:
        memory = rec["memory"]
        if has_goal_state(memory):
            continue

        activity = get_activity(memory)
        activity_to_records[activity].append(rec)

    unique_activities = list(activity_to_records.keys())
    if not unique_activities:
        print("🎉 All memories already have non-empty goal states.")
        return

    print(f"\n🔧 {len(unique_activities)} unique activities need goal states:\n")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        generate_goal_state_for_activity(client, semaphore, activity)
        for activity in unique_activities
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful_activities = 0
    failed_activities = 0

    for result in results:
        if isinstance(result, Exception):
            failed_activities += 1
            continue

        activity, goal_state = result
        if not goal_state:
            failed_activities += 1
            print(f"❌ No goal state generated for: {activity}")
            continue

        successful_activities += 1
        print(f"✅ {activity}: {len(goal_state)} goal state(s)")

        # Apply the same goal_state to all records with this activity
        for rec in activity_to_records.get(activity, []):
            rec["memory"]["goal_state"] = goal_state

    print(f"\n✅ Activities updated: {successful_activities}")
    print(f"⚠️ Activities still failed: {failed_activities}")

    # Verify missing records
    missing = []
    for rec in records:
        if not has_goal_state(rec["memory"]):
            missing.append(get_memory_id(rec["memory"]))

    if missing:
        print(f"\n⚠️ Memories still missing goal_state: {len(missing)}")
        print(f"   Examples: {missing[:20]}")

    # Write all records back to their original files
    write_memory_records(records)

    print(f"\n💾 Saved updated memories back to {MEMORIES_DIR}")


if __name__ == "__main__":
    asyncio.run(main())