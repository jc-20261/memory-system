#!/usr/bin/env python3
r"""
generate_single_term_aliases.py

Generates LLM aliases for isolated search terms extracted from memory encodings.

This script focuses only on single terms:
  - Object template names
  - Object instance IDs
  - Action template names (including sub-actions)
  - Action categories
  - Trajectories
  - Tags

No attribute-value pairs or state-change triples are processed here.

For each memory, the script:
  1. Loads the NL memory and annotated action plot from records.jsonl
     (if available) to provide context.
  2. Extracts isolated terms.
  3. Splits the terms into chunks of 50.
  4. Calls DeepSeek (temperature=0.3, reasoning_effort="none") to generate
     up to 10 aliases per term.
  5. Post-processes aliases so spaces become underscores.

Outputs:
  Explore\single_term_aliases_readable.txt
  Explore\data\single_term_aliases.json
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import AsyncOpenAI

# ----------------------------------------------------------------------
# Paths and configuration
# ----------------------------------------------------------------------
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
OUTPUT_READABLE = EXPLORE_DIR / "single_term_aliases_readable.txt"
OUTPUT_JSON = EXPLORE_DIR / "data" / "single_term_aliases.json"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"

CHUNK_SIZE = 50
MAX_ALIASES_PER_TERM = 10
TEMPERATURE = 0.3
MAX_CONCURRENT_CHUNKS = 5

# Optional filter: set to a specific memory_id to process only that memory.
MEMORY_ID_FILTER: Optional[str] = None
# Optional limit: set to 1 for testing, None for all.
LIMIT_MEMORIES: Optional[int] = None


# ----------------------------------------------------------------------
# Memory / record loading
# ----------------------------------------------------------------------
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


def get_memory_id(memory: Dict[str, Any]) -> str:
    mid = memory.get("memory_id")
    if mid:
        return str(mid)
    inner = memory.get("memory") or {}
    mid = inner.get("memory_id")
    return str(mid) if mid else "unknown"


def get_activity(memory: Dict[str, Any]) -> str:
    act = memory.get("activity")
    if act:
        return str(act)
    inner = memory.get("memory") or {}
    act = inner.get("activity")
    return str(act) if act else "unknown"


def load_record_by_memory_id(memory_id: str) -> Optional[Dict[str, Any]]:
    records_path = MEMORIES_DIR / "records.jsonl"
    if not records_path.exists():
        return None
    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line == "---":
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict) and rec.get("memory_id") == memory_id:
                    return rec
            except json.JSONDecodeError:
                continue
    return None


# ----------------------------------------------------------------------
# Isolated term extraction
# ----------------------------------------------------------------------
def extract_isolated_terms(memory: Dict[str, Any]) -> Set[str]:
    terms: Set[str] = set()
    inner = memory.get("memory") or memory

    objects = inner.get("objects") or []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        template = obj.get("template")
        if template:
            terms.add(f"template:{template}")
        obj_id = obj.get("obj_id")
        if obj_id:
            terms.add(f"obj_id:{obj_id}")

    actions = inner.get("actions") or []
    for action in actions:
        if isinstance(action, dict):
            _collect_action_isolated(action, terms)

    return terms


def _collect_action_isolated(action: Dict[str, Any], terms: Set[str]):
    if not isinstance(action, dict):
        return

    template = action.get("template")
    if template:
        terms.add(f"action:{template}")

    category = action.get("action_category")
    if category:
        terms.add(f"category:{category}")

    trajectory = action.get("kinematic_trajectory")
    if trajectory:
        terms.add(f"trajectory:{trajectory}")

    for tag in action.get("tags") or []:
        if isinstance(tag, str):
            terms.add(f"tag:{tag}")

    for sub in action.get("sub_actions") or []:
        if isinstance(sub, dict):
            _collect_action_isolated(sub, terms)


# ----------------------------------------------------------------------
# LLM prompt and call
# ----------------------------------------------------------------------
SIMPLE_ALIAS_SYSTEM = (
    "You are a synonym generator for memory search terms.\n"
    "Given a list of terms, generate up to 10 alternative names (aliases) for each term.\n"
    "Aliases should be concise, semantically similar phrases.\n"
    "Use underscores to join words instead of spaces.\n"
    "Return a JSON object where keys are the original term strings and values are arrays of alias strings.\n"
    "Do not include the original term itself. Do not include explanations."
)


async def call_llm_for_aliases(
    client: AsyncOpenAI,
    chunk: List[str],
    common_context: str,
    chunk_index: int,
    total_chunks: int,
) -> Dict[str, List[str]]:
    user_msg = common_context + "\n\n"
    user_msg += f"Generate aliases for the following {len(chunk)} terms:\n"
    user_msg += json.dumps(chunk, indent=2, ensure_ascii=False)

    try:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SIMPLE_ALIAS_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=TEMPERATURE,
            max_tokens=8000,
            extra_body={"reasoning_effort": "none"},
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        data = json.loads(content)
        if not isinstance(data, dict):
            print(f"  ⚠️ Chunk {chunk_index}/{total_chunks}: response not a dict")
            return {}

        # Post-process: spaces -> underscores, cap aliases
        cleaned = {}
        for term, aliases in data.items():
            if not isinstance(aliases, list):
                continue
            new_aliases = []
            for alias in aliases:
                if not isinstance(alias, str):
                    continue
                alias = "_".join(alias.strip().split())
                if alias and alias not in new_aliases:
                    new_aliases.append(alias)
                if len(new_aliases) >= MAX_ALIASES_PER_TERM:
                    break
            cleaned[term] = new_aliases
        return cleaned

    except Exception as e:
        print(f"  ❌ Chunk {chunk_index}/{total_chunks} failed: {e}")
        return {}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
async def main():
    print("Loading memories...")
    memories = load_memory_records()

    # Deduplicate
    unique_memories = {}
    for mem in memories:
        mid = get_memory_id(mem)
        if mid not in unique_memories:
            unique_memories[mid] = mem
    memories = list(unique_memories.values())

    if MEMORY_ID_FILTER:
        memories = [m for m in memories if get_memory_id(m) == MEMORY_ID_FILTER]
        if not memories:
            print(f"❌ Memory ID {MEMORY_ID_FILTER} not found.")
            return

    if LIMIT_MEMORIES is not None:
        memories = memories[:LIMIT_MEMORIES]

    print(f"Processing {len(memories)} memories.")

    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)

    all_results: Dict[str, Dict[str, List[str]]] = {}

    async def process_chunk(
        memory_id: str,
        chunk: List[str],
        common_context: str,
        chunk_index: int,
        total_chunks: int,
    ) -> Tuple[str, Dict[str, List[str]]]:
        async with semaphore:
            result = await call_llm_for_aliases(
                client, chunk, common_context, chunk_index, total_chunks
            )
            return memory_id, result

    for memory in memories:
        memory_id = get_memory_id(memory)
        activity = get_activity(memory)

        # Build common context
        record = load_record_by_memory_id(memory_id)
        nl_text = ""
        annotated_plot = ""
        if record:
            nl_text = record.get("nl_text", "")
            annotated_plot = record.get("state_annotated_sketch", "") or record.get("annotated_plot", "")

        common_context = "CONTEXT:\n"
        common_context += f"Activity: {activity}\n\n"
        if nl_text:
            common_context += f"NL Memory:\n{nl_text}\n\n"
        if annotated_plot:
            common_context += f"Annotated Action Plot:\n{annotated_plot}\n\n"

        isolated_terms = sorted(extract_isolated_terms(memory))
        chunks = [
            isolated_terms[i : i + CHUNK_SIZE]
            for i in range(0, len(isolated_terms), CHUNK_SIZE)
        ]

        print(f"\nMemory {memory_id} ({activity})")
        print(f"  Isolated terms: {len(isolated_terms)}")
        print(f"  Chunks: {len(chunks)}")

        tasks = []
        for idx, chunk in enumerate(chunks, 1):
            tasks.append(
                process_chunk(memory_id, chunk, common_context, idx, len(chunks))
            )
        chunk_results = await asyncio.gather(*tasks)

        merged = {}
        for _, result in chunk_results:
            merged.update(result)
        all_results[memory_id] = merged

    # Save JSON output
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Save readable output
    with open(OUTPUT_READABLE, "w", encoding="utf-8") as f:
        f.write("SINGLE TERM ALIAS GENERATION RESULTS\n")
        f.write("=" * 90 + "\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Temperature: {TEMPERATURE}\n")
        f.write(f"Reasoning: none\n")
        f.write(f"Max aliases per term: {MAX_ALIASES_PER_TERM}\n")
        f.write("-" * 90 + "\n\n")
        f.write("SYSTEM PROMPT USED\n")
        f.write("-" * 90 + "\n")
        f.write(SIMPLE_ALIAS_SYSTEM)
        f.write("\n" + "=" * 90 + "\n\n")

        for memory_id, term_aliases in all_results.items():
            f.write(f"Memory ID: {memory_id}\n")
            f.write("-" * 60 + "\n")
            for term, aliases in term_aliases.items():
                f.write(f"\nTerm: {term}\n")
                if aliases:
                    for alias in aliases:
                        f.write(f"  - {alias}\n")
                else:
                    f.write("  (no aliases generated)\n")
            f.write("\n" + "-" * 60 + "\n\n")

    print(f"\n✅ JSON aliases saved to:\n   {OUTPUT_JSON}")
    print(f"✅ Readable results saved to:\n   {OUTPUT_READABLE}")


if __name__ == "__main__":
    asyncio.run(main())