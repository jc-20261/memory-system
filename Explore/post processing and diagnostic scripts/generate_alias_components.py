#!/usr/bin/env python3
r"""
generate_component_value_aliases.py

Generates LLM aliases for:
  1. Individual attribute components (from attribute paths, excluding object IDs,
     structural containers, measurement attributes, stop words, and numeric tokens).
  2. Non-scalar values (simple strings that are not numeric, boolean, stop words,
     or units).

For each memory, the script uses the NL memory and annotated action plot as context,
then batches the components and values separately (50 terms per chunk) and calls
DeepSeek (temperature=0.3, reasoning_effort="none") to generate up to 10 aliases per term.

Aliases are post-processed to use underscores instead of spaces.

Outputs:
  Explore\component_value_aliases_readable.txt
  Explore\data\alias_expansion_component_value.json
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import AsyncOpenAI

# ----------------------------------------------------------------------
# Paths and configuration
# ----------------------------------------------------------------------
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
OUTPUT_READABLE = EXPLORE_DIR / "component_value_aliases_readable.txt"
OUTPUT_JSON = EXPLORE_DIR / "data" / "alias_expansion_component_value.json"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"

CHUNK_SIZE = 50
MAX_ALIASES_PER_TERM = 10
TEMPERATURE = 0.3
MAX_CONCURRENT_CHUNKS = 5

# Optional filter for testing
MEMORY_ID_FILTER: Optional[str] = None
LIMIT_MEMORIES: Optional[int] = None

# ----------------------------------------------------------------------
# Stop words and unit words
# ----------------------------------------------------------------------
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
    "relationr", "relative_to"
}

UNIT_WORDS = {
    "cm", "m", "mm", "km", "g", "kg", "mg", "ml", "l", "s",
    "sec", "min", "hr", "hour", "hours", "degree", "degrees",
    "percent", "%", "gram", "grams", "meter", "meters",
    "centimeter", "centimeters", "millimeter", "millimeters",
    "liter", "liters", "milliliter", "milliliters", "kilogram",
    "kilograms"
}

# ----------------------------------------------------------------------
# Banned attribute components (these will not be aliased as components)
# ----------------------------------------------------------------------
BANNED_ATTRIBUTE_COMPONENTS = {
    "length", "width", "height_thickness", "height", "mass", "weight",
    "shape", "dimensions", "volume", "duration", "repetitions",
    "cycle_duration", "start_offset", "temporal_type", "number",
    "position", "position_relative_body", "material", "materials",
    "category", "categories", "function", "functions",
    "relation", "relative_to", "relationr"
}

def is_valid_component(comp: str) -> bool:
    """Check if a component should be included for alias generation."""
    if not comp or not isinstance(comp, str):
        return False
    comp = comp.strip()
    if not comp:
        return False
    if comp.lower() in BANNED_ATTRIBUTE_COMPONENTS:
        return False
    if comp.lower() in STOP_WORDS or comp.lower() in UNIT_WORDS:
        return False
    if re.fullmatch(r"\d+", comp):
        return False
    return True

def is_valid_value(val: Any) -> bool:
    """Check if a value is a non-scalar simple string suitable for alias generation."""
    if not isinstance(val, str):
        return False
    s = val.strip()
    if not s:
        return False
    if s.lower() in STOP_WORDS or s.lower() in UNIT_WORDS:
        return False
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return False
    if s.lower() in {"true", "false", "none", "null"}:
        return False
    return True

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
# Flattening utilities
# ----------------------------------------------------------------------
def flatten_to_path_components(obj: Any, prefix: Tuple[str, ...] = None) -> List[Tuple[Tuple[str, ...], Any]]:
    if prefix is None:
        prefix = ()
    result = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = prefix + (str(k),)
            if isinstance(v, dict):
                if "value" in v and set(v.keys()) <= {"value", "unit"}:
                    val = v.get("value")
                    unit = v.get("unit", "")
                    result.append((new_prefix, f"{val}{unit}" if unit else str(val)))
                else:
                    result.extend(flatten_to_path_components(v, new_prefix))
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        result.extend(flatten_to_path_components(item, new_prefix))
                    else:
                        result.append((new_prefix, item))
            else:
                result.append((new_prefix, v))
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                result.extend(flatten_to_path_components(item, prefix))
            else:
                result.append((prefix, item))
    else:
        result.append((prefix, obj))
    return result

def collect_path_keys(node: Any, comps: List[str]):
    if isinstance(node, str):
        comps.append(node)
    elif isinstance(node, dict):
        for k, v in node.items():
            comps.append(str(k))
            if isinstance(v, dict):
                collect_path_keys(v, comps)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        collect_path_keys(item, comps)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, str):
                comps.append(item)
            elif isinstance(item, dict):
                collect_path_keys(item, comps)

def path_from_aspect_attribute(aspect: Any, attribute: Any) -> List[str]:
    comps: List[str] = []
    if aspect is not None:
        if isinstance(aspect, str):
            comps.append(aspect)
        elif isinstance(aspect, (dict, list)):
            collect_path_keys(aspect, comps)
    if attribute is not None:
        if isinstance(attribute, str):
            comps.append(attribute)
        elif isinstance(attribute, list):
            for item in attribute:
                if isinstance(item, str):
                    comps.append(item)
                elif isinstance(item, dict):
                    collect_path_keys(item, comps)
        elif isinstance(attribute, dict):
            collect_path_keys(attribute, comps)
    return comps

# ----------------------------------------------------------------------
# Extract attribute components and values from a memory
# ----------------------------------------------------------------------
def extract_attribute_components(memory: Dict[str, Any]) -> Set[str]:
    components: Set[str] = set()
    inner = memory.get("memory") or memory

    # Objects: overrides and explicit fields
    objects = inner.get("objects") or []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        # overrides
        overrides = obj.get("overrides") or {}
        for path, _ in flatten_to_path_components(overrides):
            for comp in path:
                if is_valid_component(comp):
                    components.add(comp)
        # explicit fields: dimensions, weight, mass, position, motion, shape
        for field in ("dimensions", "weight", "mass", "position", "motion", "shape"):
            if field in obj:
                field_val = obj[field]
                if isinstance(field_val, dict):
                    for path, _ in flatten_to_path_components(field_val):
                        for comp in path:
                            if is_valid_component(comp):
                                components.add(comp)
                else:
                    if is_valid_component(field):
                        components.add(field)

    # Actions: preconditions, changes, conditional changes
    actions = inner.get("actions") or []
    for action in actions:
        if isinstance(action, dict):
            _collect_action_components(action, components)

    # Goal states
    goals = memory.get("goal_state") or inner.get("goal_state") or []
    if isinstance(goals, list):
        for goal in goals:
            if isinstance(goal, dict):
                attr = goal.get("attribute")
                if attr:
                    if isinstance(attr, str):
                        comps = [attr]
                    elif isinstance(attr, (list, tuple)):
                        comps = list(attr)
                    else:
                        comps = [str(attr)]
                    for comp in comps:
                        if is_valid_component(comp):
                            components.add(comp)
    return components

def _collect_action_components(action: Dict[str, Any], components: Set[str]):
    if not isinstance(action, dict):
        return
    for prec in action.get("preconditions") or []:
        if isinstance(prec, dict):
            comps = path_from_aspect_attribute(prec.get("aspect"), prec.get("attribute"))
            for comp in comps:
                if is_valid_component(comp):
                    components.add(comp)
    for list_key in ("changes", "changes_per_cycle", "changes_total"):
        for ch in action.get(list_key) or []:
            if isinstance(ch, dict):
                comps = path_from_aspect_attribute(ch.get("aspect"), ch.get("attribute"))
                for comp in comps:
                    if is_valid_component(comp):
                        components.add(comp)
    for cond in action.get("conditional_changes") or []:
        if isinstance(cond, dict):
            for ch in cond.get("changes") or []:
                if isinstance(ch, dict):
                    comps = path_from_aspect_attribute(ch.get("aspect"), ch.get("attribute"))
                    for comp in comps:
                        if is_valid_component(comp):
                            components.add(comp)
    for sub in action.get("sub_actions") or []:
        if isinstance(sub, dict):
            _collect_action_components(sub, components)

def extract_values(memory: Dict[str, Any]) -> Set[str]:
    values: Set[str] = set()
    inner = memory.get("memory") or memory

    # Objects: overrides and explicit fields (only leaf values)
    objects = inner.get("objects") or []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        overrides = obj.get("overrides") or {}
        for path, val in flatten_to_path_components(overrides):
            if is_valid_value(val):
                values.add(val)
        # explicit fields can have values too (shape, motion, etc.)
        for field in ("dimensions", "weight", "mass", "position", "motion", "shape"):
            if field in obj:
                field_val = obj[field]
                if isinstance(field_val, dict):
                    for path, val in flatten_to_path_components(field_val):
                        if is_valid_value(val):
                            values.add(val)
                else:
                    if is_valid_value(field_val):
                        values.add(field_val)

    # Actions
    actions = inner.get("actions") or []
    for action in actions:
        if isinstance(action, dict):
            _collect_action_values(action, values)

    # Goal states
    goals = memory.get("goal_state") or inner.get("goal_state") or []
    if isinstance(goals, list):
        for goal in goals:
            if isinstance(goal, dict):
                val = goal.get("value")
                if is_valid_value(val):
                    values.add(val)
    return values

def _collect_action_values(action: Dict[str, Any], values: Set[str]):
    if not isinstance(action, dict):
        return
    for prec in action.get("preconditions") or []:
        if isinstance(prec, dict):
            if is_valid_value(prec.get("value")):
                values.add(prec["value"])
    for list_key in ("changes", "changes_per_cycle", "changes_total"):
        for ch in action.get(list_key) or []:
            if isinstance(ch, dict):
                if is_valid_value(ch.get("old")):
                    values.add(ch["old"])
                if is_valid_value(ch.get("new")):
                    values.add(ch["new"])
    for cond in action.get("conditional_changes") or []:
        if isinstance(cond, dict):
            for ch in cond.get("changes") or []:
                if isinstance(ch, dict):
                    if is_valid_value(ch.get("old")):
                        values.add(ch["old"])
                    if is_valid_value(ch.get("new")):
                        values.add(ch["new"])
    for sub in action.get("sub_actions") or []:
        if isinstance(sub, dict):
            _collect_action_values(sub, values)

# ----------------------------------------------------------------------
# LLM prompts and call
# ----------------------------------------------------------------------
COMPONENT_ALIAS_SYSTEM = (
    "You are a synonym generator for attribute component names used in a memory system.\n"
    "Given a list of attribute components (single words or short phrases), generate up to 10 alternative names or synonyms for each component.\n"
    "The aliases should be concise, and words separated by underscores.\n"
    "Return a JSON object where keys are the original component strings and values are arrays of alias strings.\n"
    "Do not include the original term itself. Do not include explanations."
)

VALUE_ALIAS_SYSTEM = (
    "You are a synonym generator for non-scalar attribute values used in a memory system.\n"
    "Given a list of values (strings), generate up to 10 alternative values or synonyms for each value.\n"
    "The aliases should be concise, and words separated by underscores.\n"
    "Return a JSON object where keys are the original value strings and values are arrays of alias strings.\n"
    "Do not include the original term itself. Do not include explanations."
)

async def call_llm_for_aliases(
    client: AsyncOpenAI,
    chunk: List[str],
    common_context: str,
    system_msg: str,
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
                {"role": "system", "content": system_msg},
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

    # Global alias containers
    global_component_aliases: Dict[str, List[str]] = {}
    global_value_aliases: Dict[str, List[str]] = {}

    async def process_chunk(
        memory_id: str,
        chunk: List[str],
        common_context: str,
        system_msg: str,
        chunk_index: int,
        total_chunks: int,
    ) -> Tuple[str, Dict[str, List[str]]]:
        async with semaphore:
            result = await call_llm_for_aliases(
                client, chunk, common_context, system_msg, chunk_index, total_chunks
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

        # Extract components and values for this memory
        components = sorted(extract_attribute_components(memory))
        values = sorted(extract_values(memory))

        print(f"\nMemory {memory_id} ({activity})")
        print(f"  Attribute components: {len(components)}")
        print(f"  Values: {len(values)}")

        # Process components
        comp_chunks = [components[i:i+CHUNK_SIZE] for i in range(0, len(components), CHUNK_SIZE)]
        comp_tasks = []
        for idx, chunk in enumerate(comp_chunks, 1):
            comp_tasks.append(
                process_chunk(memory_id, chunk, common_context, COMPONENT_ALIAS_SYSTEM, idx, len(comp_chunks))
            )
        comp_results = await asyncio.gather(*comp_tasks)
        for _, result in comp_results:
            for term, aliases in result.items():
                global_component_aliases.setdefault(term, [])
                for alias in aliases:
                    if alias not in global_component_aliases[term]:
                        global_component_aliases[term].append(alias)
                # Cap at 10
                global_component_aliases[term] = global_component_aliases[term][:MAX_ALIASES_PER_TERM]

        # Process values
        val_chunks = [values[i:i+CHUNK_SIZE] for i in range(0, len(values), CHUNK_SIZE)]
        val_tasks = []
        for idx, chunk in enumerate(val_chunks, 1):
            val_tasks.append(
                process_chunk(memory_id, chunk, common_context, VALUE_ALIAS_SYSTEM, idx, len(val_chunks))
            )
        val_results = await asyncio.gather(*val_tasks)
        for _, result in val_results:
            for term, aliases in result.items():
                global_value_aliases.setdefault(term, [])
                for alias in aliases:
                    if alias not in global_value_aliases[term]:
                        global_value_aliases[term].append(alias)
                global_value_aliases[term] = global_value_aliases[term][:MAX_ALIASES_PER_TERM]

    # Save JSON output
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "attribute_component_aliases": global_component_aliases,
        "value_aliases": global_value_aliases,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Save readable output
    with open(OUTPUT_READABLE, "w", encoding="utf-8") as f:
        f.write("COMPONENT AND VALUE ALIAS GENERATION RESULTS\n")
        f.write("=" * 90 + "\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Temperature: {TEMPERATURE}\n")
        f.write(f"Reasoning: none\n")
        f.write(f"Max aliases per term: {MAX_ALIASES_PER_TERM}\n")
        f.write("-" * 90 + "\n\n")
        f.write("SYSTEM PROMPT (Components)\n")
        f.write("-" * 90 + "\n")
        f.write(COMPONENT_ALIAS_SYSTEM)
        f.write("\n" + "=" * 90 + "\n\n")
        f.write("SYSTEM PROMPT (Values)\n")
        f.write("-" * 90 + "\n")
        f.write(VALUE_ALIAS_SYSTEM)
        f.write("\n" + "=" * 90 + "\n\n")

        f.write("ATTRIBUTE COMPONENT ALIASES\n")
        f.write("-" * 90 + "\n")
        for term, aliases in sorted(global_component_aliases.items()):
            f.write(f"\nComponent: {term}\n")
            for alias in aliases:
                f.write(f"  - {alias}\n")

        f.write("\n" + "=" * 90 + "\n")
        f.write("VALUE ALIASES\n")
        f.write("-" * 90 + "\n")
        for term, aliases in sorted(global_value_aliases.items()):
            f.write(f"\nValue: {term}\n")
            for alias in aliases:
                f.write(f"  - {alias}\n")

    print(f"\n✅ JSON aliases saved to:\n   {OUTPUT_JSON}")
    print(f"✅ Readable results saved to:\n   {OUTPUT_READABLE}")

if __name__ == "__main__":
    asyncio.run(main())