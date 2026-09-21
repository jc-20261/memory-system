#!/usr/bin/env python3
r"""
generate_aliases_test.py

Test alias generation for one memory.

This script:
- Loads the first memory (or a specified memory) from:
    F:\New folder (4)\New folder\Memories
- Extracts base terms (isolated terms and attribute–value–op triples)
  using the robust extraction logic from the count script.
- Loads the corresponding record (NL memory + annotated action plot)
  from records.jsonl in the same folder.
- Builds two prompts sharing the same common context:
    1. Simple alias generation prompt.
    2. Attribute–value–op expansion prompt.
- Splits each term list into chunks of 50 and calls DeepSeek
  (temperature=0.3, reasoning_effort="none", response_format JSON).
- Writes a readable report to:
    Explore\alias_generation_test.txt

No filtering of generated aliases is applied.
"""
import os
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from openai import AsyncOpenAI

# ----------------------------------------------------------------------
# Paths and config
# ----------------------------------------------------------------------
EXPLORE_DIR = Path(__file__).resolve().parent.parent
MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"
OUTPUT_FILE = EXPLORE_DIR / "alias_generation_test.txt"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"

CHUNK_SIZE = 50
MAX_ALIASES_PER_TERM = 10
TEMPERATURE = 0.3

# ----------------------------------------------------------------------
# Stop/unit words and simple filters (same as count script)
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
    "templates", "aliases", "categories", "functions", "materials"
}

UNIT_WORDS = {
    "cm", "m", "mm", "km", "g", "kg", "mg", "ml", "l", "s",
    "sec", "min", "hr", "hour", "hours", "degree", "degrees",
    "percent", "%", "gram", "grams", "meter", "meters",
    "centimeter", "centimeters", "millimeter", "millimeters",
    "liter", "liters", "milliliter", "milliliters", "kilogram",
    "kilograms"
}

def make_hashable(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value

def combine_value_unit(value: Any, unit: Any) -> str:
    if unit is None:
        return str(value)
    return f"{value}{unit}"

# ----------------------------------------------------------------------
# Robust extraction functions (from count script)
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
                    result.append((new_prefix, combine_value_unit(v.get("value"), v.get("unit"))))
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
        for i, item in enumerate(obj):
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

def expand_value(base_path: Tuple[str, ...], op: str, value: Any):
    if value is None:
        return
    if isinstance(value, dict) and "value" in value and set(value.keys()) <= {"value", "unit"}:
        leaf = combine_value_unit(value.get("value"), value.get("unit"))
        yield (base_path, op, leaf)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            new_path = base_path + (str(k),)
            yield from expand_value(new_path, op, v)
    elif isinstance(value, list):
        for item in value:
            yield from expand_value(base_path, op, item)
    else:
        yield (base_path, op, make_hashable(value))

def extract_entry_triples(entry: Dict[str, Any], value_key: str, op_override: str = None) -> List[Tuple[Tuple[str, ...], str, Any]]:
    obj = entry.get("object")
    aspect = entry.get("aspect")
    attribute = entry.get("attribute")
    value = entry.get(value_key)
    if value is None:
        return []
    if op_override:
        op = op_override
    elif "op" in entry:
        op = str(entry["op"])
    else:
        op = "eq"
    base_comps: List[str] = []
    if obj is not None and str(obj).strip():
        base_comps.append(str(obj))
    path_comps = base_comps + path_from_aspect_attribute(aspect, attribute)
    base_path = tuple(path_comps)
    triples = []
    for path_tuple, _, val in expand_value(base_path, op, value):
        triples.append((path_tuple, op, val))
    return triples

def extract_terms(memory: Dict[str, Any]) -> Tuple[Set[Any], Set[Tuple[Tuple[str, ...], str, Any]]]:
    isolated: Set[Any] = set()
    triples: Set[Tuple[Tuple[str, ...], str, Any]] = set()
    inner = memory.get("memory") or memory

    # Objects
    objects = inner.get("objects") or []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        template = obj.get("template")
        if template:
            isolated.add(f"template:{template}")
        obj_id = obj.get("obj_id")
        if obj_id:
            isolated.add(f"obj_id:{obj_id}")
        overrides = obj.get("overrides") or {}
        for path, val in flatten_to_path_components(overrides):
            base_path = tuple(path)
            for t in expand_value(base_path, "eq", val):
                triples.add(t)
        for field in ("dimensions", "weight", "mass", "position",
                      "position_relative_body", "motion", "shape"):
            if field in obj:
                field_val = obj[field]
                if isinstance(field_val, dict):
                    for path, val in flatten_to_path_components(field_val):
                        full_path = (field,) + tuple(path)
                        for t in expand_value(full_path, "eq", val):
                            triples.add(t)
                else:
                    triples.add(((field,), "eq", make_hashable(field_val)))

    # Actions
    actions = inner.get("actions") or []
    for action in actions:
        if isinstance(action, dict):
            _extract_action_terms(action, isolated, triples)

    # Goal states
    goals = memory.get("goal_state") or inner.get("goal_state") or []
    if isinstance(goals, list):
        for goal in goals:
            if isinstance(goal, dict):
                obj = goal.get("object")
                attr = goal.get("attribute")
                val = goal.get("value")
                op = goal.get("op", "eq")
                if obj and attr and val is not None:
                    path = (str(obj), str(attr))
                    for t in expand_value(path, str(op), val):
                        triples.add(t)
    return isolated, triples

def _extract_action_terms(action: Dict[str, Any], isolated: Set[Any], triples: Set[Tuple[Tuple[str, ...], str, Any]]):
    if not isinstance(action, dict):
        return
    template = action.get("template")
    if template:
        isolated.add(f"action:{template}")
    category = action.get("action_category")
    if category:
        isolated.add(f"category:{category}")
    trajectory = action.get("kinematic_trajectory")
    if trajectory:
        isolated.add(f"trajectory:{trajectory}")
    for tag in action.get("tags") or []:
        if isinstance(tag, str):
            isolated.add(f"tag:{tag}")
    # Preconditions
    for prec in action.get("preconditions") or []:
        if isinstance(prec, dict):
            for t in extract_entry_triples(prec, "value"):
                triples.add(t)
    # Changes
    for change_list_key in ("changes", "changes_per_cycle", "changes_total"):
        for ch in action.get(change_list_key) or []:
            if isinstance(ch, dict):
                for t in extract_entry_triples(ch, "new", op_override="new"):
                    triples.add(t)
                for t in extract_entry_triples(ch, "old", op_override="old"):
                    triples.add(t)
    # Conditional changes
    for cond in action.get("conditional_changes") or []:
        if isinstance(cond, dict):
            for ch in cond.get("changes") or []:
                if isinstance(ch, dict):
                    for t in extract_entry_triples(ch, "new", op_override="new"):
                        triples.add(t)
                    for t in extract_entry_triples(ch, "old", op_override="old"):
                        triples.add(t)
    # Sub-actions
    for sub in action.get("sub_actions") or []:
        if isinstance(sub, dict):
            _extract_action_terms(sub, isolated, triples)

# ----------------------------------------------------------------------
# Memory loading and record loading
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

def load_record_by_memory_id(memory_id: str) -> Optional[Dict[str, Any]]:
    records_path = MEMORIES_DIR / "records.jsonl"
    if not records_path.exists():
        # fallback: try records_readable.txt (not implemented, user can adapt)
        print("⚠️ records.jsonl not found; NL context will be empty.")
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
# Formatting helpers for prompts and output
# ----------------------------------------------------------------------
def format_isolated_term(term: str) -> str:
    # Keep as is for LLM? We'll strip prefix for alias generation target
    # but keep full term as identifier.
    return term

def format_triple(triple: Tuple[Tuple[str, ...], str, Any]) -> str:
    path, op, val = triple
    path_str = ".".join(path)
    return f"{path_str} {op} {val}"

def format_triple_json(triple) -> Dict[str, Any]:
    path, op, val = triple
    return {"path": list(path), "op": op, "value": val}

# ----------------------------------------------------------------------
# LLM call
# ----------------------------------------------------------------------
async def call_llm(client: AsyncOpenAI, system_msg: str, user_msg: str) -> Optional[str]:
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
        return content
    except Exception as e:
        print(f"LLM call failed: {e}")
        return None

async def process_chunks(client, chunks, system_msg, common_context, term_kind):
    results = {}
    for i, chunk in enumerate(chunks, 1):
        print(f"  Processing {term_kind} chunk {i}/{len(chunks)} ({len(chunk)} terms)...")
        user_msg = common_context + "\n\n"
        user_msg += f"List of {term_kind} terms (generate aliases for each):\n"
        if term_kind == "isolated":
            term_list = [json.dumps(t) for t in chunk]
            user_msg += json.dumps(chunk, indent=2, ensure_ascii=False)
        else:
            term_list = [format_triple_json(t) for t in chunk]
            user_msg += json.dumps(term_list, indent=2, ensure_ascii=False)
        response_text = await call_llm(client, system_msg, user_msg)
        if response_text:
            try:
                data = json.loads(response_text)
                if isinstance(data, dict):
                    results.update(data)
                else:
                    print(f"  ⚠️ Response not a dict for chunk {i}")
            except json.JSONDecodeError:
                print(f"  ⚠️ Invalid JSON in chunk {i}")
    return results

# ----------------------------------------------------------------------
# Main test
# ----------------------------------------------------------------------
async def main():
    print("Loading memories...")
    memories = load_memory_records()
    # Deduplicate
    unique = {}
    for mem in memories:
        mid = get_memory_id(mem)
        if mid not in unique:
            unique[mid] = mem
    memories = list(unique.values())
    if not memories:
        print("No memories found.")
        return

    # Choose first memory for test
    memory = memories[0]
    memory_id = get_memory_id(memory)
    activity = memory.get("activity", "unknown")
    print(f"Using memory: {memory_id} ({activity})")

    # Extract terms
    isolated, triples = extract_terms(memory)
    print(f"Isolated terms: {len(isolated)}")
    print(f"Triples: {len(triples)}")

    # Load record
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

    # Build instruction prompts
    simple_alias_system = (
        "You are a synonym generator for memory search terms.\n"
        "Given a list of base terms (strings), generate up to 10 alternative names (aliases) for each term.\n"
        "Aliases should be synonyms or alternative expressions likely to be used in a query about the same memory.\n"
        "Return JSON object where keys are the original term strings and values are lists of alias strings.\n"
        "Do not include the original term itself in the alias list. Do not add explanations."
    )
    expansion_system = (
        "You are a memory condition expander.\n"
        "Given a list of attribute-path, operator, value triples, generate up to 10 alternative conditions that express the same meaning.\n"
        "Each alternative should be a JSON object with keys 'path' (list of strings), 'op' (string), and 'value' (string).\n"
        "You may change the path, operator, and value as long as the meaning is preserved.\n"
        "Return a JSON object where keys are the original condition string (e.g., 'path op value') and values are lists of such alternative condition objects.\n"
        "Do not include the original triple itself. Do not add explanations."
    )

    # Prepare chunks
    isolated_list = sorted(list(isolated))
    triples_list = sorted(list(triples), key=lambda t: (t[0], t[1], str(t[2])))

    isolated_chunks = [isolated_list[i:i+CHUNK_SIZE] for i in range(0, len(isolated_list), CHUNK_SIZE)]
    triple_chunks = [triples_list[i:i+CHUNK_SIZE] for i in range(0, len(triples_list), CHUNK_SIZE)]

    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    print(f"Isolated chunks: {len(isolated_chunks)}")
    print(f"Triple chunks: {len(triple_chunks)}")

    # Process
    simple_results = await process_chunks(client, isolated_chunks, simple_alias_system, common_context, "isolated")
    expansion_results = await process_chunks(client, triple_chunks, expansion_system, common_context, "triple")

    # Write readable output
    output_lines = []
    output_lines.append("ALIAS GENERATION TEST REPORT")
    output_lines.append("=" * 90)
    output_lines.append(f"Memory ID: {memory_id}")
    output_lines.append(f"Activity: {activity}")
    output_lines.append(f"Generated with model {MODEL_NAME}, temperature {TEMPERATURE}, reasoning none.")
    output_lines.append("\n" + "-" * 90)
    output_lines.append("COMMON CONTEXT (shared by both prompts)")
    output_lines.append("-" * 90)
    output_lines.append(common_context)
    output_lines.append("\n" + "=" * 90)
    output_lines.append("SIMPLE ALIAS PROMPT INSTRUCTION")
    output_lines.append("-" * 90)
    output_lines.append(simple_alias_system)
    output_lines.append("\n" + "=" * 90)
    output_lines.append("SIMPLE ALIAS RESULTS")
    output_lines.append("-" * 90)
    for term in isolated_list:
        aliases = simple_results.get(term, [])
        output_lines.append(f"\nTerm: {term}")
        if aliases:
            for alias in aliases[:MAX_ALIASES_PER_TERM]:
                output_lines.append(f"  - {alias}")
        else:
            output_lines.append("  (no aliases generated)")

    output_lines.append("\n" + "=" * 90)
    output_lines.append("EXPANSION PROMPT INSTRUCTION")
    output_lines.append("-" * 90)
    output_lines.append(expansion_system)
    output_lines.append("\n" + "=" * 90)
    output_lines.append("EXPANSION RESULTS")
    output_lines.append("-" * 90)
    for triple in triples_list:
        triple_str = format_triple(triple)
        expansions = expansion_results.get(triple_str, [])
        output_lines.append(f"\nCondition: {triple_str}")
        if expansions:
            for exp in expansions[:MAX_ALIASES_PER_TERM]:
                exp_str = json.dumps(exp, ensure_ascii=False)
                output_lines.append(f"  - {exp_str}")
        else:
            output_lines.append("  (no expansions generated)")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"✅ Test report saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())