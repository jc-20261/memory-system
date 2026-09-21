#!/usr/bin/env python3
"""
populate_fields.py – Add categories, functions, and materials to object storage.

Loads object_storage.json, asks the LLM to add:

- categories
- functions
- materials

to each object template, then saves the updated object_storage.json.

This version supports:
- Flat and wrapped object-storage formats.
- Batched LLM calls.
- Configurable target groups and max concurrency.
- Materials stored as a list.
"""

import asyncio
import json
import os
import math
from pathlib import Path
from typing import Dict, Any, List

from openai import AsyncOpenAI

# =============================================================================
# Configuration
# =============================================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"

DATA_DIR = Path(__file__).parent / "data"
OBJECT_STORAGE_PATH = DATA_DIR / "object_storage.json"

# Number of groups to split the templates into.
TARGET_GROUPS = int(os.getenv("POPULATE_TARGET_GROUPS", "400"))

# Maximum number of concurrent LLM calls.
MAX_CONCURRENT = int(os.getenv("POPULATE_MAX_CONCURRENT", "20"))

# =============================================================================
# LLM prompt
# =============================================================================
SYSTEM_PROMPT = """
You are an expert object template annotator for a memory system.

You will be given a JSON object where:
- Each key is an object template name.
- Each value is the existing default attributes for that template.

For each template, return an annotation object with exactly three keys:

{
  "categories": ["concrete_category", "broader_category", "...", "object"],
  "functions": ["function1", "function2", "..."],
  "materials": ["material1", "material2", "..."]
}

Requirements:
- Categories must form a chain from concrete to abstract.
- End the category chain with "physical_object" and "object".
- Do not include "person" or "agent" as generic categories unless the object truly is a person.
- Functions should be plausible uses or behaviours of the object.
- Materials must be a list, not a composite string.
  - Example: use ["metal", "plastic"], not "metal_plastic".
  - If only one material, still use a list: ["wood"].
- Preserve the existing object template fields; do not include them in your output.
- Do not include any commentary.

Return a single JSON object mapping each template name to its annotation:

{
  "template_name_1": {
    "categories": ["...", "..."],
    "functions": ["...", "..."],
    "materials": ["...", "..."]
  },
  "template_name_2": {
    "categories": ["...", "..."],
    "functions": ["...", "..."],
    "materials": ["...", "..."]
  }
}

DEMONSTRATION:

Input:
{
  "pasta_food": {
    "shape": "rectangular_prism",
    "state": "dry"
  },
  "knife": {
    "shape": "elongated",
    "blade_edge": "sharp"
  }
}

Expected output:
{
  "pasta_food": {
    "categories": ["pasta", "grain_product", "food", "physical_object", "object"],
    "functions": ["food", "staple", "boiling", "baking"],
    "materials": ["durum_wheat_semolina"]
  },
  "knife": {
    "categories": ["tool", "cutlery", "physical_object", "object"],
    "functions": ["cutting", "slicing", "peeling"],
    "materials": ["stainless_steel", "wood"]
  }
}
"""


async def annotate_group(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    group: Dict[str, Any],
    group_index: int,
    total_groups: int,
) -> Dict[str, Any]:
    """Annotate a group of templates and return template_name -> annotation."""
    async with semaphore:
        print(f"🚀 Processing group {group_index}/{total_groups} ({len(group)} templates)...")

        user_msg = json.dumps(group, indent=2, ensure_ascii=False)

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=6000,
                extra_body={"reasoning_effort": "high"},
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            data = json.loads(content)

            if not isinstance(data, dict):
                print(f"   ❌ Group {group_index}: response is not a JSON object")
                return {}

            valid = {}
            for name, annotation in data.items():
                if name in group:
                    valid[name] = annotation

            print(f"   ✅ Group {group_index}: annotated {len(valid)}/{len(group)} templates")
            return valid

        except Exception as e:
            print(f"   ❌ Group {group_index} failed: {e}")
            return {}


async def main():
    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    if not OBJECT_STORAGE_PATH.exists():
        print(f"❌ Object storage not found: {OBJECT_STORAGE_PATH}")
        return

    with open(OBJECT_STORAGE_PATH, "r", encoding="utf-8") as f:
        object_storage = json.load(f)

    if not isinstance(object_storage, dict):
        print("❌ Object storage is not a JSON object.")
        return

    print(f"Loaded {len(object_storage)} object template entries.")

    # Normalize object storage to flat format: template_name -> defaults dict
    normalized_storage: Dict[str, Any] = {}

    for key, value in object_storage.items():
        if not isinstance(value, dict):
            normalized_storage[key] = {}
            continue

        if "defaults" in value and isinstance(value["defaults"], dict):
            defaults = value["defaults"].copy()
        else:
            defaults = value.copy()

        # Convert singular "material" to "materials" list if needed
        if "materials" not in defaults and "material" in defaults:
            material = defaults["material"]
            if isinstance(material, str):
                defaults["materials"] = [material]
            elif isinstance(material, list):
                defaults["materials"] = material
            # keep original material field too for compatibility

        normalized_storage[key] = defaults

    # Identify templates missing categories, functions, or materials
    to_annotate: Dict[str, Any] = {}

    for name, defaults in normalized_storage.items():
        has_categories = isinstance(defaults.get("categories"), list)
        has_functions = isinstance(defaults.get("functions"), list)
        has_materials = isinstance(defaults.get("materials"), list)

        if has_categories and has_functions and has_materials:
            continue

        to_annotate[name] = defaults

    if not to_annotate:
        print("🎉 All templates already have categories, functions, and materials.")
        return

    print(f"🔧 {len(to_annotate)} templates need categories/functions/materials.")

    # Build groups
    template_names = list(to_annotate.keys())
    group_count = min(TARGET_GROUPS, len(template_names))
    group_size = math.ceil(len(template_names) / group_count)

    groups = []
    for i in range(0, len(template_names), group_size):
        chunk = template_names[i : i + group_size]
        group_dict = {name: to_annotate[name] for name in chunk}
        groups.append(group_dict)

    print(f"📦 Split into {len(groups)} groups. Max concurrency: {MAX_CONCURRENT}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        annotate_group(client, semaphore, group, idx + 1, len(groups))
        for idx, group in enumerate(groups)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    annotated_count = 0
    failed_count = 0

    for idx, result in enumerate(results, 1):
        if isinstance(result, Exception):
            failed_count += len(groups[idx - 1])
            continue

        if not isinstance(result, dict):
            failed_count += len(groups[idx - 1])
            continue

        for name, annotation in result.items():
            if not isinstance(annotation, dict):
                failed_count += 1
                continue

            categories = annotation.get("categories")
            functions = annotation.get("functions")
            materials = annotation.get("materials")

            if isinstance(categories, list):
                normalized_storage[name]["categories"] = categories
            if isinstance(functions, list):
                normalized_storage[name]["functions"] = functions
            if isinstance(materials, list):
                normalized_storage[name]["materials"] = materials

            if (
                isinstance(categories, list)
                and isinstance(functions, list)
                and isinstance(materials, list)
            ):
                annotated_count += 1

    print(f"\n✅ Annotated {annotated_count} templates successfully.")
    if failed_count:
        print(f"⚠️ Failed to annotate {failed_count} templates.")

    # Save in flat format
    with open(OBJECT_STORAGE_PATH, "w", encoding="utf-8") as f:
        json.dump(normalized_storage, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved updated object storage to {OBJECT_STORAGE_PATH}")


if __name__ == "__main__":
    asyncio.run(main())