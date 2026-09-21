#!/usr/bin/env python3
r"""
3d_mgen.py – v0.2 memory generation pipeline.

Two input modes:

    Mode A – fresh generation from a seed prompt.
        Runs the full v0.1-style chain (activity → NL → sketch → enhanced →
        generalized → state-annotated), then the 3D stages on top.

    Mode B – reuse a previously generated v0.1 memory.
        Loads the v0.1 annotated action timeline and object list directly
        and jumps to the 3D stages.

Both modes converge at Stage 2 (parts prior) and share every stage after
that.

Stage 3 is split into four sub-steps internally:

    3a  Entity extraction      [LLM] – produce {protagonist, objects, actions}
    3b  Element resolution     [LLM] – one call per uncached template
    3c  Scene layout           [LLM] – world transforms + environment
    3d  Build assembly         [det] – call the pure helper

Stage list (LLM calls marked [LLM], deterministic steps [det]):

    Mode A only:
        Stage 0a  Activity generation                    [LLM]
        Stage 0b  NL memory generation                   [LLM]
        Stage 1a  Memory sketch                          [LLM]
        Stage 1b  Enhanced sketch                        [LLM]
        Stage 1c  Generalization                         [LLM]
        Stage 1d  State annotation                       [LLM]

    Both modes:
        Stage 2   Parts prior                            [LLM]
        Stage 3a  Entity extraction                      [LLM]
        Stage 3b  Element resolution                     [LLM] × N templates
        Stage 3c  Scene layout                           [LLM]
        Stage 3d  Build assembly                         [det]
        Stage 4   Operations per top-level action        [LLM] × N
        Stage 5   Labeling (incremental)                 [LLM]
        Stage 6   Reconstituted annotation               [LLM]
        Stage 7   Relation maps per frame                [det]
        Stage 8a  Initial JSON encoding                  [LLM]
        Stage 8b  JSON review (narrowed scope)           [LLM]
        Stage 8c  Assembly-JSON consistency check        [det]
        Stage 9a  Persist                                [det]
        Stage 9b  Diff vs v0.1 (Mode B only)             [det]

Retry and failure semantics:

    Every LLM call is retried up to LLM_MAX_RETRIES times on any failure
    (transport error, empty content, or JSON parse error). If all retries
    fail, the stage raises StageFailure. The batch loop catches
    StageFailure, records the failing stage and reason, and continues to
    the next activity. Failed memories are not stored in memories.jsonl
    but appear in records.jsonl with failed: true and a failure reason.

Prompt feeding policy:

    Every LLM call receives the full text of its inputs, not truncated
    excerpts. The stages that pass large inputs (Stage 3c, Stage 4, Stage
    8a) do so in full; if token budgets become a practical problem, the
    fix is a summarisation stage, not truncation of the source material.

All prompt text lives in this file. Every other module in the 3d package
is a pure helper that makes no LLM calls and holds no prompts.
"""

import argparse
import asyncio
import copy
import importlib
import json
import os
import re
import sys
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_assembly = importlib.import_module("3d_assembly")
_elements = importlib.import_module("3d_elements_storage")
_ops = importlib.import_module("3d_scene_operations")
_rmap = importlib.import_module("3d_relation_map")
_render = importlib.import_module("3d_render_assembly")
_builder = importlib.import_module("3d_assembly_builder")
_sim = importlib.import_module("3d_shape_similarity")

Assembly3D = _assembly.Assembly3D
Object3D = _assembly.Object3D
Element3D = _assembly.Element3D
Transform = _assembly.Transform
TimelineKeyframe = _assembly.TimelineKeyframe
make_element = _assembly.make_element
make_transform = _assembly.make_transform
make_assembly_id = _assembly.make_assembly_id

Elements3DStorage = _elements.Elements3DStorage
ElementEntry = _elements.ElementEntry

MoveOp = _ops.MoveOp
RotateOp = _ops.RotateOp
ScaleOp = _ops.ScaleOp
SetTransformOp = _ops.SetTransformOp
ReachArmRotateOp = _ops.ReachArmRotateOp
AttachOp = _ops.AttachOp
DetachOp = _ops.DetachOp
SplitOp = _ops.SplitOp
MergeOp = _ops.MergeOp
AddObjectOp = _ops.AddObjectOp
RemoveObjectOp = _ops.RemoveObjectOp
AddElementOp = _ops.AddElementOp
RemoveElementOp = _ops.RemoveElementOp
ChangePropertiesOp = _ops.ChangePropertiesOp
ContactOp = _ops.ContactOp
ReleaseOp = _ops.ReleaseOp
LabelOp = _ops.LabelOp
UnlabelOp = _ops.UnlabelOp
WaitOp = _ops.WaitOp
EmitEventOp = _ops.EmitEventOp
CompoundOp = _ops.CompoundOp
apply_operation = _ops.apply_operation
op_from_dict = _ops.op_from_dict
op_to_dict = _ops.op_to_dict

extract_relation_map = _rmap.extract_relation_map
RelationMap = _rmap.RelationMap

AssemblyBuilder = _builder.AssemblyBuilder
build_assembly = _builder.build_assembly

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================
MODULE_DIR = Path(__file__).parent
DATA_DIR = MODULE_DIR / "data"
DEMOS_DIR = DATA_DIR / "demos"
STORAGES_DIR = DATA_DIR / "storages"
ASSEMBLIES_DIR = MODULE_DIR / "assemblies"
RENDERS_DIR = ASSEMBLIES_DIR / "renders"
MEMORIES_DIR = MODULE_DIR / "memories"
RELATION_MAPS_DIR = MEMORIES_DIR / "relation_maps"
DIFFS_DIR = MEMORIES_DIR / "diffs"

V01_ROOT_DIR = Path(r"F:\New folder (4)\New folder")
V01_MEMORIES_DIR = V01_ROOT_DIR / "Memories"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"

OUTPUT_MAX_TOKENS = 300000
REASONING_EFFORT = "high"
DEFAULT_TEMPERATURE = 0.4

MAX_OPERATION_RETRIES = 1
MAX_SCENE_RETRIES = 1

LLM_MAX_RETRIES = 3


class StageFailure(RuntimeError):
    """
    Raised when a pipeline stage fails after all LLM retries. The batch
    loop catches this, records the failing stage and reason, and continues
    to the next activity.
    """
    def __init__(self, stage: str, memory_id: str, reason: str = ""):
        self.stage = stage
        self.memory_id = memory_id
        self.reason = reason
        msg = f"stage '{stage}' failed for memory '{memory_id}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


# =============================================================================
# Controlled property vocabularies
# =============================================================================
PROPERTY_VOCABULARIES: Dict[str, List[str]] = {
    "temperature":    ["very_cold", "cold", "cool", "room_temperature", "warm", "hot", "very_hot"],
    "moisture":       ["dry", "slightly_damp", "damp", "wet", "soaking", "dripping"],
    "cleanliness":    ["clean", "slightly_dirty", "dirty", "very_dirty", "filthy"],
    "oxidation":      ["none", "light", "moderate", "heavy", "complete"],
    "surface_state":  ["dry", "damp", "wet", "slick", "oily", "coated", "dusty", "sticky"],
    "physical_state": ["intact", "deformed", "cracked", "torn", "broken", "fragmented", "liquified", "evaporated"],
    "state":          ["open", "closed", "partially_open", "partially_closed", "locked", "unlocked"],
    "texture":        ["smooth", "rough", "granular", "fibrous", "crystalline", "powdery", "slimy", "crisp", "soft"],
    "visibility":     ["visible", "faint", "barely_visible", "not_visible", "hidden"],
}


def _format_vocabularies_for_prompt() -> str:
    lines = ["CONTROLLED PROPERTY VOCABULARIES (use these values when the phenomenon matches):"]
    for prop, values in PROPERTY_VOCABULARIES.items():
        lines.append(f"  {prop}: {' | '.join(values)}")
    lines.append(
        "For any property not listed above, or for a value that does not fit the vocabulary, "
        "prefix the value with 'custom:' (e.g. 'custom:molten'). Custom values are logged "
        "for review."
    )
    return "\n".join(lines)


PROPERTY_VOCABULARIES_BLOCK = _format_vocabularies_for_prompt()


# =============================================================================
# Demo file loading
# =============================================================================

def _load_demo(name: str) -> str:
    path = DEMOS_DIR / name
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Warning: failed to load demo '{name}': {e}")
        return ""


DEMO_NL = _load_demo("3d_nl_examples.txt")
DEMO_ACTION_PLOT = _load_demo("3d_action_plot_demo.txt")
DEMO_DECOMPOSE = _load_demo("3d_decompose_demo.txt")
DEMO_GENERALIZE = _load_demo("3d_generalize_demo.txt")
DEMO_STATE_CHANGE = _load_demo("3d_state_change_demo.txt")
DEMO_PARTS_PRIOR = _load_demo("3d_parts_prior_demo.txt")
DEMO_ENTITY_EXTRACTION = _load_demo("3d_entity_extraction_demo.txt")
DEMO_ELEMENT_DECOMPOSITION = _load_demo("3d_element_decomposition_demo.txt")
DEMO_SCENE = _load_demo("3d_scene_demo.txt")
DEMO_OPERATIONS = _load_demo("3d_operations_demo.txt")
DEMO_LABELING = _load_demo("3d_labeling_demo.txt")
DEMO_RECONSTITUTED = _load_demo("3d_reconstituted_demo.txt")
DEMO_JSON_ENCODE = _load_demo("3d_json_encode_demo.txt")
DEMO_JSON_REVIEW = _load_demo("3d_json_review_demo.txt")


# =============================================================================
# Static system prompts
# =============================================================================

SYS_ACTIVITY_GENERATION = (
    "You are a creative activity generator. Output ONLY valid JSON.\n"
    "Each activity must be reasonably specific, e.g. 'making pizza'.\n"
    "Do NOT propose overly general activities like 'making dinner'.\n"
    "Output ONLY valid JSON in the exact format:\n"
    '{"activities": ["activity1", "activity2", ...]}\n'
)

SYS_NL_GENERATION = (
    "You are a meticulous observer of physical processes. "
    "Write a detailed procedural memory in the style of the examples below. "
    "The memory should be 300-600 words, rich in sensory details, physical "
    "attributes, object states, spatial relationships, and explicit state "
    "changes. Make it as vivid and structured as the examples below while "
    "ensuring high plausibility.\n\n"
    + DEMO_NL
)

SYS_MEMORY_SKETCH = (
    "You are an expert in task and object analysis. Given a procedural memory, "
    "produce a memory sketch that includes:\n\n"
    "1. OBJECT LIST (tiered)\n"
    "   - List all objects present in the scene and involved in the actions.\n"
    "   - For composite objects, show sub-objects indented beneath the parent.\n\n"
    "2. ACTION PLOT (with natural processes in parallel)\n"
    "   - Show the main human actions and their temporal structure.\n"
    "   - Add natural processes as separate actions in parallel lanes where they occur.\n"
    "   - Use indentation for sub-actions and include durations, temporal types "
    "(seq, parallel, cyclical), and start offsets if needed.\n\n"
    + DEMO_ACTION_PLOT
)

SYS_ENHANCED_SKETCH = (
    "You are an expert in decomposition. You will be given both the original "
    "natural language memory and the memory sketch. Enhance the sketch by:\n\n"
    "1. Recursively decomposing every action into more primitive sub-actions.\n"
    "   - If an action can be broken down further, do so until all leaf actions "
    "are atomic or cyclical.\n"
    "   - Keep the same hierarchical format.\n"
    "   - Add durations and temporal types (seq, parallel, cyclical).\n"
    "   - For natural processes, decompose them as well if possible.\n\n"
    "2. Extending the object list:\n"
    "   - Add any new objects that emerge from newly decomposed actions.\n"
    "   - Break down objects further when appropriate.\n"
    "   - Ensure the object list and action plot stay synchronised.\n\n"
    "3. Cross-check against the NL memory:\n"
    "   - If the NL memory describes an object, change, or natural process that "
    "the initial sketch missed, add it here.\n"
    "   - If the NL memory uses more granular terminology than the sketch, use "
    "the memory's granularity.\n"
    "   - The NL memory is the ground truth; the sketch is a first pass.\n\n"
    + DEMO_DECOMPOSE
)

SYS_GENERALIZE = (
    "You are an expert in action generalisation. Using the enhanced action plot, "
    "perform bottom-up generalisation:\n\n"
    "- Start from the lowest-level sub-actions and replace them with the most "
    "general reusable template name that accurately describes the action.\n"
    "- Work upwards level by level.\n"
    "- For complex actions, provide a general template name and an alternative "
    "name that preserves the specific sequence of this memory.\n"
    "- Do not remove any actions or sub-actions; only adjust their names.\n"
    "- Mark complex actions with 'general: [template]' and 'alt: [specific_alt]'.\n\n"
    + DEMO_GENERALIZE
)

SYS_STATE_ANNOTATION = (
    "You are an expert in physical state changes. Annotate the action plot exhaustively.\n\n"
    "1. Add preconditions to actions – conditions that must be true before the "
    "action can start.\n"
    "2. Add state changes that result from each action. For cyclical actions, "
    "separate changes_per_cycle and changes_total. For actions with variable "
    "effects, use conditional_changes.\n"
    "3. Add sensory perception changes for the protagonist, each with a 'source'.\n"
    "4. Add natural processes that are directly linked to attribute changes.\n"
    "5. Use aspects where needed: object.aspect.attribute.\n"
    "6. Capture every observable change in the NL memory.\n"
    "7. Ensure every action (including natural processes) has its preconditions "
    "and changes fully specified.\n\n"
    + DEMO_STATE_CHANGE
)

SYS_PARTS_PRIOR = (
    "You are an expert in material and object composition. You will be given:\n"
    "  1. The natural language memory.\n"
    "  2. The memory sketch (initial object list and action plot).\n"
    "  3. The enhanced sketch (extended object list and decomposed actions).\n"
    "  4. The state-annotated action timeline (with preconditions, changes, "
    "sensory effects, natural process links).\n\n"
    "Produce an exhaustive text list of the expected parts of every object "
    "that appears in the memory.\n\n"
    "For each object, list every part, sub-part, and material composition you "
    "would expect a physical specimen to have. Be exhaustive and granular: it "
    "is better to over-list than to under-list. This list is a prior that "
    "guides downstream 3D geometry generation; it is not a constraint.\n\n"
    "Output as JSON:\n"
    "{\n"
    "  \"objects\": [\n"
    "    {\n"
    "      \"object_id\": \"apple_01\",\n"
    "      \"template_name\": \"apple\",\n"
    "      \"expected_parts\": [\"skin\", \"flesh\", \"core\", \"stem\", \"calyx\", \"seeds\", \"pith\"],\n"
    "      \"expected_materials\": [\"apple_skin\", \"apple_flesh\", \"apple_core\"],\n"
    "      \"notes\": \"...\"\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    + DEMO_PARTS_PRIOR
)

SYS_ENTITY_EXTRACTION = (
    "You are a memory structurer for a 3D memory system. You will be given:\n\n"
    "1. The natural language memory (the ground truth narrative).\n"
    "2. The memory sketch: a tiered object list plus an action plot.\n"
    "3. The enhanced sketch: an extended object list and a recursively "
    "decomposed action plot.\n"
    "4. The parts prior: an exhaustive list of expected parts for each "
    "object.\n"
    "5. The state-annotated action timeline: the same actions with "
    "preconditions, state changes, sensory effects, and natural process "
    "links.\n\n"
    "Your task is to produce a structured JSON representation of the "
    "memory at its STARTING STATE (t=0).\n\n"
    "Produce a single JSON object with three top-level keys:\n\n"
    "{\n"
    "  \"protagonist\": {...},\n"
    "  \"objects\": [...],\n"
    "  \"actions\": [...]\n"
    "}\n\n"
    "PROTAGONIST\n\n"
    "The protagonist is the person whose memory this is. It has:\n"
    "  - template: usually \"person\"\n"
    "  - obj_id: a unique identifier like \"person_01\"\n"
    "  - overrides: any protagonist attributes explicitly mentioned "
    "(e.g. intention, posture, sensory accumulators)\n\n"
    "OBJECTS\n\n"
    "List every physical object that is present at the START of the "
    "memory. Do NOT include objects that are created or introduced by an "
    "action later in the memory. Do include objects that are present from "
    "the start even if they are only used later.\n\n"
    "Each object has:\n"
    "  - obj_id: a unique identifier, e.g. \"apple_01\", \"left_hand_01\".\n"
    "  - template: the object template name.\n"
    "  - parent_object_id: the obj_id of the parent, or null for roots.\n"
    "  - number: \"single\" or \"plural\".\n"
    "  - position: {relation, relative_to} if known from the sketches.\n"
    "  - dimensions: {length, width, height_thickness}, each with value and "
    "unit (\"cm\"). If not explicitly stated, leave empty.\n"
    "  - shape: a short shape descriptor.\n"
    "  - materials: a list of material names.\n"
    "  - overrides: any additional attributes explicitly mentioned.\n\n"
    "For composite objects, list the parent first and then the children "
    "with parent_object_id set.\n\n"
    "ACTIONS\n\n"
    "List the top-level human actions the protagonist performs. Do not "
    "list natural processes, sub-actions, or sensory perceptions as "
    "top-level actions.\n\n"
    "Each action has:\n"
    "  - template: the general action template name.\n"
    "  - instance_id: a unique identifier (e.g. \"action.0\", \"action.1\").\n"
    "  - duration: the total duration in seconds.\n"
    "  - temporal_type: \"sequential\", \"parallel\", \"cyclical\", or \"once\".\n"
    "  - cycle_duration: for cyclical actions, the duration of one cycle.\n"
    "  - repetitions: for cyclical actions, the number of cycles.\n"
    "  - participants: a list of obj_ids of the objects involved.\n"
    "  - preconditions: [{object, attribute, op, value}] if specified.\n"
    "  - changes: [{object, attribute, old, new}] within this action.\n"
    "  - kinematic_trajectory: one of \"linear\", \"straight\", \"curved\", "
    "\"circular\", \"arc\", \"enveloping\", \"helical\", \"spiral\", "
    "\"deceleration\", \"angular\".\n\n"
    "Output only valid JSON. No commentary, no markdown fences.\n\n"
    + DEMO_ENTITY_EXTRACTION
)

SYS_ELEMENT_DECOMPOSITION = (
    "You are a 3D geometry designer. Given an object template's semantic "
    "description, produce a compact decomposition of the object into "
    "primitive elements that approximates its shape.\n\n"
    "Coordinate convention: Z is up, ground at z=0, lengths in cm. Each "
    "element's transform positions the element relative to the object's "
    "local origin (its centroid). Rotations are quaternions [x, y, z, w].\n\n"
    "Available element types and required parameters:\n"
    "  sphere:      radius_cm\n"
    "  ellipsoid:   radius_x_cm, radius_y_cm, radius_z_cm\n"
    "  box:         size_x_cm, size_y_cm, size_z_cm\n"
    "  cylinder:    radius_cm, height_cm\n"
    "  cone:        radius_cm, height_cm\n"
    "  torus:       major_radius_cm, minor_radius_cm\n"
    "  capsule:     radius_cm, height_cm\n"
    "  sheet:       size_x_cm, size_y_cm, thickness_cm\n"
    "  wedge:       size_x_cm, size_y_cm, size_z_cm\n"
    "  prism:       size_x_cm, size_y_cm, num_sides\n"
    "  pyramid:     base_size_x_cm, base_size_y_cm, height_cm\n"
    "  hemisphere:  radius_cm\n"
    "  frustum:     bottom_radius_cm, top_radius_cm, height_cm\n\n"
    "Element transform (optional, identity if omitted):\n"
    "  {\"translate\": [x, y, z], \"rotate\": [x, y, z, w], \"scale\": [sx, sy, sz]}\n\n"
    "Guidelines:\n"
    "- Keep decompositions compact. 1 to 6 elements per object.\n"
    "- Use a single primitive when it captures the shape well.\n"
    "- Compose when needed (an apple is a sphere, a short cylinder for the "
    "stem, and a torus for the calyx).\n"
    "- Do NOT model surface details or internal structure beyond what is "
    "needed for the object's overall geometry.\n"
    "- Do NOT model parts that are separate objects in the memory.\n"
    "- Provide a plausible default material dict, mass in grams, and notes.\n\n"
    "Output format (exact):\n"
    "{\n"
    "  \"elements\": [\n"
    "    {\"element_type\": \"...\", \"parameters\": {...}, \"transform\": {...}},\n"
    "    ...\n"
    "  ],\n"
    "  \"default_material\": {\"color\": \"...\", \"roughness\": 0.0-1.0, "
    "\"metalness\": 0.0-1.0},\n"
    "  \"default_mass_grams\": <number>,\n"
    "  \"notes\": \"...\"\n"
    "}\n\n"
    "Only the JSON. No commentary, no markdown fences.\n\n"
    + DEMO_ELEMENT_DECOMPOSITION
)

SYS_INITIAL_3D_SCENE = (
    "You are a 3D scene arranger. You will be given:\n\n"
    "1. The natural language memory (ground truth narrative).\n"
    "2. A structured list of objects present at t=0, with their "
    "templates, hierarchy, and initial position hints.\n"
    "3. The parts prior, listing every expected part per object.\n"
    "4. A summary of the geometry that will be used for each template "
    "(element types and bounding-box sizes).\n\n"
    "Your task: produce WORLD-space transforms for every object at the "
    "start of the memory, and an environment description.\n\n"
    "CRITICAL: All transforms you produce are in WORLD coordinates. Do "
    "not subtract the parent's transform. The pipeline handles "
    "parent-child conversion internally.\n\n"
    "Coordinate system: Z is up. Ground at z=0. Lengths in cm. "
    "Common reference heights: floor=0cm, table=75cm, counter=80cm, "
    "eye level=160cm. Counter top at z=80. When two objects are in "
    "contact (\"on\"), place them so their bounding volumes just touch. "
    "When something is \"in\" another object, place it inside. When a "
    "hand is \"beside the body\", place it at roughly x=±25, z=100.\n\n"
    "PLACEMENT CONSTRAINT\n\n"
    "Position every object the protagonist must manipulate within the "
    "reach envelope of the appropriate arm. The left shoulder is at "
    "(-25, -50, 155) and the right shoulder is at (25, -50, 155). Each "
    "arm is 70 cm long and the hand adds 20 cm, giving a 90 cm "
    "shoulder-to-fingertip reach. Place manipulable objects within 60 cm "
    "of at least one shoulder so the arm can reach them by rotation alone.\n\n"
    "Output JSON:\n"
    "{\n"
    "  \"object_transforms\": {\n"
    "    \"<obj_id>\": {\"translate\": [x,y,z], \"rotate\": [x,y,z,w], "
    "\"scale\": [sx,sy,sz]}\n"
    "  },\n"
    "  \"environment\": {\n"
    "    \"ground\": {\"present\": true, \"material\": \"...\"},\n"
    "    \"counter\": {\"present\": true, \"top_z_cm\": 80.0, ...},\n"
    "    ...\n"
    "  }\n"
    "}\n\n"
    "In the demonstration below, the demo shows a full assembly including "
    "element decompositions, for context. For this stage, only the "
    "object_transforms and environment sections are needed. Elements are "
    "already resolved by the pipeline.\n\n"
    + DEMO_SCENE
)

SYS_OPERATIONS_FOR_ACTION = (
    "You are a 3D animation planner. You will be given:\n\n"
    "1. The full natural language memory.\n"
    "2. The full state-annotated action timeline for context.\n"
    "3. The parts prior.\n"
    "4. The current state of the 3D assembly (all objects with world "
    "positions and materials).\n"
    "5. Every operation group emitted so far, in order.\n"
    "6. The single top-level action that you must encode now.\n\n"
    "Produce the operations that carry out that action.\n\n"
    "Every change in the NL memory – human action, machine action, or natural "
    "process – must be represented as one or more operations.\n\n"
    "OPERATION VOCABULARY\n\n"
    "Geometric operations:\n"
    "  add_object        – introduce a new object\n"
    "  remove_object     – remove an object entirely\n"
    "  add_element       – add a primitive element to an existing object\n"
    "  remove_element    – remove a primitive element\n"
    "  move              – additive translation (object_id, delta)\n"
    "  rotate            – multiplicative rotation (object_id, quaternion)\n"
    "  scale             – multiplicative scale (object_id, factor)\n"
    "  set_transform     – set full world TRS in one shot\n"
    "  reach_arm_rotate  – rotate an articulated limb about its joint so the\n"
    "                      limb points at a world-space target. Use for any\n"
    "                      reach, lift, withdraw, or retract motion of a limb.\n"
    "  attach            – make A a child of B in the scene-graph hierarchy\n"
    "  detach            – make A independent of its current parent\n"
    "  split             – binary split of one object into two along a plane\n"
    "  merge             – combine N objects into one\n\n"
    "Property and interaction operations:\n"
    "  change_properties – merge or replace an object's property dict\n"
    "  contact           – record a physical contact interaction\n"
    "  release           – terminate a prior contact\n"
    "  label             – assign a semantic name to an object or element\n"
    "  unlabel           – remove a semantic name\n"
    "  wait              – advance time without geometry change\n"
    "  emit_event        – record a perceptual or semantic marker\n"
    "  compound          – group several operations under one action\n\n"
    "ARTICULATED LIMBS\n\n"
    "Arms, hands, thumbs, neck, and head are articulated limbs. Each limb's "
    "local origin is at its joint. To make a limb reach toward a world "
    "position, use `reach_arm_rotate` with the target's world coordinates. "
    "Do NOT use `move`, `scale`, or `set_transform` on a limb; those would "
    "translate the joint off the body. To rotate a limb in place without a "
    "specific target, use `rotate`.\n\n"
    "CONTACT AND RELEASE\n\n"
    "`contact` and `release` model physical interaction. They are distinct "
    "from `attach` and `detach`, which control the scene-graph hierarchy.\n\n"
    "  attach  / detach – \"if the parent moves, does the child follow?\"\n"
    "  contact / release – \"is A touching B, and how?\"\n\n"
    "A ContactOp has these fields:\n"
    "  object_a         – actor or subject of the contact\n"
    "  object_b         – object being contacted\n"
    "  contact_type     – touch, grip, hold, push, pull, lean, support, rest_on\n"
    "  contact_points   – optional list of {point_world, normal_world}\n"
    "  force            – placeholder dict; defaults {\"magnitude_n\": 0.0, "
    "\"direction_world\": [0,0,-1]}\n"
    "  friction         – placeholder dict; defaults {\"coefficient\": 0.5, "
    "\"regime\": \"static\"}\n"
    "  starts_contact   – True for a new contact\n"
    "  ends_contact     – True for ending an existing contact\n"
    "  participants     – semantic roles\n"
    "  notes            – optional string\n\n"
    "CONTACT-TYPE-TO-ATTACHMENT CONVENTION\n\n"
    "  contact type   implies attachment?\n"
    "  ------------   -------------------\n"
    "  touch          no\n"
    "  grip           yes\n"
    "  hold           yes\n"
    "  push           optional\n"
    "  pull           optional\n"
    "  lean           no\n"
    "  support        no\n"
    "  rest_on        no\n\n"
    "When you emit a ContactOp with type `grip` or `hold`, you MUST also emit "
    "the matching `attach` in the same OperationGroup, unless object_b is "
    "already attached to object_a (or vice versa) in the current assembly.\n\n"
    "When you emit a ReleaseOp that ends a `grip` or `hold`, you MUST also "
    "emit the matching `detach` in the same OperationGroup.\n\n"
    "For push, pull, lean, support, rest_on, and touch, do NOT emit an attach "
    "or detach.\n\n"
    "PHYSICS PLACEHOLDERS\n\n"
    "The `force` and `friction` sub-dicts are placeholders. Values are "
    "recorded but not consumed. Do not invent units or fields beyond those "
    "listed.\n\n"
    "PROPERTY VOCABULARIES\n\n"
    + PROPERTY_VOCABULARIES_BLOCK + "\n\n"
    "OUTPUT FORMAT\n\n"
    "Output JSON:\n"
    "{\n"
    "  \"operations\": [ {...}, ... ],\n"
    "  \"notes\": \"...\"\n"
    "}\n\n"
    + DEMO_OPERATIONS
)

SYS_LABELING = (
    "You are a labeling engine for 3D scene elements. Given a set of new "
    "objects or elements that were just created during a scene operation, "
    "assign a semantic label to each element based on its geometry and the "
    "context.\n\n"
    "Rules:\n"
    "- Every element receives exactly one label.\n"
    "- Labels describe the role the element plays in the object, not just "
    "its geometric type.\n"
    "- The input includes the object's material and metadata (e.g. "
    "skin_present, state, moisture, clips). Use them: an apple sphere with "
    "skin_present: false is the peeled body, not the skin; a split object "
    "with clips is a cut face, not the source surface.\n"
    "- The input includes the object's parent and world position. Use them "
    "for relational labels.\n"
    "- For split or clipped objects, the elements are inherited from the "
    "source but the visible surface includes cut faces. Labels reflect "
    "roles in the split object.\n"
    "- Object groups with multiple similar elements should be labeled with "
    "the same role name, indexed (seed_1, seed_2, ...) when distinguishable.\n\n"
    "Output JSON:\n"
    "{\n"
    "  \"labels\": [\n"
    "    {\"object_id\": \"apple_01\", \"element_index\": 0, \"label\": \"skin\"},\n"
    "    ...\n"
    "  ],\n"
    "  \"notes\": \"...\"\n"
    "}\n\n"
    + DEMO_LABELING
)

SYS_RECONSTITUTED_ANNOTATION = (
    "You are producing the FINAL, GROUND-TRUTH annotation for a memory. The 3D "
    "assembly and its operations are the truth; the original annotation and "
    "parts prior are what was claimed before the 3D model was built.\n\n"
    "You will be given:\n"
    "1. The original annotated action timeline and object list.\n"
    "2. The natural language memory.\n"
    "3. The parts prior.\n"
    "4. The final assembly summary.\n"
    "5. The operation groups.\n\n"
    "Produce two artifacts:\n"
    "1. The reconstituted expanded object list: every object and sub-object "
    "that actually appears in the 3D assembly, with its semantic label, "
    "template, categories, and attributes. Parts that the annotation claimed "
    "but the geometry does not have are either dropped or flagged as unresolved.\n"
    "2. The reconstituted annotated action timeline: every action mapped to the "
    "operations that implement it, with the actual state changes derived from "
    "the assembly deltas, not just what the text claimed. Actions that have no "
    "corresponding operations are dropped.\n\n"
    "Preserve action instance IDs from the input annotation wherever the action "
    "still exists in the 3D model.\n\n"
    "Output JSON:\n"
    "{\n"
    "  \"reconstituted_object_list\": {...},\n"
    "  \"reconstituted_action_timeline\": {...},\n"
    "  \"dropped_objects\": [...],\n"
    "  \"dropped_actions\": [...],\n"
    "  \"notes\": \"...\"\n"
    "}\n\n"
    + DEMO_RECONSTITUTED
)

SYS_JSON_ENCODE = (
    "You are a memory-to-JSON encoder for the v0.2 (3D-grounded) memory system. "
    "You will be given:\n\n"
    "1. The reconstituted annotation (object list and action timeline).\n"
    "2. The object templates that were used (with defaults, categories, "
    "functions, materials, aliases, composition).\n"
    "3. The action templates that were used (with default duration, action "
    "category, kinematic trajectory, sub-action templates, alternatives, "
    "preconditions).\n"
    "4. The full operation groups, in order.\n"
    "5. The final assembly summary.\n"
    "6. The relation map summary.\n"
    "7. The natural language memory.\n\n"
    "Convert these into a final JSON memory encoding.\n\n"
    "The encoding must include:\n"
    "- activity, memory_id, 3d_assembly_id\n"
    "- object_templates: every template used, with defaults, categories, "
    "functions, materials, aliases, composition_templates\n"
    "- action_templates: every template used, with default_duration, "
    "action_category, kinematic_trajectory, sub_action_templates, "
    "alternatives, preconditions\n"
    "- memory: protagonist (with sensory accumulators), objects (as instances "
    "with overrides only), actions (nested with sub-actions where applicable)\n"
    "- goal_state: derived from the final frame\n"
    "- relation_maps: references to the per-frame map files, plus a summary\n\n"
    "Cyclical actions should be compressed: one entry with duration, "
    "cycle_duration, repetitions, changes_per_cycle, changes_total. Do not "
    "expand cycles into individual entries.\n\n"
    "Output ONLY valid JSON.\n\n"
    + DEMO_JSON_ENCODE
)

SYS_JSON_REVIEW = (
    "You are a JSON review expert for the v0.2 memory system. You will be "
    "given:\n\n"
    "1. The original NL memory.\n"
    "2. The reconstituted annotation.\n"
    "3. The initial JSON encoding.\n"
    "4. The 3D assembly summary.\n\n"
    "Perform the following NARROW checks and corrections, then output the "
    "entire edited encoding:\n\n"
    "1. Remove spurious references: delete any content that is not in the NL "
    "memory and not in the 3D assembly.\n"
    "2. Sensory sources: every sensory perception entry must have a valid "
    "'source' field referring to an object or aspect present in the encoding.\n"
    "3. Preconditions: ensure preconditions are separated from state changes "
    "and placed on the correct action.\n"
    "4. Fine non-geometric details: check the NL memory for details (surface "
    "moisture, friction, colour transitions, sensory texture, smell, sound) "
    "that are in NEITHER the 3D assembly NOR the initial JSON encoding. Add "
    "them to the JSON. Do NOT add geometric details.\n"
    "5. Pass-through: if the encoding already satisfies all checks, return it "
    "unchanged.\n\n"
    "Output ONLY the final JSON object. No commentary.\n\n"
    + DEMO_JSON_REVIEW
)


# =============================================================================
# OperationGroup
# =============================================================================

@dataclass
class OperationGroup:
    action_instance_id: str
    action_template: str
    operations: List[Any] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_instance_id": self.action_instance_id,
            "action_template": self.action_template,
            "operations": [op_to_dict(o) for o in self.operations],
            "notes": self.notes,
        }


# =============================================================================
# Token accounting
# =============================================================================

class TokenAccountant:
    def __init__(self):
        self.per_memory: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {
                "prompt_cache_hit": 0,
                "prompt_cache_miss": 0,
                "prompt_total": 0,
                "output": 0,
                "total": 0,
                "calls": 0,
            })
        )

    def record(self, memory_id: str, stage: str, usage: Dict[str, Any]):
        if not usage:
            return
        bucket = self.per_memory[memory_id][stage]
        bucket["calls"] += 1
        hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
        miss = int(usage.get("prompt_cache_miss_tokens", 0) or 0)
        pt = int(usage.get("prompt_tokens", 0) or 0)
        ot = int(usage.get("completion_tokens", 0) or 0)
        tt = int(usage.get("total_tokens", 0) or 0)
        if hit == 0 and miss == 0:
            miss = pt
        bucket["prompt_cache_hit"] += hit
        bucket["prompt_cache_miss"] += miss
        bucket["prompt_total"] += pt if pt else (hit + miss)
        bucket["output"] += ot
        bucket["total"] += tt if tt else ((hit + miss) + ot)

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        grand = {"prompt_cache_hit": 0, "prompt_cache_miss": 0,
                 "prompt_total": 0, "output": 0, "total": 0, "calls": 0}
        for mid, stages in self.per_memory.items():
            mem_total = {"prompt_cache_hit": 0, "prompt_cache_miss": 0,
                         "prompt_total": 0, "output": 0, "total": 0, "calls": 0}
            stage_summary = {}
            for stage, counts in stages.items():
                stage_summary[stage] = dict(counts)
                for k in mem_total:
                    mem_total[k] += counts[k]
            out[mid] = {"stages": stage_summary, "totals": mem_total}
            for k in grand:
                grand[k] += mem_total[k]
        out["__grand_total__"] = grand
        return out


# =============================================================================
# LLM caller
# =============================================================================

class LLMCaller:
    """
    Thin wrapper around AsyncOpenAI that:
      - retries each call up to LLM_MAX_RETRIES times on any failure
        (transport error, empty content, or JSON parse error),
      - uses the system/user message format for prompt caching,
      - records token usage into the accountant,
      - strips markdown fences and parses JSON when requested.
    """

    def __init__(self, client, model: str = MODEL_NAME,
                 max_tokens: int = OUTPUT_MAX_TOKENS,
                 temperature: float = DEFAULT_TEMPERATURE,
                 accountant: Optional[TokenAccountant] = None,
                 memory_id: Optional[str] = None):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.accountant = accountant
        self.memory_id = memory_id or "unknown"

    async def _call_once(self, stage: str, system_msg: str, user_msg: str) -> Optional[str]:
        """Single attempt. Returns None on any failure. Records token usage."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body={"reasoning_effort": REASONING_EFFORT},
            )
            content = response.choices[0].message.content
            if content is None:
                print(f"    [{stage}] empty content")
                return None
            content = content.strip()
            if self.accountant is not None:
                usage = response.usage
                u = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage or {})
                self.accountant.record(self.memory_id, stage, u)
            return content
        except Exception as e:
            print(f"    [{stage}] transport error: {e}")
            return None

    async def call_text(self, stage: str, system_msg: str, user_msg: str) -> Optional[str]:
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            content = await self._call_once(
                f"{stage}[{attempt}/{LLM_MAX_RETRIES}]", system_msg, user_msg
            )
            if content is not None:
                return content
        print(f"    [{stage}] all {LLM_MAX_RETRIES} attempts failed")
        return None

    async def call_json(self, stage: str, system_msg: str, user_msg: str) -> Optional[Dict[str, Any]]:
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            content = await self._call_once(
                f"{stage}[{attempt}/{LLM_MAX_RETRIES}]", system_msg, user_msg
            )
            if content is None:
                continue
            cleaned = content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                print(f"    [{stage}] attempt {attempt}/{LLM_MAX_RETRIES} unparseable JSON: {e}")
        print(f"    [{stage}] all {LLM_MAX_RETRIES} attempts failed")
        return None


# =============================================================================
# Pipeline
# =============================================================================

class Pipeline3DMGen:

    def __init__(self, client, storage: Elements3DStorage):
        self.client = client
        self.storage = storage
        self.accountant = TokenAccountant()

    # ------------------------------------------------------------------
    # Public entries
    # ------------------------------------------------------------------

    async def generate_from_scratch(self, seed_prompt: str,
                                    num_activities: int = 1) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        activities = await self._stage_0a_activities(seed_prompt, num_activities, memory_id="batch")
        for i, activity in enumerate(activities):
            memory_id = f"mem3d_{int(time.time()):010d}_{i:02d}"
            try:
                record = await self._run_mode_a(memory_id, activity)
                records.append(record)
            except StageFailure as e:
                print(f"    Memory '{activity}' failed at stage '{e.stage}': {e.reason}")
                records.append({
                    "memory_id": memory_id,
                    "activity": activity,
                    "mode": "A",
                    "failed": True,
                    "failed_stage": e.stage,
                    "failure_reason": e.reason,
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                print(f"    Memory '{activity}' failed with unexpected error: {e}")
                traceback.print_exc()
                records.append({
                    "memory_id": memory_id,
                    "activity": activity,
                    "mode": "A",
                    "failed": True,
                    "failed_stage": "unknown",
                    "failure_reason": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
        return records

    async def generate_from_v01(self, memory_id: str,
                                v01_memory_json: Dict[str, Any]) -> Dict[str, Any]:
        return await self._run_mode_b(memory_id, v01_memory_json)

    # ------------------------------------------------------------------
    # Mode A orchestration
    # ------------------------------------------------------------------

    async def _run_mode_a(self, memory_id: str, activity: str) -> Dict[str, Any]:
        print(f"\n  [Mode A] {memory_id}: {activity}")
        caller = LLMCaller(self.client, accountant=self.accountant, memory_id=memory_id)
        record: Dict[str, Any] = {
            "memory_id": memory_id,
            "activity": activity,
            "mode": "A",
            "timestamp": datetime.now().isoformat(),
        }

        nl_text = await self._stage_0b_nl(caller, activity)
        if not nl_text:
            raise StageFailure("stage_0b_nl", memory_id, "NL generation failed")
        record["nl_text"] = nl_text

        sketch = await self._stage_1a_sketch(caller, nl_text)
        if not sketch:
            raise StageFailure("stage_1a_sketch", memory_id, "memory sketch failed")
        record["memory_sketch"] = sketch

        enhanced = await self._stage_1b_enhance(caller, nl_text, sketch)
        if enhanced is None:
            raise StageFailure("stage_1b_enhance", memory_id, "enhanced sketch failed")
        record["enhanced_sketch"] = enhanced

        generalized = await self._stage_1c_generalize(caller, enhanced)
        if generalized is None:
            raise StageFailure("stage_1c_generalize", memory_id, "generalization failed")
        record["generalized_sketch"] = generalized

        annotated = await self._stage_1d_state_annotate(caller, generalized, nl_text)
        if annotated is None:
            raise StageFailure("stage_1d_state_annotate", memory_id, "state annotation failed")
        record["state_annotated_sketch"] = annotated

        annotation = {
            "nl_text": nl_text,
            "activity": activity,
            "memory_sketch": record["memory_sketch"],
            "enhanced_sketch": record["enhanced_sketch"],
            "generalized_sketch": record["generalized_sketch"],
            "state_annotated_sketch": record["state_annotated_sketch"],
            "source": "mode_a",
        }
        record.update(await self._run_3d_stages(memory_id, annotation))
        return record

    # ------------------------------------------------------------------
    # Mode B orchestration
    # ------------------------------------------------------------------

    async def _run_mode_b(self, memory_id: str,
                          v01_memory_json: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n  [Mode B] {memory_id}")
        caller = LLMCaller(self.client, accountant=self.accountant, memory_id=memory_id)

        nl_text = self._extract_v01_nl(v01_memory_json)
        annotation_text = self._extract_v01_annotation(v01_memory_json)

        record: Dict[str, Any] = {
            "memory_id": memory_id,
            "activity": v01_memory_json.get("activity", "unknown"),
            "mode": "B",
            "timestamp": datetime.now().isoformat(),
            "nl_text": nl_text,
            "v01_annotation": annotation_text,
            "v01_memory_json": v01_memory_json,
        }

        annotation = {
            "nl_text": nl_text,
            "activity": v01_memory_json.get("activity", "unknown"),
            "memory_sketch": v01_memory_json.get("memory_sketch", ""),
            "enhanced_sketch": v01_memory_json.get("enhanced_sketch", ""),
            "generalized_sketch": v01_memory_json.get("generalized_sketch", ""),
            "state_annotated_sketch": annotation_text,
            "source": "mode_b",
            "v01_memory_json": v01_memory_json,
        }
        record.update(await self._run_3d_stages(memory_id, annotation))

        record["diff_vs_v01"] = self._compute_diff(v01_memory_json, record)
        return record

    # ------------------------------------------------------------------
    # Shared 3D stages
    # ------------------------------------------------------------------

    async def _run_3d_stages(self, memory_id: str,
                             annotation: Dict[str, Any]) -> Dict[str, Any]:
        caller = LLMCaller(self.client, accountant=self.accountant, memory_id=memory_id)
        nl_text = annotation["nl_text"]
        memory_sketch = annotation.get("memory_sketch", "")
        enhanced_sketch = annotation.get("enhanced_sketch", "")
        generalized_sketch = annotation.get("generalized_sketch", "")
        state_annotated_sketch = annotation.get(
            "state_annotated_sketch",
            annotation.get("annotation_text", ""),
        )

        # ---- Stage 2: parts prior ----
        parts_prior = await self._stage_2_parts_prior(
            caller,
            nl_text=nl_text,
            memory_sketch=memory_sketch,
            enhanced_sketch=enhanced_sketch,
            state_annotated_sketch=state_annotated_sketch,
        )
        if parts_prior is None:
            raise StageFailure("stage_2_parts_prior", memory_id, "LLM returned no parts prior")

        # ---- Stage 3: initial 3D scene (with entity extraction) ----
        assembly, extracted_actions = await self._stage_3_initial_scene(
            caller, memory_id, nl_text, annotation, parts_prior
        )

        # ---- Stage 4: operations per top-level action ----
        operation_groups = await self._stage_4_operations_per_action(
            caller,
            assembly,
            extracted_actions,
            nl_text,
            state_annotated_sketch,
            parts_prior,
            memory_id,
        )

        per_frame_assemblies = self._apply_operation_groups(assembly, operation_groups)

        # Save each per-frame assembly to disk so it can be rendered
        # later without re-applying operations. Paths are relative to
        # this module's location.
        frames_dir = ASSEMBLIES_DIR / "frames" / memory_id
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(per_frame_assemblies):
            frame.save(frames_dir / f"frame_{i:04d}.3dassembly.json")

        # ---- Stage 5: labeling ----
        await self._stage_5_labeling(caller, per_frame_assemblies, nl_text, memory_id)

        # ---- Stage 6: reconstituted annotation ----
        reconstituted = await self._stage_6_reconstitute(
            caller,
            per_frame_assemblies,
            operation_groups,
            state_annotated_sketch,
            parts_prior,
            nl_text,
        )
        if reconstituted is None:
            raise StageFailure("stage_6_reconstitute", memory_id, "LLM returned no reconstituted annotation")

        # ---- Stage 7: relation maps ----
        relation_maps = self._stage_7_relation_maps(memory_id, per_frame_assemblies)

        # ---- Stage 8a: initial JSON ----
        initial_json = await self._stage_8a_json_encode(
            caller,
            reconstituted,
            per_frame_assemblies,
            relation_maps,
            operation_groups,
            nl_text,
        )
        if initial_json is None:
            raise StageFailure("stage_8a_json_encode", memory_id, "LLM returned no initial JSON")

        # ---- Stage 8b: JSON review ----
        reviewed_json = await self._stage_8b_json_review(
            caller,
            nl_text,
            initial_json,
            per_frame_assemblies[-1] if per_frame_assemblies else None,
            reconstituted,
        )
        if reviewed_json is None:
            raise StageFailure("stage_8b_json_review", memory_id, "LLM returned no reviewed JSON")
        final_json = reviewed_json

        # ---- Stage 8c: consistency check ----
        consistency_report = self._stage_8c_consistency_check(
            final_json, per_frame_assemblies
        )

        final_json["memory_id"] = memory_id
        final_json["activity"] = annotation.get("activity", final_json.get("activity", "unknown"))
        final_json["3d_assembly_id"] = make_assembly_id(memory_id)
        final_json.setdefault("objects", [])
        final_json.setdefault("actions", [])

        final_assembly = per_frame_assemblies[-1] if per_frame_assemblies else assembly
        assembly_path = final_assembly.save(
            ASSEMBLIES_DIR / f"{memory_id}.3dassembly.json"
        )

        render_path = None
        try:
            RENDERS_DIR.mkdir(parents=True, exist_ok=True)
            render_path = RENDERS_DIR / f"{memory_id}_solid.png"
            _render.render_assembly(
                final_assembly,
                output_path=render_path,
                backend="software",
                mode="solid",
            )
        except Exception as e:
            print(f"    Render failed: {e}")

        return {
            "parts_prior": parts_prior,
            "initial_assembly": assembly.to_dict(),
            "operation_groups": [g.to_dict() for g in operation_groups],
            "frame_count": len(per_frame_assemblies),
            "reconstituted_annotation": reconstituted,
            "relation_map_count": len(relation_maps),
            "initial_json": initial_json,
            "reviewed_json": reviewed_json,
            "final_json": final_json,
            "assembly_path": str(assembly_path),
            "render_path": str(render_path) if render_path else None,
            "consistency_report": consistency_report,
        }

    # ------------------------------------------------------------------
    # Stage 0a
    # ------------------------------------------------------------------

    async def _stage_0a_activities(self, seed_prompt: str, count: int,
                                   memory_id: str) -> List[str]:
        caller = LLMCaller(self.client, accountant=self.accountant, memory_id=memory_id)
        user_msg = f"{seed_prompt}\nGenerate {count} specific activities."
        data = await caller.call_json("stage_0a_activities", SYS_ACTIVITY_GENERATION, user_msg)
        if not isinstance(data, dict):
            return []
        acts = data.get("activities", [])
        return [str(a) for a in acts if isinstance(a, str)] if isinstance(acts, list) else []

    # ------------------------------------------------------------------
    # Stage 0b
    # ------------------------------------------------------------------

    async def _stage_0b_nl(self, caller: LLMCaller, activity: str) -> Optional[str]:
        user_msg = f"Write a memory of: {activity}\n"
        return await caller.call_text("stage_0b_nl", SYS_NL_GENERATION, user_msg)

    # ------------------------------------------------------------------
    # Stage 1a
    # ------------------------------------------------------------------

    async def _stage_1a_sketch(self, caller: LLMCaller, nl_text: str) -> Optional[str]:
        return await caller.call_text("stage_1a_sketch", SYS_MEMORY_SKETCH, nl_text)

    # ------------------------------------------------------------------
    # Stage 1b (with NL text)
    # ------------------------------------------------------------------

    async def _stage_1b_enhance(self, caller: LLMCaller,
                                 nl_text: str, sketch: str) -> Optional[str]:
        user_msg = (
            f"Original NL memory:\n{nl_text}\n\n"
            f"Memory sketch:\n{sketch}\n\n"
            "Enhance the sketch as described in the system prompt."
        )
        return await caller.call_text("stage_1b_enhance", SYS_ENHANCED_SKETCH, user_msg)

    # ------------------------------------------------------------------
    # Stage 1c
    # ------------------------------------------------------------------

    async def _stage_1c_generalize(self, caller: LLMCaller, sketch: str) -> Optional[str]:
        return await caller.call_text("stage_1c_generalize", SYS_GENERALIZE, sketch)

    # ------------------------------------------------------------------
    # Stage 1d
    # ------------------------------------------------------------------

    async def _stage_1d_state_annotate(self, caller: LLMCaller,
                                        sketch: str, nl_text: str) -> Optional[str]:
        user_msg = (
            f"Original NL memory:\n{nl_text}\n\n"
            f"Generalized action plot:\n{sketch}\n\n"
            "Annotate the action plot exhaustively."
        )
        return await caller.call_text("stage_1d_state_annotate", SYS_STATE_ANNOTATION, user_msg)

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    async def _stage_2_parts_prior(
        self,
        caller: LLMCaller,
        nl_text: str,
        memory_sketch: str,
        enhanced_sketch: str,
        state_annotated_sketch: str,
    ) -> Optional[Dict[str, Any]]:
        user_msg = (
            f"Natural language memory:\n{nl_text}\n\n"
            f"Memory sketch:\n{memory_sketch}\n\n"
            f"Enhanced sketch:\n{enhanced_sketch}\n\n"
            f"State-annotated action timeline:\n{state_annotated_sketch}\n\n"
            "Produce an exhaustive parts prior for every object."
        )
        return await caller.call_json("stage_2_parts_prior", SYS_PARTS_PRIOR, user_msg)

    # ------------------------------------------------------------------
    # Stage 3: initial 3D scene (orchestrator)
    # ------------------------------------------------------------------

    async def _stage_3_initial_scene(
        self,
        caller: LLMCaller,
        memory_id: str,
        nl_text: str,
        annotation: Dict[str, Any],
        parts_prior: Dict[str, Any],
    ) -> Tuple[Assembly3D, List[Dict[str, Any]]]:
        """
        Build the initial 3D scene. Raises StageFailure on any sub-stage
        failure. Returns (assembly, extracted_actions).
        """
        # ---- 3a: entity extraction ----
        extracted = await self._stage_3a_extract_entities(
            caller,
            nl_text=nl_text,
            memory_sketch=annotation.get("memory_sketch", ""),
            enhanced_sketch=annotation.get("enhanced_sketch", ""),
            state_annotated_sketch=annotation.get("state_annotated_sketch", ""),
            parts_prior=parts_prior,
        )
        if not isinstance(extracted, dict):
            raise StageFailure("stage_3a_extract_entities", memory_id,
                               "LLM returned no structured entities")

        protagonist_spec = extracted.get("protagonist") or {
            "template": "person",
            "obj_id": "person_01",
        }
        object_specs = extracted.get("objects") or []
        actions = extracted.get("actions") or []

        print(f"    Stage 3a extracted: "
              f"protagonist=1, objects={len(object_specs)}, actions={len(actions)}")

        if not object_specs:
            raise StageFailure("stage_3a_extract_entities", memory_id,
                               "no objects extracted")

        # ---- 3b: resolve elements ----
        element_map = await self._stage_3b_resolve_elements(
            caller,
            protagonist_spec=protagonist_spec,
            object_specs=object_specs,
            nl_text=nl_text,
            parts_prior=parts_prior,
            memory_id=memory_id,
        )

        # ---- 3c: scene layout ----
        layout = await self._stage_3c_scene_layout(
            caller,
            memory_id=memory_id,
            nl_text=nl_text,
            protagonist_spec=protagonist_spec,
            object_specs=object_specs,
            element_map=element_map,
            parts_prior=parts_prior,
        )
        if layout is None:
            raise StageFailure("stage_3c_scene_layout", memory_id,
                               "LLM returned no layout")

        # ---- 3d: build assembly ----
        try:
            assembly = build_assembly(
                memory_id=memory_id,
                protagonist_spec=protagonist_spec,
                object_specs=object_specs,
                element_map=element_map,
                scene_layout=layout.get("object_transforms", {}),
                environment=layout.get("environment", {}),
                storage=self.storage,
            )
        except Exception as e:
            raise StageFailure("stage_3d_build_assembly", memory_id,
                               f"assembly build failed: {e}")

        return assembly, actions

    # ------------------------------------------------------------------
    # Stage 3a
    # ------------------------------------------------------------------

    async def _stage_3a_extract_entities(
        self,
        caller: LLMCaller,
        nl_text: str,
        memory_sketch: str,
        enhanced_sketch: str,
        state_annotated_sketch: str,
        parts_prior: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        user_msg = (
            f"Natural language memory:\n{nl_text}\n\n"
            f"Memory sketch:\n{memory_sketch}\n\n"
            f"Enhanced sketch:\n{enhanced_sketch}\n\n"
            f"Parts prior:\n{json.dumps(parts_prior, indent=2, ensure_ascii=False)}\n\n"
            f"State-annotated action timeline:\n{state_annotated_sketch}\n\n"
            "Produce the structured JSON with protagonist, objects present "
            "at t=0, and top-level actions."
        )
        return await caller.call_json(
            "stage_3a_extract_entities",
            SYS_ENTITY_EXTRACTION,
            user_msg,
        )

    # ------------------------------------------------------------------
    # Stage 3b
    # ------------------------------------------------------------------

    async def _stage_3b_resolve_elements(
        self,
        caller: LLMCaller,
        protagonist_spec: Dict[str, Any],
        object_specs: List[Dict[str, Any]],
        nl_text: str,
        parts_prior: Dict[str, Any],
        memory_id: str,
    ) -> Dict[str, List[Element3D]]:
        """
        Walk the entity list, collect unique templates, resolve each to a
        list of Element3D via the storage cache or by calling the LLM.
        Store newly generated decompositions in the storage. Raises
        StageFailure if any template cannot be resolved.
        """
        template_context: Dict[str, Dict[str, Any]] = {}

        protag_template = protagonist_spec.get("template", "person")
        if protag_template:
            template_context.setdefault(protag_template, {
                "template_name": protag_template,
                "shape": "humanoid",
                "materials": ["biological"],
                "functions": ["agent", "observer"],
                "categories": ["person", "physical_object", "object"],
                "dimensions": {},
            })

        for spec in object_specs:
            if not isinstance(spec, dict):
                continue
            tmpl = spec.get("template") or spec.get("template_name")
            if not tmpl:
                continue
            if tmpl in template_context:
                continue
            template_context[tmpl] = {
                "template_name": tmpl,
                "shape": spec.get("shape", ""),
                "materials": spec.get("materials", []),
                "functions": spec.get("functions", []),
                "categories": [],
                "dimensions": spec.get("dimensions", {}),
            }

        print(f"    Stage 3b: {len(template_context)} unique templates to resolve")

        element_map: Dict[str, List[Element3D]] = {}

        for tmpl, context in template_context.items():
            if self.storage.has_geometry(tmpl):
                element_map[tmpl] = self.storage.get_elements(tmpl)
                continue

            print(f"      Generating elements for template: {tmpl}")
            entry = await self._stage_3b_generate_entry(
                caller, tmpl, context, nl_text, parts_prior
            )
            if entry is None:
                raise StageFailure(
                    "stage_3b_elements",
                    memory_id,
                    f"element decomposition failed for template '{tmpl}'",
                )

            ok, msg = self.storage.add_template(entry)
            if not ok:
                print(f"      Warning: could not store '{tmpl}': {msg}")
            element_map[tmpl] = entry.elements

        return element_map

    async def _stage_3b_generate_entry(
        self,
        caller: LLMCaller,
        template_name: str,
        context: Dict[str, Any],
        nl_text: str,
        parts_prior: Dict[str, Any],
    ) -> Optional[ElementEntry]:
        prior_entry = None
        for entry in (parts_prior.get("objects") or []):
            if not isinstance(entry, dict):
                continue
            entry_template = entry.get("template_name") or ""
            entry_obj_id = entry.get("object_id") or ""
            if (template_name == entry_template
                    or template_name in entry_obj_id
                    or entry_template in template_name):
                prior_entry = entry
                break

        sibling_templates = []
        for name in self.storage.list_templates():
            if name == template_name:
                continue
            dims = self.storage.get_default_dimensions(name)
            sibling_templates.append({
                "template_name": name,
                "dimensions": dims,
            })

        user_msg = (
            f"Template name: {template_name}\n\n"
            f"Template context:\n"
            f"{json.dumps(context, indent=2, ensure_ascii=False)}\n\n"
            f"Parts prior entry for this template:\n"
            f"{json.dumps(prior_entry, indent=2, ensure_ascii=False) if prior_entry else '(none)'}\n\n"
            f"Full natural language memory (for geometric grounding):\n"
            f"{nl_text}\n\n"
            f"Sibling templates already in the registry "
            f"(for size consistency):\n"
            f"{json.dumps(sibling_templates, indent=2, ensure_ascii=False)}\n\n"
            "Produce the element decomposition for this template."
        )

        data = await caller.call_json(
            f"stage_3b_elements[{template_name}]",
            SYS_ELEMENT_DECOMPOSITION,
            user_msg,
        )
        if not isinstance(data, dict):
            return None

        raw_elements = data.get("elements", [])
        elements: List[Element3D] = []
        for el in raw_elements:
            if not isinstance(el, dict):
                continue
            try:
                elements.append(Element3D.from_dict(el))
            except Exception:
                continue
        if not elements:
            return None

        entry = ElementEntry(
            template_name=template_name,
            elements=elements,
            default_material=dict(data.get("default_material", {})),
            default_mass_grams=data.get("default_mass_grams"),
            notes=data.get("notes", ""),
        )
        ok, msg = entry.validate()
        if not ok:
            print(f"      Entry for '{template_name}' failed validation: {msg}")
            return None
        return entry

    # ------------------------------------------------------------------
    # Stage 3c
    # ------------------------------------------------------------------

    async def _stage_3c_scene_layout(
        self,
        caller: LLMCaller,
        memory_id: str,
        nl_text: str,
        protagonist_spec: Dict[str, Any],
        object_specs: List[Dict[str, Any]],
        element_map: Dict[str, List[Element3D]],
        parts_prior: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        template_bbox: Dict[str, Dict[str, Any]] = {}
        for tmpl, elements in element_map.items():
            dims = self.storage.get_default_dimensions(tmpl)
            if not dims:
                dims = self._approximate_dimensions(elements)
            template_bbox[tmpl] = dims

        entity_list = {
            "protagonist": protagonist_spec,
            "objects": object_specs,
        }

        user_msg = (
            f"Memory ID: {memory_id}\n\n"
            f"Natural language memory:\n{nl_text}\n\n"
            f"Objects present at t=0:\n"
            f"{json.dumps(entity_list, indent=2, ensure_ascii=False)}\n\n"
            f"Parts prior:\n{json.dumps(parts_prior, indent=2, ensure_ascii=False)}\n\n"
            f"Resolved template bounding boxes (dimensions in cm):\n"
            f"{json.dumps(template_bbox, indent=2, ensure_ascii=False)}\n\n"
            "Produce the world transforms and environment."
        )

        data = await caller.call_json(
            "stage_3c_scene_layout",
            SYS_INITIAL_3D_SCENE,
            user_msg,
        )
        if not isinstance(data, dict):
            return None

        obj_transforms = data.get("object_transforms", {})
        if not isinstance(obj_transforms, dict):
            obj_transforms = {}

        return {
            "object_transforms": obj_transforms,
            "environment": data.get("environment", {}),
        }

    def _approximate_dimensions(self, elements: List[Element3D]) -> Dict[str, Any]:
        if not elements:
            return {}
        best = None
        for el in elements:
            p = el.parameters
            size = 0.0
            if el.element_type == "sphere":
                size = 2.0 * p.get("radius_cm", 0.0)
            elif el.element_type == "box":
                size = max(
                    p.get("size_x_cm", 0.0),
                    p.get("size_y_cm", 0.0),
                    p.get("size_z_cm", 0.0),
                )
            elif el.element_type == "cylinder":
                size = max(2.0 * p.get("radius_cm", 0.0), p.get("height_cm", 0.0))
            elif el.element_type == "capsule":
                size = 2.0 * p.get("radius_cm", 0.0) + p.get("height_cm", 0.0)
            elif el.element_type == "frustum":
                size = max(
                    2.0 * p.get("bottom_radius_cm", 0.0),
                    2.0 * p.get("top_radius_cm", 0.0),
                    p.get("height_cm", 0.0),
                )
            else:
                for v in p.values():
                    if isinstance(v, (int, float)) and v > size:
                        size = float(v)
            if best is None or size > best:
                best = size
        if best is None or best <= 0:
            return {}
        return {
            "length": {"value": best, "unit": "cm"},
            "width": {"value": best, "unit": "cm"},
            "height_thickness": {"value": best, "unit": "cm"},
        }

    # ------------------------------------------------------------------
    # Stage 4
    # ------------------------------------------------------------------

    async def _stage_4_operations_per_action(
        self,
        caller: LLMCaller,
        assembly: Assembly3D,
        extracted_actions: List[Dict[str, Any]],
        nl_text: str,
        state_annotated_sketch: str,
        parts_prior: Dict[str, Any],
        memory_id: str,
    ) -> List[OperationGroup]:
        if not extracted_actions:
            raise StageFailure("stage_4_ops", memory_id,
                               "no top-level actions to encode")

        groups: List[OperationGroup] = []
        current_assembly = assembly

        for action in extracted_actions:
            if not isinstance(action, dict):
                continue
            action_template = action.get("template", "unknown_action")
            action_id = action.get("instance_id") or f"{assembly.memory_id}.action.{len(groups)}"

            user_msg = self._build_operation_user_msg(
                assembly=current_assembly,
                action=action,
                annotation_text=state_annotated_sketch,
                nl_text=nl_text,
                parts_prior=parts_prior,
                previous_groups=groups,
            )

            ops_data = await caller.call_json(
                f"stage_4_ops[{action_template}]",
                SYS_OPERATIONS_FOR_ACTION,
                user_msg,
            )
            if not isinstance(ops_data, dict):
                raise StageFailure(
                    "stage_4_ops", memory_id,
                    f"no operations returned for action '{action_template}'",
                )

            raw_ops = ops_data.get("operations", []) or []
            parsed = []
            for raw in raw_ops:
                try:
                    parsed.append(op_from_dict(raw))
                except Exception as e:
                    raise StageFailure(
                        "stage_4_ops", memory_id,
                        f"operation parse failed for '{action_template}': {e}",
                    )

            applied_ok = False
            temp_assembly = current_assembly
            first_error: Optional[Exception] = None
            try:
                for op in parsed:
                    temp_assembly, _ = apply_operation(temp_assembly, op)
                applied_ok = True
            except Exception as e:
                first_error = e
                print(f"    Stage 4 apply failed for '{action_template}': {e}")

            if not applied_ok:
                print(f"    Stage 4 retrying action '{action_template}' once")
                user_msg_retry = (
                    user_msg + "\n\nPrevious attempt failed validation: "
                    + str(first_error) + "\nProduce corrected operations."
                )
                ops_data = await caller.call_json(
                    f"stage_4_ops_retry[{action_template}]",
                    SYS_OPERATIONS_FOR_ACTION,
                    user_msg_retry,
                )
                if not isinstance(ops_data, dict):
                    raise StageFailure(
                        "stage_4_ops", memory_id,
                        f"retry returned no operations for '{action_template}'",
                    )
                parsed = []
                for raw in ops_data.get("operations", []) or []:
                    try:
                        parsed.append(op_from_dict(raw))
                    except Exception as e:
                        raise StageFailure(
                            "stage_4_ops", memory_id,
                            f"retry parse failed for '{action_template}': {e}",
                        )
                try:
                    temp_assembly = current_assembly
                    for op in parsed:
                        temp_assembly, _ = apply_operation(temp_assembly, op)
                    applied_ok = True
                except Exception as e2:
                    raise StageFailure(
                        "stage_4_ops", memory_id,
                        f"apply failed again for '{action_template}': {e2}",
                    )

            current_assembly = temp_assembly
            groups.append(OperationGroup(
                action_instance_id=action_id,
                action_template=action_template,
                operations=parsed,
                notes=ops_data.get("notes", "") if isinstance(ops_data, dict) else "",
            ))

        return groups

    def _build_operation_user_msg(
        self,
        assembly: Assembly3D,
        action: Dict[str, Any],
        annotation_text: str,
        nl_text: str,
        parts_prior: Dict[str, Any],
        previous_groups: List[OperationGroup],
    ) -> str:
        return (
            f"Full natural language memory:\n{nl_text}\n\n"
            f"Full state-annotated action timeline:\n{annotation_text}\n\n"
            f"Parts prior:\n{json.dumps(parts_prior, indent=2, ensure_ascii=False)}\n\n"
            f"Current assembly (all objects with world positions and materials):\n"
            f"{json.dumps(self._summarise_assembly(assembly), indent=2, ensure_ascii=False)}\n\n"
            f"Operation groups emitted so far (in order):\n"
            f"{json.dumps([g.to_dict() for g in previous_groups], indent=2, ensure_ascii=False)}\n\n"
            f"Top-level action to encode now:\n"
            f"{json.dumps(action, indent=2, ensure_ascii=False)}\n\n"
            "Produce the operations for this action."
        )

    # ------------------------------------------------------------------
    # Apply operation groups -> per-frame assemblies
    # ------------------------------------------------------------------

    def _apply_operation_groups(
        self,
        initial_assembly: Assembly3D,
        groups: List[OperationGroup],
    ) -> List[Assembly3D]:
        # Frame 0 is the initial state, before any operations are applied.
        # Frames 1..N correspond to the state after each operation group.
        frames: List[Assembly3D] = [initial_assembly.copy()]
        current = initial_assembly
        for group in groups:
            for op in group.operations:
                current, _ = apply_operation(current, op)
            frames.append(current.copy())
        return frames

    # ------------------------------------------------------------------
    # Stage 5
    # ------------------------------------------------------------------

    async def _stage_5_labeling(
        self,
        caller: LLMCaller,
        per_frame_assemblies: List[Assembly3D],
        nl_text: str,
        memory_id: str,
    ):
        if not per_frame_assemblies:
            raise StageFailure("stage_5_labeling", memory_id, "no frame assemblies to label")
        initial = per_frame_assemblies[0]
        await self._label_assembly(caller, initial, nl_text,
                                   stage="stage_5_labeling_initial",
                                   memory_id=memory_id)

        for i in range(1, len(per_frame_assemblies)):
            prev = per_frame_assemblies[i - 1]
            curr = per_frame_assemblies[i]
            new_ids = [o.object_id for o in curr.objects
                       if not prev.has_object(o.object_id)]
            if not new_ids:
                continue
            await self._label_assembly(
                caller, curr, nl_text,
                stage=f"stage_5_labeling_frame_{i}",
                focus_ids=new_ids,
                memory_id=memory_id,
            )

    async def _label_assembly(
        self,
        caller: LLMCaller,
        assembly: Assembly3D,
        nl_text: str,
        stage: str,
        focus_ids: Optional[List[str]] = None,
        memory_id: str = "unknown",
    ):
        if focus_ids is not None:
            objects_to_label = [assembly.get_object(oid) for oid in focus_ids]
            objects_to_label = [o for o in objects_to_label if o is not None]
        else:
            objects_to_label = list(assembly.objects)

        world = assembly.get_all_world_matrices()

        summary = []
        for obj in objects_to_label:
            M = world.get(obj.object_id)
            pos = None
            if M is not None:
                pos = [float(M[0, 3]), float(M[1, 3]), float(M[2, 3])]

            summary.append({
                "object_id": obj.object_id,
                "template_name": obj.template_name,
                "elements": [
                    {"element_index": i,
                     "element_type": el.element_type,
                     "parameters": el.parameters}
                    for i, el in enumerate(obj.elements)
                ],
                "material": dict(obj.material or {}),
                "metadata": dict(obj.metadata or {}),
                "parent_object_id": obj.parent_object_id,
                "children": list(obj.children),
                "position_world": pos,
            })

        user_msg = (
            f"Natural language memory (for context):\n{nl_text}\n\n"
            f"Objects to label:\n"
            f"{json.dumps({'objects': summary}, indent=2, ensure_ascii=False)}"
        )

        data = await caller.call_json(stage, SYS_LABELING, user_msg)
        if not isinstance(data, dict):
            raise StageFailure(stage, memory_id, "labeling LLM returned no result")

        for entry in data.get("labels", []):
            oid = entry.get("object_id")
            ei = entry.get("element_index")
            label = entry.get("label")
            if oid is None or ei is None or label is None:
                continue
            obj = assembly.get_object(oid)
            if obj is None or ei < 0 or ei >= len(obj.elements):
                continue
            el = obj.elements[ei]
            if el.material_override is None:
                el.material_override = {}
            el.material_override["label"] = label

    # ------------------------------------------------------------------
    # Stage 6
    # ------------------------------------------------------------------

    async def _stage_6_reconstitute(
        self,
        caller: LLMCaller,
        per_frame_assemblies: List[Assembly3D],
        operation_groups: List[OperationGroup],
        original_annotation: str,
        parts_prior: Dict[str, Any],
        nl_text: str,
    ) -> Optional[Dict[str, Any]]:
        final_assembly = per_frame_assemblies[-1] if per_frame_assemblies else None
        if final_assembly is None:
            return None
        assembly_summary = self._summarise_assembly(final_assembly)
        ops_summary = [g.to_dict() for g in operation_groups]

        user_msg = (
            f"Original annotated action timeline and object list:\n"
            f"{original_annotation}\n\n"
            f"Natural language memory:\n{nl_text}\n\n"
            f"Parts prior:\n{json.dumps(parts_prior, indent=2, ensure_ascii=False)}\n\n"
            f"Final assembly summary:\n{json.dumps(assembly_summary, indent=2, ensure_ascii=False)}\n\n"
            f"Operation groups:\n{json.dumps(ops_summary, indent=2, ensure_ascii=False)}\n\n"
            "Produce the reconstituted annotation."
        )
        return await caller.call_json("stage_6_reconstitute", SYS_RECONSTITUTED_ANNOTATION, user_msg)

    # ------------------------------------------------------------------
    # Stage 7
    # ------------------------------------------------------------------

    def _stage_7_relation_maps(
        self,
        memory_id: str,
        per_frame_assemblies: List[Assembly3D],
    ) -> List[RelationMap]:
        maps_dir = RELATION_MAPS_DIR / memory_id
        maps_dir.mkdir(parents=True, exist_ok=True)
        results: List[RelationMap] = []
        category_lookup = self._build_category_lookup()
        for i, asm in enumerate(per_frame_assemblies):
            rmap = extract_relation_map(
                asm,
                category_lookup=category_lookup,
                keyframe_index=i,
                keyframe_time=float(i),
            )
            path = maps_dir / f"frame_{i:04d}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rmap.to_dict(), f, indent=2, ensure_ascii=False)
            results.append(rmap)
        return results

    # ------------------------------------------------------------------
    # Stage 8a
    # ------------------------------------------------------------------

    async def _stage_8a_json_encode(
        self,
        caller: LLMCaller,
        reconstituted: Optional[Dict[str, Any]],
        per_frame_assemblies: List[Assembly3D],
        relation_maps: List[RelationMap],
        operation_groups: List[OperationGroup],
        nl_text: str,
    ) -> Optional[Dict[str, Any]]:
        final_assembly = per_frame_assemblies[-1] if per_frame_assemblies else None
        assembly_summary = self._summarise_assembly(final_assembly) if final_assembly else {}
        relation_summary = self._summarise_relation_maps(relation_maps)

        object_templates_dump: Dict[str, Any] = {}
        for name in self.storage.list_templates():
            object_templates_dump[name] = {
                "default_material": self.storage.get_default_material(name),
                "default_mass_grams": self.storage.get_default_mass(name),
                "default_dimensions_cm": self.storage.get_default_dimensions(name),
                "composite_children_templates": self.storage.get_composite_children(name),
            }

        action_templates_dump: Dict[str, Any] = {}
        seen_templates: set = set()
        for group in operation_groups:
            for op in group.operations:
                self._collect_action_templates(op, action_templates_dump, seen_templates)

        user_msg = (
            f"Reconstituted annotation:\n"
            f"{json.dumps(reconstituted, indent=2, ensure_ascii=False)}\n\n"
            f"Object templates used (full):\n"
            f"{json.dumps(object_templates_dump, indent=2, ensure_ascii=False)}\n\n"
            f"Action templates used (full):\n"
            f"{json.dumps(action_templates_dump, indent=2, ensure_ascii=False)}\n\n"
            f"Full operation groups (in order):\n"
            f"{json.dumps([g.to_dict() for g in operation_groups], indent=2, ensure_ascii=False)}\n\n"
            f"Final assembly summary:\n"
            f"{json.dumps(assembly_summary, indent=2, ensure_ascii=False)}\n\n"
            f"Relation map summary:\n"
            f"{json.dumps(relation_summary, indent=2, ensure_ascii=False)}\n\n"
            f"Natural language memory:\n{nl_text}\n\n"
            "Produce the final v0.2 memory JSON."
        )
        return await caller.call_json("stage_8a_json_encode", SYS_JSON_ENCODE, user_msg)

    def _collect_action_templates(self, op, out: Dict[str, Any], seen: set):
        return

    # ------------------------------------------------------------------
    # Stage 8b
    # ------------------------------------------------------------------

    async def _stage_8b_json_review(
        self,
        caller: LLMCaller,
        nl_text: str,
        initial_json: Dict[str, Any],
        final_assembly: Optional[Assembly3D],
        reconstituted: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        assembly_summary = self._summarise_assembly(final_assembly) if final_assembly else {}
        user_msg = (
            f"Original NL memory:\n{nl_text}\n\n"
            f"Reconstituted annotation:\n"
            f"{json.dumps(reconstituted, indent=2, ensure_ascii=False)}\n\n"
            f"Initial JSON encoding:\n"
            f"{json.dumps(initial_json, indent=2, ensure_ascii=False)}\n\n"
            f"3D assembly summary:\n"
            f"{json.dumps(assembly_summary, indent=2, ensure_ascii=False)}\n\n"
            "Perform the review and output the edited JSON."
        )
        return await caller.call_json("stage_8b_json_review", SYS_JSON_REVIEW, user_msg)

    # ------------------------------------------------------------------
    # Stage 8c
    # ------------------------------------------------------------------

    def _stage_8c_consistency_check(
        self,
        final_json: Dict[str, Any],
        per_frame_assemblies: List[Assembly3D],
    ) -> Dict[str, Any]:
        report = {
            "missing_assembly_id": None,
            "objects_in_json_not_in_assembly": [],
            "objects_in_assembly_not_in_json": [],
            "actions_without_operations": [],
            "goal_objects_missing": [],
        }
        final_assembly = per_frame_assemblies[-1] if per_frame_assemblies else None
        if final_assembly is None:
            report["missing_assembly_id"] = "no_assembly"
            return report

        asm_ids = {o.object_id for o in final_assembly.objects}
        json_obj_ids = set()
        for obj in final_json.get("objects", []) or []:
            if isinstance(obj, dict) and obj.get("obj_id"):
                json_obj_ids.add(obj["obj_id"])

        report["objects_in_json_not_in_assembly"] = sorted(json_obj_ids - asm_ids)
        report["objects_in_assembly_not_in_json"] = sorted(asm_ids - json_obj_ids)

        for goal in final_json.get("goal_state", []) or []:
            if isinstance(goal, dict) and goal.get("object") not in asm_ids:
                report["goal_objects_missing"].append(goal.get("object"))

        return report

    # ------------------------------------------------------------------
    # Diff vs v0.1
    # ------------------------------------------------------------------

    def _compute_diff(self, v01_json: Dict[str, Any],
                      v02_record: Dict[str, Any]) -> Dict[str, Any]:
        v01_objects = set()
        for obj in (v01_json.get("memory", {}) or {}).get("objects", []) or []:
            if isinstance(obj, dict) and obj.get("obj_id"):
                v01_objects.add(obj["obj_id"])

        v02_json = v02_record.get("final_json", {}) or {}
        v02_objects = set()
        for obj in v02_json.get("objects", []) or []:
            if isinstance(obj, dict) and obj.get("obj_id"):
                v02_objects.add(obj["obj_id"])

        v01_actions = set()
        for act in (v01_json.get("memory", {}) or {}).get("actions", []) or []:
            if isinstance(act, dict) and act.get("template"):
                v01_actions.add(act["template"])

        v02_actions = set()
        for act in v02_json.get("actions", []) or []:
            if isinstance(act, dict) and act.get("template"):
                v02_actions.add(act["template"])

        return {
            "objects_added_in_v02": sorted(v02_objects - v01_objects),
            "objects_removed_in_v02": sorted(v01_objects - v02_objects),
            "objects_shared": sorted(v01_objects & v02_objects),
            "actions_added_in_v02": sorted(v02_actions - v01_actions),
            "actions_removed_in_v02": sorted(v01_actions - v02_actions),
            "actions_shared": sorted(v01_actions & v02_actions),
            "consistency_report": v02_record.get("consistency_report", {}),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_v01_nl(self, v01_json: Dict[str, Any]) -> str:
        for key in ("nl_text", "nl", "nl_memory"):
            v = v01_json.get(key)
            if isinstance(v, str) and v.strip():
                return v
        inner = v01_json.get("memory", {}) or {}
        for key in ("nl_text", "nl", "nl_memory"):
            v = inner.get(key)
            if isinstance(v, str) and v.strip():
                return v
        return "(NL memory text not found in v0.1 record)"

    def _extract_v01_annotation(self, v01_json: Dict[str, Any]) -> str:
        for key in ("state_annotated_sketch", "annotated_plot", "generalized_sketch"):
            v = v01_json.get(key)
            if isinstance(v, str) and v.strip():
                return v
        inner = v01_json.get("memory", {}) or {}
        return json.dumps({
            "objects": inner.get("objects", []),
            "actions": inner.get("actions", []),
        }, indent=2, ensure_ascii=False)

    def _summarise_assembly(self, assembly: Optional[Assembly3D]) -> Dict[str, Any]:
        if assembly is None:
            return {}
        out = {"objects": []}
        world = assembly.get_all_world_matrices()
        for obj in assembly.objects:
            M = world.get(obj.object_id)
            pos = None
            if M is not None:
                pos = [float(M[0, 3]), float(M[1, 3]), float(M[2, 3])]
            out["objects"].append({
                "object_id": obj.object_id,
                "template_name": obj.template_name,
                "parent": obj.parent_object_id,
                "children": list(obj.children),
                "element_count": len(obj.elements),
                "position_world": pos,
                "metadata": dict(obj.metadata or {}),
                "material": dict(obj.material or {}),
            })
        return out

    def _summarise_relation_maps(self, maps: List[RelationMap]) -> Dict[str, Any]:
        out = {"frame_count": len(maps), "final_frame_links": []}
        if maps:
            for l in maps[-1].links:
                out["final_frame_links"].append({
                    "predicate": l.predicate,
                    "source": l.source_point_id,
                    "target": l.target_point_id,
                    "verified": l.verified,
                })
        return out

    def _build_category_lookup(self) -> Dict[str, List[str]]:
        lookup: Dict[str, List[str]] = {}
        for name in self.storage.list_templates():
            entry = self.storage.get_template(name)
            if entry is None:
                continue
            lookup[name] = [name] + list(entry.composite_children_templates or [])
        return lookup


# =============================================================================
# Persistence
# =============================================================================

def _ensure_dirs():
    for d in (DATA_DIR, DEMOS_DIR, STORAGES_DIR, ASSEMBLIES_DIR,
              RENDERS_DIR, MEMORIES_DIR, RELATION_MAPS_DIR, DIFFS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _overwrite_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_memories(memories: List[Dict[str, Any]]):
    with open(MEMORIES_DIR / "memories.jsonl", "w", encoding="utf-8") as f:
        for m in memories:
            f.write(json.dumps(m, ensure_ascii=False) + "\n\n---\n")


def _write_records(records: List[Dict[str, Any]]):
    with open(MEMORIES_DIR / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n\n---\n")

    with open(MEMORIES_DIR / "records_readable.txt", "w", encoding="utf-8") as f:
        for r in records:
            f.write("=" * 80 + "\n")
            f.write(f"Memory: {r.get('memory_id','?')}  "
                    f"Activity: {r.get('activity','?')}  "
                    f"Mode: {r.get('mode','?')}\n")
            if r.get("failed"):
                f.write(f"FAILED at stage: {r.get('failed_stage','?')}\n")
                reason = r.get("failure_reason", "")
                if reason:
                    f.write(f"Reason: {reason}\n")
            f.write("=" * 80 + "\n\n")
            for key in ("nl_text", "memory_sketch", "enhanced_sketch",
                        "generalized_sketch", "state_annotated_sketch",
                        "parts_prior", "reconstituted_annotation",
                        "initial_json", "final_json", "diff_vs_v01",
                        "consistency_report"):
                if key not in r:
                    continue
                f.write(f"--- {key} ---\n")
                v = r[key]
                if isinstance(v, str):
                    f.write(v + "\n\n")
                else:
                    f.write(json.dumps(v, indent=2, ensure_ascii=False) + "\n\n")
            f.write("\n\n")


def _write_token_usage(accountant: TokenAccountant):
    with open(MEMORIES_DIR / "token_usage.json", "w", encoding="utf-8") as f:
        json.dump(accountant.summary(), f, indent=2, ensure_ascii=False)


# =============================================================================
# Main
# =============================================================================

async def _main_async(args):
    _ensure_dirs()

    if not OPENAI_AVAILABLE:
        print("openai library not installed. pip install openai")
        return

    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    storage = Elements3DStorage()
    storage.load()

    pipeline = Pipeline3DMGen(client, storage)

    records: List[Dict[str, Any]] = []

    if args.mode == "A":
        if args.activity:
            activities = [args.activity]
        else:
            seed = args.seed or (
                "household activities: kitchen tasks, gardening, and home maintenance"
            )
            activities = await pipeline._stage_0a_activities(
                seed, args.count, memory_id="batch"
            )

        for i, activity in enumerate(activities):
            memory_id = f"mem3d_{int(time.time()):010d}_{i:02d}"
            try:
                record = await pipeline._run_mode_a(memory_id, activity)
                records.append(record)
            except StageFailure as e:
                print(f"    Memory '{activity}' failed at stage '{e.stage}': {e.reason}")
                records.append({
                    "memory_id": memory_id,
                    "activity": activity,
                    "mode": "A",
                    "failed": True,
                    "failed_stage": e.stage,
                    "failure_reason": e.reason,
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                print(f"    Memory '{activity}' failed with unexpected error: {e}")
                traceback.print_exc()
                records.append({
                    "memory_id": memory_id,
                    "activity": activity,
                    "mode": "A",
                    "failed": True,
                    "failed_stage": "unknown",
                    "failure_reason": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
    elif args.mode == "B":
        if not args.memory_file:
            print("Mode B requires --memory-file.")
            return
        memory_json = json.loads(Path(args.memory_file).read_text(encoding="utf-8"))
        mid = memory_json.get("memory_id") or f"mem3d_from_v01_{int(time.time())}"
        try:
            rec = await pipeline.generate_from_v01(mid, memory_json)
            records.append(rec)
        except StageFailure as e:
            records.append({
                "memory_id": mid,
                "activity": memory_json.get("activity", "unknown"),
                "mode": "B",
                "failed": True,
                "failed_stage": e.stage,
                "failure_reason": e.reason,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            records.append({
                "memory_id": mid,
                "activity": memory_json.get("activity", "unknown"),
                "mode": "B",
                "failed": True,
                "failed_stage": "unknown",
                "failure_reason": str(e),
                "timestamp": datetime.now().isoformat(),
            })
    else:
        print(f"Unknown mode '{args.mode}'.")
        return

    memories = [r.get("final_json", {}) for r in records
                if not r.get("failed") and r.get("final_json")]

    _write_memories(memories)
    _write_records(records)
    _write_token_usage(pipeline.accountant)

    storage.save()

    print()
    print("=" * 70)
    succeeded = [r for r in records if not r.get("failed")]
    failed = [r for r in records if r.get("failed")]
    print(f"Run complete.")
    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Failed:    {len(failed)}")
    if failed:
        by_stage: Dict[str, List[Dict[str, Any]]] = {}
        for r in failed:
            by_stage.setdefault(r.get("failed_stage", "unknown"), []).append(r)
        print()
        print("  Failures by stage:")
        for stage, items in sorted(by_stage.items()):
            print(f"    {stage}: {len(items)}")
            for r in items:
                print(f"      - {r.get('activity','?')} ({r.get('memory_id','?')})")
                reason = r.get("failure_reason", "")
                if reason:
                    print(f"        {reason}")
    print(f"Memories file:    {MEMORIES_DIR / 'memories.jsonl'}")
    print(f"Records file:     {MEMORIES_DIR / 'records.jsonl'}")
    print(f"Readable records: {MEMORIES_DIR / 'records_readable.txt'}")
    print(f"Token usage:      {MEMORIES_DIR / 'token_usage.json'}")
    totals = pipeline.accountant.summary().get("__grand_total__", {})
    print(f"Tokens: cache_hit={totals.get('prompt_cache_hit',0)}  "
          f"cache_miss={totals.get('prompt_cache_miss',0)}  "
          f"output={totals.get('output',0)}  "
          f"total={totals.get('total',0)}  "
          f"calls={totals.get('calls',0)}")


def main():
    parser = argparse.ArgumentParser(description="3d_mgen – v0.2 memory generation pipeline")
    parser.add_argument("--mode", choices=["A", "B"], default="A")
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--activity", type=str, default=None)
    parser.add_argument("--memory-file", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()