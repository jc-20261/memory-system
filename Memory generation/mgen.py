#!/usr/bin/env python3
r"""
Full Memory Generation Pipeline – updated with all approved changes.

Changes include:
- Fixed TemplateMatchLogger bool JSON serialization bug.
- Fixed singleton ObjectStorage/ActionStorage state reset.
- Removed duplicate search indexing.
- Cached SentenceTransformer in TemplateMatcher.
- Added categories/functions/materials generation to object templates.
- Extended ObjectStorage to store categories, functions, materials, aliases.
- Updated persistence to save/load all template metadata.
- Added pickle output for Memory objects and storages.
- Added startup loading from pickle if available, falling back to JSON.
- Added object-template auto-expansion with blank defaults for newly discovered attributes.
- Added goal_state generation and review in JSON encoding/review stages.
- Added fully resolved GoalState objects with runtime Object references in pickles.
"""

import asyncio, json, os, sys, time, random, shutil, copy, traceback, pickle
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from openai import AsyncOpenAI

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

# =============================================================================
# Configuration
# =============================================================================
from pathlib import Path as _Path
SCRIPT_DIR = _Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent
DEMO_DIR   = SCRIPT_DIR
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"
MAX_CONCURRENT = 10
MAX_TOTAL_MEMORIES = 2
INITIAL_BATCH_SIZE = 1
SNAPSHOT_MILESTONES = {1, 10, 50, 100, 200, 300, 1000, 10000, 100000}

TEMPLATE_NAME_THRESHOLD = 0.85
TEMPLATE_ACTION_NAME_THRESHOLD = 0.85
TEMPLATE_ACTION_TRAJ_THRESHOLD = 0.96
TEMPLATE_ACTION_COMBINED_THRESHOLD = 0.92

ALLOWED_TRAJECTORIES = {
    "linear", "straight", "circular", "arc", "curved",
    "enveloping", "helical", "spiral", "deceleration", "angular"
}

OUTPUT_MAX_TOKENS = 393216
REVIEW_MIN_OUTPUT_RATIO = 0.9

# =============================================================================
# Advanced Object / Action / Memory classes
# =============================================================================
class Object:
    def __init__(self, template_name: str, obj_id: str, number: str = "single",
                 position: dict = None, position_relative_body: dict = None,
                 dimensions: dict = None, weight: dict = None,
                 shape: str = "", motion: dict = None,
                 overrides: dict = None, aliases: list = None,
                 composite_of: list = None, functions: list = None):
        self.template_name = template_name
        self.obj_id = obj_id
        self.number = number
        self.position = position or {}
        self.position_relative_body = position_relative_body or {}
        self.dimensions = dimensions or {}
        self.weight = weight or {}
        self.shape = shape or ""
        self.motion = motion or {}
        self.overrides = overrides or {}
        self.aliases = aliases or []
        self.composite_of = composite_of or []
        self.functions = functions or []
        self.categories: List[str] = []

    @property
    def attributes(self):
        store = ObjectStorage()
        base = store.get_defaults(self.template_name).copy()
        merged = {**base, **self.overrides}
        return merged


class GoalState:
    """A goal condition referencing an actual runtime Object instance."""

    def __init__(
        self,
        target_object: Object,
        attribute: str,
        value: Any,
        aspect: Optional[str] = None,
        op: str = "eq",
    ):
        self.target_object = target_object
        self.attribute = attribute
        self.value = value
        self.aspect = aspect
        self.op = op

    def __repr__(self):
        target = getattr(self.target_object, "obj_id", self.target_object)
        return f"GoalState({target}.{self.attribute} {self.op} {self.value})"


class Protagonist(Object):
    def __init__(self, obj_id: str = "person_01"):
        super().__init__("person", obj_id, number="single")
        self.overrides.update({
            "intention": None, "cheerfulness": 0.5, "focus": 0.8,
            "fatigue": 0.2, "satisfaction": 0.3, "sweat_level": 0.0,
            "breathing_rate": "normal", "energy_level": 0.8,
            "mental_state": {},
            "sensory_perception_touch": [],
            "sensory_perception_sight": [],
            "sensory_perception_smell": [],
            "sensory_perception_taste": [],
            "sensory_perception_sound": []
        })
        self.anatomical_features: List[str] = []


class ObjectStorage:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._defaults: Dict[str, dict] = {}
        self._functions: Dict[str, List[str]] = defaultdict(list)
        self._aliases: Dict[str, List[str]] = defaultdict(list)
        self._categories: Dict[str, List[str]] = defaultdict(list)
        self._materials: Dict[str, List[str]] = defaultdict(list)
        self._initialized = True

    def add_template(self, name, defaults, functions=None, aliases=None, categories=None, materials=None):
        self._defaults[name] = defaults
        if functions:
            self._functions[name] = functions
        if aliases:
            self._aliases[name] = aliases
        if categories:
            self._categories[name] = categories
        if materials:
            self._materials[name] = materials

    def get_defaults(self, name):
        return self._defaults.get(name, {})

    def get_categories(self, name):
        return self._categories.get(name, [])

    def get_functions(self, name):
        return self._functions.get(name, [])

    def get_aliases(self, name):
        return self._aliases.get(name, [])

    def get_materials(self, name):
        return self._materials.get(name, [])

    def list_templates(self):
        return list(self._defaults.keys())

    def instantiate(self, template_name, obj_id, overrides=None, **kwargs):
        return Object(template_name, obj_id, overrides=overrides or {}, **kwargs)

    def expand_template_from_instance_attributes(self, template_name: str, instance_attributes: Dict[str, Any]):
        """
        Add missing attribute paths to the template with blank defaults.

        For nested dicts, recurse and leave empty dicts/None as appropriate.
        """
        if template_name not in self._defaults:
            return

        template = self._defaults[template_name]

        def ensure_path(current_template: dict, current_instance: dict, path_parts: List[str]):
            for key, value in current_instance.items():
                if isinstance(value, dict) and isinstance(current_template.get(key), dict):
                    ensure_path(current_template[key], value, path_parts + [str(key)])
                elif key not in current_template:
                    # Add blank default
                    if isinstance(value, dict):
                        current_template[key] = {}
                    elif isinstance(value, list):
                        current_template[key] = []
                    else:
                        current_template[key] = None

        ensure_path(template, instance_attributes, [])


@dataclass
class Action:
    template: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    duration: int = 1
    participants: List[Object] = field(default_factory=list)
    changes: List[Dict] = field(default_factory=list)
    changes_per_cycle: List[Dict] = field(default_factory=list)
    changes_total: List[Dict] = field(default_factory=list)
    conditional_changes: List[Dict] = field(default_factory=list)
    preconditions: List[Dict] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    action_category: str = "HUMAN_INTERACTION"
    kinematic_trajectory: str = "straight"
    temporal_type: str = "sequential"
    sub_actions: List['Action'] = field(default_factory=list)
    repetitions: int = 1
    cycle_duration: float = 0.0

    @property
    def effective_duration(self):
        if self.temporal_type == "cyclical":
            return self.cycle_duration * self.repetitions
        return self.duration


class ActionStorage:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._templates: Dict[str, dict] = {}
        self._sub_action_sequences: Dict[str, List[str]] = {}
        self._initialized = True

    def add_template(self, name, sub_actions=None, **kwargs):
        self._templates[name] = kwargs
        if sub_actions is not None:
            self._sub_action_sequences[name] = sub_actions

    def get_template(self, name):
        return self._templates.get(name, {})

    def list_templates(self):
        return list(self._templates.keys())

    def get_sub_action_sequence(self, name):
        return self._sub_action_sequences.get(name, [])

    def instantiate(self, template_name, overrides, auto_add=True):
        if template_name not in self._templates:
            if auto_add:
                self._templates[template_name] = {}
        dur = overrides.get("duration", 1)
        try:
            dur = int(dur)
        except:
            dur = 1
        traj = overrides.get("kinematic_trajectory", "straight")
        if traj not in ALLOWED_TRAJECTORIES:
            traj = "straight"
        cat = overrides.get("action_category", "HUMAN_INTERACTION")
        return Action(
            template=template_name,
            overrides=overrides,
            duration=dur,
            kinematic_trajectory=traj,
            temporal_type=overrides.get("temporal_type", "sequential"),
            repetitions=overrides.get("repetitions", 1),
            cycle_duration=overrides.get("cycle_duration", 0.0),
            tags=overrides.get("tags") or [],
            action_category=cat,
            preconditions=overrides.get("preconditions") or [],
            changes=overrides.get("changes") or [],
            changes_per_cycle=overrides.get("changes_per_cycle") or [],
            changes_total=overrides.get("changes_total") or [],
            conditional_changes=overrides.get("conditional_changes") or []
        )


class Memory:
    def __init__(self, memory_id, protagonist):
        self.id = memory_id
        self.protagonist = protagonist
        self.objects = {protagonist.obj_id: protagonist}
        self.actions = []
        self.activity = ""
        self.goal_states: List[GoalState] = []
        self.indexed_terms = set()

    def add_object(self, obj):
        self.objects[obj.obj_id] = obj

    def add_action(self, action):
        self.actions.append(action)


# ----------------------------------------------------------------------
# Search Engine (generation-side)
# ----------------------------------------------------------------------
class SearchEngine:
    def __init__(self, max_memories=10000):
        self.memories = deque(maxlen=max_memories)
        self.index = defaultdict(list)

    def add_memory(self, mem):
        if len(self.memories) >= self.memories.maxlen:
            self.memories.popleft()
        self.memories.append(mem)
        indexed_terms = self._index_memory(mem)
        return indexed_terms

    def _index_memory(self, mem):
        terms = {}
        if mem.activity:
            terms[f"activity:{mem.activity}"] = 1.0
        for obj in mem.objects.values():
            flat = self._flatten_dict(obj.overrides)
            for key, val in flat.items():
                terms[f"{obj.template_name}.{key}={val}"] = 1.0
                terms[f"{key}={val}"] = 1.0
            for dim, d in (obj.dimensions or {}).items():
                terms[f"dimension.{dim}.value={d['value']}"] = 1.0
                terms[f"dimension.{dim}.unit={d['unit']}"] = 1.0
            if obj.weight:
                terms[f"weight.value={obj.weight['value']}"] = 1.0
                terms[f"weight.unit={obj.weight['unit']}"] = 1.0
            if obj.position:
                terms[f"position={obj.position.get('relation')}:{obj.position.get('relative_to')}"] = 1.0
            if obj.position_relative_body:
                terms[f"position_relative_body={obj.position_relative_body.get('relation')}"] = 1.0
            if obj.motion:
                terms[f"motion.type={obj.motion.get('type')}"] = 1.0
                terms[f"motion.velocity={obj.motion.get('velocity')}"] = 1.0
            if obj.shape:
                terms[f"shape={obj.shape}"] = 1.0
            for alias in obj.aliases:
                terms[f"alias:{alias}"] = 1.0
            terms[f"template:{obj.template_name}"] = 1.0
        for act in mem.actions:
            self._index_action(act, terms)
        for term, weight in terms.items():
            self.index[term].append((mem, weight))
        return set(terms.keys())

    def _flatten_dict(self, d, parent_key="", sep="."):
        items = {}
        if not isinstance(d, dict):
            return items
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key, sep))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.update(self._flatten_dict(item, f"{new_key}.{i}", sep))
                    else:
                        items[f"{new_key}.{i}"] = item
            else:
                items[new_key] = v
        return items

    def _index_action(self, action, terms, scale=1.0):
        terms[f"action:{action.template}"] = terms.get(f"action:{action.template}", 0) + 2.0 * scale
        terms[f"trajectory:{action.kinematic_trajectory}"] = terms.get(f"trajectory:{action.kinematic_trajectory}", 0) + 2.0 * scale
        terms[f"category:{action.action_category}"] = terms.get(f"category:{action.action_category}", 0) + 2.0 * scale
        for tag in (action.tags or []):
            terms[tag] = terms.get(tag, 0) + 2.0 * scale
        for prec in (action.preconditions or []):
            self._index_condition(prec, terms, "precondition")
        for ch in (action.changes or []) + (action.changes_per_cycle or []) + (action.changes_total or []):
            self._index_change(ch, terms)
        for cond in (action.conditional_changes or []):
            if isinstance(cond, dict):
                changes = cond.get("changes") or []
                for ch in changes:
                    self._index_change(ch, terms)
        for sub in (action.sub_actions or []):
            self._index_action(sub, terms, scale * 0.8)

    def _index_condition(self, cond, terms, prefix):
        if not isinstance(cond, dict):
            return
        obj = cond.get("object", "?")
        aspect = cond.get("aspect", "")
        attr = cond.get("attribute", "?")
        val = cond.get("value", "?")
        full = f"{obj}.{aspect}.{attr}" if aspect else f"{obj}.{attr}"
        terms[f"{prefix}:{full}={val}"] = 1.0
        terms[f"{prefix}:{attr}={val}"] = 1.0

    def _index_change(self, ch, terms):
        if not isinstance(ch, dict):
            return
        obj = ch.get("object", "?")
        aspect = ch.get("aspect", "")
        attr = ch.get("attribute", "?")
        old = ch.get("old", "?")
        new = ch.get("new", "?")
        full = f"{obj}.{aspect}.{attr}" if aspect else f"{obj}.{attr}"
        terms[f"state_change:{full}:{old}->{new}"] = 1.0
        terms[f"state_change:{attr}:{new}"] = 1.0


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
class PersistenceManager:
    def __init__(self, base_dir, milestones=None):
        self.base_dir = Path(base_dir)
        self.main_dir = self.base_dir / "main"
        self.snapshots_dir = self.base_dir / "snapshots"
        self.main_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.milestones = milestones or {1, 10, 50, 100, 200, 300, 1000, 10000}
        self.memory_objects_dir = self.main_dir / "memory_objects"
        self.memory_objects_dir.mkdir(parents=True, exist_ok=True)

    def save_final_json(self, json_data):
        with open(self.main_dir / "memories_final.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(json_data, ensure_ascii=False) + "\n\n---\n")
            f.flush()

    def save_record(self, record):
        with open(self.main_dir / "records.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n\n---\n")
            f.flush()
        self._save_readable_record(record)

    def _save_readable_record(self, record):
        with open(self.main_dir / "records_readable.txt", "a", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"Activity: {record.get('activity','')}\n")
            f.write(f"Memory ID: {record.get('memory_id','')}\n")
            f.write("=" * 70 + "\n\n")
            for key, label in [
                ("nl_text", "NL Memory"),
                ("memory_sketch", "Memory Sketch"),
                ("enhanced_sketch", "Enhanced Sketch"),
                ("generalized_sketch", "Generalized Sketch"),
                ("state_annotated_sketch", "State Annotated Sketch"),
                ("initial_json", "Initial JSON"),
                ("json_review", "JSON Review"),
                ("final_json", "Final JSON"),
            ]:
                if key in record:
                    f.write(f"--- {label} ---\n")
                    if isinstance(record[key], str):
                        f.write(record[key] + "\n\n")
                    else:
                        f.write(json.dumps(record[key], indent=2, ensure_ascii=False) + "\n\n")
            if "timings" in record:
                f.write("--- Timings ---\n")
                f.write(json.dumps(record["timings"], indent=2, ensure_ascii=False) + "\n\n")
            f.write("\n")

    def save_stage_timings_jsonl(self, memory_id, activity, timings):
        with open(self.main_dir / "stage_timings.jsonl", "a", encoding="utf-8") as f:
            entry = {"memory_id": memory_id, "activity": activity, "timings": timings}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()

    def save_run_summary_readable(self, run_timings, total_elapsed_seconds):
        with open(self.main_dir / "stage_timings_readable.txt", "w", encoding="utf-8") as f:
            f.write("Total run elapsed time: {:.2f} seconds\n\n".format(total_elapsed_seconds))
            for entry in run_timings:
                f.write(json.dumps(entry, indent=2, ensure_ascii=False))
                f.write("\n\n---\n\n")

    def save_object_storage(self, obj_storage):
        flat = {}
        for name in obj_storage.list_templates():
            defaults = copy.deepcopy(obj_storage.get_defaults(name))
            entry = defaults
            cats = obj_storage.get_categories(name)
            funcs = obj_storage.get_functions(name)
            aliases = obj_storage.get_aliases(name)
            mats = obj_storage.get_materials(name)
            if cats:
                entry["categories"] = copy.deepcopy(cats)
            if funcs:
                entry["functions"] = copy.deepcopy(funcs)
            if aliases:
                entry["aliases"] = copy.deepcopy(aliases)
            if mats:
                entry["materials"] = copy.deepcopy(mats)
            flat[name] = entry
        with open(self.main_dir / "object_storage.json", "w", encoding="utf-8") as f:
            json.dump(flat, f, indent=2, ensure_ascii=False)
            f.flush()

    def save_action_storage(self, act_storage):
        data = {
            "templates": act_storage._templates,
            "sub_action_sequences": act_storage._sub_action_sequences
        }
        with open(self.main_dir / "action_storage.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()

    def save_pipeline_state(self, state):
        with open(self.main_dir / "pipeline_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
            f.flush()

    def save_initial_activities(self, activities):
        with open(self.main_dir / "initial_activities.json", "w", encoding="utf-8") as f:
            json.dump(activities, f, indent=2, ensure_ascii=False)
            f.flush()

    def maybe_snapshot(self, count):
        if count in self.milestones:
            snap_dir = self.snapshots_dir / f"gen_{count}"
            snap_dir.mkdir(parents=True, exist_ok=True)
            for f in self.main_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, snap_dir / f.name)
            print(f"  📸 Snapshot at {count}")

    def load_activity_history(self):
        path = self.main_dir / "activity_history.jsonl"
        if not path.exists():
            return []
        activities = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    activities.append(line)
        return activities

    def save_activity_to_history(self, activity):
        path = self.main_dir / "activity_history.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(activity + "\n")
            f.flush()

    def load_object_storage(self):
        path = self.main_dir / "object_storage.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_action_storage(self):
        path = self.main_dir / "action_storage.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_all_search_terms_history(self, memory_id, terms):
        path = self.main_dir / "all_search_terms_history.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            entry = {"memory_id": memory_id, "terms": list(terms)}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()

    def save_run_initial_search_terms(self, terms):
        path = self.main_dir / "run_initial_search_terms.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(terms), f, indent=2, ensure_ascii=False)
            f.flush()

    def load_run_initial_search_terms(self):
        path = self.main_dir / "run_initial_search_terms.json"
        if not path.exists():
            return set()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data)

    def save_memory_object(self, memory):
        path = self.memory_objects_dir / f"{memory.id}.pkl"
        with open(path, "wb") as f:
            pickle.dump(memory, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_storage_objects(self, obj_storage, act_storage):
        with open(self.memory_objects_dir / "object_storage.pkl", "wb") as f:
            pickle.dump(obj_storage, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.memory_objects_dir / "action_storage.pkl", "wb") as f:
            pickle.dump(act_storage, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_storage_objects(self):
        obj_storage = None
        act_storage = None
        obj_pkl = self.memory_objects_dir / "object_storage.pkl"
        act_pkl = self.memory_objects_dir / "action_storage.pkl"
        if obj_pkl.exists():
            try:
                with open(obj_pkl, "rb") as f:
                    obj_storage = pickle.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load object_storage pickle: {e}")
        if act_pkl.exists():
            try:
                with open(act_pkl, "rb") as f:
                    act_storage = pickle.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load action_storage pickle: {e}")
        return obj_storage, act_storage


# ----------------------------------------------------------------------
# API Logger & Template Match Logger
# ----------------------------------------------------------------------
class APILogger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, activity, purpose, messages, response_content, usage):
        try:
            usage_dict = usage.model_dump()
        except:
            usage_dict = {"raw": str(usage)}
        entry = {"timestamp": datetime.now().isoformat(),
                 "activity": activity,
                 "purpose": purpose,
                 "messages": messages,
                 "response": response_content,
                 "usage": usage_dict}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()


class TemplateMatchLogger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, match_type, proposed, best_match, score, matched):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": match_type,
            "proposed": str(proposed),
            "best_match": str(best_match),
            "score": float(round(score, 4)),
            "matched": bool(matched),
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()


# ----------------------------------------------------------------------
# Template Matcher
# ----------------------------------------------------------------------
class TemplateMatcher:
    _embedding_model = None

    def __init__(self, object_storage, action_storage, match_logger):
        self.object_storage = object_storage
        self.action_storage = action_storage
        self.logger = match_logger
        self._obj_names, self._obj_emb, self._obj_bm25 = [], [], None
        self._act_names, self._act_emb, self._act_traj, self._act_bm25 = [], [], [], None
        self._build_index()

    @classmethod
    def _get_embedding_model(cls):
        if cls._embedding_model is None:
            if ST_AVAILABLE:
                cls._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._embedding_model

    def _build_index(self):
        if not ST_AVAILABLE or not BM25_AVAILABLE:
            print("⚠️ TemplateMatcher disabled: missing sentence-transformers or rank-bm25")
            return
        model = self._get_embedding_model()
        if model is None:
            return
        self._obj_names = self.object_storage.list_templates()
        if self._obj_names:
            self._obj_emb = model.encode(self._obj_names, normalize_embeddings=True)
            self._obj_bm25 = BM25Okapi([n.split() for n in self._obj_names])
        self._act_names = self.action_storage.list_templates()
        if self._act_names:
            self._act_emb = model.encode(self._act_names, normalize_embeddings=True)
            self._act_bm25 = BM25Okapi([n.split() for n in self._act_names])
            self._act_traj = [self.action_storage.get_template(n).get("kinematic_trajectory","straight") for n in self._act_names]

    def is_active(self):
        return (self._obj_bm25 is not None or self._act_bm25 is not None)

    def find_matching_object(self, proposed_name):
        if self._obj_bm25 is None or self._obj_emb is None:
            return None
        if proposed_name in self._obj_names:
            self.logger.log("object", proposed_name, proposed_name, 1.0, True)
            return proposed_name
        tokens = proposed_name.split()
        bm25_scores = self._obj_bm25.get_scores(tokens)
        model = self._get_embedding_model()
        proposed_emb = model.encode(proposed_name, normalize_embeddings=True)
        emb_scores = np.dot(self._obj_emb, proposed_emb)
        combined = (bm25_scores / (bm25_scores.max() or 1) + emb_scores) / 2
        best_idx = np.argmax(combined)
        best_score = combined[best_idx]
        best_name = self._obj_names[best_idx]
        matched = best_score >= TEMPLATE_NAME_THRESHOLD
        self.logger.log("object", proposed_name, best_name, best_score, matched)
        return best_name if matched else None

    def find_matching_action(self, proposed_name, proposed_traj, proposed_sub_actions=None):
        if proposed_sub_actions:
            seq = [sa.get("template") for sa in proposed_sub_actions]
            for name in self._act_names:
                stored_seq = self.action_storage.get_sub_action_sequence(name)
                if stored_seq and stored_seq == seq:
                    self.logger.log("complex", proposed_name, name, 1.0, True)
                    return name
        if self._act_bm25 is None or self._act_emb is None:
            return None
        if proposed_name in self._act_names:
            idx = self._act_names.index(proposed_name)
            traj_sim = self._traj_sim(proposed_traj, self._act_traj[idx])
            combined = 0.5 + 0.5 * traj_sim
            matched = combined >= TEMPLATE_ACTION_COMBINED_THRESHOLD
            self.logger.log("action", proposed_name, proposed_name, combined, matched)
            return proposed_name if matched else None
        tokens = proposed_name.split()
        bm25_scores = self._act_bm25.get_scores(tokens)
        model = self._get_embedding_model()
        proposed_emb = model.encode(proposed_name, normalize_embeddings=True)
        emb_scores = np.dot(self._act_emb, proposed_emb)
        name_scores = (bm25_scores / (bm25_scores.max() or 1) + emb_scores) / 2
        best_idx = np.argmax(name_scores)
        best_name = self._act_names[best_idx]
        best_name_score = name_scores[best_idx]
        if best_name_score < TEMPLATE_ACTION_NAME_THRESHOLD:
            self.logger.log("action", proposed_name, best_name, best_name_score, False)
            return None
        traj_sim = self._traj_sim(proposed_traj, self._act_traj[best_idx])
        combined = (best_name_score + traj_sim) / 2
        matched = combined >= TEMPLATE_ACTION_COMBINED_THRESHOLD
        self.logger.log("action", proposed_name, best_name, combined, matched)
        return best_name if matched else None

    def _traj_sim(self, t1, t2):
        if t1 == t2:
            return 1.0
        if not ST_AVAILABLE:
            return 0.0
        model = self._get_embedding_model()
        e1 = model.encode(t1, normalize_embeddings=True)
        e2 = model.encode(t2, normalize_embeddings=True)
        sim = float(np.dot(e1, e2))
        return 0.0 if sim < 0.7 else sim


# ----------------------------------------------------------------------
# Action Decomposer (safety net)
# ----------------------------------------------------------------------
class ActionDecomposer:
    def __init__(self, llm, logger=None):
        self.llm = llm
        self.logger = logger

    async def decompose_action(self, action):
        return action


# ----------------------------------------------------------------------
# Memory Generation Pipeline
# ----------------------------------------------------------------------
class MemoryGenerationPipeline:
    def __init__(self, llm, object_storage, action_storage, search_engine,
                 persistence, template_matcher,
                 nl_examples_text="", extra_context="",
                 action_plot_demo="", decompose_demo="", generalize_demo="",
                 state_change_demo="", json_encode_demo="", json_review_demo="",
                 api_logger=None):
        self.llm = llm
        self.object_storage = object_storage
        self.action_storage = action_storage
        self.search_engine = search_engine
        self.persistence = persistence
        self.template_matcher = template_matcher
        self.api_logger = api_logger
        self.decomposer = ActionDecomposer(llm, api_logger)

        self.generated_count = 0
        self.all_searchable_terms = set()
        self.new_actions = set()
        self.new_objects = set()
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.id_lock = asyncio.Lock()
        self.success_count = 0
        self.fail_count = 0
        self.generation_times = []
        self.run_timings = []
        self.initial_batch_search_terms = set()
        self.expansion_pool = set()

        self.proposed_activities = set()
        self.proposed_activities.update(self.persistence.load_activity_history())

        # ---- Static prompts ----
        self.nl_system_msg = (
            "You are a meticulous observer of physical processes. "
            "Write a detailed procedural memory in the style of the examples below. "
            "The memory should be 300–600 words, rich in sensory details, physical attributes, "
            "object states, spatial relationships, and explicit state changes. "
            "Make it as vivid and structured as the examples below while ensuring high plausibility. \n\n"
            + nl_examples_text
        )

        self.sketch_system_msg = (
            "You are an expert in task and object analysis. Given a procedural memory, produce a memory sketch that includes:\n\n"
            "1. OBJECT LIST (tiered)\n"
            "   - List all objects present in the scene and involved in the actions.\n"
            "   - For composite objects, show sub‑objects indented beneath the parent.\n"
            "   - Example:\n"
            "     apple_01\n"
            "     ├── apple_skin_01\n"
            "     ├── apple_flesh_01\n"
            "     └── apple_core_01\n"
            "     knife_01\n"
            "     ├── knife_blade_01\n"
            "     └── knife_handle_01\n"
            "     person_01\n"
            "     ├── left_hand_01\n"
            "     ├── right_hand_01\n"
            "     └── eyes_01\n\n"
            "2. ACTION PLOT (with natural processes in parallel)\n"
            "   - Show the main human actions and their temporal structure.\n"
            "   - Add natural processes as separate actions in parallel lanes where they occur.\n"
            "   - Mark action category where relevant. Categories include:\n"
            "       HUMAN_INTERACTION\n"
            "       ├── MANIPULATION\n"
            "       ├── TOOL_USAGE\n"
            "       ├── BODY_MOVEMENT\n"
            "       └── PERCEPTION\n"
            "       NATURAL_PROCESS\n"
            "       ├── GRAVITY\n"
            "       ├── OXIDATION\n"
            "       ├── LIQUID_FLOW\n"
            "       ├── STRESS_TEAR\n"
            "       ├── LIGHT_PASSAGE\n"
            "       ├── PRESSURE_RELEASE\n"
            "       ├── CHEMICAL_CHANGE\n"
            "       └── BIOLOGICAL_PROCESS\n"
            "   - Use indentation for sub‑actions and include durations, temporal types (seq, parallel, cyclical), and start offsets if needed.\n"
            + (action_plot_demo if action_plot_demo else "")
        )

        self.enhance_system_msg = (
            "You are an expert in decomposition. Enhance the memory sketch by:\n\n"
            "1. Recursively decomposing every action into more primitive sub‑actions.\n"
            "   - If an action can be broken down further, do so until all leaf actions are atomic or cyclical.\n"
            "   - Keep the same hierarchical format.\n"
            "   - Add durations and temporal types (seq, parallel, cyclical).\n"
            "   - Mark cyclical actions with 'cyclical' and include cycle_duration and repetitions if known.\n"
            "   - For natural processes, decompose them as well if possible, for example:\n"
            "       juice_seepage → cell_rupture + liquid_flow\n"
            "       stress_tear → tear_initiation + tear_propagation\n"
            "       light_passage → photon_transmission\n"
            "       gravity_pull → downward_acceleration\n\n"
            "2. Extending the object list:\n"
            "   - Add any new objects that emerge from newly decomposed actions.\n"
            "   - Break down objects further when appropriate, for example:\n"
            "       knife_01 → knife_blade_01 + knife_handle_01\n"
            "       hand_01 → fingers, palm, thumb\n"
            "   - Ensure the object list and action plot stay synchronised.\n\n"
            "   Example of a decomposed sub‑action:\n"
            "   rotate_apple_by_hand (cyclical, 40s)\n"
            "   ├── adjust_grip_for_rotation (2s)\n"
            "   │   ├── release_slight_pressure (0.5s)\n"
            "   │   ├── reposition_thumb (1.0s)\n"
            "   │   └── reapply_pressure (0.5s)\n"
            "   ├── apply_continuous_torque_thumb (cyclical, 36s)\n"
            "   │   ├── flex_thumb_interphalangeal_joint (cyclical, 1s)\n"
            "   │   └── extend_thumb_interphalangeal_joint (cyclical, 1s)\n"
            "   ├── monitor_rotation_visually (cyclical, 40s)\n"
            "   │   ├── saccade_scene (cyclical, 0.3s)\n"
            "   │   └── fixate_on_surface (cyclical, 0.5s)\n"
            "   └── stop_rotation (2s)\n"
            "       ├── relax_torque_muscles (1.0s)\n"
            "       └── return_hand_to_neutral (1.0s)\n"
            + (decompose_demo if decompose_demo else "")
        )

        self.generalize_system_msg = (
            "You are an expert in action generalisation. Using the enhanced action plot, perform bottom‑up generalisation:\n\n"
            "- Start from the lowest‑level sub‑actions and replace them with the most general reusable template name that accurately describes the action.\n"
            "- Work upwards level by level.\n"
            "- For complex actions, provide a general template name and an alternative name that preserves the specific sequence of this memory.\n"
            "- Do not remove any actions or sub‑actions; only adjust their names.\n"
            "- Mark complex actions with 'general: [template]' and 'alt: [specific_alt]'.\n\n"
            "Example replacements:\n"
            "    specific: draw_knife_along_surface\n"
            "    general:  draw_tool_along_surface\n"
            "    alternative: draw_knife_along_surface\n\n"
            "    specific: rotate_apple_by_hand\n"
            "    general:  rotate_object_by_hand\n"
            "    alternative: rotate_apple_by_hand\n\n"
            "    specific: continuous_peeling\n"
            "    general:  continuous_peeling\n"
            "    alternative: continuous_peeling_apple_rotational\n\n"
            "The output must be the full action plot with the same size and structure as the input, only names changed/added.\n"
            + (generalize_demo if generalize_demo else "")
        )

        self.state_change_system_msg = (
            "You are an expert in physical state changes. Annotate the action plot exhaustively.\n\n"
            "1. Add preconditions to actions – conditions that must be true before the action can start.\n"
            "   - Format:\n"
            "     preconditions:\n"
            "       - object.aspect.attribute = value\n"
            "   or \n"
            "     preconditions:\n"
            "       - object.attribute = value\n"
            "   - Distinguish preconditions from state changes; preconditions are checked, not caused.\n\n"
            "2. Add state changes that result from each action.\n"
            "   - For simple actions, use 'changes:' list.\n"
            "   - For cyclical actions, separate:\n"
            "       changes_per_cycle:  (what happens each cycle)\n"
            "       changes_total:      (net change after all cycles)\n"
            "   - For actions with variable effects, use 'conditional_changes:' based on an attribute, e.g.:\n"
            "       conditional_changes:\n"
            "         if pressing_force = low:\n"
            "           - peel_strip_01.width = 1 cm\n"
            "           - peel_strip_01.thickness_height = 0.1 cm\n"
            "           - peel_strip_01.side_2.tear_probability = 0.3\n"
            "           - apple_flesh_01.surface.indentation.width = 1 cm\n"
            "           - apple_flesh_01.surface.indentation.thickness_height = 0.07 cm\n"
            "           - peel_strip_01.transparency = medium\n"
            "         if pressing_force = high:\n"
            "           - peel_strip_01.width = 2 cm\n"
            "           - peel_strip_01.thickness_height = 0.3 cm\n"
            "           - peel_strip_01.side_2.tear_probability = 0.1\n"
            "           - apple_flesh_01.surface.indentation.width = 2 cm\n"
            "           - apple_flesh_01.surface.indentation.thickness_height = 0.2 cm\n"
            "           - peel_strip_01.transparency = very low\n"
            "         if pressing_force = medium:\n"
            "           - ... (analogous)\n\n"
            "3. Add sensory perception changes for the protagonist:\n"
            "   - Every sensory entry must include a 'source'.\n"
            "   - Example:\n"
            "     person_01.sensory_perception_touch += slight_wetness (source: juice_01)\n"
            "     person_01.sensory_perception_sight += spiral_peel_strip (source: peel_strip_01)\n\n"
            "4. Add natural processes that are directly linked to attribute changes:\n"
            "   - If a state change modifies 'tear_probability', ensure 'stress_tear' appears.\n"
            "   - If 'transparency' is modified, ensure 'light_passage' appears.\n"
            "   - If 'juice_surface' becomes true, ensure 'juice_seepage' is present.\n\n"
            "5. Use aspects where needed:\n"
            "   - Format: object.aspect.attribute\n"
            "   - Example (where the aspect in this case is the surface of an object. There are other types of aspects, such as interior, side[01], side[02], edge, etc):\n"
            "     apple_flesh_01.surface.cell_structure: intact -> ruptured\n"
            "     apple_01.surface.coverage.type = droplets\n"
            "     apple_01.surface.coverage.composition = apple_juice\n"
            "     left_hand_01.surface.coverage[0].type = film_patch\n"
            "     left_hand_01.surface.coverage[1].type = droplets\n\n"
            "6. Capture every observable change in the NL memory, including (in terms of the NL memory of peeling an apple; please also see the demonstration below):\n"
            "   - Initial surface dampness: apple_skin_01.surface_moisture = slightly_damp\n"
            "   - Friction/slip resistance: apple_skin_01.surface_friction = moderate\n"
            "   - Peel strip shape: shape = ['strip', 'spiral']\n"
            "   - Hand coverage: left_hand_01.surface.coverage = [ film_patch of apple_juice, water droplets ]\n"
            "   - Apple surface droplets: apple_01.surface.coverage.type = droplets; composition = apple_juice\n"
            "   - Peel strip underside composition: peel_strip_01.side_2.composition = apple_pith\n"
            "   - Fingerprint indentations: apple_01.surface.indentations[0].type = fingerprint, height_thickness = 0.1 cm, width = 1.0 cm\n\n"
            "7. Ensure every action (including natural processes) has its preconditions and changes fully specified.\n"
            + (state_change_demo if state_change_demo else "")
        )

        self.json_encode_system_msg = (
            "You are a memory‑to‑JSON encoder. Convert the annotated action plot and the original NL memory into a rich JSON encoding.\n\n"
            "The JSON must have four top‑level keys:\n"
            "  'object_templates'\n"
            "  'action_templates'\n"
            "  'memory'\n"
            "  'goal_state'\n\n"
            "Requirements:\n\n"
            "A. Object templates:\n"
            "   - For every object type used, provide a template with invariant defaults.\n"
            "   - Templates must include:\n"
            "       - 'defaults': containing:\n"
            "           - dimensions (length, width, height_thickness) with value and unit\n"
            "           - mass (value and unit)\n"
            "           - shape\n"
            "           - position (relation, relative_to)\n"
            "           - any other invariant attributes (color, material, etc.)\n"
            "       - 'functions': list of plausible functions (up to 10)\n"
            "       - 'categories': hierarchical category chain from specific to abstract, ending with 'physical_object' and 'object'\n"
            "       - 'aliases': list of alternate names\n"
            "       - 'materials': list of materials (NOT a composite string)\n"
            "       - If the object is composite, include 'composition_templates' listing sub‑object template names.\n\n"
            "B. Action templates:\n"
            "   - For every action template used, provide a record with:\n"
            "       - default_duration\n"
            "       - action_category (from the allowed categories)\n"
            "       - kinematic_trajectory (from the allowed set)\n"
            "       - sub_action_templates (if complex)\n"
            "       - alternatives (if any)\n"
            "       - default preconditions if applicable\n\n"
            "C. Memory instance:\n"
            "   - Include 'memory_id'.\n"
            "   - 'protagonist': object with overrides, anatomical_features, and sensory perceptions (each entry with source).\n"
            "   - 'objects': list of object instances. Each instance must have:\n"
            "       - template (referencing template name)\n"
            "       - obj_id\n"
            "       - number ('single' or 'plural')\n"
            "       - position (or position_relative_body for body parts)\n"
            "       - dimensions (length, width, height_thickness) with value/unit\n"
            "       - mass (value/unit)\n"
            "       - shape\n"
            "       - overrides: all variable attributes, including nested aspects, sensory info, surface coverage, indentations, etc.\n"
            "   - 'actions': list of action instances. Each action must have:\n"
            "       - template\n"
            "       - overrides (participants, specific parameters)\n"
            "       - duration\n"
            "       - participants (list of obj_ids)\n"
            "       - preconditions (list of {object, aspect, attribute, op, value})\n"
            "       - changes (or changes_per_cycle/changes_total for cyclical actions)\n"
            "       - conditional_changes (list of {condition, changes})\n"
            "       - kinematic_trajectory\n"
            "       - temporal_type\n"
            "       - sub_actions (recursive)\n\n"
            "D. Goal state:\n"
            "   - 'goal_state' must be a list of goal objects.\n"
            "   - Each goal object must have:\n"
            "       - 'object': object template name or generic name that appears in the memory's object templates or instances\n"
            "       - 'attribute': attribute path, e.g. 'skin.present'\n"
            "       - 'value': desired value\n"
            "       - 'aspect' (optional)\n"
            "   - Use object names that are present in the memory encoding.\n"
            "   - Do not invent a new object template unless absolutely necessary.\n\n"
            "Important:\n"
            "- Use nested attributes for tiered paths; do not use dotted strings.\n"
            "- For multiple instances of the same attribute, use arrays.\n"
            "- Ensure every object has dimensions, mass, shape, position.\n"
            "- Ensure all actions have correct action_category, including NATURAL_PROCESS.\n"
            "- Include sensory perception changes with sources.\n"
            "- Include categories, functions, materials for every object template.\n"
            "- Output ONLY valid JSON.\n"
            + (json_encode_demo if json_encode_demo else "")
        )

        self.json_review_system_msg = (
            "You are a JSON review expert. Compare the initial JSON encoding with the original NL memory and consider the question: 'Is the NL memory fully captured by the json encoding in its current form, i.e. every object, action, attribute and every state change?' Do not output an answer (text or json) to the question, but keep it in mind to perform the following checks and corrections. Your objective is to edit the initial json encoding based on the following, and output the entire edited encoding in full, which should be almost the same as the initial encoding just with further refinements. It should not be a summary or report, but the entire json encoding, just post review/editing. The checks and corrections you should consider are below:\n\n"
            "1. Alias consistency:\n"
            "   - Ensure object names match existing objects; unify any aliases to the canonical obj_id.\n"
            "   - Example: 'spiral_peel_strip' → 'peel_strip_01'.\n\n"
            "2. Sensory sources:\n"
            "   - Every sensory perception entry must have a 'source' field.\n"
            "   - The source must refer to an object or aspect present in the encoding.\n\n"
            "3. Missing state changes:\n"
            "   - Add any state changes that are missing compared to the NL memory. As an example, in terms of the NL memory of peeling an apple, in the demo file below, the following was added especially:\n"
            "       - juice_01.volume and position changes during juice_seepage\n"
            "       - apple_01.surface.coverage changes (droplets, film)\n"
            "       - apple_01.surface.indentations (fingerprints)\n"
            "       - apple_01.surface.slipperiness changes\n"
            "       - left_hand_01.surface.coverage (film_patch, droplets)\n"
            "       - knife_01.surface.coverage (film of apple_juice, volume 3 ml)\n\n"
            "4. Cyclical actions:\n"
            "   - Ensure every cyclical action has both 'changes_per_cycle' and 'changes_total' where applicable.\n\n"
            "5. Preconditions:\n"
            "   - Ensure they are separated from state changes and correctly placed on the appropriate action.\n"
            "   - As an example, in the demo below for the memory of peeling an apple, the following change was made: For draw_knife_along_surface, a pre-condition was added as, to draw it, knife blade needed to be in contact under the apple skin:\n"
            "       {'object': 'knife_01', 'attribute': 'blade_contact_point', 'value': 'under_skin'}\n\n"
            "6. Natural processes:\n"
            "   - Confirm that natural processes are present with correct action_category and preconditions/effects.\n"
            "   - As an example, in the demo below for the memory of peeling an apple, the following was included: juice_seepage, flesh_oxidation, stress_tear, light_passage, gravity_pull.\n\n"
            "7. Fine details from the NL memory needed to be added. As an example, in the demo below for the memory of peeling an apple, the following were added:\n"
            "   - Initial surface dampness: apple_skin_01.surface_moisture = slightly_damp\n"
            "   - Friction/slip resistance: apple_skin_01.surface_friction = moderate\n"
            "   - Peel strip shape: ['strip', 'spiral']\n"
            "   - Hand coverage: two coverage entries (film_patch of apple_juice and droplets of water)\n"
            "   - Apple surface coverage: droplets of apple_juice\n"
            "   - Peel strip underside: side_2.composition = apple_pith (or apple_flesh if more accurate)\n"
            "   - Apple core shape: five_pointed, visibility faint\n\n"
            "8. Remove any spurious references (e.g., refrigerator door sound if not in NL memory).\n\n"
            "9. Goal state consistency:\n"
            "   - Ensure every goal-state object matches an object template or instance in the memory.\n"
            "   - Ensure the attribute path and value in the goal state make sense for the object.\n"
            "   - If a goal-state object is a specific obj_id, change it to the object template/generic name where possible.\n"
            "   - If a goal-state object is missing from the memory, correct it to the closest matching existing object or remove it.\n\n"
            "10. If the encoding already satisfies all of the above, return it unchanged.\n\n"
            "Output ONLY the final JSON encoding (or the original if no changes needed). Return ONLY the final JSON object with the same top‑level keys as the initial encoding (object_templates, action_templates, memory, goal_state). Do not include any review commentary, change lists, or wrapper objects. Do not add keys like final_json_encoding, review_applied, remedy_steps_required, etc. The final json encoding you are outputting should be almost the same as the initial json encoding that you are reviewing, just with any changes you made after review. Please provide the final json encoding in full. \n"
            + (json_review_demo if json_review_demo else "")
        )

    async def run(self, initial_prompt):
        run_start_time = time.perf_counter()
        print(f"\n{'='*60}\n🚀 PHASE 1: Generating {INITIAL_BATCH_SIZE} initial memories...\n{'='*60}\n")
        initial_activities = await self._brainstorm_activities_normal(initial_prompt, INITIAL_BATCH_SIZE)
        print(f"  📋 Generated {len(initial_activities)} initial activity ideas\n")
        self.persistence.save_initial_activities(initial_activities)
        for act in initial_activities:
            self.proposed_activities.add(act)
            self.persistence.save_activity_to_history(act)

        all_activity_ideas = list(initial_activities)
        total_activities = len(all_activity_ideas)

        tasks = [self._generate_one_memory(act, "initial") for act in all_activity_ideas]
        await asyncio.gather(*tasks, return_exceptions=True)

        self.persistence.save_run_initial_search_terms(self.initial_batch_search_terms)
        self.expansion_pool = self.initial_batch_search_terms

        print(f"\n{'='*60}\n🔁 Expansion activity generation (target: {MAX_TOTAL_MEMORIES})...\n{'='*60}\n")
        expansion_activities = []
        while total_activities < MAX_TOTAL_MEMORIES:
            term = self._random_pick_single_term()
            if not term:
                print("  ⚠️ No searchable terms available yet; waiting...")
                await asyncio.sleep(5)
                continue
            proposed = await self._propose_expansion_activities_normal(term)
            if proposed:
                for act in proposed:
                    self.proposed_activities.add(act)
                    self.persistence.save_activity_to_history(act)
                expansion_activities.extend(proposed)
                total_activities += len(proposed)
                print(f"    🎲 term '{term}' → {len(proposed)} activities (total ideas: {total_activities})")
            else:
                print(f"    🎲 term '{term}' → no proposals, retrying")

        print(f"\n  📋 Total activities to generate: {total_activities}\n")
        print(f"{'='*60}\n🧠 Generating expansion memories...\n{'='*60}\n")
        tasks = [self._generate_one_memory(act, "expansion") for act in expansion_activities]
        await asyncio.gather(*tasks, return_exceptions=True)

        run_end_time = time.perf_counter()
        total_elapsed = run_end_time - run_start_time
        print(f"\n🏁 Pipeline finished! Total memories: {self.generated_count}, Success: {self.success_count}, Failed: {self.fail_count}")
        if self.generation_times:
            avg = sum(self.generation_times) / len(self.generation_times)
            print(f"⏱️ Average generation time: {avg:.1f}s over {len(self.generation_times)} memories")
        print(f"⏱️ Total run elapsed time: {total_elapsed:.1f}s")

        self.persistence.save_run_summary_readable(self.run_timings, total_elapsed)

    async def _brainstorm_activities_normal(self, prompt, count):
        if self.proposed_activities:
            exclusion = "\n".join(sorted(self.proposed_activities))
            exclusion_block = f"\n\nPreviously proposed activities (DO NOT repeat any of these):\n{exclusion}\n"
        else:
            exclusion_block = ""
        system_msg = (
            "You are a creative activity generator. Output ONLY valid JSON.\n"
            "Each activity must be reasonably specific, e.g., 'making pizza'.\n"
            "Do NOT propose overly general activities like 'making dinner',\n"
            "because these are massively variable and not suitable for a single memory.\n"
            "Output ONLY valid JSON in the exact format:\n"
            '{"activities": ["activity1", "activity2", ...]}\n'
            + exclusion_block
        )
        user_msg = f"{prompt}\nGenerate {count} specific activities."
        messages = [{"role":"system","content": system_msg}, {"role":"user","content": user_msg}]
        try:
            r = await self.llm.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                response_format={"type":"json_object"}, temperature=0.8,
                max_tokens=OUTPUT_MAX_TOKENS
            )
            content = r.choices[0].message.content
            if self.api_logger:
                self.api_logger.log("activity_generation", "brainstorm_initial", messages, content, r.usage)
            data = json.loads(content) if isinstance(content, str) else content
            if isinstance(data, dict):
                if "activities" in data:
                    activities = data["activities"]
                    if isinstance(activities, list):
                        return [a if isinstance(a, str) else str(a) for a in activities]
                if "activity" in data and isinstance(data["activity"], str):
                    return [data["activity"]]
            return []
        except Exception as e:
            print(f"  ❌ Initial brainstorm failed: {e}")
            return []

    async def _propose_expansion_activities_normal(self, term):
        if self.proposed_activities:
            exclusion = "\n".join(sorted(self.proposed_activities))
            exclusion_block = f"\n\nPreviously proposed activities (DO NOT repeat any of these):\n{exclusion}\n"
        else:
            exclusion_block = ""
        system_msg = (
            "You are a creative activity generator. Output ONLY valid JSON.\n"
            "Propose up to 2 highly plausible, reasonably specific activities "
            "that involve the given search term. Avoid overly general activities.\n"
            "Output ONLY valid JSON in the exact format:\n"
            '{"activities": ["activity1", "activity2"]}\n'
            + exclusion_block
        )
        user_msg = f"Search term: {term}\nPropose up to 2 specific activities involving this term."
        messages = [{"role":"system","content": system_msg}, {"role":"user","content": user_msg}]
        try:
            r = await self.llm.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                response_format={"type":"json_object"}, temperature=0.8,
                max_tokens=OUTPUT_MAX_TOKENS
            )
            content = r.choices[0].message.content
            if self.api_logger:
                self.api_logger.log("activity_generation", "expansion_activity", messages, content, r.usage)
            data = json.loads(content) if isinstance(content, str) else content
            if isinstance(data, dict):
                if "activities" in data:
                    activities = data["activities"]
                    if isinstance(activities, list):
                        return [a if isinstance(a, str) else str(a) for a in activities][:2]
                if "activity" in data and isinstance(data["activity"], str):
                    return [data["activity"]]
            return []
        except Exception as e:
            print(f"    ❌ Expansion activity generation failed for '{term}': {e}")
            return []

    def _random_pick_single_term(self):
        if self.expansion_pool:
            return random.choice(list(self.expansion_pool))
        return None

    async def _generate_one_memory(self, activity, domain):
        async with self.semaphore:
            if self.generated_count >= MAX_TOTAL_MEMORIES:
                return

            start_time = time.perf_counter()

            nl = await self._write_nl_memory(activity)
            if not nl:
                print(f"    ❌ [{activity}] NL generation failed"); self.fail_count += 1; return
            print(f"    ✓ [{activity}] NL ({len(nl)} chars)")

            timings = {}
            record = await self._encode_memory_advanced(nl, activity, timings)
            if not record:
                print(f"    ❌ [{activity}] Multi‑step encoding failed"); self.fail_count += 1; return

            final_json = record.get("final_json")
            if not final_json:
                print(f"    ❌ [{activity}] Final JSON missing"); self.fail_count += 1; return

            final_json["activity"] = activity

            async with self.id_lock:
                memory_id = f"mem_{self.generated_count:06d}"
                self.generated_count += 1

            final_json.pop("memory_id", None)
            final_json.pop("activity", None)
            ordered_final = {
                "activity": activity,
                "memory_id": memory_id,
                **final_json
            }
            final_json = ordered_final
            record["memory_id"] = memory_id

            try:
                ingest_start = time.perf_counter()
                ingest_json = copy.deepcopy(final_json)
                memory = await self._ingest(ingest_json, timings)
                ingest_elapsed = time.perf_counter() - ingest_start
                timings.setdefault("ingest_total", {"duration_seconds": ingest_elapsed})

                post_start = time.perf_counter()
                self._post_process(memory, final_json, record)
                post_elapsed = time.perf_counter() - post_start
                timings["post_process"] = {"duration_seconds": post_elapsed}

                self.success_count += 1
                elapsed = time.perf_counter() - start_time
                self.generation_times.append(elapsed)
                record["elapsed_seconds"] = round(elapsed, 2)
                record["timings"] = timings

                memory_terms = getattr(memory, "indexed_terms", set())
                filtered_terms = {t for t in memory_terms if not t.startswith("activity:")}
                self.persistence.save_all_search_terms_history(memory_id, memory_terms)
                if domain == "initial":
                    self.initial_batch_search_terms.update(filtered_terms)

                self.run_timings.append({
                    "memory_id": memory_id,
                    "activity": activity,
                    "stages": timings
                })

                self.persistence.save_stage_timings_jsonl(memory_id, activity, timings)

                # Save class-object pickle version
                self.persistence.save_memory_object(memory)
                self.persistence.save_storage_objects(self.object_storage, self.action_storage)

                print(f"    ✅ [{activity}] Memory created ({memory.id}) in {elapsed:.1f}s")
            except Exception as e:
                self.fail_count += 1
                print(f"    ❌ [{activity}] EXCEPTION: {type(e).__name__}: {e}")
                traceback.print_exc()

    async def _write_nl_memory(self, activity):
        user_msg = f"Write a memory of: {activity}\n"
        content, _ = await self._call_llm_high("nl_generation", self.nl_system_msg, user_msg, activity, max_tokens=OUTPUT_MAX_TOKENS)
        return content

    async def _encode_memory_advanced(self, nl_text, activity, timings: dict):
        # Stage 1: Memory sketch
        t0 = time.perf_counter()
        sketch, sketch_usage = await self._call_llm_high("memory_sketch", self.sketch_system_msg, nl_text, activity)
        timings["memory_sketch"] = {
            "duration_seconds": time.perf_counter() - t0,
            "usage": sketch_usage
        }
        if not sketch:
            print("    ❌ memory_sketch failed or returned empty")
            return None

        # Stage 2: Enhanced sketch
        t0 = time.perf_counter()
        enhanced, enhanced_usage = await self._call_llm_high("enhance_sketch", self.enhance_system_msg, sketch, activity)
        timings["enhanced_sketch"] = {
            "duration_seconds": time.perf_counter() - t0,
            "usage": enhanced_usage
        }
        if enhanced:
            sketch = enhanced
        else:
            print("    ⚠️ enhance_sketch returned empty, using previous")

        # Stage 3: Generalisation
        t0 = time.perf_counter()
        generalized, gen_usage = await self._call_llm_high("generalize", self.generalize_system_msg, sketch, activity)
        timings["generalization"] = {
            "duration_seconds": time.perf_counter() - t0,
            "usage": gen_usage
        }
        if generalized:
            sketch = generalized
        else:
            print("    ⚠️ generalize returned empty, using previous")

        # Stage 4: State changes
        t0 = time.perf_counter()
        annotated, state_usage = await self._call_llm_high("state_changes", self.state_change_system_msg, sketch, activity)
        timings["state_changes"] = {
            "duration_seconds": time.perf_counter() - t0,
            "usage": state_usage
        }
        if annotated:
            sketch = annotated
        else:
            print("    ⚠️ state_changes returned empty, using previous")

        # Stage 5: JSON encoding
        t0 = time.perf_counter()
        json_text, json_usage = await self._call_llm_high("json_encode_initial", self.json_encode_system_msg, sketch, activity, max_tokens=OUTPUT_MAX_TOKENS)
        timings["json_encoding"] = {
            "duration_seconds": time.perf_counter() - t0,
            "usage": json_usage
        }
        if not json_text:
            print("    ❌ json_encode_initial failed or returned empty")
            return None
        initial_json = self._parse_json(json_text)
        if not initial_json:
            print("    ❌ initial JSON parse failed")
            return None

        # Stage 6: JSON review
        t0 = time.perf_counter()
        review_user_msg = f"Original NL memory:\n{nl_text}\n\nInitial JSON encoding:\n{json_text}"
        review_text, review_usage = await self._call_llm_high("json_review", self.json_review_system_msg, review_user_msg, activity, max_tokens=OUTPUT_MAX_TOKENS)
        timings["json_review"] = {
            "duration_seconds": time.perf_counter() - t0,
            "usage": review_usage
        }

        review_json = self._parse_json(review_text) if review_text else None
        final_json = self._validate_review_output(review_json, initial_json, timings)

        record = {
            "activity": activity,
            "nl_text": nl_text,
            "memory_sketch": sketch,
            "enhanced_sketch": enhanced,
            "generalized_sketch": generalized,
            "state_annotated_sketch": annotated,
            "initial_json": initial_json,
            "json_review": review_text,
            "final_json": final_json
        }
        return record

    def _validate_review_output(self, review_json, initial_json, timings):
        candidate = None
        if isinstance(review_json, dict):
            if "final_json_encoding" in review_json:
                candidate = review_json["final_json_encoding"]
            else:
                candidate = review_json

        if not isinstance(candidate, dict):
            return initial_json
        if not (("object_templates" in candidate) and ("action_templates" in candidate) and ("memory" in candidate)):
            return initial_json

        memory_block = candidate.get("memory")
        if not isinstance(memory_block, dict) or ("objects" not in memory_block and "actions" not in memory_block):
            return initial_json

        init_usage = timings.get("json_encoding", {}).get("usage")
        review_usage = timings.get("json_review", {}).get("usage")
        init_output_tokens = self._output_tokens(init_usage)
        review_output_tokens = self._output_tokens(review_usage)
        if init_output_tokens is not None and review_output_tokens is not None:
            if review_output_tokens < REVIEW_MIN_OUTPUT_RATIO * init_output_tokens:
                return initial_json

        return candidate

    def _output_tokens(self, usage_dict):
        if not isinstance(usage_dict, dict):
            return None
        try:
            completion = int(usage_dict.get("completion_tokens", 0) or 0)
            reasoning = int(usage_dict.get("reasoning_tokens", 0) or 0)
            return max(0, completion - reasoning)
        except (TypeError, ValueError):
            return None

    def _parse_json(self, text):
        if not isinstance(text, str):
            return None
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except:
            return None

    async def _call_llm_high(self, purpose, system_msg, user_msg, activity, max_tokens=OUTPUT_MAX_TOKENS):
        messages = [
            {"role":"system","content": system_msg},
            {"role":"user","content": user_msg}
        ]
        try:
            r = await self.llm.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                temperature=0.4, max_tokens=max_tokens,
                extra_body={"reasoning_effort": "high"}
            )
            content = r.choices[0].message.content.strip()
            usage = r.usage
            usage_dict = usage.model_dump() if hasattr(usage, 'model_dump') else dict(usage or {})
            if self.api_logger:
                self.api_logger.log(activity, purpose, messages, content, r.usage)
            return content, usage_dict
        except Exception as e:
            print(f"    ❌ {purpose} call failed: {e}")
            return None, None

    def _normalize_template_name(self, name: Any) -> str:
        if not isinstance(name, str):
            name = str(name)
        return name.lower().strip().replace("_", " ").replace("-", " ")

    def _find_template_name(self, name: str) -> Optional[str]:
        query_norm = self._normalize_template_name(name)
        for template_name in self.object_storage.list_templates():
            if self._normalize_template_name(template_name) == query_norm:
                return template_name
        for template_name in self.object_storage.list_templates():
            for alias in self.object_storage.get_aliases(template_name):
                if self._normalize_template_name(alias) == query_norm:
                    return template_name
        return None

    def _resolve_goal_object(self, memory: 'Memory', object_name: str) -> Object:
        if object_name in memory.objects:
            return memory.objects[object_name]

        for obj in memory.objects.values():
            if obj.template_name == object_name:
                return obj

        canonical_template = self._find_template_name(object_name)
        if canonical_template:
            for obj in memory.objects.values():
                if obj.template_name == canonical_template:
                    return obj
            obj = Object(
                template_name=canonical_template,
                obj_id=f"{canonical_template}_goal_01",
                overrides={},
            )
            self._fill_defaults(obj)
            self._enforce_object_completeness(obj)
            memory.add_object(obj)
            return obj

        normalized = self._normalize_template_name(object_name).replace(" ", "_")
        fallback = Object(
            template_name=object_name,
            obj_id=f"{normalized}_goal_01",
            overrides={"unresolved": True},
        )
        memory.add_object(fallback)
        return fallback

    async def _ingest(self, json_data, timings: Optional[dict] = None):
        t0 = time.perf_counter()
        self.template_matcher._build_index()
        if timings is not None:
            timings["template_matcher_rebuild"] = {"duration_seconds": time.perf_counter() - t0}

        memory_data = json_data.get("memory", json_data)

        t0 = time.perf_counter()
        obj_templates = json_data.get("object_templates")
        if isinstance(obj_templates, dict):
            for name, tmpl in obj_templates.items():
                if isinstance(tmpl, dict):
                    defaults = tmpl.get("defaults") or {}
                    functions = tmpl.get("functions") or []
                    aliases = tmpl.get("aliases") or []
                    categories = tmpl.get("categories") or []
                    materials = tmpl.get("materials") or tmpl.get("material") or []
                    if isinstance(materials, str):
                        materials = [materials]
                    self.object_storage.add_template(name, defaults, functions=functions, aliases=aliases, categories=categories, materials=materials)
        else:
            print("    ⚠️ object_templates is not a dict, skipping")

        act_templates = json_data.get("action_templates")
        if isinstance(act_templates, dict):
            for name, tmpl in act_templates.items():
                if isinstance(tmpl, dict):
                    self.action_storage.add_template(
                        name,
                        default_duration=tmpl.get("default_duration", 1),
                        action_category=tmpl.get("action_category", "HUMAN_INTERACTION"),
                        kinematic_trajectory=tmpl.get("kinematic_trajectory", "straight"),
                        sub_actions=tmpl.get("sub_action_templates") or None,
                        **{k:v for k,v in tmpl.items() if k not in ("default_duration","action_category","kinematic_trajectory","sub_action_templates")}
                    )
        else:
            print("    ⚠️ action_templates is not a dict, skipping")
        if timings is not None:
            timings["top_level_template_processing"] = {"duration_seconds": time.perf_counter() - t0}

        t0 = time.perf_counter()
        for obj_entry in (memory_data.get("objects") or []):
            if not isinstance(obj_entry, dict):
                continue
            proposed = obj_entry.get("template")
            if not proposed:
                continue
            match = self.template_matcher.find_matching_object(proposed)
            if match:
                obj_entry["template"] = match
            else:
                self.object_storage.add_template(proposed, obj_entry.get("overrides") or {})
                self.new_objects.add(proposed)
        for ad in (memory_data.get("actions") or []):
            if isinstance(ad, dict):
                self._resolve_action_templates(ad)
        if timings is not None:
            timings["template_resolution"] = {"duration_seconds": time.perf_counter() - t0}

        t0 = time.perf_counter()
        protag = memory_data.get("protagonist") or {}
        protagonist = Protagonist(protag.get("obj_id", "person_01"))
        if "overrides" in protag:
            protagonist.overrides.update(protag["overrides"] or {})
        if "anatomical_features" in protag:
            protagonist.anatomical_features = protag["anatomical_features"] or []
        mem = Memory(json_data.get("memory_id", f"mem_{self.generated_count:06d}"), protagonist)
        mem.activity = json_data.get("activity", "")

        for od in (memory_data.get("objects") or []):
            if not isinstance(od, dict):
                continue
            template_name = od.get("template")
            obj_id = od.get("obj_id")
            if not template_name or not obj_id:
                continue

            overrides = od.get("overrides") or {}
            dimensions = od.get("dimensions") or {}
            weight = od.get("weight") or od.get("mass") or {}
            mass = od.get("mass") or od.get("weight") or {}
            position = od.get("position") or {}
            position_relative_body = od.get("position_relative_body") or {}
            motion = od.get("motion") or {}
            shape = od.get("shape", "")
            aliases = od.get("aliases") or []
            composite_of = od.get("composite_of") or []
            functions = od.get("functions") or []

            obj = Object(
                template_name=template_name,
                obj_id=obj_id,
                number=od.get("number", "single"),
                position=position,
                position_relative_body=position_relative_body,
                dimensions=dimensions,
                weight=weight,
                shape=shape,
                motion=motion,
                overrides=overrides,
                aliases=aliases,
                composite_of=composite_of,
                functions=functions,
            )
            effective_attrs = {**overrides, "dimensions": dimensions, "mass": mass, "weight": weight, "shape": shape, "position": position, "position_relative_body": position_relative_body, "motion": motion}
            self.object_storage.expand_template_from_instance_attributes(template_name, effective_attrs)

            self._fill_defaults(obj)
            self._enforce_object_completeness(obj)
            mem.add_object(obj)

        for ad in (memory_data.get("actions") or []):
            if not isinstance(ad, dict):
                continue
            action = self.action_storage.instantiate(ad.get("template"), ad)
            action.kinematic_trajectory = ad.get("kinematic_trajectory", "straight")
            action.temporal_type = ad.get("temporal_type", "sequential")
            action.repetitions = ad.get("repetitions", 1)
            action.cycle_duration = ad.get("cycle_duration", 0.0)
            participants = []
            for pid in (ad.get("participants") or []):
                if pid in mem.objects:
                    participants.append(mem.objects[pid])
                else:
                    obj = Object(pid, pid)
                    mem.objects[pid] = obj
                    participants.append(obj)
            action.participants = participants
            if "changes" in ad:
                action.changes = self._convert_changes(ad["changes"])
            if "changes_per_cycle" in ad:
                action.changes_per_cycle = self._convert_changes(ad["changes_per_cycle"])
            if "changes_total" in ad:
                action.changes_total = self._convert_changes(ad["changes_total"])
            action.conditional_changes = ad.get("conditional_changes") or []
            action.preconditions = ad.get("preconditions") or []
            action.tags = ad.get("tags") or []
            if "sub_actions" in ad and ad["sub_actions"] is not None:
                action.sub_actions = self._build_sub_actions(ad["sub_actions"], mem)
            mem.add_action(action)

        # Goal state ingestion – resolved to runtime Object instances
        goal_list = json_data.get("goal_state") or []
        if isinstance(goal_list, list):
            for goal_data in goal_list:
                if not isinstance(goal_data, dict):
                    continue
                object_name = goal_data.get("object")
                if not object_name:
                    continue
                target_obj = self._resolve_goal_object(mem, str(object_name))
                mem.goal_states.append(
                    GoalState(
                        target_object=target_obj,
                        attribute=goal_data.get("attribute", "?"),
                        value=goal_data.get("value"),
                        aspect=goal_data.get("aspect"),
                        op=goal_data.get("op", "eq"),
                    )
                )

        if timings is not None:
            timings["memory_construction"] = {"duration_seconds": time.perf_counter() - t0}

        t0 = time.perf_counter()
        for i, action in enumerate(mem.actions):
            mem.actions[i] = await self.decomposer.decompose_action(action)
        if timings is not None:
            timings["decomposer"] = {"duration_seconds": time.perf_counter() - t0}

        t0 = time.perf_counter()
        indexed_terms = self.search_engine.add_memory(mem)
        mem.indexed_terms = indexed_terms
        if timings is not None:
            timings["search_indexing"] = {"duration_seconds": time.perf_counter() - t0}

        return mem

    def _resolve_action_templates(self, act):
        tmpl = act.get("template")
        traj = act.get("kinematic_trajectory", "straight")
        sub_actions = act.get("sub_actions")
        if tmpl:
            match = self.template_matcher.find_matching_action(tmpl, traj, sub_actions)
            if match:
                act["template"] = match
            else:
                sub_seq = [sa.get("template") for sa in (sub_actions or [])] if sub_actions else []
                self.action_storage.add_template(tmpl, sub_actions=sub_seq if sub_seq else None)
                self.new_actions.add(tmpl)
        if sub_actions:
            for sa in sub_actions:
                if isinstance(sa, dict):
                    self._resolve_action_templates(sa)

    def _convert_changes(self, changes):
        result = []
        for ch in (changes or []):
            result.append(ch if isinstance(ch, dict) else ch)
        return result

    def _build_sub_actions(self, sub_list, mem):
        if not isinstance(sub_list, list):
            return []
        result = []
        for s in sub_list:
            if not isinstance(s, dict):
                continue
            action = self.action_storage.instantiate(s.get("template"), s)
            action.kinematic_trajectory = s.get("kinematic_trajectory", "straight")
            action.temporal_type = s.get("temporal_type", "sequential")
            action.repetitions = s.get("repetitions", 1)
            action.cycle_duration = s.get("cycle_duration", 0.0)
            participants = []
            for pid in (s.get("participants") or []):
                if pid in mem.objects:
                    participants.append(mem.objects[pid])
                else:
                    obj = Object(pid, pid)
                    mem.objects[pid] = obj
                    participants.append(obj)
            action.participants = participants
            if "changes" in s:
                action.changes = self._convert_changes(s["changes"])
            if "changes_per_cycle" in s:
                action.changes_per_cycle = self._convert_changes(s["changes_per_cycle"])
            if "changes_total" in s:
                action.changes_total = self._convert_changes(s["changes_total"])
            action.conditional_changes = s.get("conditional_changes") or []
            action.preconditions = s.get("preconditions") or []
            action.tags = s.get("tags") or []
            if "sub_actions" in s and isinstance(s["sub_actions"], list):
                action.sub_actions = self._build_sub_actions(s["sub_actions"], mem)
            result.append(action)
        return result

    def _fill_defaults(self, obj):
        template = self.object_storage.get_defaults(obj.template_name)
        if not template:
            return
        for field in ["dimensions", "position", "position_relative_body", "weight", "motion"]:
            if not getattr(obj, field, None):
                default_val = template.get(field)
                if default_val:
                    setattr(obj, field, copy.deepcopy(default_val))
        if not obj.shape:
            default_shape = template.get("shape", "")
            if default_shape:
                obj.shape = default_shape

    def _enforce_object_completeness(self, obj):
        template = self.object_storage.get_defaults(obj.template_name)
        default_dims = template.get("dimensions") or {}
        obj.dimensions = obj.dimensions or {}
        for dim in ["length", "width", "height_thickness"]:
            if not obj.dimensions.get(dim):
                if dim in default_dims:
                    obj.dimensions[dim] = copy.deepcopy(default_dims[dim])
                else:
                    obj.dimensions[dim] = {"value": 0, "unit": "unknown"}
        if not obj.weight:
            default_weight = template.get("weight")
            if default_weight:
                obj.weight = copy.deepcopy(default_weight)
            else:
                obj.weight = {"value": 0, "unit": "unknown"}
        if not obj.shape:
            default_shape = template.get("shape", "")
            obj.shape = default_shape if default_shape else "unknown"
        if not obj.position:
            default_position = template.get("position")
            if default_position:
                obj.position = copy.deepcopy(default_position)
            else:
                obj.position = {"relation": "unknown", "relative_to": "none"}

    def _post_process(self, memory, final_json, record):
        terms = getattr(memory, "indexed_terms", set())
        self.all_searchable_terms.update(terms)
        for a in memory.actions:
            self.new_actions.add(a.template)
            for s in a.sub_actions:
                self.new_actions.add(s.template)
        for obj in memory.objects.values():
            self.new_objects.add(obj.template_name)
        self.persistence.save_final_json(final_json)
        self.persistence.save_record(record)
        self.persistence.save_object_storage(self.object_storage)
        self.persistence.save_action_storage(self.action_storage)
        self.persistence.save_pipeline_state(self._pipeline_state())
        self.persistence.maybe_snapshot(self.generated_count)

    def _pipeline_state(self):
        return {
            "generated_count": self.generated_count,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "all_searchable_terms": list(self.all_searchable_terms),
            "new_actions": list(self.new_actions),
            "new_objects": list(self.new_objects)
        }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
async def main():
    print("\n" + "="*60 + "\n  Autonomous Memory Generation Pipeline\n" + "="*60)
    print(f"  Data: {DATA_DIR}\n  Model: {MODEL_NAME}\n  Max concurrent: {MAX_CONCURRENT}\n  Target: {MAX_TOTAL_MEMORIES}\n")

    def load_demo(filename):
        path = DEMO_DIR / filename
        return path.read_text(encoding="utf-8") if path.exists() else ""

    action_plot_demo = load_demo("action_plot_demo.txt")
    decompose_demo = load_demo("decompose_demo.txt")
    generalize_demo = load_demo("generalize_demo.txt")
    state_change_demo = load_demo("state_change_demo.txt")
    json_encode_demo = load_demo("json_encode_demo.txt")
    json_review_demo = load_demo("json_review_demo.txt")
    nl_examples = load_demo("nl_examples.txt")
    extra_context = load_demo("context.txt")

    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    try:
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"user","content":"ok"}],
            max_tokens=5
        )
        print("✅ API connection OK")
    except Exception as e:
        print(f"❌ API failed: {e}"); return

    persistence = PersistenceManager(DATA_DIR, SNAPSHOT_MILESTONES)

    obj_storage, act_storage = persistence.load_storage_objects()

    if obj_storage is None:
        obj_storage = ObjectStorage()
    if act_storage is None:
        act_storage = ActionStorage()

    if not obj_storage.list_templates():
        obj_data = persistence.load_object_storage()
        if obj_data:
            for name, tmpl in obj_data.items():
                if not isinstance(tmpl, dict):
                    continue
                defaults = {k: v for k, v in tmpl.items() if k not in ("categories", "functions", "aliases", "materials")}
                cats = tmpl.get("categories", [])
                funcs = tmpl.get("functions", [])
                aliases = tmpl.get("aliases", [])
                mats = tmpl.get("materials") or tmpl.get("material") or []
                if isinstance(mats, str):
                    mats = [mats]
                obj_storage.add_template(name, defaults, functions=funcs, aliases=aliases, categories=cats, materials=mats)
    if not act_storage.list_templates():
        act_data = persistence.load_action_storage()
        if act_data:
            act_storage._templates = act_data.get("templates", {})
            act_storage._sub_action_sequences = act_data.get("sub_action_sequences", {})

    se = SearchEngine(MAX_TOTAL_MEMORIES)

    if not obj_storage.list_templates():
        obj_storage.add_template("hand", {
            "configuration.grip.type": "open", "motion": "idle",
            "surface_moisture": "dry", "temperature": 36.0, "colour": "flesh",
            "mass": {"value":0.4,"unit":"kg"},
            "dimensions":{"length":{"value":18,"unit":"cm"},"width":{"value":8,"unit":"cm"},"height_thickness":{"value":2,"unit":"cm"}}
        }, categories=["hand", "body_part", "physical_object", "object"], functions=["grasping", "holding"], materials=["flesh"])
        obj_storage.add_template("apple", {
            "color":"red","shape":"spherical","sweetness":0.8,"flesh_color":"cream",
            "typical_weight":{"value":200,"unit":"g"},
            "composition_templates":["apple_skin","apple_flesh","apple_core"],
            "dimensions":{"length":{"value":8,"unit":"cm"},"width":{"value":8,"unit":"cm"},"height_thickness":{"value":8,"unit":"cm"}}
        }, categories=["apple", "fruit", "food", "physical_object", "object"], functions=["food", "snack"], materials=["apple_flesh", "apple_skin"])
    if not act_storage.list_templates():
        act_storage.add_template("grasp", kinematic_trajectory="enveloping", default_duration=2)
        act_storage.add_template("cut", kinematic_trajectory="straight", default_duration=5)

    match_logger = TemplateMatchLogger(Path(DATA_DIR) / "main" / "template_match_log.jsonl")
    template_matcher = TemplateMatcher(obj_storage, act_storage, match_logger)

    if template_matcher.is_active():
        print("✅ TemplateMatcher indexes ready.")
    else:
        print("⚠️ TemplateMatcher indexes unavailable: check rank-bm25 and sentence-transformers installation.")

    api_logger = APILogger(Path(DATA_DIR) / "main" / "api_log.jsonl")
    gen = MemoryGenerationPipeline(
        client, obj_storage, act_storage, se, persistence,
        template_matcher=template_matcher,
        nl_examples_text=nl_examples,
        extra_context=extra_context,
        action_plot_demo=action_plot_demo,
        decompose_demo=decompose_demo,
        generalize_demo=generalize_demo,
        state_change_demo=state_change_demo,
        json_encode_demo=json_encode_demo,
        json_review_demo=json_review_demo,
        api_logger=api_logger
    )
    await gen.run("household activities: kitchen tasks, gardening, and home maintenance")


if __name__ == "__main__":
    asyncio.run(main())