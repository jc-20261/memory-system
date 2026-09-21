#!/usr/bin/env python3
r"""
pickle_loader.py – Loads pickle memories, storage, and alias expansion file.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

from memory_model import Memory

EXPLORE_DIR = Path(__file__).parent
PICKLE_MEMORIES_DIR = EXPLORE_DIR / "pickle_memories"
PICKLE_STORAGE_DIR = EXPLORE_DIR / "pickle_storage"
ALIAS_FILE = EXPLORE_DIR / "data" / "alias_expansion.json"


class PickleMemoryLoader:
    def __init__(self):
        self.memories: List[Memory] = []
        self.object_storage: Dict[str, Any] = {}
        self.action_storage_general: Dict[str, Any] = {}
        self.action_storage_alts: Dict[str, Any] = {}
        self.verb_storage: Dict[str, Any] = {}
        self.alias_expansion: Dict[str, Any] = {}
        self._load_all()

    def _load_all(self):
        print("Loading pickle memories...")
        memory_files = sorted(PICKLE_MEMORIES_DIR.glob("*.pkl"))
        if not memory_files:
            print(f"⚠️ No pickle memories found in {PICKLE_MEMORIES_DIR}")
        for pkl in memory_files:
            with open(pkl, "rb") as f:
                mem = pickle.load(f)
            self.memories.append(mem)
        print(f"  Loaded {len(self.memories)} memories.")

        print("Loading storage pickles...")
        obj_path = PICKLE_STORAGE_DIR / "object_storage.pkl"
        if obj_path.exists():
            with open(obj_path, "rb") as f:
                self.object_storage = pickle.load(f)
        else:
            print(f"⚠️ Object storage pickle not found: {obj_path}")

        action_path = PICKLE_STORAGE_DIR / "action_storage_linked.pkl"
        if action_path.exists():
            with open(action_path, "rb") as f:
                data = pickle.load(f)
            self.action_storage_general = data.get("general_templates", {})
            self.action_storage_alts = data.get("alts", {})
        else:
            print(f"⚠️ Action storage pickle not found: {action_path}")

        verb_path = PICKLE_STORAGE_DIR / "verb_storage.pkl"
        if verb_path.exists():
            with open(verb_path, "rb") as f:
                self.verb_storage = pickle.load(f)
        else:
            print(f"⚠️ Verb storage pickle not found: {verb_path}")

        print("Loading alias expansion file...")
        if ALIAS_FILE.exists():
            with open(ALIAS_FILE, "r", encoding="utf-8") as f:
                self.alias_expansion = json.load(f)
        else:
            print(f"⚠️ Alias expansion file not found: {ALIAS_FILE}")

        print("  Storage loaded.")