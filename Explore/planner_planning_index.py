#!/usr/bin/env python3
r"""
planner_planning_index.py – Secondary planning indices + object substitution lookup.

Builds lookup structures for planning without repeatedly calling the search
engine, and provides the object-compatibility lookup used by the condition
component and the world-state validator.

Primary structures (unchanged from earlier version):
    verb_to_actions       : verb -> list of action instance IDs
    effect_index          : (object, attr_path, op, value) -> list of action IDs
    precondition_index    : same key format -> list of action IDs
    old_state_index       : same key format -> list of action IDs (old states)
    action_by_id          : action_instance_id -> ActionInstance
    object_by_id          : object_id -> Object
    object_templates      : template_name -> template defaults from object_storage

New in this version:
    helpers                : PlannerMatchHelpers instance
    memory_by_id           : memory_id -> Memory
    get_memory_objects     : runtime Object instances for a memory
    get_memory_id_for_action : memory_id from an action's instance_id
    find_object_substitute         : best memory object for a query object
    find_object_substitutes_batch  : all candidates above threshold
    get_memory_object_compat_dicts : flat dicts for object_compatibility

Substitution parameters (all overridable by Planner):
    substitution_min_threshold  : minimum object_compatibility score accepted
    substitution_max_candidates : cap on candidates considered per lookup
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from memory_model import Memory, ActionInstance, Object
from pickle_search_index import PickleSearchIndex
from planner_match_helpers import PlannerMatchHelpers


class PlanningIndex:
    """Secondary index for planning, built from a PickleSearchIndex instance."""

    # ------------------------------------------------------------------
    # Tuning parameters
    # ------------------------------------------------------------------
    memory_batch_size: int = 10
    max_memory_batches: Optional[int] = None
    planning_depth_limit: int = 60
    max_total_actions: int = 100

    # Verb component
    target_verb_cap: int = 20
    context_verb_cap: int = 10
    verb_exact_bonus: float = 1.0
    verb_fuzzy_bonus: float = 0.9

    # Object comparison
    category_match_bonus: float = 0.5
    attribute_value_match_bonus: float = 0.1
    dimension_unit_match_bonus: float = 0.2
    position_unit_match_bonus: float = 0.2
    exact_object_match_bonus: float = 0.5
    fuzzy_object_match_bonus: float = 0.25
    action_category_bonus: float = 0.3

    # State / condition component
    candidate_actions_per_subgoal: int = 10
    subgoal_match_threshold: float = 0.3
    precondition_match_threshold: float = 0.1
    old_state_match_bonus: float = 0.2
    goal_state_match_bonus: float = 0.5
    position_unit_match_bonus_cond: float = 0.2

    # Plan scoring
    verb_score_weight: float = 0.3
    object_score_weight: float = 0.2
    goal_satisfaction_weight: float = 0.4
    action_count_penalty: float = 0.005
    validation_error_penalty: float = 0.005
    plan_score_threshold: float = 0.0

    # Substitution
    substitution_min_threshold: float = 0.0
    substitution_max_candidates: int = 20

    # World state validation
    missing_object_penalty: float = 1.0
    precondition_violation_penalty: float = 1.0
    spatial_inconsistency_penalty: float = 1.0
    type_mismatch_penalty: float = 1.0

    # Defining keys used to separate instance attributes from the six
    # class-defining signals when building the object_compatibility dicts.
    _DEFINING_KEYS = {
        "categories", "materials", "functions", "shape",
        "dimensions", "position", "position_relative_body",
    }

    def __init__(self, search_index: PickleSearchIndex):
        self.search_index = search_index

        # Primary structures
        self.verb_to_actions: Dict[str, List[str]] = defaultdict(list)
        self.effect_index: Dict[Tuple, List[str]] = defaultdict(list)
        self.precondition_index: Dict[Tuple, List[str]] = defaultdict(list)
        self.old_state_index: Dict[Tuple, List[str]] = defaultdict(list)
        self.action_by_id: Dict[str, ActionInstance] = {}
        self.object_by_id: Dict[str, Object] = {}
        self.object_templates: Dict[str, Dict[str, Any]] = search_index.object_storage

        # Memory-level lookups
        self.memories: List[Memory] = list(search_index.memories)
        self.memory_by_id: Dict[str, Memory] = {m.id: m for m in self.memories}

        # Match helpers (self-contained copy of the engine primitives)
        alias_expansion = getattr(search_index, "alias_expansion", {}) or {}
        self.helpers = PlannerMatchHelpers(alias_expansion)

        # Build indices
        self._build_from_memories(self.memories)

    def _build_from_memories(self, memories: List[Memory]):
        for memory in memories:
            for obj in memory.objects.values():
                self.object_by_id[obj.obj_id] = obj
            for action in memory.actions:
                self._index_action(action)

    def _index_action(self, action: ActionInstance, depth: int = 0):
        if action is None:
            return

        instance_id = action.instance_id
        self.action_by_id[instance_id] = action

        # Verb indexing
        general_verb = getattr(getattr(action.general_template, "verb", None), "name", None)
        alt_verb = getattr(getattr(action.alt, "verb", None), "name", None)
        verbs = set()
        if general_verb:
            verbs.add(general_verb)
        if alt_verb:
            verbs.add(alt_verb)
        for verb in verbs:
            self.verb_to_actions[verb].append(instance_id)

        # Preconditions
        for prec in action.preconditions or []:
            if isinstance(prec, dict):
                key = self._condition_to_key(prec)
                if key:
                    self.precondition_index[key].append(instance_id)

        # Changes (including per-cycle/total)
        for change_list_key in ("changes", "changes_per_cycle", "changes_total"):
            changes = getattr(action, change_list_key, None) or []
            for ch in changes:
                if isinstance(ch, dict):
                    key_new = self._change_to_key(ch, value_key="new")
                    if key_new:
                        self.effect_index[key_new].append(instance_id)
                    key_old = self._change_to_key(ch, value_key="old")
                    if key_old:
                        self.old_state_index[key_old].append(instance_id)

        # Conditional changes
        for cond in getattr(action, "conditional_changes", []) or []:
            if isinstance(cond, dict):
                for ch in cond.get("changes", []) or []:
                    if isinstance(ch, dict):
                        key_new = self._change_to_key(ch, value_key="new")
                        if key_new:
                            self.effect_index[key_new].append(instance_id)
                        key_old = self._change_to_key(ch, value_key="old")
                        if key_old:
                            self.old_state_index[key_old].append(instance_id)

        # Sub-actions recursively
        for sub in action.sub_actions or []:
            self._index_action(sub, depth + 1)

    # ------------------------------------------------------------------
    # Key construction
    # ------------------------------------------------------------------
    def _condition_to_key(self, cond: Dict[str, Any]) -> Optional[Tuple]:
        return self._entry_to_key(cond, value_key="value")

    def _change_to_key(self, ch: Dict[str, Any], value_key: str) -> Optional[Tuple]:
        return self._entry_to_key(ch, value_key=value_key)

    def _entry_to_key(self, entry: Dict[str, Any], value_key: str) -> Optional[Tuple]:
        """
        Build a hashable key from a condition/change dict.

        Key format:
            (object_name, attribute_path_tuple, op, value)

        Position is preserved as a composite string so the position unit
        can be matched later.
        """
        obj = entry.get("object")
        attr = entry.get("attribute")
        val = entry.get(value_key)
        op = entry.get("op", "eq")

        if obj is None or attr is None or val is None:
            return None

        if isinstance(attr, str):
            path = tuple(attr.split("."))
        elif isinstance(attr, (list, tuple)):
            path = tuple(str(x) for x in attr)
        else:
            path = (str(attr),)

        # Special handling for the position unit
        if path == ("position",) or path == ("position_relative_body",):
            if isinstance(val, dict):
                rel = val.get("relation", "")
                rel_r = val.get("relationr", "")
                rel_to = val.get("relative_to", "")
                val_composite = f"{rel}|{rel_r}|{rel_to}"
            else:
                val_composite = str(val)
            return (str(obj), path, str(op), val_composite)

        if isinstance(val, dict):
            val_norm = str(val)
        else:
            val_norm = str(val)

        return (str(obj), path, str(op), val_norm)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_actions_for_verb(self, verb: str) -> List[ActionInstance]:
        """Return all ActionInstances associated with a verb."""
        ids = self.verb_to_actions.get(verb, [])
        return [self.action_by_id[i] for i in ids if i in self.action_by_id]

    def get_actions_producing_effect(
        self,
        obj: str,
        attr_path: Tuple[str, ...],
        op: str = "eq",
        value: Any = None,
    ) -> List[ActionInstance]:
        """Return actions whose new state matches the given effect condition."""
        if value is None:
            return []
        key = (obj, tuple(attr_path), op, str(value))
        ids = self.effect_index.get(key, [])
        return [self.action_by_id[i] for i in ids if i in self.action_by_id]

    def get_actions_with_precondition(
        self,
        obj: str,
        attr_path: Tuple[str, ...],
        op: str = "eq",
        value: Any = None,
    ) -> List[ActionInstance]:
        """Return actions that require the given precondition."""
        if value is None:
            return []
        key = (obj, tuple(attr_path), op, str(value))
        ids = self.precondition_index.get(key, [])
        return [self.action_by_id[i] for i in ids if i in self.action_by_id]

    def get_actions_with_old_state(
        self,
        obj: str,
        attr_path: Tuple[str, ...],
        op: str = "eq",
        value: Any = None,
    ) -> List[ActionInstance]:
        """Return actions whose old state matches the given condition."""
        if value is None:
            return []
        key = (obj, tuple(attr_path), op, str(value))
        ids = self.old_state_index.get(key, [])
        return [self.action_by_id[i] for i in ids if i in self.action_by_id]

    # ------------------------------------------------------------------
    # Memory lookups
    # ------------------------------------------------------------------
    def get_memory_objects(self, memory_id: str) -> List[Object]:
        """Return the runtime Object instances for a memory, or an empty list."""
        mem = self.memory_by_id.get(memory_id)
        if mem is None:
            return []
        return list(mem.objects.values())

    def get_memory_id_for_action(self, action: ActionInstance) -> str:
        """
        Extract the memory ID from an action's instance ID.

        Instance IDs have the form:
            "mem_000051.memory.actions.0.sub_actions.1"
        so the memory ID is the first dot-separated segment.
        """
        if not action or not action.instance_id:
            return ""
        parts = action.instance_id.split(".")
        return parts[0] if parts else ""

    # ------------------------------------------------------------------
    # Object compatibility and substitution
    # ------------------------------------------------------------------
    def _object_to_compat_dict(self, obj: Object) -> Dict[str, Any]:
        """
        Convert a runtime Object into the flat dict shape expected by
        PlannerMatchHelpers.object_compatibility.
        """
        merged = dict(obj.template_defaults or {})
        merged.update(obj.overrides or {})
        instance_attrs = {
            k: v for k, v in merged.items()
            if k not in self._DEFINING_KEYS
        }

        return {
            "template_name": obj.template_name or "",
            "obj_id": obj.obj_id or "",
            "categories": list(obj.categories or []),
            "materials": list(obj.materials or []),
            "functions": list(obj.functions or []),
            "shape": obj.shape or "",
            "dimensions": obj.dimensions or {},
            "position": obj.position or {},
            "attributes": instance_attrs,
        }

    def get_memory_object_compat_dicts(
        self, memory_id: str
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Return a list of (obj_id, compat_dict) for the memory's objects.
        """
        result = []
        for obj in self.get_memory_objects(memory_id):
            result.append((obj.obj_id, self._object_to_compat_dict(obj)))
        return result

    def find_object_substitute(
        self,
        query_obj: Dict[str, Any],
        allowed_memory_ids: Optional[Set[str]] = None,
        min_threshold: Optional[float] = None,
        max_candidates: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Find the single best memory object substitute for a query object.

        Args:
            query_obj: the query object as produced by the query processor.
            allowed_memory_ids: restrict scanning to these memories, or None
                for all memories.
            min_threshold: minimum compatibility score for acceptance;
                defaults to self.substitution_min_threshold.
            max_candidates: cap on the number of objects considered;
                defaults to self.substitution_max_candidates.

        Returns:
            {
                "candidate": compat_dict,
                "memory_id": ...,
                "obj_id": ...,
                "score": float,
                "signals": [str],
                "tier": "exact" | "alias" | "fuzzy_name" | "attribute",
            }
            or None if no candidate passes the threshold.
        """
        if min_threshold is None:
            min_threshold = self.substitution_min_threshold
        if max_candidates is None:
            max_candidates = self.substitution_max_candidates

        best = None
        best_score = min_threshold
        considered = 0

        for memory_id, memory in self.memory_by_id.items():
            if allowed_memory_ids is not None and memory_id not in allowed_memory_ids:
                continue
            for obj in memory.objects.values():
                considered += 1
                if considered > max_candidates:
                    break
                cand = self._object_to_compat_dict(obj)
                score, signals, tier = self.helpers.object_compatibility(query_obj, cand)
                if score > best_score:
                    best_score = score
                    best = {
                        "candidate": cand,
                        "memory_id": memory_id,
                        "obj_id": obj.obj_id,
                        "score": score,
                        "signals": signals,
                        "tier": tier,
                    }
            if considered > max_candidates:
                break

        return best

    def find_object_substitutes_batch(
        self,
        query_obj: Dict[str, Any],
        allowed_memory_ids: Optional[Set[str]] = None,
        min_threshold: Optional[float] = None,
        max_candidates: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Like find_object_substitute, but returns all candidates above
        threshold, sorted by score descending.

        Returns a list of dicts with the same shape as find_object_substitute.
        """
        if min_threshold is None:
            min_threshold = self.substitution_min_threshold
        if max_candidates is None:
            max_candidates = self.substitution_max_candidates

        results = []
        considered = 0

        for memory_id, memory in self.memory_by_id.items():
            if allowed_memory_ids is not None and memory_id not in allowed_memory_ids:
                continue
            for obj in memory.objects.values():
                considered += 1
                if considered > max_candidates:
                    break
                cand = self._object_to_compat_dict(obj)
                score, signals, tier = self.helpers.object_compatibility(query_obj, cand)
                if score > min_threshold:
                    results.append({
                        "candidate": cand,
                        "memory_id": memory_id,
                        "obj_id": obj.obj_id,
                        "score": score,
                        "signals": signals,
                        "tier": tier,
                    })
            if considered > max_candidates:
                break

        results.sort(key=lambda x: -x["score"])
        return results