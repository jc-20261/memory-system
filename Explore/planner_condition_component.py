#!/usr/bin/env python3
r"""
planner_condition_component.py – Condition component for planning.

Performs backward chaining / goal regression over state changes and
preconditions. Uses the planning-side match helpers (which replicate the
search engine's full-path matching) so scoring is consistent with search.

Design:
    For each subgoal, scan every entry in effect_index. Score each candidate
    effect using the same core-triple matching the search engine uses
    (path × op × value), then multiply by an object multiplier:

        exact object name        → object_mult = 1.0
        different object name    → object_mult = object_compatibility score
        either side unresolvable → object_mult = 0.0 (candidate rejected)

    The final score is core_score × object_mult. When the object multiplier
    came from a substitution (score < 1.0), a substitution event is recorded
    in the trace.

    When no candidate meets the gate, backward chaining terminates on that
    branch and records the failure. The plan is not produced; the trace
    records the failure point.

The same logic serves verb-guided planning and general backward chaining,
since both call generate_plan().
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from memory_model import ActionInstance, Object
from planner_planning_index import PlanningIndex
from planner_match_helpers import OBJECT_COMPAT_CONFIG


# =============================================================================
# Filtering helpers
# =============================================================================
BANNED_FALLBACK_VALUE_MULT = 0.2  # mirrors CONFIG["value_no_match_mult"]


class ConditionComponent:
    """Backward chaining over state conditions with object substitution."""

    def __init__(self, planning_index: PlanningIndex):
        self.pidx = planning_index
        self.helpers = planning_index.helpers

        # Copy tuning parameters
        self.planning_depth_limit = planning_index.planning_depth_limit
        self.candidate_actions_per_subgoal = planning_index.candidate_actions_per_subgoal
        self.subgoal_match_threshold = planning_index.subgoal_match_threshold
        self.precondition_match_threshold = planning_index.precondition_match_threshold
        self.substitution_min_threshold = planning_index.substitution_min_threshold

        # Allowed memories filter
        self.allowed_memory_ids: Optional[Set[str]] = None

        # Caches
        self._object_compat_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Memory filtering
    # ------------------------------------------------------------------
    def set_allowed_memories(self, memory_ids: List[str]):
        self.allowed_memory_ids = set(memory_ids)
        self._object_compat_cache.clear()

    def clear_allowed_memories(self):
        self.allowed_memory_ids = None
        self._object_compat_cache.clear()

    def _is_allowed_action(self, action: ActionInstance) -> bool:
        if self.allowed_memory_ids is None:
            return True
        mid = self.pidx.get_memory_id_for_action(action)
        return mid in self.allowed_memory_ids

    # ------------------------------------------------------------------
    # Object resolution to compat dict
    # ------------------------------------------------------------------
    def _resolve_object_compat(
        self,
        obj_name: str,
        query_objects: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Return a compat dict for obj_name, or None if it cannot be resolved.

        Lookup order:
            1. Query objects (match by obj_id, template_name, or name)
            2. Instance lookup in pidx.object_by_id
            3. Memory scan by template_name (first match wins)
            4. object_templates stored defaults
            5. Fallback: a minimal dict with just template_name
        """
        if not obj_name:
            return None
        if obj_name in self._object_compat_cache:
            return self._object_compat_cache[obj_name]

        result = None

        # 1. Query objects
        for q in query_objects or []:
            if not isinstance(q, dict):
                continue
            if q.get("obj_id") == obj_name or q.get("template_name") == obj_name or q.get("name") == obj_name:
                result = {
                    "template_name": q.get("template_name") or obj_name,
                    "obj_id": q.get("obj_id") or obj_name,
                    "categories": list(q.get("categories", []) or []),
                    "materials": list(q.get("materials", []) or []),
                    "functions": list(q.get("functions", []) or []),
                    "shape": q.get("shape", "") or "",
                    "dimensions": q.get("dimensions", {}) or {},
                    "position": q.get("position", {}) or {},
                    "attributes": dict(q.get("attributes", {}) or {}),
                }
                break

        # 2. Instance lookup
        if result is None:
            obj = self.pidx.object_by_id.get(obj_name)
            if obj is not None:
                result = self.pidx._object_to_compat_dict(obj)

        # 3. Memory scan by template name
        if result is None:
            for mem in self.pidx.memory_by_id.values():
                for obj in mem.objects.values():
                    if obj.template_name == obj_name or obj.obj_id == obj_name:
                        result = self.pidx._object_to_compat_dict(obj)
                        break
                if result is not None:
                    break

        # 4. Template storage
        if result is None:
            tmpl = self.pidx.object_templates.get(obj_name)
            if isinstance(tmpl, dict):
                result = {
                    "template_name": obj_name,
                    "obj_id": obj_name,
                    "categories": list(tmpl.get("categories", []) or []),
                    "materials": list(tmpl.get("materials", []) or []),
                    "functions": list(tmpl.get("functions", []) or []),
                    "shape": tmpl.get("shape", "") or "",
                    "dimensions": tmpl.get("dimensions", {}) or {},
                    "position": tmpl.get("position", {}) or {},
                    "attributes": {},
                }

        # 5. Fallback
        if result is None:
            result = {
                "template_name": obj_name,
                "obj_id": obj_name,
                "categories": [],
                "materials": [],
                "functions": [],
                "shape": "",
                "dimensions": {},
                "position": {},
                "attributes": {},
            }

        self._object_compat_cache[obj_name] = result
        return result

    # ------------------------------------------------------------------
    # Core-triple + object multiplier scoring
    # ------------------------------------------------------------------
    def _score_condition_pair(
        self,
        goal_obj: str,
        goal_path: Tuple[str, ...],
        goal_op: str,
        goal_val: Any,
        cand_obj: str,
        cand_path: Tuple[str, ...],
        cand_op: str,
        cand_val: Any,
        query_objects: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Return (final_score, details) where final_score = core × object_mult.

        details includes: object_mult, object_tier, object_signals,
        path_mult, op_mult, value_mult, core_score, substitution (bool),
        original_object, substituted_object.
        """
        # Path
        path_mult, path_alias = self.helpers.path_match_score(
            tuple(goal_path), tuple(cand_path)
        )
        op_mult = self.helpers.op_match_score(goal_op, cand_op)
        value_mult = self.helpers.value_match_score(goal_val, cand_val)

        value_fallback_used = False
        if value_mult == 0:
            # Banned-component fallback: strip unmatchable components from
            # both paths, then try path match again.
            unmatchable = self.helpers._unmatchable_attr_components
            filtered_goal = tuple(c for c in goal_path if c not in unmatchable)
            filtered_cand = tuple(c for c in cand_path if c not in unmatchable)
            if not filtered_goal or not filtered_cand:
                return 0.0, {
                    "object_mult": 0.0,
                    "object_tier": "rejected",
                    "object_signals": [],
                    "path_mult": 0.0,
                    "op_mult": op_mult,
                    "value_mult": 0.0,
                    "core_score": 0.0,
                    "substitution": False,
                    "reason": "value_no_match_banned_path",
                }
            path_mult, path_alias = self.helpers.path_match_score(
                filtered_goal, filtered_cand
            )
            if path_mult == 0:
                return 0.0, {
                    "object_mult": 0.0,
                    "object_tier": "rejected",
                    "object_signals": [],
                    "path_mult": 0.0,
                    "op_mult": op_mult,
                    "value_mult": 0.0,
                    "core_score": 0.0,
                    "substitution": False,
                    "reason": "value_no_match_filtered_path_mismatch",
                }
            value_mult = BANNED_FALLBACK_VALUE_MULT
            value_fallback_used = True

        if path_mult == 0 or op_mult == 0 or value_mult == 0:
            return 0.0, {
                "object_mult": 0.0,
                "object_tier": "rejected",
                "object_signals": [],
                "path_mult": path_mult,
                "op_mult": op_mult,
                "value_mult": value_mult,
                "core_score": 0.0,
                "substitution": False,
                "reason": "zero_component",
            }

        core_score = path_mult * op_mult * value_mult

        # Object multiplier
        substitution = False
        object_tier = "exact"
        object_signals: List[str] = []

        if not goal_obj or not cand_obj:
            object_mult = 1.0
        elif goal_obj == cand_obj:
            object_mult = 1.0
        else:
            goal_compat = self._resolve_object_compat(goal_obj, query_objects)
            cand_compat = self._resolve_object_compat(cand_obj, query_objects)
            if goal_compat is None or cand_compat is None:
                return 0.0, {
                    "object_mult": 0.0,
                    "object_tier": "rejected",
                    "object_signals": [],
                    "path_mult": path_mult,
                    "op_mult": op_mult,
                    "value_mult": value_mult,
                    "core_score": core_score,
                    "substitution": False,
                    "reason": "object_unresolvable",
                }
            object_mult, object_signals, object_tier = self.helpers.object_compatibility(
                goal_compat, cand_compat
            )
            if object_mult == 0.0:
                return 0.0, {
                    "object_mult": 0.0,
                    "object_tier": "rejected",
                    "object_signals": object_signals,
                    "path_mult": path_mult,
                    "op_mult": op_mult,
                    "value_mult": value_mult,
                    "core_score": core_score,
                    "substitution": False,
                    "reason": "object_below_gate",
                }
            substitution = True

        final_score = core_score * object_mult
        details = {
            "object_mult": object_mult,
            "object_tier": object_tier,
            "object_signals": object_signals,
            "path_mult": path_mult,
            "op_mult": op_mult,
            "value_mult": value_mult,
            "value_fallback_used": value_fallback_used,
            "core_score": core_score,
            "substitution": substitution,
            "original_object": goal_obj if substitution else None,
            "substituted_object": cand_obj if substitution else None,
        }
        return final_score, details

    # ------------------------------------------------------------------
    # Condition satisfaction check
    # ------------------------------------------------------------------
    def _is_condition_satisfied(
        self,
        condition: Dict[str, Any],
        satisfied_conditions: List[Dict[str, Any]],
        query_objects: List[Dict[str, Any]],
    ) -> bool:
        """Return True if condition matches any entry in satisfied_conditions."""
        goal_obj = condition.get("object", "")
        goal_attr = condition.get("attribute")
        goal_path = self._path_tuple(goal_attr)
        goal_op = condition.get("op", "eq")
        goal_val = condition.get("value")

        for sat in satisfied_conditions:
            sat_obj = sat.get("object", "")
            sat_attr = sat.get("attribute")
            sat_path = self._path_tuple(sat_attr)
            sat_op = sat.get("op", "eq")
            sat_val = sat.get("value")

            score, _ = self._score_condition_pair(
                goal_obj, goal_path, goal_op, goal_val,
                sat_obj, sat_path, sat_op, sat_val,
                query_objects,
            )
            if score >= self.precondition_match_threshold:
                return True
        return False

    def _path_tuple(self, attr: Any) -> Tuple[str, ...]:
        if attr is None:
            return ()
        if isinstance(attr, str):
            return tuple(attr.split("."))
        if isinstance(attr, (list, tuple)):
            return tuple(str(x) for x in attr)
        return (str(attr),)

    # ------------------------------------------------------------------
    # Candidate action retrieval for a subgoal
    # ------------------------------------------------------------------
    def _find_candidate_actions_for_goal(
        self,
        goal_condition: Dict[str, Any],
        query_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return candidate actions for the goal condition.

        Each candidate is a dict:
            {
                "action": ActionInstance,
                "score": float,
                "details": {...},
                "effect_key": (obj, path, op, val),
            }
        """
        goal_obj = goal_condition.get("object", "")
        goal_attr = goal_condition.get("attribute")
        goal_path = self._path_tuple(goal_attr)
        goal_op = goal_condition.get("op", "eq")
        goal_val = goal_condition.get("value")

        candidates: List[Dict[str, Any]] = []
        seen_action_ids: Set[str] = set()

        # Straight scan over effect_index
        for key, action_ids in self.pidx.effect_index.items():
            key_obj, key_path, key_op, key_val = key
            # Skip effects with no object (should not happen but be safe)
            if not key_obj:
                continue

            score, details = self._score_condition_pair(
                goal_obj, goal_path, goal_op, goal_val,
                key_obj, key_path, key_op, key_val,
                query_objects or [],
            )
            if score <= 0:
                continue
            if score < self.subgoal_match_threshold:
                continue

            for action_id in action_ids:
                action = self.pidx.action_by_id.get(action_id)
                if action is None:
                    continue
                if not self._is_allowed_action(action):
                    continue
                if action_id in seen_action_ids:
                    # Already recorded — keep the higher score
                    for existing in candidates:
                        if existing["action"].instance_id == action_id:
                            if score > existing["score"]:
                                existing["score"] = score
                                existing["details"] = details
                                existing["effect_key"] = key
                            break
                    continue
                seen_action_ids.add(action_id)
                candidates.append({
                    "action": action,
                    "score": score,
                    "details": details,
                    "effect_key": key,
                })

        candidates.sort(key=lambda c: -c["score"])
        return candidates[: self.candidate_actions_per_subgoal]

    # ------------------------------------------------------------------
    # Precondition / old state extraction
    # ------------------------------------------------------------------
    def _extract_preconditions(self, action: ActionInstance) -> List[Dict[str, Any]]:
        preconditions = []
        for prec in action.preconditions or []:
            if isinstance(prec, dict):
                preconditions.append(prec)
        return preconditions

    def _extract_old_states(self, action: ActionInstance) -> List[Dict[str, Any]]:
        old_states = []
        for change_list_key in ("changes", "changes_per_cycle", "changes_total"):
            changes = getattr(action, change_list_key, None) or []
            for ch in changes:
                if isinstance(ch, dict):
                    old = ch.get("old")
                    if old is not None:
                        old_states.append({
                            "object": ch.get("object"),
                            "attribute": ch.get("attribute"),
                            "op": ch.get("op", "eq"),
                            "value": old,
                        })
        for cond in getattr(action, "conditional_changes", []) or []:
            if isinstance(cond, dict):
                for ch in cond.get("changes", []) or []:
                    if isinstance(ch, dict):
                        old = ch.get("old")
                        if old is not None:
                            old_states.append({
                                "object": ch.get("object"),
                                "attribute": ch.get("attribute"),
                                "op": ch.get("op", "eq"),
                                "value": old,
                            })
        return old_states

    def _extract_new_states(self, action: ActionInstance) -> List[Dict[str, Any]]:
        new_states = []
        for change_list_key in ("changes", "changes_per_cycle", "changes_total"):
            changes = getattr(action, change_list_key, None) or []
            for ch in changes:
                if isinstance(ch, dict):
                    new = ch.get("new")
                    if new is not None:
                        new_states.append({
                            "object": ch.get("object"),
                            "attribute": ch.get("attribute"),
                            "op": ch.get("op", "eq"),
                            "value": new,
                        })
        for cond in getattr(action, "conditional_changes", []) or []:
            if isinstance(cond, dict):
                for ch in cond.get("changes", []) or []:
                    if isinstance(ch, dict):
                        new = ch.get("new")
                        if new is not None:
                            new_states.append({
                                "object": ch.get("object"),
                                "attribute": ch.get("attribute"),
                                "op": ch.get("op", "eq"),
                                "value": new,
                            })
        return new_states

    # ------------------------------------------------------------------
    # Substitution event record
    # ------------------------------------------------------------------
    def _make_substitution_event(
        self,
        level: str,
        action: ActionInstance,
        details: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not details.get("substitution"):
            return None
        return {
            "level": level,
            "action_instance_id": action.instance_id,
            "action_template": action.template_name,
            "original_object": details.get("original_object"),
            "substituted_object": details.get("substituted_object"),
            "similarity": details.get("object_mult"),
            "signals": details.get("object_signals", []),
            "tier": details.get("object_tier"),
        }

    # ------------------------------------------------------------------
    # Main backward chaining
    # ------------------------------------------------------------------
    def generate_plan(
        self,
        goal_conditions: List[Dict[str, Any]],
        initial_conditions: List[Dict[str, Any]],
        query_objects: List[Dict[str, Any]],
        max_depth: Optional[int] = None,
        trace: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[List[ActionInstance]], Dict[str, Any]]:
        """
        Attempt to generate a plan that achieves goal_conditions from
        initial_conditions.

        On success returns (plan, diagnostics).
        On failure returns (None, diagnostics with failure reason).
        """
        if max_depth is None:
            max_depth = self.planning_depth_limit

        satisfied = list(initial_conditions)
        subgoals = list(goal_conditions)
        plan_rev: List[ActionInstance] = []
        visited_subgoals: Set[str] = set()
        substitutions: List[Dict[str, Any]] = []
        failure_info: Dict[str, Any] = {}

        def _subgoal_key(cond):
            obj = cond.get("object", "")
            attr = cond.get("attribute")
            path = self._path_tuple(attr)
            op = cond.get("op", "eq")
            val = cond.get("value")
            return f"{obj}|{path}|{op}|{val}"

        def backward_chain(
            current_subgoals: List[Dict[str, Any]],
            current_satisfied: List[Dict[str, Any]],
            current_plan_rev: List[ActionInstance],
            depth: int,
        ) -> bool:
            nonlocal failure_info

            if not current_subgoals:
                return True

            if depth > max_depth:
                failure_info = {"reason": "depth_limit", "depth": depth}
                return False

            for i, subgoal in enumerate(current_subgoals):
                if self._is_condition_satisfied(subgoal, current_satisfied, query_objects):
                    continue

                subgoal_key = _subgoal_key(subgoal)
                if subgoal_key in visited_subgoals:
                    continue
                visited_subgoals.add(subgoal_key)

                candidates = self._find_candidate_actions_for_goal(subgoal, query_objects)

                step_trace = {
                    "depth": depth,
                    "subgoal": subgoal,
                    "candidates": [],
                    "selected_action": None,
                    "selected_action_template": None,
                    "new_subgoals": [],
                    "backtracked": False,
                    "failure": None,
                }

                if not candidates:
                    step_trace["failure"] = {
                        "reason": "no_candidate_actions",
                        "subgoal": subgoal,
                    }
                    if trace is not None:
                        trace.append(step_trace)
                    failure_info = {
                        "reason": "no_candidate_actions",
                        "subgoal": subgoal,
                        "depth": depth,
                    }
                    return False

                for cand in candidates:
                    action = cand["action"]
                    score = cand["score"]
                    details = cand["details"]

                    preconditions = self._extract_preconditions(action)
                    old_states = self._extract_old_states(action)

                    new_subgoals = []
                    for pc in preconditions + old_states:
                        if not self._is_condition_satisfied(pc, current_satisfied, query_objects):
                            new_subgoals.append(pc)

                    candidate_info = {
                        "action_instance_id": action.instance_id,
                        "action_template": action.template_name,
                        "match_score": score,
                        "effect_key": list(cand["effect_key"]),
                        "details": details,
                        "preconditions": preconditions,
                        "old_states": old_states,
                        "new_subgoals": new_subgoals,
                    }
                    step_trace["candidates"].append(candidate_info)

                    current_plan_rev.append(action)
                    sub_event = self._make_substitution_event("goal", action, details)

                    new_satisfied = current_satisfied + [subgoal]
                    remaining = current_subgoals[:i] + current_subgoals[i+1:] + new_subgoals

                    if backward_chain(remaining, new_satisfied, current_plan_rev, depth + 1):
                        step_trace["selected_action"] = action.instance_id
                        step_trace["selected_action_template"] = action.template_name
                        step_trace["new_subgoals"] = new_subgoals
                        if sub_event is not None:
                            substitutions.append(sub_event)
                        if trace is not None:
                            trace.append(step_trace)
                        return True

                    # Backtrack
                    current_plan_rev.pop()
                    step_trace["backtracked"] = True

                if trace is not None:
                    trace.append(step_trace)
                failure_info = {
                    "reason": "backtracked_all_candidates",
                    "subgoal": subgoal,
                    "depth": depth,
                }
                return False

            return True

        success = backward_chain(subgoals, satisfied, plan_rev, 0)

        diagnostics = {
            "success": success,
            "plan_length": len(plan_rev) if success else 0,
            "substitutions": substitutions,
            "failure": failure_info if not success else None,
            "depth_used": max_depth,
        }

        if not success:
            return None, diagnostics

        plan = list(reversed(plan_rev))
        return plan, diagnostics