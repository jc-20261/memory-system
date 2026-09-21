#!/usr/bin/env python3
r"""
planner_world_state_validator.py – Dual-mode world state validator for planning.

Provides two independent validations per plan:

    Mode 1 — Plan-context validation.
        Build a world state from the memories actually referenced by the
        plan's actions (i.e. the substituted objects and their memory
        attributes). Apply the plan's actions in sequence. This is the
        "correct" world in which the substituted actions actually make
        sense.

    Mode 2 — Query-context validation.
        Build a world state from the query's objects and initial conditions.
        Apply the plan's actions against it. Expect many failures. Failures
        are recorded but do not halt the simulation.

Both modes use the same fuzzy matching for preconditions and goal checks
(delegated to PlannerMatchHelpers.value_match_score), so validation is
consistent with the condition component.

Neither mode blocks plan generation. Precondition failures reduce
goal_satisfaction but do not discard the plan.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from memory_model import ActionInstance, Object as MemoryObject
from planner_planning_index import PlanningIndex
from planner_match_helpers import PlannerMatchHelpers


# =============================================================================
# Helpers
# =============================================================================
def _as_target_list(value) -> List[str]:
    """Return a list of non-empty string targets from a scalar or list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if x is not None and str(x) != ""]
    s = str(value)
    return [s] if s else ""


# =============================================================================
# WorldState
# =============================================================================
class WorldState:
    """Runtime world state used for plan validation."""

    def __init__(self):
        self.objects: Dict[str, Dict[str, Any]] = {}
        self.spatial_forward: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.spatial_reverse: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.current_time: float = 0.0

    def add_object(self, obj_id: str, attributes: Optional[Dict[str, Any]] = None):
        if obj_id not in self.objects:
            self.objects[obj_id] = {}
        if attributes:
            self.objects[obj_id].update(attributes)

    def get_object(self, obj_id: str) -> Optional[Dict[str, Any]]:
        return self.objects.get(obj_id)

    def object_exists(self, obj_id: str) -> bool:
        return obj_id in self.objects

    def set_attribute(self, obj_id: str, path, value: Any):
        if obj_id not in self.objects:
            self.objects[obj_id] = {}
        if isinstance(path, str):
            path = tuple(path.split("."))
        current = self.objects[obj_id]
        for comp in path[:-1]:
            if comp not in current or not isinstance(current[comp], dict):
                current[comp] = {}
            current = current[comp]
        current[path[-1]] = value

    def get_attribute(self, obj_id: str, path, default: Any = None) -> Any:
        obj = self.objects.get(obj_id)
        if obj is None:
            return default
        if isinstance(path, str):
            path = tuple(path.split("."))
        current = obj
        for comp in path:
            if not isinstance(current, dict) or comp not in current:
                return default
            current = current[comp]
        return current

    def add_spatial(self, subj_id: str, relation: str, obj_id) -> None:
        """
        Add one or more spatial edges from subj_id to the target(s).

        obj_id may be a scalar string or a list of strings. When a list is
        given, each item produces its own edge. This allows ternary or
        multi-target relations (for example 'between') to be represented
        as a set of binary edges without crashing on unhashable keys.
        """
        targets = _as_target_list(obj_id)
        if not subj_id or not relation or not targets:
            return
        subj_str = str(subj_id)
        rel_str = str(relation)
        for target in targets:
            self.spatial_forward[subj_str][rel_str].append(target)
            self.spatial_reverse[target][rel_str].append(subj_str)

    def remove_spatial(self, subj_id: str, relation: str, obj_id) -> None:
        """Symmetric removal for one or more target objects."""
        targets = _as_target_list(obj_id)
        subj_str = str(subj_id)
        rel_str = str(relation)
        for target in targets:
            if subj_str in self.spatial_forward and rel_str in self.spatial_forward[subj_str]:
                try:
                    self.spatial_forward[subj_str][rel_str].remove(target)
                    if not self.spatial_forward[subj_str][rel_str]:
                        del self.spatial_forward[subj_str][rel_str]
                except ValueError:
                    pass
            if target in self.spatial_reverse and rel_str in self.spatial_reverse[target]:
                try:
                    self.spatial_reverse[target][rel_str].remove(subj_str)
                    if not self.spatial_reverse[target][rel_str]:
                        del self.spatial_reverse[target][rel_str]
                except ValueError:
                    pass

    def copy(self) -> "WorldState":
        import copy
        new = WorldState()
        new.objects = copy.deepcopy(self.objects)
        new.spatial_forward = copy.deepcopy(self.spatial_forward)
        new.spatial_reverse = copy.deepcopy(self.spatial_reverse)
        new.current_time = self.current_time
        return new


# =============================================================================
# WorldStateValidator
# =============================================================================
class WorldStateValidator:
    """
    Validates a plan against a given world state.

    Lenient by design: precondition failures are recorded but the action's
    effects are still applied, so the plan can continue and the goal can be
    reached in the trace. Goal failures are recorded but do not invalidate
    the plan (they reduce goal_satisfaction via the planner's scoring).
    """

    def __init__(self, planning_index: PlanningIndex):
        self.pidx = planning_index
        self.helpers: PlannerMatchHelpers = planning_index.helpers
        self.threshold = 0.0  # everything passes; scoring reflects magnitude

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def validate_plan_query_context(
        self,
        plan: List[ActionInstance],
        query_objects: List[Dict[str, Any]],
        initial_conditions: List[Dict[str, Any]],
        goal_conditions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Validate the plan against the query's own world state.

        Builds a world state from the query objects and initial conditions.
        Many preconditions will fail because plan actions reference memory
        objects rather than query objects. Failures are recorded, not fatal.
        """
        state = self._build_world_state_from_query(query_objects, initial_conditions)
        return self._run_plan(state, plan, goal_conditions, mode="query_context")

    def validate_plan_plan_context(
        self,
        plan: List[ActionInstance],
        goal_conditions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Validate the plan against a world state built from the memories the
        plan's actions were drawn from.
        """
        state = self._build_world_state_from_plan(plan)
        return self._run_plan(state, plan, goal_conditions, mode="plan_context")

    # ------------------------------------------------------------------
    # World state construction
    # ------------------------------------------------------------------
    def _build_world_state_from_query(
        self,
        query_objects: List[Dict[str, Any]],
        initial_conditions: List[Dict[str, Any]],
    ) -> WorldState:
        ws = WorldState()

        for obj in query_objects or []:
            if not isinstance(obj, dict):
                continue
            obj_id = obj.get("obj_id") or obj.get("template_name")
            if not obj_id:
                continue
            attrs: Dict[str, Any] = {}
            # Flatten attributes into nested form
            for path_str, val in (obj.get("attributes") or {}).items():
                self._set_nested(attrs, str(path_str).split("."), val)
            # Also include defining attributes at the top level
            for key in ("shape",):
                v = obj.get(key)
                if v:
                    attrs[key] = v
            for key in ("dimensions", "position"):
                v = obj.get(key)
                if isinstance(v, dict) and v:
                    attrs[key] = dict(v)
            ws.add_object(obj_id, attrs)

        for cond in initial_conditions or []:
            if not isinstance(cond, dict):
                continue
            obj_id = cond.get("object")
            attr = cond.get("attribute")
            val = cond.get("value")
            aspect = cond.get("aspect")
            if not obj_id or attr is None or val is None:
                continue
            path = self._build_path(aspect, attr)
            ws.set_attribute(obj_id, path, val)
            if path == ("position",) and isinstance(val, dict):
                rel = val.get("relation", "")
                rel_to = val.get("relative_to", "")
                if rel and rel_to:
                    ws.add_spatial(obj_id, rel, rel_to)

        return ws

    def _build_world_state_from_plan(self, plan: List[ActionInstance]) -> WorldState:
        """
        Build a world state from the memories referenced by the plan's actions.

        For each action in the plan, find the memory that produced it and
        add all of that memory's objects to the world state with their
        template defaults merged with their instance overrides.
        """
        ws = WorldState()
        processed_memories = set()

        for action in plan:
            memory_id = self.pidx.get_memory_id_for_action(action)
            if not memory_id or memory_id in processed_memories:
                continue
            processed_memories.add(memory_id)

            for obj in self.pidx.get_memory_objects(memory_id):
                if obj.obj_id in ws.objects:
                    continue
                attrs = self._object_to_state_attrs(obj)
                ws.add_object(obj.obj_id, attrs)

                # Record spatial relation from the object's position
                pos = obj.position or {}
                rel = pos.get("relation")
                rel_to = pos.get("relative_to")
                if rel and rel_to:
                    ws.add_spatial(obj.obj_id, rel, rel_to)

        return ws

    def _object_to_state_attrs(self, obj: MemoryObject) -> Dict[str, Any]:
        """
        Build a nested attribute dict from a runtime Object's merged defaults
        and overrides. This is what the world state stores.
        """
        merged: Dict[str, Any] = {}
        defaults = obj.template_defaults or {}
        for k, v in defaults.items():
            if k in ("categories", "functions", "materials", "aliases"):
                continue
            self._set_nested(merged, str(k).split("."), v)
        for k, v in (obj.overrides or {}).items():
            self._set_nested(merged, str(k).split("."), v)
        return merged

    def _set_nested(self, container: Dict[str, Any], path: List[str], value: Any):
        cur = container
        for comp in path[:-1]:
            if comp not in cur or not isinstance(cur[comp], dict):
                cur[comp] = {}
            cur = cur[comp]
        if path:
            cur[path[-1]] = value

    def _build_path(self, aspect, attribute) -> Tuple[str, ...]:
        parts: List[str] = []
        if aspect:
            if isinstance(aspect, str):
                parts.extend(aspect.split("."))
            elif isinstance(aspect, (list, tuple)):
                parts.extend(str(x) for x in aspect)
        if attribute is not None:
            if isinstance(attribute, str):
                parts.extend(attribute.split("."))
            elif isinstance(attribute, (list, tuple)):
                parts.extend(str(x) for x in attribute)
            else:
                parts.append(str(attribute))
        return tuple(parts)

    # ------------------------------------------------------------------
    # Plan execution
    # ------------------------------------------------------------------
    def _run_plan(
        self,
        state: WorldState,
        plan: List[ActionInstance],
        goal_conditions: List[Dict[str, Any]],
        mode: str,
    ) -> Dict[str, Any]:
        """
        Apply the plan to the given state and check the goal conditions.

        Precondition failures are recorded but do not halt the simulation.
        Effects are always applied, so the trace shows what would happen if
        the action were executed anyway.
        """
        action_errors: Dict[int, List[str]] = {}
        all_errors: List[str] = []
        precondition_checks: List[Dict[str, Any]] = []

        for i, action in enumerate(plan, 1):
            step_precondition_checks = self._check_preconditions(state, action)
            step_errors = [c["message"] for c in step_precondition_checks if not c["passed"]]
            precondition_checks.extend(step_precondition_checks)

            # Apply effects regardless of precondition failures
            self._apply_action_effects(state, action)

            if step_errors:
                action_errors[i] = step_errors
                all_errors.extend(f"Step {i} ({action.template_name}): {e}" for e in step_errors)

        goal_checks = self._check_goal_conditions(state, goal_conditions)
        unmet_goals = [g for g in goal_checks if not g["passed"]]

        # Lenient validity: valid if preconditions passed and goals met
        valid = (not action_errors) and (not unmet_goals)

        # Partial satisfaction score: fraction of goal conditions satisfied
        # (weighted by their individual match score, so a fuzzy match counts
        # less than an exact one).
        if goal_checks:
            satisfaction = sum(g["score"] for g in goal_checks) / len(goal_checks)
        else:
            satisfaction = 1.0

        return {
            "mode": mode,
            "valid": valid,
            "errors": all_errors,
            "action_errors": action_errors,
            "unmet_goals": unmet_goals,
            "goal_checks": goal_checks,
            "precondition_checks": precondition_checks,
            "goal_satisfaction": satisfaction,
            "final_state": state,
            "objects_in_state": list(state.objects.keys()),
        }

    # ------------------------------------------------------------------
    # Precondition checks
    # ------------------------------------------------------------------
    def _check_preconditions(
        self,
        state: WorldState,
        action: ActionInstance,
    ) -> List[Dict[str, Any]]:
        results = []
        for prec in action.preconditions or []:
            if not isinstance(prec, dict):
                continue
            res = self._check_condition(state, prec, level="precondition", action=action)
            results.append(res)
        return results

    def _check_goal_conditions(
        self,
        state: WorldState,
        goal_conditions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        results = []
        for goal in goal_conditions or []:
            if not isinstance(goal, dict):
                continue
            res = self._check_condition(state, goal, level="goal", action=None)
            results.append(res)
        return results

    def _check_condition(
        self,
        state: WorldState,
        condition: Dict[str, Any],
        level: str,
        action: Optional[ActionInstance],
    ) -> Dict[str, Any]:
        """
        Check a single condition against the world state using fuzzy value
        matching. Records the condition, the object resolution, the actual
        value found, and the resulting match score.

        Object resolution:
            1. Exact obj_id in state
            2. Exact template name in state
            3. Fuzzy substitution via object_compatibility
            4. Fail (record as missing)
        """
        obj_name = condition.get("object")
        attr = condition.get("attribute")
        aspect = condition.get("aspect")
        op = condition.get("op", "eq")
        expected = condition.get("value")

        record: Dict[str, Any] = {
            "level": level,
            "condition": condition,
            "object_requested": obj_name,
            "object_resolved": None,
            "substituted": False,
            "attribute_path": None,
            "actual_value": None,
            "expected_value": expected,
            "op": op,
            "score": 0.0,
            "passed": False,
            "message": "",
        }

        if not obj_name or attr is None:
            record["message"] = "condition missing object or attribute"
            return record

        path = self._build_path(aspect, attr)
        record["attribute_path"] = list(path)

        # Object resolution
        resolved_id = self._resolve_object_in_state(obj_name, state)
        if resolved_id is None:
            record["message"] = f"object '{obj_name}' not found in world state"
            return record

        record["object_resolved"] = resolved_id
        record["substituted"] = (resolved_id != obj_name)

        actual = state.get_attribute(resolved_id, path)
        record["actual_value"] = actual

        # Compare
        score, message = self._compare_values(op, actual, expected)
        record["score"] = score
        record["passed"] = score > 0.0
        record["message"] = message

        return record

    def _resolve_object_in_state(
        self,
        obj_name: str,
        state: WorldState,
    ) -> Optional[str]:
        # 1. Exact obj_id
        if obj_name in state.objects:
            return obj_name

        # 2. Exact template name
        for obj_id, attrs in state.objects.items():
            if attrs.get("template_name") == obj_name:
                return obj_id

        # 3. Fuzzy substitution via object_compatibility
        query_compat = self._resolve_query_object_compat(obj_name)
        if query_compat is None:
            return None

        best_id = None
        best_score = 0.0
        for obj_id in state.objects:
            cand_compat = self._state_object_to_compat(state, obj_id)
            if cand_compat is None:
                continue
            score, _signals, _tier = self.helpers.object_compatibility(query_compat, cand_compat)
            if score > best_score:
                best_score = score
                best_id = obj_id

        return best_id if best_score > 0.0 else None

    def _resolve_query_object_compat(self, obj_name: str) -> Optional[Dict[str, Any]]:
        # Instance lookup
        obj = self.pidx.object_by_id.get(obj_name)
        if obj is not None:
            return self.pidx._object_to_compat_dict(obj)
        # Memory scan
        for mem in self.pidx.memory_by_id.values():
            for o in mem.objects.values():
                if o.template_name == obj_name or o.obj_id == obj_name:
                    return self.pidx._object_to_compat_dict(o)
        # Template storage
        tmpl = self.pidx.object_templates.get(obj_name)
        if isinstance(tmpl, dict):
            return {
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
        # Minimal fallback
        return {
            "template_name": obj_name,
            "obj_id": obj_name,
            "categories": [], "materials": [], "functions": [],
            "shape": "", "dimensions": {}, "position": {}, "attributes": {},
        }

    def _state_object_to_compat(
        self,
        state: WorldState,
        obj_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Build a compat dict from an object already in the world state."""
        attrs = state.objects.get(obj_id)
        if attrs is None:
            return None
        # Try the runtime Object first, for richer metadata
        runtime = self.pidx.object_by_id.get(obj_id)
        if runtime is not None:
            return self.pidx._object_to_compat_dict(runtime)
        # Fall back to the state attrs
        return {
            "template_name": attrs.get("template_name", obj_id),
            "obj_id": obj_id,
            "categories": list(attrs.get("categories", []) or []),
            "materials": list(attrs.get("materials", []) or []),
            "functions": list(attrs.get("functions", []) or []),
            "shape": attrs.get("shape", "") or "",
            "dimensions": attrs.get("dimensions", {}) or {},
            "position": attrs.get("position", {}) or {},
            "attributes": {k: v for k, v in attrs.items()
                           if k not in {"template_name", "categories", "materials",
                                        "functions", "shape", "dimensions", "position"}},
        }

    # ------------------------------------------------------------------
    # Value comparison (fuzzy)
    # ------------------------------------------------------------------
    def _compare_values(
        self,
        op: str,
        actual: Any,
        expected: Any,
    ) -> Tuple[float, str]:
        """
        Return (score, message). Score is in [0, 1].

        Uses the helpers' fuzzy value matching for eq / ne. Numeric
        comparisons (gt / gte / lt / lte) are done on normalised values.
        Missing actual → 0.0 with an explanatory message.
        """
        if actual is None:
            # The condition's expected value may itself be None (rare)
            if expected is None and op == "eq":
                return 1.0, "both None"
            return 0.0, f"value not set (expected {expected})"

        op = str(op)

        if op == "eq":
            score = self.helpers.value_match_score(actual, expected)
            if score > 0.0:
                return score, f"matched with score {score:.3f}"
            return 0.0, f"mismatch: actual={actual}, expected={expected}"

        if op == "ne":
            score = self.helpers.value_match_score(actual, expected)
            if score == 0.0:
                return 1.0, "not equal"
            return max(0.0, 1.0 - score), f"unexpectedly similar (score {score:.3f})"

        if op in ("gt", ">", "gte", ">=", "lt", "<", "lte", "<="):
            return self._compare_ordering(op, actual, expected)

        if op == "in":
            if isinstance(actual, (list, tuple)):
                score = max((self.helpers.value_match_score(expected, x) for x in actual), default=0.0)
                return score, f"in-list match score {score:.3f}" if score > 0 else "not in list"
            if isinstance(expected, (list, tuple)):
                score = max((self.helpers.value_match_score(actual, x) for x in expected), default=0.0)
                return score, f"expected-list match score {score:.3f}" if score > 0 else "not in list"
            return 0.0, "invalid 'in' comparison (neither side is a list)"

        if op == "+=":
            return 0.0, "+= comparison not supported in validation"

        # Unknown operator — default to eq
        score = self.helpers.value_match_score(actual, expected)
        return score, f"unknown op '{op}', treated as eq (score {score:.3f})"

    def _compare_ordering(
        self,
        op: str,
        actual: Any,
        expected: Any,
    ) -> Tuple[float, str]:
        a = self.helpers.normalize_value(actual)
        e = self.helpers.normalize_value(expected)
        try:
            a_num = float(a)
            e_num = float(e)
        except (TypeError, ValueError):
            return 0.0, f"cannot order-compare {actual} with {expected}"

        if op in ("gt", ">"):
            return (1.0, "greater") if a_num > e_num else (0.0, f"{a_num} not > {e_num}")
        if op in ("gte", ">="):
            return (1.0, "greater or equal") if a_num >= e_num else (0.0, f"{a_num} not >= {e_num}")
        if op in ("lt", "<"):
            return (1.0, "less") if a_num < e_num else (0.0, f"{a_num} not < {e_num}")
        if op in ("lte", "<="):
            return (1.0, "less or equal") if a_num <= e_num else (0.0, f"{a_num} not <= {e_num}")
        return 0.0, f"unknown ordering op '{op}'"

    # ------------------------------------------------------------------
    # Apply action effects
    # ------------------------------------------------------------------
    def _apply_action_effects(
        self,
        state: WorldState,
        action: ActionInstance,
    ):
        for change in action.changes or []:
            self._apply_change(state, change)
        for list_name in ("changes_per_cycle", "changes_total"):
            for change in getattr(action, list_name, []) or []:
                self._apply_change(state, change)
        for cond in getattr(action, "conditional_changes", []) or []:
            if isinstance(cond, dict):
                condition = cond.get("condition")
                if condition and isinstance(condition, dict):
                    check = self._check_condition(state, condition, "conditional", action)
                    if check["passed"]:
                        for ch in cond.get("changes", []) or []:
                            self._apply_change(state, ch)

        state.current_time += getattr(action, "effective_duration", action.duration or 0)

    def _apply_change(self, state: WorldState, change: Dict[str, Any]):
        if not isinstance(change, dict):
            return
        obj_id = change.get("object")
        if not obj_id:
            return
        # If the object is not present, add it as an empty object so the
        # change at least records something (lenient behaviour).
        if not state.object_exists(obj_id):
            state.add_object(obj_id, {})

        attr = change.get("attribute")
        aspect = change.get("aspect")
        op = change.get("op", "eq")
        new_value = change.get("new")
        if new_value is None and "value" in change:
            new_value = change.get("value")

        path = self._build_path(aspect, attr)
        if not path:
            return

        if op == "+=":
            current = state.get_attribute(obj_id, path)
            try:
                current_num = float(current) if current is not None else 0.0
                new_num = float(new_value) if new_value is not None else 0.0
                state.set_attribute(obj_id, path, current_num + new_num)
            except (TypeError, ValueError):
                state.set_attribute(obj_id, path, new_value)
        else:
            state.set_attribute(obj_id, path, new_value)

        if path == ("position",) and isinstance(new_value, dict):
            rel = new_value.get("relation", "")
            rel_to = new_value.get("relative_to", "")
            if rel and rel_to:
                state.add_spatial(obj_id, rel, rel_to)