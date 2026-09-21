#!/usr/bin/env python3
r"""
planner_verb_component.py – Verb component for planning.

Retrieves candidate actions associated with target verbs and scores them by:

  - Verb match (exact string, or fuzzy via the verb embedding group)
  - Object fit (via PlannerMatchHelpers.object_compatibility against action
    participants)
  - Optional action category match

The score returned is on the same scale as the condition component's object
multiplier, so a candidate's object fit has a single meaning throughout the
planning pipeline. No duplicate matching logic lives here — everything is
delegated to PlannerMatchHelpers.
"""

from typing import Any, Dict, List, Optional, Tuple

from memory_model import ActionInstance, Object
from planner_planning_index import PlanningIndex


class VerbComponent:
    """Retrieves and scores actions by verb and object fit."""

    def __init__(self, planning_index: PlanningIndex):
        self.pidx = planning_index
        self.helpers = planning_index.helpers

        # Copy tuning parameters from planning index
        self.target_verb_cap = planning_index.target_verb_cap
        self.context_verb_cap = planning_index.context_verb_cap
        self.verb_exact_bonus = planning_index.verb_exact_bonus
        self.verb_fuzzy_bonus = planning_index.verb_fuzzy_bonus
        self.action_category_bonus = planning_index.action_category_bonus

        # Per-call caches (cleared at the start of each retrieve_candidates call)
        self._participant_compat_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def retrieve_candidates(
        self,
        verbs: List[str],
        query_objects: List[Dict[str, Any]],
        is_target: bool = True,
        query_action_category: Optional[str] = None,
        trace: Optional[List[Dict[str, Any]]] = None,
        allowed_memory_ids: Optional[set] = None,
    ) -> List[Tuple[ActionInstance, Dict[str, float]]]:
        """
        Return ranked candidate actions for each verb, with component scores.

        Args:
            verbs: list of query verb strings (target or context).
            query_objects: resolved query objects (compat dict shape).
            is_target: True if these are target verbs, False for context.
            query_action_category: optional action category to match against.
            trace: optional list; if provided, a detailed entry is appended
                per candidate with verb/object/category breakdowns.
            allowed_memory_ids: optional set of memory IDs. When provided,
                candidates whose owning memory is not in the set are
                discarded. None means no filtering.

        Returns:
            List of (action, scores) tuples sorted by total score descending,
            capped at target_verb_cap or context_verb_cap.
        """
        cap = self.target_verb_cap if is_target else self.context_verb_cap

        # Reset caches
        self._participant_compat_cache.clear()

        candidates: List[Tuple[ActionInstance, Dict[str, float]]] = []
        seen_action_ids = set()

        for verb in verbs:
            actions = self.pidx.get_actions_for_verb(verb)
            for action in actions:
                if action.instance_id in seen_action_ids:
                    continue

                if allowed_memory_ids is not None:
                    mid = self.pidx.get_memory_id_for_action(action)
                    if mid not in allowed_memory_ids:
                        continue

                seen_action_ids.add(action.instance_id)

                scores, details = self._score_candidate_detailed(
                    verb, action, query_objects, query_action_category
                )
                candidates.append((action, scores))

                if trace is not None:
                    trace.append({
                        "verb": verb,
                        "action_instance_id": action.instance_id,
                        "action_template": action.template_name,
                        "scores": scores,
                        "details": details,
                    })

        candidates.sort(key=lambda item: item[1]["total"], reverse=True)
        return candidates[:cap]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score_candidate(
        self,
        query_verb: str,
        action: ActionInstance,
        query_objects: List[Dict[str, Any]],
        query_action_category: Optional[str] = None,
    ) -> Dict[str, float]:
        """Return the component scores for a single action candidate."""
        scores, _ = self._score_candidate_detailed(
            query_verb, action, query_objects, query_action_category
        )
        return scores

    def _score_candidate_detailed(
        self,
        query_verb: str,
        action: ActionInstance,
        query_objects: List[Dict[str, Any]],
        query_action_category: Optional[str] = None,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        verb_score, verb_details = self._score_verb_match_detailed(query_verb, action)
        object_score, object_details = self._score_object_fit_detailed(
            query_objects, action
        )

        category_score = 0.0
        if query_action_category and action.action_category == query_action_category:
            category_score = self.action_category_bonus

        total = verb_score + object_score + category_score

        scores = {
            "verb": verb_score,
            "object": object_score,
            "category": category_score,
            "total": total,
        }

        details = {
            "verb": verb_details,
            "object": object_details,
            "category": {
                "query_action_category": query_action_category,
                "action_category": action.action_category,
                "matched": bool(
                    query_action_category
                    and action.action_category == query_action_category
                ),
                "bonus": category_score,
            },
        }

        return scores, details

    # ------------------------------------------------------------------
    # Verb match
    # ------------------------------------------------------------------
    def _collect_action_verbs(self, action: ActionInstance) -> List[str]:
        """Return all verb names attached to the action (general + alt)."""
        verbs = []
        if action.general_template and getattr(action.general_template, "verb", None):
            name = action.general_template.verb.name
            if name:
                verbs.append(name)
        if action.alt and getattr(action.alt, "verb", None):
            name = action.alt.verb.name
            if name and name not in verbs:
                verbs.append(name)
        return verbs

    def _score_verb_match_detailed(
        self,
        query_verb: str,
        action: ActionInstance,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score the verb match. Exact string match gets the exact bonus.
        Otherwise, try fuzzy matching against the verb embedding group.
        """
        action_verbs = self._collect_action_verbs(action)
        if not action_verbs:
            return 0.0, {
                "matched_verb": None,
                "match_type": "none",
                "score": 0.0,
                "action_verbs": [],
            }

        # Exact string match first
        for verb in action_verbs:
            if verb == query_verb:
                return self.verb_exact_bonus, {
                    "matched_verb": verb,
                    "match_type": "exact",
                    "score": self.verb_exact_bonus,
                    "action_verbs": action_verbs,
                }

        # Fuzzy match via the verb embedding group
        best_score = 0.0
        best_verb = None
        best_sim = 0.0

        for verb in action_verbs:
            full_term = f"verb:{verb}"
            term_index = self.helpers.group_term_index.get("verb", {}).get(full_term)
            if term_index is None:
                continue
            q_vec = self.helpers._query_embedding(query_verb)
            if q_vec is None:
                continue
            sim = float(self.helpers.group_embeddings["verb"][term_index] @ q_vec)
            if sim >= self.helpers.threshold and sim > best_sim:
                best_sim = sim
                best_verb = verb
                best_score = self.verb_fuzzy_bonus

        if best_score > 0:
            return best_score, {
                "matched_verb": best_verb,
                "match_type": "fuzzy",
                "similarity": best_sim,
                "score": best_score,
                "action_verbs": action_verbs,
            }

        return 0.0, {
            "matched_verb": None,
            "match_type": "none",
            "score": 0.0,
            "action_verbs": action_verbs,
        }

    # ------------------------------------------------------------------
    # Object fit
    # ------------------------------------------------------------------
    def _participant_compat(self, participant: Object) -> Dict[str, Any]:
        """Return the compat dict for a participant, cached per call."""
        key = participant.obj_id
        if key not in self._participant_compat_cache:
            self._participant_compat_cache[key] = self.pidx._object_to_compat_dict(participant)
        return self._participant_compat_cache[key]

    def _ensure_compat_dict(self, q_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure a query object is in the compat dict shape expected by
        object_compatibility. Query objects from the query processor are
        already close; this method fills in any missing keys.
        """
        return {
            "template_name": q_obj.get("template_name") or q_obj.get("name") or "",
            "obj_id": q_obj.get("obj_id") or q_obj.get("template_name") or q_obj.get("name") or "",
            "categories": list(q_obj.get("categories", []) or []),
            "materials": list(q_obj.get("materials", []) or []),
            "functions": list(q_obj.get("functions", []) or []),
            "shape": q_obj.get("shape", "") or "",
            "dimensions": q_obj.get("dimensions", {}) or {},
            "position": q_obj.get("position", {}) or {},
            "attributes": dict(q_obj.get("attributes", {}) or {}),
        }

    def _score_object_fit_detailed(
        self,
        query_objects: List[Dict[str, Any]],
        action: ActionInstance,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score how well the action's participants match the query objects.

        For each query object, find the best-matching participant by
        object_compatibility and take that score. The total is the sum of
        best-per-query-object scores. This preserves the "more query objects
        matched = higher score" property while keeping each pair's score
        bounded to [0, 1].
        """
        if not query_objects:
            return 0.0, {"reason": "no_query_objects", "comparisons": []}

        participants = action.participants or []
        if not participants:
            return 0.0, {"reason": "no_participants", "comparisons": []}

        per_query_scores: List[float] = []
        comparisons: List[Dict[str, Any]] = []

        for q_obj in query_objects:
            q_compat = self._ensure_compat_dict(q_obj)

            best_pair_score = 0.0
            best_pair_detail: Optional[Dict[str, Any]] = None

            for p_obj in participants:
                p_compat = self._participant_compat(p_obj)
                score, signals, tier = self.helpers.object_compatibility(q_compat, p_compat)
                if score > best_pair_score:
                    best_pair_score = score
                    best_pair_detail = {
                        "participant_obj_id": p_obj.obj_id,
                        "participant_template": p_obj.template_name,
                        "score": score,
                        "signals": signals,
                        "tier": tier,
                    }

            per_query_scores.append(best_pair_score)
            comparisons.append({
                "query_object": {
                    "template_name": q_compat["template_name"],
                    "obj_id": q_compat["obj_id"],
                },
                "best_score": best_pair_score,
                "best_match": best_pair_detail,
            })

        total = sum(per_query_scores)

        return total, {
            "per_query_scores": per_query_scores,
            "comparisons": comparisons,
            "total": total,
        }