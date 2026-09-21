#!/usr/bin/env python3
r"""
planner.py – Main planner integrating verb and condition components.

Orchestrates plan generation via:

- VerbComponent for retrieving candidate actions based on target verbs.
- ConditionComponent for backward chaining over state changes and preconditions.
- WorldStateValidator for validating generated plans in two modes:
    * plan-context   (world state built from the plan's own memories)
    * query-context  (world state built from the query's objects/conditions)

Plans are ranked by a blended score. Neither validation mode discards a
plan; both contribute to ranking and both are reported in the trace.

Two-stage planning
==================
Planning runs in exactly two stages, both of which always execute:

    Stage "top10"  — the first memory_batch_size memories from search results
    Stage "all"    — every memory in search results

Both stages append to the same candidate pool. Plans are tagged with their
originating stage so the trace can distinguish them. Ranking happens once
over the combined pool at the end.

The verb component is filtered by the currently allowed memories (passed
explicitly to retrieve_candidates) so that stage "top10" genuinely
restricts its candidate set. The planning index is not modified.

Action-centric multi-goal support
=================================
For queries with more than one target action, an additional pass runs
after the two stages:

    _generate_multi_action_plans(query)

Each target action is planned independently (verb retrieval, per-action
plan), and the per-action plans are combined by Cartesian product. Each
combination competes with single-action plans in the same ranked list.

Coverage bonus
==============
When a query has N > 1 target actions, a plan whose `actions_covered`
equals N receives a coverage bonus of ×N applied to its positive score
components (verb + object + blended satisfaction) before the action-count
and error penalties are subtracted. Partial coverage receives no bonus.
Single-target queries never receive the bonus.

Coverage group
==============
Every ranked plan is tagged with a `coverage_group` field for grouped
output in the CLI and trace:

    verb_guided_all      — verb-guided plan covering all target actions
    verb_guided_partial  — verb-guided plan covering only some targets
    general              — general backward-chaining plan

Trace
=====
self.full_trace holds one entry per stage, tagged with "stage" and
"batch_index". self.multi_action_trace holds entries for the multi-action
pass. self.all_generated_plans retains the pre-threshold pool.
"""

import itertools
from typing import Any, Dict, List, Optional, Tuple

from memory_model import ActionInstance, Memory
from planner_planning_index import PlanningIndex
from planner_verb_component import VerbComponent
from planner_condition_component import ConditionComponent
from planner_world_state_validator import WorldState, WorldStateValidator


class Planner:
    """Main planning orchestration class."""

    # ------------------------------------------------------------------
    # Multi-action tuning parameters (class attributes; override on instance)
    # ------------------------------------------------------------------
    multi_action_enabled: bool = True
    multi_action_per_action_plan_cap: int = 3
    multi_action_max_combinations: int = 20
    multi_action_max_seeds_per_action: int = 10

    def __init__(
        self,
        planning_index: PlanningIndex,
        initial_world_state: Optional[WorldState] = None,
        trace_enabled: bool = True,
    ):
        self.pidx = planning_index
        self.initial_world_state = initial_world_state
        self.trace_enabled = trace_enabled

        self.verb_component = VerbComponent(planning_index)
        self.condition_component = ConditionComponent(planning_index)
        self.validator = WorldStateValidator(planning_index)

        self.memory_batch_size = planning_index.memory_batch_size
        self.planning_depth_limit = planning_index.planning_depth_limit
        self.max_total_actions = planning_index.max_total_actions
        self.verb_score_weight = planning_index.verb_score_weight
        self.object_score_weight = planning_index.object_score_weight
        self.goal_satisfaction_weight = planning_index.goal_satisfaction_weight
        self.action_count_penalty = planning_index.action_count_penalty
        self.validation_error_penalty = planning_index.validation_error_penalty
        self.plan_score_threshold = planning_index.plan_score_threshold

        self.full_trace: List[Dict[str, Any]] = []
        self.all_generated_plans: List[Dict[str, Any]] = []
        self.multi_action_trace: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plan(
        self,
        query: Dict[str, Any],
        search_results: List[Dict[str, Any]],
        max_batches: Optional[int] = None,   # kept for caller compat; unused
    ) -> List[Dict[str, Any]]:
        """
        Generate ranked plans from the structured query and search results.

        Runs two stages sequentially: the top memory_batch_size memories,
        then all memories. Both always run. Results are combined and ranked
        once.
        """
        self.full_trace = []
        self.all_generated_plans = []
        self.multi_action_trace = []

        if not query.get("goal_conditions"):
            return [{"error": "No goal conditions provided"}]

        target_verbs = query.get("target_verbs", [])
        target_actions = query.get("target_actions", [])

        memory_ids: List[str] = []
        for res in search_results:
            if isinstance(res.get("memory"), Memory):
                memory_ids.append(res["memory"].id)
            else:
                mid = res.get("memory_id")
                if mid:
                    memory_ids.append(mid)

        first_batch = memory_ids[: self.memory_batch_size]
        all_memories = list(memory_ids)

        print(f"[DIAG] Total memories available: {len(memory_ids)}")
        print(f"[DIAG] Stage 'top10' will use: {first_batch}")
        print(f"[DIAG] Stage 'all' will use:   {len(all_memories)} memories")
        print(f"[DIAG] Target verbs: {target_verbs}")
        print(f"[DIAG] Target actions: {[a.get('name') for a in target_actions]}")
        print(f"[DIAG] Goal conditions: {query.get('goal_conditions')}")
        print(f"[DIAG] Initial conditions: {query.get('initial_conditions')}")
        print(f"[DIAG] Query objects: {[o.get('template_name') for o in query.get('objects', [])]}")

        plans: List[Dict[str, Any]] = []

        # ---- Stage 1: top-N memories ----
        print(f"\n[DIAG] === Stage 'top10' ({len(first_batch)} memories) ===")
        stage1_plans = self._run_stage(
            memories=first_batch,
            query=query,
            stage_label="top10",
            batch_index=0,
        )
        print(f"[DIAG] Stage 'top10' produced {len(stage1_plans)} plans")
        plans.extend(stage1_plans)

        # ---- Stage 2: all memories ----
        print(f"\n[DIAG] === Stage 'all' ({len(all_memories)} memories) ===")
        stage2_plans = self._run_stage(
            memories=all_memories,
            query=query,
            stage_label="all",
            batch_index=1,
        )
        print(f"[DIAG] Stage 'all' produced {len(stage2_plans)} plans")
        plans.extend(stage2_plans)

        print(f"\n[DIAG] Total plans accumulated across stages: {len(plans)}")

        # ---- Action-centric multi-goal planning ----
        if self.multi_action_enabled and len(target_actions) > 1:
            print(f"\n[DIAG] === Multi-action pass ({len(target_actions)} actions) ===")
            multi_action_plans = self._generate_multi_action_plans(query)
            for p in multi_action_plans:
                p.setdefault("stage", "multi_action")
            print(f"[DIAG] Multi-action pass produced {len(multi_action_plans)} combined plans")
            plans.extend(multi_action_plans)

        # ---- Single ranking pass over all candidate plans ----
        ranked = self._rank_plans(
            plans,
            query.get("goal_conditions", []),
            query_objects=query.get("objects", []),
            initial_conditions=query.get("initial_conditions", []),
            total_target_actions=len(target_actions),
        )

        return ranked

    # ------------------------------------------------------------------
    # Stage runner
    # ------------------------------------------------------------------
    def _run_stage(
        self,
        memories: List[str],
        query: Dict[str, Any],
        stage_label: str,
        batch_index: int,
    ) -> List[Dict[str, Any]]:
        """
        Run one stage of planning: set the allowed memories, execute the
        verb-guided and general backward-chaining paths over that set, tag
        every produced plan with the stage label, and append a stage entry
        to full_trace.
        """
        self._set_allowed_memories(memories)

        plans = self._generate_plans_for_batch(
            target_verbs=query.get("target_verbs", []),
            query_objects=query.get("objects", []),
            initial_conditions=query.get("initial_conditions", []),
            goal_conditions=query.get("goal_conditions", []),
            query_action_category=query.get("action_category"),
            batch_memory_ids=list(memories),
            batch_index=batch_index,
            stage_label=stage_label,
        )

        for p in plans:
            p["stage"] = stage_label

        return plans

    def _set_allowed_memories(self, memory_ids: Optional[List[str]]):
        """
        Set the allowed memories on the condition component. The verb
        component receives the same set via an explicit parameter on each
        retrieve_candidates call. The planning index is not modified.
        """
        if memory_ids is None or len(memory_ids) == 0:
            self.condition_component.clear_allowed_memories()
        else:
            self.condition_component.set_allowed_memories(list(memory_ids))

    # ------------------------------------------------------------------
    # Batch planning (called once per stage)
    # ------------------------------------------------------------------
    def _generate_plans_for_batch(
        self,
        target_verbs: List[str],
        query_objects: List[Dict[str, Any]],
        initial_conditions: List[Dict[str, Any]],
        goal_conditions: List[Dict[str, Any]],
        query_action_category: Optional[str],
        batch_memory_ids: List[str],
        batch_index: int,
        stage_label: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        plans: List[Dict[str, Any]] = []
        batch_trace_entry: Dict[str, Any] = {
            "batch_index": batch_index,
            "stage": stage_label,
            "memories": batch_memory_ids,
            "verb_candidates": [],
            "condition_trace": [],
            "seed_plans": [],
        }

        if target_verbs:
            verb_trace: List[Dict[str, Any]] = []
            candidates = self.verb_component.retrieve_candidates(
                verbs=target_verbs,
                query_objects=query_objects,
                is_target=True,
                query_action_category=query_action_category,
                trace=verb_trace,
                allowed_memory_ids=self.condition_component.allowed_memory_ids,
            )
            batch_trace_entry["verb_candidates"] = verb_trace

            print(f"[DIAG] Verb component returned {len(candidates)} candidates")

            for idx, (action, scores) in enumerate(candidates, 1):
                plan_result = self._build_plan_from_seed(
                    seed_action=action,
                    seed_scores=scores,
                    initial_conditions=initial_conditions,
                    goal_conditions=goal_conditions,
                    query_objects=query_objects,
                    shared_verb_trace=verb_trace,
                )
                if plan_result:
                    print(
                        f"[DIAG] Candidate {idx}/{len(candidates)} "
                        f"({action.template_name}) produced a plan with "
                        f"{len(plan_result['actions'])} actions, "
                        f"score {plan_result.get('scores', {}).get('total', 0.0):.4f}"
                    )
                    plans.append(plan_result)
                    batch_trace_entry["seed_plans"].append({
                        "seed_action_instance_id": action.instance_id,
                        "seed_action_template": action.template_name,
                        "plan_length": len(plan_result["actions"]),
                        "seed_scores": scores,
                    })
                else:
                    print(
                        f"[DIAG] Candidate {idx}/{len(candidates)} "
                        f"({action.template_name}) failed backward chaining"
                    )
                    batch_trace_entry["seed_plans"].append({
                        "seed_action_instance_id": action.instance_id,
                        "seed_action_template": action.template_name,
                        "plan_length": 0,
                        "failure": "backward_chaining_failed",
                    })

        condition_trace: List[Dict[str, Any]] = []
        general_plan, general_diag = self.condition_component.generate_plan(
            goal_conditions=goal_conditions,
            initial_conditions=initial_conditions,
            query_objects=query_objects,
            max_depth=self.planning_depth_limit,
            trace=condition_trace,
        )
        batch_trace_entry["condition_trace"] = condition_trace

        if general_plan:
            print(
                f"[DIAG] General backward chaining produced a plan with "
                f"{len(general_plan)} actions"
            )
            plan = {
                "actions": general_plan,
                "scores": {"verb": 0.0, "object": 0.0, "category": 0.0, "total": 0.0},
                "plan_length": len(general_plan),
                "type": "general_backward_chaining",
                "substitutions": general_diag.get("substitutions", []),
                "trace": {
                    "type": "general_backward_chaining",
                    "batch_index": batch_index,
                    "stage": stage_label,
                    "condition_trace": condition_trace,
                } if self.trace_enabled else None,
            }
            plans.append(plan)
        else:
            print(
                f"[DIAG] General backward chaining failed: "
                f"{general_diag.get('failure')}"
            )

        print(f"[DIAG] Stage '{stage_label}' plans before ranking: {len(plans)}")

        self.full_trace.append(batch_trace_entry)

        return plans

    # ------------------------------------------------------------------
    # Seed-based plan construction
    # ------------------------------------------------------------------
    def _build_plan_from_seed(
        self,
        seed_action: ActionInstance,
        seed_scores: Dict[str, float],
        initial_conditions: List[Dict[str, Any]],
        goal_conditions: List[Dict[str, Any]],
        query_objects: List[Dict[str, Any]],
        shared_verb_trace: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Build a plan whose terminal action is the seed. Backward chaining
        satisfies the seed's preconditions and old states starting from the
        initial conditions.

        `goal_conditions` is accepted for API symmetry; the seed is the
        terminal action, and chaining is driven by the seed's own
        preconditions. Pass [] when the seed is a target action.
        """
        preconditions = self.condition_component._extract_preconditions(seed_action)
        old_states = self.condition_component._extract_old_states(seed_action)

        subgoals: List[Dict[str, Any]] = []
        for cond in preconditions + old_states:
            if not self.condition_component._is_condition_satisfied(
                cond, initial_conditions, query_objects
            ):
                subgoals.append(cond)

        condition_trace: List[Dict[str, Any]] = []
        prefix_actions: List[ActionInstance] = []
        prefix_substitutions: List[Dict[str, Any]] = []
        failure: Optional[Dict[str, Any]] = None

        if subgoals:
            plan_rev, diag = self.condition_component.generate_plan(
                goal_conditions=subgoals,
                initial_conditions=initial_conditions,
                query_objects=query_objects,
                max_depth=self.planning_depth_limit,
                trace=condition_trace,
            )
            if not plan_rev:
                failure = diag.get("failure", {"reason": "unknown"})
                return None
            prefix_actions = plan_rev
            prefix_substitutions = diag.get("substitutions", [])
        else:
            condition_trace = []

        total_actions = prefix_actions + [seed_action]
        all_substitutions = list(prefix_substitutions)

        plan_trace = {
            "type": "verb_guided",
            "seed_action_instance_id": seed_action.instance_id,
            "seed_action_template": seed_action.template_name,
            "seed_scores": seed_scores,
            "subgoals_before_backward_chaining": subgoals,
            "condition_trace": condition_trace,
            "substitutions": all_substitutions,
        }

        return {
            "actions": total_actions,
            "scores": seed_scores,
            "plan_length": len(total_actions),
            "type": "verb_guided",
            "substitutions": all_substitutions,
            "trace": plan_trace if self.trace_enabled else None,
        }

    # ------------------------------------------------------------------
    # Action-centric multi-goal planning
    # ------------------------------------------------------------------
    def _generate_multi_action_plans(
        self,
        query: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Run the verb-guided path once per target action, then combine the
        per-action plans by Cartesian product (capped).
        """
        target_actions = query.get("target_actions", [])
        if not target_actions:
            target_actions = [
                {"name": v, "verb": v, "participants": []}
                for v in query.get("target_verbs", [])
            ]
        if not target_actions:
            return []

        query_objects = query.get("objects", [])
        initial_conditions = query.get("initial_conditions", [])
        query_action_category = query.get("action_category")

        # Multi-action runs over all memories.
        self._set_allowed_memories(None)

        per_action_plans: List[List[Dict[str, Any]]] = []
        per_action_records: List[Dict[str, Any]] = []

        for a_idx, target_action in enumerate(target_actions):
            verb = target_action.get("verb") or target_action.get("name")
            target_name = target_action.get("name")

            verb_trace: List[Dict[str, Any]] = []
            candidates = self.verb_component.retrieve_candidates(
                verbs=[verb],
                query_objects=query_objects,
                is_target=True,
                query_action_category=query_action_category,
                trace=verb_trace,
                allowed_memory_ids=self.condition_component.allowed_memory_ids,
            )

            action_plans: List[Dict[str, Any]] = []
            seeds_considered = 0
            seed_records: List[Dict[str, Any]] = []

            for action, scores in candidates:
                if seeds_considered >= self.multi_action_max_seeds_per_action:
                    break
                seeds_considered += 1

                record = {
                    "seed_template": action.template_name,
                    "seed_instance_id": action.instance_id,
                    "verb_score": scores.get("verb", 0.0),
                    "object_score": scores.get("object", 0.0),
                    "category_score": scores.get("category", 0.0),
                    "total_score": scores.get("total", 0.0),
                    "plan_produced": False,
                    "plan_length": None,
                    "failure": None,
                }

                plan_result = self._build_plan_from_seed(
                    seed_action=action,
                    seed_scores=scores,
                    initial_conditions=initial_conditions,
                    goal_conditions=[],
                    query_objects=query_objects,
                    shared_verb_trace=verb_trace,
                )
                if plan_result:
                    record["plan_produced"] = True
                    record["plan_length"] = len(plan_result["actions"])
                    action_plans.append(plan_result)
                else:
                    record["failure"] = "backward_chaining_failed"
                seed_records.append(record)

            action_plans = action_plans[: self.multi_action_per_action_plan_cap]
            per_action_plans.append(action_plans)
            per_action_records.append({
                "action_index": a_idx,
                "target_action": target_action,
                "target_action_name": target_name,
                "verb_used": verb,
                "candidates_retrieved": len(candidates),
                "seeds_considered": seeds_considered,
                "plans_produced": len(action_plans),
                "plan_lengths": [len(p["actions"]) for p in action_plans],
                "seed_records": seed_records,
            })
            print(
                f"[DIAG] Action {a_idx} ({target_name}, verb={verb}): "
                f"{len(action_plans)} plans produced "
                f"(lengths: {[len(p['actions']) for p in action_plans]})"
            )

        if any(not g for g in per_action_plans):
            failed = [i for i, g in enumerate(per_action_plans) if not g]
            self.multi_action_trace.append({
                "type": "multi_action_pass",
                "status": "failed",
                "reason": "at least one target action has no plan",
                "failed_action_indices": failed,
                "action_count": len(target_actions),
                "target_actions": target_actions,
                "per_action": per_action_records,
                "per_action_plan_cap": self.multi_action_per_action_plan_cap,
                "max_combinations": self.multi_action_max_combinations,
                "combinations_generated": 0,
                "combinations": [],
            })
            return []

        combined: List[Dict[str, Any]] = []
        combination_records: List[Dict[str, Any]] = []
        truncated = False

        for combo_idx, combo in enumerate(itertools.product(*per_action_plans)):
            if combo_idx >= self.multi_action_max_combinations:
                truncated = True
                break

            combined_actions: List[ActionInstance] = []
            combined_subs: List[Dict[str, Any]] = []
            score_sums = {"verb": 0.0, "object": 0.0, "category": 0.0}
            per_action_detail: List[Dict[str, Any]] = []

            for a_idx, p in enumerate(combo):
                combined_actions.extend(p["actions"])
                combined_subs.extend(p.get("substitutions", []))
                p_scores = p.get("scores", {})
                for k in score_sums:
                    score_sums[k] += p_scores.get(k, 0.0)

                p_trace = p.get("trace") or {}
                per_action_detail.append({
                    "action_index": a_idx,
                    "target_action": target_actions[a_idx],
                    "seed_template": p_trace.get("seed_action_template"),
                    "seed_instance_id": p_trace.get("seed_action_instance_id"),
                    "seed_scores": dict(p_scores),
                    "plan_length": len(p["actions"]),
                    "substitutions": p.get("substitutions", []),
                })

            n = len(combo)
            avg_scores = {k: v / n for k, v in score_sums.items()}
            avg_scores["total"] = sum(avg_scores[k] for k in ("verb", "object", "category"))

            combined.append({
                "actions": combined_actions,
                "scores": avg_scores,
                "plan_length": len(combined_actions),
                "type": "multi_action_combined",
                "substitutions": combined_subs,
                "actions_covered": n,
                "stage": "multi_action",
                "trace": {
                    "type": "multi_action_combined",
                    "combo_index": combo_idx,
                    "per_action": per_action_detail,
                } if self.trace_enabled else None,
            })

            combination_records.append({
                "combo_index": combo_idx,
                "total_actions": len(combined_actions),
                "avg_scores": dict(avg_scores),
                "per_action": [
                    {
                        "action_index": d["action_index"],
                        "target_action_name": target_actions[d["action_index"]].get("name"),
                        "seed_template": d["seed_template"],
                        "plan_length": d["plan_length"],
                    }
                    for d in per_action_detail
                ],
            })

        self.multi_action_trace.append({
            "type": "multi_action_pass",
            "status": "ok",
            "action_count": len(target_actions),
            "target_actions": target_actions,
            "per_action": per_action_records,
            "per_action_plan_cap": self.multi_action_per_action_plan_cap,
            "max_combinations": self.multi_action_max_combinations,
            "combinations_generated": len(combined),
            "combinations_truncated_by_cap": truncated,
            "combinations": combination_records,
        })

        return combined

    # ------------------------------------------------------------------
    # Coverage group classification
    # ------------------------------------------------------------------
    def _coverage_group(self, plan: Dict[str, Any], total_target_actions: int) -> str:
        """
        Classify a plan for grouped output:
            verb_guided_all      — verb-guided plan covering every target action
            verb_guided_partial  — verb-guided plan covering only some targets
            general              — general backward-chaining plan
        """
        if plan.get("type") == "general_backward_chaining":
            return "general"

        covered = plan.get("actions_covered")
        if covered is None:
            # Single-seed verb-guided plan covers exactly one target action.
            covered = 1

        if total_target_actions <= 1:
            return "verb_guided_all"
        if covered >= total_target_actions:
            return "verb_guided_all"
        return "verb_guided_partial"

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------
    def _rank_plans(
        self,
        plans: List[Dict[str, Any]],
        goal_conditions: List[Dict[str, Any]],
        query_objects: List[Dict[str, Any]],
        initial_conditions: List[Dict[str, Any]],
        total_target_actions: int = 0,
    ) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []

        for plan in plans:
            actions = plan.get("actions", [])
            scores = plan.get("scores", {})

            plan_validation = self.validator.validate_plan_plan_context(
                plan=actions,
                goal_conditions=goal_conditions,
            )
            query_validation = self.validator.validate_plan_query_context(
                plan=actions,
                query_objects=query_objects,
                initial_conditions=initial_conditions,
                goal_conditions=goal_conditions,
            )

            plan["validation_plan_context"] = plan_validation
            plan["validation_query_context"] = query_validation
            plan["validation"] = plan_validation

            final_score, coverage_bonus = self._compute_plan_score(
                plan,
                plan_validation,
                query_validation,
                total_target_actions=total_target_actions,
            )
            plan["final_score"] = final_score

            coverage_group = self._coverage_group(plan, total_target_actions)
            plan["coverage_group"] = coverage_group

            verb_score = scores.get("verb", 0.0)
            object_score = scores.get("object", 0.0)
            plan_sat = plan_validation.get("goal_satisfaction", 0.0)
            query_sat = query_validation.get("goal_satisfaction", 0.0)
            blended_sat = 0.5 * plan_sat + 0.5 * query_sat
            action_count = len(actions)
            plan_errors = len(plan_validation.get("errors", []))
            query_errors = len(query_validation.get("errors", []))

            plan["score_breakdown"] = {
                "stage": plan.get("stage", "?"),
                "coverage_group": coverage_group,
                "verb_score_raw": verb_score,
                "verb_score_weight": self.verb_score_weight,
                "verb_score_contribution": self.verb_score_weight * verb_score,
                "object_score_raw": object_score,
                "object_score_weight": self.object_score_weight,
                "object_score_contribution": self.object_score_weight * object_score,
                "goal_satisfaction_plan_context": plan_sat,
                "goal_satisfaction_query_context": query_sat,
                "goal_satisfaction_blended": blended_sat,
                "goal_satisfaction_weight": self.goal_satisfaction_weight,
                "goal_satisfaction_contribution": self.goal_satisfaction_weight * blended_sat,
                "coverage_bonus_multiplier": coverage_bonus,
                "action_count": action_count,
                "action_count_penalty_per_action": self.action_count_penalty,
                "action_count_penalty_total": self.action_count_penalty * action_count,
                "plan_context_error_count": plan_errors,
                "query_context_error_count": query_errors,
                "error_penalty_per_error": self.validation_error_penalty,
                "error_penalty_total": self.validation_error_penalty * (plan_errors + query_errors),
                "final_score": final_score,
            }

            ranked.append(plan)

        ranked.sort(key=lambda p: p["final_score"], reverse=True)

        pre_filter = list(ranked)
        self.all_generated_plans = pre_filter
        ranked = [p for p in pre_filter if p["final_score"] >= self.plan_score_threshold]

        print(f"\n[DIAG] Plans before threshold filter: {len(pre_filter)}")
        print(
            f"[DIAG] Plans after threshold filter "
            f"(>= {self.plan_score_threshold}): {len(ranked)}"
        )

        if pre_filter and not ranked:
            print("[DIAG] All plans filtered out. Top 5 rejected:")
            for i, plan in enumerate(pre_filter[:5], 1):
                print(
                    f"[DIAG]   Rejected plan {i}: "
                    f"stage={plan.get('stage', '?')}, "
                    f"group={plan.get('coverage_group', '?')}, "
                    f"score = {plan['final_score']:.4f}, "
                    f"type = {plan.get('type', '?')}"
                )
                pv = plan.get("validation_plan_context", {})
                qv = plan.get("validation_query_context", {})
                print(
                    f"[DIAG]     plan-context: valid={pv.get('valid')}, "
                    f"sat={pv.get('goal_satisfaction'):.4f}, "
                    f"errors={len(pv.get('errors', []))}"
                )
                print(
                    f"[DIAG]     query-context: valid={qv.get('valid')}, "
                    f"sat={qv.get('goal_satisfaction'):.4f}, "
                    f"errors={len(qv.get('errors', []))}"
                )

        return ranked

    def _compute_plan_score(
        self,
        plan: Dict[str, Any],
        plan_validation: Dict[str, Any],
        query_validation: Dict[str, Any],
        total_target_actions: int = 0,
    ) -> Tuple[float, float]:
        """
        Return (final_score, coverage_bonus_applied).

        The coverage bonus is a multiplier in [1.0, total_target_actions]
        applied to the positive score components only. It fires when the
        query has N > 1 target actions and the plan's `actions_covered`
        equals N. It does not modify the penalties.

        Structure:
            positive = verb_weight*verb + object_weight*object
                     + satisfaction_weight*blended_sat
            positive *= coverage_bonus
            final    = positive - action_count_penalty*N_actions
                                - error_penalty*N_errors
        """
        scores = plan.get("scores", {})
        verb_score = scores.get("verb", 0.0)
        object_score = scores.get("object", 0.0)

        plan_sat = plan_validation.get("goal_satisfaction", 0.0)
        query_sat = query_validation.get("goal_satisfaction", 0.0)
        blended_sat = 0.5 * plan_sat + 0.5 * query_sat

        coverage_bonus = 1.0
        if total_target_actions > 1:
            covered = plan.get("actions_covered", 0) or 0
            if covered == total_target_actions:
                coverage_bonus = float(total_target_actions)

        positive = (
            self.verb_score_weight * verb_score
            + self.object_score_weight * object_score
            + self.goal_satisfaction_weight * blended_sat
        )
        positive *= coverage_bonus

        action_count = plan.get("plan_length", len(plan.get("actions", [])))
        error_count = (
            len(plan_validation.get("errors", []))
            + len(query_validation.get("errors", []))
        )

        final_score = (
            positive
            - self.action_count_penalty * action_count
            - self.validation_error_penalty * error_count
        )
        return final_score, coverage_bonus