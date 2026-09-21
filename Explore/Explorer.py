#!/usr/bin/env python3
r"""
Explorer.py – CLI for pickle search with mode switching, drill-down,
live search progress, configurable matched-term viewing, and planning.

Default matched-term view: top 20 by score contribution.
User can enter:
  - a number N to show the top N matched terms
  - 'all' to show all matched terms
  - Enter to go back
  - 'plan' to generate a plan from the current search results

Planning:
  - Top 10 plans per group displayed by default.
  - Plans are grouped into three sections:
        Verb-guided plans — covering all target actions
        Verb-guided plans — covering some target actions
        General backward-chaining plans
  - Traces are overwritten on each planning run:
        plan_trace_simple.txt   – per-action lines, per-substitution lines,
                                  dual validation summaries, stage labels.
        plan_trace_full.txt     – candidate-by-candidate details from the
                                  verb component, condition component, and
                                  both validation modes; the multi-action
                                  pass trace; grouped plan-level traces;
                                  two representative long plans drawn from
                                  the pre-threshold pool; and a stage census
                                  over the full pool.

Stages:
  The planner runs two stages, "top10" (first memory_batch_size memories)
  and "all" (every memory), plus an optional "multi_action" pass. Every
  produced plan carries a `stage` field.

Coverage groups:
  Every ranked plan carries `coverage_group`, one of:
      verb_guided_all
      verb_guided_partial
      general
  The CLI and full trace use this field to group output.

Multi-action planning:
  The planner exposes `multi_action_trace` with one entry per run of the
  action-centric multi-goal pass. Each combined plan also carries a `trace`
  dict with per-action seed detail. Explorer writes both into
  plan_trace_full.txt.

Special commands:
  - 'github' prints the repository URL.
"""

import asyncio
import json
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime

from pickle_loader import PickleMemoryLoader
from pickle_search_index import PickleSearchIndex
from pickle_search_engine import PickleSearchEngine
from pickle_query_processor import PickleQueryProcessor

from planner_planning_index import PlanningIndex
from planner import Planner
from planner_world_state_validator import WorldState

EXPLORE_DIR = Path(__file__).parent
DATA_DIR = EXPLORE_DIR / "data"
SEARCH_RESULTS_READABLE = EXPLORE_DIR / "search_results_readable.txt"
SEARCH_RESULTS_JSONL = DATA_DIR / "search_results.jsonl"

PLAN_TRACE_SIMPLE = EXPLORE_DIR / "plan_trace_simple.txt"
PLAN_TRACE_FULL = EXPLORE_DIR / "plan_trace_full.txt"


# =============================================================================
# Search result persistence
# =============================================================================
def save_search_results(query: str, mode: str, terms: List[Tuple[str, float]], results: List[Dict[str, Any]]):
    with open(SEARCH_RESULTS_READABLE, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Query: {query}\n")
        f.write(f"Mode: {mode}\n")
        f.write("Terms:\n")
        for term, _weight in terms:
            f.write(f"  - {term}\n")
        f.write("=" * 80 + "\n\n")
        for i, res in enumerate(results[:10], 1):
            f.write(f"{i}. {res['memory'].activity} (score: {res['score']:.4f})\n")
            for td in res['term_details']:
                f.write(f"   - {td['query_term']} -> {td['matched_index_term']} ({td['score_contribution']:.4f})\n")
                f.write(f"     Context: {td['source_context']}\n")
            f.write("\n")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "mode": mode,
        "terms": [t for t, _ in terms],
        "results": [
            {
                "memory_id": res['memory'].id,
                "activity": res['memory'].activity,
                "score": res['score'],
                "term_details": [
                    {
                        "query_term": td['query_term'],
                        "matched_index_term": td['matched_index_term'],
                        "score_contribution": td['score_contribution'],
                        "mode": td['mode'],
                        "source_context": td['source_context'],
                        "term_type": td['term_type'],
                    }
                    for td in res['term_details']
                ],
            }
            for res in results[:10]
        ],
    }
    with open(SEARCH_RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =============================================================================
# Search CLI helpers
# =============================================================================
def print_progress(current_query_term: str, q_idx: int, total_q: int, index_examined: int, elapsed: float):
    term_display = current_query_term
    if len(term_display) > 50:
        term_display = term_display[:47] + "..."
    print(
        f"\rTerm: {term_display} | Query {q_idx}/{total_q} | "
        f"Index terms examined: {index_examined} | Elapsed: {elapsed:.1f}s",
        end="", flush=True
    )


def print_matched_terms(memory_id: str, activity: str, term_details: List[Dict[str, Any]], limit: int):
    sorted_terms = sorted(term_details, key=lambda td: td['score_contribution'], reverse=True)
    if limit is not None:
        sorted_terms = sorted_terms[:limit]

    print(f"\nMemory: {memory_id}")
    print(f"Activity: {activity}")
    if limit is not None:
        print(f"Top {limit} matched terms:")
    else:
        print("All matched terms:")

    for i, td in enumerate(sorted_terms, 1):
        print(f"{i}. {td['query_term']} -> {td['matched_index_term']} "
              f"(score {td['score_contribution']:.4f})")


# =============================================================================
# World state from query
# =============================================================================
def build_initial_world_state(planning_query: Dict[str, Any]) -> WorldState:
    """
    Build the query-context world state. Used only for informational
    display in the CLI. The actual query-context validation is performed
    inside the planner via WorldStateValidator.
    """
    ws = WorldState()

    for obj in planning_query.get("objects", []):
        obj_id = obj.get("obj_id") or obj.get("template_name")
        if not obj_id:
            continue
        attrs = obj.get("attributes", {}) or {}
        ws.add_object(obj_id, attrs)

    for cond in planning_query.get("initial_conditions", []):
        obj_id = cond.get("object")
        attr = cond.get("attribute")
        val = cond.get("value")
        aspect = cond.get("aspect")

        if not obj_id or attr is None or val is None:
            continue

        if aspect:
            if isinstance(aspect, str):
                path = f"{aspect}.{attr}" if isinstance(attr, str) else (*aspect.split("."), attr)
            else:
                path = (*aspect, attr)
        else:
            path = attr

        ws.set_attribute(obj_id, path, val)

        if attr == "position" and isinstance(val, dict):
            rel = val.get("relation", "")
            rel_to = val.get("relative_to", "")
            if rel and rel_to:
                ws.add_spatial(obj_id, rel, rel_to)

    return ws


# =============================================================================
# Full-detail formatting helpers
# =============================================================================
def _format_action_full(action, indent: int = 0) -> List[str]:
    """Return a list of lines fully describing one ActionInstance."""
    pad = "  " * indent
    lines = []

    dur = getattr(action, "effective_duration", None) or getattr(action, "duration", None)
    cat = getattr(action, "action_category", "") or ""
    lines.append(
        f"{pad}- {action.template_name}"
        f"  (id={getattr(action, 'instance_id', '?')}, "
        f"dur={dur}s, cat={cat})"
    )

    if getattr(action, "participants", None):
        parts = ", ".join(p.obj_id for p in action.participants if p)
        lines.append(f"{pad}    participants: {parts}")

    traj = getattr(action, "kinematic_trajectory", "")
    if traj:
        lines.append(f"{pad}    trajectory: {traj}")

    tt = getattr(action, "temporal_type", "")
    if tt:
        lines.append(f"{pad}    temporal_type: {tt}")

    tags = getattr(action, "tags", None)
    if tags:
        lines.append(f"{pad}    tags: {', '.join(tags)}")

    precs = getattr(action, "preconditions", None) or []
    if precs:
        lines.append(f"{pad}    preconditions:")
        for prec in precs:
            lines.append(f"{pad}      {json.dumps(prec, ensure_ascii=False)}")

    for list_name in ("changes", "changes_per_cycle", "changes_total"):
        items = getattr(action, list_name, None) or []
        if items:
            lines.append(f"{pad}    {list_name}:")
            for ch in items:
                lines.append(f"{pad}      {json.dumps(ch, ensure_ascii=False)}")

    conds = getattr(action, "conditional_changes", None) or []
    if conds:
        lines.append(f"{pad}    conditional_changes:")
        for cc in conds:
            lines.append(f"{pad}      {json.dumps(cc, ensure_ascii=False)}")

    subs = getattr(action, "sub_actions", None) or []
    if subs:
        lines.append(f"{pad}    sub_actions:")
        for sub in subs:
            lines.extend(_format_action_full(sub, indent + 2))

    return lines


def _format_plan_full(plan: dict, index: int) -> List[str]:
    """Return a list of lines fully describing one ranked plan."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"PLAN #{index}")
    lines.append("=" * 80)
    lines.append(f"  type:           {plan.get('type', '?')}")
    lines.append(f"  stage:          {plan.get('stage', '?')}")
    lines.append(f"  coverage_group: {plan.get('coverage_group', '?')}")
    lines.append(f"  final_score:    {plan.get('final_score', 0.0):.4f}")
    lines.append(f"  action_count:   {len(plan.get('actions', []))}")
    if plan.get("plan_length") is not None:
        lines.append(f"  plan_length:    {plan['plan_length']}")
    if plan.get("actions_covered") is not None:
        lines.append(f"  actions_covered:{plan['actions_covered']}")
    if plan.get("goals_covered") is not None:
        lines.append(f"  goals_covered:  {plan['goals_covered']}")

    scores = plan.get("scores") or {}
    if scores:
        lines.append("  seed_scores:")
        for k, v in scores.items():
            lines.append(f"    {k}: {v}")

    subs = plan.get("substitutions") or []
    if subs:
        lines.append(f"  substitutions ({len(subs)}):")
        for sub in subs:
            lines.append(f"    - {json.dumps(sub, ensure_ascii=False)}")
    else:
        lines.append("  substitutions: none")

    actions = plan.get("actions") or []
    lines.append(f"  actions ({len(actions)}):")
    for i, action in enumerate(actions, 1):
        lines.append(f"    [{i}]")
        lines.extend(_format_action_full(action, indent=3))

    breakdown = plan.get("score_breakdown")
    if breakdown:
        lines.append("  score_breakdown:")
        for k, v in breakdown.items():
            lines.append(f"    {k}: {v}")

    plan_trace = plan.get("trace")
    if plan_trace:
        lines.append("  plan_trace:")
        for trace_line in json.dumps(plan_trace, indent=2, ensure_ascii=False).splitlines():
            lines.append("    " + trace_line)

    for key in ("validation_plan_context", "validation_query_context"):
        v = plan.get(key)
        if not isinstance(v, dict):
            continue
        lines.append(f"  {key}:")
        lines.append(f"    valid:             {v.get('valid')}")
        lines.append(f"    goal_satisfaction: {v.get('goal_satisfaction')}")
        errors = v.get("errors") or []
        if errors:
            lines.append(f"    errors ({len(errors)}):")
            for e in errors:
                lines.append(f"      - {e}")
        unmet = v.get("unmet_goals") or []
        if unmet:
            lines.append(f"    unmet_goals ({len(unmet)}):")
            for g in unmet:
                lines.append(f"      - {json.dumps(g, ensure_ascii=False)}")

    lines.append("")
    return lines


def _select_long_plans(pool: List[Dict[str, Any]], thresholds=(10, 20)) -> List[Tuple[int, Any]]:
    """
    For each threshold, return the shortest plan with strictly more than
    `threshold` actions. If several plans tie on that length, pick one at
    random. If no plan exceeds the threshold, return (threshold, None).
    """
    picks: List[Tuple[int, Any]] = []
    for t in thresholds:
        candidates = [p for p in pool if len(p.get("actions", [])) > t]
        if not candidates:
            picks.append((t, None))
            continue
        min_len = min(len(p["actions"]) for p in candidates)
        tied = [p for p in candidates if len(p["actions"]) == min_len]
        picks.append((t, random.choice(tied)))
    return picks


# =============================================================================
# Grouped plan printers (CLI)
# =============================================================================
def _print_plan_group(title: str, group_plans: List[Dict[str, Any]], limit: int = 10):
    print(f"\n{title}")
    if not group_plans:
        print("  (none)")
        return
    for i, plan in enumerate(group_plans[:limit], 1):
        print(f"\n  {i}. Score: {plan.get('final_score', 0.0):.4f}  "
              f"Stage: {plan.get('stage', '?')}  "
              f"Type: {plan.get('type', '?')}")
        if plan.get("actions_covered") is not None:
            print(f"     Actions covered: {plan['actions_covered']}")
        if plan.get("goals_covered") is not None:
            print(f"     Goals covered: {plan['goals_covered']}")

        actions = plan.get("actions", [])
        for j, action in enumerate(actions, 1):
            name = action.template_name if hasattr(action, "template_name") else str(action)
            print(f"       {j}. {name}")

        subs = plan.get("substitutions", [])
        if subs:
            print("     Substitutions:")
            for s in subs:
                print(
                    f"       - {s.get('original_object')} -> {s.get('substituted_object')} "
                    f"(sim {s.get('similarity'):.3f}, signals {s.get('signals')})"
                )

        pv = plan.get("validation_plan_context", {})
        qv = plan.get("validation_query_context", {})
        print(f"     Plan-context: valid={pv.get('valid')} "
              f"sat={pv.get('goal_satisfaction', 0.0):.4f} "
              f"errors={len(pv.get('errors', []))}")
        print(f"     Query-context: valid={qv.get('valid')} "
              f"sat={qv.get('goal_satisfaction', 0.0):.4f} "
              f"errors={len(qv.get('errors', []))}")


# =============================================================================
# Trace file writers
# =============================================================================
def _fmt_float(x, digits=4) -> str:
    if isinstance(x, (int, float)):
        return f"{x:.{digits}f}"
    return str(x)


def save_plan_traces_simple(plans: List[Dict[str, Any]]):
    """Overwrites plan_trace_simple.txt with grouped per-plan summaries."""
    groups = [
        ("VERB-GUIDED PLANS — COVERING ALL TARGET ACTIONS",
         [p for p in plans if p.get("coverage_group") == "verb_guided_all"]),
        ("VERB-GUIDED PLANS — COVERING SOME TARGET ACTIONS",
         [p for p in plans if p.get("coverage_group") == "verb_guided_partial"]),
        ("GENERAL BACKWARD-CHAINING PLANS",
         [p for p in plans if p.get("coverage_group") == "general"]),
    ]

    with open(PLAN_TRACE_SIMPLE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PLAN TRACES (SIMPLE)\n")
        f.write(f"Generated at: {datetime.now().isoformat()}\n")
        f.write(f"Total plans: {len(plans)}\n")
        f.write("=" * 80 + "\n\n")

        for group_title, group_plans in groups:
            f.write("#" * 80 + "\n")
            f.write(f"# {group_title}  ({len(group_plans)} plans)\n")
            f.write("#" * 80 + "\n\n")

            if not group_plans:
                f.write("  (none)\n\n")
                continue

            for i, plan in enumerate(group_plans[:10], 1):
                f.write(f"--- PLAN {i} ---\n")
                f.write("SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Type: {plan.get('type', 'unknown')}\n")
                f.write(f"Stage: {plan.get('stage', '?')}\n")
                f.write(f"Coverage group: {plan.get('coverage_group', '?')}\n")
                f.write(f"Final score: {_fmt_float(plan.get('final_score', 0.0))}\n")
                actions = plan.get("actions", [])
                f.write(f"Action count: {len(actions)}\n")
                if plan.get("actions_covered") is not None:
                    f.write(f"Actions covered: {plan['actions_covered']}\n")
                if plan.get("goals_covered") is not None:
                    f.write(f"Goals covered: {plan['goals_covered']}\n")

                f.write("\nACTIONS\n")
                f.write("-" * 40 + "\n")
                for j, action in enumerate(actions, 1):
                    name = action.template_name if hasattr(action, "template_name") else str(action)
                    f.write(f"  {j}. {name}\n")

                subs = plan.get("substitutions", [])
                f.write("\nSUBSTITUTIONS\n")
                f.write("-" * 40 + "\n")
                if not subs:
                    f.write("  (none)\n")
                else:
                    for s in subs:
                        f.write(
                            f"  [goal] action={s.get('action_template')} "
                            f"original={s.get('original_object')} "
                            f"substituted={s.get('substituted_object')} "
                            f"sim={_fmt_float(s.get('similarity'))} "
                            f"tier={s.get('tier')} "
                            f"signals={s.get('signals')}\n"
                        )

                f.write("\nSCORE BREAKDOWN\n")
                f.write("-" * 40 + "\n")
                for k, v in plan.get("score_breakdown", {}).items():
                    if isinstance(v, float):
                        f.write(f"  {k}: {_fmt_float(v)}\n")
                    else:
                        f.write(f"  {k}: {v}\n")

                pv = plan.get("validation_plan_context", {})
                qv = plan.get("validation_query_context", {})

                f.write("\nVALIDATION: PLAN CONTEXT\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Valid: {pv.get('valid')}\n")
                f.write(f"  Goal satisfaction: {_fmt_float(pv.get('goal_satisfaction', 0.0))}\n")
                if pv.get("errors"):
                    f.write("  Errors:\n")
                    for err in pv["errors"]:
                        f.write(f"    - {err}\n")
                if pv.get("unmet_goals"):
                    f.write("  Unmet goals:\n")
                    for g in pv["unmet_goals"]:
                        cond = g.get("condition", {})
                        f.write(
                            f"    - {cond.get('object')}.{cond.get('attribute')} "
                            f"{cond.get('op','eq')} {cond.get('value')} "
                            f"(actual={g.get('actual_value')}, msg={g.get('message')})\n"
                        )

                f.write("\nVALIDATION: QUERY CONTEXT\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Valid: {qv.get('valid')}\n")
                f.write(f"  Goal satisfaction: {_fmt_float(qv.get('goal_satisfaction', 0.0))}\n")
                if qv.get("errors"):
                    f.write("  Errors:\n")
                    for err in qv["errors"]:
                        f.write(f"    - {err}\n")
                if qv.get("unmet_goals"):
                    f.write("  Unmet goals:\n")
                    for g in qv["unmet_goals"]:
                        cond = g.get("condition", {})
                        f.write(
                            f"    - {cond.get('object')}.{cond.get('attribute')} "
                            f"{cond.get('op','eq')} {cond.get('value')} "
                            f"(actual={g.get('actual_value')}, msg={g.get('message')})\n"
                        )

                f.write("\n\n")


def _write_per_plan_full_block(f, plan: Dict[str, Any], index: int):
    """Write the full detail block for a single plan into an open file."""
    f.write("#" * 80 + "\n")
    f.write(f"# PLAN {index} — FULL TRACE\n")
    f.write("#" * 80 + "\n\n")

    f.write(f"Type: {plan.get('type')}\n")
    f.write(f"Stage: {plan.get('stage', '?')}\n")
    f.write(f"Coverage group: {plan.get('coverage_group', '?')}\n")
    f.write(f"Final score: {_fmt_float(plan.get('final_score', 0.0))}\n")
    f.write(f"Action count: {len(plan.get('actions', []))}\n")
    if plan.get("actions_covered") is not None:
        f.write(f"Actions covered: {plan['actions_covered']}\n")
    if plan.get("goals_covered") is not None:
        f.write(f"Goals covered: {plan['goals_covered']}\n")
    f.write("\n")

    pv = plan.get("validation_plan_context", {})
    qv = plan.get("validation_query_context", {})

    f.write("PLAN-CONTEXT VALIDATION (full)\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Valid: {pv.get('valid')}\n")
    f.write(f"  Goal satisfaction: {_fmt_float(pv.get('goal_satisfaction', 0.0))}\n")
    f.write(f"  Objects in state: {pv.get('objects_in_state')}\n")
    f.write("  Precondition checks:\n")
    for pc in pv.get("precondition_checks", []):
        cond = pc.get("condition", {})
        f.write(
            f"    [{pc.get('level')}] {cond.get('object')}.{cond.get('attribute')} "
            f"{cond.get('op','eq')} {cond.get('value')} "
            f"-> score={_fmt_float(pc.get('score', 0.0))} "
            f"passed={pc.get('passed')} "
            f"msg={pc.get('message')}\n"
        )
    f.write("  Goal checks:\n")
    for g in pv.get("goal_checks", []):
        cond = g.get("condition", {})
        f.write(
            f"    {cond.get('object')}.{cond.get('attribute')} "
            f"{cond.get('op','eq')} {cond.get('value')} "
            f"-> score={_fmt_float(g.get('score', 0.0))} "
            f"passed={g.get('passed')} "
            f"actual={g.get('actual_value')} "
            f"msg={g.get('message')}\n"
        )

    f.write("\nQUERY-CONTEXT VALIDATION (full)\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Valid: {qv.get('valid')}\n")
    f.write(f"  Goal satisfaction: {_fmt_float(qv.get('goal_satisfaction', 0.0))}\n")
    f.write(f"  Objects in state: {qv.get('objects_in_state')}\n")
    f.write("  Precondition checks:\n")
    for pc in qv.get("precondition_checks", []):
        cond = pc.get("condition", {})
        f.write(
            f"    [{pc.get('level')}] {cond.get('object')}.{cond.get('attribute')} "
            f"{cond.get('op','eq')} {cond.get('value')} "
            f"-> score={_fmt_float(pc.get('score', 0.0))} "
            f"passed={pc.get('passed')} "
            f"msg={pc.get('message')}\n"
        )
    f.write("  Goal checks:\n")
    for g in qv.get("goal_checks", []):
        cond = g.get("condition", {})
        f.write(
            f"    {cond.get('object')}.{cond.get('attribute')} "
            f"{cond.get('op','eq')} {cond.get('value')} "
            f"-> score={_fmt_float(g.get('score', 0.0))} "
            f"passed={g.get('passed')} "
            f"actual={g.get('actual_value')} "
            f"msg={g.get('message')}\n"
        )

    plan_trace = plan.get("trace")
    if plan_trace:
        f.write("\nPLAN TRACE\n")
        f.write("-" * 40 + "\n")
        f.write(json.dumps(plan_trace, indent=2, ensure_ascii=False))
        f.write("\n")

    f.write("\n\n")


def save_plan_traces_full(planner: Planner, plans: List[Dict[str, Any]]):
    """Overwrites plan_trace_full.txt with all candidate-by-candidate details."""
    with open(PLAN_TRACE_FULL, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PLAN TRACES (FULL)\n")
        f.write(f"Generated at: {datetime.now().isoformat()}\n")
        f.write(f"Total plans: {len(plans)}\n")
        f.write("=" * 80 + "\n\n")

        # ------------------------------------------------------------
        # Stage-level traces
        # ------------------------------------------------------------
        f.write("STAGE-LEVEL PLANNING TRACE\n")
        f.write("=" * 80 + "\n\n")
        for batch_entry in planner.full_trace:
            stage = batch_entry.get("stage", "?")
            batch_idx = batch_entry.get("batch_index", "?")
            f.write(f"Stage: {stage}  (batch_index={batch_idx})\n")
            f.write(f"Memories: {batch_entry.get('memories')}\n\n")

            f.write("VERB CANDIDATES\n")
            f.write("-" * 40 + "\n")
            for cand in batch_entry.get("verb_candidates", []):
                f.write(
                    f"  verb={cand.get('verb')} "
                    f"action={cand.get('action_template')} "
                    f"instance={cand.get('action_instance_id')}\n"
                )
                f.write(f"    scores: {cand.get('scores')}\n")
                det = cand.get("details", {})
                v_det = det.get("verb", {})
                f.write(
                    f"    verb match: type={v_det.get('match_type')} "
                    f"matched_verb={v_det.get('matched_verb')} "
                    f"score={_fmt_float(v_det.get('score', 0.0))}\n"
                )
                o_det = det.get("object", {})
                f.write(
                    f"    object score: {_fmt_float(o_det.get('total', 0.0))}\n"
                )
                for comp in o_det.get("comparisons", []):
                    q = comp.get("query_object", {})
                    f.write(
                        f"      query={q.get('template_name')} "
                        f"best_score={_fmt_float(comp.get('best_score', 0.0))}\n"
                    )
                    bm = comp.get("best_match")
                    if bm:
                        f.write(
                            f"        matched={bm.get('participant_obj_id')} "
                            f"tier={bm.get('tier')} "
                            f"signals={bm.get('signals')}\n"
                        )
                f.write("\n")

            f.write("SEED PLANS\n")
            f.write("-" * 40 + "\n")
            for sp in batch_entry.get("seed_plans", []):
                f.write(
                    f"  seed={sp.get('seed_action_template')} "
                    f"instance={sp.get('seed_action_instance_id')} "
                    f"plan_length={sp.get('plan_length')} "
                    f"failure={sp.get('failure')}\n"
                )
                if sp.get("seed_scores"):
                    f.write(f"    scores: {sp.get('seed_scores')}\n")
            f.write("\n")

            f.write("CONDITION COMPONENT TRACE\n")
            f.write("-" * 40 + "\n")
            for step in batch_entry.get("condition_trace", []):
                f.write(f"  depth={step.get('depth')} subgoal={step.get('subgoal')}\n")
                if step.get("failure"):
                    f.write(f"    failure: {step.get('failure')}\n")
                for c in step.get("candidates", []):
                    f.write(
                        f"    candidate action={c.get('action_template')} "
                        f"score={_fmt_float(c.get('match_score', 0.0))} "
                        f"effect_key={c.get('effect_key')}\n"
                    )
                    det = c.get("details", {})
                    f.write(
                        f"      object_mult={_fmt_float(det.get('object_mult', 0.0))} "
                        f"tier={det.get('object_tier')} "
                        f"signals={det.get('object_signals')} "
                        f"core={_fmt_float(det.get('core_score', 0.0))}\n"
                    )
                    f.write(
                        f"      path={_fmt_float(det.get('path_mult', 0.0))} "
                        f"op={_fmt_float(det.get('op_mult', 0.0))} "
                        f"value={_fmt_float(det.get('value_mult', 0.0))} "
                        f"fallback={det.get('value_fallback_used')}\n"
                    )
                if step.get("selected_action"):
                    f.write(f"    SELECTED: {step.get('selected_action')}\n")
                f.write("\n")

            f.write("\n" + "-" * 80 + "\n\n")

        # ------------------------------------------------------------
        # Plan-level traces (grouped)
        # ------------------------------------------------------------
        f.write("\nPLAN-LEVEL TRACES (grouped)\n")
        f.write("=" * 80 + "\n\n")

        groups = [
            ("Verb-guided plans — covering all target actions",
             [p for p in plans if p.get("coverage_group") == "verb_guided_all"]),
            ("Verb-guided plans — covering some target actions",
             [p for p in plans if p.get("coverage_group") == "verb_guided_partial"]),
            ("General backward-chaining plans",
             [p for p in plans if p.get("coverage_group") == "general"]),
        ]

        for group_title, group_plans in groups:
            f.write("#" * 80 + "\n")
            f.write(f"# {group_title}  ({len(group_plans)} plans)\n")
            f.write("#" * 80 + "\n\n")

            if not group_plans:
                f.write("  (none)\n\n")
                continue

            for i, plan in enumerate(group_plans[:10], 1):
                _write_per_plan_full_block(f, plan, i)

        # ------------------------------------------------------------
        # Multi-action pass trace
        # ------------------------------------------------------------
        multi_action_trace = getattr(planner, "multi_action_trace", None) or []
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("MULTI-ACTION PASS TRACE\n")
        f.write("=" * 80 + "\n\n")
        if not multi_action_trace:
            f.write("(no multi-action pass was run)\n\n")
        else:
            for pass_idx, entry in enumerate(multi_action_trace, 1):
                f.write(f"--- PASS {pass_idx} ---\n")
                f.write(f"  type:   {entry.get('type')}\n")
                f.write(f"  status: {entry.get('status')}\n")
                if entry.get("reason"):
                    f.write(f"  reason: {entry.get('reason')}\n")
                if entry.get("failed_action_indices"):
                    f.write(f"  failed_action_indices: {entry.get('failed_action_indices')}\n")
                f.write(f"  action_count: {entry.get('action_count')}\n")
                f.write(f"  per_action_plan_cap: {entry.get('per_action_plan_cap')}\n")
                f.write(f"  max_combinations: {entry.get('max_combinations')}\n")
                f.write(f"  combinations_generated: {entry.get('combinations_generated')}\n")
                f.write(f"  combinations_truncated_by_cap: {entry.get('combinations_truncated_by_cap')}\n\n")

                f.write("  TARGET ACTIONS\n")
                for ta in entry.get("target_actions", []):
                    f.write(f"    - name={ta.get('name')} verb={ta.get('verb')} "
                            f"participants={ta.get('participants')}\n")
                f.write("\n")

                f.write("  PER-ACTION DETAIL\n")
                for pa in entry.get("per_action", []):
                    f.write(
                        f"    Action {pa.get('action_index')} "
                        f"({pa.get('target_action_name')}, verb={pa.get('verb_used')}): "
                        f"candidates={pa.get('candidates_retrieved')}, "
                        f"seeds_considered={pa.get('seeds_considered')}, "
                        f"plans_produced={pa.get('plans_produced')}, "
                        f"plan_lengths={pa.get('plan_lengths')}\n"
                    )
                    f.write("      SEED RECORDS\n")
                    for sr in pa.get("seed_records", []):
                        f.write(
                            f"        seed={sr.get('seed_template')} "
                            f"instance={sr.get('seed_instance_id')} "
                            f"verb={_fmt_float(sr.get('verb_score', 0.0))} "
                            f"object={_fmt_float(sr.get('object_score', 0.0))} "
                            f"category={_fmt_float(sr.get('category_score', 0.0))} "
                            f"total={_fmt_float(sr.get('total_score', 0.0))} "
                            f"plan_produced={sr.get('plan_produced')} "
                            f"plan_length={sr.get('plan_length')} "
                            f"failure={sr.get('failure')}\n"
                        )
                    f.write("\n")

                f.write("  COMBINATIONS\n")
                for combo in entry.get("combinations", []):
                    f.write(
                        f"    Combo {combo.get('combo_index')}: "
                        f"total_actions={combo.get('total_actions')} "
                        f"avg_scores={combo.get('avg_scores')}\n"
                    )
                    for pa in combo.get("per_action", []):
                        f.write(
                            f"      action_index={pa.get('action_index')} "
                            f"({pa.get('target_action_name')}) "
                            f"seed={pa.get('seed_template')} "
                            f"plan_length={pa.get('plan_length')}\n"
                        )
                f.write("\n" + "-" * 80 + "\n\n")

        # ------------------------------------------------------------
        # Representative long plans (from the pre-threshold pool)
        # ------------------------------------------------------------
        pool = getattr(planner, "all_generated_plans", []) or []
        picks = _select_long_plans(pool, thresholds=(10, 20))

        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("REPRESENTATIVE LONG PLANS — FULL DETAIL\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total plans generated (pre-threshold): {len(pool)}\n")
        f.write(f"Total plans returned (ranked):         {len(plans)}\n\n")

        from collections import Counter

        # Plan length census over the full pre-threshold pool
        lengths = Counter(len(p.get("actions", [])) for p in pool)
        f.write("Plan length census (all generated plans):\n")
        if lengths:
            for length in sorted(lengths):
                f.write(f"  {length:4d} actions: {lengths[length]:4d} plans\n")
        else:
            f.write("  (no plans were generated)\n")
        f.write("\n")

        # Plan type census over the full pre-threshold pool
        types = Counter(p.get("type", "?") for p in pool)
        f.write("Plan type census (all generated plans):\n")
        if types:
            for t in sorted(types):
                f.write(f"  {t}: {types[t]} plans\n")
        else:
            f.write("  (no plans were generated)\n")
        f.write("\n")

        # Plan stage census over the full pre-threshold pool
        stages = Counter(p.get("stage", "?") for p in pool)
        f.write("Plan stage census (all generated plans):\n")
        if stages:
            for s in sorted(stages):
                f.write(f"  {s}: {stages[s]} plans\n")
        else:
            f.write("  (no plans were generated)\n")
        f.write("\n")

        # Plan coverage-group census over the full pre-threshold pool
        cov = Counter(p.get("coverage_group", "?") for p in pool)
        f.write("Plan coverage-group census (all generated plans):\n")
        if cov:
            for c in sorted(cov):
                f.write(f"  {c}: {cov[c]} plans\n")
        else:
            f.write("  (no plans were generated)\n")
        f.write("\n")

        for idx, (threshold, plan) in enumerate(picks, 1):
            f.write("-" * 80 + "\n")
            if plan is None:
                f.write(f"[{idx}] No plan found with more than {threshold} actions.\n")
                f.write("-" * 80 + "\n\n")
                continue

            n = len(plan.get("actions", []))
            f.write(
                f"[{idx}] Representative plan: shortest with > {threshold} actions "
                f"(this plan has {n})\n"
            )
            f.write(
                f"     stage: {plan.get('stage', '?')}  "
                f"coverage_group: {plan.get('coverage_group', '?')}  "
                f"final_score: {plan.get('final_score', 0.0):.4f}  "
                f"type: {plan.get('type', '?')}\n"
            )
            if plan.get("actions_covered") is not None:
                f.write(f"     actions_covered: {plan['actions_covered']}\n")
            if plan.get("goals_covered") is not None:
                f.write(f"     goals_covered: {plan['goals_covered']}\n")
            f.write("-" * 80 + "\n\n")

            for line in _format_plan_full(plan, index=idx):
                f.write(line + "\n")
            f.write("\n")


# =============================================================================
# Planning handler
# =============================================================================
async def handle_planning(qp: PickleQueryProcessor, index: PickleSearchIndex,
                          engine: PickleSearchEngine, search_results: List[Dict[str, Any]]):
    planning_query = qp.get_or_create_planning_data()

    if not planning_query.get("goal_conditions"):
        print("\nNo goal conditions found in planning query. Cannot generate plan.")
        return

    print("\nPlanning query:")
    print(json.dumps(planning_query, indent=2, ensure_ascii=False))

    pidx = PlanningIndex(index)
    initial_ws = build_initial_world_state(planning_query)
    planner = Planner(pidx, initial_world_state=initial_ws, trace_enabled=True)

    print("\nGenerating plans...")
    start = time.perf_counter()
    plans = planner.plan(planning_query, search_results)
    elapsed = time.perf_counter() - start

    print(f"Planning completed in {elapsed:.2f}s")

    if not plans:
        print("No plans generated.")
        return

    print("\nPlans by group (top 10 each):")

    _print_plan_group(
        "Verb-guided plans — covering all target actions:",
        [p for p in plans if p.get("coverage_group") == "verb_guided_all"],
    )
    _print_plan_group(
        "Verb-guided plans — covering some target actions:",
        [p for p in plans if p.get("coverage_group") == "verb_guided_partial"],
    )
    _print_plan_group(
        "General backward-chaining plans:",
        [p for p in plans if p.get("coverage_group") == "general"],
    )

    save_plan_traces_simple(plans)
    save_plan_traces_full(planner, plans)
    print(f"\nSimple trace saved to: {PLAN_TRACE_SIMPLE}")
    print(f"Full trace saved to:   {PLAN_TRACE_FULL}")


# =============================================================================
# Main loop
# =============================================================================
async def main():
    print("Loading pickle memory system...")
    loader = PickleMemoryLoader()
    index = PickleSearchIndex(loader.memories, loader.object_storage, loader.alias_expansion)
    engine = PickleSearchEngine(index)
    qp = PickleQueryProcessor(loader)
    qp.set_mode("llm")

    mode_commands = {"manual", "llm", "mode 1", "mode 2"}

    print("\nPickle Search CLI")
    print("Default mode is 'llm'. Type 'manual' or 'llm' to switch modes.")
    print("Type 'github' for the repository URL.")
    print("Type 'quit' to exit.\n")

    while True:
        prompt = f"[{qp.mode}]> "
        user_input = input(prompt).strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break

        if user_input.lower() == "github":
            print("https://github.com/jc-20261")
            continue

        if user_input.lower() in mode_commands:
            if user_input.lower() in {"manual", "mode 1"}:
                qp.set_mode("manual")
                print("Switched to manual query terms mode.")
            elif user_input.lower() in {"llm", "mode 2"}:
                qp.set_mode("llm")
                print("Switched to LLM query mode.")
            continue

        terms = await qp.generate_terms(user_input)

        print("Query terms:")
        print(terms)

        query_alias_map = qp.get_last_query_aliases()
        if query_alias_map:
            print(f"Query-side aliases: {len(query_alias_map)} entries")
        else:
            print("Query-side aliases: none (manual mode or alias generation returned empty)")

        search_start = time.perf_counter()
        results = engine.search(
            terms,
            progress=print_progress,
            query_alias_map=query_alias_map,
        )
        search_elapsed = time.perf_counter() - search_start

        print(f"\nSearch completed in {search_elapsed:.2f}s")

        if not results:
            print("No matches.")
            continue

        print("\nTop results:")
        for i, res in enumerate(results[:10], 1):
            print(f"{i}. {res['memory'].activity} (score: {res['score']:.4f})")

        save_search_results(user_input, qp.mode, terms, results)
        print("Search results saved.")

        while True:
            choice = input(
                "\nSelect a result number, type 'plan', or press Enter to continue: "
            ).strip()

            if not choice:
                break

            if choice.lower() == "plan":
                await handle_planning(qp, index, engine, results)
                continue

            try:
                idx = int(choice) - 1
                res = results[idx]
            except (ValueError, IndexError):
                print("Invalid selection.")
                continue

            print_matched_terms(res['memory'].id, res['memory'].activity, res['term_details'], 20)

            while True:
                detail_choice = input(
                    "\nEnter 'all' for full list, a number for top N, or press Enter to go back: "
                ).strip()

                if not detail_choice:
                    break

                if detail_choice.lower() == "all":
                    print_matched_terms(res['memory'].id, res['memory'].activity, res['term_details'], None)
                    continue

                try:
                    n = int(detail_choice)
                    if n <= 0:
                        print("Please enter a positive number.")
                        continue
                    print_matched_terms(res['memory'].id, res['memory'].activity, res['term_details'], n)
                except ValueError:
                    print("Invalid input.")
                    continue

            while True:
                context_choice = input(
                    "\nSelect a matched term number to inspect context, or press Enter to return to results: "
                ).strip()

                if not context_choice:
                    break

                try:
                    tidx = int(context_choice) - 1
                    sorted_details = sorted(
                        res['term_details'],
                        key=lambda td: td['score_contribution'],
                        reverse=True
                    )
                    td = sorted_details[tidx]
                except (ValueError, IndexError):
                    print("Invalid selection.")
                    continue

                print(f"\nQuery term: {td['query_term']}")
                print(f"Matched index term: {td['matched_index_term']}")
                print(f"Mode: {td['mode']}")
                print(f"Score contribution: {td['score_contribution']:.4f}")
                print(f"Source context:\n{td['source_context']}")


if __name__ == "__main__":
    asyncio.run(main())