# envs/your_env/server/observation.py
from __future__ import annotations

from typing import List

from ..model import IssueTriageState, Observation


def _available_labels(state: IssueTriageState) -> List[str]:
    labels: List[str] = []

    labels.extend(state.repo_rules.labels.get("type", []))
    labels.extend(state.repo_rules.labels.get("severity", []))
    labels.extend(state.repo_rules.labels.get("component", []))
    labels.extend(state.repo_rules.labels.get("status", []))
    labels.extend(state.repo_rules.labels.get("priority", []))

    for definition in state.repo_rules.label_definitions:
        labels.append(definition.name)

    # Deduplicate while preserving order.
    seen = set()
    deduped: List[str] = []
    for label in labels:
        if label and label not in seen:
            seen.add(label)
            deduped.append(label)
    return deduped


def _available_assignees(state: IssueTriageState) -> List[str]:
    assignees: List[str] = []

    assignees.extend(state.repo_rules.assignee_pool)

    for team_members in state.repo_rules.routing_rules.values():
        assignees.extend(team_members)

    for members in state.repo_rules.team_map.values():
        assignees.extend(members)

    for rule in state.repo_rules.team_routing_rules:
        assignees.extend(rule.assignee_candidates)

    seen = set()
    deduped: List[str] = []
    for name in assignees:
        if name and name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def _available_milestones(state: IssueTriageState) -> List[str]:
    milestones = list(state.repo_rules.milestones)
    seen = set()
    deduped: List[str] = []
    for milestone in milestones:
        if milestone and milestone not in seen:
            seen.add(milestone)
            deduped.append(milestone)
    return deduped


def build_observation(state: IssueTriageState) -> Observation:
    """
    Convert internal episode state into the agent-facing observation.

    Keep hidden grading data out of this view.
    """
    remaining_steps = max(0, state.max_steps - state.step_count)

    return Observation(
        episode_id=state.episode_id,
        task=state.task,
        issue=state.issue.model_copy(deep=True),
        repo_rules=state.repo_rules.model_copy(deep=True),
        available_labels=_available_labels(state),
        available_assignees=_available_assignees(state),
        available_milestones=_available_milestones(state),
        candidate_duplicates=[cand.model_copy(deep=True) for cand in state.candidate_duplicates],
        action_history=[entry.model_copy(deep=True) for entry in state.current_action_history],
        pending_missing_fields=list(state.pending_missing_fields),
        remaining_steps=remaining_steps,
        step_count=state.step_count,
        done=state.done,
        last_action_valid=state.last_action_valid,
        last_action_message=state.last_action_message,
    )