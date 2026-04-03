# envs/your_env/server/termination.py
from __future__ import annotations

from ..models import IssueStatus, IssueTriageState


def _all_required_labels_present(state: IssueTriageState) -> bool:
    target = state.hidden_target
    if target is None or not target.gold_labels:
        return True

    labels = set(state.issue.labels)
    return all(label in labels for label in target.gold_labels)


def _assignee_ok(state: IssueTriageState) -> bool:
    target = state.hidden_target
    if target is None or target.gold_assignee is None:
        return True
    return state.issue.assignees == [target.gold_assignee]


def _priority_ok(state: IssueTriageState) -> bool:
    target = state.hidden_target
    if target is None or target.gold_priority is None:
        return True
    return state.issue.priority == target.gold_priority


def _milestone_ok(state: IssueTriageState) -> bool:
    target = state.hidden_target
    if target is None or target.gold_milestone is None:
        return True
    return state.issue.milestone == target.gold_milestone


def _severity_ok(state: IssueTriageState) -> bool:
    target = state.hidden_target
    if target is None or target.gold_severity is None:
        return True
    return state.issue.severity == target.gold_severity


def _component_ok(state: IssueTriageState) -> bool:
    target = state.hidden_target
    if target is None or target.gold_component is None:
        return True
    return state.issue.component == target.gold_component


def _duplicate_ok(state: IssueTriageState) -> bool:
    target = state.hidden_target
    if target is None or target.gold_duplicate_issue_id is None:
        return True
    return target.gold_duplicate_issue_id in state.issue.linked_duplicates


def _missing_info_ok(state: IssueTriageState) -> bool:
    target = state.hidden_target
    if target is None or not target.required_missing_fields:
        return True
    requested = set(state.requested_fields)
    return all(field in requested for field in target.required_missing_fields)


def _closure_ok(state: IssueTriageState) -> bool:
    target = state.hidden_target
    if target is None:
        return True
    if target.gold_close_reason is None:
        return state.issue.status == IssueStatus.OPEN
    return state.issue.status == IssueStatus.CLOSED


def _task_goal_satisfied(state: IssueTriageState) -> bool:
    """
    Final success condition for the episode.

    If no hidden target exists, this should not auto-solve the episode.
    That keeps live URL/local runs active until max_steps or explicit done.
    """
    if state.hidden_target is None:
        return False

    return all(
        [
            _all_required_labels_present(state),
            _assignee_ok(state),
            _priority_ok(state),
            _milestone_ok(state),
            _severity_ok(state),
            _component_ok(state),
            _duplicate_ok(state),
            _missing_info_ok(state),
            _closure_ok(state),
        ]
    )


def is_episode_done(state: IssueTriageState) -> bool:
    """
    Decide whether the episode should terminate.

    Rules:
    - terminate when max steps are reached
    - terminate when the hidden target is fully satisfied
    - preserve any explicit done flag already set on state
    """
    if state.done:
        return True

    if state.step_count >= state.max_steps:
        return True

    if _task_goal_satisfied(state):
        return True

    return False


def is_success(state: IssueTriageState) -> bool:
    """
    True only when the task goal has been satisfied.
    """
    return _task_goal_satisfied(state)


def remaining_steps(state: IssueTriageState) -> int:
    """
    Convenience helper used by observation construction.
    """
    return max(0, state.max_steps - state.step_count)