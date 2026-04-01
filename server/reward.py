# envs/your_env/server/reward.py
from __future__ import annotations

from typing import List, Optional, Set

from ..model import (
    CloseReason,
    HiddenGradingTarget,
    IssueStatus,
    IssueTriageState,
    Priority,
    Reward,
    Severity,
)


def _labels_set(state: IssueTriageState) -> Set[str]:
    return set(state.issue.labels)


def _has_all(required: List[str], present: Set[str]) -> bool:
    return all(item in present for item in required)


def _intersection_score(required: List[str], present: Set[str]) -> float:
    if not required:
        return 1.0
    return len(set(required).intersection(present)) / float(len(required))


def _comment_keyword_score(state: IssueTriageState, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    text = " ".join(comment.body.lower() for comment in state.issue.comments)
    return 1.0 if all(keyword.lower() in text for keyword in keywords) else 0.0


def _closed_reason_bonus(target: HiddenGradingTarget, state: IssueTriageState) -> float:
    if target.gold_close_reason is None:
        return 1.0 if state.issue.status == IssueStatus.OPEN else 0.0
    if state.issue.status != IssueStatus.CLOSED:
        return 0.0
    # We do not store the close reason directly in state yet.
    # This keeps the environment simple and deterministic.
    # If you later add close_reason to IssueSnapshot, check it here.
    return 1.0


def _basic_progress_score(state: IssueTriageState) -> Reward:
    issue = state.issue
    score = 0.0
    if issue.labels:
        score += 0.2
    if issue.assignees:
        score += 0.2
    if issue.priority is not None:
        score += 0.2
    if issue.milestone is not None:
        score += 0.2
    if issue.comments:
        score += 0.2

    score = max(0.0, min(1.0, score))
    return Reward(
        total=score,
        type_score=0.0,
        severity_score=0.0,
        component_score=0.0,
        assignee_score=0.0,
        priority_score=0.0,
        milestone_score=0.0,
        missing_info_score=0.0,
        duplicate_score=0.0,
        closure_score=0.0,
        comment_score=0.0,
        invalid_action_penalty=0.0,
        destructive_action_penalty=0.0,
    )


def compute_reward(state: IssueTriageState) -> Reward:
    """
    Compute a dense reward for the current state.

    If hidden_target is present, reward is shaped against the target.
    Otherwise, return a basic progress score.
    """
    target = state.hidden_target
    if target is None:
        return _basic_progress_score(state)

    issue = state.issue
    labels = _labels_set(state)

    # Labels
    type_score = _intersection_score([x for x in target.gold_labels if x.startswith("type:")], labels)
    severity_score = 1.0 if target.gold_severity is None or issue.severity == target.gold_severity else 0.0
    component_score = 1.0 if target.gold_component is None or issue.component == target.gold_component else 0.0

    # Assignment / routing
    if target.gold_assignee is None:
        assignee_score = 1.0 if not issue.assignees else 0.0
    else:
        assignee_score = 1.0 if issue.assignees == [target.gold_assignee] else 0.0

    # Priority / milestone
    priority_score = 1.0 if target.gold_priority is None or issue.priority == target.gold_priority else 0.0
    milestone_score = 1.0 if target.gold_milestone is None or issue.milestone == target.gold_milestone else 0.0

    # Missing info
    missing_info_score = _intersection_score(target.required_missing_fields, state.requested_fields)

    # Duplicate
    if target.gold_duplicate_issue_id is None:
        duplicate_score = 1.0 if not issue.linked_duplicates else 0.0
    else:
        duplicate_score = 1.0 if target.gold_duplicate_issue_id in issue.linked_duplicates else 0.0

    # Closure
    closure_score = _closed_reason_bonus(target, state)

    # Comment quality
    comment_score = _comment_keyword_score(state, target.expected_comment_keywords)

    # Dense weighted score
    total = (
        0.15 * type_score
        + 0.15 * severity_score
        + 0.15 * component_score
        + 0.15 * assignee_score
        + 0.10 * priority_score
        + 0.10 * milestone_score
        + 0.10 * missing_info_score
        + 0.05 * duplicate_score
        + 0.03 * closure_score
        + 0.02 * comment_score
    )

    # Small penalties for obviously bad states
    invalid_action_penalty = 0.0
    destructive_action_penalty = 0.0

    if state.last_action_valid is False:
        invalid_action_penalty = -0.10
        total += invalid_action_penalty

    # Penalize closing a non-duplicate when the hidden target says it should stay open.
    if target.gold_close_reason is None and issue.status == IssueStatus.CLOSED:
        destructive_action_penalty = -0.15
        total += destructive_action_penalty

    total = max(-1.0, min(1.0, total))

    return Reward(
        total=total,
        type_score=type_score,
        severity_score=severity_score,
        component_score=component_score,
        assignee_score=assignee_score,
        priority_score=priority_score,
        milestone_score=milestone_score,
        missing_info_score=missing_info_score,
        duplicate_score=duplicate_score,
        closure_score=closure_score,
        comment_score=comment_score,
        invalid_action_penalty=invalid_action_penalty,
        destructive_action_penalty=destructive_action_penalty,
    )