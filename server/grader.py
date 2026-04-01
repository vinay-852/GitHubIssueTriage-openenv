# envs/your_env/server/grader.py
from __future__ import annotations

from typing import List, Set

from ..model import GraderResult, HiddenGradingTarget, IssueStatus, IssueTriageState


def _labels_set(state: IssueTriageState) -> Set[str]:
    return set(state.issue.labels)


def _comment_text(state: IssueTriageState) -> str:
    return " ".join(comment.body.lower() for comment in state.issue.comments)


def _requested_fields_set(state: IssueTriageState) -> Set[str]:
    return set(state.requested_fields)


def _close_reason(state: IssueTriageState) -> str:
    return str(state.issue.metadata.get("close_reason", "")).lower()


def _matched_comment_keywords(state: IssueTriageState, keywords: List[str]) -> bool:
    if not keywords:
        return True
    text = _comment_text(state)
    return all(keyword.lower() in text for keyword in keywords)


def grade_episode(state: IssueTriageState) -> GraderResult:
    """
    Deterministic grader for a completed or in-progress episode.

    Score range: 0.0 to 1.0
    """
    target = state.hidden_target
    if target is None:
        # Fallback grading when no hidden target exists.
        score = 0.0
        if state.issue.labels:
            score += 0.2
        if state.issue.assignees:
            score += 0.2
        if state.issue.priority is not None:
            score += 0.2
        if state.issue.milestone is not None:
            score += 0.2
        if state.issue.comments:
            score += 0.2
        return GraderResult(
            score=max(0.0, min(1.0, score)),
            matched_labels=list(state.issue.labels),
            matched_assignee=bool(state.issue.assignees),
            matched_priority=state.issue.priority is not None,
            matched_milestone=state.issue.milestone is not None,
            duplicate_matched=bool(state.issue.linked_duplicates),
            missing_fields_requested=bool(state.requested_fields),
            closed_correctly=state.issue.status == IssueStatus.CLOSED,
            comment_accepted=bool(state.issue.comments),
            notes=["No hidden_target present; using fallback grading."],
        )

    labels = _labels_set(state)

    # Label match
    matched_labels = [label for label in target.gold_labels if label in labels]
    labels_ok = len(target.gold_labels) == 0 or len(matched_labels) == len(target.gold_labels)

    # Assignee
    if target.gold_assignee is None:
        assignee_ok = len(state.issue.assignees) == 0
    else:
        assignee_ok = state.issue.assignees == [target.gold_assignee]

    # Priority
    if target.gold_priority is None:
        priority_ok = state.issue.priority is None
    else:
        priority_ok = state.issue.priority == target.gold_priority

    # Milestone
    if target.gold_milestone is None:
        milestone_ok = state.issue.milestone is None
    else:
        milestone_ok = state.issue.milestone == target.gold_milestone

    # Severity / component
    if target.gold_severity is None:
        severity_ok = state.issue.severity is None
    else:
        severity_ok = state.issue.severity == target.gold_severity

    if target.gold_component is None:
        component_ok = state.issue.component is None
    else:
        component_ok = state.issue.component == target.gold_component

    # Duplicate handling
    if target.gold_duplicate_issue_id is None:
        duplicate_ok = len(state.issue.linked_duplicates) == 0
    else:
        duplicate_ok = target.gold_duplicate_issue_id in state.issue.linked_duplicates

    # Missing info
    if target.required_missing_fields:
        requested = _requested_fields_set(state)
        missing_info_ok = all(field in requested for field in target.required_missing_fields)
        missing_info_partial = len(requested.intersection(set(target.required_missing_fields))) / len(
            target.required_missing_fields
        )
    else:
        missing_info_ok = True
        missing_info_partial = 1.0

    # Close decision
    if target.gold_close_reason is None:
        closed_ok = state.issue.status == IssueStatus.OPEN
    else:
        closed_ok = state.issue.status == IssueStatus.CLOSED
        if closed_ok:
            close_reason_ok = _close_reason(state) == target.gold_close_reason.value
        else:
            close_reason_ok = False

    # Comment quality
    comment_ok = _matched_comment_keywords(state, target.expected_comment_keywords)

    # Partial scoring
    score = 0.0
    score += 0.18 if labels_ok else 0.18 * (len(matched_labels) / max(1, len(target.gold_labels)))
    score += 0.18 if assignee_ok else 0.0
    score += 0.12 if priority_ok else 0.0
    score += 0.10 if milestone_ok else 0.0
    score += 0.10 if severity_ok else 0.0
    score += 0.10 if component_ok else 0.0
    score += 0.12 if duplicate_ok else 0.0
    score += 0.06 if missing_info_ok else 0.06 * missing_info_partial
    score += 0.04 if closed_ok else 0.0
    score += 0.02 if comment_ok else 0.0

    # Slight bonus if close reason matches exactly when needed
    if target.gold_close_reason is not None and state.issue.status == IssueStatus.CLOSED:
        if _close_reason(state) == target.gold_close_reason.value:
            score += 0.02

    score = max(0.0, min(1.0, score))

    notes: List[str] = []
    if not labels_ok:
        notes.append("Label set incomplete or incorrect.")
    if not assignee_ok:
        notes.append("Assignee does not match target.")
    if not priority_ok:
        notes.append("Priority does not match target.")
    if not milestone_ok:
        notes.append("Milestone does not match target.")
    if not severity_ok:
        notes.append("Severity does not match target.")
    if not component_ok:
        notes.append("Component does not match target.")
    if not duplicate_ok:
        notes.append("Duplicate target not linked correctly.")
    if not missing_info_ok:
        notes.append("Required missing fields were not fully requested.")
    if not closed_ok:
        notes.append("Closure state does not match target.")
    if not comment_ok:
        notes.append("Comment keywords did not match target.")

    return GraderResult(
        score=score,
        matched_labels=matched_labels,
        matched_assignee=assignee_ok,
        matched_priority=priority_ok,
        matched_milestone=milestone_ok,
        duplicate_matched=duplicate_ok,
        missing_fields_requested=missing_info_ok,
        closed_correctly=closed_ok,
        comment_accepted=comment_ok,
        notes=notes,
    )


def is_success(state: IssueTriageState) -> bool:
    """
    Strict success check for completed episodes.
    """
    result = grade_episode(state)
    return result.score >= 0.95