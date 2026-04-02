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


def _grade_labels(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, float, List[str], List[str]]:
    if not target.gold_labels:
        return True, 1.0, [], []

    labels = _labels_set(state)
    matched = [label for label in target.gold_labels if label in labels]
    partial = len(matched) / float(len(target.gold_labels))
    ok = partial == 1.0
    notes = [] if ok else ["Label set incomplete or incorrect."]
    return ok, partial, matched, notes


def _grade_assignee(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, List[str]]:
    if target.gold_assignee is None:
        ok = len(state.issue.assignees) == 0
    else:
        ok = state.issue.assignees == [target.gold_assignee]
    notes = [] if ok else ["Assignee does not match target."]
    return ok, notes


def _grade_priority(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, List[str]]:
    if target.gold_priority is None:
        ok = state.issue.priority is None
    else:
        ok = state.issue.priority == target.gold_priority
    notes = [] if ok else ["Priority does not match target."]
    return ok, notes


def _grade_milestone(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, List[str]]:
    if target.gold_milestone is None:
        ok = state.issue.milestone is None
    else:
        ok = state.issue.milestone == target.gold_milestone
    notes = [] if ok else ["Milestone does not match target."]
    return ok, notes


def _grade_severity(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, List[str]]:
    if target.gold_severity is None:
        ok = state.issue.severity is None
    else:
        ok = state.issue.severity == target.gold_severity
    notes = [] if ok else ["Severity does not match target."]
    return ok, notes


def _grade_component(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, List[str]]:
    if target.gold_component is None:
        ok = state.issue.component is None
    else:
        ok = state.issue.component == target.gold_component
    notes = [] if ok else ["Component does not match target."]
    return ok, notes


def _grade_duplicate(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, List[str]]:
    if target.gold_duplicate_issue_id is None:
        ok = len(state.issue.linked_duplicates) == 0
    else:
        ok = target.gold_duplicate_issue_id in state.issue.linked_duplicates
    notes = [] if ok else ["Duplicate target not linked correctly."]
    return ok, notes


def _grade_missing_info(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, float, List[str]]:
    if not target.required_missing_fields:
        return True, 1.0, []

    requested = _requested_fields_set(state)
    required = set(target.required_missing_fields)
    matched = requested.intersection(required)
    partial = len(matched) / float(len(required)) if required else 1.0
    ok = partial == 1.0
    notes = [] if ok else ["Required missing fields were not fully requested."]
    return ok, partial, notes


def _grade_closure(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, float, List[str]]:
    if target.gold_close_reason is None:
        ok = state.issue.status == IssueStatus.OPEN
        notes = [] if ok else ["Closure state does not match target."]
        return ok, 0.0, notes

    ok = state.issue.status == IssueStatus.CLOSED
    notes: List[str] = []
    bonus = 0.0
    if ok:
        if _close_reason(state) == target.gold_close_reason.value:
            bonus = 0.02
        else:
            notes.append("Close reason does not match target.")
    else:
        notes.append("Closure state does not match target.")
    return ok, bonus, notes


def _grade_comment(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, List[str]]:
    ok = _matched_comment_keywords(state, target.expected_comment_keywords)
    notes = [] if ok else ["Comment keywords did not match target."]
    return ok, notes


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

    labels_ok, labels_partial, matched_labels, label_notes = _grade_labels(state, target)
    assignee_ok, assignee_notes = _grade_assignee(state, target)
    priority_ok, priority_notes = _grade_priority(state, target)
    milestone_ok, milestone_notes = _grade_milestone(state, target)
    severity_ok, severity_notes = _grade_severity(state, target)
    component_ok, component_notes = _grade_component(state, target)
    duplicate_ok, duplicate_notes = _grade_duplicate(state, target)
    missing_info_ok, missing_info_partial, missing_notes = _grade_missing_info(state, target)
    closure_ok, closure_bonus, closure_notes = _grade_closure(state, target)
    comment_ok, comment_notes = _grade_comment(state, target)

    score = 0.0
    score += 0.18 if labels_ok else 0.18 * labels_partial
    score += 0.18 if assignee_ok else 0.0
    score += 0.12 if priority_ok else 0.0
    score += 0.10 if milestone_ok else 0.0
    score += 0.10 if severity_ok else 0.0
    score += 0.10 if component_ok else 0.0
    score += 0.12 if duplicate_ok else 0.0
    score += 0.06 if missing_info_ok else 0.06 * missing_info_partial
    score += 0.04 if closure_ok else 0.0
    score += 0.02 if comment_ok else 0.0
    score += closure_bonus

    score = max(0.0, min(1.0, score))

    notes: List[str] = []
    for bucket in (
        label_notes,
        assignee_notes,
        priority_notes,
        milestone_notes,
        severity_notes,
        component_notes,
        duplicate_notes,
        missing_notes,
        closure_notes,
        comment_notes,
    ):
        for message in bucket:
            if message and message not in notes:
                notes.append(message)

    return GraderResult(
        score=score,
        matched_labels=matched_labels,
        matched_assignee=assignee_ok,
        matched_priority=priority_ok,
        matched_milestone=milestone_ok,
        duplicate_matched=duplicate_ok,
        missing_fields_requested=missing_info_ok,
        closed_correctly=closure_ok,
        comment_accepted=comment_ok,
        notes=notes,
    )


def is_success(state: IssueTriageState) -> bool:
    """
    Strict success check for completed episodes.
    """
    result = grade_episode(state)
    return result.score >= 0.95
