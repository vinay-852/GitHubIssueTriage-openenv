# envs/your_env/server/transitions.py
from __future__ import annotations

from typing import List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field

from ..model import (
    Action,
    ActionType,
    AddLabelAction,
    AssignUserAction,
    CloseIssueAction,
    CloseReason,
    CommentAction,
    HiddenGradingTarget,
    IssueComment,
    IssueSnapshot,
    IssueStatus,
    IssueTriageState,
    MarkDuplicateAction,
    NoopAction,
    Priority,
    ReadAssigneePoolAction,
    ReadIssueAction,
    ReadLabelDefinitionsAction,
    ReadMilestonesAction,
    ReadRepoRulesAction,
    ReadTeamRoutingAction,
    RemoveLabelAction,
    RequestInfoAction,
    ReopenIssueAction,
    SearchSimilarIssuesAction,
    SetMilestoneAction,
    SetPriorityAction,
    Severity,
    TimelineEvent,
)


class TransitionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_valid: bool = True
    action_effect: str = ""
    changed_fields: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _available_labels(state: IssueTriageState) -> List[str]:
    labels: List[str] = []
    labels.extend(state.repo_rules.labels.get("type", []))
    labels.extend(state.repo_rules.labels.get("severity", []))
    labels.extend(state.repo_rules.labels.get("component", []))
    labels.extend(state.repo_rules.labels.get("status", []))
    labels.extend(state.repo_rules.labels.get("priority", []))

    for definition in state.repo_rules.label_definitions:
        labels.append(definition.name)

    for alias, canonical in state.repo_rules.label_aliases.items():
        labels.append(alias)
        labels.append(canonical)

    return _dedupe_keep_order(labels)


def _available_assignees(state: IssueTriageState) -> List[str]:
    assignees: List[str] = []
    assignees.extend(state.repo_rules.assignee_pool)

    for members in state.repo_rules.routing_rules.values():
        assignees.extend(members)

    for members in state.repo_rules.team_map.values():
        assignees.extend(members)

    for rule in state.repo_rules.team_routing_rules:
        assignees.extend(rule.assignee_candidates)

    return _dedupe_keep_order(assignees)


def _available_milestones(state: IssueTriageState) -> List[str]:
    return _dedupe_keep_order(list(state.repo_rules.milestones))


def _label_conflicts(state: IssueTriageState, new_label: str) -> List[str]:
    conflicts: List[str] = []
    for definition in state.repo_rules.label_definitions:
        if definition.name == new_label:
            conflicts.extend(definition.mutually_exclusive_with)
            break
    return conflicts


def _ensure_comment(
    state: IssueTriageState,
    *,
    body: str,
    internal: bool = False,
    actor: str = "triage-agent",
) -> None:
    state.issue.comments.append(
        IssueComment(
            comment_id=f"c_{len(state.issue.comments) + 1}",
            author=actor,
            body=body,
            created_at="",
            internal=internal,
        )
    )
    state.issue.timeline.append(
        TimelineEvent(
            event_id=f"t_{len(state.issue.timeline) + 1}",
            event_type="commented",
            actor=actor,
            created_at="",
            payload={"internal": str(internal).lower()},
        )
    )


def _mark_close_reason(state: IssueTriageState, reason: str) -> None:
    state.issue.metadata["close_reason"] = reason


def _set_status_duplicate_label(state: IssueTriageState) -> None:
    dup_label = state.repo_rules.duplicate_policy.get("label", "status:duplicate")
    if dup_label not in state.issue.labels:
        state.issue.labels.append(dup_label)


def _requested_fields_template(state: IssueTriageState, fields: List[str]) -> str:
    template = state.repo_rules.response_templates.get(
        "missing_info",
        "Please provide the following details: {fields}",
    )
    return template.format(fields=", ".join(fields))


def _duplicate_comment_template(state: IssueTriageState, issue_id: str) -> str:
    template = state.repo_rules.response_templates.get(
        "duplicate",
        "This issue appears to be a duplicate of #{issue_id}.",
    )
    return template.format(issue_id=issue_id)


def _close_comment_template(state: IssueTriageState, reason: str) -> str:
    template = state.repo_rules.response_templates.get(
        "closed",
        "Closing this issue as {reason}.",
    )
    return template.format(reason=reason)


def _search_similar_issues_text(state: IssueTriageState, query: str) -> str:
    query_tokens = {t for t in query.lower().split() if t}
    ranked = []
    for cand in state.candidate_duplicates:
        hay = " ".join([cand.title, cand.short_summary, " ".join(cand.labels)]).lower()
        tokens = {t for t in hay.split() if t}
        overlap = len(query_tokens.intersection(tokens))
        ranked.append((overlap, cand.issue_id, cand.similarity_score))
    ranked.sort(key=lambda x: (x[0], x[2]), reverse=True)
    top = [f"{issue_id}:{score:.2f}" for _, issue_id, score in ranked[:3]]
    return ", ".join(top) if top else "no_matches"


def _is_valid_milestone(state: IssueTriageState, milestone: str) -> bool:
    return milestone in set(_available_milestones(state))


def _is_valid_assignee(state: IssueTriageState, username: str) -> bool:
    return username in set(_available_assignees(state))


def _is_valid_label(state: IssueTriageState, label: str) -> bool:
    return label in set(_available_labels(state))


def _maybe_conflicts_allow(state: IssueTriageState, new_label: str) -> bool:
    if not state.repo_rules.strict_mode:
        return True
    conflicts = _label_conflicts(state, new_label)
    current = set(state.issue.labels)
    return not bool(current.intersection(conflicts))


def _set_issue_component_from_label(state: IssueTriageState, label: str) -> None:
    if label.startswith("component:"):
        state.issue.component = label.split(":", 1)[1].strip()


def _set_issue_severity_from_label(state: IssueTriageState, label: str) -> None:
    if label.startswith("severity:"):
        raw = label.split(":", 1)[1].strip().lower()
        try:
            state.issue.severity = Severity(raw)
        except Exception:
            pass


def _set_issue_priority_from_label(state: IssueTriageState, label: str) -> None:
    if label.startswith("priority:"):
        raw = label.split(":", 1)[1].strip().lower()
        try:
            state.issue.priority = Priority(raw)
        except Exception:
            pass


def _apply_status_label(state: IssueTriageState, label: str) -> None:
    if label.startswith("status:"):
        if label == "status:duplicate" and label not in state.issue.labels:
            state.issue.labels.append(label)


def _update_pending_missing_fields(state: IssueTriageState, requested: List[str]) -> None:
    state.requested_fields = _dedupe_keep_order(state.requested_fields + requested)
    if state.hidden_target and state.hidden_target.required_missing_fields:
        needed = set(state.hidden_target.required_missing_fields)
        state.pending_missing_fields = [
            f for f in state.hidden_target.required_missing_fields if f not in set(state.requested_fields)
        ]
    else:
        state.pending_missing_fields = []


def _handle_read_action(effect: str, notes: Optional[List[str]] = None) -> TransitionResult:
    return TransitionResult(
        action_valid=True,
        action_effect=effect,
        changed_fields=[],
        notes=notes or [],
    )


def _handle_add_label(state: IssueTriageState, action: AddLabelAction) -> TransitionResult:
    label = action.label.strip()
    if not _is_valid_label(state, label):
        return TransitionResult(
            action_valid=False,
            action_effect=f"invalid_label:{label}",
            changed_fields=[],
            notes=[f"Label '{label}' is not allowed by repo rules."],
        )

    if not _maybe_conflicts_allow(state, label):
        return TransitionResult(
            action_valid=False,
            action_effect=f"conflicting_label:{label}",
            changed_fields=[],
            notes=[f"Label '{label}' conflicts with existing labels."],
        )

    changed: List[str] = []

    if label not in state.issue.labels:
        state.issue.labels.append(label)
        changed.append("issue.labels")

    _set_issue_component_from_label(state, label)
    _set_issue_severity_from_label(state, label)
    _set_issue_priority_from_label(state, label)
    _apply_status_label(state, label)

    if label.startswith("component:"):
        changed.append("issue.component")
    if label.startswith("severity:"):
        changed.append("issue.severity")
    if label.startswith("priority:"):
        changed.append("issue.priority")

    return TransitionResult(
        action_valid=True,
        action_effect=f"label_added:{label}",
        changed_fields=_dedupe_keep_order(changed),
        notes=[],
    )


def _handle_remove_label(state: IssueTriageState, action: RemoveLabelAction) -> TransitionResult:
    label = action.label.strip()
    if label not in state.issue.labels:
        return TransitionResult(
            action_valid=False,
            action_effect=f"label_not_present:{label}",
            changed_fields=[],
            notes=[f"Label '{label}' is not currently applied."],
        )

    state.issue.labels = [x for x in state.issue.labels if x != label]
    return TransitionResult(
        action_valid=True,
        action_effect=f"label_removed:{label}",
        changed_fields=["issue.labels"],
        notes=[],
    )


def _handle_assign_user(state: IssueTriageState, action: AssignUserAction) -> TransitionResult:
    username = action.username.strip()
    if not _is_valid_assignee(state, username):
        return TransitionResult(
            action_valid=False,
            action_effect=f"invalid_assignee:{username}",
            changed_fields=[],
            notes=[f"Assignee '{username}' is not available in this repo."],
        )

    state.issue.assignees = [username]
    return TransitionResult(
        action_valid=True,
        action_effect=f"assignee_set:{username}",
        changed_fields=["issue.assignees"],
        notes=[],
    )


def _handle_set_priority(state: IssueTriageState, action: SetPriorityAction) -> TransitionResult:
    state.issue.priority = action.priority
    return TransitionResult(
        action_valid=True,
        action_effect=f"priority_set:{action.priority.value}",
        changed_fields=["issue.priority"],
        notes=[],
    )


def _handle_set_milestone(state: IssueTriageState, action: SetMilestoneAction) -> TransitionResult:
    milestone = action.milestone.strip()
    if not _is_valid_milestone(state, milestone):
        return TransitionResult(
            action_valid=False,
            action_effect=f"invalid_milestone:{milestone}",
            changed_fields=[],
            notes=[f"Milestone '{milestone}' is not valid for this repo."],
        )

    state.issue.milestone = milestone
    return TransitionResult(
        action_valid=True,
        action_effect=f"milestone_set:{milestone}",
        changed_fields=["issue.milestone"],
        notes=[],
    )


def _handle_comment(state: IssueTriageState, action: CommentAction) -> TransitionResult:
    text = action.text.strip()
    if not text:
        return TransitionResult(
            action_valid=False,
            action_effect="empty_comment",
            changed_fields=[],
            notes=["Comment text cannot be empty."],
        )

    _ensure_comment(state, body=text, internal=False, actor="triage-agent")
    state.public_notes.append(text)
    return TransitionResult(
        action_valid=True,
        action_effect="comment_added",
        changed_fields=["issue.comments", "public_notes"],
        notes=[],
    )


def _handle_request_info(state: IssueTriageState, action: RequestInfoAction) -> TransitionResult:
    fields = [f.strip() for f in action.fields if f and f.strip()]
    if not fields:
        return TransitionResult(
            action_valid=False,
            action_effect="no_fields_requested",
            changed_fields=[],
            notes=["request_info requires at least one field."],
        )

    _update_pending_missing_fields(state, fields)

    text = _requested_fields_template(state, fields)
    _ensure_comment(state, body=text, internal=False, actor="triage-agent")
    state.public_notes.append(text)

    return TransitionResult(
        action_valid=True,
        action_effect="requested_info",
        changed_fields=["requested_fields", "pending_missing_fields", "issue.comments", "public_notes"],
        notes=[f"Requested fields: {', '.join(fields)}"],
    )


def _handle_mark_duplicate(state: IssueTriageState, action: MarkDuplicateAction) -> TransitionResult:
    issue_id = action.issue_id.strip()
    candidate_ids = {cand.issue_id for cand in state.candidate_duplicates}

    if issue_id not in candidate_ids:
        return TransitionResult(
            action_valid=False,
            action_effect=f"unknown_duplicate_target:{issue_id}",
            changed_fields=[],
            notes=[f"Issue '{issue_id}' is not one of the known duplicate candidates."],
        )

    if issue_id not in state.issue.linked_duplicates:
        state.issue.linked_duplicates.append(issue_id)

    _set_status_duplicate_label(state)
    state.public_notes.append(_duplicate_comment_template(state, issue_id))
    _ensure_comment(state, body=_duplicate_comment_template(state, issue_id), internal=False, actor="triage-agent")

    changed = ["issue.linked_duplicates", "issue.labels", "issue.comments", "public_notes"]

    dup_policy = (state.repo_rules.duplicate_policy.get("action") or "").lower()
    should_close = "close" in dup_policy or dup_policy == "mark_duplicate_and_close"

    if should_close:
        state.issue.status = IssueStatus.CLOSED
        _mark_close_reason(state, "duplicate")
        _ensure_comment(state, body=_close_comment_template(state, "duplicate"), internal=False, actor="triage-agent")
        changed.extend(["issue.status", "issue.metadata", "issue.comments"])

    return TransitionResult(
        action_valid=True,
        action_effect=f"marked_duplicate:{issue_id}",
        changed_fields=_dedupe_keep_order(changed),
        notes=[f"Linked duplicate issue '{issue_id}'."],
    )


def _handle_close_issue(state: IssueTriageState, action: CloseIssueAction) -> TransitionResult:
    reason = action.reason.value

    closure_policy = {x.lower() for x in state.repo_rules.closure_policy}
    if closure_policy and reason not in closure_policy and not state.repo_rules.strict_mode:
        # In relaxed mode, allow closure even if not in policy.
        pass
    elif closure_policy and reason not in closure_policy:
        return TransitionResult(
            action_valid=False,
            action_effect=f"disallowed_close_reason:{reason}",
            changed_fields=[],
            notes=[f"Close reason '{reason}' is not allowed by repo rules."],
        )

    if action.reason == CloseReason.DUPLICATE and not state.issue.linked_duplicates and state.repo_rules.strict_mode:
        return TransitionResult(
            action_valid=False,
            action_effect="cannot_close_duplicate_without_link",
            changed_fields=[],
            notes=["Cannot close as duplicate before linking a canonical issue."],
        )

    state.issue.status = IssueStatus.CLOSED
    _mark_close_reason(state, reason)
    close_comment = _close_comment_template(state, reason)
    _ensure_comment(state, body=close_comment, internal=False, actor="triage-agent")
    state.public_notes.append(close_comment)

    return TransitionResult(
        action_valid=True,
        action_effect=f"closed:{reason}",
        changed_fields=["issue.status", "issue.metadata", "issue.comments", "public_notes"],
        notes=[],
    )


def _handle_reopen_issue(state: IssueTriageState, action: ReopenIssueAction) -> TransitionResult:
    state.issue.status = IssueStatus.OPEN
    state.issue.metadata["close_reason"] = ""
    return TransitionResult(
        action_valid=True,
        action_effect="reopened",
        changed_fields=["issue.status", "issue.metadata"],
        notes=[],
    )


def _handle_search_similar_issues(state: IssueTriageState, action: SearchSimilarIssuesAction) -> TransitionResult:
    text = _search_similar_issues_text(state, action.query)
    return TransitionResult(
        action_valid=True,
        action_effect=f"similar_issues:{text}",
        changed_fields=[],
        notes=[text] if text else [],
    )


def apply_action_to_state(state: IssueTriageState, action: Action) -> TransitionResult:
    """
    Apply one action to the current episode state.

    This module performs all mutation work:
    - validation
    - label changes
    - assignment
    - priority / milestone changes
    - duplicate marking
    - close / reopen
    - comment + request-info behavior
    - read actions as no-op responses
    """
    # Keep state-level validity in sync so reward can see it.
    result: TransitionResult

    if action.type == ActionType.READ_ISSUE:
        result = _handle_read_action("issue_read")
    elif action.type == ActionType.READ_REPO_RULES:
        result = _handle_read_action("repo_rules_read")
    elif action.type == ActionType.READ_LABEL_DEFINITIONS:
        result = _handle_read_action("label_definitions_read")
    elif action.type == ActionType.READ_TEAM_ROUTING:
        result = _handle_read_action("team_routing_read")
    elif action.type == ActionType.READ_ASSIGNEE_POOL:
        result = _handle_read_action("assignee_pool_read")
    elif action.type == ActionType.READ_MILESTONES:
        result = _handle_read_action("milestones_read")
    elif action.type == ActionType.SEARCH_SIMILAR_ISSUES:
        result = _handle_search_similar_issues(state, action)  # type: ignore[arg-type]
    elif action.type == ActionType.ADD_LABEL:
        result = _handle_add_label(state, action)  # type: ignore[arg-type]
    elif action.type == ActionType.REMOVE_LABEL:
        result = _handle_remove_label(state, action)  # type: ignore[arg-type]
    elif action.type == ActionType.ASSIGN_USER:
        result = _handle_assign_user(state, action)  # type: ignore[arg-type]
    elif action.type == ActionType.SET_PRIORITY:
        result = _handle_set_priority(state, action)  # type: ignore[arg-type]
    elif action.type == ActionType.SET_MILESTONE:
        result = _handle_set_milestone(state, action)  # type: ignore[arg-type]
    elif action.type == ActionType.COMMENT:
        result = _handle_comment(state, action)  # type: ignore[arg-type]
    elif action.type == ActionType.REQUEST_INFO:
        result = _handle_request_info(state, action)  # type: ignore[arg-type]
    elif action.type == ActionType.MARK_DUPLICATE:
        result = _handle_mark_duplicate(state, action)  # type: ignore[arg-type]
    elif action.type == ActionType.CLOSE_ISSUE:
        result = _handle_close_issue(state, action)  # type: ignore[arg-type]
    elif action.type == ActionType.REOPEN_ISSUE:
        result = _handle_reopen_issue(state, action)  # type: ignore[arg-type]
    elif action.type == ActionType.NOOP:
        result = TransitionResult(action_valid=True, action_effect="noop", changed_fields=[], notes=[])
    else:
        result = TransitionResult(
            action_valid=False,
            action_effect=f"unsupported_action:{action.type}",
            changed_fields=[],
            notes=[f"Unsupported action type: {action.type}"],
        )

    state.last_action_valid = result.action_valid
    state.last_action_message = result.action_effect
    return result