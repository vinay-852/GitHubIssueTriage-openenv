# envs/your_env/server/actions.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Union

from pydantic import TypeAdapter

try:
    from GitHubIssueTriage.models import Action, ActionType, validate_action_payload
except ImportError:  # pragma: no cover
    from models import Action, ActionType, validate_action_payload

ACTION_ADAPTER = TypeAdapter(Action)


@dataclass
class ParsedActionResult:
    action: Action
    valid: bool
    effect: str = ""
    notes: list[str] = field(default_factory=list)


def parse_action(action: Union[Action, Dict[str, Any]]) -> Action:
    """
    Validate and normalize a raw action dict into a typed Action model.
    If the action is already a typed Action, it is returned unchanged.
    """
    if isinstance(action, dict):
        raw_action = action
    elif hasattr(action, "model_dump"):
        raw_action = action.model_dump(exclude_none=True, mode="json")
    else:
        return ACTION_ADAPTER.validate_python(action)

    sanitized = _sanitize_raw_action(raw_action)
    try:
        return ACTION_ADAPTER.validate_python(sanitized)
    except Exception as e:
        # Provide detailed error information
        import json
        print(f"[ACTION_PARSE_ERROR] Validation failed for action: {json.dumps(sanitized)}", flush=True)
        print(f"[ACTION_PARSE_ERROR] Error: {str(e)}", flush=True)
        raise ValueError(f"Invalid action format: {sanitized}. Error: {str(e)}") from e


def get_action_type(action: Union[Action, Dict[str, Any]]) -> ActionType:
    """
    Convenience helper for routing logic.
    """
    return parse_action(action).type


def action_to_dict(action: Union[Action, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize an action into a plain dictionary.
    """
    return parse_action(action).model_dump()


def is_read_action(action: Union[Action, Dict[str, Any]]) -> bool:
    """
    Returns True for read-only actions.
    """
    action_type = get_action_type(action)
    return action_type in {
        ActionType.READ_ISSUE,
        ActionType.READ_REPO_RULES,
        ActionType.READ_LABEL_DEFINITIONS,
        ActionType.READ_TEAM_ROUTING,
        ActionType.READ_ASSIGNEE_POOL,
        ActionType.READ_MILESTONES,
        ActionType.SEARCH_SIMILAR_ISSUES,
    }


def is_mutating_action(action: Union[Action, Dict[str, Any]]) -> bool:
    """
    Returns True for actions that can change issue state.
    """
    return not is_read_action(action)


def parse_and_validate_action(
    action: Union[Action, Dict[str, Any]],
    allowed_actions: Iterable[ActionType],
) -> ParsedActionResult:
    """
    Parse an action, ensuring it is permitted for the current task and payload is well-formed.
    """
    parsed = parse_action(action)
    allowed_set = set(allowed_actions)

    if allowed_set and parsed.type not in allowed_set:
        return ParsedActionResult(
            action=parsed,
            valid=False,
            effect=f"action_disallowed:{parsed.type.value}",
            notes=[f"Action '{parsed.type.value}' is not allowed for this task."],
        )

    is_valid, message = validate_action_payload(parsed)
    if not is_valid:
        return ParsedActionResult(
            action=parsed,
            valid=False,
            effect="action_validation_failed",
            notes=[message],
        )

    return ParsedActionResult(action=parsed, valid=True)


FIELD_WHITELIST: Dict[ActionType, set[str]] = {
    ActionType.READ_ISSUE: {"type", "issue_id"},
    ActionType.READ_REPO_RULES: {"type"},
    ActionType.READ_LABEL_DEFINITIONS: {"type"},
    ActionType.READ_TEAM_ROUTING: {"type"},
    ActionType.READ_ASSIGNEE_POOL: {"type"},
    ActionType.READ_MILESTONES: {"type"},
    ActionType.SEARCH_SIMILAR_ISSUES: {"type", "query"},
    ActionType.ADD_LABEL: {"type", "label"},
    ActionType.REMOVE_LABEL: {"type", "label"},
    ActionType.ASSIGN_USER: {"type", "username"},
    ActionType.SET_PRIORITY: {"type", "priority"},
    ActionType.SET_MILESTONE: {"type", "milestone"},
    ActionType.COMMENT: {"type", "text"},
    ActionType.REQUEST_INFO: {"type", "fields"},
    ActionType.MARK_DUPLICATE: {"type", "issue_id"},
    ActionType.CLOSE_ISSUE: {"type", "reason"},
    ActionType.REOPEN_ISSUE: {"type", "reason"},
    ActionType.NOOP: {"type"},
}


def _sanitize_raw_action(raw_action: Dict[str, Any]) -> Dict[str, Any]:
    raw_type = raw_action.get("type")
    if not isinstance(raw_type, str):
        return raw_action

    try:
        action_type = ActionType(raw_type)
    except ValueError:
        return raw_action

    allowed_fields = FIELD_WHITELIST.get(action_type)
    if not allowed_fields:
        return raw_action

    sanitized = {key: value for key, value in raw_action.items() if key in allowed_fields}
    sanitized["type"] = raw_type
    return sanitized