# envs/your_env/server/actions.py
from __future__ import annotations

from typing import Any, Dict, Union

from pydantic import TypeAdapter

from ..model import Action, ActionType

ACTION_ADAPTER = TypeAdapter(Action)


def parse_action(action: Union[Action, Dict[str, Any]]) -> Action:
    """
    Validate and normalize a raw action dict into a typed Action model.
    If the action is already a typed Action, it is returned unchanged.
    """
    if isinstance(action, dict):
        return ACTION_ADAPTER.validate_python(action)
    return action


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