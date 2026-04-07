from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import TypeAdapter, ValidationError

from models import Action, ActionType

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("API_BASE_URL")

ACTION_ADAPTER = TypeAdapter(Action)

SYSTEM_PROMPT = """
You are a GitHub Issue Triage Manager agent operating inside a strict Pydantic-validated action environment.

You MUST output exactly one JSON object and nothing else.

Hard requirements:
- The JSON object must conform to exactly one of the Action models in the discriminated union.
- The action discriminator key is "type", not "action".
- Never output plain strings, nested wrappers, explanations, markdown, or extra keys.
- Every action must include all required fields for that action.
- Use only these exact action type values:
  read_issue
  read_repo_rules
  read_label_definitions
  read_team_routing
  read_assignee_pool
  read_milestones
  search_similar_issues
  add_label
  remove_label
  assign_user
  set_priority
  set_milestone
  comment
  request_info
  provide_info
  mark_duplicate
  close_issue
  reopen_issue
  noop

Field requirements:
- read_issue MUST include "issue_id"
- request_info MUST include "fields"
- provide_info MUST include "fields"
- add_label MUST include "label"
- remove_label MUST include "label"
- assign_user MUST include "username"
- set_priority MUST include "priority"
- set_milestone MUST include "milestone"
- comment MUST include "text"
- mark_duplicate MUST include "issue_id"
- close_issue MUST include "reason"
- reopen_issue MUST include "reason"

Context rules:
- Read the issue and repo rules early if you need context before taking a triage action.
- If required information is missing, choose REQUEST_INFO and list only the missing fields.
- If a strong duplicate is present, choose MARK_DUPLICATE with the correct issue_id.
- Follow repo rules for labels, priority, milestone, severity, routing, and closure.
- Take only one action per step.
- Keep COMMENT short and focused.
""".strip()


class IssueTriageAgent:
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model_name: Optional[str] = None,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.api_base_url = api_base_url if api_base_url is not None else BASE_URL
        self.api_key = api_key if api_key is not None else API_KEY
        self.model_name = model_name or os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
        self.temperature = float(os.getenv("TEMPERATURE", "0.0"))
        self.max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "200"))

        self.client: Optional[OpenAI] = client
        if self.client is None and self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)

    def _fallback_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        task = observation.get("task", {}) or {}
        issue = observation.get("issue", {}) or {}
        repo_rules = observation.get("repo_rules", {}) or {}
        issue_id = task.get("issue_id")
        allowed_actions = set(task.get("allowed_actions") or [])

        history = observation.get("action_history", []) or []
        history_types = set()
        for entry in history:
            if not isinstance(entry, dict):
                continue
            raw_type = entry.get("action_type")
            if hasattr(raw_type, "value"):
                normalized = str(raw_type.value).strip().lower()
            else:
                normalized = str(raw_type or "").strip().lower()
                if normalized.startswith("actiontype."):
                    normalized = normalized.split(".", 1)[1]
            if normalized:
                history_types.add(normalized)

        if (
            issue_id
            and ActionType.READ_ISSUE.value in allowed_actions
            and ActionType.READ_ISSUE.value not in history_types
        ):
            return {"type": ActionType.READ_ISSUE.value, "issue_id": issue_id}

        if (
            ActionType.READ_REPO_RULES.value in allowed_actions
            and ActionType.READ_REPO_RULES.value not in history_types
        ):
            return {"type": ActionType.READ_REPO_RULES.value}

        pending_fields = observation.get("pending_missing_fields") or []
        if (
            ActionType.REQUEST_INFO.value in allowed_actions
            and pending_fields
            and ActionType.REQUEST_INFO.value not in history_types
        ):
            safe_fields = [f for f in pending_fields if isinstance(f, str) and f.strip()]
            if safe_fields:
                return {"type": ActionType.REQUEST_INFO.value, "fields": safe_fields}

        if ActionType.ADD_LABEL.value in allowed_actions:
            labels = set(issue.get("labels") or [])
            objective_summary = observation.get("objective_summary") or []
            for line in objective_summary:
                if not isinstance(line, str) or not line.startswith("Labels needed:"):
                    continue
                needed = [x.strip() for x in line.split(":", 1)[1].split(",") if x.strip()]
                for label in needed:
                    if label not in labels:
                        return {"type": ActionType.ADD_LABEL.value, "label": label}

        if ActionType.ASSIGN_USER.value in allowed_actions and not (issue.get("assignees") or []):
            component = str(issue.get("component") or "").strip()
            routing = repo_rules.get("routing_rules", {}) or {}
            if component in routing and isinstance(routing[component], list):
                for candidate in routing[component]:
                    if isinstance(candidate, str) and candidate.strip():
                        return {"type": ActionType.ASSIGN_USER.value, "username": candidate.strip()}
            pool = repo_rules.get("assignee_pool", []) or []
            for candidate in pool:
                if isinstance(candidate, str) and candidate.strip():
                    return {"type": ActionType.ASSIGN_USER.value, "username": candidate.strip()}

        if ActionType.MARK_DUPLICATE.value in allowed_actions and not (issue.get("linked_duplicates") or []):
            candidates = observation.get("candidate_duplicates") or []
            best = None
            best_score = -1.0
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                score = float(cand.get("similarity_score") or 0.0)
                issue_ref = cand.get("issue_id")
                if isinstance(issue_ref, str) and score > best_score:
                    best_score = score
                    best = issue_ref
            if best:
                return {"type": ActionType.MARK_DUPLICATE.value, "issue_id": best}

        if (
            ActionType.CLOSE_ISSUE.value in allowed_actions
            and str(issue.get("status") or "").lower() == "open"
            and (issue.get("linked_duplicates") or [])
        ):
            return {"type": ActionType.CLOSE_ISSUE.value, "reason": "duplicate"}

        if ActionType.COMMENT.value in allowed_actions:
            comments = issue.get("comments") or []
            if not comments:
                return {"type": ActionType.COMMENT.value, "text": "triaged and policy checks applied"}

        if ActionType.NOOP.value in allowed_actions:
            return {"type": ActionType.NOOP.value}

        if ActionType.READ_LABEL_DEFINITIONS.value in allowed_actions:
            return {"type": ActionType.READ_LABEL_DEFINITIONS.value}
        if ActionType.READ_MILESTONES.value in allowed_actions:
            return {"type": ActionType.READ_MILESTONES.value}
        if ActionType.READ_TEAM_ROUTING.value in allowed_actions:
            return {"type": ActionType.READ_TEAM_ROUTING.value}

        if allowed_actions:
            first = sorted(allowed_actions)[0]
            payload: Dict[str, Any] = {"type": first}
            if first == ActionType.READ_ISSUE.value and issue_id:
                payload["issue_id"] = issue_id
            elif first == ActionType.ADD_LABEL.value:
                payload["label"] = "type:bug"
            elif first == ActionType.ASSIGN_USER.value:
                payload["username"] = "devon"
            elif first == ActionType.SET_PRIORITY.value:
                payload["priority"] = "p2"
            elif first == ActionType.SET_MILESTONE.value:
                payload["milestone"] = "backlog"
            elif first == ActionType.COMMENT.value:
                payload["text"] = "triage update"
            elif first == ActionType.REQUEST_INFO.value:
                payload["fields"] = ["steps_to_reproduce"]
            elif first == ActionType.MARK_DUPLICATE.value:
                payload["issue_id"] = "issue_099"
            elif first == ActionType.CLOSE_ISSUE.value:
                payload["reason"] = "duplicate"
            return payload

        return {"type": ActionType.NOOP.value}

    def _build_messages(self, observation: Dict[str, Any]) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "instruction": "Choose the next best triage action.",
                        "observation": observation,
                    },
                    indent=2,
                ),
            },
        ]

    def _strip_code_fences(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2 and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def _extract_json_object(self, text: str) -> str:
        text = self._strip_code_fences(text)

        if text.startswith("{") and text.endswith("}"):
            return text

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]

        raise ValueError("No JSON object found in model output.")

    def _sanitize_action_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)

        if "type" not in data:
            if "action" in data:
                data["type"] = data.pop("action")
            elif "action_type" in data:
                data["type"] = data.pop("action_type")

        payload = data.pop("action_payload", None)
        if isinstance(payload, dict):
            for key, value in payload.items():
                data.setdefault(key, value)

        for key in [
            "outcome",
            "success",
            "timestamp",
            "step_index",
            "message",
            "analysis",
            "thought",
            "reasoning",
        ]:
            data.pop(key, None)

        return data

    def _parse_action_json(self, raw_text: str) -> Dict[str, Any]:
        json_text = self._extract_json_object(raw_text)
        data = json.loads(json_text)
        if not isinstance(data, dict):
            raise ValueError("Model output was not a JSON object.")

        sanitized = self._sanitize_action_dict(data)
        action = ACTION_ADAPTER.validate_python(sanitized)
        return action.model_dump(mode="json", exclude_none=True)

    def next_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self.client is None:
            return self._fallback_action(observation)

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=self._build_messages(observation),
                stream=True,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            parts: List[str] = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    parts.append(delta)

            raw_text = "".join(parts).strip()
            return self._parse_action_json(raw_text)

        except Exception as e:
            return self._fallback_action(observation)
