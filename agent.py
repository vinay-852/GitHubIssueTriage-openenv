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
    def __init__(self) -> None:
        self.api_base_url = BASE_URL
        self.api_key = API_KEY
        self.model_name = os.getenv("MODEL_NAME", "oca/gpt-5")
        self.temperature = float(os.getenv("TEMPERATURE", "0.2"))
        self.max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "200"))

        print(f"Using API base URL: {self.api_base_url}")
        print(f"Using model: {self.model_name}")
        print(f"Using API key: {'Yes' if self.api_key else 'No'}")

        self.client: Optional[OpenAI] = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)
        else:
            print("OPENAI_API_KEY is not set. Falling back to rule-based actions.")

    def _fallback_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        task = observation.get("task", {})
        issue_id = task.get("issue_id")

        history = observation.get("action_history", [])
        history_types = {
            (entry.get("action_type") or "").lower() for entry in history if isinstance(entry, dict)
        }

        if issue_id and ActionType.READ_ISSUE.value not in history_types:
            return {"type": ActionType.READ_ISSUE.value, "issue_id": issue_id}

        if ActionType.READ_REPO_RULES.value not in history_types:
            return {"type": ActionType.READ_REPO_RULES.value}

        pending_fields = observation.get("pending_missing_fields") or []
        if pending_fields:
            safe_fields = [f for f in pending_fields if isinstance(f, str) and f.strip()]
            if safe_fields:
                return {"type": ActionType.REQUEST_INFO.value, "fields": safe_fields}

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
            print(f"[Debug] LLM Raw Stream Output: {raw_text}")

            return self._parse_action_json(raw_text)

        except Exception as e:
            print(f"Error occurred in next_action_streaming: {e}")
            return self._fallback_action(observation)