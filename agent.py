from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, model_validator

load_dotenv()


class LLMAction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal[
        "read_issue",
        "read_repo_rules",
        "read_label_definitions",
        "read_team_routing",
        "read_assignee_pool",
        "read_milestones",
        "search_similar_issues",
        "add_label",
        "remove_label",
        "assign_user",
        "set_priority",
        "set_milestone",
        "comment",
        "request_info",
        "mark_duplicate",
        "close_issue",
        "reopen_issue",
        "noop",
    ]

    issue_id: Optional[str] = None
    query: Optional[str] = None
    label: Optional[str] = None
    username: Optional[str] = None
    priority: Optional[str] = None
    milestone: Optional[str] = None
    text: Optional[str] = None
    fields: Optional[List[str]] = None
    reason: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        data = dict(data)

        # Accept multiple model output styles.
        if "type" not in data:
            if "action" in data:
                data["type"] = data.pop("action")
            elif "action_type" in data:
                data["type"] = data.pop("action_type")

        # Flatten nested payloads.
        payload = data.pop("action_payload", None)
        if isinstance(payload, dict):
            for k, v in payload.items():
                data.setdefault(k, v)

        # Drop common non-schema leakage.
        for k in ["outcome", "success", "timestamp", "step_index"]:
            data.pop(k, None)

        return data


SYSTEM_PROMPT = """
You are a GitHub Issue Triage Manager agent operating inside a strict Pydantic-validated action environment.

You MUST output exactly one JSON object and nothing else.

Hard requirements:
- The JSON object must conform to exactly one of the Action models in the discriminated union.
- The action discriminator key is "type", not "action".
- Never output plain strings, nested wrappers, explanations, markdown, or extra keys.
- Every action must include all required fields for that action.
- In particular:
  - read_issue MUST include "issue_id"
  - request_info MUST include "fields"
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
- The current issue context is already available to you; use the current issue_id when required.
- Read the issue and repo rules early if you need context before taking a triage action.
- If required information is missing, request it with REQUEST_INFO and list only the missing fields.
- If a strong duplicate is present, use MARK_DUPLICATE with the correct issue_id.
- Follow repo rules for labels, priority, milestone, severity, routing, and closure.
- Take only one action per step.

Reasoning rules:
- Prefer valid, minimal, deterministic actions.
- Never invent missing values.
- Never emit an action that would fail schema validation.
- Before returning, ensure the JSON validates against the Action schema.
""".strip()


class IssueTriageAgent:
    def __init__(self) -> None:
        self.api_base_url = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
        self.api_key = (
            os.getenv("HF_TOKEN")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("API_KEY")
        )
        self.model_name = os.getenv("MODEL_NAME")

        self.temperature = float(os.getenv("TEMPERATURE", "0.2"))
        self.max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "200"))

        print(f"Using API base URL: {self.api_base_url}")
        print(f"Using model: {self.model_name}")
        print(f"Using API key: {'Yes' if self.api_key else 'No'}")

        if not self.api_key:
            raise RuntimeError("Missing HF_TOKEN / OPENAI_API_KEY / API_KEY.")
        if not self.model_name:
            raise RuntimeError("Missing MODEL_NAME.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)

    def next_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
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
                ],
                response_format={"type": "json_object"},
                temperature=self.temperature,
            )

            content = response.choices[0].message.content or "{}"
            print(f"LLM Raw Output: {content}")

            action = LLMAction.model_validate_json(content)
            payload = action.model_dump(exclude_none=True)
            return payload if "type" in payload else self._fallback()

        except Exception as e:
            print(f"Error occurred: {e}")
            return self._fallback()

    def _fallback(self) -> Dict[str, Any]:
        return {"type": "noop"}