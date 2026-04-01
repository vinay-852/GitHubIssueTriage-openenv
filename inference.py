#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load environment variables from project/local .env files if available.
if load_dotenv is not None:
    load_dotenv(dotenv_path=ROOT / ".env")
    load_dotenv()

try:
    from envs.GitHubIssueTriageManager.server.environment import GitHubIssueTriageEnvironment
    from envs.GitHubIssueTriageManager.server.grader import grade_episode
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import envs.GitHubIssueTriageManager.server.environment / grader. "
        "Check package paths and file names."
    ) from exc


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
DATA_DIR = os.getenv("DATA_DIR") or os.getenv("TRIAGE_DATA_DIR") or "data"
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "180"))

ALLOWED_ACTION_TYPES = {
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
}

SYSTEM_PROMPT = """
You are a GitHub Issue Triage Manager agent.

You MUST output exactly one valid JSON object and nothing else.

The JSON must represent one next action only.

Allowed actions:
- {"type":"read_issue","issue_id":"..."}
- {"type":"read_repo_rules"}
- {"type":"read_label_definitions"}
- {"type":"read_team_routing"}
- {"type":"read_assignee_pool"}
- {"type":"read_milestones"}
- {"type":"search_similar_issues","query":"..."}
- {"type":"add_label","label":"..."}
- {"type":"remove_label","label":"..."}
- {"type":"assign_user","username":"..."}
- {"type":"set_priority","priority":"p0|p1|p2|p3"}
- {"type":"set_milestone","milestone":"..."}
- {"type":"comment","text":"..."}
- {"type":"request_info","fields":["...","..."]}
- {"type":"mark_duplicate","issue_id":"..."}
- {"type":"close_issue","reason":"duplicate|invalid|wontfix|resolved|stale|not_enough_info"}
- {"type":"reopen_issue","reason":"..."}
- {"type":"noop"}

Decision rules:
1. Read the issue and repo rules early.
2. If required details are missing, request them.
3. If there is a strong duplicate candidate, mark it.
4. Use repo labels and routing rules before guessing.
5. Keep comments short and specific.
6. Do not output explanations, markdown, or multiple actions.

Output JSON only.
""".strip()


def make_client() -> OpenAI:
    if not MODEL_NAME:
        raise RuntimeError("MODEL_NAME is required.")
    if not API_KEY:
        raise RuntimeError("HF_TOKEN, OPENAI_API_KEY, or API_KEY is required.")
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def load_environment() -> GitHubIssueTriageEnvironment:
    data_path = Path(DATA_DIR)
    if data_path.exists():
        return GitHubIssueTriageEnvironment(data_dir=data_path, strict_mode=True)
    return GitHubIssueTriageEnvironment(episodes=[], strict_mode=True)

def _enum_value(x: Any) -> Any:
    return getattr(x, "value", x)

def summarize_observation(obs) -> Dict[str, Any]:
    issue = obs.issue
    task = obs.task
    repo = obs.repo_rules

    history = []
    for h in obs.action_history[-8:]:
        history.append(
            {
                "step_index": h.step_index,
                "action_type": _enum_value(h.action_type),
                "outcome": h.outcome,
                "success": h.success,
            }
        )

    candidates = []
    for c in obs.candidate_duplicates[:5]:
        candidates.append(
            {
                "issue_id": c.issue_id,
                "title": c.title,
                "short_summary": c.short_summary,
                "similarity_score": c.similarity_score,
                "labels": c.labels,
                "status": _enum_value(c.status),
            }
        )

    return {
        "task": {
            "task_id": task.task_id,
            "difficulty": _enum_value(task.difficulty),
            "goal_type": _enum_value(task.goal_type),
            "max_steps": task.max_steps,
            "allowed_actions": [_enum_value(a) for a in task.allowed_actions],
            "repo_rules_url": task.repo_rules_url,
        },
        "issue": {
            "issue_id": issue.issue_id,
            "issue_url": issue.issue_url,
            "repo_id": issue.repo_id,
            "title": issue.title,
            "body": issue.body,
            "author": issue.author,
            "status": _enum_value(issue.status),
            "labels": issue.labels,
            "assignees": issue.assignees,
            "milestone": issue.milestone,
            "priority": _enum_value(issue.priority) if issue.priority else None,
            "severity": _enum_value(issue.severity) if issue.severity else None,
            "component": issue.component,
            "linked_duplicates": issue.linked_duplicates,
        },
        "repo_rules": {
            "repo_id": repo.repo_id,
            "repo_name": repo.repo_name,
            "source_url": repo.source_url,
            "source_kind": repo.source_kind,
            "strict_mode": repo.strict_mode,
            "labels": repo.labels,
            "routing_rules": repo.routing_rules,
            "milestones": repo.milestones,
            "missing_info": repo.missing_info,
            "duplicate_policy": repo.duplicate_policy,
            "closure_policy": repo.closure_policy,
            "response_templates": repo.response_templates,
            "assignee_pool": repo.assignee_pool,
        },
        "candidate_duplicates": candidates,
        "pending_missing_fields": obs.pending_missing_fields,
        "remaining_steps": obs.remaining_steps,
        "step_count": obs.step_count,
        "last_action_valid": obs.last_action_valid,
        "last_action_message": obs.last_action_message,
        "history": history,
    }


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    raise ValueError(f"Could not parse JSON object from model output: {text[:300]}")


def validate_action(action: Dict[str, Any], obs) -> bool:
    if not isinstance(action, dict):
        return False

    action_type = action.get("type")
    if action_type not in ALLOWED_ACTION_TYPES:
        return False

    # Required keys per action type.
    required: Dict[str, List[str]] = {
        "read_issue": ["issue_id"],
        "search_similar_issues": ["query"],
        "add_label": ["label"],
        "remove_label": ["label"],
        "assign_user": ["username"],
        "set_priority": ["priority"],
        "set_milestone": ["milestone"],
        "comment": ["text"],
        "request_info": ["fields"],
        "mark_duplicate": ["issue_id"],
        "close_issue": ["reason"],
        "reopen_issue": ["reason"],
        "noop": [],
        "read_repo_rules": [],
        "read_label_definitions": [],
        "read_team_routing": [],
        "read_assignee_pool": [],
        "read_milestones": [],
    }

    for key in required.get(action_type, []):
        if key not in action:
            return False

    # Light sanity checks.
    if action_type == "read_issue":
        if str(action.get("issue_id")) != str(obs.issue.issue_id):
            return False

    if action_type == "request_info":
        fields = action.get("fields")
        if not isinstance(fields, list) or not fields:
            return False
        if not all(isinstance(x, str) and x.strip() for x in fields):
            return False

    if action_type == "close_issue":
        if str(action.get("reason")) not in {
            "duplicate",
            "invalid",
            "wontfix",
            "resolved",
            "stale",
            "not_enough_info",
        }:
            return False

    if action_type == "set_priority":
        if str(action.get("priority")) not in {"p0", "p1", "p2", "p3"}:
            return False

    return True

def first_missing_context_action(obs) -> Optional[Dict[str, Any]]:
    history_types = {
        _enum_value(h.action_type) for h in obs.action_history
    }

    if "read_issue" not in history_types:
        return {"type": "read_issue", "issue_id": obs.issue.issue_id}
    if "read_repo_rules" not in history_types:
        return {"type": "read_repo_rules"}
    if "read_label_definitions" not in history_types:
        return {"type": "read_label_definitions"}
    if "read_assignee_pool" not in history_types:
        return {"type": "read_assignee_pool"}
    if "read_milestones" not in history_types:
        return {"type": "read_milestones"}

    if _enum_value(obs.task.goal_type) == "duplicate_resolution" and "search_similar_issues" not in history_types:
        query = f"{obs.issue.title}\n{obs.issue.body}"
        return {"type": "search_similar_issues", "query": query}

    return None


def heuristic_fallback(obs) -> Dict[str, Any]:
    text = f"{obs.issue.title}\n{obs.issue.body}".lower()
    labels = obs.repo_rules.labels or {}

    if _enum_value(obs.task.goal_type) == "needs_info" and obs.pending_missing_fields:
        return {"type": "request_info", "fields": obs.pending_missing_fields[:4]}

    if _enum_value(obs.task.goal_type) == "duplicate_resolution" and obs.candidate_duplicates:
        best = max(obs.candidate_duplicates, key=lambda c: c.similarity_score)
        return {"type": "mark_duplicate", "issue_id": best.issue_id}

    if any(word in text for word in ["crash", "error", "fail", "failure", "bug", "broken", "500"]):
        type_labels = labels.get("type", [])
        label = type_labels[0] if type_labels else "type:bug"
        return {"type": "add_label", "label": label}

    if any(word in text for word in ["feature", "enhancement", "request", "support for"]):
        type_labels = labels.get("type", [])
        label = type_labels[1] if len(type_labels) > 1 else (type_labels[0] if type_labels else "type:feature")
        return {"type": "add_label", "label": label}

    if obs.repo_rules.assignee_pool:
        return {"type": "assign_user", "username": obs.repo_rules.assignee_pool[0]}

    return {"type": "noop"}


def build_user_prompt(obs) -> str:
    payload = summarize_observation(obs)
    return (
        "You must output JSON.\n"
        "Choose the single best next action.\n\n"
        f"Observation:\n{json.dumps(payload, indent=2)}\n"
    )


def choose_action(client: OpenAI, obs) -> Dict[str, Any]:
    preflight = first_missing_context_action(obs)
    if preflight is not None:
        return preflight

    user_prompt = build_user_prompt(obs)

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            text={"format": {"type": "json_object"}},
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        text = getattr(response, "output_text", "") or ""
    except Exception:
        return heuristic_fallback(obs)

    if not text.strip():
        return heuristic_fallback(obs)

    try:
        action = extract_json_object(text)
    except Exception:
        return heuristic_fallback(obs)

    if not validate_action(action, obs):
        return heuristic_fallback(obs)

    return action


# ---------------------------------------------------------------------
# Episode run
# ---------------------------------------------------------------------
def run_episode(client: OpenAI, env: GitHubIssueTriageEnvironment, task_id: str) -> Tuple[float, Dict[str, Any]]:
    obs = env.reset(task_id=task_id)

    done = False
    steps = 0

    while not done and steps < MAX_STEPS:
        action = choose_action(client, obs)
        result = env.step(action)
        obs = result.observation
        done = result.done
        steps += 1

    final_state = env.state()
    grader_result = grade_episode(final_state)
    return grader_result.score, grader_result.model_dump()


def main() -> int:
    client = make_client()
    env = load_environment()

    episodes = getattr(env, "_episodes_source", [])
    if not episodes:
        raise RuntimeError(
            f"No episodes loaded. Put repo_rules.json, tasks.json, and issues.json in {DATA_DIR}, "
            f"or set DATA_DIR/TRIAGE_DATA_DIR appropriately."
        )

    task_ids = [ep.task.task_id for ep in episodes]

    results: List[Dict[str, Any]] = []
    scores: List[float] = []

    for task_id in task_ids:
        score, grader = run_episode(client, env, task_id)
        scores.append(score)
        record = {
            "task_id": task_id,
            "score": score,
            "grader": grader,
        }
        results.append(record)
        print(json.dumps(record, indent=2))

    average = sum(scores) / len(scores) if scores else 0.0
    summary = {
        "tasks": len(scores),
        "average_score": average,
        "scores": scores,
    }
    print(json.dumps(summary, indent=2))

    out_path = os.getenv("INFERENCE_RESULTS_PATH")
    if out_path:
        Path(out_path).write_text(
            json.dumps({"results": results, "summary": summary}, indent=2),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())