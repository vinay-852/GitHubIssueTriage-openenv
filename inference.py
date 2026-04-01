from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from envs.GitHubIssueTriageManager.server.environment import GitHubIssueTriageEnvironment
    from envs.GitHubIssueTriageManager.server.grader import grade_episode
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import your environment. "
        "Check that envs/GitHubIssueTriageManager/server/environment.py and grader.py exist "
        "and that your package paths are correct."
    ) from exc


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
DATA_DIR = os.getenv("DATA_DIR") or os.getenv("TRIAGE_DATA_DIR") or "data"
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "180"))

SYSTEM_PROMPT = """
You are a GitHub Issue Triage Manager agent.

Return exactly one JSON object and nothing else.

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

Rules:
- Prefer read actions first if context is missing.
- Use request_info for missing reproduction details.
- Use mark_duplicate only when there is a strong duplicate candidate.
- Keep comments short and specific.
- Output JSON only, no markdown, no explanation.
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


def summarize_observation(obs) -> Dict[str, Any]:
    issue = obs.issue
    task = obs.task
    repo = obs.repo_rules

    history = []
    for h in obs.action_history[-6:]:
        history.append(
            {
                "step_index": h.step_index,
                "action_type": h.action_type.value if hasattr(h.action_type, "value") else str(h.action_type),
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
                "status": c.status.value if hasattr(c.status, "value") else str(c.status),
            }
        )

    return {
        "task": {
            "task_id": task.task_id,
            "difficulty": task.difficulty.value if hasattr(task.difficulty, "value") else str(task.difficulty),
            "goal_type": task.goal_type.value if hasattr(task.goal_type, "value") else str(task.goal_type),
            "max_steps": task.max_steps,
            "allowed_actions": [
                a.value if hasattr(a, "value") else str(a) for a in task.allowed_actions
            ],
        },
        "issue": {
            "issue_id": issue.issue_id,
            "issue_url": issue.issue_url,
            "repo_id": issue.repo_id,
            "title": issue.title,
            "body": issue.body,
            "author": issue.author,
            "status": issue.status.value if hasattr(issue.status, "value") else str(issue.status),
            "labels": issue.labels,
            "assignees": issue.assignees,
            "milestone": issue.milestone,
            "priority": issue.priority.value if issue.priority else None,
            "severity": issue.severity.value if issue.severity else None,
            "component": issue.component,
            "linked_duplicates": issue.linked_duplicates,
        },
        "repo_rules": {
            "repo_id": repo.repo_id,
            "repo_name": repo.repo_name,
            "labels": repo.labels,
            "routing_rules": repo.routing_rules,
            "milestones": repo.milestones,
            "missing_info": repo.missing_info,
            "duplicate_policy": repo.duplicate_policy,
            "closure_policy": repo.closure_policy,
            "response_templates": repo.response_templates,
            "assignee_pool": repo.assignee_pool,
            "strict_mode": repo.strict_mode,
            "source_url": repo.source_url,
            "source_kind": repo.source_kind,
        },
        "candidate_duplicates": candidates,
        "pending_missing_fields": obs.pending_missing_fields,
        "remaining_steps": obs.remaining_steps,
        "step_count": obs.step_count,
        "last_action_valid": obs.last_action_valid,
        "last_action_message": obs.last_action_message,
        "history": history,
    }


def first_missing_action(obs) -> Optional[Dict[str, Any]]:
    history_types = [
        h.action_type.value if hasattr(h.action_type, "value") else str(h.action_type)
        for h in obs.action_history
    ]
    issue_id = obs.issue.issue_id

    if "read_issue" not in history_types:
        return {"type": "read_issue", "issue_id": issue_id}
    if "read_repo_rules" not in history_types:
        return {"type": "read_repo_rules"}
    if "read_label_definitions" not in history_types:
        return {"type": "read_label_definitions"}
    if "read_assignee_pool" not in history_types:
        return {"type": "read_assignee_pool"}
    if "read_milestones" not in history_types:
        return {"type": "read_milestones"}

    if obs.task.goal_type.value == "duplicate_resolution" and "search_similar_issues" not in history_types:
        query = f"{obs.issue.title}\n{obs.issue.body}"
        return {"type": "search_similar_issues", "query": query}

    return None


def heuristic_fallback(obs) -> Dict[str, Any]:
    text = f"{obs.issue.title}\n{obs.issue.body}".lower()
    labels = obs.repo_rules.labels or {}

    if obs.task.goal_type.value == "needs_info" and obs.pending_missing_fields:
        return {"type": "request_info", "fields": obs.pending_missing_fields[:4]}

    if obs.task.goal_type.value == "duplicate_resolution" and obs.candidate_duplicates:
        best = max(obs.candidate_duplicates, key=lambda c: c.similarity_score)
        return {"type": "mark_duplicate", "issue_id": best.issue_id}

    if any(word in text for word in ["crash", "error", "fail", "failure", "bug", "broken", "500"]):
        type_label = next(iter(labels.get("type", ["type:bug"])), "type:bug")
        return {"type": "add_label", "label": type_label}

    if any(word in text for word in ["feature", "enhancement", "add support", "request"]):
        type_label = next(iter(labels.get("type", ["type:feature"])), "type:feature")
        return {"type": "add_label", "label": type_label}

    if obs.repo_rules.assignee_pool:
        return {"type": "assign_user", "username": obs.repo_rules.assignee_pool[0]}

    return {"type": "noop"}


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()

    # Strip code fences if the model emits them anyway.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Last resort: try literal_eval on a JSON-like dict.
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    raise ValueError(f"Could not parse JSON action from model output: {text[:300]}")


def choose_action(client: OpenAI, obs) -> Dict[str, Any]:
    preflight = first_missing_action(obs)
    if preflight is not None:
        return preflight

    payload = summarize_observation(obs)

    user_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Observation JSON:\n{json.dumps(payload, indent=2)}\n\n"
        f"Return one action JSON object only."
    )

    response = client.responses.create(
        model=MODEL_NAME,
        input=user_prompt,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    text = getattr(response, "output_text", "") or ""
    if not text.strip():
        return heuristic_fallback(obs)

    try:
        action = extract_json_object(text)
    except Exception:
        return heuristic_fallback(obs)

    if not isinstance(action, dict) or "type" not in action:
        return heuristic_fallback(obs)

    return action


def run_episode(client: OpenAI, env: GitHubIssueTriageEnvironment, task_id: str) -> Tuple[float, Dict[str, Any]]:
    obs = env.reset(task_id=task_id)
    done = False
    step = 0

    while not done and step < MAX_STEPS:
        action = choose_action(client, obs)
        result = env.step(action)
        obs = result.observation
        done = result.done
        step += 1

    final_state = env.state()
    grader_result = grade_episode(final_state)
    return grader_result.score, grader_result.model_dump()


def main() -> int:
    client = make_client()
    env = load_environment()

    episodes = getattr(env, "_episodes_source", [])
    if not episodes:
        raise RuntimeError(
            f"No episodes loaded. Set DATA_DIR/TRIAGE_DATA_DIR to a folder containing "
            f"repo_rules.json, tasks.json, and issues.json."
        )

    task_ids = [ep.task.task_id for ep in episodes]

    results: List[Dict[str, Any]] = []
    scores: List[float] = []

    for task_id in task_ids:
        score, grader = run_episode(client, env, task_id)
        scores.append(score)
        results.append(
            {
                "task_id": task_id,
                "score": score,
                "grader": grader,
            }
        )
        print(json.dumps(results[-1], indent=2))

    average = sum(scores) / len(scores) if scores else 0.0
    summary = {
        "tasks": len(scores),
        "average_score": average,
        "scores": scores,
    }

    print(json.dumps(summary, indent=2))

    # Optional machine-readable output for CI or the validator.
    out_path = os.getenv("INFERENCE_RESULTS_PATH")
    if out_path:
        Path(out_path).write_text(json.dumps({"results": results, "summary": summary}, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())