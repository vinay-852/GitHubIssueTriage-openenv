from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List
from urllib.parse import urlparse

from openai import OpenAI

from agent import IssueTriageAgent
from models import ActionType
from server.GitHubIssueTriage_environment import GitHubIssueTriageEnvironment
from server.grader import grade_episode
from server.loader import load_episode_from_source, load_episode_bundle

try:
    from client import GithubissuetriageEnv
except ImportError:  # pragma: no cover
    GithubissuetriageEnv = None


DATA_DIR = Path("data")
DEFAULT_REPO_RULES = DATA_DIR / "repo_rules.json"
DEFAULT_ISSUES = DATA_DIR / "issues.json"
DEFAULT_TASKS = DATA_DIR / "tasks.json"

# Required inference-time model wiring.
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional for OpenEnv.from_docker_image() based workflows.
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def _structured_print(label: str, payload: Dict) -> None:
    print(f"[{label}] {json.dumps(payload, sort_keys=True)}", flush=True)


def _observation_snapshot(observation) -> Dict:
    return {
        "step_count": observation.step_count,
        "remaining_steps": observation.remaining_steps,
        "done": observation.done,
        "last_action_message": observation.last_action_message,
        "objective_summary": observation.objective_summary,
        "pending_missing_fields": getattr(observation, "pending_missing_fields", []),
        "provided_fields": getattr(observation, "provided_fields", {}),
    }


def _parse_github_issue_url(issue_url: str) -> tuple[str, str]:
    parsed = urlparse(issue_url)
    parts = [p for p in parsed.path.split("/") if p]

    if len(parts) < 4 or parts[2] != "issues":
        raise ValueError(
            f"Unsupported GitHub issue URL: {issue_url}. "
            "Expected https://github.com/OWNER/REPO/issues/123"
        )

    owner = parts[0]
    repo = parts[1]
    issue_id = parts[3]
    return f"{owner}/{repo}", issue_id


def _collect_provided_info(fields: List[str]) -> Dict[str, str]:
    print("\nThe environment requested more information.", file=sys.stderr, flush=True)
    print("Using default values for the fields.\n", file=sys.stderr, flush=True)

    # Default values for common fields
    defaults = {
        "steps_to_reproduce": "1. Open the application\n2. Navigate to the checkout page\n3. Attempt to complete a purchase\n4. Observe the 500 error",
        "expected_behavior": "The checkout should complete successfully without errors.",
        "actual_behavior": "A 500 internal server error occurs.",
        "environment": "Production environment, API version v2",
        "browser": "Chrome 120.0",
        "os": "macOS 14.0",
        "additional_context": "This affects enterprise customers with active sessions.",
    }

    values: Dict[str, str] = {}
    for field in fields:
        value = defaults.get(field, f"Default value for {field}")
        values[field] = value
        print(f"{field}: {value}", file=sys.stderr, flush=True)
    return values


def _build_provide_info_action(fields: List[str]) -> Dict[str, object]:
    return {
        "type": ActionType.PROVIDE_INFO.value,
        "fields": _collect_provided_info(fields),
    }


class EpisodeEnv:
    """Adapter that makes the remote sync client behave like the local env."""

    def __init__(self, sync_client):
        self._client = sync_client
        self._last_observation = None

    def __enter__(self):
        self._client.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._client.__exit__(exc_type, exc, tb)

    def reset(self, **kwargs):
        result = self._client.reset(**kwargs)
        self._last_observation = result.observation
        return self._last_observation

    def step(self, action, **kwargs):
        result = self._client.step(action, **kwargs)
        reward_value = float(result.reward or 0.0)
        self._last_observation = result.observation

        reward_proxy = SimpleNamespace(
            total=reward_value,
            model_dump=lambda: {"total": reward_value},
        )
        info_proxy = SimpleNamespace(
            action_valid=getattr(result.observation, "last_action_valid", True),
            action_effect=getattr(result.observation, "last_action_message", ""),
            grader_notes=[],
            changed_fields=[],
            reward_breakdown={"total": reward_value},
            reward_components={},
        )

        return SimpleNamespace(
            observation=result.observation,
            reward=reward_proxy,
            done=result.done,
            info=info_proxy,
        )

    @property
    def state(self):
        if self._last_observation is None:
            return SimpleNamespace()
        return self._last_observation


def run_episode(env, agent: IssueTriageAgent) -> Dict[str, float]:
    observation = env.reset()

    _structured_print(
        "START",
        {
            "task_id": observation.task.task_id,
            "episode_id": observation.episode_id,
            "difficulty": observation.task.difficulty.value,
            "max_steps": observation.task.max_steps,
            "objective_summary": observation.objective_summary,
        },
    )

    step_index = 0

    while True:
        action = agent.next_action(observation.model_dump())

        try:
            step_result = env.step(action)
        except RuntimeError as e:
            error_msg = str(e)
            print(f"[ERROR] Environment step failed: {error_msg}", file=sys.stderr, flush=True)
            print(
                f"[ERROR] Action was: {json.dumps(action, sort_keys=True)}",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"[ERROR] Observation state: "
                f"episode_id={observation.episode_id}, step={step_index}",
                file=sys.stderr,
                flush=True,
            )
            raise

        _structured_print(
            "STEP",
            {
                "task_id": observation.task.task_id,
                "step": step_index,
                "action": action,
                "reward": step_result.reward.total,
                "reward_breakdown": step_result.reward.model_dump(),
                "action_valid": step_result.info.action_valid,
                "action_effect": step_result.info.action_effect,
                "grader_notes": step_result.info.grader_notes,
                "observation": _observation_snapshot(step_result.observation),
            },
        )

        observation = step_result.observation
        step_index += 1

        if step_result.done:
            break

        if action.get("type") == ActionType.REQUEST_INFO.value:
            requested_fields = action.get("fields") or list(
                getattr(observation, "pending_missing_fields", [])
            )
            if requested_fields:
                provide_action = _build_provide_info_action(list(requested_fields))
                try:
                    provide_result = env.step(provide_action)
                except RuntimeError as e:
                    error_msg = str(e)
                    print(
                        f"[ERROR] Environment step failed on PROVIDE_INFO: {error_msg}",
                        file=sys.stderr,
                        flush=True,
                    )
                    print(
                        f"[ERROR] Action was: {json.dumps(provide_action, sort_keys=True)}",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise

                _structured_print(
                    "STEP",
                    {
                        "task_id": observation.task.task_id,
                        "step": step_index,
                        "action": provide_action,
                        "reward": provide_result.reward.total,
                        "reward_breakdown": provide_result.reward.model_dump(),
                        "action_valid": provide_result.info.action_valid,
                        "action_effect": provide_result.info.action_effect,
                        "grader_notes": provide_result.info.grader_notes,
                        "observation": _observation_snapshot(provide_result.observation),
                    },
                )

                observation = provide_result.observation
                step_index += 1

                if provide_result.done:
                    break

    final_state = env.state
    grade = grade_episode(final_state)

    result_payload = {
        "task_id": observation.task.task_id,
        "steps_taken": step_index,
        "score": grade.score,
        "matched_labels": grade.matched_labels,
        "matched_assignee": grade.matched_assignee,
        "matched_priority": grade.matched_priority,
        "matched_milestone": grade.matched_milestone,
        "duplicate_matched": grade.duplicate_matched,
        "missing_fields_requested": grade.missing_fields_requested,
        "closed_correctly": grade.closed_correctly,
        "comment_accepted": grade.comment_accepted,
        "notes": grade.notes,
    }
    _structured_print("END", result_payload)

    return {"score": grade.score, "steps": float(step_index)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GitHub issue triage using local data files or a GitHub issue URL."
    )
    parser.add_argument(
        "--repo-rules",
        default=str(DEFAULT_REPO_RULES),
        help=f"Path to repo_rules.json (default: {DEFAULT_REPO_RULES})",
    )
    parser.add_argument(
        "--tasks-file",
        default=str(DEFAULT_TASKS),
        help=f"Path to tasks.json (default: {DEFAULT_TASKS}). If provided, runs all tasks.",
    )
    parser.add_argument(
        "--issue-file",
        default=str(DEFAULT_ISSUES),
        help=f"Path to issues.json or single issue JSON file (default: {DEFAULT_ISSUES})",
    )
    parser.add_argument(
        "--issue-url",
        default=None,
        help="Optional GitHub issue URL. If provided, this overrides --issue-file.",
    )
    parser.add_argument(
        "--live-github",
        action="store_true",
        help="Fetch issue data directly from GitHub",
    )
    parser.add_argument(
        "--transport",
        choices=["local", "remote"],
        default="local",
        help="Run against the local environment or the remote sync client.",
    )
    parser.add_argument(
        "--base-url",
        default="https://vinay-pepakayala-githubissuetriagemanager.hf.space",
        help="Remote environment base URL",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum steps for the generated episode",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="Optional override for generated task id",
    )
    args = parser.parse_args()

    repo_rules_path = Path(args.repo_rules).resolve()
    if not repo_rules_path.exists():
        raise FileNotFoundError(f"Repo rules file not found: {repo_rules_path}")

    tasks_path = Path(args.tasks_file).resolve()
    if tasks_path.exists():
        # Load all episodes from tasks
        episodes = load_episode_bundle(
            repo_rules_path=str(repo_rules_path),
            tasks_path=str(tasks_path),
            issues_path=str(DEFAULT_ISSUES),
            live_github=args.live_github,
        )
    else:
        # Fallback to single episode
        if args.issue_url:
            issue_source: str = args.issue_url
        else:
            issue_path = Path(args.issue_file).resolve()
            if not issue_path.exists():
                raise FileNotFoundError(f"Issue file not found: {issue_path}")
            issue_source = str(issue_path)

        episode = load_episode_from_source(
            repo_rules_path=str(repo_rules_path),
            issue_source=issue_source,
            live_github=args.live_github,
            task_id=args.task_id,
            max_steps=args.max_steps,
        )
        episodes = [episode]

    model_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None
    agent = IssueTriageAgent(
        client=model_client,
        model_name=MODEL_NAME,
        api_base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    if args.transport == "local":
        env = GitHubIssueTriageEnvironment(
            episodes=episodes,
            strict_mode=True,
            live_github=args.live_github,
        )
        # Run all episodes
        results = []
        for i in range(len(episodes)):
            result = run_episode(env, agent)
            task = episodes[i].task
            results.append({
                'episode': i+1,
                'task_id': task.task_id,
                'difficulty': task.difficulty.value,
                'score': result['score'],
                'steps': result['steps']
            })
        
        # Print table after all runs
        print(
            f"Model: {agent.model_name}, Base URL: {agent.api_base_url}, Temp: {agent.temperature}, Max Tokens: {agent.max_tokens}",
            file=sys.stderr,
            flush=True,
        )
        print("Episode | Task ID              | Difficulty | Score  | Steps", file=sys.stderr, flush=True)
        print("--------|-----------------------|------------|--------|------", file=sys.stderr, flush=True)
        for r in results:
            print(
                f"{r['episode']:7} | {r['task_id'][:21]:21} | {r['difficulty']:10} | {r['score']:.3f} | {int(r['steps']):5}",
                file=sys.stderr,
                flush=True,
            )
        return

    if GithubissuetriageEnv is None:
        raise ImportError("Remote client is not available in this environment.")

    with GithubissuetriageEnv(
        base_url=args.base_url,
        episodes=[episode],
        strict_mode=True,
        live_github=args.live_github,
    ).sync() as client:
        env = EpisodeEnv(client)
        print(run_episode(env, agent), file=sys.stderr, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Keep the validator run alive and surface a machine-readable error.
        print(
            f"[FATAL] inference.py handled exception: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )