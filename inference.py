from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from agent import IssueTriageAgent
from server.GitHubIssueTriage_environment import GitHubIssueTriageEnvironment
from server.grader import grade_episode
from server.loader import load_episode_bundle, load_episode_from_source

DATA_DIR = Path("data")
DEFAULT_REPO_RULES = DATA_DIR / "repo_rules.json"
DEFAULT_ISSUES = DATA_DIR / "issues.json"
DEFAULT_TASKS = DATA_DIR / "tasks.json"

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = OPENAI_API_KEY or HF_TOKEN

BENCHMARK = "GitHubIssueTriage"
SUCCESS_SCORE_THRESHOLD = 0.80
# Keep a visible margin away from 0 and 1 to survive downstream rounding.
SCORE_EPSILON = 1e-3


def _emit(tag: str, payload: Dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, separators=(',', ':'), ensure_ascii=True)}", flush=True)


def log_start(task: str, env: str, model: str) -> None:
    _emit("START", {"task": task, "env": env, "model": model})


def log_step(step: int, action: Dict[str, Any], reward: float, done: bool, error: Optional[str]) -> None:
    _emit(
        "STEP",
        {
            "step": int(step),
            "action": action,
            "reward": float(_strict_open01(reward)),
            "done": bool(done),
            "error": error,
        },
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    _emit(
        "END",
        {
            "success": bool(success),
            "steps": int(steps),
            "score": float(_strict_open01(score)),
            "rewards": [float(_strict_open01(r)) for r in rewards],
        },
    )


def _strict_open01(value: float, epsilon: float = SCORE_EPSILON) -> float:
    try:
        bounded = max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        bounded = 0.5
    if not math.isfinite(bounded):
        bounded = 0.5

    try:
        eps = max(1e-6, min(0.49, float(epsilon)))
    except (TypeError, ValueError):
        eps = SCORE_EPSILON
    if not math.isfinite(eps):
        eps = SCORE_EPSILON

    if bounded <= 0.0:
        return eps
    if bounded >= 1.0:
        return 1.0 - eps
    return bounded


def _load_episodes(args: argparse.Namespace):
    repo_rules_path = Path(args.repo_rules).resolve()
    if not repo_rules_path.exists():
        raise FileNotFoundError(f"Repo rules file not found: {repo_rules_path}")

    issue_path = Path(args.issue_file).resolve()
    tasks_path = Path(args.tasks_file).resolve()

    if tasks_path.exists():
        return load_episode_bundle(
            repo_rules_path=str(repo_rules_path),
            tasks_path=str(tasks_path),
            issues_path=str(issue_path),
            live_github=args.live_github,
        )

    if args.issue_url:
        episode = load_episode_from_source(
            repo_rules_path=str(repo_rules_path),
            issue_source=args.issue_url,
            live_github=args.live_github,
            task_id=args.task_id,
            max_steps=args.max_steps,
        )
        return [episode]

    if not issue_path.exists():
        raise FileNotFoundError(f"Issue file not found: {issue_path}")

    return load_episode_bundle(
        repo_rules_path=str(repo_rules_path),
        tasks_path=None,
        issues_path=str(issue_path),
        live_github=args.live_github,
    )


def run_episode(env: GitHubIssueTriageEnvironment, agent: IssueTriageAgent, task_id: str, max_steps: int) -> Dict[str, Any]:
    result = env.reset(task_id=task_id)

    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    for step in range(1, max_steps + 1):
        if result.done:
            break

        action = agent.next_action(result.model_dump())

        try:
            step_result = env.step(action)
            reward = _strict_open01(float(step_result.reward.total))
            done = bool(step_result.done)
            error = None

            rewards.append(reward)
            steps_taken = step
            result = step_result.observation

            log_step(step=step, action=action, reward=reward, done=done, error=error)

            if done:
                break

        except Exception as exc:  # pragma: no cover
            error = f"{exc.__class__.__name__}: {exc}"
            log_step(step=step, action=action, reward=SCORE_EPSILON, done=True, error=error)
            steps_taken = step
            break

    grade = grade_episode(env.state)
    score = _strict_open01(float(grade.score))
    if not (0.0 < score < 1.0):
        raise RuntimeError(f"Task score out of strict range (0,1): {score!r}")
    success = score >= SUCCESS_SCORE_THRESHOLD

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "difficulty": env.state.task.difficulty.value,
        "score": score,
        "steps": steps_taken,
        "success": success,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible baseline inference for GitHubIssueTriage.")
    parser.add_argument("--repo-rules", default=str(DEFAULT_REPO_RULES))
    parser.add_argument("--tasks-file", default=str(DEFAULT_TASKS))
    parser.add_argument("--issue-file", default=str(DEFAULT_ISSUES))
    parser.add_argument("--issue-url", default=None)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--live-github", action="store_true")
    args = parser.parse_args()

    episodes = _load_episodes(args)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
    agent = IssueTriageAgent(
        client=client,
        model_name=MODEL_NAME,
        api_base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    env = GitHubIssueTriageEnvironment(
        episodes=episodes,
        strict_mode=True,
        live_github=args.live_github,
    )

    for ep in episodes:
        task_id = ep.task.task_id
        run_episode(env, agent, task_id=task_id, max_steps=ep.task.max_steps)


if __name__ == "__main__":
    main()
