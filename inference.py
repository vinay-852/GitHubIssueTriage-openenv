from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

from agent import IssueTriageAgent
from envs.GitHubIssueTriageManager.server.environment import GitHubIssueTriageEnvironment
from envs.GitHubIssueTriageManager.server.grader import grade_episode


def _structured_print(label: str, payload: Dict) -> None:
    print(f"[{label}] {json.dumps(payload, sort_keys=True)}", flush=True)


def _load_task_ids(data_dir: Path) -> List[str]:
    tasks_path = data_dir / "tasks.json"
    if not tasks_path.exists():
        raise FileNotFoundError(f"Unable to locate tasks file at {tasks_path}.")
    data = json.loads(tasks_path.read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    return [task["task_id"] for task in tasks if "task_id" in task]


def _observation_snapshot(observation) -> Dict:
    return {
        "step_count": observation.step_count,
        "remaining_steps": observation.remaining_steps,
        "done": observation.done,
        "last_action_message": observation.last_action_message,
        "objective_summary": observation.objective_summary,
    }


def run_episode(
    env: GitHubIssueTriageEnvironment,
    agent: IssueTriageAgent,
    task_id: str,
) -> Dict[str, float]:
    observation = env.reset(task_id=task_id)
    _structured_print(
        "START",
        {
            "task_id": task_id,
            "episode_id": observation.episode_id,
            "difficulty": observation.task.difficulty.value,
            "max_steps": observation.task.max_steps,
            "objective_summary": observation.objective_summary,
        },
    )

    step_index = 0
    rewards: List[float] = []

    while True:
        action = agent.next_action(observation.model_dump())
        step_result = env.step(action)

        rewards.append(step_result.reward.total)
        _structured_print(
            "STEP",
            {
                "task_id": task_id,
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

    final_state = env.state()
    grade = grade_episode(final_state)
    result_payload = {
        "task_id": task_id,
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


def summarize_results(results: Dict[str, Dict[str, float]]) -> None:
    scores = [value["score"] for value in results.values()]
    _structured_print(
        "SUMMARY",
        {
            "by_task": {task_id: round(value["score"], 4) for task_id, value in results.items()},
            "mean_score": round(mean(scores), 4) if scores else 0.0,
            "total_tasks": len(results),
        },
    )


def main() -> None:
    data_dir = Path(os.getenv("DATA_DIR", "data")).resolve()
    env = GitHubIssueTriageEnvironment(
        data_dir=str(data_dir),
        strict_mode=True,
        live_github=False,
    )
    agent = IssueTriageAgent()

    task_ids = _load_task_ids(data_dir)
    episode_results: Dict[str, Dict[str, float]] = {}

    for task_id in task_ids:
        episode_results[task_id] = run_episode(env, agent, task_id)

    summarize_results(episode_results)


if __name__ == "__main__":
    main()
