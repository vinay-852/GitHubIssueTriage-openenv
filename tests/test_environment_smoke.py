from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from envs.GitHubIssueTriageManager.server.environment import GitHubIssueTriageEnvironment


def _build_env() -> GitHubIssueTriageEnvironment:
    return GitHubIssueTriageEnvironment(data_dir="data", strict_mode=True, live_github=False)


def test_disallowed_action_is_rejected() -> None:
    env = _build_env()
    observation = env.reset(task_id="triage_easy_api_p1")
    assert observation.task.task_id == "triage_easy_api_p1"

    result = env.step({"type": "close_issue", "reason": "duplicate"})

    assert result.info.action_valid is False
    assert result.info.action_effect.startswith("action_disallowed")
    assert "Action 'close_issue' is not allowed for this task." in result.info.grader_notes[0]
    assert result.observation.step_count == 1
    assert result.observation.last_action_valid is False


def test_request_info_updates_objectives() -> None:
    env = _build_env()
    result = env.reset(task_id="needs_info_sso")
    assert any("Request info fields" in item for item in result.objective_summary)

    step = env.step(
        {
            "type": "request_info",
            "fields": ["steps_to_reproduce", "expected_behavior", "actual_behavior", "environment"],
        }
    )

    assert step.info.action_valid is True
    assert not any("Request info fields" in item for item in step.observation.objective_summary)
    assert "Requested fields" in " ".join(step.info.grader_notes)


def test_seeded_reset_is_deterministic() -> None:
    env_a = _build_env()
    first_a = env_a.reset(seed=42)

    env_b = _build_env()
    first_b = env_b.reset(seed=42)

    assert first_a.task.task_id == first_b.task.task_id
