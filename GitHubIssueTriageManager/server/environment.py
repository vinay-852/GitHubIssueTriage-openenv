# envs/your_env/server/environment.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ..model import (
    Action,
    IssueTriageState,
    Observation,
    ResetResult,
    Reward,
    StatePayload,
    StepInfo,
    StepResult,
)
from .actions import parse_action
from .loader import load_episode_bundle, load_episode_bundle_from_paths
from .observation import build_observation
from .reward import compute_reward
from .termination import is_episode_done
from .transitions import apply_action_to_state


class GitHubIssueTriageEnvironment:
    """
    Thin orchestration environment.

    Heavy logic should live in helper modules:
    - loader.py
    - actions.py
    - transitions.py
    - observation.py
    - reward.py
    - termination.py
    """

    def __init__(
        self,
        *,
        episodes: Optional[list[IssueTriageState]] = None,
        repo_rules_source: Optional[Union[str, Path]] = None,
        tasks_source: Optional[Union[str, Path]] = None,
        issues_source: Optional[Union[str, Path]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        strict_mode: bool = True,
    ) -> None:
        self.strict_mode = strict_mode

        self._episodes_source: list[IssueTriageState] = episodes or []
        self._episode_index: int = -1
        self._state: Optional[IssueTriageState] = None
        self._last_reward_total: float = 0.0

        if not self._episodes_source:
            if data_dir is not None:
                self._episodes_source = load_episode_bundle_from_paths(data_dir)
            elif repo_rules_source and tasks_source and issues_source:
                self._episodes_source = load_episode_bundle(
                    repo_rules_path=repo_rules_source,
                    tasks_path=tasks_source,
                    issues_path=issues_source,
                )

    def reset(self, task_id: Optional[str] = None) -> Observation:
        if not self._episodes_source:
            raise RuntimeError(
                "No episodes loaded. Pass episodes=..., data_dir=..., or repo_rules_source/tasks_source/issues_source."
            )

        if task_id is None:
            self._episode_index = (self._episode_index + 1) % len(self._episodes_source)
            base_state = self._episodes_source[self._episode_index]
        else:
            match_idx = None
            for idx, ep in enumerate(self._episodes_source):
                if ep.task.task_id == task_id or ep.episode_id == task_id:
                    match_idx = idx
                    break
            if match_idx is None:
                raise KeyError(f"Unknown task_id or episode_id: {task_id}")
            self._episode_index = match_idx
            base_state = self._episodes_source[match_idx]

        self._state = base_state.model_copy(deep=True)
        self._state.step_count = 0
        self._state.done = False
        self._state.current_action_history = []
        self._state.pending_missing_fields = (
            list(self._state.hidden_target.required_missing_fields)
            if self._state.hidden_target
            else []
        )
        self._state.requested_fields = []
        self._state.public_notes = []
        self._state.last_action_valid = True
        self._state.last_action_message = ""
        self._state.internal_score_cache = None

        self._last_reward_total = 0.0
        return build_observation(self._state)

    def step(self, action: Action | dict) -> StepResult:
        state = self._require_state()

        if state.done:
            obs = build_observation(state)
            reward = compute_reward(state)
            return StepResult(
                observation=obs,
                reward=reward,
                done=True,
                info=StepInfo(
                    action_valid=False,
                    action_effect="episode_already_done",
                    changed_fields=[],
                    reward_breakdown=reward.model_dump(),
                    grader_notes=["Episode already completed."],
                ),
            )

        parsed_action = parse_action(action)

        transition = apply_action_to_state(state, parsed_action)
        state.step_count += 1

        if is_episode_done(state):
            state.done = True

        reward = compute_reward(state)
        obs = build_observation(state)

        state.internal_score_cache = reward.total
        state.last_action_valid = bool(getattr(transition, "action_valid", True))
        state.last_action_message = str(getattr(transition, "action_effect", ""))

        info = StepInfo(
            action_valid=bool(getattr(transition, "action_valid", True)),
            action_effect=str(getattr(transition, "action_effect", "")),
            changed_fields=list(getattr(transition, "changed_fields", [])),
            reward_breakdown=reward.model_dump(),
            grader_notes=list(getattr(transition, "notes", [])),
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=state.done,
            info=info,
        )

    def state(self) -> IssueTriageState:
        return self._require_state().model_copy(deep=True)

    def snapshot(self) -> StatePayload:
        return StatePayload(state=self.state())

    def reset_result(self, task_id: Optional[str] = None) -> ResetResult:
        obs = self.reset(task_id=task_id)
        return ResetResult(observation=obs, state=self.state())

    def _require_state(self) -> IssueTriageState:
        if self._state is None:
            raise RuntimeError("Environment has not been reset yet.")
        return self._state