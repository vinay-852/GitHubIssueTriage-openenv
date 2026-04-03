# envs/your_env/server/environment.py
from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

from openenv.core.env_server import Environment

from ..models import (
    Action,
    Difficulty,
    HistoryEntry,
    Observation,
    ResetResult,
    State,
    StatePayload,
    StepInfo,
    StepResult,
)
from .actions import ParsedActionResult, parse_and_validate_action
from .loader import load_episode_bundle, load_episode_bundle_from_paths
from .observation import build_observation
from .reward import compute_reward
from .termination import is_episode_done
from .transitions import apply_action_to_state


class GitHubIssueTriageEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        *,
        episodes: Optional[list[State]] = None,
        repo_rules_source: Optional[Union[str, Path]] = None,
        tasks_source: Optional[Union[str, Path]] = None,
        issues_source: Optional[Union[str, Path]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        strict_mode: bool = True,
        live_github: bool = False,
    ) -> None:
        self.strict_mode = strict_mode
        self.live_github = live_github

        self._episodes_source: list[State] = episodes or []
        self._episode_index: int = -1
        self._state: Optional[State] = None
        self._seed: Optional[int] = None
        self._global_sequence: List[int] = []
        self._global_position: int = -1
        self._difficulty_sequences: Dict[Difficulty, List[int]] = {}
        self._difficulty_positions: Dict[Difficulty, int] = {}

        if not self._episodes_source:
            if data_dir is not None:
                self._episodes_source = load_episode_bundle_from_paths(
                    data_dir,
                    live_github=live_github,
                )
            elif repo_rules_source and tasks_source and issues_source:
                self._episodes_source = load_episode_bundle(
                    repo_rules_path=repo_rules_source,
                    tasks_path=tasks_source,
                    issues_path=issues_source,
                    live_github=live_github,
                )

        self._initialize_sequences()

    def _initialize_sequences(self, seed: Optional[int] = None) -> None:
        if not self._episodes_source:
            self._global_sequence = []
            self._global_position = -1
            self._difficulty_sequences = {}
            self._difficulty_positions = {}
            return

        rng = random.Random(seed) if seed is not None else None

        indices = list(range(len(self._episodes_source)))
        if rng is not None:
            rng.shuffle(indices)

        self._global_sequence = indices
        self._global_position = -1

        self._difficulty_sequences = {}
        self._difficulty_positions = {}
        for difficulty in Difficulty:
            seq = [
                idx
                for idx in indices
                if self._episodes_source[idx].task.difficulty == difficulty
            ]
            if rng is not None:
                rng.shuffle(seq)
            self._difficulty_sequences[difficulty] = seq
            self._difficulty_positions[difficulty] = -1

    def _set_seed(self, seed: Optional[int]) -> None:
        self._seed = seed
        self._initialize_sequences(seed)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _record_history(
        self,
        state: State,
        *,
        action: Action,
        outcome: str,
        success: bool,
    ) -> None:
        state.current_action_history.append(
            HistoryEntry(
                step_index=state.step_count,
                action_type=action.type,
                action_payload=action.model_dump(),
                outcome=outcome,
                success=success,
                timestamp=self._timestamp(),
            )
        )

    def _normalize_difficulty(
        self, difficulty: Optional[Union[str, Difficulty]]
    ) -> Optional[Difficulty]:
        if difficulty is None:
            return None
        if isinstance(difficulty, Difficulty):
            return difficulty
        try:
            return Difficulty(difficulty.strip().lower())
        except Exception as exc:
            raise KeyError(f"Unknown difficulty: {difficulty}") from exc

    def _next_index(self, *, difficulty: Optional[Difficulty] = None) -> int:
        if difficulty is None:
            if not self._global_sequence:
                raise RuntimeError("No episodes available.")
            self._global_position = (self._global_position + 1) % len(self._global_sequence)
            return self._global_sequence[self._global_position]

        seq = self._difficulty_sequences.get(difficulty, [])
        if not seq:
            raise KeyError(f"No episodes available for difficulty '{difficulty.value}'.")
        position = (self._difficulty_positions[difficulty] + 1) % len(seq)
        self._difficulty_positions[difficulty] = position
        return seq[position]

    def reset(
        self,
        task_id: Optional[str] = None,
        difficulty: Optional[Union[str, Difficulty]] = None,
        seed: Optional[int] = None,
    ) -> Observation:
        if not self._episodes_source:
            raise RuntimeError(
                "No episodes loaded. Pass episodes=..., data_dir=..., "
                "or repo_rules_source/tasks_source/issues_source."
            )

        if seed is not None:
            self._set_seed(seed)

        difficulty_enum = self._normalize_difficulty(difficulty)
        if task_id is None:
            if difficulty_enum is None:
                index = self._next_index()
            else:
                index = self._next_index(difficulty=difficulty_enum)
            self._episode_index = index
            base_state = self._episodes_source[index]
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

        return build_observation(self._state)

    def step(self, action: Action | dict) -> StepResult:
        state = self._require_state()

        if state.done:
            obs = build_observation(state)
            reward = compute_reward(state)
            reward_dump = reward.model_dump()
            reward_components = (
                reward_dump.pop("components", {})
                if isinstance(reward_dump.get("components"), dict)
                else {}
            )
            reward_breakdown = {
                key: float(value)
                for key, value in reward_dump.items()
                if isinstance(value, (int, float))
            }

            return StepResult(
                observation=obs,
                reward=reward,
                done=True,
                info=StepInfo(
                    action_valid=False,
                    action_effect="episode_already_done",
                    changed_fields=[],
                    reward_breakdown=reward_breakdown,
                    reward_components=reward_components,
                    grader_notes=["Episode already completed."],
                ),
            )

        validation: ParsedActionResult = parse_and_validate_action(
            action, state.task.allowed_actions
        )
        parsed_action = validation.action

        if not validation.valid:
            action_effect = validation.effect or "action_validation_failed"
            notes = validation.notes or ["Action failed validation."]

            self._record_history(
                state,
                action=parsed_action,
                outcome=action_effect,
                success=False,
            )

            state.step_count += 1
            if is_episode_done(state):
                state.done = True

            state.last_action_valid = False
            state.last_action_message = notes[0]

            reward = compute_reward(state)
            obs = build_observation(state)
            state.internal_score_cache = reward.total

            reward_dump = reward.model_dump()
            reward_components = (
                reward_dump.pop("components", {})
                if isinstance(reward_dump.get("components"), dict)
                else {}
            )
            reward_breakdown = {
                key: float(value)
                for key, value in reward_dump.items()
                if isinstance(value, (int, float))
            }

            info = StepInfo(
                action_valid=False,
                action_effect=action_effect,
                changed_fields=[],
                reward_breakdown=reward_breakdown,
                reward_components=reward_components,
                grader_notes=notes,
            )

            return StepResult(
                observation=obs,
                reward=reward,
                done=state.done,
                info=info,
            )

        transition = apply_action_to_state(state, parsed_action)

        state.step_count += 1
        if is_episode_done(state):
            state.done = True

        reward = compute_reward(state)
        obs = build_observation(state)

        transition_notes = list(getattr(transition, "notes", []))
        transition_effect = str(getattr(transition, "action_effect", ""))

        state.internal_score_cache = reward.total
        state.last_action_valid = bool(getattr(transition, "action_valid", True))
        state.last_action_message = (
            transition_notes[0] if transition_notes else transition_effect
        )

        reward_dump = reward.model_dump()
        reward_components = (
            reward_dump.pop("components", {})
            if isinstance(reward_dump.get("components"), dict)
            else {}
        )
        reward_breakdown = {
            key: float(value)
            for key, value in reward_dump.items()
            if isinstance(value, (int, float))
        }

        info = StepInfo(
            action_valid=bool(getattr(transition, "action_valid", True)),
            action_effect=transition_effect,
            changed_fields=list(getattr(transition, "changed_fields", [])),
            reward_breakdown=reward_breakdown,
            reward_components=reward_components,
            grader_notes=transition_notes,
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=state.done,
            info=info,
        )

    @property
    def state(self) -> State:
        return self._require_state().model_copy(deep=True)

    def snapshot(self) -> StatePayload:
        return StatePayload(state=self.state)

    def reset_result(
        self,
        task_id: Optional[str] = None,
        difficulty: Optional[Union[str, Difficulty]] = None,
        seed: Optional[int] = None,
    ) -> ResetResult:
        obs = self.reset(task_id=task_id, difficulty=difficulty, seed=seed)
        return ResetResult(observation=obs, state=self.state)

    def _require_state(self) -> State:
        if self._state is None:
            raise RuntimeError("Environment has not been reset yet.")
        return self._state