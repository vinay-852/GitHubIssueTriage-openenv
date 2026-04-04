# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Githubissuetriage Environment Client."""

import argparse
from contextlib import AbstractContextManager
from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core.sync_client import SyncEnvClient

try:
    from GitHubIssueTriage.models import Action, Observation
except ImportError:  # pragma: no cover
    from models import Action, Observation


class GithubissuetriageEnv(
    EnvClient[Action, Observation, State]
):
    """
    Client for the Githubissuetriage Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server over websocket
        >>> with GithubissuetriageEnv(base_url="http://localhost:8000").session() as session:
        ...     result = session.initial_result
        ...     print(result.observation.echoed_message)
        ...
        ...     result = session.step(Action(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = GithubissuetriageEnv.from_docker_image("GitHubIssueTriage-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(Action(message="Test"))
        ... finally:
        ...     client.close()
    """

    def __init__(
        self,
        base_url: str,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
        max_message_size_mb: float = 100.0,
        provider=None,
        mode: str | None = None,
        **kwargs,
    ) -> None:
        self.episodes = kwargs.pop("episodes", None)
        self.strict_mode = kwargs.pop("strict_mode", None)
        self.live_github = kwargs.pop("live_github", None)
        self.extra_env_kwargs = kwargs

        super().__init__(
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
            max_message_size_mb=max_message_size_mb,
            provider=provider,
            mode=mode,
        )

    class Session(AbstractContextManager["GithubissuetriageEnv.Session"]):
        """Context manager that opens a websocket connection and resets the env."""

        def __init__(
            self,
            client: "GithubissuetriageEnv",
            *,
            task_id: str | None = None,
            difficulty: str | None = None,
            seed: int | None = None,
        ) -> None:
            self._client = client
            self._task_id = task_id
            self._difficulty = difficulty
            self._seed = seed
            self._session = client.sync()
            self.initial_result = None

        def __enter__(self) -> "GithubissuetriageEnv.Session":
            self._session.__enter__()
            self.initial_result = self._session.reset(
                task_id=self._task_id,
                difficulty=self._difficulty,
                seed=self._seed,
            )
            return self

        def __exit__(self, exc_type, exc, tb) -> bool | None:
            return self._session.__exit__(exc_type, exc, tb)

        def reset(self, **kwargs):
            return self._session.reset(**kwargs)

        def step(self, action, **kwargs):
            return self._session.step(action, **kwargs)

        def close(self) -> None:
            self._session.close()

    def websocket_session(self) -> SyncEnvClient[Action, Observation, State]:
        """
        Return a synchronous websocket session wrapper for this environment client.

        This is the easiest way to use the environment from regular Python code:

            with GithubissuetriageEnv(base_url="http://localhost:8000").websocket_session() as client:
                result = client.reset()
        """
        return self.sync()

    def session(
        self,
        *,
        task_id: str | None = None,
        difficulty: str | None = None,
        seed: int | None = None,
    ) -> "GithubissuetriageEnv.Session":
        """Open a websocket session and reset the environment on entry."""
        return GithubissuetriageEnv.Session(
            self,
            task_id=task_id,
            difficulty=difficulty,
            seed=seed,
        )

    def _step_payload(self, action: Action | Dict) -> Dict:
        """
        Convert Action to JSON payload for step message.

        Args:
            action: Action instance or dict

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        if isinstance(action, dict):
            action_dict = action.copy()
        elif hasattr(action, "model_dump"):
            action_dict = action.model_dump(exclude_none=True, mode="json")
        else:
            raise TypeError(f"Unsupported action type: {type(action).__name__}")
        
        # Ensure 'type' field exists and is a string
        if "type" not in action_dict:
            raise ValueError(f"Action missing 'type' field: {action_dict}")
        
        # Convert ActionType enum to string if needed
        if hasattr(action_dict["type"], "value"):
            action_dict["type"] = action_dict["type"].value
        
        # Log the exact payload being sent
        import json
        print(f"[CLIENT_STEP_PAYLOAD] {json.dumps(action_dict)}", flush=True)
        
        return action_dict

    def _parse_result(self, payload: Dict) -> StepResult[Observation]:
        """
        Parse server response into StepResult[Observation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with Observation
        """
        try:
            print(f"[CLIENT_PARSE_RESULT] Received payload keys: {list(payload.keys())}", flush=True)
            
            # Extract observation data - server sends nested 'observation' object
            obs_data = payload.get("observation", payload)
            if not isinstance(obs_data, dict):
                obs_data = {}

            # Some servers wrap step data as {"observation": {"observation": ..., "info": ...}}
            if (
                "observation" in obs_data
                and isinstance(obs_data.get("observation"), dict)
                and "episode_id" not in obs_data
            ):
                obs_data = obs_data["observation"]

            raw_reward = payload.get("reward", obs_data.get("reward"))
            if isinstance(raw_reward, dict):
                reward_value = raw_reward.get("total")
            else:
                reward_value = raw_reward

            print(f"[CLIENT_PARSE_RESULT] Observation keys: {list(obs_data.keys())}", flush=True)

            # Build observation with all required fields
            observation = Observation.model_validate(
                {
                    **obs_data,
                    "reward": reward_value,
                    "done": payload.get("done", obs_data.get("done", False)),
                }
            )

            return StepResult(
                observation=observation,
                reward=reward_value,
                done=payload.get("done", obs_data.get("done", False)),
            )
        except Exception as e:
            # Log the error but provide helpful debugging info
            print(f"[DEBUG] _parse_result failed: {e}", flush=True)
            print(f"[DEBUG] Payload structure: {list(payload.keys())}", flush=True)
            if "observation" in payload:
                print(f"[DEBUG] Observation keys: {list(payload['observation'].keys())}", flush=True)
            raise

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


def main() -> None:
    """Open a websocket session and reset the environment from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Environment server URL (http:// or ws://).",
    )
    parser.add_argument("--task-id", default=None, help="Optional task or episode id.")
    parser.add_argument(
        "--difficulty",
        default=None,
        help="Optional difficulty filter (easy, medium, hard).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional reset seed.")
    args = parser.parse_args()

    session = GithubissuetriageEnv(base_url=args.base_url).session(
        task_id=args.task_id,
        difficulty=args.difficulty,
        seed=args.seed,
    )
    with session:
        print(session.initial_result.observation.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
