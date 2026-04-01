from __future__ import annotations

import os
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..model import Observation, StatePayload, StepResult
from .environment import GitHubIssueTriageEnvironment


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: Optional[str] = None


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Dict[str, Any]


class ReloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Optional[str] = None
    repo_rules_source: Optional[str] = None
    tasks_source: Optional[str] = None
    issues_source: Optional[str] = None
    strict_mode: Optional[bool] = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    ready: bool
    init_error: Optional[str] = None
    loaded_episodes: int = 0
    current_episode_id: Optional[str] = None
    current_task_id: Optional[str] = None


class TaskListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: List[str] = Field(default_factory=list)


class RootResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    endpoints: List[str] = Field(default_factory=list)


class _ServerState:
    def __init__(self) -> None:
        self.lock = RLock()
        self.env: Optional[GitHubIssueTriageEnvironment] = None
        self.init_error: Optional[str] = None


SERVER_STATE = _ServerState()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_data_dir() -> Optional[str]:
    """
    Prefer a bundled data directory if present.
    """
    package_data = Path(__file__).resolve().parent.parent / "data"
    if package_data.exists():
        return str(package_data)
    return None


def _build_environment(
    *,
    data_dir: Optional[str] = None,
    repo_rules_source: Optional[str] = None,
    tasks_source: Optional[str] = None,
    issues_source: Optional[str] = None,
    strict_mode: Optional[bool] = None,
) -> GitHubIssueTriageEnvironment:
    """
    Build one environment instance from either:
    - data_dir
    - repo_rules_source + tasks_source + issues_source
    - or bundled default data directory
    """
    if strict_mode is None:
        strict_mode = _env_bool("STRICT_MODE", True)

    if data_dir is None:
        data_dir = os.getenv("DATA_DIR") or os.getenv("TRIAGE_DATA_DIR") or _default_data_dir()

    if repo_rules_source is None:
        repo_rules_source = os.getenv("REPO_RULES_SOURCE")

    if tasks_source is None:
        tasks_source = os.getenv("TASKS_SOURCE")

    if issues_source is None:
        issues_source = os.getenv("ISSUES_SOURCE")

    if data_dir:
        return GitHubIssueTriageEnvironment(
            data_dir=data_dir,
            strict_mode=strict_mode,
        )

    any_sources = any([repo_rules_source, tasks_source, issues_source])
    all_sources = all([repo_rules_source, tasks_source, issues_source])

    if any_sources and not all_sources:
        raise ValueError(
            "Provide either data_dir OR all of repo_rules_source, tasks_source, and issues_source."
        )

    if all_sources:
        return GitHubIssueTriageEnvironment(
            repo_rules_source=repo_rules_source,
            tasks_source=tasks_source,
            issues_source=issues_source,
            strict_mode=strict_mode,
        )

    # Empty environment is allowed to boot the server, but reset() will fail until data is configured.
    return GitHubIssueTriageEnvironment(
        episodes=[],
        strict_mode=strict_mode,
    )


def _load_on_startup() -> None:
    with SERVER_STATE.lock:
        try:
            SERVER_STATE.env = _build_environment()
            SERVER_STATE.init_error = None
        except Exception as exc:
            SERVER_STATE.env = None
            SERVER_STATE.init_error = str(exc)


def _get_env() -> GitHubIssueTriageEnvironment:
    with SERVER_STATE.lock:
        if SERVER_STATE.env is None:
            if SERVER_STATE.init_error:
                raise HTTPException(status_code=500, detail=SERVER_STATE.init_error)
            try:
                SERVER_STATE.env = _build_environment()
            except Exception as exc:
                SERVER_STATE.init_error = str(exc)
                raise HTTPException(status_code=500, detail=str(exc))
        return SERVER_STATE.env


def _health_payload() -> HealthResponse:
    env = SERVER_STATE.env
    if env is None:
        return HealthResponse(
            ok=False,
            ready=False,
            init_error=SERVER_STATE.init_error,
            loaded_episodes=0,
            current_episode_id=None,
            current_task_id=None,
        )

    current_episode_id: Optional[str] = None
    current_task_id: Optional[str] = None
    try:
        current_state = env.state()
        current_episode_id = current_state.episode_id
        current_task_id = current_state.task.task_id
    except Exception:
        current_episode_id = None
        current_task_id = None

    loaded_episodes = len(getattr(env, "_episodes_source", []))
    ready = loaded_episodes > 0

    return HealthResponse(
        ok=True,
        ready=ready,
        init_error=SERVER_STATE.init_error,
        loaded_episodes=loaded_episodes,
        current_episode_id=current_episode_id,
        current_task_id=current_task_id,
    )


app = FastAPI(
    title="GitHub Issue Triage Manager",
    version="0.1.0",
    description="OpenEnv-compatible backend for GitHub issue triage tasks.",
)


@app.on_event("startup")
def startup_event() -> None:
    _load_on_startup()


@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    return RootResponse(
        name="GitHub Issue Triage Manager",
        version="0.1.0",
        endpoints=[
            "GET /",
            "GET /health",
            "GET /tasks",
            "POST /reload",
            "POST /reset",
            "POST /step",
            "GET /state",
        ],
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return _health_payload()


@app.get("/tasks", response_model=TaskListResponse)
def list_tasks() -> TaskListResponse:
    env = _get_env()
    task_ids = [ep.task.task_id for ep in getattr(env, "_episodes_source", [])]
    return TaskListResponse(tasks=task_ids)


@app.post("/reload", response_model=HealthResponse)
def reload_environment(request: Optional[ReloadRequest] = Body(default=None)) -> HealthResponse:
    with SERVER_STATE.lock:
        try:
            SERVER_STATE.env = _build_environment(
                data_dir=request.data_dir if request else None,
                repo_rules_source=request.repo_rules_source if request else None,
                tasks_source=request.tasks_source if request else None,
                issues_source=request.issues_source if request else None,
                strict_mode=request.strict_mode if request else None,
            )
            SERVER_STATE.init_error = None
        except Exception as exc:
            SERVER_STATE.env = None
            SERVER_STATE.init_error = str(exc)
            raise HTTPException(status_code=400, detail=str(exc))
    return _health_payload()


@app.post("/reset", response_model=Observation)
def reset(request: Optional[ResetRequest] = Body(default=None)) -> Observation:
    env = _get_env()
    with SERVER_STATE.lock:
        try:
            return env.reset(task_id=request.task_id if request else None)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=StepResult)
def step(request: StepRequest) -> StepResult:
    env = _get_env()
    with SERVER_STATE.lock:
        try:
            return env.step(request.action)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state", response_model=StatePayload)
def state() -> StatePayload:
    env = _get_env()
    with SERVER_STATE.lock:
        try:
            return env.snapshot()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)