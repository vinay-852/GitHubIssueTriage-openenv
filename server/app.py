# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Githubissuetriage Environment.

This module creates an HTTP server that exposes the GithubissuetriageEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e


from pydantic import BaseModel, ConfigDict


class ActionPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str


try:
    from GitHubIssueTriage.models import Observation
    from GitHubIssueTriage.server.GitHubIssueTriage_environment import GitHubIssueTriageEnvironment
except ImportError:  # pragma: no cover
    from models import Observation
    from server.GitHubIssueTriage_environment import GitHubIssueTriageEnvironment



# Create the app with web interface and README integration
app = create_app(
    GitHubIssueTriageEnvironment,
    ActionPayload,
    Observation,
    env_name="GitHubIssueTriage",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


# Add custom error handler for validation errors
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import json
import traceback

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Custom handler to provide detailed validation error messages."""
    error_msg = str(exc)
    error_details = exc.errors()
    print(f"[VALIDATION_ERROR] {error_msg}", flush=True)
    print(f"[VALIDATION_DETAILS] {json.dumps(error_details, default=str)}", flush=True)
    return JSONResponse(
        status_code=422,
        content={
            "detail": error_details,
            "message": f"Validation error: {error_msg}",
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Custom handler to log all unhandled exceptions."""
    error_trace = traceback.format_exc()
    print(f"[UNHANDLED_ERROR] {str(exc)}", flush=True)
    print(f"[TRACEBACK] {error_trace}", flush=True)
    return JSONResponse(
        status_code=500,
        content={
            "message": str(exc),
            "type": type(exc).__name__,
        }
    )


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m GitHubIssueTriage.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn GitHubIssueTriage.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
