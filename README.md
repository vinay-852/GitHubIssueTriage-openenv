---
title: GitHubIssueTriage Environment Server
emoji: "🧭"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# 🧭 GitHubIssueTriage Environment

GitHubIssueTriage is an OpenEnv-based environment designed to train and evaluate agents on real-world GitHub issue triage workflows.

Instead of a simple echo setup, this environment simulates practical issue triaging scenarios where agents must analyze, decide, and act based on repository rules.

---

## 🚀 Overview

Each episode represents a realistic triage workflow where the agent is expected to:

- Understand issue details and repository policies  
- Apply labels, assign users, set priorities, and milestones  
- Request missing information when needed  
- Detect duplicates and decide when to close or reopen issues  
- Optimize decisions using reward-based feedback  

---

## 🧩 Environment Structure

Each episode includes:

- **repo_rules** → Defines triage policies (labels, workflows, templates)  
- **issue** → Current issue snapshot  
- **task** → Allowed actions, constraints, and objective  
- **hidden_target** → Expected correct outcome for evaluation  

### ⏹ Episode ends when:
- Maximum steps are reached, or  
- Hidden target is fully satisfied  

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Run the Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### 📌 Available Endpoints

- `/web` → Interactive OpenEnv UI  
- `/docs` → API documentation (FastAPI/OpenAPI)  
- `/health` → Health check  
- `/ws` → WebSocket endpoint  

By default, `/web` loads a demo episode.  
To use custom data, set:

```bash
GITHUB_ISSUE_TRIAGE_DATA_DIR=<your_data_folder>
```

---

### 🐳 Optional: Run with Docker

```bash
docker build -t github-issue-triage-env -f server/Dockerfile .
docker run --rm -p 8000:8000 github-issue-triage-env
```

Or with the default root Dockerfile:

```bash
docker build -t github-issue-triage-env .
docker run --rm -p 8000:8000 github-issue-triage-env
```

---

## ⚙️ Configuration

Create a `.env` file in the root directory:

```dotenv
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
HF_TOKEN=YOUR_HF_OR_PROVIDER_TOKEN
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=openai/gpt-oss-120b
TEMPERATURE=0.0
MAX_OUTPUT_TOKENS=200
```

`inference.py` reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` (and also supports `OPENAI_API_KEY`).

---

## 🤖 Running Inference

Run all tasks using:

```bash
python inference.py
```

### 📊 Output includes:
- Episode  
- Task ID  
- Difficulty  
- Score  
- Steps  
- Structured logs on stdout with strict `[START]`, `[STEP]`, and `[END]` entries

---

### 🔧 CLI Options

- `--repo-rules` → Custom repo rules  
- `--tasks-file` → Run specific tasks  
- `--issue-file` → Single issue fallback  
- `--issue-url` → Load GitHub issue directly  
- `--live-github` → Fetch live data  
- `--task-id` → Override generated task id for single-issue mode  
- `--max-steps` → Override step limit for single-issue mode  

---

### 🧪 Examples

Run all tasks:
```bash
python inference.py
```

Run a single issue:
```bash
python inference.py --issue-url https://github.com/OWNER/REPO/issues/123
```

Use a custom tasks file:
```bash
python inference.py --tasks-file data/tasks.json
```

---

## 📈 Baseline Scores

| Task ID              | Difficulty | Score | Steps |
|----------------------|------------|--------|-------|
| triage_easy_api_p1   | easy       | 0.938  | 6     |
| needs_info_sso       | medium     | 0.975  | 5     |
| duplicate_ui_crash   | hard       | 0.972  | 6     |

Use these values to compare performance across models or configurations.

---

## 📂 Data Loading Methods

### A. Load from JSON Bundle

```python
from server.loader import load_episode_bundle
from server.GitHubIssueTriage_environment import GitHubIssueTriageEnvironment

episodes = load_episode_bundle(
    repo_rules_path="data/repo_rules.json",
    tasks_path="data/tasks.json",
    issues_path="data/issues.json",
    live_github=False,
)

env = GitHubIssueTriageEnvironment(episodes=episodes)
obs = env.reset()
```

---

### B. Load from Folder

```python
from server.GitHubIssueTriage_environment import GitHubIssueTriageEnvironment

env = GitHubIssueTriageEnvironment(data_dir="data")
obs = env.reset()
```

---

### C. Load Single Issue

```python
from server.loader import load_episode_from_source
from server.GitHubIssueTriage_environment import GitHubIssueTriageEnvironment

state = load_episode_from_source(
    repo_rules_path="data/url_repo_rules.json",
    issue_source="https://github.com/owner/repo/issues/123",
    live_github=True,
    max_steps=10,
)

env = GitHubIssueTriageEnvironment(episodes=[state], live_github=True)
obs = env.reset()
```

---

## 🎯 Action Space

### 🔍 Read Actions
- read_issue, read_repo_rules, read_label_definitions, search_similar_issues

### ⚙️ Triage Actions
- add_label, remove_label, assign_user, set_priority, set_milestone

### 💬 Communication Actions
- comment, request_info, provide_info

### 🔄 Lifecycle Actions
- mark_duplicate, close_issue, reopen_issue, noop

All actions are validated against rules and constraints.

---

## 🧠 Reward System

The reward system is dense and deterministic, based on:

- Label accuracy  
- Severity/component match  
- Assignment and prioritization  
- Missing information handling  
- Duplicate detection  
- Closure correctness  
- Comment quality  

### ❌ Penalties
- Invalid actions  
- Incorrect closures  

---

### 📏 Evaluation Example

```python
from server.grader import grade_episode

result = grade_episode(env._state)
print(result.score, result.notes)
```

---

## ☁️ Deployment (Hugging Face Spaces)

```bash
openenv push --repo-id <your-namespace>/GitHubIssueTriageManager
```

This builds and deploys the environment automatically.

---

## 📁 Project Structure

```text
.
├── client.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
└── server/
    ├── actions.py
    ├── app.py
    ├── GitHubIssueTriage_environment.py
    ├── grader.py
    ├── loader.py
    ├── observation.py
    ├── reward.py
    ├── termination.py
    ├── transitions.py
    └── Dockerfile
```
