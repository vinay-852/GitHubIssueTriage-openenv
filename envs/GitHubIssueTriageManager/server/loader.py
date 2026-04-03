from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.request import Request, urlopen

from ..models import (
    ActionType,
    Difficulty,
    DuplicateCandidate,
    GoalType,
    HiddenGradingTarget,
    IssueComment,
    IssueSnapshot,
    IssueStatus,
    IssueTriageState,
    Priority,
    RepoRules,
    Severity,
    TaskSpec,
    TimelineEvent,
    build_initial_state,
)

JsonLike = Dict[str, Any]

_GITHUB_ISSUE_WEB_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/issues/(?P<number>\d+)(?:/.*)?$"
)

_GITHUB_BLOB_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/blob/(?P<branch>[^/]+)/(?P<path>.+)$"
)


def _is_url(value: Union[str, Path]) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def _headers() -> Dict[str, str]:
    headers = {
        "User-Agent": "openenv-github-issue-triage-loader/1.0",
        "Accept": "application/vnd.github+json, application/json",
    }

    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _load_text_source(source: Union[str, Path]) -> str:
    if _is_url(source):
        req = Request(str(source), headers=_headers())
        with urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8")
    with Path(source).open("r", encoding="utf-8") as f:
        return f.read()


def _load_json_source(source: Union[str, Path]) -> Any:
    return json.loads(_load_text_source(source))


def _unwrap_payload(data: Any, key: str) -> List[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if key in data and isinstance(data[key], list):
            return data[key]
        if key in data and isinstance(data[key], dict):
            return [data[key]]
    raise ValueError(f"Unsupported JSON shape. Expected a list or a wrapper with key '{key}'.")


def _normalize_repo_rules_payload(data: Any) -> JsonLike:
    if isinstance(data, dict) and "repo_rules" in data and isinstance(data["repo_rules"], dict):
        return data["repo_rules"]
    if isinstance(data, dict):
        return data
    raise ValueError("repo_rules source must be a JSON object.")


def _convert_blob_url_to_raw(url: str) -> Optional[str]:
    m = _GITHUB_BLOB_RE.match(url)
    if not m:
        return None
    owner = m.group("owner")
    repo = m.group("repo")
    branch = m.group("branch")
    path = m.group("path")
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"


def _github_issue_api_url_from_web_url(url: str) -> Optional[str]:
    m = _GITHUB_ISSUE_WEB_RE.match(url)
    if not m:
        return None
    owner = m.group("owner")
    repo = m.group("repo")
    number = m.group("number")
    return f"https://api.github.com/repos/{owner}/{repo}/issues/{number}"


def _fetch_json(url: str) -> Any:
    req = Request(url, headers=_headers())
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load_json_maybe_github(source: Union[str, Path]) -> Any:
    """
    Accepts:
      - local JSON file path
      - raw GitHub JSON URL
      - github.com blob URL
      - any direct JSON URL
    """
    if not _is_url(source):
        return _load_json_source(source)

    url = str(source)
    raw_blob = _convert_blob_url_to_raw(url)
    if raw_blob is not None:
        url = raw_blob

    return _fetch_json(url)


def _parse_issue_comments(raw_comments: Any) -> List[IssueComment]:
    comments: List[IssueComment] = []
    if not isinstance(raw_comments, list):
        return comments

    for item in raw_comments:
        if not isinstance(item, dict):
            continue

        comments.append(
            IssueComment(
                comment_id=str(item.get("comment_id") or item.get("id") or f"c_{len(comments)}"),
                author=str(
                    item.get("author")
                    or (item.get("user") or {}).get("login")
                    or item.get("user_login")
                    or "unknown"
                ),
                body=str(item.get("body") or ""),
                created_at=str(item.get("created_at") or item.get("createdAt") or ""),
                edited_at=item.get("edited_at") or item.get("updated_at"),
                internal=bool(item.get("internal", False)),
            )
        )
    return comments


def _parse_timeline_events(raw_events: Any) -> List[TimelineEvent]:
    events: List[TimelineEvent] = []
    if not isinstance(raw_events, list):
        return events

    for item in raw_events:
        if not isinstance(item, dict):
            continue

        payload = item.get("payload")
        if not isinstance(payload, dict):
            payload = {}

        events.append(
            TimelineEvent(
                event_id=str(item.get("event_id") or item.get("id") or f"t_{len(events)}"),
                event_type=str(item.get("event_type") or item.get("type") or "event"),
                actor=str(
                    item.get("actor")
                    or (item.get("user") or {}).get("login")
                    or item.get("user_login")
                    or "unknown"
                ),
                created_at=str(item.get("created_at") or item.get("createdAt") or ""),
                payload={str(k): str(v) for k, v in payload.items()},
            )
        )
    return events


def _issue_status(value: Any) -> IssueStatus:
    raw = str(value or "open").lower()
    if raw == "closed":
        return IssueStatus.CLOSED
    return IssueStatus.OPEN


def _priority(value: Any) -> Optional[Priority]:
    if value is None:
        return None
    try:
        return Priority(str(value).lower())
    except Exception:
        return None


def _severity(value: Any) -> Optional[Severity]:
    if value is None:
        return None
    try:
        return Severity(str(value).lower())
    except Exception:
        return None


def _normalize_issue_snapshot(data: JsonLike) -> IssueSnapshot:
    """
    Accepts either:
      - your internal IssueSnapshot shape
      - GitHub REST issue payload
      - a small custom JSON issue object
    """
    issue_url = data.get("issue_url") or data.get("html_url") or data.get("url")

    labels_raw = data.get("labels", [])
    labels: List[str] = []
    if isinstance(labels_raw, list):
        for item in labels_raw:
            if isinstance(item, str):
                labels.append(item)
            elif isinstance(item, dict):
                labels.append(str(item.get("name") or item.get("label") or ""))
    labels = [x for x in labels if x]

    assignees_raw = data.get("assignees", [])
    assignees: List[str] = []
    if isinstance(assignees_raw, list):
        for item in assignees_raw:
            if isinstance(item, str):
                assignees.append(item)
            elif isinstance(item, dict):
                assignees.append(str(item.get("login") or item.get("username") or ""))
    assignees = [x for x in assignees if x]

    comments = _parse_issue_comments(data.get("comments", []))
    timeline = _parse_timeline_events(data.get("timeline", []))

    linked_duplicates_raw = data.get("linked_duplicates", [])
    linked_duplicates = [str(x) for x in linked_duplicates_raw] if isinstance(linked_duplicates_raw, list) else []

    milestone_value = data.get("milestone")
    if isinstance(milestone_value, dict):
        milestone_value = milestone_value.get("title") or milestone_value.get("name")

    repo_value = data.get("repo_id") or data.get("repository_id") or ""
    if not repo_value:
        repository = data.get("repository")
        if isinstance(repository, dict):
            repo_value = repository.get("full_name") or repository.get("name") or ""

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    return IssueSnapshot(
        issue_id=str(data.get("issue_id") or data.get("number") or data.get("id")),
        repo_id=str(repo_value),
        issue_url=str(issue_url) if issue_url else None,
        title=str(data.get("title") or ""),
        body=str(data.get("body") or ""),
        author=str(
            data.get("author")
            or (data.get("user") or {}).get("login")
            or data.get("user_login")
            or "unknown"
        ),
        created_at=str(data.get("created_at") or data.get("createdAt") or ""),
        updated_at=data.get("updated_at") or data.get("updatedAt"),
        status=_issue_status(data.get("status") or data.get("state")),
        labels=labels,
        assignees=assignees,
        milestone=str(milestone_value) if milestone_value else None,
        priority=_priority(data.get("priority")),
        severity=_severity(data.get("severity")),
        component=(str(data.get("component")) if data.get("component") is not None else None),
        comments=comments,
        timeline=timeline,
        linked_duplicates=linked_duplicates,
        is_locked=bool(data.get("is_locked", False)),
        metadata={str(k): str(v) for k, v in metadata.items()},
    )


def _fetch_github_issue(issue_url: str) -> JsonLike:
    api_url = _github_issue_api_url_from_web_url(issue_url)
    if api_url is None:
        raise ValueError(f"Not a supported GitHub issue URL: {issue_url}")

    issue_payload = _fetch_json(api_url)

    comments_url = issue_payload.get("comments_url")
    comments: List[Any] = []
    if comments_url:
        try:
            comments = _fetch_json(comments_url)
        except Exception:
            comments = []

    normalized: JsonLike = dict(issue_payload)
    normalized["issue_url"] = issue_url
    normalized["comments"] = comments if isinstance(comments, list) else []
    normalized.setdefault(
        "repo_id",
        issue_payload.get("repository_url")
        or (issue_payload.get("repository") or {}).get("full_name")
        or "",
    )
    normalized.setdefault("issue_id", issue_payload.get("number") or issue_payload.get("id"))
    normalized.setdefault("author", (issue_payload.get("user") or {}).get("login", "unknown"))
    normalized.setdefault("status", issue_payload.get("state", "open"))
    normalized.setdefault("labels", issue_payload.get("labels", []))
    normalized.setdefault("assignees", issue_payload.get("assignees", []))
    normalized.setdefault("milestone", issue_payload.get("milestone"))
    normalized.setdefault("body", issue_payload.get("body", ""))
    normalized.setdefault("title", issue_payload.get("title", ""))
    normalized.setdefault("created_at", issue_payload.get("created_at", ""))
    normalized.setdefault("updated_at", issue_payload.get("updated_at"))
    return normalized


def _load_issue_item(item: Any, *, live_github: bool = False) -> IssueSnapshot:
    if isinstance(item, IssueSnapshot):
        return item.model_copy(deep=True)

    if isinstance(item, str):
        if _is_url(item):
            if live_github and _GITHUB_ISSUE_WEB_RE.match(item):
                return _normalize_issue_snapshot(_fetch_github_issue(item))

            data = _load_json_maybe_github(item)
            if isinstance(data, dict):
                return _normalize_issue_snapshot(data)

            raise ValueError(f"Issue URL did not resolve to a JSON object: {item}")

        raise ValueError(f"Unsupported string issue source: {item}")

    if isinstance(item, dict):
        issue_url = item.get("issue_url") or item.get("url")
        if live_github and isinstance(issue_url, str) and _GITHUB_ISSUE_WEB_RE.match(issue_url):
            return _normalize_issue_snapshot(_fetch_github_issue(issue_url))

        return _normalize_issue_snapshot(item)

    raise ValueError(f"Unsupported issue source item: {type(item).__name__}")


def load_repo_rules(repo_rules_path: Union[str, Path]) -> RepoRules:
    raw = _load_json_maybe_github(repo_rules_path)
    payload = _normalize_repo_rules_payload(raw)
    if not isinstance(payload, dict):
        raise ValueError("repo_rules must be a JSON object.")
    return RepoRules.model_validate(payload)


def load_tasks(tasks_path: Union[str, Path]) -> List[TaskSpec]:
    raw = _load_json_maybe_github(tasks_path)
    task_items = _unwrap_payload(raw, "tasks")

    task_field_names = set(TaskSpec.model_fields.keys())
    tasks: List[TaskSpec] = []

    for item in task_items:
        if not isinstance(item, dict):
            continue
        task_data = {k: v for k, v in item.items() if k in task_field_names}
        tasks.append(TaskSpec.model_validate(task_data))

    return tasks


def load_issues(issues_path: Union[str, Path], *, live_github: bool = False) -> List[IssueSnapshot]:
    raw = _load_json_maybe_github(issues_path)

    if isinstance(raw, list):
        return [_load_issue_item(item, live_github=live_github) for item in raw]

    if isinstance(raw, dict) and "issues" in raw:
        issues_raw = raw["issues"]
        if isinstance(issues_raw, list):
            return [_load_issue_item(item, live_github=live_github) for item in issues_raw]
        if isinstance(issues_raw, dict):
            return [_load_issue_item(issues_raw, live_github=live_github)]

    if isinstance(raw, dict):
        return [_load_issue_item(raw, live_github=live_github)]

    raise ValueError("issues source must be a list, an object with key 'issues', or a single issue object.")


def _build_issue_index(issues: Sequence[IssueSnapshot]) -> Dict[str, IssueSnapshot]:
    index: Dict[str, IssueSnapshot] = {}
    for issue in issues:
        index[issue.issue_id] = issue
    return index


def _parse_hidden_target(raw_task: dict) -> Optional[HiddenGradingTarget]:
    hidden = raw_task.get("hidden_target")
    if not hidden:
        return None
    if isinstance(hidden, HiddenGradingTarget):
        return hidden.model_copy(deep=True)
    if isinstance(hidden, dict):
        return HiddenGradingTarget.model_validate(hidden)
    raise ValueError("hidden_target must be a dict or HiddenGradingTarget.")


def _parse_candidate_duplicates(raw_task: dict) -> List[DuplicateCandidate]:
    raw_candidates = raw_task.get("candidate_duplicates") or []
    if not isinstance(raw_candidates, list):
        return []
    candidates: List[DuplicateCandidate] = []
    for item in raw_candidates:
        if isinstance(item, DuplicateCandidate):
            candidates.append(item.model_copy(deep=True))
        elif isinstance(item, dict):
            candidates.append(DuplicateCandidate.model_validate(item))
    return candidates


def _generate_hidden_target_from_issue(issue: IssueSnapshot) -> HiddenGradingTarget:
    """
    Auto-generate a HiddenGradingTarget from issue metadata and comments.
    
    This extracts:
      - gold_labels: from issue.labels and inferred from priority/severity/component
      - gold_priority: from issue.priority or extracted from comments
      - gold_severity: from issue.severity
      - gold_component: from issue.component
      - gold_assignee: from first assignee if available
    """
    gold_labels: List[str] = []
    
    # Extract explicit labels from the issue
    if issue.labels:
        gold_labels.extend(issue.labels)
    
    # Infer labels from scalar fields
    if issue.priority:
        gold_labels.append(f"priority:{issue.priority.value}")
    
    if issue.severity:
        gold_labels.append(f"severity:{issue.severity.value}")
    
    if issue.component:
        gold_labels.append(f"component:{issue.component}")
    
    # Extract priority (can be overridden by comments)
    gold_priority = issue.priority
    
    # Try to extract priority from comments if not already set
    if not gold_priority and issue.comments:
        for comment in issue.comments:
            # Look for priority mentions in comment body
            comment_lower = comment.body.lower()
            for priority in Priority:
                if priority.value in comment_lower:
                    gold_priority = priority
                    break
            if gold_priority:
                break
    
    # Extract first assignee if available
    gold_assignee = issue.assignees[0] if issue.assignees else None
    
    return HiddenGradingTarget(
        gold_labels=gold_labels,
        gold_assignee=gold_assignee,
        gold_priority=gold_priority,
        gold_milestone=issue.milestone,
        gold_severity=issue.severity,
        gold_component=issue.component,
        gold_duplicate_issue_id=issue.linked_duplicates[0] if issue.linked_duplicates else None,
        gold_close_reason=None,
        required_missing_fields=[],
        expected_requests=[],
        expected_comment_keywords=[],
        expected_response_style=None,
    )


def load_episode_bundle(
    *,
    repo_rules_path: Union[str, Path],
    tasks_path: Union[str, Path],
    issues_path: Union[str, Path],
    live_github: bool = False,
) -> List[IssueTriageState]:
    """
    Main loader used by the environment.

    Supports:
      - local JSON files
      - GitHub raw URLs
      - github.com blob URLs
      - single GitHub issue URLs inside issues.json or issue entries
    """
    repo_rules = load_repo_rules(repo_rules_path)
    tasks_raw = _load_json_maybe_github(tasks_path)
    issues = load_issues(issues_path, live_github=live_github)
    issue_index = _build_issue_index(issues)

    task_items = _unwrap_payload(tasks_raw, "tasks")

    episodes: List[IssueTriageState] = []
    task_field_names = set(TaskSpec.model_fields.keys())

    for raw_task in task_items:
        if not isinstance(raw_task, dict):
            continue

        task_data = {k: v for k, v in raw_task.items() if k in task_field_names}
        task = TaskSpec.model_validate(task_data)

        if task.issue_id not in issue_index:
            raise ValueError(
                f"Issue {task.issue_id!r} referenced by task {task.task_id!r} was not found in issues source."
            )

        issue = issue_index[task.issue_id].model_copy(deep=True)

        episode_id = str(raw_task.get("episode_id") or f"ep_{task.task_id}")
        hidden_target = _parse_hidden_target(raw_task)
        candidate_duplicates = _parse_candidate_duplicates(raw_task)

        state = build_initial_state(
            episode_id=episode_id,
            task=task,
            repo_rules=repo_rules,
            issue=issue,
            candidate_duplicates=candidate_duplicates,
            hidden_target=hidden_target,
        )
        episodes.append(state)

    return episodes


def load_episode_bundle_from_paths(
    data_dir: Union[str, Path],
    *,
    live_github: bool = False,
) -> List[IssueTriageState]:
    """
    Convenience helper when your data is stored in a folder like:
      data/
        repo_rules.json
        tasks.json
        issues.json
    """
    base = Path(data_dir)
    repo_rules_path = base / "repo_rules.json"
    tasks_path = base / "tasks.json"
    issues_path = base / "issues.json"

    missing = [str(p) for p in [repo_rules_path, tasks_path, issues_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")

    return load_episode_bundle(
        repo_rules_path=repo_rules_path,
        tasks_path=tasks_path,
        issues_path=issues_path,
        live_github=live_github,
    )


def load_single_episode(
    *,
    repo_rules_path: Union[str, Path],
    task: dict,
    issue: Union[dict, str],
    candidate_duplicates: Optional[List[dict]] = None,
    live_github: bool = False,
) -> IssueTriageState:
    """
    Helper for tests, ad-hoc episodes, or GitHub-URL-backed issue data.
    """
    repo_rules = load_repo_rules(repo_rules_path)
    task_field_names = set(TaskSpec.model_fields.keys())
    task_data = {k: v for k, v in task.items() if k in task_field_names}
    task_obj = TaskSpec.model_validate(task_data)

    issue_obj = _load_issue_item(issue, live_github=live_github)

    dup_objs = [DuplicateCandidate.model_validate(x) for x in (candidate_duplicates or [])]
    hidden_target = _parse_hidden_target(task)

    return build_initial_state(
        episode_id=str(task.get("episode_id") or f"ep_{task_obj.task_id}"),
        task=task_obj,
        repo_rules=repo_rules,
        issue=issue_obj,
        candidate_duplicates=dup_objs,
        hidden_target=hidden_target,
    )


def load_episode_from_source(
    *,
    repo_rules_path: Union[str, Path],
    issue_source: Union[str, Path, Dict[str, Any]],
    live_github: bool = False,
    task_id: Optional[str] = None,
    max_steps: int = 10,
) -> IssueTriageState:
    """
    Build a single episode directly from repo rules + one issue source.
    This is the no-tasks.json path.
    """
    repo_rules = load_repo_rules(repo_rules_path)

    if isinstance(issue_source, dict):
        issue = _normalize_issue_snapshot(issue_source)
    else:
        issue = _load_issue_item(issue_source, live_github=live_github)

    generated_task_id = task_id or f"triage_{issue.repo_id.replace('/', '_')}_{issue.issue_id}"

    task = TaskSpec(
        task_id=generated_task_id,
        difficulty=Difficulty.EASY,
        goal_type=GoalType.TRIAGE_ONLY,
        repo_id=issue.repo_id,
        issue_id=issue.issue_id,
        max_steps=max_steps,
        success_criteria=[],
        allowed_actions=[
            ActionType.READ_ISSUE,
            ActionType.READ_REPO_RULES,
            ActionType.READ_LABEL_DEFINITIONS,
            ActionType.READ_TEAM_ROUTING,
            ActionType.READ_ASSIGNEE_POOL,
            ActionType.READ_MILESTONES,
            ActionType.SEARCH_SIMILAR_ISSUES,
            ActionType.ADD_LABEL,
            ActionType.REMOVE_LABEL,
            ActionType.ASSIGN_USER,
            ActionType.SET_PRIORITY,
            ActionType.SET_MILESTONE,
            ActionType.COMMENT,
            ActionType.REQUEST_INFO,
            ActionType.PROVIDE_INFO,
            ActionType.MARK_DUPLICATE,
            ActionType.CLOSE_ISSUE,
            ActionType.REOPEN_ISSUE,
            ActionType.NOOP,
        ],
        hidden_grading_flags={},
        repo_rules_url=None,
    )

    # Auto-generate hidden target from issue if not explicitly provided
    hidden_target = _generate_hidden_target_from_issue(issue)

    return build_initial_state(
        episode_id=f"ep_{generated_task_id}",
        task=task,
        repo_rules=repo_rules,
        issue=issue,
        candidate_duplicates=[],
        hidden_target=hidden_target,
    )