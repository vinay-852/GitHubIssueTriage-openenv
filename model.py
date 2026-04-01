from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, Annotated

from pydantic import BaseModel, Field, ConfigDict


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class GoalType(str, Enum):
    TRIAGE_ONLY = "triage_only"
    NEEDS_INFO = "needs_info"
    DUPLICATE_RESOLUTION = "duplicate_resolution"


class IssueStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


class LabelCategory(str, Enum):
    TYPE = "type"
    SEVERITY = "severity"
    COMPONENT = "component"
    STATUS = "status"
    PRIORITY = "priority"
    CUSTOM = "custom"


class Priority(str, Enum):
    P0 = "p0"
    P1 = "p1"
    P2 = "p2"
    P3 = "p3"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CloseReason(str, Enum):
    DUPLICATE = "duplicate"
    INVALID = "invalid"
    WONTFIX = "wontfix"
    RESOLVED = "resolved"
    STALE = "stale"
    NOT_ENOUGH_INFO = "not_enough_info"


class ActionType(str, Enum):
    READ_ISSUE = "read_issue"
    READ_REPO_RULES = "read_repo_rules"
    READ_LABEL_DEFINITIONS = "read_label_definitions"
    READ_TEAM_ROUTING = "read_team_routing"
    READ_ASSIGNEE_POOL = "read_assignee_pool"
    READ_MILESTONES = "read_milestones"
    SEARCH_SIMILAR_ISSUES = "search_similar_issues"
    ADD_LABEL = "add_label"
    REMOVE_LABEL = "remove_label"
    ASSIGN_USER = "assign_user"
    SET_PRIORITY = "set_priority"
    SET_MILESTONE = "set_milestone"
    COMMENT = "comment"
    REQUEST_INFO = "request_info"
    MARK_DUPLICATE = "mark_duplicate"
    CLOSE_ISSUE = "close_issue"
    REOPEN_ISSUE = "reopen_issue"
    NOOP = "noop"


# -----------------------------
# Core domain models
# -----------------------------

class LabelDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    category: LabelCategory
    description: str = ""
    mutually_exclusive_with: List[str] = Field(default_factory=list)
    implies: List[str] = Field(default_factory=list)
    required_before: List[str] = Field(default_factory=list)
    valid_issue_types: List[str] = Field(default_factory=list)


class TeamRoutingRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component: str
    issue_type: Optional[str] = None
    severity_threshold: Optional[Severity] = None
    assigned_team: str
    assignee_candidates: List[str] = Field(default_factory=list)
    default_milestone: Optional[str] = None
    escalation_path: Optional[str] = None


class RepoRules(BaseModel):
    """
    JSON-first policy document used by the environment and grader.
    This is the deterministic source of truth for triage behavior.
    """
    model_config = ConfigDict(extra="forbid")

    repo_id: str
    repo_name: str

    # Source tracking
    source_url: Optional[str] = None
    source_kind: Literal["json", "readme", "mixed"] = "json"
    version: str = "v1"
    strict_mode: bool = True

    # Canonical policy data
    labels: Dict[str, List[str]] = Field(default_factory=dict)
    severity_policy: Dict[str, str] = Field(default_factory=dict)
    priority_policy: Dict[str, str] = Field(default_factory=dict)
    routing_rules: Dict[str, List[str]] = Field(default_factory=dict)
    milestones: List[str] = Field(default_factory=list)
    missing_info: Dict[str, List[str]] = Field(default_factory=dict)
    duplicate_policy: Dict[str, str] = Field(default_factory=dict)
    closure_policy: List[str] = Field(default_factory=list)
    response_templates: Dict[str, str] = Field(default_factory=dict)

    # Optional richer structures for internal use
    label_definitions: List[LabelDefinition] = Field(default_factory=list)
    team_routing_rules: List[TeamRoutingRule] = Field(default_factory=list)

    # Convenience lookup data
    assignee_pool: List[str] = Field(default_factory=list)
    team_map: Dict[str, List[str]] = Field(default_factory=dict)
    required_fields_by_issue_type: Dict[str, List[str]] = Field(default_factory=dict)
    label_aliases: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, str] = Field(default_factory=dict)


class IssueComment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comment_id: str
    author: str
    body: str
    created_at: str
    edited_at: Optional[str] = None
    internal: bool = False


class TimelineEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: str
    actor: str
    created_at: str
    payload: Dict[str, str] = Field(default_factory=dict)


class IssueSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_id: str
    repo_id: str
    issue_url: Optional[str] = None
    title: str
    body: str
    author: str
    created_at: str
    updated_at: Optional[str] = None
    status: IssueStatus = IssueStatus.OPEN
    labels: List[str] = Field(default_factory=list)
    assignees: List[str] = Field(default_factory=list)
    milestone: Optional[str] = None
    priority: Optional[Priority] = None
    severity: Optional[Severity] = None
    component: Optional[str] = None
    comments: List[IssueComment] = Field(default_factory=list)
    timeline: List[TimelineEvent] = Field(default_factory=list)
    linked_duplicates: List[str] = Field(default_factory=list)
    is_locked: bool = False
    metadata: Dict[str, str] = Field(default_factory=dict)


class DuplicateCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_id: str
    title: str
    short_summary: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    labels: List[str] = Field(default_factory=list)
    status: IssueStatus = IssueStatus.OPEN
    reason: Optional[str] = None


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    difficulty: Difficulty
    goal_type: GoalType
    repo_id: str
    issue_id: str
    max_steps: int = Field(ge=1)
    success_criteria: List[str] = Field(default_factory=list)
    allowed_actions: List[ActionType] = Field(default_factory=list)
    hidden_grading_flags: Dict[str, bool] = Field(default_factory=dict)
    repo_rules_url: Optional[str] = None


class HistoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: int
    action_type: ActionType
    action_payload: Dict[str, Any] = Field(default_factory=dict)
    outcome: str = ""
    success: bool = True
    timestamp: Optional[str] = None


class HiddenGradingTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gold_labels: List[str] = Field(default_factory=list)
    gold_assignee: Optional[str] = None
    gold_priority: Optional[Priority] = None
    gold_milestone: Optional[str] = None
    gold_severity: Optional[Severity] = None
    gold_component: Optional[str] = None
    gold_duplicate_issue_id: Optional[str] = None
    gold_close_reason: Optional[CloseReason] = None
    required_missing_fields: List[str] = Field(default_factory=list)
    expected_requests: List[str] = Field(default_factory=list)
    expected_comment_keywords: List[str] = Field(default_factory=list)
    expected_response_style: Optional[str] = None


class IssueTriageState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    task: TaskSpec
    repo_rules: RepoRules
    issue: IssueSnapshot
    candidate_duplicates: List[DuplicateCandidate] = Field(default_factory=list)

    step_count: int = 0
    max_steps: int = 0
    done: bool = False

    current_action_history: List[HistoryEntry] = Field(default_factory=list)
    pending_missing_fields: List[str] = Field(default_factory=list)
    requested_fields: List[str] = Field(default_factory=list)
    public_notes: List[str] = Field(default_factory=list)

    hidden_target: Optional[HiddenGradingTarget] = None
    internal_score_cache: Optional[float] = None
    last_action_valid: bool = True
    last_action_message: str = ""

    metadata: Dict[str, str] = Field(default_factory=dict)


# -----------------------------
# Actions
# -----------------------------

class ReadIssueAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.READ_ISSUE] = ActionType.READ_ISSUE
    issue_id: str


class ReadRepoRulesAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.READ_REPO_RULES] = ActionType.READ_REPO_RULES


class ReadLabelDefinitionsAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.READ_LABEL_DEFINITIONS] = ActionType.READ_LABEL_DEFINITIONS


class ReadTeamRoutingAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.READ_TEAM_ROUTING] = ActionType.READ_TEAM_ROUTING


class ReadAssigneePoolAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.READ_ASSIGNEE_POOL] = ActionType.READ_ASSIGNEE_POOL


class ReadMilestonesAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.READ_MILESTONES] = ActionType.READ_MILESTONES


class SearchSimilarIssuesAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.SEARCH_SIMILAR_ISSUES] = ActionType.SEARCH_SIMILAR_ISSUES
    query: str = ""


class AddLabelAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.ADD_LABEL] = ActionType.ADD_LABEL
    label: str


class RemoveLabelAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.REMOVE_LABEL] = ActionType.REMOVE_LABEL
    label: str


class AssignUserAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.ASSIGN_USER] = ActionType.ASSIGN_USER
    username: str


class SetPriorityAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.SET_PRIORITY] = ActionType.SET_PRIORITY
    priority: Priority


class SetMilestoneAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.SET_MILESTONE] = ActionType.SET_MILESTONE
    milestone: str


class CommentAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.COMMENT] = ActionType.COMMENT
    text: str


class RequestInfoAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.REQUEST_INFO] = ActionType.REQUEST_INFO
    fields: List[str] = Field(default_factory=list)


class MarkDuplicateAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.MARK_DUPLICATE] = ActionType.MARK_DUPLICATE
    issue_id: str


class CloseIssueAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.CLOSE_ISSUE] = ActionType.CLOSE_ISSUE
    reason: CloseReason


class ReopenIssueAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.REOPEN_ISSUE] = ActionType.REOPEN_ISSUE
    reason: str = ""


class NoopAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.NOOP] = ActionType.NOOP


Action = Annotated[
    Union[
        ReadIssueAction,
        ReadRepoRulesAction,
        ReadLabelDefinitionsAction,
        ReadTeamRoutingAction,
        ReadAssigneePoolAction,
        ReadMilestonesAction,
        SearchSimilarIssuesAction,
        AddLabelAction,
        RemoveLabelAction,
        AssignUserAction,
        SetPriorityAction,
        SetMilestoneAction,
        CommentAction,
        RequestInfoAction,
        MarkDuplicateAction,
        CloseIssueAction,
        ReopenIssueAction,
        NoopAction,
    ],
    Field(discriminator="type"),
]


# -----------------------------
# Observation / Reward / Results
# -----------------------------

class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    task: TaskSpec
    issue: IssueSnapshot
    repo_rules: RepoRules
    available_labels: List[str] = Field(default_factory=list)
    available_assignees: List[str] = Field(default_factory=list)
    available_milestones: List[str] = Field(default_factory=list)
    candidate_duplicates: List[DuplicateCandidate] = Field(default_factory=list)
    action_history: List[HistoryEntry] = Field(default_factory=list)
    pending_missing_fields: List[str] = Field(default_factory=list)
    remaining_steps: int = 0
    step_count: int = 0
    done: bool = False
    last_action_valid: bool = True
    last_action_message: str = ""


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: float = Field(ge=-1.0, le=1.0)
    type_score: float = Field(default=0.0, ge=0.0, le=1.0)
    severity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    component_score: float = Field(default=0.0, ge=0.0, le=1.0)
    assignee_score: float = Field(default=0.0, ge=0.0, le=1.0)
    priority_score: float = Field(default=0.0, ge=0.0, le=1.0)
    milestone_score: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_info_score: float = Field(default=0.0, ge=0.0, le=1.0)
    duplicate_score: float = Field(default=0.0, ge=0.0, le=1.0)
    closure_score: float = Field(default=0.0, ge=0.0, le=1.0)
    comment_score: float = Field(default=0.0, ge=0.0, le=1.0)
    invalid_action_penalty: float = Field(default=0.0, le=0.0)
    destructive_action_penalty: float = Field(default=0.0, le=0.0)


class StepInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_valid: bool = True
    action_effect: str = ""
    changed_fields: List[str] = Field(default_factory=list)
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
    grader_notes: List[str] = Field(default_factory=list)


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: Observation
    reward: Reward
    done: bool
    info: StepInfo


class GraderResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    matched_labels: List[str] = Field(default_factory=list)
    matched_assignee: bool = False
    matched_priority: bool = False
    matched_milestone: bool = False
    duplicate_matched: bool = False
    missing_fields_requested: bool = False
    closed_correctly: bool = False
    comment_accepted: bool = False
    notes: List[str] = Field(default_factory=list)


# -----------------------------
# Helper container models
# -----------------------------

class ResetResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: Observation
    state: IssueTriageState


class StatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: IssueTriageState


def build_initial_state(
    *,
    episode_id: str,
    task: TaskSpec,
    repo_rules: RepoRules,
    issue: IssueSnapshot,
    candidate_duplicates: Optional[List[DuplicateCandidate]] = None,
    hidden_target: Optional[HiddenGradingTarget] = None,
) -> IssueTriageState:
    candidate_duplicates = candidate_duplicates or []
    return IssueTriageState(
        episode_id=episode_id,
        task=task,
        repo_rules=repo_rules,
        issue=issue,
        candidate_duplicates=candidate_duplicates,
        step_count=0,
        max_steps=task.max_steps,
        done=False,
        current_action_history=[],
        pending_missing_fields=[],
        requested_fields=[],
        public_notes=[],
        hidden_target=hidden_target,
        internal_score_cache=None,
        last_action_valid=True,
        last_action_message="",
        metadata={},
    )