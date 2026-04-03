#!/usr/bin/env python3
"""Quick test to verify auto-generation of hidden target from issue."""

from envs.GitHubIssueTriageManager.server.loader import _generate_hidden_target_from_issue
from envs.GitHubIssueTriageManager.models import IssueSnapshot, IssueStatus, Priority, Severity


def test_auto_generate_hidden_target():
    """Test that hidden target is auto-generated from issue metadata."""
    
    issue = IssueSnapshot(
        issue_id="test_issue_001",
        repo_id="test-repo",
        title="Test Issue",
        body="Test body",
        author="test_user",
        created_at="2026-04-01T00:00:00Z",
        status=IssueStatus.OPEN,
        labels=["type:enhancement"],
        assignees=["alice"],
        milestone="v2.0",
        priority=Priority.P2,
        severity=Severity.MEDIUM,
        component="voice-ui",
        linked_duplicates=["duplicate_123"],
    )
    
    hidden_target = _generate_hidden_target_from_issue(issue)
    
    print("✅ Generated HiddenGradingTarget:")
    print(f"   gold_labels: {hidden_target.gold_labels}")
    print(f"   gold_priority: {hidden_target.gold_priority}")
    print(f"   gold_severity: {hidden_target.gold_severity}")
    print(f"   gold_component: {hidden_target.gold_component}")
    print(f"   gold_assignee: {hidden_target.gold_assignee}")
    print(f"   gold_milestone: {hidden_target.gold_milestone}")
    print(f"   gold_duplicate_issue_id: {hidden_target.gold_duplicate_issue_id}")
    
    # Verify the values
    assert "type:enhancement" in hidden_target.gold_labels, "Custom labels should be included"
    assert "priority:p2" in hidden_target.gold_labels, "Priority should be converted to label"
    assert "severity:medium" in hidden_target.gold_labels, "Severity should be converted to label"
    assert "component:voice-ui" in hidden_target.gold_labels, "Component should be converted to label"
    assert hidden_target.gold_priority == Priority.P2, "Priority should match issue priority"
    assert hidden_target.gold_severity == Severity.MEDIUM, "Severity should match issue severity"
    assert hidden_target.gold_component == "voice-ui", "Component should match issue component"
    assert hidden_target.gold_assignee == "alice", "Assignee should be the first one"
    assert hidden_target.gold_milestone == "v2.0", "Milestone should match"
    assert hidden_target.gold_duplicate_issue_id == "duplicate_123", "Duplicate ID should match"
    
    print("\n✅ All assertions passed!")


if __name__ == "__main__":
    test_auto_generate_hidden_target()
