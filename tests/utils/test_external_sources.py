"""Unit tests for external source handlers.

Tests for GitHub, GitLab, JIRA, Linear, Asana, Notion, and other
external platform handlers with proper mocking and edge cases.
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
from freezegun import freeze_time

from tenets.utils.external_sources import (
    AsanaHandler,
    ExternalContent,
    ExternalSourceManager,
    GitHubHandler,
    GitLabHandler,
    JiraHandler,
    LinearHandler,
    NotionHandler,
)


@pytest.fixture
def mock_cache_manager():
    """Create a mock cache manager."""
    cache = MagicMock()
    cache.general.get.return_value = None
    cache.general.put.return_value = None
    return cache


@pytest.fixture
def mock_requests():
    """Mock requests library."""
    with patch("tenets.utils.external_sources.requests") as mock:
        mock.get = MagicMock()
        mock.post = MagicMock()
        yield mock


class TestExternalContent:
    """Test ExternalContent dataclass."""

    def test_external_content_creation(self):
        """Test creating external content."""
        content = ExternalContent(
            title="Test Issue",
            body="Issue description",
            metadata={"labels": ["bug"]},
            source_type="github",
            url="https://github.com/org/repo/issues/1",
            ttl_hours=24,
        )

        assert content.title == "Test Issue"
        assert content.body == "Issue description"
        assert content.metadata["labels"] == ["bug"]
        assert content.source_type == "github"
        assert content.url == "https://github.com/org/repo/issues/1"
        assert content.ttl_hours == 24
        assert content.cached_at is None

    def test_external_content_with_cached_at(self):
        """Test external content with cached timestamp."""
        now = datetime.now()
        content = ExternalContent(
            title="Test",
            body="Body",
            metadata={},
            source_type="test",
            url="https://example.com",
            cached_at=now,
        )

        assert content.cached_at == now


class TestExternalSourceHandler:
    """Test base ExternalSourceHandler class."""

    def test_load_api_tokens(self, monkeypatch):
        """Test loading API tokens from environment."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123")
        monkeypatch.setenv("GITLAB_TOKEN", "glpat-test456")
        monkeypatch.setenv("JIRA_TOKEN", "jira_test789")

        handler = GitHubHandler()  # Use concrete subclass

        assert handler._api_tokens["github"] == "ghp_test123"
        assert handler._api_tokens["gitlab"] == "glpat-test456"
        assert handler._api_tokens["jira"] == "jira_test789"

    def test_get_cached_content_no_cache(self, mock_cache_manager):
        """Test getting cached content when cache is disabled."""
        handler = GitHubHandler(cache_manager=None)
        result = handler.get_cached_content("https://example.com")

        assert result is None

    @freeze_time("2024-01-01 12:00:00")
    def test_get_cached_content_valid(self, mock_cache_manager):
        """Test getting valid cached content."""
        cached_data = {
            "title": "Cached Issue",
            "body": "Cached body",
            "metadata": {},
            "source_type": "github",
            "url": "https://github.com/org/repo/issues/1",
            "cached_at": "2024-01-01T11:00:00",
            "ttl_hours": 2,
        }
        mock_cache_manager.general.get.return_value = cached_data

        handler = GitHubHandler(cache_manager=mock_cache_manager)
        result = handler.get_cached_content("https://github.com/org/repo/issues/1")

        assert result is not None
        assert result.title == "Cached Issue"
        assert result.body == "Cached body"

    @freeze_time("2024-01-01 12:00:00")
    def test_get_cached_content_expired(self, mock_cache_manager):
        """Test getting expired cached content."""
        cached_data = {
            "title": "Old Issue",
            "body": "Old body",
            "metadata": {},
            "source_type": "github",
            "url": "https://github.com/org/repo/issues/1",
            "cached_at": "2024-01-01T08:00:00",  # 4 hours ago
            "ttl_hours": 2,  # Only valid for 2 hours
        }
        mock_cache_manager.general.get.return_value = cached_data

        handler = GitHubHandler(cache_manager=mock_cache_manager)
        result = handler.get_cached_content("https://github.com/org/repo/issues/1")

        assert result is None

    @freeze_time("2024-01-01 12:00:00")
    def test_cache_content(self, mock_cache_manager):
        """Test caching content."""
        content = ExternalContent(
            title="New Issue",
            body="New body",
            metadata={},
            source_type="github",
            url="https://github.com/org/repo/issues/1",
            ttl_hours=6,
        )

        handler = GitHubHandler(cache_manager=mock_cache_manager)
        handler.cache_content("https://github.com/org/repo/issues/1", content)

        mock_cache_manager.general.put.assert_called_once()
        call_args = mock_cache_manager.general.put.call_args

        assert call_args[0][0] == "external_content:https://github.com/org/repo/issues/1"
        assert call_args[1]["ttl"] == 6 * 3600  # 6 hours in seconds

        cached_data = call_args[0][1]
        assert cached_data["title"] == "New Issue"
        assert cached_data["body"] == "New body"
        assert "cached_at" in cached_data


class TestGitHubHandler:
    """Test GitHub handler."""

    def test_can_handle(self):
        """Test URL recognition."""
        handler = GitHubHandler()

        assert handler.can_handle("https://github.com/org/repo/issues/123")
        assert handler.can_handle("https://github.com/org/repo/pull/456")
        assert handler.can_handle("https://gist.github.com/user/abc123")
        assert not handler.can_handle("https://gitlab.com/org/repo")
        assert not handler.can_handle("https://example.com")

    def test_extract_identifier_issue(self):
        """Test extracting GitHub issue identifier."""
        handler = GitHubHandler()
        url = "https://github.com/microsoft/vscode/issues/12345"

        identifier, metadata = handler.extract_identifier(url)

        assert identifier == "microsoft/vscode#12345"
        assert metadata["owner"] == "microsoft"
        assert metadata["repo"] == "vscode"
        assert metadata["type"] == "issue"
        assert metadata["number"] == "12345"

    def test_extract_identifier_pull_request(self):
        """Test extracting GitHub PR identifier."""
        handler = GitHubHandler()
        url = "https://github.com/python/cpython/pull/98765"

        identifier, metadata = handler.extract_identifier(url)

        assert identifier == "python/cpython#98765"
        assert metadata["owner"] == "python"
        assert metadata["repo"] == "cpython"
        assert metadata["type"] == "pull_request"
        assert metadata["number"] == "98765"

    def test_extract_identifier_discussion(self):
        """Test extracting GitHub discussion identifier."""
        handler = GitHubHandler()
        url = "https://github.com/facebook/react/discussions/5678"

        identifier, metadata = handler.extract_identifier(url)

        assert identifier == "facebook/react/discussions/5678"
        assert metadata["owner"] == "facebook"
        assert metadata["repo"] == "react"
        assert metadata["type"] == "discussion"
        assert metadata["number"] == "5678"

    def test_extract_identifier_commit(self):
        """Test extracting GitHub commit identifier."""
        handler = GitHubHandler()
        url = "https://github.com/torvalds/linux/commit/abc123def456"

        identifier, metadata = handler.extract_identifier(url)

        assert identifier == "torvalds/linux@abc123d"
        assert metadata["owner"] == "torvalds"
        assert metadata["repo"] == "linux"
        assert metadata["type"] == "commit"
        assert metadata["sha"] == "abc123d"

    def test_extract_identifier_gist(self):
        """Test extracting GitHub gist identifier."""
        handler = GitHubHandler()
        url = "https://gist.github.com/user/1234567890abcdef"

        identifier, metadata = handler.extract_identifier(url)

        assert identifier == "gist:1234567890abcdef"
        assert metadata["type"] == "gist"
        assert metadata["gist_id"] == "1234567890abcdef"

    @patch("tenets.utils.external_sources.REQUESTS_AVAILABLE", True)
    def test_fetch_content_issue_success(self, mock_requests, monkeypatch):
        """Test successfully fetching GitHub issue content."""
        monkeypatch.setenv("GITHUB_TOKEN", "test_token")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "title": "Bug: Application crashes",
            "body": "The app crashes when clicking the button",
            "state": "open",
            "labels": [{"name": "bug"}, {"name": "critical"}],
            "assignees": [{"login": "johndoe"}],
            "created_at": "2024-01-01T10:00:00Z",
            "updated_at": "2024-01-01T11:00:00Z",
        }
        mock_requests.get.return_value = mock_response

        handler = GitHubHandler()
        metadata = {"owner": "org", "repo": "repo", "type": "issue", "number": "123"}

        content = handler.fetch_content("https://github.com/org/repo/issues/123", metadata)

        assert content is not None
        assert content.title == "Bug: Application crashes"
        assert content.body == "The app crashes when clicking the button"
        assert content.source_type == "github"
        assert content.metadata["state"] == "open"
        assert content.metadata["labels"] == ["bug", "critical"]
        assert content.metadata["assignees"] == ["johndoe"]
        assert content.ttl_hours == 6  # Open issues have shorter TTL

        # Verify API call
        mock_requests.get.assert_called_once_with(
            "https://api.github.com/repos/org/repo/issues/123",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Tenets-PromptParser/1.0",
                "Authorization": "token test_token",
            },
            timeout=10,
        )

    @patch("tenets.utils.external_sources.REQUESTS_AVAILABLE", True)
    def test_fetch_content_pull_request_success(self, mock_requests):
        """Test successfully fetching GitHub PR content."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "title": "Feature: Add dark mode",
            "body": "This PR adds dark mode support",
            "state": "closed",
            "merged": True,
            "draft": False,
            "labels": [{"name": "enhancement"}],
            "assignees": [],
        }
        mock_requests.get.return_value = mock_response

        handler = GitHubHandler()
        metadata = {"owner": "org", "repo": "repo", "type": "pull_request", "number": "456"}

        content = handler.fetch_content("https://github.com/org/repo/pull/456", metadata)

        assert content is not None
        assert content.title == "Feature: Add dark mode"
        assert content.metadata["merged"] is True
        assert content.metadata["draft"] is False
        assert content.ttl_hours == 24  # Closed items have longer TTL

    @patch("tenets.utils.external_sources.REQUESTS_AVAILABLE", True)
    def test_fetch_content_api_error(self, mock_requests):
        """Test handling API errors."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("Not found")
        mock_requests.get.return_value = mock_response

        handler = GitHubHandler()
        metadata = {"owner": "org", "repo": "repo", "type": "issue", "number": "999"}

        content = handler.fetch_content("https://github.com/org/repo/issues/999", metadata)

        assert content is None

    @patch("tenets.utils.external_sources.REQUESTS_AVAILABLE", False)
    def test_fetch_content_no_requests_library(self):
        """Test handling when requests library is not available."""
        handler = GitHubHandler()
        metadata = {"owner": "org", "repo": "repo", "type": "issue", "number": "123"}

        content = handler.fetch_content("https://github.com/org/repo/issues/123", metadata)

        assert content is None


class TestGitLabHandler:
    """Test GitLab handler."""

    def test_can_handle(self):
        """Test URL recognition."""
        handler = GitLabHandler()

        assert handler.can_handle("https://gitlab.com/org/repo/-/issues/123")
        assert handler.can_handle("https://gitlab.com/org/repo/-/merge_requests/456")
        assert handler.can_handle("https://gitlab.example.com/project")
        assert not handler.can_handle("https://github.com/org/repo")
        assert not handler.can_handle("https://example.com")

    def test_extract_identifier_issue(self):
        """Test extracting GitLab issue identifier."""
        handler = GitLabHandler()
        url = "https://gitlab.com/gitlab-org/gitlab/-/issues/12345"

        identifier, metadata = handler.extract_identifier(url)

        assert identifier == "gitlab-org/gitlab#12345"
        assert metadata["project"] == "gitlab-org/gitlab"
        assert metadata["type"] == "issue"
        assert metadata["iid"] == "12345"

    def test_extract_identifier_merge_request(self):
        """Test extracting GitLab MR identifier."""
        handler = GitLabHandler()
        url = "https://gitlab.com/gitlab-org/gitlab/-/merge_requests/98765"

        identifier, metadata = handler.extract_identifier(url)

        assert identifier == "gitlab-org/gitlab!98765"
        assert metadata["project"] == "gitlab-org/gitlab"
        assert metadata["type"] == "merge_request"
        assert metadata["iid"] == "98765"

    def test_extract_identifier_snippet(self):
        """Test extracting GitLab snippet identifier."""
        handler = GitLabHandler()
        url = "https://gitlab.com/mygroup/myproject/-/snippets/5678"

        identifier, metadata = handler.extract_identifier(url)

        assert identifier == "mygroup/myproject$$5678"
        assert metadata["project"] == "mygroup/myproject"
        assert metadata["type"] == "snippet"
        assert metadata["id"] == "5678"

    @patch("tenets.utils.external_sources.REQUESTS_AVAILABLE", True)
    def test_fetch_content_issue_success(self, mock_requests, monkeypatch):
        """Test successfully fetching GitLab issue content."""
        monkeypatch.setenv("GITLAB_TOKEN", "glpat-test123")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "title": "Feature Request: Add export",
            "description": "Need ability to export data",
            "state": "opened",
            "labels": ["feature", "backend"],
            "author": {"username": "alice"},
            "created_at": "2024-01-01T10:00:00Z",
            "updated_at": "2024-01-01T11:00:00Z",
        }
        mock_requests.get.return_value = mock_response

        handler = GitLabHandler()
        metadata = {"project": "org/repo", "type": "issue", "iid": "123"}

        content = handler.fetch_content("https://gitlab.com/org/repo/-/issues/123", metadata)

        assert content is not None
        assert content.title == "Feature Request: Add export"
        assert content.body == "Need ability to export data"
        assert content.source_type == "gitlab"
        assert content.metadata["state"] == "opened"
        assert content.metadata["labels"] == ["feature", "backend"]
        assert content.metadata["author"] == "alice"
        assert content.ttl_hours == 6  # Open issues have shorter TTL


class TestJiraHandler:
    """Test JIRA handler."""

    def test_can_handle(self):
        """Test URL recognition."""
        handler = JiraHandler()

        assert handler.can_handle("https://company.atlassian.net/browse/PROJ-123")
        assert handler.can_handle("https://jira.company.com/browse/TASK-456")
        assert handler.can_handle("https://example.com/jira/browse/BUG-789")
        assert not handler.can_handle("https://github.com/org/repo")
        assert not handler.can_handle("https://example.com")

    def test_extract_identifier(self):
        """Test extracting JIRA ticket identifier."""
        handler = JiraHandler()
        url = "https://company.atlassian.net/browse/PROJ-12345"

        identifier, metadata = handler.extract_identifier(url)

        assert identifier == "PROJ-12345"
        assert metadata["ticket"] == "PROJ-12345"
        assert metadata["project"] == "PROJ"
        assert metadata["instance"] == "https://company.atlassian.net"

    @patch("tenets.utils.external_sources.REQUESTS_AVAILABLE", True)
    def test_fetch_content_success(self, mock_requests, monkeypatch):
        """Test successfully fetching JIRA ticket content."""
        monkeypatch.setenv("JIRA_TOKEN", "jira_token")
        monkeypatch.setenv("JIRA_EMAIL", "user@example.com")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "fields": {
                "summary": "Database connection timeout",
                "description": "Connection to database times out after 30s",
                "status": {"name": "In Progress"},
                "priority": {"name": "High"},
                "assignee": {"displayName": "John Doe"},
                "reporter": {"displayName": "Jane Smith"},
                "issuetype": {"name": "Bug"},
                "labels": ["database", "performance"],
                "created": "2024-01-01T10:00:00.000+0000",
                "updated": "2024-01-01T11:00:00.000+0000",
            }
        }
        mock_requests.get.return_value = mock_response

        handler = JiraHandler()
        metadata = {
            "ticket": "PROJ-123",
            "project": "PROJ",
            "instance": "https://company.atlassian.net",
        }

        content = handler.fetch_content("https://company.atlassian.net/browse/PROJ-123", metadata)

        assert content is not None
        assert content.title == "Database connection timeout"
        assert content.body == "Connection to database times out after 30s"
        assert content.source_type == "jira"
        assert content.metadata["status"] == "In Progress"
        assert content.metadata["priority"] == "High"
        assert content.metadata["assignee"] == "John Doe"
        assert content.metadata["labels"] == ["database", "performance"]

        # Verify API call with authentication
        mock_requests.get.assert_called_once()
        call_args = mock_requests.get.call_args
        assert call_args[1]["auth"] == ("user@example.com", "jira_token")


class TestLinearHandler:
    """Test Linear handler."""

    def test_can_handle(self):
        """Test URL recognition."""
        handler = LinearHandler()

        assert handler.can_handle("https://linear.app/team/issue/TEAM-123")
        assert handler.can_handle("https://linear.app/company/issue/ENG-456")
        assert not handler.can_handle("https://github.com/org/repo")
        assert not handler.can_handle("https://example.com")

    def test_extract_identifier(self):
        """Test extracting Linear issue identifier."""
        handler = LinearHandler()
        url = "https://linear.app/myteam/issue/ENG-1234"

        identifier, metadata = handler.extract_identifier(url)

        assert identifier == "ENG-1234"
        assert metadata["team"] == "myteam"
        assert metadata["issue_id"] == "ENG-1234"

    @patch("tenets.utils.external_sources.REQUESTS_AVAILABLE", True)
    def test_fetch_content_success(self, mock_requests, monkeypatch):
        """Test successfully fetching Linear issue content."""
        monkeypatch.setenv("LINEAR_API_KEY", "lin_api_test123")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "issue": {
                    "title": "Implement user settings",
                    "description": "Add user preferences page",
                    "state": {"name": "In Progress", "type": "started"},
                    "assignee": {"name": "Alice Developer"},
                    "priority": 2,
                    "labels": {"nodes": [{"name": "frontend"}, {"name": "feature"}]},
                    "createdAt": "2024-01-01T10:00:00.000Z",
                    "updatedAt": "2024-01-01T11:00:00.000Z",
                }
            }
        }
        mock_requests.post.return_value = mock_response

        handler = LinearHandler()
        metadata = {"team": "myteam", "issue_id": "ENG-123"}

        content = handler.fetch_content("https://linear.app/myteam/issue/ENG-123", metadata)

        assert content is not None
        assert content.title == "Implement user settings"
        assert content.body == "Add user preferences page"
        assert content.source_type == "linear"
        assert content.metadata["state"] == "In Progress"
        assert content.metadata["assignee"] == "Alice Developer"
        assert content.metadata["labels"] == ["frontend", "feature"]

        # Verify GraphQL API call
        mock_requests.post.assert_called_once_with(
            "https://api.linear.app/graphql",
            headers={"Authorization": "Bearer lin_api_test123", "Content-Type": "application/json"},
            json={"query": pytest.Any(str), "variables": {"id": "ENG-123"}},
            timeout=10,
        )


class TestAsanaHandler:
    """Test Asana handler."""

    def test_can_handle(self):
        """Test URL recognition."""
        handler = AsanaHandler()

        assert handler.can_handle("https://app.asana.com/0/123456/789012")
        assert not handler.can_handle("https://github.com/org/repo")
        assert not handler.can_handle("https://example.com")

    def test_extract_identifier(self):
        """Test extracting Asana task identifier."""
        handler = AsanaHandler()
        url = "https://app.asana.com/0/123456789/987654321"

        identifier, metadata = handler.extract_identifier(url)

        assert identifier == "987654321"
        assert metadata["project_id"] == "123456789"
        assert metadata["task_id"] == "987654321"


class TestNotionHandler:
    """Test Notion handler."""

    def test_can_handle(self):
        """Test URL recognition."""
        handler = NotionHandler()

        assert handler.can_handle("https://notion.so/Page-Title-abc123def456")
        assert handler.can_handle("https://www.notion.so/workspace/Page-789xyz")
        assert handler.can_handle("https://notion.site/public-page")
        assert not handler.can_handle("https://github.com/org/repo")
        assert not handler.can_handle("https://example.com")

    def test_extract_identifier_with_hyphens(self):
        """Test extracting Notion page identifier with hyphens."""
        handler = NotionHandler()
        url = "https://notion.so/Page-Title-12345678-1234-1234-1234-123456789012"

        identifier, metadata = handler.extract_identifier(url)

        assert identifier == "12345678-1234-1234-1234-123456789012"
        assert metadata["page_id"] == "12345678-1234-1234-1234-123456789012"

    def test_extract_identifier_without_hyphens(self):
        """Test extracting Notion page identifier without hyphens."""
        handler = NotionHandler()
        url = "https://notion.so/Page-12345678123412341234123456789012"

        identifier, metadata = handler.extract_identifier(url)

        # Should add hyphens to make proper UUID
        assert identifier == "12345678-1234-1234-1234-123456789012"
        assert metadata["page_id"] == "12345678-1234-1234-1234-123456789012"


class TestExternalSourceManager:
    """Test ExternalSourceManager."""

    @pytest.fixture
    def manager(self, mock_cache_manager):
        """Create manager instance."""
        return ExternalSourceManager(cache_manager=mock_cache_manager)

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert len(manager.handlers) > 0
        assert any(isinstance(h, GitHubHandler) for h in manager.handlers)
        assert any(isinstance(h, GitLabHandler) for h in manager.handlers)
        assert any(isinstance(h, JiraHandler) for h in manager.handlers)

    def test_process_url_github(self, manager, mock_requests):
        """Test processing GitHub URL."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "title": "Test Issue",
            "body": "Issue body",
            "state": "open",
            "labels": [],
            "assignees": [],
        }
        mock_requests.get.return_value = mock_response

        with patch("tenets.utils.external_sources.REQUESTS_AVAILABLE", True):
            content = manager.process_url("https://github.com/org/repo/issues/123")

        assert content is not None
        assert content.title == "Test Issue"
        assert content.source_type == "github"

    def test_process_url_no_handler(self, manager):
        """Test processing URL with no matching handler."""
        content = manager.process_url("https://unknown-site.com/something")

        assert content is None

    def test_extract_reference_github(self, manager):
        """Test extracting GitHub reference from text."""
        text = "Check out this issue: https://github.com/org/repo/issues/123"

        result = manager.extract_reference(text)

        assert result is not None
        url, identifier, metadata = result
        assert url == "https://github.com/org/repo/issues/123"
        assert identifier == "org/repo#123"
        assert metadata["type"] == "issue"

    def test_extract_reference_no_url(self, manager):
        """Test extracting reference when no URL present."""
        text = "This is just plain text without any URLs"

        result = manager.extract_reference(text)

        assert result is None

    def test_extract_reference_unsupported_url(self, manager):
        """Test extracting reference with unsupported URL."""
        text = "Visit https://example.com for more info"

        result = manager.extract_reference(text)

        assert result is None
