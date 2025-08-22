"""External source handlers for various platforms.

This module provides handlers for fetching and parsing content from external
sources like GitHub, GitLab, JIRA, Linear, Asana, Notion, etc.
"""

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from tenets.storage.cache import CacheManager
from tenets.utils.logger import get_logger

# Optional imports
try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    BeautifulSoup = None


@dataclass
class ExternalContent:
    """Parsed content from an external source."""

    title: str
    body: str
    metadata: Dict[str, Any]
    source_type: str
    url: str
    cached_at: Optional[datetime] = None
    ttl_hours: int = 24


class ExternalSourceHandler(ABC):
    """Base class for external source handlers."""

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """Initialize handler with optional cache.

        Args:
            cache_manager: Optional cache manager for caching fetched content
        """
        self.logger = get_logger(self.__class__.__name__)
        self.cache = cache_manager
        self._api_tokens = self._load_api_tokens()

    def _load_api_tokens(self) -> Dict[str, str]:
        """Load API tokens from environment variables."""
        return {
            "github": os.environ.get("GITHUB_TOKEN", ""),
            "gitlab": os.environ.get("GITLAB_TOKEN", ""),
            "bitbucket": os.environ.get("BITBUCKET_TOKEN", ""),
            "jira": os.environ.get("JIRA_TOKEN", ""),
            "linear": os.environ.get("LINEAR_API_KEY", ""),
            "asana": os.environ.get("ASANA_TOKEN", ""),
            "notion": os.environ.get("NOTION_TOKEN", ""),
            "slack": os.environ.get("SLACK_TOKEN", ""),
            "confluence": os.environ.get("CONFLUENCE_TOKEN", ""),
            "trello": os.environ.get("TRELLO_API_KEY", ""),
            "monday": os.environ.get("MONDAY_TOKEN", ""),
            "clickup": os.environ.get("CLICKUP_TOKEN", ""),
        }

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """Check if this handler can process the given URL."""
        pass

    @abstractmethod
    def extract_identifier(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """Extract identifier and metadata from URL.

        Returns:
            Tuple of (identifier, metadata)
        """
        pass

    @abstractmethod
    def fetch_content(self, url: str, metadata: Dict[str, Any]) -> Optional[ExternalContent]:
        """Fetch content from the external source."""
        pass

    def get_cached_content(self, url: str) -> Optional[ExternalContent]:
        """Get cached content if available and valid.

        Args:
            url: URL to check cache for

        Returns:
            Cached content or None if not cached/expired
        """
        if not self.cache:
            return None

        cache_key = f"external_content:{url}"
        cached = self.cache.general.get(cache_key)

        if cached and isinstance(cached, dict):
            # Check if cache is still valid
            cached_at = cached.get("cached_at")
            ttl_hours = cached.get("ttl_hours", 24)

            if cached_at:
                cached_time = (
                    datetime.fromisoformat(cached_at) if isinstance(cached_at, str) else cached_at
                )
                if datetime.now() - cached_time < timedelta(hours=ttl_hours):
                    self.logger.debug(f"Using cached content for {url}")
                    return ExternalContent(**cached)
                else:
                    self.logger.debug(f"Cache expired for {url}")

        return None

    def cache_content(self, url: str, content: ExternalContent) -> None:
        """Cache fetched content.

        Args:
            url: URL as cache key
            content: Content to cache
        """
        if not self.cache:
            return

        cache_key = f"external_content:{url}"
        content.cached_at = datetime.now()

        self.cache.general.put(
            cache_key,
            {
                "title": content.title,
                "body": content.body,
                "metadata": content.metadata,
                "source_type": content.source_type,
                "url": content.url,
                "cached_at": content.cached_at.isoformat(),
                "ttl_hours": content.ttl_hours,
            },
            ttl=content.ttl_hours * 3600,  # Convert to seconds
        )
        self.logger.debug(f"Cached content for {url} (TTL: {content.ttl_hours}h)")

    def process(self, url: str) -> Optional[ExternalContent]:
        """Process URL with caching support.

        Args:
            url: URL to process

        Returns:
            External content or None if failed
        """
        # Check cache first
        cached = self.get_cached_content(url)
        if cached:
            return cached

        # Extract identifier and metadata
        try:
            identifier, metadata = self.extract_identifier(url)
            metadata["identifier"] = identifier
        except Exception as e:
            self.logger.error(f"Failed to extract identifier from {url}: {e}")
            return None

        # Fetch fresh content
        try:
            content = self.fetch_content(url, metadata)
            if content:
                # Cache the content
                self.cache_content(url, content)
                return content
        except Exception as e:
            self.logger.error(f"Failed to fetch content from {url}: {e}")

        return None


class GitHubHandler(ExternalSourceHandler):
    """Handler for GitHub issues, PRs, discussions, and gists."""

    def can_handle(self, url: str) -> bool:
        """Check if URL is a GitHub URL."""
        return "github.com" in url

    def extract_identifier(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """Extract GitHub identifier from URL."""
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")

        metadata = {"platform": "github"}
        identifier = ""

        if len(path_parts) >= 4:
            owner, repo = path_parts[0], path_parts[1]
            metadata["owner"] = owner
            metadata["repo"] = repo

            if path_parts[2] == "issues" and len(path_parts) >= 4:
                metadata["type"] = "issue"
                metadata["number"] = path_parts[3]
                identifier = f"{owner}/{repo}#{path_parts[3]}"
            elif path_parts[2] == "pull" and len(path_parts) >= 4:
                metadata["type"] = "pull_request"
                metadata["number"] = path_parts[3]
                identifier = f"{owner}/{repo}#{path_parts[3]}"
            elif path_parts[2] == "discussions" and len(path_parts) >= 4:
                metadata["type"] = "discussion"
                metadata["number"] = path_parts[3]
                identifier = f"{owner}/{repo}/discussions/{path_parts[3]}"
            elif path_parts[2] == "commit" and len(path_parts) >= 4:
                metadata["type"] = "commit"
                metadata["sha"] = path_parts[3][:7]  # Short SHA
                identifier = f"{owner}/{repo}@{path_parts[3][:7]}"
        elif "gist.github.com" in parsed.netloc and len(path_parts) >= 2:
            metadata["type"] = "gist"
            metadata["gist_id"] = path_parts[-1]
            identifier = f"gist:{path_parts[-1]}"

        return identifier, metadata

    def fetch_content(self, url: str, metadata: Dict[str, Any]) -> Optional[ExternalContent]:
        """Fetch content from GitHub API."""
        if not REQUESTS_AVAILABLE:
            self.logger.warning("requests library not available")
            return None

        api_url = None
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Tenets-PromptParser/1.0",
        }

        # Add auth token if available
        token = self._api_tokens.get("github")
        if token:
            headers["Authorization"] = f"token {token}"

        # Build API URL based on type
        if metadata.get("type") == "issue":
            api_url = f"https://api.github.com/repos/{metadata['owner']}/{metadata['repo']}/issues/{metadata['number']}"
        elif metadata.get("type") == "pull_request":
            api_url = f"https://api.github.com/repos/{metadata['owner']}/{metadata['repo']}/pulls/{metadata['number']}"
        elif metadata.get("type") == "discussion":
            # GraphQL would be better but REST API works too
            api_url = f"https://api.github.com/repos/{metadata['owner']}/{metadata['repo']}/discussions/{metadata['number']}"
        elif metadata.get("type") == "gist":
            api_url = f"https://api.github.com/gists/{metadata['gist_id']}"

        if not api_url:
            return None

        try:
            response = requests.get(api_url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Extract content based on type
            title = data.get("title", "")
            body = data.get("body", "")

            # Add additional context
            if metadata.get("type") in ["issue", "pull_request"]:
                state = data.get("state", "")
                labels = [label["name"] for label in data.get("labels", [])]
                assignees = [a["login"] for a in data.get("assignees", [])]

                metadata.update(
                    {
                        "state": state,
                        "labels": labels,
                        "assignees": assignees,
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                    }
                )

                # Add PR-specific info
                if metadata.get("type") == "pull_request":
                    metadata["merged"] = data.get("merged", False)
                    metadata["draft"] = data.get("draft", False)

            return ExternalContent(
                title=title,
                body=body,
                metadata=metadata,
                source_type="github",
                url=url,
                ttl_hours=(
                    6 if metadata.get("state") == "open" else 24
                ),  # Shorter TTL for open items
            )

        except Exception as e:
            self.logger.error(f"GitHub API request failed: {e}")
            return None


class GitLabHandler(ExternalSourceHandler):
    """Handler for GitLab issues, MRs, and snippets."""

    def can_handle(self, url: str) -> bool:
        """Check if URL is a GitLab URL."""
        return "gitlab.com" in url or "gitlab" in url

    def extract_identifier(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """Extract GitLab identifier from URL."""
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")

        metadata = {"platform": "gitlab"}
        identifier = ""

        # Handle different GitLab URL structures
        if "/-/" in parsed.path:
            # New GitLab URL format: /namespace/project/-/issues/123
            split_idx = path_parts.index("-")
            project_path = "/".join(path_parts[:split_idx])
            resource_parts = path_parts[split_idx + 1 :]

            metadata["project"] = project_path

            if len(resource_parts) >= 2:
                resource_type = resource_parts[0]
                resource_id = resource_parts[1]

                if resource_type == "issues":
                    metadata["type"] = "issue"
                    metadata["iid"] = resource_id
                    identifier = f"{project_path}#{resource_id}"
                elif resource_type == "merge_requests":
                    metadata["type"] = "merge_request"
                    metadata["iid"] = resource_id
                    identifier = f"{project_path}!{resource_id}"
                elif resource_type == "snippets":
                    metadata["type"] = "snippet"
                    metadata["id"] = resource_id
                    identifier = f"{project_path}$${resource_id}"

        return identifier, metadata

    def fetch_content(self, url: str, metadata: Dict[str, Any]) -> Optional[ExternalContent]:
        """Fetch content from GitLab API."""
        if not REQUESTS_AVAILABLE:
            self.logger.warning("requests library not available")
            return None

        # Determine API base URL
        parsed = urlparse(url)
        api_base = f"https://{parsed.netloc}/api/v4"

        headers = {
            "User-Agent": "Tenets-PromptParser/1.0",
        }

        # Add auth token if available
        token = self._api_tokens.get("gitlab")
        if token:
            headers["PRIVATE-TOKEN"] = token

        # Build API URL
        project_encoded = metadata["project"].replace("/", "%2F")

        if metadata.get("type") == "issue":
            api_url = f"{api_base}/projects/{project_encoded}/issues/{metadata['iid']}"
        elif metadata.get("type") == "merge_request":
            api_url = f"{api_base}/projects/{project_encoded}/merge_requests/{metadata['iid']}"
        elif metadata.get("type") == "snippet":
            api_url = f"{api_base}/projects/{project_encoded}/snippets/{metadata['id']}"
        else:
            return None

        try:
            response = requests.get(api_url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            title = data.get("title", "")
            body = data.get("description", "")

            # Add metadata
            metadata.update(
                {
                    "state": data.get("state"),
                    "labels": data.get("labels", []),
                    "author": data.get("author", {}).get("username"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                }
            )

            return ExternalContent(
                title=title,
                body=body,
                metadata=metadata,
                source_type="gitlab",
                url=url,
                ttl_hours=6 if metadata.get("state") == "opened" else 24,
            )

        except Exception as e:
            self.logger.error(f"GitLab API request failed: {e}")
            return None


class JiraHandler(ExternalSourceHandler):
    """Handler for JIRA tickets."""

    def can_handle(self, url: str) -> bool:
        """Check if URL is a JIRA URL."""
        return "atlassian.net" in url or "/browse/" in url or "jira" in url.lower()

    def extract_identifier(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """Extract JIRA ticket identifier from URL."""
        # JIRA ticket pattern: PROJECT-123
        ticket_pattern = r"([A-Z][A-Z0-9]*-\d+)"
        match = re.search(ticket_pattern, url)

        metadata = {"platform": "jira"}
        identifier = ""

        if match:
            ticket_id = match.group(1)
            metadata["ticket"] = ticket_id
            metadata["project"] = ticket_id.split("-")[0]
            identifier = ticket_id

            # Extract instance URL
            parsed = urlparse(url)
            metadata["instance"] = f"{parsed.scheme}://{parsed.netloc}"

        return identifier, metadata

    def fetch_content(self, url: str, metadata: Dict[str, Any]) -> Optional[ExternalContent]:
        """Fetch content from JIRA API."""
        if not REQUESTS_AVAILABLE:
            self.logger.warning("requests library not available")
            return None

        api_url = f"{metadata['instance']}/rest/api/latest/issue/{metadata['ticket']}"

        headers = {
            "Accept": "application/json",
            "User-Agent": "Tenets-PromptParser/1.0",
        }

        # JIRA often requires authentication
        token = self._api_tokens.get("jira")
        jira_email = os.environ.get("JIRA_EMAIL", "")

        auth = None
        if token and jira_email:
            # Use basic auth with email and token
            auth = (jira_email, token)

        try:
            response = requests.get(api_url, headers=headers, auth=auth, timeout=10)
            response.raise_for_status()
            data = response.json()

            fields = data.get("fields", {})
            title = fields.get("summary", "")
            body = fields.get("description", "")

            # Add metadata
            metadata.update(
                {
                    "status": fields.get("status", {}).get("name"),
                    "priority": fields.get("priority", {}).get("name"),
                    "assignee": (
                        fields.get("assignee", {}).get("displayName")
                        if fields.get("assignee")
                        else None
                    ),
                    "reporter": fields.get("reporter", {}).get("displayName"),
                    "issue_type": fields.get("issuetype", {}).get("name"),
                    "labels": fields.get("labels", []),
                    "created": fields.get("created"),
                    "updated": fields.get("updated"),
                }
            )

            return ExternalContent(
                title=title,
                body=body,
                metadata=metadata,
                source_type="jira",
                url=url,
                ttl_hours=12,  # JIRA tickets change frequently
            )

        except Exception as e:
            self.logger.error(f"JIRA API request failed: {e}")
            return None


class LinearHandler(ExternalSourceHandler):
    """Handler for Linear issues."""

    def can_handle(self, url: str) -> bool:
        """Check if URL is a Linear URL."""
        return "linear.app" in url

    def extract_identifier(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """Extract Linear identifier from URL."""
        # Linear URL: linear.app/team/issue/TEAM-123
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")

        metadata = {"platform": "linear"}
        identifier = ""

        if len(path_parts) >= 3:
            team = path_parts[0]
            issue_id = path_parts[-1]  # Last part is usually the issue ID

            metadata["team"] = team
            metadata["issue_id"] = issue_id
            identifier = issue_id

        return identifier, metadata

    def fetch_content(self, url: str, metadata: Dict[str, Any]) -> Optional[ExternalContent]:
        """Fetch content from Linear API using GraphQL."""
        if not REQUESTS_AVAILABLE:
            self.logger.warning("requests library not available")
            return None

        api_url = "https://api.linear.app/graphql"
        token = self._api_tokens.get("linear")

        if not token:
            self.logger.warning("Linear API key not configured")
            return None

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # GraphQL query to fetch issue details
        query = """
        query GetIssue($id: ID!) {
                issue(id: $id) {
                    title
                    description
                    state {
                        name
                        type
                    }
                    assignee {
                        name
                    }
                    priority
                    labels {
                        nodes {
                            name
                        }
                    }
                    createdAt
                    updatedAt
                }
            }
        """

        try:
            response = requests.post(
                api_url,
                headers=headers,
                json={"query": query, "variables": {"id": metadata["issue_id"]}},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            issue = data.get("data", {}).get("issue", {})
            if not issue:
                return None

            title = issue.get("title", "")
            body = issue.get("description", "")

            # Add metadata
            metadata.update(
                {
                    "state": issue.get("state", {}).get("name"),
                    "state_type": issue.get("state", {}).get("type"),
                    "assignee": (
                        issue.get("assignee", {}).get("name") if issue.get("assignee") else None
                    ),
                    "priority": issue.get("priority"),
                    "labels": [label["name"] for label in issue.get("labels", {}).get("nodes", [])],
                    "created_at": issue.get("createdAt"),
                    "updated_at": issue.get("updatedAt"),
                }
            )

            return ExternalContent(
                title=title,
                body=body,
                metadata=metadata,
                source_type="linear",
                url=url,
                ttl_hours=6,  # Linear issues update frequently
            )

        except Exception as e:
            self.logger.error(f"Linear API request failed: {e}")
            return None


class AsanaHandler(ExternalSourceHandler):
    """Handler for Asana tasks."""

    def can_handle(self, url: str) -> bool:
        """Check if URL is an Asana URL."""
        return "app.asana.com" in url

    def extract_identifier(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """Extract Asana task identifier from URL."""
        # Asana URL: app.asana.com/0/project_id/task_id
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")

        metadata = {"platform": "asana"}
        identifier = ""

        if len(path_parts) >= 3:
            project_id = path_parts[1]
            task_id = path_parts[2].split("/")[0]  # Remove any trailing parts

            metadata["project_id"] = project_id
            metadata["task_id"] = task_id
            identifier = task_id

        return identifier, metadata

    def fetch_content(self, url: str, metadata: Dict[str, Any]) -> Optional[ExternalContent]:
        """Fetch content from Asana API."""
        if not REQUESTS_AVAILABLE:
            self.logger.warning("requests library not available")
            return None

        token = self._api_tokens.get("asana")
        if not token:
            self.logger.warning("Asana token not configured")
            return None

        api_url = f"https://app.asana.com/api/1.0/tasks/{metadata['task_id']}"

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

        try:
            response = requests.get(api_url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json().get("data", {})

            title = data.get("name", "")
            body = data.get("notes", "")

            # Add metadata
            metadata.update(
                {
                    "completed": data.get("completed"),
                    "assignee": (
                        data.get("assignee", {}).get("name") if data.get("assignee") else None
                    ),
                    "due_on": data.get("due_on"),
                    "tags": [tag["name"] for tag in data.get("tags", [])],
                    "created_at": data.get("created_at"),
                    "modified_at": data.get("modified_at"),
                }
            )

            return ExternalContent(
                title=title,
                body=body,
                metadata=metadata,
                source_type="asana",
                url=url,
                ttl_hours=12,
            )

        except Exception as e:
            self.logger.error(f"Asana API request failed: {e}")
            return None


class NotionHandler(ExternalSourceHandler):
    """Handler for Notion pages and databases."""

    def can_handle(self, url: str) -> bool:
        """Check if URL is a Notion URL."""
        return "notion.so" in url or "notion.site" in url

    def extract_identifier(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """Extract Notion page/database identifier from URL."""
        # Notion URLs contain a UUID at the end
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        metadata = {"platform": "notion"}
        identifier = ""

        # Extract the UUID (last 32 characters, may have hyphens)
        uuid_pattern = (
            r"([a-f0-9]{32}|[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})$"
        )
        match = re.search(uuid_pattern, path.replace("-", ""))

        if match:
            page_id = match.group(1)
            if len(page_id) == 32:
                # Add hyphens to make it a proper UUID
                page_id = f"{page_id[:8]}-{page_id[8:12]}-{page_id[12:16]}-{page_id[16:20]}-{page_id[20:]}"

            metadata["page_id"] = page_id
            identifier = page_id

        return identifier, metadata

    def fetch_content(self, url: str, metadata: Dict[str, Any]) -> Optional[ExternalContent]:
        """Fetch content from Notion API."""
        if not REQUESTS_AVAILABLE:
            self.logger.warning("requests library not available")
            return None

        token = self._api_tokens.get("notion")
        if not token:
            self.logger.warning("Notion token not configured")
            return None

        # Notion API requires version header
        headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        }

        # First, get page metadata
        page_url = f"https://api.notion.com/v1/pages/{metadata['page_id']}"

        try:
            response = requests.get(page_url, headers=headers, timeout=10)
            response.raise_for_status()
            page_data = response.json()

            # Extract title from properties
            title = ""
            properties = page_data.get("properties", {})
            for prop in properties.values():
                if prop.get("type") == "title" and prop.get("title"):
                    title = "".join([t.get("plain_text", "") for t in prop["title"]])
                    break

            # Get page content blocks
            blocks_url = f"https://api.notion.com/v1/blocks/{metadata['page_id']}/children"
            response = requests.get(blocks_url, headers=headers, timeout=10)
            response.raise_for_status()
            blocks_data = response.json()

            # Extract text from blocks (simplified)
            body_parts = []
            for block in blocks_data.get("results", []):
                block_type = block.get("type")
                if block_type in [
                    "paragraph",
                    "heading_1",
                    "heading_2",
                    "heading_3",
                    "bulleted_list_item",
                    "numbered_list_item",
                ]:
                    text_content = block.get(block_type, {}).get("rich_text", [])
                    text = "".join([t.get("plain_text", "") for t in text_content])
                    if text:
                        body_parts.append(text)

            body = "\n".join(body_parts)

            # Add metadata
            metadata.update(
                {
                    "created_time": page_data.get("created_time"),
                    "last_edited_time": page_data.get("last_edited_time"),
                    "archived": page_data.get("archived"),
                }
            )

            return ExternalContent(
                title=title,
                body=body,
                metadata=metadata,
                source_type="notion",
                url=url,
                ttl_hours=24,  # Notion content typically doesn't change as frequently
            )

        except Exception as e:
            self.logger.error(f"Notion API request failed: {e}")
            return None


class ExternalSourceManager:
    """Manages all external source handlers."""

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """Initialize with all available handlers.

        Args:
            cache_manager: Optional cache manager for handlers
        """
        self.logger = get_logger(__name__)
        self.cache_manager = cache_manager

        # Initialize all handlers
        self.handlers = [
            GitHubHandler(cache_manager),
            GitLabHandler(cache_manager),
            JiraHandler(cache_manager),
            LinearHandler(cache_manager),
            AsanaHandler(cache_manager),
            NotionHandler(cache_manager),
        ]

        # Add more handlers as needed:
        # - BitbucketHandler
        # - ConfluenceHandler
        # - SlackHandler
        # - TrelloHandler
        # - MondayHandler
        # - ClickUpHandler

    def process_url(self, url: str) -> Optional[ExternalContent]:
        """Process a URL with the appropriate handler.

        Args:
            url: URL to process

        Returns:
            External content or None if no handler can process it
        """
        for handler in self.handlers:
            if handler.can_handle(url):
                self.logger.info(f"Processing {url} with {handler.__class__.__name__}")
                return handler.process(url)

        self.logger.debug(f"No handler found for URL: {url}")
        return None

    def extract_reference(self, text: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """Extract external reference from text.

        Args:
            text: Text that may contain a URL

        Returns:
            Tuple of (url, identifier, metadata) or None
        """
        # Find URLs in text
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        match = re.search(url_pattern, text)

        if not match:
            return None

        url = match.group(0)

        # Find handler and extract identifier
        for handler in self.handlers:
            if handler.can_handle(url):
                try:
                    identifier, metadata = handler.extract_identifier(url)
                    return url, identifier, metadata
                except Exception as e:
                    self.logger.error(f"Failed to extract identifier from {url}: {e}")

        return None
