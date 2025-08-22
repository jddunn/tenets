"""Compatibility shim for external source handlers.

This module was relocated to ``tenets.utils.external_sources``.
We re-export the public API here to maintain backward compatibility
with code/tests that still import from ``tenets.core.prompt.external_sources``.
"""

from tenets.utils.external_sources import (
    AsanaHandler,
    ExternalContent,
    ExternalSourceHandler,
    ExternalSourceManager,
    GitHubHandler,
    GitLabHandler,
    JiraHandler,
    LinearHandler,
    NotionHandler,
)

__all__ = [
    "AsanaHandler",
    "ExternalContent",
    "ExternalSourceHandler",
    "ExternalSourceManager",
    "GitHubHandler",
    "GitLabHandler",
    "JiraHandler",
    "LinearHandler",
    "NotionHandler",
]
