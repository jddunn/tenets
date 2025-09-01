"""Shared fixtures for instiller tests."""

import tempfile
from pathlib import Path

import pytest

from tenets.config import TenetsConfig
from tenets.models.tenet import Priority, Tenet, TenetCategory


@pytest.fixture
def test_config():
    """Create a test configuration with temporary directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TenetsConfig()
        config.cache_dir = tmpdir
        config.cache.directory = tmpdir
        config.max_tenets_per_context = 5
        config.tenet_injection_config = {
            "min_distance_between": 500,
            "prefer_natural_breaks": True,
            "reinforce_at_end": True,
        }
        # Explicitly disable system instruction to avoid interference
        config.tenet.system_instruction = None
        config.tenet.system_instruction_enabled = False
        yield config


@pytest.fixture
def basic_tenets():
    """Create a set of basic tenets for testing."""
    return [
        Tenet(
            content="Always validate user input",
            priority=Priority.CRITICAL,
            category=TenetCategory.SECURITY,
        ),
        Tenet(
            content="Use descriptive variable names",
            priority=Priority.HIGH,
            category=TenetCategory.STYLE,
        ),
        Tenet(
            content="Add unit tests for new functions",
            priority=Priority.HIGH,
            category=TenetCategory.TESTING,
        ),
        Tenet(
            content="Document complex logic",
            priority=Priority.MEDIUM,
            category=TenetCategory.DOCUMENTATION,
        ),
        Tenet(
            content="Optimize only when necessary",
            priority=Priority.LOW,
            category=TenetCategory.PERFORMANCE,
        ),
    ]


@pytest.fixture
def sample_markdown_content():
    """Sample markdown content for testing injection."""
    return """# API Documentation

## Overview
This document describes the REST API endpoints for our service.

## Authentication

All endpoints require authentication using JWT tokens.

```python
headers = {
    'Authorization': f'Bearer {token}'
}
```

## Endpoints

### GET /users
Returns a list of users.

### POST /users
Creates a new user.

## Error Handling

All errors return appropriate HTTP status codes.

## Rate Limiting

API calls are limited to 100 requests per minute.
"""


@pytest.fixture
def sample_code_content():
    """Sample code content for testing."""
    return """import os
import sys
from typing import List, Optional

class UserService:
    '''Service for managing users.'''
    
    def __init__(self, db_connection):
        self.db = db_connection
        
    def get_users(self) -> List[dict]:
        '''Get all users from database.'''
        query = "SELECT * FROM users"
        return self.db.execute(query)
        
    def create_user(self, name: str, email: str) -> dict:
        '''Create a new user.'''
        # TODO: Add validation
        query = "INSERT INTO users (name, email) VALUES (?, ?)"
        self.db.execute(query, (name, email))
        return {"name": name, "email": email}
        
    def delete_user(self, user_id: int) -> bool:
        '''Delete a user by ID.'''
        query = "DELETE FROM users WHERE id = ?"
        result = self.db.execute(query, (user_id,))
        return result.rowcount > 0
"""


@pytest.fixture
def mock_context_result():
    """Create a mock ContextResult for testing."""
    from tenets.models.context import ContextResult

    return ContextResult(
        files=["api.py", "models.py", "tests/test_api.py"],
        context="# API Implementation\n\nMain API code here.",
        format="markdown",
        metadata={
            "total_tokens": 500,
            "session": "test-session",
            "timestamp": "2024-01-01T00:00:00",
        },
    )
