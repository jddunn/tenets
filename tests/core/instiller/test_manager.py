"""Tests for TenetManager."""

import pytest
import tempfile
import sqlite3
import json
import yaml
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from tenets.core.instiller.manager import TenetManager
from tenets.models.tenet import (
    Tenet, Priority, TenetStatus, TenetCategory, TenetCollection
)
from tenets.config import TenetsConfig


@pytest.fixture
def config():
    """Create test configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TenetsConfig()
        config.cache_dir = tmpdir
        yield config


@pytest.fixture
def manager(config):
    """Create TenetManager instance."""
    return TenetManager(config)


@pytest.fixture
def sample_tenets(manager):
    """Create sample tenets in manager."""
    tenets = []
    tenets.append(manager.add_tenet(
        "Always use type hints",
        priority=Priority.HIGH,
        category=TenetCategory.STYLE
    ))
    tenets.append(manager.add_tenet(
        "Handle exceptions properly",
        priority=Priority.CRITICAL,
        category=TenetCategory.QUALITY
    ))
    tenets.append(manager.add_tenet(
        "Write tests for new features",
        priority=Priority.MEDIUM,
        category=TenetCategory.TESTING
    ))
    return tenets


class TestTenetManager:
    """Test suite for TenetManager."""
    
    def test_initialization(self, config):
        """Test TenetManager initialization."""
        manager = TenetManager(config)
        
        assert manager.config == config
        assert manager.storage_path.exists()
        assert manager.db_path.exists()
        assert isinstance(manager._tenet_cache, dict)
        
    def test_database_initialization(self, manager):
        """Test database schema creation."""
        # Check tables exist
        with sqlite3.connect(manager.db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            
            assert "tenets" in tables
            assert "tenet_sessions" in tables
            assert "tenet_metrics" in tables
            
    def test_add_tenet_basic(self, manager):
        """Test adding a basic tenet."""
        tenet = manager.add_tenet(
            "Use meaningful variable names",
            priority="high"
        )
        
        assert isinstance(tenet, Tenet)
        assert tenet.content == "Use meaningful variable names"
        assert tenet.priority == Priority.HIGH
        assert tenet.id in manager._tenet_cache
        
    def test_add_tenet_with_category(self, manager):
        """Test adding tenet with category."""
        tenet = manager.add_tenet(
            "Follow PEP 8",
            priority=Priority.MEDIUM,
            category=TenetCategory.STYLE
        )
        
        assert tenet.category == TenetCategory.STYLE
        
    def test_add_tenet_with_session(self, manager):
        """Test adding tenet bound to session."""
        tenet = manager.add_tenet(
            "Session-specific rule",
            session="test-session",
            author="test-user"
        )
        
        assert "test-session" in tenet.session_bindings
        assert tenet.author == "test-user"
        
    def test_get_tenet_by_id(self, manager, sample_tenets):
        """Test getting tenet by full ID."""
        tenet = manager.get_tenet(sample_tenets[0].id)
        
        assert tenet is not None
        assert tenet.content == "Always use type hints"
        
    def test_get_tenet_by_partial_id(self, manager, sample_tenets):
        """Test getting tenet by partial ID."""
        partial_id = sample_tenets[0].id[:8]
        tenet = manager.get_tenet(partial_id)
        
        assert tenet is not None
        assert tenet.content == "Always use type hints"
        
    def test_get_tenet_not_found(self, manager):
        """Test getting non-existent tenet."""
        tenet = manager.get_tenet("nonexistent")
        
        assert tenet is None
        
    def test_list_tenets_all(self, manager, sample_tenets):
        """Test listing all tenets."""
        tenets = manager.list_tenets()
        
        assert len(tenets) == 3
        assert all('content' in t for t in tenets)
        
    def test_list_tenets_pending_only(self, manager, sample_tenets):
        """Test listing only pending tenets."""
        # Instill one tenet
        sample_tenets[0].instill()
        manager._save_tenet(sample_tenets[0])
        
        tenets = manager.list_tenets(pending_only=True)
        
        assert len(tenets) == 2
        
    def test_list_tenets_instilled_only(self, manager, sample_tenets):
        """Test listing only instilled tenets."""
        # Instill one tenet
        sample_tenets[0].instill()
        manager._save_tenet(sample_tenets[0])
        
        tenets = manager.list_tenets(instilled_only=True)
        
        assert len(tenets) == 1
        assert tenets[0]['instilled'] == True
        
    def test_list_tenets_by_category(self, manager, sample_tenets):
        """Test listing tenets by category."""
        tenets = manager.list_tenets(category=TenetCategory.STYLE)
        
        assert len(tenets) == 1
        assert tenets[0]['content'] == "Always use type hints"
        
    def test_list_tenets_by_session(self, manager):
        """Test listing tenets by session."""
        manager.add_tenet("Global tenet")
        manager.add_tenet("Session tenet", session="test-session")
        
        tenets = manager.list_tenets(session="test-session")
        
        # Should include global and session-specific
        assert len(tenets) >= 1
        
    def test_get_pending_tenets(self, manager, sample_tenets):
        """Test getting pending tenets."""
        pending = manager.get_pending_tenets()
        
        assert len(pending) == 3
        # Should be sorted by priority
        assert pending[0].priority == Priority.CRITICAL
        
    def test_get_pending_tenets_by_session(self, manager):
        """Test getting pending tenets for session."""
        manager.add_tenet("Global", priority=Priority.HIGH)
        session_tenet = manager.add_tenet(
            "Session-specific",
            priority=Priority.CRITICAL,
            session="test"
        )
        
        pending = manager.get_pending_tenets(session="test")
        
        assert session_tenet in pending
        
    def test_remove_tenet(self, manager, sample_tenets):
        """Test removing (archiving) a tenet."""
        tenet_id = sample_tenets[0].id
        
        removed = manager.remove_tenet(tenet_id)
        
        assert removed == True
        assert tenet_id not in manager._tenet_cache
        
        # Check it's archived in DB
        with sqlite3.connect(manager.db_path) as conn:
            cursor = conn.execute(
                "SELECT status FROM tenets WHERE id = ?",
                (tenet_id,)
            )
            row = cursor.fetchone()
            assert row[0] == TenetStatus.ARCHIVED.value
            
    def test_remove_tenet_not_found(self, manager):
        """Test removing non-existent tenet."""
        removed = manager.remove_tenet("nonexistent")
        
        assert removed == False
        
    def test_instill_tenets(self, manager, sample_tenets):
        """Test instilling pending tenets."""
        result = manager.instill_tenets()
        
        assert result['count'] == 3
        assert len(result['tenets']) == 3
        assert result['strategy'] == 'priority-based'
        
        # Check status updated
        for tenet in sample_tenets:
            fresh = manager.get_tenet(tenet.id)
            assert fresh.status == TenetStatus.INSTILLED
            
    def test_instill_tenets_force(self, manager, sample_tenets):
        """Test force reinstilling tenets."""
        # First instill
        manager.instill_tenets()
        
        # Force reinstill
        result = manager.instill_tenets(force=True)
        
        assert result['count'] == 3
        
    def test_instill_tenets_by_session(self, manager):
        """Test instilling tenets for specific session."""
        manager.add_tenet("Global")
        manager.add_tenet("Session", session="test")
        
        result = manager.instill_tenets(session="test")
        
        assert result['session'] == "test"
        
    def test_get_tenets_for_injection(self, manager, sample_tenets):
        """Test getting tenets for injection."""
        # Instill tenets first
        for tenet in sample_tenets:
            tenet.instill()
            manager._save_tenet(tenet)
        
        tenets = manager.get_tenets_for_injection(
            context_length=10000,
            max_tenets=2
        )
        
        assert len(tenets) <= 2
        # Should be sorted by priority
        if len(tenets) > 0:
            assert tenets[0].priority.weight >= tenets[-1].priority.weight
            
    def test_export_tenets_yaml(self, manager, sample_tenets):
        """Test exporting tenets as YAML."""
        export = manager.export_tenets(format="yaml")
        
        data = yaml.safe_load(export)
        assert 'version' in data
        assert 'tenets' in data
        assert len(data['tenets']) == 3
        
    def test_export_tenets_json(self, manager, sample_tenets):
        """Test exporting tenets as JSON."""
        export = manager.export_tenets(format="json")
        
        data = json.loads(export)
        assert 'version' in data
        assert 'tenets' in data
        assert len(data['tenets']) == 3
        
    def test_export_tenets_by_session(self, manager):
        """Test exporting tenets filtered by session."""
        manager.add_tenet("Global")
        manager.add_tenet("Session", session="test")
        
        export = manager.export_tenets(format="json", session="test")
        data = json.loads(export)
        
        # Should include session-bound tenet
        assert any("Session" in t['content'] for t in data['tenets'])
        
    def test_import_tenets_yaml(self, manager, tmp_path):
        """Test importing tenets from YAML."""
        # Create export file
        export_data = {
            'version': '1.0',
            'tenets': [
                {
                    'id': 'imported1',
                    'content': 'Imported tenet 1',
                    'priority': 'high',
                    'status': 'pending'
                }
            ]
        }
        
        import_file = tmp_path / "import.yaml"
        with open(import_file, 'w') as f:
            yaml.dump(export_data, f)
        
        count = manager.import_tenets(import_file)
        
        assert count == 1
        assert manager.get_tenet('imported1') is not None
        
    def test_import_tenets_json(self, manager, tmp_path):
        """Test importing tenets from JSON."""
        export_data = {
            'version': '1.0',
            'tenets': [
                {
                    'id': 'imported2',
                    'content': 'Imported tenet 2',
                    'priority': 'medium',
                    'status': 'pending'
                }
            ]
        }
        
        import_file = tmp_path / "import.json"
        with open(import_file, 'w') as f:
            json.dump(export_data, f)
        
        count = manager.import_tenets(import_file)
        
        assert count == 1
        assert manager.get_tenet('imported2') is not None
        
    def test_import_tenets_with_session(self, manager, tmp_path):
        """Test importing tenets with session binding."""
        export_data = {
            'version': '1.0',
            'tenets': [{'id': 'test', 'content': 'Test', 'priority': 'low'}]
        }
        
        import_file = tmp_path / "import.json"
        with open(import_file, 'w') as f:
            json.dump(export_data, f)
        
        count = manager.import_tenets(import_file, session="import-session")
        
        tenet = manager.get_tenet('test')
        assert "import-session" in tenet.session_bindings
        
    def test_import_duplicate_tenets(self, manager, tmp_path):
        """Test importing duplicate tenets are skipped."""
        # Add existing tenet
        existing = manager.add_tenet("Existing")
        
        export_data = {
            'version': '1.0',
            'tenets': [
                {
                    'id': existing.id,
                    'content': 'Should be skipped',
                    'priority': 'high'
                }
            ]
        }
        
        import_file = tmp_path / "import.json"
        with open(import_file, 'w') as f:
            json.dump(export_data, f)
        
        count = manager.import_tenets(import_file)
        
        assert count == 0  # Should skip duplicate
        
    def test_create_collection(self, manager, sample_tenets):
        """Test creating a tenet collection."""
        collection = manager.create_collection(
            name="Best Practices",
            description="Core best practices",
            tenet_ids=[t.id for t in sample_tenets[:2]]
        )
        
        assert isinstance(collection, TenetCollection)
        assert collection.name == "Best Practices"
        assert len(collection.tenets) == 2
        
        # Check saved to disk
        collection_file = manager.storage_path / "collection_best_practices.json"
        assert collection_file.exists()
        
    def test_analyze_tenet_effectiveness(self, manager, sample_tenets):
        """Test effectiveness analysis."""
        # Update some metrics
        sample_tenets[0].metrics.injection_count = 10
        sample_tenets[0].metrics.compliance_score = 0.9
        sample_tenets[1].metrics.injection_count = 5
        sample_tenets[1].metrics.compliance_score = 0.3
        sample_tenets[1].metrics.reinforcement_needed = True
        
        for tenet in sample_tenets:
            manager._save_tenet(tenet)
        
        analysis = manager.analyze_tenet_effectiveness()
        
        assert analysis['total_tenets'] == 3
        assert 'by_status' in analysis
        assert 'by_priority' in analysis
        assert 'most_injected' in analysis
        assert 'need_reinforcement' in analysis
        
        # Check most injected
        assert len(analysis['most_injected']) > 0
        assert analysis['most_injected'][0]['count'] == 10
        
        # Check needs reinforcement
        assert len(analysis['need_reinforcement']) == 1
        
    def test_analyze_effectiveness_empty(self, manager):
        """Test effectiveness analysis with no tenets."""
        # Clear cache
        manager._tenet_cache = {}
        
        analysis = manager.analyze_tenet_effectiveness()
        
        assert analysis['total_tenets'] == 0
        assert analysis['status'] == 'No tenets configured'
        
    def test_save_and_load_tenet_persistence(self, config):
        """Test tenet persistence across manager instances."""
        # Create first manager and add tenet
        manager1 = TenetManager(config)
        tenet = manager1.add_tenet(
            "Persistent tenet",
            priority=Priority.HIGH,
            category=TenetCategory.QUALITY
        )
        tenet_id = tenet.id
        
        # Create second manager - should load from DB
        manager2 = TenetManager(config)
        
        loaded_tenet = manager2.get_tenet(tenet_id)
        assert loaded_tenet is not None
        assert loaded_tenet.content == "Persistent tenet"
        assert loaded_tenet.priority == Priority.HIGH
        
    def test_session_binding_persistence(self, manager):
        """Test session bindings are persisted."""
        tenet = manager.add_tenet("Test", session="session1")
        tenet.bind_to_session("session2")
        manager._save_tenet(tenet)
        
        # Check in database
        with sqlite3.connect(manager.db_path) as conn:
            cursor = conn.execute(
                "SELECT session_id FROM tenet_sessions WHERE tenet_id = ?",
                (tenet.id,)
            )
            sessions = [row[0] for row in cursor.fetchall()]
            
            assert "session1" in sessions
            assert "session2" in sessions
            
    def test_metrics_persistence(self, manager):
        """Test metrics are persisted."""
        tenet = manager.add_tenet("Test")
        tenet.metrics.injection_count = 5
        tenet.metrics.compliance_score = 0.8
        manager._save_tenet(tenet)
        
        # Check in database
        with sqlite3.connect(manager.db_path) as conn:
            cursor = conn.execute(
                "SELECT injection_count, compliance_score FROM tenet_metrics WHERE tenet_id = ?",
                (tenet.id,)
            )
            row = cursor.fetchone()
            
            assert row[0] == 5
            assert row[1] == 0.8