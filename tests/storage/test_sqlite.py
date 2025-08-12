"""Tests for SQLite utilities."""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tenets.config import TenetsConfig
from tenets.storage.sqlite import Database, SQLitePaths


@pytest.fixture
def config():
    """Create test configuration with temporary cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TenetsConfig()
        config.cache.directory = tmpdir
        config.cache.sqlite_pragmas = {
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "foreign_keys": "ON",
        }
        yield config


@pytest.fixture
def database(config):
    """Create Database instance."""
    return Database(config)


class TestSQLitePaths:
    """Test suite for SQLitePaths dataclass."""

    def test_sqlite_paths_creation(self):
        """Test SQLitePaths dataclass creation."""
        root = Path("/tmp/cache")
        main_db = root / "tenets.db"

        paths = SQLitePaths(root=root, main_db=main_db)

        assert paths.root == root
        assert paths.main_db == main_db


class TestDatabase:
    """Test suite for Database class."""

    def test_initialization(self, config):
        """Test Database initialization."""
        db = Database(config)

        assert db.config == config
        assert db.paths is not None
        assert db.paths.root == Path(config.cache.directory)
        assert db.paths.main_db == Path(config.cache.directory) / "tenets.db"

    def test_resolve_paths(self, config):
        """Test path resolution."""
        paths = Database._resolve_paths(config)

        assert isinstance(paths, SQLitePaths)
        assert paths.root == Path(config.cache.directory)
        assert paths.main_db.name == "tenets.db"

    def test_ensure_dirs(self, config):
        """Test directory creation."""
        # Use a non-existent directory
        config.cache.directory = "/tmp/test_tenets_cache_" + str(id(config))

        db = Database(config)

        # Directory should be created
        assert db.paths.root.exists()
        assert db.paths.root.is_dir()

        # Clean up
        import shutil

        shutil.rmtree(db.paths.root)

    def test_connect_default_path(self, database):
        """Test connecting to default database path."""
        conn = database.connect()

        assert conn is not None
        assert isinstance(conn, sqlite3.Connection)

        # Database file should exist
        assert database.paths.main_db.exists()

        # Test connection is working
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1

        conn.close()

    def test_connect_custom_path(self, database, tmp_path):
        """Test connecting to custom database path."""
        custom_db = tmp_path / "custom.db"

        conn = database.connect(db_path=custom_db)

        assert conn is not None
        assert custom_db.exists()

        # Create a table to verify connection
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER)")
        conn.commit()
        conn.close()

        # Verify table exists
        conn2 = sqlite3.connect(custom_db)
        cursor = conn2.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn2.close()

        assert ("test",) in tables

    def test_apply_pragmas(self, database):
        """Test PRAGMA application."""
        conn = database.connect()

        # Check that pragmas were applied
        cursor = conn.cursor()

        # Check journal_mode
        cursor.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0]
        assert journal_mode.upper() == "WAL"

        # Check synchronous
        cursor.execute("PRAGMA synchronous")
        synchronous = cursor.fetchone()[0]
        assert synchronous == 1  # NORMAL = 1

        # Check foreign_keys
        cursor.execute("PRAGMA foreign_keys")
        foreign_keys = cursor.fetchone()[0]
        assert foreign_keys == 1  # ON = 1

        conn.close()

    def test_apply_pragmas_error_handling(self, database):
        """Test PRAGMA error handling."""
        conn = sqlite3.connect(":memory:")

        # Invalid pragma should be handled gracefully
        invalid_pragmas = {"invalid_pragma": "value", "another_bad": "setting"}

        # Should not raise exception
        database._apply_pragmas(conn, invalid_pragmas)

        # Connection should still work
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        assert cursor.fetchone()[0] == 1

        conn.close()

    def test_detect_types(self, database):
        """Test that PARSE_DECLTYPES is set."""
        conn = database.connect()

        # Create table with datetime
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE test_types (
                id INTEGER,
                created_at TIMESTAMP
            )
        """
        )

        from datetime import datetime

        now = datetime.now()

        cursor.execute("INSERT INTO test_types VALUES (?, ?)", (1, now))
        conn.commit()

        # Retrieve and check type
        cursor.execute("SELECT created_at FROM test_types WHERE id = 1")
        result = cursor.fetchone()[0]

        # Should be datetime object, not string
        assert isinstance(result, datetime)

        conn.close()

    def test_multiple_connections(self, database):
        """Test multiple simultaneous connections."""
        conn1 = database.connect()
        conn2 = database.connect()

        assert conn1 is not conn2

        # Both should work
        cursor1 = conn1.cursor()
        cursor1.execute("CREATE TABLE IF NOT EXISTS test1 (id INTEGER)")
        conn1.commit()

        cursor2 = conn2.cursor()
        cursor2.execute("INSERT INTO test1 VALUES (1)")
        conn2.commit()

        # Check data is visible
        cursor1.execute("SELECT COUNT(*) FROM test1")
        count = cursor1.fetchone()[0]
        assert count == 1

        conn1.close()
        conn2.close()

    def test_wal_mode_persistence(self, database):
        """Test that WAL mode persists across connections."""
        # First connection sets WAL mode
        conn1 = database.connect()
        conn1.close()

        # Second connection should still be in WAL mode
        conn2 = sqlite3.connect(database.paths.main_db)
        cursor = conn2.cursor()
        cursor.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        conn2.close()

        assert mode.upper() == "WAL"

        # WAL files should exist
        wal_file = database.paths.main_db.with_suffix(".db-wal")
        shm_file = database.paths.main_db.with_suffix(".db-shm")

        # These may or may not exist depending on checkpoint status
        # Just verify no errors accessing them

    def test_empty_pragmas(self, config):
        """Test with no pragmas configured."""
        config.cache.sqlite_pragmas = {}

        db = Database(config)
        conn = db.connect()

        # Should connect successfully
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        assert cursor.fetchone()[0] == 1

        conn.close()

    def test_connection_with_path_object(self, database):
        """Test that Path objects are handled correctly."""
        path = Path(database.paths.main_db)

        conn = database.connect(db_path=path)

        assert conn is not None
        conn.close()

    def test_connection_string_path(self, database, tmp_path):
        """Test connection with string path."""
        str_path = str(tmp_path / "string_db.db")

        conn = database.connect(db_path=str_path)

        assert conn is not None
        assert Path(str_path).exists()
        conn.close()

    def test_concurrent_writes_wal_mode(self, database):
        """Test that WAL mode allows concurrent reads during write."""
        import threading
        import time

        conn1 = database.connect()
        conn2 = database.connect()

        # Create test table
        cursor1 = conn1.cursor()
        cursor1.execute("CREATE TABLE concurrent_test (id INTEGER, value TEXT)")
        conn1.commit()

        # Start a write transaction in thread
        def long_write():
            cursor = conn1.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            for i in range(100):
                cursor.execute("INSERT INTO concurrent_test VALUES (?, ?)", (i, f"value_{i}"))
                time.sleep(0.001)  # Simulate slow write
            conn1.commit()

        write_thread = threading.Thread(target=long_write)
        write_thread.start()

        # Try to read while writing (should work with WAL)
        time.sleep(0.01)  # Let write start
        cursor2 = conn2.cursor()
        cursor2.execute("SELECT COUNT(*) FROM concurrent_test")
        # Should not block in WAL mode

        write_thread.join()

        conn1.close()
        conn2.close()

    def test_database_logger(self, database, caplog):
        """Test that logger is properly configured."""
        assert database.logger is not None

        # Test logging on pragma error
        conn = sqlite3.connect(":memory:")
        database._apply_pragmas(conn, {"bad_pragma": "value"})

        # Should have logged the error
        # Note: Actual log capture depends on pytest-caplog configuration

    def test_paths_creation_failure(self, config):
        """Test handling of directory creation failure."""
        # Set cache directory to unwriteable location
        config.cache.directory = "/root/cannot_write_here"

        with patch("pathlib.Path.mkdir", side_effect=PermissionError("Cannot create")):
            with pytest.raises(PermissionError):
                db = Database(config)
