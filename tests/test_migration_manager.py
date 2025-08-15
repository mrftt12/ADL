"""
Tests for migration manager functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from automl_framework.migrations.migration_manager import MigrationManager
from automl_framework.core.database import DatabaseManager


class TestMigrationManager:
    """Test cases for MigrationManager."""
    
    @patch('automl_framework.migrations.migration_manager.get_postgres_session')
    @patch('automl_framework.migrations.migration_manager.get_mongo_collection')
    def test_initialize_migration_tracking(self, mock_get_mongo_collection, mock_get_postgres_session):
        """Test initializing migration tracking."""
        # Mock PostgreSQL session
        mock_session = Mock()
        mock_get_postgres_session.return_value.__enter__.return_value = mock_session
        
        # Mock MongoDB collection
        mock_collection = Mock()
        mock_get_mongo_collection.return_value = mock_collection
        
        # Create migration manager
        db_manager = Mock()
        migration_manager = MigrationManager(db_manager)
        
        # Initialize migration tracking
        migration_manager.initialize_migration_tracking()
        
        # Verify PostgreSQL table creation was called
        mock_session.execute.assert_called()
        mock_session.commit.assert_called()
        
        # Verify MongoDB index creation was called
        mock_collection.create_index.assert_called_with("version", unique=True)
    
    @patch('automl_framework.migrations.migration_manager.get_postgres_session')
    def test_get_applied_postgres_migrations(self, mock_get_postgres_session):
        """Test getting applied PostgreSQL migrations."""
        # Mock session and result
        mock_session = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [("001_initial_schema",), ("002_add_indexes",)]
        mock_session.execute.return_value = mock_result
        mock_get_postgres_session.return_value.__enter__.return_value = mock_session
        
        # Create migration manager
        db_manager = Mock()
        migration_manager = MigrationManager(db_manager)
        
        # Get applied migrations
        migrations = migration_manager.get_applied_postgres_migrations()
        
        assert migrations == ["001_initial_schema", "002_add_indexes"]
        mock_session.execute.assert_called()
    
    @patch('automl_framework.migrations.migration_manager.get_mongo_collection')
    def test_get_applied_mongo_migrations(self, mock_get_mongo_collection):
        """Test getting applied MongoDB migrations."""
        # Mock collection and cursor
        mock_collection = Mock()
        mock_cursor = Mock()
        mock_cursor.sort.return_value = [
            {"version": "001_initial_collections"},
            {"version": "002_add_indexes"}
        ]
        mock_collection.find.return_value = mock_cursor
        mock_get_mongo_collection.return_value = mock_collection
        
        # Create migration manager
        db_manager = Mock()
        migration_manager = MigrationManager(db_manager)
        
        # Get applied migrations
        migrations = migration_manager.get_applied_mongo_migrations()
        
        assert migrations == ["001_initial_collections", "002_add_indexes"]
        mock_collection.find.assert_called_with({}, {"version": 1})
        mock_cursor.sort.assert_called_with("applied_at", 1)
    
    @patch('automl_framework.migrations.migration_manager.get_postgres_session')
    def test_apply_postgres_migration(self, mock_get_postgres_session):
        """Test applying a PostgreSQL migration."""
        # Mock session
        mock_session = Mock()
        mock_get_postgres_session.return_value.__enter__.return_value = mock_session
        
        # Create migration manager
        db_manager = Mock()
        migration_manager = MigrationManager(db_manager)
        
        # Apply migration
        version = "001_test_migration"
        name = "Test Migration"
        sql_script = "CREATE TABLE test_table (id INTEGER);"
        
        migration_manager.apply_postgres_migration(version, name, sql_script)
        
        # Verify SQL execution
        assert mock_session.execute.call_count == 2  # One for script, one for recording
        mock_session.commit.assert_called()
    
    @patch('automl_framework.migrations.migration_manager.get_mongo_collection')
    def test_apply_mongo_migration(self, mock_get_mongo_collection):
        """Test applying a MongoDB migration."""
        # Mock collection
        mock_collection = Mock()
        mock_get_mongo_collection.return_value = mock_collection
        
        # Create migration manager
        db_manager = Mock()
        migration_manager = MigrationManager(db_manager)
        
        # Create mock migration function
        mock_migration_func = Mock()
        
        # Apply migration
        version = "001_test_migration"
        name = "Test Migration"
        
        migration_manager.apply_mongo_migration(version, name, mock_migration_func)
        
        # Verify migration function was called
        mock_migration_func.assert_called_once()
        
        # Verify migration was recorded
        mock_collection.insert_one.assert_called_once()
        call_args = mock_collection.insert_one.call_args[0][0]
        assert call_args["version"] == version
        assert call_args["name"] == name
        assert "applied_at" in call_args