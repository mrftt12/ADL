"""
Database migration management for the AutoML framework.

This module provides utilities for managing database schema migrations
for both PostgreSQL and MongoDB.
"""

import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

from sqlalchemy import text, MetaData, Table, Column, String, DateTime, Integer
from sqlalchemy.exc import SQLAlchemyError

from automl_framework.core.database import DatabaseManager, get_postgres_session, get_mongo_collection

logger = logging.getLogger(__name__)


class MigrationManager:
    """Manages database migrations for both PostgreSQL and MongoDB."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.migrations_dir = Path(__file__).parent / "scripts"
        self.migrations_dir.mkdir(exist_ok=True)
    
    def initialize_migration_tracking(self):
        """Initialize migration tracking tables/collections."""
        self._init_postgres_migration_table()
        self._init_mongo_migration_collection()
    
    def _init_postgres_migration_table(self):
        """Create migration tracking table in PostgreSQL."""
        try:
            with get_postgres_session() as session:
                # Create migrations table if it doesn't exist
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        id SERIAL PRIMARY KEY,
                        version VARCHAR(50) UNIQUE NOT NULL,
                        name VARCHAR(200) NOT NULL,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        checksum VARCHAR(64)
                    )
                """))
                session.commit()
                logger.info("PostgreSQL migration tracking table initialized")
        except SQLAlchemyError as e:
            logger.error(f"Failed to initialize PostgreSQL migration table: {e}")
            raise
    
    def _init_mongo_migration_collection(self):
        """Create migration tracking collection in MongoDB."""
        try:
            collection = get_mongo_collection("schema_migrations")
            # Create index on version field
            collection.create_index("version", unique=True)
            logger.info("MongoDB migration tracking collection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB migration collection: {e}")
            raise
    
    def get_applied_postgres_migrations(self) -> List[str]:
        """Get list of applied PostgreSQL migrations."""
        try:
            with get_postgres_session() as session:
                result = session.execute(text(
                    "SELECT version FROM schema_migrations ORDER BY applied_at"
                ))
                return [row[0] for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get applied PostgreSQL migrations: {e}")
            return []
    
    def get_applied_mongo_migrations(self) -> List[str]:
        """Get list of applied MongoDB migrations."""
        try:
            collection = get_mongo_collection("schema_migrations")
            cursor = collection.find({}, {"version": 1}).sort("applied_at", 1)
            return [doc["version"] for doc in cursor]
        except Exception as e:
            logger.error(f"Failed to get applied MongoDB migrations: {e}")
            return []
    
    def apply_postgres_migration(self, version: str, name: str, sql_script: str):
        """Apply a PostgreSQL migration."""
        try:
            with get_postgres_session() as session:
                # Execute migration script
                session.execute(text(sql_script))
                
                # Record migration
                session.execute(text("""
                    INSERT INTO schema_migrations (version, name, applied_at)
                    VALUES (:version, :name, :applied_at)
                """), {
                    "version": version,
                    "name": name,
                    "applied_at": datetime.utcnow()
                })
                
                session.commit()
                logger.info(f"Applied PostgreSQL migration: {version} - {name}")
        except SQLAlchemyError as e:
            logger.error(f"Failed to apply PostgreSQL migration {version}: {e}")
            raise
    
    def apply_mongo_migration(self, version: str, name: str, migration_func):
        """Apply a MongoDB migration."""
        try:
            # Execute migration function
            migration_func()
            
            # Record migration
            collection = get_mongo_collection("schema_migrations")
            collection.insert_one({
                "version": version,
                "name": name,
                "applied_at": datetime.utcnow()
            })
            
            logger.info(f"Applied MongoDB migration: {version} - {name}")
        except Exception as e:
            logger.error(f"Failed to apply MongoDB migration {version}: {e}")
            raise
    
    def run_initial_setup(self):
        """Run initial database setup."""
        logger.info("Starting initial database setup...")
        
        # Initialize migration tracking
        self.initialize_migration_tracking()
        
        # Apply initial PostgreSQL migrations
        self._apply_initial_postgres_migrations()
        
        # Apply initial MongoDB migrations
        self._apply_initial_mongo_migrations()
        
        logger.info("Initial database setup completed")
    
    def _apply_initial_postgres_migrations(self):
        """Apply initial PostgreSQL schema migrations."""
        applied_migrations = self.get_applied_postgres_migrations()
        
        # Migration 001: Create initial tables
        if "001_initial_schema" not in applied_migrations:
            initial_schema_sql = """
            -- Users table
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                full_name VARCHAR(100),
                is_active BOOLEAN DEFAULT TRUE NOT NULL,
                is_admin BOOLEAN DEFAULT FALSE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                last_login TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
            
            -- Datasets table
            CREATE TABLE IF NOT EXISTS datasets (
                id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(200) NOT NULL,
                file_path VARCHAR(500) NOT NULL,
                data_type VARCHAR(20) NOT NULL,
                size INTEGER NOT NULL,
                target_column VARCHAR(100),
                features_json JSONB,
                metadata_json JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                owner_id INTEGER NOT NULL REFERENCES users(id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_datasets_owner_id ON datasets(owner_id);
            CREATE INDEX IF NOT EXISTS idx_datasets_data_type ON datasets(data_type);
            CREATE INDEX IF NOT EXISTS idx_dataset_owner_created ON datasets(owner_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_dataset_type_size ON datasets(data_type, size);
            
            -- Experiments table
            CREATE TABLE IF NOT EXISTS experiments (
                id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(200) NOT NULL,
                status VARCHAR(20) DEFAULT 'created' NOT NULL,
                task_type VARCHAR(50),
                config_json JSONB,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                dataset_id VARCHAR(50) NOT NULL REFERENCES datasets(id),
                owner_id INTEGER NOT NULL REFERENCES users(id),
                best_model_id INTEGER
            );
            
            CREATE INDEX IF NOT EXISTS idx_experiments_dataset_id ON experiments(dataset_id);
            CREATE INDEX IF NOT EXISTS idx_experiments_owner_id ON experiments(owner_id);
            CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
            CREATE INDEX IF NOT EXISTS idx_experiment_owner_status ON experiments(owner_id, status);
            CREATE INDEX IF NOT EXISTS idx_experiment_dataset_created ON experiments(dataset_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_experiment_status_created ON experiments(status, created_at);
            
            -- Trained models table
            CREATE TABLE IF NOT EXISTS trained_models (
                id SERIAL PRIMARY KEY,
                name VARCHAR(200) NOT NULL,
                architecture_id VARCHAR(50) NOT NULL,
                model_path VARCHAR(500),
                hyperparameters_json JSONB,
                accuracy FLOAT,
                loss FLOAT,
                precision FLOAT,
                recall FLOAT,
                f1_score FLOAT,
                training_time FLOAT,
                inference_time FLOAT,
                additional_metrics_json JSONB,
                training_config_json JSONB,
                epochs_trained INTEGER,
                best_epoch INTEGER,
                parameter_count INTEGER,
                model_size_mb FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                experiment_id VARCHAR(50) NOT NULL REFERENCES experiments(id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_trained_models_experiment_id ON trained_models(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_trained_models_architecture_id ON trained_models(architecture_id);
            CREATE INDEX IF NOT EXISTS idx_model_experiment_accuracy ON trained_models(experiment_id, accuracy);
            CREATE INDEX IF NOT EXISTS idx_model_architecture_performance ON trained_models(architecture_id, accuracy, f1_score);
            CREATE INDEX IF NOT EXISTS idx_model_created ON trained_models(created_at);
            
            -- Add foreign key constraint for best_model_id
            ALTER TABLE experiments ADD CONSTRAINT fk_experiments_best_model 
                FOREIGN KEY (best_model_id) REFERENCES trained_models(id);
            
            -- Hyperparameter trials table
            CREATE TABLE IF NOT EXISTS hyperparameter_trials (
                id SERIAL PRIMARY KEY,
                trial_number INTEGER NOT NULL,
                hyperparameters_json JSONB NOT NULL,
                objective_value FLOAT,
                status VARCHAR(20) DEFAULT 'completed' NOT NULL,
                duration FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                experiment_id VARCHAR(50) NOT NULL REFERENCES experiments(id),
                UNIQUE(experiment_id, trial_number)
            );
            
            CREATE INDEX IF NOT EXISTS idx_hyperparameter_trials_experiment_id ON hyperparameter_trials(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_trial_experiment_number ON hyperparameter_trials(experiment_id, trial_number);
            CREATE INDEX IF NOT EXISTS idx_trial_experiment_objective ON hyperparameter_trials(experiment_id, objective_value);
            """
            
            self.apply_postgres_migration("001_initial_schema", "Create initial database schema", initial_schema_sql)
    
    def _apply_initial_mongo_migrations(self):
        """Apply initial MongoDB migrations."""
        applied_migrations = self.get_applied_mongo_migrations()
        
        # Migration 001: Create initial collections and indexes
        if "001_initial_collections" not in applied_migrations:
            def create_initial_collections():
                # Create architectures collection indexes
                arch_collection = get_mongo_collection("architectures")
                arch_collection.create_index("id", unique=True)
                arch_collection.create_index("task_type")
                arch_collection.create_index("parameter_count")
                arch_collection.create_index("created_at")
                
                # Create training_logs collection indexes
                logs_collection = get_mongo_collection("training_logs")
                logs_collection.create_index("experiment_id")
                logs_collection.create_index("status")
                logs_collection.create_index("start_time")
                logs_collection.create_index([("experiment_id", 1), ("start_time", 1)])
                
                logger.info("Created initial MongoDB collections and indexes")
            
            self.apply_mongo_migration("001_initial_collections", "Create initial collections and indexes", create_initial_collections)
    
    def create_sample_data(self):
        """Create sample data for development and testing."""
        logger.info("Creating sample data...")
        
        try:
            with get_postgres_session() as session:
                # Create sample user
                session.execute(text("""
                    INSERT INTO users (username, email, password_hash, full_name, is_admin)
                    VALUES ('admin', 'admin@automl.com', 'hashed_password', 'Admin User', TRUE)
                    ON CONFLICT (username) DO NOTHING
                """))
                
                # Create sample dataset
                session.execute(text("""
                    INSERT INTO datasets (id, name, file_path, data_type, size, owner_id)
                    VALUES ('sample-dataset-1', 'Sample Dataset', '/data/sample.csv', 'tabular', 1000, 1)
                    ON CONFLICT (id) DO NOTHING
                """))
                
                session.commit()
                logger.info("Sample data created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create sample data: {e}")
            raise


def run_migrations():
    """Run all pending migrations."""
    db_manager = DatabaseManager()
    migration_manager = MigrationManager(db_manager)
    
    try:
        migration_manager.run_initial_setup()
        logger.info("All migrations completed successfully")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        db_manager.close_all()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run migrations
    run_migrations()