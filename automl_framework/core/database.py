"""
Database connection utilities for PostgreSQL and MongoDB.

This module provides connection management and configuration for both
relational (PostgreSQL) and document (MongoDB) databases used by the AutoML framework.
"""

import os
from typing import Optional, Dict, Any
from contextlib import contextmanager
import logging

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import QueuePool
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

logger = logging.getLogger(__name__)

# SQLAlchemy base for ORM models
Base = declarative_base()

class DatabaseConfig:
    """Configuration class for database connections."""
    
    def __init__(self):
        # PostgreSQL configuration
        self.postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        self.postgres_port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.postgres_db = os.getenv("POSTGRES_DB", "automl")
        self.postgres_user = os.getenv("POSTGRES_USER", "automl_user")
        self.postgres_password = os.getenv("POSTGRES_PASSWORD", "automl_pass")
        
        # MongoDB configuration
        self.mongo_host = os.getenv("MONGO_HOST", "localhost")
        self.mongo_port = int(os.getenv("MONGO_PORT", "27017"))
        self.mongo_db = os.getenv("MONGO_DB", "automl")
        self.mongo_user = os.getenv("MONGO_USER", "")
        self.mongo_password = os.getenv("MONGO_PASSWORD", "")
        
        # Connection pool settings
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
        self.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    
    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def mongo_url(self) -> str:
        """Get MongoDB connection URL."""
        if self.mongo_user and self.mongo_password:
            return (
                f"mongodb://{self.mongo_user}:{self.mongo_password}"
                f"@{self.mongo_host}:{self.mongo_port}/{self.mongo_db}"
            )
        return f"mongodb://{self.mongo_host}:{self.mongo_port}/{self.mongo_db}"


class PostgreSQLManager:
    """Manager for PostgreSQL database connections."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._engine = None
        self._session_factory = None
    
    @property
    def engine(self):
        """Get or create SQLAlchemy engine."""
        if self._engine is None:
            self._engine = create_engine(
                self.config.postgres_url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_pre_ping=True,  # Verify connections before use
                echo=os.getenv("SQL_ECHO", "false").lower() == "true"
            )
            logger.info("PostgreSQL engine created")
        return self._engine
    
    @property
    def session_factory(self):
        """Get or create session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory
    
    @contextmanager
    def get_session(self):
        """Context manager for database sessions."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def create_tables(self):
        """Create all tables defined in the Base metadata."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("PostgreSQL tables created successfully")
        except Exception as e:
            logger.error(f"Error creating PostgreSQL tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all tables defined in the Base metadata."""
        try:
            Base.metadata.drop_all(self.engine)
            logger.info("PostgreSQL tables dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping PostgreSQL tables: {e}")
            raise
    
    def close(self):
        """Close database connections."""
        if self._engine:
            self._engine.dispose()
            logger.info("PostgreSQL connections closed")


class MongoDBManager:
    """Manager for MongoDB database connections."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._client = None
        self._database = None
    
    @property
    def client(self) -> MongoClient:
        """Get or create MongoDB client."""
        if self._client is None:
            self._client = MongoClient(
                self.config.mongo_url,
                maxPoolSize=self.config.pool_size,
                serverSelectionTimeoutMS=self.config.pool_timeout * 1000
            )
            logger.info("MongoDB client created")
        return self._client
    
    @property
    def database(self) -> Database:
        """Get MongoDB database."""
        if self._database is None:
            self._database = self.client[self.config.mongo_db]
        return self._database
    
    def get_collection(self, collection_name: str) -> Collection:
        """Get a specific collection."""
        return self.database[collection_name]
    
    def create_indexes(self):
        """Create indexes for MongoDB collections."""
        try:
            # Architectures collection indexes
            architectures = self.get_collection("architectures")
            architectures.create_index("id", unique=True)
            architectures.create_index("task_type")
            architectures.create_index("parameter_count")
            
            # Training logs collection indexes
            training_logs = self.get_collection("training_logs")
            training_logs.create_index("experiment_id")
            training_logs.create_index("timestamp")
            training_logs.create_index([("experiment_id", 1), ("timestamp", 1)])
            
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating MongoDB indexes: {e}")
            raise
    
    def drop_collections(self):
        """Drop all collections."""
        try:
            collection_names = self.database.list_collection_names()
            for name in collection_names:
                self.database.drop_collection(name)
            logger.info("MongoDB collections dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping MongoDB collections: {e}")
            raise
    
    def close(self):
        """Close database connections."""
        if self._client:
            self._client.close()
            logger.info("MongoDB connections closed")


class DatabaseManager:
    """Unified database manager for both PostgreSQL and MongoDB."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.postgres = PostgreSQLManager(self.config)
        self.mongodb = MongoDBManager(self.config)
    
    def initialize(self):
        """Initialize both databases."""
        try:
            # Initialize PostgreSQL
            self.postgres.create_tables()
            
            # Initialize MongoDB
            self.mongodb.create_indexes()
            
            logger.info("Database initialization completed successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of both database connections."""
        health = {"postgresql": False, "mongodb": False}
        
        # Check PostgreSQL
        try:
            with self.postgres.get_session() as session:
                session.execute("SELECT 1")
            health["postgresql"] = True
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
        
        # Check MongoDB
        try:
            self.mongodb.client.admin.command('ping')
            health["mongodb"] = True
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
        
        return health
    
    def close_all(self):
        """Close all database connections."""
        self.postgres.close()
        self.mongodb.close()
        logger.info("All database connections closed")


# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions for getting database sessions/connections
def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    return db_manager

def get_postgres_session():
    """Get PostgreSQL session context manager."""
    return db_manager.postgres.get_session()

def get_mongo_collection(collection_name: str) -> Collection:
    """Get MongoDB collection."""
    return db_manager.mongodb.get_collection(collection_name)

def get_db_session():
    """Alias for get_postgres_session for backward compatibility."""
    return get_postgres_session()