"""
SQLAlchemy ORM models for PostgreSQL database.

This module contains the database models for experiments, datasets, models, and users
that are stored in PostgreSQL.
"""

from datetime import datetime
from typing import Optional, Dict, Any
import json

from sqlalchemy import (
    Column, Integer, String, DateTime, Text, Boolean, Float, 
    ForeignKey, JSON, Enum as SQLEnum, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship

from automl_framework.core.database import Base
from automl_framework.models.data_models import DataType, ExperimentStatus, TaskType


class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime)
    
    # Relationships
    datasets = relationship("DatasetORM", back_populates="owner", cascade="all, delete-orphan")
    experiments = relationship("ExperimentORM", back_populates="owner", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"


class DatasetORM(Base):
    """Dataset model for storing dataset metadata."""
    
    __tablename__ = "datasets"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(200), nullable=False)
    file_path = Column(String(500), nullable=False)
    data_type = Column(SQLEnum(DataType), nullable=False, index=True)
    size = Column(Integer, nullable=False)
    target_column = Column(String(100))
    features_json = Column(JSON)  # Serialized features list
    metadata_json = Column(JSON)  # Additional metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign keys
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Relationships
    owner = relationship("User", back_populates="datasets")
    experiments = relationship("ExperimentORM", back_populates="dataset", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_dataset_owner_created", "owner_id", "created_at"),
        Index("idx_dataset_type_size", "data_type", "size"),
    )
    
    @property
    def features(self):
        """Get features as Python objects."""
        if self.features_json:
            return json.loads(self.features_json) if isinstance(self.features_json, str) else self.features_json
        return []
    
    @features.setter
    def features(self, value):
        """Set features from Python objects."""
        self.features_json = json.dumps(value) if value else None
    
    @property
    def dataset_metadata(self):
        """Get metadata as Python dict."""
        if self.metadata_json:
            return json.loads(self.metadata_json) if isinstance(self.metadata_json, str) else self.metadata_json
        return {}
    
    @dataset_metadata.setter
    def dataset_metadata(self, value):
        """Set metadata from Python dict."""
        self.metadata_json = json.dumps(value) if value else None
    
    def __repr__(self):
        return f"<DatasetORM(id='{self.id}', name='{self.name}', type={self.data_type})>"


class ExperimentORM(Base):
    """Experiment model for storing experiment metadata and results."""
    
    __tablename__ = "experiments"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(200), nullable=False)
    status = Column(SQLEnum(ExperimentStatus), default=ExperimentStatus.CREATED, nullable=False, index=True)
    task_type = Column(SQLEnum(TaskType), index=True)
    config_json = Column(JSON)  # Experiment configuration
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Foreign keys
    dataset_id = Column(String(50), ForeignKey("datasets.id"), nullable=False, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    best_model_id = Column(Integer, ForeignKey("trained_models.id"))
    
    # Relationships
    dataset = relationship("DatasetORM", back_populates="experiments")
    owner = relationship("User", back_populates="experiments")
    best_model = relationship("TrainedModelORM", foreign_keys=[best_model_id])
    models = relationship("TrainedModelORM", foreign_keys="TrainedModelORM.experiment_id", back_populates="experiment")
    
    # Indexes
    __table_args__ = (
        Index("idx_experiment_owner_status", "owner_id", "status"),
        Index("idx_experiment_dataset_created", "dataset_id", "created_at"),
        Index("idx_experiment_status_created", "status", "created_at"),
    )
    
    @property
    def config(self):
        """Get config as Python dict."""
        if self.config_json:
            return json.loads(self.config_json) if isinstance(self.config_json, str) else self.config_json
        return {}
    
    @config.setter
    def config(self, value):
        """Set config from Python dict."""
        self.config_json = json.dumps(value) if value else None
    
    @property
    def duration(self) -> Optional[float]:
        """Get experiment duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def __repr__(self):
        return f"<ExperimentORM(id='{self.id}', name='{self.name}', status={self.status})>"


class TrainedModelORM(Base):
    """Trained model metadata and performance metrics."""
    
    __tablename__ = "trained_models"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    architecture_id = Column(String(50), nullable=False, index=True)  # Reference to MongoDB
    model_path = Column(String(500))  # Path to saved model files
    hyperparameters_json = Column(JSON)
    
    # Performance metrics
    accuracy = Column(Float)
    loss = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    training_time = Column(Float)  # in seconds
    inference_time = Column(Float)  # in milliseconds
    additional_metrics_json = Column(JSON)
    
    # Training metadata
    training_config_json = Column(JSON)
    epochs_trained = Column(Integer)
    best_epoch = Column(Integer)
    parameter_count = Column(Integer)
    model_size_mb = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Foreign keys
    experiment_id = Column(String(50), ForeignKey("experiments.id"), nullable=False, index=True)
    
    # Relationships
    experiment = relationship("ExperimentORM", foreign_keys=[experiment_id], back_populates="models")
    
    # Indexes
    __table_args__ = (
        Index("idx_model_experiment_accuracy", "experiment_id", "accuracy"),
        Index("idx_model_architecture_performance", "architecture_id", "accuracy", "f1_score"),
        Index("idx_model_created", "created_at"),
    )
    
    @property
    def hyperparameters(self):
        """Get hyperparameters as Python dict."""
        if self.hyperparameters_json:
            return json.loads(self.hyperparameters_json) if isinstance(self.hyperparameters_json, str) else self.hyperparameters_json
        return {}
    
    @hyperparameters.setter
    def hyperparameters(self, value):
        """Set hyperparameters from Python dict."""
        self.hyperparameters_json = json.dumps(value) if value else None
    
    @property
    def training_config(self):
        """Get training config as Python dict."""
        if self.training_config_json:
            return json.loads(self.training_config_json) if isinstance(self.training_config_json, str) else self.training_config_json
        return {}
    
    @training_config.setter
    def training_config(self, value):
        """Set training config from Python dict."""
        self.training_config_json = json.dumps(value) if value else None
    
    @property
    def additional_metrics(self):
        """Get additional metrics as Python dict."""
        if self.additional_metrics_json:
            return json.loads(self.additional_metrics_json) if isinstance(self.additional_metrics_json, str) else self.additional_metrics_json
        return {}
    
    @additional_metrics.setter
    def additional_metrics(self, value):
        """Set additional metrics from Python dict."""
        self.additional_metrics_json = json.dumps(value) if value else None
    
    def __repr__(self):
        return f"<TrainedModelORM(id={self.id}, name='{self.name}', accuracy={self.accuracy})>"


class HyperparameterTrialORM(Base):
    """Hyperparameter optimization trial results."""
    
    __tablename__ = "hyperparameter_trials"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trial_number = Column(Integer, nullable=False)
    hyperparameters_json = Column(JSON, nullable=False)
    objective_value = Column(Float)  # The metric being optimized
    status = Column(String(20), default="completed", nullable=False)  # completed, failed, pruned
    duration = Column(Float)  # Trial duration in seconds
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Foreign keys
    experiment_id = Column(String(50), ForeignKey("experiments.id"), nullable=False, index=True)
    
    # Relationships
    experiment = relationship("ExperimentORM")
    
    # Indexes
    __table_args__ = (
        Index("idx_trial_experiment_number", "experiment_id", "trial_number"),
        Index("idx_trial_experiment_objective", "experiment_id", "objective_value"),
        UniqueConstraint("experiment_id", "trial_number", name="uq_experiment_trial"),
    )
    
    @property
    def hyperparameters(self):
        """Get hyperparameters as Python dict."""
        if self.hyperparameters_json:
            return json.loads(self.hyperparameters_json) if isinstance(self.hyperparameters_json, str) else self.hyperparameters_json
        return {}
    
    @hyperparameters.setter
    def hyperparameters(self, value):
        """Set hyperparameters from Python dict."""
        self.hyperparameters_json = json.dumps(value) if value else None
    
    def __repr__(self):
        return f"<HyperparameterTrialORM(id={self.id}, trial={self.trial_number}, objective={self.objective_value})>"