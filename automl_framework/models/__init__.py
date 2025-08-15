"""
Models package for the AutoML framework.

This package contains all data models and database schemas used throughout the system.
"""

from .data_models import (
    # Data classes
    Dataset,
    Architecture,
    Experiment,
    TrainingConfig,
    PerformanceMetrics,
    ExperimentResults,
    Feature,
    Layer,
    Connection,
    
    # Enums
    DataType,
    ExperimentStatus,
    TaskType,
    LayerType,
)

from .orm_models import (
    # ORM Models
    User,
    DatasetORM,
    ExperimentORM,
    TrainedModelORM,
    HyperparameterTrialORM,
)

from .mongo_schemas import (
    # MongoDB Schemas
    ArchitectureDocument,
    TrainingLogDocument,
    TrainingLogEntry,
    
    # Repositories
    ArchitectureRepository,
    TrainingLogRepository,
    architecture_repo,
    training_log_repo,
)

__all__ = [
    # Data classes
    "Dataset",
    "Architecture", 
    "Experiment",
    "TrainingConfig",
    "PerformanceMetrics",
    "ExperimentResults",
    "Feature",
    "Layer",
    "Connection",
    
    # Enums
    "DataType",
    "ExperimentStatus",
    "TaskType",
    "LayerType",
    
    # ORM Models
    "User",
    "DatasetORM",
    "ExperimentORM",
    "TrainedModelORM",
    "HyperparameterTrialORM",
    
    # MongoDB Schemas
    "ArchitectureDocument",
    "TrainingLogDocument",
    "TrainingLogEntry",
    
    # Repositories
    "ArchitectureRepository",
    "TrainingLogRepository",
    "architecture_repo",
    "training_log_repo",
]