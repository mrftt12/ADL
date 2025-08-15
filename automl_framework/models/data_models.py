"""
Core data models for the AutoML framework.

This module contains the main data structures used throughout the system,
including Dataset, Architecture, Experiment, and TrainingConfig classes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import re


class DataType(Enum):
    """Enumeration of supported data types."""
    IMAGE = "image"
    TEXT = "text"
    TABULAR = "tabular"
    TIME_SERIES = "time_series"


class ExperimentStatus(Enum):
    """Enumeration of experiment status values."""
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Enumeration of machine learning task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    NATURAL_LANGUAGE_PROCESSING = "nlp"
    TIME_SERIES_FORECASTING = "time_series_forecasting"


class LayerType(Enum):
    """Enumeration of neural network layer types."""
    DENSE = "dense"
    CONV2D = "conv2d"
    CONV1D = "conv1d"
    LSTM = "lstm"
    GRU = "gru"
    ATTENTION = "attention"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    POOLING = "pooling"


@dataclass
class Feature:
    """Represents a feature in a dataset."""
    name: str
    data_type: str
    is_categorical: bool = False
    unique_values: Optional[int] = None
    missing_percentage: float = 0.0
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate feature data."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Feature name must be a non-empty string")
        
        if not self.data_type or not isinstance(self.data_type, str):
            raise ValueError("Feature data_type must be a non-empty string")
        
        if not 0 <= self.missing_percentage <= 100:
            raise ValueError("Missing percentage must be between 0 and 100")
        
        if self.unique_values is not None and self.unique_values < 0:
            raise ValueError("Unique values count cannot be negative")


@dataclass
class Layer:
    """Represents a neural network layer."""
    layer_type: LayerType
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    
    def validate(self) -> None:
        """Validate layer configuration."""
        if not isinstance(self.layer_type, LayerType):
            raise ValueError("layer_type must be a LayerType enum")
        
        if not isinstance(self.parameters, dict):
            raise ValueError("parameters must be a dictionary")


@dataclass
class Connection:
    """Represents a connection between neural network layers."""
    from_layer: int
    to_layer: int
    connection_type: str = "sequential"
    
    def validate(self) -> None:
        """Validate connection."""
        if self.from_layer < 0 or self.to_layer < 0:
            raise ValueError("Layer indices must be non-negative")
        
        if self.from_layer == self.to_layer:
            raise ValueError("Layer cannot connect to itself")
        
        if not self.connection_type:
            raise ValueError("Connection type must be specified")


@dataclass
class Dataset:
    """Represents a dataset in the AutoML system."""
    id: str
    name: str
    file_path: str
    data_type: DataType
    size: int
    features: List[Feature] = field(default_factory=list)
    target_column: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def validate(self) -> None:
        """Validate dataset data."""
        if not self.id or not isinstance(self.id, str):
            raise ValueError("Dataset ID must be a non-empty string")
        
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Dataset name must be a non-empty string")
        
        if not self.file_path or not isinstance(self.file_path, str):
            raise ValueError("File path must be a non-empty string")
        
        if not isinstance(self.data_type, DataType):
            raise ValueError("data_type must be a DataType enum")
        
        if self.size <= 0:
            raise ValueError("Dataset size must be positive")
        
        # Validate features
        for feature in self.features:
            feature.validate()
        
        # Validate target column exists in features if specified
        if self.target_column:
            feature_names = [f.name for f in self.features]
            if self.target_column not in feature_names:
                raise ValueError(f"Target column '{self.target_column}' not found in features")


@dataclass
class Architecture:
    """Represents a neural network architecture."""
    id: str
    layers: List[Layer]
    connections: List[Connection] = field(default_factory=list)
    input_shape: Tuple[int, ...] = field(default_factory=tuple)
    output_shape: Tuple[int, ...] = field(default_factory=tuple)
    parameter_count: int = 0
    flops: int = 0
    task_type: Optional[TaskType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate architecture configuration."""
        if not self.id or not isinstance(self.id, str):
            raise ValueError("Architecture ID must be a non-empty string")
        
        if not self.layers:
            raise ValueError("Architecture must have at least one layer")
        
        # Validate all layers
        for i, layer in enumerate(self.layers):
            try:
                layer.validate()
            except ValueError as e:
                raise ValueError(f"Layer {i} validation failed: {e}")
        
        # Validate all connections
        max_layer_index = len(self.layers) - 1
        for connection in self.connections:
            connection.validate()
            if connection.from_layer > max_layer_index or connection.to_layer > max_layer_index:
                raise ValueError(f"Connection references non-existent layer")
        
        if self.parameter_count < 0:
            raise ValueError("Parameter count cannot be negative")
        
        if self.flops < 0:
            raise ValueError("FLOPS count cannot be negative")


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int
    learning_rate: float
    optimizer: str
    epochs: int
    early_stopping_patience: int = 10
    regularization: Dict[str, float] = field(default_factory=dict)
    validation_split: float = 0.2
    shuffle: bool = True
    random_seed: Optional[int] = None
    
    def validate(self) -> None:
        """Validate training configuration."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if not self.optimizer or not isinstance(self.optimizer, str):
            raise ValueError("Optimizer must be a non-empty string")
        
        valid_optimizers = ["adam", "sgd", "rmsprop", "adagrad", "adamw"]
        if self.optimizer.lower() not in valid_optimizers:
            raise ValueError(f"Optimizer must be one of: {valid_optimizers}")
        
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")
        
        if self.early_stopping_patience < 0:
            raise ValueError("Early stopping patience cannot be negative")
        
        if not 0 < self.validation_split < 1:
            raise ValueError("Validation split must be between 0 and 1")
        
        # Validate regularization parameters
        for reg_type, reg_value in self.regularization.items():
            if reg_value < 0:
                raise ValueError(f"Regularization value for {reg_type} cannot be negative")


@dataclass
class PerformanceMetrics:
    """Performance metrics for model evaluation."""
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    training_time: Optional[float] = None
    inference_time: Optional[float] = None
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate performance metrics."""
        metrics = [self.accuracy, self.precision, self.recall, self.f1_score]
        for metric in metrics:
            if metric is not None and not 0 <= metric <= 1:
                raise ValueError("Accuracy, precision, recall, and F1 score must be between 0 and 1")
        
        if self.loss is not None and self.loss < 0:
            raise ValueError("Loss cannot be negative")
        
        if self.training_time is not None and self.training_time < 0:
            raise ValueError("Training time cannot be negative")
        
        if self.inference_time is not None and self.inference_time < 0:
            raise ValueError("Inference time cannot be negative")


@dataclass
class ExperimentResults:
    """Results of an AutoML experiment."""
    best_architecture: Optional[Architecture] = None
    best_hyperparameters: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    training_history: List[Dict[str, float]] = field(default_factory=list)
    model_path: Optional[str] = None
    
    def validate(self) -> None:
        """Validate experiment results."""
        if self.best_architecture:
            self.best_architecture.validate()
        
        if self.performance_metrics:
            self.performance_metrics.validate()


@dataclass
class Experiment:
    """Represents an AutoML experiment."""
    id: str
    name: str
    dataset_id: str
    status: ExperimentStatus = ExperimentStatus.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    results: Optional[ExperimentResults] = None
    config: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def validate(self) -> None:
        """Validate experiment data."""
        if not self.id or not isinstance(self.id, str):
            raise ValueError("Experiment ID must be a non-empty string")
        
        # Validate ID format (alphanumeric with hyphens/underscores)
        if not re.match(r'^[a-zA-Z0-9_-]+$', self.id):
            raise ValueError("Experiment ID must contain only alphanumeric characters, hyphens, and underscores")
        
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Experiment name must be a non-empty string")
        
        if not self.dataset_id or not isinstance(self.dataset_id, str):
            raise ValueError("Dataset ID must be a non-empty string")
        
        if not isinstance(self.status, ExperimentStatus):
            raise ValueError("status must be an ExperimentStatus enum")
        
        # Validate completed_at is after created_at if both exist
        if self.completed_at and self.completed_at < self.created_at:
            raise ValueError("Completion time cannot be before creation time")
        
        # Validate results if present
        if self.results:
            self.results.validate()
        
        # Status-specific validations
        if self.status == ExperimentStatus.COMPLETED and not self.completed_at:
            raise ValueError("Completed experiments must have a completion time")
        
        if self.status == ExperimentStatus.FAILED and not self.error_message:
            raise ValueError("Failed experiments must have an error message")