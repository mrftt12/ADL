"""
Core interfaces and abstract base classes for AutoML Framework components
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd


# Enums
class DataType(Enum):
    IMAGE = "image"
    TEXT = "text"
    TABULAR = "tabular"
    TIME_SERIES = "time_series"


class ExperimentStatus(Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    OBJECT_DETECTION = "object_detection"
    TEXT_CLASSIFICATION = "text_classification"


# Core Data Models
@dataclass
class Feature:
    name: str
    data_type: str
    is_categorical: bool
    missing_values: int
    unique_values: int


@dataclass
class DatasetMetadata:
    id: str
    name: str
    data_type: DataType
    task_type: TaskType
    size: int
    features: List[Feature]
    target_column: Optional[str]
    class_distribution: Optional[Dict[str, int]]
    statistics: Dict[str, Any]


@dataclass
class ProcessedData:
    train_data: pd.DataFrame
    validation_data: pd.DataFrame
    test_data: pd.DataFrame
    preprocessing_pipeline: Any
    feature_names: List[str]


@dataclass
class Layer:
    layer_type: str
    parameters: Dict[str, Any]
    input_shape: Optional[Tuple[int, ...]]
    output_shape: Optional[Tuple[int, ...]]


@dataclass
class Connection:
    from_layer: int
    to_layer: int
    connection_type: str


@dataclass
class Architecture:
    id: str
    layers: List[Layer]
    connections: List[Connection]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    parameter_count: int
    flops: int


@dataclass
class SearchSpace:
    layer_types: List[str]
    max_layers: int
    parameter_ranges: Dict[str, Tuple[Any, Any]]
    constraints: Dict[str, Any]


@dataclass
class HyperparameterSpace:
    parameters: Dict[str, Tuple[Any, Any]]
    parameter_types: Dict[str, str]
    constraints: List[Dict[str, Any]]


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    optimizer: str
    epochs: int
    early_stopping_patience: int
    regularization: Dict[str, float]
    use_mixed_precision: bool = False


@dataclass
class PerformanceMetrics:
    accuracy: float
    loss: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    inference_time: float
    additional_metrics: Dict[str, float]


@dataclass
class TrainedModel:
    id: str
    architecture: Architecture
    config: TrainingConfig
    metrics: PerformanceMetrics
    model_path: str
    preprocessing_pipeline_path: str


@dataclass
class TrainingJob:
    id: str
    experiment_id: str
    architecture: Architecture
    config: TrainingConfig
    status: str
    created_at: str
    gpu_allocation: List[int]


@dataclass
class Trial:
    id: str
    parameters: Dict[str, Any]
    metrics: PerformanceMetrics
    status: str
    duration: float


# Core Interfaces
class IDataProcessor(ABC):
    """Interface for data processing and preprocessing"""
    
    @abstractmethod
    def analyze_dataset(self, dataset_path: str) -> DatasetMetadata:
        """Analyze dataset characteristics and return metadata"""
        pass
    
    @abstractmethod
    def create_preprocessing_pipeline(self, metadata: DatasetMetadata) -> Any:
        """Create preprocessing pipeline based on dataset metadata"""
        pass
    
    @abstractmethod
    def apply_preprocessing(self, pipeline: Any, data: pd.DataFrame) -> ProcessedData:
        """Apply preprocessing pipeline to data"""
        pass


class INASService(ABC):
    """Interface for Neural Architecture Search service"""
    
    @abstractmethod
    def define_search_space(self, task_type: TaskType, data_type: DataType) -> SearchSpace:
        """Define search space for architecture search"""
        pass
    
    @abstractmethod
    def search_architectures(self, search_space: SearchSpace, dataset_metadata: DatasetMetadata) -> List[Architecture]:
        """Search for optimal architectures"""
        pass
    
    @abstractmethod
    def evaluate_architecture(self, architecture: Architecture, dataset_metadata: DatasetMetadata) -> PerformanceMetrics:
        """Evaluate architecture performance"""
        pass


class IHyperparameterOptimizer(ABC):
    """Interface for hyperparameter optimization service"""
    
    @abstractmethod
    def define_search_space(self, architecture: Architecture) -> HyperparameterSpace:
        """Define hyperparameter search space"""
        pass
    
    @abstractmethod
    def optimize(self, objective_function: Callable, search_space: HyperparameterSpace, max_trials: int) -> TrainingConfig:
        """Optimize hyperparameters using specified objective function"""
        pass
    
    @abstractmethod
    def get_optimization_history(self) -> List[Trial]:
        """Get history of optimization trials"""
        pass


class IModelTrainer(ABC):
    """Interface for model training service"""
    
    @abstractmethod
    def train_model(self, architecture: Architecture, config: TrainingConfig, processed_data: ProcessedData) -> TrainedModel:
        """Train model with given architecture and configuration"""
        pass
    
    @abstractmethod
    def monitor_training(self, training_job: TrainingJob) -> Dict[str, Any]:
        """Monitor training progress and return current metrics"""
        pass
    
    @abstractmethod
    def save_checkpoint(self, model: Any, epoch: int, checkpoint_path: str) -> str:
        """Save model checkpoint"""
        pass


class IModelEvaluator(ABC):
    """Interface for model evaluation service"""
    
    @abstractmethod
    def evaluate_model(self, model: TrainedModel, test_data: pd.DataFrame) -> PerformanceMetrics:
        """Evaluate trained model on test data"""
        pass
    
    @abstractmethod
    def compare_models(self, models: List[TrainedModel]) -> Dict[str, Any]:
        """Compare multiple models and return ranking"""
        pass
    
    @abstractmethod
    def generate_report(self, model: TrainedModel, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        pass


class IResourceScheduler(ABC):
    """Interface for resource scheduling and management"""
    
    @abstractmethod
    def allocate_resources(self, job_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate computational resources for a job"""
        pass
    
    @abstractmethod
    def release_resources(self, job_id: str) -> bool:
        """Release resources allocated to a job"""
        pass
    
    @abstractmethod
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource utilization status"""
        pass


class IExperimentManager(ABC):
    """Interface for experiment orchestration and management"""
    
    @abstractmethod
    def create_experiment(self, dataset_path: str, experiment_config: Dict[str, Any]) -> str:
        """Create new AutoML experiment"""
        pass
    
    @abstractmethod
    def run_experiment(self, experiment_id: str) -> bool:
        """Run AutoML experiment pipeline"""
        pass
    
    @abstractmethod
    def get_experiment_status(self, experiment_id: str) -> ExperimentStatus:
        """Get current experiment status"""
        pass
    
    @abstractmethod
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment results and best model"""
        pass