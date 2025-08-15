"""
MongoDB document schemas for the AutoML framework.

This module defines the document structures for MongoDB collections
used to store architectures and training logs.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pymongo.collection import Collection

from automl_framework.models.data_models import Architecture, LayerType, TaskType
from automl_framework.core.database import get_mongo_collection


@dataclass
class ArchitectureDocument:
    """MongoDB document schema for neural network architectures."""
    
    id: str
    name: Optional[str] = None
    task_type: Optional[str] = None
    layers: List[Dict[str, Any]] = field(default_factory=list)
    connections: List[Dict[str, Any]] = field(default_factory=list)
    input_shape: List[int] = field(default_factory=list)
    output_shape: List[int] = field(default_factory=list)
    parameter_count: int = 0
    flops: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def from_architecture(cls, architecture: Architecture) -> "ArchitectureDocument":
        """Create document from Architecture dataclass."""
        return cls(
            id=architecture.id,
            task_type=architecture.task_type.value if architecture.task_type else None,
            layers=[
                {
                    "layer_type": layer.layer_type.value,
                    "parameters": layer.parameters,
                    "input_shape": list(layer.input_shape) if layer.input_shape else None,
                    "output_shape": list(layer.output_shape) if layer.output_shape else None,
                }
                for layer in architecture.layers
            ],
            connections=[
                {
                    "from_layer": conn.from_layer,
                    "to_layer": conn.to_layer,
                    "connection_type": conn.connection_type,
                }
                for conn in architecture.connections
            ],
            input_shape=list(architecture.input_shape),
            output_shape=list(architecture.output_shape),
            parameter_count=architecture.parameter_count,
            flops=architecture.flops,
            metadata=architecture.metadata,
        )
    
    def to_architecture(self) -> Architecture:
        """Convert document to Architecture dataclass."""
        from automl_framework.models.data_models import Layer, Connection
        
        layers = []
        for layer_data in self.layers:
            layer = Layer(
                layer_type=LayerType(layer_data["layer_type"]),
                parameters=layer_data.get("parameters", {}),
                input_shape=tuple(layer_data["input_shape"]) if layer_data.get("input_shape") else None,
                output_shape=tuple(layer_data["output_shape"]) if layer_data.get("output_shape") else None,
            )
            layers.append(layer)
        
        connections = []
        for conn_data in self.connections:
            connection = Connection(
                from_layer=conn_data["from_layer"],
                to_layer=conn_data["to_layer"],
                connection_type=conn_data.get("connection_type", "sequential"),
            )
            connections.append(connection)
        
        return Architecture(
            id=self.id,
            layers=layers,
            connections=connections,
            input_shape=tuple(self.input_shape),
            output_shape=tuple(self.output_shape),
            parameter_count=self.parameter_count,
            flops=self.flops,
            task_type=TaskType(self.task_type) if self.task_type else None,
            metadata=self.metadata,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return asdict(self)


@dataclass
class TrainingLogEntry:
    """Individual training log entry."""
    
    epoch: int
    step: int
    timestamp: datetime
    metrics: Dict[str, float] = field(default_factory=dict)
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return asdict(self)


@dataclass
class TrainingLogDocument:
    """MongoDB document schema for training logs."""
    
    experiment_id: str
    model_id: Optional[int] = None
    architecture_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed
    total_epochs: Optional[int] = None
    best_epoch: Optional[int] = None
    best_metric_value: Optional[float] = None
    best_metric_name: str = "accuracy"
    
    # Training configuration
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    # Log entries
    entries: List[Dict[str, Any]] = field(default_factory=list)
    
    # Summary statistics
    final_metrics: Dict[str, float] = field(default_factory=dict)
    training_duration: Optional[float] = None  # in seconds
    
    # System information
    gpu_info: Dict[str, Any] = field(default_factory=dict)
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    def add_log_entry(self, entry: TrainingLogEntry):
        """Add a training log entry."""
        self.entries.append(entry.to_dict())
    
    def complete_training(self, final_metrics: Dict[str, float]):
        """Mark training as completed with final metrics."""
        self.end_time = datetime.utcnow()
        self.status = "completed"
        self.final_metrics = final_metrics
        if self.start_time:
            self.training_duration = (self.end_time - self.start_time).total_seconds()
    
    def fail_training(self, error_message: str):
        """Mark training as failed."""
        self.end_time = datetime.utcnow()
        self.status = "failed"
        self.final_metrics["error"] = error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return asdict(self)


class ArchitectureRepository:
    """Repository for managing architecture documents in MongoDB."""
    
    def __init__(self, collection: Optional[Collection] = None):
        self.collection = collection or get_mongo_collection("architectures")
    
    def save(self, architecture: Architecture) -> str:
        """Save architecture to MongoDB."""
        doc = ArchitectureDocument.from_architecture(architecture)
        result = self.collection.replace_one(
            {"id": doc.id},
            doc.to_dict(),
            upsert=True
        )
        return doc.id
    
    def find_by_id(self, architecture_id: str) -> Optional[Architecture]:
        """Find architecture by ID."""
        doc_data = self.collection.find_one({"id": architecture_id})
        if doc_data:
            doc = ArchitectureDocument(**doc_data)
            return doc.to_architecture()
        return None
    
    def find_by_task_type(self, task_type: TaskType, limit: int = 100) -> List[Architecture]:
        """Find architectures by task type."""
        cursor = self.collection.find(
            {"task_type": task_type.value}
        ).limit(limit)
        
        architectures = []
        for doc_data in cursor:
            doc = ArchitectureDocument(**doc_data)
            architectures.append(doc.to_architecture())
        
        return architectures
    
    def find_by_parameter_range(self, min_params: int, max_params: int) -> List[Architecture]:
        """Find architectures within parameter count range."""
        cursor = self.collection.find({
            "parameter_count": {"$gte": min_params, "$lte": max_params}
        })
        
        architectures = []
        for doc_data in cursor:
            doc = ArchitectureDocument(**doc_data)
            architectures.append(doc.to_architecture())
        
        return architectures
    
    def delete(self, architecture_id: str) -> bool:
        """Delete architecture by ID."""
        result = self.collection.delete_one({"id": architecture_id})
        return result.deleted_count > 0
    
    def list_all(self, skip: int = 0, limit: int = 100) -> List[Architecture]:
        """List all architectures with pagination."""
        cursor = self.collection.find().skip(skip).limit(limit)
        
        architectures = []
        for doc_data in cursor:
            doc = ArchitectureDocument(**doc_data)
            architectures.append(doc.to_architecture())
        
        return architectures


class TrainingLogRepository:
    """Repository for managing training log documents in MongoDB."""
    
    def __init__(self, collection: Optional[Collection] = None):
        self.collection = collection or get_mongo_collection("training_logs")
    
    def create_log(self, experiment_id: str, **kwargs) -> str:
        """Create a new training log."""
        log_doc = TrainingLogDocument(experiment_id=experiment_id, **kwargs)
        result = self.collection.insert_one(log_doc.to_dict())
        return str(result.inserted_id)
    
    def add_log_entry(self, experiment_id: str, entry: TrainingLogEntry):
        """Add a log entry to existing training log."""
        self.collection.update_one(
            {"experiment_id": experiment_id, "status": "running"},
            {"$push": {"entries": entry.to_dict()}}
        )
    
    def update_best_metrics(self, experiment_id: str, epoch: int, metric_value: float, metric_name: str = "accuracy"):
        """Update best metrics for a training log."""
        self.collection.update_one(
            {"experiment_id": experiment_id},
            {
                "$set": {
                    "best_epoch": epoch,
                    "best_metric_value": metric_value,
                    "best_metric_name": metric_name
                }
            }
        )
    
    def complete_training(self, experiment_id: str, final_metrics: Dict[str, float]):
        """Mark training as completed."""
        end_time = datetime.utcnow()
        
        # Calculate duration
        log_doc = self.collection.find_one({"experiment_id": experiment_id})
        duration = None
        if log_doc and log_doc.get("start_time"):
            start_time = log_doc["start_time"]
            duration = (end_time - start_time).total_seconds()
        
        self.collection.update_one(
            {"experiment_id": experiment_id},
            {
                "$set": {
                    "end_time": end_time,
                    "status": "completed",
                    "final_metrics": final_metrics,
                    "training_duration": duration
                }
            }
        )
    
    def fail_training(self, experiment_id: str, error_message: str):
        """Mark training as failed."""
        self.collection.update_one(
            {"experiment_id": experiment_id},
            {
                "$set": {
                    "end_time": datetime.utcnow(),
                    "status": "failed",
                    "final_metrics.error": error_message
                }
            }
        )
    
    def find_by_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Find training log by experiment ID."""
        return self.collection.find_one({"experiment_id": experiment_id})
    
    def find_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Find training logs by status."""
        return list(self.collection.find({"status": status}))
    
    def get_training_history(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get training history entries for an experiment."""
        log_doc = self.collection.find_one({"experiment_id": experiment_id})
        if log_doc:
            return log_doc.get("entries", [])
        return []
    
    def delete_by_experiment(self, experiment_id: str) -> bool:
        """Delete training log by experiment ID."""
        result = self.collection.delete_one({"experiment_id": experiment_id})
        return result.deleted_count > 0


# Repository instances
architecture_repo = ArchitectureRepository()
training_log_repo = TrainingLogRepository()