"""
Tests for MongoDB document schemas.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from automl_framework.models.mongo_schemas import (
    ArchitectureDocument, TrainingLogDocument, TrainingLogEntry,
    ArchitectureRepository, TrainingLogRepository
)
from automl_framework.models.data_models import (
    Architecture, Layer, Connection, LayerType, TaskType
)


class TestArchitectureDocument:
    """Test cases for ArchitectureDocument."""
    
    def test_from_architecture(self):
        """Test creating document from Architecture dataclass."""
        # Create test architecture
        layers = [
            Layer(layer_type=LayerType.DENSE, parameters={"units": 64}),
            Layer(layer_type=LayerType.DENSE, parameters={"units": 32}),
        ]
        connections = [Connection(from_layer=0, to_layer=1)]
        
        architecture = Architecture(
            id="arch-123",
            layers=layers,
            connections=connections,
            input_shape=(784,),
            output_shape=(10,),
            parameter_count=25000,
            flops=50000,
            task_type=TaskType.CLASSIFICATION,
            metadata={"created_by": "test"}
        )
        
        # Convert to document
        doc = ArchitectureDocument.from_architecture(architecture)
        
        assert doc.id == "arch-123"
        assert doc.task_type == "classification"
        assert len(doc.layers) == 2
        assert doc.layers[0]["layer_type"] == "dense"
        assert doc.layers[0]["parameters"] == {"units": 64}
        assert len(doc.connections) == 1
        assert doc.connections[0]["from_layer"] == 0
        assert doc.connections[0]["to_layer"] == 1
        assert doc.input_shape == [784]
        assert doc.output_shape == [10]
        assert doc.parameter_count == 25000
        assert doc.flops == 50000
        assert doc.metadata == {"created_by": "test"}
    
    def test_to_architecture(self):
        """Test converting document back to Architecture dataclass."""
        # Create test document
        doc = ArchitectureDocument(
            id="arch-123",
            task_type="classification",
            layers=[
                {
                    "layer_type": "dense",
                    "parameters": {"units": 64},
                    "input_shape": [784],
                    "output_shape": [64]
                },
                {
                    "layer_type": "dense",
                    "parameters": {"units": 32},
                    "input_shape": [64],
                    "output_shape": [32]
                }
            ],
            connections=[
                {
                    "from_layer": 0,
                    "to_layer": 1,
                    "connection_type": "sequential"
                }
            ],
            input_shape=[784],
            output_shape=[10],
            parameter_count=25000,
            flops=50000,
            metadata={"created_by": "test"}
        )
        
        # Convert to architecture
        architecture = doc.to_architecture()
        
        assert architecture.id == "arch-123"
        assert architecture.task_type == TaskType.CLASSIFICATION
        assert len(architecture.layers) == 2
        assert architecture.layers[0].layer_type == LayerType.DENSE
        assert architecture.layers[0].parameters == {"units": 64}
        assert architecture.layers[0].input_shape == (784,)
        assert len(architecture.connections) == 1
        assert architecture.connections[0].from_layer == 0
        assert architecture.connections[0].to_layer == 1
        assert architecture.input_shape == (784,)
        assert architecture.output_shape == (10,)
        assert architecture.parameter_count == 25000
        assert architecture.flops == 50000
        assert architecture.metadata == {"created_by": "test"}
    
    def test_to_dict(self):
        """Test converting document to dictionary."""
        doc = ArchitectureDocument(
            id="arch-123",
            task_type="classification",
            parameter_count=25000
        )
        
        doc_dict = doc.to_dict()
        
        assert isinstance(doc_dict, dict)
        assert doc_dict["id"] == "arch-123"
        assert doc_dict["task_type"] == "classification"
        assert doc_dict["parameter_count"] == 25000
        assert "created_at" in doc_dict


class TestTrainingLogEntry:
    """Test cases for TrainingLogEntry."""
    
    def test_create_log_entry(self):
        """Test creating a training log entry."""
        timestamp = datetime.utcnow()
        entry = TrainingLogEntry(
            epoch=1,
            step=100,
            timestamp=timestamp,
            metrics={"accuracy": 0.85, "loss": 0.15},
            loss=0.15,
            learning_rate=0.001,
            batch_size=32
        )
        
        assert entry.epoch == 1
        assert entry.step == 100
        assert entry.timestamp == timestamp
        assert entry.metrics == {"accuracy": 0.85, "loss": 0.15}
        assert entry.loss == 0.15
        assert entry.learning_rate == 0.001
        assert entry.batch_size == 32
    
    def test_to_dict(self):
        """Test converting log entry to dictionary."""
        timestamp = datetime.utcnow()
        entry = TrainingLogEntry(
            epoch=1,
            step=100,
            timestamp=timestamp,
            metrics={"accuracy": 0.85}
        )
        
        entry_dict = entry.to_dict()
        
        assert isinstance(entry_dict, dict)
        assert entry_dict["epoch"] == 1
        assert entry_dict["step"] == 100
        assert entry_dict["timestamp"] == timestamp
        assert entry_dict["metrics"] == {"accuracy": 0.85}


class TestTrainingLogDocument:
    """Test cases for TrainingLogDocument."""
    
    def test_create_training_log(self):
        """Test creating a training log document."""
        start_time = datetime.utcnow()
        log_doc = TrainingLogDocument(
            experiment_id="exp-123",
            model_id=1,
            architecture_id="arch-456",
            start_time=start_time,
            total_epochs=100,
            hyperparameters={"learning_rate": 0.001},
            training_config={"batch_size": 32}
        )
        
        assert log_doc.experiment_id == "exp-123"
        assert log_doc.model_id == 1
        assert log_doc.architecture_id == "arch-456"
        assert log_doc.start_time == start_time
        assert log_doc.status == "running"
        assert log_doc.total_epochs == 100
        assert log_doc.hyperparameters == {"learning_rate": 0.001}
        assert log_doc.training_config == {"batch_size": 32}
        assert log_doc.entries == []
    
    def test_add_log_entry(self):
        """Test adding log entries to training log."""
        log_doc = TrainingLogDocument(experiment_id="exp-123")
        
        entry = TrainingLogEntry(
            epoch=1,
            step=100,
            timestamp=datetime.utcnow(),
            metrics={"accuracy": 0.85}
        )
        
        log_doc.add_log_entry(entry)
        
        assert len(log_doc.entries) == 1
        assert log_doc.entries[0]["epoch"] == 1
        assert log_doc.entries[0]["step"] == 100
        assert log_doc.entries[0]["metrics"] == {"accuracy": 0.85}
    
    def test_complete_training(self):
        """Test completing training."""
        start_time = datetime.utcnow()
        log_doc = TrainingLogDocument(
            experiment_id="exp-123",
            start_time=start_time
        )
        
        final_metrics = {"accuracy": 0.95, "loss": 0.05}
        log_doc.complete_training(final_metrics)
        
        assert log_doc.status == "completed"
        assert log_doc.final_metrics == final_metrics
        assert log_doc.end_time is not None
        assert log_doc.training_duration is not None
        assert log_doc.training_duration > 0
    
    def test_fail_training(self):
        """Test failing training."""
        log_doc = TrainingLogDocument(experiment_id="exp-123")
        
        error_message = "Training failed due to OOM error"
        log_doc.fail_training(error_message)
        
        assert log_doc.status == "failed"
        assert log_doc.final_metrics["error"] == error_message
        assert log_doc.end_time is not None
    
    def test_to_dict(self):
        """Test converting training log to dictionary."""
        log_doc = TrainingLogDocument(
            experiment_id="exp-123",
            status="completed",
            total_epochs=100
        )
        
        log_dict = log_doc.to_dict()
        
        assert isinstance(log_dict, dict)
        assert log_dict["experiment_id"] == "exp-123"
        assert log_dict["status"] == "completed"
        assert log_dict["total_epochs"] == 100


class TestArchitectureRepository:
    """Test cases for ArchitectureRepository."""
    
    @patch('automl_framework.models.mongo_schemas.get_mongo_collection')
    def test_save_architecture(self, mock_get_collection):
        """Test saving architecture to MongoDB."""
        # Mock collection
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        mock_collection.replace_one.return_value = Mock()
        
        # Create repository and architecture
        repo = ArchitectureRepository()
        architecture = Architecture(
            id="arch-123",
            layers=[Layer(layer_type=LayerType.DENSE, parameters={"units": 64})],
            input_shape=(784,),
            output_shape=(10,)
        )
        
        # Save architecture
        result_id = repo.save(architecture)
        
        assert result_id == "arch-123"
        mock_collection.replace_one.assert_called_once()
        
        # Check the call arguments
        call_args = mock_collection.replace_one.call_args
        assert call_args[0][0] == {"id": "arch-123"}  # filter
        assert call_args[1]["upsert"] is True
    
    @patch('automl_framework.models.mongo_schemas.get_mongo_collection')
    def test_find_by_id(self, mock_get_collection):
        """Test finding architecture by ID."""
        # Mock collection
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        
        # Mock document data
        mock_doc_data = {
            "id": "arch-123",
            "task_type": "classification",
            "layers": [
                {
                    "layer_type": "dense",
                    "parameters": {"units": 64},
                    "input_shape": [784],
                    "output_shape": [64]
                }
            ],
            "connections": [],
            "input_shape": [784],
            "output_shape": [10],
            "parameter_count": 25000,
            "flops": 50000,
            "metadata": {},
            "created_at": datetime.utcnow()
        }
        mock_collection.find_one.return_value = mock_doc_data
        
        # Create repository and find architecture
        repo = ArchitectureRepository()
        architecture = repo.find_by_id("arch-123")
        
        assert architecture is not None
        assert architecture.id == "arch-123"
        assert architecture.task_type == TaskType.CLASSIFICATION
        assert len(architecture.layers) == 1
        assert architecture.layers[0].layer_type == LayerType.DENSE
        
        mock_collection.find_one.assert_called_once_with({"id": "arch-123"})
    
    @patch('automl_framework.models.mongo_schemas.get_mongo_collection')
    def test_find_by_id_not_found(self, mock_get_collection):
        """Test finding architecture by ID when not found."""
        # Mock collection
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        mock_collection.find_one.return_value = None
        
        # Create repository and find architecture
        repo = ArchitectureRepository()
        architecture = repo.find_by_id("nonexistent")
        
        assert architecture is None
        mock_collection.find_one.assert_called_once_with({"id": "nonexistent"})
    
    @patch('automl_framework.models.mongo_schemas.get_mongo_collection')
    def test_delete_architecture(self, mock_get_collection):
        """Test deleting architecture."""
        # Mock collection
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        mock_collection.delete_one.return_value = Mock(deleted_count=1)
        
        # Create repository and delete architecture
        repo = ArchitectureRepository()
        result = repo.delete("arch-123")
        
        assert result is True
        mock_collection.delete_one.assert_called_once_with({"id": "arch-123"})


class TestTrainingLogRepository:
    """Test cases for TrainingLogRepository."""
    
    @patch('automl_framework.models.mongo_schemas.get_mongo_collection')
    def test_create_log(self, mock_get_collection):
        """Test creating a training log."""
        # Mock collection
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        mock_collection.insert_one.return_value = Mock(inserted_id="log_id_123")
        
        # Create repository and log
        repo = TrainingLogRepository()
        log_id = repo.create_log(
            experiment_id="exp-123",
            model_id=1,
            architecture_id="arch-456"
        )
        
        assert log_id == "log_id_123"
        mock_collection.insert_one.assert_called_once()
        
        # Check the inserted document structure
        call_args = mock_collection.insert_one.call_args[0][0]
        assert call_args["experiment_id"] == "exp-123"
        assert call_args["model_id"] == 1
        assert call_args["architecture_id"] == "arch-456"
        assert call_args["status"] == "running"
    
    @patch('automl_framework.models.mongo_schemas.get_mongo_collection')
    def test_add_log_entry(self, mock_get_collection):
        """Test adding log entry to training log."""
        # Mock collection
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        
        # Create repository and add log entry
        repo = TrainingLogRepository()
        entry = TrainingLogEntry(
            epoch=1,
            step=100,
            timestamp=datetime.utcnow(),
            metrics={"accuracy": 0.85}
        )
        
        repo.add_log_entry("exp-123", entry)
        
        mock_collection.update_one.assert_called_once()
        
        # Check the update query
        call_args = mock_collection.update_one.call_args
        assert call_args[0][0] == {"experiment_id": "exp-123", "status": "running"}
        assert "$push" in call_args[0][1]
        assert "entries" in call_args[0][1]["$push"]
    
    @patch('automl_framework.models.mongo_schemas.get_mongo_collection')
    def test_complete_training(self, mock_get_collection):
        """Test completing training."""
        # Mock collection
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        
        # Mock existing log document
        start_time = datetime.utcnow()
        mock_collection.find_one.return_value = {"start_time": start_time}
        
        # Create repository and complete training
        repo = TrainingLogRepository()
        final_metrics = {"accuracy": 0.95, "loss": 0.05}
        repo.complete_training("exp-123", final_metrics)
        
        mock_collection.update_one.assert_called_once()
        
        # Check the update query
        call_args = mock_collection.update_one.call_args
        assert call_args[0][0] == {"experiment_id": "exp-123"}
        update_doc = call_args[0][1]["$set"]
        assert update_doc["status"] == "completed"
        assert update_doc["final_metrics"] == final_metrics
        assert "end_time" in update_doc
        assert "training_duration" in update_doc
    
    @patch('automl_framework.models.mongo_schemas.get_mongo_collection')
    def test_fail_training(self, mock_get_collection):
        """Test failing training."""
        # Mock collection
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        
        # Create repository and fail training
        repo = TrainingLogRepository()
        error_message = "Training failed due to OOM error"
        repo.fail_training("exp-123", error_message)
        
        mock_collection.update_one.assert_called_once()
        
        # Check the update query
        call_args = mock_collection.update_one.call_args
        assert call_args[0][0] == {"experiment_id": "exp-123"}
        update_doc = call_args[0][1]["$set"]
        assert update_doc["status"] == "failed"
        assert update_doc["final_metrics.error"] == error_message
        assert "end_time" in update_doc