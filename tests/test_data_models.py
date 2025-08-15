"""
Tests for core data models and validation.
"""

import pytest
from datetime import datetime, timedelta
from automl_framework.models.data_models import (
    Dataset, Architecture, Experiment, TrainingConfig, PerformanceMetrics,
    ExperimentResults, Feature, Layer, Connection,
    DataType, ExperimentStatus, TaskType, LayerType
)


class TestFeature:
    """Test cases for Feature class."""
    
    def test_valid_feature(self):
        """Test creating a valid feature."""
        feature = Feature(
            name="age",
            data_type="numeric",
            is_categorical=False,
            unique_values=100,
            missing_percentage=5.0
        )
        feature.validate()  # Should not raise
    
    def test_invalid_feature_name(self):
        """Test feature with invalid name."""
        feature = Feature(name="", data_type="numeric")
        with pytest.raises(ValueError, match="Feature name must be a non-empty string"):
            feature.validate()
    
    def test_invalid_missing_percentage(self):
        """Test feature with invalid missing percentage."""
        feature = Feature(name="test", data_type="numeric", missing_percentage=150.0)
        with pytest.raises(ValueError, match="Missing percentage must be between 0 and 100"):
            feature.validate()


class TestLayer:
    """Test cases for Layer class."""
    
    def test_valid_layer(self):
        """Test creating a valid layer."""
        layer = Layer(
            layer_type=LayerType.DENSE,
            parameters={"units": 64, "activation": "relu"},
            input_shape=(784,),
            output_shape=(64,)
        )
        layer.validate()  # Should not raise
    
    def test_invalid_layer_type(self):
        """Test layer with invalid type."""
        layer = Layer(layer_type="invalid")
        with pytest.raises(ValueError, match="layer_type must be a LayerType enum"):
            layer.validate()


class TestConnection:
    """Test cases for Connection class."""
    
    def test_valid_connection(self):
        """Test creating a valid connection."""
        connection = Connection(from_layer=0, to_layer=1)
        connection.validate()  # Should not raise
    
    def test_self_connection(self):
        """Test connection to same layer."""
        connection = Connection(from_layer=1, to_layer=1)
        with pytest.raises(ValueError, match="Layer cannot connect to itself"):
            connection.validate()
    
    def test_negative_layer_index(self):
        """Test connection with negative layer index."""
        connection = Connection(from_layer=-1, to_layer=1)
        with pytest.raises(ValueError, match="Layer indices must be non-negative"):
            connection.validate()


class TestDataset:
    """Test cases for Dataset class."""
    
    def test_valid_dataset(self):
        """Test creating a valid dataset."""
        features = [
            Feature(name="feature1", data_type="numeric"),
            Feature(name="target", data_type="categorical")
        ]
        dataset = Dataset(
            id="test-dataset-1",
            name="Test Dataset",
            file_path="/path/to/data.csv",
            data_type=DataType.TABULAR,
            size=1000,
            features=features,
            target_column="target"
        )
        dataset.validate()  # Should not raise
    
    def test_invalid_dataset_id(self):
        """Test dataset with invalid ID."""
        dataset = Dataset(
            id="",
            name="Test",
            file_path="/path/to/data.csv",
            data_type=DataType.TABULAR,
            size=1000
        )
        with pytest.raises(ValueError, match="Dataset ID must be a non-empty string"):
            dataset.validate()
    
    def test_invalid_dataset_size(self):
        """Test dataset with invalid size."""
        dataset = Dataset(
            id="test-1",
            name="Test",
            file_path="/path/to/data.csv",
            data_type=DataType.TABULAR,
            size=0
        )
        with pytest.raises(ValueError, match="Dataset size must be positive"):
            dataset.validate()
    
    def test_invalid_target_column(self):
        """Test dataset with non-existent target column."""
        features = [Feature(name="feature1", data_type="numeric")]
        dataset = Dataset(
            id="test-1",
            name="Test",
            file_path="/path/to/data.csv",
            data_type=DataType.TABULAR,
            size=1000,
            features=features,
            target_column="nonexistent"
        )
        with pytest.raises(ValueError, match="Target column 'nonexistent' not found in features"):
            dataset.validate()


class TestArchitecture:
    """Test cases for Architecture class."""
    
    def test_valid_architecture(self):
        """Test creating a valid architecture."""
        layers = [
            Layer(layer_type=LayerType.DENSE, parameters={"units": 64}),
            Layer(layer_type=LayerType.DENSE, parameters={"units": 32}),
            Layer(layer_type=LayerType.DENSE, parameters={"units": 10})
        ]
        connections = [
            Connection(from_layer=0, to_layer=1),
            Connection(from_layer=1, to_layer=2)
        ]
        architecture = Architecture(
            id="arch-1",
            layers=layers,
            connections=connections,
            input_shape=(784,),
            output_shape=(10,),
            parameter_count=25000,
            flops=50000
        )
        architecture.validate()  # Should not raise
    
    def test_empty_layers(self):
        """Test architecture with no layers."""
        architecture = Architecture(id="arch-1", layers=[])
        with pytest.raises(ValueError, match="Architecture must have at least one layer"):
            architecture.validate()
    
    def test_invalid_connection_reference(self):
        """Test architecture with connection referencing non-existent layer."""
        layers = [Layer(layer_type=LayerType.DENSE, parameters={"units": 64})]
        connections = [Connection(from_layer=0, to_layer=5)]  # Layer 5 doesn't exist
        architecture = Architecture(id="arch-1", layers=layers, connections=connections)
        with pytest.raises(ValueError, match="Connection references non-existent layer"):
            architecture.validate()
    
    def test_negative_parameter_count(self):
        """Test architecture with negative parameter count."""
        layers = [Layer(layer_type=LayerType.DENSE, parameters={"units": 64})]
        architecture = Architecture(id="arch-1", layers=layers, parameter_count=-100)
        with pytest.raises(ValueError, match="Parameter count cannot be negative"):
            architecture.validate()


class TestTrainingConfig:
    """Test cases for TrainingConfig class."""
    
    def test_valid_training_config(self):
        """Test creating a valid training configuration."""
        config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer="adam",
            epochs=100,
            early_stopping_patience=10,
            validation_split=0.2
        )
        config.validate()  # Should not raise
    
    def test_invalid_batch_size(self):
        """Test training config with invalid batch size."""
        config = TrainingConfig(
            batch_size=0,
            learning_rate=0.001,
            optimizer="adam",
            epochs=100
        )
        with pytest.raises(ValueError, match="Batch size must be positive"):
            config.validate()
    
    def test_invalid_learning_rate(self):
        """Test training config with invalid learning rate."""
        config = TrainingConfig(
            batch_size=32,
            learning_rate=0,
            optimizer="adam",
            epochs=100
        )
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            config.validate()
    
    def test_invalid_optimizer(self):
        """Test training config with invalid optimizer."""
        config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer="invalid_optimizer",
            epochs=100
        )
        with pytest.raises(ValueError, match="Optimizer must be one of"):
            config.validate()
    
    def test_invalid_validation_split(self):
        """Test training config with invalid validation split."""
        config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer="adam",
            epochs=100,
            validation_split=1.5
        )
        with pytest.raises(ValueError, match="Validation split must be between 0 and 1"):
            config.validate()


class TestPerformanceMetrics:
    """Test cases for PerformanceMetrics class."""
    
    def test_valid_performance_metrics(self):
        """Test creating valid performance metrics."""
        metrics = PerformanceMetrics(
            accuracy=0.95,
            loss=0.05,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            training_time=3600.0,
            inference_time=0.01
        )
        metrics.validate()  # Should not raise
    
    def test_invalid_accuracy(self):
        """Test performance metrics with invalid accuracy."""
        metrics = PerformanceMetrics(accuracy=1.5)
        with pytest.raises(ValueError, match="Accuracy, precision, recall, and F1 score must be between 0 and 1"):
            metrics.validate()
    
    def test_negative_loss(self):
        """Test performance metrics with negative loss."""
        metrics = PerformanceMetrics(loss=-0.1)
        with pytest.raises(ValueError, match="Loss cannot be negative"):
            metrics.validate()
    
    def test_negative_training_time(self):
        """Test performance metrics with negative training time."""
        metrics = PerformanceMetrics(training_time=-100)
        with pytest.raises(ValueError, match="Training time cannot be negative"):
            metrics.validate()


class TestExperiment:
    """Test cases for Experiment class."""
    
    def test_valid_experiment(self):
        """Test creating a valid experiment."""
        experiment = Experiment(
            id="exp-123",
            name="Test Experiment",
            dataset_id="dataset-456",
            status=ExperimentStatus.CREATED
        )
        experiment.validate()  # Should not raise
    
    def test_invalid_experiment_id_format(self):
        """Test experiment with invalid ID format."""
        experiment = Experiment(
            id="exp 123!",  # Contains space and special character
            name="Test",
            dataset_id="dataset-456"
        )
        with pytest.raises(ValueError, match="Experiment ID must contain only alphanumeric characters"):
            experiment.validate()
    
    def test_invalid_completion_time(self):
        """Test experiment with completion time before creation time."""
        now = datetime.now()
        experiment = Experiment(
            id="exp-123",
            name="Test",
            dataset_id="dataset-456",
            created_at=now,
            completed_at=now - timedelta(hours=1)  # Completed before created
        )
        with pytest.raises(ValueError, match="Completion time cannot be before creation time"):
            experiment.validate()
    
    def test_completed_experiment_without_completion_time(self):
        """Test completed experiment without completion time."""
        experiment = Experiment(
            id="exp-123",
            name="Test",
            dataset_id="dataset-456",
            status=ExperimentStatus.COMPLETED
            # Missing completed_at
        )
        with pytest.raises(ValueError, match="Completed experiments must have a completion time"):
            experiment.validate()
    
    def test_failed_experiment_without_error_message(self):
        """Test failed experiment without error message."""
        experiment = Experiment(
            id="exp-123",
            name="Test",
            dataset_id="dataset-456",
            status=ExperimentStatus.FAILED
            # Missing error_message
        )
        with pytest.raises(ValueError, match="Failed experiments must have an error message"):
            experiment.validate()


class TestEnums:
    """Test cases for enum classes."""
    
    def test_data_type_enum(self):
        """Test DataType enum values."""
        assert DataType.IMAGE.value == "image"
        assert DataType.TEXT.value == "text"
        assert DataType.TABULAR.value == "tabular"
        assert DataType.TIME_SERIES.value == "time_series"
    
    def test_experiment_status_enum(self):
        """Test ExperimentStatus enum values."""
        assert ExperimentStatus.CREATED.value == "created"
        assert ExperimentStatus.RUNNING.value == "running"
        assert ExperimentStatus.COMPLETED.value == "completed"
        assert ExperimentStatus.FAILED.value == "failed"
    
    def test_task_type_enum(self):
        """Test TaskType enum values."""
        assert TaskType.CLASSIFICATION.value == "classification"
        assert TaskType.REGRESSION.value == "regression"
        assert TaskType.OBJECT_DETECTION.value == "object_detection"
    
    def test_layer_type_enum(self):
        """Test LayerType enum values."""
        assert LayerType.DENSE.value == "dense"
        assert LayerType.CONV2D.value == "conv2d"
        assert LayerType.LSTM.value == "lstm"