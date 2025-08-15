"""
Test utilities for AutoML framework testing.

Provides mock datasets, synthetic test cases, and common testing utilities.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from typing import Dict, List, Tuple, Any, Optional
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta

from automl_framework.models.data_models import (
    Dataset, Architecture, Experiment, TrainingConfig, PerformanceMetrics,
    Feature, Layer, Connection, DataType, ExperimentStatus, TaskType, LayerType
)


class MockDatasetGenerator:
    """Generates mock datasets for testing."""
    
    @staticmethod
    def create_tabular_dataset(
        n_samples: int = 1000,
        n_features: int = 10,
        n_classes: int = 3,
        missing_rate: float = 0.1,
        categorical_features: int = 2
    ) -> Tuple[pd.DataFrame, str]:
        """Create a mock tabular dataset."""
        np.random.seed(42)
        
        data = {}
        
        # Numerical features
        for i in range(n_features - categorical_features):
            feature_data = np.random.normal(0, 1, n_samples)
            # Add missing values
            missing_mask = np.random.random(n_samples) < missing_rate
            feature_data[missing_mask] = np.nan
            data[f'numeric_feature_{i}'] = feature_data
        
        # Categorical features
        categories = ['A', 'B', 'C', 'D', 'E']
        for i in range(categorical_features):
            feature_data = np.random.choice(categories, n_samples)
            # Add missing values
            missing_mask = np.random.random(n_samples) < missing_rate
            feature_data[missing_mask] = None
            data[f'categorical_feature_{i}'] = feature_data
        
        # Target variable
        data['target'] = np.random.randint(0, n_classes, n_samples)
        
        df = pd.DataFrame(data)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return df, temp_file.name
    
    @staticmethod
    def create_image_dataset(
        n_samples: int = 100,
        image_shape: Tuple[int, int, int] = (32, 32, 3),
        n_classes: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Create a mock image dataset."""
        np.random.seed(42)
        
        # Generate random images
        images = np.random.randint(0, 256, (n_samples,) + image_shape, dtype=np.uint8)
        labels = np.random.randint(0, n_classes, n_samples)
        
        # Save to temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Save images and labels
        np.save(os.path.join(temp_dir, 'images.npy'), images)
        np.save(os.path.join(temp_dir, 'labels.npy'), labels)
        
        return images, labels, temp_dir
    
    @staticmethod
    def create_time_series_dataset(
        n_samples: int = 1000,
        sequence_length: int = 100,
        n_features: int = 5,
        n_classes: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Create a mock time series dataset."""
        np.random.seed(42)
        
        # Generate time series data with some patterns
        data = []
        labels = []
        
        for i in range(n_samples):
            # Create time series with trend and noise
            t = np.linspace(0, 10, sequence_length)
            series = np.zeros((sequence_length, n_features))
            
            for j in range(n_features):
                # Add trend, seasonality, and noise
                trend = np.random.normal(0, 0.1) * t
                seasonality = np.sin(2 * np.pi * t / 10) * np.random.normal(0, 0.5)
                noise = np.random.normal(0, 0.2, sequence_length)
                series[:, j] = trend + seasonality + noise
            
            data.append(series)
            labels.append(np.random.randint(0, n_classes))
        
        data = np.array(data)
        labels = np.array(labels)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
        np.savez(temp_file.name, data=data, labels=labels)
        temp_file.close()
        
        return data, labels, temp_file.name


class MockArchitectureGenerator:
    """Generates mock architectures for testing."""
    
    @staticmethod
    def create_simple_mlp(
        input_size: int = 784,
        hidden_sizes: List[int] = [128, 64],
        output_size: int = 10
    ) -> Architecture:
        """Create a simple MLP architecture."""
        layers = []
        connections = []
        
        # Input layer (implicit)
        current_size = input_size
        
        # Hidden layers
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(Layer(
                layer_type=LayerType.DENSE,
                parameters={
                    'units': hidden_size,
                    'activation': 'relu',
                    'use_bias': True
                }
            ))
            
            if i > 0:
                connections.append(Connection(i-1, i))
            
            current_size = hidden_size
        
        # Output layer
        layers.append(Layer(
            layer_type=LayerType.DENSE,
            parameters={
                'units': output_size,
                'activation': 'softmax',
                'use_bias': True
            }
        ))
        
        if len(hidden_sizes) > 0:
            connections.append(Connection(len(hidden_sizes)-1, len(hidden_sizes)))
        
        return Architecture(
            id=f"mlp_{len(hidden_sizes)}_{output_size}",
            layers=layers,
            connections=connections,
            input_shape=(input_size,),
            output_shape=(output_size,),
            parameter_count=MockArchitectureGenerator._calculate_mlp_params(
                input_size, hidden_sizes, output_size
            ),
            flops=MockArchitectureGenerator._calculate_mlp_flops(
                input_size, hidden_sizes, output_size
            )
        )
    
    @staticmethod
    def create_simple_cnn(
        input_shape: Tuple[int, int, int] = (32, 32, 3),
        conv_filters: List[int] = [32, 64],
        dense_units: List[int] = [128],
        output_size: int = 10
    ) -> Architecture:
        """Create a simple CNN architecture."""
        layers = []
        connections = []
        layer_idx = 0
        
        # Convolutional layers
        for i, filters in enumerate(conv_filters):
            # Conv layer
            layers.append(Layer(
                layer_type=LayerType.CONV2D,
                parameters={
                    'filters': filters,
                    'kernel_size': (3, 3),
                    'activation': 'relu',
                    'padding': 'same'
                }
            ))
            
            if layer_idx > 0:
                connections.append(Connection(layer_idx-1, layer_idx))
            layer_idx += 1
            
            # Pooling layer
            layers.append(Layer(
                layer_type=LayerType.POOLING,
                parameters={
                    'pool_type': 'max',
                    'pool_size': (2, 2)
                }
            ))
            connections.append(Connection(layer_idx-1, layer_idx))
            layer_idx += 1
        
        # Flatten layer
        layers.append(Layer(
            layer_type=LayerType.FLATTEN,
            parameters={}
        ))
        connections.append(Connection(layer_idx-1, layer_idx))
        layer_idx += 1
        
        # Dense layers
        for units in dense_units:
            layers.append(Layer(
                layer_type=LayerType.DENSE,
                parameters={
                    'units': units,
                    'activation': 'relu'
                }
            ))
            connections.append(Connection(layer_idx-1, layer_idx))
            layer_idx += 1
        
        # Output layer
        layers.append(Layer(
            layer_type=LayerType.DENSE,
            parameters={
                'units': output_size,
                'activation': 'softmax'
            }
        ))
        connections.append(Connection(layer_idx-1, layer_idx))
        
        return Architecture(
            id=f"cnn_{len(conv_filters)}_{len(dense_units)}_{output_size}",
            layers=layers,
            connections=connections,
            input_shape=input_shape,
            output_shape=(output_size,),
            parameter_count=100000,  # Approximate
            flops=1000000  # Approximate
        )
    
    @staticmethod
    def _calculate_mlp_params(input_size: int, hidden_sizes: List[int], output_size: int) -> int:
        """Calculate parameters for MLP."""
        total_params = 0
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            total_params += current_size * hidden_size + hidden_size  # weights + bias
            current_size = hidden_size
        
        total_params += current_size * output_size + output_size  # output layer
        
        return total_params
    
    @staticmethod
    def _calculate_mlp_flops(input_size: int, hidden_sizes: List[int], output_size: int) -> int:
        """Calculate FLOPs for MLP (approximate)."""
        total_flops = 0
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            total_flops += current_size * hidden_size  # matrix multiplication
            current_size = hidden_size
        
        total_flops += current_size * output_size  # output layer
        
        return total_flops


class MockExperimentGenerator:
    """Generates mock experiments for testing."""
    
    @staticmethod
    def create_experiment(
        status: ExperimentStatus = ExperimentStatus.CREATED,
        with_results: bool = False
    ) -> Experiment:
        """Create a mock experiment."""
        experiment_id = f"exp_{np.random.randint(1000, 9999)}"
        
        experiment = Experiment(
            id=experiment_id,
            name=f"Test Experiment {experiment_id}",
            dataset_id="test_dataset_123",
            status=status,
            created_at=datetime.now() - timedelta(hours=1)
        )
        
        if status == ExperimentStatus.COMPLETED:
            experiment.completed_at = datetime.now()
            
            if with_results:
                experiment.results = MockExperimentGenerator._create_experiment_results()
        
        elif status == ExperimentStatus.FAILED:
            experiment.error_message = "Mock error for testing"
            experiment.completed_at = datetime.now()
        
        return experiment
    
    @staticmethod
    def _create_experiment_results() -> Dict[str, Any]:
        """Create mock experiment results."""
        return {
            'best_architecture': MockArchitectureGenerator.create_simple_mlp(),
            'best_hyperparameters': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'optimizer': 'adam'
            },
            'performance_metrics': PerformanceMetrics(
                accuracy=0.95,
                loss=0.05,
                precision=0.92,
                recall=0.88,
                f1_score=0.90,
                training_time=3600.0,
                inference_time=0.01
            ),
            'training_history': {
                'train_loss': [0.8, 0.6, 0.4, 0.2, 0.1],
                'val_loss': [0.9, 0.7, 0.5, 0.3, 0.15],
                'train_accuracy': [0.6, 0.7, 0.8, 0.9, 0.95],
                'val_accuracy': [0.55, 0.65, 0.75, 0.85, 0.90]
            }
        }


class MockServiceFactory:
    """Factory for creating mock services."""
    
    @staticmethod
    def create_mock_data_processing_service():
        """Create mock data processing service."""
        mock_service = Mock()
        mock_service.analyze_dataset.return_value = {
            'data_type': DataType.TABULAR,
            'n_samples': 1000,
            'n_features': 10,
            'missing_values': 0.1,
            'feature_types': ['numeric'] * 8 + ['categorical'] * 2
        }
        mock_service.preprocess_data.return_value = (
            np.random.random((1000, 10)),  # X
            np.random.randint(0, 3, 1000)  # y
        )
        return mock_service
    
    @staticmethod
    def create_mock_nas_service():
        """Create mock NAS service."""
        mock_service = Mock()
        mock_service.search_architectures.return_value = [
            MockArchitectureGenerator.create_simple_mlp(),
            MockArchitectureGenerator.create_simple_cnn()
        ]
        return mock_service
    
    @staticmethod
    def create_mock_hyperparameter_optimizer():
        """Create mock hyperparameter optimizer."""
        mock_optimizer = Mock()
        mock_optimizer.optimize.return_value = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'optimizer': 'adam',
            'weight_decay': 1e-4
        }
        mock_optimizer.get_optimization_history.return_value = [
            {'params': {'lr': 0.01}, 'score': 0.8},
            {'params': {'lr': 0.001}, 'score': 0.95}
        ]
        return mock_optimizer
    
    @staticmethod
    def create_mock_training_service():
        """Create mock training service."""
        mock_service = Mock()
        mock_service.train_model.return_value = {
            'model': Mock(),
            'training_history': {
                'loss': [0.8, 0.6, 0.4, 0.2],
                'accuracy': [0.6, 0.7, 0.8, 0.9]
            },
            'final_metrics': PerformanceMetrics(
                accuracy=0.9,
                loss=0.2,
                precision=0.88,
                recall=0.92,
                f1_score=0.90,
                training_time=1800.0,
                inference_time=0.005
            )
        }
        return mock_service
    
    @staticmethod
    def create_mock_evaluation_service():
        """Create mock evaluation service."""
        mock_service = Mock()
        mock_service.evaluate_model.return_value = PerformanceMetrics(
            accuracy=0.95,
            loss=0.05,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            training_time=3600.0,
            inference_time=0.01
        )
        return mock_service


class TestDataManager:
    """Manages test data lifecycle."""
    
    def __init__(self):
        self.temp_files = []
        self.temp_dirs = []
    
    def create_tabular_dataset(self, **kwargs) -> Tuple[pd.DataFrame, str]:
        """Create tabular dataset and track temp file."""
        df, temp_file = MockDatasetGenerator.create_tabular_dataset(**kwargs)
        self.temp_files.append(temp_file)
        return df, temp_file
    
    def create_image_dataset(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, str]:
        """Create image dataset and track temp dir."""
        images, labels, temp_dir = MockDatasetGenerator.create_image_dataset(**kwargs)
        self.temp_dirs.append(temp_dir)
        return images, labels, temp_dir
    
    def create_time_series_dataset(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, str]:
        """Create time series dataset and track temp file."""
        data, labels, temp_file = MockDatasetGenerator.create_time_series_dataset(**kwargs)
        self.temp_files.append(temp_file)
        return data, labels, temp_file
    
    def cleanup(self):
        """Clean up all temporary files and directories."""
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except (OSError, FileNotFoundError):
                pass
        
        for temp_dir in self.temp_dirs:
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except (OSError, FileNotFoundError):
                pass
        
        self.temp_files.clear()
        self.temp_dirs.clear()


@pytest.fixture
def test_data_manager():
    """Pytest fixture for test data manager."""
    manager = TestDataManager()
    yield manager
    manager.cleanup()


@pytest.fixture
def mock_tabular_dataset(test_data_manager):
    """Pytest fixture for mock tabular dataset."""
    return test_data_manager.create_tabular_dataset()


@pytest.fixture
def mock_image_dataset(test_data_manager):
    """Pytest fixture for mock image dataset."""
    return test_data_manager.create_image_dataset()


@pytest.fixture
def mock_time_series_dataset(test_data_manager):
    """Pytest fixture for mock time series dataset."""
    return test_data_manager.create_time_series_dataset()


@pytest.fixture
def mock_simple_mlp():
    """Pytest fixture for simple MLP architecture."""
    return MockArchitectureGenerator.create_simple_mlp()


@pytest.fixture
def mock_simple_cnn():
    """Pytest fixture for simple CNN architecture."""
    return MockArchitectureGenerator.create_simple_cnn()


@pytest.fixture
def mock_experiment():
    """Pytest fixture for mock experiment."""
    return MockExperimentGenerator.create_experiment()


@pytest.fixture
def mock_completed_experiment():
    """Pytest fixture for completed mock experiment."""
    return MockExperimentGenerator.create_experiment(
        status=ExperimentStatus.COMPLETED,
        with_results=True
    )


# Performance benchmarking utilities
class PerformanceBenchmark:
    """Utilities for performance benchmarking and regression testing."""
    
    @staticmethod
    def benchmark_function(func, *args, **kwargs) -> Dict[str, float]:
        """Benchmark a function's execution time and memory usage."""
        import time
        import tracemalloc
        
        # Start memory tracking
        tracemalloc.start()
        
        # Measure execution time
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'execution_time': end_time - start_time,
            'peak_memory_mb': peak / 1024 / 1024,
            'current_memory_mb': current / 1024 / 1024,
            'result': result
        }
    
    @staticmethod
    def assert_performance_regression(
        current_time: float,
        baseline_time: float,
        tolerance: float = 0.2
    ):
        """Assert that performance hasn't regressed beyond tolerance."""
        regression_ratio = (current_time - baseline_time) / baseline_time
        assert regression_ratio <= tolerance, (
            f"Performance regression detected: {regression_ratio:.2%} slower than baseline"
        )


# Assertion helpers
def assert_architecture_valid(architecture: Architecture):
    """Assert that an architecture is valid."""
    assert isinstance(architecture, Architecture)
    assert architecture.id is not None
    assert len(architecture.layers) > 0
    assert architecture.input_shape is not None
    assert architecture.output_shape is not None
    assert architecture.parameter_count >= 0
    assert architecture.flops >= 0


def assert_metrics_valid(metrics: PerformanceMetrics):
    """Assert that performance metrics are valid."""
    assert isinstance(metrics, PerformanceMetrics)
    assert 0 <= metrics.accuracy <= 1
    assert metrics.loss >= 0
    assert 0 <= metrics.precision <= 1
    assert 0 <= metrics.recall <= 1
    assert 0 <= metrics.f1_score <= 1
    assert metrics.training_time >= 0
    assert metrics.inference_time >= 0


def assert_experiment_valid(experiment: Experiment):
    """Assert that an experiment is valid."""
    assert isinstance(experiment, Experiment)
    assert experiment.id is not None
    assert experiment.name is not None
    assert experiment.dataset_id is not None
    assert isinstance(experiment.status, ExperimentStatus)
    assert experiment.created_at is not None
    
    if experiment.status == ExperimentStatus.COMPLETED:
        assert experiment.completed_at is not None
        assert experiment.completed_at >= experiment.created_at
    
    if experiment.status == ExperimentStatus.FAILED:
        assert experiment.error_message is not None