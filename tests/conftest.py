"""
Pytest configuration and shared fixtures for AutoML framework tests.

Provides common fixtures, test configuration, and utilities
used across all test modules.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

# Import test utilities
from tests.test_utils import (
    TestDataManager, MockDatasetGenerator, MockArchitectureGenerator,
    MockExperimentGenerator, MockServiceFactory, PerformanceBenchmark
)


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Set random seeds for reproducible tests
    np.random.seed(42)
    
    # Configure test environment
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'ERROR'  # Reduce log noise during testing


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and organize tests."""
    for item in items:
        # Add markers based on test file names
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "test_api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        elif "test_database" in item.nodeid:
            item.add_marker(pytest.mark.database)
        elif "test_performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        else:
            item.add_marker(pytest.mark.unit)
        
        # Add mock marker for tests using mocks
        if "mock" in item.nodeid.lower() or hasattr(item, 'fixturenames') and any('mock' in name for name in item.fixturenames):
            item.add_marker(pytest.mark.mock)


# Shared fixtures
@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    return {
        'database_url': 'sqlite:///:memory:',
        'test_data_dir': tempfile.mkdtemp(),
        'log_level': 'ERROR',
        'timeout': 30,
        'max_workers': 2
    }


@pytest.fixture(scope="session")
def test_database():
    """Test database fixture."""
    from automl_framework.core.database import DatabaseManager
    
    db_manager = DatabaseManager(database_url="sqlite:///:memory:")
    db_manager.create_tables()
    
    yield db_manager
    
    # Cleanup
    db_manager.engine.dispose()


@pytest.fixture
def temp_directory():
    """Temporary directory fixture."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_tabular_data():
    """Sample tabular dataset fixture."""
    np.random.seed(42)
    
    data = {
        'numeric_feature_1': np.random.normal(0, 1, 1000),
        'numeric_feature_2': np.random.uniform(0, 100, 1000),
        'categorical_feature_1': np.random.choice(['A', 'B', 'C'], 1000),
        'categorical_feature_2': np.random.choice(['X', 'Y', 'Z'], 1000),
        'target': np.random.randint(0, 3, 1000)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_image_data():
    """Sample image dataset fixture."""
    np.random.seed(42)
    
    # Generate random images (32x32x3)
    images = np.random.randint(0, 256, (100, 32, 32, 3), dtype=np.uint8)
    labels = np.random.randint(0, 10, 100)
    
    return images, labels


@pytest.fixture
def sample_time_series_data():
    """Sample time series dataset fixture."""
    np.random.seed(42)
    
    # Generate time series data
    n_samples = 100
    sequence_length = 50
    n_features = 5
    
    data = np.random.normal(0, 1, (n_samples, sequence_length, n_features))
    labels = np.random.randint(0, 2, n_samples)
    
    return data, labels


@pytest.fixture
def mock_architecture():
    """Mock architecture fixture."""
    return MockArchitectureGenerator.create_simple_mlp()


@pytest.fixture
def mock_cnn_architecture():
    """Mock CNN architecture fixture."""
    return MockArchitectureGenerator.create_simple_cnn()


@pytest.fixture
def mock_experiment():
    """Mock experiment fixture."""
    return MockExperimentGenerator.create_experiment()


@pytest.fixture
def mock_completed_experiment():
    """Mock completed experiment fixture."""
    return MockExperimentGenerator.create_experiment(
        status='completed',
        with_results=True
    )


@pytest.fixture
def mock_services():
    """Mock services fixture."""
    return {
        'data_processing': MockServiceFactory.create_mock_data_processing_service(),
        'nas': MockServiceFactory.create_mock_nas_service(),
        'hyperparameter_optimizer': MockServiceFactory.create_mock_hyperparameter_optimizer(),
        'training': MockServiceFactory.create_mock_training_service(),
        'evaluation': MockServiceFactory.create_mock_evaluation_service()
    }


@pytest.fixture
def performance_benchmark():
    """Performance benchmark fixture."""
    return PerformanceBenchmark()


# Test data fixtures with automatic cleanup
@pytest.fixture
def test_data_manager():
    """Test data manager fixture with automatic cleanup."""
    manager = TestDataManager()
    yield manager
    manager.cleanup()


@pytest.fixture
def tabular_dataset_file(test_data_manager):
    """Tabular dataset file fixture."""
    df, temp_file = test_data_manager.create_tabular_dataset(
        n_samples=1000,
        n_features=10,
        n_classes=3
    )
    return df, temp_file


@pytest.fixture
def image_dataset_dir(test_data_manager):
    """Image dataset directory fixture."""
    images, labels, temp_dir = test_data_manager.create_image_dataset(
        n_samples=100,
        image_shape=(32, 32, 3),
        n_classes=10
    )
    return images, labels, temp_dir


@pytest.fixture
def time_series_dataset_file(test_data_manager):
    """Time series dataset file fixture."""
    data, labels, temp_file = test_data_manager.create_time_series_dataset(
        n_samples=500,
        sequence_length=100,
        n_features=5
    )
    return data, labels, temp_file


# Mock patches for external dependencies
@pytest.fixture
def mock_torch():
    """Mock PyTorch fixture."""
    with patch('torch.cuda.is_available', return_value=False):
        with patch('torch.cuda.device_count', return_value=0):
            with patch('torch.nn.Module') as mock_module:
                yield mock_module


@pytest.fixture
def mock_sklearn():
    """Mock scikit-learn fixture."""
    with patch('sklearn.model_selection.train_test_split') as mock_split:
        mock_split.return_value = (
            np.random.random((800, 10)),  # X_train
            np.random.random((200, 10)),  # X_test
            np.random.randint(0, 2, 800), # y_train
            np.random.randint(0, 2, 200)  # y_test
        )
        yield mock_split


@pytest.fixture
def mock_database_operations():
    """Mock database operations fixture."""
    with patch('automl_framework.core.database.DatabaseManager') as mock_db:
        mock_session = Mock()
        mock_db.return_value.get_session.return_value.__enter__.return_value = mock_session
        mock_db.return_value.get_session.return_value.__exit__.return_value = None
        yield mock_session


# Performance test fixtures
@pytest.fixture
def performance_test_data():
    """Performance test data fixture."""
    sizes = [1000, 5000, 10000]
    datasets = {}
    
    for size in sizes:
        datasets[size] = {
            'X': np.random.random((size, 20)),
            'y': np.random.randint(0, 5, size)
        }
    
    return datasets


# API test fixtures
@pytest.fixture
def api_client():
    """API test client fixture."""
    from fastapi.testclient import TestClient
    from automl_framework.api.main import app
    
    return TestClient(app)


@pytest.fixture
def api_headers():
    """API headers fixture."""
    return {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_environment():
    """Automatic cleanup fixture."""
    # Setup
    original_env = os.environ.copy()
    
    yield
    
    # Cleanup
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
    
    # Clean up any temporary files
    import gc
    gc.collect()


# Skip markers for conditional tests
def pytest_runtest_setup(item):
    """Setup function to handle conditional test skipping."""
    # Skip GPU tests if no GPU available
    if item.get_closest_marker("gpu"):
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("PyTorch not available")
    
    # Skip slow tests in fast mode
    if item.get_closest_marker("slow"):
        if item.config.getoption("--fast", default=False):
            pytest.skip("Skipping slow test in fast mode")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Run only fast tests"
    )
    
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )
    
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="Run performance tests"
    )


# Custom assertions
def assert_dataframe_equal(df1, df2, check_dtype=True, check_index=True):
    """Custom assertion for DataFrame equality."""
    pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype, check_index=check_index)


def assert_array_close(arr1, arr2, rtol=1e-5, atol=1e-8):
    """Custom assertion for array closeness."""
    np.testing.assert_allclose(arr1, arr2, rtol=rtol, atol=atol)


def assert_performance_acceptable(execution_time, max_time, operation_name="Operation"):
    """Custom assertion for performance requirements."""
    assert execution_time <= max_time, (
        f"{operation_name} took {execution_time:.3f}s, "
        f"which exceeds the maximum allowed time of {max_time:.3f}s"
    )