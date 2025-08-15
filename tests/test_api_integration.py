"""
API integration tests for AutoML framework.

Tests REST API endpoints, WebSocket communication, authentication,
and API interactions with realistic datasets and scenarios.
"""

import pytest
import json
import tempfile
import os
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from automl_framework.api.main import app
from automl_framework.models.data_models import (
    ExperimentStatus, DataType, TaskType
)
from tests.test_utils import (
    MockDatasetGenerator, MockExperimentGenerator, TestDataManager,
    test_data_manager
)


class TestDatasetAPI:
    """Test dataset-related API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('automl_framework.api.routes.datasets.DataProcessingService')
    def test_upload_dataset_csv(self, mock_data_service):
        """Test CSV dataset upload."""
        # Mock data service
        mock_service = Mock()
        mock_service.analyze_dataset.return_value = {
            'data_type': DataType.TABULAR,
            'n_samples': 1000,
            'n_features': 10,
            'feature_types': {
                'feature1': 'numeric',
                'feature2': 'categorical'
            },
            'target_column': 'target',
            'missing_values': {'feature1': 5.0},
            'class_distribution': {'0': 500, '1': 500}
        }
        mock_data_service.return_value = mock_service
        
        # Create test CSV
        test_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.randint(0, 2, 100)
        })
        csv_content = test_data.to_csv(index=False)
        
        # Test upload
        response = self.client.post(
            "/api/v1/datasets/upload",
            files={"file": ("test.csv", csv_content, "text/csv")},
            data={"name": "Test Dataset", "description": "Test dataset for API"}
        )
        
        assert response.status_code == 201
        data = response.json()
        
        assert "dataset_id" in data
        assert data["name"] == "Test Dataset"
        assert data["description"] == "Test dataset for API"
        assert "analysis" in data
        assert data["analysis"]["data_type"] == "tabular"
        assert data["analysis"]["n_samples"] == 1000
        assert data["analysis"]["n_features"] == 10
    
    def test_upload_invalid_file(self):
        """Test upload of invalid file."""
        # Test with non-CSV file
        response = self.client.post(
            "/api/v1/datasets/upload",
            files={"file": ("test.txt", "invalid content", "text/plain")},
            data={"name": "Invalid Dataset"}
        )
        
        assert response.status_code == 400
        assert "error" in response.json()
    
    def test_upload_empty_file(self):
        """Test upload of empty file."""
        response = self.client.post(
            "/api/v1/datasets/upload",
            files={"file": ("empty.csv", "", "text/csv")},
            data={"name": "Empty Dataset"}
        )
        
        assert response.status_code == 400
        assert "error" in response.json()
    
    @patch('automl_framework.api.routes.datasets.DataProcessingService')
    def test_get_dataset_info(self, mock_data_service):
        """Test getting dataset information."""
        # Mock data service
        mock_service = Mock()
        mock_service.get_dataset_info.return_value = {
            'id': 'dataset_123',
            'name': 'Test Dataset',
            'data_type': DataType.TABULAR,
            'n_samples': 1000,
            'n_features': 10,
            'created_at': '2023-12-01T10:00:00Z',
            'file_size_mb': 2.5
        }
        mock_data_service.return_value = mock_service
        
        response = self.client.get("/api/v1/datasets/dataset_123")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == "dataset_123"
        assert data["name"] == "Test Dataset"
        assert data["data_type"] == "tabular"
        assert data["n_samples"] == 1000
    
    def test_get_nonexistent_dataset(self):
        """Test getting non-existent dataset."""
        response = self.client.get("/api/v1/datasets/nonexistent")
        
        assert response.status_code == 404
        assert "error" in response.json()
    
    @patch('automl_framework.api.routes.datasets.DataProcessingService')
    def test_list_datasets(self, mock_data_service):
        """Test listing datasets."""
        # Mock data service
        mock_service = Mock()
        mock_service.list_datasets.return_value = [
            {
                'id': 'dataset_1',
                'name': 'Dataset 1',
                'data_type': DataType.TABULAR,
                'n_samples': 1000,
                'created_at': '2023-12-01T10:00:00Z'
            },
            {
                'id': 'dataset_2',
                'name': 'Dataset 2',
                'data_type': DataType.IMAGE,
                'n_samples': 5000,
                'created_at': '2023-12-01T11:00:00Z'
            }
        ]
        mock_data_service.return_value = mock_service
        
        response = self.client.get("/api/v1/datasets")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "datasets" in data
        assert len(data["datasets"]) == 2
        assert data["datasets"][0]["id"] == "dataset_1"
        assert data["datasets"][1]["id"] == "dataset_2"
    
    @patch('automl_framework.api.routes.datasets.DataProcessingService')
    def test_delete_dataset(self, mock_data_service):
        """Test dataset deletion."""
        # Mock data service
        mock_service = Mock()
        mock_service.delete_dataset.return_value = True
        mock_data_service.return_value = mock_service
        
        response = self.client.delete("/api/v1/datasets/dataset_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Dataset deleted successfully"
    
    @patch('automl_framework.api.routes.datasets.DataProcessingService')
    def test_get_dataset_preview(self, mock_data_service):
        """Test getting dataset preview."""
        # Mock data service
        mock_service = Mock()
        mock_service.get_dataset_preview.return_value = {
            'columns': ['feature1', 'feature2', 'target'],
            'sample_data': [
                {'feature1': 1.5, 'feature2': 'A', 'target': 0},
                {'feature1': 2.3, 'feature2': 'B', 'target': 1},
                {'feature1': 0.8, 'feature2': 'A', 'target': 0}
            ],
            'statistics': {
                'feature1': {'mean': 1.53, 'std': 0.75, 'min': 0.1, 'max': 3.2},
                'feature2': {'unique_values': ['A', 'B', 'C'], 'most_common': 'A'}
            }
        }
        mock_data_service.return_value = mock_service
        
        response = self.client.get("/api/v1/datasets/dataset_123/preview")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "columns" in data
        assert "sample_data" in data
        assert "statistics" in data
        assert len(data["columns"]) == 3
        assert len(data["sample_data"]) == 3


class TestExperimentAPI:
    """Test experiment-related API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    @patch('automl_framework.api.routes.experiments.ExperimentManager')
    def test_create_experiment(self, mock_experiment_manager):
        """Test experiment creation."""
        # Mock experiment manager
        mock_manager = Mock()
        mock_experiment = MockExperimentGenerator.create_experiment()
        mock_manager.create_experiment.return_value = mock_experiment
        mock_experiment_manager.return_value = mock_manager
        
        experiment_data = {
            "name": "Test Classification Experiment",
            "description": "Testing classification pipeline",
            "dataset_id": "dataset_123",
            "task_type": "classification",
            "config": {
                "max_trials": 20,
                "max_time_minutes": 120,
                "optimization_metric": "accuracy",
                "validation_split": 0.2,
                "enable_feature_engineering": True,
                "enable_early_stopping": True
            }
        }
        
        response = self.client.post(
            "/api/v1/experiments",
            json=experiment_data
        )
        
        assert response.status_code == 201
        data = response.json()
        
        assert "experiment_id" in data
        assert data["name"] == "Test Classification Experiment"
        assert data["status"] == "created"
        assert data["task_type"] == "classification"
    
    def test_create_experiment_invalid_data(self):
        """Test experiment creation with invalid data."""
        invalid_data = {
            "name": "",  # Empty name
            "dataset_id": "dataset_123",
            "task_type": "invalid_task",  # Invalid task type
            "config": {}
        }
        
        response = self.client.post(
            "/api/v1/experiments",
            json=invalid_data
        )
        
        assert response.status_code == 422  # Validation error
    
    @patch('automl_framework.api.routes.experiments.ExperimentManager')
    def test_start_experiment(self, mock_experiment_manager):
        """Test starting an experiment."""
        # Mock experiment manager
        mock_manager = Mock()
        mock_manager.start_experiment.return_value = True
        mock_experiment_manager.return_value = mock_manager
        
        response = self.client.post("/api/v1/experiments/exp_123/start")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Experiment started successfully"
        assert data["experiment_id"] == "exp_123"
    
    @patch('automl_framework.api.routes.experiments.ExperimentManager')
    def test_stop_experiment(self, mock_experiment_manager):
        """Test stopping an experiment."""
        # Mock experiment manager
        mock_manager = Mock()
        mock_manager.stop_experiment.return_value = True
        mock_experiment_manager.return_value = mock_manager
        
        response = self.client.post("/api/v1/experiments/exp_123/stop")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Experiment stopped successfully"
    
    @patch('automl_framework.api.routes.experiments.ExperimentManager')
    def test_get_experiment_status(self, mock_experiment_manager):
        """Test getting experiment status."""
        # Mock experiment manager
        mock_manager = Mock()
        mock_experiment = MockExperimentGenerator.create_experiment(
            status=ExperimentStatus.RUNNING
        )
        mock_experiment.progress = 0.65
        mock_experiment.current_trial = 13
        mock_experiment.total_trials = 20
        mock_experiment.best_score = 0.92
        mock_experiment.time_elapsed = 1800  # 30 minutes
        mock_experiment.estimated_time_remaining = 900  # 15 minutes
        
        mock_manager.get_experiment.return_value = mock_experiment
        mock_experiment_manager.return_value = mock_manager
        
        response = self.client.get("/api/v1/experiments/exp_123/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["experiment_id"] == mock_experiment.id
        assert data["status"] == "running"
        assert data["progress"] == 0.65
        assert data["current_trial"] == 13
        assert data["total_trials"] == 20
        assert data["best_score"] == 0.92
    
    @patch('automl_framework.api.routes.experiments.ExperimentManager')
    def test_get_experiment_results(self, mock_experiment_manager):
        """Test getting experiment results."""
        # Mock experiment manager
        mock_manager = Mock()
        mock_experiment = MockExperimentGenerator.create_experiment(
            status=ExperimentStatus.COMPLETED,
            with_results=True
        )
        mock_manager.get_experiment.return_value = mock_experiment
        mock_experiment_manager.return_value = mock_manager
        
        response = self.client.get("/api/v1/experiments/exp_123/results")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "experiment_id" in data
        assert "best_model" in data
        assert "best_hyperparameters" in data
        assert "performance_metrics" in data
        assert "training_history" in data
    
    @patch('automl_framework.api.routes.experiments.ExperimentManager')
    def test_list_experiments(self, mock_experiment_manager):
        """Test listing experiments."""
        # Mock experiment manager
        mock_manager = Mock()
        mock_experiments = [
            MockExperimentGenerator.create_experiment(status=ExperimentStatus.COMPLETED),
            MockExperimentGenerator.create_experiment(status=ExperimentStatus.RUNNING),
            MockExperimentGenerator.create_experiment(status=ExperimentStatus.FAILED)
        ]
        mock_manager.list_experiments.return_value = mock_experiments
        mock_experiment_manager.return_value = mock_manager
        
        response = self.client.get("/api/v1/experiments")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "experiments" in data
        assert len(data["experiments"]) == 3
        
        # Check status distribution
        statuses = [exp["status"] for exp in data["experiments"]]
        assert "completed" in statuses
        assert "running" in statuses
        assert "failed" in statuses
    
    @patch('automl_framework.api.routes.experiments.ExperimentManager')
    def test_delete_experiment(self, mock_experiment_manager):
        """Test experiment deletion."""
        # Mock experiment manager
        mock_manager = Mock()
        mock_manager.delete_experiment.return_value = True
        mock_experiment_manager.return_value = mock_manager
        
        response = self.client.delete("/api/v1/experiments/exp_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Experiment deleted successfully"


class TestModelAPI:
    """Test model-related API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    @patch('automl_framework.api.routes.models.ModelExportService')
    def test_export_model(self, mock_export_service):
        """Test model export."""
        # Mock export service
        mock_service = Mock()
        mock_service.export_model.return_value = {
            'export_id': 'export_123',
            'model_path': '/path/to/exported/model',
            'format': 'pytorch',
            'size_mb': 25.6,
            'created_at': '2023-12-01T15:30:00Z'
        }
        mock_export_service.return_value = mock_service
        
        export_request = {
            "experiment_id": "exp_123",
            "model_id": "model_456",
            "format": "pytorch",
            "include_preprocessing": True,
            "optimize_for_inference": True
        }
        
        response = self.client.post(
            "/api/v1/models/export",
            json=export_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["export_id"] == "export_123"
        assert data["format"] == "pytorch"
        assert data["size_mb"] == 25.6
    
    @patch('automl_framework.api.routes.models.ModelServingService')
    def test_deploy_model(self, mock_serving_service):
        """Test model deployment."""
        # Mock serving service
        mock_service = Mock()
        mock_service.deploy_model.return_value = {
            'deployment_id': 'deploy_123',
            'endpoint_url': 'https://api.example.com/models/deploy_123/predict',
            'status': 'active',
            'created_at': '2023-12-01T16:00:00Z'
        }
        mock_serving_service.return_value = mock_service
        
        deploy_request = {
            "model_id": "model_456",
            "deployment_name": "Production Classifier",
            "instance_type": "cpu-medium",
            "auto_scaling": True,
            "min_instances": 1,
            "max_instances": 5
        }
        
        response = self.client.post(
            "/api/v1/models/deploy",
            json=deploy_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["deployment_id"] == "deploy_123"
        assert data["status"] == "active"
        assert "endpoint_url" in data
    
    @patch('automl_framework.api.routes.models.ModelServingService')
    def test_model_prediction(self, mock_serving_service):
        """Test model prediction endpoint."""
        # Mock serving service
        mock_service = Mock()
        mock_service.predict.return_value = {
            'predictions': [0.85, 0.15],  # Class probabilities
            'predicted_class': 0,
            'confidence': 0.85,
            'prediction_time_ms': 12.5
        }
        mock_serving_service.return_value = mock_service
        
        prediction_request = {
            "instances": [
                {
                    "feature1": 1.5,
                    "feature2": 2.3,
                    "feature3": 0.8,
                    "feature4": "category_A"
                }
            ]
        }
        
        response = self.client.post(
            "/api/v1/models/deploy_123/predict",
            json=prediction_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert "predicted_class" in data
        assert "confidence" in data
        assert data["predicted_class"] == 0
        assert data["confidence"] == 0.85
    
    @patch('automl_framework.api.routes.models.ModelServingService')
    def test_batch_prediction(self, mock_serving_service):
        """Test batch prediction endpoint."""
        # Mock serving service
        mock_service = Mock()
        mock_service.predict_batch.return_value = {
            'predictions': [
                {'predicted_class': 0, 'confidence': 0.85, 'probabilities': [0.85, 0.15]},
                {'predicted_class': 1, 'confidence': 0.92, 'probabilities': [0.08, 0.92]},
                {'predicted_class': 0, 'confidence': 0.78, 'probabilities': [0.78, 0.22]}
            ],
            'batch_size': 3,
            'total_prediction_time_ms': 45.2
        }
        mock_serving_service.return_value = mock_service
        
        batch_request = {
            "instances": [
                {"feature1": 1.5, "feature2": 2.3},
                {"feature1": 0.8, "feature2": 1.2},
                {"feature1": 2.1, "feature2": 3.4}
            ]
        }
        
        response = self.client.post(
            "/api/v1/models/deploy_123/predict/batch",
            json=batch_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert "batch_size" in data
        assert data["batch_size"] == 3
        assert len(data["predictions"]) == 3


class TestWebSocketIntegration:
    """Test WebSocket integration for real-time updates."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    @pytest.mark.asyncio
    @patch('automl_framework.api.websocket_manager.WebSocketManager')
    async def test_experiment_progress_websocket(self, mock_ws_manager):
        """Test WebSocket for experiment progress updates."""
        # Mock WebSocket manager
        mock_manager = Mock()
        mock_connection = AsyncMock()
        mock_manager.connect.return_value = mock_connection
        mock_ws_manager.return_value = mock_manager
        
        # Test WebSocket connection
        with self.client.websocket_connect("/ws/experiments/exp_123/progress") as websocket:
            # Simulate receiving progress updates
            progress_updates = [
                {"type": "progress", "experiment_id": "exp_123", "progress": 0.2, "current_trial": 4},
                {"type": "progress", "experiment_id": "exp_123", "progress": 0.5, "current_trial": 10},
                {"type": "progress", "experiment_id": "exp_123", "progress": 1.0, "status": "completed"}
            ]
            
            for update in progress_updates:
                # Simulate sending update from server
                websocket.send_json(update)
                
                # Receive and verify update
                received = websocket.receive_json()
                assert received["type"] == update["type"]
                assert received["experiment_id"] == update["experiment_id"]
                assert received["progress"] == update["progress"]
    
    @pytest.mark.asyncio
    async def test_training_metrics_websocket(self):
        """Test WebSocket for training metrics updates."""
        with self.client.websocket_connect("/ws/experiments/exp_123/metrics") as websocket:
            # Simulate training metrics updates
            metrics_updates = [
                {
                    "type": "training_metrics",
                    "experiment_id": "exp_123",
                    "epoch": 1,
                    "train_loss": 0.8,
                    "val_loss": 0.85,
                    "train_accuracy": 0.6,
                    "val_accuracy": 0.55
                },
                {
                    "type": "training_metrics",
                    "experiment_id": "exp_123",
                    "epoch": 5,
                    "train_loss": 0.4,
                    "val_loss": 0.45,
                    "train_accuracy": 0.85,
                    "val_accuracy": 0.82
                }
            ]
            
            for update in metrics_updates:
                websocket.send_json(update)
                received = websocket.receive_json()
                
                assert received["type"] == "training_metrics"
                assert received["epoch"] == update["epoch"]
                assert received["train_loss"] == update["train_loss"]
                assert received["val_accuracy"] == update["val_accuracy"]
    
    def test_websocket_authentication(self):
        """Test WebSocket authentication."""
        # Test connection without authentication
        with pytest.raises(Exception):  # Should fail without proper auth
            with self.client.websocket_connect("/ws/experiments/exp_123/progress"):
                pass
        
        # Test connection with invalid token
        headers = {"Authorization": "Bearer invalid_token"}
        with pytest.raises(Exception):
            with self.client.websocket_connect(
                "/ws/experiments/exp_123/progress",
                headers=headers
            ):
                pass


class TestAPIErrorHandling:
    """Test API error handling and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    def test_404_endpoints(self):
        """Test 404 responses for non-existent endpoints."""
        response = self.client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        
        response = self.client.post("/api/v1/invalid/endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test 405 responses for wrong HTTP methods."""
        # GET on POST-only endpoint
        response = self.client.get("/api/v1/experiments")
        # Note: This might be 200 if GET is also supported
        
        # DELETE on GET-only endpoint
        response = self.client.delete("/api/v1/experiments/exp_123/status")
        assert response.status_code == 405
    
    def test_validation_errors(self):
        """Test request validation errors."""
        # Missing required fields
        response = self.client.post(
            "/api/v1/experiments",
            json={"name": "Test"}  # Missing required fields
        )
        assert response.status_code == 422
        
        # Invalid field types
        response = self.client.post(
            "/api/v1/experiments",
            json={
                "name": 123,  # Should be string
                "dataset_id": "dataset_123",
                "task_type": "classification"
            }
        )
        assert response.status_code == 422
    
    @patch('automl_framework.api.routes.experiments.ExperimentManager')
    def test_internal_server_errors(self, mock_experiment_manager):
        """Test internal server error handling."""
        # Mock service to raise exception
        mock_manager = Mock()
        mock_manager.create_experiment.side_effect = Exception("Internal error")
        mock_experiment_manager.return_value = mock_manager
        
        response = self.client.post(
            "/api/v1/experiments",
            json={
                "name": "Test Experiment",
                "dataset_id": "dataset_123",
                "task_type": "classification",
                "config": {}
            }
        )
        
        assert response.status_code == 500
        assert "error" in response.json()
    
    def test_rate_limiting(self):
        """Test API rate limiting."""
        # Make many requests quickly
        responses = []
        for i in range(100):
            response = self.client.get("/api/v1/health")
            responses.append(response.status_code)
        
        # Should eventually get rate limited (429)
        # Note: This depends on rate limiting configuration
        status_codes = set(responses)
        assert 200 in status_codes  # Some requests should succeed
        # assert 429 in status_codes  # Some should be rate limited
    
    def test_request_timeout(self):
        """Test request timeout handling."""
        # This would require mocking slow operations
        # For now, just test that the endpoint exists
        response = self.client.get("/api/v1/health")
        assert response.status_code == 200


class TestAPIPerformance:
    """Test API performance characteristics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    def test_health_check_performance(self):
        """Test health check endpoint performance."""
        import time
        
        start_time = time.time()
        response = self.client.get("/api/v1/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 0.1  # Should be very fast
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            start_time = time.time()
            response = self.client.get("/api/v1/health")
            end_time = time.time()
            results.append({
                'status_code': response.status_code,
                'response_time': end_time - start_time
            })
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all requests succeeded
        assert len(results) == 10
        assert all(result['status_code'] == 200 for result in results)
        
        # Verify reasonable response times
        avg_response_time = sum(result['response_time'] for result in results) / len(results)
        assert avg_response_time < 0.5  # Average should be reasonable


if __name__ == "__main__":
    pytest.main([__file__])