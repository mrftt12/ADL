"""
End-to-end integration tests for AutoML framework.

Tests complete AutoML pipeline from data upload to model deployment,
including all service interactions and data flow.
"""

import pytest
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from automl_framework.services.data_processing import DataProcessingService
from automl_framework.services.nas_service import NASService
from automl_framework.services.hyperparameter_optimization import HyperparameterOptimizer
from automl_framework.services.training_service import TrainingService
from automl_framework.services.evaluation_service import EvaluationService
from automl_framework.services.experiment_manager import ExperimentManager
from automl_framework.services.model_export import ModelExportService
from automl_framework.services.model_serving import ModelServingService

from automl_framework.models.data_models import (
    Dataset, Experiment, ExperimentStatus, TaskType, DataType
)

from tests.test_utils import (
    MockDatasetGenerator, MockArchitectureGenerator, TestDataManager,
    assert_experiment_valid, assert_metrics_valid, test_data_manager
)


class TestEndToEndAutoMLPipeline:
    """Test complete AutoML pipeline end-to-end."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize services
        self.data_service = DataProcessingService()
        self.nas_service = NASService()
        self.hpo_service = HyperparameterOptimizer()
        self.training_service = TrainingService()
        self.evaluation_service = EvaluationService()
        self.experiment_manager = ExperimentManager()
        self.export_service = ModelExportService()
        self.serving_service = ModelServingService()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('automl_framework.services.training_service.torch')
    @patch('automl_framework.services.nas_service.torch')
    def test_complete_classification_pipeline(self, mock_nas_torch, mock_training_torch, test_data_manager):
        """Test complete AutoML pipeline for classification task."""
        # Step 1: Create and upload dataset
        df, dataset_path = test_data_manager.create_tabular_dataset(
            n_samples=1000,
            n_features=20,
            n_classes=3,
            missing_rate=0.1,
            categorical_features=5
        )
        
        # Step 2: Create experiment
        experiment = Experiment(
            id="test_exp_001",
            name="End-to-End Classification Test",
            dataset_id="test_dataset_001",
            status=ExperimentStatus.CREATED,
            created_at=datetime.now()
        )
        
        # Step 3: Process dataset
        processed_data = self.data_service.process_dataset(
            dataset_path,
            task_type=TaskType.CLASSIFICATION,
            validation_split=0.2,
            test_split=0.1,
            enable_feature_engineering=True
        )
        
        assert 'X_train' in processed_data
        assert 'X_val' in processed_data
        assert 'X_test' in processed_data
        assert 'y_train' in processed_data
        assert 'y_val' in processed_data
        assert 'y_test' in processed_data
        assert 'preprocessing_pipeline' in processed_data
        
        # Verify data shapes
        assert len(processed_data['X_train']) > 0
        assert len(processed_data['X_val']) > 0
        assert len(processed_data['X_test']) > 0
        assert processed_data['X_train'].shape[1] == processed_data['X_val'].shape[1]
        
        # Step 4: Neural Architecture Search (mocked)
        with patch.object(self.nas_service, 'search_architectures') as mock_nas:
            mock_architectures = [
                MockArchitectureGenerator.create_simple_mlp(
                    input_size=processed_data['X_train'].shape[1],
                    hidden_sizes=[128, 64],
                    output_size=3
                ),
                MockArchitectureGenerator.create_simple_mlp(
                    input_size=processed_data['X_train'].shape[1],
                    hidden_sizes=[256, 128, 64],
                    output_size=3
                )
            ]
            mock_nas.return_value = mock_architectures
            
            candidate_architectures = self.nas_service.search_architectures(
                dataset_metadata={
                    'input_shape': (processed_data['X_train'].shape[1],),
                    'output_shape': (3,),
                    'task_type': TaskType.CLASSIFICATION
                },
                max_architectures=2,
                search_time_limit=300
            )
            
            assert len(candidate_architectures) == 2
            for arch in candidate_architectures:
                assert arch.input_shape == (processed_data['X_train'].shape[1],)
                assert arch.output_shape == (3,)
        
        # Step 5: Hyperparameter Optimization (mocked)
        with patch.object(self.hpo_service, 'optimize_hyperparameters') as mock_hpo:
            mock_hpo.return_value = {
                'learning_rate': 0.001,
                'batch_size': 32,
                'optimizer': 'adam',
                'weight_decay': 1e-4,
                'dropout_rate': 0.3,
                'epochs': 50
            }
            
            best_hyperparameters = self.hpo_service.optimize_hyperparameters(
                architecture=candidate_architectures[0],
                dataset=processed_data,
                optimization_budget=20
            )
            
            assert 'learning_rate' in best_hyperparameters
            assert 'batch_size' in best_hyperparameters
            assert 'optimizer' in best_hyperparameters
        
        # Step 6: Model Training (mocked)
        with patch.object(self.training_service, 'train_model') as mock_training:
            from automl_framework.models.data_models import PerformanceMetrics
            
            mock_training.return_value = {
                'model': Mock(),
                'training_history': {
                    'train_loss': [0.8, 0.6, 0.4, 0.3, 0.2],
                    'val_loss': [0.9, 0.7, 0.5, 0.4, 0.3],
                    'train_accuracy': [0.6, 0.7, 0.8, 0.85, 0.9],
                    'val_accuracy': [0.55, 0.65, 0.75, 0.8, 0.85]
                },
                'final_metrics': PerformanceMetrics(
                    accuracy=0.85,
                    loss=0.3,
                    precision=0.83,
                    recall=0.87,
                    f1_score=0.85,
                    training_time=1800.0,
                    inference_time=0.005
                ),
                'best_epoch': 5,
                'stopped_early': False
            }
            
            training_config = self.training_service.create_training_config(
                candidate_architectures[0],
                best_hyperparameters,
                TaskType.CLASSIFICATION
            )
            
            training_result = self.training_service.train_model(
                candidate_architectures[0],
                training_config,
                processed_data['X_train'],
                processed_data['y_train'],
                processed_data['X_val'],
                processed_data['y_val']
            )
            
            assert 'model' in training_result
            assert 'training_history' in training_result
            assert 'final_metrics' in training_result
            assert_metrics_valid(training_result['final_metrics'])
        
        # Step 7: Model Evaluation (mocked)
        with patch.object(self.evaluation_service, 'evaluate_model') as mock_evaluation:
            mock_evaluation.return_value = PerformanceMetrics(
                accuracy=0.87,
                loss=0.28,
                precision=0.85,
                recall=0.89,
                f1_score=0.87,
                training_time=1800.0,
                inference_time=0.005
            )
            
            test_metrics = self.evaluation_service.evaluate_model(
                training_result['model'],
                processed_data['X_test'],
                processed_data['y_test'],
                task_type=TaskType.CLASSIFICATION
            )
            
            assert_metrics_valid(test_metrics)
            assert test_metrics.accuracy > 0.8  # Should achieve good performance
        
        # Step 8: Model Export (mocked)
        with patch.object(self.export_service, 'export_model') as mock_export:
            export_path = os.path.join(self.temp_dir, 'exported_model')
            mock_export.return_value = {
                'model_path': export_path,
                'preprocessing_path': os.path.join(export_path, 'preprocessing.pkl'),
                'metadata_path': os.path.join(export_path, 'metadata.json'),
                'format': 'pytorch',
                'size_mb': 15.2
            }
            
            export_result = self.export_service.export_model(
                model=training_result['model'],
                architecture=candidate_architectures[0],
                preprocessing_pipeline=processed_data['preprocessing_pipeline'],
                export_format='pytorch',
                export_path=export_path
            )
            
            assert 'model_path' in export_result
            assert 'preprocessing_path' in export_result
            assert 'metadata_path' in export_result
        
        # Step 9: Update experiment status
        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = datetime.now()
        experiment.results = {
            'best_architecture': candidate_architectures[0],
            'best_hyperparameters': best_hyperparameters,
            'training_metrics': training_result['final_metrics'],
            'test_metrics': test_metrics,
            'model_export': export_result
        }
        
        assert_experiment_valid(experiment)
        assert experiment.status == ExperimentStatus.COMPLETED
        assert experiment.results is not None
    
    @patch('automl_framework.services.training_service.torch')
    def test_regression_pipeline(self, mock_torch, test_data_manager):
        """Test AutoML pipeline for regression task."""
        # Create regression dataset
        np.random.seed(42)
        n_samples = 1000
        n_features = 15
        
        X = np.random.normal(0, 1, (n_samples, n_features))
        # Create target with some relationship to features
        y = (X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 0.5 + 
             np.random.normal(0, 0.1, n_samples))
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['target'] = y
        
        # Save dataset
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            # Process dataset for regression
            processed_data = self.data_service.process_dataset(
                temp_file.name,
                task_type=TaskType.REGRESSION,
                validation_split=0.2,
                test_split=0.1
            )
            
            assert 'X_train' in processed_data
            assert 'y_train' in processed_data
            
            # Verify target is continuous
            assert processed_data['y_train'].dtype in [np.float32, np.float64]
            
            # Mock architecture search for regression
            with patch.object(self.nas_service, 'search_architectures') as mock_nas:
                regression_arch = MockArchitectureGenerator.create_simple_mlp(
                    input_size=processed_data['X_train'].shape[1],
                    hidden_sizes=[64, 32],
                    output_size=1  # Single output for regression
                )
                mock_nas.return_value = [regression_arch]
                
                architectures = self.nas_service.search_architectures(
                    dataset_metadata={
                        'input_shape': (processed_data['X_train'].shape[1],),
                        'output_shape': (1,),
                        'task_type': TaskType.REGRESSION
                    }
                )
                
                assert len(architectures) == 1
                assert architectures[0].output_shape == (1,)
            
            # Mock training for regression
            with patch.object(self.training_service, 'train_model') as mock_training:
                from automl_framework.models.data_models import PerformanceMetrics
                
                mock_training.return_value = {
                    'model': Mock(),
                    'training_history': {
                        'train_loss': [1.2, 0.8, 0.6, 0.4, 0.3],
                        'val_loss': [1.3, 0.9, 0.7, 0.5, 0.4]
                    },
                    'final_metrics': PerformanceMetrics(
                        accuracy=0.0,  # Not applicable for regression
                        loss=0.3,
                        precision=0.0,  # Not applicable
                        recall=0.0,     # Not applicable
                        f1_score=0.0,   # Not applicable
                        training_time=1200.0,
                        inference_time=0.003
                    )
                }
                
                training_result = self.training_service.train_model(
                    architectures[0],
                    Mock(),  # training config
                    processed_data['X_train'],
                    processed_data['y_train'],
                    processed_data['X_val'],
                    processed_data['y_val']
                )
                
                assert 'model' in training_result
                assert training_result['final_metrics'].loss < 1.0
        
        finally:
            os.unlink(temp_file.name)
    
    def test_pipeline_error_handling(self, test_data_manager):
        """Test error handling throughout the pipeline."""
        # Test with corrupted dataset
        corrupted_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        corrupted_file.write("invalid,csv,data\n1,2\n3,4,5,6\n")  # Inconsistent columns
        corrupted_file.close()
        
        try:
            # Should handle corrupted data gracefully
            with pytest.raises((ValueError, pd.errors.ParserError)):
                self.data_service.process_dataset(corrupted_file.name)
        
        finally:
            os.unlink(corrupted_file.name)
        
        # Test with empty dataset
        empty_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        empty_file.write("col1,col2,target\n")  # Header only
        empty_file.close()
        
        try:
            with pytest.raises(ValueError, match="Dataset is empty"):
                self.data_service.process_dataset(empty_file.name)
        
        finally:
            os.unlink(empty_file.name)
    
    def test_pipeline_with_different_data_types(self, test_data_manager):
        """Test pipeline with different data types."""
        # Test with mostly categorical data
        categorical_data = {
            'category_1': np.random.choice(['A', 'B', 'C', 'D'], 1000),
            'category_2': np.random.choice(['X', 'Y', 'Z'], 1000),
            'category_3': np.random.choice(['P', 'Q', 'R', 'S', 'T'], 1000),
            'numeric_1': np.random.normal(0, 1, 1000),
            'numeric_2': np.random.uniform(0, 100, 1000),
            'target': np.random.randint(0, 2, 1000)
        }
        
        df = pd.DataFrame(categorical_data)
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            processed_data = self.data_service.process_dataset(
                temp_file.name,
                task_type=TaskType.CLASSIFICATION
            )
            
            # Should handle categorical encoding
            assert 'X_train' in processed_data
            assert processed_data['X_train'].shape[1] > 5  # Should have more features after encoding
            
            # Verify no categorical data remains
            assert not any(processed_data['X_train'].dtypes == 'object')
        
        finally:
            os.unlink(temp_file.name)


class TestAPIIntegration:
    """Test API integration with AutoML services."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from automl_framework.api.main import app
        from fastapi.testclient import TestClient
        
        self.client = TestClient(app)
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('automl_framework.api.routes.experiments.ExperimentManager')
    def test_create_experiment_api(self, mock_experiment_manager):
        """Test experiment creation through API."""
        # Mock experiment manager
        mock_manager = Mock()
        mock_experiment = Mock()
        mock_experiment.id = "exp_123"
        mock_experiment.name = "Test Experiment"
        mock_experiment.status = ExperimentStatus.CREATED
        mock_manager.create_experiment.return_value = mock_experiment
        mock_experiment_manager.return_value = mock_manager
        
        # Test API call
        response = self.client.post(
            "/api/v1/experiments",
            json={
                "name": "Test Experiment",
                "dataset_id": "dataset_123",
                "task_type": "classification",
                "config": {
                    "max_trials": 10,
                    "max_time_minutes": 60
                }
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "exp_123"
        assert data["name"] == "Test Experiment"
        assert data["status"] == "created"
    
    @patch('automl_framework.api.routes.datasets.DataProcessingService')
    def test_upload_dataset_api(self, mock_data_service):
        """Test dataset upload through API."""
        # Mock data service
        mock_service = Mock()
        mock_service.analyze_dataset.return_value = {
            'data_type': DataType.TABULAR,
            'n_samples': 1000,
            'n_features': 10,
            'feature_types': {'feature1': 'numeric'},
            'target_column': 'target'
        }
        mock_data_service.return_value = mock_service
        
        # Create test CSV file
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'target': [0, 1, 0, 1, 0]
        })
        
        csv_content = test_data.to_csv(index=False)
        
        # Test file upload
        response = self.client.post(
            "/api/v1/datasets/upload",
            files={"file": ("test.csv", csv_content, "text/csv")},
            data={"name": "Test Dataset"}
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "dataset_id" in data
        assert data["name"] == "Test Dataset"
        assert "analysis" in data
    
    @patch('automl_framework.api.routes.experiments.ExperimentManager')
    def test_start_experiment_api(self, mock_experiment_manager):
        """Test starting experiment through API."""
        # Mock experiment manager
        mock_manager = Mock()
        mock_manager.start_experiment.return_value = True
        mock_experiment_manager.return_value = mock_manager
        
        # Test API call
        response = self.client.post("/api/v1/experiments/exp_123/start")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Experiment started successfully"
    
    @patch('automl_framework.api.routes.experiments.ExperimentManager')
    def test_get_experiment_status_api(self, mock_experiment_manager):
        """Test getting experiment status through API."""
        # Mock experiment manager
        mock_manager = Mock()
        mock_experiment = Mock()
        mock_experiment.id = "exp_123"
        mock_experiment.status = ExperimentStatus.RUNNING
        mock_experiment.progress = 0.5
        mock_manager.get_experiment.return_value = mock_experiment
        mock_experiment_manager.return_value = mock_manager
        
        # Test API call
        response = self.client.get("/api/v1/experiments/exp_123/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "exp_123"
        assert data["status"] == "running"
        assert data["progress"] == 0.5


class TestDatabaseIntegration:
    """Test database integration and data persistence."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from automl_framework.core.database import DatabaseManager
        
        # Use in-memory SQLite for testing
        self.db_manager = DatabaseManager(database_url="sqlite:///:memory:")
        self.db_manager.create_tables()
    
    def test_experiment_persistence(self):
        """Test experiment data persistence."""
        from automl_framework.models.orm_models import ExperimentORM
        
        # Create experiment
        experiment_data = {
            'id': 'exp_test_001',
            'name': 'Test Experiment',
            'dataset_id': 'dataset_001',
            'status': ExperimentStatus.CREATED.value,
            'created_at': datetime.now(),
            'config': {'max_trials': 10}
        }
        
        # Save to database
        with self.db_manager.get_session() as session:
            experiment_orm = ExperimentORM(**experiment_data)
            session.add(experiment_orm)
            session.commit()
            
            # Retrieve from database
            retrieved = session.query(ExperimentORM).filter_by(id='exp_test_001').first()
            
            assert retrieved is not None
            assert retrieved.name == 'Test Experiment'
            assert retrieved.status == ExperimentStatus.CREATED.value
    
    def test_dataset_metadata_persistence(self):
        """Test dataset metadata persistence."""
        from automl_framework.models.orm_models import DatasetORM
        
        # Create dataset metadata
        dataset_data = {
            'id': 'dataset_test_001',
            'name': 'Test Dataset',
            'file_path': '/path/to/dataset.csv',
            'data_type': DataType.TABULAR.value,
            'size': 1000,
            'created_at': datetime.now(),
            'metadata': {
                'n_features': 10,
                'feature_types': {'feature1': 'numeric'}
            }
        }
        
        # Save to database
        with self.db_manager.get_session() as session:
            dataset_orm = DatasetORM(**dataset_data)
            session.add(dataset_orm)
            session.commit()
            
            # Retrieve from database
            retrieved = session.query(DatasetORM).filter_by(id='dataset_test_001').first()
            
            assert retrieved is not None
            assert retrieved.name == 'Test Dataset'
            assert retrieved.data_type == DataType.TABULAR.value
            assert retrieved.metadata['n_features'] == 10
    
    def test_model_results_persistence(self):
        """Test model results persistence."""
        from automl_framework.models.orm_models import ModelORM
        
        # Create model results
        model_data = {
            'id': 'model_test_001',
            'experiment_id': 'exp_test_001',
            'architecture_id': 'arch_001',
            'hyperparameters': {
                'learning_rate': 0.001,
                'batch_size': 32
            },
            'metrics': {
                'accuracy': 0.95,
                'loss': 0.05
            },
            'training_time': 1800.0,
            'created_at': datetime.now()
        }
        
        # Save to database
        with self.db_manager.get_session() as session:
            model_orm = ModelORM(**model_data)
            session.add(model_orm)
            session.commit()
            
            # Retrieve from database
            retrieved = session.query(ModelORM).filter_by(id='model_test_001').first()
            
            assert retrieved is not None
            assert retrieved.experiment_id == 'exp_test_001'
            assert retrieved.metrics['accuracy'] == 0.95
            assert retrieved.hyperparameters['learning_rate'] == 0.001


class TestMultiServiceCommunication:
    """Test communication between multiple services."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.experiment_manager = ExperimentManager()
        self.data_service = DataProcessingService()
        self.nas_service = NASService()
        self.training_service = TrainingService()
    
    @patch('automl_framework.services.experiment_manager.ResourceScheduler')
    @patch('automl_framework.services.training_service.torch')
    def test_experiment_service_coordination(self, mock_torch, mock_scheduler):
        """Test coordination between experiment manager and other services."""
        # Mock resource scheduler
        mock_scheduler_instance = Mock()
        mock_scheduler_instance.allocate_resources.return_value = {
            'gpu_ids': [0],
            'memory_gb': 8,
            'cpu_cores': 4
        }
        mock_scheduler.return_value = mock_scheduler_instance
        
        # Mock experiment workflow
        with patch.object(self.data_service, 'process_dataset') as mock_data:
            mock_data.return_value = {
                'X_train': np.random.random((100, 10)),
                'y_train': np.random.randint(0, 2, 100),
                'X_val': np.random.random((20, 10)),
                'y_val': np.random.randint(0, 2, 20),
                'preprocessing_pipeline': Mock()
            }
            
            with patch.object(self.nas_service, 'search_architectures') as mock_nas:
                mock_nas.return_value = [
                    MockArchitectureGenerator.create_simple_mlp()
                ]
                
                with patch.object(self.training_service, 'train_model') as mock_training:
                    from automl_framework.models.data_models import PerformanceMetrics
                    
                    mock_training.return_value = {
                        'model': Mock(),
                        'final_metrics': PerformanceMetrics(
                            accuracy=0.9, loss=0.1, precision=0.88,
                            recall=0.92, f1_score=0.90,
                            training_time=100.0, inference_time=0.01
                        )
                    }
                    
                    # Test experiment execution
                    experiment_config = {
                        'dataset_path': '/path/to/dataset.csv',
                        'task_type': TaskType.CLASSIFICATION,
                        'max_trials': 5,
                        'max_time_minutes': 30
                    }
                    
                    result = self.experiment_manager.run_experiment(
                        experiment_id="test_exp_001",
                        config=experiment_config
                    )
                    
                    # Verify service interactions
                    mock_data.assert_called_once()
                    mock_nas.assert_called_once()
                    mock_training.assert_called()
                    
                    assert 'best_model' in result
                    assert 'final_metrics' in result
    
    def test_service_error_propagation(self):
        """Test error propagation between services."""
        # Test data service error propagation
        with patch.object(self.data_service, 'process_dataset') as mock_data:
            mock_data.side_effect = ValueError("Invalid dataset format")
            
            with pytest.raises(ValueError, match="Invalid dataset format"):
                self.experiment_manager.run_experiment(
                    experiment_id="test_exp_error",
                    config={'dataset_path': '/invalid/path.csv'}
                )
        
        # Test NAS service error propagation
        with patch.object(self.data_service, 'process_dataset') as mock_data:
            mock_data.return_value = {'X_train': np.random.random((100, 10))}
            
            with patch.object(self.nas_service, 'search_architectures') as mock_nas:
                mock_nas.side_effect = RuntimeError("Architecture search failed")
                
                with pytest.raises(RuntimeError, match="Architecture search failed"):
                    self.experiment_manager.run_experiment(
                        experiment_id="test_exp_nas_error",
                        config={'dataset_path': '/path/to/dataset.csv'}
                    )


if __name__ == "__main__":
    pytest.main([__file__])