"""
Unit tests for ExperimentManager class.
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from automl_framework.services.experiment_manager import ExperimentManager, PipelineJob, JobDependency
from automl_framework.models.data_models import (
    Experiment, ExperimentStatus, ExperimentResults, 
    DataType, TaskType, Architecture, TrainingConfig, PerformanceMetrics
)
from automl_framework.core.exceptions import ExperimentError, ValidationError


class TestExperimentManager:
    """Test cases for ExperimentManager class."""
    
    @pytest.fixture
    def experiment_manager(self):
        """Create ExperimentManager instance for testing."""
        return ExperimentManager(max_concurrent_experiments=2)
    
    @pytest.fixture
    def sample_experiment_config(self):
        """Create sample experiment configuration."""
        # Create a temporary CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write("feature1,feature2,target\n1,2,0\n3,4,1\n5,6,0\n")
        temp_file.close()
        
        return {
            'name': 'Test Experiment',
            'dataset_path': temp_file.name,
            'task_type': 'classification',
            'data_type': 'tabular',
            'max_trials': 10
        }
    
    @pytest.fixture
    def mock_services(self):
        """Mock all required services."""
        with patch('automl_framework.services.experiment_manager.get_service_registry') as mock_registry:
            # Create mock services
            mock_data_processor = Mock()
            mock_nas_service = Mock()
            mock_hpo_service = Mock()
            mock_trainer = Mock()
            mock_evaluator = Mock()
            
            # Configure mock registry
            mock_registry.return_value.get_service.side_effect = lambda name: {
                'data_processor': mock_data_processor,
                'nas_service': mock_nas_service,
                'hyperparameter_optimizer': mock_hpo_service,
                'model_trainer': mock_trainer,
                'model_evaluator': mock_evaluator
            }[name]
            
            yield {
                'data_processor': mock_data_processor,
                'nas_service': mock_nas_service,
                'hpo_service': mock_hpo_service,
                'trainer': mock_trainer,
                'evaluator': mock_evaluator
            }
    
    def test_init(self, experiment_manager):
        """Test ExperimentManager initialization."""
        assert experiment_manager.max_concurrent_experiments == 2
        assert len(experiment_manager.active_experiments) == 0
        assert len(experiment_manager.experiment_jobs) == 0
        assert len(experiment_manager.job_dependencies) == 0
        assert experiment_manager.executor is not None
    
    @patch('automl_framework.services.experiment_manager.get_mongo_collection')
    def test_create_experiment_success(self, mock_mongo, experiment_manager, sample_experiment_config):
        """Test successful experiment creation."""
        # Mock MongoDB collection
        mock_collection = Mock()
        mock_mongo.return_value = mock_collection
        
        # Create experiment
        experiment_id = experiment_manager.create_experiment(
            sample_experiment_config['dataset_path'],
            sample_experiment_config
        )
        
        # Verify experiment ID is valid UUID
        assert uuid.UUID(experiment_id)
        
        # Verify experiment was saved to database
        mock_collection.replace_one.assert_called_once()
        
        # Verify progress tracking initialized
        assert experiment_id in experiment_manager.experiment_progress
        assert 'data_processing' in experiment_manager.experiment_progress[experiment_id]
        assert 'overall' in experiment_manager.experiment_progress[experiment_id]
    
    def test_create_experiment_invalid_config(self, experiment_manager):
        """Test experiment creation with invalid configuration."""
        invalid_config = {'name': 'Test'}  # Missing dataset_path
        
        with pytest.raises(ExperimentError):  # ValidationError is wrapped in ExperimentError
            experiment_manager.create_experiment('', invalid_config)
    
    def test_create_experiment_nonexistent_dataset(self, experiment_manager):
        """Test experiment creation with nonexistent dataset."""
        config = {
            'name': 'Test',
            'dataset_path': '/nonexistent/file.csv'
        }
        
        with pytest.raises(ExperimentError):  # ValidationError is wrapped in ExperimentError
            experiment_manager.create_experiment(config['dataset_path'], config)
    
    @patch('automl_framework.services.experiment_manager.get_mongo_collection')
    def test_run_experiment_success(self, mock_mongo, experiment_manager, sample_experiment_config, mock_services):
        """Test successful experiment execution."""
        # Mock MongoDB collection
        mock_collection = Mock()
        mock_mongo.return_value = mock_collection
        
        # Create experiment first
        experiment_id = experiment_manager.create_experiment(
            sample_experiment_config['dataset_path'],
            sample_experiment_config
        )
        
        # Mock database load
        mock_experiment = Experiment(
            id=experiment_id,
            name='Test Experiment',
            dataset_id='test_dataset',
            status=ExperimentStatus.CREATED,
            config=sample_experiment_config
        )
        mock_collection.find_one.return_value = {
            'id': experiment_id,
            'name': 'Test Experiment',
            'dataset_id': 'test_dataset',
            'status': 'created',
            'config': sample_experiment_config,
            'created_at': datetime.now().isoformat()
        }
        
        # Run experiment
        result = experiment_manager.run_experiment(experiment_id)
        
        assert result is True
        assert experiment_id in experiment_manager.active_experiments
        assert experiment_id in experiment_manager.experiment_futures
    
    @patch('automl_framework.services.experiment_manager.get_mongo_collection')
    def test_run_experiment_not_found(self, mock_mongo, experiment_manager):
        """Test running nonexistent experiment."""
        mock_collection = Mock()
        mock_collection.find_one.return_value = None
        mock_mongo.return_value = mock_collection
        
        with pytest.raises(ExperimentError):
            experiment_manager.run_experiment('nonexistent_id')
    
    def test_run_experiment_max_concurrent_limit(self, experiment_manager, sample_experiment_config):
        """Test maximum concurrent experiments limit."""
        # Fill up the concurrent experiment slots
        experiment_manager.active_experiments = {
            'exp1': Mock(),
            'exp2': Mock()
        }
        
        with pytest.raises(ExperimentError, match="Maximum concurrent experiments limit reached"):
            experiment_manager.run_experiment('exp3')
    
    @patch('automl_framework.services.experiment_manager.get_mongo_collection')
    def test_get_experiment_status(self, mock_mongo, experiment_manager, sample_experiment_config):
        """Test getting experiment status."""
        # Mock MongoDB collection
        mock_collection = Mock()
        mock_mongo.return_value = mock_collection
        
        # Create experiment
        experiment_id = experiment_manager.create_experiment(
            sample_experiment_config['dataset_path'],
            sample_experiment_config
        )
        
        # Mock database response
        mock_collection.find_one.return_value = {
            'id': experiment_id,
            'name': 'Test Experiment',
            'dataset_id': 'test_dataset',
            'status': 'created',
            'config': sample_experiment_config,
            'created_at': datetime.now().isoformat()
        }
        
        status = experiment_manager.get_experiment_status(experiment_id)
        assert status == ExperimentStatus.CREATED
    
    def test_get_experiment_status_not_found(self, experiment_manager):
        """Test getting status of nonexistent experiment."""
        with patch('automl_framework.services.experiment_manager.get_mongo_collection') as mock_mongo:
            mock_collection = Mock()
            mock_collection.find_one.return_value = None
            mock_mongo.return_value = mock_collection
            
            with pytest.raises(ExperimentError):
                experiment_manager.get_experiment_status('nonexistent_id')
    
    @patch('automl_framework.services.experiment_manager.get_mongo_collection')
    def test_get_experiment_results_completed(self, mock_mongo, experiment_manager):
        """Test getting results of completed experiment."""
        mock_collection = Mock()
        mock_mongo.return_value = mock_collection
        
        # Mock completed experiment
        experiment_id = 'test_exp_id'
        mock_collection.find_one.return_value = {
            'id': experiment_id,
            'name': 'Test Experiment',
            'dataset_id': 'test_dataset',
            'status': 'completed',
            'created_at': datetime.now().isoformat(),
            'completed_at': datetime.now().isoformat(),
            'results': {
                'best_architecture': None,
                'best_hyperparameters': None,
                'performance_metrics': None,
                'training_history': [],
                'model_path': None
            }
        }
        
        results = experiment_manager.get_experiment_results(experiment_id)
        
        assert results['experiment_id'] == experiment_id
        assert results['status'] == 'completed'
        assert results['results'] is not None
    
    @patch('automl_framework.services.experiment_manager.get_mongo_collection')
    def test_get_experiment_results_running(self, mock_mongo, experiment_manager):
        """Test getting results of running experiment."""
        mock_collection = Mock()
        mock_mongo.return_value = mock_collection
        
        # Mock running experiment
        experiment_id = 'test_exp_id'
        mock_collection.find_one.return_value = {
            'id': experiment_id,
            'name': 'Test Experiment',
            'dataset_id': 'test_dataset',
            'status': 'running',
            'created_at': datetime.now().isoformat()
        }
        
        # Initialize progress
        experiment_manager.experiment_progress[experiment_id] = {
            'data_processing': 50.0,
            'overall': 10.0
        }
        
        results = experiment_manager.get_experiment_results(experiment_id)
        
        assert results['experiment_id'] == experiment_id
        assert results['status'] == 'running'
        assert results['results'] is None
        assert results['progress']['data_processing'] == 50.0
    
    def test_get_experiment_progress(self, experiment_manager):
        """Test getting experiment progress."""
        experiment_id = 'test_exp_id'
        progress_data = {
            'data_processing': 75.0,
            'architecture_search': 50.0,
            'overall': 25.0
        }
        experiment_manager.experiment_progress[experiment_id] = progress_data
        
        progress = experiment_manager.get_experiment_progress(experiment_id)
        assert progress == progress_data
    
    def test_get_experiment_progress_not_found(self, experiment_manager):
        """Test getting progress of nonexistent experiment."""
        progress = experiment_manager.get_experiment_progress('nonexistent_id')
        assert progress == {}
    
    @patch('automl_framework.services.experiment_manager.get_mongo_collection')
    def test_cancel_experiment(self, mock_mongo, experiment_manager):
        """Test cancelling a running experiment."""
        mock_collection = Mock()
        mock_mongo.return_value = mock_collection
        
        # Add active experiment
        experiment_id = 'test_exp_id'
        mock_experiment = Mock()
        mock_experiment.status = ExperimentStatus.RUNNING
        experiment_manager.active_experiments[experiment_id] = mock_experiment
        
        # Add mock future
        mock_future = Mock()
        experiment_manager.experiment_futures[experiment_id] = mock_future
        
        result = experiment_manager.cancel_experiment(experiment_id)
        
        assert result is True
        assert experiment_id not in experiment_manager.active_experiments
        assert experiment_id not in experiment_manager.experiment_futures
        mock_future.cancel.assert_called_once()
        assert mock_experiment.status == ExperimentStatus.CANCELLED
    
    def test_cancel_experiment_not_active(self, experiment_manager):
        """Test cancelling non-active experiment."""
        result = experiment_manager.cancel_experiment('nonexistent_id')
        assert result is False
    
    @patch('automl_framework.services.experiment_manager.get_mongo_collection')
    def test_list_experiments(self, mock_mongo, experiment_manager):
        """Test listing all experiments."""
        mock_collection = Mock()
        mock_mongo.return_value = mock_collection
        
        # Mock experiments in database
        mock_collection.find.return_value = [
            {
                'id': 'exp1',
                'name': 'Experiment 1',
                'status': 'completed',
                'created_at': datetime.now().isoformat(),
                'completed_at': datetime.now().isoformat()
            },
            {
                'id': 'exp2',
                'name': 'Experiment 2',
                'status': 'running',
                'created_at': datetime.now().isoformat()
            }
        ]
        
        experiments = experiment_manager.list_experiments()
        
        assert len(experiments) == 2
        assert experiments[0]['id'] == 'exp1'
        assert experiments[0]['status'] == 'completed'
        assert experiments[1]['id'] == 'exp2'
        assert experiments[1]['status'] == 'running'
    
    @patch('automl_framework.services.experiment_manager.get_mongo_collection')
    def test_list_experiments_with_filter(self, mock_mongo, experiment_manager):
        """Test listing experiments with status filter."""
        mock_collection = Mock()
        mock_mongo.return_value = mock_collection
        
        # Mock experiments in database
        mock_collection.find.return_value = [
            {
                'id': 'exp1',
                'name': 'Experiment 1',
                'status': 'completed',
                'created_at': datetime.now().isoformat(),
                'completed_at': datetime.now().isoformat()
            },
            {
                'id': 'exp2',
                'name': 'Experiment 2',
                'status': 'running',
                'created_at': datetime.now().isoformat()
            }
        ]
        
        experiments = experiment_manager.list_experiments(status_filter=ExperimentStatus.COMPLETED)
        
        assert len(experiments) == 1
        assert experiments[0]['id'] == 'exp1'
        assert experiments[0]['status'] == 'completed'
    
    def test_add_progress_callback(self, experiment_manager):
        """Test adding progress callback."""
        experiment_id = 'test_exp_id'
        callback = Mock()
        
        experiment_manager.add_progress_callback(experiment_id, callback)
        
        assert experiment_id in experiment_manager.progress_callbacks
        assert callback in experiment_manager.progress_callbacks[experiment_id]
    
    def test_update_progress(self, experiment_manager):
        """Test updating experiment progress."""
        experiment_id = 'test_exp_id'
        experiment_manager.experiment_progress[experiment_id] = {
            'data_processing': 0.0,
            'architecture_search': 0.0,
            'hyperparameter_optimization': 0.0,
            'model_training': 0.0,
            'model_evaluation': 0.0,
            'overall': 0.0
        }
        
        # Add progress callback
        callback = Mock()
        experiment_manager.add_progress_callback(experiment_id, callback)
        
        # Update progress
        experiment_manager._update_progress(experiment_id, 'data_processing', 50.0)
        
        assert experiment_manager.experiment_progress[experiment_id]['data_processing'] == 50.0
        assert experiment_manager.experiment_progress[experiment_id]['overall'] == 10.0  # 50/5 stages
        callback.assert_called_once()
    
    def test_validate_experiment_config_valid(self, experiment_manager, sample_experiment_config):
        """Test validating valid experiment configuration."""
        # Should not raise any exception
        experiment_manager._validate_experiment_config(sample_experiment_config)
    
    def test_validate_experiment_config_missing_dataset_path(self, experiment_manager):
        """Test validating config missing dataset_path."""
        config = {'name': 'Test'}
        
        with pytest.raises(ValidationError, match="Missing required field: dataset_path"):
            experiment_manager._validate_experiment_config(config)
    
    def test_validate_experiment_config_nonexistent_file(self, experiment_manager):
        """Test validating config with nonexistent dataset file."""
        config = {'dataset_path': '/nonexistent/file.csv'}
        
        with pytest.raises(ValidationError, match="Dataset file not found"):
            experiment_manager._validate_experiment_config(config)
    
    def test_shutdown(self, experiment_manager):
        """Test shutting down experiment manager."""
        # Add some active experiments
        experiment_manager.active_experiments['exp1'] = Mock()
        experiment_manager.active_experiments['exp2'] = Mock()
        
        with patch.object(experiment_manager, 'cancel_experiment') as mock_cancel:
            experiment_manager.shutdown()
            
            # Verify all experiments were cancelled
            assert mock_cancel.call_count == 2
            
            # Verify executor was shutdown
            assert experiment_manager.executor._shutdown


class TestPipelineJob:
    """Test cases for PipelineJob class."""
    
    def test_init(self):
        """Test PipelineJob initialization."""
        job_function = Mock()
        job = PipelineJob(
            job_id='test_job',
            job_type='test_type',
            job_function=job_function,
            args=(1, 2),
            kwargs={'key': 'value'}
        )
        
        assert job.job_id == 'test_job'
        assert job.job_type == 'test_type'
        assert job.job_function == job_function
        assert job.args == (1, 2)
        assert job.kwargs == {'key': 'value'}
        assert job.status == 'pending'
        assert job.result is None
        assert job.error is None
        assert job.progress == 0.0


class TestJobDependency:
    """Test cases for JobDependency class."""
    
    def test_init(self):
        """Test JobDependency initialization."""
        dependency = JobDependency('job1', ['job2', 'job3'])
        
        assert dependency.job_id == 'job1'
        assert dependency.depends_on == ['job2', 'job3']
        assert dependency.completed is False
        assert dependency.result is None
        assert dependency.error is None


if __name__ == '__main__':
    pytest.main([__file__])