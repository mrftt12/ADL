"""
Database integration tests for AutoML framework.

Tests database operations, data persistence, transactions,
and database schema integrity.
"""

import pytest
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import json

from automl_framework.core.database import DatabaseManager
from automl_framework.models.orm_models import (
    ExperimentORM, DatasetORM, ModelORM, UserORM, 
    ArchitectureORM, HyperparameterTrialORM
)
from automl_framework.models.data_models import (
    ExperimentStatus, DataType, TaskType
)
from tests.test_utils import (
    MockExperimentGenerator, MockArchitectureGenerator
)


class TestDatabaseConnection:
    """Test database connection and basic operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use in-memory SQLite for testing
        self.db_manager = DatabaseManager(database_url="sqlite:///:memory:")
        self.db_manager.create_tables()
    
    def test_database_connection(self):
        """Test database connection."""
        assert self.db_manager.engine is not None
        assert self.db_manager.SessionLocal is not None
    
    def test_create_tables(self):
        """Test table creation."""
        # Tables should be created without errors
        self.db_manager.create_tables()
        
        # Verify tables exist by trying to query them
        with self.db_manager.get_session() as session:
            # Should not raise exceptions
            session.query(ExperimentORM).count()
            session.query(DatasetORM).count()
            session.query(ModelORM).count()
            session.query(UserORM).count()
    
    def test_session_management(self):
        """Test database session management."""
        # Test context manager
        with self.db_manager.get_session() as session:
            assert session is not None
            # Session should be active
            assert session.is_active
        
        # Session should be closed after context
        # Note: SQLAlchemy sessions don't have a simple "is_closed" check
    
    def test_transaction_rollback(self):
        """Test transaction rollback on error."""
        with pytest.raises(Exception):
            with self.db_manager.get_session() as session:
                # Create valid experiment
                experiment = ExperimentORM(
                    id='test_exp_001',
                    name='Test Experiment',
                    dataset_id='dataset_001',
                    status=ExperimentStatus.CREATED.value,
                    created_at=datetime.now()
                )
                session.add(experiment)
                session.flush()  # Flush to database
                
                # Verify it exists
                assert session.query(ExperimentORM).filter_by(id='test_exp_001').first() is not None
                
                # Cause an error
                raise Exception("Test error")
        
        # After rollback, experiment should not exist
        with self.db_manager.get_session() as session:
            assert session.query(ExperimentORM).filter_by(id='test_exp_001').first() is None


class TestExperimentPersistence:
    """Test experiment data persistence."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.db_manager = DatabaseManager(database_url="sqlite:///:memory:")
        self.db_manager.create_tables()
    
    def test_create_experiment(self):
        """Test creating experiment in database."""
        experiment_data = {
            'id': 'exp_test_001',
            'name': 'Test Classification Experiment',
            'description': 'Testing experiment persistence',
            'dataset_id': 'dataset_001',
            'status': ExperimentStatus.CREATED.value,
            'task_type': TaskType.CLASSIFICATION.value,
            'created_at': datetime.now(),
            'config': {
                'max_trials': 20,
                'max_time_minutes': 120,
                'optimization_metric': 'accuracy'
            }
        }
        
        with self.db_manager.get_session() as session:
            experiment = ExperimentORM(**experiment_data)
            session.add(experiment)
            session.commit()
            
            # Verify experiment was saved
            saved_experiment = session.query(ExperimentORM).filter_by(id='exp_test_001').first()
            assert saved_experiment is not None
            assert saved_experiment.name == 'Test Classification Experiment'
            assert saved_experiment.status == ExperimentStatus.CREATED.value
            assert saved_experiment.config['max_trials'] == 20
    
    def test_update_experiment_status(self):
        """Test updating experiment status."""
        # Create experiment
        with self.db_manager.get_session() as session:
            experiment = ExperimentORM(
                id='exp_update_001',
                name='Update Test',
                dataset_id='dataset_001',
                status=ExperimentStatus.CREATED.value,
                created_at=datetime.now()
            )
            session.add(experiment)
            session.commit()
        
        # Update status
        with self.db_manager.get_session() as session:
            experiment = session.query(ExperimentORM).filter_by(id='exp_update_001').first()
            experiment.status = ExperimentStatus.RUNNING.value
            experiment.started_at = datetime.now()
            session.commit()
        
        # Verify update
        with self.db_manager.get_session() as session:
            updated_experiment = session.query(ExperimentORM).filter_by(id='exp_update_001').first()
            assert updated_experiment.status == ExperimentStatus.RUNNING.value
            assert updated_experiment.started_at is not None
    
    def test_experiment_completion(self):
        """Test marking experiment as completed."""
        # Create and start experiment
        with self.db_manager.get_session() as session:
            experiment = ExperimentORM(
                id='exp_complete_001',
                name='Completion Test',
                dataset_id='dataset_001',
                status=ExperimentStatus.RUNNING.value,
                created_at=datetime.now() - timedelta(hours=1),
                started_at=datetime.now() - timedelta(minutes=30)
            )
            session.add(experiment)
            session.commit()
        
        # Complete experiment
        completion_time = datetime.now()
        results = {
            'best_accuracy': 0.95,
            'best_model_id': 'model_001',
            'total_trials': 15,
            'best_hyperparameters': {
                'learning_rate': 0.001,
                'batch_size': 32
            }
        }
        
        with self.db_manager.get_session() as session:
            experiment = session.query(ExperimentORM).filter_by(id='exp_complete_001').first()
            experiment.status = ExperimentStatus.COMPLETED.value
            experiment.completed_at = completion_time
            experiment.results = results
            session.commit()
        
        # Verify completion
        with self.db_manager.get_session() as session:
            completed_experiment = session.query(ExperimentORM).filter_by(id='exp_complete_001').first()
            assert completed_experiment.status == ExperimentStatus.COMPLETED.value
            assert completed_experiment.completed_at == completion_time
            assert completed_experiment.results['best_accuracy'] == 0.95
    
    def test_experiment_failure(self):
        """Test handling experiment failure."""
        with self.db_manager.get_session() as session:
            experiment = ExperimentORM(
                id='exp_fail_001',
                name='Failure Test',
                dataset_id='dataset_001',
                status=ExperimentStatus.RUNNING.value,
                created_at=datetime.now() - timedelta(minutes=30),
                started_at=datetime.now() - timedelta(minutes=15)
            )
            session.add(experiment)
            session.commit()
        
        # Mark as failed
        error_message = "Training failed due to insufficient memory"
        with self.db_manager.get_session() as session:
            experiment = session.query(ExperimentORM).filter_by(id='exp_fail_001').first()
            experiment.status = ExperimentStatus.FAILED.value
            experiment.completed_at = datetime.now()
            experiment.error_message = error_message
            session.commit()
        
        # Verify failure
        with self.db_manager.get_session() as session:
            failed_experiment = session.query(ExperimentORM).filter_by(id='exp_fail_001').first()
            assert failed_experiment.status == ExperimentStatus.FAILED.value
            assert failed_experiment.error_message == error_message
    
    def test_list_experiments_by_status(self):
        """Test querying experiments by status."""
        # Create experiments with different statuses
        experiments_data = [
            {'id': 'exp_created_001', 'status': ExperimentStatus.CREATED.value},
            {'id': 'exp_running_001', 'status': ExperimentStatus.RUNNING.value},
            {'id': 'exp_completed_001', 'status': ExperimentStatus.COMPLETED.value},
            {'id': 'exp_failed_001', 'status': ExperimentStatus.FAILED.value},
            {'id': 'exp_running_002', 'status': ExperimentStatus.RUNNING.value}
        ]
        
        with self.db_manager.get_session() as session:
            for exp_data in experiments_data:
                experiment = ExperimentORM(
                    id=exp_data['id'],
                    name=f"Test {exp_data['id']}",
                    dataset_id='dataset_001',
                    status=exp_data['status'],
                    created_at=datetime.now()
                )
                session.add(experiment)
            session.commit()
        
        # Query by status
        with self.db_manager.get_session() as session:
            running_experiments = session.query(ExperimentORM).filter_by(
                status=ExperimentStatus.RUNNING.value
            ).all()
            
            assert len(running_experiments) == 2
            running_ids = [exp.id for exp in running_experiments]
            assert 'exp_running_001' in running_ids
            assert 'exp_running_002' in running_ids
            
            completed_experiments = session.query(ExperimentORM).filter_by(
                status=ExperimentStatus.COMPLETED.value
            ).all()
            
            assert len(completed_experiments) == 1
            assert completed_experiments[0].id == 'exp_completed_001'


class TestDatasetPersistence:
    """Test dataset data persistence."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.db_manager = DatabaseManager(database_url="sqlite:///:memory:")
        self.db_manager.create_tables()
    
    def test_create_dataset(self):
        """Test creating dataset in database."""
        dataset_data = {
            'id': 'dataset_test_001',
            'name': 'Test Tabular Dataset',
            'description': 'Dataset for testing persistence',
            'file_path': '/path/to/dataset.csv',
            'data_type': DataType.TABULAR.value,
            'size': 10000,
            'created_at': datetime.now(),
            'metadata': {
                'n_features': 15,
                'n_classes': 3,
                'feature_types': {
                    'feature1': 'numeric',
                    'feature2': 'categorical'
                },
                'missing_values': {
                    'feature1': 5.2,
                    'feature2': 0.0
                },
                'target_column': 'target'
            }
        }
        
        with self.db_manager.get_session() as session:
            dataset = DatasetORM(**dataset_data)
            session.add(dataset)
            session.commit()
            
            # Verify dataset was saved
            saved_dataset = session.query(DatasetORM).filter_by(id='dataset_test_001').first()
            assert saved_dataset is not None
            assert saved_dataset.name == 'Test Tabular Dataset'
            assert saved_dataset.data_type == DataType.TABULAR.value
            assert saved_dataset.size == 10000
            assert saved_dataset.metadata['n_features'] == 15
            assert saved_dataset.metadata['target_column'] == 'target'
    
    def test_dataset_file_operations(self):
        """Test dataset file path operations."""
        with self.db_manager.get_session() as session:
            dataset = DatasetORM(
                id='dataset_file_001',
                name='File Test Dataset',
                file_path='/original/path/dataset.csv',
                data_type=DataType.TABULAR.value,
                size=5000,
                created_at=datetime.now()
            )
            session.add(dataset)
            session.commit()
        
        # Update file path (e.g., after moving file)
        new_path = '/new/path/dataset.csv'
        with self.db_manager.get_session() as session:
            dataset = session.query(DatasetORM).filter_by(id='dataset_file_001').first()
            dataset.file_path = new_path
            dataset.updated_at = datetime.now()
            session.commit()
        
        # Verify update
        with self.db_manager.get_session() as session:
            updated_dataset = session.query(DatasetORM).filter_by(id='dataset_file_001').first()
            assert updated_dataset.file_path == new_path
            assert updated_dataset.updated_at is not None
    
    def test_list_datasets_by_type(self):
        """Test querying datasets by type."""
        datasets_data = [
            {'id': 'dataset_tabular_001', 'data_type': DataType.TABULAR.value},
            {'id': 'dataset_image_001', 'data_type': DataType.IMAGE.value},
            {'id': 'dataset_text_001', 'data_type': DataType.TEXT.value},
            {'id': 'dataset_tabular_002', 'data_type': DataType.TABULAR.value}
        ]
        
        with self.db_manager.get_session() as session:
            for ds_data in datasets_data:
                dataset = DatasetORM(
                    id=ds_data['id'],
                    name=f"Test {ds_data['id']}",
                    file_path=f"/path/{ds_data['id']}.csv",
                    data_type=ds_data['data_type'],
                    size=1000,
                    created_at=datetime.now()
                )
                session.add(dataset)
            session.commit()
        
        # Query by type
        with self.db_manager.get_session() as session:
            tabular_datasets = session.query(DatasetORM).filter_by(
                data_type=DataType.TABULAR.value
            ).all()
            
            assert len(tabular_datasets) == 2
            tabular_ids = [ds.id for ds in tabular_datasets]
            assert 'dataset_tabular_001' in tabular_ids
            assert 'dataset_tabular_002' in tabular_ids
            
            image_datasets = session.query(DatasetORM).filter_by(
                data_type=DataType.IMAGE.value
            ).all()
            
            assert len(image_datasets) == 1
            assert image_datasets[0].id == 'dataset_image_001'


class TestModelPersistence:
    """Test model data persistence."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.db_manager = DatabaseManager(database_url="sqlite:///:memory:")
        self.db_manager.create_tables()
    
    def test_create_model(self):
        """Test creating model in database."""
        model_data = {
            'id': 'model_test_001',
            'experiment_id': 'exp_001',
            'architecture_id': 'arch_001',
            'name': 'Best Classification Model',
            'hyperparameters': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'optimizer': 'adam',
                'epochs': 50
            },
            'metrics': {
                'accuracy': 0.95,
                'precision': 0.93,
                'recall': 0.97,
                'f1_score': 0.95,
                'loss': 0.05
            },
            'training_time': 1800.0,
            'model_path': '/path/to/model.pth',
            'created_at': datetime.now()
        }
        
        with self.db_manager.get_session() as session:
            model = ModelORM(**model_data)
            session.add(model)
            session.commit()
            
            # Verify model was saved
            saved_model = session.query(ModelORM).filter_by(id='model_test_001').first()
            assert saved_model is not None
            assert saved_model.name == 'Best Classification Model'
            assert saved_model.experiment_id == 'exp_001'
            assert saved_model.hyperparameters['learning_rate'] == 0.001
            assert saved_model.metrics['accuracy'] == 0.95
            assert saved_model.training_time == 1800.0
    
    def test_model_versioning(self):
        """Test model versioning."""
        base_model_data = {
            'experiment_id': 'exp_001',
            'architecture_id': 'arch_001',
            'hyperparameters': {'learning_rate': 0.001},
            'metrics': {'accuracy': 0.90},
            'training_time': 1200.0,
            'created_at': datetime.now()
        }
        
        # Create multiple versions
        versions = [
            {'id': 'model_v1_001', 'version': '1.0', 'metrics': {'accuracy': 0.90}},
            {'id': 'model_v2_001', 'version': '1.1', 'metrics': {'accuracy': 0.93}},
            {'id': 'model_v3_001', 'version': '1.2', 'metrics': {'accuracy': 0.95}}
        ]
        
        with self.db_manager.get_session() as session:
            for version_data in versions:
                model_data = {**base_model_data, **version_data}
                model = ModelORM(**model_data)
                session.add(model)
            session.commit()
        
        # Query models by experiment
        with self.db_manager.get_session() as session:
            experiment_models = session.query(ModelORM).filter_by(
                experiment_id='exp_001'
            ).order_by(ModelORM.created_at).all()
            
            assert len(experiment_models) == 3
            
            # Verify versions are in order
            versions_found = [model.version for model in experiment_models]
            assert versions_found == ['1.0', '1.1', '1.2']
            
            # Verify accuracy improvement
            accuracies = [model.metrics['accuracy'] for model in experiment_models]
            assert accuracies == [0.90, 0.93, 0.95]
    
    def test_best_model_selection(self):
        """Test selecting best model by metric."""
        models_data = [
            {
                'id': 'model_001',
                'experiment_id': 'exp_001',
                'metrics': {'accuracy': 0.85, 'f1_score': 0.83}
            },
            {
                'id': 'model_002',
                'experiment_id': 'exp_001',
                'metrics': {'accuracy': 0.92, 'f1_score': 0.90}
            },
            {
                'id': 'model_003',
                'experiment_id': 'exp_001',
                'metrics': {'accuracy': 0.88, 'f1_score': 0.91}
            }
        ]
        
        with self.db_manager.get_session() as session:
            for model_data in models_data:
                model = ModelORM(
                    id=model_data['id'],
                    experiment_id=model_data['experiment_id'],
                    architecture_id='arch_001',
                    hyperparameters={},
                    metrics=model_data['metrics'],
                    training_time=1000.0,
                    created_at=datetime.now()
                )
                session.add(model)
            session.commit()
        
        # Find best model by accuracy
        with self.db_manager.get_session() as session:
            # This would typically be done with a custom query
            models = session.query(ModelORM).filter_by(experiment_id='exp_001').all()
            best_accuracy_model = max(models, key=lambda m: m.metrics['accuracy'])
            
            assert best_accuracy_model.id == 'model_002'
            assert best_accuracy_model.metrics['accuracy'] == 0.92
            
            # Find best model by F1 score
            best_f1_model = max(models, key=lambda m: m.metrics['f1_score'])
            
            assert best_f1_model.id == 'model_003'
            assert best_f1_model.metrics['f1_score'] == 0.91


class TestArchitecturePersistence:
    """Test architecture data persistence."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.db_manager = DatabaseManager(database_url="sqlite:///:memory:")
        self.db_manager.create_tables()
    
    def test_create_architecture(self):
        """Test creating architecture in database."""
        architecture = MockArchitectureGenerator.create_simple_mlp()
        
        architecture_data = {
            'id': architecture.id,
            'name': 'Simple MLP',
            'description': 'Multi-layer perceptron for classification',
            'architecture_json': {
                'layers': [
                    {'type': 'dense', 'units': 128, 'activation': 'relu'},
                    {'type': 'dense', 'units': 64, 'activation': 'relu'},
                    {'type': 'dense', 'units': 10, 'activation': 'softmax'}
                ],
                'connections': [
                    {'from': 0, 'to': 1},
                    {'from': 1, 'to': 2}
                ]
            },
            'input_shape': architecture.input_shape,
            'output_shape': architecture.output_shape,
            'parameter_count': architecture.parameter_count,
            'flops': architecture.flops,
            'task_type': TaskType.CLASSIFICATION.value,
            'created_at': datetime.now()
        }
        
        with self.db_manager.get_session() as session:
            arch_orm = ArchitectureORM(**architecture_data)
            session.add(arch_orm)
            session.commit()
            
            # Verify architecture was saved
            saved_arch = session.query(ArchitectureORM).filter_by(id=architecture.id).first()
            assert saved_arch is not None
            assert saved_arch.name == 'Simple MLP'
            assert saved_arch.parameter_count == architecture.parameter_count
            assert saved_arch.input_shape == architecture.input_shape
            assert len(saved_arch.architecture_json['layers']) == 3


class TestHyperparameterTrialPersistence:
    """Test hyperparameter trial persistence."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.db_manager = DatabaseManager(database_url="sqlite:///:memory:")
        self.db_manager.create_tables()
    
    def test_create_hyperparameter_trial(self):
        """Test creating hyperparameter trial in database."""
        trial_data = {
            'id': 'trial_001',
            'experiment_id': 'exp_001',
            'trial_number': 1,
            'hyperparameters': {
                'learning_rate': 0.01,
                'batch_size': 64,
                'optimizer': 'sgd',
                'weight_decay': 1e-4
            },
            'metrics': {
                'accuracy': 0.87,
                'loss': 0.35,
                'val_accuracy': 0.85,
                'val_loss': 0.38
            },
            'status': 'completed',
            'training_time': 1200.0,
            'created_at': datetime.now() - timedelta(minutes=30),
            'completed_at': datetime.now()
        }
        
        with self.db_manager.get_session() as session:
            trial = HyperparameterTrialORM(**trial_data)
            session.add(trial)
            session.commit()
            
            # Verify trial was saved
            saved_trial = session.query(HyperparameterTrialORM).filter_by(id='trial_001').first()
            assert saved_trial is not None
            assert saved_trial.experiment_id == 'exp_001'
            assert saved_trial.trial_number == 1
            assert saved_trial.hyperparameters['learning_rate'] == 0.01
            assert saved_trial.metrics['accuracy'] == 0.87
            assert saved_trial.status == 'completed'
    
    def test_hyperparameter_optimization_history(self):
        """Test storing hyperparameter optimization history."""
        experiment_id = 'exp_hpo_001'
        
        # Create multiple trials
        trials_data = [
            {
                'id': 'trial_001',
                'trial_number': 1,
                'hyperparameters': {'learning_rate': 0.1, 'batch_size': 32},
                'metrics': {'accuracy': 0.75, 'loss': 0.8}
            },
            {
                'id': 'trial_002',
                'trial_number': 2,
                'hyperparameters': {'learning_rate': 0.01, 'batch_size': 64},
                'metrics': {'accuracy': 0.85, 'loss': 0.4}
            },
            {
                'id': 'trial_003',
                'trial_number': 3,
                'hyperparameters': {'learning_rate': 0.001, 'batch_size': 32},
                'metrics': {'accuracy': 0.92, 'loss': 0.2}
            }
        ]
        
        with self.db_manager.get_session() as session:
            for trial_data in trials_data:
                trial = HyperparameterTrialORM(
                    id=trial_data['id'],
                    experiment_id=experiment_id,
                    trial_number=trial_data['trial_number'],
                    hyperparameters=trial_data['hyperparameters'],
                    metrics=trial_data['metrics'],
                    status='completed',
                    training_time=1000.0,
                    created_at=datetime.now(),
                    completed_at=datetime.now()
                )
                session.add(trial)
            session.commit()
        
        # Query optimization history
        with self.db_manager.get_session() as session:
            trials = session.query(HyperparameterTrialORM).filter_by(
                experiment_id=experiment_id
            ).order_by(HyperparameterTrialORM.trial_number).all()
            
            assert len(trials) == 3
            
            # Verify improvement over trials
            accuracies = [trial.metrics['accuracy'] for trial in trials]
            assert accuracies == [0.75, 0.85, 0.92]
            
            # Find best trial
            best_trial = max(trials, key=lambda t: t.metrics['accuracy'])
            assert best_trial.id == 'trial_003'
            assert best_trial.hyperparameters['learning_rate'] == 0.001


class TestDatabaseConstraints:
    """Test database constraints and data integrity."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.db_manager = DatabaseManager(database_url="sqlite:///:memory:")
        self.db_manager.create_tables()
    
    def test_unique_constraints(self):
        """Test unique constraints."""
        # Create first experiment
        with self.db_manager.get_session() as session:
            experiment1 = ExperimentORM(
                id='exp_unique_001',
                name='First Experiment',
                dataset_id='dataset_001',
                status=ExperimentStatus.CREATED.value,
                created_at=datetime.now()
            )
            session.add(experiment1)
            session.commit()
        
        # Try to create experiment with same ID
        with pytest.raises(Exception):  # Should raise integrity error
            with self.db_manager.get_session() as session:
                experiment2 = ExperimentORM(
                    id='exp_unique_001',  # Same ID
                    name='Second Experiment',
                    dataset_id='dataset_002',
                    status=ExperimentStatus.CREATED.value,
                    created_at=datetime.now()
                )
                session.add(experiment2)
                session.commit()
    
    def test_foreign_key_constraints(self):
        """Test foreign key constraints."""
        # This would test relationships between tables
        # For example, models should reference valid experiments
        
        # Create model without corresponding experiment
        with pytest.raises(Exception):  # Should raise foreign key error
            with self.db_manager.get_session() as session:
                model = ModelORM(
                    id='model_orphan_001',
                    experiment_id='nonexistent_exp',  # Non-existent experiment
                    architecture_id='arch_001',
                    hyperparameters={},
                    metrics={},
                    training_time=1000.0,
                    created_at=datetime.now()
                )
                session.add(model)
                session.commit()
    
    def test_data_type_validation(self):
        """Test data type validation."""
        # Test invalid enum values
        with pytest.raises(Exception):
            with self.db_manager.get_session() as session:
                experiment = ExperimentORM(
                    id='exp_invalid_001',
                    name='Invalid Experiment',
                    dataset_id='dataset_001',
                    status='invalid_status',  # Invalid status
                    created_at=datetime.now()
                )
                session.add(experiment)
                session.commit()


class TestDatabasePerformance:
    """Test database performance characteristics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.db_manager = DatabaseManager(database_url="sqlite:///:memory:")
        self.db_manager.create_tables()
    
    def test_bulk_insert_performance(self):
        """Test bulk insert performance."""
        import time
        
        # Create many experiments
        experiments = []
        for i in range(1000):
            experiment = ExperimentORM(
                id=f'exp_bulk_{i:04d}',
                name=f'Bulk Experiment {i}',
                dataset_id='dataset_001',
                status=ExperimentStatus.CREATED.value,
                created_at=datetime.now()
            )
            experiments.append(experiment)
        
        # Measure bulk insert time
        start_time = time.time()
        with self.db_manager.get_session() as session:
            session.add_all(experiments)
            session.commit()
        end_time = time.time()
        
        insert_time = end_time - start_time
        assert insert_time < 5.0  # Should complete within 5 seconds
        
        # Verify all experiments were inserted
        with self.db_manager.get_session() as session:
            count = session.query(ExperimentORM).count()
            assert count == 1000
    
    def test_query_performance(self):
        """Test query performance."""
        import time
        
        # Create test data
        with self.db_manager.get_session() as session:
            for i in range(100):
                experiment = ExperimentORM(
                    id=f'exp_query_{i:03d}',
                    name=f'Query Test {i}',
                    dataset_id='dataset_001',
                    status=ExperimentStatus.COMPLETED.value if i % 2 == 0 else ExperimentStatus.RUNNING.value,
                    created_at=datetime.now() - timedelta(days=i),
                    config={'trial': i}
                )
                session.add(experiment)
            session.commit()
        
        # Test various queries
        start_time = time.time()
        with self.db_manager.get_session() as session:
            # Simple filter
            completed = session.query(ExperimentORM).filter_by(
                status=ExperimentStatus.COMPLETED.value
            ).all()
            
            # Date range query
            recent = session.query(ExperimentORM).filter(
                ExperimentORM.created_at >= datetime.now() - timedelta(days=30)
            ).all()
            
            # JSON query (if supported)
            # specific_trials = session.query(ExperimentORM).filter(
            #     ExperimentORM.config['trial'].astext.cast(Integer) > 50
            # ).all()
        
        end_time = time.time()
        query_time = end_time - start_time
        
        assert query_time < 1.0  # Should be fast
        assert len(completed) == 50  # Half should be completed
        assert len(recent) <= 31  # Should be recent experiments


if __name__ == "__main__":
    pytest.main([__file__])