"""
Tests for SQLAlchemy ORM models.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from automl_framework.core.database import Base
from automl_framework.models.orm_models import (
    User, DatasetORM, ExperimentORM, TrainedModelORM, HyperparameterTrialORM
)
from automl_framework.models.data_models import DataType, ExperimentStatus, TaskType


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    session.close()


class TestUser:
    """Test cases for User ORM model."""
    
    def test_create_user(self, db_session):
        """Test creating a user."""
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User"
        )
        
        db_session.add(user)
        db_session.commit()
        
        assert user.id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True
        assert user.is_admin is False
        assert user.created_at is not None
    
    def test_unique_username_constraint(self, db_session):
        """Test that usernames must be unique."""
        user1 = User(username="testuser", email="test1@example.com", password_hash="hash1")
        user2 = User(username="testuser", email="test2@example.com", password_hash="hash2")
        
        db_session.add(user1)
        db_session.commit()
        
        db_session.add(user2)
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_unique_email_constraint(self, db_session):
        """Test that emails must be unique."""
        user1 = User(username="user1", email="test@example.com", password_hash="hash1")
        user2 = User(username="user2", email="test@example.com", password_hash="hash2")
        
        db_session.add(user1)
        db_session.commit()
        
        db_session.add(user2)
        with pytest.raises(IntegrityError):
            db_session.commit()


class TestDatasetORM:
    """Test cases for DatasetORM model."""
    
    def test_create_dataset(self, db_session):
        """Test creating a dataset."""
        # Create user first
        user = User(username="testuser", email="test@example.com", password_hash="hash")
        db_session.add(user)
        db_session.commit()
        
        # Create dataset
        dataset = DatasetORM(
            id="test-dataset-1",
            name="Test Dataset",
            file_path="/path/to/data.csv",
            data_type=DataType.TABULAR,
            size=1000,
            owner_id=user.id
        )
        
        db_session.add(dataset)
        db_session.commit()
        
        assert dataset.id == "test-dataset-1"
        assert dataset.name == "Test Dataset"
        assert dataset.data_type == DataType.TABULAR
        assert dataset.size == 1000
        assert dataset.owner_id == user.id
        assert dataset.created_at is not None
    
    def test_features_property(self, db_session):
        """Test features JSON property."""
        user = User(username="testuser", email="test@example.com", password_hash="hash")
        db_session.add(user)
        db_session.commit()
        
        dataset = DatasetORM(
            id="test-dataset-1",
            name="Test Dataset",
            file_path="/path/to/data.csv",
            data_type=DataType.TABULAR,
            size=1000,
            owner_id=user.id
        )
        
        # Test setting features
        features_data = [{"name": "feature1", "type": "numeric"}]
        dataset.features = features_data
        
        db_session.add(dataset)
        db_session.commit()
        
        # Test getting features
        retrieved_dataset = db_session.query(DatasetORM).filter_by(id="test-dataset-1").first()
        assert retrieved_dataset.features == features_data
    
    def test_metadata_property(self, db_session):
        """Test metadata JSON property."""
        user = User(username="testuser", email="test@example.com", password_hash="hash")
        db_session.add(user)
        db_session.commit()
        
        dataset = DatasetORM(
            id="test-dataset-1",
            name="Test Dataset",
            file_path="/path/to/data.csv",
            data_type=DataType.TABULAR,
            size=1000,
            owner_id=user.id
        )
        
        # Test setting metadata
        metadata = {"source": "test", "version": "1.0"}
        dataset.dataset_metadata = metadata
        
        db_session.add(dataset)
        db_session.commit()
        
        # Test getting metadata
        retrieved_dataset = db_session.query(DatasetORM).filter_by(id="test-dataset-1").first()
        assert retrieved_dataset.dataset_metadata == metadata


class TestExperimentORM:
    """Test cases for ExperimentORM model."""
    
    def test_create_experiment(self, db_session):
        """Test creating an experiment."""
        # Create user and dataset first
        user = User(username="testuser", email="test@example.com", password_hash="hash")
        db_session.add(user)
        db_session.commit()
        
        dataset = DatasetORM(
            id="test-dataset-1",
            name="Test Dataset",
            file_path="/path/to/data.csv",
            data_type=DataType.TABULAR,
            size=1000,
            owner_id=user.id
        )
        db_session.add(dataset)
        db_session.commit()
        
        # Create experiment
        experiment = ExperimentORM(
            id="test-exp-1",
            name="Test Experiment",
            dataset_id=dataset.id,
            owner_id=user.id,
            task_type=TaskType.CLASSIFICATION
        )
        
        db_session.add(experiment)
        db_session.commit()
        
        assert experiment.id == "test-exp-1"
        assert experiment.name == "Test Experiment"
        assert experiment.status == ExperimentStatus.CREATED
        assert experiment.task_type == TaskType.CLASSIFICATION
        assert experiment.dataset_id == dataset.id
        assert experiment.owner_id == user.id
        assert experiment.created_at is not None
    
    def test_config_property(self, db_session):
        """Test config JSON property."""
        user = User(username="testuser", email="test@example.com", password_hash="hash")
        db_session.add(user)
        db_session.commit()
        
        dataset = DatasetORM(
            id="test-dataset-1",
            name="Test Dataset",
            file_path="/path/to/data.csv",
            data_type=DataType.TABULAR,
            size=1000,
            owner_id=user.id
        )
        db_session.add(dataset)
        db_session.commit()
        
        experiment = ExperimentORM(
            id="test-exp-1",
            name="Test Experiment",
            dataset_id=dataset.id,
            owner_id=user.id
        )
        
        # Test setting config
        config = {"max_trials": 100, "timeout": 3600}
        experiment.config = config
        
        db_session.add(experiment)
        db_session.commit()
        
        # Test getting config
        retrieved_exp = db_session.query(ExperimentORM).filter_by(id="test-exp-1").first()
        assert retrieved_exp.config == config
    
    def test_duration_property(self, db_session):
        """Test duration property calculation."""
        user = User(username="testuser", email="test@example.com", password_hash="hash")
        db_session.add(user)
        db_session.commit()
        
        dataset = DatasetORM(
            id="test-dataset-1",
            name="Test Dataset",
            file_path="/path/to/data.csv",
            data_type=DataType.TABULAR,
            size=1000,
            owner_id=user.id
        )
        db_session.add(dataset)
        db_session.commit()
        
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=2)
        
        experiment = ExperimentORM(
            id="test-exp-1",
            name="Test Experiment",
            dataset_id=dataset.id,
            owner_id=user.id,
            started_at=start_time,
            completed_at=end_time
        )
        
        db_session.add(experiment)
        db_session.commit()
        
        assert experiment.duration == 7200.0  # 2 hours in seconds


class TestTrainedModelORM:
    """Test cases for TrainedModelORM model."""
    
    def test_create_trained_model(self, db_session):
        """Test creating a trained model."""
        # Create user, dataset, and experiment first
        user = User(username="testuser", email="test@example.com", password_hash="hash")
        db_session.add(user)
        db_session.commit()
        
        dataset = DatasetORM(
            id="test-dataset-1",
            name="Test Dataset",
            file_path="/path/to/data.csv",
            data_type=DataType.TABULAR,
            size=1000,
            owner_id=user.id
        )
        db_session.add(dataset)
        db_session.commit()
        
        experiment = ExperimentORM(
            id="test-exp-1",
            name="Test Experiment",
            dataset_id=dataset.id,
            owner_id=user.id
        )
        db_session.add(experiment)
        db_session.commit()
        
        # Create trained model
        model = TrainedModelORM(
            name="Test Model",
            architecture_id="arch-123",
            experiment_id=experiment.id,
            accuracy=0.95,
            loss=0.05,
            training_time=3600.0,
            parameter_count=1000000
        )
        
        db_session.add(model)
        db_session.commit()
        
        assert model.id is not None
        assert model.name == "Test Model"
        assert model.architecture_id == "arch-123"
        assert model.experiment_id == experiment.id
        assert model.accuracy == 0.95
        assert model.loss == 0.05
        assert model.training_time == 3600.0
        assert model.parameter_count == 1000000
    
    def test_hyperparameters_property(self, db_session):
        """Test hyperparameters JSON property."""
        user = User(username="testuser", email="test@example.com", password_hash="hash")
        db_session.add(user)
        db_session.commit()
        
        dataset = DatasetORM(
            id="test-dataset-1",
            name="Test Dataset",
            file_path="/path/to/data.csv",
            data_type=DataType.TABULAR,
            size=1000,
            owner_id=user.id
        )
        db_session.add(dataset)
        db_session.commit()
        
        experiment = ExperimentORM(
            id="test-exp-1",
            name="Test Experiment",
            dataset_id=dataset.id,
            owner_id=user.id
        )
        db_session.add(experiment)
        db_session.commit()
        
        model = TrainedModelORM(
            name="Test Model",
            architecture_id="arch-123",
            experiment_id=experiment.id
        )
        
        # Test setting hyperparameters
        hyperparams = {"learning_rate": 0.001, "batch_size": 32}
        model.hyperparameters = hyperparams
        
        db_session.add(model)
        db_session.commit()
        
        # Test getting hyperparameters
        retrieved_model = db_session.query(TrainedModelORM).filter_by(name="Test Model").first()
        assert retrieved_model.hyperparameters == hyperparams


class TestHyperparameterTrialORM:
    """Test cases for HyperparameterTrialORM model."""
    
    def test_create_hyperparameter_trial(self, db_session):
        """Test creating a hyperparameter trial."""
        # Create user, dataset, and experiment first
        user = User(username="testuser", email="test@example.com", password_hash="hash")
        db_session.add(user)
        db_session.commit()
        
        dataset = DatasetORM(
            id="test-dataset-1",
            name="Test Dataset",
            file_path="/path/to/data.csv",
            data_type=DataType.TABULAR,
            size=1000,
            owner_id=user.id
        )
        db_session.add(dataset)
        db_session.commit()
        
        experiment = ExperimentORM(
            id="test-exp-1",
            name="Test Experiment",
            dataset_id=dataset.id,
            owner_id=user.id
        )
        db_session.add(experiment)
        db_session.commit()
        
        # Create hyperparameter trial
        trial = HyperparameterTrialORM(
            trial_number=1,
            experiment_id=experiment.id,
            objective_value=0.95,
            duration=120.5
        )
        
        # Set hyperparameters
        hyperparams = {"learning_rate": 0.001, "batch_size": 32}
        trial.hyperparameters = hyperparams
        
        db_session.add(trial)
        db_session.commit()
        
        assert trial.id is not None
        assert trial.trial_number == 1
        assert trial.experiment_id == experiment.id
        assert trial.objective_value == 0.95
        assert trial.duration == 120.5
        assert trial.hyperparameters == hyperparams
        assert trial.status == "completed"
    
    def test_unique_trial_number_per_experiment(self, db_session):
        """Test that trial numbers must be unique per experiment."""
        user = User(username="testuser", email="test@example.com", password_hash="hash")
        db_session.add(user)
        db_session.commit()
        
        dataset = DatasetORM(
            id="test-dataset-1",
            name="Test Dataset",
            file_path="/path/to/data.csv",
            data_type=DataType.TABULAR,
            size=1000,
            owner_id=user.id
        )
        db_session.add(dataset)
        db_session.commit()
        
        experiment = ExperimentORM(
            id="test-exp-1",
            name="Test Experiment",
            dataset_id=dataset.id,
            owner_id=user.id
        )
        db_session.add(experiment)
        db_session.commit()
        
        # Create first trial
        trial1 = HyperparameterTrialORM(
            trial_number=1,
            experiment_id=experiment.id,
            objective_value=0.95
        )
        trial1.hyperparameters = {"lr": 0.001}
        
        db_session.add(trial1)
        db_session.commit()
        
        # Try to create second trial with same number
        trial2 = HyperparameterTrialORM(
            trial_number=1,
            experiment_id=experiment.id,
            objective_value=0.90
        )
        trial2.hyperparameters = {"lr": 0.01}
        
        db_session.add(trial2)
        with pytest.raises(IntegrityError):
            db_session.commit()