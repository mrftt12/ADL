"""
Comprehensive unit tests for checkpoint manager.

Tests cover checkpoint creation, loading, cleanup, and recovery scenarios
with various model types and training configurations.
"""

import pytest
import os
import tempfile
import shutil
import json
import pickle
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from automl_framework.services.checkpoint_manager import (
    CheckpointManager, CheckpointMetadata, CheckpointConfig
)
from automl_framework.models.data_models import (
    Architecture, TrainingConfig, PerformanceMetrics
)
from tests.test_utils import (
    MockArchitectureGenerator, TestDataManager,
    mock_simple_mlp, test_data_manager
)


class TestCheckpointMetadata:
    """Test CheckpointMetadata class."""
    
    def test_metadata_creation(self):
        """Test checkpoint metadata creation."""
        metadata = CheckpointMetadata(
            checkpoint_id="ckpt_001",
            experiment_id="exp_123",
            epoch=10,
            timestamp=datetime.now(),
            model_architecture_id="arch_456",
            training_config_hash="config_hash_789",
            metrics={'accuracy': 0.95, 'loss': 0.05},
            file_size_mb=150.5,
            checkpoint_path="/path/to/checkpoint.pth"
        )
        
        assert metadata.checkpoint_id == "ckpt_001"
        assert metadata.experiment_id == "exp_123"
        assert metadata.epoch == 10
        assert metadata.metrics['accuracy'] == 0.95
        assert metadata.file_size_mb == 150.5
    
    def test_metadata_validation(self):
        """Test metadata validation."""
        # Valid metadata
        valid_metadata = CheckpointMetadata(
            checkpoint_id="ckpt_001",
            experiment_id="exp_123",
            epoch=10,
            timestamp=datetime.now(),
            model_architecture_id="arch_456",
            training_config_hash="config_hash_789",
            checkpoint_path="/path/to/checkpoint.pth"
        )
        
        valid_metadata.validate()  # Should not raise
        
        # Invalid metadata - negative epoch
        invalid_metadata = CheckpointMetadata(
            checkpoint_id="ckpt_001",
            experiment_id="exp_123",
            epoch=-1,  # Invalid
            timestamp=datetime.now(),
            model_architecture_id="arch_456",
            training_config_hash="config_hash_789",
            checkpoint_path="/path/to/checkpoint.pth"
        )
        
        with pytest.raises(ValueError, match="Epoch must be non-negative"):
            invalid_metadata.validate()
    
    def test_metadata_serialization(self):
        """Test metadata serialization and deserialization."""
        metadata = CheckpointMetadata(
            checkpoint_id="ckpt_001",
            experiment_id="exp_123",
            epoch=10,
            timestamp=datetime.now(),
            model_architecture_id="arch_456",
            training_config_hash="config_hash_789",
            metrics={'accuracy': 0.95},
            checkpoint_path="/path/to/checkpoint.pth"
        )
        
        # Serialize to dict
        metadata_dict = metadata.to_dict()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict['checkpoint_id'] == "ckpt_001"
        assert metadata_dict['epoch'] == 10
        
        # Deserialize from dict
        restored_metadata = CheckpointMetadata.from_dict(metadata_dict)
        assert restored_metadata.checkpoint_id == metadata.checkpoint_id
        assert restored_metadata.epoch == metadata.epoch
        assert restored_metadata.metrics == metadata.metrics


class TestCheckpointConfig:
    """Test CheckpointConfig class."""
    
    def test_config_creation(self):
        """Test checkpoint configuration creation."""
        config = CheckpointConfig(
            save_frequency=5,
            max_checkpoints=10,
            save_best_only=True,
            monitor_metric='val_accuracy',
            save_optimizer_state=True,
            compression_enabled=True,
            cleanup_old_checkpoints=True
        )
        
        assert config.save_frequency == 5
        assert config.max_checkpoints == 10
        assert config.save_best_only is True
        assert config.monitor_metric == 'val_accuracy'
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        valid_config = CheckpointConfig(
            save_frequency=5,
            max_checkpoints=10
        )
        valid_config.validate()  # Should not raise
        
        # Invalid config - negative frequency
        invalid_config = CheckpointConfig(
            save_frequency=-1,  # Invalid
            max_checkpoints=10
        )
        
        with pytest.raises(ValueError, match="Save frequency must be positive"):
            invalid_config.validate()
        
        # Invalid config - zero max checkpoints
        invalid_config = CheckpointConfig(
            save_frequency=5,
            max_checkpoints=0  # Invalid
        )
        
        with pytest.raises(ValueError, match="Max checkpoints must be positive"):
            invalid_config.validate()


class TestCheckpointManager:
    """Test CheckpointManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = CheckpointConfig(
            save_frequency=2,
            max_checkpoints=5,
            save_best_only=False
        )
        self.manager = CheckpointManager(
            checkpoint_dir=self.temp_dir,
            config=self.config
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_manager_initialization(self):
        """Test checkpoint manager initialization."""
        assert self.manager.checkpoint_dir == self.temp_dir
        assert self.manager.config == self.config
        assert os.path.exists(self.temp_dir)
        assert len(self.manager.checkpoints) == 0
    
    def test_should_save_checkpoint(self):
        """Test checkpoint saving decision logic."""
        # Should save at frequency intervals
        assert self.manager.should_save_checkpoint(2) is True
        assert self.manager.should_save_checkpoint(4) is True
        assert self.manager.should_save_checkpoint(1) is False
        assert self.manager.should_save_checkpoint(3) is False
        
        # Always save at epoch 0
        assert self.manager.should_save_checkpoint(0) is True
    
    def test_should_save_checkpoint_best_only(self):
        """Test checkpoint saving with best_only mode."""
        config = CheckpointConfig(
            save_frequency=1,
            save_best_only=True,
            monitor_metric='val_accuracy'
        )
        manager = CheckpointManager(self.temp_dir, config)
        
        # First checkpoint should always save
        assert manager.should_save_checkpoint(1, {'val_accuracy': 0.8}) is True
        
        # Better metric should save
        assert manager.should_save_checkpoint(2, {'val_accuracy': 0.85}) is True
        
        # Worse metric should not save
        assert manager.should_save_checkpoint(3, {'val_accuracy': 0.82}) is False
    
    def test_generate_checkpoint_id(self):
        """Test checkpoint ID generation."""
        checkpoint_id = self.manager._generate_checkpoint_id("exp_123", 10)
        
        assert isinstance(checkpoint_id, str)
        assert "exp_123" in checkpoint_id
        assert "epoch_10" in checkpoint_id
        assert len(checkpoint_id) > 20  # Should include timestamp
    
    def test_create_checkpoint_path(self):
        """Test checkpoint path creation."""
        checkpoint_id = "ckpt_exp_123_epoch_10_20231201_120000"
        checkpoint_path = self.manager._create_checkpoint_path(checkpoint_id)
        
        expected_path = os.path.join(self.temp_dir, f"{checkpoint_id}.pth")
        assert checkpoint_path == expected_path
    
    def test_save_checkpoint(self, mock_simple_mlp):
        """Test saving checkpoint."""
        # Mock model and optimizer
        mock_model = Mock()
        mock_model.state_dict.return_value = {'layer1.weight': [1, 2, 3]}
        
        mock_optimizer = Mock()
        mock_optimizer.state_dict.return_value = {'lr': 0.001}
        
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer='adam',
            epochs=10
        )
        
        metrics = {'accuracy': 0.95, 'loss': 0.05}
        
        checkpoint_id = self.manager.save_checkpoint(
            experiment_id="exp_123",
            epoch=10,
            model=mock_model,
            optimizer=mock_optimizer,
            architecture=mock_simple_mlp,
            training_config=training_config,
            metrics=metrics,
            training_history={'loss': [0.8, 0.6, 0.4]}
        )
        
        assert isinstance(checkpoint_id, str)
        assert len(self.manager.checkpoints) == 1
        
        # Verify checkpoint file exists
        checkpoint_path = self.manager._create_checkpoint_path(checkpoint_id)
        assert os.path.exists(checkpoint_path)
        
        # Verify metadata
        metadata = self.manager.checkpoints[0]
        assert metadata.checkpoint_id == checkpoint_id
        assert metadata.experiment_id == "exp_123"
        assert metadata.epoch == 10
        assert metadata.metrics == metrics
    
    def test_load_checkpoint(self, mock_simple_mlp):
        """Test loading checkpoint."""
        # First save a checkpoint
        mock_model = Mock()
        mock_model.state_dict.return_value = {'layer1.weight': [1, 2, 3]}
        
        mock_optimizer = Mock()
        mock_optimizer.state_dict.return_value = {'lr': 0.001}
        
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer='adam',
            epochs=10
        )
        
        checkpoint_id = self.manager.save_checkpoint(
            experiment_id="exp_123",
            epoch=10,
            model=mock_model,
            optimizer=mock_optimizer,
            architecture=mock_simple_mlp,
            training_config=training_config,
            metrics={'accuracy': 0.95},
            training_history={'loss': [0.8, 0.6]}
        )
        
        # Now load the checkpoint
        loaded_data = self.manager.load_checkpoint(checkpoint_id)
        
        assert isinstance(loaded_data, dict)
        assert 'model_state_dict' in loaded_data
        assert 'optimizer_state_dict' in loaded_data
        assert 'architecture' in loaded_data
        assert 'training_config' in loaded_data
        assert 'epoch' in loaded_data
        assert 'metrics' in loaded_data
        assert 'training_history' in loaded_data
        
        assert loaded_data['epoch'] == 10
        assert loaded_data['metrics']['accuracy'] == 0.95
    
    def test_load_nonexistent_checkpoint(self):
        """Test loading non-existent checkpoint."""
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            self.manager.load_checkpoint("nonexistent_checkpoint")
    
    def test_list_checkpoints(self, mock_simple_mlp):
        """Test listing checkpoints."""
        # Save multiple checkpoints
        mock_model = Mock()
        mock_model.state_dict.return_value = {'layer1.weight': [1, 2, 3]}
        
        mock_optimizer = Mock()
        mock_optimizer.state_dict.return_value = {'lr': 0.001}
        
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer='adam',
            epochs=10
        )
        
        checkpoint_ids = []
        for epoch in [2, 4, 6]:
            checkpoint_id = self.manager.save_checkpoint(
                experiment_id="exp_123",
                epoch=epoch,
                model=mock_model,
                optimizer=mock_optimizer,
                architecture=mock_simple_mlp,
                training_config=training_config,
                metrics={'accuracy': 0.8 + epoch * 0.02}
            )
            checkpoint_ids.append(checkpoint_id)
        
        # List all checkpoints
        all_checkpoints = self.manager.list_checkpoints()
        assert len(all_checkpoints) == 3
        
        # List checkpoints for specific experiment
        exp_checkpoints = self.manager.list_checkpoints(experiment_id="exp_123")
        assert len(exp_checkpoints) == 3
        
        # List checkpoints for non-existent experiment
        no_checkpoints = self.manager.list_checkpoints(experiment_id="exp_999")
        assert len(no_checkpoints) == 0
    
    def test_get_best_checkpoint(self, mock_simple_mlp):
        """Test getting best checkpoint."""
        mock_model = Mock()
        mock_model.state_dict.return_value = {'layer1.weight': [1, 2, 3]}
        
        mock_optimizer = Mock()
        mock_optimizer.state_dict.return_value = {'lr': 0.001}
        
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer='adam',
            epochs=10
        )
        
        # Save checkpoints with different accuracies
        checkpoint_ids = []
        accuracies = [0.85, 0.92, 0.88, 0.95, 0.90]  # Best is 0.95 at index 3
        
        for i, accuracy in enumerate(accuracies):
            checkpoint_id = self.manager.save_checkpoint(
                experiment_id="exp_123",
                epoch=(i + 1) * 2,
                model=mock_model,
                optimizer=mock_optimizer,
                architecture=mock_simple_mlp,
                training_config=training_config,
                metrics={'accuracy': accuracy, 'loss': 1.0 - accuracy}
            )
            checkpoint_ids.append(checkpoint_id)
        
        # Get best checkpoint by accuracy (maximize)
        best_checkpoint = self.manager.get_best_checkpoint(
            experiment_id="exp_123",
            metric='accuracy',
            mode='max'
        )
        
        assert best_checkpoint is not None
        assert best_checkpoint.metrics['accuracy'] == 0.95
        assert best_checkpoint.epoch == 8  # (3 + 1) * 2
        
        # Get best checkpoint by loss (minimize)
        best_checkpoint = self.manager.get_best_checkpoint(
            experiment_id="exp_123",
            metric='loss',
            mode='min'
        )
        
        assert best_checkpoint is not None
        assert best_checkpoint.metrics['loss'] == 0.05  # 1.0 - 0.95
    
    def test_cleanup_old_checkpoints(self, mock_simple_mlp):
        """Test cleanup of old checkpoints."""
        # Set max_checkpoints to 3
        config = CheckpointConfig(
            save_frequency=1,
            max_checkpoints=3,
            cleanup_old_checkpoints=True
        )
        manager = CheckpointManager(self.temp_dir, config)
        
        mock_model = Mock()
        mock_model.state_dict.return_value = {'layer1.weight': [1, 2, 3]}
        
        mock_optimizer = Mock()
        mock_optimizer.state_dict.return_value = {'lr': 0.001}
        
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer='adam',
            epochs=10
        )
        
        # Save 5 checkpoints (more than max_checkpoints)
        checkpoint_ids = []
        for epoch in range(1, 6):
            checkpoint_id = manager.save_checkpoint(
                experiment_id="exp_123",
                epoch=epoch,
                model=mock_model,
                optimizer=mock_optimizer,
                architecture=mock_simple_mlp,
                training_config=training_config,
                metrics={'accuracy': 0.8 + epoch * 0.02}
            )
            checkpoint_ids.append(checkpoint_id)
        
        # Should only keep the last 3 checkpoints
        assert len(manager.checkpoints) == 3
        
        # Verify that the oldest checkpoints were removed
        remaining_epochs = [ckpt.epoch for ckpt in manager.checkpoints]
        assert remaining_epochs == [3, 4, 5]
        
        # Verify that old checkpoint files were deleted
        for i in range(2):  # First 2 checkpoints should be deleted
            checkpoint_path = manager._create_checkpoint_path(checkpoint_ids[i])
            assert not os.path.exists(checkpoint_path)
    
    def test_delete_checkpoint(self, mock_simple_mlp):
        """Test deleting specific checkpoint."""
        # Save a checkpoint
        mock_model = Mock()
        mock_model.state_dict.return_value = {'layer1.weight': [1, 2, 3]}
        
        mock_optimizer = Mock()
        mock_optimizer.state_dict.return_value = {'lr': 0.001}
        
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer='adam',
            epochs=10
        )
        
        checkpoint_id = self.manager.save_checkpoint(
            experiment_id="exp_123",
            epoch=10,
            model=mock_model,
            optimizer=mock_optimizer,
            architecture=mock_simple_mlp,
            training_config=training_config,
            metrics={'accuracy': 0.95}
        )
        
        # Verify checkpoint exists
        assert len(self.manager.checkpoints) == 1
        checkpoint_path = self.manager._create_checkpoint_path(checkpoint_id)
        assert os.path.exists(checkpoint_path)
        
        # Delete checkpoint
        self.manager.delete_checkpoint(checkpoint_id)
        
        # Verify checkpoint was removed
        assert len(self.manager.checkpoints) == 0
        assert not os.path.exists(checkpoint_path)
    
    def test_delete_nonexistent_checkpoint(self):
        """Test deleting non-existent checkpoint."""
        with pytest.raises(ValueError, match="Checkpoint not found"):
            self.manager.delete_checkpoint("nonexistent_checkpoint")
    
    def test_get_checkpoint_info(self, mock_simple_mlp):
        """Test getting checkpoint information."""
        # Save a checkpoint
        mock_model = Mock()
        mock_model.state_dict.return_value = {'layer1.weight': [1, 2, 3]}
        
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer='adam',
            epochs=10
        )
        
        checkpoint_id = self.manager.save_checkpoint(
            experiment_id="exp_123",
            epoch=10,
            model=mock_model,
            optimizer=None,
            architecture=mock_simple_mlp,
            training_config=training_config,
            metrics={'accuracy': 0.95, 'loss': 0.05}
        )
        
        # Get checkpoint info
        info = self.manager.get_checkpoint_info(checkpoint_id)
        
        assert isinstance(info, dict)
        assert info['checkpoint_id'] == checkpoint_id
        assert info['experiment_id'] == "exp_123"
        assert info['epoch'] == 10
        assert info['metrics']['accuracy'] == 0.95
        assert 'file_size_mb' in info
        assert 'created_at' in info
    
    def test_checkpoint_compression(self, mock_simple_mlp):
        """Test checkpoint compression."""
        config = CheckpointConfig(
            save_frequency=1,
            compression_enabled=True
        )
        manager = CheckpointManager(self.temp_dir, config)
        
        # Create larger mock model state
        large_state = {f'layer_{i}.weight': list(range(1000)) for i in range(10)}
        
        mock_model = Mock()
        mock_model.state_dict.return_value = large_state
        
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer='adam',
            epochs=10
        )
        
        checkpoint_id = manager.save_checkpoint(
            experiment_id="exp_123",
            epoch=10,
            model=mock_model,
            optimizer=None,
            architecture=mock_simple_mlp,
            training_config=training_config,
            metrics={'accuracy': 0.95}
        )
        
        # Verify checkpoint was saved and can be loaded
        loaded_data = manager.load_checkpoint(checkpoint_id)
        assert 'model_state_dict' in loaded_data
        assert loaded_data['epoch'] == 10
    
    def test_checkpoint_metadata_persistence(self, mock_simple_mlp):
        """Test checkpoint metadata persistence."""
        # Save a checkpoint
        mock_model = Mock()
        mock_model.state_dict.return_value = {'layer1.weight': [1, 2, 3]}
        
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer='adam',
            epochs=10
        )
        
        checkpoint_id = self.manager.save_checkpoint(
            experiment_id="exp_123",
            epoch=10,
            model=mock_model,
            optimizer=None,
            architecture=mock_simple_mlp,
            training_config=training_config,
            metrics={'accuracy': 0.95}
        )
        
        # Create new manager instance (simulating restart)
        new_manager = CheckpointManager(self.temp_dir, self.config)
        
        # Verify metadata was loaded
        assert len(new_manager.checkpoints) == 1
        assert new_manager.checkpoints[0].checkpoint_id == checkpoint_id
        assert new_manager.checkpoints[0].epoch == 10
    
    def test_concurrent_checkpoint_access(self, mock_simple_mlp):
        """Test concurrent checkpoint operations."""
        import threading
        import time
        
        mock_model = Mock()
        mock_model.state_dict.return_value = {'layer1.weight': [1, 2, 3]}
        
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer='adam',
            epochs=10
        )
        
        checkpoint_ids = []
        errors = []
        
        def save_checkpoint(epoch):
            try:
                checkpoint_id = self.manager.save_checkpoint(
                    experiment_id="exp_123",
                    epoch=epoch,
                    model=mock_model,
                    optimizer=None,
                    architecture=mock_simple_mlp,
                    training_config=training_config,
                    metrics={'accuracy': 0.8 + epoch * 0.01}
                )
                checkpoint_ids.append(checkpoint_id)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads to save checkpoints concurrently
        threads = []
        for epoch in range(1, 6):
            thread = threading.Thread(target=save_checkpoint, args=(epoch,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0
        assert len(checkpoint_ids) == 5
        assert len(self.manager.checkpoints) == 5


if __name__ == "__main__":
    pytest.main([__file__])