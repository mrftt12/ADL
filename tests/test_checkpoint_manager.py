"""
Unit tests for checkpoint manager
"""

import pytest
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import torch
import torch.nn as nn

from automl_framework.services.checkpoint_manager import (
    CheckpointManager, CheckpointManagerRegistry, CheckpointConfig,
    CheckpointMetadata, get_checkpoint_manager, cleanup_job_checkpoints
)
from automl_framework.core.interfaces import Architecture, TrainingConfig, Layer, Connection
from automl_framework.core.exceptions import CheckpointError


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.linear(x)


class TestCheckpointConfig:
    """Test checkpoint configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = CheckpointConfig()
        
        assert config.save_frequency == 1
        assert config.max_checkpoints == 10
        assert config.save_best_only is False
        assert config.monitor_metric == "val_loss"
        assert config.mode == "min"
        assert config.save_optimizer_state is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = CheckpointConfig(
            save_frequency=5,
            max_checkpoints=3,
            save_best_only=True,
            monitor_metric="val_accuracy",
            mode="max"
        )
        
        assert config.save_frequency == 5
        assert config.max_checkpoints == 3
        assert config.save_best_only is True
        assert config.monitor_metric == "val_accuracy"
        assert config.mode == "max"


class TestCheckpointManager:
    """Test checkpoint manager functionality"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.job_id = "test_job_123"
        
        # Mock the config to use our temp directory
        with patch('automl_framework.services.checkpoint_manager.get_config') as mock_config:
            mock_config.return_value.checkpoint_storage_path = self.temp_dir
            self.manager = CheckpointManager(self.job_id)
    
    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test checkpoint manager initialization"""
        assert self.manager.job_id == self.job_id
        assert self.manager.checkpoint_dir.exists()
        assert len(self.manager.checkpoints) == 0
    
    def test_detect_framework_pytorch(self):
        """Test framework detection for PyTorch"""
        model = SimpleModel()
        framework = self.manager._detect_framework(model)
        assert framework == "pytorch"
    
    def test_detect_framework_unknown(self):
        """Test framework detection for unknown model"""
        model = object()  # Unknown model type
        
        with pytest.raises(CheckpointError):
            self.manager._detect_framework(model)
    
    def test_generate_checkpoint_id(self):
        """Test checkpoint ID generation"""
        checkpoint_id = self.manager._generate_checkpoint_id(5, 100)
        
        assert "epoch_0005" in checkpoint_id
        assert "step_000100" in checkpoint_id
        assert len(checkpoint_id.split('_')) == 4  # epoch, step, timestamp
    
    def test_hash_architecture(self):
        """Test architecture hashing"""
        architecture = Architecture(
            id="test_arch",
            layers=[Layer("conv2d", {"filters": 32}, (28, 28, 1), (26, 26, 32))],
            connections=[Connection(0, 1, "sequential")],
            input_shape=(28, 28, 1),
            output_shape=(10,),
            parameter_count=1000,
            flops=50000
        )
        
        hash1 = self.manager._hash_architecture(architecture)
        hash2 = self.manager._hash_architecture(architecture)
        
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length
    
    def test_should_save_checkpoint_frequency(self):
        """Test checkpoint save frequency logic"""
        config = CheckpointConfig(save_frequency=3)
        self.manager.config = config
        
        assert not self.manager._should_save_checkpoint(1)
        assert not self.manager._should_save_checkpoint(2)
        assert self.manager._should_save_checkpoint(3)
        assert not self.manager._should_save_checkpoint(4)
        assert self.manager._should_save_checkpoint(6)
    
    def test_should_save_checkpoint_best_only(self):
        """Test save best only logic"""
        config = CheckpointConfig(save_best_only=True, monitor_metric="val_loss", mode="min")
        self.manager.config = config
        
        # First checkpoint should always save
        assert self.manager._should_save_checkpoint(1, {"val_loss": 1.0})
        
        # Set best metric value
        self.manager.best_metric_value = 1.0
        
        # Better metric should save
        assert self.manager._should_save_checkpoint(2, {"val_loss": 0.8})
        
        # Worse metric should not save
        assert not self.manager._should_save_checkpoint(3, {"val_loss": 1.2})
    
    def test_is_best_checkpoint(self):
        """Test best checkpoint detection"""
        config = CheckpointConfig(monitor_metric="val_accuracy", mode="max")
        self.manager.config = config
        
        # First checkpoint is always best
        assert self.manager._is_best_checkpoint({"val_accuracy": 0.8})
        
        # Set current best
        self.manager.best_metric_value = 0.8
        
        # Better accuracy should be best
        assert self.manager._is_best_checkpoint({"val_accuracy": 0.9})
        
        # Worse accuracy should not be best
        assert not self.manager._is_best_checkpoint({"val_accuracy": 0.7})
    
    def test_save_pytorch_checkpoint(self):
        """Test saving PyTorch checkpoint"""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint_id = self.manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            step=100,
            metrics={"val_loss": 0.5, "val_accuracy": 0.8},
            force_save=True
        )
        
        assert checkpoint_id is not None
        assert checkpoint_id in self.manager.checkpoints
        
        metadata = self.manager.checkpoints[checkpoint_id]
        assert metadata.epoch == 5
        assert metadata.step == 100
        assert metadata.framework == "pytorch"
        assert metadata.metrics["val_loss"] == 0.5
        assert Path(metadata.file_path).exists()
    
    def test_load_pytorch_checkpoint(self):
        """Test loading PyTorch checkpoint"""
        # First save a checkpoint
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint_id = self.manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            step=100,
            metrics={"val_loss": 0.5},
            force_save=True
        )
        
        # Create new model and optimizer instances
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        # Load the checkpoint
        loaded_model, loaded_optimizer, checkpoint_info = self.manager.load_checkpoint(
            checkpoint_id=checkpoint_id,
            model=new_model,
            optimizer=new_optimizer
        )
        
        assert loaded_model is new_model
        assert loaded_optimizer is new_optimizer
        assert checkpoint_info['metadata']['epoch'] == 5
        assert checkpoint_info['checkpoint_data']['metrics']['val_loss'] == 0.5
    
    def test_load_best_checkpoint(self):
        """Test loading best checkpoint"""
        model = SimpleModel()
        
        # Save multiple checkpoints
        self.manager.save_checkpoint(model, epoch=1, metrics={"val_loss": 1.0}, force_save=True)
        self.manager.save_checkpoint(model, epoch=2, metrics={"val_loss": 0.5}, force_save=True)  # Best
        self.manager.save_checkpoint(model, epoch=3, metrics={"val_loss": 0.8}, force_save=True)
        
        # Load best checkpoint
        new_model = SimpleModel()
        loaded_model, _, checkpoint_info = self.manager.load_checkpoint(
            load_best=True,
            model=new_model
        )
        
        assert loaded_model is new_model
        assert checkpoint_info['metadata']['epoch'] == 2
        assert checkpoint_info['metadata']['is_best'] is True
    
    def test_list_checkpoints(self):
        """Test listing checkpoints"""
        model = SimpleModel()
        
        # Save some checkpoints
        self.manager.save_checkpoint(model, epoch=1, metrics={"val_loss": 1.0}, force_save=True)
        self.manager.save_checkpoint(model, epoch=2, metrics={"val_loss": 0.5}, force_save=True)
        
        # List with metadata
        checkpoints_full = self.manager.list_checkpoints(include_metadata=True)
        assert len(checkpoints_full) == 2
        assert all('checkpoint_id' in cp for cp in checkpoints_full)
        assert all('metrics' in cp for cp in checkpoints_full)
        
        # List without metadata
        checkpoints_simple = self.manager.list_checkpoints(include_metadata=False)
        assert len(checkpoints_simple) == 2
        assert all('checkpoint_id' in cp for cp in checkpoints_simple)
        assert all('architecture' not in cp for cp in checkpoints_simple)
    
    def test_delete_checkpoint(self):
        """Test deleting a checkpoint"""
        model = SimpleModel()
        
        checkpoint_id = self.manager.save_checkpoint(
            model, epoch=1, metrics={"val_loss": 1.0}, force_save=True
        )
        
        assert checkpoint_id in self.manager.checkpoints
        
        # Delete the checkpoint
        result = self.manager.delete_checkpoint(checkpoint_id)
        
        assert result is True
        assert checkpoint_id not in self.manager.checkpoints
        
        # Try to delete non-existent checkpoint
        result = self.manager.delete_checkpoint("nonexistent")
        assert result is False
    
    def test_cleanup_old_checkpoints(self):
        """Test cleanup of old checkpoints"""
        config = CheckpointConfig(max_checkpoints=3)
        self.manager.config = config
        
        model = SimpleModel()
        
        # Save more checkpoints than the limit
        checkpoint_ids = []
        for i in range(5):
            checkpoint_id = self.manager.save_checkpoint(
                model, epoch=i+1, metrics={"val_loss": 1.0 - i*0.1}, force_save=True
            )
            checkpoint_ids.append(checkpoint_id)
            time.sleep(0.01)  # Ensure different timestamps
        
        # Should only keep the most recent checkpoints
        assert len(self.manager.checkpoints) == 3
        
        # The last 3 checkpoints should remain
        remaining_ids = set(self.manager.checkpoints.keys())
        expected_ids = set(checkpoint_ids[-3:])
        assert remaining_ids == expected_ids
    
    def test_cleanup_all_checkpoints(self):
        """Test cleanup of all checkpoints"""
        model = SimpleModel()
        
        # Save some checkpoints
        self.manager.save_checkpoint(model, epoch=1, force_save=True)
        self.manager.save_checkpoint(model, epoch=2, force_save=True)
        
        assert len(self.manager.checkpoints) == 2
        
        # Cleanup all
        deleted_count = self.manager.cleanup_all_checkpoints()
        
        assert deleted_count == 2
        assert len(self.manager.checkpoints) == 0
    
    def test_get_checkpoint_info(self):
        """Test getting checkpoint information"""
        model = SimpleModel()
        
        checkpoint_id = self.manager.save_checkpoint(
            model, epoch=5, metrics={"val_loss": 0.5}, force_save=True
        )
        
        info = self.manager.get_checkpoint_info(checkpoint_id)
        
        assert info is not None
        assert info['checkpoint_id'] == checkpoint_id
        assert info['epoch'] == 5
        assert info['file_exists'] is True
        assert info['file_size_mb'] > 0
        assert 'age_hours' in info
    
    def test_save_and_load_metadata(self):
        """Test saving and loading checkpoint metadata"""
        model = SimpleModel()
        
        # Save a checkpoint
        checkpoint_id = self.manager.save_checkpoint(
            model, epoch=5, metrics={"val_loss": 0.5}, force_save=True
        )
        
        # Create new manager instance (simulates restart)
        with patch('automl_framework.services.checkpoint_manager.get_config') as mock_config:
            mock_config.return_value.checkpoint_storage_path = self.temp_dir
            new_manager = CheckpointManager(self.job_id)
        
        # Should load existing checkpoints
        assert len(new_manager.checkpoints) == 1
        assert checkpoint_id in new_manager.checkpoints
        assert new_manager.checkpoints[checkpoint_id].epoch == 5


class TestCheckpointManagerRegistry:
    """Test checkpoint manager registry"""
    
    def setup_method(self):
        self.registry = CheckpointManagerRegistry()
    
    def test_get_manager(self):
        """Test getting checkpoint manager"""
        with patch('automl_framework.services.checkpoint_manager.CheckpointManager') as mock_manager:
            mock_instance = Mock()
            mock_manager.return_value = mock_instance
            
            manager = self.registry.get_manager("job1")
            
            assert manager is mock_instance
            assert "job1" in self.registry.managers
            mock_manager.assert_called_once_with("job1", None)
    
    def test_get_existing_manager(self):
        """Test getting existing manager"""
        with patch('automl_framework.services.checkpoint_manager.CheckpointManager') as mock_manager:
            mock_instance = Mock()
            mock_manager.return_value = mock_instance
            
            # Get manager twice
            manager1 = self.registry.get_manager("job1")
            manager2 = self.registry.get_manager("job1")
            
            assert manager1 is manager2
            assert len(self.registry.managers) == 1
            mock_manager.assert_called_once()  # Should only be called once
    
    def test_remove_manager(self):
        """Test removing manager"""
        with patch('automl_framework.services.checkpoint_manager.CheckpointManager') as mock_manager:
            mock_instance = Mock()
            mock_manager.return_value = mock_instance
            
            # Create manager
            self.registry.get_manager("job1")
            assert "job1" in self.registry.managers
            
            # Remove manager
            result = self.registry.remove_manager("job1")
            
            assert result is True
            assert "job1" not in self.registry.managers
    
    def test_remove_nonexistent_manager(self):
        """Test removing non-existent manager"""
        result = self.registry.remove_manager("nonexistent")
        assert result is False
    
    def test_remove_manager_with_cleanup(self):
        """Test removing manager with checkpoint cleanup"""
        with patch('automl_framework.services.checkpoint_manager.CheckpointManager') as mock_manager:
            mock_instance = Mock()
            mock_manager.return_value = mock_instance
            
            # Create manager
            self.registry.get_manager("job1")
            
            # Remove with cleanup
            result = self.registry.remove_manager("job1", cleanup_checkpoints=True)
            
            assert result is True
            mock_instance.cleanup_all_checkpoints.assert_called_once()
    
    def test_list_jobs(self):
        """Test listing jobs"""
        with patch('automl_framework.services.checkpoint_manager.CheckpointManager'):
            self.registry.get_manager("job1")
            self.registry.get_manager("job2")
            
            jobs = self.registry.list_jobs()
            
            assert set(jobs) == {"job1", "job2"}
    
    def test_get_all_checkpoints(self):
        """Test getting all checkpoints"""
        with patch('automl_framework.services.checkpoint_manager.CheckpointManager') as mock_manager:
            mock_instance1 = Mock()
            mock_instance1.list_checkpoints.return_value = [{"checkpoint_id": "cp1"}]
            mock_instance2 = Mock()
            mock_instance2.list_checkpoints.return_value = [{"checkpoint_id": "cp2"}]
            
            mock_manager.side_effect = [mock_instance1, mock_instance2]
            
            self.registry.get_manager("job1")
            self.registry.get_manager("job2")
            
            all_checkpoints = self.registry.get_all_checkpoints()
            
            assert "job1" in all_checkpoints
            assert "job2" in all_checkpoints
            assert all_checkpoints["job1"] == [{"checkpoint_id": "cp1"}]
            assert all_checkpoints["job2"] == [{"checkpoint_id": "cp2"}]
    
    def test_cleanup_completed_jobs(self):
        """Test cleanup of completed jobs"""
        with patch('automl_framework.services.checkpoint_manager.CheckpointManager'):
            # Create managers for multiple jobs
            self.registry.get_manager("job1")
            self.registry.get_manager("job2")
            self.registry.get_manager("job3")
            
            # Cleanup with only job2 still active
            self.registry.cleanup_completed_jobs(["job2"])
            
            # Only job2 should remain
            assert self.registry.list_jobs() == ["job2"]


class TestGlobalFunctions:
    """Test global utility functions"""
    
    def test_get_checkpoint_manager(self):
        """Test global get_checkpoint_manager function"""
        with patch('automl_framework.services.checkpoint_manager.checkpoint_registry') as mock_registry:
            mock_manager = Mock()
            mock_registry.get_manager.return_value = mock_manager
            
            manager = get_checkpoint_manager("job1")
            
            assert manager is mock_manager
            mock_registry.get_manager.assert_called_once_with("job1", None)
    
    def test_cleanup_job_checkpoints(self):
        """Test global cleanup_job_checkpoints function"""
        with patch('automl_framework.services.checkpoint_manager.checkpoint_registry') as mock_registry:
            mock_registry.remove_manager.return_value = True
            
            result = cleanup_job_checkpoints("job1")
            
            assert result is True
            mock_registry.remove_manager.assert_called_once_with("job1", cleanup_checkpoints=True)


@pytest.fixture
def sample_architecture():
    """Sample architecture for testing"""
    return Architecture(
        id="test_arch",
        layers=[
            Layer("conv2d", {"filters": 32, "kernel_size": 3}, (28, 28, 1), (26, 26, 32)),
            Layer("dense", {"units": 10}, (832,), (10,))
        ],
        connections=[Connection(0, 1, "sequential")],
        input_shape=(28, 28, 1),
        output_shape=(10,),
        parameter_count=1000,
        flops=50000
    )


@pytest.fixture
def sample_training_config():
    """Sample training configuration for testing"""
    return TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        optimizer="adam",
        epochs=10,
        early_stopping_patience=5,
        regularization={"l2": 0.01}
    )


class TestIntegration:
    """Integration tests for checkpoint manager"""
    
    def test_full_checkpoint_workflow(self, sample_architecture, sample_training_config):
        """Test complete checkpoint workflow"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            with patch('automl_framework.services.checkpoint_manager.get_config') as mock_config:
                mock_config.return_value.checkpoint_storage_path = temp_dir
                
                # Create manager
                manager = CheckpointManager("integration_test")
                
                # Create model
                model = SimpleModel()
                optimizer = torch.optim.Adam(model.parameters())
                
                # Save multiple checkpoints
                checkpoint_ids = []
                for epoch in range(1, 4):
                    checkpoint_id = manager.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        step=epoch * 100,
                        metrics={"val_loss": 1.0 - epoch * 0.2, "val_accuracy": epoch * 0.3},
                        architecture=sample_architecture,
                        config=sample_training_config,
                        tags=[f"epoch_{epoch}"],
                        force_save=True
                    )
                    checkpoint_ids.append(checkpoint_id)
                
                # Verify checkpoints were saved
                assert len(manager.checkpoints) == 3
                assert manager.best_checkpoint is not None
                assert manager.best_checkpoint.epoch == 3  # Best val_loss
                
                # Load best checkpoint
                new_model = SimpleModel()
                new_optimizer = torch.optim.Adam(new_model.parameters())
                
                loaded_model, loaded_optimizer, info = manager.load_checkpoint(
                    load_best=True,
                    model=new_model,
                    optimizer=new_optimizer
                )
                
                assert loaded_model is new_model
                assert loaded_optimizer is new_optimizer
                assert info['metadata']['epoch'] == 3
                assert info['metadata']['is_best'] is True
                
                # Test checkpoint info
                checkpoint_info = manager.get_checkpoint_info(checkpoint_ids[0])
                assert checkpoint_info is not None
                assert checkpoint_info['epoch'] == 1
                assert checkpoint_info['file_exists'] is True
                
                # Cleanup
                deleted_count = manager.cleanup_all_checkpoints()
                assert deleted_count == 3
                assert len(manager.checkpoints) == 0
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__])