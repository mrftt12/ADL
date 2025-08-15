"""
Unit tests for training service
"""

import pytest
import time
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

import torch
import torch.nn as nn

from automl_framework.services.training_service import (
    GPUManager, DistributedTrainer, TrainingJobManager, ModelTrainingService,
    GPUInfo, TrainingProgress
)
from automl_framework.core.interfaces import (
    Architecture, TrainingConfig, ProcessedData, TrainingJob, Layer, Connection
)
from automl_framework.core.exceptions import TrainingError, ResourceError


class TestGPUManager:
    """Test GPU management functionality"""
    
    def setup_method(self):
        self.gpu_manager = GPUManager()
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.set_device')
    @patch('torch.cuda.memory_reserved')
    def test_get_available_gpus(self, mock_memory_reserved, mock_set_device, 
                               mock_get_props, mock_device_count, mock_cuda_available):
        """Test getting available GPU information"""
        # Mock CUDA availability
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2
        
        # Mock GPU properties
        mock_props = Mock()
        mock_props.name = "Tesla V100"
        mock_props.total_memory = 16 * 1024**3  # 16GB
        mock_get_props.return_value = mock_props
        
        # Mock memory info
        mock_memory_reserved.return_value = 2 * 1024**3  # 2GB reserved
        
        gpus = self.gpu_manager.get_available_gpus()
        
        assert len(gpus) == 2
        assert all(isinstance(gpu, GPUInfo) for gpu in gpus)
        assert gpus[0].name == "Tesla V100"
        assert gpus[0].memory_total == 16 * 1024**3
    
    @patch('torch.cuda.is_available')
    def test_get_available_gpus_no_cuda(self, mock_cuda_available):
        """Test getting GPUs when CUDA is not available"""
        mock_cuda_available.return_value = False
        
        gpus = self.gpu_manager.get_available_gpus()
        
        assert len(gpus) == 0
    
    def test_allocate_gpus_success(self):
        """Test successful GPU allocation"""
        # Mock available GPUs
        with patch.object(self.gpu_manager, 'get_available_gpus') as mock_get_gpus:
            mock_get_gpus.return_value = [
                GPUInfo(0, "GPU0", 1000, 800, 200, 20.0, 60, True),
                GPUInfo(1, "GPU1", 1000, 900, 100, 10.0, 55, True),
                GPUInfo(2, "GPU2", 1000, 200, 800, 80.0, 75, False)
            ]
            
            allocated = self.gpu_manager.allocate_gpus("job1", 2)
            
            assert allocated == [0, 1]
            assert self.gpu_manager._allocated_gpus["job1"] == [0, 1]
    
    def test_allocate_gpus_insufficient(self):
        """Test GPU allocation when insufficient GPUs available"""
        with patch.object(self.gpu_manager, 'get_available_gpus') as mock_get_gpus:
            mock_get_gpus.return_value = [
                GPUInfo(0, "GPU0", 1000, 800, 200, 20.0, 60, True),
                GPUInfo(1, "GPU1", 1000, 200, 800, 80.0, 75, False)
            ]
            
            with pytest.raises(ResourceError):
                self.gpu_manager.allocate_gpus("job1", 2)
    
    def test_release_gpus(self):
        """Test GPU release"""
        # First allocate
        self.gpu_manager._allocated_gpus["job1"] = [0, 1]
        
        result = self.gpu_manager.release_gpus("job1")
        
        assert result is True
        assert "job1" not in self.gpu_manager._allocated_gpus
    
    def test_release_gpus_nonexistent(self):
        """Test releasing GPUs for non-existent job"""
        result = self.gpu_manager.release_gpus("nonexistent")
        
        assert result is False


class TestDistributedTrainer:
    """Test distributed training functionality"""
    
    def setup_method(self):
        self.trainer = DistributedTrainer("pytorch")
    
    @patch('torch.cuda.is_available')
    def test_init_pytorch_cuda(self, mock_cuda_available):
        """Test PyTorch initialization with CUDA"""
        mock_cuda_available.return_value = True
        
        trainer = DistributedTrainer("pytorch")
        
        assert trainer.backend == "pytorch"
        assert trainer.device.type == "cuda"
    
    @patch('torch.cuda.is_available')
    def test_init_pytorch_cpu(self, mock_cuda_available):
        """Test PyTorch initialization without CUDA"""
        mock_cuda_available.return_value = False
        
        trainer = DistributedTrainer("pytorch")
        
        assert trainer.backend == "pytorch"
        assert trainer.device.type == "cpu"
    
    def test_init_unsupported_backend(self):
        """Test initialization with unsupported backend"""
        with pytest.raises(ValueError):
            DistributedTrainer("unsupported")
    
    def test_get_pytorch_optimizer(self):
        """Test PyTorch optimizer creation"""
        model = nn.Linear(10, 1)
        config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer="adam",
            epochs=10,
            early_stopping_patience=5,
            regularization={}
        )
        
        optimizer = self.trainer._get_pytorch_optimizer(model, config)
        
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]['lr'] == 0.001
    
    def test_get_pytorch_criterion(self):
        """Test PyTorch criterion creation"""
        config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer="adam",
            epochs=10,
            early_stopping_patience=5,
            regularization={}
        )
        
        criterion = self.trainer._get_pytorch_criterion(config)
        
        assert isinstance(criterion, nn.CrossEntropyLoss)
    
    def test_calculate_metrics(self):
        """Test metrics calculation"""
        # Create mock tensors
        output = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        target = torch.tensor([1, 0, 1])
        
        metrics = self.trainer._calculate_metrics(output, target)
        
        assert 'accuracy' in metrics
        assert 0.0 <= metrics['accuracy'] <= 1.0


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.linear(x)


class TestTrainingJobManager:
    """Test training job management"""
    
    def setup_method(self):
        self.job_manager = TrainingJobManager()
    
    def teardown_method(self):
        self.job_manager.shutdown()
    
    def test_submit_job(self):
        """Test job submission"""
        job = TrainingJob(
            id="test_job",
            experiment_id="exp1",
            architecture=Mock(),
            config=Mock(),
            status="queued",
            created_at="2023-01-01 00:00:00",
            gpu_allocation=[]
        )
        
        job_id = self.job_manager.submit_job(job)
        
        assert job_id == "test_job"
        assert not self.job_manager.job_queue.empty()
    
    def test_get_job_status_nonexistent(self):
        """Test getting status of non-existent job"""
        status = self.job_manager.get_job_status("nonexistent")
        
        assert status is None
    
    def test_cancel_job(self):
        """Test job cancellation"""
        job = TrainingJob(
            id="test_job",
            experiment_id="exp1",
            architecture=Mock(),
            config=Mock(),
            status="running",
            created_at="2023-01-01 00:00:00",
            gpu_allocation=[0]
        )
        
        self.job_manager.active_jobs["test_job"] = job
        
        result = self.job_manager.cancel_job("test_job")
        
        assert result is True
        assert job.status == "cancelled"


class TestModelTrainingService:
    """Test main training service"""
    
    def setup_method(self):
        self.service = ModelTrainingService("pytorch")
    
    def teardown_method(self):
        self.service.job_manager.shutdown()
    
    def test_init(self):
        """Test service initialization"""
        assert self.service.backend == "pytorch"
        assert isinstance(self.service.distributed_trainer, DistributedTrainer)
        assert isinstance(self.service.job_manager, TrainingJobManager)
    
    def test_save_checkpoint_pytorch(self):
        """Test checkpoint saving for PyTorch"""
        model = SimpleModel()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pth")
            
            result_path = self.service.save_checkpoint(model, 5, checkpoint_path)
            
            assert result_path == checkpoint_path
            assert os.path.exists(checkpoint_path)
            
            # Verify checkpoint content
            checkpoint = torch.load(checkpoint_path)
            assert checkpoint['epoch'] == 5
            assert 'model_state_dict' in checkpoint
    
    def test_save_checkpoint_error(self):
        """Test checkpoint saving error handling"""
        model = SimpleModel()
        invalid_path = "/invalid/path/checkpoint.pth"
        
        with pytest.raises(TrainingError):
            self.service.save_checkpoint(model, 5, invalid_path)
    
    def test_register_progress_callback(self):
        """Test progress callback registration"""
        callback = Mock()
        
        self.service.register_progress_callback("job1", callback)
        
        assert "job1" in self.service.progress_callbacks
        assert self.service.progress_callbacks["job1"] == callback
    
    def test_get_training_statistics(self):
        """Test getting training statistics"""
        with patch.object(self.service.distributed_trainer.gpu_manager, 'get_available_gpus') as mock_get_gpus:
            mock_get_gpus.return_value = [
                GPUInfo(0, "GPU0", 1000, 800, 200, 20.0, 60, True),
                GPUInfo(1, "GPU1", 1000, 200, 800, 80.0, 75, False)
            ]
            
            stats = self.service.get_training_statistics()
            
            assert 'active_jobs' in stats
            assert 'queued_jobs' in stats
            assert 'completed_jobs' in stats
            assert 'available_gpus' in stats
            assert 'total_gpus' in stats
            assert 'gpu_utilization' in stats
            
            assert stats['available_gpus'] == 1
            assert stats['total_gpus'] == 2


class TestTrainingProgress:
    """Test training progress tracking"""
    
    def test_training_progress_creation(self):
        """Test creating training progress object"""
        progress = TrainingProgress(
            epoch=5,
            total_epochs=10,
            batch=100,
            total_batches=200,
            train_loss=0.5,
            val_loss=0.6,
            train_metrics={'accuracy': 0.8},
            val_metrics={'accuracy': 0.75},
            learning_rate=0.001,
            elapsed_time=300.0,
            estimated_remaining_time=300.0
        )
        
        assert progress.epoch == 5
        assert progress.total_epochs == 10
        assert progress.train_loss == 0.5
        assert progress.val_loss == 0.6
        assert progress.train_metrics['accuracy'] == 0.8


@pytest.fixture
def sample_architecture():
    """Sample architecture for testing"""
    return Architecture(
        id="arch1",
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


@pytest.fixture
def sample_processed_data():
    """Sample processed data for testing"""
    import pandas as pd
    
    return ProcessedData(
        train_data=pd.DataFrame({'feature1': [1, 2, 3], 'target': [0, 1, 0]}),
        validation_data=pd.DataFrame({'feature1': [4, 5], 'target': [1, 0]}),
        test_data=pd.DataFrame({'feature1': [6, 7], 'target': [0, 1]}),
        preprocessing_pipeline=Mock(),
        feature_names=['feature1']
    )


class TestIntegration:
    """Integration tests for training service"""
    
    def test_full_training_workflow(self, sample_architecture, sample_training_config, sample_processed_data):
        """Test complete training workflow"""
        service = ModelTrainingService("pytorch")
        
        try:
            # Mock the job execution to complete quickly
            with patch.object(service.job_manager, '_execute_job') as mock_execute:
                mock_execute.return_value = {
                    'job_id': 'test_job',
                    'status': 'completed',
                    'metrics': {
                        'accuracy': 0.95,
                        'loss': 0.1,
                        'training_time': 600
                    }
                }
                
                # This would normally take a long time, so we'll mock the waiting
                with patch('time.sleep'):
                    trained_model = service.train_model(
                        sample_architecture,
                        sample_training_config,
                        sample_processed_data
                    )
                
                assert trained_model is not None
                assert trained_model.architecture == sample_architecture
                assert trained_model.config == sample_training_config
                assert trained_model.metrics.accuracy == 0.95
        
        finally:
            service.job_manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])