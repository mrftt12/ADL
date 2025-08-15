"""
Comprehensive unit tests for model training service.

Tests cover distributed training, monitoring, checkpointing, and various
training scenarios with different architectures and configurations.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta

from automl_framework.services.training_service import (
    TrainingService, DistributedTrainer, TrainingMonitor, 
    EarlyStopping, LearningRateScheduler
)
from automl_framework.services.checkpoint_manager import CheckpointManager
from automl_framework.models.data_models import (
    Architecture, TrainingConfig, PerformanceMetrics, 
    ExperimentStatus, TaskType
)
from tests.test_utils import (
    MockArchitectureGenerator, MockDatasetGenerator, TestDataManager,
    assert_architecture_valid, assert_metrics_valid, PerformanceBenchmark,
    mock_simple_mlp, mock_simple_cnn, test_data_manager
)


class TestTrainingMonitor:
    """Test TrainingMonitor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = TrainingMonitor()
    
    def test_monitor_initialization(self):
        """Test training monitor initialization."""
        assert self.monitor.metrics_history == []
        assert self.monitor.current_epoch == 0
        assert self.monitor.best_metric == float('-inf')
        assert self.monitor.start_time is None
    
    def test_start_monitoring(self):
        """Test starting training monitoring."""
        self.monitor.start_monitoring()
        
        assert self.monitor.start_time is not None
        assert isinstance(self.monitor.start_time, datetime)
    
    def test_log_epoch_metrics(self):
        """Test logging epoch metrics."""
        self.monitor.start_monitoring()
        
        metrics = {
            'train_loss': 0.5,
            'train_accuracy': 0.8,
            'val_loss': 0.6,
            'val_accuracy': 0.75
        }
        
        self.monitor.log_epoch_metrics(1, metrics)
        
        assert len(self.monitor.metrics_history) == 1
        assert self.monitor.current_epoch == 1
        assert self.monitor.metrics_history[0]['epoch'] == 1
        assert self.monitor.metrics_history[0]['metrics'] == metrics
    
    def test_update_best_metric(self):
        """Test updating best metric tracking."""
        self.monitor.start_monitoring()
        
        # First epoch
        self.monitor.log_epoch_metrics(1, {'val_accuracy': 0.8})
        assert self.monitor.best_metric == 0.8
        assert self.monitor.best_epoch == 1
        
        # Better epoch
        self.monitor.log_epoch_metrics(2, {'val_accuracy': 0.85})
        assert self.monitor.best_metric == 0.85
        assert self.monitor.best_epoch == 2
        
        # Worse epoch
        self.monitor.log_epoch_metrics(3, {'val_accuracy': 0.82})
        assert self.monitor.best_metric == 0.85  # Should remain unchanged
        assert self.monitor.best_epoch == 2
    
    def test_get_training_summary(self):
        """Test getting training summary."""
        self.monitor.start_monitoring()
        
        # Log some metrics
        for epoch in range(1, 4):
            metrics = {
                'train_loss': 1.0 - epoch * 0.2,
                'val_accuracy': 0.5 + epoch * 0.1
            }
            self.monitor.log_epoch_metrics(epoch, metrics)
        
        summary = self.monitor.get_training_summary()
        
        assert isinstance(summary, dict)
        assert 'total_epochs' in summary
        assert 'best_metric' in summary
        assert 'best_epoch' in summary
        assert 'training_time' in summary
        assert 'final_metrics' in summary
        
        assert summary['total_epochs'] == 3
        assert summary['best_metric'] == 0.8
        assert summary['best_epoch'] == 3
    
    def test_calculate_training_time(self):
        """Test training time calculation."""
        self.monitor.start_monitoring()
        
        # Simulate some training time
        import time
        time.sleep(0.1)
        
        training_time = self.monitor.get_training_time()
        assert training_time > 0
        assert training_time < 1.0  # Should be less than 1 second
    
    def test_export_metrics_history(self):
        """Test exporting metrics history."""
        self.monitor.start_monitoring()
        
        # Log metrics
        for epoch in range(1, 4):
            metrics = {'train_loss': 1.0 - epoch * 0.2}
            self.monitor.log_epoch_metrics(epoch, metrics)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.monitor.export_metrics_history(temp_file)
            
            # Verify file was created and contains data
            assert os.path.exists(temp_file)
            
            import json
            with open(temp_file, 'r') as f:
                exported_data = json.load(f)
            
            assert 'metrics_history' in exported_data
            assert len(exported_data['metrics_history']) == 3
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestEarlyStopping:
    """Test EarlyStopping class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.early_stopping = EarlyStopping(
            patience=3,
            min_delta=0.01,
            monitor='val_loss',
            mode='min'
        )
    
    def test_early_stopping_initialization(self):
        """Test early stopping initialization."""
        assert self.early_stopping.patience == 3
        assert self.early_stopping.min_delta == 0.01
        assert self.early_stopping.monitor == 'val_loss'
        assert self.early_stopping.mode == 'min'
        assert self.early_stopping.best_score is None
        assert self.early_stopping.counter == 0
    
    def test_early_stopping_improvement(self):
        """Test early stopping with improvement."""
        # First epoch - should not stop
        should_stop = self.early_stopping.check_early_stopping({'val_loss': 1.0})
        assert should_stop is False
        assert self.early_stopping.best_score == 1.0
        assert self.early_stopping.counter == 0
        
        # Second epoch - improvement
        should_stop = self.early_stopping.check_early_stopping({'val_loss': 0.8})
        assert should_stop is False
        assert self.early_stopping.best_score == 0.8
        assert self.early_stopping.counter == 0
    
    def test_early_stopping_no_improvement(self):
        """Test early stopping without improvement."""
        # Initialize with first score
        self.early_stopping.check_early_stopping({'val_loss': 1.0})
        
        # No improvement for patience epochs
        for i in range(self.early_stopping.patience):
            should_stop = self.early_stopping.check_early_stopping({'val_loss': 1.1})
            if i < self.early_stopping.patience - 1:
                assert should_stop is False
            else:
                assert should_stop is True
        
        assert self.early_stopping.counter == self.early_stopping.patience
    
    def test_early_stopping_maximize_mode(self):
        """Test early stopping in maximize mode."""
        early_stopping = EarlyStopping(
            patience=2,
            monitor='val_accuracy',
            mode='max'
        )
        
        # First epoch
        early_stopping.check_early_stopping({'val_accuracy': 0.8})
        assert early_stopping.best_score == 0.8
        
        # Improvement
        should_stop = early_stopping.check_early_stopping({'val_accuracy': 0.85})
        assert should_stop is False
        assert early_stopping.best_score == 0.85
        assert early_stopping.counter == 0
        
        # No improvement
        should_stop = early_stopping.check_early_stopping({'val_accuracy': 0.82})
        assert should_stop is False
        assert early_stopping.counter == 1
    
    def test_early_stopping_min_delta(self):
        """Test early stopping with minimum delta."""
        early_stopping = EarlyStopping(
            patience=2,
            min_delta=0.05,
            monitor='val_loss',
            mode='min'
        )
        
        # Initialize
        early_stopping.check_early_stopping({'val_loss': 1.0})
        
        # Small improvement (less than min_delta)
        should_stop = early_stopping.check_early_stopping({'val_loss': 0.98})
        assert should_stop is False
        assert early_stopping.counter == 1  # Should count as no improvement
        
        # Significant improvement
        should_stop = early_stopping.check_early_stopping({'val_loss': 0.9})
        assert should_stop is False
        assert early_stopping.counter == 0  # Should reset counter


class TestLearningRateScheduler:
    """Test LearningRateScheduler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scheduler = LearningRateScheduler(
            initial_lr=0.01,
            schedule_type='step',
            step_size=10,
            gamma=0.1
        )
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        assert self.scheduler.initial_lr == 0.01
        assert self.scheduler.schedule_type == 'step'
        assert self.scheduler.current_lr == 0.01
    
    def test_step_schedule(self):
        """Test step learning rate schedule."""
        # Before step
        for epoch in range(9):
            lr = self.scheduler.get_lr(epoch)
            assert lr == 0.01
        
        # At step
        lr = self.scheduler.get_lr(10)
        assert lr == 0.001  # 0.01 * 0.1
        
        # After step
        lr = self.scheduler.get_lr(15)
        assert lr == 0.001
        
        # Next step
        lr = self.scheduler.get_lr(20)
        assert lr == 0.0001  # 0.001 * 0.1
    
    def test_exponential_schedule(self):
        """Test exponential learning rate schedule."""
        scheduler = LearningRateScheduler(
            initial_lr=0.01,
            schedule_type='exponential',
            gamma=0.95
        )
        
        lr_epoch_0 = scheduler.get_lr(0)
        lr_epoch_1 = scheduler.get_lr(1)
        lr_epoch_2 = scheduler.get_lr(2)
        
        assert lr_epoch_0 == 0.01
        assert abs(lr_epoch_1 - 0.01 * 0.95) < 1e-6
        assert abs(lr_epoch_2 - 0.01 * 0.95 ** 2) < 1e-6
    
    def test_cosine_schedule(self):
        """Test cosine annealing learning rate schedule."""
        scheduler = LearningRateScheduler(
            initial_lr=0.01,
            schedule_type='cosine',
            T_max=100,
            eta_min=0.001
        )
        
        lr_epoch_0 = scheduler.get_lr(0)
        lr_epoch_50 = scheduler.get_lr(50)  # Middle of cycle
        lr_epoch_100 = scheduler.get_lr(100)  # End of cycle
        
        assert lr_epoch_0 == 0.01
        assert lr_epoch_50 < lr_epoch_0  # Should decrease
        assert lr_epoch_100 == 0.001  # Should reach minimum
    
    def test_plateau_schedule(self):
        """Test reduce on plateau schedule."""
        scheduler = LearningRateScheduler(
            initial_lr=0.01,
            schedule_type='plateau',
            patience=3,
            factor=0.5
        )
        
        # No improvement for patience epochs
        for i in range(3):
            lr = scheduler.step_plateau(0.5)  # Same loss
            if i < 2:
                assert lr == 0.01
            else:
                assert lr == 0.005  # Should reduce
    
    def test_warmup_schedule(self):
        """Test learning rate warmup."""
        scheduler = LearningRateScheduler(
            initial_lr=0.01,
            schedule_type='warmup',
            warmup_epochs=5,
            target_lr=0.1
        )
        
        # During warmup
        lr_epoch_0 = scheduler.get_lr(0)
        lr_epoch_2 = scheduler.get_lr(2)
        lr_epoch_5 = scheduler.get_lr(5)
        
        assert lr_epoch_0 < lr_epoch_2 < lr_epoch_5
        assert lr_epoch_5 == 0.1  # Should reach target


class TestDistributedTrainer:
    """Test DistributedTrainer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trainer = DistributedTrainer()
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_check_gpu_availability(self, mock_device_count, mock_cuda_available):
        """Test GPU availability checking."""
        # Mock GPU available
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2
        
        gpu_info = self.trainer._check_gpu_availability()
        
        assert gpu_info['cuda_available'] is True
        assert gpu_info['device_count'] == 2
        
        # Mock no GPU
        mock_cuda_available.return_value = False
        mock_device_count.return_value = 0
        
        gpu_info = self.trainer._check_gpu_availability()
        
        assert gpu_info['cuda_available'] is False
        assert gpu_info['device_count'] == 0
    
    def test_calculate_batch_size_per_gpu(self):
        """Test batch size calculation per GPU."""
        # Single GPU
        batch_size = self.trainer._calculate_batch_size_per_gpu(
            total_batch_size=64,
            num_gpus=1
        )
        assert batch_size == 64
        
        # Multiple GPUs
        batch_size = self.trainer._calculate_batch_size_per_gpu(
            total_batch_size=64,
            num_gpus=4
        )
        assert batch_size == 16
        
        # Uneven division
        batch_size = self.trainer._calculate_batch_size_per_gpu(
            total_batch_size=65,
            num_gpus=4
        )
        assert batch_size == 17  # Rounded up
    
    @patch('automl_framework.services.training_service.torch')
    def test_setup_distributed_training(self, mock_torch):
        """Test distributed training setup."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        
        config = self.trainer._setup_distributed_training(
            num_gpus=2,
            backend='nccl'
        )
        
        assert isinstance(config, dict)
        assert 'world_size' in config
        assert 'backend' in config
        assert config['world_size'] == 2
        assert config['backend'] == 'nccl'
    
    def test_estimate_memory_usage(self, mock_simple_mlp):
        """Test memory usage estimation."""
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer='adam',
            epochs=10
        )
        
        memory_estimate = self.trainer._estimate_memory_usage(
            mock_simple_mlp,
            training_config
        )
        
        assert isinstance(memory_estimate, dict)
        assert 'model_memory_mb' in memory_estimate
        assert 'batch_memory_mb' in memory_estimate
        assert 'optimizer_memory_mb' in memory_estimate
        assert 'total_memory_mb' in memory_estimate
        
        assert memory_estimate['total_memory_mb'] > 0
    
    @patch('automl_framework.services.training_service.torch')
    def test_create_data_loaders(self, mock_torch):
        """Test data loader creation."""
        # Mock data
        X_train = np.random.random((1000, 10))
        y_train = np.random.randint(0, 2, 1000)
        X_val = np.random.random((200, 10))
        y_val = np.random.randint(0, 2, 200)
        
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer='adam',
            epochs=10
        )
        
        data_loaders = self.trainer._create_data_loaders(
            X_train, y_train, X_val, y_val, training_config
        )
        
        assert isinstance(data_loaders, dict)
        assert 'train_loader' in data_loaders
        assert 'val_loader' in data_loaders


class TestTrainingService:
    """Test TrainingService integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = TrainingService()
    
    @patch('automl_framework.services.training_service.DistributedTrainer')
    @patch('automl_framework.services.training_service.TrainingMonitor')
    def test_train_model_basic(self, mock_monitor_class, mock_trainer_class, mock_simple_mlp):
        """Test basic model training."""
        # Mock trainer and monitor
        mock_trainer = Mock()
        mock_monitor = Mock()
        mock_trainer_class.return_value = mock_trainer
        mock_monitor_class.return_value = mock_monitor
        
        # Mock training data
        X_train = np.random.random((1000, 784))
        y_train = np.random.randint(0, 10, 1000)
        X_val = np.random.random((200, 784))
        y_val = np.random.randint(0, 10, 200)
        
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer='adam',
            epochs=10
        )
        
        # Mock trainer return value
        mock_trainer.train.return_value = {
            'model': Mock(),
            'training_history': {'loss': [0.8, 0.6, 0.4]},
            'final_metrics': PerformanceMetrics(
                accuracy=0.9,
                loss=0.1,
                precision=0.88,
                recall=0.92,
                f1_score=0.90,
                training_time=1800.0,
                inference_time=0.005
            )
        }
        
        result = self.service.train_model(
            mock_simple_mlp,
            training_config,
            X_train, y_train,
            X_val, y_val
        )
        
        assert isinstance(result, dict)
        assert 'model' in result
        assert 'training_history' in result
        assert 'final_metrics' in result
        assert_metrics_valid(result['final_metrics'])
    
    def test_create_training_config(self, mock_simple_mlp):
        """Test training configuration creation."""
        hyperparameters = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'optimizer': 'adam',
            'epochs': 100,
            'weight_decay': 1e-4
        }
        
        config = self.service.create_training_config(
            mock_simple_mlp,
            hyperparameters,
            TaskType.CLASSIFICATION
        )
        
        assert isinstance(config, TrainingConfig)
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.optimizer == 'adam'
        assert config.epochs == 100
    
    def test_validate_training_config(self):
        """Test training configuration validation."""
        # Valid config
        valid_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer='adam',
            epochs=10
        )
        
        is_valid, errors = self.service.validate_training_config(valid_config)
        assert is_valid is True
        assert len(errors) == 0
        
        # Invalid config
        invalid_config = TrainingConfig(
            batch_size=0,  # Invalid
            learning_rate=-0.001,  # Invalid
            optimizer='invalid_optimizer',  # Invalid
            epochs=0  # Invalid
        )
        
        is_valid, errors = self.service.validate_training_config(invalid_config)
        assert is_valid is False
        assert len(errors) > 0
    
    def test_estimate_training_time(self, mock_simple_mlp):
        """Test training time estimation."""
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer='adam',
            epochs=100
        )
        
        # Mock dataset size
        dataset_size = 10000
        
        time_estimate = self.service.estimate_training_time(
            mock_simple_mlp,
            training_config,
            dataset_size
        )
        
        assert isinstance(time_estimate, dict)
        assert 'estimated_time_seconds' in time_estimate
        assert 'estimated_time_per_epoch' in time_estimate
        assert 'factors' in time_estimate
        
        assert time_estimate['estimated_time_seconds'] > 0
    
    @patch('automl_framework.services.training_service.CheckpointManager')
    def test_train_with_checkpointing(self, mock_checkpoint_class, mock_simple_mlp):
        """Test training with checkpointing enabled."""
        mock_checkpoint_manager = Mock()
        mock_checkpoint_class.return_value = mock_checkpoint_manager
        
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer='adam',
            epochs=10,
            save_checkpoints=True,
            checkpoint_frequency=5
        )
        
        # Mock data
        X_train = np.random.random((100, 784))
        y_train = np.random.randint(0, 10, 100)
        X_val = np.random.random((20, 784))
        y_val = np.random.randint(0, 10, 20)
        
        with patch.object(self.service, '_train_model_internal') as mock_train:
            mock_train.return_value = {
                'model': Mock(),
                'training_history': {'loss': [0.5]},
                'final_metrics': PerformanceMetrics(
                    accuracy=0.9, loss=0.1, precision=0.88,
                    recall=0.92, f1_score=0.90,
                    training_time=100.0, inference_time=0.01
                )
            }
            
            result = self.service.train_model(
                mock_simple_mlp, training_config,
                X_train, y_train, X_val, y_val
            )
            
            # Verify checkpoint manager was used
            mock_checkpoint_class.assert_called_once()
    
    def test_resume_training_from_checkpoint(self, mock_simple_mlp):
        """Test resuming training from checkpoint."""
        checkpoint_path = "/path/to/checkpoint.pth"
        
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer='adam',
            epochs=20,
            resume_from_checkpoint=checkpoint_path
        )
        
        with patch.object(self.service, '_load_checkpoint') as mock_load:
            mock_load.return_value = {
                'epoch': 10,
                'model_state': {},
                'optimizer_state': {},
                'training_history': {'loss': [0.8, 0.6, 0.4]}
            }
            
            with patch.object(self.service, '_train_model_internal') as mock_train:
                mock_train.return_value = {
                    'model': Mock(),
                    'training_history': {'loss': [0.8, 0.6, 0.4, 0.2]},
                    'final_metrics': PerformanceMetrics(
                        accuracy=0.95, loss=0.05, precision=0.92,
                        recall=0.88, f1_score=0.90,
                        training_time=200.0, inference_time=0.01
                    )
                }
                
                # Mock data
                X_train = np.random.random((100, 784))
                y_train = np.random.randint(0, 10, 100)
                
                result = self.service.train_model(
                    mock_simple_mlp, training_config,
                    X_train, y_train
                )
                
                # Verify checkpoint was loaded
                mock_load.assert_called_once_with(checkpoint_path)
    
    def test_training_with_early_stopping(self, mock_simple_mlp):
        """Test training with early stopping."""
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer='adam',
            epochs=100,
            early_stopping_patience=5,
            early_stopping_min_delta=0.01
        )
        
        # Mock data
        X_train = np.random.random((100, 784))
        y_train = np.random.randint(0, 10, 100)
        X_val = np.random.random((20, 784))
        y_val = np.random.randint(0, 10, 20)
        
        with patch.object(self.service, '_train_model_internal') as mock_train:
            # Simulate early stopping after 15 epochs
            mock_train.return_value = {
                'model': Mock(),
                'training_history': {'loss': [0.8, 0.6, 0.4, 0.35, 0.34]},
                'final_metrics': PerformanceMetrics(
                    accuracy=0.9, loss=0.34, precision=0.88,
                    recall=0.92, f1_score=0.90,
                    training_time=150.0, inference_time=0.01
                ),
                'stopped_early': True,
                'stopped_epoch': 15
            }
            
            result = self.service.train_model(
                mock_simple_mlp, training_config,
                X_train, y_train, X_val, y_val
            )
            
            assert result['stopped_early'] is True
            assert result['stopped_epoch'] == 15
    
    def test_training_performance_benchmark(self, mock_simple_mlp):
        """Test training performance benchmarking."""
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer='adam',
            epochs=5  # Small number for testing
        )
        
        # Mock data
        X_train = np.random.random((100, 784))
        y_train = np.random.randint(0, 10, 100)
        
        # Benchmark training
        benchmark = PerformanceBenchmark()
        
        with patch.object(self.service, '_train_model_internal') as mock_train:
            mock_train.return_value = {
                'model': Mock(),
                'training_history': {'loss': [0.8, 0.6, 0.4]},
                'final_metrics': PerformanceMetrics(
                    accuracy=0.9, loss=0.1, precision=0.88,
                    recall=0.92, f1_score=0.90,
                    training_time=10.0, inference_time=0.01
                )
            }
            
            benchmark_result = benchmark.benchmark_function(
                self.service.train_model,
                mock_simple_mlp, training_config, X_train, y_train
            )
            
            assert 'execution_time' in benchmark_result
            assert 'peak_memory_mb' in benchmark_result
            assert benchmark_result['execution_time'] > 0
    
    def test_multi_gpu_training_setup(self, mock_simple_mlp):
        """Test multi-GPU training setup."""
        training_config = TrainingConfig(
            batch_size=64,
            learning_rate=0.001,
            optimizer='adam',
            epochs=10,
            use_multi_gpu=True
        )
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=4):
                with patch.object(self.service, '_train_model_internal') as mock_train:
                    mock_train.return_value = {
                        'model': Mock(),
                        'training_history': {'loss': [0.5]},
                        'final_metrics': PerformanceMetrics(
                            accuracy=0.9, loss=0.1, precision=0.88,
                            recall=0.92, f1_score=0.90,
                            training_time=50.0, inference_time=0.01
                        )
                    }
                    
                    # Mock data
                    X_train = np.random.random((1000, 784))
                    y_train = np.random.randint(0, 10, 1000)
                    
                    result = self.service.train_model(
                        mock_simple_mlp, training_config, X_train, y_train
                    )
                    
                    # Training should complete successfully
                    assert 'model' in result
                    assert 'final_metrics' in result


if __name__ == "__main__":
    pytest.main([__file__])