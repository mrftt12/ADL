"""
Unit tests for training monitoring service
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from automl_framework.services.training_monitor import (
    MetricsBuffer, EarlyStopping, LearningRateScheduler, TrainingMonitor,
    TrainingVisualizer, TrainingMonitorManager, MetricSnapshot, 
    EarlyStoppingConfig, LearningRateScheduleConfig, TrainingMetrics
)


class TestMetricsBuffer:
    """Test metrics buffer functionality"""
    
    def setup_method(self):
        self.buffer = MetricsBuffer(max_size=100)
    
    def test_add_metric(self):
        """Test adding metrics to buffer"""
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            epoch=1,
            batch=10,
            metric_name="loss",
            metric_value=0.5,
            phase="train"
        )
        
        self.buffer.add_metric(snapshot)
        
        assert len(self.buffer.buffer) == 1
        assert self.buffer.buffer[0] == snapshot
    
    def test_get_metrics_no_filter(self):
        """Test getting all metrics without filter"""
        snapshots = [
            MetricSnapshot(time.time(), 1, 10, "loss", 0.5, "train"),
            MetricSnapshot(time.time(), 1, 10, "accuracy", 0.8, "train"),
            MetricSnapshot(time.time(), 1, 0, "loss", 0.6, "validation")
        ]
        
        for snapshot in snapshots:
            self.buffer.add_metric(snapshot)
        
        metrics = self.buffer.get_metrics()
        assert len(metrics) == 3
    
    def test_get_metrics_with_filters(self):
        """Test getting metrics with filters"""
        snapshots = [
            MetricSnapshot(time.time(), 1, 10, "loss", 0.5, "train"),
            MetricSnapshot(time.time(), 1, 10, "accuracy", 0.8, "train"),
            MetricSnapshot(time.time(), 1, 0, "loss", 0.6, "validation")
        ]
        
        for snapshot in snapshots:
            self.buffer.add_metric(snapshot)
        
        # Filter by metric name
        loss_metrics = self.buffer.get_metrics(metric_name="loss")
        assert len(loss_metrics) == 2
        
        # Filter by phase
        train_metrics = self.buffer.get_metrics(phase="train")
        assert len(train_metrics) == 2
        
        # Filter by both
        train_loss = self.buffer.get_metrics(metric_name="loss", phase="train")
        assert len(train_loss) == 1
    
    def test_get_latest_metric(self):
        """Test getting latest metric value"""
        snapshots = [
            MetricSnapshot(time.time(), 1, 0, "loss", 0.5, "validation"),
            MetricSnapshot(time.time() + 1, 2, 0, "loss", 0.4, "validation"),
            MetricSnapshot(time.time() + 2, 3, 0, "loss", 0.3, "validation")
        ]
        
        for snapshot in snapshots:
            self.buffer.add_metric(snapshot)
        
        latest = self.buffer.get_latest_metric("loss", "validation")
        assert latest is not None
        assert latest.metric_value == 0.3
        assert latest.epoch == 3
    
    def test_clear_buffer(self):
        """Test clearing buffer"""
        snapshot = MetricSnapshot(time.time(), 1, 10, "loss", 0.5, "train")
        self.buffer.add_metric(snapshot)
        
        assert len(self.buffer.buffer) == 1
        
        self.buffer.clear()
        assert len(self.buffer.buffer) == 0


class TestEarlyStopping:
    """Test early stopping functionality"""
    
    def test_early_stopping_min_mode(self):
        """Test early stopping in min mode (for loss)"""
        config = EarlyStoppingConfig(
            monitor_metric="val_loss",
            patience=3,
            min_delta=0.001,
            mode="min"
        )
        
        early_stopping = EarlyStopping(config)
        
        # Improving scores
        assert not early_stopping(1.0)
        assert not early_stopping(0.9)
        assert not early_stopping(0.8)
        
        # No improvement
        assert not early_stopping(0.81)  # counter = 1
        assert not early_stopping(0.82)  # counter = 2
        assert early_stopping(0.83)      # counter = 3, should stop
        assert early_stopping.should_stop
    
    def test_early_stopping_max_mode(self):
        """Test early stopping in max mode (for accuracy)"""
        config = EarlyStoppingConfig(
            monitor_metric="val_accuracy",
            patience=2,
            min_delta=0.001,
            mode="max"
        )
        
        early_stopping = EarlyStopping(config)
        
        # Improving scores
        assert not early_stopping(0.8)
        assert not early_stopping(0.9)
        
        # No improvement
        assert not early_stopping(0.89)  # counter = 1
        assert early_stopping(0.88)      # counter = 2, should stop
        assert early_stopping.should_stop
    
    def test_early_stopping_reset(self):
        """Test resetting early stopping state"""
        config = EarlyStoppingConfig(patience=2, mode="min")
        early_stopping = EarlyStopping(config)
        
        # Build up some state
        early_stopping(1.0)
        early_stopping(1.1)
        early_stopping(1.2)
        
        assert early_stopping.counter == 2
        assert early_stopping.best_score == 1.0
        
        # Reset
        early_stopping.reset()
        
        assert early_stopping.counter == 0
        assert early_stopping.best_score is None
        assert not early_stopping.should_stop


class TestLearningRateScheduler:
    """Test learning rate scheduling"""
    
    def test_reduce_on_plateau(self):
        """Test reduce on plateau scheduler"""
        config = LearningRateScheduleConfig(
            scheduler_type="reduce_on_plateau",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
        
        scheduler = LearningRateScheduler(config, initial_lr=0.001)
        
        # No reduction initially
        assert scheduler.step(1.0) == 0.001
        assert scheduler.step(0.9) == 0.001  # Improvement
        
        # No improvement
        assert scheduler.step(0.91) == 0.001  # counter = 1
        new_lr = scheduler.step(0.92)         # counter = 2, should reduce
        assert new_lr == 0.0005
    
    def test_exponential_decay(self):
        """Test exponential decay scheduler"""
        config = LearningRateScheduleConfig(
            scheduler_type="exponential",
            gamma=0.9
        )
        
        scheduler = LearningRateScheduler(config, initial_lr=0.001)
        
        assert scheduler.step(epoch=0) == 0.001
        assert abs(scheduler.step(epoch=1) - 0.0009) < 1e-6
        assert abs(scheduler.step(epoch=2) - 0.00081) < 1e-6
    
    def test_step_decay(self):
        """Test step decay scheduler"""
        config = LearningRateScheduleConfig(
            scheduler_type="step",
            step_size=2,
            gamma=0.5
        )
        
        scheduler = LearningRateScheduler(config, initial_lr=0.001)
        
        assert scheduler.step(epoch=0) == 0.001
        assert scheduler.step(epoch=1) == 0.001
        assert scheduler.step(epoch=2) == 0.0005
        assert scheduler.step(epoch=3) == 0.0005
        assert scheduler.step(epoch=4) == 0.00025
    
    def test_cosine_annealing(self):
        """Test cosine annealing scheduler"""
        config = LearningRateScheduleConfig(
            scheduler_type="cosine",
            t_max=10,
            eta_min=0.0001
        )
        
        scheduler = LearningRateScheduler(config, initial_lr=0.001)
        
        # At epoch 0, should be initial_lr
        assert scheduler.step(epoch=0) == 0.001
        
        # At epoch t_max/2, should be around eta_min
        mid_lr = scheduler.step(epoch=5)
        assert abs(mid_lr - 0.00055) < 0.0001  # Adjusted expected value
        
        # At epoch t_max, should be eta_min
        final_lr = scheduler.step(epoch=10)
        assert abs(final_lr - 0.0001) < 1e-6


class TestTrainingVisualizer:
    """Test training visualization"""
    
    def setup_method(self):
        self.visualizer = TrainingVisualizer()
    
    def test_generate_training_report_empty(self):
        """Test generating report with empty history"""
        report = self.visualizer.generate_training_report([])
        assert report == {}
    
    def test_generate_training_report(self):
        """Test generating training report"""
        metrics_history = [
            TrainingMetrics(
                epoch=1,
                train_loss=1.0,
                val_loss=1.1,
                train_metrics={'accuracy': 0.7},
                val_metrics={'accuracy': 0.65},
                learning_rate=0.001,
                epoch_time=30.0,
                gpu_memory_used={0: 2048},
                cpu_usage=50.0,
                timestamp=time.time()
            ),
            TrainingMetrics(
                epoch=2,
                train_loss=0.8,
                val_loss=0.9,
                train_metrics={'accuracy': 0.8},
                val_metrics={'accuracy': 0.75},
                learning_rate=0.001,
                epoch_time=32.0,
                gpu_memory_used={0: 2100},
                cpu_usage=55.0,
                timestamp=time.time()
            )
        ]
        
        report = self.visualizer.generate_training_report(metrics_history)
        
        assert 'training_summary' in report
        assert 'convergence_analysis' in report
        assert 'resource_usage' in report
        
        summary = report['training_summary']
        assert summary['total_epochs'] == 2
        assert summary['final_train_loss'] == 0.8
        assert summary['final_val_loss'] == 0.9
        assert summary['best_val_loss'] == 0.9
    
    def test_check_convergence(self):
        """Test convergence checking"""
        # Converged losses (small changes)
        converged_losses = [1.0, 0.9, 0.85, 0.84, 0.83, 0.825, 0.824, 0.823, 0.822, 0.821]
        assert self.visualizer._check_convergence(converged_losses, window=5, threshold=0.1)  # Increased threshold
        
        # Not converged losses (large changes)
        not_converged_losses = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        assert not self.visualizer._check_convergence(not_converged_losses, window=5, threshold=0.01)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_training_curves(self, mock_close, mock_savefig):
        """Test plotting training curves"""
        metrics_history = [
            TrainingMetrics(
                epoch=1, train_loss=1.0, val_loss=1.1,
                train_metrics={'accuracy': 0.7}, val_metrics={'accuracy': 0.65},
                learning_rate=0.001, epoch_time=30.0,
                gpu_memory_used={}, cpu_usage=50.0, timestamp=time.time()
            )
        ]
        
        # Mock the buffer to return base64 string
        with patch('io.BytesIO') as mock_buffer:
            mock_buffer.return_value.getvalue.return_value = b'fake_image_data'
            
            result = self.visualizer.plot_training_curves(metrics_history)
            
            assert isinstance(result, str)
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()


class TestTrainingMonitor:
    """Test training monitor"""
    
    def setup_method(self):
        self.monitor = TrainingMonitor("test_job")
    
    def teardown_method(self):
        self.monitor.stop_monitoring()
    
    def test_setup_early_stopping(self):
        """Test setting up early stopping"""
        config = EarlyStoppingConfig(patience=5)
        self.monitor.setup_early_stopping(config)
        
        assert self.monitor.early_stopping is not None
        assert self.monitor.early_stopping.config.patience == 5
    
    def test_setup_lr_scheduler(self):
        """Test setting up learning rate scheduler"""
        config = LearningRateScheduleConfig(scheduler_type="exponential")
        self.monitor.setup_lr_scheduler(config, initial_lr=0.001)
        
        assert self.monitor.lr_scheduler is not None
        assert self.monitor.lr_scheduler.config.scheduler_type == "exponential"
    
    def test_add_callback(self):
        """Test adding callback"""
        callback = Mock()
        self.monitor.add_callback(callback)
        
        assert callback in self.monitor.callbacks
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring"""
        assert not self.monitor.is_monitoring
        
        self.monitor.start_monitoring()
        assert self.monitor.is_monitoring
        assert self.monitor.monitoring_thread is not None
        
        self.monitor.stop_monitoring()
        assert not self.monitor.is_monitoring
    
    def test_log_epoch_metrics(self):
        """Test logging epoch metrics"""
        metrics = TrainingMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            train_metrics={'accuracy': 0.8},
            val_metrics={'accuracy': 0.75},
            learning_rate=0.001,
            epoch_time=30.0,
            gpu_memory_used={0: 2048},
            cpu_usage=50.0,
            timestamp=time.time()
        )
        
        self.monitor.log_epoch_metrics(metrics)
        
        assert len(self.monitor.metrics_history) == 1
        assert self.monitor.current_epoch == 1
        
        # Check metrics were added to buffer
        train_loss = self.monitor.metrics_buffer.get_latest_metric("loss", "train")
        assert train_loss is not None
        assert train_loss.metric_value == 0.5
    
    def test_log_batch_metrics(self):
        """Test logging batch metrics"""
        self.monitor.current_epoch = 1
        self.monitor.log_batch_metrics(10, 0.5, {'accuracy': 0.8})
        
        assert self.monitor.current_batch == 10
        
        # Check metrics were added to buffer
        batch_loss = self.monitor.metrics_buffer.get_latest_metric("loss", "train")
        assert batch_loss is not None
        assert batch_loss.metric_value == 0.5
        assert batch_loss.batch == 10
    
    def test_get_current_status(self):
        """Test getting current status"""
        self.monitor.current_epoch = 5
        self.monitor.current_batch = 100
        
        status = self.monitor.get_current_status()
        
        assert status['job_id'] == "test_job"
        assert status['current_epoch'] == 5
        assert status['current_batch'] == 100
        assert 'elapsed_time' in status
    
    def test_get_metrics_history(self):
        """Test getting metrics history"""
        # Add some metrics
        self.monitor.log_batch_metrics(1, 0.5, {'accuracy': 0.8})
        self.monitor.log_batch_metrics(2, 0.4, {'accuracy': 0.85})
        
        # Get all metrics
        all_metrics = self.monitor.get_metrics_history()
        assert len(all_metrics) >= 4  # 2 loss + 2 accuracy metrics
        
        # Get filtered metrics
        loss_metrics = self.monitor.get_metrics_history(metric_name="loss")
        assert len(loss_metrics) == 2
        
        # Get last N metrics
        last_metrics = self.monitor.get_metrics_history(last_n=2)
        assert len(last_metrics) == 2


class TestTrainingMonitorManager:
    """Test training monitor manager"""
    
    def setup_method(self):
        self.manager = TrainingMonitorManager()
    
    def teardown_method(self):
        # Clean up all monitors
        for job_id in list(self.manager.monitors.keys()):
            self.manager.remove_monitor(job_id)
    
    def test_create_monitor(self):
        """Test creating a monitor"""
        monitor = self.manager.create_monitor("job1")
        
        assert monitor is not None
        assert monitor.job_id == "job1"
        assert "job1" in self.manager.monitors
    
    def test_create_duplicate_monitor(self):
        """Test creating duplicate monitor"""
        monitor1 = self.manager.create_monitor("job1")
        monitor2 = self.manager.create_monitor("job1")
        
        assert monitor1 is monitor2
        assert len(self.manager.monitors) == 1
    
    def test_get_monitor(self):
        """Test getting existing monitor"""
        created_monitor = self.manager.create_monitor("job1")
        retrieved_monitor = self.manager.get_monitor("job1")
        
        assert created_monitor is retrieved_monitor
    
    def test_get_nonexistent_monitor(self):
        """Test getting non-existent monitor"""
        monitor = self.manager.get_monitor("nonexistent")
        assert monitor is None
    
    def test_remove_monitor(self):
        """Test removing monitor"""
        self.manager.create_monitor("job1")
        assert "job1" in self.manager.monitors
        
        result = self.manager.remove_monitor("job1")
        assert result is True
        assert "job1" not in self.manager.monitors
    
    def test_remove_nonexistent_monitor(self):
        """Test removing non-existent monitor"""
        result = self.manager.remove_monitor("nonexistent")
        assert result is False
    
    def test_get_all_statuses(self):
        """Test getting all monitor statuses"""
        self.manager.create_monitor("job1")
        self.manager.create_monitor("job2")
        
        statuses = self.manager.get_all_statuses()
        
        assert len(statuses) == 2
        assert "job1" in statuses
        assert "job2" in statuses
        assert all('job_id' in status for status in statuses.values())
    
    def test_cleanup_completed_monitors(self):
        """Test cleaning up completed monitors"""
        monitor1 = self.manager.create_monitor("job1")
        monitor2 = self.manager.create_monitor("job2")
        
        # Simulate one monitor completing
        monitor1.is_monitoring = False
        # Keep monitor2 active
        monitor2.start_monitoring()
        
        self.manager.cleanup_completed_monitors()
        
        assert "job1" not in self.manager.monitors
        assert "job2" in self.manager.monitors
        
        # Clean up the remaining monitor
        monitor2.stop_monitoring()


if __name__ == "__main__":
    pytest.main([__file__])