"""
Training monitoring and early stopping service
"""

import time
import json
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

from ..core.interfaces import TrainingJob, PerformanceMetrics
from ..core.config import get_config
from ..utils.logging import get_logger


@dataclass
class MetricSnapshot:
    """Single metric measurement at a point in time"""
    timestamp: float
    epoch: int
    batch: int
    metric_name: str
    metric_value: float
    phase: str  # 'train' or 'validation'


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping"""
    monitor_metric: str = "val_loss"
    patience: int = 10
    min_delta: float = 0.001
    mode: str = "min"  # 'min' or 'max'
    restore_best_weights: bool = True
    baseline: Optional[float] = None


@dataclass
class LearningRateScheduleConfig:
    """Configuration for learning rate scheduling"""
    scheduler_type: str = "reduce_on_plateau"  # 'reduce_on_plateau', 'cosine', 'exponential', 'step'
    monitor_metric: str = "val_loss"
    factor: float = 0.5
    patience: int = 5
    min_lr: float = 1e-7
    cooldown: int = 0
    # For step scheduler
    step_size: int = 30
    gamma: float = 0.1
    # For cosine scheduler
    t_max: int = 100
    eta_min: float = 0


@dataclass
class TrainingMetrics:
    """Comprehensive training metrics for an epoch"""
    epoch: int
    train_loss: float
    val_loss: Optional[float]
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    learning_rate: float
    epoch_time: float
    gpu_memory_used: Dict[int, float]
    cpu_usage: float
    timestamp: float


class MetricsBuffer:
    """Thread-safe buffer for storing training metrics"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add_metric(self, snapshot: MetricSnapshot):
        """Add a metric snapshot to the buffer"""
        with self.lock:
            self.buffer.append(snapshot)
    
    def get_metrics(self, job_id: str = None, metric_name: str = None, 
                   phase: str = None, last_n: int = None) -> List[MetricSnapshot]:
        """Get metrics with optional filtering"""
        with self.lock:
            metrics = list(self.buffer)
            
            # Apply filters
            if metric_name:
                metrics = [m for m in metrics if m.metric_name == metric_name]
            if phase:
                metrics = [m for m in metrics if m.phase == phase]
            if last_n:
                metrics = metrics[-last_n:]
            
            return metrics
    
    def get_latest_metric(self, metric_name: str, phase: str = "validation") -> Optional[MetricSnapshot]:
        """Get the latest value for a specific metric"""
        with self.lock:
            for snapshot in reversed(self.buffer):
                if snapshot.metric_name == metric_name and snapshot.phase == phase:
                    return snapshot
            return None
    
    def clear(self):
        """Clear all metrics"""
        with self.lock:
            self.buffer.clear()


class EarlyStopping:
    """Early stopping implementation with configurable criteria"""
    
    def __init__(self, config: EarlyStoppingConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
        
        # Set comparison function based on mode
        if config.mode == "min":
            self.is_better = lambda current, best: current < best - config.min_delta
        else:
            self.is_better = lambda current, best: current > best + config.min_delta
    
    def __call__(self, current_score: float, model_weights: Any = None) -> bool:
        """
        Check if training should stop early
        
        Args:
            current_score: Current metric value
            model_weights: Current model weights (optional)
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            if model_weights is not None and self.config.restore_best_weights:
                self.best_weights = self._copy_weights(model_weights)
        
        elif self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            if model_weights is not None and self.config.restore_best_weights:
                self.best_weights = self._copy_weights(model_weights)
        
        else:
            self.counter += 1
            
        if self.counter >= self.config.patience:
            self.should_stop = True
            self.logger.info(
                f"Early stopping triggered. Best {self.config.monitor_metric}: {self.best_score:.6f}"
            )
        
        return self.should_stop
    
    def _copy_weights(self, weights: Any) -> Any:
        """Copy model weights (implementation depends on framework)"""
        # This would be implemented based on the specific framework
        # For PyTorch: return copy.deepcopy(weights.state_dict())
        # For TensorFlow: return weights.get_weights()
        return weights
    
    def get_best_weights(self) -> Any:
        """Get the best model weights"""
        return self.best_weights
    
    def reset(self):
        """Reset early stopping state"""
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.should_stop = False


class LearningRateScheduler:
    """Learning rate scheduling with multiple strategies"""
    
    def __init__(self, config: LearningRateScheduleConfig, initial_lr: float):
        self.config = config
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.logger = get_logger(__name__)
        
        # State for different schedulers
        self.best_score = None
        self.patience_counter = 0
        self.cooldown_counter = 0
        self.step_counter = 0
    
    def step(self, current_score: Optional[float] = None, epoch: int = 0) -> float:
        """
        Update learning rate based on scheduler type
        
        Args:
            current_score: Current metric value (for plateau scheduler)
            epoch: Current epoch number
            
        Returns:
            New learning rate
        """
        if self.config.scheduler_type == "reduce_on_plateau":
            self.current_lr = self._reduce_on_plateau(current_score)
        elif self.config.scheduler_type == "exponential":
            self.current_lr = self._exponential_decay(epoch)
        elif self.config.scheduler_type == "step":
            self.current_lr = self._step_decay(epoch)
        elif self.config.scheduler_type == "cosine":
            self.current_lr = self._cosine_annealing(epoch)
        
        return self.current_lr
    
    def _reduce_on_plateau(self, current_score: Optional[float]) -> float:
        """Reduce learning rate when metric plateaus"""
        if current_score is None:
            return self.current_lr
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self.current_lr
        
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score - 1e-4:  # Improvement
            self.best_score = current_score
            self.patience_counter = 0
        else:  # No improvement
            self.patience_counter += 1
        
        if self.patience_counter >= self.config.patience:
            new_lr = max(self.current_lr * self.config.factor, self.config.min_lr)
            if new_lr < self.current_lr:
                self.logger.info(f"Reducing learning rate from {self.current_lr:.6f} to {new_lr:.6f}")
                self.patience_counter = 0
                self.cooldown_counter = self.config.cooldown
                self.current_lr = new_lr
                return new_lr
        
        return self.current_lr
    
    def _exponential_decay(self, epoch: int) -> float:
        """Exponential learning rate decay"""
        return self.initial_lr * (self.config.gamma ** epoch)
    
    def _step_decay(self, epoch: int) -> float:
        """Step-wise learning rate decay"""
        return self.initial_lr * (self.config.gamma ** (epoch // self.config.step_size))
    
    def _cosine_annealing(self, epoch: int) -> float:
        """Cosine annealing learning rate schedule"""
        return self.config.eta_min + (self.initial_lr - self.config.eta_min) * \
               (1 + np.cos(np.pi * epoch / self.config.t_max)) / 2
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.current_lr


class TrainingVisualizer:
    """Generate training visualizations and reports"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def plot_training_curves(self, metrics_history: List[TrainingMetrics]) -> str:
        """
        Generate training curves plot
        
        Returns:
            Base64 encoded PNG image
        """
        if not metrics_history:
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Progress', fontsize=16)
        
        epochs = [m.epoch for m in metrics_history]
        train_losses = [m.train_loss for m in metrics_history]
        val_losses = [m.val_loss for m in metrics_history if m.val_loss is not None]
        learning_rates = [m.learning_rate for m in metrics_history]
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, label='Training Loss', color='blue')
        if val_losses:
            val_epochs = [m.epoch for m in metrics_history if m.val_loss is not None]
            axes[0, 0].plot(val_epochs, val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves (if available)
        train_accs = [m.train_metrics.get('accuracy', 0) for m in metrics_history]
        val_accs = [m.val_metrics.get('accuracy', 0) for m in metrics_history if m.val_metrics]
        
        if any(train_accs):
            axes[0, 1].plot(epochs, train_accs, label='Training Accuracy', color='blue')
            if val_accs:
                val_epochs = [m.epoch for m in metrics_history if m.val_metrics]
                axes[0, 1].plot(val_epochs, val_accs, label='Validation Accuracy', color='red')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Accuracy Curves')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(epochs, learning_rates, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Training time per epoch
        epoch_times = [m.epoch_time for m in metrics_history]
        axes[1, 1].plot(epochs, epoch_times, color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('Training Time per Epoch')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return image_base64
    
    def generate_training_report(self, metrics_history: List[TrainingMetrics]) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        if not metrics_history:
            return {}
        
        final_metrics = metrics_history[-1]
        
        # Calculate statistics
        train_losses = [m.train_loss for m in metrics_history]
        val_losses = [m.val_loss for m in metrics_history if m.val_loss is not None]
        
        report = {
            'training_summary': {
                'total_epochs': len(metrics_history),
                'final_train_loss': final_metrics.train_loss,
                'final_val_loss': final_metrics.val_loss,
                'best_val_loss': min(val_losses) if val_losses else None,
                'final_learning_rate': final_metrics.learning_rate,
                'total_training_time': sum(m.epoch_time for m in metrics_history),
                'average_epoch_time': np.mean([m.epoch_time for m in metrics_history])
            },
            'convergence_analysis': {
                'loss_improvement': train_losses[0] - train_losses[-1] if len(train_losses) > 1 else 0,
                'loss_stability': np.std(train_losses[-10:]) if len(train_losses) >= 10 else 0,
                'converged': self._check_convergence(train_losses)
            },
            'resource_usage': {
                'peak_gpu_memory': max(
                    max(m.gpu_memory_used.values()) if m.gpu_memory_used else 0 
                    for m in metrics_history
                ),
                'average_cpu_usage': np.mean([m.cpu_usage for m in metrics_history]),
                'gpu_utilization_efficiency': self._calculate_gpu_efficiency(metrics_history)
            },
            'training_curves_plot': self.plot_training_curves(metrics_history)
        }
        
        return report
    
    def _check_convergence(self, losses: List[float], window: int = 10, threshold: float = 0.001) -> bool:
        """Check if training has converged"""
        if len(losses) < window * 2:
            return False
        
        recent_losses = losses[-window:]
        previous_losses = losses[-window*2:-window]
        
        recent_mean = np.mean(recent_losses)
        previous_mean = np.mean(previous_losses)
        
        if previous_mean == 0:
            return False
        
        improvement = abs(previous_mean - recent_mean) / previous_mean
        return improvement < threshold
    
    def _calculate_gpu_efficiency(self, metrics_history: List[TrainingMetrics]) -> float:
        """Calculate GPU utilization efficiency"""
        if not metrics_history or not any(m.gpu_memory_used for m in metrics_history):
            return 0.0
        
        # Simplified efficiency calculation
        total_usage = 0
        count = 0
        
        for metrics in metrics_history:
            if metrics.gpu_memory_used:
                total_usage += sum(metrics.gpu_memory_used.values())
                count += len(metrics.gpu_memory_used)
        
        return (total_usage / count) if count > 0 else 0.0


class TrainingMonitor:
    """Main training monitoring service"""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.logger = get_logger(__name__)
        self.config = get_config()
        
        # Components
        self.metrics_buffer = MetricsBuffer()
        self.early_stopping = None
        self.lr_scheduler = None
        self.visualizer = TrainingVisualizer()
        
        # State
        self.is_monitoring = False
        self.monitoring_thread = None
        self.callbacks: List[Callable] = []
        self.metrics_history: List[TrainingMetrics] = []
        
        # Real-time metrics
        self.current_epoch = 0
        self.current_batch = 0
        self.start_time = time.time()
    
    def setup_early_stopping(self, config: EarlyStoppingConfig):
        """Setup early stopping with configuration"""
        self.early_stopping = EarlyStopping(config)
        self.logger.info(f"Early stopping configured: monitor={config.monitor_metric}, patience={config.patience}")
    
    def setup_lr_scheduler(self, config: LearningRateScheduleConfig, initial_lr: float):
        """Setup learning rate scheduler"""
        self.lr_scheduler = LearningRateScheduler(config, initial_lr)
        self.logger.info(f"LR scheduler configured: type={config.scheduler_type}")
    
    def add_callback(self, callback: Callable):
        """Add a callback function for real-time updates"""
        self.callbacks.append(callback)
    
    def start_monitoring(self):
        """Start monitoring training progress"""
        self.is_monitoring = True
        self.start_time = time.time()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info(f"Started monitoring for job {self.job_id}")
    
    def stop_monitoring(self):
        """Stop monitoring training progress"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info(f"Stopped monitoring for job {self.job_id}")
    
    def log_epoch_metrics(self, epoch_metrics: TrainingMetrics):
        """Log metrics for a completed epoch"""
        self.metrics_history.append(epoch_metrics)
        self.current_epoch = epoch_metrics.epoch
        
        # Add individual metrics to buffer
        timestamp = time.time()
        
        # Training metrics
        self.metrics_buffer.add_metric(MetricSnapshot(
            timestamp=timestamp,
            epoch=epoch_metrics.epoch,
            batch=0,
            metric_name="loss",
            metric_value=epoch_metrics.train_loss,
            phase="train"
        ))
        
        for metric_name, value in epoch_metrics.train_metrics.items():
            self.metrics_buffer.add_metric(MetricSnapshot(
                timestamp=timestamp,
                epoch=epoch_metrics.epoch,
                batch=0,
                metric_name=metric_name,
                metric_value=value,
                phase="train"
            ))
        
        # Validation metrics
        if epoch_metrics.val_loss is not None:
            self.metrics_buffer.add_metric(MetricSnapshot(
                timestamp=timestamp,
                epoch=epoch_metrics.epoch,
                batch=0,
                metric_name="loss",
                metric_value=epoch_metrics.val_loss,
                phase="validation"
            ))
        
        for metric_name, value in epoch_metrics.val_metrics.items():
            self.metrics_buffer.add_metric(MetricSnapshot(
                timestamp=timestamp,
                epoch=epoch_metrics.epoch,
                batch=0,
                metric_name=metric_name,
                metric_value=value,
                phase="validation"
            ))
        
        # Check early stopping
        should_stop = False
        if self.early_stopping and epoch_metrics.val_loss is not None:
            should_stop = self.early_stopping(epoch_metrics.val_loss)
        
        # Update learning rate
        new_lr = epoch_metrics.learning_rate
        if self.lr_scheduler:
            new_lr = self.lr_scheduler.step(epoch_metrics.val_loss, epoch_metrics.epoch)
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback({
                    'job_id': self.job_id,
                    'epoch': epoch_metrics.epoch,
                    'metrics': epoch_metrics,
                    'should_stop': should_stop,
                    'new_lr': new_lr
                })
            except Exception as e:
                self.logger.error(f"Error in callback: {e}")
    
    def log_batch_metrics(self, batch: int, loss: float, metrics: Dict[str, float]):
        """Log metrics for a training batch"""
        self.current_batch = batch
        timestamp = time.time()
        
        # Log batch loss
        self.metrics_buffer.add_metric(MetricSnapshot(
            timestamp=timestamp,
            epoch=self.current_epoch,
            batch=batch,
            metric_name="loss",
            metric_value=loss,
            phase="train"
        ))
        
        # Log batch metrics
        for metric_name, value in metrics.items():
            self.metrics_buffer.add_metric(MetricSnapshot(
                timestamp=timestamp,
                epoch=self.current_epoch,
                batch=batch,
                metric_name=metric_name,
                metric_value=value,
                phase="train"
            ))
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current training status"""
        elapsed_time = time.time() - self.start_time
        
        # Get latest metrics
        latest_loss = self.metrics_buffer.get_latest_metric("loss", "train")
        latest_val_loss = self.metrics_buffer.get_latest_metric("loss", "validation")
        
        return {
            'job_id': self.job_id,
            'is_monitoring': self.is_monitoring,
            'current_epoch': self.current_epoch,
            'current_batch': self.current_batch,
            'elapsed_time': elapsed_time,
            'latest_train_loss': latest_loss.metric_value if latest_loss else None,
            'latest_val_loss': latest_val_loss.metric_value if latest_val_loss else None,
            'early_stopping_counter': self.early_stopping.counter if self.early_stopping else 0,
            'should_stop': self.early_stopping.should_stop if self.early_stopping else False,
            'current_lr': self.lr_scheduler.get_lr() if self.lr_scheduler else None
        }
    
    def get_metrics_history(self, metric_name: str = None, phase: str = None, 
                           last_n: int = None) -> List[MetricSnapshot]:
        """Get historical metrics"""
        return self.metrics_buffer.get_metrics(
            job_id=self.job_id,
            metric_name=metric_name,
            phase=phase,
            last_n=last_n
        )
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        return self.visualizer.generate_training_report(self.metrics_history)
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                current_status = self.get_current_status()
                
                # Log system status periodically
                if self.current_epoch > 0 and self.current_epoch % 10 == 0:
                    self.logger.info(f"Training progress - Epoch: {self.current_epoch}, "
                                   f"Loss: {current_status.get('latest_train_loss', 'N/A')}")
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)


class TrainingMonitorManager:
    """Manages multiple training monitors"""
    
    def __init__(self):
        self.monitors: Dict[str, TrainingMonitor] = {}
        self.logger = get_logger(__name__)
    
    def create_monitor(self, job_id: str) -> TrainingMonitor:
        """Create a new training monitor for a job"""
        if job_id in self.monitors:
            self.logger.warning(f"Monitor for job {job_id} already exists")
            return self.monitors[job_id]
        
        monitor = TrainingMonitor(job_id)
        self.monitors[job_id] = monitor
        self.logger.info(f"Created monitor for job {job_id}")
        return monitor
    
    def get_monitor(self, job_id: str) -> Optional[TrainingMonitor]:
        """Get existing monitor for a job"""
        return self.monitors.get(job_id)
    
    def remove_monitor(self, job_id: str) -> bool:
        """Remove and cleanup monitor for a job"""
        if job_id in self.monitors:
            monitor = self.monitors[job_id]
            monitor.stop_monitoring()
            del self.monitors[job_id]
            self.logger.info(f"Removed monitor for job {job_id}")
            return True
        return False
    
    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active monitors"""
        return {
            job_id: monitor.get_current_status()
            for job_id, monitor in self.monitors.items()
        }
    
    def cleanup_completed_monitors(self):
        """Remove monitors for completed jobs"""
        completed_jobs = []
        for job_id, monitor in self.monitors.items():
            if not monitor.is_monitoring:
                completed_jobs.append(job_id)
        
        for job_id in completed_jobs:
            self.remove_monitor(job_id)


# Global monitor manager instance
monitor_manager = TrainingMonitorManager()


def get_monitor_manager() -> TrainingMonitorManager:
    """Get global monitor manager instance"""
    return monitor_manager