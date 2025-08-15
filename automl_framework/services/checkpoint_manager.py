"""
Checkpoint management service for model training
"""

import os
import json
import time
import shutil
import pickle
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

import torch
import torch.nn as nn

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..core.interfaces import TrainingJob, Architecture, TrainingConfig
from ..core.config import get_config
from ..core.exceptions import CheckpointError
from ..utils.logging import get_logger


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint"""
    checkpoint_id: str
    job_id: str
    epoch: int
    step: int
    timestamp: float
    model_state_size: int
    optimizer_state_size: int
    metrics: Dict[str, float]
    architecture_hash: str
    config_hash: str
    framework: str  # 'pytorch' or 'tensorflow'
    file_path: str
    is_best: bool = False
    tags: List[str] = None


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management"""
    save_frequency: int = 1  # Save every N epochs
    max_checkpoints: int = 10  # Maximum number of checkpoints to keep
    save_best_only: bool = False  # Only save checkpoints that improve metrics
    monitor_metric: str = "val_loss"  # Metric to monitor for best checkpoint
    mode: str = "min"  # 'min' or 'max' for metric comparison
    save_optimizer_state: bool = True  # Whether to save optimizer state
    compression: bool = True  # Whether to compress checkpoint files
    backup_to_cloud: bool = False  # Whether to backup to cloud storage
    cleanup_on_completion: bool = False  # Whether to cleanup old checkpoints when training completes


class CheckpointManager:
    """Manages model checkpoints during training"""
    
    def __init__(self, job_id: str, config: CheckpointConfig = None):
        self.job_id = job_id
        self.config = config or CheckpointConfig()
        self.logger = get_logger(__name__)
        self.system_config = get_config()
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(self.system_config.checkpoint_storage_path) / job_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.checkpoints: Dict[str, CheckpointMetadata] = {}
        self.best_checkpoint: Optional[CheckpointMetadata] = None
        self.best_metric_value: Optional[float] = None
        self.last_save_time = 0
        self.lock = threading.Lock()
        
        # Load existing checkpoints
        self._load_existing_checkpoints()
    
    def save_checkpoint(
        self,
        model: Union[nn.Module, Any],
        optimizer: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Dict[str, float] = None,
        architecture: Optional[Architecture] = None,
        config: Optional[TrainingConfig] = None,
        tags: List[str] = None,
        force_save: bool = False
    ) -> str:
        """
        Save a model checkpoint
        
        Args:
            model: The model to save
            optimizer: The optimizer state to save (optional)
            epoch: Current epoch number
            step: Current step number
            metrics: Current metrics
            architecture: Model architecture (optional)
            config: Training configuration (optional)
            tags: Tags for the checkpoint (optional)
            force_save: Force save even if conditions not met
            
        Returns:
            Checkpoint ID
        """
        with self.lock:
            try:
                # Check if we should save this checkpoint
                if not force_save and not self._should_save_checkpoint(epoch, metrics):
                    return None
                
                # Generate checkpoint ID
                checkpoint_id = self._generate_checkpoint_id(epoch, step)
                
                # Determine framework
                framework = self._detect_framework(model)
                
                # Create checkpoint file path
                checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.ckpt"
                
                # Save the checkpoint
                if framework == "pytorch":
                    self._save_pytorch_checkpoint(
                        checkpoint_path, model, optimizer, epoch, step, metrics, architecture, config
                    )
                elif framework == "tensorflow":
                    self._save_tensorflow_checkpoint(
                        checkpoint_path, model, optimizer, epoch, step, metrics, architecture, config
                    )
                else:
                    raise CheckpointError(f"Unsupported framework: {framework}")
                
                # Create metadata
                metadata = CheckpointMetadata(
                    checkpoint_id=checkpoint_id,
                    job_id=self.job_id,
                    epoch=epoch,
                    step=step,
                    timestamp=time.time(),
                    model_state_size=self._get_file_size(checkpoint_path),
                    optimizer_state_size=0,  # Would be calculated separately
                    metrics=metrics or {},
                    architecture_hash=self._hash_architecture(architecture) if architecture else "",
                    config_hash=self._hash_config(config) if config else "",
                    framework=framework,
                    file_path=str(checkpoint_path),
                    is_best=False,
                    tags=tags or []
                )
                
                # Check if this is the best checkpoint
                if self._is_best_checkpoint(metrics):
                    metadata.is_best = True
                    self.best_checkpoint = metadata
                    self.best_metric_value = metrics.get(self.config.monitor_metric) if metrics else None
                
                # Store metadata
                self.checkpoints[checkpoint_id] = metadata
                self._save_metadata()
                
                # Cleanup old checkpoints if necessary
                self._cleanup_old_checkpoints()
                
                self.logger.info(f"Saved checkpoint {checkpoint_id} for job {self.job_id}")
                return checkpoint_id
                
            except Exception as e:
                self.logger.error(f"Error saving checkpoint: {e}")
                raise CheckpointError(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(
        self,
        checkpoint_id: str = None,
        load_best: bool = False,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None
    ) -> Tuple[Any, Optional[Any], Dict[str, Any]]:
        """
        Load a checkpoint
        
        Args:
            checkpoint_id: Specific checkpoint ID to load (optional)
            load_best: Whether to load the best checkpoint
            model: Model to load state into (optional)
            optimizer: Optimizer to load state into (optional)
            
        Returns:
            Tuple of (model, optimizer, metadata)
        """
        with self.lock:
            try:
                # Determine which checkpoint to load
                if load_best and self.best_checkpoint:
                    metadata = self.best_checkpoint
                elif checkpoint_id and checkpoint_id in self.checkpoints:
                    metadata = self.checkpoints[checkpoint_id]
                elif self.checkpoints:
                    # Load the latest checkpoint
                    metadata = max(self.checkpoints.values(), key=lambda x: x.timestamp)
                else:
                    raise CheckpointError("No checkpoints available to load")
                
                checkpoint_path = Path(metadata.file_path)
                if not checkpoint_path.exists():
                    raise CheckpointError(f"Checkpoint file not found: {checkpoint_path}")
                
                # Load based on framework
                if metadata.framework == "pytorch":
                    loaded_model, loaded_optimizer, checkpoint_data = self._load_pytorch_checkpoint(
                        checkpoint_path, model, optimizer
                    )
                elif metadata.framework == "tensorflow":
                    loaded_model, loaded_optimizer, checkpoint_data = self._load_tensorflow_checkpoint(
                        checkpoint_path, model, optimizer
                    )
                else:
                    raise CheckpointError(f"Unsupported framework: {metadata.framework}")
                
                self.logger.info(f"Loaded checkpoint {metadata.checkpoint_id} for job {self.job_id}")
                
                return loaded_model, loaded_optimizer, {
                    'metadata': asdict(metadata),
                    'checkpoint_data': checkpoint_data
                }
                
            except Exception as e:
                self.logger.error(f"Error loading checkpoint: {e}")
                raise CheckpointError(f"Failed to load checkpoint: {e}")
    
    def list_checkpoints(self, include_metadata: bool = True) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        with self.lock:
            if include_metadata:
                return [asdict(metadata) for metadata in self.checkpoints.values()]
            else:
                return [
                    {
                        'checkpoint_id': metadata.checkpoint_id,
                        'epoch': metadata.epoch,
                        'timestamp': metadata.timestamp,
                        'is_best': metadata.is_best,
                        'metrics': metadata.metrics
                    }
                    for metadata in self.checkpoints.values()
                ]
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint"""
        with self.lock:
            try:
                if checkpoint_id not in self.checkpoints:
                    return False
                
                metadata = self.checkpoints[checkpoint_id]
                checkpoint_path = Path(metadata.file_path)
                
                # Delete the file
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                
                # Remove from tracking
                del self.checkpoints[checkpoint_id]
                
                # Update best checkpoint if necessary
                if self.best_checkpoint and self.best_checkpoint.checkpoint_id == checkpoint_id:
                    self._update_best_checkpoint()
                
                self._save_metadata()
                
                self.logger.info(f"Deleted checkpoint {checkpoint_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error deleting checkpoint {checkpoint_id}: {e}")
                return False
    
    def cleanup_all_checkpoints(self) -> int:
        """Delete all checkpoints for this job"""
        with self.lock:
            deleted_count = 0
            
            for checkpoint_id in list(self.checkpoints.keys()):
                if self.delete_checkpoint(checkpoint_id):
                    deleted_count += 1
            
            # Remove the checkpoint directory if empty
            try:
                if self.checkpoint_dir.exists() and not any(self.checkpoint_dir.iterdir()):
                    self.checkpoint_dir.rmdir()
            except Exception as e:
                self.logger.warning(f"Could not remove checkpoint directory: {e}")
            
            self.logger.info(f"Cleaned up {deleted_count} checkpoints for job {self.job_id}")
            return deleted_count
    
    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a checkpoint"""
        with self.lock:
            if checkpoint_id not in self.checkpoints:
                return None
            
            metadata = self.checkpoints[checkpoint_id]
            checkpoint_path = Path(metadata.file_path)
            
            info = asdict(metadata)
            info.update({
                'file_exists': checkpoint_path.exists(),
                'file_size_mb': metadata.model_state_size / (1024 * 1024),
                'age_hours': (time.time() - metadata.timestamp) / 3600
            })
            
            return info
    
    def _should_save_checkpoint(self, epoch: int, metrics: Dict[str, float] = None) -> bool:
        """Determine if a checkpoint should be saved"""
        # Check frequency
        if epoch % self.config.save_frequency != 0:
            return False
        
        # Check if save_best_only is enabled
        if self.config.save_best_only and metrics:
            return self._is_best_checkpoint(metrics)
        
        return True
    
    def _is_best_checkpoint(self, metrics: Dict[str, float] = None) -> bool:
        """Check if current metrics represent the best checkpoint"""
        if not metrics or self.config.monitor_metric not in metrics:
            return False
        
        current_value = metrics[self.config.monitor_metric]
        
        if self.best_metric_value is None:
            return True
        
        if self.config.mode == "min":
            return current_value < self.best_metric_value
        else:
            return current_value > self.best_metric_value
    
    def _generate_checkpoint_id(self, epoch: int, step: int) -> str:
        """Generate a unique checkpoint ID"""
        timestamp = int(time.time())
        return f"epoch_{epoch:04d}_step_{step:06d}_{timestamp}"
    
    def _detect_framework(self, model: Any) -> str:
        """Detect the ML framework being used"""
        if isinstance(model, nn.Module):
            return "pytorch"
        elif TF_AVAILABLE and isinstance(model, tf.keras.Model):
            return "tensorflow"
        else:
            # Try to infer from model attributes
            if hasattr(model, 'state_dict'):
                return "pytorch"
            elif hasattr(model, 'get_weights'):
                return "tensorflow"
            else:
                raise CheckpointError("Could not detect model framework")
    
    def _save_pytorch_checkpoint(
        self,
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: Optional[Any],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        architecture: Optional[Architecture],
        config: Optional[TrainingConfig]
    ):
        """Save PyTorch checkpoint"""
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'metrics': metrics or {},
            'timestamp': time.time(),
            'job_id': self.job_id
        }
        
        if optimizer and self.config.save_optimizer_state:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        if architecture:
            checkpoint_data['architecture'] = asdict(architecture)
        
        if config:
            checkpoint_data['training_config'] = asdict(config)
        
        torch.save(checkpoint_data, checkpoint_path)
    
    def _save_tensorflow_checkpoint(
        self,
        checkpoint_path: Path,
        model: Any,
        optimizer: Optional[Any],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        architecture: Optional[Architecture],
        config: Optional[TrainingConfig]
    ):
        """Save TensorFlow checkpoint"""
        if not TF_AVAILABLE:
            raise CheckpointError("TensorFlow not available")
        
        # Save model weights
        model.save_weights(str(checkpoint_path))
        
        # Save additional metadata
        metadata_path = checkpoint_path.with_suffix('.json')
        metadata = {
            'epoch': epoch,
            'step': step,
            'metrics': metrics or {},
            'timestamp': time.time(),
            'job_id': self.job_id
        }
        
        if architecture:
            metadata['architecture'] = asdict(architecture)
        
        if config:
            metadata['training_config'] = asdict(config)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_pytorch_checkpoint(
        self,
        checkpoint_path: Path,
        model: Optional[nn.Module],
        optimizer: Optional[Any]
    ) -> Tuple[Any, Optional[Any], Dict[str, Any]]:
        """Load PyTorch checkpoint"""
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        if model is not None:
            model.load_state_dict(checkpoint_data['model_state_dict'])
        
        loaded_optimizer = None
        if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            loaded_optimizer = optimizer
        
        return model, loaded_optimizer, checkpoint_data
    
    def _load_tensorflow_checkpoint(
        self,
        checkpoint_path: Path,
        model: Optional[Any],
        optimizer: Optional[Any]
    ) -> Tuple[Any, Optional[Any], Dict[str, Any]]:
        """Load TensorFlow checkpoint"""
        if not TF_AVAILABLE:
            raise CheckpointError("TensorFlow not available")
        
        if model is not None:
            model.load_weights(str(checkpoint_path))
        
        # Load metadata
        metadata_path = checkpoint_path.with_suffix('.json')
        checkpoint_data = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                checkpoint_data = json.load(f)
        
        return model, optimizer, checkpoint_data
    
    def _hash_architecture(self, architecture: Architecture) -> str:
        """Generate hash for architecture"""
        arch_str = json.dumps(asdict(architecture), sort_keys=True)
        return hashlib.md5(arch_str.encode()).hexdigest()
    
    def _hash_config(self, config: TrainingConfig) -> str:
        """Generate hash for training config"""
        config_str = json.dumps(asdict(config), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes"""
        try:
            return file_path.stat().st_size
        except Exception:
            return 0
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if we exceed the maximum"""
        if len(self.checkpoints) <= self.config.max_checkpoints:
            return
        
        # Sort checkpoints by timestamp (oldest first)
        sorted_checkpoints = sorted(
            self.checkpoints.values(),
            key=lambda x: x.timestamp
        )
        
        # Keep the best checkpoint and recent ones
        checkpoints_to_delete = []
        for metadata in sorted_checkpoints[:-self.config.max_checkpoints]:
            if not metadata.is_best:  # Don't delete the best checkpoint
                checkpoints_to_delete.append(metadata.checkpoint_id)
        
        # Delete old checkpoints
        for checkpoint_id in checkpoints_to_delete:
            self.delete_checkpoint(checkpoint_id)
    
    def _update_best_checkpoint(self):
        """Update the best checkpoint after deletion"""
        self.best_checkpoint = None
        self.best_metric_value = None
        
        for metadata in self.checkpoints.values():
            if self.config.monitor_metric in metadata.metrics:
                metric_value = metadata.metrics[self.config.monitor_metric]
                
                if (self.best_metric_value is None or
                    (self.config.mode == "min" and metric_value < self.best_metric_value) or
                    (self.config.mode == "max" and metric_value > self.best_metric_value)):
                    
                    self.best_checkpoint = metadata
                    self.best_metric_value = metric_value
                    metadata.is_best = True
                else:
                    metadata.is_best = False
    
    def _save_metadata(self):
        """Save checkpoint metadata to disk"""
        metadata_file = self.checkpoint_dir / "checkpoints_metadata.json"
        
        try:
            metadata_dict = {
                checkpoint_id: asdict(metadata)
                for checkpoint_id, metadata in self.checkpoints.items()
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Could not save checkpoint metadata: {e}")
    
    def _load_existing_checkpoints(self):
        """Load existing checkpoint metadata from disk"""
        metadata_file = self.checkpoint_dir / "checkpoints_metadata.json"
        
        if not metadata_file.exists():
            return
        
        try:
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            
            for checkpoint_id, metadata_data in metadata_dict.items():
                metadata = CheckpointMetadata(**metadata_data)
                self.checkpoints[checkpoint_id] = metadata
                
                # Update best checkpoint
                if metadata.is_best:
                    self.best_checkpoint = metadata
                    if self.config.monitor_metric in metadata.metrics:
                        self.best_metric_value = metadata.metrics[self.config.monitor_metric]
            
            self.logger.info(f"Loaded {len(self.checkpoints)} existing checkpoints for job {self.job_id}")
            
        except Exception as e:
            self.logger.warning(f"Could not load existing checkpoint metadata: {e}")


class CheckpointManagerRegistry:
    """Registry for managing multiple checkpoint managers"""
    
    def __init__(self):
        self.managers: Dict[str, CheckpointManager] = {}
        self.lock = threading.Lock()
        self.logger = get_logger(__name__)
    
    def get_manager(self, job_id: str, config: CheckpointConfig = None) -> CheckpointManager:
        """Get or create a checkpoint manager for a job"""
        with self.lock:
            if job_id not in self.managers:
                self.managers[job_id] = CheckpointManager(job_id, config)
                self.logger.info(f"Created checkpoint manager for job {job_id}")
            
            return self.managers[job_id]
    
    def remove_manager(self, job_id: str, cleanup_checkpoints: bool = False) -> bool:
        """Remove a checkpoint manager"""
        with self.lock:
            if job_id not in self.managers:
                return False
            
            manager = self.managers[job_id]
            
            if cleanup_checkpoints:
                manager.cleanup_all_checkpoints()
            
            del self.managers[job_id]
            self.logger.info(f"Removed checkpoint manager for job {job_id}")
            return True
    
    def list_jobs(self) -> List[str]:
        """List all jobs with active checkpoint managers"""
        with self.lock:
            return list(self.managers.keys())
    
    def get_all_checkpoints(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all checkpoints for all jobs"""
        with self.lock:
            return {
                job_id: manager.list_checkpoints(include_metadata=False)
                for job_id, manager in self.managers.items()
            }
    
    def cleanup_completed_jobs(self, active_job_ids: List[str]):
        """Cleanup checkpoint managers for completed jobs"""
        with self.lock:
            completed_jobs = [
                job_id for job_id in self.managers.keys()
                if job_id not in active_job_ids
            ]
            
            for job_id in completed_jobs:
                self.remove_manager(job_id, cleanup_checkpoints=True)
            
            if completed_jobs:
                self.logger.info(f"Cleaned up checkpoint managers for {len(completed_jobs)} completed jobs")


# Global checkpoint manager registry
checkpoint_registry = CheckpointManagerRegistry()


def get_checkpoint_manager(job_id: str, config: CheckpointConfig = None) -> CheckpointManager:
    """Get checkpoint manager for a job"""
    return checkpoint_registry.get_manager(job_id, config)


def cleanup_job_checkpoints(job_id: str) -> bool:
    """Cleanup all checkpoints for a job"""
    return checkpoint_registry.remove_manager(job_id, cleanup_checkpoints=True)