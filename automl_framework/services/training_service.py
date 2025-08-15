"""
Model training service with distributed training support
"""

import os
import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Empty
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp_torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

try:
    import tensorflow as tf
    from tensorflow.distribute import MirroredStrategy, MultiWorkerMirroredStrategy
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..core.interfaces import (
    IModelTrainer, Architecture, TrainingConfig, ProcessedData, 
    TrainedModel, TrainingJob, PerformanceMetrics
)
from ..core.config import get_config
from ..core.exceptions import TrainingError, ResourceError
from ..utils.logging import get_logger


@dataclass
class GPUInfo:
    """GPU information and status"""
    id: int
    name: str
    memory_total: int
    memory_free: int
    memory_used: int
    utilization: float
    temperature: int
    is_available: bool


@dataclass
class TrainingProgress:
    """Training progress information"""
    epoch: int
    total_epochs: int
    batch: int
    total_batches: int
    train_loss: float
    val_loss: Optional[float]
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    learning_rate: float
    elapsed_time: float
    estimated_remaining_time: float


class GPUManager:
    """Manages GPU allocation and monitoring"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self._gpu_lock = threading.Lock()
        self._allocated_gpus: Dict[str, List[int]] = {}
        
    def get_available_gpus(self) -> List[GPUInfo]:
        """Get list of available GPUs with their status"""
        gpus = []
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    # Get GPU properties
                    props = torch.cuda.get_device_properties(i)
                    
                    # Get memory info
                    torch.cuda.set_device(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory
                    memory_free = torch.cuda.memory_reserved(i)
                    memory_used = memory_total - memory_free
                    
                    # Get utilization (simplified - in production use nvidia-ml-py)
                    utilization = (memory_used / memory_total) * 100
                    
                    gpu_info = GPUInfo(
                        id=i,
                        name=props.name,
                        memory_total=memory_total,
                        memory_free=memory_free,
                        memory_used=memory_used,
                        utilization=utilization,
                        temperature=0,  # Would need nvidia-ml-py for real temperature
                        is_available=memory_free > (memory_total * 0.1)  # 10% free threshold
                    )
                    gpus.append(gpu_info)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get info for GPU {i}: {e}")
                    
        return gpus
    
    def allocate_gpus(self, job_id: str, num_gpus: int) -> List[int]:
        """Allocate GPUs for a training job"""
        with self._gpu_lock:
            available_gpus = self.get_available_gpus()
            free_gpus = [gpu.id for gpu in available_gpus if gpu.is_available]
            
            # Check if we have enough GPUs
            if len(free_gpus) < num_gpus:
                raise ResourceError(
                    f"Not enough GPUs available. Requested: {num_gpus}, Available: {len(free_gpus)}"
                )
            
            # Allocate the requested number of GPUs
            allocated = free_gpus[:num_gpus]
            self._allocated_gpus[job_id] = allocated
            
            self.logger.info(f"Allocated GPUs {allocated} to job {job_id}")
            return allocated
    
    def release_gpus(self, job_id: str) -> bool:
        """Release GPUs allocated to a job"""
        with self._gpu_lock:
            if job_id in self._allocated_gpus:
                gpus = self._allocated_gpus.pop(job_id)
                self.logger.info(f"Released GPUs {gpus} from job {job_id}")
                return True
            return False
    
    def get_allocated_gpus(self, job_id: str) -> List[int]:
        """Get GPUs allocated to a specific job"""
        return self._allocated_gpus.get(job_id, [])


class DistributedTrainer:
    """Distributed training coordinator for multi-GPU training"""
    
    def __init__(self, backend: str = "pytorch"):
        self.backend = backend.lower()
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.gpu_manager = GPUManager()
        
        if self.backend == "pytorch":
            self._init_pytorch()
        elif self.backend == "tensorflow" and TF_AVAILABLE:
            self._init_tensorflow()
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def _init_pytorch(self):
        """Initialize PyTorch distributed training"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU training")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
    
    def _init_tensorflow(self):
        """Initialize TensorFlow distributed training"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                self.logger.error(f"GPU configuration error: {e}")
    
    def setup_distributed_pytorch(self, rank: int, world_size: int, gpu_ids: List[int]):
        """Setup PyTorch distributed training"""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Initialize the process group
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            rank=rank,
            world_size=world_size
        )
        
        # Set the GPU for this process
        if torch.cuda.is_available() and rank < len(gpu_ids):
            torch.cuda.set_device(gpu_ids[rank])
    
    def train_pytorch_distributed(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        config: TrainingConfig,
        gpu_ids: List[int],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Train PyTorch model with distributed training"""
        world_size = len(gpu_ids)
        
        if world_size == 1:
            # Single GPU training
            return self._train_pytorch_single(
                model, train_loader, val_loader, config, gpu_ids[0], progress_callback
            )
        else:
            # Multi-GPU distributed training
            return self._train_pytorch_multi(
                model, train_loader, val_loader, config, gpu_ids, progress_callback
            )
    
    def _train_pytorch_single(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        gpu_id: int,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Single GPU PyTorch training"""
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Setup optimizer
        optimizer = self._get_pytorch_optimizer(model, config)
        criterion = self._get_pytorch_criterion(config)
        scheduler = self._get_pytorch_scheduler(optimizer, config)
        
        # Training loop
        training_history = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.epochs):
            start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_metrics = {}
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate metrics
                batch_metrics = self._calculate_metrics(output, target)
                for key, value in batch_metrics.items():
                    train_metrics[key] = train_metrics.get(key, 0) + value
            
            # Average metrics
            train_loss /= len(train_loader)
            for key in train_metrics:
                train_metrics[key] /= len(train_loader)
            
            # Validation phase
            val_loss, val_metrics = self._validate_pytorch(model, val_loader, criterion, device)
            
            # Learning rate scheduling
            if scheduler:
                scheduler.step(val_loss)
            
            # Progress tracking
            epoch_time = time.time() - start_time
            progress = TrainingProgress(
                epoch=epoch + 1,
                total_epochs=config.epochs,
                batch=len(train_loader),
                total_batches=len(train_loader),
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=optimizer.param_groups[0]['lr'],
                elapsed_time=epoch_time,
                estimated_remaining_time=(config.epochs - epoch - 1) * epoch_time
            )
            
            training_history.append(asdict(progress))
            
            if progress_callback:
                progress_callback(progress)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= config.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        return {
            'training_history': training_history,
            'best_val_loss': best_val_loss,
            'final_metrics': val_metrics
        }
    
    def _train_pytorch_multi(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        gpu_ids: List[int],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Multi-GPU distributed PyTorch training"""
        world_size = len(gpu_ids)
        
        # Use multiprocessing to spawn training processes
        mp_torch.spawn(
            self._train_pytorch_worker,
            args=(world_size, model, train_loader, val_loader, config, gpu_ids, progress_callback),
            nprocs=world_size,
            join=True
        )
        
        # Return results (simplified - in production, you'd collect results from workers)
        return {'status': 'completed', 'world_size': world_size}
    
    def _train_pytorch_worker(
        self,
        rank: int,
        world_size: int,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        gpu_ids: List[int],
        progress_callback: Optional[Callable] = None
    ):
        """Worker process for distributed PyTorch training"""
        self.setup_distributed_pytorch(rank, world_size, gpu_ids)
        
        device = torch.device(f"cuda:{gpu_ids[rank]}")
        model = model.to(device)
        model = DDP(model, device_ids=[gpu_ids[rank]])
        
        # Setup distributed sampler
        train_sampler = DistributedSampler(train_loader.dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # Training logic similar to single GPU but with distributed considerations
        optimizer = self._get_pytorch_optimizer(model, config)
        criterion = self._get_pytorch_criterion(config)
        
        for epoch in range(config.epochs):
            train_sampler.set_epoch(epoch)
            
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Cleanup
        dist.destroy_process_group()
    
    def train_tensorflow_distributed(
        self,
        model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        config: TrainingConfig,
        gpu_ids: List[int],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Train TensorFlow model with distributed training"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        # Setup distributed strategy
        if len(gpu_ids) > 1:
            strategy = MirroredStrategy(devices=[f"/gpu:{i}" for i in gpu_ids])
        else:
            strategy = tf.distribute.get_strategy()  # Default strategy
        
        with strategy.scope():
            # Create and compile model within strategy scope
            optimizer = self._get_tensorflow_optimizer(config)
            loss_fn = self._get_tensorflow_loss(config)
            metrics = self._get_tensorflow_metrics(config)
            
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
            
            # Setup callbacks
            callbacks = self._get_tensorflow_callbacks(config, progress_callback)
            
            # Train the model
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=config.epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            return {
                'training_history': history.history,
                'final_metrics': history.history
            }
    
    def _get_pytorch_optimizer(self, model: nn.Module, config: TrainingConfig):
        """Get PyTorch optimizer"""
        if config.optimizer.lower() == 'adam':
            return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        elif config.optimizer.lower() == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
        elif config.optimizer.lower() == 'adamw':
            return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        else:
            return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    def _get_pytorch_criterion(self, config: TrainingConfig):
        """Get PyTorch loss criterion"""
        # This would be determined by the task type
        return nn.CrossEntropyLoss()
    
    def _get_pytorch_scheduler(self, optimizer, config: TrainingConfig):
        """Get PyTorch learning rate scheduler"""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    
    def _get_tensorflow_optimizer(self, config: TrainingConfig):
        """Get TensorFlow optimizer"""
        if config.optimizer.lower() == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        elif config.optimizer.lower() == 'sgd':
            return tf.keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=0.9)
        elif config.optimizer.lower() == 'adamw':
            return tf.keras.optimizers.AdamW(learning_rate=config.learning_rate)
        else:
            return tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    
    def _get_tensorflow_loss(self, config: TrainingConfig):
        """Get TensorFlow loss function"""
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    def _get_tensorflow_metrics(self, config: TrainingConfig):
        """Get TensorFlow metrics"""
        return ['accuracy', 'precision', 'recall']
    
    def _get_tensorflow_callbacks(self, config: TrainingConfig, progress_callback: Optional[Callable]):
        """Get TensorFlow callbacks"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.early_stopping_patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        if progress_callback:
            # Custom callback for progress reporting
            class ProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = TrainingProgress(
                        epoch=epoch + 1,
                        total_epochs=config.epochs,
                        batch=0,
                        total_batches=0,
                        train_loss=logs.get('loss', 0),
                        val_loss=logs.get('val_loss', 0),
                        train_metrics={'accuracy': logs.get('accuracy', 0)},
                        val_metrics={'val_accuracy': logs.get('val_accuracy', 0)},
                        learning_rate=self.model.optimizer.learning_rate.numpy(),
                        elapsed_time=0,
                        estimated_remaining_time=0
                    )
                    progress_callback(progress)
            
            callbacks.append(ProgressCallback())
        
        return callbacks
    
    def _validate_pytorch(self, model: nn.Module, val_loader: DataLoader, criterion, device):
        """Validate PyTorch model"""
        model.eval()
        val_loss = 0.0
        val_metrics = {}
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                # Calculate metrics
                batch_metrics = self._calculate_metrics(output, target)
                for key, value in batch_metrics.items():
                    val_metrics[key] = val_metrics.get(key, 0) + value
        
        val_loss /= len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
        
        return val_loss, val_metrics
    
    def _calculate_metrics(self, output: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Calculate training metrics"""
        with torch.no_grad():
            pred = output.argmax(dim=1)
            correct = pred.eq(target).sum().item()
            total = target.size(0)
            accuracy = correct / total
            
            return {'accuracy': accuracy}


class TrainingJobManager:
    """Manages training job lifecycle and scheduling"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.gpu_manager = GPUManager()
        self.job_queue = Queue()
        self.active_jobs: Dict[str, TrainingJob] = {}
        self.job_results: Dict[str, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=self.config.resources.max_concurrent_experiments)
        self._shutdown = False
        
        # Start job scheduler thread
        self.scheduler_thread = threading.Thread(target=self._job_scheduler, daemon=True)
        self.scheduler_thread.start()
    
    def submit_job(self, job: TrainingJob) -> str:
        """Submit a training job to the queue"""
        self.job_queue.put(job)
        self.logger.info(f"Submitted training job {job.id} to queue")
        return job.id
    
    def get_job_status(self, job_id: str) -> Optional[str]:
        """Get status of a training job"""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id].status
        elif job_id in self.job_results:
            return "completed"
        else:
            return None
    
    def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get results of a completed training job"""
        return self.job_results.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job"""
        if job_id in self.active_jobs:
            self.active_jobs[job_id].status = "cancelled"
            self.gpu_manager.release_gpus(job_id)
            return True
        return False
    
    def _job_scheduler(self):
        """Background job scheduler"""
        while not self._shutdown:
            try:
                # Get next job from queue (with timeout)
                job = self.job_queue.get(timeout=1.0)
                
                # Check if we can run this job
                if len(self.active_jobs) >= self.config.resources.max_concurrent_experiments:
                    # Put job back in queue and wait
                    self.job_queue.put(job)
                    time.sleep(5)
                    continue
                
                # Try to allocate GPUs
                try:
                    num_gpus = min(len(job.gpu_allocation), self.config.resources.max_gpu_per_experiment)
                    if num_gpus == 0:
                        num_gpus = 1  # Default to 1 GPU
                    
                    gpu_ids = self.gpu_manager.allocate_gpus(job.id, num_gpus)
                    job.gpu_allocation = gpu_ids
                    job.status = "running"
                    
                    # Add to active jobs
                    self.active_jobs[job.id] = job
                    
                    # Submit to executor
                    future = self.executor.submit(self._execute_job, job)
                    future.add_done_callback(lambda f, job_id=job.id: self._job_completed(job_id, f))
                    
                except ResourceError as e:
                    self.logger.warning(f"Cannot allocate resources for job {job.id}: {e}")
                    # Put job back in queue
                    self.job_queue.put(job)
                    time.sleep(10)  # Wait before retrying
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in job scheduler: {e}")
    
    def _execute_job(self, job: TrainingJob) -> Dict[str, Any]:
        """Execute a training job"""
        try:
            self.logger.info(f"Starting execution of job {job.id}")
            
            # This is a placeholder - actual implementation would:
            # 1. Load the model architecture
            # 2. Load the dataset
            # 3. Create the trainer
            # 4. Run training
            # 5. Return results
            
            # Simulate training time
            time.sleep(10)
            
            return {
                'job_id': job.id,
                'status': 'completed',
                'metrics': {
                    'accuracy': 0.95,
                    'loss': 0.1,
                    'training_time': 600
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error executing job {job.id}: {e}")
            return {
                'job_id': job.id,
                'status': 'failed',
                'error': str(e)
            }
    
    def _job_completed(self, job_id: str, future: Future):
        """Handle job completion"""
        try:
            result = future.result()
            self.job_results[job_id] = result
            
            # Remove from active jobs
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            # Release GPUs
            self.gpu_manager.release_gpus(job_id)
            
            self.logger.info(f"Job {job_id} completed with status: {result.get('status')}")
            
        except Exception as e:
            self.logger.error(f"Error handling job completion for {job_id}: {e}")
    
    def shutdown(self):
        """Shutdown the job manager"""
        self._shutdown = True
        self.executor.shutdown(wait=True)


class ModelTrainingService(IModelTrainer):
    """Main model training service implementation"""
    
    def __init__(self, backend: str = "pytorch"):
        self.backend = backend
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.distributed_trainer = DistributedTrainer(backend)
        self.job_manager = TrainingJobManager()
        self.progress_callbacks: Dict[str, Callable] = {}
    
    def train_model(
        self, 
        architecture: Architecture, 
        config: TrainingConfig, 
        processed_data: ProcessedData
    ) -> TrainedModel:
        """Train model with given architecture and configuration"""
        try:
            # Create training job
            job = TrainingJob(
                id=f"job_{int(time.time())}",
                experiment_id="",  # Would be set by experiment manager
                architecture=architecture,
                config=config,
                status="queued",
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                gpu_allocation=[]
            )
            
            # Submit job
            job_id = self.job_manager.submit_job(job)
            
            # Wait for completion (simplified - in production, this would be async)
            while True:
                status = self.job_manager.get_job_status(job_id)
                if status in ["completed", "failed", "cancelled"]:
                    break
                time.sleep(1)
            
            # Get results
            results = self.job_manager.get_job_results(job_id)
            if not results or results.get('status') != 'completed':
                raise TrainingError(f"Training failed: {results.get('error', 'Unknown error')}")
            
            # Create trained model
            metrics = PerformanceMetrics(
                accuracy=results['metrics']['accuracy'],
                loss=results['metrics']['loss'],
                precision=0.0,  # Would be calculated
                recall=0.0,     # Would be calculated
                f1_score=0.0,   # Would be calculated
                training_time=results['metrics']['training_time'],
                inference_time=0.0,  # Would be measured
                additional_metrics={}
            )
            
            trained_model = TrainedModel(
                id=f"model_{job_id}",
                architecture=architecture,
                config=config,
                metrics=metrics,
                model_path=f"{self.config.model_storage_path}/model_{job_id}",
                preprocessing_pipeline_path=f"{self.config.model_storage_path}/pipeline_{job_id}"
            )
            
            return trained_model
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise TrainingError(f"Training failed: {e}")
    
    def monitor_training(self, training_job: TrainingJob) -> Dict[str, Any]:
        """Monitor training progress and return current metrics"""
        status = self.job_manager.get_job_status(training_job.id)
        
        if status == "completed":
            return self.job_manager.get_job_results(training_job.id)
        elif status == "running":
            return {
                'status': 'running',
                'job_id': training_job.id,
                'gpu_allocation': training_job.gpu_allocation
            }
        else:
            return {
                'status': status or 'unknown',
                'job_id': training_job.id
            }
    
    def save_checkpoint(self, model: Any, epoch: int, checkpoint_path: str) -> str:
        """Save model checkpoint"""
        try:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            if self.backend == "pytorch":
                if hasattr(model, 'state_dict'):
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'timestamp': time.time()
                    }, checkpoint_path)
                else:
                    torch.save(model, checkpoint_path)
            
            elif self.backend == "tensorflow" and TF_AVAILABLE:
                model.save_weights(checkpoint_path)
            
            self.logger.info(f"Saved checkpoint at epoch {epoch} to {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            raise TrainingError(f"Failed to save checkpoint: {e}")
    
    def register_progress_callback(self, job_id: str, callback: Callable):
        """Register a progress callback for a training job"""
        self.progress_callbacks[job_id] = callback
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training service statistics"""
        gpu_info = self.distributed_trainer.gpu_manager.get_available_gpus()
        
        return {
            'active_jobs': len(self.job_manager.active_jobs),
            'queued_jobs': self.job_manager.job_queue.qsize(),
            'completed_jobs': len(self.job_manager.job_results),
            'available_gpus': len([gpu for gpu in gpu_info if gpu.is_available]),
            'total_gpus': len(gpu_info),
            'gpu_utilization': sum(gpu.utilization for gpu in gpu_info) / len(gpu_info) if gpu_info else 0
        }