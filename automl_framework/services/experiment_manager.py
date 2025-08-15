"""
Experiment Manager for orchestrating AutoML pipeline execution.

This module provides the ExperimentManager class that coordinates the entire
AutoML pipeline, managing experiment lifecycle, job scheduling, and progress tracking.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, Future
import threading
import time

from automl_framework.core.interfaces import (
    IExperimentManager,
    IDataProcessor,
    INASService,
    IHyperparameterOptimizer,
    IModelTrainer,
    IModelEvaluator,
    IResourceScheduler
)
from automl_framework.models.data_models import (
    Experiment,
    ExperimentStatus,
    ExperimentResults,
    Dataset,
    Architecture,
    TrainingConfig,
    PerformanceMetrics,
    DataType,
    TaskType
)
from automl_framework.core.database import get_postgres_session
# from automl_framework.core.database import get_mongo_collection
from automl_framework.core.registry import get_service_registry
from automl_framework.core.exceptions import (
    ExperimentError,
    ResourceError,
    ValidationError
)

logger = logging.getLogger(__name__)


class JobDependency:
    """Represents a dependency between jobs in the pipeline."""
    
    def __init__(self, job_id: str, depends_on: List[str]):
        self.job_id = job_id
        self.depends_on = depends_on
        self.completed = False
        self.result = None
        self.error = None


class PipelineJob:
    """Represents a job in the AutoML pipeline."""
    
    def __init__(self, job_id: str, job_type: str, job_function: Callable, 
                 args: tuple = (), kwargs: dict = None):
        self.job_id = job_id
        self.job_type = job_type
        self.job_function = job_function
        self.args = args
        self.kwargs = kwargs or {}
        self.status = "pending"
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.progress = 0.0


class ExperimentManager(IExperimentManager):
    """
    Manages AutoML experiment orchestration and lifecycle.
    
    Coordinates the entire AutoML pipeline including data processing,
    architecture search, hyperparameter optimization, training, and evaluation.
    """
    
    def __init__(self, max_concurrent_experiments: int = 5):
        self.max_concurrent_experiments = max_concurrent_experiments
        self.active_experiments: Dict[str, Experiment] = {}
        self.experiment_jobs: Dict[str, List[PipelineJob]] = {}
        self.job_dependencies: Dict[str, JobDependency] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_experiments)
        self.experiment_futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        
        # Progress tracking
        self.experiment_progress: Dict[str, Dict[str, float]] = {}
        self.progress_callbacks: Dict[str, List[Callable]] = {}
        
        # WebSocket event broadcasting
        self._websocket_enabled = False
        self._websocket_manager = None
        
        # In-memory storage for experiments (replace with proper database later)
        self.experiments_storage: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"ExperimentManager initialized with max {max_concurrent_experiments} concurrent experiments")
    
    def create_experiment(self, dataset_path: str, experiment_config: Dict[str, Any]) -> str:
        """
        Create a new AutoML experiment.
        
        Args:
            dataset_path: Path to the dataset file
            experiment_config: Configuration parameters for the experiment
            
        Returns:
            str: Unique experiment ID
            
        Raises:
            ValidationError: If configuration is invalid
            ExperimentError: If experiment creation fails
        """
        try:
            # Generate unique experiment ID
            experiment_id = str(uuid.uuid4())
            
            # Validate experiment configuration
            self._validate_experiment_config(experiment_config)
            
            # Create experiment object
            experiment = Experiment(
                id=experiment_id,
                name=experiment_config.get('name', f'Experiment_{experiment_id[:8]}'),
                dataset_id=experiment_config.get('dataset_id', experiment_id),  # Use experiment_id as dataset_id if not provided
                status=ExperimentStatus.CREATED,
                config=experiment_config,
                created_at=datetime.now()
            )
            
            # Validate experiment
            experiment.validate()
            
            # Store experiment in database
            self._save_experiment_to_db(experiment)
            
            # Initialize progress tracking
            self.experiment_progress[experiment_id] = {
                'data_processing': 0.0,
                'architecture_search': 0.0,
                'hyperparameter_optimization': 0.0,
                'model_training': 0.0,
                'model_evaluation': 0.0,
                'overall': 0.0
            }
            
            logger.info(f"Created experiment {experiment_id} with name '{experiment.name}'")
            
            # Broadcast experiment creation event
            asyncio.create_task(self._broadcast_experiment_event(
                'created', 
                experiment_id, 
                {
                    'id': experiment_id,
                    'name': experiment.name,
                    'status': experiment.status.value,
                    'created_at': experiment.created_at.isoformat()
                }
            ))
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise ExperimentError(f"Experiment creation failed: {e}")
    
    def run_experiment(self, experiment_id: str) -> bool:
        """
        Run an AutoML experiment pipeline.
        
        Args:
            experiment_id: ID of the experiment to run
            
        Returns:
            bool: True if experiment started successfully
            
        Raises:
            ExperimentError: If experiment cannot be started
        """
        try:
            with self._lock:
                if len(self.active_experiments) >= self.max_concurrent_experiments:
                    raise ExperimentError("Maximum concurrent experiments limit reached")
                
                if experiment_id in self.active_experiments:
                    raise ExperimentError(f"Experiment {experiment_id} is already running")
            
            # Load experiment from database
            experiment = self._load_experiment_from_db(experiment_id)
            if not experiment:
                raise ExperimentError(f"Experiment {experiment_id} not found")
            
            if experiment.status != ExperimentStatus.CREATED:
                raise ExperimentError(f"Experiment {experiment_id} cannot be started (status: {experiment.status})")
            
            # Update experiment status
            experiment.status = ExperimentStatus.RUNNING
            self._save_experiment_to_db(experiment)
            
            # Add to active experiments
            with self._lock:
                self.active_experiments[experiment_id] = experiment
            
            # Submit experiment to executor
            future = self.executor.submit(self._run_experiment_pipeline, experiment_id)
            self.experiment_futures[experiment_id] = future
            
            logger.info(f"Started experiment {experiment_id}")
            
            # Broadcast experiment start event
            asyncio.create_task(self._broadcast_experiment_event(
                'started',
                experiment_id,
                {
                    'id': experiment_id,
                    'name': experiment.name,
                    'status': experiment.status.value,
                    'started_at': datetime.now().isoformat()
                }
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start experiment {experiment_id}: {e}")
            raise ExperimentError(f"Failed to start experiment: {e}")
    
    def get_experiment_status(self, experiment_id: str) -> ExperimentStatus:
        """
        Get current experiment status.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            ExperimentStatus: Current status of the experiment
        """
        # Check active experiments first
        if experiment_id in self.active_experiments:
            return self.active_experiments[experiment_id].status
        
        # Load from database
        experiment = self._load_experiment_from_db(experiment_id)
        if experiment:
            return experiment.status
        
        raise ExperimentError(f"Experiment {experiment_id} not found")
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment results and best model.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Dict[str, Any]: Experiment results including best model and metrics
        """
        experiment = self._load_experiment_from_db(experiment_id)
        if not experiment:
            raise ExperimentError(f"Experiment {experiment_id} not found")
        
        if experiment.status != ExperimentStatus.COMPLETED:
            return {
                'experiment_id': experiment_id,
                'status': experiment.status.value,
                'progress': self.get_experiment_progress(experiment_id),
                'results': None
            }
        
        results = {
            'experiment_id': experiment_id,
            'status': experiment.status.value,
            'name': experiment.name,
            'created_at': experiment.created_at.isoformat(),
            'completed_at': experiment.completed_at.isoformat() if experiment.completed_at else None,
            'progress': self.get_experiment_progress(experiment_id),
            'results': asdict(experiment.results) if experiment.results and hasattr(experiment.results, '__dataclass_fields__') else None
        }
        
        return results
    
    def get_experiment_progress(self, experiment_id: str) -> Dict[str, float]:
        """
        Get detailed progress information for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Dict[str, float]: Progress information for each pipeline stage
        """
        return self.experiment_progress.get(experiment_id, {})
    
    def cancel_experiment(self, experiment_id: str) -> bool:
        """
        Cancel a running experiment.
        
        Args:
            experiment_id: ID of the experiment to cancel
            
        Returns:
            bool: True if experiment was cancelled successfully
        """
        try:
            with self._lock:
                if experiment_id not in self.active_experiments:
                    return False
                
                # Cancel the future
                if experiment_id in self.experiment_futures:
                    future = self.experiment_futures[experiment_id]
                    future.cancel()
                
                # Update experiment status
                experiment = self.active_experiments[experiment_id]
                experiment.status = ExperimentStatus.CANCELLED
                experiment.completed_at = datetime.now()
                
                # Save to database
                self._save_experiment_to_db(experiment)
                
                # Remove from active experiments
                del self.active_experiments[experiment_id]
                if experiment_id in self.experiment_futures:
                    del self.experiment_futures[experiment_id]
            
            logger.info(f"Cancelled experiment {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel experiment {experiment_id}: {e}")
            return False
    
    def list_experiments(self, status_filter: Optional[ExperimentStatus] = None) -> List[Dict[str, Any]]:
        """
        List all experiments with optional status filtering.
        
        Args:
            status_filter: Optional status to filter by
            
        Returns:
            List[Dict[str, Any]]: List of experiment summaries
        """
        experiments = self._load_all_experiments_from_db()
        
        if status_filter:
            experiments = [exp for exp in experiments if exp.status == status_filter]
        
        return [
            {
                'id': exp.id,
                'name': exp.name,
                'status': exp.status.value,
                'created_at': exp.created_at.isoformat(),
                'completed_at': exp.completed_at.isoformat() if exp.completed_at else None,
                'progress': self.get_experiment_progress(exp.id)
            }
            for exp in experiments
        ]
    
    def add_progress_callback(self, experiment_id: str, callback: Callable[[str, Dict[str, float]], None]):
        """
        Add a progress callback for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            callback: Callback function that receives experiment_id and progress dict
        """
        if experiment_id not in self.progress_callbacks:
            self.progress_callbacks[experiment_id] = []
        self.progress_callbacks[experiment_id].append(callback)
    
    def enable_websocket_events(self, websocket_manager=None):
        """Enable WebSocket event broadcasting."""
        self._websocket_enabled = True
        if websocket_manager:
            self._websocket_manager = websocket_manager
    
    async def _broadcast_experiment_event(self, event_type: str, experiment_id: str, data: Dict[str, Any]):
        """Broadcast experiment event via WebSocket if enabled."""
        if not self._websocket_enabled:
            return
        
        try:
            # Import here to avoid circular imports
            from automl_framework.api.routes.websocket import (
                broadcast_experiment_created,
                broadcast_experiment_started,
                broadcast_experiment_progress,
                broadcast_experiment_completed,
                broadcast_experiment_failed
            )
            
            # Get user ID from experiment
            experiment = self.active_experiments.get(experiment_id) or self._load_experiment_from_db(experiment_id)
            user_id = experiment.config.get('user_id', 'unknown') if experiment else 'unknown'
            
            if event_type == 'created':
                await broadcast_experiment_created(experiment_id, data, user_id)
            elif event_type == 'started':
                await broadcast_experiment_started(experiment_id, data, user_id)
            elif event_type == 'progress':
                await broadcast_experiment_progress(experiment_id, data, user_id)
            elif event_type == 'completed':
                await broadcast_experiment_completed(experiment_id, data, user_id)
            elif event_type == 'failed':
                await broadcast_experiment_failed(experiment_id, data, user_id)
                
        except Exception as e:
            logger.error(f"Failed to broadcast WebSocket event: {e}")
    
    def _run_experiment_pipeline(self, experiment_id: str):
        """
        Execute the complete AutoML pipeline for an experiment.
        
        Args:
            experiment_id: ID of the experiment to run
        """
        try:
            experiment = self.active_experiments[experiment_id]
            logger.info(f"Starting pipeline for experiment {experiment_id}")
            
            # Create pipeline jobs
            self._create_pipeline_jobs(experiment_id, experiment)
            
            # Execute pipeline stages
            results = {}
            
            # Stage 1: Data Processing
            self._update_progress(experiment_id, 'data_processing', 0.0)
            processed_data = self._execute_data_processing(experiment_id, experiment)
            results['processed_data'] = processed_data
            self._update_progress(experiment_id, 'data_processing', 100.0)
            
            # Stage 2: Architecture Search
            self._update_progress(experiment_id, 'architecture_search', 0.0)
            best_architectures = self._execute_architecture_search(experiment_id, experiment, processed_data)
            results['architectures'] = best_architectures
            self._update_progress(experiment_id, 'architecture_search', 100.0)
            
            # Stage 3: Hyperparameter Optimization
            self._update_progress(experiment_id, 'hyperparameter_optimization', 0.0)
            best_configs = self._execute_hyperparameter_optimization(experiment_id, best_architectures)
            results['hyperparameters'] = best_configs
            self._update_progress(experiment_id, 'hyperparameter_optimization', 100.0)
            
            # Stage 4: Model Training
            self._update_progress(experiment_id, 'model_training', 0.0)
            trained_models = self._execute_model_training(experiment_id, best_architectures, best_configs, processed_data)
            results['trained_models'] = trained_models
            self._update_progress(experiment_id, 'model_training', 100.0)
            
            # Stage 5: Model Evaluation
            self._update_progress(experiment_id, 'model_evaluation', 0.0)
            evaluation_results = self._execute_model_evaluation(experiment_id, trained_models, processed_data)
            results['evaluation'] = evaluation_results
            self._update_progress(experiment_id, 'model_evaluation', 100.0)
            
            # Complete experiment
            self._complete_experiment(experiment_id, results)
            
        except Exception as e:
            logger.error(f"Pipeline failed for experiment {experiment_id}: {e}")
            self._fail_experiment(experiment_id, str(e))
        finally:
            # Clean up
            with self._lock:
                if experiment_id in self.active_experiments:
                    del self.active_experiments[experiment_id]
                if experiment_id in self.experiment_futures:
                    del self.experiment_futures[experiment_id]
    
    def _create_pipeline_jobs(self, experiment_id: str, experiment: Experiment):
        """Create pipeline jobs with dependencies."""
        jobs = []
        
        # Data processing job
        data_job = PipelineJob(
            job_id=f"{experiment_id}_data_processing",
            job_type="data_processing",
            job_function=self._execute_data_processing,
            args=(experiment_id, experiment)
        )
        jobs.append(data_job)
        
        # Architecture search job (depends on data processing)
        nas_job = PipelineJob(
            job_id=f"{experiment_id}_architecture_search",
            job_type="architecture_search",
            job_function=self._execute_architecture_search,
            args=(experiment_id, experiment)
        )
        jobs.append(nas_job)
        
        # Store jobs
        self.experiment_jobs[experiment_id] = jobs
        
        # Create dependencies
        self.job_dependencies[nas_job.job_id] = JobDependency(
            nas_job.job_id, [data_job.job_id]
        )
    
    def _execute_data_processing(self, experiment_id: str, experiment: Experiment, processed_data=None):
        """Execute data processing stage."""
        try:
            registry = get_service_registry()
            data_processor = registry.get_service('data_processor')
            
            # Get dataset path from experiment config
            dataset_path = experiment.config.get('dataset_path', '')
            
            # Analyze dataset
            dataset_metadata = data_processor.analyze_dataset(dataset_path)
            
            # Create preprocessing pipeline
            pipeline = data_processor.create_preprocessing_pipeline(dataset_metadata)
            
            # Load and preprocess data
            import pandas as pd
            raw_data = pd.read_csv(dataset_path)
            processed_data = data_processor.apply_preprocessing(pipeline, raw_data)
            
            logger.info(f"Data processing completed for experiment {experiment_id}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Data processing failed for experiment {experiment_id}: {e}")
            raise
    
    def _execute_architecture_search(self, experiment_id: str, experiment: Experiment, processed_data):
        """Execute architecture search stage."""
        try:
            registry = get_service_registry()
            nas_service = registry.get_service('nas_service')
            
            # Determine task type from config or data
            task_type = TaskType(experiment.config.get('task_type', 'classification'))
            data_type = DataType(experiment.config.get('data_type', 'tabular'))
            
            # Define search space
            search_space = nas_service.define_search_space(task_type, data_type)
            
            # Search for architectures
            architectures = nas_service.search_architectures(search_space, processed_data)
            
            logger.info(f"Architecture search completed for experiment {experiment_id}, found {len(architectures)} architectures")
            return architectures[:5]  # Return top 5 architectures
            
        except Exception as e:
            logger.error(f"Architecture search failed for experiment {experiment_id}: {e}")
            raise
    
    def _execute_hyperparameter_optimization(self, experiment_id: str, architectures: List[Architecture]):
        """Execute hyperparameter optimization stage."""
        try:
            registry = get_service_registry()
            hpo_service = registry.get_service('hyperparameter_optimizer')
            
            best_configs = []
            for arch in architectures:
                # Define hyperparameter search space
                search_space = hpo_service.define_search_space(arch)
                
                # Optimize hyperparameters
                def objective_function(params):
                    # This would normally train and evaluate a model
                    # For now, return a dummy score
                    return 0.85
                
                best_config = hpo_service.optimize(objective_function, search_space, max_trials=20)
                best_configs.append(best_config)
            
            logger.info(f"Hyperparameter optimization completed for experiment {experiment_id}")
            return best_configs
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed for experiment {experiment_id}: {e}")
            raise
    
    def _execute_model_training(self, experiment_id: str, architectures: List[Architecture], 
                              configs: List[TrainingConfig], processed_data):
        """Execute model training stage."""
        try:
            registry = get_service_registry()
            trainer = registry.get_service('model_trainer')
            
            trained_models = []
            for arch, config in zip(architectures, configs):
                trained_model = trainer.train_model(arch, config, processed_data)
                trained_models.append(trained_model)
            
            logger.info(f"Model training completed for experiment {experiment_id}, trained {len(trained_models)} models")
            return trained_models
            
        except Exception as e:
            logger.error(f"Model training failed for experiment {experiment_id}: {e}")
            raise
    
    def _execute_model_evaluation(self, experiment_id: str, trained_models, processed_data):
        """Execute model evaluation stage."""
        try:
            registry = get_service_registry()
            evaluator = registry.get_service('model_evaluator')
            
            evaluation_results = []
            for model in trained_models:
                metrics = evaluator.evaluate_model(model, processed_data.test_data)
                evaluation_results.append(metrics)
            
            logger.info(f"Model evaluation completed for experiment {experiment_id}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed for experiment {experiment_id}: {e}")
            raise
    
    def _complete_experiment(self, experiment_id: str, results: Dict[str, Any]):
        """Complete an experiment with results."""
        try:
            experiment = self.active_experiments[experiment_id]
            
            # Find best model
            best_model = None
            best_metrics = None
            if results.get('trained_models') and results.get('evaluation'):
                models = results['trained_models']
                evaluations = results['evaluation']
                
                # Find model with highest accuracy
                best_idx = 0
                best_accuracy = 0
                for i, metrics in enumerate(evaluations):
                    if metrics.accuracy and metrics.accuracy > best_accuracy:
                        best_accuracy = metrics.accuracy
                        best_idx = i
                
                best_model = models[best_idx]
                best_metrics = evaluations[best_idx]
            
            # Create experiment results
            experiment_results = ExperimentResults(
                best_architecture=best_model.architecture if best_model else None,
                best_hyperparameters=asdict(best_model.config) if best_model else None,
                performance_metrics=best_metrics,
                training_history=[],
                model_path=best_model.model_path if best_model else None
            )
            
            # Update experiment
            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = datetime.now()
            experiment.results = experiment_results
            
            # Save to database
            self._save_experiment_to_db(experiment)
            
            # Update progress
            self._update_progress(experiment_id, 'overall', 100.0)
            
            logger.info(f"Experiment {experiment_id} completed successfully")
            
            # Broadcast completion event
            asyncio.create_task(self._broadcast_experiment_event(
                'completed',
                experiment_id,
                {
                    'experiment_id': experiment_id,
                    'status': experiment.status.value,
                    'completed_at': experiment.completed_at.isoformat(),
                    'results': asdict(experiment_results) if hasattr(experiment_results, '__dataclass_fields__') else None
                }
            ))
            
        except Exception as e:
            logger.error(f"Failed to complete experiment {experiment_id}: {e}")
            self._fail_experiment(experiment_id, str(e))
    
    def _fail_experiment(self, experiment_id: str, error_message: str):
        """Mark an experiment as failed."""
        try:
            if experiment_id in self.active_experiments:
                experiment = self.active_experiments[experiment_id]
            else:
                experiment = self._load_experiment_from_db(experiment_id)
            
            if experiment:
                experiment.status = ExperimentStatus.FAILED
                experiment.completed_at = datetime.now()
                experiment.error_message = error_message
                self._save_experiment_to_db(experiment)
            
            logger.error(f"Experiment {experiment_id} failed: {error_message}")
            
            # Broadcast failure event
            asyncio.create_task(self._broadcast_experiment_event(
                'failed',
                experiment_id,
                {
                    'experiment_id': experiment_id,
                    'status': experiment.status.value,
                    'error_message': error_message,
                    'failed_at': experiment.completed_at.isoformat() if experiment.completed_at else datetime.now().isoformat()
                }
            ))
            
        except Exception as e:
            logger.error(f"Failed to mark experiment {experiment_id} as failed: {e}")
    
    def _update_progress(self, experiment_id: str, stage: str, progress: float):
        """Update progress for an experiment stage."""
        if experiment_id not in self.experiment_progress:
            return
        
        self.experiment_progress[experiment_id][stage] = progress
        
        # Calculate overall progress
        stages = ['data_processing', 'architecture_search', 'hyperparameter_optimization', 
                 'model_training', 'model_evaluation']
        total_progress = sum(self.experiment_progress[experiment_id][s] for s in stages) / len(stages)
        self.experiment_progress[experiment_id]['overall'] = total_progress
        
        # Call progress callbacks
        if experiment_id in self.progress_callbacks:
            for callback in self.progress_callbacks[experiment_id]:
                try:
                    callback(experiment_id, self.experiment_progress[experiment_id])
                except Exception as e:
                    logger.error(f"Progress callback failed: {e}")
        
        # Broadcast progress update
        asyncio.create_task(self._broadcast_experiment_event(
            'progress',
            experiment_id,
            {
                'experiment_id': experiment_id,
                'stage': stage,
                'progress': progress,
                'overall_progress': total_progress
            }
        ))
    
    def _validate_experiment_config(self, config: Dict[str, Any]):
        """Validate experiment configuration."""
        required_fields = ['dataset_path']
        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required field: {field}")
        
        # Validate dataset path exists
        import os
        if not os.path.exists(config['dataset_path']):
            raise ValidationError(f"Dataset file not found: {config['dataset_path']}")
    
    def _save_experiment_to_db(self, experiment: Experiment):
        """Save experiment to database."""
        try:
            # For now, use in-memory storage instead of MongoDB
            experiment_doc = {
                'id': experiment.id,
                'name': experiment.name,
                'dataset_id': experiment.dataset_id,
                'status': experiment.status.value,
                'created_at': experiment.created_at.isoformat(),
                'completed_at': experiment.completed_at.isoformat() if experiment.completed_at else None,
                'config': experiment.config,
                'error_message': experiment.error_message,
                'results': asdict(experiment.results) if experiment.results and hasattr(experiment.results, '__dataclass_fields__') else None
            }
            
            self.experiments_storage[experiment.id] = experiment_doc
            logger.debug(f"Saved experiment {experiment.id} to in-memory storage")
            
        except Exception as e:
            logger.error(f"Failed to save experiment to database: {e}")
            raise
    
    def _load_experiment_from_db(self, experiment_id: str) -> Optional[Experiment]:
        """Load experiment from database."""
        try:
            doc = self.experiments_storage.get(experiment_id)
            
            if doc:
                # Convert datetime strings back to datetime objects
                if 'created_at' in doc and isinstance(doc['created_at'], str):
                    doc['created_at'] = datetime.fromisoformat(doc['created_at'])
                if 'completed_at' in doc and isinstance(doc['completed_at'], str):
                    doc['completed_at'] = datetime.fromisoformat(doc['completed_at'])
                
                # Convert status string back to enum
                if 'status' in doc and isinstance(doc['status'], str):
                    doc['status'] = ExperimentStatus(doc['status'])
                
                # Ensure required fields are present
                if 'dataset_id' not in doc:
                    doc['dataset_id'] = doc.get('id', 'unknown')
                
                # Convert results dict back to ExperimentResults if present
                if 'results' in doc and doc['results'] and isinstance(doc['results'], dict):
                    doc['results'] = ExperimentResults(**doc['results'])
                
                return Experiment(**doc)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load experiment from database: {e}")
            return None
    
    def _load_all_experiments_from_db(self) -> List[Experiment]:
        """Load all experiments from database."""
        try:
            experiments = []
            for doc in self.experiments_storage.values():
                # Make a copy to avoid modifying the stored data
                doc_copy = doc.copy()
                
                # Convert datetime strings back to datetime objects
                if 'created_at' in doc_copy and isinstance(doc_copy['created_at'], str):
                    doc_copy['created_at'] = datetime.fromisoformat(doc_copy['created_at'])
                if 'completed_at' in doc_copy and isinstance(doc_copy['completed_at'], str):
                    doc_copy['completed_at'] = datetime.fromisoformat(doc_copy['completed_at'])
                
                # Convert status string back to enum
                if 'status' in doc_copy and isinstance(doc_copy['status'], str):
                    doc_copy['status'] = ExperimentStatus(doc_copy['status'])
                
                # Ensure required fields are present
                if 'dataset_id' not in doc_copy:
                    doc_copy['dataset_id'] = doc_copy.get('id', 'unknown')
                
                # Convert results dict back to ExperimentResults if present
                if 'results' in doc_copy and doc_copy['results'] and isinstance(doc_copy['results'], dict):
                    doc_copy['results'] = ExperimentResults(**doc_copy['results'])
                
                experiments.append(Experiment(**doc_copy))
            
            return experiments
            
        except Exception as e:
            logger.error(f"Failed to load experiments from database: {e}")
            return []
    
    def shutdown(self):
        """Shutdown the experiment manager."""
        logger.info("Shutting down ExperimentManager")
        
        # Cancel all running experiments
        with self._lock:
            for experiment_id in list(self.active_experiments.keys()):
                self.cancel_experiment(experiment_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("ExperimentManager shutdown complete")