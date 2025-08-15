"""
Logging infrastructure for AutoML Framework
"""

import logging
import logging.handlers
import sys
from typing import Optional
from pathlib import Path
import json
from datetime import datetime

from automl_framework.core.config import get_config


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'experiment_id'):
            log_entry['experiment_id'] = record.experiment_id
        if hasattr(record, 'job_id'):
            log_entry['job_id'] = record.job_id
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        return json.dumps(log_entry)


class AutoMLLogger:
    """Custom logger for AutoML Framework with structured logging support"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._configured = False
    
    def _configure_logger(self):
        """Configure logger with handlers and formatters"""
        if self._configured:
            return
        
        config = get_config()
        log_config = config.logging
        
        # Set log level
        level = getattr(logging, log_config.level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Use JSON formatter for structured logging
        json_formatter = JSONFormatter()
        console_handler.setFormatter(json_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_config.file_path:
            log_path = Path(log_config.file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_config.file_path,
                maxBytes=log_config.max_file_size_mb * 1024 * 1024,
                backupCount=log_config.backup_count
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(json_formatter)
            self.logger.addHandler(file_handler)
        
        self._configured = True
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional context"""
        self._configure_logger()
        extra = self._build_extra(**kwargs)
        self.logger.debug(message, extra=extra)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional context"""
        self._configure_logger()
        extra = self._build_extra(**kwargs)
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional context"""
        self._configure_logger()
        extra = self._build_extra(**kwargs)
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional context"""
        self._configure_logger()
        extra = self._build_extra(**kwargs)
        self.logger.error(message, extra=extra)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with optional context"""
        self._configure_logger()
        extra = self._build_extra(**kwargs)
        self.logger.critical(message, extra=extra)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback and optional context"""
        self._configure_logger()
        extra = self._build_extra(**kwargs)
        self.logger.exception(message, extra=extra)
    
    def _build_extra(self, **kwargs) -> dict:
        """Build extra context for logging"""
        extra = {}
        for key, value in kwargs.items():
            if key in ['experiment_id', 'job_id', 'user_id', 'model_id', 'dataset_id']:
                extra[key] = value
        return extra


class ExperimentLogger(AutoMLLogger):
    """Specialized logger for experiment tracking"""
    
    def __init__(self, experiment_id: str):
        super().__init__(f"automl.experiment.{experiment_id}")
        self.experiment_id = experiment_id
    
    def log_experiment_start(self, dataset_name: str, config: dict):
        """Log experiment start"""
        self.info(
            f"Starting experiment with dataset: {dataset_name}",
            experiment_id=self.experiment_id,
            dataset_name=dataset_name,
            config=config
        )
    
    def log_experiment_complete(self, best_model_id: str, metrics: dict):
        """Log experiment completion"""
        self.info(
            f"Experiment completed. Best model: {best_model_id}",
            experiment_id=self.experiment_id,
            best_model_id=best_model_id,
            metrics=metrics
        )
    
    def log_experiment_error(self, error: Exception, stage: str):
        """Log experiment error"""
        self.error(
            f"Experiment failed at stage: {stage}",
            experiment_id=self.experiment_id,
            stage=stage,
            error_type=type(error).__name__,
            error_message=str(error)
        )
    
    def log_stage_start(self, stage: str, details: Optional[dict] = None):
        """Log experiment stage start"""
        self.info(
            f"Starting stage: {stage}",
            experiment_id=self.experiment_id,
            stage=stage,
            details=details or {}
        )
    
    def log_stage_complete(self, stage: str, results: Optional[dict] = None):
        """Log experiment stage completion"""
        self.info(
            f"Completed stage: {stage}",
            experiment_id=self.experiment_id,
            stage=stage,
            results=results or {}
        )


class TrainingLogger(AutoMLLogger):
    """Specialized logger for training jobs"""
    
    def __init__(self, job_id: str):
        super().__init__(f"automl.training.{job_id}")
        self.job_id = job_id
    
    def log_training_start(self, architecture_id: str, config: dict):
        """Log training start"""
        self.info(
            f"Starting training job for architecture: {architecture_id}",
            job_id=self.job_id,
            architecture_id=architecture_id,
            config=config
        )
    
    def log_epoch_metrics(self, epoch: int, metrics: dict):
        """Log epoch metrics"""
        self.info(
            f"Epoch {epoch} metrics",
            job_id=self.job_id,
            epoch=epoch,
            metrics=metrics
        )
    
    def log_training_complete(self, final_metrics: dict):
        """Log training completion"""
        self.info(
            "Training completed",
            job_id=self.job_id,
            final_metrics=final_metrics
        )
    
    def log_checkpoint_saved(self, epoch: int, checkpoint_path: str):
        """Log checkpoint save"""
        self.info(
            f"Checkpoint saved at epoch {epoch}",
            job_id=self.job_id,
            epoch=epoch,
            checkpoint_path=checkpoint_path
        )


def get_logger(name: str) -> AutoMLLogger:
    """Get logger instance for given name"""
    return AutoMLLogger(name)


def get_experiment_logger(experiment_id: str) -> ExperimentLogger:
    """Get experiment logger instance"""
    return ExperimentLogger(experiment_id)


def get_training_logger(job_id: str) -> TrainingLogger:
    """Get training logger instance"""
    return TrainingLogger(job_id)