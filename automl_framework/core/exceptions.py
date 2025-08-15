"""
Custom exceptions for AutoML Framework
"""

from typing import Optional, Dict, Any


class AutoMLException(Exception):
    """Base exception class for AutoML Framework"""
    
    def __init__(
        self, 
        message: str, 
        error_code: str, 
        recoverable: bool = False,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.recoverable = recoverable
        self.context = context or {}


class DataProcessingError(AutoMLException):
    """Exception raised during data processing operations"""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="DATA_PROCESSING_ERROR",
            recoverable=False,
            context=context
        )


class ArchitectureSearchError(AutoMLException):
    """Exception raised during neural architecture search"""
    
    def __init__(self, message: str, recoverable: bool = True, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="ARCHITECTURE_SEARCH_ERROR",
            recoverable=recoverable,
            context=context
        )


class HyperparameterOptimizationError(AutoMLException):
    """Exception raised during hyperparameter optimization"""
    
    def __init__(self, message: str, recoverable: bool = True, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="HYPERPARAMETER_OPTIMIZATION_ERROR",
            recoverable=recoverable,
            context=context
        )


class TrainingError(AutoMLException):
    """Exception raised during model training"""
    
    def __init__(self, message: str, recoverable: bool = True, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="TRAINING_ERROR",
            recoverable=recoverable,
            context=context
        )


class ResourceError(AutoMLException):
    """Exception raised when resource allocation fails"""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RESOURCE_ERROR",
            recoverable=True,
            context=context
        )


class ValidationError(AutoMLException):
    """Exception raised during input validation"""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            recoverable=False,
            context=context
        )


class CheckpointError(AutoMLException):
    """Exception raised during checkpoint operations"""
    
    def __init__(self, message: str, recoverable: bool = True, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CHECKPOINT_ERROR",
            recoverable=recoverable,
            context=context
        )


class ExperimentError(AutoMLException):
    """Exception raised during experiment management operations"""
    
    def __init__(self, message: str, recoverable: bool = False, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="EXPERIMENT_ERROR",
            recoverable=recoverable,
            context=context
        )