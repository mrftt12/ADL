"""
Services package for AutoML Framework
"""

from .evaluation_service import (
    ComprehensiveEvaluator,
    ModelComparator,
    ConfusionMatrixResult,
    StatisticalTestResult,
    ModelComparisonResult,
    CrossValidationResult,
    MetricType
)

from .experiment_manager import (
    ExperimentManager,
    PipelineJob,
    JobDependency
)

from .resource_scheduler import (
    ResourceScheduler,
    ResourceRequirement,
    ResourceAllocation,
    ScheduledJob,
    SystemResources,
    JobPriority,
    JobStatus,
    ResourceType
)

__all__ = [
    'ComprehensiveEvaluator',
    'ModelComparator',
    'ConfusionMatrixResult',
    'StatisticalTestResult',
    'ModelComparisonResult',
    'CrossValidationResult',
    'MetricType',
    'ExperimentManager',
    'PipelineJob',
    'JobDependency',
    'ResourceScheduler',
    'ResourceRequirement',
    'ResourceAllocation',
    'ScheduledJob',
    'SystemResources',
    'JobPriority',
    'JobStatus',
    'ResourceType'
]