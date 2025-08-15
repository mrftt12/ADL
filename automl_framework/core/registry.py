"""
Service registry for managing AutoML Framework components
"""

from typing import Dict, Type, Any, Optional
from automl_framework.core.interfaces import (
    IDataProcessor,
    INASService,
    IHyperparameterOptimizer,
    IModelTrainer,
    IModelEvaluator,
    IResourceScheduler,
    IExperimentManager
)


class ServiceRegistry:
    """Registry for managing service instances and dependencies"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._service_types: Dict[str, Type] = {
            'data_processor': IDataProcessor,
            'nas_service': INASService,
            'hyperparameter_optimizer': IHyperparameterOptimizer,
            'model_trainer': IModelTrainer,
            'model_evaluator': IModelEvaluator,
            'resource_scheduler': IResourceScheduler,
            'experiment_manager': IExperimentManager
        }
    
    def register_service(self, service_name: str, service_instance: Any) -> None:
        """Register a service instance"""
        if service_name not in self._service_types:
            raise ValueError(f"Unknown service type: {service_name}")
        
        expected_type = self._service_types[service_name]
        if not isinstance(service_instance, expected_type):
            raise TypeError(f"Service {service_name} must implement {expected_type.__name__}")
        
        self._services[service_name] = service_instance
    
    def get_service(self, service_name: str) -> Any:
        """Get a registered service instance"""
        if service_name not in self._services:
            raise ValueError(f"Service {service_name} not registered")
        return self._services[service_name]
    
    def is_registered(self, service_name: str) -> bool:
        """Check if a service is registered"""
        return service_name in self._services
    
    def list_services(self) -> Dict[str, str]:
        """List all registered services"""
        return {
            name: type(service).__name__ 
            for name, service in self._services.items()
        }
    
    def unregister_service(self, service_name: str) -> None:
        """Unregister a service"""
        if service_name in self._services:
            del self._services[service_name]


# Global service registry instance
service_registry = ServiceRegistry()


def get_service_registry() -> ServiceRegistry:
    """Get the global service registry"""
    return service_registry


def initialize_service_registry() -> None:
    """Initialize the service registry with default implementations"""
    from automl_framework.services.data_processing import DatasetAnalyzer
    from automl_framework.services.nas_service import NASService
    from automl_framework.services.hyperparameter_optimization import HyperparameterOptimizationService
    from automl_framework.services.training_service import ModelTrainingService
    from automl_framework.services.evaluation_service import ComprehensiveEvaluator
    from automl_framework.services.resource_scheduler import ResourceScheduler
    from automl_framework.services.experiment_manager import ExperimentManager
    
    # Create a simple data processor wrapper
    class DataProcessorWrapper(IDataProcessor):
        def __init__(self):
            self.analyzer = DatasetAnalyzer()
        
        def analyze_dataset(self, dataset_path: str, target_column: str = None):
            return self.analyzer.analyze_dataset(dataset_path, target_column)
        
        def create_preprocessing_pipeline(self, metadata):
            # This would create a preprocessing pipeline
            return None
        
        def apply_preprocessing(self, pipeline, data):
            # This would apply preprocessing
            return data
    
    # Create a model evaluator wrapper
    class ModelEvaluatorWrapper(IModelEvaluator):
        def __init__(self):
            from automl_framework.models.data_models import TaskType
            self.evaluator = ComprehensiveEvaluator(TaskType.CLASSIFICATION)
        
        def evaluate_model(self, model, test_data):
            # This would evaluate the model
            # For now, return a mock performance metrics
            from automl_framework.models.data_models import PerformanceMetrics
            return PerformanceMetrics(
                accuracy=0.85,
                precision=0.83,
                recall=0.87,
                f1_score=0.85,
                roc_auc=0.89
            )
        
        def compare_models(self, models):
            # Mock model comparison
            return {
                "best_model": models[0] if models else None,
                "rankings": [{"model": m, "score": 0.85} for m in models]
            }
        
        def generate_report(self, model, metrics):
            # Mock report generation
            return {
                "model_summary": "Mock model report",
                "metrics": metrics.__dict__ if hasattr(metrics, '__dict__') else metrics,
                "recommendations": ["Model performs well on test data"]
            }
    
    # Register services
    if not service_registry.is_registered('data_processor'):
        service_registry.register_service('data_processor', DataProcessorWrapper())
    
    if not service_registry.is_registered('nas_service'):
        service_registry.register_service('nas_service', NASService())
    
    if not service_registry.is_registered('hyperparameter_optimizer'):
        service_registry.register_service('hyperparameter_optimizer', HyperparameterOptimizationService())
    
    if not service_registry.is_registered('model_trainer'):
        service_registry.register_service('model_trainer', ModelTrainingService())
    
    if not service_registry.is_registered('model_evaluator'):
        service_registry.register_service('model_evaluator', ModelEvaluatorWrapper())
    
    if not service_registry.is_registered('resource_scheduler'):
        resource_scheduler = ResourceScheduler()
        resource_scheduler.enable_websocket_events()
        service_registry.register_service('resource_scheduler', resource_scheduler)
    
    if not service_registry.is_registered('experiment_manager'):
        experiment_manager = ExperimentManager()
        experiment_manager.enable_websocket_events()
        service_registry.register_service('experiment_manager', experiment_manager)