"""
FastAPI application for AutoML Framework REST API.

This module provides REST API endpoints for dataset upload, experiment creation,
monitoring, and result retrieval for the AutoML framework.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from automl_framework.core.exceptions import (
    ExperimentError,
    ResourceError,
    ValidationError,
    AutoMLException
)
from automl_framework.models.data_models import TaskType, DataType
from automl_framework.api.routes import auth, experiments, datasets, resources, websocket, model_serving, model_monitoring, monitoring
from automl_framework.api.websocket_manager import websocket_manager
from automl_framework.api.middleware import (
    MetricsMiddleware,
    AlertingMiddleware,
    HealthCheckMiddleware,
    RequestLoggingMiddleware
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AutoML Framework API",
    description="REST API for automated machine learning pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add monitoring middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(MetricsMiddleware)
app.add_middleware(AlertingMiddleware, error_threshold=10, response_time_threshold=5000.0)
app.add_middleware(HealthCheckMiddleware, check_interval=300)

# Include routers
app.include_router(auth.router)
app.include_router(experiments.router)
app.include_router(datasets.router)
app.include_router(resources.router)
app.include_router(websocket.router)
app.include_router(model_serving.router)
app.include_router(model_monitoring.router)
app.include_router(monitoring.router)

# Pydantic models for error responses
from pydantic import BaseModel
from typing import Optional

class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None

# Error handlers
@app.exception_handler(ExperimentError)
async def experiment_error_handler(request, exc: ExperimentError):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="ExperimentError",
            message=str(exc),
            details={"recoverable": getattr(exc, 'recoverable', False)}
        ).dict()
    )

@app.exception_handler(ResourceError)
async def resource_error_handler(request, exc: ResourceError):
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            error="ResourceError",
            message=str(exc),
            details={"recoverable": getattr(exc, 'recoverable', True)}
        ).dict()
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="ValidationError",
            message=str(exc)
        ).dict()
    )

@app.exception_handler(AutoMLException)
async def automl_error_handler(request, exc: AutoMLException):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="AutoMLException",
            message=str(exc),
            details={"recoverable": getattr(exc, 'recoverable', False)}
        ).dict()
    )

# Application lifecycle events
@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup with comprehensive error handling."""
    from automl_framework.core.registry import initialize_service_registry, get_service_registry
    from automl_framework.monitoring.metrics import initialize_metrics_collection
    from automl_framework.monitoring.alerts import initialize_alert_system
    from automl_framework.core.environment import initialize_environment, get_environment_manager
    from automl_framework.api.auth import initialize_authentication
    
    startup_errors = []
    env_config = None
    
    # Initialize environment detection first with error handling
    try:
        logger.info("Starting environment detection...")
        env_config = initialize_environment()
        env_manager = get_environment_manager()
        
        # Log detailed environment information
        logger.info(f"Environment detected: {env_config.name}")
        logger.info(f"GPU available: {env_config.gpu_available}")
        logger.info(f"Database available: {env_config.database_available}")
        logger.info(f"Authentication backend: {env_manager.get_auth_backend()}")
        logger.info(f"Resource limits: {env_manager.get_resource_limits()}")
        
        if not env_config.gpu_available:
            logger.info("Running in CPU-only mode - GPU-related services will be skipped")
        
        if not env_config.database_available:
            logger.warning("Database not available - using in-memory storage for authentication")
        
    except Exception as e:
        error_msg = f"Failed to initialize environment detection: {e}"
        logger.error(error_msg, exc_info=True)
        startup_errors.append(("environment_detection", error_msg))
        
        # Create fallback environment config
        try:
            from automl_framework.core.environment import EnvironmentConfig
            env_config = EnvironmentConfig(
                name="unknown",
                supports_gpu=False,
                gpu_available=False,
                supports_persistent_storage=False,
                default_auth_backend="memory",
                max_memory_mb=1024,
                max_cpu_cores=1,
                database_available=False
            )
            logger.warning("Using fallback environment configuration")
        except Exception as fallback_error:
            logger.error(f"Failed to create fallback environment config: {fallback_error}")
            env_config = None
    
    # Initialize authentication system with error handling
    try:
        logger.info("Initializing authentication system...")
        auth_status = initialize_authentication()
        
        if auth_status["status"] == "success":
            logger.info(f"Authentication initialized successfully: backend={auth_status['backend']}")
            logger.info(f"Demo users available: {auth_status['demo_users_available']}")
        else:
            logger.error(f"Authentication initialization failed: {auth_status.get('error', 'Unknown error')}")
            startup_errors.append(("authentication", auth_status.get('error', 'Unknown error')))
            
    except Exception as e:
        error_msg = f"Failed to initialize authentication: {e}"
        logger.error(error_msg, exc_info=True)
        startup_errors.append(("authentication", error_msg))
    
    # Initialize service registry with error handling
    try:
        logger.info("Initializing service registry...")
        initialize_service_registry()
        logger.info("Service registry initialized successfully")
    except Exception as e:
        error_msg = f"Failed to initialize service registry: {e}"
        logger.error(error_msg, exc_info=True)
        startup_errors.append(("service_registry", error_msg))
    
    # Initialize monitoring systems with error handling
    try:
        logger.info("Initializing monitoring systems...")
        initialize_metrics_collection()
        logger.info("Metrics collection initialized successfully")
    except Exception as e:
        error_msg = f"Failed to initialize metrics collection: {e}"
        logger.warning(error_msg, exc_info=True)
        startup_errors.append(("metrics_collection", error_msg))
    
    try:
        initialize_alert_system()
        logger.info("Alert system initialized successfully")
    except Exception as e:
        error_msg = f"Failed to initialize alert system: {e}"
        logger.warning(error_msg, exc_info=True)
        startup_errors.append(("alert_system", error_msg))
    
    # Start WebSocket manager with error handling
    try:
        logger.info("Starting WebSocket manager...")
        await websocket_manager.start()
        logger.info("WebSocket manager started successfully")
    except Exception as e:
        error_msg = f"Failed to start WebSocket manager: {e}"
        logger.error(error_msg, exc_info=True)
        startup_errors.append(("websocket_manager", error_msg))
    
    # Enable WebSocket events in services with error handling
    try:
        logger.info("Configuring service integrations...")
        registry = get_service_registry()
        
        # Enable WebSocket events in experiment manager
        try:
            experiment_manager = registry.get_service('experiment_manager')
            if experiment_manager:
                experiment_manager.enable_websocket_events(websocket_manager)
                logger.info("WebSocket events enabled for experiment manager")
            else:
                logger.warning("Experiment manager service not found in registry")
        except Exception as e:
            error_msg = f"Failed to configure experiment manager: {e}"
            logger.warning(error_msg, exc_info=True)
            startup_errors.append(("experiment_manager_config", error_msg))
        
        # Enable WebSocket events in resource scheduler
        try:
            resource_scheduler = registry.get_service('resource_scheduler')
            if resource_scheduler:
                # Configure resource scheduler for CPU-only mode if needed
                if env_config and not env_config.gpu_available:
                    logger.info("Configuring resource scheduler for CPU-only mode")
                    # The resource scheduler will automatically detect GPU availability
                    # through its own system resource detection
                
                resource_scheduler.enable_websocket_events(websocket_manager)
                # Start async monitoring now that event loop is available
                resource_scheduler.start_async_monitoring()
                logger.info("WebSocket events enabled for resource scheduler")
            else:
                logger.warning("Resource scheduler service not found in registry")
        except Exception as e:
            error_msg = f"Failed to configure resource scheduler: {e}"
            logger.warning(error_msg, exc_info=True)
            startup_errors.append(("resource_scheduler_config", error_msg))
            
    except Exception as e:
        error_msg = f"Failed to configure service integrations: {e}"
        logger.error(error_msg, exc_info=True)
        startup_errors.append(("service_integrations", error_msg))
    
    # GPU initialization with error handling
    if env_config and env_config.gpu_available:
        try:
            logger.info("Initializing GPU services...")
            # Test GPU availability one more time during startup
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"GPU initialization successful: {gpu_count} GPU(s) available")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                logger.warning("GPU was detected during environment setup but is not available during startup")
                startup_errors.append(("gpu_initialization", "GPU not available during startup"))
        except ImportError:
            logger.warning("PyTorch not available - GPU services will be disabled")
            startup_errors.append(("gpu_initialization", "PyTorch not installed"))
        except Exception as e:
            error_msg = f"GPU initialization failed: {e}"
            logger.warning(error_msg, exc_info=True)
            startup_errors.append(("gpu_initialization", error_msg))
    else:
        logger.info("GPU services disabled - running in CPU-only mode")
    
    # Database connectivity test with error handling
    if env_config and env_config.database_available:
        try:
            logger.info("Testing database connectivity...")
            from automl_framework.core.database import get_database_manager
            db_manager = get_database_manager()
            health_status = db_manager.health_check()
            
            healthy_dbs = [db for db, status in health_status.items() if status]
            if healthy_dbs:
                logger.info(f"Database connectivity confirmed: {healthy_dbs}")
            else:
                logger.warning("No healthy databases found during startup")
                startup_errors.append(("database_connectivity", "No healthy databases found"))
                
        except Exception as e:
            error_msg = f"Database connectivity test failed: {e}"
            logger.warning(error_msg, exc_info=True)
            startup_errors.append(("database_connectivity", error_msg))
    else:
        logger.info("Database services disabled - using in-memory storage")
    
    # Log startup summary
    if startup_errors:
        logger.warning(f"Application startup completed with {len(startup_errors)} errors/warnings:")
        for component, error in startup_errors:
            logger.warning(f"  - {component}: {error}")
        logger.warning("Application will continue with reduced functionality")
    else:
        logger.info("Application startup completed successfully with no errors")
    
    # Log final configuration summary
    if env_config:
        logger.info("Final startup configuration:")
        logger.info(f"  Environment: {env_config.name}")
        logger.info(f"  GPU Available: {env_config.gpu_available}")
        logger.info(f"  Database Available: {env_config.database_available}")
        logger.info(f"  Max Memory: {env_config.max_memory_mb}MB")
        logger.info(f"  Max CPU Cores: {env_config.max_cpu_cores}")
    
    logger.info("Application startup process complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup services on application shutdown."""
    from automl_framework.monitoring.metrics import get_metrics_collector
    
    # Stop monitoring systems
    try:
        metrics_collector = get_metrics_collector()
        metrics_collector.stop_collection()
    except Exception as e:
        logger.warning(f"Error stopping metrics collection: {e}")
    
    await websocket_manager.stop()
    logger.info("Application shutdown complete")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint with comprehensive system status."""
    from automl_framework.core.environment import get_environment_manager
    from automl_framework.api.auth import get_auth_backend_info, test_authentication
    
    try:
        env_manager = get_environment_manager()
        env_config = env_manager.config
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "environment": env_config.name,
            "gpu_available": env_config.gpu_available,
            "database_available": env_config.database_available
        }
        
        # Add authentication backend information
        try:
            auth_info = get_auth_backend_info()
            health_status["authentication"] = {
                "backend": auth_info.get("backend", "unknown"),
                "database_available": auth_info.get("database_available", False),
                "user_count": auth_info.get("user_count", "unknown")
            }
        except Exception as e:
            logger.debug(f"Could not get auth backend info for health check: {e}")
            health_status["authentication"] = {"status": "error", "error": str(e)}
        
        # Test authentication functionality
        try:
            auth_test = test_authentication()
            health_status["authentication"]["test_result"] = auth_test["status"]
            if auth_test["status"] != "success":
                health_status["authentication"]["test_error"] = auth_test.get("error", "Unknown error")
        except Exception as e:
            logger.debug(f"Authentication test failed during health check: {e}")
            health_status["authentication"]["test_result"] = "error"
            health_status["authentication"]["test_error"] = str(e)
        
        # Add resource information if available
        try:
            from automl_framework.core.registry import get_service_registry
            registry = get_service_registry()
            resource_scheduler = registry.get_service('resource_scheduler')
            if resource_scheduler:
                resource_status = resource_scheduler.get_resource_status()
                health_status["resources"] = {
                    "running_jobs": resource_status.get("running_jobs", 0),
                    "queued_jobs": resource_status.get("queued_jobs", 0)
                }
        except Exception as e:
            logger.debug(f"Could not get resource status for health check: {e}")
            health_status["resources"] = {"status": "error", "error": str(e)}
        
        # Add GPU information if available
        if env_config.gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    health_status["gpu"] = {
                        "available": True,
                        "count": gpu_count,
                        "devices": []
                    }
                    for i in range(min(gpu_count, 4)):  # Limit to first 4 GPUs for health check
                        try:
                            gpu_name = torch.cuda.get_device_name(i)
                            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                            health_status["gpu"]["devices"].append({
                                "id": i,
                                "name": gpu_name,
                                "memory_gb": round(gpu_memory, 1)
                            })
                        except Exception as gpu_error:
                            logger.debug(f"Could not get info for GPU {i}: {gpu_error}")
                else:
                    health_status["gpu"] = {"available": False, "reason": "CUDA not available"}
            except ImportError:
                health_status["gpu"] = {"available": False, "reason": "PyTorch not installed"}
            except Exception as e:
                health_status["gpu"] = {"available": False, "reason": str(e)}
        else:
            health_status["gpu"] = {"available": False, "reason": "GPU not detected during startup"}
        
        # Add database connectivity information
        if env_config.database_available:
            try:
                from automl_framework.core.database import get_database_manager
                db_manager = get_database_manager()
                db_health = db_manager.health_check()
                health_status["databases"] = db_health
            except Exception as e:
                logger.debug(f"Could not get database status for health check: {e}")
                health_status["databases"] = {"status": "error", "error": str(e)}
        else:
            health_status["databases"] = {"status": "disabled", "reason": "Database not available"}
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "degraded",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

# Configuration endpoint
@app.get("/api/v1/config")
async def get_api_config():
    """
    Get API configuration and available options.
    """
    return {
        "version": "1.0.0",
        "supported_data_types": [dt.value for dt in DataType],
        "supported_task_types": [tt.value for tt in TaskType],
        "max_file_size_mb": 100,
        "supported_file_formats": [
            ".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json",
            ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif",
            ".txt", ".jsonl"
        ]
    }

# Legacy model endpoints (redirects to new model serving endpoints)
@app.get("/api/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get trained model information (legacy endpoint)."""
    return {
        "model_id": model_id,
        "status": "available",
        "message": "Please use /models/{model_id}/{version}/info endpoint for detailed model information"
    }

@app.post("/api/v1/models/{model_id}/predict")
async def predict_model(model_id: str, data: Dict[str, Any]):
    """Make predictions using a trained model (legacy endpoint)."""
    return {
        "model_id": model_id,
        "predictions": [],
        "message": "Please use /models/predict endpoint for model predictions"
    }

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the application
    uvicorn.run(
        "automl_framework.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )