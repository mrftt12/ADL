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
    """Initialize services on application startup."""
    from automl_framework.core.registry import initialize_service_registry, get_service_registry
    from automl_framework.monitoring.metrics import initialize_metrics_collection
    from automl_framework.monitoring.alerts import initialize_alert_system
    
    # Initialize service registry
    initialize_service_registry()
    
    # Initialize monitoring systems
    initialize_metrics_collection()
    initialize_alert_system()
    
    # Start WebSocket manager
    await websocket_manager.start()
    
    # Enable WebSocket events in services
    try:
        registry = get_service_registry()
        
        # Enable WebSocket events in experiment manager
        experiment_manager = registry.get_service('experiment_manager')
        if experiment_manager:
            experiment_manager.enable_websocket_events(websocket_manager)
            logger.info("WebSocket events enabled for experiment manager")
        
        # Enable WebSocket events in resource scheduler
        resource_scheduler = registry.get_service('resource_scheduler')
        if resource_scheduler:
            resource_scheduler.enable_websocket_events(websocket_manager)
            # Start async monitoring now that event loop is available
            resource_scheduler.start_async_monitoring()
            logger.info("WebSocket events enabled for resource scheduler")
            
    except Exception as e:
        logger.warning(f"Failed to enable WebSocket events in some services: {e}")
    
    logger.info("Application startup complete")

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
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

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