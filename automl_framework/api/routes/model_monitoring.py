"""
FastAPI routes for model monitoring and versioning
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ...services.model_monitoring import (
    ModelMonitoringService,
    ModelVersion,
    PerformanceSnapshot,
    Alert,
    ABTestConfig,
    ModelStatus,
    AlertType
)
from ...services.model_export import ModelExportService
from ...services.model_serving import ModelServingService
from ...core.exceptions import AutoMLException


logger = logging.getLogger(__name__)

# Initialize services (these would typically be dependency injected)
export_service = ModelExportService()
serving_service = ModelServingService(export_service)
monitoring_service = ModelMonitoringService(export_service, serving_service)

router = APIRouter(prefix="/monitoring", tags=["model-monitoring"])


# Pydantic models for API
class ModelVersionModel(BaseModel):
    """API model for model versions."""
    model_id: str
    version: str
    status: str
    created_at: str
    deployed_at: Optional[str] = None
    deprecated_at: Optional[str] = None
    traffic_percentage: float
    description: Optional[str] = None
    tags: List[str] = []


class RegisterVersionRequest(BaseModel):
    """API model for registering model versions."""
    model_id: str = Field(..., description="Model identifier")
    version: str = Field(..., description="Model version")
    description: Optional[str] = Field(None, description="Version description")
    tags: Optional[List[str]] = Field(None, description="Version tags")


class DeployVersionRequest(BaseModel):
    """API model for deploying model versions."""
    traffic_percentage: float = Field(100.0, description="Traffic percentage", ge=0, le=100)


class PerformanceSnapshotModel(BaseModel):
    """API model for performance snapshots."""
    model_id: str
    version: str
    timestamp: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    avg_response_time: float
    error_rate: float
    prediction_count: int
    confidence_distribution: Dict[str, float] = {}


class AlertModel(BaseModel):
    """API model for alerts."""
    id: str
    model_id: str
    version: str
    alert_type: str
    severity: str
    message: str
    timestamp: str
    resolved: bool
    resolved_at: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ABTestRequest(BaseModel):
    """API model for creating A/B tests."""
    test_id: str = Field(..., description="Test identifier")
    model_id: str = Field(..., description="Model identifier")
    control_version: str = Field(..., description="Control version")
    treatment_version: str = Field(..., description="Treatment version")
    traffic_split: float = Field(50.0, description="Traffic split percentage", ge=0, le=100)
    duration_hours: Optional[int] = Field(None, description="Test duration in hours")
    success_metric: str = Field("accuracy", description="Success metric")
    min_sample_size: int = Field(1000, description="Minimum sample size")


class ABTestResultsModel(BaseModel):
    """API model for A/B test results."""
    test_id: str
    model_id: str
    control_version: str
    treatment_version: str
    start_time: str
    end_time: Optional[str] = None
    control_metrics: Dict[str, Any]
    treatment_metrics: Dict[str, Any]
    sufficient_sample_size: bool
    active: bool


class MonitoringSummaryModel(BaseModel):
    """API model for monitoring summary."""
    model_id: str
    total_versions: int
    active_version: Optional[str] = None
    unresolved_alerts: int
    recent_performance: Dict[str, Any] = {}
    last_updated: str


@router.post("/versions", response_model=ModelVersionModel)
async def register_model_version(request: RegisterVersionRequest) -> ModelVersionModel:
    """
    Register a new model version.
    
    Args:
        request: Version registration request
        
    Returns:
        Created model version
    """
    try:
        # Get model metadata from export service
        models = export_service.list_exported_models(request.model_id)
        matching_models = [m for m in models if m['version'] == request.version]
        
        if not matching_models:
            raise HTTPException(
                status_code=404,
                detail=f"Exported model {request.model_id}:{request.version} not found"
            )
        
        model_path = matching_models[0]['path']
        _, metadata = export_service.load_exported_model(model_path)
        
        # Register version
        model_version = monitoring_service.register_model_version(
            model_id=request.model_id,
            version=request.version,
            metadata=metadata,
            description=request.description,
            tags=request.tags
        )
        
        return ModelVersionModel(
            model_id=model_version.model_id,
            version=model_version.version,
            status=model_version.status.value,
            created_at=model_version.created_at.isoformat(),
            deployed_at=model_version.deployed_at.isoformat() if model_version.deployed_at else None,
            deprecated_at=model_version.deprecated_at.isoformat() if model_version.deprecated_at else None,
            traffic_percentage=model_version.traffic_percentage,
            description=model_version.description,
            tags=model_version.tags
        )
        
    except AutoMLException as e:
        logger.error(f"Failed to register model version: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error registering model version: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{model_id}/{version}/deploy")
async def deploy_model_version(
    model_id: str,
    version: str,
    request: DeployVersionRequest
):
    """
    Deploy a model version.
    
    Args:
        model_id: Model identifier
        version: Model version
        request: Deployment request
        
    Returns:
        Deployment confirmation
    """
    try:
        monitoring_service.deploy_model_version(
            model_id=model_id,
            version=version,
            traffic_percentage=request.traffic_percentage
        )
        
        return {
            "message": f"Model {model_id}:{version} deployed with {request.traffic_percentage}% traffic"
        }
        
    except AutoMLException as e:
        logger.error(f"Failed to deploy model version: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error deploying model version: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{model_id}/{version}/deprecate")
async def deprecate_model_version(model_id: str, version: str):
    """
    Deprecate a model version.
    
    Args:
        model_id: Model identifier
        version: Model version
        
    Returns:
        Deprecation confirmation
    """
    try:
        monitoring_service.deprecate_model_version(model_id, version)
        
        return {"message": f"Model {model_id}:{version} deprecated"}
        
    except AutoMLException as e:
        logger.error(f"Failed to deprecate model version: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error deprecating model version: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{model_id}/versions", response_model=List[ModelVersionModel])
async def get_model_versions(model_id: str) -> List[ModelVersionModel]:
    """
    Get all versions of a model.
    
    Args:
        model_id: Model identifier
        
    Returns:
        List of model versions
    """
    try:
        versions = monitoring_service.get_model_versions(model_id)
        
        return [
            ModelVersionModel(
                model_id=v.model_id,
                version=v.version,
                status=v.status.value,
                created_at=v.created_at.isoformat(),
                deployed_at=v.deployed_at.isoformat() if v.deployed_at else None,
                deprecated_at=v.deprecated_at.isoformat() if v.deprecated_at else None,
                traffic_percentage=v.traffic_percentage,
                description=v.description,
                tags=v.tags
            )
            for v in versions
        ]
        
    except Exception as e:
        logger.error(f"Failed to get model versions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{model_id}/performance", response_model=List[PerformanceSnapshotModel])
async def get_performance_history(
    model_id: str,
    version: Optional[str] = Query(None, description="Specific version"),
    hours: int = Query(24, description="Time window in hours")
) -> List[PerformanceSnapshotModel]:
    """
    Get performance history for a model.
    
    Args:
        model_id: Model identifier
        version: Optional specific version
        hours: Time window in hours
        
    Returns:
        List of performance snapshots
    """
    try:
        from datetime import timedelta
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        history = monitoring_service.get_performance_history(
            model_id=model_id,
            version=version,
            start_time=start_time,
            end_time=end_time
        )
        
        return [
            PerformanceSnapshotModel(
                model_id=snap.model_id,
                version=snap.version,
                timestamp=snap.timestamp.isoformat(),
                accuracy=snap.accuracy,
                precision=snap.precision,
                recall=snap.recall,
                f1_score=snap.f1_score,
                avg_response_time=snap.avg_response_time,
                error_rate=snap.error_rate,
                prediction_count=snap.prediction_count,
                confidence_distribution=snap.confidence_distribution
            )
            for snap in history
        ]
        
    except Exception as e:
        logger.error(f"Failed to get performance history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{model_id}/metrics")
async def get_current_metrics(model_id: str, version: str, window_hours: int = Query(24)):
    """
    Get current performance metrics for a model version.
    
    Args:
        model_id: Model identifier
        version: Model version
        window_hours: Time window in hours
        
    Returns:
        Current performance metrics
    """
    try:
        metrics = monitoring_service.compute_performance_metrics(
            model_id=model_id,
            version=version,
            window_hours=window_hours
        )
        
        return PerformanceSnapshotModel(
            model_id=metrics.model_id,
            version=metrics.version,
            timestamp=metrics.timestamp.isoformat(),
            accuracy=metrics.accuracy,
            precision=metrics.precision,
            recall=metrics.recall,
            f1_score=metrics.f1_score,
            avg_response_time=metrics.avg_response_time,
            error_rate=metrics.error_rate,
            prediction_count=metrics.prediction_count,
            confidence_distribution=metrics.confidence_distribution
        )
        
    except Exception as e:
        logger.error(f"Failed to get current metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/alerts", response_model=List[AlertModel])
async def get_alerts(
    model_id: Optional[str] = Query(None, description="Model ID filter"),
    unresolved_only: bool = Query(True, description="Only unresolved alerts")
) -> List[AlertModel]:
    """
    Get monitoring alerts.
    
    Args:
        model_id: Optional model ID filter
        unresolved_only: Only return unresolved alerts
        
    Returns:
        List of alerts
    """
    try:
        alerts = monitoring_service.get_alerts(
            model_id=model_id,
            unresolved_only=unresolved_only
        )
        
        return [
            AlertModel(
                id=alert.id,
                model_id=alert.model_id,
                version=alert.version,
                alert_type=alert.alert_type.value,
                severity=alert.severity,
                message=alert.message,
                timestamp=alert.timestamp.isoformat(),
                resolved=alert.resolved,
                resolved_at=alert.resolved_at.isoformat() if alert.resolved_at else None,
                metadata=alert.metadata
            )
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """
    Resolve an alert.
    
    Args:
        alert_id: Alert identifier
        
    Returns:
        Resolution confirmation
    """
    try:
        monitoring_service.resolve_alert(alert_id)
        return {"message": f"Alert {alert_id} resolved"}
        
    except AutoMLException as e:
        logger.error(f"Failed to resolve alert: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error resolving alert: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ab-tests", response_model=ABTestResultsModel)
async def create_ab_test(request: ABTestRequest) -> ABTestResultsModel:
    """
    Create an A/B test between two model versions.
    
    Args:
        request: A/B test request
        
    Returns:
        A/B test configuration
    """
    try:
        ab_test = monitoring_service.create_ab_test(
            test_id=request.test_id,
            model_id=request.model_id,
            control_version=request.control_version,
            treatment_version=request.treatment_version,
            traffic_split=request.traffic_split,
            duration_hours=request.duration_hours,
            success_metric=request.success_metric,
            min_sample_size=request.min_sample_size
        )
        
        # Get initial results
        results = monitoring_service.get_ab_test_results(request.test_id)
        
        return ABTestResultsModel(**results)
        
    except AutoMLException as e:
        logger.error(f"Failed to create A/B test: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error creating A/B test: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/ab-tests/{test_id}", response_model=ABTestResultsModel)
async def get_ab_test_results(test_id: str) -> ABTestResultsModel:
    """
    Get A/B test results.
    
    Args:
        test_id: Test identifier
        
    Returns:
        A/B test results
    """
    try:
        results = monitoring_service.get_ab_test_results(test_id)
        return ABTestResultsModel(**results)
        
    except AutoMLException as e:
        logger.error(f"Failed to get A/B test results: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting A/B test results: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{model_id}/summary", response_model=MonitoringSummaryModel)
async def get_monitoring_summary(model_id: str) -> MonitoringSummaryModel:
    """
    Get monitoring summary for a model.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Monitoring summary
    """
    try:
        summary = monitoring_service.get_monitoring_summary(model_id)
        return MonitoringSummaryModel(**summary)
        
    except Exception as e:
        logger.error(f"Failed to get monitoring summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/cleanup")
async def cleanup_old_data():
    """
    Clean up old monitoring data.
    
    Returns:
        Cleanup confirmation
    """
    try:
        monitoring_service.cleanup_old_data()
        return {"message": "Old monitoring data cleaned up"}
        
    except Exception as e:
        logger.error(f"Failed to cleanup old data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def monitoring_health_check():
    """
    Health check endpoint for monitoring service.
    
    Returns:
        Service health status
    """
    try:
        # Get some basic stats
        all_alerts = monitoring_service.get_alerts(unresolved_only=False)
        unresolved_alerts = monitoring_service.get_alerts(unresolved_only=True)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "total_alerts": len(all_alerts),
            "unresolved_alerts": len(unresolved_alerts)
        }
        
    except Exception as e:
        logger.error(f"Monitoring health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }