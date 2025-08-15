"""
API endpoints for monitoring and observability
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from automl_framework.monitoring.metrics import get_metrics_collector, SystemMetrics, ApplicationMetrics
from automl_framework.monitoring.health_checks import get_health_checker, HealthStatus, HealthCheckResult
from automl_framework.monitoring.alerts import get_alert_manager, Alert, AlertSeverity, AlertStatus
from automl_framework.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


# Pydantic models for API responses
class MetricsResponse(BaseModel):
    """Metrics API response"""
    counters: Dict[str, int]
    gauges: Dict[str, float]
    api_requests_per_minute: float
    timestamp: str


class SystemMetricsResponse(BaseModel):
    """System metrics API response"""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    gpu_utilization: List[float]
    gpu_memory_used: List[float]
    gpu_memory_total: List[float]


class HealthCheckResponse(BaseModel):
    """Health check API response"""
    name: str
    status: str
    message: str
    response_time_ms: float
    timestamp: float
    details: Optional[Dict[str, Any]] = None


class HealthSummaryResponse(BaseModel):
    """Health summary API response"""
    overall_status: str
    timestamp: float
    checks: Dict[str, Dict[str, Any]]


class AlertResponse(BaseModel):
    """Alert API response"""
    id: str
    name: str
    severity: str
    status: str
    message: str
    source: str
    timestamp: str
    resolved_at: Optional[str] = None
    acknowledged_at: Optional[str] = None
    acknowledged_by: Optional[str] = None
    metadata: Dict[str, Any]


class AlertSummaryResponse(BaseModel):
    """Alert summary API response"""
    active_alerts_count: int
    recent_alerts_count: int
    severity_breakdown: Dict[str, int]
    alert_channels_count: int
    alert_rules_count: int
    timestamp: str


@router.get("/health", response_model=HealthSummaryResponse)
async def get_health_status():
    """
    Get overall system health status
    """
    try:
        health_checker = get_health_checker()
        await health_checker.run_all_checks()
        summary = health_checker.get_health_summary()
        
        return HealthSummaryResponse(**summary)
        
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get health status")


@router.get("/health/{check_name}", response_model=HealthCheckResponse)
async def get_health_check(check_name: str):
    """
    Get specific health check result
    """
    try:
        health_checker = get_health_checker()
        result = await health_checker.run_check(check_name)
        
        if result is None:
            raise HTTPException(status_code=404, detail=f"Health check '{check_name}' not found")
        
        return HealthCheckResponse(
            name=result.name,
            status=result.status.value,
            message=result.message,
            response_time_ms=result.response_time_ms,
            timestamp=result.timestamp,
            details=result.details
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running health check {check_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to run health check")


@router.get("/metrics", response_model=MetricsResponse)
async def get_current_metrics():
    """
    Get current system metrics
    """
    try:
        metrics_collector = get_metrics_collector()
        metrics = metrics_collector.get_current_metrics()
        
        return MetricsResponse(**metrics)
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")


@router.get("/metrics/system", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """
    Get system resource metrics
    """
    try:
        metrics_collector = get_metrics_collector()
        system_metrics = metrics_collector.get_system_metrics()
        
        return SystemMetricsResponse(
            cpu_percent=system_metrics.cpu_percent,
            memory_percent=system_metrics.memory_percent,
            memory_used_gb=system_metrics.memory_used_gb,
            memory_total_gb=system_metrics.memory_total_gb,
            disk_percent=system_metrics.disk_percent,
            disk_used_gb=system_metrics.disk_used_gb,
            disk_total_gb=system_metrics.disk_total_gb,
            gpu_utilization=system_metrics.gpu_utilization,
            gpu_memory_used=system_metrics.gpu_memory_used,
            gpu_memory_total=system_metrics.gpu_memory_total
        )
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system metrics")


@router.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """
    Get metrics in Prometheus format
    """
    try:
        metrics_collector = get_metrics_collector()
        prometheus_metrics = metrics_collector.export_prometheus_metrics()
        
        return prometheus_metrics
        
    except Exception as e:
        logger.error(f"Error exporting Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to export Prometheus metrics")


@router.get("/metrics/history/{metric_name}")
async def get_metric_history(
    metric_name: str,
    hours: int = Query(1, ge=1, le=24, description="Hours of history to retrieve")
):
    """
    Get metric history for specified time period
    """
    try:
        metrics_collector = get_metrics_collector()
        history = metrics_collector.get_metric_history(metric_name, hours)
        
        return {
            "metric_name": metric_name,
            "hours": hours,
            "data_points": [
                {
                    "value": point.value,
                    "timestamp": point.timestamp.isoformat(),
                    "labels": point.labels
                }
                for point in history
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting metric history for {metric_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metric history")


@router.get("/alerts", response_model=List[AlertResponse])
async def get_active_alerts():
    """
    Get all active alerts
    """
    try:
        alert_manager = get_alert_manager()
        active_alerts = alert_manager.get_active_alerts()
        
        return [
            AlertResponse(
                id=alert.id,
                name=alert.name,
                severity=alert.severity.value,
                status=alert.status.value,
                message=alert.message,
                source=alert.source,
                timestamp=alert.timestamp.isoformat(),
                resolved_at=alert.resolved_at.isoformat() if alert.resolved_at else None,
                acknowledged_at=alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                acknowledged_by=alert.acknowledged_by,
                metadata=alert.metadata
            )
            for alert in active_alerts
        ]
        
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get active alerts")


@router.get("/alerts/history", response_model=List[AlertResponse])
async def get_alert_history(
    hours: int = Query(24, ge=1, le=168, description="Hours of history to retrieve")
):
    """
    Get alert history for specified time period
    """
    try:
        alert_manager = get_alert_manager()
        alert_history = alert_manager.get_alert_history(hours)
        
        return [
            AlertResponse(
                id=alert.id,
                name=alert.name,
                severity=alert.severity.value,
                status=alert.status.value,
                message=alert.message,
                source=alert.source,
                timestamp=alert.timestamp.isoformat(),
                resolved_at=alert.resolved_at.isoformat() if alert.resolved_at else None,
                acknowledged_at=alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                acknowledged_by=alert.acknowledged_by,
                metadata=alert.metadata
            )
            for alert in alert_history
        ]
        
    except Exception as e:
        logger.error(f"Error getting alert history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get alert history")


@router.get("/alerts/summary", response_model=AlertSummaryResponse)
async def get_alert_summary():
    """
    Get alert summary
    """
    try:
        alert_manager = get_alert_manager()
        summary = alert_manager.get_alert_summary()
        
        return AlertSummaryResponse(**summary)
        
    except Exception as e:
        logger.error(f"Error getting alert summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get alert summary")


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, acknowledged_by: str):
    """
    Acknowledge an alert
    """
    try:
        alert_manager = get_alert_manager()
        success = alert_manager.acknowledge_alert(alert_id, acknowledged_by)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")
        
        return {"message": f"Alert {alert_id} acknowledged by {acknowledged_by}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """
    Resolve an alert
    """
    try:
        alert_manager = get_alert_manager()
        success = alert_manager.resolve_alert(alert_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")
        
        return {"message": f"Alert {alert_id} resolved"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve alert")


@router.get("/status")
async def get_service_status():
    """
    Get comprehensive service status
    """
    try:
        # Get health status
        health_checker = get_health_checker()
        await health_checker.run_all_checks()
        health_summary = health_checker.get_health_summary()
        
        # Get metrics
        metrics_collector = get_metrics_collector()
        current_metrics = metrics_collector.get_current_metrics()
        system_metrics = metrics_collector.get_system_metrics()
        
        # Get alert summary
        alert_manager = get_alert_manager()
        alert_summary = alert_manager.get_alert_summary()
        
        return {
            "service": "automl-framework",
            "version": "1.0.0",
            "status": health_summary["overall_status"],
            "timestamp": current_metrics["timestamp"],
            "health": health_summary,
            "metrics": {
                "system": {
                    "cpu_percent": system_metrics.cpu_percent,
                    "memory_percent": system_metrics.memory_percent,
                    "disk_percent": system_metrics.disk_percent,
                    "gpu_count": len(system_metrics.gpu_utilization)
                },
                "application": {
                    "api_requests_per_minute": current_metrics["api_requests_per_minute"],
                    "total_requests": current_metrics["counters"].get("api.requests.total", 0)
                }
            },
            "alerts": alert_summary
        }
        
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get service status")