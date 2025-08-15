"""
Resource management routes for AutoML Framework API.

This module provides endpoints for monitoring system resources and job status.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from automl_framework.api.auth import get_current_user, User
from automl_framework.core.registry import get_service_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/resources", tags=["resources"])

class ResourceStatusResponse(BaseModel):
    system_resources: Dict[str, Any]
    running_jobs: int
    queued_jobs: int
    total_jobs_completed: int
    resource_utilization: Dict[str, float]

@router.get("/status", response_model=ResourceStatusResponse)
async def get_resource_status(current_user: User = Depends(get_current_user)):
    """
    Get current system resource status and utilization.
    """
    try:
        registry = get_service_registry()
        resource_scheduler = registry.get_service('resource_scheduler')
        
        status = resource_scheduler.get_resource_status()
        
        return ResourceStatusResponse(
            system_resources=status["system_resources"],
            running_jobs=status["running_jobs"],
            queued_jobs=status["queued_jobs"],
            total_jobs_completed=status["total_jobs_completed"],
            resource_utilization=status["resource_utilization"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get resource status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get resource status: {str(e)}")

@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get status of a specific resource job.
    """
    try:
        registry = get_service_registry()
        resource_scheduler = registry.get_service('resource_scheduler')
        
        job_status = resource_scheduler.get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return job_status
        
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@router.get("/jobs")
async def list_jobs(
    status: str = None,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user)
):
    """
    List all resource jobs with optional filtering.
    """
    try:
        registry = get_service_registry()
        resource_scheduler = registry.get_service('resource_scheduler')
        
        # Get resource status which includes job information
        resource_status = resource_scheduler.get_resource_status()
        
        # For now, return basic job information
        # In a full implementation, this would query job history
        return {
            "jobs": [],
            "total": 0,
            "running_jobs": resource_status["running_jobs"],
            "queued_jobs": resource_status["queued_jobs"],
            "completed_jobs": resource_status["total_jobs_completed"]
        }
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")

@router.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Cancel a running or queued job.
    """
    try:
        registry = get_service_registry()
        resource_scheduler = registry.get_service('resource_scheduler')
        
        success = resource_scheduler.cancel_job(job_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
        
        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": "Job cancelled successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

@router.get("/metrics")
async def get_resource_metrics(current_user: User = Depends(get_current_user)):
    """
    Get detailed resource metrics and statistics.
    """
    try:
        registry = get_service_registry()
        resource_scheduler = registry.get_service('resource_scheduler')
        
        status = resource_scheduler.get_resource_status()
        
        return {
            "timestamp": status["system_resources"].get("last_updated"),
            "cpu": {
                "total_cores": status["system_resources"].get("total_cpu_cores", 0),
                "available_cores": status["system_resources"].get("available_cpu_cores", 0),
                "utilization_percent": status["resource_utilization"].get("cpu", 0)
            },
            "memory": {
                "total_gb": status["system_resources"].get("total_memory_gb", 0),
                "available_gb": status["system_resources"].get("available_memory_gb", 0),
                "utilization_percent": status["resource_utilization"].get("memory", 0)
            },
            "gpu": {
                "total_count": status["system_resources"].get("total_gpu_count", 0),
                "available_count": status["system_resources"].get("available_gpu_count", 0),
                "utilization_percent": status["resource_utilization"].get("gpu", 0),
                "gpu_info": status["system_resources"].get("gpu_info", [])
            },
            "storage": {
                "total_gb": status["system_resources"].get("total_storage_gb", 0),
                "available_gb": status["system_resources"].get("available_storage_gb", 0),
                "utilization_percent": status["resource_utilization"].get("storage", 0)
            },
            "jobs": {
                "running": status["running_jobs"],
                "queued": status["queued_jobs"],
                "completed": status["total_jobs_completed"]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get resource metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get resource metrics: {str(e)}")