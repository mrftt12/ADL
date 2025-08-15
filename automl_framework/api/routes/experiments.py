"""
Experiment management routes for AutoML Framework API.

This module provides endpoints for creating, managing, and monitoring experiments.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from automl_framework.api.auth import get_current_active_user, User
from automl_framework.core.registry import get_service_registry
from automl_framework.core.exceptions import ExperimentError, ValidationError
from automl_framework.models.data_models import ExperimentStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])

class ExperimentCreateRequest(BaseModel):
    name: str = Field(..., description="Experiment name")
    dataset_path: Optional[str] = Field(None, description="Path to dataset file")
    task_type: Optional[str] = Field("classification", description="ML task type")
    data_type: Optional[str] = Field("tabular", description="Data type")
    target_column: Optional[str] = Field(None, description="Target column name")
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")

class ExperimentResponse(BaseModel):
    id: str
    name: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    progress: Dict[str, float] = Field(default_factory=dict)
    results: Optional[Dict[str, Any]] = None

class ExperimentListResponse(BaseModel):
    experiments: List[ExperimentResponse]
    total: int

@router.post("", response_model=ExperimentResponse)
async def create_experiment(
    request: ExperimentCreateRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Create a new AutoML experiment.
    """
    try:
        registry = get_service_registry()
        experiment_manager = registry.get_service('experiment_manager')
        
        # Prepare experiment configuration
        config = {
            "name": request.name,
            "dataset_path": request.dataset_path,
            "task_type": request.task_type,
            "data_type": request.data_type,
            "target_column": request.target_column,
            "user_id": current_user.id,
            **request.config
        }
        
        # Validate dataset path
        if request.dataset_path:
            import os
            from pathlib import Path
            
            if not os.path.exists(request.dataset_path):
                # Try to find in uploads directory
                upload_dir = Path("data/uploads")
                dataset_files = list(upload_dir.glob(f"{request.dataset_path}.*"))
                if dataset_files:
                    config["dataset_path"] = str(dataset_files[0])
                else:
                    raise HTTPException(status_code=404, detail="Dataset file not found")
        
        # Create experiment
        experiment_id = experiment_manager.create_experiment(
            dataset_path=config["dataset_path"],
            experiment_config=config
        )
        
        # Get experiment details
        experiment_data = experiment_manager.get_experiment_results(experiment_id)
        
        return ExperimentResponse(
            id=experiment_id,
            name=request.name,
            status=experiment_data["status"],
            created_at=experiment_data.get("created_at", datetime.now().isoformat()),
            progress=experiment_data.get("progress", {})
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ExperimentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Experiment creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create experiment: {str(e)}")

@router.post("/{experiment_id}/start")
async def start_experiment(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """
    Start running an AutoML experiment.
    """
    try:
        registry = get_service_registry()
        experiment_manager = registry.get_service('experiment_manager')
        
        # Start experiment
        success = experiment_manager.run_experiment(experiment_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to start experiment")
        
        return {
            "experiment_id": experiment_id,
            "status": "started",
            "message": "Experiment started successfully"
        }
        
    except ExperimentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start experiment: {str(e)}")

@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get experiment details and results.
    """
    try:
        registry = get_service_registry()
        experiment_manager = registry.get_service('experiment_manager')
        
        experiment_data = experiment_manager.get_experiment_results(experiment_id)
        
        return ExperimentResponse(
            id=experiment_id,
            name=experiment_data.get("name", f"Experiment {experiment_id[:8]}"),
            status=experiment_data["status"],
            created_at=experiment_data.get("created_at", ""),
            completed_at=experiment_data.get("completed_at"),
            progress=experiment_data.get("progress", {}),
            results=experiment_data.get("results")
        )
        
    except ExperimentError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get experiment: {str(e)}")

@router.get("", response_model=ExperimentListResponse)
async def list_experiments(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_active_user)
):
    """
    List all experiments with optional filtering.
    """
    try:
        registry = get_service_registry()
        experiment_manager = registry.get_service('experiment_manager')
        
        # Parse status filter
        status_filter = None
        if status:
            try:
                status_filter = ExperimentStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        experiments = experiment_manager.list_experiments(status_filter)
        
        # Apply pagination
        total = len(experiments)
        experiments = experiments[offset:offset + limit]
        
        experiment_responses = [
            ExperimentResponse(
                id=exp["id"],
                name=exp["name"],
                status=exp["status"],
                created_at=exp["created_at"],
                completed_at=exp.get("completed_at"),
                progress=exp.get("progress", {})
            )
            for exp in experiments
        ]
        
        return ExperimentListResponse(
            experiments=experiment_responses,
            total=total
        )
        
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list experiments: {str(e)}")

@router.delete("/{experiment_id}")
async def cancel_experiment(
    experiment_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Cancel a running experiment.
    """
    try:
        registry = get_service_registry()
        experiment_manager = registry.get_service('experiment_manager')
        
        success = experiment_manager.cancel_experiment(experiment_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Experiment not found or cannot be cancelled")
        
        return {
            "experiment_id": experiment_id,
            "status": "cancelled",
            "message": "Experiment cancelled successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to cancel experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel experiment: {str(e)}")

@router.get("/{experiment_id}/progress")
async def get_experiment_progress(
    experiment_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get detailed progress information for an experiment.
    """
    try:
        registry = get_service_registry()
        experiment_manager = registry.get_service('experiment_manager')
        
        progress = experiment_manager.get_experiment_progress(experiment_id)
        
        return {
            "experiment_id": experiment_id,
            "progress": progress,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get progress for experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get progress: {str(e)}")