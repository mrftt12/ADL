"""
Dataset management routes for AutoML Framework API.

This module provides endpoints for dataset upload, analysis, and management.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from automl_framework.api.auth import get_current_active_user, User
from automl_framework.core.registry import get_service_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

class DatasetMetadataResponse(BaseModel):
    dataset_id: str
    filename: str
    file_path: str
    size_bytes: int
    metadata: Dict[str, Any]

class DatasetAnalysisResponse(BaseModel):
    dataset_id: str
    metadata: Dict[str, Any]

@router.post("/upload", response_model=DatasetMetadataResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user)
):
    """
    Upload a dataset file for analysis and preprocessing.
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size (limit to 100MB)
        max_size = 100 * 1024 * 1024  # 100MB
        content = await file.read()
        if len(content) > max_size:
            raise HTTPException(status_code=413, detail="File too large (max 100MB)")
        
        # Create upload directory
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        filename = f"{file_id}{file_extension}"
        file_path = upload_dir / filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Analyze dataset
        registry = get_service_registry()
        data_processor = registry.get_service('data_processor')
        
        try:
            metadata = data_processor.analyze_dataset(str(file_path))
            
            return DatasetMetadataResponse(
                dataset_id=file_id,
                filename=file.filename,
                file_path=str(file_path),
                size_bytes=len(content),
                metadata={
                    "name": metadata.name,
                    "data_type": metadata.data_type.value,
                    "task_type": metadata.task_type.value,
                    "size": metadata.size,
                    "features": [
                        {
                            "name": f.name,
                            "data_type": f.data_type,
                            "is_categorical": f.is_categorical,
                            "unique_values": f.unique_values,
                            "missing_percentage": f.missing_percentage
                        }
                        for f in metadata.features
                    ],
                    "statistics": metadata.statistics
                }
            )
        except Exception as e:
            # Clean up file if analysis fails
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=400, detail=f"Dataset analysis failed: {str(e)}")
            
    except Exception as e:
        logger.error(f"Dataset upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/{dataset_id}/analyze", response_model=DatasetAnalysisResponse)
async def analyze_dataset(
    dataset_id: str,
    target_column: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """
    Analyze an uploaded dataset and return metadata.
    """
    try:
        # Find dataset file
        upload_dir = Path("data/uploads")
        dataset_files = list(upload_dir.glob(f"{dataset_id}.*"))
        
        if not dataset_files:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset_path = dataset_files[0]
        
        # Analyze dataset
        registry = get_service_registry()
        data_processor = registry.get_service('data_processor')
        metadata = data_processor.analyze_dataset(str(dataset_path), target_column)
        
        return DatasetAnalysisResponse(
            dataset_id=dataset_id,
            metadata={
                "name": metadata.name,
                "data_type": metadata.data_type.value,
                "task_type": metadata.task_type.value,
                "size": metadata.size,
                "target_column": metadata.target_column,
                "class_distribution": metadata.class_distribution,
                "features": [
                    {
                        "name": f.name,
                        "data_type": f.data_type,
                        "is_categorical": f.is_categorical,
                        "unique_values": f.unique_values,
                        "missing_percentage": f.missing_percentage,
                        "statistics": f.statistics
                    }
                    for f in metadata.features
                ],
                "statistics": metadata.statistics
            }
        )
        
    except Exception as e:
        logger.error(f"Dataset analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/{dataset_id}")
async def get_dataset_info(
    dataset_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get information about an uploaded dataset.
    """
    try:
        # Find dataset file
        upload_dir = Path("data/uploads")
        dataset_files = list(upload_dir.glob(f"{dataset_id}.*"))
        
        if not dataset_files:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset_path = dataset_files[0]
        
        return {
            "dataset_id": dataset_id,
            "filename": dataset_path.name,
            "file_path": str(dataset_path),
            "size_bytes": dataset_path.stat().st_size,
            "created_at": dataset_path.stat().st_ctime,
            "modified_at": dataset_path.stat().st_mtime
        }
        
    except Exception as e:
        logger.error(f"Failed to get dataset info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dataset info: {str(e)}")

@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete an uploaded dataset.
    """
    try:
        # Find dataset file
        upload_dir = Path("data/uploads")
        dataset_files = list(upload_dir.glob(f"{dataset_id}.*"))
        
        if not dataset_files:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Delete file
        for dataset_path in dataset_files:
            dataset_path.unlink()
        
        return {
            "dataset_id": dataset_id,
            "message": "Dataset deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to delete dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")

@router.get("")
async def list_datasets(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_active_user)
):
    """
    List all uploaded datasets.
    """
    try:
        upload_dir = Path("data/uploads")
        if not upload_dir.exists():
            return {"datasets": [], "total": 0}
        
        # Get all dataset files
        dataset_files = list(upload_dir.glob("*"))
        dataset_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        
        # Apply pagination
        total = len(dataset_files)
        dataset_files = dataset_files[offset:offset + limit]
        
        datasets = []
        for file_path in dataset_files:
            dataset_id = file_path.stem
            datasets.append({
                "dataset_id": dataset_id,
                "filename": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "created_at": file_path.stat().st_ctime,
                "modified_at": file_path.stat().st_mtime
            })
        
        return {
            "datasets": datasets,
            "total": total
        }
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")