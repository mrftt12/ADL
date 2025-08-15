"""
FastAPI routes for model serving and inference
"""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from ...services.model_serving import ModelServingService, PredictionRequest, PredictionResponse
from ...services.model_export import ModelExportService
from ...core.exceptions import AutoMLException


logger = logging.getLogger(__name__)

# Initialize services (these would typically be dependency injected)
export_service = ModelExportService()
serving_service = ModelServingService(export_service)

router = APIRouter(prefix="/models", tags=["model-serving"])


# Pydantic models for API
class PredictionRequestModel(BaseModel):
    """API model for prediction requests."""
    model_id: str = Field(..., description="Model identifier")
    version: str = Field(..., description="Model version")
    input_data: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(..., description="Input data for prediction")
    format: str = Field("json", description="Input data format")
    preprocessing: bool = Field(True, description="Apply preprocessing pipeline")
    return_probabilities: bool = Field(False, description="Return prediction probabilities")
    batch_size: Optional[int] = Field(None, description="Batch size for processing")


class PredictionResponseModel(BaseModel):
    """API model for prediction responses."""
    model_id: str
    version: str
    predictions: List[Any]
    probabilities: Optional[List[Any]] = None
    prediction_time: float
    preprocessing_time: float
    inference_time: float
    metadata: Dict[str, Any]


class BatchPredictionRequestModel(BaseModel):
    """API model for batch prediction requests."""
    requests: List[PredictionRequestModel] = Field(..., description="List of prediction requests")


class ModelInfoModel(BaseModel):
    """API model for model information."""
    model_id: str
    version: str
    framework: str
    export_format: str
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    feature_names: Optional[List[str]] = None
    cached: bool
    last_accessed: Optional[str] = None
    access_count: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    export_timestamp: Optional[str] = None


class CacheStatsModel(BaseModel):
    """API model for cache statistics."""
    cached_models: int
    cache_usage_mb: float
    cache_limit_mb: int
    cache_utilization: float
    total_predictions: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    average_inference_time: float


@router.post("/predict", response_model=PredictionResponseModel)
async def predict(request: PredictionRequestModel) -> PredictionResponseModel:
    """
    Make predictions using a deployed model.
    
    Args:
        request: Prediction request
        
    Returns:
        Prediction response
    """
    try:
        # Convert API request to service request
        service_request = PredictionRequest(
            model_id=request.model_id,
            version=request.version,
            input_data=request.input_data,
            format=request.format,
            preprocessing=request.preprocessing,
            return_probabilities=request.return_probabilities,
            batch_size=request.batch_size
        )
        
        # Make prediction
        response = await serving_service.predict_async(service_request)
        
        # Convert service response to API response
        return PredictionResponseModel(
            model_id=response.model_id,
            version=response.version,
            predictions=response.predictions.tolist() if isinstance(response.predictions, np.ndarray) else response.predictions,
            probabilities=response.probabilities.tolist() if isinstance(response.probabilities, np.ndarray) else response.probabilities,
            prediction_time=response.prediction_time,
            preprocessing_time=response.preprocessing_time,
            inference_time=response.inference_time,
            metadata=response.metadata or {}
        )
        
    except AutoMLException as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/predict/batch", response_model=List[PredictionResponseModel])
async def predict_batch(request: BatchPredictionRequestModel) -> List[PredictionResponseModel]:
    """
    Make batch predictions using deployed models.
    
    Args:
        request: Batch prediction request
        
    Returns:
        List of prediction responses
    """
    try:
        # Convert API requests to service requests
        service_requests = [
            PredictionRequest(
                model_id=req.model_id,
                version=req.version,
                input_data=req.input_data,
                format=req.format,
                preprocessing=req.preprocessing,
                return_probabilities=req.return_probabilities,
                batch_size=req.batch_size
            )
            for req in request.requests
        ]
        
        # Make batch predictions
        responses = serving_service.predict_batch(service_requests)
        
        # Convert service responses to API responses
        return [
            PredictionResponseModel(
                model_id=response.model_id,
                version=response.version,
                predictions=response.predictions.tolist() if isinstance(response.predictions, np.ndarray) else response.predictions,
                probabilities=response.probabilities.tolist() if isinstance(response.probabilities, np.ndarray) else response.probabilities,
                prediction_time=response.prediction_time,
                preprocessing_time=response.preprocessing_time,
                inference_time=response.inference_time,
                metadata=response.metadata or {}
            )
            for response in responses
        ]
        
    except AutoMLException as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during batch prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{model_id}/{version}/info", response_model=ModelInfoModel)
async def get_model_info(model_id: str, version: str) -> ModelInfoModel:
    """
    Get information about a deployed model.
    
    Args:
        model_id: Model identifier
        version: Model version
        
    Returns:
        Model information
    """
    try:
        info = serving_service.get_model_info(model_id, version)
        return ModelInfoModel(**info)
        
    except AutoMLException as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/available", response_model=List[Dict[str, Any]])
async def list_available_models() -> List[Dict[str, Any]]:
    """
    List all available models for serving.
    
    Returns:
        List of available models
    """
    try:
        return serving_service.list_available_models()
        
    except Exception as e:
        logger.error(f"Failed to list available models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/cache/stats", response_model=CacheStatsModel)
async def get_cache_stats() -> CacheStatsModel:
    """
    Get cache and serving statistics.
    
    Returns:
        Cache statistics
    """
    try:
        stats = serving_service.get_cache_stats()
        return CacheStatsModel(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{model_id}/{version}/warmup")
async def warmup_model(model_id: str, version: str, background_tasks: BackgroundTasks):
    """
    Warm up a model by loading it into cache.
    
    Args:
        model_id: Model identifier
        version: Model version
        background_tasks: FastAPI background tasks
    """
    try:
        background_tasks.add_task(serving_service.warmup_model, model_id, version)
        return {"message": f"Model {model_id}:{version} warmup started"}
        
    except Exception as e:
        logger.error(f"Failed to warmup model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/cache")
async def clear_cache(model_id: Optional[str] = None, version: Optional[str] = None):
    """
    Clear model cache.
    
    Args:
        model_id: Optional model ID to clear
        version: Optional version to clear
        
    Returns:
        Number of models removed from cache
    """
    try:
        count = serving_service.clear_cache(model_id, version)
        return {"message": f"Cleared {count} models from cache"}
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/predict/csv")
async def predict_csv(
    model_id: str,
    version: str,
    file: bytes,
    preprocessing: bool = True,
    return_probabilities: bool = False
):
    """
    Make predictions from CSV data.
    
    Args:
        model_id: Model identifier
        version: Model version
        file: CSV file bytes
        preprocessing: Apply preprocessing pipeline
        return_probabilities: Return prediction probabilities
        
    Returns:
        Prediction response
    """
    try:
        # Parse CSV data
        import io
        csv_data = pd.read_csv(io.BytesIO(file))
        
        # Convert to list of dictionaries
        input_data = csv_data.to_dict('records')
        
        # Create prediction request
        service_request = PredictionRequest(
            model_id=model_id,
            version=version,
            input_data=input_data,
            format="csv",
            preprocessing=preprocessing,
            return_probabilities=return_probabilities
        )
        
        # Make prediction
        response = await serving_service.predict_async(service_request)
        
        # Convert service response to API response
        return PredictionResponseModel(
            model_id=response.model_id,
            version=response.version,
            predictions=response.predictions.tolist() if isinstance(response.predictions, np.ndarray) else response.predictions,
            probabilities=response.probabilities.tolist() if isinstance(response.probabilities, np.ndarray) else response.probabilities,
            prediction_time=response.prediction_time,
            preprocessing_time=response.preprocessing_time,
            inference_time=response.inference_time,
            metadata=response.metadata or {}
        )
        
    except AutoMLException as e:
        logger.error(f"CSV prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during CSV prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for model serving service.
    
    Returns:
        Service health status
    """
    try:
        stats = serving_service.get_cache_stats()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "cached_models": stats["cached_models"],
            "total_predictions": stats["total_predictions"]
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }