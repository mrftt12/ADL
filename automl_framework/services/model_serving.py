"""
Model Serving Service for AutoML Framework

This service handles model loading, caching, and inference for deployed models.
Provides both real-time and batch prediction capabilities.
"""

import os
import json
import pickle
import joblib
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# Framework-specific imports (will be conditionally imported)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from .model_export import ModelExportService, ModelMetadata, ExportFormat, ModelFramework
from ..core.exceptions import AutoMLException


logger = logging.getLogger(__name__)


@dataclass
class PredictionRequest:
    """Request for model prediction."""
    model_id: str
    version: str
    input_data: Union[Dict[str, Any], List[Dict[str, Any]], np.ndarray, pd.DataFrame]
    format: str = "json"  # json, csv, numpy
    preprocessing: bool = True
    return_probabilities: bool = False
    batch_size: Optional[int] = None


@dataclass
class PredictionResponse:
    """Response from model prediction."""
    model_id: str
    version: str
    predictions: Union[List[Any], np.ndarray]
    probabilities: Optional[Union[List[Any], np.ndarray]] = None
    prediction_time: float = 0.0
    preprocessing_time: float = 0.0
    inference_time: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class ModelCache:
    """Cache entry for loaded models."""
    model: Any
    metadata: ModelMetadata
    preprocessing_pipeline: Optional[Any]
    last_accessed: datetime
    access_count: int = 0
    memory_usage_mb: float = 0.0


class ModelServingService:
    """Service for serving trained models for inference."""
    
    def __init__(
        self,
        model_export_service: ModelExportService,
        cache_size_mb: int = 1024,
        cache_ttl_hours: int = 24,
        max_workers: int = 4
    ):
        """
        Initialize model serving service.
        
        Args:
            model_export_service: Model export service instance
            cache_size_mb: Maximum cache size in MB
            cache_ttl_hours: Cache time-to-live in hours
            max_workers: Maximum number of worker threads for batch processing
        """
        self.export_service = model_export_service
        self.cache_size_mb = cache_size_mb
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.max_workers = max_workers
        
        # Model cache
        self._model_cache: Dict[str, ModelCache] = {}
        self._cache_lock = threading.RLock()
        
        # Thread pool for batch processing
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Statistics
        self._stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_inference_time': 0.0,
            'average_inference_time': 0.0
        }
        
        logger.info(f"Model serving service initialized with cache size: {cache_size_mb}MB, TTL: {cache_ttl_hours}h")
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Make predictions using a deployed model.
        
        Args:
            request: Prediction request
            
        Returns:
            Prediction response
        """
        start_time = datetime.now()
        
        try:
            # Load model from cache or disk
            model_key = f"{request.model_id}:{request.version}"
            model_cache = self._get_cached_model(model_key, request.model_id, request.version)
            
            # Prepare input data
            preprocessing_start = datetime.now()
            processed_input = self._preprocess_input(
                request.input_data,
                model_cache.preprocessing_pipeline if request.preprocessing else None,
                model_cache.metadata
            )
            preprocessing_time = (datetime.now() - preprocessing_start).total_seconds()
            
            # Make predictions
            inference_start = datetime.now()
            predictions, probabilities = self._make_predictions(
                model_cache.model,
                processed_input,
                model_cache.metadata,
                request.return_probabilities,
                request.batch_size
            )
            inference_time = (datetime.now() - inference_start).total_seconds()
            
            # Update statistics
            total_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(total_time)
            
            # Update cache access
            with self._cache_lock:
                model_cache.last_accessed = datetime.now()
                model_cache.access_count += 1
            
            return PredictionResponse(
                model_id=request.model_id,
                version=request.version,
                predictions=predictions,
                probabilities=probabilities if request.return_probabilities else None,
                prediction_time=total_time,
                preprocessing_time=preprocessing_time,
                inference_time=inference_time,
                metadata={
                    'input_shape': getattr(processed_input, 'shape', None),
                    'prediction_count': len(predictions) if isinstance(predictions, (list, np.ndarray)) else 1
                }
            )
            
        except Exception as e:
            logger.error(f"Prediction failed for model {request.model_id}:{request.version}: {e}")
            raise AutoMLException(f"Prediction failed: {e}", "PREDICTION_FAILED")
    
    async def predict_async(self, request: PredictionRequest) -> PredictionResponse:
        """
        Make asynchronous predictions using a deployed model.
        
        Args:
            request: Prediction request
            
        Returns:
            Prediction response
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.predict, request)
    
    def predict_batch(self, requests: List[PredictionRequest]) -> List[PredictionResponse]:
        """
        Make batch predictions for multiple requests.
        
        Args:
            requests: List of prediction requests
            
        Returns:
            List of prediction responses
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.predict, request) for request in requests]
            return [future.result() for future in futures]
    
    def _get_cached_model(self, model_key: str, model_id: str, version: str) -> ModelCache:
        """Get model from cache or load from disk."""
        with self._cache_lock:
            # Check if model is in cache and not expired
            if model_key in self._model_cache:
                cache_entry = self._model_cache[model_key]
                if datetime.now() - cache_entry.last_accessed < self.cache_ttl:
                    self._stats['cache_hits'] += 1
                    return cache_entry
                else:
                    # Remove expired entry
                    del self._model_cache[model_key]
            
            # Cache miss - load model from disk
            self._stats['cache_misses'] += 1
            
            # Find model path
            model_paths = self.export_service.list_exported_models(model_id)
            matching_models = [m for m in model_paths if m['version'] == version]
            
            if not matching_models:
                raise AutoMLException(f"Model {model_id}:{version} not found", "MODEL_NOT_FOUND")
            
            model_path = matching_models[0]['path']
            
            # Load model and metadata
            model, metadata = self.export_service.load_exported_model(model_path)
            
            # Load preprocessing pipeline if exists
            preprocessing_pipeline = None
            pipeline_path = Path(model_path) / "preprocessing_pipeline.joblib"
            if not pipeline_path.exists():
                pipeline_path = Path(model_path) / "preprocessing_pipeline.pkl"
            
            if pipeline_path.exists():
                try:
                    if pipeline_path.suffix == '.joblib':
                        preprocessing_pipeline = joblib.load(pipeline_path)
                    else:
                        with open(pipeline_path, 'rb') as f:
                            preprocessing_pipeline = pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load preprocessing pipeline: {e}")
            
            # Calculate memory usage (rough estimate)
            memory_usage = self._estimate_memory_usage(model)
            
            # Check if we need to evict models from cache
            self._evict_if_needed(memory_usage)
            
            # Create cache entry
            cache_entry = ModelCache(
                model=model,
                metadata=metadata,
                preprocessing_pipeline=preprocessing_pipeline,
                last_accessed=datetime.now(),
                access_count=1,
                memory_usage_mb=memory_usage
            )
            
            self._model_cache[model_key] = cache_entry
            return cache_entry
    
    def _preprocess_input(
        self,
        input_data: Union[Dict[str, Any], List[Dict[str, Any]], np.ndarray, pd.DataFrame],
        preprocessing_pipeline: Optional[Any],
        metadata: ModelMetadata
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Preprocess input data for model inference."""
        
        # Convert input to appropriate format
        if isinstance(input_data, dict):
            # Single prediction - convert to DataFrame
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            # Multiple predictions - convert to DataFrame
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        elif isinstance(input_data, np.ndarray):
            # Already in numpy format
            if preprocessing_pipeline:
                logger.warning("Preprocessing pipeline provided but input is numpy array")
            return input_data
        else:
            raise AutoMLException(f"Unsupported input data type: {type(input_data)}", "INVALID_INPUT")
        
        # Validate input features
        if metadata.feature_names:
            missing_features = set(metadata.feature_names) - set(df.columns)
            if missing_features:
                raise AutoMLException(f"Missing required features: {missing_features}", "MISSING_FEATURES")
            
            # Reorder columns to match training order
            df = df[metadata.feature_names]
        
        # Apply preprocessing pipeline if provided
        if preprocessing_pipeline:
            try:
                if hasattr(preprocessing_pipeline, 'transform'):
                    processed_data = preprocessing_pipeline.transform(df)
                else:
                    # Custom pipeline
                    processed_data = preprocessing_pipeline(df)
                
                # Convert to numpy if needed
                if isinstance(processed_data, pd.DataFrame):
                    processed_data = processed_data.values
                
                return processed_data
            except Exception as e:
                raise AutoMLException(f"Preprocessing failed: {e}", "PREPROCESSING_FAILED")
        
        return df.values
    
    def _make_predictions(
        self,
        model: Any,
        input_data: Union[np.ndarray, pd.DataFrame],
        metadata: ModelMetadata,
        return_probabilities: bool = False,
        batch_size: Optional[int] = None
    ) -> Tuple[Union[List[Any], np.ndarray], Optional[Union[List[Any], np.ndarray]]]:
        """Make predictions using the loaded model."""
        
        try:
            if metadata.framework == ModelFramework.PYTORCH:
                return self._predict_pytorch(model, input_data, return_probabilities, batch_size)
            elif metadata.framework == ModelFramework.TENSORFLOW:
                return self._predict_tensorflow(model, input_data, return_probabilities, batch_size)
            elif metadata.framework == ModelFramework.SKLEARN:
                return self._predict_sklearn(model, input_data, return_probabilities, batch_size)
            elif metadata.export_format == ExportFormat.ONNX:
                return self._predict_onnx(model, input_data, return_probabilities, batch_size)
            else:
                raise AutoMLException(f"Unsupported model framework: {metadata.framework}", "UNSUPPORTED_FRAMEWORK")
                
        except Exception as e:
            if isinstance(e, AutoMLException):
                raise
            raise AutoMLException(f"Model inference failed: {e}", "INFERENCE_FAILED")
    
    def _predict_pytorch(
        self,
        model: Any,
        input_data: np.ndarray,
        return_probabilities: bool,
        batch_size: Optional[int]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions using PyTorch model."""
        if not TORCH_AVAILABLE:
            raise AutoMLException("PyTorch not available", "FRAMEWORK_UNAVAILABLE")
        
        import torch
        
        # Convert to tensor
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.FloatTensor(input_data)
        else:
            input_tensor = torch.FloatTensor(input_data.values)
        
        # Set model to evaluation mode
        if hasattr(model, 'eval'):
            model.eval()
        
        with torch.no_grad():
            if batch_size and len(input_tensor) > batch_size:
                # Process in batches
                predictions = []
                probabilities = [] if return_probabilities else None
                
                for i in range(0, len(input_tensor), batch_size):
                    batch = input_tensor[i:i + batch_size]
                    output = model(batch)
                    
                    if return_probabilities:
                        probs = torch.softmax(output, dim=1)
                        probabilities.extend(probs.numpy())
                        predictions.extend(torch.argmax(output, dim=1).numpy())
                    else:
                        predictions.extend(torch.argmax(output, dim=1).numpy())
                
                predictions = np.array(predictions)
                probabilities = np.array(probabilities) if probabilities else None
            else:
                # Process all at once
                output = model(input_tensor)
                
                if return_probabilities:
                    probabilities = torch.softmax(output, dim=1).numpy()
                    predictions = torch.argmax(output, dim=1).numpy()
                else:
                    predictions = torch.argmax(output, dim=1).numpy()
                    probabilities = None
        
        return predictions, probabilities
    
    def _predict_tensorflow(
        self,
        model: Any,
        input_data: np.ndarray,
        return_probabilities: bool,
        batch_size: Optional[int]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions using TensorFlow model."""
        if not TF_AVAILABLE:
            raise AutoMLException("TensorFlow not available", "FRAMEWORK_UNAVAILABLE")
        
        import tensorflow as tf
        
        # Convert to tensor
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data.values
        
        input_tensor = tf.constant(input_data, dtype=tf.float32)
        
        # Make predictions
        if batch_size and len(input_tensor) > batch_size:
            # Process in batches
            predictions = []
            probabilities = [] if return_probabilities else None
            
            for i in range(0, len(input_tensor), batch_size):
                batch = input_tensor[i:i + batch_size]
                output = model(batch)
                
                if return_probabilities:
                    probs = tf.nn.softmax(output)
                    probabilities.extend(probs.numpy())
                    predictions.extend(tf.argmax(output, axis=1).numpy())
                else:
                    predictions.extend(tf.argmax(output, axis=1).numpy())
            
            predictions = np.array(predictions)
            probabilities = np.array(probabilities) if probabilities else None
        else:
            # Process all at once
            output = model(input_tensor)
            
            if return_probabilities:
                probabilities = tf.nn.softmax(output).numpy()
                predictions = tf.argmax(output, axis=1).numpy()
            else:
                predictions = tf.argmax(output, axis=1).numpy()
                probabilities = None
        
        return predictions, probabilities
    
    def _predict_sklearn(
        self,
        model: Any,
        input_data: Union[np.ndarray, pd.DataFrame],
        return_probabilities: bool,
        batch_size: Optional[int]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions using scikit-learn model."""
        
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data.values
        
        # Make predictions
        predictions = model.predict(input_data)
        
        # Get probabilities if requested and available
        probabilities = None
        if return_probabilities and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_data)
        
        return predictions, probabilities
    
    def _predict_onnx(
        self,
        model: Any,
        input_data: np.ndarray,
        return_probabilities: bool,
        batch_size: Optional[int]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions using ONNX model."""
        if not ONNX_AVAILABLE:
            raise AutoMLException("ONNX Runtime not available", "FRAMEWORK_UNAVAILABLE")
        
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data.values
        
        # Get input name
        input_name = model.get_inputs()[0].name
        
        # Make predictions
        if batch_size and len(input_data) > batch_size:
            # Process in batches
            predictions = []
            probabilities = [] if return_probabilities else None
            
            for i in range(0, len(input_data), batch_size):
                batch = input_data[i:i + batch_size].astype(np.float32)
                output = model.run(None, {input_name: batch})
                
                if return_probabilities and len(output) > 1:
                    probabilities.extend(output[1])  # Assuming second output is probabilities
                    predictions.extend(np.argmax(output[0], axis=1))
                else:
                    predictions.extend(np.argmax(output[0], axis=1))
            
            predictions = np.array(predictions)
            probabilities = np.array(probabilities) if probabilities else None
        else:
            # Process all at once
            input_data = input_data.astype(np.float32)
            output = model.run(None, {input_name: input_data})
            
            if return_probabilities and len(output) > 1:
                probabilities = output[1]  # Assuming second output is probabilities
                predictions = np.argmax(output[0], axis=1)
            else:
                predictions = np.argmax(output[0], axis=1)
                probabilities = None
        
        return predictions, probabilities
    
    def _estimate_memory_usage(self, model: Any) -> float:
        """Estimate memory usage of a model in MB."""
        try:
            import sys
            return sys.getsizeof(model) / (1024 * 1024)
        except Exception:
            return 50.0  # Default estimate
    
    def _evict_if_needed(self, new_model_size_mb: float) -> None:
        """Evict models from cache if needed to make space."""
        with self._cache_lock:
            current_usage = sum(cache.memory_usage_mb for cache in self._model_cache.values())
            
            if current_usage + new_model_size_mb > self.cache_size_mb:
                # Sort by last accessed time (oldest first)
                sorted_models = sorted(
                    self._model_cache.items(),
                    key=lambda x: x[1].last_accessed
                )
                
                # Evict oldest models until we have enough space
                for model_key, cache_entry in sorted_models:
                    if current_usage + new_model_size_mb <= self.cache_size_mb:
                        break
                    
                    logger.info(f"Evicting model {model_key} from cache")
                    del self._model_cache[model_key]
                    current_usage -= cache_entry.memory_usage_mb
    
    def _update_stats(self, inference_time: float) -> None:
        """Update inference statistics."""
        self._stats['total_predictions'] += 1
        self._stats['total_inference_time'] += inference_time
        self._stats['average_inference_time'] = (
            self._stats['total_inference_time'] / self._stats['total_predictions']
        )
    
    def get_model_info(self, model_id: str, version: str) -> Dict[str, Any]:
        """
        Get information about a deployed model.
        
        Args:
            model_id: Model identifier
            version: Model version
            
        Returns:
            Model information dictionary
        """
        model_key = f"{model_id}:{version}"
        
        # Check if model is in cache
        with self._cache_lock:
            if model_key in self._model_cache:
                cache_entry = self._model_cache[model_key]
                return {
                    'model_id': model_id,
                    'version': version,
                    'framework': cache_entry.metadata.framework.value,
                    'export_format': cache_entry.metadata.export_format.value,
                    'input_shape': cache_entry.metadata.input_shape,
                    'output_shape': cache_entry.metadata.output_shape,
                    'feature_names': cache_entry.metadata.feature_names,
                    'cached': True,
                    'last_accessed': cache_entry.last_accessed.isoformat(),
                    'access_count': cache_entry.access_count,
                    'memory_usage_mb': cache_entry.memory_usage_mb
                }
        
        # Model not in cache - get info from export service
        model_paths = self.export_service.list_exported_models(model_id)
        matching_models = [m for m in model_paths if m['version'] == version]
        
        if not matching_models:
            raise AutoMLException(f"Model {model_id}:{version} not found", "MODEL_NOT_FOUND")
        
        model_info = matching_models[0]
        return {
            'model_id': model_id,
            'version': version,
            'framework': model_info['framework'],
            'export_format': model_info['export_format'],
            'model_size_mb': model_info.get('model_size_mb'),
            'cached': False,
            'export_timestamp': model_info['export_timestamp']
        }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models for serving.
        
        Returns:
            List of available model information
        """
        return self.export_service.list_exported_models()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache and serving statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._cache_lock:
            cached_models = len(self._model_cache)
            total_cache_usage = sum(cache.memory_usage_mb for cache in self._model_cache.values())
            
            return {
                'cached_models': cached_models,
                'cache_usage_mb': total_cache_usage,
                'cache_limit_mb': self.cache_size_mb,
                'cache_utilization': total_cache_usage / self.cache_size_mb if self.cache_size_mb > 0 else 0,
                'total_predictions': self._stats['total_predictions'],
                'cache_hits': self._stats['cache_hits'],
                'cache_misses': self._stats['cache_misses'],
                'cache_hit_rate': (
                    self._stats['cache_hits'] / (self._stats['cache_hits'] + self._stats['cache_misses'])
                    if (self._stats['cache_hits'] + self._stats['cache_misses']) > 0 else 0
                ),
                'average_inference_time': self._stats['average_inference_time']
            }
    
    def clear_cache(self, model_id: Optional[str] = None, version: Optional[str] = None) -> int:
        """
        Clear model cache.
        
        Args:
            model_id: Optional model ID to clear (clears all if None)
            version: Optional version to clear (clears all versions if None)
            
        Returns:
            Number of models removed from cache
        """
        with self._cache_lock:
            if model_id is None:
                # Clear entire cache
                count = len(self._model_cache)
                self._model_cache.clear()
                logger.info(f"Cleared entire model cache ({count} models)")
                return count
            
            # Clear specific model(s)
            keys_to_remove = []
            for key in self._model_cache.keys():
                key_model_id, key_version = key.split(':', 1)
                if key_model_id == model_id and (version is None or key_version == version):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._model_cache[key]
            
            logger.info(f"Cleared {len(keys_to_remove)} models from cache")
            return len(keys_to_remove)
    
    def warmup_model(self, model_id: str, version: str) -> None:
        """
        Warm up a model by loading it into cache.
        
        Args:
            model_id: Model identifier
            version: Model version
        """
        model_key = f"{model_id}:{version}"
        self._get_cached_model(model_key, model_id, version)
        logger.info(f"Warmed up model {model_key}")
    
    def shutdown(self) -> None:
        """Shutdown the serving service and cleanup resources."""
        logger.info("Shutting down model serving service")
        self._executor.shutdown(wait=True)
        self.clear_cache()