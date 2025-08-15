"""
Model Export Service for AutoML Framework

This service handles model serialization, export to various formats,
and bundling with preprocessing pipelines for deployment.
"""

import os
import json
import pickle
import joblib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd

# Framework-specific imports (will be conditionally imported)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from ..models.data_models import Architecture, TrainingConfig, PerformanceMetrics
from ..core.exceptions import AutoMLException


logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported model export formats."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    PICKLE = "pickle"
    JOBLIB = "joblib"
    TFLITE = "tflite"


class ModelFramework(Enum):
    """Supported ML frameworks."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    UNKNOWN = "unknown"


@dataclass
class ModelMetadata:
    """Metadata for exported models."""
    model_id: str
    model_name: str
    version: str
    framework: ModelFramework
    architecture: Architecture
    training_config: TrainingConfig
    performance_metrics: PerformanceMetrics
    export_format: ExportFormat
    export_timestamp: datetime
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    preprocessing_steps: List[str]
    feature_names: List[str]
    target_names: Optional[List[str]] = None
    model_size_mb: Optional[float] = None
    additional_info: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        data = asdict(self)
        # Convert enums to strings
        data['framework'] = self.framework.value
        data['export_format'] = self.export_format.value
        data['export_timestamp'] = self.export_timestamp.isoformat()
        
        # Convert architecture and other complex objects to serializable format
        if 'architecture' in data and data['architecture']:
            arch_data = data['architecture']
            # Convert LayerType enums to strings
            if 'layers' in arch_data:
                for layer in arch_data['layers']:
                    if 'layer_type' in layer and hasattr(layer['layer_type'], 'value'):
                        layer['layer_type'] = layer['layer_type'].value
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary."""
        data = data.copy()
        data['framework'] = ModelFramework(data['framework'])
        data['export_format'] = ExportFormat(data['export_format'])
        data['export_timestamp'] = datetime.fromisoformat(data['export_timestamp'])
        return cls(**data)


@dataclass
class ExportedModel:
    """Container for exported model and its components."""
    model_path: str
    metadata: ModelMetadata
    preprocessing_pipeline_path: Optional[str] = None
    config_path: Optional[str] = None
    requirements_path: Optional[str] = None


class ModelExportService:
    """Service for exporting trained models in various formats."""
    
    def __init__(self, export_base_dir: str = "models/exports"):
        """
        Initialize model export service.
        
        Args:
            export_base_dir: Base directory for exported models
        """
        self.export_base_dir = Path(export_base_dir)
        self.export_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Check available frameworks
        self.frameworks_available = {
            'pytorch': TORCH_AVAILABLE,
            'tensorflow': TF_AVAILABLE,
            'onnx': ONNX_AVAILABLE
        }
        
        logger.info(f"Model export service initialized. Available frameworks: {self.frameworks_available}")
    
    def export_model(
        self,
        model: Any,
        model_id: str,
        model_name: str,
        version: str,
        architecture: Architecture,
        training_config: TrainingConfig,
        performance_metrics: PerformanceMetrics,
        export_formats: List[ExportFormat],
        preprocessing_pipeline: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        output_shape: Optional[Tuple[int, ...]] = None
    ) -> List[ExportedModel]:
        """
        Export model in specified formats.
        
        Args:
            model: Trained model object
            model_id: Unique model identifier
            model_name: Human-readable model name
            version: Model version
            architecture: Model architecture
            training_config: Training configuration used
            performance_metrics: Model performance metrics
            export_formats: List of formats to export to
            preprocessing_pipeline: Optional preprocessing pipeline
            feature_names: List of feature names
            target_names: List of target names
            input_shape: Model input shape
            output_shape: Model output shape
            
        Returns:
            List of exported model containers
        """
        logger.info(f"Exporting model {model_id} in formats: {[f.value for f in export_formats]}")
        
        # Detect model framework
        framework = self._detect_framework(model)
        
        # Create export directory for this model
        model_export_dir = self.export_base_dir / model_id / version
        model_export_dir.mkdir(parents=True, exist_ok=True)
        
        exported_models = []
        
        for export_format in export_formats:
            try:
                exported_model = self._export_single_format(
                    model=model,
                    model_id=model_id,
                    model_name=model_name,
                    version=version,
                    framework=framework,
                    architecture=architecture,
                    training_config=training_config,
                    performance_metrics=performance_metrics,
                    export_format=export_format,
                    export_dir=model_export_dir,
                    preprocessing_pipeline=preprocessing_pipeline,
                    feature_names=feature_names or [],
                    target_names=target_names,
                    input_shape=input_shape or (),
                    output_shape=output_shape or ()
                )
                exported_models.append(exported_model)
                logger.info(f"Successfully exported model in {export_format.value} format")
                
            except Exception as e:
                logger.error(f"Failed to export model in {export_format.value} format: {e}")
                # Continue with other formats
                continue
        
        if not exported_models:
            raise AutoMLException(
                f"Failed to export model in any of the requested formats: {[f.value for f in export_formats]}",
                "EXPORT_FAILED"
            )
        
        return exported_models
    
    def _detect_framework(self, model: Any) -> ModelFramework:
        """Detect the ML framework of the model."""
        try:
            if TORCH_AVAILABLE:
                import torch
                if isinstance(model, torch.nn.Module):
                    return ModelFramework.PYTORCH
        except (ImportError, TypeError):
            pass
        
        try:
            if TF_AVAILABLE:
                import tensorflow as tf
                if isinstance(model, (tf.keras.Model, tf.Module)):
                    return ModelFramework.TENSORFLOW
        except (ImportError, TypeError):
            pass
        
        if hasattr(model, 'predict') and hasattr(model, 'fit'):
            # Likely sklearn-compatible model
            return ModelFramework.SKLEARN
        else:
            return ModelFramework.UNKNOWN
    
    def _export_single_format(
        self,
        model: Any,
        model_id: str,
        model_name: str,
        version: str,
        framework: ModelFramework,
        architecture: Architecture,
        training_config: TrainingConfig,
        performance_metrics: PerformanceMetrics,
        export_format: ExportFormat,
        export_dir: Path,
        preprocessing_pipeline: Optional[Any],
        feature_names: List[str],
        target_names: Optional[List[str]],
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...]
    ) -> ExportedModel:
        """Export model in a single format."""
        
        # Create format-specific subdirectory
        format_dir = export_dir / export_format.value
        format_dir.mkdir(exist_ok=True)
        
        # Export model based on format
        if export_format == ExportFormat.PYTORCH:
            model_path = self._export_pytorch(model, format_dir, framework)
        elif export_format == ExportFormat.TENSORFLOW:
            model_path = self._export_tensorflow(model, format_dir, framework)
        elif export_format == ExportFormat.ONNX:
            model_path = self._export_onnx(model, format_dir, framework, input_shape)
        elif export_format == ExportFormat.PICKLE:
            model_path = self._export_pickle(model, format_dir)
        elif export_format == ExportFormat.JOBLIB:
            model_path = self._export_joblib(model, format_dir)
        elif export_format == ExportFormat.TFLITE:
            model_path = self._export_tflite(model, format_dir, framework)
        else:
            raise AutoMLException(f"Unsupported export format: {export_format.value}", "UNSUPPORTED_FORMAT")
        
        # Export preprocessing pipeline if provided
        preprocessing_pipeline_path = None
        if preprocessing_pipeline is not None:
            preprocessing_pipeline_path = self._export_preprocessing_pipeline(
                preprocessing_pipeline, format_dir
            )
        
        # Calculate model size
        model_size_mb = self._calculate_model_size(model_path)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            version=version,
            framework=framework,
            architecture=architecture,
            training_config=training_config,
            performance_metrics=performance_metrics,
            export_format=export_format,
            export_timestamp=datetime.now(),
            input_shape=input_shape,
            output_shape=output_shape,
            preprocessing_steps=self._get_preprocessing_steps(preprocessing_pipeline),
            feature_names=feature_names,
            target_names=target_names,
            model_size_mb=model_size_mb
        )
        
        # Save metadata
        metadata_path = format_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Create requirements file
        requirements_path = self._create_requirements_file(format_dir, framework, export_format)
        
        return ExportedModel(
            model_path=str(model_path),
            metadata=metadata,
            preprocessing_pipeline_path=preprocessing_pipeline_path,
            config_path=str(metadata_path),
            requirements_path=str(requirements_path)
        )
    
    def _export_pytorch(self, model: Any, export_dir: Path, framework: ModelFramework) -> str:
        """Export PyTorch model."""
        if not TORCH_AVAILABLE:
            raise AutoMLException("PyTorch not available for export", "FRAMEWORK_UNAVAILABLE")
        
        if framework != ModelFramework.PYTORCH:
            raise AutoMLException("Model is not a PyTorch model", "FRAMEWORK_MISMATCH")
        
        model_path = export_dir / "model.pth"
        
        # Save model state dict
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'model_config': getattr(model, 'config', {})
        }, model_path)
        
        return str(model_path)
    
    def _export_tensorflow(self, model: Any, export_dir: Path, framework: ModelFramework) -> str:
        """Export TensorFlow model."""
        if not TF_AVAILABLE:
            raise AutoMLException("TensorFlow not available for export", "FRAMEWORK_UNAVAILABLE")
        
        if framework != ModelFramework.TENSORFLOW:
            raise AutoMLException("Model is not a TensorFlow model", "FRAMEWORK_MISMATCH")
        
        model_path = export_dir / "model"
        
        # Save in SavedModel format
        tf.saved_model.save(model, str(model_path))
        
        return str(model_path)
    
    def _export_onnx(self, model: Any, export_dir: Path, framework: ModelFramework, input_shape: Tuple[int, ...]) -> str:
        """Export model to ONNX format."""
        if not ONNX_AVAILABLE:
            raise AutoMLException("ONNX not available for export", "FRAMEWORK_UNAVAILABLE")
        
        model_path = export_dir / "model.onnx"
        
        if framework == ModelFramework.PYTORCH and TORCH_AVAILABLE:
            # Export PyTorch model to ONNX
            dummy_input = torch.randn(1, *input_shape)
            torch.onnx.export(
                model,
                dummy_input,
                str(model_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
        elif framework == ModelFramework.TENSORFLOW and TF_AVAILABLE:
            # Export TensorFlow model to ONNX (requires tf2onnx)
            try:
                import tf2onnx
                spec = (tf.TensorSpec((None, *input_shape), tf.float32, name="input"),)
                output_path = str(model_path)
                model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=11)
                with open(output_path, "wb") as f:
                    f.write(model_proto.SerializeToString())
            except ImportError:
                raise AutoMLException("tf2onnx required for TensorFlow to ONNX conversion", "DEPENDENCY_MISSING")
        else:
            raise AutoMLException(f"ONNX export not supported for framework: {framework.value}", "UNSUPPORTED_CONVERSION")
        
        return str(model_path)
    
    def _export_pickle(self, model: Any, export_dir: Path) -> str:
        """Export model using pickle."""
        model_path = export_dir / "model.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return str(model_path)
    
    def _export_joblib(self, model: Any, export_dir: Path) -> str:
        """Export model using joblib."""
        model_path = export_dir / "model.joblib"
        
        joblib.dump(model, model_path)
        
        return str(model_path)
    
    def _export_tflite(self, model: Any, export_dir: Path, framework: ModelFramework) -> str:
        """Export model to TensorFlow Lite format."""
        if not TF_AVAILABLE:
            raise AutoMLException("TensorFlow not available for TFLite export", "FRAMEWORK_UNAVAILABLE")
        
        if framework != ModelFramework.TENSORFLOW:
            raise AutoMLException("TFLite export only supported for TensorFlow models", "FRAMEWORK_MISMATCH")
        
        model_path = export_dir / "model.tflite"
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        with open(model_path, 'wb') as f:
            f.write(tflite_model)
        
        return str(model_path)
    
    def _export_preprocessing_pipeline(self, pipeline: Any, export_dir: Path) -> str:
        """Export preprocessing pipeline."""
        pipeline_path = export_dir / "preprocessing_pipeline.joblib"
        
        try:
            joblib.dump(pipeline, pipeline_path)
        except Exception as e:
            logger.warning(f"Failed to export preprocessing pipeline with joblib: {e}")
            # Fallback to pickle
            pipeline_path = export_dir / "preprocessing_pipeline.pkl"
            with open(pipeline_path, 'wb') as f:
                pickle.dump(pipeline, f)
        
        return str(pipeline_path)
    
    def _calculate_model_size(self, model_path: str) -> float:
        """Calculate model size in MB."""
        try:
            if os.path.isfile(model_path):
                size_bytes = os.path.getsize(model_path)
            else:
                # For directories (like TensorFlow SavedModel)
                size_bytes = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(model_path)
                    for filename in filenames
                )
            return size_bytes / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.warning(f"Failed to calculate model size: {e}")
            return None
    
    def _get_preprocessing_steps(self, pipeline: Any) -> List[str]:
        """Extract preprocessing steps from pipeline."""
        if pipeline is None:
            return []
        
        try:
            if hasattr(pipeline, 'steps') and pipeline.steps:
                # sklearn Pipeline
                return [step[0] for step in pipeline.steps]
            elif hasattr(pipeline, 'transformers') and pipeline.transformers:
                # Custom pipeline
                return [getattr(t, '__class__', type(t)).__name__ for t in pipeline.transformers]
            else:
                return [getattr(pipeline, '__class__', type(pipeline)).__name__]
        except Exception:
            return ["unknown_preprocessing"]
    
    def _create_requirements_file(self, export_dir: Path, framework: ModelFramework, export_format: ExportFormat) -> str:
        """Create requirements.txt file for the exported model."""
        requirements_path = export_dir / "requirements.txt"
        
        requirements = []
        
        # Base requirements
        requirements.append("numpy")
        requirements.append("pandas")
        
        # Framework-specific requirements
        if framework == ModelFramework.PYTORCH:
            requirements.append("torch")
            requirements.append("torchvision")
        elif framework == ModelFramework.TENSORFLOW:
            requirements.append("tensorflow")
        elif framework == ModelFramework.SKLEARN:
            requirements.append("scikit-learn")
        
        # Format-specific requirements
        if export_format == ExportFormat.ONNX:
            requirements.append("onnx")
            requirements.append("onnxruntime")
        elif export_format == ExportFormat.JOBLIB:
            requirements.append("joblib")
        
        # Additional common requirements
        requirements.extend([
            "scipy",
            "matplotlib",
            "seaborn"
        ])
        
        with open(requirements_path, 'w') as f:
            for req in sorted(set(requirements)):
                f.write(f"{req}\n")
        
        return str(requirements_path)
    
    def load_exported_model(self, model_path: str) -> Tuple[Any, ModelMetadata]:
        """
        Load an exported model and its metadata.
        
        Args:
            model_path: Path to the exported model directory or file
            
        Returns:
            Tuple of (model, metadata)
        """
        model_path = Path(model_path)
        
        # Find metadata file
        if model_path.is_dir():
            metadata_path = model_path / "metadata.json"
        else:
            metadata_path = model_path.parent / "metadata.json"
        
        if not metadata_path.exists():
            raise AutoMLException(f"Metadata file not found: {metadata_path}", "METADATA_NOT_FOUND")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        metadata = ModelMetadata.from_dict(metadata_dict)
        
        # Load model based on format
        if metadata.export_format == ExportFormat.PYTORCH:
            model = self._load_pytorch_model(model_path, metadata)
        elif metadata.export_format == ExportFormat.TENSORFLOW:
            model = self._load_tensorflow_model(model_path, metadata)
        elif metadata.export_format == ExportFormat.ONNX:
            model = self._load_onnx_model(model_path, metadata)
        elif metadata.export_format == ExportFormat.PICKLE:
            model = self._load_pickle_model(model_path, metadata)
        elif metadata.export_format == ExportFormat.JOBLIB:
            model = self._load_joblib_model(model_path, metadata)
        else:
            raise AutoMLException(f"Unsupported model format for loading: {metadata.export_format.value}", "UNSUPPORTED_FORMAT")
        
        return model, metadata
    
    def _load_pytorch_model(self, model_path: Path, metadata: ModelMetadata) -> Any:
        """Load PyTorch model."""
        if not TORCH_AVAILABLE:
            raise AutoMLException("PyTorch not available for loading", "FRAMEWORK_UNAVAILABLE")
        
        if model_path.is_dir():
            model_file = model_path / "model.pth"
        else:
            model_file = model_path
        
        checkpoint = torch.load(model_file, map_location='cpu')
        
        # Note: This is a simplified loader. In practice, you'd need to reconstruct
        # the model architecture and then load the state dict
        return checkpoint
    
    def _load_tensorflow_model(self, model_path: Path, metadata: ModelMetadata) -> Any:
        """Load TensorFlow model."""
        if not TF_AVAILABLE:
            raise AutoMLException("TensorFlow not available for loading", "FRAMEWORK_UNAVAILABLE")
        
        if model_path.is_dir():
            model_dir = model_path / "model"
        else:
            model_dir = model_path
        
        return tf.saved_model.load(str(model_dir))
    
    def _load_onnx_model(self, model_path: Path, metadata: ModelMetadata) -> Any:
        """Load ONNX model."""
        if not ONNX_AVAILABLE:
            raise AutoMLException("ONNX not available for loading", "FRAMEWORK_UNAVAILABLE")
        
        if model_path.is_dir():
            model_file = model_path / "model.onnx"
        else:
            model_file = model_path
        
        return ort.InferenceSession(str(model_file))
    
    def _load_pickle_model(self, model_path: Path, metadata: ModelMetadata) -> Any:
        """Load pickled model."""
        if model_path.is_dir():
            model_file = model_path / "model.pkl"
        else:
            model_file = model_path
        
        with open(model_file, 'rb') as f:
            return pickle.load(f)
    
    def _load_joblib_model(self, model_path: Path, metadata: ModelMetadata) -> Any:
        """Load joblib model."""
        if model_path.is_dir():
            model_file = model_path / "model.joblib"
        else:
            model_file = model_path
        
        return joblib.load(model_file)
    
    def list_exported_models(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all exported models or models for a specific ID.
        
        Args:
            model_id: Optional model ID to filter by
            
        Returns:
            List of model information dictionaries
        """
        models = []
        
        search_dirs = [self.export_base_dir / model_id] if model_id else list(self.export_base_dir.iterdir())
        
        for model_dir in search_dirs:
            if not model_dir.is_dir():
                continue
            
            for version_dir in model_dir.iterdir():
                if not version_dir.is_dir():
                    continue
                
                for format_dir in version_dir.iterdir():
                    if not format_dir.is_dir():
                        continue
                    
                    metadata_path = format_dir / "metadata.json"
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            
                            models.append({
                                'model_id': metadata['model_id'],
                                'model_name': metadata['model_name'],
                                'version': metadata['version'],
                                'framework': metadata['framework'],
                                'export_format': metadata['export_format'],
                                'export_timestamp': metadata['export_timestamp'],
                                'model_size_mb': metadata.get('model_size_mb'),
                                'path': str(format_dir)
                            })
                        except Exception as e:
                            logger.warning(f"Failed to read metadata from {metadata_path}: {e}")
        
        return sorted(models, key=lambda x: x['export_timestamp'], reverse=True)