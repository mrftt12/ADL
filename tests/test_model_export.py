"""
Unit tests for Model Export Service
"""

import os
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from automl_framework.services.model_export import (
    ModelExportService,
    ExportFormat,
    ModelFramework,
    ModelMetadata,
    ExportedModel
)
from automl_framework.models.data_models import (
    Architecture,
    TrainingConfig,
    PerformanceMetrics,
    Layer,
    LayerType
)
from automl_framework.core.exceptions import AutoMLException


class SimpleTestModel:
    """Simple test model that can be serialized."""
    def __init__(self):
        self.weights = [1, 2, 3]
    
    def predict(self, x):
        return x
    
    def fit(self, x, y):
        pass


class UnknownTestModel:
    """Test model without predict/fit methods."""
    pass


class TestModelExportService:
    """Test cases for ModelExportService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.export_service = ModelExportService(export_base_dir=self.temp_dir)
        
        # Create test data
        self.test_architecture = Architecture(
            id="test_arch",
            layers=[
                Layer(layer_type=LayerType.DENSE, parameters={"units": 64}),
                Layer(layer_type=LayerType.DENSE, parameters={"units": 10})
            ]
        )
        
        self.test_training_config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            optimizer="adam",
            epochs=100
        )
        
        self.test_performance_metrics = PerformanceMetrics(
            accuracy=0.95,
            loss=0.05,
            precision=0.94,
            recall=0.96,
            f1_score=0.95,
            training_time=3600.0,
            inference_time=0.01
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test service initialization."""
        assert self.export_service.export_base_dir == Path(self.temp_dir)
        assert self.export_service.export_base_dir.exists()
        assert isinstance(self.export_service.frameworks_available, dict)
    
    def test_detect_framework_sklearn(self):
        """Test framework detection for sklearn-like models."""
        mock_model = Mock()
        mock_model.predict = Mock()
        mock_model.fit = Mock()
        
        framework = self.export_service._detect_framework(mock_model)
        assert framework == ModelFramework.SKLEARN
    
    def test_detect_framework_unknown(self):
        """Test framework detection for unknown models."""
        mock_model = UnknownTestModel()
        
        framework = self.export_service._detect_framework(mock_model)
        assert framework == ModelFramework.UNKNOWN
    
    @patch('automl_framework.services.model_export.TORCH_AVAILABLE', True)
    @patch('torch.nn.Module')
    def test_detect_framework_pytorch(self, mock_torch_module):
        """Test framework detection for PyTorch models."""
        mock_model = Mock(spec=mock_torch_module)
        
        with patch('torch.nn.Module', mock_torch_module):
            framework = self.export_service._detect_framework(mock_model)
            assert framework == ModelFramework.PYTORCH
    
    def test_export_pickle_format(self):
        """Test exporting model in pickle format."""
        mock_model = SimpleTestModel()
        
        exported_models = self.export_service.export_model(
            model=mock_model,
            model_id="test_model",
            model_name="Test Model",
            version="1.0.0",
            architecture=self.test_architecture,
            training_config=self.test_training_config,
            performance_metrics=self.test_performance_metrics,
            export_formats=[ExportFormat.PICKLE],
            feature_names=["feature1", "feature2"],
            input_shape=(10,),
            output_shape=(1,)
        )
        
        assert len(exported_models) == 1
        exported_model = exported_models[0]
        
        assert isinstance(exported_model, ExportedModel)
        assert exported_model.model_path.endswith("model.pkl")
        assert Path(exported_model.model_path).exists()
        assert exported_model.metadata.export_format == ExportFormat.PICKLE
        assert exported_model.metadata.framework == ModelFramework.SKLEARN
        assert exported_model.config_path.endswith("metadata.json")
        assert Path(exported_model.config_path).exists()
    
    def test_export_joblib_format(self):
        """Test exporting model in joblib format."""
        mock_model = SimpleTestModel()
        
        exported_models = self.export_service.export_model(
            model=mock_model,
            model_id="test_model",
            model_name="Test Model",
            version="1.0.0",
            architecture=self.test_architecture,
            training_config=self.test_training_config,
            performance_metrics=self.test_performance_metrics,
            export_formats=[ExportFormat.JOBLIB],
            feature_names=["feature1", "feature2"]
        )
        
        assert len(exported_models) == 1
        exported_model = exported_models[0]
        
        assert exported_model.model_path.endswith("model.joblib")
        assert Path(exported_model.model_path).exists()
        assert exported_model.metadata.export_format == ExportFormat.JOBLIB
    
    def test_export_multiple_formats(self):
        """Test exporting model in multiple formats."""
        mock_model = SimpleTestModel()
        
        exported_models = self.export_service.export_model(
            model=mock_model,
            model_id="test_model",
            model_name="Test Model",
            version="1.0.0",
            architecture=self.test_architecture,
            training_config=self.test_training_config,
            performance_metrics=self.test_performance_metrics,
            export_formats=[ExportFormat.PICKLE, ExportFormat.JOBLIB],
            feature_names=["feature1", "feature2"]
        )
        
        assert len(exported_models) == 2
        formats = [model.metadata.export_format for model in exported_models]
        assert ExportFormat.PICKLE in formats
        assert ExportFormat.JOBLIB in formats
    
    def test_export_with_preprocessing_pipeline(self):
        """Test exporting model with preprocessing pipeline."""
        mock_model = SimpleTestModel()
        
        # Create a simple serializable pipeline
        class SimplePipeline:
            def __init__(self):
                self.steps = [("scaler", "StandardScaler"), ("encoder", "LabelEncoder")]
        
        mock_pipeline = SimplePipeline()
        
        exported_models = self.export_service.export_model(
            model=mock_model,
            model_id="test_model",
            model_name="Test Model",
            version="1.0.0",
            architecture=self.test_architecture,
            training_config=self.test_training_config,
            performance_metrics=self.test_performance_metrics,
            export_formats=[ExportFormat.PICKLE],
            preprocessing_pipeline=mock_pipeline,
            feature_names=["feature1", "feature2"]
        )
        
        assert len(exported_models) == 1
        exported_model = exported_models[0]
        
        assert exported_model.preprocessing_pipeline_path is not None
        assert Path(exported_model.preprocessing_pipeline_path).exists()
        assert exported_model.metadata.preprocessing_steps == ["scaler", "encoder"]
    
    def test_metadata_serialization(self):
        """Test metadata serialization and deserialization."""
        metadata = ModelMetadata(
            model_id="test_model",
            model_name="Test Model",
            version="1.0.0",
            framework=ModelFramework.SKLEARN,
            architecture=self.test_architecture,
            training_config=self.test_training_config,
            performance_metrics=self.test_performance_metrics,
            export_format=ExportFormat.PICKLE,
            export_timestamp=datetime.now(),
            input_shape=(10,),
            output_shape=(1,),
            preprocessing_steps=["scaler"],
            feature_names=["feature1", "feature2"]
        )
        
        # Test to_dict
        metadata_dict = metadata.to_dict()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict['framework'] == 'sklearn'
        assert metadata_dict['export_format'] == 'pickle'
        assert isinstance(metadata_dict['export_timestamp'], str)
        
        # Test from_dict
        reconstructed_metadata = ModelMetadata.from_dict(metadata_dict)
        assert reconstructed_metadata.model_id == metadata.model_id
        assert reconstructed_metadata.framework == metadata.framework
        assert reconstructed_metadata.export_format == metadata.export_format
    
    def test_calculate_model_size(self):
        """Test model size calculation."""
        # Create a temporary file
        test_file = Path(self.temp_dir) / "test_model.pkl"
        test_content = b"test model content"
        with open(test_file, 'wb') as f:
            f.write(test_content)
        
        size_mb = self.export_service._calculate_model_size(str(test_file))
        expected_size_mb = len(test_content) / (1024 * 1024)
        assert abs(size_mb - expected_size_mb) < 0.001
    
    def test_get_preprocessing_steps_sklearn_pipeline(self):
        """Test extracting preprocessing steps from sklearn pipeline."""
        mock_pipeline = Mock()
        mock_pipeline.steps = [("scaler", Mock()), ("encoder", Mock())]
        
        steps = self.export_service._get_preprocessing_steps(mock_pipeline)
        assert steps == ["scaler", "encoder"]
    
    def test_get_preprocessing_steps_custom_pipeline(self):
        """Test extracting preprocessing steps from custom pipeline."""
        class StandardScaler:
            pass
        
        class LabelEncoder:
            pass
        
        class CustomPipeline:
            def __init__(self):
                self.transformers = [StandardScaler(), LabelEncoder()]
        
        mock_pipeline = CustomPipeline()
        
        steps = self.export_service._get_preprocessing_steps(mock_pipeline)
        assert steps == ["StandardScaler", "LabelEncoder"]
    
    def test_get_preprocessing_steps_none(self):
        """Test extracting preprocessing steps when pipeline is None."""
        steps = self.export_service._get_preprocessing_steps(None)
        assert steps == []
    
    def test_create_requirements_file(self):
        """Test creating requirements file."""
        export_dir = Path(self.temp_dir) / "test_export"
        export_dir.mkdir()
        
        requirements_path = self.export_service._create_requirements_file(
            export_dir, ModelFramework.SKLEARN, ExportFormat.JOBLIB
        )
        
        assert Path(requirements_path).exists()
        
        with open(requirements_path, 'r') as f:
            requirements = f.read()
        
        assert "numpy" in requirements
        assert "pandas" in requirements
        assert "scikit-learn" in requirements
        assert "joblib" in requirements
    
    def test_list_exported_models_empty(self):
        """Test listing exported models when none exist."""
        models = self.export_service.list_exported_models()
        assert models == []
    
    def test_list_exported_models_with_models(self):
        """Test listing exported models."""
        # Create a mock exported model structure
        model_dir = Path(self.temp_dir) / "test_model" / "1.0.0" / "pickle"
        model_dir.mkdir(parents=True)
        
        metadata = {
            "model_id": "test_model",
            "model_name": "Test Model",
            "version": "1.0.0",
            "framework": "sklearn",
            "export_format": "pickle",
            "export_timestamp": "2023-01-01T00:00:00",
            "model_size_mb": 1.5
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        models = self.export_service.list_exported_models()
        assert len(models) == 1
        assert models[0]["model_id"] == "test_model"
        assert models[0]["model_name"] == "Test Model"
        assert models[0]["version"] == "1.0.0"
        assert models[0]["framework"] == "sklearn"
        assert models[0]["export_format"] == "pickle"
    
    def test_load_exported_model_metadata_not_found(self):
        """Test loading exported model when metadata is missing."""
        with pytest.raises(AutoMLException) as exc_info:
            self.export_service.load_exported_model("/nonexistent/path")
        
        assert exc_info.value.error_code == "METADATA_NOT_FOUND"
    
    def test_export_unsupported_format(self):
        """Test exporting with unsupported format."""
        mock_model = Mock()
        
        # Create a mock unsupported format
        class UnsupportedFormat:
            value = 'unsupported'
        
        unsupported_format = UnsupportedFormat()
        
        with pytest.raises(AutoMLException) as exc_info:
            self.export_service._export_single_format(
                model=mock_model,
                model_id="test",
                model_name="Test",
                version="1.0",
                framework=ModelFramework.SKLEARN,
                architecture=self.test_architecture,
                training_config=self.test_training_config,
                performance_metrics=self.test_performance_metrics,
                export_format=unsupported_format,
                export_dir=Path(self.temp_dir),
                preprocessing_pipeline=None,
                feature_names=[],
                target_names=None,
                input_shape=(),
                output_shape=()
            )
        
        assert exc_info.value.error_code == "UNSUPPORTED_FORMAT"
    
    @patch('automl_framework.services.model_export.TORCH_AVAILABLE', False)
    def test_export_pytorch_unavailable(self):
        """Test PyTorch export when PyTorch is not available."""
        mock_model = Mock()
        
        with pytest.raises(AutoMLException) as exc_info:
            self.export_service._export_pytorch(
                mock_model, Path(self.temp_dir), ModelFramework.PYTORCH
            )
        
        assert exc_info.value.error_code == "FRAMEWORK_UNAVAILABLE"
    
    def test_export_pytorch_framework_mismatch(self):
        """Test PyTorch export with wrong framework."""
        mock_model = Mock()
        
        with pytest.raises(AutoMLException) as exc_info:
            self.export_service._export_pytorch(
                mock_model, Path(self.temp_dir), ModelFramework.SKLEARN
            )
        
        assert exc_info.value.error_code == "FRAMEWORK_MISMATCH"
    
    @patch('automl_framework.services.model_export.TF_AVAILABLE', False)
    def test_export_tensorflow_unavailable(self):
        """Test TensorFlow export when TensorFlow is not available."""
        mock_model = Mock()
        
        with pytest.raises(AutoMLException) as exc_info:
            self.export_service._export_tensorflow(
                mock_model, Path(self.temp_dir), ModelFramework.TENSORFLOW
            )
        
        assert exc_info.value.error_code == "FRAMEWORK_UNAVAILABLE"
    
    @patch('automl_framework.services.model_export.ONNX_AVAILABLE', False)
    def test_export_onnx_unavailable(self):
        """Test ONNX export when ONNX is not available."""
        mock_model = Mock()
        
        with pytest.raises(AutoMLException) as exc_info:
            self.export_service._export_onnx(
                mock_model, Path(self.temp_dir), ModelFramework.PYTORCH, (10,)
            )
        
        assert exc_info.value.error_code == "FRAMEWORK_UNAVAILABLE"
    
    def test_export_all_formats_fail(self):
        """Test export when all formats fail."""
        mock_model = Mock()
        
        # Mock all export methods to raise exceptions
        with patch.object(self.export_service, '_export_pickle', side_effect=Exception("Export failed")):
            with pytest.raises(AutoMLException) as exc_info:
                self.export_service.export_model(
                    model=mock_model,
                    model_id="test_model",
                    model_name="Test Model",
                    version="1.0.0",
                    architecture=self.test_architecture,
                    training_config=self.test_training_config,
                    performance_metrics=self.test_performance_metrics,
                    export_formats=[ExportFormat.PICKLE]
                )
        
        assert exc_info.value.error_code == "EXPORT_FAILED"


if __name__ == "__main__":
    pytest.main([__file__])