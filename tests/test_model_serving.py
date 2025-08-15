"""
Unit tests for Model Serving Service
"""

import tempfile
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import pandas as pd

from automl_framework.services.model_serving import (
    ModelServingService,
    PredictionRequest,
    PredictionResponse,
    ModelCache
)
from automl_framework.services.model_export import (
    ModelExportService,
    ModelMetadata,
    ExportFormat,
    ModelFramework
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
    """Simple test model for serving tests."""
    def __init__(self):
        self.weights = np.array([[1, 2], [3, 4]])
    
    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        return np.dot(x, self.weights).sum(axis=1)
    
    def predict_proba(self, x):
        predictions = self.predict(x)
        # Simple softmax-like transformation
        exp_preds = np.exp(predictions - np.max(predictions))
        return exp_preds / exp_preds.sum()


class TestModelServingService:
    """Test cases for ModelServingService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.export_service = ModelExportService(export_base_dir=self.temp_dir)
        self.serving_service = ModelServingService(
            model_export_service=self.export_service,
            cache_size_mb=100,
            cache_ttl_hours=1,
            max_workers=2
        )
        
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
        
        self.test_metadata = ModelMetadata(
            model_id="test_model",
            model_name="Test Model",
            version="1.0.0",
            framework=ModelFramework.SKLEARN,
            architecture=self.test_architecture,
            training_config=self.test_training_config,
            performance_metrics=self.test_performance_metrics,
            export_format=ExportFormat.PICKLE,
            export_timestamp=datetime.now(),
            input_shape=(2,),
            output_shape=(1,),
            preprocessing_steps=[],
            feature_names=["feature1", "feature2"]
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.serving_service.shutdown()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test service initialization."""
        assert self.serving_service.cache_size_mb == 100
        assert self.serving_service.cache_ttl == timedelta(hours=1)
        assert self.serving_service.max_workers == 2
        assert len(self.serving_service._model_cache) == 0
    
    def test_predict_simple(self):
        """Test simple prediction."""
        # Mock the export service to return a test model
        test_model = SimpleTestModel()
        
        with patch.object(self.export_service, 'list_exported_models') as mock_list:
            with patch.object(self.export_service, 'load_exported_model') as mock_load:
                mock_list.return_value = [{'model_id': 'test_model', 'version': '1.0.0', 'path': '/test/path'}]
                mock_load.return_value = (test_model, self.test_metadata)
                
                request = PredictionRequest(
                    model_id="test_model",
                    version="1.0.0",
                    input_data={"feature1": 1.0, "feature2": 2.0},
                    preprocessing=False
                )
                
                response = self.serving_service.predict(request)
                
                assert isinstance(response, PredictionResponse)
                assert response.model_id == "test_model"
                assert response.version == "1.0.0"
                assert len(response.predictions) == 1
                assert response.prediction_time > 0
    
    def test_predict_with_dataframe_input(self):
        """Test prediction with DataFrame input."""
        test_model = SimpleTestModel()
        
        with patch.object(self.export_service, 'list_exported_models') as mock_list:
            with patch.object(self.export_service, 'load_exported_model') as mock_load:
                mock_list.return_value = [{'model_id': 'test_model', 'version': '1.0.0', 'path': '/test/path'}]
                mock_load.return_value = (test_model, self.test_metadata)
                
                # Test with list of dictionaries
                request = PredictionRequest(
                    model_id="test_model",
                    version="1.0.0",
                    input_data=[
                        {"feature1": 1.0, "feature2": 2.0},
                        {"feature1": 3.0, "feature2": 4.0}
                    ],
                    preprocessing=False
                )
                
                response = self.serving_service.predict(request)
                
                assert len(response.predictions) == 2
                assert all(isinstance(pred, (int, float, np.number)) for pred in response.predictions)
    
    def test_predict_with_probabilities(self):
        """Test prediction with probabilities."""
        test_model = SimpleTestModel()
        
        with patch.object(self.export_service, 'list_exported_models') as mock_list:
            with patch.object(self.export_service, 'load_exported_model') as mock_load:
                mock_list.return_value = [{'model_id': 'test_model', 'version': '1.0.0', 'path': '/test/path'}]
                mock_load.return_value = (test_model, self.test_metadata)
                
                request = PredictionRequest(
                    model_id="test_model",
                    version="1.0.0",
                    input_data={"feature1": 1.0, "feature2": 2.0},
                    preprocessing=False,
                    return_probabilities=True
                )
                
                response = self.serving_service.predict(request)
                
                assert response.probabilities is not None
                assert len(response.probabilities) > 0
    
    def test_predict_model_not_found(self):
        """Test prediction with non-existent model."""
        with patch.object(self.export_service, 'list_exported_models') as mock_list:
            mock_list.return_value = []
            
            request = PredictionRequest(
                model_id="nonexistent_model",
                version="1.0.0",
                input_data={"feature1": 1.0, "feature2": 2.0}
            )
            
            with pytest.raises(AutoMLException) as exc_info:
                self.serving_service.predict(request)
            
            assert exc_info.value.error_code == "MODEL_NOT_FOUND"
    
    def test_predict_missing_features(self):
        """Test prediction with missing required features."""
        test_model = SimpleTestModel()
        
        with patch.object(self.export_service, 'list_exported_models') as mock_list:
            with patch.object(self.export_service, 'load_exported_model') as mock_load:
                mock_list.return_value = [{'model_id': 'test_model', 'version': '1.0.0', 'path': '/test/path'}]
                mock_load.return_value = (test_model, self.test_metadata)
                
                request = PredictionRequest(
                    model_id="test_model",
                    version="1.0.0",
                    input_data={"feature1": 1.0},  # Missing feature2
                    preprocessing=False
                )
                
                with pytest.raises(AutoMLException) as exc_info:
                    self.serving_service.predict(request)
                
                assert exc_info.value.error_code == "MISSING_FEATURES"
    
    def test_cache_functionality(self):
        """Test model caching functionality."""
        test_model = SimpleTestModel()
        
        with patch.object(self.export_service, 'list_exported_models') as mock_list:
            with patch.object(self.export_service, 'load_exported_model') as mock_load:
                mock_list.return_value = [{'model_id': 'test_model', 'version': '1.0.0', 'path': '/test/path'}]
                mock_load.return_value = (test_model, self.test_metadata)
                
                request = PredictionRequest(
                    model_id="test_model",
                    version="1.0.0",
                    input_data={"feature1": 1.0, "feature2": 2.0},
                    preprocessing=False
                )
                
                # First prediction - should load model
                response1 = self.serving_service.predict(request)
                assert mock_load.call_count == 1
                
                # Second prediction - should use cached model
                response2 = self.serving_service.predict(request)
                assert mock_load.call_count == 1  # Should not load again
                
                # Check cache stats
                stats = self.serving_service.get_cache_stats()
                assert stats['cached_models'] == 1
                assert stats['cache_hits'] >= 1
    
    def test_cache_eviction(self):
        """Test cache eviction when memory limit is reached."""
        # Create a service with very small cache
        small_cache_service = ModelServingService(
            model_export_service=self.export_service,
            cache_size_mb=0.001,  # Very small cache
            cache_ttl_hours=1
        )
        
        test_model = SimpleTestModel()
        
        with patch.object(self.export_service, 'list_exported_models') as mock_list:
            with patch.object(self.export_service, 'load_exported_model') as mock_load:
                mock_list.return_value = [
                    {'model_id': 'model1', 'version': '1.0.0', 'path': '/test/path1'},
                    {'model_id': 'model2', 'version': '1.0.0', 'path': '/test/path2'}
                ]
                mock_load.return_value = (test_model, self.test_metadata)
                
                # Load first model
                request1 = PredictionRequest(
                    model_id="model1",
                    version="1.0.0",
                    input_data={"feature1": 1.0, "feature2": 2.0},
                    preprocessing=False
                )
                small_cache_service.predict(request1)
                
                # Load second model - should evict first
                request2 = PredictionRequest(
                    model_id="model2",
                    version="1.0.0",
                    input_data={"feature1": 1.0, "feature2": 2.0},
                    preprocessing=False
                )
                small_cache_service.predict(request2)
                
                # Cache should only have one model due to eviction
                stats = small_cache_service.get_cache_stats()
                assert stats['cached_models'] <= 1
        
        small_cache_service.shutdown()
    
    def test_get_model_info(self):
        """Test getting model information."""
        test_model = SimpleTestModel()
        
        with patch.object(self.export_service, 'list_exported_models') as mock_list:
            with patch.object(self.export_service, 'load_exported_model') as mock_load:
                mock_list.return_value = [{'model_id': 'test_model', 'version': '1.0.0', 'path': '/test/path'}]
                mock_load.return_value = (test_model, self.test_metadata)
                
                # Get info for uncached model
                info = self.serving_service.get_model_info("test_model", "1.0.0")
                assert info['model_id'] == "test_model"
                assert info['version'] == "1.0.0"
                assert info['cached'] == False
                
                # Load model into cache
                request = PredictionRequest(
                    model_id="test_model",
                    version="1.0.0",
                    input_data={"feature1": 1.0, "feature2": 2.0},
                    preprocessing=False
                )
                self.serving_service.predict(request)
                
                # Get info for cached model
                info = self.serving_service.get_model_info("test_model", "1.0.0")
                assert info['cached'] == True
                assert 'access_count' in info
                assert 'memory_usage_mb' in info
    
    def test_list_available_models(self):
        """Test listing available models."""
        with patch.object(self.export_service, 'list_exported_models') as mock_list:
            mock_list.return_value = [
                {'model_id': 'model1', 'version': '1.0.0'},
                {'model_id': 'model2', 'version': '1.0.0'}
            ]
            
            models = self.serving_service.list_available_models()
            assert len(models) == 2
            assert models[0]['model_id'] == 'model1'
            assert models[1]['model_id'] == 'model2'
    
    def test_clear_cache(self):
        """Test clearing cache."""
        test_model = SimpleTestModel()
        
        with patch.object(self.export_service, 'list_exported_models') as mock_list:
            with patch.object(self.export_service, 'load_exported_model') as mock_load:
                mock_list.return_value = [{'model_id': 'test_model', 'version': '1.0.0', 'path': '/test/path'}]
                mock_load.return_value = (test_model, self.test_metadata)
                
                # Load model into cache
                request = PredictionRequest(
                    model_id="test_model",
                    version="1.0.0",
                    input_data={"feature1": 1.0, "feature2": 2.0},
                    preprocessing=False
                )
                self.serving_service.predict(request)
                
                # Verify model is cached
                stats = self.serving_service.get_cache_stats()
                assert stats['cached_models'] == 1
                
                # Clear cache
                count = self.serving_service.clear_cache()
                assert count == 1
                
                # Verify cache is empty
                stats = self.serving_service.get_cache_stats()
                assert stats['cached_models'] == 0
    
    def test_warmup_model(self):
        """Test model warmup."""
        test_model = SimpleTestModel()
        
        with patch.object(self.export_service, 'list_exported_models') as mock_list:
            with patch.object(self.export_service, 'load_exported_model') as mock_load:
                mock_list.return_value = [{'model_id': 'test_model', 'version': '1.0.0', 'path': '/test/path'}]
                mock_load.return_value = (test_model, self.test_metadata)
                
                # Warmup model
                self.serving_service.warmup_model("test_model", "1.0.0")
                
                # Verify model is cached
                stats = self.serving_service.get_cache_stats()
                assert stats['cached_models'] == 1
    
    def test_predict_batch(self):
        """Test batch predictions."""
        test_model = SimpleTestModel()
        
        with patch.object(self.export_service, 'list_exported_models') as mock_list:
            with patch.object(self.export_service, 'load_exported_model') as mock_load:
                mock_list.return_value = [{'model_id': 'test_model', 'version': '1.0.0', 'path': '/test/path'}]
                mock_load.return_value = (test_model, self.test_metadata)
                
                requests = [
                    PredictionRequest(
                        model_id="test_model",
                        version="1.0.0",
                        input_data={"feature1": 1.0, "feature2": 2.0},
                        preprocessing=False
                    ),
                    PredictionRequest(
                        model_id="test_model",
                        version="1.0.0",
                        input_data={"feature1": 3.0, "feature2": 4.0},
                        preprocessing=False
                    )
                ]
                
                responses = self.serving_service.predict_batch(requests)
                
                assert len(responses) == 2
                assert all(isinstance(r, PredictionResponse) for r in responses)
                assert all(r.model_id == "test_model" for r in responses)
    
    def test_preprocess_input_numpy(self):
        """Test preprocessing with numpy input."""
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        processed = self.serving_service._preprocess_input(
            input_data, None, self.test_metadata
        )
        
        assert isinstance(processed, np.ndarray)
        np.testing.assert_array_equal(processed, input_data)
    
    def test_preprocess_input_invalid_type(self):
        """Test preprocessing with invalid input type."""
        with pytest.raises(AutoMLException) as exc_info:
            self.serving_service._preprocess_input(
                "invalid_input", None, self.test_metadata
            )
        
        assert exc_info.value.error_code == "INVALID_INPUT"
    
    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        test_model = SimpleTestModel()
        memory_usage = self.serving_service._estimate_memory_usage(test_model)
        
        assert isinstance(memory_usage, float)
        assert memory_usage > 0
    
    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        # Create service with very short TTL
        short_ttl_service = ModelServingService(
            model_export_service=self.export_service,
            cache_ttl_hours=0.001  # Very short TTL
        )
        
        test_model = SimpleTestModel()
        
        with patch.object(self.export_service, 'list_exported_models') as mock_list:
            with patch.object(self.export_service, 'load_exported_model') as mock_load:
                mock_list.return_value = [{'model_id': 'test_model', 'version': '1.0.0', 'path': '/test/path'}]
                mock_load.return_value = (test_model, self.test_metadata)
                
                request = PredictionRequest(
                    model_id="test_model",
                    version="1.0.0",
                    input_data={"feature1": 1.0, "feature2": 2.0},
                    preprocessing=False
                )
                
                # First prediction
                short_ttl_service.predict(request)
                assert mock_load.call_count == 1
                
                # Wait for TTL to expire (simulate by manually setting old timestamp)
                import time
                time.sleep(0.01)  # Small delay
                
                # Manually expire the cache entry
                for cache_entry in short_ttl_service._model_cache.values():
                    cache_entry.last_accessed = datetime.now() - timedelta(hours=1)
                
                # Second prediction should reload model
                short_ttl_service.predict(request)
                assert mock_load.call_count == 2
        
        short_ttl_service.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])