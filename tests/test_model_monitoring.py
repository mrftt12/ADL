"""
Unit tests for Model Monitoring Service
"""

import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pytest

from automl_framework.services.model_monitoring import (
    ModelMonitoringService,
    ModelVersion,
    ModelStatus,
    PredictionLog,
    PerformanceSnapshot,
    Alert,
    AlertType,
    ABTestConfig
)
from automl_framework.services.model_export import (
    ModelExportService,
    ModelMetadata,
    ExportFormat,
    ModelFramework
)
from automl_framework.services.model_serving import ModelServingService
from automl_framework.models.data_models import (
    Architecture,
    TrainingConfig,
    PerformanceMetrics,
    Layer,
    LayerType
)
from automl_framework.core.exceptions import AutoMLException


class TestModelMonitoringService:
    """Test cases for ModelMonitoringService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.export_service = ModelExportService(export_base_dir=self.temp_dir)
        self.serving_service = ModelServingService(self.export_service)
        self.monitoring_service = ModelMonitoringService(
            export_service=self.export_service,
            serving_service=self.serving_service,
            monitoring_data_dir=self.temp_dir + "/monitoring"
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
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test service initialization."""
        assert self.monitoring_service.export_service == self.export_service
        assert self.monitoring_service.serving_service == self.serving_service
        assert self.monitoring_service.retention_days == 90
        assert 'accuracy_drop' in self.monitoring_service.alert_thresholds
    
    def test_register_model_version(self):
        """Test registering a model version."""
        model_version = self.monitoring_service.register_model_version(
            model_id="test_model",
            version="1.0.0",
            metadata=self.test_metadata,
            performance_baseline=self.test_performance_metrics,
            description="Test version",
            tags=["test", "v1"]
        )
        
        assert isinstance(model_version, ModelVersion)
        assert model_version.model_id == "test_model"
        assert model_version.version == "1.0.0"
        assert model_version.status == ModelStatus.INACTIVE
        assert model_version.description == "Test version"
        assert model_version.tags == ["test", "v1"]
        assert model_version.metadata == self.test_metadata
    
    def test_deploy_model_version(self):
        """Test deploying a model version."""
        # Register version first
        self.monitoring_service.register_model_version(
            model_id="test_model",
            version="1.0.0",
            metadata=self.test_metadata
        )
        
        # Deploy version
        self.monitoring_service.deploy_model_version(
            model_id="test_model",
            version="1.0.0",
            traffic_percentage=80.0
        )
        
        # Check deployment
        versions = self.monitoring_service.get_model_versions("test_model")
        deployed_version = versions[0]
        
        assert deployed_version.status == ModelStatus.ACTIVE
        assert deployed_version.traffic_percentage == 80.0
        assert deployed_version.deployed_at is not None
    
    def test_deploy_nonexistent_version(self):
        """Test deploying a non-existent model version."""
        with pytest.raises(AutoMLException) as exc_info:
            self.monitoring_service.deploy_model_version(
                model_id="nonexistent_model",
                version="1.0.0"
            )
        
        assert exc_info.value.error_code == "VERSION_NOT_FOUND"
    
    def test_deprecate_model_version(self):
        """Test deprecating a model version."""
        # Register and deploy version first
        self.monitoring_service.register_model_version(
            model_id="test_model",
            version="1.0.0",
            metadata=self.test_metadata
        )
        self.monitoring_service.deploy_model_version("test_model", "1.0.0")
        
        # Deprecate version
        self.monitoring_service.deprecate_model_version("test_model", "1.0.0")
        
        # Check deprecation
        versions = self.monitoring_service.get_model_versions("test_model")
        deprecated_version = versions[0]
        
        assert deprecated_version.status == ModelStatus.DEPRECATED
        assert deprecated_version.traffic_percentage == 0.0
        assert deprecated_version.deprecated_at is not None
    
    def test_get_model_versions(self):
        """Test getting model versions."""
        # Register multiple versions
        for i in range(3):
            version = f"1.{i}.0"
            self.monitoring_service.register_model_version(
                model_id="test_model",
                version=version,
                metadata=self.test_metadata
            )
        
        versions = self.monitoring_service.get_model_versions("test_model")
        assert len(versions) == 3
        assert all(v.model_id == "test_model" for v in versions)
    
    def test_get_active_version(self):
        """Test getting active model version."""
        # Register versions
        self.monitoring_service.register_model_version(
            model_id="test_model",
            version="1.0.0",
            metadata=self.test_metadata
        )
        self.monitoring_service.register_model_version(
            model_id="test_model",
            version="1.1.0",
            metadata=self.test_metadata
        )
        
        # No active version initially
        active = self.monitoring_service.get_active_version("test_model")
        assert active is None
        
        # Deploy one version
        self.monitoring_service.deploy_model_version("test_model", "1.1.0")
        
        # Should return active version
        active = self.monitoring_service.get_active_version("test_model")
        assert active is not None
        assert active.version == "1.1.0"
        assert active.status == ModelStatus.ACTIVE
    
    def test_log_prediction(self):
        """Test logging predictions."""
        # Register version first
        self.monitoring_service.register_model_version(
            model_id="test_model",
            version="1.0.0",
            metadata=self.test_metadata
        )
        
        # Log prediction
        self.monitoring_service.log_prediction(
            model_id="test_model",
            version="1.0.0",
            input_data={"feature1": 1.0, "feature2": 2.0},
            prediction=1,
            confidence=0.95,
            response_time=0.1
        )
        
        # Check logs
        key = "test_model:1.0.0"
        logs = list(self.monitoring_service._prediction_logs[key])
        assert len(logs) == 1
        
        log = logs[0]
        assert log.model_id == "test_model"
        assert log.version == "1.0.0"
        assert log.prediction == 1
        assert log.confidence == 0.95
        assert log.response_time == 0.1
    
    def test_log_prediction_with_error(self):
        """Test logging predictions with errors."""
        # Register version first
        self.monitoring_service.register_model_version(
            model_id="test_model",
            version="1.0.0",
            metadata=self.test_metadata
        )
        
        # Log prediction with error
        self.monitoring_service.log_prediction(
            model_id="test_model",
            version="1.0.0",
            input_data={"feature1": 1.0, "feature2": 2.0},
            prediction=None,
            error="Prediction failed"
        )
        
        # Check that alert was created
        alerts = self.monitoring_service.get_alerts("test_model")
        assert len(alerts) > 0
        assert alerts[0].alert_type == AlertType.HIGH_ERROR_RATE
    
    def test_compute_performance_metrics(self):
        """Test computing performance metrics."""
        # Register version
        self.monitoring_service.register_model_version(
            model_id="test_model",
            version="1.0.0",
            metadata=self.test_metadata
        )
        
        # Log some predictions
        for i in range(10):
            self.monitoring_service.log_prediction(
                model_id="test_model",
                version="1.0.0",
                input_data={"feature1": float(i), "feature2": float(i+1)},
                prediction=i % 2,
                confidence=0.8 + (i * 0.01),
                response_time=0.1 + (i * 0.01)
            )
        
        # Compute metrics
        metrics = self.monitoring_service.compute_performance_metrics(
            model_id="test_model",
            version="1.0.0",
            window_hours=1
        )
        
        assert isinstance(metrics, PerformanceSnapshot)
        assert metrics.model_id == "test_model"
        assert metrics.version == "1.0.0"
        assert metrics.prediction_count == 10
        assert metrics.error_rate == 0.0  # No errors
        assert metrics.avg_response_time > 0
        assert 'mean' in metrics.confidence_distribution
    
    def test_get_performance_history(self):
        """Test getting performance history."""
        # Register version
        self.monitoring_service.register_model_version(
            model_id="test_model",
            version="1.0.0",
            metadata=self.test_metadata
        )
        
        # Create some performance snapshots
        for i in range(3):
            snapshot = PerformanceSnapshot(
                model_id="test_model",
                version="1.0.0",
                timestamp=datetime.now() - timedelta(hours=i),
                prediction_count=100 + i,
                error_rate=0.01 * i,
                avg_response_time=0.1 + (i * 0.01)
            )
            key = "test_model:1.0.0"
            self.monitoring_service._performance_history[key].append(snapshot)
        
        # Get history
        history = self.monitoring_service.get_performance_history("test_model", "1.0.0")
        assert len(history) == 3
        assert all(h.model_id == "test_model" for h in history)
        assert all(h.version == "1.0.0" for h in history)
    
    def test_create_ab_test(self):
        """Test creating A/B test."""
        # Register versions
        self.monitoring_service.register_model_version(
            model_id="test_model",
            version="1.0.0",
            metadata=self.test_metadata
        )
        self.monitoring_service.register_model_version(
            model_id="test_model",
            version="1.1.0",
            metadata=self.test_metadata
        )
        
        # Create A/B test
        ab_test = self.monitoring_service.create_ab_test(
            test_id="test_ab_1",
            model_id="test_model",
            control_version="1.0.0",
            treatment_version="1.1.0",
            traffic_split=30.0,
            duration_hours=24
        )
        
        assert isinstance(ab_test, ABTestConfig)
        assert ab_test.test_id == "test_ab_1"
        assert ab_test.model_id == "test_model"
        assert ab_test.control_version == "1.0.0"
        assert ab_test.treatment_version == "1.1.0"
        assert ab_test.traffic_split == 30.0
        assert ab_test.active == True
    
    def test_get_ab_test_results(self):
        """Test getting A/B test results."""
        # Register versions
        self.monitoring_service.register_model_version(
            model_id="test_model",
            version="1.0.0",
            metadata=self.test_metadata
        )
        self.monitoring_service.register_model_version(
            model_id="test_model",
            version="1.1.0",
            metadata=self.test_metadata
        )
        
        # Create A/B test
        self.monitoring_service.create_ab_test(
            test_id="test_ab_1",
            model_id="test_model",
            control_version="1.0.0",
            treatment_version="1.1.0"
        )
        
        # Get results
        results = self.monitoring_service.get_ab_test_results("test_ab_1")
        
        assert results['test_id'] == "test_ab_1"
        assert results['model_id'] == "test_model"
        assert results['control_version'] == "1.0.0"
        assert results['treatment_version'] == "1.1.0"
        assert 'control_metrics' in results
        assert 'treatment_metrics' in results
    
    def test_get_ab_test_results_not_found(self):
        """Test getting results for non-existent A/B test."""
        with pytest.raises(AutoMLException) as exc_info:
            self.monitoring_service.get_ab_test_results("nonexistent_test")
        
        assert exc_info.value.error_code == "TEST_NOT_FOUND"
    
    def test_get_alerts(self):
        """Test getting alerts."""
        # Create some alerts manually
        alert1 = Alert(
            id="alert1",
            model_id="test_model",
            version="1.0.0",
            alert_type=AlertType.HIGH_ERROR_RATE,
            severity="high",
            message="High error rate detected",
            timestamp=datetime.now()
        )
        alert2 = Alert(
            id="alert2",
            model_id="test_model",
            version="1.0.0",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity="medium",
            message="Performance degradation detected",
            timestamp=datetime.now(),
            resolved=True,
            resolved_at=datetime.now()
        )
        
        self.monitoring_service._alerts.extend([alert1, alert2])
        
        # Get all alerts
        all_alerts = self.monitoring_service.get_alerts("test_model", unresolved_only=False)
        assert len(all_alerts) == 2
        
        # Get only unresolved alerts
        unresolved_alerts = self.monitoring_service.get_alerts("test_model", unresolved_only=True)
        assert len(unresolved_alerts) == 1
        assert unresolved_alerts[0].id == "alert1"
    
    def test_resolve_alert(self):
        """Test resolving alerts."""
        # Create alert
        alert = Alert(
            id="alert1",
            model_id="test_model",
            version="1.0.0",
            alert_type=AlertType.HIGH_ERROR_RATE,
            severity="high",
            message="High error rate detected",
            timestamp=datetime.now()
        )
        self.monitoring_service._alerts.append(alert)
        
        # Resolve alert
        self.monitoring_service.resolve_alert("alert1")
        
        # Check resolution
        assert alert.resolved == True
        assert alert.resolved_at is not None
    
    def test_resolve_alert_not_found(self):
        """Test resolving non-existent alert."""
        with pytest.raises(AutoMLException) as exc_info:
            self.monitoring_service.resolve_alert("nonexistent_alert")
        
        assert exc_info.value.error_code == "ALERT_NOT_FOUND"
    
    def test_hash_input(self):
        """Test input hashing."""
        input1 = {"feature1": 1.0, "feature2": 2.0}
        input2 = {"feature2": 2.0, "feature1": 1.0}  # Different order
        input3 = {"feature1": 1.0, "feature2": 3.0}  # Different values
        
        hash1 = self.monitoring_service._hash_input(input1)
        hash2 = self.monitoring_service._hash_input(input2)
        hash3 = self.monitoring_service._hash_input(input3)
        
        assert hash1 == hash2  # Same content, different order
        assert hash1 != hash3  # Different content
        assert isinstance(hash1, str)
        assert len(hash1) == 16  # MD5 hash truncated to 16 chars
    
    def test_cleanup_old_data(self):
        """Test cleaning up old data."""
        # Create old prediction logs
        old_log = PredictionLog(
            model_id="test_model",
            version="1.0.0",
            timestamp=datetime.now() - timedelta(days=100),  # Very old
            input_hash="test_hash",
            prediction=1
        )
        recent_log = PredictionLog(
            model_id="test_model",
            version="1.0.0",
            timestamp=datetime.now(),  # Recent
            input_hash="test_hash2",
            prediction=2
        )
        
        key = "test_model:1.0.0"
        self.monitoring_service._prediction_logs[key].extend([old_log, recent_log])
        
        # Cleanup
        self.monitoring_service.cleanup_old_data()
        
        # Check that old data was removed
        remaining_logs = list(self.monitoring_service._prediction_logs[key])
        assert len(remaining_logs) == 1
        assert remaining_logs[0].prediction == 2  # Recent log remains
    
    def test_get_monitoring_summary(self):
        """Test getting monitoring summary."""
        # Register and deploy version
        self.monitoring_service.register_model_version(
            model_id="test_model",
            version="1.0.0",
            metadata=self.test_metadata
        )
        self.monitoring_service.deploy_model_version("test_model", "1.0.0")
        
        # Create some alerts
        alert = Alert(
            id="alert1",
            model_id="test_model",
            version="1.0.0",
            alert_type=AlertType.HIGH_ERROR_RATE,
            severity="high",
            message="Test alert",
            timestamp=datetime.now()
        )
        self.monitoring_service._alerts.append(alert)
        
        # Get summary
        summary = self.monitoring_service.get_monitoring_summary("test_model")
        
        assert summary['model_id'] == "test_model"
        assert summary['total_versions'] == 1
        assert summary['active_version'] == "1.0.0"
        assert summary['unresolved_alerts'] == 1
        assert 'recent_performance' in summary
        assert 'last_updated' in summary


if __name__ == "__main__":
    pytest.main([__file__])