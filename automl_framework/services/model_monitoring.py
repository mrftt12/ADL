"""
Model Monitoring and Versioning Service for AutoML Framework

This service handles model version tracking, performance monitoring,
A/B testing, and automated retraining triggers.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict, deque
import statistics

import numpy as np
import pandas as pd

from .model_export import ModelExportService, ModelMetadata
from .model_serving import ModelServingService, PredictionRequest, PredictionResponse
from ..models.data_models import PerformanceMetrics
from ..core.exceptions import AutoMLException


logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model deployment status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    TESTING = "testing"
    FAILED = "failed"


class AlertType(Enum):
    """Types of monitoring alerts."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    HIGH_ERROR_RATE = "high_error_rate"
    DRIFT_DETECTED = "drift_detected"
    RESOURCE_USAGE = "resource_usage"
    PREDICTION_VOLUME = "prediction_volume"


@dataclass
class ModelVersion:
    """Model version information."""
    model_id: str
    version: str
    status: ModelStatus
    created_at: datetime
    deployed_at: Optional[datetime] = None
    deprecated_at: Optional[datetime] = None
    metadata: Optional[ModelMetadata] = None
    performance_baseline: Optional[PerformanceMetrics] = None
    traffic_percentage: float = 100.0
    description: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class PredictionLog:
    """Log entry for model predictions."""
    model_id: str
    version: str
    timestamp: datetime
    input_hash: str
    prediction: Any
    confidence: Optional[float] = None
    response_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PerformanceSnapshot:
    """Performance metrics snapshot."""
    model_id: str
    version: str
    timestamp: datetime
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    prediction_count: int = 0
    confidence_distribution: Dict[str, float] = None
    
    def __post_init__(self):
        if self.confidence_distribution is None:
            self.confidence_distribution = {}


@dataclass
class Alert:
    """Monitoring alert."""
    id: str
    model_id: str
    version: str
    alert_type: AlertType
    severity: str  # low, medium, high, critical
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ABTestConfig:
    """A/B test configuration."""
    test_id: str
    model_id: str
    control_version: str
    treatment_version: str
    traffic_split: float  # Percentage for treatment (0-100)
    start_time: datetime
    end_time: Optional[datetime] = None
    success_metric: str = "accuracy"
    min_sample_size: int = 1000
    significance_level: float = 0.05
    active: bool = True


class ModelMonitoringService:
    """Service for monitoring deployed models and managing versions."""
    
    def __init__(
        self,
        export_service: ModelExportService,
        serving_service: ModelServingService,
        monitoring_data_dir: str = "monitoring_data",
        alert_thresholds: Optional[Dict[str, float]] = None,
        retention_days: int = 90
    ):
        """
        Initialize model monitoring service.
        
        Args:
            export_service: Model export service instance
            serving_service: Model serving service instance
            monitoring_data_dir: Directory for monitoring data
            alert_thresholds: Alert threshold configuration
            retention_days: Data retention period in days
        """
        self.export_service = export_service
        self.serving_service = serving_service
        self.monitoring_data_dir = Path(monitoring_data_dir)
        self.monitoring_data_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'accuracy_drop': 0.05,  # 5% drop in accuracy
            'error_rate': 0.10,     # 10% error rate
            'response_time': 5.0,   # 5 seconds response time
            'drift_threshold': 0.1   # Statistical drift threshold
        }
        
        # In-memory storage (in production, use persistent storage)
        self._model_versions: Dict[str, List[ModelVersion]] = defaultdict(list)
        self._prediction_logs: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._performance_history: Dict[str, List[PerformanceSnapshot]] = defaultdict(list)
        self._alerts: List[Alert] = []
        self._ab_tests: Dict[str, ABTestConfig] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Model monitoring service initialized with data dir: {monitoring_data_dir}")
    
    def register_model_version(
        self,
        model_id: str,
        version: str,
        metadata: ModelMetadata,
        performance_baseline: Optional[PerformanceMetrics] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model_id: Model identifier
            version: Model version
            metadata: Model metadata
            performance_baseline: Baseline performance metrics
            description: Version description
            tags: Version tags
            
        Returns:
            Created model version
        """
        with self._lock:
            model_version = ModelVersion(
                model_id=model_id,
                version=version,
                status=ModelStatus.INACTIVE,
                created_at=datetime.now(),
                metadata=metadata,
                performance_baseline=performance_baseline,
                description=description,
                tags=tags or []
            )
            
            self._model_versions[model_id].append(model_version)
            
            # Save to disk
            self._save_model_version(model_version)
            
            logger.info(f"Registered model version {model_id}:{version}")
            return model_version
    
    def deploy_model_version(
        self,
        model_id: str,
        version: str,
        traffic_percentage: float = 100.0
    ) -> None:
        """
        Deploy a model version.
        
        Args:
            model_id: Model identifier
            version: Model version
            traffic_percentage: Percentage of traffic to route to this version
        """
        with self._lock:
            model_version = self._get_model_version(model_id, version)
            if not model_version:
                raise AutoMLException(f"Model version {model_id}:{version} not found", "VERSION_NOT_FOUND")
            
            # Update status
            model_version.status = ModelStatus.ACTIVE
            model_version.deployed_at = datetime.now()
            model_version.traffic_percentage = traffic_percentage
            
            # If deploying with 100% traffic, deactivate other versions
            if traffic_percentage == 100.0:
                for mv in self._model_versions[model_id]:
                    if mv.version != version and mv.status == ModelStatus.ACTIVE:
                        mv.status = ModelStatus.INACTIVE
                        mv.traffic_percentage = 0.0
            
            # Save changes
            self._save_model_version(model_version)
            
            logger.info(f"Deployed model version {model_id}:{version} with {traffic_percentage}% traffic")
    
    def deprecate_model_version(self, model_id: str, version: str) -> None:
        """
        Deprecate a model version.
        
        Args:
            model_id: Model identifier
            version: Model version
        """
        with self._lock:
            model_version = self._get_model_version(model_id, version)
            if not model_version:
                raise AutoMLException(f"Model version {model_id}:{version} not found", "VERSION_NOT_FOUND")
            
            model_version.status = ModelStatus.DEPRECATED
            model_version.deprecated_at = datetime.now()
            model_version.traffic_percentage = 0.0
            
            self._save_model_version(model_version)
            
            logger.info(f"Deprecated model version {model_id}:{version}")
    
    def log_prediction(
        self,
        model_id: str,
        version: str,
        input_data: Any,
        prediction: Any,
        confidence: Optional[float] = None,
        response_time: float = 0.0,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a model prediction.
        
        Args:
            model_id: Model identifier
            version: Model version
            input_data: Input data for prediction
            prediction: Model prediction
            confidence: Prediction confidence
            response_time: Response time in seconds
            error: Error message if prediction failed
            metadata: Additional metadata
        """
        # Create input hash for tracking
        input_hash = self._hash_input(input_data)
        
        log_entry = PredictionLog(
            model_id=model_id,
            version=version,
            timestamp=datetime.now(),
            input_hash=input_hash,
            prediction=prediction,
            confidence=confidence,
            response_time=response_time,
            error=error,
            metadata=metadata or {}
        )
        
        with self._lock:
            key = f"{model_id}:{version}"
            self._prediction_logs[key].append(log_entry)
        
        # Check for alerts
        self._check_alerts(model_id, version, log_entry)
    
    def get_model_versions(self, model_id: str) -> List[ModelVersion]:
        """
        Get all versions of a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of model versions
        """
        with self._lock:
            return list(self._model_versions.get(model_id, []))
    
    def get_active_version(self, model_id: str) -> Optional[ModelVersion]:
        """
        Get the active version of a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Active model version or None
        """
        with self._lock:
            for version in self._model_versions.get(model_id, []):
                if version.status == ModelStatus.ACTIVE:
                    return version
            return None
    
    def get_performance_history(
        self,
        model_id: str,
        version: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[PerformanceSnapshot]:
        """
        Get performance history for a model.
        
        Args:
            model_id: Model identifier
            version: Optional specific version
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List of performance snapshots
        """
        with self._lock:
            key = f"{model_id}:{version}" if version else model_id
            history = []
            
            if version:
                history = self._performance_history.get(key, [])
            else:
                # Get history for all versions
                for v in self._model_versions.get(model_id, []):
                    version_key = f"{model_id}:{v.version}"
                    history.extend(self._performance_history.get(version_key, []))
            
            # Apply time filters
            if start_time or end_time:
                filtered_history = []
                for snapshot in history:
                    if start_time and snapshot.timestamp < start_time:
                        continue
                    if end_time and snapshot.timestamp > end_time:
                        continue
                    filtered_history.append(snapshot)
                history = filtered_history
            
            return sorted(history, key=lambda x: x.timestamp)
    
    def compute_performance_metrics(
        self,
        model_id: str,
        version: str,
        window_hours: int = 24
    ) -> PerformanceSnapshot:
        """
        Compute performance metrics for a model version.
        
        Args:
            model_id: Model identifier
            version: Model version
            window_hours: Time window in hours
            
        Returns:
            Performance snapshot
        """
        with self._lock:
            key = f"{model_id}:{version}"
            logs = list(self._prediction_logs.get(key, []))
            
            # Filter logs by time window
            cutoff_time = datetime.now() - timedelta(hours=window_hours)
            recent_logs = [log for log in logs if log.timestamp >= cutoff_time]
            
            if not recent_logs:
                return PerformanceSnapshot(
                    model_id=model_id,
                    version=version,
                    timestamp=datetime.now()
                )
            
            # Compute metrics
            total_predictions = len(recent_logs)
            error_count = sum(1 for log in recent_logs if log.error is not None)
            error_rate = error_count / total_predictions if total_predictions > 0 else 0.0
            
            response_times = [log.response_time for log in recent_logs if log.response_time > 0]
            avg_response_time = statistics.mean(response_times) if response_times else 0.0
            
            # Confidence distribution
            confidences = [log.confidence for log in recent_logs if log.confidence is not None]
            confidence_distribution = {}
            if confidences:
                confidence_distribution = {
                    'mean': statistics.mean(confidences),
                    'std': statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
                    'min': min(confidences),
                    'max': max(confidences)
                }
            
            snapshot = PerformanceSnapshot(
                model_id=model_id,
                version=version,
                timestamp=datetime.now(),
                avg_response_time=avg_response_time,
                error_rate=error_rate,
                prediction_count=total_predictions,
                confidence_distribution=confidence_distribution
            )
            
            # Store snapshot
            self._performance_history[key].append(snapshot)
            
            return snapshot
    
    def create_ab_test(
        self,
        test_id: str,
        model_id: str,
        control_version: str,
        treatment_version: str,
        traffic_split: float = 50.0,
        duration_hours: Optional[int] = None,
        success_metric: str = "accuracy",
        min_sample_size: int = 1000
    ) -> ABTestConfig:
        """
        Create an A/B test between two model versions.
        
        Args:
            test_id: Test identifier
            model_id: Model identifier
            control_version: Control version
            treatment_version: Treatment version
            traffic_split: Percentage of traffic for treatment
            duration_hours: Test duration in hours
            success_metric: Metric to optimize
            min_sample_size: Minimum sample size
            
        Returns:
            A/B test configuration
        """
        end_time = None
        if duration_hours:
            end_time = datetime.now() + timedelta(hours=duration_hours)
        
        ab_test = ABTestConfig(
            test_id=test_id,
            model_id=model_id,
            control_version=control_version,
            treatment_version=treatment_version,
            traffic_split=traffic_split,
            start_time=datetime.now(),
            end_time=end_time,
            success_metric=success_metric,
            min_sample_size=min_sample_size
        )
        
        with self._lock:
            self._ab_tests[test_id] = ab_test
            
            # Update traffic routing
            control_mv = self._get_model_version(model_id, control_version)
            treatment_mv = self._get_model_version(model_id, treatment_version)
            
            if control_mv:
                control_mv.traffic_percentage = 100.0 - traffic_split
                control_mv.status = ModelStatus.ACTIVE
            
            if treatment_mv:
                treatment_mv.traffic_percentage = traffic_split
                treatment_mv.status = ModelStatus.TESTING
        
        logger.info(f"Created A/B test {test_id} for model {model_id}")
        return ab_test
    
    def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """
        Get A/B test results.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Test results dictionary
        """
        with self._lock:
            ab_test = self._ab_tests.get(test_id)
            if not ab_test:
                raise AutoMLException(f"A/B test {test_id} not found", "TEST_NOT_FOUND")
            
            # Get performance data for both versions
            control_perf = self.compute_performance_metrics(
                ab_test.model_id, ab_test.control_version
            )
            treatment_perf = self.compute_performance_metrics(
                ab_test.model_id, ab_test.treatment_version
            )
            
            # Simple statistical comparison (in production, use proper statistical tests)
            control_sample_size = control_perf.prediction_count
            treatment_sample_size = treatment_perf.prediction_count
            
            results = {
                'test_id': test_id,
                'model_id': ab_test.model_id,
                'control_version': ab_test.control_version,
                'treatment_version': ab_test.treatment_version,
                'start_time': ab_test.start_time.isoformat(),
                'end_time': ab_test.end_time.isoformat() if ab_test.end_time else None,
                'control_metrics': {
                    'sample_size': control_sample_size,
                    'error_rate': control_perf.error_rate,
                    'avg_response_time': control_perf.avg_response_time
                },
                'treatment_metrics': {
                    'sample_size': treatment_sample_size,
                    'error_rate': treatment_perf.error_rate,
                    'avg_response_time': treatment_perf.avg_response_time
                },
                'sufficient_sample_size': (
                    control_sample_size >= ab_test.min_sample_size and
                    treatment_sample_size >= ab_test.min_sample_size
                ),
                'active': ab_test.active
            }
            
            return results
    
    def get_alerts(
        self,
        model_id: Optional[str] = None,
        unresolved_only: bool = True
    ) -> List[Alert]:
        """
        Get monitoring alerts.
        
        Args:
            model_id: Optional model ID filter
            unresolved_only: Only return unresolved alerts
            
        Returns:
            List of alerts
        """
        with self._lock:
            alerts = list(self._alerts)
            
            if model_id:
                alerts = [a for a in alerts if a.model_id == model_id]
            
            if unresolved_only:
                alerts = [a for a in alerts if not a.resolved]
            
            return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def resolve_alert(self, alert_id: str) -> None:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert identifier
        """
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    logger.info(f"Resolved alert {alert_id}")
                    return
            
            raise AutoMLException(f"Alert {alert_id} not found", "ALERT_NOT_FOUND")
    
    def _get_model_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        for mv in self._model_versions.get(model_id, []):
            if mv.version == version:
                return mv
        return None
    
    def _hash_input(self, input_data: Any) -> str:
        """Create a hash of input data for tracking."""
        import hashlib
        
        try:
            if isinstance(input_data, dict):
                data_str = json.dumps(input_data, sort_keys=True)
            elif isinstance(input_data, (list, tuple)):
                data_str = json.dumps(input_data)
            else:
                data_str = str(input_data)
            
            return hashlib.md5(data_str.encode()).hexdigest()[:16]
        except Exception:
            return "unknown"
    
    def _check_alerts(self, model_id: str, version: str, log_entry: PredictionLog) -> None:
        """Check if any alerts should be triggered."""
        # Error rate alert
        if log_entry.error:
            self._create_alert(
                model_id=model_id,
                version=version,
                alert_type=AlertType.HIGH_ERROR_RATE,
                severity="high",
                message=f"Prediction error: {log_entry.error}"
            )
        
        # Response time alert
        if log_entry.response_time > self.alert_thresholds.get('response_time', 5.0):
            self._create_alert(
                model_id=model_id,
                version=version,
                alert_type=AlertType.RESOURCE_USAGE,
                severity="medium",
                message=f"High response time: {log_entry.response_time:.2f}s"
            )
    
    def _create_alert(
        self,
        model_id: str,
        version: str,
        alert_type: AlertType,
        severity: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create a new alert."""
        import uuid
        
        alert = Alert(
            id=str(uuid.uuid4()),
            model_id=model_id,
            version=version,
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self._alerts.append(alert)
        
        logger.warning(f"Alert created: {alert.message}")
        return alert
    
    def _save_model_version(self, model_version: ModelVersion) -> None:
        """Save model version to disk."""
        try:
            version_dir = self.monitoring_data_dir / "versions" / model_version.model_id
            version_dir.mkdir(parents=True, exist_ok=True)
            
            version_file = version_dir / f"{model_version.version}.json"
            
            # Convert to serializable format
            data = asdict(model_version)
            data['created_at'] = model_version.created_at.isoformat()
            if model_version.deployed_at:
                data['deployed_at'] = model_version.deployed_at.isoformat()
            if model_version.deprecated_at:
                data['deprecated_at'] = model_version.deprecated_at.isoformat()
            data['status'] = model_version.status.value
            
            with open(version_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save model version: {e}")
    
    def cleanup_old_data(self) -> None:
        """Clean up old monitoring data based on retention policy."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        with self._lock:
            # Clean up prediction logs
            for key in list(self._prediction_logs.keys()):
                logs = self._prediction_logs[key]
                # Keep only recent logs
                recent_logs = deque([log for log in logs if log.timestamp >= cutoff_date], maxlen=10000)
                self._prediction_logs[key] = recent_logs
            
            # Clean up performance history
            for key in list(self._performance_history.keys()):
                history = self._performance_history[key]
                recent_history = [snap for snap in history if snap.timestamp >= cutoff_date]
                self._performance_history[key] = recent_history
            
            # Clean up resolved alerts
            self._alerts = [alert for alert in self._alerts 
                          if not alert.resolved or 
                          (alert.resolved_at and alert.resolved_at >= cutoff_date)]
        
        logger.info(f"Cleaned up monitoring data older than {self.retention_days} days")
    
    def get_monitoring_summary(self, model_id: str) -> Dict[str, Any]:
        """
        Get monitoring summary for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Monitoring summary
        """
        with self._lock:
            versions = self.get_model_versions(model_id)
            active_version = self.get_active_version(model_id)
            alerts = self.get_alerts(model_id, unresolved_only=True)
            
            # Get recent performance
            recent_performance = {}
            if active_version:
                perf = self.compute_performance_metrics(model_id, active_version.version)
                recent_performance = {
                    'prediction_count': perf.prediction_count,
                    'error_rate': perf.error_rate,
                    'avg_response_time': perf.avg_response_time
                }
            
            return {
                'model_id': model_id,
                'total_versions': len(versions),
                'active_version': active_version.version if active_version else None,
                'unresolved_alerts': len(alerts),
                'recent_performance': recent_performance,
                'last_updated': datetime.now().isoformat()
            }