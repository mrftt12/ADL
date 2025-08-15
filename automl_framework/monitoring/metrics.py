"""
Metrics collection and monitoring for AutoML Framework
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

from automl_framework.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_memory_used: List[float] = field(default_factory=list)
    gpu_memory_total: List[float] = field(default_factory=list)


@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    active_experiments: int = 0
    queued_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    api_requests_total: int = 0
    api_requests_per_minute: float = 0.0
    websocket_connections: int = 0
    database_connections: int = 0


class MetricsCollector:
    """Collects and stores metrics for monitoring"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._collection_thread = None
        self._stop_collection = False
        
        # Request tracking for rate calculation
        self._request_timestamps = deque(maxlen=1000)
        
    def start_collection(self, interval_seconds: int = 30):
        """Start automatic metrics collection"""
        if self._collection_thread and self._collection_thread.is_alive():
            return
        
        self._stop_collection = False
        self._collection_thread = threading.Thread(
            target=self._collect_system_metrics_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._collection_thread.start()
        logger.info("Started metrics collection", interval_seconds=interval_seconds)
    
    def stop_collection(self):
        """Stop automatic metrics collection"""
        self._stop_collection = True
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        logger.info("Stopped metrics collection")
    
    def _collect_system_metrics_loop(self, interval_seconds: int):
        """Background thread for collecting system metrics"""
        while not self._stop_collection:
            try:
                self._collect_system_metrics()
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(interval_seconds)
    
    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_gauge('system.cpu.percent', cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_gauge('system.memory.percent', memory.percent)
            self.record_gauge('system.memory.used_gb', memory.used / (1024**3))
            self.record_gauge('system.memory.total_gb', memory.total / (1024**3))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_gauge('system.disk.percent', disk.percent)
            self.record_gauge('system.disk.used_gb', disk.used / (1024**3))
            self.record_gauge('system.disk.total_gb', disk.total / (1024**3))
            
            # Network metrics
            network = psutil.net_io_counters()
            self.record_counter('system.network.bytes_sent', network.bytes_sent)
            self.record_counter('system.network.bytes_recv', network.bytes_recv)
            
            # GPU metrics (if available)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    self.record_gauge(f'system.gpu.{i}.utilization', gpu.load * 100)
                    self.record_gauge(f'system.gpu.{i}.memory_used_mb', gpu.memoryUsed)
                    self.record_gauge(f'system.gpu.{i}.memory_total_mb', gpu.memoryTotal)
                    self.record_gauge(f'system.gpu.{i}.temperature', gpu.temperature)
            except ImportError:
                pass  # GPUtil not available
            except Exception as e:
                logger.warning(f"Error collecting GPU metrics: {e}")
                
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def record_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Record a counter metric"""
        with self._lock:
            self.counters[name] += value
            self._add_metric_point(name, self.counters[name], labels or {})
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric"""
        with self._lock:
            self.gauges[name] = value
            self._add_metric_point(name, value, labels or {})
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric"""
        with self._lock:
            self.histograms[name].append(value)
            # Keep only recent values
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
            self._add_metric_point(name, value, labels or {})
    
    def record_api_request(self, endpoint: str, method: str, status_code: int, duration_ms: float):
        """Record API request metrics"""
        labels = {
            'endpoint': endpoint,
            'method': method,
            'status_code': str(status_code)
        }
        
        self.record_counter('api.requests.total', labels=labels)
        self.record_histogram('api.request.duration_ms', duration_ms, labels=labels)
        
        # Track request timestamps for rate calculation
        with self._lock:
            self._request_timestamps.append(datetime.now())
    
    def record_experiment_metric(self, experiment_id: str, metric_name: str, value: float):
        """Record experiment-specific metric"""
        labels = {'experiment_id': experiment_id}
        self.record_gauge(f'experiment.{metric_name}', value, labels=labels)
    
    def record_training_metric(self, job_id: str, epoch: int, metric_name: str, value: float):
        """Record training job metric"""
        labels = {'job_id': job_id, 'epoch': str(epoch)}
        self.record_gauge(f'training.{metric_name}', value, labels=labels)
    
    def _add_metric_point(self, name: str, value: float, labels: Dict[str, str]):
        """Add metric point to time series"""
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels
        )
        self.metrics[name].append(point)
        
        # Clean old metrics
        self._cleanup_old_metrics(name)
    
    def _cleanup_old_metrics(self, name: str):
        """Remove metrics older than retention period"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        metrics_deque = self.metrics[name]
        
        while metrics_deque and metrics_deque[0].timestamp < cutoff_time:
            metrics_deque.popleft()
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values"""
        with self._lock:
            # Calculate API request rate
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)
            recent_requests = sum(1 for ts in self._request_timestamps if ts > minute_ago)
            
            return {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'api_requests_per_minute': recent_requests,
                'timestamp': now.isoformat()
            }
    
    def get_metric_history(self, name: str, hours: int = 1) -> List[MetricPoint]:
        """Get metric history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            if name not in self.metrics:
                return []
            
            return [
                point for point in self.metrics[name]
                if point.timestamp > cutoff_time
            ]
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        with self._lock:
            gpu_util = []
            gpu_mem_used = []
            gpu_mem_total = []
            
            # Collect GPU metrics
            for key, value in self.gauges.items():
                if 'gpu' in key and 'utilization' in key:
                    gpu_util.append(value)
                elif 'gpu' in key and 'memory_used' in key:
                    gpu_mem_used.append(value)
                elif 'gpu' in key and 'memory_total' in key:
                    gpu_mem_total.append(value)
            
            return SystemMetrics(
                cpu_percent=self.gauges.get('system.cpu.percent', 0.0),
                memory_percent=self.gauges.get('system.memory.percent', 0.0),
                memory_used_gb=self.gauges.get('system.memory.used_gb', 0.0),
                memory_total_gb=self.gauges.get('system.memory.total_gb', 0.0),
                disk_percent=self.gauges.get('system.disk.percent', 0.0),
                disk_used_gb=self.gauges.get('system.disk.used_gb', 0.0),
                disk_total_gb=self.gauges.get('system.disk.total_gb', 0.0),
                gpu_utilization=gpu_util,
                gpu_memory_used=gpu_mem_used,
                gpu_memory_total=gpu_mem_total
            )
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        with self._lock:
            # Export counters
            for name, value in self.counters.items():
                prometheus_name = name.replace('.', '_')
                lines.append(f'# TYPE {prometheus_name} counter')
                lines.append(f'{prometheus_name} {value}')
            
            # Export gauges
            for name, value in self.gauges.items():
                prometheus_name = name.replace('.', '_')
                lines.append(f'# TYPE {prometheus_name} gauge')
                lines.append(f'{prometheus_name} {value}')
        
        return '\n'.join(lines)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
        _metrics_collector.start_collection()
    return _metrics_collector


def initialize_metrics_collection():
    """Initialize metrics collection"""
    collector = get_metrics_collector()
    logger.info("Metrics collection initialized")
    return collector