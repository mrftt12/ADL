"""
Health check system for AutoML Framework services
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import aiohttp
import psutil

from automl_framework.utils.logging import get_logger
from automl_framework.core.database import get_database_manager
from automl_framework.core.config import get_config

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    response_time_ms: float
    timestamp: float
    details: Optional[Dict[str, Any]] = None


class HealthCheck:
    """Base health check class"""
    
    def __init__(self, name: str, timeout_seconds: float = 5.0):
        self.name = name
        self.timeout_seconds = timeout_seconds
    
    async def check(self) -> HealthCheckResult:
        """Perform health check"""
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                self._perform_check(),
                timeout=self.timeout_seconds
            )
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=result.get('status', HealthStatus.UNKNOWN),
                message=result.get('message', ''),
                response_time_ms=response_time,
                timestamp=time.time(),
                details=result.get('details')
            )
            
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout_seconds}s",
                response_time_ms=response_time,
                timestamp=time.time()
            )
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time_ms=response_time,
                timestamp=time.time()
            )
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Override this method to implement specific health check logic"""
        raise NotImplementedError


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity"""
    
    def __init__(self):
        super().__init__("database", timeout_seconds=10.0)
    
    async def _perform_check(self) -> Dict[str, Any]:
        try:
            db_manager = get_database_manager()
            
            # Use the health_check method from DatabaseManager
            health_status = db_manager.health_check()
            
            if health_status.get('postgresql') and health_status.get('mongodb'):
                return {
                    'status': HealthStatus.HEALTHY,
                    'message': 'Database connections are healthy',
                    'details': health_status
                }
            else:
                return {
                    'status': HealthStatus.UNHEALTHY,
                    'message': 'One or more database connections failed',
                    'details': health_status
                }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Database connection failed: {str(e)}',
                'details': {'error': str(e)}
            }


class RedisHealthCheck(HealthCheck):
    """Health check for Redis connectivity"""
    
    def __init__(self):
        super().__init__("redis", timeout_seconds=5.0)
    
    async def _perform_check(self) -> Dict[str, Any]:
        try:
            import redis.asyncio as redis
            config = get_config()
            
            redis_client = redis.from_url(config.redis_url)
            await redis_client.ping()
            await redis_client.close()
            
            return {
                'status': HealthStatus.HEALTHY,
                'message': 'Redis connection is healthy'
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Redis connection failed: {str(e)}',
                'details': {'error': str(e)}
            }


class SystemResourcesHealthCheck(HealthCheck):
    """Health check for system resources"""
    
    def __init__(self, cpu_threshold: float = 90.0, memory_threshold: float = 90.0, disk_threshold: float = 90.0):
        super().__init__("system_resources", timeout_seconds=5.0)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    async def _perform_check(self) -> Dict[str, Any]:
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Determine overall status
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_percent > self.cpu_threshold:
                status = HealthStatus.DEGRADED
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > self.memory_threshold:
                status = HealthStatus.DEGRADED
                issues.append(f"High memory usage: {memory_percent:.1f}%")
            
            if disk_percent > self.disk_threshold:
                status = HealthStatus.DEGRADED
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            
            message = "System resources are healthy"
            if issues:
                message = f"System resource issues: {', '.join(issues)}"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'disk_percent': disk_percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_total_gb': memory.total / (1024**3),
                    'disk_used_gb': disk.used / (1024**3),
                    'disk_total_gb': disk.total / (1024**3)
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'System resources check failed: {str(e)}',
                'details': {'error': str(e)}
            }


class ServiceHealthCheck(HealthCheck):
    """Health check for external services via HTTP"""
    
    def __init__(self, name: str, url: str, expected_status: int = 200):
        super().__init__(name, timeout_seconds=10.0)
        self.url = url
        self.expected_status = expected_status
    
    async def _perform_check(self) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url) as response:
                    if response.status == self.expected_status:
                        return {
                            'status': HealthStatus.HEALTHY,
                            'message': f'Service is responding with status {response.status}',
                            'details': {
                                'status_code': response.status,
                                'url': self.url
                            }
                        }
                    else:
                        return {
                            'status': HealthStatus.UNHEALTHY,
                            'message': f'Service returned unexpected status {response.status}',
                            'details': {
                                'status_code': response.status,
                                'expected_status': self.expected_status,
                                'url': self.url
                            }
                        }
                        
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Service health check failed: {str(e)}',
                'details': {
                    'error': str(e),
                    'url': self.url
                }
            }


class GPUHealthCheck(HealthCheck):
    """Health check for GPU availability and status"""
    
    def __init__(self):
        super().__init__("gpu", timeout_seconds=5.0)
    
    async def _perform_check(self) -> Dict[str, Any]:
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            if not gpus:
                return {
                    'status': HealthStatus.HEALTHY,
                    'message': 'No GPUs detected',
                    'details': {'gpu_count': 0}
                }
            
            gpu_details = []
            issues = []
            
            for i, gpu in enumerate(gpus):
                gpu_info = {
                    'id': i,
                    'name': gpu.name,
                    'utilization': gpu.load * 100,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'temperature': gpu.temperature
                }
                gpu_details.append(gpu_info)
                
                # Check for issues
                if gpu.temperature > 85:
                    issues.append(f"GPU {i} temperature high: {gpu.temperature}°C")
                
                if gpu_info['memory_percent'] > 95:
                    issues.append(f"GPU {i} memory usage high: {gpu_info['memory_percent']:.1f}%")
            
            status = HealthStatus.HEALTHY
            message = f"{len(gpus)} GPU(s) available and healthy"
            
            if issues:
                status = HealthStatus.DEGRADED
                message = f"GPU issues detected: {', '.join(issues)}"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'gpu_count': len(gpus),
                    'gpus': gpu_details
                }
            }
            
        except ImportError:
            return {
                'status': HealthStatus.UNKNOWN,
                'message': 'GPUtil not available - cannot check GPU status',
                'details': {'error': 'GPUtil not installed'}
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'GPU health check failed: {str(e)}',
                'details': {'error': str(e)}
            }


class HealthChecker:
    """Manages and runs health checks"""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks"""
        self.register_check(DatabaseHealthCheck())
        self.register_check(RedisHealthCheck())
        self.register_check(SystemResourcesHealthCheck())
        self.register_check(GPUHealthCheck())
    
    def register_check(self, health_check: HealthCheck):
        """Register a health check"""
        self.health_checks[health_check.name] = health_check
        logger.info(f"Registered health check: {health_check.name}")
    
    def unregister_check(self, name: str):
        """Unregister a health check"""
        if name in self.health_checks:
            del self.health_checks[name]
            if name in self.last_results:
                del self.last_results[name]
            logger.info(f"Unregistered health check: {name}")
    
    async def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check"""
        if name not in self.health_checks:
            return None
        
        result = await self.health_checks[name].check()
        self.last_results[name] = result
        
        logger.info(
            f"Health check completed: {name}",
            status=result.status.value,
            response_time_ms=result.response_time_ms
        )
        
        return result
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        tasks = []
        for name in self.health_checks:
            tasks.append(self.run_check(name))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        check_results = {}
        for i, (name, result) in enumerate(zip(self.health_checks.keys(), results)):
            if isinstance(result, Exception):
                logger.error(f"Health check {name} failed with exception: {result}")
                check_results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(result)}",
                    response_time_ms=0.0,
                    timestamp=time.time()
                )
            else:
                check_results[name] = result
        
        return check_results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status"""
        if not self.last_results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self.last_results.values()]
        
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary"""
        overall_status = self.get_overall_status()
        
        return {
            'overall_status': overall_status.value,
            'timestamp': time.time(),
            'checks': {
                name: {
                    'status': result.status.value,
                    'message': result.message,
                    'response_time_ms': result.response_time_ms,
                    'timestamp': result.timestamp
                }
                for name, result in self.last_results.items()
            }
        }


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker instance"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker