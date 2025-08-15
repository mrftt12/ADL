"""
Middleware for API monitoring and observability
"""

import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from automl_framework.monitoring.metrics import get_metrics_collector
from automl_framework.monitoring.alerts import get_alert_manager
from automl_framework.utils.logging import get_logger

logger = get_logger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect API metrics"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Record request start time
        start_time = time.time()
        
        # Get metrics collector
        metrics_collector = get_metrics_collector()
        
        # Extract request info
        method = request.method
        path = request.url.path
        
        # Normalize path for metrics (remove IDs and query params)
        normalized_path = self._normalize_path(path)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Record metrics
            metrics_collector.record_api_request(
                endpoint=normalized_path,
                method=method,
                status_code=response.status_code,
                duration_ms=response_time_ms
            )
            
            # Log request
            logger.info(
                f"{method} {path} - {response.status_code}",
                method=method,
                path=path,
                status_code=response.status_code,
                response_time_ms=response_time_ms
            )
            
            return response
            
        except Exception as e:
            # Calculate response time for failed requests
            response_time_ms = (time.time() - start_time) * 1000
            
            # Record error metrics
            metrics_collector.record_api_request(
                endpoint=normalized_path,
                method=method,
                status_code=500,
                duration_ms=response_time_ms
            )
            
            # Record error counter
            metrics_collector.record_counter('api.errors.total')
            
            # Log error
            logger.error(
                f"{method} {path} - Error: {str(e)}",
                method=method,
                path=path,
                error=str(e),
                response_time_ms=response_time_ms
            )
            
            raise
    
    def _normalize_path(self, path: str) -> str:
        """Normalize API path for metrics grouping"""
        # Remove query parameters
        if '?' in path:
            path = path.split('?')[0]
        
        # Replace common ID patterns with placeholders
        import re
        
        # UUID patterns
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{id}', path)
        
        # Numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        # File extensions
        path = re.sub(r'\.[a-zA-Z0-9]+$', '.{ext}', path)
        
        return path


class AlertingMiddleware(BaseHTTPMiddleware):
    """Middleware to trigger alerts based on API metrics"""
    
    def __init__(self, app, error_threshold: int = 10, response_time_threshold: float = 5000.0):
        super().__init__(app)
        self.error_threshold = error_threshold
        self.response_time_threshold = response_time_threshold
        self.error_count = 0
        self.last_alert_check = time.time()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Check response time
            response_time_ms = (time.time() - start_time) * 1000
            
            if response_time_ms > self.response_time_threshold:
                await self._check_slow_response_alert(request.url.path, response_time_ms)
            
            # Reset error count on successful request
            if response.status_code < 400:
                self.error_count = 0
            else:
                self.error_count += 1
                await self._check_error_rate_alert()
            
            return response
            
        except Exception as e:
            self.error_count += 1
            await self._check_error_rate_alert()
            raise
    
    async def _check_slow_response_alert(self, path: str, response_time_ms: float):
        """Check for slow response alert"""
        try:
            alert_manager = get_alert_manager()
            metrics = {
                'api.slow_response.path': path,
                'api.slow_response.time_ms': response_time_ms,
                'api.slow_response.threshold_ms': self.response_time_threshold
            }
            await alert_manager.check_alerts(metrics)
        except Exception as e:
            logger.error(f"Error checking slow response alert: {e}")
    
    async def _check_error_rate_alert(self):
        """Check for high error rate alert"""
        try:
            # Only check every 60 seconds to avoid spam
            current_time = time.time()
            if current_time - self.last_alert_check < 60:
                return
            
            self.last_alert_check = current_time
            
            if self.error_count >= self.error_threshold:
                alert_manager = get_alert_manager()
                metrics = {
                    'api.error_count': self.error_count,
                    'api.error_threshold': self.error_threshold
                }
                await alert_manager.check_alerts(metrics)
                
        except Exception as e:
            logger.error(f"Error checking error rate alert: {e}")


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Middleware to perform periodic health checks"""
    
    def __init__(self, app, check_interval: int = 300):  # 5 minutes
        super().__init__(app)
        self.check_interval = check_interval
        self.last_health_check = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Perform periodic health checks
        current_time = time.time()
        if current_time - self.last_health_check > self.check_interval:
            await self._perform_health_checks()
            self.last_health_check = current_time
        
        return await call_next(request)
    
    async def _perform_health_checks(self):
        """Perform background health checks"""
        try:
            from automl_framework.monitoring.health_checks import get_health_checker
            
            health_checker = get_health_checker()
            results = await health_checker.run_all_checks()
            
            # Check for unhealthy services and trigger alerts
            alert_manager = get_alert_manager()
            
            for name, result in results.items():
                if result.status.value in ['unhealthy', 'degraded']:
                    metrics = {
                        f'health.{name}.status': result.status.value,
                        f'health.{name}.response_time_ms': result.response_time_ms
                    }
                    await alert_manager.check_alerts(metrics)
                    
        except Exception as e:
            logger.error(f"Error performing background health checks: {e}")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request logging"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        import uuid
        request_id = str(uuid.uuid4())[:8]
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Extract client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log request start
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=client_ip,
            user_agent=user_agent
        )
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Log successful response
            response_time_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                response_time_ms=response_time_ms
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Log error response
            response_time_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {str(e)}",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                error=str(e),
                response_time_ms=response_time_ms
            )
            
            raise