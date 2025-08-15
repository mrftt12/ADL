# AutoML Framework Monitoring and Observability

This document describes the comprehensive monitoring and observability infrastructure for the AutoML Framework.

## Overview

The monitoring system provides:

- **Metrics Collection**: System and application metrics
- **Health Checks**: Service health monitoring
- **Alerting**: Automated alert generation and notification
- **Logging**: Structured logging with JSON format
- **Dashboards**: Visual monitoring dashboards
- **API Endpoints**: Programmatic access to monitoring data

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│  Metrics        │───▶│   Prometheus    │
│   Services      │    │  Collector      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│  Health Checks  │───▶│  Alert Manager  │◀────────────┘
│                 │    │                 │
└─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Notifications  │
                       │  (Email/Slack)  │
                       └─────────────────┘
```

## Components

### 1. Metrics Collection

**Location**: `automl_framework/monitoring/metrics.py`

Collects various types of metrics:

- **System Metrics**: CPU, memory, disk, GPU utilization
- **Application Metrics**: API requests, experiment counts, job status
- **Custom Metrics**: Domain-specific metrics for ML workflows

**Key Features**:
- Automatic background collection
- Prometheus export format
- Time-series data storage
- Configurable retention periods

**Usage**:
```python
from automl_framework.monitoring.metrics import get_metrics_collector

collector = get_metrics_collector()
collector.record_counter('api.requests.total')
collector.record_gauge('system.cpu.percent', 75.5)
collector.record_histogram('api.response_time_ms', 150.0)
```

### 2. Health Checks

**Location**: `automl_framework/monitoring/health_checks.py`

Monitors service health:

- **Database Health**: PostgreSQL and MongoDB connectivity
- **Redis Health**: Cache service availability
- **System Resources**: CPU, memory, disk usage thresholds
- **GPU Health**: GPU temperature and memory usage
- **Custom Checks**: Service-specific health validation

**Health Check Types**:
- `DatabaseHealthCheck`: Database connectivity
- `RedisHealthCheck`: Redis connectivity
- `SystemResourcesHealthCheck`: Resource usage
- `GPUHealthCheck`: GPU status
- `ServiceHealthCheck`: HTTP endpoint checks

### 3. Alert Management

**Location**: `automl_framework/monitoring/alerts.py`

Automated alerting system:

- **Alert Rules**: Configurable conditions for triggering alerts
- **Severity Levels**: Low, Medium, High, Critical
- **Notification Channels**: Email, Slack, Webhooks
- **Alert Lifecycle**: Active, Acknowledged, Resolved states

**Built-in Alert Rules**:
- High CPU/Memory/Disk usage
- Database connection failures
- High API error rates
- GPU overheating
- Training job failure rates

### 4. API Endpoints

**Location**: `automl_framework/api/routes/monitoring.py`

REST API for monitoring data:

- `GET /api/v1/monitoring/health` - Overall health status
- `GET /api/v1/monitoring/metrics` - Current metrics
- `GET /api/v1/monitoring/alerts` - Active alerts
- `GET /api/v1/monitoring/status` - Comprehensive status

### 5. Middleware

**Location**: `automl_framework/api/middleware.py`

Automatic monitoring integration:

- **MetricsMiddleware**: Collects API request metrics
- **AlertingMiddleware**: Triggers alerts based on API behavior
- **HealthCheckMiddleware**: Periodic health checks
- **RequestLoggingMiddleware**: Structured request logging

## Configuration

### Environment Variables

```bash
# Logging
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/automl.log

# Metrics
METRICS_RETENTION_HOURS=24
METRICS_COLLECTION_INTERVAL=30

# Alerts
ALERT_EMAIL_ENABLED=true
ALERT_EMAIL_SMTP_HOST=smtp.gmail.com
ALERT_EMAIL_SMTP_PORT=587
ALERT_SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

### Logging Configuration

**File**: `config/logging.yaml`

Structured logging with JSON format:
- Console output for development
- File rotation for production
- Separate error logs
- API request logs

### Alert Configuration

Configure alert channels in your application:

```python
from automl_framework.monitoring.alerts import get_alert_manager, EmailAlertChannel

alert_manager = get_alert_manager()

# Add email notifications
email_channel = EmailAlertChannel(
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    username="alerts@company.com",
    password="password",
    from_email="alerts@company.com",
    to_emails=["admin@company.com"]
)
alert_manager.add_channel(email_channel)
```

## Deployment

### Local Development

The monitoring stack is included in `docker-compose.yml`:

```bash
# Deploy with monitoring
./scripts/deploy-local.sh

# Access monitoring services
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3001 (admin/admin)
# Alertmanager: http://localhost:9093
```

### Production (Kubernetes)

Monitoring services are defined in Kubernetes manifests:

```bash
# Deploy to Kubernetes
./scripts/deploy-k8s.sh

# Port forward monitoring services
kubectl port-forward service/prometheus 9090:9090 -n automl
kubectl port-forward service/grafana 3001:3000 -n automl
```

## Dashboards

### CLI Dashboard

**File**: `scripts/monitoring-dashboard.py`

Simple command-line monitoring dashboard:

```bash
# Run monitoring dashboard
python scripts/monitoring-dashboard.py

# Custom API URL and refresh interval
python scripts/monitoring-dashboard.py --api-url http://api:8000 --refresh 10
```

### Grafana Dashboard

**File**: `docker/grafana-dashboard.json`

Pre-configured Grafana dashboard with:
- System resource utilization
- API request rates and response times
- Active experiments and training jobs
- Error rates by endpoint
- Database connection metrics

### Prometheus Metrics

Access raw metrics at:
- `http://localhost:8000/api/v1/monitoring/metrics/prometheus`
- `http://localhost:9090` (Prometheus UI)

## Alert Rules

### System Alerts

- **High CPU Usage**: CPU > 90% for 5 minutes
- **High Memory Usage**: Memory > 90% for 5 minutes
- **Critical Disk Usage**: Disk > 95% for 2 minutes
- **GPU Overheating**: GPU temperature > 85°C

### Application Alerts

- **High API Error Rate**: Error rate > 10% for 2 minutes
- **Slow API Responses**: 95th percentile > 5 seconds
- **Database Connection Failure**: Connection down for 1 minute
- **High Training Failure Rate**: Failure rate > 50%

### Custom Alerts

Add custom alert rules:

```python
from automl_framework.monitoring.alerts import AlertRule, AlertSeverity

custom_rule = AlertRule(
    name="custom_metric_threshold",
    condition=lambda metrics: metrics.get('custom.metric', 0) > 100,
    severity=AlertSeverity.HIGH,
    message_template="Custom metric exceeded threshold: {custom.metric}",
    cooldown_minutes=15
)

alert_manager.register_rule(custom_rule)
```

## Troubleshooting

### Common Issues

1. **Metrics not appearing**:
   - Check if metrics collector is initialized
   - Verify API middleware is configured
   - Check Prometheus scrape configuration

2. **Alerts not firing**:
   - Verify alert rules are registered
   - Check alert channel configuration
   - Review alert cooldown periods

3. **Health checks failing**:
   - Check service connectivity
   - Verify database credentials
   - Review timeout settings

### Debug Commands

```bash
# Check API health
curl http://localhost:8000/api/v1/monitoring/health

# Get current metrics
curl http://localhost:8000/api/v1/monitoring/metrics

# View active alerts
curl http://localhost:8000/api/v1/monitoring/alerts

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# View container logs
docker-compose logs -f api
docker-compose logs -f prometheus
```

## Best Practices

### Metrics

1. **Use appropriate metric types**:
   - Counters for cumulative values
   - Gauges for current values
   - Histograms for distributions

2. **Add meaningful labels**:
   - Include context like experiment_id, user_id
   - Keep cardinality reasonable

3. **Set retention policies**:
   - Configure appropriate retention periods
   - Clean up old metrics regularly

### Alerts

1. **Avoid alert fatigue**:
   - Set appropriate thresholds
   - Use cooldown periods
   - Group related alerts

2. **Make alerts actionable**:
   - Include context in messages
   - Provide troubleshooting steps
   - Link to relevant dashboards

3. **Test alert channels**:
   - Verify email/Slack configuration
   - Test alert delivery regularly
   - Monitor alert manager health

### Logging

1. **Use structured logging**:
   - JSON format for machine parsing
   - Include request IDs for tracing
   - Add contextual information

2. **Set appropriate log levels**:
   - DEBUG for development
   - INFO for production
   - ERROR for issues requiring attention

3. **Rotate log files**:
   - Configure file rotation
   - Set size and count limits
   - Archive old logs

## Security Considerations

1. **Secure monitoring endpoints**:
   - Use authentication for sensitive metrics
   - Restrict access to monitoring dashboards
   - Encrypt communication channels

2. **Protect sensitive data**:
   - Avoid logging sensitive information
   - Sanitize metrics labels
   - Use secure alert channels

3. **Monitor the monitors**:
   - Set up monitoring for monitoring services
   - Alert on monitoring system failures
   - Maintain monitoring infrastructure

## Performance Impact

The monitoring system is designed to have minimal performance impact:

- **Metrics collection**: ~1-2% CPU overhead
- **Health checks**: Run every 5 minutes by default
- **Logging**: Asynchronous with buffering
- **Alerts**: Evaluated every 30 seconds

Adjust collection intervals and retention periods based on your performance requirements.