# Docker Deployment Guide

This guide covers deploying the AutoML Framework using Docker and Docker Compose for both development and production environments.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Docker Compose Services](#docker-compose-services)
- [Environment Configuration](#environment-configuration)
- [Production Deployment](#production-deployment)
- [Scaling and Load Balancing](#scaling-and-load-balancing)
- [Monitoring and Logging](#monitoring-and-logging)
- [Backup and Recovery](#backup-and-recovery)
- [Troubleshooting](#troubleshooting)

## ⚡ Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 8GB RAM minimum (16GB recommended)
- 20GB free disk space

### Basic Deployment

```bash
# Clone repository
git clone <repository-url>
cd automl-framework

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### Using Development Scripts

```bash
# Start with development script (recommended)
./scripts/dev-start.sh

# Start specific profile
./scripts/dev-start.sh -p api --no-monitoring

# Clean start
./scripts/dev-start.sh --clean
```

## 🐳 Docker Compose Services

### Core Application Services

#### API Service
```yaml
api:
  build:
    context: .
    dockerfile: Dockerfile
  ports:
    - "8000:8000"
  environment:
    - DATABASE_URL=postgresql://automl:automl_password@postgres:5432/automl
    - MONGODB_URL=mongodb://automl:automl_password@mongodb:27017/automl
    - REDIS_URL=redis://:automl_password@redis:6379/0
  depends_on:
    - postgres
    - mongodb
    - redis
```

#### Worker Services
```yaml
training-worker:
  build:
    context: .
    dockerfile: Dockerfile.worker
  environment:
    - WORKER_TYPE=training
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
```

#### Frontend Service
```yaml
frontend:
  build:
    context: .
    dockerfile: Dockerfile.frontend
  ports:
    - "3000:80"
  depends_on:
    - api
```

### Database Services

#### PostgreSQL
```yaml
postgres:
  image: postgres:14-alpine
  environment:
    POSTGRES_DB: automl
    POSTGRES_USER: automl
    POSTGRES_PASSWORD: automl_password
  volumes:
    - postgres_data:/var/lib/postgresql/data
    - ./docker/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
  ports:
    - "5432:5432"
```

#### MongoDB
```yaml
mongodb:
  image: mongo:6.0
  environment:
    MONGO_INITDB_ROOT_USERNAME: automl
    MONGO_INITDB_ROOT_PASSWORD: automl_password
    MONGO_INITDB_DATABASE: automl
  volumes:
    - mongodb_data:/data/db
    - ./docker/init-mongo.js:/docker-entrypoint-initdb.d/init-mongo.js:ro
  ports:
    - "27017:27017"
```

#### Redis
```yaml
redis:
  image: redis:7-alpine
  command: redis-server --appendonly yes --requirepass automl_password
  volumes:
    - redis_data:/data
  ports:
    - "6379:6379"
```

### Monitoring Services

#### Prometheus
```yaml
prometheus:
  image: prom/prometheus:latest
  ports:
    - "9090:9090"
  volumes:
    - ./docker/prometheus.yml:/etc/prometheus/prometheus.yml
    - prometheus_data:/prometheus
  command:
    - '--config.file=/etc/prometheus/prometheus.yml'
    - '--storage.tsdb.path=/prometheus'
    - '--web.enable-lifecycle'
```

#### Grafana
```yaml
grafana:
  image: grafana/grafana:latest
  ports:
    - "3001:3000"
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=admin
  volumes:
    - grafana_data:/var/lib/grafana
    - ./docker/grafana-dashboard.json:/etc/grafana/provisioning/dashboards/automl-dashboard.json
```

## ⚙️ Environment Configuration

### Development Environment

Create `.env.dev` for development:

```bash
# Application
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DEBUG=true

# Database URLs
DATABASE_URL=postgresql://automl:automl_password@postgres:5432/automl
MONGODB_URL=mongodb://automl:automl_password@mongodb:27017/automl
REDIS_URL=redis://:automl_password@redis:6379/0

# Security
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET_KEY=dev-jwt-secret-change-in-production

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Production Environment

Create `.env.prod` for production:

```bash
# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Database URLs (use strong passwords!)
DATABASE_URL=postgresql://automl:${POSTGRES_PASSWORD}@postgres:5432/automl
MONGODB_URL=mongodb://automl:${MONGO_PASSWORD}@mongodb:27017/automl
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0

# Security (generate strong keys!)
SECRET_KEY=${SECRET_KEY}
JWT_SECRET_KEY=${JWT_SECRET_KEY}

# Performance
WORKER_CONCURRENCY=4
API_WORKERS=4

# Monitoring
PROMETHEUS_RETENTION=30d
GRAFANA_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
```

### Docker Compose Override

Create `docker-compose.override.yml` for local customizations:

```yaml
version: '3.8'

services:
  api:
    volumes:
      - ./automl_framework:/app/automl_framework:ro
    environment:
      - LOG_LEVEL=DEBUG
    
  postgres:
    ports:
      - "5433:5432"  # Avoid conflicts with local PostgreSQL
    
  mongodb:
    ports:
      - "27018:27017"  # Avoid conflicts with local MongoDB
```

## 🚀 Production Deployment

### Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    restart: unless-stopped

  postgres:
    image: postgres:14-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: automl
      POSTGRES_USER: automl
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    secrets:
      - postgres_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt
  mongodb_password:
    file: ./secrets/mongodb_password.txt
  redis_password:
    file: ./secrets/redis_password.txt
```

### SSL/TLS Configuration

```bash
# Generate SSL certificates (for development)
mkdir -p ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/automl.key \
  -out ssl/automl.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=automl.local"

# For production, use Let's Encrypt or your certificate authority
```

### Nginx Configuration

Create `docker/nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }
    
    upstream frontend {
        server frontend:80;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=1r/s;
    
    server {
        listen 80;
        server_name automl.local;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name automl.local;
        
        ssl_certificate /etc/nginx/ssl/automl.crt;
        ssl_certificate_key /etc/nginx/ssl/automl.key;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        
        # API routes
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # File upload routes (with higher limits)
        location /api/v1/datasets/upload {
            limit_req zone=upload burst=5 nodelay;
            client_max_body_size 1G;
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 300s;
            proxy_send_timeout 300s;
        }
        
        # WebSocket routes
        location /ws/ {
            proxy_pass http://api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Frontend routes
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

### Deployment Script

Create `scripts/deploy-prod.sh`:

```bash
#!/bin/bash

set -e

echo "🚀 Deploying AutoML Framework to production..."

# Load environment variables
if [[ -f ".env.prod" ]]; then
    export $(grep -v '^#' .env.prod | xargs)
fi

# Build images
echo "🔨 Building production images..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build --no-cache

# Run database migrations
echo "📊 Running database migrations..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml run --rm api python -m automl_framework.migrations.migration_manager

# Start services
echo "🚀 Starting production services..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 30

# Health check
echo "🏥 Running health checks..."
if curl -f https://automl.local/api/v1/health > /dev/null 2>&1; then
    echo "✅ Production deployment successful!"
else
    echo "❌ Health check failed"
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs
    exit 1
fi

echo "🎉 AutoML Framework is now running in production!"
echo "🌐 Access the application at: https://automl.local"
```

## 📈 Scaling and Load Balancing

### Horizontal Scaling

Scale specific services:

```bash
# Scale API service
docker-compose up -d --scale api=3

# Scale worker services
docker-compose up -d --scale training-worker=2 --scale nas-worker=2

# Scale with resource limits
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale api=3
```

### Docker Swarm Deployment

Initialize Docker Swarm:

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml -c docker-compose.prod.yml automl

# Scale services
docker service scale automl_api=3
docker service scale automl_training-worker=2

# Check service status
docker service ls
docker service ps automl_api
```

### Kubernetes Deployment

See `k8s/` directory for Kubernetes manifests:

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Scale deployments
kubectl scale deployment automl-api --replicas=3
kubectl scale deployment automl-workers --replicas=2

# Check status
kubectl get pods
kubectl get services
```

## 📊 Monitoring and Logging

### Centralized Logging

Configure log aggregation:

```yaml
# docker-compose.logging.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:8.5.0
    volumes:
      - ./docker/logstash.conf:/usr/share/logstash/pipeline/logstash.conf

  kibana:
    image: docker.elastic.co/kibana/kibana:8.5.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200

volumes:
  elasticsearch_data:
```

### Log Configuration

Configure structured logging:

```python
# automl_framework/utils/logging.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'experiment_id'):
            log_entry['experiment_id'] = record.experiment_id
            
        return json.dumps(log_entry)
```

### Metrics Collection

Configure custom metrics:

```python
# automl_framework/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
experiment_counter = Counter('automl_experiments_total', 'Total experiments created')
training_duration = Histogram('automl_training_duration_seconds', 'Training duration')
active_experiments = Gauge('automl_active_experiments', 'Currently active experiments')

# Use in code
experiment_counter.inc()
with training_duration.time():
    # Training code here
    pass
```

## 💾 Backup and Recovery

### Database Backup

Create backup scripts:

```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# PostgreSQL backup
docker-compose exec -T postgres pg_dump -U automl automl > "$BACKUP_DIR/postgres.sql"

# MongoDB backup
docker-compose exec -T mongodb mongodump --uri="mongodb://automl:automl_password@localhost:27017/automl" --out="$BACKUP_DIR/mongodb"

# Redis backup
docker-compose exec -T redis redis-cli --rdb "$BACKUP_DIR/redis.rdb"

# Compress backup
tar -czf "$BACKUP_DIR.tar.gz" -C "$BACKUP_DIR" .
rm -rf "$BACKUP_DIR"

echo "Backup completed: $BACKUP_DIR.tar.gz"
```

### Automated Backups

Add to crontab:

```bash
# Run daily backups at 2 AM
0 2 * * * /path/to/automl-framework/scripts/backup.sh
```

### Recovery Process

```bash
#!/bin/bash
# scripts/restore.sh

BACKUP_FILE="$1"

if [[ -z "$BACKUP_FILE" ]]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

# Extract backup
RESTORE_DIR="/tmp/restore_$(date +%s)"
mkdir -p "$RESTORE_DIR"
tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR"

# Stop services
docker-compose down

# Restore PostgreSQL
docker-compose up -d postgres
sleep 10
docker-compose exec -T postgres psql -U automl -d automl < "$RESTORE_DIR/postgres.sql"

# Restore MongoDB
docker-compose up -d mongodb
sleep 10
docker-compose exec -T mongodb mongorestore --uri="mongodb://automl:automl_password@localhost:27017/automl" "$RESTORE_DIR/mongodb/automl"

# Restore Redis
docker-compose up -d redis
sleep 5
docker-compose exec -T redis redis-cli --rdb "$RESTORE_DIR/redis.rdb"

# Start all services
docker-compose up -d

echo "Restore completed from: $BACKUP_FILE"
```

## 🔧 Troubleshooting

### Common Issues

#### Out of Memory

```bash
# Check memory usage
docker stats

# Increase memory limits
# In docker-compose.yml:
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
```

#### Disk Space Issues

```bash
# Check disk usage
docker system df

# Clean up unused resources
docker system prune -f
docker volume prune -f

# Remove old images
docker image prune -a -f
```

#### Network Issues

```bash
# Check network connectivity
docker network ls
docker network inspect automl-framework_automl-network

# Recreate network
docker-compose down
docker network prune -f
docker-compose up -d
```

#### Service Dependencies

```bash
# Check service startup order
docker-compose logs postgres
docker-compose logs api

# Restart with proper dependencies
docker-compose down
docker-compose up -d postgres mongodb redis
sleep 30
docker-compose up -d
```

### Performance Tuning

#### Database Performance

```sql
-- PostgreSQL tuning
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
SELECT pg_reload_conf();
```

#### Container Resource Limits

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

#### Volume Performance

```yaml
volumes:
  postgres_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /fast/ssd/postgres_data
```

This Docker deployment guide provides comprehensive instructions for deploying the AutoML Framework in various environments. For specific deployment scenarios or issues, refer to the troubleshooting section or consult the development team.