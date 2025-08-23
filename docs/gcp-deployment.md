# Google Cloud Platform Deployment Guide

This guide covers deploying the AutoML Framework to Google Cloud Platform using Cloud Build, Cloud Run, and Google Kubernetes Engine.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Cloud Build Configurations](#cloud-build-configurations)
- [Deployment Options](#deployment-options)
- [Environment Variables](#environment-variables)
- [Monitoring and Logging](#monitoring-and-logging)
- [Troubleshooting](#troubleshooting)

## 🌟 Overview

The AutoML Framework can be deployed to GCP using several approaches:

1. **Cloud Run** - Serverless containers (recommended for development)
2. **Google Kubernetes Engine (GKE)** - Full container orchestration (recommended for production)
3. **Compute Engine** - Virtual machines with Docker Compose

## 📋 Prerequisites

### Required Tools

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Install kubectl
gcloud components install kubectl

# Install Docker (for local testing)
# Follow instructions at: https://docs.docker.com/get-docker/
```

### GCP Setup

1. **Create a GCP Project**
   ```bash
   gcloud projects create your-automl-project
   gcloud config set project your-automl-project
   ```

2. **Enable Billing**
   - Go to [GCP Console](https://console.cloud.google.com)
   - Enable billing for your project

3. **Run Setup Script**
   ```bash
   ./scripts/setup-gcp.sh --project-id your-automl-project --all
   ```

## ⚡ Quick Start

### Option 1: Automated Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd automl-framework

# 2. Run the setup script
./scripts/setup-gcp.sh --project-id YOUR_PROJECT_ID --all

# 3. Trigger a build
gcloud builds submit --config cloudbuild.yaml

# 4. Check deployment status
gcloud run services list
```

### Option 2: Manual Setup

```bash
# 1. Set your project
gcloud config set project YOUR_PROJECT_ID

# 2. Enable APIs
gcloud services enable cloudbuild.googleapis.com container.googleapis.com run.googleapis.com

# 3. Build and deploy
gcloud builds submit --config cloudbuild.yaml
```

## 🔧 Cloud Build Configurations

We provide several Cloud Build configurations for different use cases:

### 1. `cloudbuild.yaml` - Production Deployment

**Use Case**: Full production deployment to Cloud Run and GKE
**Triggers**: Main branch pushes
**Features**:
- Builds all Docker images (API, Worker, Frontend)
- Deploys to Cloud Run
- Updates GKE deployments
- Includes security scanning

```bash
# Manual trigger
gcloud builds submit --config cloudbuild.yaml

# With custom substitutions
gcloud builds submit --config cloudbuild.yaml \
  --substitutions _REGION=us-west1,_DATABASE_URL=your-db-url
```

### 2. `cloudbuild-dev.yaml` - Development Deployment

**Use Case**: Development and testing
**Triggers**: Develop branch pushes
**Features**:
- Runs tests before building
- Deploys to development Cloud Run services
- Smaller resource allocations

```bash
gcloud builds submit --config cloudbuild-dev.yaml
```

### 3. `cloudbuild-gke.yaml` - GKE-Only Deployment

**Use Case**: Kubernetes-native deployment
**Features**:
- Builds images and deploys to GKE
- Creates Kubernetes manifests from docker-compose
- Manages secrets and ConfigMaps

```bash
gcloud builds submit --config cloudbuild-gke.yaml \
  --substitutions _CLUSTER=your-cluster,_ZONE=us-central1-a
```

### 4. `cloudbuild-build-only.yaml` - CI/CD Pipeline

**Use Case**: Build images for external deployment
**Features**:
- Only builds and pushes images
- Creates deployment manifests
- Security scanning
- Multi-tag support

```bash
gcloud builds submit --config cloudbuild-build-only.yaml
```

## 🚀 Deployment Options

### Cloud Run Deployment

**Pros**: Serverless, auto-scaling, pay-per-use
**Cons**: Limited to stateless services, cold starts

```bash
# Deploy API to Cloud Run
gcloud run deploy automl-api \
  --image gcr.io/YOUR_PROJECT_ID/automl-api:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10

# Deploy Frontend to Cloud Run
gcloud run deploy automl-frontend \
  --image gcr.io/YOUR_PROJECT_ID/automl-frontend:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 512Mi \
  --port 8080
```

### GKE Deployment

**Pros**: Full Kubernetes features, persistent storage, complex networking
**Cons**: More complex, always-on costs

```bash
# Create GKE cluster
gcloud container clusters create automl-cluster \
  --zone us-central1-a \
  --machine-type e2-standard-4 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10

# Deploy using kubectl
kubectl apply -f k8s/
```

### Hybrid Deployment

**Recommended Approach**: Use Cloud Run for API/Frontend, GKE for workers

```yaml
# Cloud Run for stateless services
- API Service → Cloud Run
- Frontend → Cloud Run

# GKE for stateful/compute-intensive services  
- Training Workers → GKE
- Database Services → GKE
- Monitoring → GKE
```

## 🔐 Environment Variables

### Required Variables

```bash
# Database connections
DATABASE_URL=postgresql://user:pass@host:5432/db
MONGODB_URL=mongodb://user:pass@host:27017/db
REDIS_URL=redis://:pass@host:6379/0

# Application settings
ENVIRONMENT=production
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret
```

### Cloud Build Substitutions

```yaml
substitutions:
  _REGION: 'us-central1'
  _ZONE: 'us-central1-a'
  _CLUSTER: 'automl-cluster'
  _NAMESPACE: 'automl-production'
  _DATABASE_URL: 'postgresql://...'
  _MONGODB_URL: 'mongodb://...'
  _REDIS_URL: 'redis://...'
```

### Setting Environment Variables

```bash
# For Cloud Run
gcloud run services update automl-api \
  --set-env-vars="DATABASE_URL=postgresql://...,LOG_LEVEL=INFO"

# For GKE (using ConfigMap)
kubectl create configmap automl-config \
  --from-literal=LOG_LEVEL=INFO \
  --from-literal=ENVIRONMENT=production

# For GKE (using Secret)
kubectl create secret generic automl-secrets \
  --from-literal=DATABASE_URL=postgresql://... \
  --from-literal=JWT_SECRET_KEY=your-secret
```

## 📊 Monitoring and Logging

### Cloud Logging

```bash
# View API logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=automl-api" --limit 50

# View build logs
gcloud logging read "resource.type=build" --limit 20

# Create log-based metrics
gcloud logging metrics create error_rate \
  --description="Error rate for AutoML API" \
  --log-filter="resource.type=cloud_run_revision AND severity>=ERROR"
```

### Cloud Monitoring

```bash
# Create alerting policy
gcloud alpha monitoring policies create --policy-from-file=monitoring-policy.json
```

Example monitoring policy:
```json
{
  "displayName": "AutoML High Error Rate",
  "conditions": [
    {
      "displayName": "High error rate",
      "conditionThreshold": {
        "filter": "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"automl-api\"",
        "comparison": "COMPARISON_GREATER_THAN",
        "thresholdValue": 10,
        "duration": "300s"
      }
    }
  ],
  "combiner": "OR",
  "enabled": true
}
```

### Application Monitoring

The AutoML Framework includes built-in monitoring:

- **Prometheus metrics**: Available at `/api/v1/monitoring/metrics/prometheus`
- **Health checks**: Available at `/health`
- **Grafana dashboards**: Deployed with monitoring stack

## 🔧 Troubleshooting

### Common Issues

#### Build Failures

```bash
# Check build logs
gcloud builds log BUILD_ID

# Common fixes
# 1. Increase timeout
options:
  timeout: '2400s'

# 2. Use larger machine
options:
  machineType: 'E2_HIGHCPU_8'

# 3. Check Docker context
.dockerignore
```

#### Deployment Failures

```bash
# Check Cloud Run logs
gcloud run services logs read automl-api --region us-central1

# Check GKE pod status
kubectl get pods -n automl-production
kubectl describe pod POD_NAME -n automl-production
kubectl logs POD_NAME -n automl-production
```

#### Database Connection Issues

```bash
# Test database connectivity
gcloud sql connect INSTANCE_NAME --user=automl

# Check firewall rules
gcloud compute firewall-rules list

# Verify service account permissions
gcloud projects get-iam-policy PROJECT_ID
```

#### Memory/Resource Issues

```bash
# Increase Cloud Run memory
gcloud run services update automl-api \
  --memory 4Gi \
  --cpu 2

# Scale GKE nodes
gcloud container clusters resize automl-cluster \
  --num-nodes 5 \
  --zone us-central1-a
```

### Debug Commands

```bash
# Check service status
gcloud run services list
kubectl get all -n automl-production

# View recent builds
gcloud builds list --limit 10

# Check container images
gcloud container images list --repository gcr.io/PROJECT_ID

# Test endpoints
curl -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  https://automl-api-xxx-uc.a.run.app/health
```

### Performance Optimization

#### Cloud Run Optimization

```yaml
# Optimize for cold starts
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/execution-environment: gen2
    spec:
      containerConcurrency: 100
      timeoutSeconds: 300
```

#### GKE Optimization

```yaml
# Resource requests and limits
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"

# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: automl-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: automl-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 📚 Additional Resources

- [Cloud Build Documentation](https://cloud.google.com/build/docs)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
- [Container Registry](https://cloud.google.com/container-registry/docs)
- [Cloud SQL](https://cloud.google.com/sql/docs)
- [Memorystore](https://cloud.google.com/memorystore/docs)

## 🆘 Support

For deployment issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Review Cloud Build logs
3. Verify GCP permissions and quotas
4. Check service health endpoints
5. Review monitoring dashboards

This deployment guide provides comprehensive coverage for deploying the AutoML Framework to Google Cloud Platform using modern cloud-native practices.