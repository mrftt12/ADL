#!/bin/bash

# Deploy AutoML Framework to Kubernetes

set -e

echo "🚀 Deploying AutoML Framework to Kubernetes..."

# Check if kubectl is installed and configured
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if kustomize is installed
if ! command -v kustomize &> /dev/null; then
    echo "❌ kustomize is not installed. Please install kustomize first."
    exit 1
fi

# Check cluster connection
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot connect to Kubernetes cluster. Please check your kubeconfig."
    exit 1
fi

# Create secrets directory if it doesn't exist
mkdir -p k8s/secrets

# Generate secrets if they don't exist
if [ ! -f k8s/secrets/database-password ]; then
    echo "🔐 Generating database password..."
    openssl rand -base64 32 > k8s/secrets/database-password
fi

if [ ! -f k8s/secrets/mongodb-password ]; then
    echo "🔐 Generating MongoDB password..."
    openssl rand -base64 32 > k8s/secrets/mongodb-password
fi

if [ ! -f k8s/secrets/redis-password ]; then
    echo "🔐 Generating Redis password..."
    openssl rand -base64 32 > k8s/secrets/redis-password
fi

if [ ! -f k8s/secrets/jwt-secret ]; then
    echo "🔐 Generating JWT secret..."
    openssl rand -base64 64 > k8s/secrets/jwt-secret
fi

# Build and push Docker images
echo "🔨 Building Docker images..."
docker build -t automl/api:latest -f Dockerfile .
docker build -t automl/worker:latest -f Dockerfile.worker .
docker build -t automl/frontend:latest -f Dockerfile.frontend .

# Tag and push images (adjust registry as needed)
REGISTRY=${REGISTRY:-"your-registry.com"}
if [ "$REGISTRY" != "your-registry.com" ]; then
    echo "📤 Pushing images to registry..."
    docker tag automl/api:latest $REGISTRY/automl/api:latest
    docker tag automl/worker:latest $REGISTRY/automl/worker:latest
    docker tag automl/frontend:latest $REGISTRY/automl/frontend:latest
    
    docker push $REGISTRY/automl/api:latest
    docker push $REGISTRY/automl/worker:latest
    docker push $REGISTRY/automl/frontend:latest
fi

# Apply Kubernetes manifests
echo "📋 Applying Kubernetes manifests..."
cd k8s
kustomize build . | kubectl apply -f -

# Wait for deployments to be ready
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/postgres -n automl
kubectl wait --for=condition=available --timeout=300s deployment/mongodb -n automl
kubectl wait --for=condition=available --timeout=300s deployment/redis -n automl
kubectl wait --for=condition=available --timeout=300s deployment/automl-api -n automl
kubectl wait --for=condition=available --timeout=300s deployment/automl-frontend -n automl

# Check pod status
echo "🔍 Checking pod status..."
kubectl get pods -n automl

# Get service information
echo ""
echo "🎉 AutoML Framework deployed successfully!"
echo ""
echo "📊 Services:"
kubectl get services -n automl

echo ""
echo "🌐 Ingress:"
kubectl get ingress -n automl

echo ""
echo "💾 Storage:"
kubectl get pvc -n automl

echo ""
echo "🔧 Management commands:"
echo "  - View pods: kubectl get pods -n automl"
echo "  - View logs: kubectl logs -f deployment/automl-api -n automl"
echo "  - Port forward API: kubectl port-forward service/automl-api-service 8000:8000 -n automl"
echo "  - Port forward Frontend: kubectl port-forward service/automl-frontend-service 3000:80 -n automl"
echo "  - Scale API: kubectl scale deployment automl-api --replicas=5 -n automl"
echo ""

# Run health check
echo "🏥 Running health check..."
if kubectl port-forward service/automl-api-service 8000:8000 -n automl &
then
    PF_PID=$!
    sleep 5
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API health check passed"
    else
        echo "❌ API health check failed"
    fi
    kill $PF_PID
fi