# AutoML Framework API Documentation

This document provides comprehensive documentation for the AutoML Framework REST API and WebSocket endpoints.

## 📋 Table of Contents

- [API Overview](#api-overview)
- [Authentication](#authentication)
- [Core Endpoints](#core-endpoints)
- [WebSocket API](#websocket-api)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)
- [SDK Usage](#sdk-usage)

## 🌐 API Overview

### Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

### API Version

Current API version: `v1`

All endpoints are prefixed with `/api/v1/`

### Content Types

- **Request**: `application/json` (except file uploads)
- **Response**: `application/json`
- **File Upload**: `multipart/form-data`

### Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

## 🔐 Authentication

### JWT Token Authentication

The API uses JWT (JSON Web Token) for authentication.

#### Login

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "your_password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {
    "id": "user-123",
    "email": "user@example.com",
    "username": "john_doe"
  }
}
```

#### Using the Token

Include the token in the Authorization header:

```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

#### Register

```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "username": "john_doe",
  "password": "secure_password"
}
```

#### Refresh Token

```http
POST /api/v1/auth/refresh
Authorization: Bearer <your_token>
```

## 🔧 Core Endpoints

### Datasets

#### Upload Dataset

```http
POST /api/v1/datasets/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <dataset_file>
name: "My Dataset" (optional)
description: "Dataset description" (optional)
```

**Response:**
```json
{
  "dataset_id": "dataset-123",
  "name": "My Dataset",
  "file_path": "data/uploads/dataset-123.csv",
  "data_type": "TABULAR",
  "size": 1000,
  "features": ["feature1", "feature2", "target"],
  "target_column": "target",
  "metadata": {
    "columns": 3,
    "rows": 1000,
    "missing_values": 5,
    "data_types": {
      "feature1": "float64",
      "feature2": "float64",
      "target": "object"
    }
  },
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### List Datasets

```http
GET /api/v1/datasets
Authorization: Bearer <token>

# Query parameters:
# - page: int = 1
# - limit: int = 20
# - data_type: str = None
# - search: str = None
```

**Response:**
```json
{
  "datasets": [
    {
      "dataset_id": "dataset-123",
      "name": "My Dataset",
      "data_type": "TABULAR",
      "size": 1000,
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "limit": 20,
  "pages": 1
}
```

#### Get Dataset Details

```http
GET /api/v1/datasets/{dataset_id}
Authorization: Bearer <token>
```

#### Delete Dataset

```http
DELETE /api/v1/datasets/{dataset_id}
Authorization: Bearer <token>
```

### Experiments

#### Create Experiment

```http
POST /api/v1/experiments
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "My AutoML Experiment",
  "dataset_id": "dataset-123",
  "task_type": "classification",
  "config": {
    "max_trials": 50,
    "max_epochs": 100,
    "early_stopping_patience": 10,
    "nas_strategy": "darts",
    "hpo_strategy": "bayesian",
    "optimization_metric": "accuracy",
    "cross_validation_folds": 5,
    "test_split": 0.2,
    "random_seed": 42
  }
}
```

**Response:**
```json
{
  "experiment_id": "exp-456",
  "name": "My AutoML Experiment",
  "dataset_id": "dataset-123",
  "status": "CREATED",
  "task_type": "classification",
  "config": {
    "max_trials": 50,
    "max_epochs": 100,
    "early_stopping_patience": 10,
    "nas_strategy": "darts",
    "hpo_strategy": "bayesian",
    "optimization_metric": "accuracy"
  },
  "created_at": "2024-01-15T11:00:00Z",
  "estimated_duration": "2-4 hours"
}
```

#### Start Experiment

```http
POST /api/v1/experiments/{experiment_id}/start
Authorization: Bearer <token>
```

#### List Experiments

```http
GET /api/v1/experiments
Authorization: Bearer <token>

# Query parameters:
# - page: int = 1
# - limit: int = 20
# - status: str = None
# - dataset_id: str = None
```

#### Get Experiment Details

```http
GET /api/v1/experiments/{experiment_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "experiment_id": "exp-456",
  "name": "My AutoML Experiment",
  "dataset_id": "dataset-123",
  "status": "RUNNING",
  "progress": {
    "current_trial": 15,
    "total_trials": 50,
    "completion_percentage": 30,
    "elapsed_time": "45 minutes",
    "estimated_remaining": "1.5 hours"
  },
  "current_best": {
    "trial_id": "trial-789",
    "architecture": {
      "layers": [
        {"type": "dense", "units": 128, "activation": "relu"},
        {"type": "dropout", "rate": 0.3},
        {"type": "dense", "units": 64, "activation": "relu"},
        {"type": "dense", "units": 3, "activation": "softmax"}
      ]
    },
    "hyperparameters": {
      "learning_rate": 0.001,
      "batch_size": 32,
      "optimizer": "adam"
    },
    "metrics": {
      "accuracy": 0.95,
      "loss": 0.15,
      "f1_score": 0.94,
      "precision": 0.96,
      "recall": 0.93
    }
  },
  "created_at": "2024-01-15T11:00:00Z",
  "started_at": "2024-01-15T11:05:00Z"
}
```

#### Stop Experiment

```http
POST /api/v1/experiments/{experiment_id}/stop
Authorization: Bearer <token>
```

#### Get Experiment Results

```http
GET /api/v1/experiments/{experiment_id}/results
Authorization: Bearer <token>
```

**Response:**
```json
{
  "experiment_id": "exp-456",
  "status": "COMPLETED",
  "best_model": {
    "model_id": "model-789",
    "trial_id": "trial-789",
    "architecture": {
      "layers": [
        {"type": "dense", "units": 128, "activation": "relu"},
        {"type": "dropout", "rate": 0.3},
        {"type": "dense", "units": 64, "activation": "relu"},
        {"type": "dense", "units": 3, "activation": "softmax"}
      ],
      "parameter_count": 15432,
      "flops": 2.1e6
    },
    "hyperparameters": {
      "learning_rate": 0.001,
      "batch_size": 32,
      "optimizer": "adam",
      "epochs": 85
    },
    "performance": {
      "train_metrics": {
        "accuracy": 0.98,
        "loss": 0.08,
        "f1_score": 0.97
      },
      "validation_metrics": {
        "accuracy": 0.95,
        "loss": 0.15,
        "f1_score": 0.94
      },
      "test_metrics": {
        "accuracy": 0.94,
        "loss": 0.17,
        "f1_score": 0.93,
        "precision": 0.96,
        "recall": 0.91,
        "confusion_matrix": [[45, 2, 1], [1, 38, 3], [0, 2, 46]]
      }
    },
    "training_history": {
      "epochs": 85,
      "training_time": "2.3 hours",
      "convergence_epoch": 78
    }
  },
  "all_trials": [
    {
      "trial_id": "trial-789",
      "rank": 1,
      "metrics": {"accuracy": 0.94, "loss": 0.17},
      "hyperparameters": {"learning_rate": 0.001, "batch_size": 32}
    }
  ],
  "summary": {
    "total_trials": 50,
    "successful_trials": 47,
    "failed_trials": 3,
    "total_training_time": "3.2 hours",
    "best_accuracy": 0.94,
    "architecture_search_time": "45 minutes",
    "hyperparameter_optimization_time": "2.35 hours"
  }
}
```

### Models

#### List Models

```http
GET /api/v1/models
Authorization: Bearer <token>
```

#### Get Model Details

```http
GET /api/v1/models/{model_id}
Authorization: Bearer <token>
```

#### Export Model

```http
POST /api/v1/models/{model_id}/export
Authorization: Bearer <token>
Content-Type: application/json

{
  "format": "onnx",  # Options: "onnx", "tensorflow", "pytorch", "pickle"
  "include_preprocessing": true,
  "optimize_for_inference": true
}
```

**Response:**
```json
{
  "export_id": "export-123",
  "model_id": "model-789",
  "format": "onnx",
  "file_path": "exports/model-789.onnx",
  "file_size": "15.2 MB",
  "preprocessing_pipeline": "exports/model-789_preprocessing.pkl",
  "metadata": {
    "input_shape": [1, 4],
    "output_shape": [1, 3],
    "class_names": ["setosa", "versicolor", "virginica"]
  },
  "download_url": "/api/v1/models/exports/export-123/download",
  "created_at": "2024-01-15T15:30:00Z",
  "expires_at": "2024-01-22T15:30:00Z"
}
```

#### Download Exported Model

```http
GET /api/v1/models/exports/{export_id}/download
Authorization: Bearer <token>
```

#### Model Inference

```http
POST /api/v1/models/{model_id}/predict
Authorization: Bearer <token>
Content-Type: application/json

{
  "instances": [
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 2.8, 4.8, 1.8]
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "class": "setosa",
      "confidence": 0.98,
      "probabilities": {
        "setosa": 0.98,
        "versicolor": 0.015,
        "virginica": 0.005
      }
    },
    {
      "class": "virginica",
      "confidence": 0.92,
      "probabilities": {
        "setosa": 0.02,
        "versicolor": 0.06,
        "virginica": 0.92
      }
    }
  ],
  "model_id": "model-789",
  "inference_time": "12ms"
}
```

### System

#### Health Check

```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T12:00:00Z",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "mongodb": "healthy",
    "redis": "healthy",
    "gpu": "available"
  },
  "system": {
    "cpu_usage": "45%",
    "memory_usage": "60%",
    "disk_usage": "30%",
    "gpu_usage": "20%"
  }
}
```

#### System Metrics

```http
GET /api/v1/metrics
Authorization: Bearer <token>
```

## 🔌 WebSocket API

### Real-time Experiment Updates

Connect to experiment updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/experiments/exp-456');

ws.onopen = function(event) {
    console.log('Connected to experiment updates');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Update received:', data);
};

ws.onclose = function(event) {
    console.log('Connection closed');
};
```

### Message Types

#### Training Progress

```json
{
  "type": "training_progress",
  "experiment_id": "exp-456",
  "trial_id": "trial-789",
  "epoch": 25,
  "total_epochs": 100,
  "metrics": {
    "loss": 0.25,
    "accuracy": 0.89,
    "val_loss": 0.31,
    "val_accuracy": 0.85
  },
  "timestamp": "2024-01-15T12:30:00Z"
}
```

#### Architecture Search Update

```json
{
  "type": "nas_update",
  "experiment_id": "exp-456",
  "current_architecture": 15,
  "total_architectures": 50,
  "best_architecture": {
    "architecture_id": "arch-123",
    "performance": 0.92,
    "layers": [
      {"type": "conv2d", "filters": 32, "kernel_size": 3},
      {"type": "maxpool2d", "pool_size": 2},
      {"type": "dense", "units": 128}
    ]
  },
  "timestamp": "2024-01-15T12:35:00Z"
}
```

#### Hyperparameter Optimization Update

```json
{
  "type": "hpo_update",
  "experiment_id": "exp-456",
  "trial_id": "trial-789",
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "dropout_rate": 0.3
  },
  "objective_value": 0.94,
  "best_value": 0.95,
  "timestamp": "2024-01-15T12:40:00Z"
}
```

#### Experiment Status Change

```json
{
  "type": "status_change",
  "experiment_id": "exp-456",
  "old_status": "RUNNING",
  "new_status": "COMPLETED",
  "message": "Experiment completed successfully",
  "timestamp": "2024-01-15T14:00:00Z"
}
```

## ❌ Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request data",
    "details": {
      "field": "dataset_id",
      "issue": "Dataset not found"
    },
    "timestamp": "2024-01-15T12:00:00Z",
    "request_id": "req-123"
  }
}
```

### HTTP Status Codes

| Code | Description | Example |
|------|-------------|---------|
| 200 | Success | Request completed successfully |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request data |
| 401 | Unauthorized | Invalid or missing authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource already exists |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Request validation failed |
| `AUTHENTICATION_ERROR` | Authentication failed |
| `AUTHORIZATION_ERROR` | Insufficient permissions |
| `RESOURCE_NOT_FOUND` | Requested resource not found |
| `RESOURCE_CONFLICT` | Resource already exists |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `DATASET_ERROR` | Dataset processing error |
| `EXPERIMENT_ERROR` | Experiment execution error |
| `MODEL_ERROR` | Model training or inference error |
| `SYSTEM_ERROR` | Internal system error |

## 🚦 Rate Limiting

### Rate Limits

| Endpoint Category | Limit | Window |
|-------------------|-------|--------|
| Authentication | 10 requests | 1 minute |
| Dataset Upload | 5 requests | 1 minute |
| Experiment Creation | 10 requests | 1 hour |
| Model Inference | 100 requests | 1 minute |
| General API | 1000 requests | 1 hour |

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248000
X-RateLimit-Window: 3600
```

## 📚 Examples

### Python Example

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:8000/api/v1"

# Login
login_data = {
    "email": "user@example.com",
    "password": "password"
}
response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
token = response.json()["access_token"]

# Headers for authenticated requests
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Upload dataset
with open("dataset.csv", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{BASE_URL}/datasets/upload", 
                           files=files, 
                           headers={"Authorization": f"Bearer {token}"})
dataset_id = response.json()["dataset_id"]

# Create experiment
experiment_data = {
    "name": "My Experiment",
    "dataset_id": dataset_id,
    "task_type": "classification",
    "config": {
        "max_trials": 20,
        "max_epochs": 50
    }
}
response = requests.post(f"{BASE_URL}/experiments", 
                        json=experiment_data, 
                        headers=headers)
experiment_id = response.json()["experiment_id"]

# Start experiment
response = requests.post(f"{BASE_URL}/experiments/{experiment_id}/start", 
                        headers=headers)

# Monitor experiment
response = requests.get(f"{BASE_URL}/experiments/{experiment_id}", 
                       headers=headers)
print(json.dumps(response.json(), indent=2))
```

### JavaScript Example

```javascript
class AutoMLClient {
    constructor(baseUrl, token) {
        this.baseUrl = baseUrl;
        this.token = token;
    }
    
    async request(method, endpoint, data = null) {
        const url = `${this.baseUrl}/api/v1${endpoint}`;
        const options = {
            method,
            headers: {
                'Authorization': `Bearer ${this.token}`,
                'Content-Type': 'application/json'
            }
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(url, options);
        return response.json();
    }
    
    async uploadDataset(file, name) {
        const formData = new FormData();
        formData.append('file', file);
        if (name) formData.append('name', name);
        
        const response = await fetch(`${this.baseUrl}/api/v1/datasets/upload`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.token}`
            },
            body: formData
        });
        
        return response.json();
    }
    
    async createExperiment(config) {
        return this.request('POST', '/experiments', config);
    }
    
    async getExperiment(experimentId) {
        return this.request('GET', `/experiments/${experimentId}`);
    }
    
    connectToExperiment(experimentId, callbacks) {
        const ws = new WebSocket(`ws://localhost:8000/ws/experiments/${experimentId}`);
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (callbacks[data.type]) {
                callbacks[data.type](data);
            }
        };
        
        return ws;
    }
}

// Usage
const client = new AutoMLClient('http://localhost:8000', 'your-token');

// Upload dataset
const fileInput = document.getElementById('file-input');
const dataset = await client.uploadDataset(fileInput.files[0], 'My Dataset');

// Create experiment
const experiment = await client.createExperiment({
    name: 'My Experiment',
    dataset_id: dataset.dataset_id,
    task_type: 'classification'
});

// Monitor experiment
const ws = client.connectToExperiment(experiment.experiment_id, {
    training_progress: (data) => console.log('Training progress:', data),
    status_change: (data) => console.log('Status changed:', data)
});
```

## 🛠️ SDK Usage

### Python SDK

```bash
pip install automl-framework-sdk
```

```python
from automl_framework_sdk import AutoMLClient

# Initialize client
client = AutoMLClient(
    base_url="http://localhost:8000",
    email="user@example.com",
    password="password"
)

# Upload dataset
dataset = client.upload_dataset("dataset.csv", name="My Dataset")

# Create and start experiment
experiment = client.create_experiment(
    name="My Experiment",
    dataset_id=dataset.id,
    task_type="classification",
    max_trials=50
)

# Monitor experiment with callback
def on_progress(progress):
    print(f"Progress: {progress.completion_percentage}%")

experiment.start()
experiment.monitor(on_progress=on_progress)

# Get results
results = experiment.get_results()
print(f"Best accuracy: {results.best_model.performance.test_metrics.accuracy}")

# Export model
model_file = experiment.best_model.export(format="onnx")
print(f"Model exported to: {model_file}")
```

This API documentation provides comprehensive coverage of all available endpoints and usage patterns. For the most up-to-date information, always refer to the interactive documentation at `/docs` when running the service.