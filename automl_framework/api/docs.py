"""
API documentation configuration and customization.

This module provides OpenAPI/Swagger documentation customization
and API documentation utilities.
"""

from typing import Any, Dict

# OpenAPI custom configuration
openapi_tags = [
    {
        "name": "authentication",
        "description": "User authentication and authorization endpoints",
    },
    {
        "name": "datasets",
        "description": "Dataset upload, analysis, and management operations",
    },
    {
        "name": "experiments",
        "description": "AutoML experiment creation, monitoring, and management",
    },
    {
        "name": "resources",
        "description": "System resource monitoring and job management",
    },
    {
        "name": "models",
        "description": "Trained model management and inference endpoints",
    },
]

def get_openapi_config() -> Dict[str, Any]:
    """
    Get custom OpenAPI configuration.
    
    Returns:
        Dict containing OpenAPI configuration
    """
    return {
        "title": "AutoML Framework API",
        "description": """
        ## AutoML Framework REST API
        
        This API provides comprehensive endpoints for automated machine learning workflows including:
        
        * **Dataset Management**: Upload, analyze, and manage datasets
        * **Experiment Management**: Create, run, and monitor AutoML experiments
        * **Resource Monitoring**: Track system resources and job status
        * **Model Management**: Access trained models and make predictions
        * **Authentication**: Secure access with JWT tokens
        
        ### Getting Started
        
        1. **Authentication**: Login with credentials to get an access token
        2. **Upload Dataset**: Upload your dataset for analysis
        3. **Create Experiment**: Configure and create an AutoML experiment
        4. **Monitor Progress**: Track experiment progress and resource usage
        5. **Get Results**: Retrieve trained models and performance metrics
        
        ### Authentication
        
        Most endpoints require authentication using JWT Bearer tokens:
        
        ```
        Authorization: Bearer <your-jwt-token>
        ```
        
        Use the `/api/v1/auth/login` endpoint to obtain a token.
        
        ### Rate Limiting
        
        API requests are rate-limited to prevent abuse:
        - 100 requests per 15-minute window per user
        - Higher limits available for authenticated users
        
        ### Error Handling
        
        The API uses standard HTTP status codes and returns detailed error messages:
        - `400` - Bad Request (validation errors)
        - `401` - Unauthorized (authentication required)
        - `403` - Forbidden (insufficient permissions)
        - `404` - Not Found (resource doesn't exist)
        - `422` - Unprocessable Entity (validation failed)
        - `500` - Internal Server Error (system error)
        
        ### Support
        
        For support and documentation, visit our GitHub repository or contact the development team.
        """,
        "version": "1.0.0",
        "contact": {
            "name": "AutoML Framework Team",
            "email": "support@automl-framework.com",
            "url": "https://github.com/automl-framework/automl-framework"
        },
        "license": {
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        },
        "servers": [
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            },
            {
                "url": "https://api.automl-framework.com",
                "description": "Production server"
            }
        ],
        "tags": openapi_tags
    }

def get_api_examples() -> Dict[str, Any]:
    """
    Get API usage examples for documentation.
    
    Returns:
        Dict containing API usage examples
    """
    return {
        "authentication": {
            "login": {
                "summary": "Login to get access token",
                "value": {
                    "username": "demo_user",
                    "password": "secret"
                }
            }
        },
        "dataset_upload": {
            "csv_upload": {
                "summary": "Upload CSV dataset",
                "description": "Upload a CSV file for analysis",
                "value": {
                    "file": "dataset.csv",
                    "name": "Customer Data",
                    "description": "Customer segmentation dataset"
                }
            }
        },
        "experiment_creation": {
            "classification": {
                "summary": "Create classification experiment",
                "value": {
                    "name": "Customer Segmentation",
                    "dataset_path": "dataset_id_123",
                    "task_type": "classification",
                    "data_type": "tabular",
                    "target_column": "segment",
                    "config": {
                        "max_trials": 50,
                        "timeout_minutes": 120
                    }
                }
            },
            "regression": {
                "summary": "Create regression experiment",
                "value": {
                    "name": "Price Prediction",
                    "dataset_path": "dataset_id_456",
                    "task_type": "regression",
                    "data_type": "tabular",
                    "target_column": "price",
                    "config": {
                        "max_trials": 30,
                        "timeout_minutes": 60
                    }
                }
            }
        }
    }