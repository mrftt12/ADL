"""
API server startup script for AutoML Framework.

This module provides server configuration and startup utilities.
"""

import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from automl_framework.api.main import app
from automl_framework.api.docs import get_openapi_config
from automl_framework.core.config import get_config
from automl_framework.core.registry import initialize_service_registry

logger = logging.getLogger(__name__)

def setup_logging():
    """Configure logging for the API server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/api.log", mode="a")
        ]
    )
    
    # Set specific log levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("automl_framework").setLevel(logging.DEBUG)

def setup_directories():
    """Create necessary directories for the API server."""
    directories = [
        "data/uploads",
        "logs",
        "models",
        "checkpoints"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def initialize_services():
    """Initialize AutoML services."""
    try:
        logger.info("Initializing AutoML services...")
        initialize_service_registry()
        logger.info("AutoML services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

def customize_openapi():
    """Customize OpenAPI documentation."""
    config = get_openapi_config()
    
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = app.openapi()
        openapi_schema.update(config)
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    # Setup logging
    setup_logging()
    logger.info("Starting AutoML Framework API server...")
    
    # Setup directories
    setup_directories()
    
    # Initialize services
    initialize_services()
    
    # Customize OpenAPI documentation
    customize_openapi()
    
    logger.info("API server setup completed")
    return app

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1
):
    """
    Run the API server with uvicorn.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
        workers: Number of worker processes
    """
    # Create the app
    app = create_app()
    
    # Configure uvicorn
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,  # Reload doesn't work with multiple workers
        log_level="info",
        access_log=True,
        use_colors=True
    )
    
    # Run the server
    server = uvicorn.Server(config)
    logger.info(f"Starting server on {host}:{port}")
    server.run()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AutoML Framework API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )