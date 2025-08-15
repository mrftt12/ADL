#!/usr/bin/env python3
"""
Simple script to run the AutoML Framework API server.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ.setdefault("PYTHONPATH", str(project_root))

if __name__ == "__main__":
    from automl_framework.api.server import run_server
    
    print("Starting AutoML Framework API Server...")
    print("Visit http://localhost:8000/docs for interactive documentation")
    print("Press Ctrl+C to stop the server")
    
    try:
        run_server(
            host="0.0.0.0",
            port=8000,
            reload=True  # Enable auto-reload for development
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server failed to start: {e}")
        sys.exit(1)