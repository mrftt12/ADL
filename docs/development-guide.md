# AutoML Framework Development Guide

This guide provides detailed instructions for setting up and developing the AutoML Framework locally.

## 📋 Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Service Architecture](#service-architecture)
- [Development Workflows](#development-workflows)
- [Testing](#testing)
- [Debugging](#debugging)
- [Performance Optimization](#performance-optimization)
- [Contributing Guidelines](#contributing-guidelines)

## 🛠️ Development Environment Setup

### Prerequisites

Ensure you have the following installed:

- **Docker Desktop** (latest version)
- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **Git** for version control
- **NVIDIA GPU drivers** (optional, for GPU acceleration)

### Initial Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd automl-framework
   ```

2. **Set Up Development Environment**
   ```bash
   # Make scripts executable
   chmod +x scripts/*.sh
   
   # Start development environment
   ./scripts/dev-start.sh
   ```

3. **Verify Installation**
   ```bash
   # Check service health
   ./scripts/dev-health.sh
   
   # View service URLs
   echo "API: http://localhost:8000/docs"
   echo "Frontend: http://localhost:3000"
   echo "Grafana: http://localhost:3001"
   ```

### Development Modes

#### Docker Mode (Recommended for New Developers)

```bash
# Start all services in Docker
./scripts/dev-start.sh -m docker -p full

# Benefits:
# - Consistent environment across machines
# - No local dependency conflicts
# - Easy cleanup and reset
```

#### Native Mode (For Active Development)

```bash
# Start databases in Docker, services natively
./scripts/dev-start.sh -m hybrid -p api

# Benefits:
# - Faster code reload
# - Direct debugging access
# - IDE integration
```

#### Minimal Mode (For Backend Development)

```bash
# Start only databases
./scripts/dev-start.sh -p minimal

# Then manually start services you're working on
python run_api.py
```

## 🏗️ Service Architecture

### Core Services

| Service | Port | Description | Development Notes |
|---------|------|-------------|-------------------|
| **API Gateway** | 8000 | FastAPI application | Hot reload enabled in dev mode |
| **Frontend** | 3000 | React application | Vite dev server with HMR |
| **Training Worker** | - | Background training service | Connects to job queue |
| **NAS Worker** | - | Architecture search service | GPU-intensive operations |
| **HPO Worker** | - | Hyperparameter optimization | CPU/GPU hybrid workload |

### Infrastructure Services

| Service | Port | Description | Development Notes |
|---------|------|-------------|-------------------|
| **PostgreSQL** | 5432 | Primary database | Persistent data storage |
| **MongoDB** | 27017 | Document storage | Architecture definitions |
| **Redis** | 6379 | Cache and job queue | In-memory operations |
| **Prometheus** | 9090 | Metrics collection | Monitoring and alerting |
| **Grafana** | 3001 | Visualization dashboard | admin/admin credentials |

### Service Dependencies

```mermaid
graph TD
    A[API Gateway] --> B[PostgreSQL]
    A --> C[MongoDB]
    A --> D[Redis]
    
    E[Training Worker] --> B
    E --> C
    E --> D
    
    F[NAS Worker] --> C
    F --> D
    
    G[HPO Worker] --> B
    G --> D
    
    H[Frontend] --> A
    
    I[Prometheus] --> A
    I --> E
    I --> F
    I --> G
    
    J[Grafana] --> I
```

## 🔄 Development Workflows

### Starting Development

1. **Start Core Services**
   ```bash
   ./scripts/dev-start.sh -p minimal
   ```

2. **Start Your Service**
   ```bash
   # API development
   python run_api.py
   
   # Frontend development
   cd ui && npm run dev
   
   # Worker development
   python -m automl_framework.services.training_service
   ```

3. **Make Changes and Test**
   ```bash
   # Run tests
   pytest tests/test_your_feature.py -v
   
   # Check code style
   black automl_framework/
   flake8 automl_framework/
   ```

### Hot Reload Configuration

#### Python Services (FastAPI)

```python
# run_api.py automatically enables reload in development
if __name__ == "__main__":
    run_server(
        host="0.0.0.0",
        port=8000,
        reload=True  # Enables hot reload
    )
```

#### Frontend (React + Vite)

```bash
# Vite automatically enables HMR
cd ui
npm run dev  # Starts with hot module replacement
```

### Database Development

#### Running Migrations

```bash
# Initialize fresh databases
./scripts/dev-migrate.sh --init

# Run pending migrations only
./scripts/dev-migrate.sh --migrate

# Reset databases (destroys data!)
./scripts/dev-migrate.sh --reset

# Add sample data
./scripts/dev-migrate.sh --seed
```

#### Creating New Migrations

```python
# 1. Update ORM models in automl_framework/models/orm_models.py
# 2. Create migration script in automl_framework/migrations/scripts/
# 3. Update migration manager to include new script
```

#### Database Schema Changes

```bash
# After modifying models, update the database
./scripts/dev-migrate.sh --migrate

# If you need to reset everything
./scripts/dev-migrate.sh --reset
./scripts/dev-migrate.sh --init
./scripts/dev-migrate.sh --seed
```

## 🧪 Testing

### Test Structure

```
tests/
├── unit/                 # Unit tests for individual components
├── integration/          # Integration tests for service communication
├── end_to_end/          # Full pipeline tests
├── fixtures/            # Test data and fixtures
└── conftest.py          # Pytest configuration
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/end_to_end/

# Run with coverage
pytest --cov=automl_framework --cov-report=html

# Run specific test file
pytest tests/test_training_service.py -v

# Run tests matching pattern
pytest -k "test_nas" -v
```

### Writing Tests

#### Unit Test Example

```python
# tests/unit/test_data_processing.py
import pytest
from automl_framework.services.data_processing import DatasetAnalyzer

class TestDatasetAnalyzer:
    def test_analyze_csv_dataset(self):
        analyzer = DatasetAnalyzer()
        metadata = analyzer.analyze_dataset("tests/fixtures/sample.csv")
        
        assert metadata.data_type == "TABULAR"
        assert len(metadata.features) > 0
        assert metadata.size > 0
```

#### Integration Test Example

```python
# tests/integration/test_api_integration.py
import pytest
from fastapi.testclient import TestClient
from automl_framework.api.main import app

client = TestClient(app)

def test_create_experiment():
    # Upload dataset first
    with open("tests/fixtures/sample.csv", "rb") as f:
        response = client.post("/api/v1/datasets/upload", files={"file": f})
    
    dataset_id = response.json()["dataset_id"]
    
    # Create experiment
    experiment_data = {
        "name": "Test Experiment",
        "dataset_id": dataset_id,
        "task_type": "classification"
    }
    
    response = client.post("/api/v1/experiments", json=experiment_data)
    assert response.status_code == 201
```

### Test Data Management

```bash
# Create test fixtures
mkdir -p tests/fixtures
echo "feature1,feature2,target" > tests/fixtures/sample.csv
echo "1.0,2.0,A" >> tests/fixtures/sample.csv
echo "2.0,3.0,B" >> tests/fixtures/sample.csv
```

## 🐛 Debugging

### Logging Configuration

#### Python Services

```python
# Set log level in environment
export LOG_LEVEL=DEBUG

# Or in code
import logging
logging.getLogger("automl_framework").setLevel(logging.DEBUG)
```

#### Viewing Logs

```bash
# Follow all service logs
./scripts/dev-start.sh --logs

# View specific service logs
docker-compose logs -f api
docker-compose logs -f training-worker

# View native service logs
tail -f logs/api.log
tail -f logs/automl.log
```

### Debugging Tools

#### Python Debugger

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use modern debugger
import ipdb; ipdb.set_trace()
```

#### VS Code Debugging

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug API",
            "type": "python",
            "request": "launch",
            "program": "run_api.py",
            "env": {
                "LOG_LEVEL": "DEBUG"
            }
        }
    ]
}
```

#### Database Debugging

```bash
# Connect to PostgreSQL
PGPASSWORD=automl_password psql -h localhost -U automl -d automl

# Connect to MongoDB
mongosh "mongodb://automl:automl_password@localhost:27017/automl"

# Connect to Redis
redis-cli -a automl_password
```

### Common Debugging Scenarios

#### Service Won't Start

```bash
# Check port conflicts
lsof -i :8000

# Check logs for errors
./scripts/dev-health.sh
docker-compose logs api

# Reset environment
./scripts/dev-stop.sh --clean
./scripts/dev-start.sh
```

#### Database Connection Issues

```bash
# Check database status
./scripts/dev-health.sh

# Test connections manually
python -c "from automl_framework.core.database import get_db_session; print('DB OK')"

# Reset databases
./scripts/dev-migrate.sh --reset
```

#### GPU Issues

```bash
# Check GPU availability
nvidia-smi

# Test PyTorch GPU access
python -c "import torch; print(torch.cuda.is_available())"

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## ⚡ Performance Optimization

### Development Performance

#### Faster Startup

```bash
# Start only what you need
./scripts/dev-start.sh -p minimal

# Skip monitoring services
./scripts/dev-start.sh --no-monitoring

# Use hybrid mode for faster code reload
./scripts/dev-start.sh -m hybrid
```

#### Efficient Testing

```bash
# Run tests in parallel
pytest -n auto

# Run only changed tests
pytest --lf  # Last failed
pytest --ff  # Failed first

# Skip slow tests during development
pytest -m "not slow"
```

### Production Performance

#### Database Optimization

```sql
-- Add indexes for common queries
CREATE INDEX idx_experiments_user_id ON experiments(user_id);
CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_datasets_created_at ON datasets(created_at);
```

#### Caching Strategy

```python
# Use Redis for caching expensive operations
from automl_framework.core.cache import cache_result

@cache_result(ttl=3600)  # Cache for 1 hour
def expensive_computation(data):
    # Expensive operation here
    return result
```

#### Resource Management

```bash
# Monitor resource usage
./scripts/dev-health.sh --watch

# Adjust worker concurrency
export WORKER_CONCURRENCY=4

# Limit GPU memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## 🤝 Contributing Guidelines

### Code Style

#### Python

```bash
# Format code
black automl_framework/ tests/
isort automl_framework/ tests/

# Lint code
flake8 automl_framework/ tests/
mypy automl_framework/

# Check docstrings
pydocstyle automl_framework/
```

#### TypeScript/React

```bash
cd ui

# Format code
npm run format

# Lint code
npm run lint

# Type check
npm run type-check
```

### Commit Guidelines

```bash
# Use conventional commits
git commit -m "feat: add neural architecture search service"
git commit -m "fix: resolve database connection timeout"
git commit -m "docs: update API documentation"
git commit -m "test: add integration tests for training service"
```

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes and Test**
   ```bash
   # Make your changes
   # Add tests
   pytest
   
   # Check code style
   black automl_framework/
   flake8 automl_framework/
   ```

3. **Update Documentation**
   ```bash
   # Update relevant docs
   # Add API documentation if needed
   # Update README if necessary
   ```

4. **Submit Pull Request**
   - Provide clear description
   - Link related issues
   - Include test results
   - Request review from maintainers

### Development Best Practices

#### Code Organization

```python
# Follow the established structure
automl_framework/
├── api/           # FastAPI routes and middleware
├── core/          # Core utilities and configuration
├── models/        # Data models and schemas
├── services/      # Business logic services
├── utils/         # Utility functions
└── migrations/    # Database migrations
```

#### Error Handling

```python
# Use custom exceptions
from automl_framework.core.exceptions import AutoMLException

def risky_operation():
    try:
        # Operation that might fail
        pass
    except Exception as e:
        raise AutoMLException(
            message="Operation failed",
            error_code="OPERATION_ERROR",
            recoverable=True
        ) from e
```

#### Configuration Management

```python
# Use centralized configuration
from automl_framework.core.config import get_config

config = get_config()
database_url = config.database_url
```

#### Async/Await Patterns

```python
# Use async for I/O operations
async def process_dataset(dataset_id: str):
    async with get_async_db_session() as session:
        dataset = await session.get(Dataset, dataset_id)
        # Process dataset
        return result
```

This development guide should help you get started with contributing to the AutoML Framework. For specific questions or issues, please check the troubleshooting section or create an issue on GitHub.