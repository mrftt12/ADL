# Design Document

## Overview

This design addresses the two critical issues preventing successful local Docker Compose deployment:
1. Authentication system failures where logged-in users aren't recognized
2. GPU-related errors when running in Docker containers without GPU support

The solution implements environment detection and graceful fallbacks to ensure the application works seamlessly in Docker Compose while maintaining full functionality in GPU-enabled environments.

## Architecture

### Environment Detection System

```python
class EnvironmentDetector:
    @staticmethod
    def is_docker() -> bool:
        """Detect if running in Docker container"""
        return os.path.exists('/.dockerenv') or os.getenv('DOCKER_CONTAINER') == 'true'
    
    @staticmethod
    def has_gpu_support() -> bool:
        """Detect if GPU support is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    @staticmethod
    def get_environment_config() -> Dict[str, Any]:
        """Get environment-specific configuration"""
```

### Configuration Management

The system will use a layered configuration approach:
1. **Base Configuration**: Default settings for all environments
2. **Environment Overrides**: Docker Compose specific settings
3. **Runtime Detection**: Automatic GPU availability detection and config selection

### Authentication System Improvements

```python
class AuthenticationManager:
    def __init__(self):
        self.use_persistent_db = self._should_use_database()
        self.fallback_to_memory = True
        self.initialize_default_users()
    
    def _should_use_database(self) -> bool:
        """Determine if database should be used based on availability"""
        # Try to connect to database, fall back to memory if unavailable
```

## Components and Interfaces

### 1. Environment Configuration Module

**Location**: `automl_framework/core/environment.py`

**Responsibilities**:
- Detect deployment environment (Docker, local, etc.)
- Detect GPU availability
- Load environment-specific configurations
- Provide fallback mechanisms

**Interface**:
```python
class EnvironmentConfig:
    def get_auth_config(self) -> AuthConfig
    def get_resource_config(self) -> ResourceConfig
    def is_gpu_available(self) -> bool
    def is_docker_environment(self) -> bool
    def get_database_config(self) -> DatabaseConfig
```

### 2. Enhanced Authentication System

**Location**: `automl_framework/api/auth.py` (modifications)

**Improvements**:
- Database connection health checks
- Graceful fallback to in-memory storage when database unavailable
- Better error handling for authentication failures
- Automatic initialization of demo users

### 3. Resource Configuration Manager

**Location**: `automl_framework/core/config.py` (modifications)

**Responsibilities**:
- Detect GPU availability at runtime
- Provide CPU-only configurations when GPU unavailable
- Log warnings for unsupported features
- Gracefully handle PyTorch/CUDA initialization failures

## Data Models

### Environment Configuration

```python
@dataclass
class EnvironmentConfig:
    name: str  # "docker", "local", "kubernetes"
    supports_gpu: bool
    gpu_available: bool
    supports_persistent_storage: bool
    default_auth_backend: str
    max_memory_mb: int
    max_cpu_cores: int
    database_available: bool
```

### Authentication Configuration

```python
@dataclass
class AuthConfig:
    backend_type: str  # "memory", "database", "external"
    jwt_secret: str
    token_expiry_minutes: int
    enable_signup: bool
    default_users: List[Dict[str, Any]]
```

## Error Handling

### Authentication Errors

1. **Database Connection Failures**:
   - Log warning about database unavailability
   - Fall back to in-memory user storage
   - Continue with demo users

2. **JWT Token Issues**:
   - Use environment-specific secret keys
   - Provide clear error messages for token validation failures

### Resource Configuration Errors

1. **GPU Initialization Failures**:
   - Detect GPU availability at runtime
   - Skip GPU initialization if unavailable
   - Log informational message about CPU-only mode
   - Handle PyTorch CUDA initialization errors gracefully

2. **Configuration Loading Errors**:
   - Use sensible defaults for missing configurations
   - Log warnings for unsupported settings
   - Continue with reduced functionality

## Testing Strategy

### Unit Tests

1. **Environment Detection Tests**:
   - Test Docker environment detection with .dockerenv file
   - Test GPU availability detection
   - Test configuration loading for each environment

2. **Authentication Tests**:
   - Test in-memory authentication when database unavailable
   - Test database authentication when available
   - Test fallback mechanisms and demo user initialization

3. **Configuration Tests**:
   - Test GPU setting filtering when GPU unavailable
   - Test CPU-only configuration generation
   - Test PyTorch/CUDA error handling

### Integration Tests

1. **Docker Compose Simulation**:
   - Test with Docker Compose environment
   - Test full application startup without GPU
   - Verify authentication flow works with in-memory storage

2. **API Endpoint Tests**:
   - Test login/logout functionality
   - Test project creation with CPU-only resources
   - Test health check endpoint

### Deployment Tests

1. **Docker Compose Deployment**:
   - Deploy with docker-compose up
   - Test authentication with demo credentials
   - Verify no GPU-related errors in logs
   - Test API functionality end-to-end

## Implementation Approach

### Phase 1: Environment Detection
- Create environment detection utilities
- Implement configuration loading system
- Add environment-specific overrides

### Phase 2: Authentication Fixes
- Modify authentication system for database availability detection
- Implement graceful database fallbacks
- Test authentication flow in Docker Compose mode

### Phase 3: Resource Configuration
- Detect GPU availability at runtime
- Implement CPU-only resource configurations
- Add appropriate logging and warnings for GPU unavailability

### Phase 4: Integration and Testing
- Test complete application startup in Docker Compose
- Verify authentication and API functionality
- Deploy and validate with docker-compose up

## Configuration Files

### Docker Compose Environment Variables

```yaml
# In docker-compose.yml
environment:
  - ENVIRONMENT=docker
  - DOCKER_CONTAINER=true
  - AUTH_BACKEND=auto
  - LOG_LEVEL=INFO
```

### Updated automl_config.yaml

```yaml
# Environment-specific overrides
docker:
  resources:
    max_gpu_per_experiment: 0  # Will be overridden if GPU detected
    default_gpu_memory_limit: "0GB"
    force_cpu_only: false  # Auto-detect
  
  auth:
    backend: "auto"  # Try database, fallback to memory
    enable_database_fallback: true
```

## Security Considerations

1. **JWT Secrets**: Use environment-specific secrets, not hardcoded values
2. **Demo Users**: Clearly mark demo credentials and consider disabling in production
3. **Rate Limiting**: Ensure rate limiting works with in-memory storage
4. **CORS Configuration**: Configure appropriately for Docker Compose local development