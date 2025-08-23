"""
Environment detection utilities for AutoML Framework.

This module provides utilities to detect the deployment environment
(Docker, local, etc.) and available resources (GPU, database, etc.)
to enable environment-specific configurations and graceful fallbacks.
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Configuration for different deployment environments."""
    name: str  # "docker", "local", "kubernetes"
    supports_gpu: bool
    gpu_available: bool
    supports_persistent_storage: bool
    default_auth_backend: str
    max_memory_mb: int
    max_cpu_cores: int
    database_available: bool


class EnvironmentDetector:
    """Utility class for detecting deployment environment and available resources."""
    
    @staticmethod
    def is_docker() -> bool:
        """
        Detect if running in Docker container.
        
        Returns:
            bool: True if running in Docker, False otherwise
        """
        detection_methods = []
        
        # Check for .dockerenv file (standard Docker indicator)
        if os.path.exists('/.dockerenv'):
            detection_methods.append("/.dockerenv file found")
            logger.debug("Docker environment detected: /.dockerenv file exists")
            return True
        
        # Check for explicit environment variable
        if os.getenv('DOCKER_CONTAINER', '').lower() == 'true':
            detection_methods.append("DOCKER_CONTAINER environment variable")
            logger.debug("Docker environment detected: DOCKER_CONTAINER=true")
            return True
        
        # Check for Docker-specific cgroup entries
        try:
            with open('/proc/1/cgroup', 'r') as f:
                content = f.read()
                if 'docker' in content or 'containerd' in content:
                    detection_methods.append("Docker cgroup entries found")
                    logger.debug("Docker environment detected: Docker cgroup entries found")
                    return True
        except (FileNotFoundError, PermissionError) as e:
            logger.debug(f"Could not read /proc/1/cgroup for Docker detection: {e}")
        
        logger.debug("Docker environment not detected - running in local/native environment")
        return False
    
    @staticmethod
    def has_gpu_support() -> bool:
        """
        Detect if GPU support is available.
        
        Returns:
            bool: True if GPU is available, False otherwise
        """
        logger.debug("Starting GPU availability detection...")
        
        try:
            import torch
            logger.debug("PyTorch imported successfully")
            
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_count = torch.cuda.device_count()
                logger.info(f"GPU support detected: {gpu_count} GPU(s) available")
                
                # Log detailed GPU information
                for i in range(gpu_count):
                    try:
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_props = torch.cuda.get_device_properties(i)
                        gpu_memory = gpu_props.total_memory / (1024**3)  # Convert to GB
                        logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB, Compute {gpu_props.major}.{gpu_props.minor})")
                    except Exception as gpu_error:
                        logger.warning(f"Could not get details for GPU {i}: {gpu_error}")
                
                # Test GPU initialization
                try:
                    test_tensor = torch.cuda.FloatTensor([1.0])
                    logger.debug("GPU initialization test successful")
                    del test_tensor
                except Exception as init_error:
                    logger.warning(f"GPU initialization test failed: {init_error}")
                    
            else:
                logger.info("GPU support not available: CUDA not available")
                logger.debug("CUDA availability check returned False")
                
            return gpu_available
            
        except ImportError as e:
            logger.info(f"GPU support not available: PyTorch not installed ({e})")
            return False
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {e}", exc_info=True)
            return False
    
    @staticmethod
    def is_kubernetes() -> bool:
        """
        Detect if running in Kubernetes environment.
        
        Returns:
            bool: True if running in Kubernetes, False otherwise
        """
        logger.debug("Checking for Kubernetes environment...")
        
        # Check for Kubernetes service account
        if os.path.exists('/var/run/secrets/kubernetes.io/serviceaccount'):
            logger.debug("Kubernetes environment detected: service account directory found")
            return True
        
        # Check for Kubernetes environment variables
        k8s_host = os.getenv('KUBERNETES_SERVICE_HOST')
        if k8s_host:
            logger.debug(f"Kubernetes environment detected: KUBERNETES_SERVICE_HOST={k8s_host}")
            return True
        
        # Check for other Kubernetes indicators
        k8s_port = os.getenv('KUBERNETES_SERVICE_PORT')
        if k8s_port:
            logger.debug(f"Kubernetes environment detected: KUBERNETES_SERVICE_PORT={k8s_port}")
            return True
        
        logger.debug("Kubernetes environment not detected")
        return False
    
    @staticmethod
    def get_environment_name() -> str:
        """
        Get the current environment name.
        
        Returns:
            str: Environment name ("docker", "kubernetes", "local")
        """
        logger.debug("Determining environment type...")
        
        if EnvironmentDetector.is_kubernetes():
            logger.info("Environment type determined: Kubernetes")
            return "kubernetes"
        elif EnvironmentDetector.is_docker():
            logger.info("Environment type determined: Docker")
            return "docker"
        else:
            logger.info("Environment type determined: Local/Native")
            return "local"
    
    @staticmethod
    def test_database_connectivity() -> bool:
        """
        Test if database is available and accessible.
        
        Returns:
            bool: True if database is accessible, False otherwise
        """
        logger.debug("Starting database connectivity tests...")
        
        try:
            # Get database URLs from environment or config
            postgres_url = os.getenv('DATABASE_URL', 'postgresql://localhost:5432/automl')
            mongodb_url = os.getenv('MONGODB_URL', 'mongodb://localhost:27017/automl')
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            
            logger.debug(f"Testing database connections:")
            logger.debug(f"  PostgreSQL: {postgres_url.split('@')[-1] if '@' in postgres_url else postgres_url}")
            logger.debug(f"  MongoDB: {mongodb_url.split('@')[-1] if '@' in mongodb_url else mongodb_url}")
            logger.debug(f"  Redis: {redis_url.split('@')[-1] if '@' in redis_url else redis_url}")
            
            databases_tested = 0
            databases_available = 0
            
            # Test PostgreSQL connection
            try:
                import psycopg2
                databases_tested += 1
                logger.debug("Testing PostgreSQL connection...")
                conn = psycopg2.connect(postgres_url)
                conn.close()
                databases_available += 1
                logger.info("PostgreSQL database connection successful")
                return True
            except ImportError:
                logger.debug("PostgreSQL driver (psycopg2) not available")
            except Exception as e:
                logger.warning(f"PostgreSQL connection failed: {e}")
            
            # Test MongoDB connection
            try:
                import pymongo
                databases_tested += 1
                logger.debug("Testing MongoDB connection...")
                client = pymongo.MongoClient(mongodb_url, serverSelectionTimeoutMS=2000)
                client.server_info()  # Force connection
                client.close()
                databases_available += 1
                logger.info("MongoDB database connection successful")
                return True
            except ImportError:
                logger.debug("MongoDB driver (pymongo) not available")
            except Exception as e:
                logger.warning(f"MongoDB connection failed: {e}")
            
            # Test Redis connection
            try:
                import redis
                databases_tested += 1
                logger.debug("Testing Redis connection...")
                r = redis.from_url(redis_url, socket_connect_timeout=2)
                r.ping()
                databases_available += 1
                logger.info("Redis database connection successful")
                return True
            except ImportError:
                logger.debug("Redis driver (redis) not available")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
            
            logger.info(f"Database connectivity test complete: {databases_available}/{databases_tested} databases available")
            return False
            
        except Exception as e:
            logger.error(f"Database connectivity test failed with unexpected error: {e}", exc_info=True)
            return False
    
    @staticmethod
    def get_environment_config() -> EnvironmentConfig:
        """
        Get environment-specific configuration.
        
        Returns:
            EnvironmentConfig: Configuration for the detected environment
        """
        logger.info("=== Environment Configuration Detection ===")
        
        env_name = EnvironmentDetector.get_environment_name()
        gpu_available = EnvironmentDetector.has_gpu_support()
        database_available = EnvironmentDetector.test_database_connectivity()
        
        # Base configuration
        config = EnvironmentConfig(
            name=env_name,
            supports_gpu=True,  # Environment supports GPU (may not be available)
            gpu_available=gpu_available,
            supports_persistent_storage=True,
            default_auth_backend="auto",
            max_memory_mb=2048,
            max_cpu_cores=2,
            database_available=database_available
        )
        
        logger.info(f"Base configuration created for {env_name} environment")
        
        # Environment-specific overrides
        if env_name == "docker":
            logger.info("Applying Docker-specific configuration overrides...")
            config.default_auth_backend = "memory" if not database_available else "database"
            config.max_memory_mb = 1024  # Conservative for Docker
            config.max_cpu_cores = 1
            logger.info(f"  Auth backend: {config.default_auth_backend} (database_available={database_available})")
            logger.info(f"  Memory limit: {config.max_memory_mb}MB")
            logger.info(f"  CPU cores: {config.max_cpu_cores}")
            
        elif env_name == "kubernetes":
            logger.info("Applying Kubernetes-specific configuration overrides...")
            config.supports_persistent_storage = True
            config.max_memory_mb = 4096
            config.max_cpu_cores = 4
            logger.info(f"  Memory limit: {config.max_memory_mb}MB")
            logger.info(f"  CPU cores: {config.max_cpu_cores}")
            logger.info(f"  Persistent storage: {config.supports_persistent_storage}")
            
        elif env_name == "local":
            logger.info("Applying local development configuration overrides...")
            config.supports_gpu = True
            config.max_memory_mb = 8192
            config.max_cpu_cores = 8
            logger.info(f"  Memory limit: {config.max_memory_mb}MB")
            logger.info(f"  CPU cores: {config.max_cpu_cores}")
            logger.info(f"  GPU support: {config.supports_gpu}")
        
        # Log final configuration summary
        logger.info("=== Final Environment Configuration ===")
        logger.info(f"Environment: {env_name}")
        logger.info(f"GPU Available: {gpu_available}")
        logger.info(f"Database Available: {database_available}")
        logger.info(f"Auth Backend: {config.default_auth_backend}")
        logger.info(f"Resource Limits: {config.max_memory_mb}MB RAM, {config.max_cpu_cores} CPU cores")
        logger.info(f"Persistent Storage: {config.supports_persistent_storage}")
        logger.info("==========================================")
        
        return config


class EnvironmentManager:
    """Manager for environment-specific configurations and behaviors."""
    
    def __init__(self):
        self._config: Optional[EnvironmentConfig] = None
        self._detector = EnvironmentDetector()
    
    @property
    def config(self) -> EnvironmentConfig:
        """Get the current environment configuration."""
        if self._config is None:
            self._config = self._detector.get_environment_config()
        return self._config
    
    def refresh_config(self) -> EnvironmentConfig:
        """Refresh the environment configuration."""
        self._config = self._detector.get_environment_config()
        return self._config
    
    def is_gpu_enabled(self) -> bool:
        """Check if GPU is enabled for the current environment."""
        return self.config.gpu_available
    
    def should_use_database(self) -> bool:
        """Check if database should be used in the current environment."""
        return self.config.database_available
    
    def get_auth_backend(self) -> str:
        """Get the appropriate authentication backend for the current environment."""
        if self.config.default_auth_backend == "auto":
            return "database" if self.config.database_available else "memory"
        return self.config.default_auth_backend
    
    def get_resource_limits(self) -> Dict[str, Any]:
        """Get resource limits for the current environment."""
        return {
            "max_memory_mb": self.config.max_memory_mb,
            "max_cpu_cores": self.config.max_cpu_cores,
            "gpu_available": self.config.gpu_available,
            "max_gpu_per_experiment": 1 if self.config.gpu_available else 0
        }


# Global environment manager instance
_environment_manager: Optional[EnvironmentManager] = None


def get_environment_manager() -> EnvironmentManager:
    """
    Get the global environment manager instance.
    
    Returns:
        EnvironmentManager: The global environment manager
    """
    global _environment_manager
    if _environment_manager is None:
        _environment_manager = EnvironmentManager()
    return _environment_manager


def initialize_environment() -> EnvironmentConfig:
    """
    Initialize environment detection and return configuration.
    
    Returns:
        EnvironmentConfig: The detected environment configuration
    """
    logger.info("Starting environment initialization...")
    
    manager = get_environment_manager()
    config = manager.refresh_config()
    
    # Log any fallback configurations being used
    fallbacks_used = []
    
    if not config.gpu_available and config.supports_gpu:
        fallbacks_used.append("GPU support disabled (hardware not available)")
    
    if not config.database_available:
        fallbacks_used.append("Database services unavailable (using in-memory storage)")
    
    if config.default_auth_backend == "memory":
        fallbacks_used.append("In-memory authentication backend selected")
    
    if config.name == "docker" and config.max_memory_mb < 2048:
        fallbacks_used.append(f"Reduced memory limit for Docker environment ({config.max_memory_mb}MB)")
    
    if fallbacks_used:
        logger.warning("=== Fallback Configurations Active ===")
        for fallback in fallbacks_used:
            logger.warning(f"  - {fallback}")
        logger.warning("Application will run with reduced functionality")
        logger.warning("======================================")
    
    logger.info("=== Environment Initialization Summary ===")
    logger.info(f"Environment: {config.name}")
    logger.info(f"GPU Support: {config.gpu_available}")
    logger.info(f"Database Available: {config.database_available}")
    logger.info(f"Auth Backend: {manager.get_auth_backend()}")
    logger.info(f"Resource Limits: {config.max_memory_mb}MB RAM, {config.max_cpu_cores} cores")
    logger.info(f"Fallbacks Active: {len(fallbacks_used)}")
    logger.info("=========================================")
    
    return config