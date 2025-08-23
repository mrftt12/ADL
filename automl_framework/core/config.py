"""
Configuration management for AutoML Framework
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    postgresql_url: str = "postgresql://localhost:5432/automl"
    mongodb_url: str = "mongodb://localhost:27017/automl"
    redis_url: str = "redis://localhost:6379/0"


@dataclass
class ResourceConfig:
    """Resource management configuration"""
    max_concurrent_experiments: int = 5
    max_gpu_per_experiment: int = 4
    default_gpu_memory_limit: str = "8GB"
    job_timeout_hours: int = 24
    checkpoint_interval_minutes: int = 30


@dataclass
class TrainingConfig:
    """Default training configuration"""
    default_epochs: int = 100
    early_stopping_patience: int = 10
    default_batch_size: int = 32
    default_learning_rate: float = 0.001
    mixed_precision: bool = True


@dataclass
class NASConfig:
    """Neural Architecture Search configuration"""
    search_time_limit_hours: int = 12
    max_architectures_to_evaluate: int = 50
    population_size: int = 20
    mutation_rate: float = 0.1


@dataclass
class HPOConfig:
    """Hyperparameter Optimization configuration"""
    max_trials: int = 100
    optimization_time_limit_hours: int = 6
    n_startup_trials: int = 10
    n_ei_candidates: int = 24


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    max_upload_size_mb: int = 1000
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size_mb: int = 100
    backup_count: int = 5


@dataclass
class AutoMLConfig:
    """Main configuration class for AutoML Framework"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    nas: NASConfig = field(default_factory=NASConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Storage paths
    data_storage_path: str = "./data"
    model_storage_path: str = "./models"
    checkpoint_storage_path: str = "./checkpoints"
    log_storage_path: str = "./logs"


class ConfigManager:
    """Configuration manager for loading and managing settings"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._config: Optional[AutoMLConfig] = None
        self._environment_manager = None
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        return os.environ.get("AUTOML_CONFIG_PATH", "config/automl_config.yaml")
    
    def load_config(self) -> AutoMLConfig:
        """Load configuration from file or environment variables"""
        if self._config is not None:
            return self._config
        
        # Initialize environment detection
        self._initialize_environment_manager()
        
        # Start with default configuration
        config = AutoMLConfig()
        
        # Load from file if exists
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                config = self._merge_config(config, file_config)
        
        # Apply environment-specific overrides
        config = self._apply_environment_overrides(config)
        
        # Override with environment variables
        config = self._load_from_env(config)
        
        # Apply GPU availability-based resource configuration
        config = self._apply_gpu_based_config(config)
        
        # Validate and apply fallbacks
        config = self._validate_and_apply_fallbacks(config)
        
        # Create necessary directories
        self._create_directories(config)
        
        self._config = config
        return config
    
    def _merge_config(self, base_config: AutoMLConfig, file_config: Dict[str, Any]) -> AutoMLConfig:
        """Merge file configuration with base configuration"""
        # This is a simplified merge - in production, you'd want more sophisticated merging
        for section, values in file_config.items():
            if hasattr(base_config, section) and isinstance(values, dict):
                section_obj = getattr(base_config, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
        return base_config
    
    def _load_from_env(self, config: AutoMLConfig) -> AutoMLConfig:
        """Load configuration from environment variables"""
        logger.debug("Loading configuration from environment variables")
        
        # Database
        if os.getenv("DATABASE_URL"):  # Common PostgreSQL env var
            config.database.postgresql_url = os.getenv("DATABASE_URL")
        if os.getenv("POSTGRESQL_URL"):
            config.database.postgresql_url = os.getenv("POSTGRESQL_URL")
        if os.getenv("MONGODB_URL"):
            config.database.mongodb_url = os.getenv("MONGODB_URL")
        if os.getenv("REDIS_URL"):
            config.database.redis_url = os.getenv("REDIS_URL")
        
        # API
        if os.getenv("API_HOST"):
            config.api.host = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            config.api.port = int(os.getenv("API_PORT"))
        if os.getenv("API_WORKERS"):
            config.api.workers = int(os.getenv("API_WORKERS"))
        
        # Resources
        if os.getenv("MAX_CONCURRENT_EXPERIMENTS"):
            config.resources.max_concurrent_experiments = int(os.getenv("MAX_CONCURRENT_EXPERIMENTS"))
        if os.getenv("MAX_GPU_PER_EXPERIMENT"):
            config.resources.max_gpu_per_experiment = int(os.getenv("MAX_GPU_PER_EXPERIMENT"))
        if os.getenv("DEFAULT_GPU_MEMORY_LIMIT"):
            config.resources.default_gpu_memory_limit = os.getenv("DEFAULT_GPU_MEMORY_LIMIT")
        if os.getenv("JOB_TIMEOUT_HOURS"):
            config.resources.job_timeout_hours = int(os.getenv("JOB_TIMEOUT_HOURS"))
        
        # Training
        if os.getenv("DEFAULT_BATCH_SIZE"):
            config.training.default_batch_size = int(os.getenv("DEFAULT_BATCH_SIZE"))
        if os.getenv("DEFAULT_EPOCHS"):
            config.training.default_epochs = int(os.getenv("DEFAULT_EPOCHS"))
        if os.getenv("MIXED_PRECISION"):
            config.training.mixed_precision = os.getenv("MIXED_PRECISION").lower() == "true"
        
        # Logging
        if os.getenv("LOG_LEVEL"):
            config.logging.level = os.getenv("LOG_LEVEL")
        if os.getenv("LOG_FILE_PATH"):
            config.logging.file_path = os.getenv("LOG_FILE_PATH")
        
        # Storage paths
        if os.getenv("DATA_STORAGE_PATH"):
            config.data_storage_path = os.getenv("DATA_STORAGE_PATH")
        if os.getenv("MODEL_STORAGE_PATH"):
            config.model_storage_path = os.getenv("MODEL_STORAGE_PATH")
        if os.getenv("CHECKPOINT_STORAGE_PATH"):
            config.checkpoint_storage_path = os.getenv("CHECKPOINT_STORAGE_PATH")
        if os.getenv("LOG_STORAGE_PATH"):
            config.log_storage_path = os.getenv("LOG_STORAGE_PATH")
        
        # Docker-specific environment variables
        if os.getenv("DOCKER_CONTAINER") == "true":
            logger.info("Docker container detected via environment variable")
            # Force certain settings for Docker
            if os.getenv("FORCE_CPU_ONLY") == "true":
                config.resources.max_gpu_per_experiment = 0
                config.resources.default_gpu_memory_limit = "0GB"
                config.training.mixed_precision = False
                logger.info("Forced CPU-only mode via environment variable")
        
        return config
    
    def _initialize_environment_manager(self) -> None:
        """Initialize environment manager for environment detection"""
        try:
            from .environment import get_environment_manager
            self._environment_manager = get_environment_manager()
            logger.info("Environment manager initialized successfully")
        except ImportError as e:
            logger.warning(f"Could not import environment manager: {e}")
            self._environment_manager = None
    
    def _apply_environment_overrides(self, config: AutoMLConfig) -> AutoMLConfig:
        """Apply environment-specific configuration overrides"""
        if self._environment_manager is None:
            logger.warning("Environment manager not available, skipping environment overrides")
            return config
        
        env_config = self._environment_manager.config
        logger.info(f"Applying configuration overrides for environment: {env_config.name}")
        
        # Docker-specific overrides
        if env_config.name == "docker":
            logger.info("Applying Docker-specific configuration overrides")
            
            # Resource limits for Docker
            config.resources.max_concurrent_experiments = min(config.resources.max_concurrent_experiments, 2)
            config.resources.job_timeout_hours = min(config.resources.job_timeout_hours, 12)
            
            # API configuration for Docker
            config.api.workers = min(config.api.workers, 2)
            config.api.cors_origins = ["*"]  # Allow all origins for local development
            
            # Training configuration adjustments
            config.training.default_batch_size = min(config.training.default_batch_size, 16)
            config.training.mixed_precision = False  # Disable mixed precision in Docker by default
            
            # NAS and HPO limits for resource-constrained environments
            config.nas.search_time_limit_hours = min(config.nas.search_time_limit_hours, 6)
            config.nas.max_architectures_to_evaluate = min(config.nas.max_architectures_to_evaluate, 20)
            config.hpo.max_trials = min(config.hpo.max_trials, 50)
            config.hpo.optimization_time_limit_hours = min(config.hpo.optimization_time_limit_hours, 3)
            
            # Database configuration for Docker
            if not env_config.database_available:
                logger.warning("Database not available in Docker environment, using fallback configurations")
                # Keep original URLs but they will be handled by fallback mechanisms
        
        # Kubernetes-specific overrides
        elif env_config.name == "kubernetes":
            logger.info("Applying Kubernetes-specific configuration overrides")
            config.resources.max_concurrent_experiments = min(config.resources.max_concurrent_experiments, 10)
            config.api.workers = min(config.api.workers, 8)
        
        # Local development overrides
        elif env_config.name == "local":
            logger.info("Using local development configuration")
            # Keep default settings for local development
        
        return config
    
    def _apply_gpu_based_config(self, config: AutoMLConfig) -> AutoMLConfig:
        """Apply GPU availability-based resource configuration"""
        if self._environment_manager is None:
            return config
        
        gpu_available = self._environment_manager.is_gpu_enabled()
        
        if not gpu_available:
            logger.warning("GPU not available, applying CPU-only configuration")
            
            # Force CPU-only settings
            config.resources.max_gpu_per_experiment = 0
            config.resources.default_gpu_memory_limit = "0GB"
            
            # Disable mixed precision training (requires GPU)
            config.training.mixed_precision = False
            
            # Reduce resource-intensive settings
            config.training.default_batch_size = min(config.training.default_batch_size, 16)
            config.nas.population_size = min(config.nas.population_size, 10)
            config.hpo.n_ei_candidates = min(config.hpo.n_ei_candidates, 12)
            
            logger.info("Configuration adjusted for CPU-only mode")
        else:
            logger.info(f"GPU available, using GPU-enabled configuration with {config.resources.max_gpu_per_experiment} max GPUs per experiment")
        
        return config
    
    def _validate_and_apply_fallbacks(self, config: AutoMLConfig) -> AutoMLConfig:
        """Validate configuration and apply fallbacks for invalid settings"""
        logger.info("Validating configuration and applying fallbacks")
        
        # Validate GPU settings
        config = self._validate_gpu_settings(config)
        
        # Validate resource limits
        config = self._validate_resource_limits(config)
        
        # Validate training settings
        config = self._validate_training_settings(config)
        
        # Validate storage paths
        config = self._validate_storage_paths(config)
        
        # Validate database settings
        config = self._validate_database_settings(config)
        
        logger.info("Configuration validation completed")
        return config
    
    def _validate_gpu_settings(self, config: AutoMLConfig) -> AutoMLConfig:
        """Validate GPU settings and apply CPU-only fallbacks when necessary"""
        gpu_available = self.is_gpu_available()
        
        if not gpu_available and config.resources.max_gpu_per_experiment > 0:
            logger.warning(f"GPU not available but max_gpu_per_experiment is {config.resources.max_gpu_per_experiment}, setting to 0")
            config.resources.max_gpu_per_experiment = 0
            config.resources.default_gpu_memory_limit = "0GB"
        
        if not gpu_available and config.training.mixed_precision:
            logger.warning("GPU not available but mixed_precision is enabled, disabling mixed precision")
            config.training.mixed_precision = False
        
        # Validate GPU memory limit format
        if config.resources.default_gpu_memory_limit and not config.resources.default_gpu_memory_limit.endswith(('GB', 'MB', '0GB')):
            logger.warning(f"Invalid GPU memory limit format: {config.resources.default_gpu_memory_limit}, using default 8GB")
            config.resources.default_gpu_memory_limit = "8GB" if gpu_available else "0GB"
        
        return config
    
    def _validate_resource_limits(self, config: AutoMLConfig) -> AutoMLConfig:
        """Validate resource limits and apply reasonable fallbacks"""
        # Validate concurrent experiments
        if config.resources.max_concurrent_experiments <= 0:
            logger.warning("max_concurrent_experiments must be positive, setting to 1")
            config.resources.max_concurrent_experiments = 1
        elif config.resources.max_concurrent_experiments > 20:
            logger.warning("max_concurrent_experiments is very high, limiting to 20")
            config.resources.max_concurrent_experiments = 20
        
        # Validate job timeout
        if config.resources.job_timeout_hours <= 0:
            logger.warning("job_timeout_hours must be positive, setting to 24")
            config.resources.job_timeout_hours = 24
        elif config.resources.job_timeout_hours > 168:  # 1 week
            logger.warning("job_timeout_hours is very high (>1 week), limiting to 168 hours")
            config.resources.job_timeout_hours = 168
        
        # Validate checkpoint interval
        if config.resources.checkpoint_interval_minutes <= 0:
            logger.warning("checkpoint_interval_minutes must be positive, setting to 30")
            config.resources.checkpoint_interval_minutes = 30
        
        return config
    
    def _validate_training_settings(self, config: AutoMLConfig) -> AutoMLConfig:
        """Validate training settings and apply reasonable fallbacks"""
        # Validate epochs
        if config.training.default_epochs <= 0:
            logger.warning("default_epochs must be positive, setting to 100")
            config.training.default_epochs = 100
        
        # Validate batch size
        if config.training.default_batch_size <= 0:
            logger.warning("default_batch_size must be positive, setting to 32")
            config.training.default_batch_size = 32
        elif config.training.default_batch_size > 1024:
            logger.warning("default_batch_size is very high, limiting to 1024")
            config.training.default_batch_size = 1024
        
        # Validate learning rate
        if config.training.default_learning_rate <= 0:
            logger.warning("default_learning_rate must be positive, setting to 0.001")
            config.training.default_learning_rate = 0.001
        elif config.training.default_learning_rate > 1.0:
            logger.warning("default_learning_rate is very high, limiting to 1.0")
            config.training.default_learning_rate = 1.0
        
        # Validate early stopping patience
        if config.training.early_stopping_patience <= 0:
            logger.warning("early_stopping_patience must be positive, setting to 10")
            config.training.early_stopping_patience = 10
        
        return config
    
    def _validate_storage_paths(self, config: AutoMLConfig) -> AutoMLConfig:
        """Validate storage paths and ensure they are accessible"""
        paths_to_validate = [
            ("data_storage_path", config.data_storage_path),
            ("model_storage_path", config.model_storage_path),
            ("checkpoint_storage_path", config.checkpoint_storage_path),
            ("log_storage_path", config.log_storage_path)
        ]
        
        for path_name, path_value in paths_to_validate:
            if not path_value:
                fallback_path = f"./{path_name.replace('_storage_path', '')}"
                logger.warning(f"{path_name} is empty, using fallback: {fallback_path}")
                setattr(config, path_name, fallback_path)
            else:
                # Ensure path is not absolute in Docker to avoid permission issues
                if self._environment_manager and self._environment_manager.config.name == "docker":
                    if os.path.isabs(path_value) and not path_value.startswith('/app'):
                        relative_path = f"./{os.path.basename(path_value)}"
                        logger.warning(f"Converting absolute path {path_value} to relative path {relative_path} for Docker")
                        setattr(config, path_name, relative_path)
        
        return config
    
    def _validate_database_settings(self, config: AutoMLConfig) -> AutoMLConfig:
        """Validate database settings and apply fallbacks"""
        if not self.should_use_database():
            logger.info("Database not available, configuration will use fallback mechanisms")
            # Don't modify URLs here, let the services handle fallbacks
            return config
        
        # Validate database URLs format
        if config.database.postgresql_url and not config.database.postgresql_url.startswith(('postgresql://', 'postgres://')):
            logger.warning(f"Invalid PostgreSQL URL format: {config.database.postgresql_url}")
        
        if config.database.mongodb_url and not config.database.mongodb_url.startswith('mongodb://'):
            logger.warning(f"Invalid MongoDB URL format: {config.database.mongodb_url}")
        
        if config.database.redis_url and not config.database.redis_url.startswith('redis://'):
            logger.warning(f"Invalid Redis URL format: {config.database.redis_url}")
        
        return config
    
    def _create_directories(self, config: AutoMLConfig) -> None:
        """Create necessary directories if they don't exist"""
        directories = [
            config.data_storage_path,
            config.model_storage_path,
            config.checkpoint_storage_path,
            config.log_storage_path
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_config(self) -> AutoMLConfig:
        """Get current configuration"""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def reload_config(self) -> AutoMLConfig:
        """Reload configuration from file"""
        self._config = None
        self._environment_manager = None
        return self.load_config()
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about the current environment"""
        if self._environment_manager is None:
            return {"environment": "unknown", "gpu_available": False, "database_available": False}
        
        env_config = self._environment_manager.config
        return {
            "environment": env_config.name,
            "gpu_available": env_config.gpu_available,
            "database_available": env_config.database_available,
            "auth_backend": self._environment_manager.get_auth_backend(),
            "resource_limits": self._environment_manager.get_resource_limits()
        }
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available in the current environment"""
        if self._environment_manager is None:
            return False
        return self._environment_manager.is_gpu_enabled()
    
    def should_use_database(self) -> bool:
        """Check if database should be used in the current environment"""
        if self._environment_manager is None:
            return True  # Default to trying database
        return self._environment_manager.should_use_database()
    
    def get_configuration_warnings(self) -> list[str]:
        """Get list of configuration warnings and fallbacks applied"""
        warnings = []
        
        if self._config is None:
            return warnings
        
        # Check for GPU-related warnings
        if not self.is_gpu_available():
            if self._config.resources.max_gpu_per_experiment > 0:
                warnings.append("GPU not available but GPU resources configured - using CPU-only mode")
            if self._config.training.mixed_precision:
                warnings.append("Mixed precision training disabled due to GPU unavailability")
        
        # Check for database warnings
        if not self.should_use_database():
            warnings.append("Database not available - using in-memory fallbacks")
        
        # Check for resource limit warnings
        env_info = self.get_environment_info()
        if env_info.get("environment") == "docker":
            warnings.append("Running in Docker environment - resource limits applied")
        
        return warnings
    
    def log_configuration_summary(self) -> None:
        """Log a summary of the current configuration"""
        if self._config is None:
            logger.warning("No configuration loaded")
            return
        
        env_info = self.get_environment_info()
        
        logger.info("=== Configuration Summary ===")
        logger.info(f"Environment: {env_info.get('environment', 'unknown')}")
        logger.info(f"GPU Available: {env_info.get('gpu_available', False)}")
        logger.info(f"Database Available: {env_info.get('database_available', False)}")
        logger.info(f"Auth Backend: {env_info.get('auth_backend', 'unknown')}")
        logger.info(f"Max Concurrent Experiments: {self._config.resources.max_concurrent_experiments}")
        logger.info(f"Max GPU per Experiment: {self._config.resources.max_gpu_per_experiment}")
        logger.info(f"Default Batch Size: {self._config.training.default_batch_size}")
        logger.info(f"Mixed Precision: {self._config.training.mixed_precision}")
        
        # Log warnings
        warnings = self.get_configuration_warnings()
        if warnings:
            logger.warning("Configuration Warnings:")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        
        logger.info("=== End Configuration Summary ===")
    
    def validate_runtime_requirements(self) -> Dict[str, bool]:
        """Validate that runtime requirements are met"""
        results = {
            "gpu_requirements_met": True,
            "database_requirements_met": True,
            "storage_requirements_met": True,
            "resource_requirements_met": True
        }
        
        if self._config is None:
            return {key: False for key in results.keys()}
        
        # Check GPU requirements
        if self._config.resources.max_gpu_per_experiment > 0 and not self.is_gpu_available():
            results["gpu_requirements_met"] = False
            logger.error("GPU required but not available")
        
        # Check database requirements
        if not self.should_use_database():
            logger.warning("Database not available, using fallback mechanisms")
            # This is not a failure, just a fallback
        
        # Check storage requirements
        try:
            self._create_directories(self._config)
            results["storage_requirements_met"] = True
        except Exception as e:
            logger.error(f"Failed to create storage directories: {e}")
            results["storage_requirements_met"] = False
        
        # Check resource requirements
        if self._config.resources.max_concurrent_experiments <= 0:
            results["resource_requirements_met"] = False
            logger.error("Invalid resource configuration")
        
        return results


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> AutoMLConfig:
    """Get global configuration instance"""
    config = config_manager.get_config()
    # Log configuration summary on first load
    if not hasattr(get_config, '_logged'):
        config_manager.log_configuration_summary()
        get_config._logged = True
    return config


def validate_configuration() -> bool:
    """Validate that the current configuration meets runtime requirements"""
    results = config_manager.validate_runtime_requirements()
    all_valid = all(results.values())
    
    if not all_valid:
        logger.error("Configuration validation failed:")
        for requirement, met in results.items():
            if not met:
                logger.error(f"  - {requirement}: FAILED")
    else:
        logger.info("Configuration validation passed")
    
    return all_valid


def get_environment_info() -> Dict[str, Any]:
    """Get information about the current environment"""
    return config_manager.get_environment_info()


def is_gpu_available() -> bool:
    """Check if GPU is available in the current environment"""
    return config_manager.is_gpu_available()


def should_use_database() -> bool:
    """Check if database should be used in the current environment"""
    return config_manager.should_use_database()


def get_cpu_only_experiment_config() -> Dict[str, Any]:
    """
    Get CPU-only experiment configuration for environments without GPU.
    
    Returns:
        Dict[str, Any]: CPU-only experiment configuration
    """
    config = get_config()
    
    cpu_only_config = {
        "resources": {
            "cpu_cores": min(4, config.resources.max_concurrent_experiments),
            "memory_gb": 4.0,
            "gpu_count": 0,
            "gpu_memory_gb": 0.0,
            "storage_gb": 2.0,
            "estimated_duration_minutes": 120  # Longer for CPU-only
        },
        "training": {
            "batch_size": min(16, config.training.default_batch_size),
            "epochs": config.training.default_epochs,
            "learning_rate": config.training.default_learning_rate,
            "mixed_precision": False,  # Not supported on CPU
            "early_stopping_patience": config.training.early_stopping_patience
        },
        "optimization": {
            "max_trials": min(25, config.hpo.max_trials),  # Reduced for CPU
            "optimization_time_limit_hours": min(2, config.hpo.optimization_time_limit_hours),
            "n_startup_trials": min(5, config.hpo.n_startup_trials),
            "n_ei_candidates": min(12, config.hpo.n_ei_candidates)
        },
        "nas": {
            "search_time_limit_hours": min(3, config.nas.search_time_limit_hours),
            "max_architectures_to_evaluate": min(10, config.nas.max_architectures_to_evaluate),
            "population_size": min(8, config.nas.population_size)
        }
    }
    
    logger.info("Generated CPU-only experiment configuration")
    return cpu_only_config


def validate_experiment_config_for_environment(experiment_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and adjust experiment configuration for current environment.
    
    Args:
        experiment_config: Original experiment configuration
        
    Returns:
        Dict[str, Any]: Validated and adjusted configuration
    """
    if not is_gpu_available():
        logger.info("GPU not available, adjusting experiment configuration for CPU-only mode")
        
        # Force CPU-only resource requirements
        if "resources" in experiment_config:
            experiment_config["resources"]["gpu_count"] = 0
            experiment_config["resources"]["gpu_memory_gb"] = 0.0
            # Increase CPU resources to compensate
            experiment_config["resources"]["cpu_cores"] = min(
                experiment_config["resources"].get("cpu_cores", 2) * 2, 8
            )
            experiment_config["resources"]["memory_gb"] = min(
                experiment_config["resources"].get("memory_gb", 2.0) * 1.5, 8.0
            )
            # Increase estimated duration for CPU processing
            experiment_config["resources"]["estimated_duration_minutes"] = int(
                experiment_config["resources"].get("estimated_duration_minutes", 60) * 2
            )
        
        # Adjust training settings for CPU
        if "training" in experiment_config:
            experiment_config["training"]["mixed_precision"] = False
            experiment_config["training"]["batch_size"] = min(
                experiment_config["training"].get("batch_size", 32), 16
            )
        
        # Reduce optimization complexity for CPU
        if "optimization" in experiment_config:
            experiment_config["optimization"]["max_trials"] = min(
                experiment_config["optimization"].get("max_trials", 100), 25
            )
            experiment_config["optimization"]["optimization_time_limit_hours"] = min(
                experiment_config["optimization"].get("optimization_time_limit_hours", 6), 2
            )
        
        # Reduce NAS complexity for CPU
        if "nas" in experiment_config:
            experiment_config["nas"]["max_architectures_to_evaluate"] = min(
                experiment_config["nas"].get("max_architectures_to_evaluate", 50), 10
            )
            experiment_config["nas"]["search_time_limit_hours"] = min(
                experiment_config["nas"].get("search_time_limit_hours", 12), 3
            )
        
        logger.info("Experiment configuration adjusted for CPU-only mode")
    
    return experiment_config