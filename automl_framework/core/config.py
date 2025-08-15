"""
Configuration management for AutoML Framework
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


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
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        return os.environ.get("AUTOML_CONFIG_PATH", "config/automl_config.yaml")
    
    def load_config(self) -> AutoMLConfig:
        """Load configuration from file or environment variables"""
        if self._config is not None:
            return self._config
        
        # Start with default configuration
        config = AutoMLConfig()
        
        # Load from file if exists
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                config = self._merge_config(config, file_config)
        
        # Override with environment variables
        config = self._load_from_env(config)
        
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
        # Database
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
        
        # Resources
        if os.getenv("MAX_CONCURRENT_EXPERIMENTS"):
            config.resources.max_concurrent_experiments = int(os.getenv("MAX_CONCURRENT_EXPERIMENTS"))
        if os.getenv("MAX_GPU_PER_EXPERIMENT"):
            config.resources.max_gpu_per_experiment = int(os.getenv("MAX_GPU_PER_EXPERIMENT"))
        
        # Storage paths
        if os.getenv("DATA_STORAGE_PATH"):
            config.data_storage_path = os.getenv("DATA_STORAGE_PATH")
        if os.getenv("MODEL_STORAGE_PATH"):
            config.model_storage_path = os.getenv("MODEL_STORAGE_PATH")
        
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
        return self.load_config()


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> AutoMLConfig:
    """Get global configuration instance"""
    return config_manager.get_config()