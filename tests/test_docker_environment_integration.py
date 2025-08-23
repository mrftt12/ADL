"""
Integration tests for Docker environment detection and configuration.

Tests environment detection, GPU availability detection, and configuration
loading for Docker environments as specified in requirements 3.1 and 3.2.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, mock_open, MagicMock
from pathlib import Path

from automl_framework.core.environment import (
    EnvironmentDetector,
    EnvironmentManager,
    EnvironmentConfig,
    get_environment_manager,
    initialize_environment
)


class TestEnvironmentDetection:
    """Test suite for environment detection functionality."""
    
    def test_docker_detection_with_dockerenv_file(self, temp_directory):
        """Test Docker environment detection with .dockerenv file."""
        # Create a temporary .dockerenv file
        dockerenv_path = "/.dockerenv"
        
        with patch('os.path.exists') as mock_exists:
            # Mock .dockerenv file exists
            mock_exists.side_effect = lambda path: path == dockerenv_path
            
            result = EnvironmentDetector.is_docker()
            
            assert result is True
            mock_exists.assert_called_with(dockerenv_path)
    
    def test_docker_detection_with_environment_variable(self):
        """Test Docker environment detection with DOCKER_CONTAINER environment variable."""
        with patch.dict(os.environ, {'DOCKER_CONTAINER': 'true'}):
            with patch('os.path.exists', return_value=False):
                result = EnvironmentDetector.is_docker()
                
                assert result is True
    
    def test_docker_detection_with_cgroup_entries(self):
        """Test Docker environment detection with Docker cgroup entries."""
        cgroup_content = """
        12:perf_event:/docker/1234567890abcdef
        11:net_cls,net_prio:/docker/1234567890abcdef
        10:hugetlb:/docker/1234567890abcdef
        """
        
        with patch('os.path.exists', return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                with patch('builtins.open', mock_open(read_data=cgroup_content)):
                    result = EnvironmentDetector.is_docker()
                    
                    assert result is True
    
    def test_docker_detection_with_containerd_cgroup(self):
        """Test Docker environment detection with containerd cgroup entries."""
        cgroup_content = """
        12:perf_event:/system.slice/containerd.service
        11:net_cls,net_prio:/system.slice/containerd.service
        10:hugetlb:/system.slice/containerd.service
        """
        
        with patch('os.path.exists', return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                with patch('builtins.open', mock_open(read_data=cgroup_content)):
                    result = EnvironmentDetector.is_docker()
                    
                    assert result is True
    
    def test_docker_detection_false_when_not_docker(self):
        """Test Docker environment detection returns False when not in Docker."""
        with patch('os.path.exists', return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                with patch('builtins.open', side_effect=FileNotFoundError()):
                    result = EnvironmentDetector.is_docker()
                    
                    assert result is False
    
    def test_docker_detection_handles_cgroup_permission_error(self):
        """Test Docker detection handles permission errors gracefully."""
        with patch('os.path.exists', return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                with patch('builtins.open', side_effect=PermissionError()):
                    result = EnvironmentDetector.is_docker()
                    
                    assert result is False


class TestGPUDetection:
    """Test suite for GPU availability detection."""
    
    def test_gpu_detection_with_cuda_available(self):
        """Test GPU detection when CUDA is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.get_device_name.side_effect = ["GPU 0", "GPU 1"]
        
        # Mock device properties
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1024**3  # 8GB
        mock_props.major = 8
        mock_props.minor = 6
        mock_torch.cuda.get_device_properties.return_value = mock_props
        
        # Mock tensor creation
        mock_tensor = MagicMock()
        mock_torch.cuda.FloatTensor.return_value = mock_tensor
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            result = EnvironmentDetector.has_gpu_support()
            
            assert result is True
            mock_torch.cuda.is_available.assert_called_once()
            mock_torch.cuda.device_count.assert_called_once()
    
    def test_gpu_detection_with_cuda_unavailable(self):
        """Test GPU detection when CUDA is unavailable."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            result = EnvironmentDetector.has_gpu_support()
            
            assert result is False
            mock_torch.cuda.is_available.assert_called_once()
    
    def test_gpu_detection_without_pytorch(self):
        """Test GPU detection when PyTorch is not installed."""
        with patch.dict('sys.modules', {}, clear=True):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'torch'")):
                result = EnvironmentDetector.has_gpu_support()
                
                assert result is False
    
    def test_gpu_detection_with_initialization_error(self):
        """Test GPU detection handles initialization errors gracefully."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        
        # Mock device properties
        mock_props = MagicMock()
        mock_props.total_memory = 4 * 1024**3  # 4GB
        mock_props.major = 7
        mock_props.minor = 5
        mock_torch.cuda.get_device_properties.return_value = mock_props
        
        # Mock tensor creation failure
        mock_torch.cuda.FloatTensor.side_effect = RuntimeError("CUDA initialization failed")
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            result = EnvironmentDetector.has_gpu_support()
            
            # Should still return True even if initialization test fails
            assert result is True
    
    def test_gpu_detection_with_unexpected_error(self):
        """Test GPU detection handles unexpected errors gracefully."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = Exception("Unexpected error")
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            result = EnvironmentDetector.has_gpu_support()
            
            assert result is False


class TestConfigurationLoading:
    """Test suite for Docker environment configuration loading."""
    
    def test_docker_environment_config_creation(self):
        """Test configuration creation for Docker environment."""
        with patch.object(EnvironmentDetector, 'get_environment_name', return_value='docker'):
            with patch.object(EnvironmentDetector, 'has_gpu_support', return_value=False):
                with patch.object(EnvironmentDetector, 'test_database_connectivity', return_value=False):
                    config = EnvironmentDetector.get_environment_config()
                    
                    assert config.name == 'docker'
                    assert config.gpu_available is False
                    assert config.database_available is False
                    assert config.default_auth_backend == 'memory'
                    assert config.max_memory_mb == 1024  # Docker conservative limit
                    assert config.max_cpu_cores == 1
    
    def test_docker_environment_config_with_database(self):
        """Test Docker configuration when database is available."""
        with patch.object(EnvironmentDetector, 'get_environment_name', return_value='docker'):
            with patch.object(EnvironmentDetector, 'has_gpu_support', return_value=False):
                with patch.object(EnvironmentDetector, 'test_database_connectivity', return_value=True):
                    config = EnvironmentDetector.get_environment_config()
                    
                    assert config.name == 'docker'
                    assert config.database_available is True
                    assert config.default_auth_backend == 'database'
    
    def test_docker_environment_config_with_gpu(self):
        """Test Docker configuration when GPU is available."""
        with patch.object(EnvironmentDetector, 'get_environment_name', return_value='docker'):
            with patch.object(EnvironmentDetector, 'has_gpu_support', return_value=True):
                with patch.object(EnvironmentDetector, 'test_database_connectivity', return_value=False):
                    config = EnvironmentDetector.get_environment_config()
                    
                    assert config.name == 'docker'
                    assert config.gpu_available is True
                    assert config.supports_gpu is True
    
    def test_local_environment_config_creation(self):
        """Test configuration creation for local environment."""
        with patch.object(EnvironmentDetector, 'get_environment_name', return_value='local'):
            with patch.object(EnvironmentDetector, 'has_gpu_support', return_value=True):
                with patch.object(EnvironmentDetector, 'test_database_connectivity', return_value=True):
                    config = EnvironmentDetector.get_environment_config()
                    
                    assert config.name == 'local'
                    assert config.gpu_available is True
                    assert config.database_available is True
                    assert config.max_memory_mb == 8192  # Local development limit
                    assert config.max_cpu_cores == 8
    
    def test_kubernetes_environment_config_creation(self):
        """Test configuration creation for Kubernetes environment."""
        with patch.object(EnvironmentDetector, 'get_environment_name', return_value='kubernetes'):
            with patch.object(EnvironmentDetector, 'has_gpu_support', return_value=True):
                with patch.object(EnvironmentDetector, 'test_database_connectivity', return_value=True):
                    config = EnvironmentDetector.get_environment_config()
                    
                    assert config.name == 'kubernetes'
                    assert config.supports_persistent_storage is True
                    assert config.max_memory_mb == 4096
                    assert config.max_cpu_cores == 4


class TestEnvironmentManager:
    """Test suite for EnvironmentManager functionality."""
    
    def test_environment_manager_initialization(self):
        """Test EnvironmentManager initialization."""
        with patch.object(EnvironmentDetector, 'get_environment_config') as mock_get_config:
            mock_config = EnvironmentConfig(
                name='docker',
                supports_gpu=True,
                gpu_available=False,
                supports_persistent_storage=True,
                default_auth_backend='memory',
                max_memory_mb=1024,
                max_cpu_cores=1,
                database_available=False
            )
            mock_get_config.return_value = mock_config
            
            manager = EnvironmentManager()
            config = manager.config
            
            assert config.name == 'docker'
            assert config.gpu_available is False
            assert config.default_auth_backend == 'memory'
    
    def test_environment_manager_gpu_enabled_check(self):
        """Test EnvironmentManager GPU enabled check."""
        with patch.object(EnvironmentDetector, 'get_environment_config') as mock_get_config:
            mock_config = EnvironmentConfig(
                name='docker',
                supports_gpu=True,
                gpu_available=True,
                supports_persistent_storage=True,
                default_auth_backend='database',
                max_memory_mb=1024,
                max_cpu_cores=1,
                database_available=True
            )
            mock_get_config.return_value = mock_config
            
            manager = EnvironmentManager()
            
            assert manager.is_gpu_enabled() is True
    
    def test_environment_manager_database_usage_check(self):
        """Test EnvironmentManager database usage check."""
        with patch.object(EnvironmentDetector, 'get_environment_config') as mock_get_config:
            mock_config = EnvironmentConfig(
                name='docker',
                supports_gpu=True,
                gpu_available=False,
                supports_persistent_storage=True,
                default_auth_backend='memory',
                max_memory_mb=1024,
                max_cpu_cores=1,
                database_available=False
            )
            mock_get_config.return_value = mock_config
            
            manager = EnvironmentManager()
            
            assert manager.should_use_database() is False
    
    def test_environment_manager_auth_backend_selection(self):
        """Test EnvironmentManager authentication backend selection."""
        with patch.object(EnvironmentDetector, 'get_environment_config') as mock_get_config:
            # Test auto backend with database available
            mock_config = EnvironmentConfig(
                name='docker',
                supports_gpu=True,
                gpu_available=False,
                supports_persistent_storage=True,
                default_auth_backend='auto',
                max_memory_mb=1024,
                max_cpu_cores=1,
                database_available=True
            )
            mock_get_config.return_value = mock_config
            
            manager = EnvironmentManager()
            
            assert manager.get_auth_backend() == 'database'
            
            # Test auto backend with database unavailable
            mock_config.database_available = False
            manager._config = None  # Force refresh
            
            assert manager.get_auth_backend() == 'memory'
    
    def test_environment_manager_resource_limits(self):
        """Test EnvironmentManager resource limits."""
        with patch.object(EnvironmentDetector, 'get_environment_config') as mock_get_config:
            mock_config = EnvironmentConfig(
                name='docker',
                supports_gpu=True,
                gpu_available=True,
                supports_persistent_storage=True,
                default_auth_backend='database',
                max_memory_mb=2048,
                max_cpu_cores=2,
                database_available=True
            )
            mock_get_config.return_value = mock_config
            
            manager = EnvironmentManager()
            limits = manager.get_resource_limits()
            
            assert limits['max_memory_mb'] == 2048
            assert limits['max_cpu_cores'] == 2
            assert limits['gpu_available'] is True
            assert limits['max_gpu_per_experiment'] == 1
    
    def test_environment_manager_resource_limits_no_gpu(self):
        """Test EnvironmentManager resource limits without GPU."""
        with patch.object(EnvironmentDetector, 'get_environment_config') as mock_get_config:
            mock_config = EnvironmentConfig(
                name='docker',
                supports_gpu=True,
                gpu_available=False,
                supports_persistent_storage=True,
                default_auth_backend='memory',
                max_memory_mb=1024,
                max_cpu_cores=1,
                database_available=False
            )
            mock_get_config.return_value = mock_config
            
            manager = EnvironmentManager()
            limits = manager.get_resource_limits()
            
            assert limits['gpu_available'] is False
            assert limits['max_gpu_per_experiment'] == 0


class TestEnvironmentInitialization:
    """Test suite for environment initialization functionality."""
    
    def test_initialize_environment_docker_no_fallbacks(self):
        """Test environment initialization for Docker with no fallbacks needed."""
        with patch.object(EnvironmentDetector, 'get_environment_config') as mock_get_config:
            mock_config = EnvironmentConfig(
                name='docker',
                supports_gpu=True,
                gpu_available=True,
                supports_persistent_storage=True,
                default_auth_backend='database',
                max_memory_mb=2048,
                max_cpu_cores=2,
                database_available=True
            )
            mock_get_config.return_value = mock_config
            
            with patch('automl_framework.core.environment.get_environment_manager') as mock_get_manager:
                mock_manager = Mock()
                mock_manager.refresh_config.return_value = mock_config
                mock_manager.get_auth_backend.return_value = 'database'
                mock_get_manager.return_value = mock_manager
                
                config = initialize_environment()
                
                assert config.name == 'docker'
                assert config.gpu_available is True
                assert config.database_available is True
    
    def test_initialize_environment_docker_with_fallbacks(self):
        """Test environment initialization for Docker with fallbacks."""
        with patch.object(EnvironmentDetector, 'get_environment_config') as mock_get_config:
            mock_config = EnvironmentConfig(
                name='docker',
                supports_gpu=True,
                gpu_available=False,  # GPU fallback
                supports_persistent_storage=True,
                default_auth_backend='memory',  # Auth fallback
                max_memory_mb=1024,  # Memory fallback
                max_cpu_cores=1,
                database_available=False  # Database fallback
            )
            mock_get_config.return_value = mock_config
            
            with patch('automl_framework.core.environment.get_environment_manager') as mock_get_manager:
                mock_manager = Mock()
                mock_manager.refresh_config.return_value = mock_config
                mock_manager.get_auth_backend.return_value = 'memory'
                mock_get_manager.return_value = mock_manager
                
                config = initialize_environment()
                
                assert config.name == 'docker'
                assert config.gpu_available is False
                assert config.database_available is False
                assert config.default_auth_backend == 'memory'
                assert config.max_memory_mb == 1024


class TestKubernetesDetection:
    """Test suite for Kubernetes environment detection."""
    
    def test_kubernetes_detection_with_service_account(self):
        """Test Kubernetes detection with service account directory."""
        service_account_path = '/var/run/secrets/kubernetes.io/serviceaccount'
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.side_effect = lambda path: path == service_account_path
            
            result = EnvironmentDetector.is_kubernetes()
            
            assert result is True
            mock_exists.assert_called_with(service_account_path)
    
    def test_kubernetes_detection_with_service_host(self):
        """Test Kubernetes detection with KUBERNETES_SERVICE_HOST."""
        with patch.dict(os.environ, {'KUBERNETES_SERVICE_HOST': '10.96.0.1'}):
            with patch('os.path.exists', return_value=False):
                result = EnvironmentDetector.is_kubernetes()
                
                assert result is True
    
    def test_kubernetes_detection_with_service_port(self):
        """Test Kubernetes detection with KUBERNETES_SERVICE_PORT."""
        with patch.dict(os.environ, {'KUBERNETES_SERVICE_PORT': '443'}):
            with patch('os.path.exists', return_value=False):
                result = EnvironmentDetector.is_kubernetes()
                
                assert result is True
    
    def test_kubernetes_detection_false_when_not_kubernetes(self):
        """Test Kubernetes detection returns False when not in Kubernetes."""
        with patch('os.path.exists', return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                result = EnvironmentDetector.is_kubernetes()
                
                assert result is False


class TestEnvironmentNameDetection:
    """Test suite for environment name detection."""
    
    def test_get_environment_name_kubernetes(self):
        """Test environment name detection for Kubernetes."""
        with patch.object(EnvironmentDetector, 'is_kubernetes', return_value=True):
            with patch.object(EnvironmentDetector, 'is_docker', return_value=True):
                result = EnvironmentDetector.get_environment_name()
                
                assert result == 'kubernetes'
    
    def test_get_environment_name_docker(self):
        """Test environment name detection for Docker."""
        with patch.object(EnvironmentDetector, 'is_kubernetes', return_value=False):
            with patch.object(EnvironmentDetector, 'is_docker', return_value=True):
                result = EnvironmentDetector.get_environment_name()
                
                assert result == 'docker'
    
    def test_get_environment_name_local(self):
        """Test environment name detection for local environment."""
        with patch.object(EnvironmentDetector, 'is_kubernetes', return_value=False):
            with patch.object(EnvironmentDetector, 'is_docker', return_value=False):
                result = EnvironmentDetector.get_environment_name()
                
                assert result == 'local'


# Integration test markers
pytestmark = [
    pytest.mark.integration,
    pytest.mark.docker,
    pytest.mark.environment
]