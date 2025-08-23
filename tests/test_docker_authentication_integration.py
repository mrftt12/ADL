"""
Integration tests for Docker authentication fallback functionality.

Tests authentication system behavior when database is unavailable,
demo user login functionality, and JWT token generation/validation
in memory mode as specified in requirements 1.2, 1.3, and 1.5.
"""

import pytest
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from automl_framework.api.auth import (
    AuthenticationManager,
    get_auth_manager,
    authenticate_user,
    create_access_token,
    verify_token,
    get_user,
    create_user,
    initialize_authentication,
    test_authentication as auth_test_function,
    get_auth_backend_info,
    User,
    UserInDB,
    TokenData
)
from automl_framework.core.environment import EnvironmentConfig


class TestAuthenticationFallback:
    """Test suite for authentication fallback functionality."""
    
    def test_authentication_manager_database_unavailable(self):
        """Test AuthenticationManager initialization when database is unavailable."""
        # Mock environment config for Docker
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='auto',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=False
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            # Mock database connection test to fail
            with patch.object(AuthenticationManager, '_test_database_connection', return_value=False):
                auth_manager = AuthenticationManager()
                
                assert auth_manager._database_available is False
                assert auth_manager._auth_backend == 'memory'
                assert len(auth_manager._in_memory_users) > 0
                assert 'demo_user' in auth_manager._in_memory_users
    
    def test_authentication_manager_database_available(self):
        """Test AuthenticationManager initialization when database is available."""
        # Mock environment config for Docker with database
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='auto',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=True
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            # Mock database connection test to succeed
            with patch.object(AuthenticationManager, '_test_database_connection', return_value=True):
                with patch.object(AuthenticationManager, '_initialize_database_users'):
                    auth_manager = AuthenticationManager()
                    
                    assert auth_manager._database_available is True
                    assert auth_manager._auth_backend == 'database'
    
    def test_authentication_manager_fallback_on_initialization_error(self):
        """Test AuthenticationManager falls back to memory on initialization errors."""
        # Mock environment config
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='database',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=True
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            # Mock database connection test to raise exception
            with patch.object(AuthenticationManager, '_test_database_connection', side_effect=Exception("Database error")):
                auth_manager = AuthenticationManager()
                
                # Should fallback to memory mode
                assert auth_manager._database_available is False
                assert auth_manager._auth_backend == 'memory'
                assert len(auth_manager._in_memory_users) > 0
    
    def test_database_connection_test_success(self):
        """Test database connection test when database is available."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='auto',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=True
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            # Mock successful database manager
            with patch('automl_framework.core.database.get_database_manager') as mock_get_db:
                mock_db_manager = Mock()
                mock_db_manager.health_check.return_value = {
                    'postgresql': True,
                    'mongodb': False,
                    'redis': False
                }
                mock_get_db.return_value = mock_db_manager
                
                auth_manager = AuthenticationManager()
                result = auth_manager._test_database_connection()
                
                assert result is True
    
    def test_database_connection_test_failure(self):
        """Test database connection test when database is unavailable."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='auto',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=False
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            # Mock failed database manager
            with patch('automl_framework.core.database.get_database_manager') as mock_get_db:
                mock_db_manager = Mock()
                mock_db_manager.health_check.return_value = {
                    'postgresql': False,
                    'mongodb': False,
                    'redis': False
                }
                mock_get_db.return_value = mock_db_manager
                
                auth_manager = AuthenticationManager()
                result = auth_manager._test_database_connection()
                
                assert result is False
    
    def test_database_connection_test_exception(self):
        """Test database connection test handles exceptions gracefully."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='auto',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=False
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            # Mock database manager to raise exception
            with patch('automl_framework.core.database.get_database_manager', side_effect=ImportError("Database module not found")):
                auth_manager = AuthenticationManager()
                result = auth_manager._test_database_connection()
                
                assert result is False


class TestDemoUserLogin:
    """Test suite for demo user login functionality."""
    
    def test_demo_user_initialization(self):
        """Test demo users are properly initialized in memory storage."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='memory',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=False
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            auth_manager = AuthenticationManager()
            
            # Check demo users exist
            assert 'demo_user' in auth_manager._in_memory_users
            assert 'admin' in auth_manager._in_memory_users
            
            # Check demo user properties
            demo_user = auth_manager._in_memory_users['demo_user']
            assert demo_user['username'] == 'demo_user'
            assert demo_user['email'] == 'demo@example.com'
            assert demo_user['is_active'] is True
            assert demo_user['is_admin'] is False
            assert 'hashed_password' in demo_user
            
            # Check admin user properties
            admin_user = auth_manager._in_memory_users['admin']
            assert admin_user['username'] == 'admin'
            assert admin_user['is_admin'] is True
    
    def test_demo_user_authentication_success(self):
        """Test successful authentication with demo user credentials."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='memory',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=False
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            auth_manager = AuthenticationManager()
            
            # Test demo user authentication
            user = auth_manager.authenticate_user('demo_user', 'secret')
            
            assert user is not None
            assert user.username == 'demo_user'
            assert user.email == 'demo@example.com'
            assert user.is_active is True
            assert user.is_admin is False
    
    def test_demo_user_authentication_wrong_password(self):
        """Test authentication failure with wrong password."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='memory',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=False
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            auth_manager = AuthenticationManager()
            
            # Test authentication with wrong password
            user = auth_manager.authenticate_user('demo_user', 'wrong_password')
            
            assert user is None
    
    def test_demo_user_authentication_nonexistent_user(self):
        """Test authentication failure with nonexistent user."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='memory',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=False
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            auth_manager = AuthenticationManager()
            
            # Test authentication with nonexistent user
            user = auth_manager.authenticate_user('nonexistent_user', 'secret')
            
            assert user is None
    
    def test_admin_user_authentication(self):
        """Test successful authentication with admin user credentials."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='memory',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=False
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            auth_manager = AuthenticationManager()
            
            # Test admin user authentication
            user = auth_manager.authenticate_user('admin', 'secret')
            
            assert user is not None
            assert user.username == 'admin'
            assert user.is_active is True
            assert user.is_admin is True


class TestJWTTokenGeneration:
    """Test suite for JWT token generation and validation in memory mode."""
    
    def test_create_access_token(self):
        """Test JWT token creation."""
        token_data = {
            'sub': 'demo_user',
            'user_id': 'user_123'
        }
        expires_delta = timedelta(minutes=30)
        
        token = create_access_token(token_data, expires_delta)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_verify_token_valid(self):
        """Test JWT token verification with valid token."""
        token_data = {
            'sub': 'demo_user',
            'user_id': 'user_123'
        }
        expires_delta = timedelta(minutes=30)
        
        token = create_access_token(token_data, expires_delta)
        verified_token = verify_token(token)
        
        assert verified_token is not None
        assert verified_token.username == 'demo_user'
        assert verified_token.user_id == 'user_123'
    
    def test_verify_token_expired(self):
        """Test JWT token verification with expired token."""
        token_data = {
            'sub': 'demo_user',
            'user_id': 'user_123'
        }
        expires_delta = timedelta(seconds=-1)  # Already expired
        
        token = create_access_token(token_data, expires_delta)
        verified_token = verify_token(token)
        
        assert verified_token is None
    
    def test_verify_token_invalid(self):
        """Test JWT token verification with invalid token."""
        invalid_token = "invalid.token.here"
        
        verified_token = verify_token(invalid_token)
        
        assert verified_token is None
    
    def test_verify_token_empty(self):
        """Test JWT token verification with empty token."""
        verified_token = verify_token("")
        
        assert verified_token is None
        
        verified_token = verify_token(None)
        
        assert verified_token is None
    
    def test_verify_token_malformed(self):
        """Test JWT token verification with malformed token."""
        malformed_tokens = [
            "not.a.jwt",
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9",  # Missing parts
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.invalid",  # Invalid payload
        ]
        
        for token in malformed_tokens:
            verified_token = verify_token(token)
            assert verified_token is None


class TestMemoryModeUserManagement:
    """Test suite for user management in memory mode."""
    
    def test_get_user_from_memory(self):
        """Test getting user from memory storage."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='memory',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=False
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            auth_manager = AuthenticationManager()
            
            # Test getting existing user
            user = auth_manager.get_user('demo_user')
            
            assert user is not None
            assert user.username == 'demo_user'
            assert user.email == 'demo@example.com'
            
            # Test getting nonexistent user
            user = auth_manager.get_user('nonexistent')
            
            assert user is None
    
    def test_create_user_in_memory(self):
        """Test creating new user in memory storage."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='memory',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=False
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            auth_manager = AuthenticationManager()
            
            # Create new user
            new_user = auth_manager.create_user('test_user', 'test@example.com', 'password123')
            
            assert new_user is not None
            assert new_user.username == 'test_user'
            assert new_user.email == 'test@example.com'
            assert new_user.is_active is True
            assert new_user.is_admin is False
            
            # Verify user was added to memory storage
            assert 'test_user' in auth_manager._in_memory_users
            
            # Verify user can be retrieved
            retrieved_user = auth_manager.get_user('test_user')
            assert retrieved_user is not None
            assert retrieved_user.username == 'test_user'
    
    def test_create_user_duplicate_username(self):
        """Test creating user with duplicate username fails."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='memory',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=False
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            auth_manager = AuthenticationManager()
            
            # Try to create user with existing username
            with pytest.raises(ValueError, match="Username already exists"):
                auth_manager.create_user('demo_user', 'new@example.com', 'password123')
    
    def test_create_user_duplicate_email(self):
        """Test creating user with duplicate email fails."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='memory',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=False
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            auth_manager = AuthenticationManager()
            
            # Try to create user with existing email
            with pytest.raises(ValueError, match="Email already exists"):
                auth_manager.create_user('new_user', 'demo@example.com', 'password123')
    
    def test_create_user_invalid_username(self):
        """Test creating user with invalid username fails."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='memory',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=False
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            auth_manager = AuthenticationManager()
            
            # Test short username
            with pytest.raises(ValueError, match="Username must be at least 3 characters long"):
                auth_manager.create_user('ab', 'test@example.com', 'password123')
            
            # Test invalid characters
            with pytest.raises(ValueError, match="Username can only contain letters, numbers, and underscores"):
                auth_manager.create_user('user@name', 'test@example.com', 'password123')
    
    def test_create_user_invalid_email(self):
        """Test creating user with invalid email fails."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='memory',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=False
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            auth_manager = AuthenticationManager()
            
            # Test invalid email format
            with pytest.raises(ValueError, match="Invalid email format"):
                auth_manager.create_user('test_user', 'invalid-email', 'password123')


class TestAuthenticationIntegration:
    """Test suite for authentication integration functionality."""
    
    def test_initialize_authentication_success(self):
        """Test successful authentication initialization."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='memory',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=False
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            # Mock database connection to fail
            with patch.object(AuthenticationManager, '_test_database_connection', return_value=False):
                # Reset global auth manager
                import automl_framework.api.auth
                automl_framework.api.auth._auth_manager = None
                
                result = initialize_authentication()
                
                assert result['status'] == 'success'
                assert result['backend'] == 'memory'
                assert result['database_available'] is False
                assert result['demo_users_available']['demo_user'] is True
                assert result['demo_users_available']['admin'] is True
    
    def test_test_authentication_success(self):
        """Test authentication test with demo credentials."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='memory',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=False
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            # Reset global auth manager
            import automl_framework.api.auth
            automl_framework.api.auth._auth_manager = None
            
            result = auth_test_function('demo_user', 'secret')
            
            assert result['status'] == 'success'
            assert result['username'] == 'demo_user'
            assert result['token_valid'] is True
            assert result['backend'] == 'memory'
    
    def test_test_authentication_failure(self):
        """Test authentication test with wrong credentials."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='memory',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=False
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            # Reset global auth manager
            import automl_framework.api.auth
            automl_framework.api.auth._auth_manager = None
            
            result = auth_test_function('demo_user', 'wrong_password')
            
            assert result['status'] == 'failed'
            assert result['step'] == 'authentication'
    
    def test_get_auth_backend_info(self):
        """Test getting authentication backend information."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='memory',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=False
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            # Mock database connection to fail
            with patch.object(AuthenticationManager, '_test_database_connection', return_value=False):
                # Reset global auth manager
                import automl_framework.api.auth
                automl_framework.api.auth._auth_manager = None
                
                info = get_auth_backend_info()
                
                assert info['backend'] == 'memory'
                assert info['database_available'] is False
                assert info['environment'] == 'docker'
                assert info['user_count'] == 2  # demo_user and admin


class TestDatabaseFallback:
    """Test suite for database fallback scenarios."""
    
    def test_user_lookup_database_fallback(self):
        """Test user lookup falls back to memory when database fails."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='database',  # Try database first
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=True
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            # Mock database connection to succeed initially
            with patch.object(AuthenticationManager, '_test_database_connection', return_value=True):
                # Mock database user initialization to fallback to memory
                with patch.object(AuthenticationManager, '_initialize_database_users') as mock_init_db:
                    mock_init_db.side_effect = Exception("Database initialization failed")
                    
                    auth_manager = AuthenticationManager()
                    
                    # Should have fallen back to memory
                    assert auth_manager._auth_backend == 'memory'
                    
                    # Test user lookup works with memory fallback
                    user = auth_manager.get_user('demo_user')
                    assert user is not None
                    assert user.username == 'demo_user'
    
    def test_authentication_database_fallback(self):
        """Test authentication falls back to memory when database fails."""
        mock_env_config = EnvironmentConfig(
            name='docker',
            supports_gpu=True,
            gpu_available=False,
            supports_persistent_storage=True,
            default_auth_backend='database',
            max_memory_mb=1024,
            max_cpu_cores=1,
            database_available=True
        )
        
        with patch('automl_framework.api.auth.get_environment_manager') as mock_get_env:
            mock_env_manager = Mock()
            mock_env_manager.config = mock_env_config
            mock_get_env.return_value = mock_env_manager
            
            # Mock database operations to fail
            with patch.object(AuthenticationManager, '_test_database_connection', return_value=True):
                # Mock database user initialization to fail, forcing fallback to memory
                with patch.object(AuthenticationManager, '_initialize_database_users') as mock_init_db:
                    mock_init_db.side_effect = Exception("Database initialization failed")
                    
                    auth_manager = AuthenticationManager()
                    
                    # Should have fallen back to memory mode
                    assert auth_manager._auth_backend == 'memory'
                    
                    # Authentication should work via memory fallback
                    user = auth_manager.authenticate_user('demo_user', 'secret')
                    assert user is not None
                    assert user.username == 'demo_user'


# Integration test markers
pytestmark = [
    pytest.mark.integration,
    pytest.mark.docker,
    pytest.mark.authentication
]