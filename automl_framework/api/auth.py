"""
Authentication middleware and utilities for AutoML Framework API.

This module provides JWT-based authentication, user management,
and authorization for API endpoints with environment-aware backend selection.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from automl_framework.core.environment import get_environment_manager

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = "your-secret-key-change-in-production"  # Change in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[str] = None

class User(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    is_active: bool = True
    is_admin: bool = False

class UserInDB(User):
    hashed_password: str


class AuthenticationManager:
    """
    Environment-aware authentication manager that handles database connectivity
    and graceful fallback to in-memory storage.
    """
    
    def __init__(self):
        self.environment_manager = get_environment_manager()
        self._database_available = None
        self._auth_backend = None
        self._in_memory_users = {}
        self._initialize_authentication()
    
    def _initialize_authentication(self):
        """Initialize authentication system with environment detection."""
        logger.info("=== Authentication System Initialization ===")
        
        try:
            # Test database connectivity
            logger.info("Testing database connectivity for authentication...")
            self._database_available = self._test_database_connection()
            logger.info(f"Database connectivity test result: {self._database_available}")
            
            # Determine authentication backend
            logger.info("Selecting authentication backend...")
            self._auth_backend = self._select_auth_backend()
            logger.info(f"Authentication backend selected: {self._auth_backend}")
            
            # Initialize users based on backend
            logger.info(f"Initializing users with {self._auth_backend} backend...")
            self._initialize_users()
            
            logger.info("=== Authentication Initialization Complete ===")
            logger.info(f"Backend: {self._auth_backend}")
            logger.info(f"Database Available: {self._database_available}")
            logger.info(f"Environment: {self.environment_manager.config.name}")
            logger.info("============================================")
            
        except Exception as e:
            logger.error(f"Authentication initialization failed: {e}", exc_info=True)
            # Fallback to in-memory authentication
            logger.warning("=== Authentication Fallback Mode ===")
            self._database_available = False
            self._auth_backend = "memory"
            logger.warning(f"Falling back to in-memory authentication due to initialization failure")
            
            try:
                self._initialize_users()
                logger.warning("In-memory authentication fallback successful")
            except Exception as fallback_error:
                logger.error(f"Authentication fallback also failed: {fallback_error}", exc_info=True)
            
            logger.warning("=====================================")
    
    def _test_database_connection(self) -> bool:
        """
        Test database connectivity for authentication.
        
        Returns:
            bool: True if database is available and accessible
        """
        try:
            from automl_framework.core.database import get_database_manager
            
            db_manager = get_database_manager()
            health_status = db_manager.health_check()
            
            # Consider database available if at least one database is healthy
            database_available = any(health_status.values())
            
            if database_available:
                logger.info("Database connectivity test passed")
            else:
                logger.warning("Database connectivity test failed - no databases available")
            
            return database_available
            
        except Exception as e:
            logger.warning(f"Database connectivity test failed: {e}")
            return False
    
    def _select_auth_backend(self) -> str:
        """
        Select appropriate authentication backend based on environment and database availability.
        
        Returns:
            str: Selected authentication backend ("database" or "memory")
        """
        env_config = self.environment_manager.config
        
        logger.debug(f"Environment default auth backend: {env_config.default_auth_backend}")
        logger.debug(f"Database available: {self._database_available}")
        
        if env_config.default_auth_backend == "auto":
            backend = "database" if self._database_available else "memory"
            logger.info(f"Auto-selecting authentication backend: {backend} (database_available={self._database_available})")
        else:
            backend = env_config.default_auth_backend
            logger.info(f"Using configured authentication backend: {backend}")
        
        # Override to memory if database is not available
        if backend == "database" and not self._database_available:
            logger.warning("Database backend requested but database unavailable, falling back to memory")
            logger.warning("This may happen in Docker environments without database services")
            backend = "memory"
        
        logger.info(f"Final authentication backend selection: {backend}")
        return backend
    
    def _initialize_users(self):
        """Initialize users based on selected authentication backend."""
        if self._auth_backend == "memory":
            self._initialize_memory_users()
        else:
            self._initialize_database_users()
    
    def _initialize_memory_users(self):
        """Initialize in-memory user storage with demo users."""
        logger.info("Initializing in-memory user storage...")
        
        self._in_memory_users = {
            "demo_user": {
                "id": "user_123",
                "username": "demo_user",
                "email": "demo@example.com",
                "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
                "is_active": True,
                "is_admin": False,
            },
            "admin": {
                "id": "admin_456",
                "username": "admin",
                "email": "admin@example.com",
                "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
                "is_active": True,
                "is_admin": True,
            }
        }
        
        logger.info(f"In-memory authentication initialized with {len(self._in_memory_users)} demo users:")
        for username, user_data in self._in_memory_users.items():
            logger.info(f"  - {username} (ID: {user_data['id']}, Admin: {user_data['is_admin']})")
        
        logger.info("Demo credentials: username='demo_user', password='secret'")
        logger.info("Admin credentials: username='admin', password='secret'")
    
    def _initialize_database_users(self):
        """Initialize database-backed user storage."""
        try:
            # TODO: Initialize database user tables and default users
            # For now, we'll use in-memory as fallback for Docker environments
            logger.warning("Database user initialization not yet implemented")
            logger.warning("Falling back to in-memory user storage for demo users")
            self._auth_backend = "memory"
            self._initialize_memory_users()
        except Exception as e:
            logger.error(f"Database user initialization failed: {e}")
            logger.warning("Falling back to in-memory user storage")
            self._auth_backend = "memory"
            self._initialize_memory_users()
    
    def get_user(self, username: str) -> Optional[UserInDB]:
        """
        Get user from the appropriate backend.
        
        Args:
            username: Username to look up
            
        Returns:
            UserInDB: User object if found, None otherwise
        """
        try:
            if self._auth_backend == "memory":
                return self._get_user_from_memory(username)
            else:
                return self._get_user_from_database(username)
        except Exception as e:
            logger.error(f"Error getting user {username}: {e}")
            # Try fallback to memory if database fails
            if self._auth_backend != "memory":
                logger.warning(f"Database lookup failed for user {username}, trying memory fallback")
                return self._get_user_from_memory(username)
            return None
    
    def _get_user_from_memory(self, username: str) -> Optional[UserInDB]:
        """Get user from in-memory storage."""
        if username in self._in_memory_users:
            user_dict = self._in_memory_users[username]
            return UserInDB(**user_dict)
        return None
    
    def _get_user_from_database(self, username: str) -> Optional[UserInDB]:
        """Get user from database storage."""
        try:
            # TODO: Implement database user lookup
            # For now, fallback to memory
            return self._get_user_from_memory(username)
        except Exception as e:
            logger.error(f"Database user lookup failed: {e}")
            return None
    
    def create_user(self, username: str, email: str, password: str) -> UserInDB:
        """
        Create a new user in the appropriate backend.
        
        Args:
            username: Username for new user
            email: Email for new user
            password: Plain text password
            
        Returns:
            UserInDB: Created user object
            
        Raises:
            ValueError: If user creation fails due to validation or conflicts
        """
        try:
            if self._auth_backend == "memory":
                return self._create_user_in_memory(username, email, password)
            else:
                return self._create_user_in_database(username, email, password)
        except Exception as e:
            logger.error(f"Error creating user {username}: {e}")
            # Try fallback to memory if database fails
            if self._auth_backend != "memory":
                logger.warning(f"Database user creation failed for {username}, trying memory fallback")
                return self._create_user_in_memory(username, email, password)
            raise
    
    def _create_user_in_memory(self, username: str, email: str, password: str) -> UserInDB:
        """Create user in in-memory storage."""
        import uuid
        import re
        
        # Validate username
        if len(username) < 3:
            raise ValueError("Username must be at least 3 characters long")
        if not re.match("^[a-zA-Z0-9_]+$", username):
            raise ValueError("Username can only contain letters, numbers, and underscores")
        
        # Validate email
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            raise ValueError("Invalid email format")
        
        # Check if username already exists
        if username in self._in_memory_users:
            raise ValueError("Username already exists")
        
        # Check if email already exists
        for existing_user in self._in_memory_users.values():
            if existing_user.get("email") == email:
                raise ValueError("Email already exists")
        
        # Create new user
        user_id = f"user_{str(uuid.uuid4())[:8]}"
        hashed_password = get_password_hash(password)
        
        user_data = {
            "id": user_id,
            "username": username,
            "email": email,
            "hashed_password": hashed_password,
            "is_active": True,
            "is_admin": False
        }
        
        # Add to in-memory database
        self._in_memory_users[username] = user_data
        
        return UserInDB(**user_data)
    
    def _create_user_in_database(self, username: str, email: str, password: str) -> UserInDB:
        """Create user in database storage."""
        try:
            # TODO: Implement database user creation
            # For now, fallback to memory
            return self._create_user_in_memory(username, email, password)
        except Exception as e:
            logger.error(f"Database user creation failed: {e}")
            raise
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """
        Authenticate user with username and password.
        
        Args:
            username: Username to authenticate
            password: Plain text password
            
        Returns:
            UserInDB: Authenticated user object if successful, None otherwise
        """
        try:
            user = self.get_user(username)
            if not user:
                logger.warning(f"Authentication failed: user {username} not found")
                return None
            
            if not verify_password(password, user.hashed_password):
                logger.warning(f"Authentication failed: invalid password for user {username}")
                return None
            
            logger.info(f"User {username} authenticated successfully using {self._auth_backend} backend")
            return user
            
        except Exception as e:
            logger.error(f"Authentication error for user {username}: {e}")
            return None
    
    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about the current authentication backend.
        
        Returns:
            Dict containing backend information
        """
        return {
            "backend": self._auth_backend,
            "database_available": self._database_available,
            "environment": self.environment_manager.config.name,
            "user_count": len(self._in_memory_users) if self._auth_backend == "memory" else "unknown"
        }


# Global authentication manager instance
_auth_manager: Optional[AuthenticationManager] = None


def get_auth_manager() -> AuthenticationManager:
    """
    Get the global authentication manager instance.
    
    Returns:
        AuthenticationManager: The global authentication manager
    """
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthenticationManager()
    return _auth_manager


# Legacy compatibility - maintain existing interface
fake_users_db = {}  # Will be populated by auth manager

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)

def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database."""
    auth_manager = get_auth_manager()
    return auth_manager.get_user(username)

def create_user(username: str, email: str, password: str) -> UserInDB:
    """Create a new user."""
    auth_manager = get_auth_manager()
    return auth_manager.create_user(username, email, password)

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user with username and password."""
    auth_manager = get_auth_manager()
    return auth_manager.authenticate_user(username, password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire.timestamp()})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[TokenData]:
    """Verify JWT token and extract user data."""
    try:
        if not token or not token.strip():
            logger.warning("Token verification failed: empty token")
            return None
            
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        
        if username is None:
            logger.warning("Token verification failed: no username in token")
            return None
            
        # Check token expiration
        exp = payload.get("exp")
        if exp and datetime.utcnow().timestamp() > exp:
            logger.warning(f"Token verification failed: token expired for user {username}")
            return None
            
        token_data = TokenData(username=username, user_id=user_id)
        logger.debug(f"Token verified successfully for user {username}")
        return token_data
        
    except JWTError as e:
        logger.warning(f"Token verification failed: JWT error - {e}")
        return None
    except Exception as e:
        logger.error(f"Token verification failed: unexpected error - {e}")
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """
    Get current authenticated user from JWT token.
    Raises exception if no valid credentials provided.
    """
    if not credentials:
        logger.warning("Authentication failed: no credentials provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token_data = verify_token(credentials.credentials)
        if token_data is None:
            logger.warning("Authentication failed: invalid token")
            raise credentials_exception
        
        user = get_user(username=token_data.username)
        if user is None:
            logger.warning(f"Authentication failed: user {token_data.username} not found")
            raise credentials_exception
        
        if not user.is_active:
            logger.warning(f"Authentication failed: user {user.username} is inactive")
            raise HTTPException(status_code=400, detail="Inactive user")
        
        logger.debug(f"User {user.username} authenticated successfully")
        return User(
            id=user.id,
            username=user.username,
            email=user.email,
            is_active=user.is_active,
            is_admin=user.is_admin
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise credentials_exception

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user (requires authentication)."""
    if current_user.username == "anonymous":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return current_user

async def get_admin_user(current_user: User = Depends(get_current_active_user)) -> User:
    """Get current admin user (requires admin privileges)."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

class AuthenticationMiddleware:
    """
    Custom authentication middleware for request processing.
    """
    
    def __init__(self):
        self.public_endpoints = {
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/auth/login",
            "/api/v1/config"
        }
    
    def is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public (doesn't require authentication)."""
        return any(path.startswith(endpoint) for endpoint in self.public_endpoints)
    
    async def authenticate_request(self, request) -> Optional[User]:
        """Authenticate request and return user if valid."""
        # Extract authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            logger.debug("No authorization header found")
            return None
        
        try:
            # Parse authorization header
            parts = auth_header.split()
            if len(parts) != 2:
                logger.warning(f"Invalid authorization header format: {len(parts)} parts")
                return None
                
            scheme, token = parts
            if scheme.lower() != "bearer":
                logger.warning(f"Invalid authorization scheme: {scheme}")
                return None
            
            # Verify token
            token_data = verify_token(token)
            if not token_data:
                logger.warning("Token verification failed")
                return None
            
            # Get user
            user = get_user(token_data.username)
            if not user:
                logger.warning(f"User not found: {token_data.username}")
                return None
            
            # Check if user is active
            if not user.is_active:
                logger.warning(f"User is inactive: {user.username}")
                return None
            
            logger.debug(f"Request authenticated successfully for user: {user.username}")
            return User(
                id=user.id,
                username=user.username,
                email=user.email,
                is_active=user.is_active,
                is_admin=user.is_admin
            )
            
        except ValueError as e:
            logger.warning(f"Authentication failed: invalid format - {e}")
            return None
        except Exception as e:
            logger.error(f"Authentication failed: unexpected error - {e}")
            return None

# Rate limiting (simple in-memory implementation)
class RateLimiter:
    """
    Simple rate limiter for API endpoints.
    """
    
    def __init__(self, max_requests: int = 100, window_minutes: int = 15):
        self.max_requests = max_requests
        self.window_minutes = window_minutes
        self.requests = {}  # {user_id: [(timestamp, count), ...]}
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if user is allowed to make request."""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=self.window_minutes)
        
        # Clean old entries
        if user_id in self.requests:
            self.requests[user_id] = [
                (timestamp, count) for timestamp, count in self.requests[user_id]
                if timestamp > window_start
            ]
        else:
            self.requests[user_id] = []
        
        # Count requests in current window
        total_requests = sum(count for _, count in self.requests[user_id])
        
        if total_requests >= self.max_requests:
            return False
        
        # Add current request
        self.requests[user_id].append((now, 1))
        return True

async def get_current_user_websocket(token: str) -> Optional[User]:
    """
    Get current authenticated user from JWT token for WebSocket connections.
    
    Args:
        token: JWT token string
        
    Returns:
        User object if authentication successful, None otherwise
    """
    try:
        token_data = verify_token(token)
        if not token_data:
            return None
        
        user = get_user(token_data.username)
        if not user:
            return None
        
        return User(
            id=user.id,
            username=user.username,
            email=user.email,
            is_active=user.is_active,
            is_admin=user.is_admin
        )
    except Exception as e:
        logger.warning(f"WebSocket authentication failed: {e}")
        return None

# Global rate limiter instance
rate_limiter = RateLimiter()

def get_auth_backend_info() -> Dict[str, Any]:
    """
    Get information about the current authentication backend.
    Useful for debugging and health checks.
    
    Returns:
        Dict containing backend information
    """
    try:
        auth_manager = get_auth_manager()
        return auth_manager.get_backend_info()
    except Exception as e:
        logger.error(f"Error getting auth backend info: {e}")
        return {
            "backend": "unknown",
            "database_available": False,
            "environment": "unknown",
            "user_count": "unknown",
            "error": str(e)
        }

def initialize_authentication() -> Dict[str, Any]:
    """
    Initialize authentication system and return status.
    This can be called during application startup to ensure proper initialization.
    
    Returns:
        Dict containing initialization status
    """
    try:
        auth_manager = get_auth_manager()
        backend_info = auth_manager.get_backend_info()
        
        # Verify demo users are available
        demo_user = auth_manager.get_user("demo_user")
        admin_user = auth_manager.get_user("admin")
        
        status = {
            "status": "success",
            "backend": backend_info["backend"],
            "database_available": backend_info["database_available"],
            "environment": backend_info["environment"],
            "demo_users_available": {
                "demo_user": demo_user is not None,
                "admin": admin_user is not None
            }
        }
        
        logger.info(f"Authentication initialization successful: {status}")
        return status
        
    except Exception as e:
        logger.error(f"Authentication initialization failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "backend": "unknown",
            "database_available": False
        }

def test_authentication(username: str = "demo_user", password: str = "secret") -> Dict[str, Any]:
    """
    Test authentication with demo credentials.
    Useful for health checks and debugging.
    
    Args:
        username: Username to test (default: demo_user)
        password: Password to test (default: secret)
        
    Returns:
        Dict containing test results
    """
    try:
        # Test user lookup
        user = get_user(username)
        if not user:
            return {
                "status": "failed",
                "error": f"User {username} not found",
                "step": "user_lookup"
            }
        
        # Test authentication
        authenticated_user = authenticate_user(username, password)
        if not authenticated_user:
            return {
                "status": "failed",
                "error": f"Authentication failed for user {username}",
                "step": "authentication"
            }
        
        # Test token creation
        token_data = {"sub": username, "user_id": user.id}
        token = create_access_token(token_data, expires_delta=timedelta(minutes=30))
        if not token:
            return {
                "status": "failed",
                "error": "Token creation failed",
                "step": "token_creation"
            }
        
        # Test token verification
        verified_token = verify_token(token)
        if not verified_token or verified_token.username != username:
            return {
                "status": "failed",
                "error": "Token verification failed",
                "step": "token_verification"
            }
        
        return {
            "status": "success",
            "username": username,
            "user_id": user.id,
            "token_valid": True,
            "backend": get_auth_manager().get_backend_info()["backend"]
        }
        
    except Exception as e:
        logger.error(f"Authentication test failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "step": "unknown"
        }