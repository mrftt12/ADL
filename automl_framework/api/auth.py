"""
Authentication middleware and utilities for AutoML Framework API.

This module provides JWT-based authentication, user management,
and authorization for API endpoints.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

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

# Mock user database (replace with real database in production)
fake_users_db = {
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

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)

def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database."""
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)
    return None

def create_user(username: str, email: str, password: str) -> UserInDB:
    """Create a new user."""
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
    if username in fake_users_db:
        raise ValueError("Username already exists")
    
    # Check if email already exists
    for existing_user in fake_users_db.values():
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
    
    # Add to fake database
    fake_users_db[username] = user_data
    
    return UserInDB(**user_data)

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user with username and password."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[TokenData]:
    """Verify JWT token and extract user data."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        if username is None:
            return None
        token_data = TokenData(username=username, user_id=user_id)
        return token_data
    except JWTError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """
    Get current authenticated user from JWT token.
    Raises exception if no valid credentials provided.
    """
    if not credentials:
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
    
    token_data = verify_token(credentials.credentials)
    if token_data is None:
        raise credentials_exception
    
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    
    return User(
        id=user.id,
        username=user.username,
        email=user.email,
        is_active=user.is_active,
        is_admin=user.is_admin
    )

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
            return None
        
        try:
            scheme, token = auth_header.split()
            if scheme.lower() != "bearer":
                return None
            
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
            logger.warning(f"Authentication failed: {e}")
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