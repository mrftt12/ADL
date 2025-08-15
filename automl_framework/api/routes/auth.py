"""
Authentication routes for AutoML Framework API.

This module provides login, logout, and user management endpoints.
"""

from datetime import timedelta
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from automl_framework.api.auth import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    get_admin_user,
    get_user,
    create_user,
    Token,
    User,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")

class LoginRequest(BaseModel):
    username: str
    password: str

class SignupRequest(BaseModel):
    username: str
    email: str
    password: str
    confirm_password: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str = None
    is_active: bool
    is_admin: bool

@router.post("/signup", response_model=UserResponse)
async def signup(request: SignupRequest):
    """
    User registration endpoint.
    """
    # Validate input
    if request.password != request.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match"
        )
    
    if len(request.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters long"
        )
    
    # Check if user already exists
    existing_user = get_user(request.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    # Create new user
    try:
        new_user = create_user(
            username=request.username,
            email=request.email,
            password=request.password
        )
        
        return UserResponse(
            id=new_user.id,
            username=new_user.username,
            email=new_user.email,
            is_active=new_user.is_active,
            is_admin=new_user.is_admin
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint that returns JWT access token.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_active_user)):
    """
    Logout endpoint (token invalidation would be handled by client).
    """
    return {"message": "Successfully logged out"}

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """
    Get current user information.
    """
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        is_active=current_user.is_active,
        is_admin=current_user.is_admin
    )

@router.get("/status")
async def auth_status():
    """
    Get authentication system status (public endpoint).
    """
    return {
        "status": "active",
        "login_endpoint": "/api/v1/auth/login",
        "signup_endpoint": "/api/v1/auth/signup",
        "demo_credentials": {
            "username": "demo_user",
            "password": "secret"
        }
    }

@router.get("/users", response_model=list[UserResponse])
async def list_users(admin_user: User = Depends(get_admin_user)):
    """
    List all users (admin only).
    """
    # This would query the actual user database
    # For now, return mock data
    return [
        UserResponse(
            id="user_123",
            username="demo_user",
            email="demo@example.com",
            is_active=True,
            is_admin=False
        ),
        UserResponse(
            id="admin_456",
            username="admin",
            email="admin@example.com",
            is_active=True,
            is_admin=True
        )
    ]