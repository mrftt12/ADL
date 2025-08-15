# Authentication Issue Fix

## Problem
Users were getting "authentication required" errors when trying to create projects, even though they were logged in.

## Root Cause
The API endpoints were using `get_current_user` instead of `get_current_active_user`. The `get_current_user` function was designed to return an "anonymous" user when no credentials were provided, which was causing confusion.

```python
# PROBLEMATIC CODE in get_current_user:
if not credentials:
    # Return anonymous user for endpoints that don't require authentication
    return User(
        id="anonymous",
        username="anonymous",
        is_active=True,
        is_admin=False
    )
```

This meant that:
- Endpoints appeared to work without authentication
- Users thought they were authenticated when they weren't
- The UI showed confusing "authentication required" messages

## Solution

### 1. Updated Experiments Endpoints
Changed all experiment endpoints from `get_current_user` to `get_current_active_user`:

**Before:**
```python
async def create_experiment(
    request: ExperimentCreateRequest,
    current_user: User = Depends(get_current_user)  # ❌ Allows anonymous users
):
```

**After:**
```python
async def create_experiment(
    request: ExperimentCreateRequest,
    current_user: User = Depends(get_current_active_user)  # ✅ Requires authentication
):
```

### 2. Updated Dataset Endpoints
Applied the same fix to all dataset endpoints:
- `upload_dataset`
- `analyze_dataset` 
- `get_dataset_info`
- `delete_dataset`
- `list_datasets`

### 3. Authentication Behavior Now
- ✅ **Unauthenticated requests**: Return `403 Forbidden`
- ✅ **Authenticated requests**: Work properly (`200 OK`)
- ✅ **Invalid token requests**: Return `401 Unauthorized`

## Files Modified

### Backend
- `automl_framework/api/routes/experiments.py` - Updated all endpoints to use `get_current_active_user`
- `automl_framework/api/routes/datasets.py` - Updated all endpoints to use `get_current_active_user`

### No Frontend Changes Needed
The frontend authentication logic was already correct. The issue was purely on the backend.

## Test Results

### Before Fix
```bash
# Unauthenticated request would work (wrong!)
POST /api/v1/experiments (no auth) → 200 OK with anonymous user
```

### After Fix
```bash
# Proper authentication enforcement
POST /api/v1/experiments (no auth) → 403 Forbidden
POST /api/v1/experiments (valid token) → 200 OK  
POST /api/v1/experiments (invalid token) → 401 Unauthorized
```

## Usage
Users can now:
1. Sign in to the application
2. Create projects/experiments successfully
3. Get proper error messages if authentication fails
4. Have their authentication state properly validated

## Authentication Flow
1. User signs in → Token stored in localStorage
2. API client includes token in all requests
3. Backend validates token and requires active user
4. Requests succeed only with valid authentication

The "authentication required" error should no longer appear for properly logged-in users.