"""
Simple test script for AutoML Framework API endpoints.

This script tests the basic functionality of the REST API.
"""

import requests
import json
import time
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_config_endpoint():
    """Test the configuration endpoint."""
    print("\nTesting config endpoint...")
    response = requests.get(f"{BASE_URL}/api/v1/config")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_authentication():
    """Test authentication endpoints."""
    print("\nTesting authentication...")
    
    # Test login
    login_data = {
        "username": "demo_user",
        "password": "secret"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/auth/login",
        data=login_data
    )
    
    print(f"Login status: {response.status_code}")
    if response.status_code == 200:
        token_data = response.json()
        print(f"Token received: {token_data['access_token'][:20]}...")
        return token_data['access_token']
    else:
        print(f"Login failed: {response.text}")
        return None

def test_dataset_upload(token=None):
    """Test dataset upload (mock)."""
    print("\nTesting dataset operations...")
    
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    # Test dataset list (should be empty initially)
    response = requests.get(f"{BASE_URL}/api/v1/datasets", headers=headers)
    print(f"Dataset list status: {response.status_code}")
    if response.status_code == 200:
        print(f"Datasets: {response.json()}")
    
    return response.status_code == 200

def test_experiments(token=None):
    """Test experiment endpoints."""
    print("\nTesting experiment operations...")
    
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    # Test experiment list
    response = requests.get(f"{BASE_URL}/api/v1/experiments", headers=headers)
    print(f"Experiment list status: {response.status_code}")
    if response.status_code == 200:
        print(f"Experiments: {response.json()}")
    
    return response.status_code == 200

def test_resources(token=None):
    """Test resource monitoring endpoints."""
    print("\nTesting resource monitoring...")
    
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    # Test resource status
    response = requests.get(f"{BASE_URL}/api/v1/resources/status", headers=headers)
    print(f"Resource status: {response.status_code}")
    if response.status_code == 200:
        print(f"Resources: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def main():
    """Run all API tests."""
    print("AutoML Framework API Test Suite")
    print("=" * 40)
    
    # Test basic endpoints
    if not test_health_check():
        print("❌ Health check failed")
        return
    print("✅ Health check passed")
    
    if not test_config_endpoint():
        print("❌ Config endpoint failed")
        return
    print("✅ Config endpoint passed")
    
    # Test authentication
    token = test_authentication()
    if token:
        print("✅ Authentication passed")
    else:
        print("❌ Authentication failed")
        # Continue with anonymous access
    
    # Test other endpoints
    if test_dataset_upload(token):
        print("✅ Dataset endpoints accessible")
    else:
        print("❌ Dataset endpoints failed")
    
    if test_experiments(token):
        print("✅ Experiment endpoints accessible")
    else:
        print("❌ Experiment endpoints failed")
    
    if test_resources(token):
        print("✅ Resource endpoints accessible")
    else:
        print("❌ Resource endpoints failed")
    
    print("\n" + "=" * 40)
    print("API test suite completed!")
    print("\nTo start the API server, run:")
    print("python automl_framework/api/server.py")
    print("\nThen visit http://localhost:8000/docs for interactive documentation")

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
        print("Make sure the server is running on http://localhost:8000")
        print("Start it with: python automl_framework/api/server.py")