#!/usr/bin/env python3
"""
Integration test for user signup functionality.

This script tests the complete signup flow including validation,
user creation, and subsequent login.
"""

import requests
import json
import random
import string

def generate_random_user():
    """Generate random user data for testing."""
    random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return {
        'username': f'testuser_{random_id}',
        'email': f'test_{random_id}@example.com',
        'password': 'testpassword123',
        'confirm_password': 'testpassword123'
    }

def test_signup_success():
    """Test successful user signup."""
    print("Testing successful signup...")
    
    user_data = generate_random_user()
    
    response = requests.post(
        'http://localhost:8000/api/v1/auth/signup',
        json=user_data,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Signup successful!")
        print(f"   User ID: {data['id']}")
        print(f"   Username: {data['username']}")
        print(f"   Email: {data['email']}")
        
        # Test login with new user
        login_response = requests.post(
            'http://localhost:8000/api/v1/auth/login',
            data={
                'username': user_data['username'],
                'password': user_data['password']
            }
        )
        
        if login_response.status_code == 200:
            print("✅ Login with new user successful!")
            return True
        else:
            print(f"❌ Login failed: {login_response.text}")
            return False
    else:
        print(f"❌ Signup failed: {response.text}")
        return False

def test_signup_validation():
    """Test signup validation."""
    print("\nTesting signup validation...")
    
    # Test password mismatch
    user_data = generate_random_user()
    user_data['confirm_password'] = 'different_password'
    
    response = requests.post(
        'http://localhost:8000/api/v1/auth/signup',
        json=user_data,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 400 and 'Passwords do not match' in response.text:
        print("✅ Password mismatch validation works")
    else:
        print(f"❌ Password mismatch validation failed: {response.text}")
        return False
    
    # Test duplicate username
    existing_user = generate_random_user()
    
    # Create user first
    requests.post(
        'http://localhost:8000/api/v1/auth/signup',
        json=existing_user,
        headers={'Content-Type': 'application/json'}
    )
    
    # Try to create same user again
    response = requests.post(
        'http://localhost:8000/api/v1/auth/signup',
        json=existing_user,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 400 and 'Username already exists' in response.text:
        print("✅ Duplicate username validation works")
    else:
        print(f"❌ Duplicate username validation failed: {response.text}")
        return False
    
    # Test short password
    user_data = generate_random_user()
    user_data['password'] = '123'
    user_data['confirm_password'] = '123'
    
    response = requests.post(
        'http://localhost:8000/api/v1/auth/signup',
        json=user_data,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 400 and 'at least 6 characters' in response.text:
        print("✅ Short password validation works")
    else:
        print(f"❌ Short password validation failed: {response.text}")
        return False
    
    return True

def test_api_client_integration():
    """Test that the API client can handle signup."""
    print("\nTesting API client integration...")
    
    # This would test the TypeScript API client, but we'll simulate it
    user_data = generate_random_user()
    
    # Simulate what the UI would send
    response = requests.post(
        'http://localhost:8000/api/v1/auth/signup',
        json=user_data,
        headers={
            'Content-Type': 'application/json',
            'Origin': 'http://localhost:5173'  # Simulate browser request
        }
    )
    
    if response.status_code == 200:
        print("✅ API client integration works")
        
        # Check CORS headers
        if 'access-control-allow-origin' in response.headers:
            print("✅ CORS headers present")
        else:
            print("❌ CORS headers missing")
            return False
        
        return True
    else:
        print(f"❌ API client integration failed: {response.text}")
        return False

def main():
    """Run all signup tests."""
    print("AutoML Framework Signup Integration Tests")
    print("=" * 50)
    
    try:
        # Test API health first
        health_response = requests.get('http://localhost:8000/health')
        if health_response.status_code != 200:
            print("❌ API server is not running or unhealthy")
            print("Please start the server with: python -m automl_framework.api.main")
            return
        
        print("✅ API server is healthy")
        
        # Run tests
        tests = [
            test_signup_success,
            test_signup_validation,
            test_api_client_integration
        ]
        
        passed = 0
        for test in tests:
            if test():
                passed += 1
        
        print(f"\nTest Results: {passed}/{len(tests)} tests passed")
        
        if passed == len(tests):
            print("🎉 All signup tests passed!")
        else:
            print("⚠️  Some tests failed")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("Please start the server with: python -m automl_framework.api.main")
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    main()