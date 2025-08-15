#!/usr/bin/env python3
"""
Test the complete project creation flow including authentication,
dataset selection, and experiment creation.
"""

import requests
import json

def test_complete_flow():
    """Test the complete project creation flow."""
    print("Testing Complete Project Creation Flow")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Step 1: Test authentication
    print("1. Testing authentication...")
    login_response = requests.post(
        f"{base_url}/api/v1/auth/login",
        data={'username': 'demo_user', 'password': 'secret'}
    )
    
    if login_response.status_code != 200:
        print(f"❌ Login failed: {login_response.text}")
        return False
    
    token_data = login_response.json()
    token = token_data['access_token']
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    print("✅ Authentication successful")
    
    # Step 2: Test user info
    print("2. Testing user info...")
    user_response = requests.get(f"{base_url}/api/v1/auth/me", headers=headers)
    if user_response.status_code == 200:
        user_data = user_response.json()
        print(f"✅ User info: {user_data['username']} (ID: {user_data['id']})")
    else:
        print(f"❌ Failed to get user info: {user_response.text}")
        return False
    
    # Step 3: Test dataset listing
    print("3. Testing dataset listing...")
    datasets_response = requests.get(f"{base_url}/api/v1/datasets", headers=headers)
    if datasets_response.status_code == 200:
        datasets_data = datasets_response.json()
        datasets = datasets_data['datasets']
        print(f"✅ Found {len(datasets)} datasets")
        
        if datasets:
            for dataset in datasets[:3]:  # Show first 3
                size_mb = dataset['size_bytes'] / 1024 / 1024
                print(f"   - {dataset['filename']} ({size_mb:.2f} MB)")
        else:
            print("   No datasets found")
    else:
        print(f"❌ Failed to list datasets: {datasets_response.text}")
        return False
    
    # Step 4: Test experiment creation
    print("4. Testing experiment creation...")
    
    if not datasets:
        print("   No datasets available, creating with test dataset...")
        dataset_path = "data/uploads/test_data.csv"
    else:
        # Use the first available dataset
        selected_dataset = datasets[0]
        dataset_path = f"data/uploads/{selected_dataset['filename']}"
        print(f"   Using dataset: {selected_dataset['filename']}")
    
    experiment_data = {
        'name': 'Test Project from API',
        'dataset_path': dataset_path,
        'task_type': 'classification',
        'data_type': 'tabular',
        'target_column': 'target',
        'config': {
            'description': 'Test project created via API',
            'user_id': user_data['id']
        }
    }
    
    experiment_response = requests.post(
        f"{base_url}/api/v1/experiments",
        json=experiment_data,
        headers=headers
    )
    
    if experiment_response.status_code == 200:
        experiment = experiment_response.json()
        print(f"✅ Experiment created: {experiment['name']} (ID: {experiment['id']})")
        print(f"   Status: {experiment['status']}")
        
        # Step 5: Test getting the created experiment
        print("5. Testing experiment retrieval...")
        get_response = requests.get(
            f"{base_url}/api/v1/experiments/{experiment['id']}",
            headers=headers
        )
        
        if get_response.status_code == 200:
            retrieved_experiment = get_response.json()
            print(f"✅ Retrieved experiment: {retrieved_experiment['name']}")
        else:
            print(f"❌ Failed to retrieve experiment: {get_response.text}")
            return False
        
        # Step 6: Test listing experiments
        print("6. Testing experiment listing...")
        list_response = requests.get(f"{base_url}/api/v1/experiments", headers=headers)
        
        if list_response.status_code == 200:
            experiments_data = list_response.json()
            experiments = experiments_data['experiments']
            print(f"✅ Found {len(experiments)} experiments")
            
            for exp in experiments:
                print(f"   - {exp['name']} ({exp['status']})")
        else:
            print(f"❌ Failed to list experiments: {list_response.text}")
            return False
        
        return True
    else:
        print(f"❌ Failed to create experiment: {experiment_response.text}")
        return False

def test_authentication_errors():
    """Test authentication error handling."""
    print("\nTesting Authentication Error Handling")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Test without authentication
    print("1. Testing unauthenticated request...")
    response = requests.get(f"{base_url}/api/v1/experiments")
    if response.status_code == 401:
        print("✅ Correctly rejected unauthenticated request")
    else:
        print(f"❌ Unexpected response: {response.status_code}")
        return False
    
    # Test with invalid token
    print("2. Testing invalid token...")
    headers = {'Authorization': 'Bearer invalid_token'}
    response = requests.get(f"{base_url}/api/v1/experiments", headers=headers)
    if response.status_code == 401:
        print("✅ Correctly rejected invalid token")
    else:
        print(f"❌ Unexpected response: {response.status_code}")
        return False
    
    return True

def main():
    """Run all tests."""
    try:
        # Test API health first
        health_response = requests.get('http://localhost:8000/health')
        if health_response.status_code != 200:
            print("❌ API server is not running or unhealthy")
            print("Please start the server with: python -m automl_framework.api.main")
            return
        
        print("✅ API server is healthy")
        
        # Run tests
        success = True
        
        if not test_complete_flow():
            success = False
        
        if not test_authentication_errors():
            success = False
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 All tests passed! Project creation flow is working correctly.")
            print("\nThe issues have been resolved:")
            print("✅ Authentication is working properly")
            print("✅ Dataset selection from uploaded datasets works")
            print("✅ Experiment creation is successful")
            print("✅ Error handling is implemented")
        else:
            print("⚠️  Some tests failed")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("Please start the server with: python -m automl_framework.api.main")
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    main()