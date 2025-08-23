#!/usr/bin/env python3
"""
AutoML Framework API Demo - Fixed Version
This script demonstrates the complete workflow with proper dataset path handling.
"""

import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta

# Configuration
API_BASE_URL = "http://localhost:8000"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "secret"

def login(username: str, password: str) -> str:
    """Login to the API and return access token."""
    login_url = f"{API_BASE_URL}/api/v1/auth/login"
    
    login_data = {
        "username": username,
        "password": password
    }
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    response = requests.post(login_url, data=login_data, headers=headers)
    
    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data["access_token"]
        print(f"✅ Successfully logged in as {username}")
        return access_token
    else:
        print(f"❌ Login failed: {response.status_code}")
        print(f"Error: {response.text}")
        return None

def generate_time_series_data(n_points=800, start_date='2020-01-01'):
    """Generate synthetic time series data for demonstration."""
    
    # Create date range
    dates = pd.date_range(start=start_date, periods=n_points, freq='D')
    
    # Generate base trend
    trend = np.linspace(100, 200, n_points)
    
    # Add seasonal component (yearly cycle)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n_points) / 365.25)
    
    # Add weekly pattern
    weekly = 10 * np.sin(2 * np.pi * np.arange(n_points) / 7)
    
    # Add noise
    noise = np.random.normal(0, 5, n_points)
    
    # Combine components
    values = trend + seasonal + weekly + noise
    
    # Add some external features
    temperature = 20 + 15 * np.sin(2 * np.pi * np.arange(n_points) / 365.25) + np.random.normal(0, 2, n_points)
    day_of_week = np.arange(n_points) % 7
    month = dates.month
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'temperature': temperature,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': (day_of_week >= 5).astype(int)
    })
    
    return df

def upload_dataset(file_path: str, name: str, description: str, token: str):
    """Upload dataset to the AutoML Framework."""
    upload_url = f"{API_BASE_URL}/api/v1/datasets/upload"
    
    # Prepare files and data for upload
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/csv')}
        data = {
            'name': name,
            'description': description
        }
        
        headers = {'Authorization': f'Bearer {token}'}
        
        response = requests.post(upload_url, files=files, data=data, headers=headers)
    
    return response

def create_experiment(name: str, dataset_file_path: str, task_type: str, data_type: str, 
                     target_column: str, config: dict, token: str):
    """Create a new experiment."""
    create_url = f"{API_BASE_URL}/api/v1/experiments"
    
    experiment_data = {
        "name": name,
        "dataset_path": dataset_file_path,  # Use the actual file path from upload response
        "task_type": task_type,
        "data_type": data_type,
        "target_column": target_column,
        "config": config
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    response = requests.post(create_url, json=experiment_data, headers=headers)
    return response

def main():
    print("🚀 AutoML Framework API Demo - Fixed Version")
    print("=" * 60)
    
    # Step 1: Login
    print("\n1. 🔐 Authenticating...")
    access_token = login(ADMIN_USERNAME, ADMIN_PASSWORD)
    if not access_token:
        print("❌ Authentication failed. Exiting.")
        return
    
    # Step 2: Generate data
    print("\n2. 📊 Generating synthetic time series data...")
    ts_data = generate_time_series_data(n_points=800)
    print(f"✅ Generated dataset with {len(ts_data)} data points")
    print(f"Date range: {ts_data['date'].min()} to {ts_data['date'].max()}")
    
    # Step 3: Save and upload dataset
    print("\n3. 📤 Uploading dataset...")
    csv_filename = "time_series_demo_data.csv"
    ts_data.to_csv(csv_filename, index=False)
    print(f"💾 Saved dataset to {csv_filename}")
    
    upload_response = upload_dataset(
        file_path=csv_filename,
        name="Time Series Demo Dataset",
        description="Synthetic time series data with trend, seasonality, and external features for forecasting demo",
        token=access_token
    )
    
    print(f"Upload response status: {upload_response.status_code}")
    
    if upload_response.status_code == 200:
        upload_data = upload_response.json()
        dataset_id = upload_data['dataset_id']
        dataset_file_path = upload_data['file_path']  # THIS IS THE KEY FIX!
        print(f"✅ Dataset uploaded successfully!")
        print(f"Dataset ID: {dataset_id}")
        print(f"File Path: {dataset_file_path}")  # This is what we need for the experiment
        print(f"Filename: {upload_data['filename']}")
        print(f"Size: {upload_data['size_bytes']} bytes")
    else:
        print(f"❌ Dataset upload failed: {upload_response.status_code}")
        print(f"Error: {upload_response.text}")
        return
    
    # Step 4: Create experiment
    print("\n4. 🧪 Creating time series forecasting experiment...")
    experiment_config = {
        "forecast_horizon": 30,
        "lookback_window": 60,
        "validation_split": 0.2,
        "max_trials": 3,  # Small number for demo
        "max_epochs": 10,  # Small number for demo
        "features": {
            "date_column": "date",
            "value_column": "value",
            "external_features": ["temperature", "day_of_week", "month", "is_weekend"]
        },
        "model_types": ["lstm"],
        "optimization_metric": "mae",
        "early_stopping": {
            "patience": 5,
            "min_delta": 0.001
        }
    }
    
    experiment_response = create_experiment(
        name="Time Series Forecasting Demo",
        dataset_file_path=dataset_file_path,  # Use the actual file path from upload
        task_type="time_series_forecasting",
        data_type="time_series",
        target_column="value",
        config=experiment_config,
        token=access_token
    )
    
    print(f"Experiment response status: {experiment_response.status_code}")
    
    if experiment_response.status_code == 200:
        experiment_data = experiment_response.json()
        experiment_id = experiment_data['id']
        print(f"✅ Experiment created successfully!")
        print(f"Experiment ID: {experiment_id}")
        print(f"Name: {experiment_data['name']}")
        print(f"Status: {experiment_data['status']}")
        print(f"Created at: {experiment_data['created_at']}")
    else:
        print(f"❌ Experiment creation failed: {experiment_response.status_code}")
        print(f"Error: {experiment_response.text}")
        return
    
    # Step 5: List experiments
    print("\n5. 📋 Listing all experiments...")
    list_url = f"{API_BASE_URL}/api/v1/experiments"
    headers = {"Authorization": f"Bearer {access_token}"}
    list_response = requests.get(list_url, headers=headers)
    
    if list_response.status_code == 200:
        experiments_data = list_response.json()
        experiments = experiments_data.get('experiments', [])
        total = experiments_data.get('total', 0)
        
        print(f"✅ Found {total} experiment(s)")
        
        if experiments:
            print(f"\n📊 Experiments Summary:")
            print("-" * 80)
            print(f"{'ID':<20} {'Name':<30} {'Status':<12} {'Created':<20}")
            print("-" * 80)
            
            for exp in experiments:
                exp_id = exp['id'][:18] + '...' if len(exp['id']) > 20 else exp['id']
                exp_name = exp['name'][:28] + '...' if len(exp['name']) > 30 else exp['name']
                exp_status = exp['status']
                exp_created = exp['created_at'][:19] if exp['created_at'] else 'N/A'
                
                print(f"{exp_id:<20} {exp_name:<30} {exp_status:<12} {exp_created:<20}")
            
            print("-" * 80)
    else:
        print(f"❌ Failed to list experiments: {list_response.status_code}")
        print(f"Error: {list_response.text}")
    
    print("\n🎉 Demo completed successfully!")
    print("\n" + "=" * 60)
    print("KEY FIX APPLIED:")
    print("=" * 60)
    print("✅ Used actual file_path from upload response instead of constructing path")
    print(f"✅ Correct path format: {dataset_file_path}")
    print("✅ This ensures the experiment can find the uploaded dataset file")
    print("=" * 60)

if __name__ == "__main__":
    main()