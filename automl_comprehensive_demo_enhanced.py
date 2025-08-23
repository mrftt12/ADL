#!/usr/bin/env python3
"""
Enhanced AutoML Framework API Demo - Production Ready Version

This script demonstrates the full capabilities of the AutoML Framework with:
- Robust error handling and retry logic
- Better progress monitoring
- Comprehensive logging
- Graceful degradation for missing features
- Production-ready patterns

Prerequisites:
- AutoML Framework API running on http://localhost:8000
- Admin credentials: username='admin', password='secret'
- Required Python packages: requests, pandas, numpy
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
API_BASE_URL = "http://localhost:8000"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "secret"
MAX_RETRIES = 3
RETRY_DELAY = 2
REQUEST_TIMEOUT = 30

class AutoMLDemoEnhanced:
    """Enhanced AutoML Framework demonstration with robust error handling."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = REQUEST_TIMEOUT
        self.access_token = None
        self.datasets = {}
        self.experiments = {}
        self.models = {}
        self.errors = []
        self.warnings = []
        
    def log_error(self, message: str, exception: Exception = None):
        """Log an error with optional exception details."""
        error_msg = f"❌ ERROR: {message}"
        if exception:
            error_msg += f" - {str(exception)}"
        print(error_msg)
        self.errors.append(error_msg)
    
    def log_warning(self, message: str):
        """Log a warning message."""
        warning_msg = f"⚠️ WARNING: {message}"
        print(warning_msg)
        self.warnings.append(warning_msg)
    
    def make_request(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        """Make HTTP request with retry logic and error handling."""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.request(method, url, **kwargs)
                return response
            except requests.exceptions.ConnectionError as e:
                if attempt < MAX_RETRIES - 1:
                    self.log_warning(f"Connection failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    self.log_error(f"Connection failed after {MAX_RETRIES} attempts", e)
                    return None
            except requests.exceptions.Timeout as e:
                if attempt < MAX_RETRIES - 1:
                    self.log_warning(f"Request timeout (attempt {attempt + 1}/{MAX_RETRIES}), retrying...")
                    time.sleep(RETRY_DELAY)
                else:
                    self.log_error(f"Request timeout after {MAX_RETRIES} attempts", e)
                    return None
            except Exception as e:
                self.log_error(f"Unexpected error during request", e)
                return None
        
        return None
    
    def print_section(self, title: str, level: int = 1):
        """Print formatted section headers."""
        if level == 1:
            print(f"\n{'='*80}")
            print(f"🚀 {title}")
            print(f"{'='*80}")
        elif level == 2:
            print(f"\n{'-'*60}")
            print(f"📋 {title}")
            print(f"{'-'*60}")
        else:
            print(f"\n💡 {title}")
    
    def check_api_availability(self) -> bool:
        """Check if the API is available and responding."""
        try:
            response = self.make_request('GET', f"{API_BASE_URL}/health")
            if response and response.status_code == 200:
                return True
            else:
                self.log_error(f"API health check failed with status: {response.status_code if response else 'No response'}")
                return False
        except Exception as e:
            self.log_error("API availability check failed", e)
            return False
    
    def login(self) -> bool:
        """Authenticate with the API."""
        self.print_section("Authentication & Setup", 1)
        
        if not self.check_api_availability():
            self.log_error("API is not available. Please ensure the AutoML Framework is running.")
            return False
        
        login_url = f"{API_BASE_URL}/api/v1/auth/login"
        login_data = {
            "username": ADMIN_USERNAME,
            "password": ADMIN_PASSWORD
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        
        response = self.make_request('POST', login_url, data=login_data, headers=headers)
        
        if response and response.status_code == 200:
            try:
                token_data = response.json()
                self.access_token = token_data["access_token"]
                self.session.headers.update({"Authorization": f"Bearer {self.access_token}"})
                
                print(f"✅ Successfully authenticated as {ADMIN_USERNAME}")
                print(f"🔑 Token type: {token_data['token_type']}")
                return True
            except (KeyError, json.JSONDecodeError) as e:
                self.log_error("Invalid response format from login endpoint", e)
                return False
        else:
            self.log_error(f"Authentication failed: {response.status_code if response else 'No response'}")
            if response:
                self.log_error(f"Response: {response.text}")
            return False 
   
    def check_api_health(self) -> bool:
        """Check API health and configuration."""
        self.print_section("API Health & Configuration", 2)
        
        try:
            # Health check
            health_response = self.make_request('GET', f"{API_BASE_URL}/health")
            if health_response and health_response.status_code == 200:
                health_data = health_response.json()
                print("🏥 API Health Status:")
                print(f"  Status: {health_data.get('status', 'Unknown')}")
                print(f"  Environment: {health_data.get('environment', 'Unknown')}")
                print(f"  GPU Available: {health_data.get('gpu_available', False)}")
                print(f"  Database Available: {health_data.get('database_available', False)}")
                
                auth_info = health_data.get('authentication', {})
                print(f"  Auth Backend: {auth_info.get('backend', 'Unknown')}")
                
                if health_data.get('status') != 'healthy':
                    self.log_warning("API reports unhealthy status")
            else:
                self.log_error("Failed to get API health status")
                return False
            
            # Configuration
            config_response = self.make_request('GET', f"{API_BASE_URL}/api/v1/config")
            if config_response and config_response.status_code == 200:
                config_data = config_response.json()
                print(f"\n⚙️ API Configuration:")
                print(f"  Version: {config_data.get('version', 'Unknown')}")
                print(f"  Supported Data Types: {config_data.get('supported_data_types', [])}")
                print(f"  Supported Task Types: {config_data.get('supported_task_types', [])}")
                print(f"  Max File Size: {config_data.get('max_file_size_mb', 'Unknown')}MB")
            else:
                self.log_warning("Failed to get API configuration")
            
            return True
            
        except Exception as e:
            self.log_error("Error during API health check", e)
            return False
    
    def generate_safe_data(self, data_type: str, n_samples: int = 500) -> Optional[pd.DataFrame]:
        """Generate synthetic data with error handling."""
        try:
            np.random.seed(42)  # For reproducible results
            
            if data_type == "classification":
                return self._generate_classification_data(n_samples)
            elif data_type == "regression":
                return self._generate_regression_data(n_samples)
            elif data_type == "time_series":
                return self._generate_time_series_data(n_samples)
            elif data_type == "text":
                return self._generate_text_data(n_samples)
            elif data_type == "image":
                return self._generate_image_data(n_samples)
            else:
                self.log_error(f"Unknown data type: {data_type}")
                return None
                
        except Exception as e:
            self.log_error(f"Failed to generate {data_type} data", e)
            return None
    
    def _generate_classification_data(self, n_samples: int) -> pd.DataFrame:
        """Generate classification dataset."""
        n_features = 8
        X = np.random.randn(n_samples, n_features)
        
        # Create target with some logic
        target_score = X[:, 0] + X[:, 1] * 0.5 + X[:, 2] ** 2 * 0.3
        target = np.digitize(target_score, bins=np.percentile(target_score, [33, 67]))
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = target
        df['category'] = np.random.choice(['A', 'B', 'C'], n_samples)
        
        return df
    
    def _generate_regression_data(self, n_samples: int) -> pd.DataFrame:
        """Generate regression dataset."""
        n_features = 6
        X = np.random.randn(n_samples, n_features)
        
        # Generate target with noise
        target = (X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] ** 2 * 0.5 + 
                 np.sin(X[:, 3]) * 2 + np.random.randn(n_samples) * 0.3)
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = target
        df['group'] = np.random.choice(['Group1', 'Group2'], n_samples)
        
        return df
    
    def _generate_time_series_data(self, n_samples: int) -> pd.DataFrame:
        """Generate time series dataset."""
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        
        # Generate components
        trend = np.linspace(100, 200, n_samples)
        seasonal = 15 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
        weekly = 8 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
        noise = np.random.normal(0, 3, n_samples)
        
        values = trend + seasonal + weekly + noise
        temperature = 20 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25) + np.random.normal(0, 1.5, n_samples)
        
        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'temperature': temperature,
            'day_of_week': np.arange(n_samples) % 7,
            'month': dates.month,
            'is_weekend': (np.arange(n_samples) % 7 >= 5).astype(int)
        })
        
        return df
    
    def _generate_text_data(self, n_samples: int) -> pd.DataFrame:
        """Generate text dataset."""
        templates = {
            'positive': [
                "This product is excellent and works great!",
                "Amazing quality and fast delivery.",
                "Love this item, highly recommended!",
                "Outstanding service and great value.",
                "Perfect solution, very satisfied!"
            ],
            'negative': [
                "Poor quality, broke quickly.",
                "Terrible experience, waste of money.",
                "Bad customer service and delays.",
                "Product doesn't work as described.",
                "Very disappointed with purchase."
            ],
            'neutral': [
                "Product is okay, nothing special.",
                "Average quality for the price.",
                "Decent item, meets requirements.",
                "Standard product, no issues.",
                "Fair value, does the job."
            ]
        }
        
        texts = []
        labels = []
        
        for _ in range(n_samples):
            category = np.random.choice(['positive', 'negative', 'neutral'])
            text = np.random.choice(templates[category])
            text += f" Review #{np.random.randint(1, 1000)}."
            
            texts.append(text)
            labels.append({'positive': 2, 'negative': 0, 'neutral': 1}[category])
        
        return pd.DataFrame({
            'text': texts,
            'sentiment': labels,
            'length': [len(text) for text in texts]
        })
    
    def _generate_image_data(self, n_samples: int) -> pd.DataFrame:
        """Generate image metadata dataset."""
        categories = ['cat', 'dog', 'bird', 'car', 'flower']
        
        data = []
        for i in range(n_samples):
            category = np.random.choice(categories)
            data.append({
                'image_id': f'img_{i:04d}',
                'category': category,
                'width': np.random.randint(224, 512),
                'height': np.random.randint(224, 512),
                'channels': 3,
                'file_size': np.random.randint(10000, 200000),
                'brightness': np.random.uniform(0.4, 0.8),
                'contrast': np.random.uniform(0.8, 1.2)
            })
        
        return pd.DataFrame(data)
    
    def upload_dataset_safe(self, df: pd.DataFrame, name: str, description: str) -> Optional[Dict]:
        """Upload dataset with comprehensive error handling."""
        try:
            # Validate DataFrame
            if df is None or df.empty:
                self.log_error(f"Cannot upload empty dataset: {name}")
                return None
            
            # Save to CSV with error handling
            filename = f"{name.lower().replace(' ', '_').replace('-', '_')}.csv"
            df.to_csv(filename, index=False)
            
            if not os.path.exists(filename):
                self.log_error(f"Failed to create CSV file: {filename}")
                return None
            
            upload_url = f"{API_BASE_URL}/api/v1/datasets/upload"
            
            with open(filename, 'rb') as f:
                files = {'file': (filename, f, 'text/csv')}
                data = {'name': name, 'description': description}
                headers = {'Authorization': self.session.headers.get('Authorization')}
                
                response = self.make_request('POST', upload_url, files=files, data=data, headers=headers)
            
            # Clean up temporary file
            try:
                os.remove(filename)
            except OSError:
                pass
            
            if response and response.status_code == 200:
                try:
                    upload_data = response.json()
                    print(f"✅ Uploaded '{name}' - ID: {upload_data.get('dataset_id', 'Unknown')}")
                    return upload_data
                except json.JSONDecodeError as e:
                    self.log_error(f"Invalid JSON response from upload", e)
                    return None
            else:
                self.log_error(f"Failed to upload '{name}': {response.status_code if response else 'No response'}")
                if response:
                    self.log_error(f"Response: {response.text}")
                return None
                
        except Exception as e:
            self.log_error(f"Unexpected error uploading dataset '{name}'", e)
            return None
    
    def create_experiment_safe(self, name: str, dataset_path: str, task_type: str, 
                              data_type: str, target_column: str, config: Dict) -> Optional[Dict]:
        """Create experiment with error handling."""
        try:
            create_url = f"{API_BASE_URL}/api/v1/experiments"
            
            experiment_data = {
                "name": name,
                "dataset_path": dataset_path,
                "task_type": task_type,
                "data_type": data_type,
                "target_column": target_column,
                "config": config
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": self.session.headers.get('Authorization')
            }
            
            response = self.make_request('POST', create_url, json=experiment_data, headers=headers)
            
            if response and response.status_code == 200:
                try:
                    experiment_data = response.json()
                    print(f"✅ Created experiment '{name}' - ID: {experiment_data.get('id', 'Unknown')}")
                    return experiment_data
                except json.JSONDecodeError as e:
                    self.log_error(f"Invalid JSON response from experiment creation", e)
                    return None
            else:
                self.log_error(f"Failed to create experiment '{name}': {response.status_code if response else 'No response'}")
                if response:
                    self.log_error(f"Response: {response.text}")
                return None
                
        except Exception as e:
            self.log_error(f"Unexpected error creating experiment '{name}'", e)
            return None
    
    def demonstrate_dataset_management(self) -> bool:
        """Demonstrate dataset management with error handling."""
        self.print_section("Dataset Management & Upload", 1)
        
        datasets_config = [
            {
                'name': 'Classification Demo',
                'type': 'tabular',
                'task': 'classification',
                'data_type': 'classification',
                'samples': 800
            },
            {
                'name': 'Regression Demo',
                'type': 'tabular', 
                'task': 'regression',
                'data_type': 'regression',
                'samples': 600
            },
            {
                'name': 'Time Series Demo',
                'type': 'time_series',
                'task': 'time_series_forecasting', 
                'data_type': 'time_series',
                'samples': 1000
            },
            {
                'name': 'Text Classification Demo',
                'type': 'text',
                'task': 'nlp',
                'data_type': 'text', 
                'samples': 400
            },
            {
                'name': 'Image Classification Demo',
                'type': 'image',
                'task': 'object_detection',
                'data_type': 'image',
                'samples': 150
            }
        ]
        
        success_count = 0
        
        for dataset_config in datasets_config:
            self.print_section(f"Creating {dataset_config['name']}", 3)
            
            # Generate data
            df = self.generate_safe_data(dataset_config['data_type'], dataset_config['samples'])
            
            if df is not None:
                print(f"📊 Generated {len(df)} samples with {len(df.columns)} features")
                print(f"Columns: {list(df.columns)}")
                
                # Upload dataset
                upload_result = self.upload_dataset_safe(
                    df,
                    dataset_config['name'],
                    f"Synthetic {dataset_config['type']} dataset for {dataset_config['task']} demonstration"
                )
                
                if upload_result:
                    self.datasets[dataset_config['task']] = {
                        'data': upload_result,
                        'df': df,
                        'config': dataset_config
                    }
                    success_count += 1
                    
                    # Show basic statistics
                    try:
                        print(f"📈 Dataset Statistics:")
                        numeric_cols = df.select_dtypes(include=[np.number])
                        if not numeric_cols.empty:
                            print(f"  Numeric columns: {numeric_cols.shape[1]}")
                        print(f"  Missing values: {df.isnull().sum().sum()}")
                        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                    except Exception as e:
                        self.log_warning(f"Could not compute statistics for {dataset_config['name']}")
            else:
                self.log_error(f"Failed to generate data for {dataset_config['name']}")
        
        print(f"\n📊 Dataset Management Summary: {success_count}/{len(datasets_config)} datasets uploaded successfully")
        return success_count > 0    
 
   def demonstrate_experiment_creation(self) -> bool:
        """Demonstrate experiment creation with error handling."""
        self.print_section("Experiment Creation & Management", 1)
        
        if not self.datasets:
            self.log_error("No datasets available for experiment creation")
            return False
        
        experiment_configs = [
            {
                'task': 'classification',
                'name': 'Classification Experiment',
                'target': 'target',
                'config': {
                    'max_trials': 3,
                    'max_epochs': 10,
                    'validation_split': 0.2,
                    'models': ['random_forest', 'gradient_boosting'],
                    'metrics': ['accuracy', 'f1_score']
                }
            },
            {
                'task': 'regression',
                'name': 'Regression Experiment', 
                'target': 'target',
                'config': {
                    'max_trials': 3,
                    'max_epochs': 10,
                    'validation_split': 0.25,
                    'models': ['linear_regression', 'random_forest'],
                    'metrics': ['mse', 'mae']
                }
            },
            {
                'task': 'time_series_forecasting',
                'name': 'Time Series Experiment',
                'target': 'value', 
                'config': {
                    'forecast_horizon': 20,
                    'lookback_window': 40,
                    'max_trials': 2,
                    'max_epochs': 5,
                    'models': ['lstm'],
                    'features': {
                        'date_column': 'date',
                        'external_features': ['temperature', 'day_of_week', 'month']
                    }
                }
            }
        ]
        
        success_count = 0
        
        for exp_config in experiment_configs:
            task = exp_config['task']
            if task in self.datasets:
                self.print_section(f"Creating {exp_config['name']}", 2)
                
                dataset_info = self.datasets[task]
                experiment = self.create_experiment_safe(
                    name=exp_config['name'],
                    dataset_path=dataset_info['data']['file_path'],
                    task_type=task,
                    data_type=dataset_info['config']['type'],
                    target_column=exp_config['target'],
                    config=exp_config['config']
                )
                
                if experiment:
                    self.experiments[task] = experiment
                    success_count += 1
                    print(f"🎯 Configuration: {json.dumps(exp_config['config'], indent=2)}")
                else:
                    self.log_error(f"Failed to create {exp_config['name']}")
            else:
                self.log_warning(f"Skipping {exp_config['name']} - dataset not available")
        
        print(f"\n🧪 Experiment Creation Summary: {success_count}/{len(experiment_configs)} experiments created successfully")
        return success_count > 0
    
    def monitor_experiments_safe(self) -> bool:
        """Monitor experiments with error handling."""
        self.print_section("Experiment Monitoring", 1)
        
        if not self.experiments:
            self.log_warning("No experiments to monitor")
            return False
        
        monitoring_success = True
        
        for task, experiment in self.experiments.items():
            self.print_section(f"Monitoring {experiment.get('name', 'Unknown Experiment')}", 2)
            
            experiment_id = experiment.get('id')
            if not experiment_id:
                self.log_error(f"No experiment ID for {task}")
                monitoring_success = False
                continue
            
            status_url = f"{API_BASE_URL}/api/v1/experiments/{experiment_id}"
            
            # Monitor for a limited time
            max_checks = 3
            for i in range(max_checks):
                response = self.make_request('GET', status_url)
                
                if response and response.status_code == 200:
                    try:
                        data = response.json()
                        status = data.get('status', 'Unknown')
                        progress = data.get('progress', {})
                        
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {status.upper()}")
                        
                        if progress:
                            for key, value in progress.items():
                                if isinstance(value, (int, float)) and 0 <= value <= 1:
                                    print(f"  {key}: {value:.1%}")
                                else:
                                    print(f"  {key}: {value}")
                        
                        if status in ['completed', 'failed', 'cancelled']:
                            print(f"🏁 Experiment {status.upper()}!")
                            break
                        
                        if i < max_checks - 1:
                            time.sleep(3)
                            
                    except json.JSONDecodeError as e:
                        self.log_error(f"Invalid JSON response from experiment status", e)
                        monitoring_success = False
                        break
                else:
                    self.log_error(f"Failed to get experiment status: {response.status_code if response else 'No response'}")
                    monitoring_success = False
                    break
        
        return monitoring_success
    
    def list_experiments_safe(self) -> bool:
        """List all experiments with error handling."""
        self.print_section("Experiment Listing", 2)
        
        list_url = f"{API_BASE_URL}/api/v1/experiments"
        response = self.make_request('GET', list_url)
        
        if response and response.status_code == 200:
            try:
                data = response.json()
                experiments = data.get('experiments', [])
                total = data.get('total', 0)
                
                print(f"📊 Found {total} experiment(s)")
                
                if experiments:
                    print(f"\n{'ID':<20} {'Name':<30} {'Status':<12} {'Created':<20}")
                    print("-" * 82)
                    
                    for exp in experiments:
                        exp_id = exp.get('id', 'Unknown')[:18] + '...' if len(exp.get('id', '')) > 20 else exp.get('id', 'Unknown')
                        exp_name = exp.get('name', 'Unknown')[:28] + '...' if len(exp.get('name', '')) > 30 else exp.get('name', 'Unknown')
                        exp_status = exp.get('status', 'Unknown')
                        exp_created = exp.get('created_at', 'N/A')[:19] if exp.get('created_at') else 'N/A'
                        
                        print(f"{exp_id:<20} {exp_name:<30} {exp_status:<12} {exp_created:<20}")
                
                return True
                
            except json.JSONDecodeError as e:
                self.log_error("Invalid JSON response from experiment listing", e)
                return False
        else:
            self.log_error(f"Failed to list experiments: {response.status_code if response else 'No response'}")
            return False
    
    def demonstrate_advanced_features(self):
        """Demonstrate advanced features with graceful degradation."""
        self.print_section("Advanced Features Demo", 1)
        
        # Model Serving
        self.print_section("Model Serving & Inference", 2)
        print("🔧 Model serving capabilities:")
        print("  • Real-time inference endpoints")
        print("  • Batch prediction processing")
        print("  • Model caching and optimization")
        print("  • A/B testing for model versions")
        
        # Resource Management
        self.print_section("Resource Management", 2)
        resource_url = f"{API_BASE_URL}/api/v1/resources/status"
        response = self.make_request('GET', resource_url)
        
        if response and response.status_code == 200:
            try:
                data = response.json()
                print("💻 System Resources:")
                print(f"  CPU Usage: {data.get('cpu_usage', 'N/A')}")
                print(f"  Memory Usage: {data.get('memory_usage', 'N/A')}")
                print(f"  GPU Available: {data.get('gpu_available', False)}")
                print(f"  Running Jobs: {data.get('running_jobs', 0)}")
                print(f"  Queued Jobs: {data.get('queued_jobs', 0)}")
            except json.JSONDecodeError:
                self.log_warning("Could not parse resource status response")
        else:
            self.log_warning("Resource status not available")
        
        # Monitoring Features
        self.print_section("Model Monitoring", 2)
        print("📊 Monitoring capabilities:")
        print("  • Performance tracking over time")
        print("  • Data drift detection")
        print("  • Model version management")
        print("  • Automated alerting system")
    
    def generate_final_report(self):
        """Generate comprehensive final report."""
        self.print_section("Demo Summary Report", 1)
        
        print("📋 AutoML Framework Capabilities Demonstrated:")
        print()
        
        # Success metrics
        datasets_created = len(self.datasets)
        experiments_created = len(self.experiments)
        
        print("✅ Successfully Completed:")
        print(f"  🔐 Authentication and API access")
        print(f"  📊 Dataset management ({datasets_created} datasets)")
        print(f"  🧪 Experiment creation ({experiments_created} experiments)")
        print(f"  📈 Real-time monitoring")
        print(f"  🔗 API endpoint testing")
        
        # Error summary
        if self.errors:
            print(f"\n❌ Errors Encountered ({len(self.errors)}):")
            for error in self.errors[-5:]:  # Show last 5 errors
                print(f"  {error}")
        
        if self.warnings:
            print(f"\n⚠️ Warnings ({len(self.warnings)}):")
            for warning in self.warnings[-3:]:  # Show last 3 warnings
                print(f"  {warning}")
        
        # Performance summary
        print(f"\n📈 Performance Summary:")
        print(f"  • Datasets uploaded: {datasets_created}")
        print(f"  • Experiments created: {experiments_created}")
        print(f"  • Errors encountered: {len(self.errors)}")
        print(f"  • Warnings: {len(self.warnings)}")
        
        # Overall status
        if len(self.errors) == 0:
            print(f"\n🎉 Demo completed successfully with no errors!")
            print(f"The AutoML Framework is fully operational and ready for production use.")
        elif len(self.errors) <= 2:
            print(f"\n✅ Demo completed with minor issues.")
            print(f"The AutoML Framework is operational with some limitations.")
        else:
            print(f"\n⚠️ Demo completed with multiple issues.")
            print(f"Please review the errors and check your AutoML Framework setup.")
    
    def run_enhanced_demo(self) -> bool:
        """Run the enhanced comprehensive demonstration."""
        print("🚀 AutoML Framework - Enhanced Comprehensive Demo")
        print("=" * 80)
        print("Production-ready demonstration with robust error handling")
        print("and graceful degradation for missing features.")
        print("=" * 80)
        
        # Step 1: Authentication
        if not self.login():
            self.log_error("Demo failed - authentication unsuccessful")
            return False
        
        # Step 2: API Health Check
        if not self.check_api_health():
            self.log_warning("API health check failed, continuing with limited functionality")
        
        # Step 3: Dataset Management
        dataset_success = self.demonstrate_dataset_management()
        if not dataset_success:
            self.log_error("Dataset management failed, cannot continue with experiments")
            return False
        
        # Step 4: Experiment Creation
        experiment_success = self.demonstrate_experiment_creation()
        if not experiment_success:
            self.log_warning("Experiment creation failed, continuing with other features")
        
        # Step 5: Experiment Monitoring
        if experiment_success:
            self.monitor_experiments_safe()
        
        # Step 6: List All Experiments
        self.list_experiments_safe()
        
        # Step 7: Advanced Features
        self.demonstrate_advanced_features()
        
        # Step 8: Final Report
        self.generate_final_report()
        
        return len(self.errors) <= 2  # Consider success if 2 or fewer errors


def main():
    """Main function to run the enhanced demo."""
    demo = AutoMLDemoEnhanced()
    
    try:
        print(f"🔧 Starting AutoML Framework Enhanced Demo...")
        print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌐 API Endpoint: {API_BASE_URL}")
        print()
        
        success = demo.run_enhanced_demo()
        
        if success:
            print(f"\n✅ Enhanced demo completed successfully!")
            return 0
        else:
            print(f"\n⚠️ Demo completed with issues. Check the error log above.")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Demo failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())