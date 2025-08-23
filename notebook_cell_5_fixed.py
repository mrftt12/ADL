# CORRECTED CELL 5 FOR THE NOTEBOOK
# Replace your cell 5 with this code:

def create_experiment(name: str, dataset_file_path: str, task_type: str, data_type: str, 
                     target_column: str, config: dict):
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
        "Authorization": session.headers.get('Authorization')
    }
    
    response = requests.post(create_url, json=experiment_data, headers=headers)
    return response

# Check if we have both dataset_id AND dataset_file_path from the upload
if 'dataset_id' in locals() and dataset_id and 'dataset_file_path' in locals() and dataset_file_path:
    # Configure the time series forecasting experiment
    experiment_config = {
        "forecast_horizon": 30,  # Predict 30 days ahead
        "lookback_window": 60,   # Use 60 days of history
        "validation_split": 0.2,
        "max_trials": 3,         # Limit trials for demo
        "max_epochs": 10,        # Limit epochs for demo
        "features": {
            "date_column": "date",
            "value_column": "value",
            "external_features": ["temperature", "day_of_week", "month", "is_weekend"]
        },
        "model_types": ["lstm"],  # Just LSTM for simplicity
        "optimization_metric": "mae",  # Mean Absolute Error
        "early_stopping": {
            "patience": 5,
            "min_delta": 0.001
        }
    }
    
    print("🚀 Creating time series forecasting experiment...")
    print(f"Using dataset file path: {dataset_file_path}")  # Show the actual path being used
    
    experiment_response = create_experiment(
        name="Time Series Forecasting Demo",
        dataset_file_path=dataset_file_path,  # KEY FIX: Use the actual file path from upload
        task_type="time_series_forecasting",
        data_type="time_series",
        target_column="value",
        config=experiment_config
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
        print(f"\nExperiment Configuration:")
        print(json.dumps(experiment_config, indent=2))
    else:
        print(f"❌ Experiment creation failed: {experiment_response.status_code}")
        print(f"Error: {experiment_response.text}")
        experiment_id = None
else:
    print("⚠️ Skipping experiment creation due to dataset upload failure")
    print("Make sure both dataset_id and dataset_file_path are available from the upload step")
    experiment_id = None