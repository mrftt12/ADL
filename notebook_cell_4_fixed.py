# CORRECTED CELL 4 FOR THE NOTEBOOK
# Replace your cell 4 with this code:

# Save dataset to CSV for upload
csv_filename = "time_series_demo_data.csv"
ts_data.to_csv(csv_filename, index=False)
print(f"💾 Saved dataset to {csv_filename}")

def upload_dataset(file_path: str, name: str, description: str):
    """Upload dataset to the AutoML Framework."""
    upload_url = f"{API_BASE_URL}/api/v1/datasets/upload"
    
    # Prepare files and data for upload
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/csv')}
        data = {
            'name': name,
            'description': description
        }
        
        # Create headers with only Authorization, let requests handle Content-Type for multipart
        headers = {'Authorization': session.headers.get('Authorization')}
        
        response = requests.post(upload_url, files=files, data=data, headers=headers)
    
    return response

# Upload the dataset
print("📤 Uploading dataset to AutoML Framework...")
upload_response = upload_dataset(
    file_path=csv_filename,
    name="Time Series Demo Dataset",
    description="Synthetic time series data with trend, seasonality, and external features for forecasting demo"
)

print(f"Upload response status: {upload_response.status_code}")
print(f"Upload response headers: {dict(upload_response.headers)}")

if upload_response.status_code == 200:
    upload_data = upload_response.json()
    dataset_id = upload_data['dataset_id']
    dataset_file_path = upload_data['file_path']  # KEY FIX: Capture the actual file path
    print(f"✅ Dataset uploaded successfully!")
    print(f"Dataset ID: {dataset_id}")
    print(f"File Path: {dataset_file_path}")  # This is what we need for the experiment
    print(f"Filename: {upload_data['filename']}")
    print(f"Size: {upload_data['size_bytes']} bytes")
    print(f"Metadata: {json.dumps(upload_data['metadata'], indent=2)}")
else:
    print(f"❌ Dataset upload failed: {upload_response.status_code}")
    print(f"Error: {upload_response.text}")
    dataset_id = None
    dataset_file_path = None  # Make sure this is also set to None on failure