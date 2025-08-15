#!/usr/bin/env python3
"""
Test experiment WebSocket integration by creating a real experiment
and monitoring its progress through WebSocket events.
"""

import asyncio
import json
import logging
import websockets
import requests
import tempfile
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000"
WS_BASE_URL = "ws://localhost:8000"

async def create_test_dataset():
    """Create a simple test dataset for the experiment."""
    # Create a simple CSV dataset
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'feature3': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    
    df = pd.DataFrame(data)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return temp_file.name

async def authenticate_and_get_token():
    """Authenticate and get access token."""
    try:
        login_data = {
            'username': 'demo_user',
            'password': 'secret'
        }
        
        response = requests.post(f"{API_BASE_URL}/api/v1/auth/login", data=login_data)
        if response.status_code == 200:
            data = response.json()
            return data['access_token']
        else:
            logger.error(f"Authentication failed: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None

async def upload_dataset(token, dataset_path):
    """Upload dataset to the API."""
    try:
        headers = {'Authorization': f'Bearer {token}'}
        
        with open(dataset_path, 'rb') as f:
            files = {'file': f}
            data = {
                'name': 'test_dataset',
                'description': 'Test dataset for WebSocket experiment'
            }
            
            response = requests.post(
                f"{API_BASE_URL}/api/v1/datasets/upload",
                headers=headers,
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Dataset uploaded successfully: {result}")
                return result
            else:
                logger.error(f"Dataset upload failed: {response.text}")
                return None
                
    except Exception as e:
        logger.error(f"Dataset upload error: {e}")
        return None

async def create_experiment(token, dataset_info):
    """Create an experiment."""
    try:
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        experiment_data = {
            'name': 'WebSocket Test Experiment',
            'dataset_path': dataset_info.get('file_path'),
            'task_type': 'classification',
            'data_type': 'tabular',
            'target_column': 'target',
            'config': {
                'user_id': 'demo_user',
                'max_trials': 5,
                'timeout_minutes': 10
            }
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/experiments",
            headers=headers,
            json=experiment_data
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Experiment created successfully: {result}")
            return result
        else:
            logger.error(f"Experiment creation failed: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Experiment creation error: {e}")
        return None

async def start_experiment(token, experiment_id):
    """Start an experiment."""
    try:
        headers = {'Authorization': f'Bearer {token}'}
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/experiments/{experiment_id}/start",
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Experiment started successfully: {result}")
            return True
        else:
            logger.error(f"Experiment start failed: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Experiment start error: {e}")
        return False

async def monitor_experiment_websocket(token, experiment_id):
    """Monitor experiment progress via WebSocket."""
    try:
        ws_url = f"{WS_BASE_URL}/ws/connect?token={token}"
        websocket = await websockets.connect(ws_url)
        
        logger.info("WebSocket connected for experiment monitoring")
        
        # Subscribe to the specific experiment
        await websocket.send(json.dumps({
            'type': 'subscribe',
            'subscription': experiment_id
        }))
        
        # Also subscribe to all events to catch general updates
        await websocket.send(json.dumps({
            'type': 'subscribe',
            'subscription': 'all'
        }))
        
        experiment_events = []
        monitoring_duration = 60  # Monitor for 60 seconds
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < monitoring_duration:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                
                # Log all messages
                msg_type = data.get('type') or data.get('event_type', 'unknown')
                logger.info(f"Received: {msg_type}")
                
                # Track experiment-specific events
                if data.get('experiment_id') == experiment_id:
                    experiment_events.append(data)
                    logger.info(f"Experiment Event: {data.get('event_type')} - {data.get('data', {})}")
                
                # Check for experiment completion
                if (data.get('event_type') in ['experiment_completed', 'experiment_failed'] and 
                    data.get('experiment_id') == experiment_id):
                    logger.info("Experiment completed, stopping monitoring")
                    break
                    
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.info("WebSocket connection closed")
                break
        
        await websocket.close()
        logger.info(f"Monitoring completed. Received {len(experiment_events)} experiment events")
        
        return experiment_events
        
    except Exception as e:
        logger.error(f"WebSocket monitoring error: {e}")
        return []

async def main():
    """Main test function."""
    logger.info("Starting Experiment WebSocket Integration Test")
    logger.info("=" * 60)
    
    try:
        # Step 1: Create test dataset
        logger.info("1. Creating test dataset...")
        dataset_path = await create_test_dataset()
        logger.info(f"Test dataset created: {dataset_path}")
        
        # Step 2: Authenticate
        logger.info("2. Authenticating...")
        token = await authenticate_and_get_token()
        if not token:
            logger.error("Authentication failed")
            return
        
        # Step 3: Upload dataset
        logger.info("3. Uploading dataset...")
        dataset_info = await upload_dataset(token, dataset_path)
        if not dataset_info:
            logger.error("Dataset upload failed")
            return
        
        # Step 4: Create experiment
        logger.info("4. Creating experiment...")
        experiment_info = await create_experiment(token, dataset_info)
        if not experiment_info:
            logger.error("Experiment creation failed")
            return
        
        experiment_id = experiment_info.get('id')
        logger.info(f"Experiment ID: {experiment_id}")
        
        # Step 5: Start WebSocket monitoring in background
        logger.info("5. Starting WebSocket monitoring...")
        monitoring_task = asyncio.create_task(
            monitor_experiment_websocket(token, experiment_id)
        )
        
        # Give WebSocket time to connect and subscribe
        await asyncio.sleep(2)
        
        # Step 6: Start experiment
        logger.info("6. Starting experiment...")
        if await start_experiment(token, experiment_id):
            logger.info("Experiment started successfully")
        else:
            logger.error("Failed to start experiment")
            monitoring_task.cancel()
            return
        
        # Step 7: Wait for monitoring to complete
        logger.info("7. Waiting for experiment completion...")
        experiment_events = await monitoring_task
        
        # Step 8: Analyze results
        logger.info("8. Analyzing WebSocket events...")
        event_types = {}
        for event in experiment_events:
            event_type = event.get('event_type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        logger.info(f"Event summary: {event_types}")
        
        # Check if we received the expected events
        expected_events = ['experiment_started', 'experiment_progress']
        received_events = set(event_types.keys())
        
        logger.info("=" * 60)
        logger.info("Test Results:")
        logger.info(f"Total experiment events received: {len(experiment_events)}")
        logger.info(f"Event types: {list(received_events)}")
        
        for expected in expected_events:
            if expected in received_events:
                logger.info(f"✓ {expected} events received")
            else:
                logger.info(f"✗ {expected} events NOT received")
        
        if experiment_events:
            logger.info("✓ WebSocket experiment monitoring is working!")
        else:
            logger.info("✗ No experiment events received via WebSocket")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
    
    finally:
        # Cleanup
        try:
            import os
            if 'dataset_path' in locals():
                os.unlink(dataset_path)
                logger.info("Cleaned up test dataset")
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())