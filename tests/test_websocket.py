#!/usr/bin/env python3
"""
Test script for WebSocket functionality in AutoML Framework.

This script tests the WebSocket implementation by creating an experiment
and monitoring real-time updates.
"""

import asyncio
import json
import logging
import websockets
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000"
WS_BASE_URL = "ws://localhost:8000"
TEST_TOKEN = "test_token"  # You'll need to get a real token

async def test_websocket_connection():
    """Test basic WebSocket connection and message handling."""
    
    # For this test, we'll use a mock token
    # In real usage, you'd get this from the login endpoint
    ws_url = f"{WS_BASE_URL}/ws/connect?token={TEST_TOKEN}"
    
    try:
        async with websockets.connect(ws_url) as websocket:
            logger.info("WebSocket connected successfully")
            
            # Send subscription message
            subscribe_message = {
                "type": "subscribe",
                "subscription": "all"
            }
            await websocket.send(json.dumps(subscribe_message))
            logger.info("Subscribed to all events")
            
            # Listen for messages for 30 seconds
            timeout = 30
            start_time = datetime.now()
            
            while (datetime.now() - start_time).total_seconds() < timeout:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    
                    logger.info(f"Received message: {data.get('type', 'unknown')}")
                    
                    if data.get('type') == 'ping':
                        # Respond to ping
                        pong_message = {
                            "type": "pong",
                            "timestamp": data.get('timestamp')
                        }
                        await websocket.send(json.dumps(pong_message))
                        logger.info("Sent pong response")
                    
                    elif data.get('event_type'):
                        logger.info(f"Received event: {data['event_type']} - {data.get('data', {})}")
                    
                except asyncio.TimeoutError:
                    # No message received, continue
                    continue
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message: {e}")
                    continue
            
            logger.info("Test completed successfully")
            
    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"WebSocket connection closed: {e}")
    except Exception as e:
        logger.error(f"WebSocket test failed: {e}")

async def test_experiment_events():
    """Test experiment-related WebSocket events."""
    
    # This would require a real authentication token and running API server
    logger.info("Experiment events test would require running API server")
    logger.info("To test manually:")
    logger.info("1. Start the API server: python -m automl_framework.api.main")
    logger.info("2. Login to get a token")
    logger.info("3. Connect to WebSocket with the token")
    logger.info("4. Create and start an experiment")
    logger.info("5. Monitor real-time progress updates")

if __name__ == "__main__":
    print("WebSocket Test Script")
    print("====================")
    
    print("\nTesting basic WebSocket connection...")
    try:
        asyncio.run(test_websocket_connection())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {e}")
    
    print("\nExperiment events test info:")
    asyncio.run(test_experiment_events())