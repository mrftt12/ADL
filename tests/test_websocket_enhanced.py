#!/usr/bin/env python3
"""
Enhanced WebSocket integration test for AutoML Framework.

This script tests the real-time WebSocket communication features including:
- Multi-user session management
- Event-driven updates for training progress
- Notification system for experiment completion
- Resource monitoring updates
"""

import asyncio
import json
import logging
import websockets
import requests
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000"
WS_BASE_URL = "ws://localhost:8000"

class WebSocketTestClient:
    """Test client for WebSocket functionality."""
    
    def __init__(self, user_credentials: Dict[str, str]):
        self.credentials = user_credentials
        self.token = None
        self.websocket = None
        self.received_messages = []
        
    async def authenticate(self):
        """Authenticate and get access token."""
        try:
            # Login to get token
            login_data = {
                'username': self.credentials['username'],
                'password': self.credentials['password']
            }
            
            response = requests.post(f"{API_BASE_URL}/api/v1/auth/login", data=login_data)
            if response.status_code == 200:
                data = response.json()
                self.token = data['access_token']
                logger.info(f"Authenticated user: {self.credentials['username']}")
                return True
            else:
                logger.error(f"Authentication failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    async def connect_websocket(self):
        """Connect to WebSocket endpoint."""
        if not self.token:
            logger.error("No authentication token available")
            return False
        
        try:
            # Simple WebSocket URL without client_info for now
            ws_url = f"{WS_BASE_URL}/ws/connect?token={self.token}"
            self.websocket = await websockets.connect(ws_url)
            logger.info(f"WebSocket connected for user: {self.credentials['username']}")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def send_message(self, message: Dict[str, Any]):
        """Send message to WebSocket."""
        if self.websocket:
            await self.websocket.send(json.dumps(message))
    
    async def receive_messages(self, duration: int = 30):
        """Receive messages for specified duration."""
        if not self.websocket:
            return
        
        try:
            end_time = asyncio.get_event_loop().time() + duration
            
            while asyncio.get_event_loop().time() < end_time:
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    self.received_messages.append(data)
                    logger.info(f"[{self.credentials['username']}] Received: {data.get('type', 'unknown')}")
                    
                    # Handle specific message types
                    if data.get('type') == 'connection_established':
                        logger.info(f"Connection established: {data.get('connection_id')}")
                    elif data.get('event_type'):
                        logger.info(f"Event: {data['event_type']} - {data.get('data', {})}")
                    
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.info("WebSocket connection closed")
                    break
                    
        except Exception as e:
            logger.error(f"Error receiving messages: {e}")
    
    async def test_subscription_management(self):
        """Test subscription and unsubscription functionality."""
        logger.info("Testing subscription management...")
        
        # Subscribe to all events
        await self.send_message({
            'type': 'subscribe',
            'subscription': 'all'
        })
        
        await asyncio.sleep(1)
        
        # Subscribe to specific experiment (dummy ID)
        await self.send_message({
            'type': 'subscribe',
            'subscription': 'experiment_123'
        })
        
        await asyncio.sleep(1)
        
        # Get session info
        await self.send_message({
            'type': 'get_session_info'
        })
        
        await asyncio.sleep(1)
        
        # Update preferences
        await self.send_message({
            'type': 'update_preferences',
            'preferences': {
                'notifications_enabled': True,
                'auto_subscribe_experiments': True
            }
        })
        
        await asyncio.sleep(1)
    
    async def test_resource_monitoring(self):
        """Test resource status monitoring."""
        logger.info("Testing resource monitoring...")
        
        # Request resource status
        await self.send_message({
            'type': 'get_resource_status'
        })
        
        await asyncio.sleep(2)
    
    async def close(self):
        """Close WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            logger.info(f"WebSocket closed for user: {self.credentials['username']}")


async def test_multi_user_sessions():
    """Test multi-user session management."""
    logger.info("=== Testing Multi-User Sessions ===")
    
    # Create multiple test clients
    users = [
        {'username': 'demo_user', 'password': 'secret'},
        {'username': 'admin', 'password': 'secret'}
    ]
    
    clients = []
    
    try:
        # Create and authenticate clients
        for user in users:
            client = WebSocketTestClient(user)
            if await client.authenticate():
                if await client.connect_websocket():
                    clients.append(client)
        
        if not clients:
            logger.error("No clients connected successfully")
            return
        
        # Test concurrent operations
        tasks = []
        
        for client in clients:
            # Start message receiving
            tasks.append(asyncio.create_task(client.receive_messages(20)))
            
            # Test subscription management
            tasks.append(asyncio.create_task(client.test_subscription_management()))
            
            # Test resource monitoring
            tasks.append(asyncio.create_task(client.test_resource_monitoring()))
        
        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Print results
        for client in clients:
            logger.info(f"User {client.credentials['username']} received {len(client.received_messages)} messages")
            
            # Show message types received
            message_types = {}
            for msg in client.received_messages:
                msg_type = msg.get('type') or msg.get('event_type', 'unknown')
                message_types[msg_type] = message_types.get(msg_type, 0) + 1
            
            logger.info(f"Message types: {message_types}")
    
    finally:
        # Clean up
        for client in clients:
            await client.close()


async def test_experiment_simulation():
    """Simulate experiment events to test real-time updates."""
    logger.info("=== Testing Experiment Event Simulation ===")
    
    # This would normally be done by creating actual experiments
    # For now, we'll test the WebSocket infrastructure
    
    client = WebSocketTestClient({'username': 'demo_user', 'password': 'secret'})
    
    try:
        if await client.authenticate():
            if await client.connect_websocket():
                # Subscribe to experiment events
                await client.send_message({
                    'type': 'subscribe',
                    'subscription': 'all'
                })
                
                # Listen for events
                await client.receive_messages(10)
                
                logger.info(f"Received {len(client.received_messages)} messages during simulation")
    
    finally:
        await client.close()


async def test_connection_statistics():
    """Test connection statistics (admin only)."""
    logger.info("=== Testing Connection Statistics ===")
    
    admin_client = WebSocketTestClient({'username': 'admin', 'password': 'secret'})
    
    try:
        if await admin_client.authenticate():
            if await admin_client.connect_websocket():
                # Request connection stats (admin only)
                await admin_client.send_message({
                    'type': 'get_connection_stats'
                })
                
                # Listen for response
                await admin_client.receive_messages(5)
                
                # Check if we received stats
                stats_received = any(
                    msg.get('type') == 'connection_stats' 
                    for msg in admin_client.received_messages
                )
                
                if stats_received:
                    logger.info("✓ Connection statistics received successfully")
                else:
                    logger.warning("✗ Connection statistics not received")
    
    finally:
        await admin_client.close()


async def main():
    """Run all WebSocket tests."""
    logger.info("Starting Enhanced WebSocket Integration Tests")
    logger.info("=" * 50)
    
    try:
        # Test multi-user sessions
        await test_multi_user_sessions()
        
        await asyncio.sleep(2)
        
        # Test experiment simulation
        await test_experiment_simulation()
        
        await asyncio.sleep(2)
        
        # Test connection statistics
        await test_connection_statistics()
        
        logger.info("=" * 50)
        logger.info("All WebSocket tests completed")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())