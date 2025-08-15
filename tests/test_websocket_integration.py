"""
Integration tests for WebSocket functionality in AutoML Framework.

Tests WebSocket connection, event broadcasting, and real-time updates.
"""

import asyncio
import json
import pytest
import websockets
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from automl_framework.api.websocket_manager import (
    WebSocketManager, 
    WebSocketEvent, 
    EventType,
    websocket_manager
)
from automl_framework.api.routes.websocket import (
    broadcast_experiment_created,
    broadcast_experiment_progress,
    broadcast_resource_update
)


class TestWebSocketManager:
    """Test WebSocket manager functionality."""
    
    @pytest.fixture
    def ws_manager(self):
        """Create a fresh WebSocket manager for testing."""
        return WebSocketManager()
    
    @pytest.mark.asyncio
    async def test_websocket_manager_initialization(self, ws_manager):
        """Test WebSocket manager initialization."""
        assert ws_manager.connections == {}
        assert ws_manager.user_connections == {}
        assert ws_manager.experiment_subscribers == {}
        assert ws_manager.global_subscribers == set()
    
    @pytest.mark.asyncio
    async def test_websocket_manager_start_stop(self, ws_manager):
        """Test WebSocket manager start and stop."""
        await ws_manager.start()
        assert ws_manager.event_processor_task is not None
        assert ws_manager.ping_task is not None
        
        await ws_manager.stop()
        assert ws_manager.event_processor_task.cancelled()
        assert ws_manager.ping_task.cancelled()
    
    @pytest.mark.asyncio
    async def test_websocket_connection_lifecycle(self, ws_manager):
        """Test WebSocket connection lifecycle."""
        # Mock WebSocket
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        mock_websocket.close = AsyncMock()
        
        # Connect
        connection_id = await ws_manager.connect(mock_websocket, "test_user")
        
        assert connection_id in ws_manager.connections
        assert "test_user" in ws_manager.user_connections
        assert connection_id in ws_manager.user_connections["test_user"]
        mock_websocket.accept.assert_called_once()
        
        # Disconnect
        await ws_manager.disconnect(connection_id)
        
        assert connection_id not in ws_manager.connections
        assert "test_user" not in ws_manager.user_connections
        mock_websocket.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_subscription_management(self, ws_manager):
        """Test subscription management."""
        # Mock WebSocket
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        # Connect
        connection_id = await ws_manager.connect(mock_websocket, "test_user")
        
        # Subscribe to all events
        await ws_manager.subscribe(connection_id, "all")
        assert connection_id in ws_manager.global_subscribers
        
        # Subscribe to specific experiment
        await ws_manager.subscribe(connection_id, "exp_123")
        assert "exp_123" in ws_manager.experiment_subscribers
        assert connection_id in ws_manager.experiment_subscribers["exp_123"]
        
        # Unsubscribe
        await ws_manager.unsubscribe(connection_id, "all")
        assert connection_id not in ws_manager.global_subscribers
        
        await ws_manager.unsubscribe(connection_id, "exp_123")
        assert "exp_123" not in ws_manager.experiment_subscribers
    
    @pytest.mark.asyncio
    async def test_event_broadcasting(self, ws_manager):
        """Test event broadcasting to subscribers."""
        # Mock WebSocket
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        # Connect and subscribe
        connection_id = await ws_manager.connect(mock_websocket, "test_user")
        await ws_manager.subscribe(connection_id, "all")
        
        # Start the manager to enable event processing
        await ws_manager.start()
        
        # Create and broadcast event
        event = WebSocketEvent(
            event_type=EventType.EXPERIMENT_CREATED,
            data={"experiment_id": "exp_123", "name": "Test Experiment"},
            timestamp=datetime.now(),
            user_id="test_user"
        )
        
        await ws_manager.broadcast_event(event)
        
        # Wait a bit for event processing
        await asyncio.sleep(0.1)
        
        # Verify event was sent
        mock_websocket.send_text.assert_called()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data['event_type'] == 'experiment_created'
        assert sent_data['data']['experiment_id'] == 'exp_123'
        
        await ws_manager.stop()
    
    @pytest.mark.asyncio
    async def test_user_specific_events(self, ws_manager):
        """Test user-specific event delivery."""
        # Mock WebSockets for two users
        mock_ws1 = AsyncMock()
        mock_ws1.accept = AsyncMock()
        mock_ws1.send_text = AsyncMock()
        
        mock_ws2 = AsyncMock()
        mock_ws2.accept = AsyncMock()
        mock_ws2.send_text = AsyncMock()
        
        # Connect two users
        conn1 = await ws_manager.connect(mock_ws1, "user1")
        conn2 = await ws_manager.connect(mock_ws2, "user2")
        
        await ws_manager.start()
        
        # Send event to specific user
        event = WebSocketEvent(
            event_type=EventType.NOTIFICATION,
            data={"message": "Hello user1"},
            timestamp=datetime.now(),
            user_id="user1"
        )
        
        await ws_manager.send_to_user("user1", event)
        await asyncio.sleep(0.1)
        
        # Verify only user1 received the event
        mock_ws1.send_text.assert_called()
        mock_ws2.send_text.assert_not_called()
        
        await ws_manager.stop()


class TestWebSocketEvents:
    """Test WebSocket event broadcasting functions."""
    
    @pytest.mark.asyncio
    async def test_broadcast_experiment_created(self):
        """Test experiment creation event broadcasting."""
        with patch('automl_framework.api.routes.websocket.websocket_manager') as mock_manager:
            mock_manager.broadcast_event = AsyncMock()
            
            await broadcast_experiment_created(
                "exp_123",
                {"name": "Test Experiment", "status": "created"},
                "user_123"
            )
            
            mock_manager.broadcast_event.assert_called_once()
            event = mock_manager.broadcast_event.call_args[0][0]
            assert event.event_type == EventType.EXPERIMENT_CREATED
            assert event.experiment_id == "exp_123"
            assert event.user_id == "user_123"
    
    @pytest.mark.asyncio
    async def test_broadcast_experiment_progress(self):
        """Test experiment progress event broadcasting."""
        with patch('automl_framework.api.routes.websocket.websocket_manager') as mock_manager:
            mock_manager.broadcast_event = AsyncMock()
            
            await broadcast_experiment_progress(
                "exp_123",
                {"stage": "training", "progress": 50.0},
                "user_123"
            )
            
            mock_manager.broadcast_event.assert_called_once()
            event = mock_manager.broadcast_event.call_args[0][0]
            assert event.event_type == EventType.EXPERIMENT_PROGRESS
            assert event.data["progress"] == 50.0
    
    @pytest.mark.asyncio
    async def test_broadcast_resource_update(self):
        """Test resource update event broadcasting."""
        with patch('automl_framework.api.routes.websocket.websocket_manager') as mock_manager:
            mock_manager.broadcast_event = AsyncMock()
            
            await broadcast_resource_update({
                "cpu_usage": 75.0,
                "memory_usage": 60.0,
                "gpu_usage": 90.0
            })
            
            mock_manager.broadcast_event.assert_called_once()
            event = mock_manager.broadcast_event.call_args[0][0]
            assert event.event_type == EventType.RESOURCE_UPDATE
            assert event.data["cpu_usage"] == 75.0


class TestWebSocketIntegration:
    """Integration tests for WebSocket functionality."""
    
    @pytest.mark.asyncio
    async def test_experiment_manager_websocket_integration(self):
        """Test experiment manager WebSocket integration."""
        from automl_framework.services.experiment_manager import ExperimentManager
        
        # Create experiment manager with WebSocket enabled
        exp_manager = ExperimentManager()
        exp_manager.enable_websocket_events()
        
        # Mock the WebSocket broadcasting
        with patch.object(exp_manager, '_broadcast_experiment_event', new_callable=AsyncMock) as mock_broadcast:
            # Create experiment
            experiment_id = exp_manager.create_experiment(
                dataset_path="test_data.csv",
                experiment_config={
                    "name": "Test Experiment",
                    "task_type": "classification",
                    "user_id": "test_user"
                }
            )
            
            # Verify creation event was broadcast
            mock_broadcast.assert_called_with(
                'created',
                experiment_id,
                {
                    'id': experiment_id,
                    'name': 'Test Experiment',
                    'status': 'created',
                    'created_at': mock_broadcast.call_args[0][2]['created_at']
                }
            )
    
    @pytest.mark.asyncio
    async def test_resource_scheduler_websocket_integration(self):
        """Test resource scheduler WebSocket integration."""
        from automl_framework.services.resource_scheduler import ResourceScheduler
        
        # Create resource scheduler with WebSocket enabled
        scheduler = ResourceScheduler()
        scheduler.enable_websocket_events()
        
        # Mock the WebSocket broadcasting
        with patch.object(scheduler, '_broadcast_job_event', new_callable=AsyncMock) as mock_broadcast:
            # Allocate resources (which should queue the job)
            job_requirements = {
                'job_id': 'test_job_123',
                'user_id': 'test_user',
                'cpu_cores': 2,
                'memory_gb': 4.0,
                'gpu_count': 1
            }
            
            result = scheduler.allocate_resources(job_requirements)
            
            # If job was queued, verify event was broadcast
            if result.get('status') == 'queued':
                mock_broadcast.assert_called_with(
                    'queued',
                    'test_job_123',
                    {
                        'status': 'queued',
                        'estimated_wait_minutes': mock_broadcast.call_args[0][2]['estimated_wait_minutes'],
                        'queue_position': mock_broadcast.call_args[0][2]['queue_position'],
                        'requirements': mock_broadcast.call_args[0][2]['requirements']
                    }
                )


@pytest.mark.asyncio
async def test_websocket_connection_stats():
    """Test WebSocket connection statistics."""
    ws_manager = WebSocketManager()
    
    # Mock WebSocket connections
    mock_ws1 = AsyncMock()
    mock_ws1.accept = AsyncMock()
    mock_ws2 = AsyncMock()
    mock_ws2.accept = AsyncMock()
    
    # Connect multiple users
    await ws_manager.connect(mock_ws1, "user1")
    await ws_manager.connect(mock_ws2, "user1")  # Same user, different connection
    conn3 = await ws_manager.connect(AsyncMock(), "user2")
    
    # Subscribe to different events
    await ws_manager.subscribe(conn3, "all")
    await ws_manager.subscribe(conn3, "exp_123")
    
    # Get stats
    stats = await ws_manager.get_connection_stats()
    
    assert stats['total_connections'] == 3
    assert stats['unique_users'] == 2
    assert stats['global_subscribers'] == 1
    assert stats['experiment_subscriptions'] == 1
    assert stats['connections_by_user']['user1'] == 2
    assert stats['connections_by_user']['user2'] == 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])