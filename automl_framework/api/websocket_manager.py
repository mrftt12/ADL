"""
WebSocket Manager for real-time updates in AutoML Framework.

This module provides WebSocket connection management, event broadcasting,
and real-time communication for experiment monitoring and resource updates.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events that can be broadcast."""
    EXPERIMENT_CREATED = "experiment_created"
    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_PROGRESS = "experiment_progress"
    EXPERIMENT_COMPLETED = "experiment_completed"
    EXPERIMENT_FAILED = "experiment_failed"
    EXPERIMENT_CANCELLED = "experiment_cancelled"
    RESOURCE_UPDATE = "resource_update"
    JOB_QUEUED = "job_queued"
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    NOTIFICATION = "notification"


@dataclass
class WebSocketEvent:
    """Represents an event to be broadcast via WebSocket."""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None
    experiment_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            'event_type': self.event_type.value,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'experiment_id': self.experiment_id
        }


@dataclass
class WebSocketConnection:
    """Represents a WebSocket connection."""
    connection_id: str
    websocket: WebSocket
    user_id: str
    subscriptions: Set[str]  # Set of experiment IDs or 'all' for global updates
    connected_at: datetime
    last_ping: datetime
    
    async def send_event(self, event: WebSocketEvent) -> bool:
        """Send an event to this connection."""
        try:
            await self.websocket.send_text(json.dumps(event.to_dict()))
            return True
        except Exception as e:
            logger.error(f"Failed to send event to connection {self.connection_id}: {e}")
            return False
    
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send a raw message to this connection."""
        try:
            await self.websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Failed to send message to connection {self.connection_id}: {e}")
            return False


class WebSocketManager:
    """
    Manages WebSocket connections and event broadcasting.
    
    Handles connection lifecycle, subscription management, and real-time
    event distribution to connected clients.
    """
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.experiment_subscribers: Dict[str, Set[str]] = {}  # experiment_id -> connection_ids
        self.global_subscribers: Set[str] = set()  # connection_ids subscribed to all events
        self._lock = asyncio.Lock()
        
        # Event queue for reliable delivery
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.event_processor_task: Optional[asyncio.Task] = None
        
        # Connection monitoring
        self.ping_interval = 30  # seconds
        self.ping_task: Optional[asyncio.Task] = None
        
        # Enhanced session management
        self.user_sessions: Dict[str, Dict[str, Any]] = {}  # user_id -> session_data
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}  # connection_id -> metadata
        
        # Event history for new connections
        self.event_history: Dict[str, List[WebSocketEvent]] = {}  # experiment_id -> events
        self.max_history_size = 100
        
        # Connection statistics
        self.connection_stats = {
            'total_connections': 0,
            'peak_connections': 0,
            'total_events_sent': 0,
            'failed_deliveries': 0
        }
        
        logger.info("WebSocketManager initialized")
    
    async def start(self):
        """Start the WebSocket manager background tasks."""
        self.event_processor_task = asyncio.create_task(self._process_events())
        self.ping_task = asyncio.create_task(self._ping_connections())
        logger.info("WebSocketManager started")
    
    async def stop(self):
        """Stop the WebSocket manager and close all connections."""
        if self.event_processor_task:
            self.event_processor_task.cancel()
        if self.ping_task:
            self.ping_task.cancel()
        
        # Close all connections
        for connection in list(self.connections.values()):
            await self.disconnect(connection.connection_id)
        
        logger.info("WebSocketManager stopped")
    
    async def connect(self, websocket: WebSocket, user_id: str, metadata: Dict[str, Any] = None) -> str:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: FastAPI WebSocket instance
            user_id: ID of the user connecting
            metadata: Optional connection metadata (browser info, etc.)
            
        Returns:
            str: Unique connection ID
        """
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        connection = WebSocketConnection(
            connection_id=connection_id,
            websocket=websocket,
            user_id=user_id,
            subscriptions=set(),
            connected_at=datetime.now(),
            last_ping=datetime.now()
        )
        
        async with self._lock:
            self.connections[connection_id] = connection
            
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)
            
            # Store connection metadata
            self.connection_metadata[connection_id] = metadata or {}
            
            # Initialize user session if not exists
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = {
                    'first_connected': datetime.now(),
                    'active_experiments': set(),
                    'preferences': {}
                }
            
            # Update connection statistics
            self.connection_stats['total_connections'] += 1
            current_connections = len(self.connections)
            if current_connections > self.connection_stats['peak_connections']:
                self.connection_stats['peak_connections'] = current_connections
        
        # Send welcome message with session info
        session_info = self.user_sessions[user_id].copy()
        session_info['first_connected'] = session_info['first_connected'].isoformat()
        session_info['active_experiments'] = list(session_info['active_experiments'])
        
        await connection.send_message({
            'type': 'connection_established',
            'connection_id': connection_id,
            'user_id': user_id,
            'session_info': session_info,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"WebSocket connection established: {connection_id} for user {user_id}")
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """
        Disconnect a WebSocket connection.
        
        Args:
            connection_id: ID of the connection to disconnect
        """
        async with self._lock:
            if connection_id not in self.connections:
                return
            
            connection = self.connections[connection_id]
            user_id = connection.user_id
            
            # Remove from all subscriptions
            for experiment_id in connection.subscriptions:
                if experiment_id in self.experiment_subscribers:
                    self.experiment_subscribers[experiment_id].discard(connection_id)
                    if not self.experiment_subscribers[experiment_id]:
                        del self.experiment_subscribers[experiment_id]
            
            self.global_subscribers.discard(connection_id)
            
            # Remove from user connections
            if user_id in self.user_connections:
                self.user_connections[user_id].discard(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            # Close WebSocket
            try:
                await connection.websocket.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket {connection_id}: {e}")
            
            del self.connections[connection_id]
        
        logger.info(f"WebSocket connection disconnected: {connection_id}")
    
    async def subscribe(self, connection_id: str, subscription: str):
        """
        Subscribe a connection to specific events.
        
        Args:
            connection_id: ID of the connection
            subscription: 'all' for global events or experiment_id for specific experiment
        """
        async with self._lock:
            if connection_id not in self.connections:
                return
            
            connection = self.connections[connection_id]
            connection.subscriptions.add(subscription)
            
            if subscription == 'all':
                self.global_subscribers.add(connection_id)
            else:
                # Assume it's an experiment ID
                if subscription not in self.experiment_subscribers:
                    self.experiment_subscribers[subscription] = set()
                self.experiment_subscribers[subscription].add(connection_id)
                
                # Add to user's active experiments
                user_id = connection.user_id
                if user_id in self.user_sessions:
                    self.user_sessions[user_id]['active_experiments'].add(subscription)
                
                # Send recent event history for this experiment
                await self._send_event_history(connection_id, subscription)
        
        logger.debug(f"Connection {connection_id} subscribed to {subscription}")
    
    async def _send_event_history(self, connection_id: str, experiment_id: str):
        """Send recent event history for an experiment to a newly subscribed connection."""
        if experiment_id in self.event_history and connection_id in self.connections:
            connection = self.connections[connection_id]
            recent_events = self.event_history[experiment_id][-10:]  # Last 10 events
            
            for event in recent_events:
                await connection.send_event(event)
    
    async def unsubscribe(self, connection_id: str, subscription: str):
        """
        Unsubscribe a connection from specific events.
        
        Args:
            connection_id: ID of the connection
            subscription: 'all' for global events or experiment_id for specific experiment
        """
        async with self._lock:
            if connection_id not in self.connections:
                return
            
            connection = self.connections[connection_id]
            connection.subscriptions.discard(subscription)
            
            if subscription == 'all':
                self.global_subscribers.discard(connection_id)
            else:
                if subscription in self.experiment_subscribers:
                    self.experiment_subscribers[subscription].discard(connection_id)
                    if not self.experiment_subscribers[subscription]:
                        del self.experiment_subscribers[subscription]
        
        logger.debug(f"Connection {connection_id} unsubscribed from {subscription}")
    
    async def broadcast_event(self, event: WebSocketEvent):
        """
        Broadcast an event to relevant subscribers.
        
        Args:
            event: Event to broadcast
        """
        await self.event_queue.put(event)
    
    async def send_to_user(self, user_id: str, event: WebSocketEvent):
        """
        Send an event to all connections for a specific user.
        
        Args:
            user_id: ID of the user
            event: Event to send
        """
        event.user_id = user_id
        await self.event_queue.put(event)
    
    async def send_notification(self, user_id: str, message: str, level: str = "info", 
                              title: str = None, action_url: str = None, persistent: bool = False):
        """
        Send a notification to a specific user.
        
        Args:
            user_id: ID of the user
            message: Notification message
            level: Notification level (info, warning, error, success)
            title: Optional notification title
            action_url: Optional URL for notification action
            persistent: Whether notification should persist until dismissed
        """
        event = WebSocketEvent(
            event_type=EventType.NOTIFICATION,
            data={
                'message': message,
                'level': level,
                'title': title,
                'action_url': action_url,
                'persistent': persistent,
                'id': str(uuid.uuid4())
            },
            timestamp=datetime.now(),
            user_id=user_id
        )
        await self.send_to_user(user_id, event)
    
    async def send_experiment_notification(self, user_id: str, experiment_id: str, 
                                         event_type: str, data: Dict[str, Any]):
        """
        Send an experiment-specific notification with enhanced context.
        
        Args:
            user_id: ID of the user
            experiment_id: ID of the experiment
            event_type: Type of experiment event
            data: Event data
        """
        # Create contextual notification message
        messages = {
            'started': f"Experiment '{data.get('name', experiment_id)}' has started",
            'progress': f"Experiment '{data.get('name', experiment_id)}' progress: {data.get('overall_progress', 0):.1f}%",
            'completed': f"Experiment '{data.get('name', experiment_id)}' completed successfully",
            'failed': f"Experiment '{data.get('name', experiment_id)}' failed: {data.get('error_message', 'Unknown error')}"
        }
        
        levels = {
            'started': 'info',
            'progress': 'info',
            'completed': 'success',
            'failed': 'error'
        }
        
        message = messages.get(event_type, f"Experiment {event_type}")
        level = levels.get(event_type, 'info')
        
        await self.send_notification(
            user_id=user_id,
            message=message,
            level=level,
            title=f"Experiment {event_type.title()}",
            action_url=f"/experiments/{experiment_id}",
            persistent=(event_type in ['completed', 'failed'])
        )
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about current connections."""
        async with self._lock:
            return {
                'current_connections': len(self.connections),
                'unique_users': len(self.user_connections),
                'global_subscribers': len(self.global_subscribers),
                'experiment_subscriptions': len(self.experiment_subscribers),
                'connections_by_user': {
                    user_id: len(conn_ids) 
                    for user_id, conn_ids in self.user_connections.items()
                },
                'total_connections': self.connection_stats['total_connections'],
                'peak_connections': self.connection_stats['peak_connections'],
                'total_events_sent': self.connection_stats['total_events_sent'],
                'failed_deliveries': self.connection_stats['failed_deliveries'],
                'active_experiments': len(self.event_history)
            }
    
    async def get_user_session_info(self, user_id: str) -> Dict[str, Any]:
        """Get session information for a specific user."""
        async with self._lock:
            if user_id not in self.user_sessions:
                return {}
            
            session = self.user_sessions[user_id]
            connections = self.user_connections.get(user_id, set())
            
            return {
                'user_id': user_id,
                'first_connected': session['first_connected'].isoformat(),
                'active_connections': len(connections),
                'active_experiments': list(session['active_experiments']),
                'preferences': session['preferences'],
                'connection_ids': list(connections)
            }
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Update user preferences for WebSocket sessions."""
        async with self._lock:
            if user_id in self.user_sessions:
                self.user_sessions[user_id]['preferences'].update(preferences)
    
    async def broadcast_to_all_users(self, event: WebSocketEvent):
        """Broadcast an event to all connected users."""
        await self.event_queue.put(event)
    
    async def get_experiment_subscribers(self, experiment_id: str) -> List[str]:
        """Get list of connection IDs subscribed to a specific experiment."""
        async with self._lock:
            return list(self.experiment_subscribers.get(experiment_id, set()))
    
    async def _process_events(self):
        """Background task to process and distribute events."""
        while True:
            try:
                event = await self.event_queue.get()
                await self._distribute_event(event)
                self.event_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _distribute_event(self, event: WebSocketEvent):
        """Distribute an event to relevant connections."""
        target_connections = set()
        
        async with self._lock:
            # Add global subscribers
            target_connections.update(self.global_subscribers)
            
            # Add experiment-specific subscribers
            if event.experiment_id and event.experiment_id in self.experiment_subscribers:
                target_connections.update(self.experiment_subscribers[event.experiment_id])
            
            # Add user-specific connections
            if event.user_id and event.user_id in self.user_connections:
                target_connections.update(self.user_connections[event.user_id])
            
            # Store event in history for experiment-specific events
            if event.experiment_id:
                if event.experiment_id not in self.event_history:
                    self.event_history[event.experiment_id] = []
                
                self.event_history[event.experiment_id].append(event)
                
                # Limit history size
                if len(self.event_history[event.experiment_id]) > self.max_history_size:
                    self.event_history[event.experiment_id] = self.event_history[event.experiment_id][-self.max_history_size:]
        
        # Send to all target connections
        failed_connections = []
        successful_deliveries = 0
        
        for connection_id in target_connections:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                success = await connection.send_event(event)
                if success:
                    successful_deliveries += 1
                else:
                    failed_connections.append(connection_id)
        
        # Update statistics
        self.connection_stats['total_events_sent'] += successful_deliveries
        self.connection_stats['failed_deliveries'] += len(failed_connections)
        
        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)
        
        if target_connections:
            logger.debug(f"Distributed {event.event_type.value} to {successful_deliveries}/{len(target_connections)} connections")
    
    async def _ping_connections(self):
        """Background task to ping connections and remove stale ones."""
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                
                current_time = datetime.now()
                stale_connections = []
                
                async with self._lock:
                    for connection_id, connection in self.connections.items():
                        # Check if connection is stale
                        if (current_time - connection.last_ping).total_seconds() > self.ping_interval * 2:
                            stale_connections.append(connection_id)
                        else:
                            # Send ping
                            success = await connection.send_message({
                                'type': 'ping',
                                'timestamp': current_time.isoformat()
                            })
                            if success:
                                connection.last_ping = current_time
                            else:
                                stale_connections.append(connection_id)
                
                # Remove stale connections
                for connection_id in stale_connections:
                    await self.disconnect(connection_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ping task: {e}")


# Global WebSocket manager instance
websocket_manager = WebSocketManager()