"""
WebSocket routes for real-time updates in AutoML Framework.

This module provides WebSocket endpoints for real-time experiment monitoring,
resource updates, and notifications.
"""

import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.security import HTTPBearer

from automl_framework.api.auth import get_current_user_websocket, User
from automl_framework.api.websocket_manager import websocket_manager, EventType, WebSocketEvent
from automl_framework.core.registry import get_service_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])

security = HTTPBearer()


@router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(..., description="Authentication token"),
    client_info: str = Query(None, description="Client information (browser, version, etc.)")
):
    """
    WebSocket endpoint for real-time updates.
    
    Clients can connect to this endpoint to receive real-time updates about:
    - Experiment progress and status changes
    - Resource utilization updates
    - Job queue status
    - System notifications
    """
    connection_id = None
    try:
        # Authenticate user
        user = await get_current_user_websocket(token)
        if not user:
            await websocket.close(code=4001, reason="Authentication failed")
            return
        
        # Parse client info
        metadata = {}
        if client_info:
            try:
                metadata = json.loads(client_info)
            except json.JSONDecodeError:
                metadata = {'raw_info': client_info}
        
        # Establish connection
        connection_id = await websocket_manager.connect(websocket, user.id, metadata)
        
        # Send initial system status
        await send_initial_status(connection_id)
        
        # Handle incoming messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await handle_websocket_message(connection_id, user, message)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {connection_id}")
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': 'Invalid JSON format'
                }))
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': str(e)
                }))
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        if websocket.client_state.name != 'DISCONNECTED':
            await websocket.close(code=4000, reason=str(e))
    
    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id)


async def handle_websocket_message(connection_id: str, user: User, message: Dict[str, Any]):
    """
    Handle incoming WebSocket messages from clients.
    
    Args:
        connection_id: ID of the WebSocket connection
        user: Authenticated user
        message: Parsed message from client
    """
    message_type = message.get('type')
    
    if message_type == 'subscribe':
        # Subscribe to specific events
        subscription = message.get('subscription', 'all')
        await websocket_manager.subscribe(connection_id, subscription)
        
        # Send confirmation
        await websocket_manager.connections[connection_id].send_message({
            'type': 'subscription_confirmed',
            'subscription': subscription,
            'timestamp': datetime.now().isoformat()
        })
        
    elif message_type == 'unsubscribe':
        # Unsubscribe from specific events
        subscription = message.get('subscription', 'all')
        await websocket_manager.unsubscribe(connection_id, subscription)
        
        # Send confirmation
        await websocket_manager.connections[connection_id].send_message({
            'type': 'unsubscription_confirmed',
            'subscription': subscription,
            'timestamp': datetime.now().isoformat()
        })
        
    elif message_type == 'get_experiment_status':
        # Get current status of an experiment
        experiment_id = message.get('experiment_id')
        if experiment_id:
            await send_experiment_status(connection_id, experiment_id)
        
    elif message_type == 'get_resource_status':
        # Get current resource status
        await send_resource_status(connection_id)
        
    elif message_type == 'get_connection_stats':
        # Get WebSocket connection statistics (admin only)
        if user.is_admin:
            stats = await websocket_manager.get_connection_stats()
            await websocket_manager.connections[connection_id].send_message({
                'type': 'connection_stats',
                'data': stats,
                'timestamp': datetime.now().isoformat()
            })
        
    elif message_type == 'update_preferences':
        # Update user preferences
        preferences = message.get('preferences', {})
        await websocket_manager.update_user_preferences(user.id, preferences)
        
        await websocket_manager.connections[connection_id].send_message({
            'type': 'preferences_updated',
            'preferences': preferences,
            'timestamp': datetime.now().isoformat()
        })
        
    elif message_type == 'get_session_info':
        # Get user session information
        session_info = await websocket_manager.get_user_session_info(user.id)
        await websocket_manager.connections[connection_id].send_message({
            'type': 'session_info',
            'data': session_info,
            'timestamp': datetime.now().isoformat()
        })
        
    elif message_type == 'pong':
        # Handle pong response
        if connection_id in websocket_manager.connections:
            websocket_manager.connections[connection_id].last_ping = datetime.now()
        
    else:
        # Unknown message type
        await websocket_manager.connections[connection_id].send_message({
            'type': 'error',
            'message': f'Unknown message type: {message_type}',
            'timestamp': datetime.now().isoformat()
        })


async def send_experiment_status(connection_id: str, experiment_id: str):
    """Send current experiment status to a connection."""
    try:
        registry = get_service_registry()
        experiment_manager = registry.get_service('experiment_manager')
        
        # Get experiment data
        experiment_data = experiment_manager.get_experiment_results(experiment_id)
        
        # Send status update
        await websocket_manager.connections[connection_id].send_message({
            'type': 'experiment_status',
            'experiment_id': experiment_id,
            'data': experiment_data
        })
        
    except Exception as e:
        logger.error(f"Failed to send experiment status: {e}")
        await websocket_manager.connections[connection_id].send_message({
            'type': 'error',
            'message': f'Failed to get experiment status: {str(e)}'
        })


async def send_resource_status(connection_id: str):
    """Send current resource status to a connection."""
    try:
        registry = get_service_registry()
        resource_scheduler = registry.get_service('resource_scheduler')
        
        # Get resource status
        resource_status = resource_scheduler.get_resource_status()
        
        # Send resource update
        await websocket_manager.connections[connection_id].send_message({
            'type': 'resource_status',
            'data': resource_status
        })
        
    except Exception as e:
        logger.error(f"Failed to send resource status: {e}")
        await websocket_manager.connections[connection_id].send_message({
            'type': 'error',
            'message': f'Failed to get resource status: {str(e)}'
        })


# Helper functions for broadcasting events from other services

async def broadcast_experiment_created(experiment_id: str, experiment_data: Dict[str, Any], user_id: str):
    """Broadcast experiment creation event."""
    event = WebSocketEvent(
        event_type=EventType.EXPERIMENT_CREATED,
        data=experiment_data,
        timestamp=datetime.now(),
        user_id=user_id,
        experiment_id=experiment_id
    )
    await websocket_manager.broadcast_event(event)


async def broadcast_experiment_started(experiment_id: str, experiment_data: Dict[str, Any], user_id: str):
    """Broadcast experiment start event."""
    event = WebSocketEvent(
        event_type=EventType.EXPERIMENT_STARTED,
        data=experiment_data,
        timestamp=datetime.now(),
        user_id=user_id,
        experiment_id=experiment_id
    )
    await websocket_manager.broadcast_event(event)


async def broadcast_experiment_progress(experiment_id: str, progress_data: Dict[str, Any], user_id: str):
    """Broadcast experiment progress update."""
    event = WebSocketEvent(
        event_type=EventType.EXPERIMENT_PROGRESS,
        data=progress_data,
        timestamp=datetime.now(),
        user_id=user_id,
        experiment_id=experiment_id
    )
    await websocket_manager.broadcast_event(event)


async def broadcast_experiment_completed(experiment_id: str, results_data: Dict[str, Any], user_id: str):
    """Broadcast experiment completion event."""
    event = WebSocketEvent(
        event_type=EventType.EXPERIMENT_COMPLETED,
        data=results_data,
        timestamp=datetime.now(),
        user_id=user_id,
        experiment_id=experiment_id
    )
    await websocket_manager.broadcast_event(event)


async def broadcast_experiment_failed(experiment_id: str, error_data: Dict[str, Any], user_id: str):
    """Broadcast experiment failure event."""
    event = WebSocketEvent(
        event_type=EventType.EXPERIMENT_FAILED,
        data=error_data,
        timestamp=datetime.now(),
        user_id=user_id,
        experiment_id=experiment_id
    )
    await websocket_manager.broadcast_event(event)


async def broadcast_resource_update(resource_data: Dict[str, Any]):
    """Broadcast resource status update."""
    event = WebSocketEvent(
        event_type=EventType.RESOURCE_UPDATE,
        data=resource_data,
        timestamp=datetime.now()
    )
    await websocket_manager.broadcast_event(event)


async def broadcast_job_queued(job_id: str, job_data: Dict[str, Any], user_id: str):
    """Broadcast job queued event."""
    event = WebSocketEvent(
        event_type=EventType.JOB_QUEUED,
        data={'job_id': job_id, **job_data},
        timestamp=datetime.now(),
        user_id=user_id
    )
    await websocket_manager.broadcast_event(event)


async def broadcast_job_started(job_id: str, job_data: Dict[str, Any], user_id: str):
    """Broadcast job started event."""
    event = WebSocketEvent(
        event_type=EventType.JOB_STARTED,
        data={'job_id': job_id, **job_data},
        timestamp=datetime.now(),
        user_id=user_id
    )
    await websocket_manager.broadcast_event(event)


async def broadcast_job_completed(job_id: str, job_data: Dict[str, Any], user_id: str):
    """Broadcast job completed event."""
    event = WebSocketEvent(
        event_type=EventType.JOB_COMPLETED,
        data={'job_id': job_id, **job_data},
        timestamp=datetime.now(),
        user_id=user_id
    )
    await websocket_manager.broadcast_event(event)


async def send_initial_status(connection_id: str):
    """Send initial system status to a newly connected client."""
    try:
        # Send resource status
        await send_resource_status(connection_id)
        
        # Send connection statistics
        stats = await websocket_manager.get_connection_stats()
        await websocket_manager.connections[connection_id].send_message({
            'type': 'initial_status',
            'connection_stats': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to send initial status: {e}")


# Enhanced broadcasting functions with better error handling and logging

async def broadcast_experiment_created(experiment_id: str, experiment_data: Dict[str, Any], user_id: str):
    """Broadcast experiment creation event."""
    event = WebSocketEvent(
        event_type=EventType.EXPERIMENT_CREATED,
        data=experiment_data,
        timestamp=datetime.now(),
        user_id=user_id,
        experiment_id=experiment_id
    )
    await websocket_manager.broadcast_event(event)
    
    # Send user notification
    await websocket_manager.send_experiment_notification(
        user_id, experiment_id, 'created', experiment_data
    )


async def broadcast_experiment_started(experiment_id: str, experiment_data: Dict[str, Any], user_id: str):
    """Broadcast experiment start event."""
    event = WebSocketEvent(
        event_type=EventType.EXPERIMENT_STARTED,
        data=experiment_data,
        timestamp=datetime.now(),
        user_id=user_id,
        experiment_id=experiment_id
    )
    await websocket_manager.broadcast_event(event)
    
    # Send user notification
    await websocket_manager.send_experiment_notification(
        user_id, experiment_id, 'started', experiment_data
    )


async def broadcast_experiment_progress(experiment_id: str, progress_data: Dict[str, Any], user_id: str):
    """Broadcast experiment progress update."""
    event = WebSocketEvent(
        event_type=EventType.EXPERIMENT_PROGRESS,
        data=progress_data,
        timestamp=datetime.now(),
        user_id=user_id,
        experiment_id=experiment_id
    )
    await websocket_manager.broadcast_event(event)
    
    # Send progress notification for major milestones
    overall_progress = progress_data.get('overall_progress', 0)
    if overall_progress > 0 and overall_progress % 25 == 0:  # Every 25%
        await websocket_manager.send_experiment_notification(
            user_id, experiment_id, 'progress', progress_data
        )


async def broadcast_experiment_completed(experiment_id: str, results_data: Dict[str, Any], user_id: str):
    """Broadcast experiment completion event."""
    event = WebSocketEvent(
        event_type=EventType.EXPERIMENT_COMPLETED,
        data=results_data,
        timestamp=datetime.now(),
        user_id=user_id,
        experiment_id=experiment_id
    )
    await websocket_manager.broadcast_event(event)
    
    # Send completion notification
    await websocket_manager.send_experiment_notification(
        user_id, experiment_id, 'completed', results_data
    )


async def broadcast_experiment_failed(experiment_id: str, error_data: Dict[str, Any], user_id: str):
    """Broadcast experiment failure event."""
    event = WebSocketEvent(
        event_type=EventType.EXPERIMENT_FAILED,
        data=error_data,
        timestamp=datetime.now(),
        user_id=user_id,
        experiment_id=experiment_id
    )
    await websocket_manager.broadcast_event(event)
    
    # Send failure notification
    await websocket_manager.send_experiment_notification(
        user_id, experiment_id, 'failed', error_data
    )


async def broadcast_resource_update(resource_data: Dict[str, Any]):
    """Broadcast resource status update."""
    event = WebSocketEvent(
        event_type=EventType.RESOURCE_UPDATE,
        data=resource_data,
        timestamp=datetime.now()
    )
    await websocket_manager.broadcast_event(event)


async def broadcast_job_queued(job_id: str, job_data: Dict[str, Any], user_id: str):
    """Broadcast job queued event."""
    event = WebSocketEvent(
        event_type=EventType.JOB_QUEUED,
        data={'job_id': job_id, **job_data},
        timestamp=datetime.now(),
        user_id=user_id
    )
    await websocket_manager.broadcast_event(event)


async def broadcast_job_started(job_id: str, job_data: Dict[str, Any], user_id: str):
    """Broadcast job started event."""
    event = WebSocketEvent(
        event_type=EventType.JOB_STARTED,
        data={'job_id': job_id, **job_data},
        timestamp=datetime.now(),
        user_id=user_id
    )
    await websocket_manager.broadcast_event(event)


async def broadcast_job_completed(job_id: str, job_data: Dict[str, Any], user_id: str):
    """Broadcast job completed event."""
    event = WebSocketEvent(
        event_type=EventType.JOB_COMPLETED,
        data={'job_id': job_id, **job_data},
        timestamp=datetime.now(),
        user_id=user_id
    )
    await websocket_manager.broadcast_event(event)


# Import datetime here to avoid circular imports
from datetime import datetime