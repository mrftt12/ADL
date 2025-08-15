/**
 * React hook for WebSocket real-time updates
 */

import { useEffect, useCallback, useRef, useState } from 'react';
import { apiClient, WebSocketEvent, WebSocketEventHandler } from '../lib/api-client';

export interface UseWebSocketOptions {
  autoConnect?: boolean;
  subscribeToAll?: boolean;
  experimentIds?: string[];
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    autoConnect = true,
    subscribeToAll = false,
    experimentIds = []
  } = options;

  const eventHandlersRef = useRef<Map<string, WebSocketEventHandler[]>>(new Map());
  const isConnectedRef = useRef(false);
  const [connectionStats, setConnectionStats] = useState<any>(null);
  const [sessionInfo, setSessionInfo] = useState<any>(null);

  // Connect to WebSocket
  const connect = useCallback(async () => {
    if (!apiClient.isAuthenticated()) {
      console.warn('Cannot connect WebSocket: not authenticated');
      return false;
    }

    try {
      await apiClient.connectWebSocket();
      isConnectedRef.current = true;

      // Subscribe to events based on options
      if (subscribeToAll) {
        apiClient.subscribeToAllEvents();
      }

      experimentIds.forEach(id => {
        apiClient.subscribeToExperiment(id);
      });

      // Request session info
      apiClient.sendWebSocketMessage({
        type: 'get_session_info'
      });

      console.log('WebSocket connected successfully');
      return true;
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      isConnectedRef.current = false;
      return false;
    }
  }, [subscribeToAll, experimentIds]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    apiClient.disconnectWebSocket();
    isConnectedRef.current = false;
    console.log('WebSocket disconnected');
  }, []);

  // Subscribe to specific event type
  const addEventListener = useCallback((eventType: string, handler: WebSocketEventHandler) => {
    if (!eventHandlersRef.current.has(eventType)) {
      eventHandlersRef.current.set(eventType, []);
    }
    eventHandlersRef.current.get(eventType)!.push(handler);
    apiClient.onWebSocketEvent(eventType, handler);
  }, []);

  // Unsubscribe from specific event type
  const removeEventListener = useCallback((eventType: string, handler: WebSocketEventHandler) => {
    const handlers = eventHandlersRef.current.get(eventType);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
    apiClient.offWebSocketEvent(eventType, handler);
  }, []);

  // Subscribe to experiment updates
  const subscribeToExperiment = useCallback((experimentId: string) => {
    apiClient.subscribeToExperiment(experimentId);
  }, []);

  // Unsubscribe from experiment updates
  const unsubscribeFromExperiment = useCallback((experimentId: string) => {
    apiClient.unsubscribeFromExperiment(experimentId);
  }, []);

  // Check if WebSocket is connected
  const isConnected = useCallback(() => {
    return apiClient.isWebSocketConnected();
  }, []);

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect && apiClient.isAuthenticated()) {
      connect();
    }

    // Cleanup on unmount
    return () => {
      // Remove all event listeners
      eventHandlersRef.current.forEach((handlers, eventType) => {
        handlers.forEach(handler => {
          apiClient.offWebSocketEvent(eventType, handler);
        });
      });
      eventHandlersRef.current.clear();

      if (isConnectedRef.current) {
        disconnect();
      }
    };
  }, [autoConnect, connect, disconnect]);

  // Update user preferences
  const updatePreferences = useCallback((preferences: Record<string, any>) => {
    apiClient.sendWebSocketMessage({
      type: 'update_preferences',
      preferences
    });
  }, []);

  // Get connection statistics (admin only)
  const getConnectionStats = useCallback(() => {
    apiClient.sendWebSocketMessage({
      type: 'get_connection_stats'
    });
  }, []);

  // Handle system messages
  useEffect(() => {
    const handleSystemMessage = (event: WebSocketEvent) => {
      if (event.event_type === 'session_info') {
        setSessionInfo(event.data);
      } else if (event.event_type === 'connection_stats') {
        setConnectionStats(event.data);
      }
    };

    addEventListener('session_info', handleSystemMessage);
    addEventListener('connection_stats', handleSystemMessage);

    return () => {
      removeEventListener('session_info', handleSystemMessage);
      removeEventListener('connection_stats', handleSystemMessage);
    };
  }, [addEventListener, removeEventListener]);

  return {
    connect,
    disconnect,
    addEventListener,
    removeEventListener,
    subscribeToExperiment,
    unsubscribeFromExperiment,
    isConnected,
    updatePreferences,
    getConnectionStats,
    connectionStats,
    sessionInfo
  };
}

// Specific hooks for common use cases

export function useExperimentUpdates(experimentId: string | null) {
  const { addEventListener, removeEventListener, subscribeToExperiment, unsubscribeFromExperiment } = useWebSocket({
    autoConnect: true,
    experimentIds: experimentId ? [experimentId] : []
  });

  const onExperimentProgress = useCallback((handler: (data: any) => void) => {
    const eventHandler: WebSocketEventHandler = (event) => {
      if (event.event_type === 'experiment_progress' && 
          (!experimentId || event.experiment_id === experimentId)) {
        handler(event.data);
      }
    };
    addEventListener('experiment_progress', eventHandler);
    return () => removeEventListener('experiment_progress', eventHandler);
  }, [experimentId, addEventListener, removeEventListener]);

  const onExperimentCompleted = useCallback((handler: (data: any) => void) => {
    const eventHandler: WebSocketEventHandler = (event) => {
      if (event.event_type === 'experiment_completed' && 
          (!experimentId || event.experiment_id === experimentId)) {
        handler(event.data);
      }
    };
    addEventListener('experiment_completed', eventHandler);
    return () => removeEventListener('experiment_completed', eventHandler);
  }, [experimentId, addEventListener, removeEventListener]);

  const onExperimentFailed = useCallback((handler: (data: any) => void) => {
    const eventHandler: WebSocketEventHandler = (event) => {
      if (event.event_type === 'experiment_failed' && 
          (!experimentId || event.experiment_id === experimentId)) {
        handler(event.data);
      }
    };
    addEventListener('experiment_failed', eventHandler);
    return () => removeEventListener('experiment_failed', eventHandler);
  }, [experimentId, addEventListener, removeEventListener]);

  const onExperimentStarted = useCallback((handler: (data: any) => void) => {
    const eventHandler: WebSocketEventHandler = (event) => {
      if (event.event_type === 'experiment_started' && 
          (!experimentId || event.experiment_id === experimentId)) {
        handler(event.data);
      }
    };
    addEventListener('experiment_started', eventHandler);
    return () => removeEventListener('experiment_started', eventHandler);
  }, [experimentId, addEventListener, removeEventListener]);

  return {
    onExperimentProgress,
    onExperimentCompleted,
    onExperimentFailed,
    onExperimentStarted,
    subscribeToExperiment,
    unsubscribeFromExperiment
  };
}

export function useResourceUpdates() {
  const { addEventListener, removeEventListener } = useWebSocket({
    autoConnect: true,
    subscribeToAll: true
  });

  const onResourceUpdate = useCallback((handler: (data: any) => void) => {
    const eventHandler: WebSocketEventHandler = (event) => {
      if (event.event_type === 'resource_update') {
        handler(event.data);
      }
    };
    addEventListener('resource_update', eventHandler);
    return () => removeEventListener('resource_update', eventHandler);
  }, [addEventListener, removeEventListener]);

  const onJobQueued = useCallback((handler: (data: any) => void) => {
    const eventHandler: WebSocketEventHandler = (event) => {
      if (event.event_type === 'job_queued') {
        handler(event.data);
      }
    };
    addEventListener('job_queued', eventHandler);
    return () => removeEventListener('job_queued', eventHandler);
  }, [addEventListener, removeEventListener]);

  const onJobStarted = useCallback((handler: (data: any) => void) => {
    const eventHandler: WebSocketEventHandler = (event) => {
      if (event.event_type === 'job_started') {
        handler(event.data);
      }
    };
    addEventListener('job_started', eventHandler);
    return () => removeEventListener('job_started', eventHandler);
  }, [addEventListener, removeEventListener]);

  const onJobCompleted = useCallback((handler: (data: any) => void) => {
    const eventHandler: WebSocketEventHandler = (event) => {
      if (event.event_type === 'job_completed') {
        handler(event.data);
      }
    };
    addEventListener('job_completed', eventHandler);
    return () => removeEventListener('job_completed', eventHandler);
  }, [addEventListener, removeEventListener]);

  return {
    onResourceUpdate,
    onJobQueued,
    onJobStarted,
    onJobCompleted
  };
}

export function useNotifications() {
  const { addEventListener, removeEventListener } = useWebSocket({
    autoConnect: true,
    subscribeToAll: true
  });

  const onNotification = useCallback((handler: (data: { message: string; level: string }) => void) => {
    const eventHandler: WebSocketEventHandler = (event) => {
      if (event.event_type === 'notification') {
        handler(event.data);
      }
    };
    addEventListener('notification', eventHandler);
    return () => removeEventListener('notification', eventHandler);
  }, [addEventListener, removeEventListener]);

  return {
    onNotification
  };
}