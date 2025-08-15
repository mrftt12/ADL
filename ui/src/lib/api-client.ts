/**
 * API client for AutoML Framework backend
 * 
 * This module provides a centralized client for communicating with the AutoML backend API,
 * replacing Supabase calls with direct API calls to our FastAPI backend.
 */

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Types
export interface Dataset {
  id: string;
  name: string;
  description?: string;
  file_path: string;
  file_size: number;
  num_samples?: number;
  num_features?: number;
  target_column?: string;
  data_types: any;
  preprocessing_config: any;
  created_at: string;
}

export interface Experiment {
  id: string;
  name: string;
  status: 'created' | 'running' | 'completed' | 'failed' | 'cancelled';
  created_at: string;
  completed_at?: string;
  progress: {
    data_processing: number;
    architecture_search: number;
    hyperparameter_optimization: number;
    model_training: number;
    model_evaluation: number;
    overall: number;
  };
  results?: {
    best_architecture?: any;
    best_hyperparameters?: any;
    performance_metrics?: any;
    model_path?: string;
  };
}

export interface Project {
  id: string;
  name: string;
  dataset_name: string;
  dataset_id: string;
  created_at: string;
  experiments?: Experiment[];
}

export interface ApiResponse<T> {
  data?: T;
  error?: string;
  message?: string;
}

// WebSocket Event Types
export interface WebSocketEvent {
  event_type: string;
  data: any;
  timestamp: string;
  user_id?: string;
  experiment_id?: string;
}

export type WebSocketEventHandler = (event: WebSocketEvent) => void;

// WebSocket Manager Class
class WebSocketManager {
  private ws: WebSocket | null = null;
  private baseUrl: string;
  private token: string | null = null;
  private eventHandlers: Map<string, WebSocketEventHandler[]> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isConnecting = false;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl.replace('http', 'ws');
  }

  connect(token: string): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      if (this.isConnecting) {
        reject(new Error('Connection already in progress'));
        return;
      }

      this.isConnecting = true;
      this.token = token;
      
      const wsUrl = `${this.baseUrl}/ws/connect?token=${encodeURIComponent(token)}`;
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        resolve();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        this.isConnecting = false;
        this.ws = null;
        
        if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.scheduleReconnect();
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.isConnecting = false;
        reject(error);
      };
    });
  }

  disconnect() {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.eventHandlers.clear();
  }

  subscribe(subscription: string) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'subscribe',
        subscription: subscription
      }));
    }
  }

  unsubscribe(subscription: string) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'unsubscribe',
        subscription: subscription
      }));
    }
  }

  addEventListener(eventType: string, handler: WebSocketEventHandler) {
    if (!this.eventHandlers.has(eventType)) {
      this.eventHandlers.set(eventType, []);
    }
    this.eventHandlers.get(eventType)!.push(handler);
  }

  removeEventListener(eventType: string, handler: WebSocketEventHandler) {
    const handlers = this.eventHandlers.get(eventType);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  private handleMessage(data: any) {
    if (data.type === 'ping') {
      // Respond to ping
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({
          type: 'pong',
          timestamp: data.timestamp
        }));
      }
      return;
    }

    if (data.event_type) {
      // Handle WebSocket events
      const handlers = this.eventHandlers.get(data.event_type);
      if (handlers) {
        handlers.forEach(handler => {
          try {
            handler(data as WebSocketEvent);
          } catch (error) {
            console.error('Error in WebSocket event handler:', error);
          }
        });
      }

      // Also trigger 'all' handlers
      const allHandlers = this.eventHandlers.get('all');
      if (allHandlers) {
        allHandlers.forEach(handler => {
          try {
            handler(data as WebSocketEvent);
          } catch (error) {
            console.error('Error in WebSocket event handler:', error);
          }
        });
      }
    }
  }

  private scheduleReconnect() {
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`Scheduling WebSocket reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);
    
    setTimeout(() => {
      if (this.token) {
        this.connect(this.token).catch(error => {
          console.error('WebSocket reconnect failed:', error);
        });
      }
    }, delay);
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  sendMessage(message: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }
}

// API Client Class
class AutoMLApiClient {
  private baseUrl: string;
  private token: string | null = null;
  private websocketManager: WebSocketManager;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
    this.websocketManager = new WebSocketManager(baseUrl);
    this.loadToken();
  }

  private loadToken() {
    this.token = localStorage.getItem('automl_token');
  }

  private saveToken(token: string) {
    this.token = token;
    localStorage.setItem('automl_token', token);
  }

  private clearToken() {
    this.token = null;
    localStorage.removeItem('automl_token');
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`;
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        return {
          error: errorData.message || errorData.detail || `HTTP ${response.status}`,
        };
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      return {
        error: error instanceof Error ? error.message : 'Network error',
      };
    }
  }

  // Authentication methods
  async signup(userData: {
    username: string;
    email: string;
    password: string;
    confirm_password: string;
  }): Promise<ApiResponse<{
    id: string;
    username: string;
    email: string;
    is_active: boolean;
    is_admin: boolean;
  }>> {
    return this.request<{
      id: string;
      username: string;
      email: string;
      is_active: boolean;
      is_admin: boolean;
    }>('/api/v1/auth/signup', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  }

  async login(username: string, password: string): Promise<ApiResponse<{ access_token: string; token_type: string }>> {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const response = await fetch(`${this.baseUrl}/api/v1/auth/login`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return {
        error: errorData.detail || 'Login failed',
      };
    }

    const data = await response.json();
    this.saveToken(data.access_token);
    
    // Automatically connect WebSocket after successful login
    try {
      await this.connectWebSocket();
    } catch (error) {
      console.warn('Failed to connect WebSocket after login:', error);
    }
    
    return { data };
  }

  async logout(): Promise<ApiResponse<{ message: string }>> {
    const result = await this.request<{ message: string }>('/api/v1/auth/logout', {
      method: 'POST',
    });
    this.clearToken();
    return result;
  }

  async getCurrentUser(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/v1/auth/me');
  }

  // Dataset methods
  async uploadDataset(
    file: File,
    name?: string,
    description?: string
  ): Promise<ApiResponse<any>> {
    const formData = new FormData();
    formData.append('file', file);
    if (name) formData.append('name', name);
    if (description) formData.append('description', description);

    const url = `${this.baseUrl}/api/v1/datasets/upload`;
    const headers: HeadersInit = {};

    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`;
    }

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        return {
          error: errorData.detail || errorData.message || `HTTP ${response.status}`,
        };
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      return {
        error: error instanceof Error ? error.message : 'Upload failed',
      };
    }
  }

  async getDatasets(): Promise<ApiResponse<{ datasets: any[]; total: number }>> {
    return this.request<{ datasets: any[]; total: number }>('/api/v1/datasets');
  }

  async getDataset(datasetId: string): Promise<ApiResponse<any>> {
    return this.request<any>(`/api/v1/datasets/${datasetId}`);
  }

  async analyzeDataset(datasetId: string, targetColumn?: string): Promise<ApiResponse<any>> {
    const params = new URLSearchParams();
    if (targetColumn) params.append('target_column', targetColumn);
    
    const endpoint = `/api/v1/datasets/${datasetId}/analyze${params.toString() ? `?${params.toString()}` : ''}`;
    return this.request<any>(endpoint);
  }

  async deleteDataset(datasetId: string): Promise<ApiResponse<{ message: string }>> {
    return this.request<{ message: string }>(`/api/v1/datasets/${datasetId}`, {
      method: 'DELETE',
    });
  }

  // Experiment methods
  async createExperiment(experimentData: {
    name: string;
    dataset_path?: string;
    task_type?: string;
    data_type?: string;
    target_column?: string;
    config?: any;
  }): Promise<ApiResponse<Experiment>> {
    return this.request<Experiment>('/api/v1/experiments', {
      method: 'POST',
      body: JSON.stringify(experimentData),
    });
  }

  async getExperiments(
    status?: string,
    limit: number = 50,
    offset: number = 0
  ): Promise<ApiResponse<{ experiments: Experiment[]; total: number }>> {
    const params = new URLSearchParams();
    if (status) params.append('status', status);
    params.append('limit', limit.toString());
    params.append('offset', offset.toString());

    return this.request<{ experiments: Experiment[]; total: number }>(
      `/api/v1/experiments?${params.toString()}`
    );
  }

  async getExperiment(experimentId: string): Promise<ApiResponse<Experiment>> {
    return this.request<Experiment>(`/api/v1/experiments/${experimentId}`);
  }

  async startExperiment(experimentId: string): Promise<ApiResponse<{ message: string }>> {
    return this.request<{ message: string }>(`/api/v1/experiments/${experimentId}/start`, {
      method: 'POST',
    });
  }

  async cancelExperiment(experimentId: string): Promise<ApiResponse<{ message: string }>> {
    return this.request<{ message: string }>(`/api/v1/experiments/${experimentId}`, {
      method: 'DELETE',
    });
  }

  async getExperimentProgress(experimentId: string): Promise<ApiResponse<{
    experiment_id: string;
    progress: any;
    timestamp: string;
  }>> {
    return this.request<{
      experiment_id: string;
      progress: any;
      timestamp: string;
    }>(`/api/v1/experiments/${experimentId}/progress`);
  }

  // Resource monitoring methods
  async getResourceStatus(): Promise<ApiResponse<{
    system_resources: any;
    running_jobs: number;
    queued_jobs: number;
    total_jobs_completed: number;
    resource_utilization: any;
  }>> {
    return this.request<{
      system_resources: any;
      running_jobs: number;
      queued_jobs: number;
      total_jobs_completed: number;
      resource_utilization: any;
    }>('/api/v1/resources/status');
  }

  async getResourceMetrics(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/v1/resources/metrics');
  }

  async getJobStatus(jobId: string): Promise<ApiResponse<any>> {
    return this.request<any>(`/api/v1/resources/jobs/${jobId}`);
  }

  // Configuration methods
  async getApiConfig(): Promise<ApiResponse<{
    version: string;
    supported_data_types: string[];
    supported_task_types: string[];
    max_file_size_mb: number;
    supported_file_formats: string[];
  }>> {
    return this.request<{
      version: string;
      supported_data_types: string[];
      supported_task_types: string[];
      max_file_size_mb: number;
      supported_file_formats: string[];
    }>('/api/v1/config');
  }

  // Health check
  async healthCheck(): Promise<ApiResponse<{ status: string; timestamp: string }>> {
    return this.request<{ status: string; timestamp: string }>('/health');
  }

  // WebSocket methods
  async connectWebSocket(): Promise<void> {
    if (!this.token) {
      throw new Error('Authentication required for WebSocket connection');
    }
    return this.websocketManager.connect(this.token);
  }

  disconnectWebSocket() {
    this.websocketManager.disconnect();
  }

  subscribeToExperiment(experimentId: string) {
    this.websocketManager.subscribe(experimentId);
  }

  subscribeToAllEvents() {
    this.websocketManager.subscribe('all');
  }

  unsubscribeFromExperiment(experimentId: string) {
    this.websocketManager.unsubscribe(experimentId);
  }

  unsubscribeFromAllEvents() {
    this.websocketManager.unsubscribe('all');
  }

  onWebSocketEvent(eventType: string, handler: WebSocketEventHandler) {
    this.websocketManager.addEventListener(eventType, handler);
  }

  offWebSocketEvent(eventType: string, handler: WebSocketEventHandler) {
    this.websocketManager.removeEventListener(eventType, handler);
  }

  isWebSocketConnected(): boolean {
    return this.websocketManager.isConnected();
  }

  sendWebSocketMessage(message: any) {
    if (this.websocketManager.isConnected()) {
      this.websocketManager.sendMessage(message);
    }
  }

  // Utility methods
  isAuthenticated(): boolean {
    return !!this.token;
  }

  getToken(): string | null {
    return this.token;
  }
}

// Create and export singleton instance
export const apiClient = new AutoMLApiClient();

// Export the class for testing or custom instances
export { AutoMLApiClient };

// Helper functions for common operations
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const formatDuration = (seconds: number): string => {
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  if (minutes < 60) return `${minutes}m ${remainingSeconds}s`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return `${hours}h ${remainingMinutes}m`;
};

export const getStatusColor = (status: string): string => {
  switch (status.toLowerCase()) {
    case 'completed':
      return 'bg-green-500/10 text-green-600 border-green-500/20';
    case 'running':
      return 'bg-blue-500/10 text-blue-600 border-blue-500/20';
    case 'failed':
      return 'bg-red-500/10 text-red-600 border-red-500/20';
    case 'cancelled':
      return 'bg-yellow-500/10 text-yellow-600 border-yellow-500/20';
    case 'created':
    case 'queued':
      return 'bg-gray-500/10 text-gray-600 border-gray-500/20';
    default:
      return 'bg-gray-500/10 text-gray-600 border-gray-500/20';
  }
};