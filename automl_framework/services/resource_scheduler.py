"""
Resource Scheduler for managing GPU and compute allocation in AutoML Framework.

This module provides the ResourceScheduler class that handles job queuing,
resource allocation, monitoring, and fair sharing across multiple experiments.
"""

import asyncio
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable
import psutil
import subprocess

from automl_framework.core.interfaces import IResourceScheduler
from automl_framework.core.exceptions import ResourceError

logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"


class JobStatus(Enum):
    """Job execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResourceRequirement:
    """Represents resource requirements for a job."""
    cpu_cores: int = 1
    memory_gb: float = 1.0
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    storage_gb: float = 1.0
    estimated_duration_minutes: int = 60
    
    def validate(self) -> None:
        """Validate resource requirements."""
        if self.cpu_cores <= 0:
            raise ValueError("CPU cores must be positive")
        if self.memory_gb <= 0:
            raise ValueError("Memory must be positive")
        if self.gpu_count < 0:
            raise ValueError("GPU count cannot be negative")
        if self.gpu_memory_gb < 0:
            raise ValueError("GPU memory cannot be negative")
        if self.storage_gb <= 0:
            raise ValueError("Storage must be positive")
        if self.estimated_duration_minutes <= 0:
            raise ValueError("Estimated duration must be positive")


@dataclass
class ResourceAllocation:
    """Represents allocated resources for a job."""
    job_id: str
    cpu_cores: List[int] = field(default_factory=list)
    memory_gb: float = 0.0
    gpu_ids: List[int] = field(default_factory=list)
    gpu_memory_gb: float = 0.0
    storage_path: str = ""
    allocated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert allocation to dictionary."""
        return {
            'job_id': self.job_id,
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'gpu_ids': self.gpu_ids,
            'gpu_memory_gb': self.gpu_memory_gb,
            'storage_path': self.storage_path,
            'allocated_at': self.allocated_at.isoformat()
        }


@dataclass
class ScheduledJob:
    """Represents a job in the scheduler queue."""
    job_id: str
    user_id: str
    requirements: ResourceRequirement
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.QUEUED
    submitted_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    allocation: Optional[ResourceAllocation] = None
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_wait_time(self) -> timedelta:
        """Get current wait time for the job."""
        if self.started_at:
            return self.started_at - self.submitted_at
        return datetime.now() - self.submitted_at
    
    def get_execution_time(self) -> Optional[timedelta]:
        """Get execution time if job is running or completed."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.now()
        return end_time - self.started_at


@dataclass
class SystemResources:
    """Represents current system resource status."""
    total_cpu_cores: int
    available_cpu_cores: int
    total_memory_gb: float
    available_memory_gb: float
    total_gpu_count: int
    available_gpu_count: int
    gpu_info: List[Dict[str, Any]] = field(default_factory=list)
    total_storage_gb: float = 0.0
    available_storage_gb: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system resources to dictionary."""
        return {
            'total_cpu_cores': self.total_cpu_cores,
            'available_cpu_cores': self.available_cpu_cores,
            'total_memory_gb': self.total_memory_gb,
            'available_memory_gb': self.available_memory_gb,
            'total_gpu_count': self.total_gpu_count,
            'available_gpu_count': self.available_gpu_count,
            'gpu_info': self.gpu_info,
            'total_storage_gb': self.total_storage_gb,
            'available_storage_gb': self.available_storage_gb,
            'last_updated': self.last_updated.isoformat()
        }


class ResourceScheduler(IResourceScheduler):
    """
    Manages computational resource allocation and job scheduling.
    
    Provides GPU and CPU allocation, job queuing with priority management,
    resource monitoring, and fair sharing across multiple users.
    """
    
    def __init__(self, 
                 max_concurrent_jobs: int = 10,
                 resource_check_interval: int = 30,
                 enable_fair_sharing: bool = True):
        """
        Initialize the resource scheduler.
        
        Args:
            max_concurrent_jobs: Maximum number of concurrent jobs
            resource_check_interval: Interval in seconds for resource monitoring
            enable_fair_sharing: Whether to enable fair sharing between users
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.resource_check_interval = resource_check_interval
        self.enable_fair_sharing = enable_fair_sharing
        
        # Job management
        self.job_queue: deque[ScheduledJob] = deque()
        self.running_jobs: Dict[str, ScheduledJob] = {}
        self.completed_jobs: Dict[str, ScheduledJob] = {}
        self.job_history: List[ScheduledJob] = []
        
        # Resource tracking
        self.system_resources: Optional[SystemResources] = None
        self.allocated_resources: Dict[str, ResourceAllocation] = {}
        
        # User tracking for fair sharing
        self.user_job_counts: Dict[str, int] = {}
        self.user_resource_usage: Dict[str, Dict[str, float]] = {}
        
        # Threading
        self._lock = threading.RLock()
        self._stop_monitoring = threading.Event()
        self._monitor_thread = None
        
        # Callbacks
        self.job_callbacks: Dict[str, List[Callable]] = {}
        
        # WebSocket event broadcasting
        self._websocket_enabled = False
        self._websocket_manager = None
        
        # Resource monitoring for WebSocket updates
        self._resource_update_interval = 10  # seconds
        self._resource_update_task = None
        self._should_start_monitoring = False
        
        # Initialize system resources
        self._update_system_resources()
        
        # Start monitoring thread
        self._start_monitoring()
        
        logger.info(f"ResourceScheduler initialized with max {max_concurrent_jobs} concurrent jobs")
    
    def allocate_resources(self, job_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allocate computational resources for a job.
        
        Args:
            job_requirements: Dictionary containing resource requirements
            
        Returns:
            Dict[str, Any]: Allocation details or error information
            
        Raises:
            ResourceError: If resource allocation fails
        """
        try:
            # Parse job requirements
            requirements = self._parse_job_requirements(job_requirements)
            requirements.validate()
            
            # Create scheduled job
            job_id = job_requirements.get('job_id', str(uuid.uuid4()))
            user_id = job_requirements.get('user_id', 'default')
            priority = JobPriority(job_requirements.get('priority', JobPriority.NORMAL.value))
            callback = job_requirements.get('callback')
            metadata = job_requirements.get('metadata', {})
            
            job = ScheduledJob(
                job_id=job_id,
                user_id=user_id,
                requirements=requirements,
                priority=priority,
                callback=callback,
                metadata=metadata
            )
            
            with self._lock:
                # Check if resources are immediately available
                if self._can_allocate_immediately(job):
                    allocation = self._allocate_job_resources(job)
                    job.allocation = allocation
                    job.status = JobStatus.RUNNING
                    job.started_at = datetime.now()
                    
                    self.running_jobs[job_id] = job
                    self._update_user_tracking(user_id, 1)
                    
                    logger.info(f"Immediately allocated resources for job {job_id}")
                    return allocation.to_dict()
                else:
                    # Add to queue
                    self._add_to_queue(job)
                    estimated_wait = self._estimate_wait_time(job)
                    
                    logger.info(f"Job {job_id} queued with estimated wait time: {estimated_wait}")
                    
                    # Broadcast job queued event
                    asyncio.create_task(self._broadcast_job_event(
                        'queued',
                        job_id,
                        {
                            'status': 'queued',
                            'estimated_wait_minutes': estimated_wait,
                            'queue_position': self._get_queue_position(job_id),
                            'requirements': asdict(requirements)
                        }
                    ))
                    
                    return {
                        'job_id': job_id,
                        'status': 'queued',
                        'estimated_wait_minutes': estimated_wait,
                        'queue_position': self._get_queue_position(job_id)
                    }
                    
        except Exception as e:
            logger.error(f"Failed to allocate resources: {e}")
            raise ResourceError(f"Resource allocation failed: {e}")
    
    def release_resources(self, job_id: str) -> bool:
        """
        Release resources allocated to a job.
        
        Args:
            job_id: ID of the job to release resources for
            
        Returns:
            bool: True if resources were released successfully
        """
        try:
            with self._lock:
                # Check running jobs
                if job_id in self.running_jobs:
                    job = self.running_jobs[job_id]
                    
                    # Release allocated resources
                    if job.allocation:
                        self._release_job_resources(job.allocation)
                        del self.allocated_resources[job_id]
                    
                    # Update job status
                    job.status = JobStatus.COMPLETED
                    job.completed_at = datetime.now()
                    
                    # Move to completed jobs
                    self.completed_jobs[job_id] = job
                    del self.running_jobs[job_id]
                    
                    # Update user tracking
                    self._update_user_tracking(job.user_id, -1)
                    
                    # Add to history
                    self.job_history.append(job)
                    
                    # Execute callback if provided
                    if job.callback:
                        try:
                            job.callback(job_id, 'completed')
                        except Exception as e:
                            logger.error(f"Job callback failed for {job_id}: {e}")
                    
                    logger.info(f"Released resources for job {job_id}")
                    
                    # Broadcast job completed event
                    asyncio.create_task(self._broadcast_job_event(
                        'completed',
                        job_id,
                        {
                            'status': 'completed',
                            'completed_at': job.completed_at.isoformat(),
                            'execution_time': str(job.get_execution_time()) if job.get_execution_time() else None
                        }
                    ))
                    
                    # Broadcast resource update
                    asyncio.create_task(self._broadcast_resource_update())
                    
                    # Try to schedule queued jobs
                    self._schedule_queued_jobs()
                    
                    return True
                
                # Check if job is in queue and remove it
                elif self._remove_from_queue(job_id):
                    logger.info(f"Removed job {job_id} from queue")
                    return True
                
                logger.warning(f"Job {job_id} not found for resource release")
                return False
                
        except Exception as e:
            logger.error(f"Failed to release resources for job {job_id}: {e}")
            return False
    
    def get_resource_status(self) -> Dict[str, Any]:
        """
        Get current resource utilization status.
        
        Returns:
            Dict[str, Any]: Current resource status and utilization
        """
        with self._lock:
            self._update_system_resources()
            
            return {
                'system_resources': self.system_resources.to_dict() if self.system_resources else {},
                'running_jobs': len(self.running_jobs),
                'queued_jobs': len(self.job_queue),
                'total_jobs_completed': len(self.completed_jobs),
                'resource_utilization': self._calculate_resource_utilization(),
                'user_statistics': self._get_user_statistics(),
                'queue_statistics': self._get_queue_statistics()
            }
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        with self._lock:
            # Check running jobs
            if job_id in self.running_jobs:
                job = self.running_jobs[job_id]
                return {
                    'job_id': job_id,
                    'status': job.status.value,
                    'user_id': job.user_id,
                    'priority': job.priority.value,
                    'submitted_at': job.submitted_at.isoformat(),
                    'started_at': job.started_at.isoformat() if job.started_at else None,
                    'execution_time': str(job.get_execution_time()) if job.get_execution_time() else None,
                    'allocation': job.allocation.to_dict() if job.allocation else None
                }
            
            # Check completed jobs
            if job_id in self.completed_jobs:
                job = self.completed_jobs[job_id]
                return {
                    'job_id': job_id,
                    'status': job.status.value,
                    'user_id': job.user_id,
                    'priority': job.priority.value,
                    'submitted_at': job.submitted_at.isoformat(),
                    'started_at': job.started_at.isoformat() if job.started_at else None,
                    'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                    'execution_time': str(job.get_execution_time()) if job.get_execution_time() else None
                }
            
            # Check queue
            for job in self.job_queue:
                if job.job_id == job_id:
                    return {
                        'job_id': job_id,
                        'status': job.status.value,
                        'user_id': job.user_id,
                        'priority': job.priority.value,
                        'submitted_at': job.submitted_at.isoformat(),
                        'wait_time': str(job.get_wait_time()),
                        'queue_position': self._get_queue_position(job_id),
                        'estimated_wait_minutes': self._estimate_wait_time(job)
                    }
            
            return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job."""
        try:
            with self._lock:
                # Check running jobs
                if job_id in self.running_jobs:
                    job = self.running_jobs[job_id]
                    
                    # Release resources
                    if job.allocation:
                        self._release_job_resources(job.allocation)
                        del self.allocated_resources[job_id]
                    
                    # Update status
                    job.status = JobStatus.CANCELLED
                    job.completed_at = datetime.now()
                    
                    # Move to completed
                    self.completed_jobs[job_id] = job
                    del self.running_jobs[job_id]
                    
                    # Update user tracking
                    self._update_user_tracking(job.user_id, -1)
                    
                    logger.info(f"Cancelled running job {job_id}")
                    
                    # Try to schedule queued jobs
                    self._schedule_queued_jobs()
                    
                    return True
                
                # Check queue
                elif self._remove_from_queue(job_id):
                    logger.info(f"Cancelled queued job {job_id}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def set_job_priority(self, job_id: str, priority: int) -> bool:
        """Set priority for a queued job."""
        try:
            with self._lock:
                for job in self.job_queue:
                    if job.job_id == job_id:
                        job.priority = JobPriority(priority)
                        # Re-sort queue based on new priority
                        self._sort_queue()
                        logger.info(f"Updated priority for job {job_id} to {priority}")
                        return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to set priority for job {job_id}: {e}")
            return False
    
    def add_job_callback(self, job_id: str, callback: Callable):
        """Add a callback for job status changes."""
        if job_id not in self.job_callbacks:
            self.job_callbacks[job_id] = []
        self.job_callbacks[job_id].append(callback)
    
    def enable_websocket_events(self, websocket_manager=None):
        """Enable WebSocket event broadcasting."""
        self._websocket_enabled = True
        if websocket_manager:
            self._websocket_manager = websocket_manager
        
        # Mark that we want to start resource monitoring
        # The actual task will be created when the event loop is available
        self._should_start_monitoring = True
    
    def start_async_monitoring(self):
        """Start async resource monitoring task if event loop is available."""
        if self._should_start_monitoring and not self._resource_update_task:
            try:
                self._resource_update_task = asyncio.create_task(self._periodic_resource_updates())
                logger.info("Started async resource monitoring task")
            except RuntimeError as e:
                if "no running event loop" in str(e):
                    logger.debug("Event loop not available yet, will start monitoring later")
                else:
                    logger.error(f"Failed to start async monitoring: {e}")
    
    async def _broadcast_job_event(self, event_type: str, job_id: str, job_data: Dict[str, Any]):
        """Broadcast job event via WebSocket if enabled."""
        if not self._websocket_enabled:
            return
        
        try:
            # Import here to avoid circular imports
            from automl_framework.api.routes.websocket import (
                broadcast_job_queued,
                broadcast_job_started,
                broadcast_job_completed
            )
            
            # Get user ID from job
            job = None
            for j in list(self.running_jobs.values()) + list(self.completed_jobs.values()) + list(self.job_queue):
                if j.job_id == job_id:
                    job = j
                    break
            
            user_id = job.user_id if job else 'unknown'
            
            if event_type == 'queued':
                await broadcast_job_queued(job_id, job_data, user_id)
            elif event_type == 'started':
                await broadcast_job_started(job_id, job_data, user_id)
            elif event_type == 'completed':
                await broadcast_job_completed(job_id, job_data, user_id)
                
        except Exception as e:
            logger.error(f"Failed to broadcast job WebSocket event: {e}")
    
    async def _broadcast_resource_update(self):
        """Broadcast resource status update via WebSocket if enabled."""
        if not self._websocket_enabled:
            return
        
        try:
            # Import here to avoid circular imports
            from automl_framework.api.routes.websocket import broadcast_resource_update
            
            resource_status = self.get_resource_status()
            await broadcast_resource_update(resource_status)
            
        except Exception as e:
            logger.error(f"Failed to broadcast resource WebSocket event: {e}")
    
    async def _periodic_resource_updates(self):
        """Periodically broadcast resource status updates."""
        while self._websocket_enabled:
            try:
                await asyncio.sleep(self._resource_update_interval)
                await self._broadcast_resource_update()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic resource updates: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    def _parse_job_requirements(self, requirements: Dict[str, Any]) -> ResourceRequirement:
        """Parse job requirements from dictionary."""
        return ResourceRequirement(
            cpu_cores=requirements.get('cpu_cores', 1),
            memory_gb=requirements.get('memory_gb', 1.0),
            gpu_count=requirements.get('gpu_count', 0),
            gpu_memory_gb=requirements.get('gpu_memory_gb', 0.0),
            storage_gb=requirements.get('storage_gb', 1.0),
            estimated_duration_minutes=requirements.get('estimated_duration_minutes', 60)
        )
    
    def _can_allocate_immediately(self, job: ScheduledJob) -> bool:
        """Check if job can be allocated resources immediately."""
        if len(self.running_jobs) >= self.max_concurrent_jobs:
            logger.debug(f"Cannot allocate job {job.job_id}: max concurrent jobs reached")
            return False
        
        if not self.system_resources:
            logger.debug(f"Cannot allocate job {job.job_id}: no system resources")
            return False
        
        req = job.requirements
        sys_res = self.system_resources
        
        # Check CPU cores
        if req.cpu_cores > sys_res.available_cpu_cores:
            logger.debug(f"Cannot allocate job {job.job_id}: insufficient CPU cores ({req.cpu_cores} > {sys_res.available_cpu_cores})")
            return False
        
        # Check memory
        if req.memory_gb > sys_res.available_memory_gb:
            logger.debug(f"Cannot allocate job {job.job_id}: insufficient memory ({req.memory_gb} > {sys_res.available_memory_gb})")
            return False
        
        # Check GPU
        if req.gpu_count > sys_res.available_gpu_count:
            logger.debug(f"Cannot allocate job {job.job_id}: insufficient GPUs ({req.gpu_count} > {sys_res.available_gpu_count})")
            return False
        
        # Check fair sharing constraints
        if self.enable_fair_sharing and not self._check_fair_sharing(job):
            logger.debug(f"Cannot allocate job {job.job_id}: fair sharing constraints")
            return False
        
        logger.debug(f"Can allocate job {job.job_id} immediately")
        return True
    
    def _allocate_job_resources(self, job: ScheduledJob) -> ResourceAllocation:
        """Allocate specific resources for a job."""
        req = job.requirements
        
        # Allocate CPU cores (simplified - just track count)
        cpu_cores = list(range(req.cpu_cores))
        
        # Allocate GPUs
        gpu_ids = []
        if req.gpu_count > 0:
            available_gpus = [i for i in range(self.system_resources.total_gpu_count)
                            if i not in [gpu_id for alloc in self.allocated_resources.values() 
                                       for gpu_id in alloc.gpu_ids]]
            gpu_ids = available_gpus[:req.gpu_count]
        
        # Create allocation
        allocation = ResourceAllocation(
            job_id=job.job_id,
            cpu_cores=cpu_cores,
            memory_gb=req.memory_gb,
            gpu_ids=gpu_ids,
            gpu_memory_gb=req.gpu_memory_gb,
            storage_path=f"/tmp/automl_job_{job.job_id}"
        )
        
        # Update system resources
        self.system_resources.available_cpu_cores -= req.cpu_cores
        self.system_resources.available_memory_gb -= req.memory_gb
        self.system_resources.available_gpu_count -= req.gpu_count
        
        # Track allocation
        self.allocated_resources[job.job_id] = allocation
        
        return allocation
    
    def _release_job_resources(self, allocation: ResourceAllocation):
        """Release resources from an allocation."""
        if not self.system_resources:
            return
        
        # Find the job to get requirements
        job = None
        for j in list(self.running_jobs.values()) + list(self.completed_jobs.values()):
            if j.job_id == allocation.job_id:
                job = j
                break
        
        if job:
            req = job.requirements
            self.system_resources.available_cpu_cores += req.cpu_cores
            self.system_resources.available_memory_gb += req.memory_gb
            self.system_resources.available_gpu_count += req.gpu_count
    
    def _add_to_queue(self, job: ScheduledJob):
        """Add job to the priority queue."""
        self.job_queue.append(job)
        self._sort_queue()
    
    def _sort_queue(self):
        """Sort queue by priority and submission time."""
        self.job_queue = deque(sorted(
            self.job_queue,
            key=lambda j: (-j.priority.value, j.submitted_at)
        ))
    
    def _remove_from_queue(self, job_id: str) -> bool:
        """Remove job from queue."""
        for i, job in enumerate(self.job_queue):
            if job.job_id == job_id:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                self.completed_jobs[job_id] = job
                del self.job_queue[i]
                return True
        return False
    
    def _get_queue_position(self, job_id: str) -> int:
        """Get position of job in queue."""
        for i, job in enumerate(self.job_queue):
            if job.job_id == job_id:
                return i + 1
        return -1
    
    def _estimate_wait_time(self, job: ScheduledJob) -> int:
        """Estimate wait time for a job in minutes."""
        position = self._get_queue_position(job.job_id)
        if position <= 0:
            return 0
        
        # Simple estimation based on average job duration and queue position
        avg_duration = 60  # Default 60 minutes
        if self.job_history:
            completed_durations = [
                j.get_execution_time().total_seconds() / 60
                for j in self.job_history[-10:]  # Last 10 jobs
                if j.get_execution_time()
            ]
            if completed_durations:
                avg_duration = sum(completed_durations) / len(completed_durations)
        
        # Estimate based on position and available slots
        available_slots = max(1, self.max_concurrent_jobs - len(self.running_jobs))
        estimated_wait = (position - 1) // available_slots * avg_duration
        
        return int(estimated_wait)
    
    def _schedule_queued_jobs(self):
        """Try to schedule jobs from the queue."""
        scheduled_count = 0
        jobs_to_remove = []
        
        for job in list(self.job_queue):
            if len(self.running_jobs) >= self.max_concurrent_jobs:
                break
            
            if self._can_allocate_immediately(job):
                try:
                    allocation = self._allocate_job_resources(job)
                    job.allocation = allocation
                    job.status = JobStatus.RUNNING
                    job.started_at = datetime.now()
                    
                    self.running_jobs[job.job_id] = job
                    self._update_user_tracking(job.user_id, 1)
                    jobs_to_remove.append(job)
                    scheduled_count += 1
                    
                    # Execute callback if provided
                    if job.callback:
                        try:
                            job.callback(job.job_id, 'started')
                        except Exception as e:
                            logger.error(f"Job callback failed for {job.job_id}: {e}")
                    
                    logger.info(f"Scheduled job {job.job_id} from queue")
                    
                    # Broadcast job started event
                    asyncio.create_task(self._broadcast_job_event(
                        'started',
                        job.job_id,
                        {
                            'status': 'running',
                            'started_at': job.started_at.isoformat(),
                            'allocation': allocation.to_dict()
                        }
                    ))
                    
                except Exception as e:
                    logger.error(f"Failed to schedule job {job.job_id}: {e}")
        
        # Remove scheduled jobs from queue
        for job in jobs_to_remove:
            self.job_queue.remove(job)
        
        if scheduled_count > 0:
            logger.info(f"Scheduled {scheduled_count} jobs from queue")
    
    def _check_fair_sharing(self, job: ScheduledJob) -> bool:
        """Check if job allocation respects fair sharing constraints."""
        if not self.enable_fair_sharing:
            return True
        
        user_id = job.user_id
        current_user_jobs = self.user_job_counts.get(user_id, 0)
        
        # Simple fair sharing: limit users to reasonable number of concurrent jobs
        max_user_jobs = max(1, self.max_concurrent_jobs // 3)  # Allow up to 1/3 of total slots per user
        
        return current_user_jobs < max_user_jobs
    
    def _update_user_tracking(self, user_id: str, delta: int):
        """Update user job count tracking."""
        if user_id not in self.user_job_counts:
            self.user_job_counts[user_id] = 0
        self.user_job_counts[user_id] += delta
        self.user_job_counts[user_id] = max(0, self.user_job_counts[user_id])
    
    def _update_system_resources(self):
        """Update current system resource information."""
        try:
            # Get CPU information
            cpu_count = psutil.cpu_count(logical=True)
            cpu_percent = psutil.cpu_percent(interval=1)
            available_cpu = max(1, int(cpu_count * (100 - cpu_percent) / 100))
            
            # Get memory information
            memory = psutil.virtual_memory()
            total_memory_gb = memory.total / (1024**3)
            available_memory_gb = memory.available / (1024**3)
            
            # Get GPU information (simplified)
            gpu_info = self._get_gpu_info()
            total_gpus = len(gpu_info)
            available_gpus = sum(1 for gpu in gpu_info if gpu.get('available', True))
            
            # Get storage information
            disk = psutil.disk_usage('/')
            total_storage_gb = disk.total / (1024**3)
            available_storage_gb = disk.free / (1024**3)
            
            self.system_resources = SystemResources(
                total_cpu_cores=cpu_count,
                available_cpu_cores=available_cpu,
                total_memory_gb=total_memory_gb,
                available_memory_gb=available_memory_gb,
                total_gpu_count=total_gpus,
                available_gpu_count=available_gpus,
                gpu_info=gpu_info,
                total_storage_gb=total_storage_gb,
                available_storage_gb=available_storage_gb
            )
            
        except Exception as e:
            logger.error(f"Failed to update system resources: {e}")
    
    def _get_gpu_info(self) -> List[Dict[str, Any]]:
        """Get GPU information using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                gpu_info = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(', ')
                        if len(parts) >= 5:
                            gpu_info.append({
                                'index': int(parts[0]),
                                'name': parts[1],
                                'memory_total_mb': int(parts[2]),
                                'memory_used_mb': int(parts[3]),
                                'utilization_percent': int(parts[4]),
                                'available': int(parts[4]) < 80  # Consider available if < 80% utilized
                            })
                return gpu_info
            else:
                # No GPUs or nvidia-smi not available
                return []
                
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f"Could not get GPU info: {e}")
            return []
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate current resource utilization percentages."""
        if not self.system_resources:
            return {}
        
        sys_res = self.system_resources
        
        cpu_util = ((sys_res.total_cpu_cores - sys_res.available_cpu_cores) / 
                   max(1, sys_res.total_cpu_cores)) * 100
        
        memory_util = ((sys_res.total_memory_gb - sys_res.available_memory_gb) / 
                      max(1, sys_res.total_memory_gb)) * 100
        
        gpu_util = 0.0
        if sys_res.total_gpu_count > 0:
            gpu_util = ((sys_res.total_gpu_count - sys_res.available_gpu_count) / 
                       sys_res.total_gpu_count) * 100
        
        storage_util = ((sys_res.total_storage_gb - sys_res.available_storage_gb) / 
                       max(1, sys_res.total_storage_gb)) * 100
        
        return {
            'cpu_utilization_percent': round(cpu_util, 2),
            'memory_utilization_percent': round(memory_util, 2),
            'gpu_utilization_percent': round(gpu_util, 2),
            'storage_utilization_percent': round(storage_util, 2)
        }
    
    def _get_user_statistics(self) -> Dict[str, Any]:
        """Get statistics about user resource usage."""
        user_stats = {}
        
        for user_id, job_count in self.user_job_counts.items():
            user_stats[user_id] = {
                'running_jobs': job_count,
                'total_completed': len([j for j in self.completed_jobs.values() if j.user_id == user_id]),
                'queued_jobs': len([j for j in self.job_queue if j.user_id == user_id])
            }
        
        return user_stats
    
    def _get_queue_statistics(self) -> Dict[str, Any]:
        """Get statistics about the job queue."""
        if not self.job_queue:
            return {'total_queued': 0, 'average_wait_time_minutes': 0, 'priority_distribution': {}}
        
        priority_dist = {}
        total_wait_time = 0
        
        for job in self.job_queue:
            priority_name = job.priority.name
            priority_dist[priority_name] = priority_dist.get(priority_name, 0) + 1
            total_wait_time += job.get_wait_time().total_seconds() / 60
        
        return {
            'total_queued': len(self.job_queue),
            'average_wait_time_minutes': round(total_wait_time / len(self.job_queue), 2),
            'priority_distribution': priority_dist
        }
    
    def _start_monitoring(self):
        """Start the resource monitoring thread."""
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()
        logger.info("Started resource monitoring thread")
    
    def _monitor_resources(self):
        """Monitor system resources and job status."""
        while not self._stop_monitoring.is_set():
            try:
                with self._lock:
                    # Update system resources
                    self._update_system_resources()
                    
                    # Check for completed jobs (simplified)
                    completed_jobs = []
                    for job_id, job in self.running_jobs.items():
                        # In a real implementation, this would check actual job status
                        # For now, we'll use a simple time-based completion
                        if job.started_at:
                            elapsed = datetime.now() - job.started_at
                            estimated_duration = timedelta(minutes=job.requirements.estimated_duration_minutes)
                            
                            # Simulate job completion after estimated duration
                            if elapsed > estimated_duration:
                                completed_jobs.append(job_id)
                    
                    # Complete jobs that have finished
                    for job_id in completed_jobs:
                        self.release_resources(job_id)
                    
                    # Try to schedule queued jobs
                    if self.job_queue:
                        self._schedule_queued_jobs()
                
                # Sleep until next check
                self._stop_monitoring.wait(self.resource_check_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                self._stop_monitoring.wait(self.resource_check_interval)
    
    def shutdown(self):
        """Shutdown the resource scheduler."""
        logger.info("Shutting down ResourceScheduler")
        
        # Stop monitoring
        self._stop_monitoring.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        # Cancel all running jobs
        with self._lock:
            for job_id in list(self.running_jobs.keys()):
                self.cancel_job(job_id)
            
            # Cancel all queued jobs
            for job in list(self.job_queue):
                self.cancel_job(job.job_id)
        
        logger.info("ResourceScheduler shutdown complete")