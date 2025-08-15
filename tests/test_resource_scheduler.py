"""
Unit tests for ResourceScheduler class.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from automl_framework.services.resource_scheduler import (
    ResourceScheduler,
    ResourceRequirement,
    ResourceAllocation,
    ScheduledJob,
    SystemResources,
    JobPriority,
    JobStatus,
    ResourceType
)
from automl_framework.core.exceptions import ResourceError


class TestResourceRequirement:
    """Test cases for ResourceRequirement class."""
    
    def test_init_default(self):
        """Test ResourceRequirement initialization with defaults."""
        req = ResourceRequirement()
        
        assert req.cpu_cores == 1
        assert req.memory_gb == 1.0
        assert req.gpu_count == 0
        assert req.gpu_memory_gb == 0.0
        assert req.storage_gb == 1.0
        assert req.estimated_duration_minutes == 60
    
    def test_init_custom(self):
        """Test ResourceRequirement initialization with custom values."""
        req = ResourceRequirement(
            cpu_cores=4,
            memory_gb=8.0,
            gpu_count=2,
            gpu_memory_gb=16.0,
            storage_gb=100.0,
            estimated_duration_minutes=120
        )
        
        assert req.cpu_cores == 4
        assert req.memory_gb == 8.0
        assert req.gpu_count == 2
        assert req.gpu_memory_gb == 16.0
        assert req.storage_gb == 100.0
        assert req.estimated_duration_minutes == 120
    
    def test_validate_valid(self):
        """Test validation of valid requirements."""
        req = ResourceRequirement(cpu_cores=2, memory_gb=4.0, gpu_count=1)
        req.validate()  # Should not raise
    
    def test_validate_invalid_cpu(self):
        """Test validation with invalid CPU cores."""
        req = ResourceRequirement(cpu_cores=0)
        with pytest.raises(ValueError, match="CPU cores must be positive"):
            req.validate()
    
    def test_validate_invalid_memory(self):
        """Test validation with invalid memory."""
        req = ResourceRequirement(memory_gb=-1.0)
        with pytest.raises(ValueError, match="Memory must be positive"):
            req.validate()
    
    def test_validate_invalid_gpu_count(self):
        """Test validation with invalid GPU count."""
        req = ResourceRequirement(gpu_count=-1)
        with pytest.raises(ValueError, match="GPU count cannot be negative"):
            req.validate()
    
    def test_validate_invalid_duration(self):
        """Test validation with invalid duration."""
        req = ResourceRequirement(estimated_duration_minutes=0)
        with pytest.raises(ValueError, match="Estimated duration must be positive"):
            req.validate()


class TestResourceAllocation:
    """Test cases for ResourceAllocation class."""
    
    def test_init(self):
        """Test ResourceAllocation initialization."""
        allocation = ResourceAllocation(
            job_id='test_job',
            cpu_cores=[0, 1],
            memory_gb=4.0,
            gpu_ids=[0],
            gpu_memory_gb=8.0,
            storage_path='/tmp/test'
        )
        
        assert allocation.job_id == 'test_job'
        assert allocation.cpu_cores == [0, 1]
        assert allocation.memory_gb == 4.0
        assert allocation.gpu_ids == [0]
        assert allocation.gpu_memory_gb == 8.0
        assert allocation.storage_path == '/tmp/test'
        assert allocation.allocated_at is not None
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        allocation = ResourceAllocation(
            job_id='test_job',
            cpu_cores=[0, 1],
            memory_gb=4.0,
            gpu_ids=[0],
            gpu_memory_gb=8.0,
            storage_path='/tmp/test'
        )
        
        result = allocation.to_dict()
        
        assert result['job_id'] == 'test_job'
        assert result['cpu_cores'] == [0, 1]
        assert result['memory_gb'] == 4.0
        assert result['gpu_ids'] == [0]
        assert result['gpu_memory_gb'] == 8.0
        assert result['storage_path'] == '/tmp/test'
        assert 'allocated_at' in result


class TestScheduledJob:
    """Test cases for ScheduledJob class."""
    
    def test_init(self):
        """Test ScheduledJob initialization."""
        req = ResourceRequirement(cpu_cores=2, memory_gb=4.0)
        job = ScheduledJob(
            job_id='test_job',
            user_id='test_user',
            requirements=req,
            priority=JobPriority.HIGH
        )
        
        assert job.job_id == 'test_job'
        assert job.user_id == 'test_user'
        assert job.requirements == req
        assert job.priority == JobPriority.HIGH
        assert job.status == JobStatus.QUEUED
        assert job.submitted_at is not None
        assert job.started_at is None
        assert job.completed_at is None
    
    def test_get_wait_time_not_started(self):
        """Test wait time calculation for job not yet started."""
        req = ResourceRequirement()
        job = ScheduledJob('test_job', 'test_user', req)
        
        wait_time = job.get_wait_time()
        assert isinstance(wait_time, timedelta)
        assert wait_time.total_seconds() >= 0
    
    def test_get_wait_time_started(self):
        """Test wait time calculation for started job."""
        req = ResourceRequirement()
        job = ScheduledJob('test_job', 'test_user', req)
        job.started_at = datetime.now()
        
        wait_time = job.get_wait_time()
        assert isinstance(wait_time, timedelta)
        assert wait_time.total_seconds() >= 0
    
    def test_get_execution_time_not_started(self):
        """Test execution time for job not yet started."""
        req = ResourceRequirement()
        job = ScheduledJob('test_job', 'test_user', req)
        
        exec_time = job.get_execution_time()
        assert exec_time is None
    
    def test_get_execution_time_running(self):
        """Test execution time for running job."""
        req = ResourceRequirement()
        job = ScheduledJob('test_job', 'test_user', req)
        job.started_at = datetime.now() - timedelta(minutes=5)
        
        exec_time = job.get_execution_time()
        assert isinstance(exec_time, timedelta)
        assert exec_time.total_seconds() > 0
    
    def test_get_execution_time_completed(self):
        """Test execution time for completed job."""
        req = ResourceRequirement()
        job = ScheduledJob('test_job', 'test_user', req)
        job.started_at = datetime.now() - timedelta(minutes=10)
        job.completed_at = datetime.now() - timedelta(minutes=5)
        
        exec_time = job.get_execution_time()
        assert isinstance(exec_time, timedelta)
        assert abs(exec_time.total_seconds() - 300) < 10  # ~5 minutes


class TestSystemResources:
    """Test cases for SystemResources class."""
    
    def test_init(self):
        """Test SystemResources initialization."""
        resources = SystemResources(
            total_cpu_cores=8,
            available_cpu_cores=6,
            total_memory_gb=32.0,
            available_memory_gb=24.0,
            total_gpu_count=2,
            available_gpu_count=1
        )
        
        assert resources.total_cpu_cores == 8
        assert resources.available_cpu_cores == 6
        assert resources.total_memory_gb == 32.0
        assert resources.available_memory_gb == 24.0
        assert resources.total_gpu_count == 2
        assert resources.available_gpu_count == 1
        assert resources.last_updated is not None
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        resources = SystemResources(
            total_cpu_cores=8,
            available_cpu_cores=6,
            total_memory_gb=32.0,
            available_memory_gb=24.0,
            total_gpu_count=2,
            available_gpu_count=1
        )
        
        result = resources.to_dict()
        
        assert result['total_cpu_cores'] == 8
        assert result['available_cpu_cores'] == 6
        assert result['total_memory_gb'] == 32.0
        assert result['available_memory_gb'] == 24.0
        assert result['total_gpu_count'] == 2
        assert result['available_gpu_count'] == 1
        assert 'last_updated' in result


class TestResourceScheduler:
    """Test cases for ResourceScheduler class."""
    
    @pytest.fixture
    def scheduler(self):
        """Create ResourceScheduler instance for testing."""
        with patch('automl_framework.services.resource_scheduler.psutil') as mock_psutil:
            # Mock psutil calls
            mock_psutil.cpu_count.return_value = 8
            mock_psutil.cpu_percent.return_value = 0.0
            mock_psutil.virtual_memory.return_value = Mock(
                total=32 * 1024**3,  # 32GB
                available=32 * 1024**3  # 32GB available
            )
            mock_psutil.disk_usage.return_value = Mock(
                total=1000 * 1024**3,  # 1TB
                free=1000 * 1024**3  # 1TB free
            )
            
            # Mock GPU info to return 2 available GPUs
            with patch.object(ResourceScheduler, '_get_gpu_info') as mock_gpu_info:
                mock_gpu_info.return_value = [
                    {'index': 0, 'name': 'Test GPU 0', 'memory_total_mb': 8192, 'memory_used_mb': 1024, 'utilization_percent': 10, 'available': True},
                    {'index': 1, 'name': 'Test GPU 1', 'memory_total_mb': 8192, 'memory_used_mb': 1024, 'utilization_percent': 10, 'available': True}
                ]
                
                scheduler = ResourceScheduler(max_concurrent_jobs=3, resource_check_interval=1)
                
                # Stop the monitoring thread for tests
                scheduler._stop_monitoring.set()
                if scheduler._monitor_thread and scheduler._monitor_thread.is_alive():
                    scheduler._monitor_thread.join(timeout=1)
            yield scheduler
            scheduler.shutdown()
    
    def test_init(self, scheduler):
        """Test ResourceScheduler initialization."""
        assert scheduler.max_concurrent_jobs == 3
        assert scheduler.resource_check_interval == 1
        assert scheduler.enable_fair_sharing is True
        assert len(scheduler.job_queue) == 0
        assert len(scheduler.running_jobs) == 0
        assert len(scheduler.completed_jobs) == 0
    
    def test_allocate_resources_immediate(self, scheduler):
        """Test immediate resource allocation."""
        job_requirements = {
            'job_id': 'test_job_1',
            'user_id': 'test_user',
            'cpu_cores': 2,
            'memory_gb': 4.0,
            'gpu_count': 1,
            'estimated_duration_minutes': 30
        }
        
        result = scheduler.allocate_resources(job_requirements)
        
        assert result['job_id'] == 'test_job_1'
        assert 'cpu_cores' in result
        assert 'memory_gb' in result
        assert 'gpu_ids' in result
        assert len(scheduler.running_jobs) == 1
        assert 'test_job_1' in scheduler.running_jobs
    
    def test_allocate_resources_queued(self, scheduler):
        """Test resource allocation when resources are not immediately available."""
        # Disable fair sharing for this test
        scheduler.enable_fair_sharing = False
        
        # Fill up all concurrent job slots
        for i in range(3):
            job_requirements = {
                'job_id': f'running_job_{i}',
                'user_id': 'test_user',
                'cpu_cores': 1,
                'memory_gb': 1.0
            }
            scheduler.allocate_resources(job_requirements)
        
        # Try to allocate another job - should be queued
        job_requirements = {
            'job_id': 'queued_job',
            'user_id': 'test_user',
            'cpu_cores': 1,
            'memory_gb': 1.0
        }
        
        result = scheduler.allocate_resources(job_requirements)
        
        assert result['job_id'] == 'queued_job'
        assert result['status'] == 'queued'
        assert 'estimated_wait_minutes' in result
        assert 'queue_position' in result
        assert len(scheduler.job_queue) == 1
    
    def test_allocate_resources_insufficient_resources(self, scheduler):
        """Test allocation when insufficient resources are available."""
        job_requirements = {
            'job_id': 'big_job',
            'user_id': 'test_user',
            'cpu_cores': 16,  # More than available
            'memory_gb': 4.0
        }
        
        result = scheduler.allocate_resources(job_requirements)
        
        assert result['status'] == 'queued'
        assert len(scheduler.job_queue) == 1
    
    def test_allocate_resources_invalid_requirements(self, scheduler):
        """Test allocation with invalid requirements."""
        job_requirements = {
            'job_id': 'invalid_job',
            'user_id': 'test_user',
            'cpu_cores': 0,  # Invalid
            'memory_gb': 4.0
        }
        
        with pytest.raises(ResourceError):
            scheduler.allocate_resources(job_requirements)
    
    def test_release_resources_running_job(self, scheduler):
        """Test releasing resources for a running job."""
        # Allocate resources first
        job_requirements = {
            'job_id': 'test_job',
            'user_id': 'test_user',
            'cpu_cores': 2,
            'memory_gb': 4.0
        }
        scheduler.allocate_resources(job_requirements)
        
        # Release resources
        result = scheduler.release_resources('test_job')
        
        assert result is True
        assert len(scheduler.running_jobs) == 0
        assert len(scheduler.completed_jobs) == 1
        assert 'test_job' in scheduler.completed_jobs
    
    def test_release_resources_queued_job(self, scheduler):
        """Test releasing resources for a queued job."""
        # Disable fair sharing for this test
        scheduler.enable_fair_sharing = False
        
        # Fill up concurrent slots and queue a job
        for i in range(3):
            job_requirements = {
                'job_id': f'running_job_{i}',
                'user_id': 'test_user',
                'cpu_cores': 1,
                'memory_gb': 1.0
            }
            scheduler.allocate_resources(job_requirements)
        
        job_requirements = {
            'job_id': 'queued_job',
            'user_id': 'test_user',
            'cpu_cores': 1,
            'memory_gb': 1.0
        }
        scheduler.allocate_resources(job_requirements)
        
        # Release queued job
        result = scheduler.release_resources('queued_job')
        
        assert result is True
        assert len(scheduler.job_queue) == 0
        assert 'queued_job' in scheduler.completed_jobs
    
    def test_release_resources_nonexistent_job(self, scheduler):
        """Test releasing resources for nonexistent job."""
        result = scheduler.release_resources('nonexistent_job')
        assert result is False
    
    def test_get_resource_status(self, scheduler):
        """Test getting resource status."""
        status = scheduler.get_resource_status()
        
        assert 'system_resources' in status
        assert 'running_jobs' in status
        assert 'queued_jobs' in status
        assert 'total_jobs_completed' in status
        assert 'resource_utilization' in status
        assert 'user_statistics' in status
        assert 'queue_statistics' in status
    
    def test_get_job_status_running(self, scheduler):
        """Test getting status of running job."""
        job_requirements = {
            'job_id': 'test_job',
            'user_id': 'test_user',
            'cpu_cores': 1,
            'memory_gb': 1.0
        }
        scheduler.allocate_resources(job_requirements)
        
        status = scheduler.get_job_status('test_job')
        
        assert status is not None
        assert status['job_id'] == 'test_job'
        assert status['status'] == 'running'
        assert status['user_id'] == 'test_user'
        assert 'allocation' in status
    
    def test_get_job_status_queued(self, scheduler):
        """Test getting status of queued job."""
        # Disable fair sharing for this test
        scheduler.enable_fair_sharing = False
        
        # Fill up concurrent slots
        for i in range(3):
            job_requirements = {
                'job_id': f'running_job_{i}',
                'user_id': 'test_user',
                'cpu_cores': 1,
                'memory_gb': 1.0
            }
            scheduler.allocate_resources(job_requirements)
        
        # Queue a job
        job_requirements = {
            'job_id': 'queued_job',
            'user_id': 'test_user',
            'cpu_cores': 1,
            'memory_gb': 1.0
        }
        scheduler.allocate_resources(job_requirements)
        
        status = scheduler.get_job_status('queued_job')
        
        assert status is not None
        assert status['job_id'] == 'queued_job'
        assert status['status'] == 'queued'
        assert 'queue_position' in status
        assert 'estimated_wait_minutes' in status
    
    def test_get_job_status_completed(self, scheduler):
        """Test getting status of completed job."""
        job_requirements = {
            'job_id': 'test_job',
            'user_id': 'test_user',
            'cpu_cores': 1,
            'memory_gb': 1.0
        }
        scheduler.allocate_resources(job_requirements)
        scheduler.release_resources('test_job')
        
        status = scheduler.get_job_status('test_job')
        
        assert status is not None
        assert status['job_id'] == 'test_job'
        assert status['status'] == 'completed'
        assert 'completed_at' in status
    
    def test_get_job_status_nonexistent(self, scheduler):
        """Test getting status of nonexistent job."""
        status = scheduler.get_job_status('nonexistent_job')
        assert status is None
    
    def test_cancel_job_running(self, scheduler):
        """Test cancelling a running job."""
        job_requirements = {
            'job_id': 'test_job',
            'user_id': 'test_user',
            'cpu_cores': 1,
            'memory_gb': 1.0
        }
        scheduler.allocate_resources(job_requirements)
        
        result = scheduler.cancel_job('test_job')
        
        assert result is True
        assert len(scheduler.running_jobs) == 0
        assert 'test_job' in scheduler.completed_jobs
        assert scheduler.completed_jobs['test_job'].status == JobStatus.CANCELLED
    
    def test_cancel_job_queued(self, scheduler):
        """Test cancelling a queued job."""
        # Disable fair sharing for this test
        scheduler.enable_fair_sharing = False
        
        # Fill up concurrent slots
        for i in range(3):
            job_requirements = {
                'job_id': f'running_job_{i}',
                'user_id': 'test_user',
                'cpu_cores': 1,
                'memory_gb': 1.0
            }
            scheduler.allocate_resources(job_requirements)
        
        # Queue a job
        job_requirements = {
            'job_id': 'queued_job',
            'user_id': 'test_user',
            'cpu_cores': 1,
            'memory_gb': 1.0
        }
        scheduler.allocate_resources(job_requirements)
        
        result = scheduler.cancel_job('queued_job')
        
        assert result is True
        assert len(scheduler.job_queue) == 0
        assert 'queued_job' in scheduler.completed_jobs
    
    def test_cancel_job_nonexistent(self, scheduler):
        """Test cancelling nonexistent job."""
        result = scheduler.cancel_job('nonexistent_job')
        assert result is False
    
    def test_set_job_priority(self, scheduler):
        """Test setting job priority."""
        # Disable fair sharing for this test
        scheduler.enable_fair_sharing = False
        
        # Fill up concurrent slots
        for i in range(3):
            job_requirements = {
                'job_id': f'running_job_{i}',
                'user_id': 'test_user',
                'cpu_cores': 1,
                'memory_gb': 1.0
            }
            scheduler.allocate_resources(job_requirements)
        
        # Queue jobs with different priorities
        for i, priority in enumerate([JobPriority.LOW, JobPriority.NORMAL]):
            job_requirements = {
                'job_id': f'queued_job_{i}',
                'user_id': 'test_user',
                'cpu_cores': 1,
                'memory_gb': 1.0,
                'priority': priority.value
            }
            scheduler.allocate_resources(job_requirements)
        
        # Change priority of first job
        result = scheduler.set_job_priority('queued_job_0', JobPriority.HIGH.value)
        
        assert result is True
        # Check that queue is re-sorted (high priority job should be first)
        assert scheduler.job_queue[0].job_id == 'queued_job_0'
        assert scheduler.job_queue[0].priority == JobPriority.HIGH
    
    def test_set_job_priority_nonexistent(self, scheduler):
        """Test setting priority for nonexistent job."""
        result = scheduler.set_job_priority('nonexistent_job', JobPriority.HIGH.value)
        assert result is False
    
    def test_add_job_callback(self, scheduler):
        """Test adding job callback."""
        callback = Mock()
        scheduler.add_job_callback('test_job', callback)
        
        assert 'test_job' in scheduler.job_callbacks
        assert callback in scheduler.job_callbacks['test_job']
    
    def test_fair_sharing_enabled(self, scheduler):
        """Test fair sharing constraints."""
        scheduler.enable_fair_sharing = True
        scheduler.max_concurrent_jobs = 3
        
        # User should be limited to 1/3 of total slots (1 job)
        job_requirements = {
            'job_id': 'user1_job1',
            'user_id': 'user1',
            'cpu_cores': 1,
            'memory_gb': 1.0
        }
        result1 = scheduler.allocate_resources(job_requirements)
        assert 'cpu_cores' in result1  # Should be allocated immediately
        
        # Second job for same user should be queued due to fair sharing
        job_requirements = {
            'job_id': 'user1_job2',
            'user_id': 'user1',
            'cpu_cores': 1,
            'memory_gb': 1.0
        }
        result2 = scheduler.allocate_resources(job_requirements)
        assert result2['status'] == 'queued'  # Should be queued
    
    def test_fair_sharing_disabled(self, scheduler):
        """Test behavior with fair sharing disabled."""
        scheduler.enable_fair_sharing = False
        
        # User should be able to use all available slots
        for i in range(3):
            job_requirements = {
                'job_id': f'user1_job{i}',
                'user_id': 'user1',
                'cpu_cores': 1,
                'memory_gb': 1.0
            }
            result = scheduler.allocate_resources(job_requirements)
            assert 'cpu_cores' in result  # Should be allocated immediately
        
        assert len(scheduler.running_jobs) == 3
    
    @patch('automl_framework.services.resource_scheduler.psutil')
    def test_update_system_resources(self, mock_psutil, scheduler):
        """Test system resource monitoring."""
        # Mock psutil calls
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.cpu_percent.return_value = 25.0
        mock_psutil.virtual_memory.return_value = Mock(
            total=32 * 1024**3,  # 32GB
            available=24 * 1024**3  # 24GB available
        )
        mock_psutil.disk_usage.return_value = Mock(
            total=1000 * 1024**3,  # 1TB
            free=500 * 1024**3  # 500GB free
        )
        
        scheduler._update_system_resources()
        
        assert scheduler.system_resources is not None
        assert scheduler.system_resources.total_cpu_cores == 8
        assert scheduler.system_resources.available_cpu_cores == 6  # 75% of 8
        assert scheduler.system_resources.total_memory_gb == 32.0
        assert scheduler.system_resources.available_memory_gb == 24.0
    
    def test_queue_scheduling_after_job_completion(self, scheduler):
        """Test that queued jobs are scheduled when resources become available."""
        # Disable fair sharing for this test
        scheduler.enable_fair_sharing = False
        
        # Fill up all concurrent job slots
        for i in range(3):
            job_requirements = {
                'job_id': f'running_job_{i}',
                'user_id': 'test_user',
                'cpu_cores': 1,
                'memory_gb': 1.0
            }
            scheduler.allocate_resources(job_requirements)
        
        # Queue a job
        job_requirements = {
            'job_id': 'queued_job',
            'user_id': 'test_user',
            'cpu_cores': 1,
            'memory_gb': 1.0
        }
        scheduler.allocate_resources(job_requirements)
        
        assert len(scheduler.running_jobs) == 3
        assert len(scheduler.job_queue) == 1
        
        # Release one running job
        scheduler.release_resources('running_job_0')
        
        # Queued job should now be running
        assert len(scheduler.running_jobs) == 3  # 2 original + 1 from queue
        assert len(scheduler.job_queue) == 0
        assert 'queued_job' in scheduler.running_jobs
    
    def test_estimate_wait_time(self, scheduler):
        """Test wait time estimation."""
        # Disable fair sharing for this test
        scheduler.enable_fair_sharing = False
        
        # Add some job history
        for i in range(5):
            req = ResourceRequirement()
            job = ScheduledJob(f'history_job_{i}', 'user', req)
            job.started_at = datetime.now() - timedelta(minutes=70)
            job.completed_at = datetime.now() - timedelta(minutes=10)
            scheduler.job_history.append(job)
        
        # Fill up concurrent slots
        for i in range(3):
            job_requirements = {
                'job_id': f'running_job_{i}',
                'user_id': 'test_user',
                'cpu_cores': 1,
                'memory_gb': 1.0
            }
            scheduler.allocate_resources(job_requirements)
        
        # Queue a job
        job_requirements = {
            'job_id': 'queued_job',
            'user_id': 'test_user',
            'cpu_cores': 1,
            'memory_gb': 1.0
        }
        scheduler.allocate_resources(job_requirements)
        
        # Get queued job
        queued_job = scheduler.job_queue[0]
        wait_time = scheduler._estimate_wait_time(queued_job)
        
        assert isinstance(wait_time, int)
        assert wait_time >= 0
    
    def test_resource_utilization_calculation(self, scheduler):
        """Test resource utilization calculation."""
        # Allocate some resources
        job_requirements = {
            'job_id': 'test_job',
            'user_id': 'test_user',
            'cpu_cores': 2,
            'memory_gb': 8.0,
            'gpu_count': 1
        }
        scheduler.allocate_resources(job_requirements)
        
        utilization = scheduler._calculate_resource_utilization()
        
        assert 'cpu_utilization_percent' in utilization
        assert 'memory_utilization_percent' in utilization
        assert 'gpu_utilization_percent' in utilization
        assert 'storage_utilization_percent' in utilization
        
        # CPU utilization should be 25% (2 out of 8 cores)
        assert utilization['cpu_utilization_percent'] == 25.0
        # Memory utilization should be 25% (8 out of 32 GB)
        assert utilization['memory_utilization_percent'] == 25.0
        # GPU utilization should be 50% (1 out of 2 GPUs)
        assert utilization['gpu_utilization_percent'] == 50.0


if __name__ == '__main__':
    pytest.main([__file__])