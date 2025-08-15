#!/usr/bin/env python3
"""
Test script to verify the project setup is working correctly
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from automl_framework.core.config import get_config
from automl_framework.utils.logging import get_logger
from automl_framework.core.registry import get_service_registry
from automl_framework.core.interfaces import DataType, ExperimentStatus, TaskType


def test_configuration():
    """Test configuration loading"""
    print("Testing configuration loading...")
    config = get_config()
    print(f"✓ Configuration loaded successfully")
    print(f"  - Database URL: {config.database.postgresql_url}")
    print(f"  - API Port: {config.api.port}")
    print(f"  - Max concurrent experiments: {config.resources.max_concurrent_experiments}")
    return True


def test_logging():
    """Test logging system"""
    print("\nTesting logging system...")
    logger = get_logger("test")
    logger.info("Test log message", experiment_id="test-123")
    print("✓ Logging system working correctly")
    return True


def test_service_registry():
    """Test service registry"""
    print("\nTesting service registry...")
    registry = get_service_registry()
    print(f"✓ Service registry initialized")
    print(f"  - Available service types: {list(registry._service_types.keys())}")
    return True


def test_interfaces():
    """Test that interfaces and enums are properly defined"""
    print("\nTesting interfaces and data models...")
    
    # Test enums
    data_types = list(DataType)
    experiment_statuses = list(ExperimentStatus)
    task_types = list(TaskType)
    
    print(f"✓ Data types: {[dt.value for dt in data_types]}")
    print(f"✓ Experiment statuses: {[es.value for es in experiment_statuses]}")
    print(f"✓ Task types: {[tt.value for tt in task_types]}")
    return True


def main():
    """Run all tests"""
    print("AutoML Framework Setup Verification")
    print("=" * 40)
    
    tests = [
        test_configuration,
        test_logging,
        test_service_registry,
        test_interfaces
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
    
    print(f"\n{passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✓ All setup verification tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())