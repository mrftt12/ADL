#!/usr/bin/env python3
"""
Test runner script for AutoML framework.

Provides convenient commands to run different types of tests
with appropriate configurations and reporting.
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    if description:
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install pytest")
        return False


def run_unit_tests(args):
    """Run unit tests."""
    cmd = [
        "python", "-m", "pytest",
        "-m", "unit",
        "--tb=short",
        "-v"
    ]
    
    if args.coverage:
        cmd.extend(["--cov=automl_framework", "--cov-report=html", "--cov-report=term"])
    
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    if args.verbose:
        cmd.append("-vv")
    
    return run_command(cmd, "Unit Tests")


def run_integration_tests(args):
    """Run integration tests."""
    cmd = [
        "python", "-m", "pytest",
        "-m", "integration",
        "--tb=short",
        "-v"
    ]
    
    if args.coverage:
        cmd.extend(["--cov=automl_framework", "--cov-report=html", "--cov-report=term"])
    
    if args.verbose:
        cmd.append("-vv")
    
    return run_command(cmd, "Integration Tests")


def run_api_tests(args):
    """Run API tests."""
    cmd = [
        "python", "-m", "pytest",
        "-m", "api",
        "--tb=short",
        "-v"
    ]
    
    if args.verbose:
        cmd.append("-vv")
    
    return run_command(cmd, "API Tests")


def run_database_tests(args):
    """Run database tests."""
    cmd = [
        "python", "-m", "pytest",
        "-m", "database",
        "--tb=short",
        "-v"
    ]
    
    if args.verbose:
        cmd.append("-vv")
    
    return run_command(cmd, "Database Tests")


def run_performance_tests(args):
    """Run performance tests."""
    cmd = [
        "python", "-m", "pytest",
        "-m", "performance",
        "--tb=short",
        "-v",
        "--durations=0"
    ]
    
    if args.verbose:
        cmd.append("-vv")
    
    return run_command(cmd, "Performance Tests")


def run_all_tests(args):
    """Run all tests."""
    cmd = [
        "python", "-m", "pytest",
        "--tb=short",
        "-v"
    ]
    
    if args.coverage:
        cmd.extend([
            "--cov=automl_framework",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-fail-under=80"
        ])
    
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    if args.verbose:
        cmd.append("-vv")
    
    if args.fast:
        cmd.extend(["-m", "not slow"])
    
    return run_command(cmd, "All Tests")


def run_specific_test(args):
    """Run specific test file or test function."""
    cmd = [
        "python", "-m", "pytest",
        args.test,
        "--tb=short",
        "-v"
    ]
    
    if args.coverage:
        cmd.extend(["--cov=automl_framework", "--cov-report=term"])
    
    if args.verbose:
        cmd.append("-vv")
    
    return run_command(cmd, f"Specific Test: {args.test}")


def run_regression_tests(args):
    """Run regression tests."""
    cmd = [
        "python", "-m", "pytest",
        "-m", "regression",
        "--tb=short",
        "-v",
        "--durations=10"
    ]
    
    if args.verbose:
        cmd.append("-vv")
    
    return run_command(cmd, "Regression Tests")


def check_test_environment():
    """Check if test environment is properly set up."""
    print("Checking test environment...")
    
    # Check if pytest is installed
    try:
        import pytest
        print(f"✓ pytest {pytest.__version__} is installed")
    except ImportError:
        print("✗ pytest is not installed")
        print("Install with: pip install pytest")
        return False
    
    # Check if test directory exists
    if not Path("tests").exists():
        print("✗ tests directory not found")
        return False
    else:
        print("✓ tests directory found")
    
    # Check if main package exists
    if not Path("automl_framework").exists():
        print("✗ automl_framework package not found")
        return False
    else:
        print("✓ automl_framework package found")
    
    # Check for test configuration
    if Path("pytest.ini").exists():
        print("✓ pytest.ini configuration found")
    else:
        print("! pytest.ini not found (optional)")
    
    print("Test environment check completed.\n")
    return True


def generate_test_report(args):
    """Generate comprehensive test report."""
    print("Generating comprehensive test report...")
    
    # Run tests with detailed reporting
    cmd = [
        "python", "-m", "pytest",
        "--tb=short",
        "--cov=automl_framework",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
        "--cov-report=term-missing",
        "--junit-xml=test-results.xml",
        "--durations=20",
        "-v"
    ]
    
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    success = run_command(cmd, "Comprehensive Test Report")
    
    if success:
        print("\nTest report generated successfully!")
        print("- HTML coverage report: htmlcov/index.html")
        print("- XML coverage report: coverage.xml")
        print("- JUnit XML report: test-results.xml")
    
    return success


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test runner for AutoML framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py unit                    # Run unit tests
  python run_tests.py integration             # Run integration tests
  python run_tests.py all --coverage          # Run all tests with coverage
  python run_tests.py specific tests/test_data_models.py  # Run specific test file
  python run_tests.py performance             # Run performance tests
  python run_tests.py report --parallel       # Generate comprehensive report
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Test commands')
    
    # Unit tests
    unit_parser = subparsers.add_parser('unit', help='Run unit tests')
    unit_parser.add_argument('--coverage', action='store_true', help='Include coverage report')
    unit_parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    unit_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Integration tests
    integration_parser = subparsers.add_parser('integration', help='Run integration tests')
    integration_parser.add_argument('--coverage', action='store_true', help='Include coverage report')
    integration_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # API tests
    api_parser = subparsers.add_parser('api', help='Run API tests')
    api_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Database tests
    db_parser = subparsers.add_parser('database', help='Run database tests')
    db_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Performance tests
    perf_parser = subparsers.add_parser('performance', help='Run performance tests')
    perf_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # All tests
    all_parser = subparsers.add_parser('all', help='Run all tests')
    all_parser.add_argument('--coverage', action='store_true', help='Include coverage report')
    all_parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    all_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    all_parser.add_argument('--fast', action='store_true', help='Skip slow tests')
    
    # Specific test
    specific_parser = subparsers.add_parser('specific', help='Run specific test')
    specific_parser.add_argument('test', help='Test file or test function to run')
    specific_parser.add_argument('--coverage', action='store_true', help='Include coverage report')
    specific_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Regression tests
    regression_parser = subparsers.add_parser('regression', help='Run regression tests')
    regression_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Test report
    report_parser = subparsers.add_parser('report', help='Generate comprehensive test report')
    report_parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    
    # Check environment
    subparsers.add_parser('check', help='Check test environment')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Check environment first
    if not check_test_environment():
        return 1
    
    # Route to appropriate function
    success = True
    
    if args.command == 'unit':
        success = run_unit_tests(args)
    elif args.command == 'integration':
        success = run_integration_tests(args)
    elif args.command == 'api':
        success = run_api_tests(args)
    elif args.command == 'database':
        success = run_database_tests(args)
    elif args.command == 'performance':
        success = run_performance_tests(args)
    elif args.command == 'all':
        success = run_all_tests(args)
    elif args.command == 'specific':
        success = run_specific_test(args)
    elif args.command == 'regression':
        success = run_regression_tests(args)
    elif args.command == 'report':
        success = generate_test_report(args)
    elif args.command == 'check':
        success = check_test_environment()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())