#!/bin/bash

# AutoML Framework Development Stop Script
# This script stops all running AutoML Framework services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
AutoML Framework Development Stop Script

Usage: $0 [OPTIONS]

Options:
    --docker-only      Stop only Docker services
    --native-only      Stop only native services
    --clean           Remove containers and volumes
    --force           Force stop all processes
    -h, --help        Show this help message

Examples:
    $0                    # Stop all services
    $0 --docker-only      # Stop only Docker services
    $0 --clean            # Stop and clean up everything
    $0 --force            # Force stop all processes

EOF
}

# Function to stop Docker services
stop_docker_services() {
    print_status "Stopping Docker services..."
    
    cd "$PROJECT_ROOT"
    
    if [[ "$CLEAN" == "true" ]]; then
        print_status "Stopping and removing containers, networks, and volumes..."
        docker-compose down --remove-orphans --volumes
        
        # Clean up unused Docker resources
        print_status "Cleaning up unused Docker resources..."
        docker system prune -f
        
        print_success "Docker cleanup completed"
    else
        docker-compose down --remove-orphans
        print_success "Docker services stopped"
    fi
}

# Function to stop native services
stop_native_services() {
    print_status "Stopping native services..."
    
    cd "$PROJECT_ROOT"
    
    local stopped_services=()
    
    # Stop services based on PID files
    local pidfiles=(
        ".api.pid:API Service"
        ".training.pid:Training Worker"
        ".nas.pid:NAS Worker"
        ".hpo.pid:HPO Worker"
        ".frontend.pid:Frontend Service"
    )
    
    for pidfile_info in "${pidfiles[@]}"; do
        IFS=':' read -r pidfile service_name <<< "$pidfile_info"
        
        if [[ -f "$pidfile" ]]; then
            pid=$(cat "$pidfile")
            
            if kill -0 "$pid" 2>/dev/null; then
                if [[ "$FORCE" == "true" ]]; then
                    kill -9 "$pid" 2>/dev/null || true
                else
                    kill -TERM "$pid" 2>/dev/null || true
                    
                    # Wait for graceful shutdown
                    local attempts=0
                    while kill -0 "$pid" 2>/dev/null && [[ $attempts -lt 10 ]]; do
                        sleep 1
                        ((attempts++))
                    done
                    
                    # Force kill if still running
                    if kill -0 "$pid" 2>/dev/null; then
                        kill -9 "$pid" 2>/dev/null || true
                    fi
                fi
                
                stopped_services+=("$service_name")
                print_status "Stopped $service_name (PID: $pid)"
            else
                print_warning "$service_name was not running (stale PID file)"
            fi
            
            rm -f "$pidfile"
        fi
    done
    
    # Stop any remaining Python processes related to AutoML
    if [[ "$FORCE" == "true" ]]; then
        print_status "Force stopping any remaining AutoML processes..."
        
        # Find and kill AutoML-related processes
        local automl_pids=$(pgrep -f "automl_framework" 2>/dev/null || true)
        if [[ -n "$automl_pids" ]]; then
            echo "$automl_pids" | xargs kill -9 2>/dev/null || true
            print_status "Force stopped remaining AutoML processes"
        fi
        
        # Kill any uvicorn processes on port 8000
        local uvicorn_pids=$(lsof -ti:8000 2>/dev/null || true)
        if [[ -n "$uvicorn_pids" ]]; then
            echo "$uvicorn_pids" | xargs kill -9 2>/dev/null || true
            print_status "Force stopped processes on port 8000"
        fi
        
        # Kill any Node.js processes on port 3000 (frontend)
        local node_pids=$(lsof -ti:3000 2>/dev/null || true)
        if [[ -n "$node_pids" ]]; then
            echo "$node_pids" | xargs kill -9 2>/dev/null || true
            print_status "Force stopped processes on port 3000"
        fi
    fi
    
    if [[ ${#stopped_services[@]} -gt 0 ]]; then
        print_success "Stopped native services: ${stopped_services[*]}"
    else
        print_warning "No native services were running"
    fi
}

# Function to clean up temporary files
cleanup_temp_files() {
    print_status "Cleaning up temporary files..."
    
    cd "$PROJECT_ROOT"
    
    # Remove PID files
    rm -f .*.pid
    
    # Remove development environment file
    rm -f .env.dev
    
    # Clean up Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    # Clean up log files if requested
    if [[ "$CLEAN" == "true" ]]; then
        print_status "Cleaning up log files..."
        rm -rf logs/*.log 2>/dev/null || true
    fi
    
    print_success "Temporary files cleaned up"
}

# Function to check for running services
check_running_services() {
    print_status "Checking for running services..."
    
    local running_services=()
    
    # Check Docker services
    if command -v docker-compose &> /dev/null; then
        cd "$PROJECT_ROOT"
        local docker_services=$(docker-compose ps --services --filter "status=running" 2>/dev/null || true)
        if [[ -n "$docker_services" ]]; then
            running_services+=("Docker services: $(echo "$docker_services" | tr '\n' ' ')")
        fi
    fi
    
    # Check native services by port
    local ports=(8000 3000 5432 27017 6379 9090 3001)
    for port in "${ports[@]}"; do
        if lsof -ti:$port &>/dev/null; then
            local process_info=$(lsof -ti:$port | head -1 | xargs ps -p 2>/dev/null | tail -1 || echo "Unknown process")
            running_services+=("Port $port: $process_info")
        fi
    done
    
    if [[ ${#running_services[@]} -gt 0 ]]; then
        print_warning "Found running services:"
        for service in "${running_services[@]}"; do
            echo "  - $service"
        done
        return 1
    else
        print_success "No running services found"
        return 0
    fi
}

# Function to display final status
show_final_status() {
    echo ""
    print_success "AutoML Framework services stopped successfully!"
    echo ""
    
    if ! check_running_services; then
        echo ""
        print_warning "Some services may still be running. Use --force to stop them."
        echo ""
        echo "Manual cleanup commands:"
        echo "  🐳 Stop all Docker containers: docker stop \$(docker ps -q)"
        echo "  🧹 Clean Docker system: docker system prune -f"
        echo "  🔍 Check port usage: lsof -i :8000"
        echo "  ⚡ Kill process by PID: kill -9 <PID>"
    else
        echo ""
        print_status "All services have been stopped cleanly."
    fi
    
    echo ""
    echo "🔄 To restart services:"
    echo "  ./scripts/dev-start.sh"
    echo ""
}

# Main function
main() {
    local docker_only=false
    local native_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --docker-only)
                docker_only=true
                shift
                ;;
            --native-only)
                native_only=true
                shift
                ;;
            --clean)
                CLEAN="true"
                shift
                ;;
            --force)
                FORCE="true"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    print_status "Stopping AutoML Framework services..."
    
    # Stop services based on options
    if [[ "$docker_only" == "true" ]]; then
        stop_docker_services
    elif [[ "$native_only" == "true" ]]; then
        stop_native_services
    else
        # Stop both Docker and native services
        stop_docker_services
        stop_native_services
    fi
    
    # Clean up temporary files
    cleanup_temp_files
    
    # Show final status
    show_final_status
}

# Handle script interruption
trap 'print_warning "Script interrupted by user"; exit 130' INT

# Run main function
main "$@"