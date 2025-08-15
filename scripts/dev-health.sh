#!/bin/bash

# AutoML Framework Development Health Check Script
# This script checks the health status of all AutoML Framework services

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
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
AutoML Framework Development Health Check Script

Usage: $0 [OPTIONS]

Options:
    --json            Output results in JSON format
    --quiet           Only show failed checks
    --watch           Continuously monitor services (refresh every 5s)
    -h, --help        Show this help message

Examples:
    $0                # Check all services once
    $0 --quiet        # Only show problems
    $0 --watch        # Continuous monitoring
    $0 --json         # JSON output for scripting

EOF
}

# Function to check HTTP service
check_http_service() {
    local name="$1"
    local url="$2"
    local timeout="${3:-5}"
    
    if curl -f -s --max-time "$timeout" "$url" > /dev/null 2>&1; then
        if [[ "$QUIET" != "true" ]]; then
            print_success "$name is healthy ($url)"
        fi
        return 0
    else
        print_error "$name is not responding ($url)"
        return 1
    fi
}

# Function to check TCP port
check_tcp_port() {
    local name="$1"
    local host="$2"
    local port="$3"
    local timeout="${4:-3}"
    
    if timeout "$timeout" bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
        if [[ "$QUIET" != "true" ]]; then
            print_success "$name is listening ($host:$port)"
        fi
        return 0
    else
        print_error "$name is not listening ($host:$port)"
        return 1
    fi
}

# Function to check Docker service
check_docker_service() {
    local service_name="$1"
    
    if command -v docker-compose &> /dev/null; then
        cd "$PROJECT_ROOT"
        local status=$(docker-compose ps "$service_name" --format "table {{.State}}" 2>/dev/null | tail -n +2 | tr -d ' ')
        
        case "$status" in
            "Up")
                if [[ "$QUIET" != "true" ]]; then
                    print_success "Docker service $service_name is running"
                fi
                return 0
                ;;
            "Up(healthy)")
                if [[ "$QUIET" != "true" ]]; then
                    print_success "Docker service $service_name is healthy"
                fi
                return 0
                ;;
            "")
                print_error "Docker service $service_name is not found"
                return 1
                ;;
            *)
                print_error "Docker service $service_name is in state: $status"
                return 1
                ;;
        esac
    else
        print_warning "Docker Compose not available"
        return 1
    fi
}

# Function to check process by PID file
check_pid_service() {
    local name="$1"
    local pidfile="$2"
    
    if [[ -f "$pidfile" ]]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            if [[ "$QUIET" != "true" ]]; then
                print_success "$name is running (PID: $pid)"
            fi
            return 0
        else
            print_error "$name has stale PID file (PID: $pid not running)"
            return 1
        fi
    else
        print_error "$name PID file not found ($pidfile)"
        return 1
    fi
}

# Function to check database connectivity
check_database() {
    local db_type="$1"
    local connection_string="$2"
    
    case "$db_type" in
        "postgresql")
            if command -v psql &> /dev/null; then
                if PGPASSWORD=automl_password psql -h localhost -U automl -d automl -c "SELECT 1;" &> /dev/null; then
                    if [[ "$QUIET" != "true" ]]; then
                        print_success "PostgreSQL database is accessible"
                    fi
                    return 0
                else
                    print_error "PostgreSQL database is not accessible"
                    return 1
                fi
            else
                print_warning "psql command not available, skipping PostgreSQL connectivity test"
                return 1
            fi
            ;;
        "mongodb")
            if command -v mongosh &> /dev/null; then
                if mongosh "mongodb://automl:automl_password@localhost:27017/automl" --eval "db.runCommand('ping')" &> /dev/null; then
                    if [[ "$QUIET" != "true" ]]; then
                        print_success "MongoDB database is accessible"
                    fi
                    return 0
                else
                    print_error "MongoDB database is not accessible"
                    return 1
                fi
            else
                print_warning "mongosh command not available, skipping MongoDB connectivity test"
                return 1
            fi
            ;;
        "redis")
            if command -v redis-cli &> /dev/null; then
                if redis-cli -a automl_password ping &> /dev/null; then
                    if [[ "$QUIET" != "true" ]]; then
                        print_success "Redis database is accessible"
                    fi
                    return 0
                else
                    print_error "Redis database is not accessible"
                    return 1
                fi
            else
                print_warning "redis-cli command not available, skipping Redis connectivity test"
                return 1
            fi
            ;;
    esac
}

# Function to check system resources
check_system_resources() {
    local warnings=()
    
    # Check disk space
    local disk_usage=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $5}' | sed 's/%//')
    if [[ "$disk_usage" -gt 90 ]]; then
        warnings+=("Disk usage is high: ${disk_usage}%")
    elif [[ "$disk_usage" -gt 80 ]]; then
        warnings+=("Disk usage is moderate: ${disk_usage}%")
    fi
    
    # Check memory usage
    if command -v free &> /dev/null; then
        local mem_usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
        if [[ "$mem_usage" -gt 90 ]]; then
            warnings+=("Memory usage is high: ${mem_usage}%")
        fi
    fi
    
    # Check if GPU is available (if nvidia-smi exists)
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            if [[ "$QUIET" != "true" ]]; then
                print_success "GPU is available"
            fi
        else
            warnings+=("GPU is not accessible")
        fi
    fi
    
    # Report warnings
    for warning in "${warnings[@]}"; do
        print_warning "$warning"
    done
    
    if [[ ${#warnings[@]} -eq 0 && "$QUIET" != "true" ]]; then
        print_success "System resources are healthy"
    fi
    
    return ${#warnings[@]}
}

# Function to perform comprehensive health check
perform_health_check() {
    local total_checks=0
    local failed_checks=0
    local results=()
    
    if [[ "$JSON" != "true" ]]; then
        echo "🏥 AutoML Framework Health Check"
        echo "================================"
        echo ""
    fi
    
    # Check core services
    if [[ "$JSON" != "true" && "$QUIET" != "true" ]]; then
        echo "🔍 Checking Core Services..."
    fi
    
    # API Service
    ((total_checks++))
    if ! check_http_service "API Service" "http://localhost:8000/health"; then
        ((failed_checks++))
        results+=('{"service":"api","status":"failed","url":"http://localhost:8000/health"}')
    else
        results+=('{"service":"api","status":"healthy","url":"http://localhost:8000/health"}')
    fi
    
    # Frontend Service
    ((total_checks++))
    if ! check_tcp_port "Frontend Service" "localhost" "3000"; then
        ((failed_checks++))
        results+=('{"service":"frontend","status":"failed","port":"3000"}')
    else
        results+=('{"service":"frontend","status":"healthy","port":"3000"}')
    fi
    
    # Check databases
    if [[ "$JSON" != "true" && "$QUIET" != "true" ]]; then
        echo ""
        echo "💾 Checking Databases..."
    fi
    
    # PostgreSQL
    ((total_checks++))
    if ! check_tcp_port "PostgreSQL" "localhost" "5432"; then
        ((failed_checks++))
        results+=('{"service":"postgresql","status":"failed","port":"5432"}')
    else
        results+=('{"service":"postgresql","status":"healthy","port":"5432"}')
        
        # Test connectivity
        ((total_checks++))
        if ! check_database "postgresql"; then
            ((failed_checks++))
        fi
    fi
    
    # MongoDB
    ((total_checks++))
    if ! check_tcp_port "MongoDB" "localhost" "27017"; then
        ((failed_checks++))
        results+=('{"service":"mongodb","status":"failed","port":"27017"}')
    else
        results+=('{"service":"mongodb","status":"healthy","port":"27017"}')
        
        # Test connectivity
        ((total_checks++))
        if ! check_database "mongodb"; then
            ((failed_checks++))
        fi
    fi
    
    # Redis
    ((total_checks++))
    if ! check_tcp_port "Redis" "localhost" "6379"; then
        ((failed_checks++))
        results+=('{"service":"redis","status":"failed","port":"6379"}')
    else
        results+=('{"service":"redis","status":"healthy","port":"6379"}')
        
        # Test connectivity
        ((total_checks++))
        if ! check_database "redis"; then
            ((failed_checks++))
        fi
    fi
    
    # Check monitoring services (if running)
    if [[ "$JSON" != "true" && "$QUIET" != "true" ]]; then
        echo ""
        echo "📊 Checking Monitoring Services..."
    fi
    
    # Prometheus
    ((total_checks++))
    if ! check_http_service "Prometheus" "http://localhost:9090/-/healthy"; then
        ((failed_checks++))
        results+=('{"service":"prometheus","status":"failed","url":"http://localhost:9090"}')
    else
        results+=('{"service":"prometheus","status":"healthy","url":"http://localhost:9090"}')
    fi
    
    # Grafana
    ((total_checks++))
    if ! check_http_service "Grafana" "http://localhost:3001/api/health"; then
        ((failed_checks++))
        results+=('{"service":"grafana","status":"failed","url":"http://localhost:3001"}')
    else
        results+=('{"service":"grafana","status":"healthy","url":"http://localhost:3001"}')
    fi
    
    # Check system resources
    if [[ "$JSON" != "true" && "$QUIET" != "true" ]]; then
        echo ""
        echo "🖥️  Checking System Resources..."
    fi
    
    ((total_checks++))
    if ! check_system_resources; then
        ((failed_checks++))
    fi
    
    # Output results
    if [[ "$JSON" == "true" ]]; then
        echo "{"
        echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
        echo "  \"total_checks\": $total_checks,"
        echo "  \"failed_checks\": $failed_checks,"
        echo "  \"success_rate\": $(( (total_checks - failed_checks) * 100 / total_checks ))%,"
        echo "  \"services\": ["
        printf "    %s" "${results[0]}"
        for result in "${results[@]:1}"; do
            printf ",\n    %s" "$result"
        done
        echo ""
        echo "  ]"
        echo "}"
    else
        echo ""
        echo "📋 Health Check Summary"
        echo "======================"
        echo "Total checks: $total_checks"
        echo "Failed checks: $failed_checks"
        echo "Success rate: $(( (total_checks - failed_checks) * 100 / total_checks ))%"
        
        if [[ $failed_checks -eq 0 ]]; then
            echo ""
            print_success "All services are healthy! 🎉"
        else
            echo ""
            print_error "$failed_checks service(s) have issues"
            echo ""
            echo "💡 Troubleshooting tips:"
            echo "  - Check service logs: docker-compose logs [service]"
            echo "  - Restart services: ./scripts/dev-start.sh --restart"
            echo "  - Check system resources: df -h && free -h"
        fi
    fi
    
    return $failed_checks
}

# Function for continuous monitoring
watch_services() {
    if [[ "$JSON" == "true" ]]; then
        print_error "Watch mode is not compatible with JSON output"
        exit 1
    fi
    
    print_status "Starting continuous monitoring (Press Ctrl+C to stop)..."
    echo ""
    
    while true; do
        clear
        echo "🔄 AutoML Framework Health Monitor - $(date)"
        echo "=============================================="
        echo ""
        
        perform_health_check
        
        echo ""
        print_status "Refreshing in 5 seconds..."
        sleep 5
    done
}

# Main function
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --json)
                JSON="true"
                shift
                ;;
            --quiet)
                QUIET="true"
                shift
                ;;
            --watch)
                WATCH="true"
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
    
    cd "$PROJECT_ROOT"
    
    if [[ "$WATCH" == "true" ]]; then
        watch_services
    else
        perform_health_check
    fi
}

# Handle script interruption
trap 'print_warning "Health check interrupted by user"; exit 130' INT

# Run main function
main "$@"