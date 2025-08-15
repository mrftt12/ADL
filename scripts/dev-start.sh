#!/bin/bash

# AutoML Framework Development Startup Script
# This script starts all services required for local development

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
ENV_FILE="$PROJECT_ROOT/.env.dev"

# Default configuration
DEFAULT_MODE="docker"
DEFAULT_PROFILE="full"

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
AutoML Framework Development Startup Script

Usage: $0 [OPTIONS]

Options:
    -m, --mode MODE         Startup mode: docker, native, or hybrid (default: docker)
    -p, --profile PROFILE   Service profile: minimal, api, full (default: full)
    -h, --help             Show this help message
    --clean                Clean up existing containers and volumes
    --logs                 Follow logs after startup
    --no-frontend          Skip frontend service
    --no-monitoring        Skip monitoring services

Modes:
    docker    - Run all services in Docker containers (recommended)
    native    - Run services natively on the host (requires manual setup)
    hybrid    - Run databases in Docker, services natively

Profiles:
    minimal   - Only databases (PostgreSQL, MongoDB, Redis)
    api       - Databases + API service
    full      - All services including workers and monitoring

Examples:
    $0                                    # Start all services with Docker
    $0 -m native -p api                   # Start API natively with databases
    $0 -m docker -p minimal --logs       # Start only databases and follow logs
    $0 --clean                           # Clean up and start fresh

EOF
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing_deps=()
    
    if [[ "$MODE" == "docker" || "$MODE" == "hybrid" ]]; then
        if ! command -v docker &> /dev/null; then
            missing_deps+=("docker")
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            missing_deps+=("docker-compose")
        fi
    fi
    
    if [[ "$MODE" == "native" || "$MODE" == "hybrid" ]]; then
        if ! command -v python3 &> /dev/null; then
            missing_deps+=("python3")
        fi
        
        if ! command -v pip3 &> /dev/null; then
            missing_deps+=("pip3")
        fi
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_error "Please install the missing dependencies and try again."
        exit 1
    fi
    
    print_success "All prerequisites satisfied"
}

# Function to create environment file
create_env_file() {
    print_status "Creating development environment file..."
    
    cat > "$ENV_FILE" << EOF
# AutoML Framework Development Environment

# Application
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DEBUG=true

# Database URLs
DATABASE_URL=postgresql://automl:automl_password@localhost:5432/automl
MONGODB_URL=mongodb://automl:automl_password@localhost:27017/automl
REDIS_URL=redis://:automl_password@localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Frontend Configuration
FRONTEND_PORT=3000
VITE_API_URL=http://localhost:8000

# Worker Configuration
WORKER_CONCURRENCY=2
WORKER_LOG_LEVEL=INFO

# Security
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET_KEY=dev-jwt-secret-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440

# File Storage
UPLOAD_DIR=./data/uploads
MODEL_DIR=./models
CHECKPOINT_DIR=./checkpoints
LOG_DIR=./logs

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
GRAFANA_ADMIN_PASSWORD=admin

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

EOF
    
    print_success "Environment file created at $ENV_FILE"
}

# Function to setup directories
setup_directories() {
    print_status "Setting up project directories..."
    
    local directories=(
        "data/uploads"
        "logs"
        "models"
        "checkpoints"
        "data/processed"
        "data/experiments"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$PROJECT_ROOT/$dir"
        print_status "Created directory: $dir"
    done
    
    # Set appropriate permissions
    chmod -R 755 "$PROJECT_ROOT/data" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/models" "$PROJECT_ROOT/checkpoints"
    
    print_success "Project directories setup completed"
}

# Function to wait for service health
wait_for_service() {
    local service_name="$1"
    local health_url="$2"
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to be healthy..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "$health_url" > /dev/null 2>&1; then
            print_success "$service_name is healthy"
            return 0
        fi
        
        print_status "Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 2
        ((attempt++))
    done
    
    print_error "$service_name failed to become healthy after $max_attempts attempts"
    return 1
}

# Function to check database connectivity
check_database_connectivity() {
    print_status "Checking database connectivity..."
    
    # Check PostgreSQL
    if command -v psql &> /dev/null; then
        if PGPASSWORD=automl_password psql -h localhost -U automl -d automl -c "SELECT 1;" &> /dev/null; then
            print_success "PostgreSQL connection successful"
        else
            print_warning "PostgreSQL connection failed"
        fi
    fi
    
    # Check MongoDB
    if command -v mongosh &> /dev/null; then
        if mongosh "mongodb://automl:automl_password@localhost:27017/automl" --eval "db.runCommand('ping')" &> /dev/null; then
            print_success "MongoDB connection successful"
        else
            print_warning "MongoDB connection failed"
        fi
    fi
    
    # Check Redis
    if command -v redis-cli &> /dev/null; then
        if redis-cli -a automl_password ping &> /dev/null; then
            print_success "Redis connection successful"
        else
            print_warning "Redis connection failed"
        fi
    fi
}

# Function to run database migrations
run_migrations() {
    print_status "Running database migrations..."
    
    cd "$PROJECT_ROOT"
    
    # Set environment variables
    export $(grep -v '^#' "$ENV_FILE" | xargs)
    
    # Run PostgreSQL migrations
    if python3 -c "from automl_framework.migrations.migration_manager import MigrationManager; MigrationManager().run_migrations()" 2>/dev/null; then
        print_success "Database migrations completed"
    else
        print_warning "Database migrations failed or not needed"
    fi
}

# Function to start services with Docker
start_docker_services() {
    print_status "Starting services with Docker..."
    
    cd "$PROJECT_ROOT"
    
    # Determine docker-compose services based on profile
    local services=""
    case "$PROFILE" in
        "minimal")
            services="postgres mongodb redis"
            ;;
        "api")
            services="postgres mongodb redis api"
            ;;
        "full")
            services=""  # Start all services
            ;;
    esac
    
    # Add profile-specific exclusions
    local compose_args=""
    if [[ "$NO_FRONTEND" == "true" ]]; then
        compose_args="$compose_args --scale frontend=0"
    fi
    
    if [[ "$NO_MONITORING" == "true" ]]; then
        compose_args="$compose_args --scale prometheus=0 --scale grafana=0 --scale alertmanager=0 --scale node-exporter=0 --scale postgres-exporter=0 --scale redis-exporter=0 --scale cadvisor=0"
    fi
    
    # Clean up if requested
    if [[ "$CLEAN" == "true" ]]; then
        print_status "Cleaning up existing containers and volumes..."
        docker-compose down --remove-orphans --volumes
        docker system prune -f
    fi
    
    # Start services
    print_status "Building and starting Docker services..."
    if [[ -n "$services" ]]; then
        docker-compose up -d --build $compose_args $services
    else
        docker-compose up -d --build $compose_args
    fi
    
    # Wait for core services
    sleep 10
    wait_for_service "PostgreSQL" "http://localhost:5432" || true
    wait_for_service "API" "http://localhost:8000/health" || true
    
    print_success "Docker services started successfully"
}

# Function to start services natively
start_native_services() {
    print_status "Starting services natively..."
    
    cd "$PROJECT_ROOT"
    
    # Set environment variables
    export $(grep -v '^#' "$ENV_FILE" | xargs)
    
    # Install Python dependencies if needed
    if [[ ! -d "venv" ]]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        print_success "Python environment setup completed"
    else
        source venv/bin/activate
    fi
    
    # Start API service
    if [[ "$PROFILE" == "api" || "$PROFILE" == "full" ]]; then
        print_status "Starting API service..."
        python run_api.py &
        API_PID=$!
        echo $API_PID > .api.pid
        
        # Wait for API to be ready
        sleep 5
        wait_for_service "API" "http://localhost:8000/health"
    fi
    
    # Start worker services for full profile
    if [[ "$PROFILE" == "full" ]]; then
        print_status "Starting worker services..."
        
        # Training worker
        python -m automl_framework.services.training_service &
        TRAINING_PID=$!
        echo $TRAINING_PID > .training.pid
        
        # NAS worker
        python -m automl_framework.services.nas_service &
        NAS_PID=$!
        echo $NAS_PID > .nas.pid
        
        # HPO worker
        python -m automl_framework.services.hyperparameter_optimization &
        HPO_PID=$!
        echo $HPO_PID > .hpo.pid
        
        print_success "Worker services started"
    fi
    
    # Start frontend if not excluded
    if [[ "$NO_FRONTEND" != "true" && ("$PROFILE" == "full" || "$PROFILE" == "api") ]]; then
        print_status "Starting frontend service..."
        cd ui
        if [[ ! -d "node_modules" ]]; then
            npm install
        fi
        npm run dev &
        FRONTEND_PID=$!
        echo $FRONTEND_PID > ../.frontend.pid
        cd ..
        print_success "Frontend service started"
    fi
    
    print_success "Native services started successfully"
}

# Function to start hybrid services
start_hybrid_services() {
    print_status "Starting services in hybrid mode..."
    
    # Start databases with Docker
    cd "$PROJECT_ROOT"
    docker-compose up -d postgres mongodb redis
    
    # Wait for databases
    sleep 10
    
    # Start application services natively
    PROFILE="api" start_native_services
    
    print_success "Hybrid services started successfully"
}

# Function to display service information
show_service_info() {
    print_success "AutoML Framework started successfully!"
    echo ""
    echo "🔗 Service URLs:"
    
    if [[ "$PROFILE" == "api" || "$PROFILE" == "full" ]]; then
        echo "  📊 API Server: http://localhost:8000"
        echo "  📚 API Documentation: http://localhost:8000/docs"
        echo "  🔍 API Redoc: http://localhost:8000/redoc"
    fi
    
    if [[ "$NO_FRONTEND" != "true" && ("$PROFILE" == "full" || "$PROFILE" == "api") ]]; then
        echo "  🌐 Frontend: http://localhost:3000"
    fi
    
    echo ""
    echo "💾 Databases:"
    echo "  🐘 PostgreSQL: localhost:5432 (automl/automl_password)"
    echo "  🍃 MongoDB: localhost:27017 (automl/automl_password)"
    echo "  🔴 Redis: localhost:6379 (password: automl_password)"
    
    if [[ "$NO_MONITORING" != "true" && "$PROFILE" == "full" ]]; then
        echo ""
        echo "📈 Monitoring:"
        echo "  📊 Prometheus: http://localhost:9090"
        echo "  📈 Grafana: http://localhost:3001 (admin/admin)"
        echo "  🚨 Alertmanager: http://localhost:9093"
        echo "  📊 Node Exporter: http://localhost:9100"
        echo "  📊 cAdvisor: http://localhost:8080"
    fi
    
    echo ""
    echo "🛠️  Management Commands:"
    echo "  📋 View logs: $0 --logs"
    echo "  🛑 Stop services: $0 --stop"
    echo "  🔄 Restart services: $0 --restart"
    echo "  🧹 Clean up: $0 --clean"
    
    echo ""
    echo "📁 Important Directories:"
    echo "  📤 Upload: ./data/uploads"
    echo "  📊 Models: ./models"
    echo "  💾 Checkpoints: ./checkpoints"
    echo "  📝 Logs: ./logs"
    
    echo ""
    print_status "Development environment is ready!"
}

# Function to follow logs
follow_logs() {
    if [[ "$MODE" == "docker" || "$MODE" == "hybrid" ]]; then
        docker-compose logs -f
    else
        print_status "Following native service logs..."
        tail -f logs/*.log 2>/dev/null || print_warning "No log files found"
    fi
}

# Function to stop services
stop_services() {
    print_status "Stopping AutoML Framework services..."
    
    if [[ "$MODE" == "docker" || "$MODE" == "hybrid" ]]; then
        docker-compose down
    fi
    
    # Stop native services
    for pidfile in .api.pid .training.pid .nas.pid .hpo.pid .frontend.pid; do
        if [[ -f "$pidfile" ]]; then
            pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid"
                print_status "Stopped process $pid"
            fi
            rm -f "$pidfile"
        fi
    done
    
    print_success "All services stopped"
}

# Main function
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--mode)
                MODE="$2"
                shift 2
                ;;
            -p|--profile)
                PROFILE="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            --clean)
                CLEAN="true"
                shift
                ;;
            --logs)
                FOLLOW_LOGS="true"
                shift
                ;;
            --no-frontend)
                NO_FRONTEND="true"
                shift
                ;;
            --no-monitoring)
                NO_MONITORING="true"
                shift
                ;;
            --stop)
                stop_services
                exit 0
                ;;
            --restart)
                stop_services
                sleep 2
                # Continue with normal startup
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Set defaults
    MODE="${MODE:-$DEFAULT_MODE}"
    PROFILE="${PROFILE:-$DEFAULT_PROFILE}"
    
    # Validate arguments
    if [[ ! "$MODE" =~ ^(docker|native|hybrid)$ ]]; then
        print_error "Invalid mode: $MODE"
        show_usage
        exit 1
    fi
    
    if [[ ! "$PROFILE" =~ ^(minimal|api|full)$ ]]; then
        print_error "Invalid profile: $PROFILE"
        show_usage
        exit 1
    fi
    
    print_status "Starting AutoML Framework in $MODE mode with $PROFILE profile..."
    
    # Execute startup sequence
    check_prerequisites
    create_env_file
    setup_directories
    
    case "$MODE" in
        "docker")
            start_docker_services
            ;;
        "native")
            start_native_services
            ;;
        "hybrid")
            start_hybrid_services
            ;;
    esac
    
    # Run migrations
    sleep 5
    run_migrations
    
    # Check connectivity
    check_database_connectivity
    
    # Show service information
    show_service_info
    
    # Follow logs if requested
    if [[ "$FOLLOW_LOGS" == "true" ]]; then
        echo ""
        print_status "Following logs (Press Ctrl+C to exit)..."
        follow_logs
    fi
}

# Handle script interruption
trap 'print_warning "Script interrupted by user"; exit 130' INT

# Run main function
main "$@"