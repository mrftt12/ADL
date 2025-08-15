#!/bin/bash

# AutoML Framework Database Migration Script
# This script handles database initialization and migrations

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
AutoML Framework Database Migration Script

Usage: $0 [OPTIONS]

Options:
    --init            Initialize databases from scratch
    --migrate         Run pending migrations only
    --reset           Reset all databases (WARNING: destroys data)
    --seed            Seed databases with sample data
    --status          Show migration status
    --dry-run         Show what would be done without executing
    -h, --help        Show this help message

Examples:
    $0 --init         # Initialize fresh databases
    $0 --migrate      # Run pending migrations
    $0 --reset        # Reset and reinitialize databases
    $0 --seed         # Add sample data
    $0 --status       # Check migration status

EOF
}

# Function to load environment variables
load_environment() {
    if [[ -f "$ENV_FILE" ]]; then
        print_status "Loading environment from $ENV_FILE"
        export $(grep -v '^#' "$ENV_FILE" | xargs)
    else
        print_warning "Environment file not found, using defaults"
        export DATABASE_URL="postgresql://automl:automl_password@localhost:5432/automl"
        export MONGODB_URL="mongodb://automl:automl_password@localhost:27017/automl"
        export REDIS_URL="redis://:automl_password@localhost:6379/0"
    fi
}

# Function to wait for database connectivity
wait_for_database() {
    local db_type="$1"
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $db_type to be ready..."
    
    case "$db_type" in
        "postgresql")
            while [[ $attempt -le $max_attempts ]]; do
                if PGPASSWORD=automl_password psql -h localhost -U automl -d automl -c "SELECT 1;" &> /dev/null; then
                    print_success "PostgreSQL is ready"
                    return 0
                fi
                print_status "Attempt $attempt/$max_attempts: PostgreSQL not ready yet..."
                sleep 2
                ((attempt++))
            done
            ;;
        "mongodb")
            while [[ $attempt -le $max_attempts ]]; do
                if mongosh "mongodb://automl:automl_password@localhost:27017/automl" --eval "db.runCommand('ping')" &> /dev/null; then
                    print_success "MongoDB is ready"
                    return 0
                fi
                print_status "Attempt $attempt/$max_attempts: MongoDB not ready yet..."
                sleep 2
                ((attempt++))
            done
            ;;
        "redis")
            while [[ $attempt -le $max_attempts ]]; do
                if redis-cli -a automl_password ping &> /dev/null; then
                    print_success "Redis is ready"
                    return 0
                fi
                print_status "Attempt $attempt/$max_attempts: Redis not ready yet..."
                sleep 2
                ((attempt++))
            done
            ;;
    esac
    
    print_error "$db_type failed to become ready after $max_attempts attempts"
    return 1
}

# Function to initialize PostgreSQL database
init_postgresql() {
    print_status "Initializing PostgreSQL database..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "[DRY RUN] Would initialize PostgreSQL tables"
        return 0
    fi
    
    # Wait for PostgreSQL to be ready
    wait_for_database "postgresql"
    
    # Run initialization script
    if [[ -f "$PROJECT_ROOT/docker/init-db.sql" ]]; then
        print_status "Running PostgreSQL initialization script..."
        PGPASSWORD=automl_password psql -h localhost -U automl -d automl -f "$PROJECT_ROOT/docker/init-db.sql"
        print_success "PostgreSQL initialization completed"
    else
        print_warning "PostgreSQL initialization script not found"
    fi
    
    # Run Python migrations
    cd "$PROJECT_ROOT"
    if python3 -c "
from automl_framework.migrations.migration_manager import MigrationManager
manager = MigrationManager()
manager.run_migrations()
print('PostgreSQL migrations completed successfully')
" 2>/dev/null; then
        print_success "PostgreSQL migrations completed"
    else
        print_error "PostgreSQL migrations failed"
        return 1
    fi
}

# Function to initialize MongoDB database
init_mongodb() {
    print_status "Initializing MongoDB database..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "[DRY RUN] Would initialize MongoDB collections"
        return 0
    fi
    
    # Wait for MongoDB to be ready
    wait_for_database "mongodb"
    
    # Run initialization script
    if [[ -f "$PROJECT_ROOT/docker/init-mongo.js" ]]; then
        print_status "Running MongoDB initialization script..."
        mongosh "mongodb://automl:automl_password@localhost:27017/automl" "$PROJECT_ROOT/docker/init-mongo.js"
        print_success "MongoDB initialization completed"
    else
        print_warning "MongoDB initialization script not found"
    fi
    
    # Create indexes and collections
    print_status "Creating MongoDB indexes..."
    mongosh "mongodb://automl:automl_password@localhost:27017/automl" --eval "
        // Create collections with validation
        db.createCollection('architectures', {
            validator: {
                \$jsonSchema: {
                    bsonType: 'object',
                    required: ['id', 'layers', 'created_at'],
                    properties: {
                        id: { bsonType: 'string' },
                        layers: { bsonType: 'array' },
                        created_at: { bsonType: 'date' }
                    }
                }
            }
        });
        
        db.createCollection('training_logs', {
            validator: {
                \$jsonSchema: {
                    bsonType: 'object',
                    required: ['experiment_id', 'timestamp'],
                    properties: {
                        experiment_id: { bsonType: 'string' },
                        timestamp: { bsonType: 'date' }
                    }
                }
            }
        });
        
        db.createCollection('hyperparameter_trials', {
            validator: {
                \$jsonSchema: {
                    bsonType: 'object',
                    required: ['trial_id', 'experiment_id', 'parameters'],
                    properties: {
                        trial_id: { bsonType: 'string' },
                        experiment_id: { bsonType: 'string' },
                        parameters: { bsonType: 'object' }
                    }
                }
            }
        });
        
        // Create indexes
        db.architectures.createIndex({ 'id': 1 }, { unique: true });
        db.architectures.createIndex({ 'created_at': -1 });
        db.training_logs.createIndex({ 'experiment_id': 1, 'timestamp': -1 });
        db.hyperparameter_trials.createIndex({ 'experiment_id': 1 });
        db.hyperparameter_trials.createIndex({ 'trial_id': 1 }, { unique: true });
        
        print('MongoDB collections and indexes created successfully');
    "
    
    print_success "MongoDB setup completed"
}

# Function to initialize Redis
init_redis() {
    print_status "Initializing Redis..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "[DRY RUN] Would initialize Redis configuration"
        return 0
    fi
    
    # Wait for Redis to be ready
    wait_for_database "redis"
    
    # Set up Redis configuration
    print_status "Configuring Redis..."
    redis-cli -a automl_password CONFIG SET save "900 1 300 10 60 10000"
    redis-cli -a automl_password CONFIG SET maxmemory-policy allkeys-lru
    
    print_success "Redis configuration completed"
}

# Function to reset databases
reset_databases() {
    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "[DRY RUN] Would reset all databases"
        return 0
    fi
    
    print_warning "This will destroy all data in the databases!"
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Reset cancelled"
        return 0
    fi
    
    print_status "Resetting databases..."
    
    # Reset PostgreSQL
    if PGPASSWORD=automl_password psql -h localhost -U automl -d automl -c "
        DROP SCHEMA public CASCADE;
        CREATE SCHEMA public;
        GRANT ALL ON SCHEMA public TO automl;
        GRANT ALL ON SCHEMA public TO public;
    " &> /dev/null; then
        print_success "PostgreSQL reset completed"
    else
        print_error "PostgreSQL reset failed"
    fi
    
    # Reset MongoDB
    if mongosh "mongodb://automl:automl_password@localhost:27017/automl" --eval "
        db.dropDatabase();
        print('MongoDB reset completed');
    " &> /dev/null; then
        print_success "MongoDB reset completed"
    else
        print_error "MongoDB reset failed"
    fi
    
    # Reset Redis
    if redis-cli -a automl_password FLUSHALL &> /dev/null; then
        print_success "Redis reset completed"
    else
        print_error "Redis reset failed"
    fi
}

# Function to seed databases with sample data
seed_databases() {
    print_status "Seeding databases with sample data..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "[DRY RUN] Would seed databases with sample data"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    # Create sample data script
    python3 -c "
import sys
sys.path.insert(0, '.')

from datetime import datetime, timezone
from automl_framework.models.data_models import Dataset, DataType, Experiment, ExperimentStatus
from automl_framework.core.database import get_db_session
from automl_framework.models.orm_models import DatasetORM, ExperimentORM, UserORM
import json

# Create database session
session = get_db_session()

try:
    # Create sample user
    sample_user = UserORM(
        id='sample-user-1',
        email='demo@automl.com',
        username='demo_user',
        hashed_password='hashed_demo_password',
        is_active=True,
        created_at=datetime.now(timezone.utc)
    )
    session.merge(sample_user)
    
    # Create sample datasets
    sample_datasets = [
        DatasetORM(
            id='iris-dataset',
            name='Iris Classification Dataset',
            file_path='data/samples/iris.csv',
            data_type='TABULAR',
            size=150,
            features=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            target_column='species',
            metadata={'description': 'Classic iris flower classification dataset'},
            user_id='sample-user-1',
            created_at=datetime.now(timezone.utc)
        ),
        DatasetORM(
            id='housing-dataset',
            name='Boston Housing Dataset',
            file_path='data/samples/housing.csv',
            data_type='TABULAR',
            size=506,
            features=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
            target_column='MEDV',
            metadata={'description': 'Boston housing price prediction dataset'},
            user_id='sample-user-1',
            created_at=datetime.now(timezone.utc)
        )
    ]
    
    for dataset in sample_datasets:
        session.merge(dataset)
    
    # Create sample experiments
    sample_experiments = [
        ExperimentORM(
            id='iris-classification-exp',
            name='Iris Classification Experiment',
            dataset_id='iris-dataset',
            status='COMPLETED',
            user_id='sample-user-1',
            created_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            config={'task_type': 'classification', 'max_trials': 10}
        ),
        ExperimentORM(
            id='housing-regression-exp',
            name='Housing Price Prediction',
            dataset_id='housing-dataset',
            status='RUNNING',
            user_id='sample-user-1',
            created_at=datetime.now(timezone.utc),
            config={'task_type': 'regression', 'max_trials': 20}
        )
    ]
    
    for experiment in sample_experiments:
        session.merge(experiment)
    
    session.commit()
    print('Sample data seeded successfully')
    
except Exception as e:
    session.rollback()
    print(f'Error seeding data: {e}')
    sys.exit(1)
finally:
    session.close()
"
    
    if [[ $? -eq 0 ]]; then
        print_success "Database seeding completed"
    else
        print_error "Database seeding failed"
        return 1
    fi
}

# Function to show migration status
show_migration_status() {
    print_status "Checking migration status..."
    
    cd "$PROJECT_ROOT"
    
    # Check PostgreSQL status
    echo ""
    echo "📊 PostgreSQL Status:"
    if PGPASSWORD=automl_password psql -h localhost -U automl -d automl -c "
        SELECT schemaname, tablename, tableowner 
        FROM pg_tables 
        WHERE schemaname = 'public' 
        ORDER BY tablename;
    " 2>/dev/null; then
        print_success "PostgreSQL tables listed above"
    else
        print_error "Could not connect to PostgreSQL"
    fi
    
    # Check MongoDB status
    echo ""
    echo "📊 MongoDB Status:"
    if mongosh "mongodb://automl:automl_password@localhost:27017/automl" --eval "
        print('Collections:');
        db.getCollectionNames().forEach(function(collection) {
            var count = db.getCollection(collection).countDocuments();
            print('  ' + collection + ': ' + count + ' documents');
        });
    " 2>/dev/null; then
        print_success "MongoDB collections listed above"
    else
        print_error "Could not connect to MongoDB"
    fi
    
    # Check Redis status
    echo ""
    echo "📊 Redis Status:"
    if redis-cli -a automl_password INFO keyspace 2>/dev/null; then
        print_success "Redis keyspace info shown above"
    else
        print_error "Could not connect to Redis"
    fi
}

# Main function
main() {
    local action=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --init)
                action="init"
                shift
                ;;
            --migrate)
                action="migrate"
                shift
                ;;
            --reset)
                action="reset"
                shift
                ;;
            --seed)
                action="seed"
                shift
                ;;
            --status)
                action="status"
                shift
                ;;
            --dry-run)
                DRY_RUN="true"
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
    
    if [[ -z "$action" ]]; then
        print_error "No action specified"
        show_usage
        exit 1
    fi
    
    # Load environment
    load_environment
    
    # Execute action
    case "$action" in
        "init")
            print_status "Initializing databases..."
            init_postgresql
            init_mongodb
            init_redis
            print_success "Database initialization completed"
            ;;
        "migrate")
            print_status "Running migrations..."
            init_postgresql  # This includes migrations
            print_success "Migrations completed"
            ;;
        "reset")
            reset_databases
            if [[ "$DRY_RUN" != "true" ]]; then
                print_status "Reinitializing after reset..."
                init_postgresql
                init_mongodb
                init_redis
            fi
            print_success "Database reset completed"
            ;;
        "seed")
            seed_databases
            ;;
        "status")
            show_migration_status
            ;;
    esac
}

# Handle script interruption
trap 'print_warning "Migration script interrupted by user"; exit 130' INT

# Run main function
main "$@"