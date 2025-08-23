#!/bin/bash

# Google Cloud Platform Setup Script for AutoML Framework
# This script sets up the necessary GCP resources for deploying the AutoML Framework

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
AutoML Framework GCP Setup Script

Usage: $0 [OPTIONS]

Options:
    --project-id PROJECT_ID     GCP Project ID (required)
    --region REGION            GCP Region (default: us-central1)
    --zone ZONE                GCP Zone (default: us-central1-a)
    --cluster-name NAME        GKE Cluster name (default: automl-cluster)
    --enable-apis              Enable required GCP APIs
    --create-cluster           Create GKE cluster
    --setup-storage            Set up Cloud Storage buckets
    --setup-databases          Set up Cloud SQL and other managed databases
    --setup-monitoring         Set up monitoring and logging
    --all                      Run all setup steps
    -h, --help                 Show this help message

Examples:
    $0 --project-id my-project --all
    $0 --project-id my-project --enable-apis --create-cluster
    $0 --project-id my-project --region us-west1 --setup-storage

EOF
}

# Default values
PROJECT_ID=""
REGION="us-central1"
ZONE="us-central1-a"
CLUSTER_NAME="automl-cluster"
ENABLE_APIS=false
CREATE_CLUSTER=false
SETUP_STORAGE=false
SETUP_DATABASES=false
SETUP_MONITORING=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --zone)
            ZONE="$2"
            shift 2
            ;;
        --cluster-name)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        --enable-apis)
            ENABLE_APIS=true
            shift
            ;;
        --create-cluster)
            CREATE_CLUSTER=true
            shift
            ;;
        --setup-storage)
            SETUP_STORAGE=true
            shift
            ;;
        --setup-databases)
            SETUP_DATABASES=true
            shift
            ;;
        --setup-monitoring)
            SETUP_MONITORING=true
            shift
            ;;
        --all)
            ENABLE_APIS=true
            CREATE_CLUSTER=true
            SETUP_STORAGE=true
            SETUP_DATABASES=true
            SETUP_MONITORING=true
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

# Validate required parameters
if [[ -z "$PROJECT_ID" ]]; then
    print_error "Project ID is required. Use --project-id PROJECT_ID"
    exit 1
fi

# Set the project
print_status "Setting GCP project to $PROJECT_ID"
gcloud config set project "$PROJECT_ID"

# Enable APIs
if [[ "$ENABLE_APIS" == "true" ]]; then
    print_status "Enabling required GCP APIs..."
    
    apis=(
        "cloudbuild.googleapis.com"
        "container.googleapis.com"
        "containerregistry.googleapis.com"
        "run.googleapis.com"
        "sql-component.googleapis.com"
        "sqladmin.googleapis.com"
        "storage-component.googleapis.com"
        "monitoring.googleapis.com"
        "logging.googleapis.com"
        "cloudtrace.googleapis.com"
        "cloudprofiler.googleapis.com"
        "compute.googleapis.com"
        "artifactregistry.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        print_status "Enabling $api..."
        gcloud services enable "$api"
    done
    
    print_success "All required APIs enabled"
fi

# Set up Cloud Storage
if [[ "$SETUP_STORAGE" == "true" ]]; then
    print_status "Setting up Cloud Storage buckets..."
    
    # Create artifacts bucket
    ARTIFACTS_BUCKET="${PROJECT_ID}-automl-artifacts"
    if ! gsutil ls "gs://$ARTIFACTS_BUCKET" &>/dev/null; then
        print_status "Creating artifacts bucket: $ARTIFACTS_BUCKET"
        gsutil mb -l "$REGION" "gs://$ARTIFACTS_BUCKET"
        gsutil versioning set on "gs://$ARTIFACTS_BUCKET"
    else
        print_warning "Artifacts bucket already exists: $ARTIFACTS_BUCKET"
    fi
    
    # Create data bucket
    DATA_BUCKET="${PROJECT_ID}-automl-data"
    if ! gsutil ls "gs://$DATA_BUCKET" &>/dev/null; then
        print_status "Creating data bucket: $DATA_BUCKET"
        gsutil mb -l "$REGION" "gs://$DATA_BUCKET"
        
        # Create folder structure
        echo "" | gsutil cp - "gs://$DATA_BUCKET/uploads/.keep"
        echo "" | gsutil cp - "gs://$DATA_BUCKET/models/.keep"
        echo "" | gsutil cp - "gs://$DATA_BUCKET/checkpoints/.keep"
        echo "" | gsutil cp - "gs://$DATA_BUCKET/logs/.keep"
    else
        print_warning "Data bucket already exists: $DATA_BUCKET"
    fi
    
    # Create backup bucket
    BACKUP_BUCKET="${PROJECT_ID}-automl-backups"
    if ! gsutil ls "gs://$BACKUP_BUCKET" &>/dev/null; then
        print_status "Creating backup bucket: $BACKUP_BUCKET"
        gsutil mb -l "$REGION" "gs://$BACKUP_BUCKET"
        gsutil lifecycle set - "gs://$BACKUP_BUCKET" << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {"age": 30}
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "ARCHIVE"},
        "condition": {"age": 365}
      }
    ]
  }
}
EOF
    else
        print_warning "Backup bucket already exists: $BACKUP_BUCKET"
    fi
    
    print_success "Cloud Storage setup completed"
fi

# Set up databases
if [[ "$SETUP_DATABASES" == "true" ]]; then
    print_status "Setting up managed databases..."
    
    # Create Cloud SQL PostgreSQL instance
    POSTGRES_INSTANCE="automl-postgres"
    if ! gcloud sql instances describe "$POSTGRES_INSTANCE" &>/dev/null; then
        print_status "Creating Cloud SQL PostgreSQL instance: $POSTGRES_INSTANCE"
        gcloud sql instances create "$POSTGRES_INSTANCE" \
            --database-version=POSTGRES_14 \
            --tier=db-f1-micro \
            --region="$REGION" \
            --storage-type=SSD \
            --storage-size=20GB \
            --storage-auto-increase \
            --backup-start-time=03:00 \
            --enable-bin-log \
            --maintenance-window-day=SUN \
            --maintenance-window-hour=04 \
            --maintenance-release-channel=production
        
        # Set root password
        print_status "Setting PostgreSQL root password..."
        gcloud sql users set-password postgres \
            --instance="$POSTGRES_INSTANCE" \
            --password="automl_password"
        
        # Create application database
        print_status "Creating application database..."
        gcloud sql databases create automl \
            --instance="$POSTGRES_INSTANCE"
        
        # Create application user
        print_status "Creating application user..."
        gcloud sql users create automl \
            --instance="$POSTGRES_INSTANCE" \
            --password="automl_password"
    else
        print_warning "PostgreSQL instance already exists: $POSTGRES_INSTANCE"
    fi
    
    # Create Memorystore Redis instance
    REDIS_INSTANCE="automl-redis"
    if ! gcloud redis instances describe "$REDIS_INSTANCE" --region="$REGION" &>/dev/null; then
        print_status "Creating Memorystore Redis instance: $REDIS_INSTANCE"
        gcloud redis instances create "$REDIS_INSTANCE" \
            --size=1 \
            --region="$REGION" \
            --redis-version=redis_6_x \
            --auth-enabled
    else
        print_warning "Redis instance already exists: $REDIS_INSTANCE"
    fi
    
    print_success "Managed databases setup completed"
fi

# Create GKE cluster
if [[ "$CREATE_CLUSTER" == "true" ]]; then
    print_status "Creating GKE cluster: $CLUSTER_NAME"
    
    if ! gcloud container clusters describe "$CLUSTER_NAME" --zone="$ZONE" &>/dev/null; then
        gcloud container clusters create "$CLUSTER_NAME" \
            --zone="$ZONE" \
            --machine-type=e2-standard-4 \
            --num-nodes=3 \
            --enable-autoscaling \
            --min-nodes=1 \
            --max-nodes=10 \
            --enable-autorepair \
            --enable-autoupgrade \
            --enable-ip-alias \
            --network=default \
            --subnetwork=default \
            --enable-stackdriver-kubernetes \
            --enable-monitoring \
            --enable-logging \
            --addons=HorizontalPodAutoscaling,HttpLoadBalancing,NetworkPolicy \
            --enable-network-policy \
            --disk-size=50GB \
            --disk-type=pd-ssd \
            --image-type=COS_CONTAINERD \
            --enable-shielded-nodes
        
        # Get cluster credentials
        print_status "Getting cluster credentials..."
        gcloud container clusters get-credentials "$CLUSTER_NAME" --zone="$ZONE"
        
        # Create namespace
        print_status "Creating Kubernetes namespace..."
        kubectl create namespace automl-production --dry-run=client -o yaml | kubectl apply -f -
        
        print_success "GKE cluster created successfully"
    else
        print_warning "GKE cluster already exists: $CLUSTER_NAME"
        gcloud container clusters get-credentials "$CLUSTER_NAME" --zone="$ZONE"
    fi
fi

# Set up monitoring
if [[ "$SETUP_MONITORING" == "true" ]]; then
    print_status "Setting up monitoring and logging..."
    
    # Create log sink for errors
    SINK_NAME="automl-error-sink"
    if ! gcloud logging sinks describe "$SINK_NAME" &>/dev/null; then
        print_status "Creating log sink for errors..."
        gcloud logging sinks create "$SINK_NAME" \
            "storage.googleapis.com/${PROJECT_ID}-automl-logs" \
            --log-filter='severity>=ERROR'
    else
        print_warning "Log sink already exists: $SINK_NAME"
    fi
    
    # Create alerting policy for high error rate
    print_status "Setting up alerting policies..."
    cat > alerting-policy.json << EOF
{
  "displayName": "AutoML High Error Rate",
  "conditions": [
    {
      "displayName": "High error rate condition",
      "conditionThreshold": {
        "filter": "resource.type=\"gce_instance\" AND log_name=\"projects/$PROJECT_ID/logs/automl\"",
        "comparison": "COMPARISON_GREATER_THAN",
        "thresholdValue": 10,
        "duration": "300s",
        "aggregations": [
          {
            "alignmentPeriod": "60s",
            "perSeriesAligner": "ALIGN_RATE",
            "crossSeriesReducer": "REDUCE_SUM"
          }
        ]
      }
    }
  ],
  "combiner": "OR",
  "enabled": true,
  "notificationChannels": []
}
EOF
    
    gcloud alpha monitoring policies create --policy-from-file=alerting-policy.json || true
    rm -f alerting-policy.json
    
    print_success "Monitoring and logging setup completed"
fi

# Create Cloud Build triggers
print_status "Setting up Cloud Build triggers..."

# Create trigger for main branch (production)
if ! gcloud builds triggers describe automl-production &>/dev/null; then
    print_status "Creating production build trigger..."
    gcloud builds triggers create github \
        --repo-name="automl-framework" \
        --repo-owner="your-github-username" \
        --branch-pattern="^main$" \
        --build-config="cloudbuild.yaml" \
        --name="automl-production" \
        --description="Production deployment trigger for AutoML Framework"
else
    print_warning "Production build trigger already exists"
fi

# Create trigger for development branch
if ! gcloud builds triggers describe automl-development &>/dev/null; then
    print_status "Creating development build trigger..."
    gcloud builds triggers create github \
        --repo-name="automl-framework" \
        --repo-owner="your-github-username" \
        --branch-pattern="^develop$" \
        --build-config="cloudbuild-dev.yaml" \
        --name="automl-development" \
        --description="Development deployment trigger for AutoML Framework"
else
    print_warning "Development build trigger already exists"
fi

# Summary
print_success "GCP setup completed successfully!"
echo ""
echo "📋 Setup Summary:"
echo "  Project ID: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Zone: $ZONE"
if [[ "$CREATE_CLUSTER" == "true" ]]; then
    echo "  GKE Cluster: $CLUSTER_NAME"
fi
if [[ "$SETUP_STORAGE" == "true" ]]; then
    echo "  Storage Buckets:"
    echo "    - Artifacts: gs://${PROJECT_ID}-automl-artifacts"
    echo "    - Data: gs://${PROJECT_ID}-automl-data"
    echo "    - Backups: gs://${PROJECT_ID}-automl-backups"
fi
if [[ "$SETUP_DATABASES" == "true" ]]; then
    echo "  Databases:"
    echo "    - PostgreSQL: $POSTGRES_INSTANCE"
    echo "    - Redis: $REDIS_INSTANCE"
fi
echo ""
echo "🚀 Next Steps:"
echo "  1. Update cloudbuild.yaml with your project-specific settings"
echo "  2. Push your code to trigger the first build"
echo "  3. Monitor builds at: https://console.cloud.google.com/cloud-build/builds"
echo "  4. Access your deployed application via Cloud Run or GKE"
echo ""
echo "📚 Documentation:"
echo "  - Cloud Build: https://cloud.google.com/build/docs"
echo "  - GKE: https://cloud.google.com/kubernetes-engine/docs"
echo "  - Cloud Run: https://cloud.google.com/run/docs"