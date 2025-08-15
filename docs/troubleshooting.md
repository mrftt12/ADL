# AutoML Framework Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the AutoML Framework.

## 📋 Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Installation Issues](#installation-issues)
- [Service Startup Problems](#service-startup-problems)
- [Database Issues](#database-issues)
- [GPU and Training Issues](#gpu-and-training-issues)
- [Performance Problems](#performance-problems)
- [API and Frontend Issues](#api-and-frontend-issues)
- [Docker Issues](#docker-issues)
- [Network and Connectivity](#network-and-connectivity)
- [Data and Experiment Issues](#data-and-experiment-issues)
- [Getting Help](#getting-help)

## 🔍 Quick Diagnostics

### Health Check Script

Run the comprehensive health check:

```bash
./scripts/dev-health.sh
```

For continuous monitoring:

```bash
./scripts/dev-health.sh --watch
```

### System Information

```bash
# Check system resources
df -h                    # Disk space
free -h                  # Memory usage
docker system df         # Docker space usage
nvidia-smi              # GPU status (if available)

# Check running processes
ps aux | grep automl
lsof -i :8000           # API port
lsof -i :3000           # Frontend port
```

### Service Status

```bash
# Docker services
docker-compose ps

# Native services
./scripts/dev-health.sh --quiet

# Check logs
./scripts/dev-start.sh --logs
```

## 🛠️ Installation Issues

### Docker Installation Problems

#### Docker Not Found

**Error**: `docker: command not found`

**Solution**:
```bash
# macOS with Homebrew
brew install docker docker-compose

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Verify installation
docker --version
docker-compose --version
```

#### Docker Permission Denied

**Error**: `permission denied while trying to connect to the Docker daemon socket`

**Solution**:
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Restart session or run:
newgrp docker

# Test access
docker ps
```

#### Docker Compose Version Issues

**Error**: `docker-compose version 1.x is not supported`

**Solution**:
```bash
# Install Docker Compose v2
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Or use Docker Compose plugin
docker compose version
```

### Python Environment Issues

#### Python Version Incompatibility

**Error**: `Python 3.8+ required`

**Solution**:
```bash
# Check Python version
python3 --version

# Install Python 3.8+ (Ubuntu)
sudo apt-get install python3.8 python3.8-venv python3.8-dev

# Create virtual environment
python3.8 -m venv venv
source venv/bin/activate
```

#### Package Installation Failures

**Error**: `Failed building wheel for package`

**Solution**:
```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Install system dependencies (Ubuntu)
sudo apt-get install build-essential python3-dev

# Install with verbose output
pip install -v -r requirements.txt
```

### Node.js Issues

#### Node.js Version Problems

**Error**: `Node.js 16+ required`

**Solution**:
```bash
# Install Node.js via nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 18
nvm use 18

# Verify installation
node --version
npm --version
```

## 🚀 Service Startup Problems

### Services Won't Start

#### Port Already in Use

**Error**: `Port 8000 is already in use`

**Solution**:
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
export API_PORT=8001
./scripts/dev-start.sh
```

#### Insufficient Resources

**Error**: `Cannot allocate memory` or `No space left on device`

**Solution**:
```bash
# Check disk space
df -h

# Clean up Docker resources
docker system prune -f
docker volume prune -f

# Check memory usage
free -h

# Reduce service resource requirements
# Edit docker-compose.yml to lower memory limits
```

#### Service Dependencies

**Error**: `Connection refused` or `Service unavailable`

**Solution**:
```bash
# Check service startup order
docker-compose logs postgres
docker-compose logs mongodb
docker-compose logs redis

# Restart with proper dependencies
docker-compose down
docker-compose up -d postgres mongodb redis
sleep 30
docker-compose up -d api
```

### Environment Configuration Issues

#### Missing Environment Variables

**Error**: `Environment variable not set`

**Solution**:
```bash
# Check environment file
cat .env.dev

# Recreate environment file
./scripts/dev-start.sh  # This creates .env.dev automatically

# Or manually set variables
export DATABASE_URL="postgresql://automl:automl_password@localhost:5432/automl"
export MONGODB_URL="mongodb://automl:automl_password@localhost:27017/automl"
```

## 💾 Database Issues

### PostgreSQL Problems

#### Connection Refused

**Error**: `could not connect to server: Connection refused`

**Solution**:
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Check PostgreSQL logs
docker-compose logs postgres

# Restart PostgreSQL
docker-compose restart postgres

# Wait for health check
./scripts/dev-health.sh
```

#### Authentication Failed

**Error**: `FATAL: password authentication failed`

**Solution**:
```bash
# Check credentials in environment
echo $DATABASE_URL

# Reset PostgreSQL password
docker-compose down
docker volume rm automl-framework_postgres_data
docker-compose up -d postgres

# Run migrations
./scripts/dev-migrate.sh --init
```

#### Database Does Not Exist

**Error**: `database "automl" does not exist`

**Solution**:
```bash
# Initialize database
./scripts/dev-migrate.sh --init

# Or manually create
docker-compose exec postgres createdb -U automl automl
```

### MongoDB Problems

#### Connection Timeout

**Error**: `MongoServerError: connection timeout`

**Solution**:
```bash
# Check MongoDB status
docker-compose ps mongodb
docker-compose logs mongodb

# Restart MongoDB
docker-compose restart mongodb

# Test connection
mongosh "mongodb://automl:automl_password@localhost:27017/automl"
```

#### Authentication Error

**Error**: `MongoServerError: Authentication failed`

**Solution**:
```bash
# Reset MongoDB
docker-compose down
docker volume rm automl-framework_mongodb_data
docker-compose up -d mongodb

# Wait for initialization
sleep 30
./scripts/dev-migrate.sh --init
```

### Redis Problems

#### Redis Not Available

**Error**: `Redis connection failed`

**Solution**:
```bash
# Check Redis status
docker-compose ps redis
docker-compose logs redis

# Test Redis connection
redis-cli -a automl_password ping

# Restart Redis
docker-compose restart redis
```

## 🖥️ GPU and Training Issues

### GPU Not Detected

#### NVIDIA Drivers

**Error**: `CUDA device not available`

**Solution**:
```bash
# Check NVIDIA drivers
nvidia-smi

# Install NVIDIA drivers (Ubuntu)
sudo apt-get install nvidia-driver-470
sudo reboot

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

#### Docker GPU Support

**Error**: `could not select device driver "nvidia"`

**Solution**:
```bash
# Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Training Failures

#### Out of Memory

**Error**: `CUDA out of memory`

**Solution**:
```bash
# Reduce batch size
export DEFAULT_BATCH_SIZE=16

# Limit GPU memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use gradient checkpointing
export ENABLE_GRADIENT_CHECKPOINTING=true

# Monitor GPU usage
nvidia-smi -l 1
```

#### Training Stuck

**Error**: Training progress stops

**Solution**:
```bash
# Check training logs
docker-compose logs training-worker

# Check GPU utilization
nvidia-smi

# Restart training worker
docker-compose restart training-worker

# Check for deadlocks
docker-compose exec training-worker python -c "
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
"
```

## ⚡ Performance Problems

### Slow Training

#### CPU Bottleneck

**Symptoms**: Low GPU utilization, high CPU usage

**Solution**:
```bash
# Increase data loading workers
export NUM_WORKERS=4

# Use faster data loading
export USE_FAST_DATALOADER=true

# Optimize preprocessing
export CACHE_PREPROCESSED_DATA=true
```

#### I/O Bottleneck

**Symptoms**: High disk I/O wait times

**Solution**:
```bash
# Use faster storage for data
mkdir -p /tmp/automl_data
export DATA_DIR=/tmp/automl_data

# Enable data caching
export ENABLE_DATA_CACHE=true

# Use SSD for Docker volumes
# Edit docker-compose.yml to use bind mounts to SSD
```

### Memory Issues

#### High Memory Usage

**Symptoms**: System becomes unresponsive

**Solution**:
```bash
# Monitor memory usage
./scripts/dev-health.sh --watch

# Reduce worker concurrency
export WORKER_CONCURRENCY=2

# Limit model size
export MAX_MODEL_PARAMETERS=1000000

# Enable memory profiling
export ENABLE_MEMORY_PROFILING=true
```

#### Memory Leaks

**Symptoms**: Memory usage increases over time

**Solution**:
```bash
# Restart workers periodically
docker-compose restart training-worker nas-worker hpo-worker

# Enable garbage collection
export FORCE_GC_AFTER_TRIAL=true

# Monitor memory leaks
docker stats --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}"
```

## 🌐 API and Frontend Issues

### API Problems

#### API Not Responding

**Error**: `Connection refused` to API

**Solution**:
```bash
# Check API status
curl http://localhost:8000/health

# Check API logs
docker-compose logs api

# Restart API
docker-compose restart api

# Check port binding
docker-compose ps api
```

#### Authentication Issues

**Error**: `401 Unauthorized`

**Solution**:
```bash
# Check JWT configuration
echo $JWT_SECRET_KEY

# Generate new JWT secret
export JWT_SECRET_KEY=$(openssl rand -base64 32)

# Clear browser cookies/localStorage
# Restart API service
docker-compose restart api
```

### Frontend Problems

#### Frontend Won't Load

**Error**: `This site can't be reached`

**Solution**:
```bash
# Check frontend status
curl http://localhost:3000

# Check frontend logs
docker-compose logs frontend

# Rebuild frontend
docker-compose build frontend
docker-compose up -d frontend
```

#### API Connection Issues

**Error**: `Network Error` in browser console

**Solution**:
```bash
# Check API URL in frontend
# Edit ui/.env or ui/src/lib/api-client.ts

# Check CORS configuration
# Verify API allows frontend origin

# Check network connectivity
curl -I http://localhost:8000/api/v1/health
```

## 🐳 Docker Issues

### Container Problems

#### Container Exits Immediately

**Error**: Container stops right after starting

**Solution**:
```bash
# Check container logs
docker-compose logs <service_name>

# Run container interactively
docker-compose run --rm <service_name> /bin/bash

# Check Dockerfile syntax
docker build -t test .
```

#### Build Failures

**Error**: `failed to build`

**Solution**:
```bash
# Clean build cache
docker builder prune -f

# Build with no cache
docker-compose build --no-cache

# Check Dockerfile syntax
docker build --progress=plain -t test .
```

### Volume Issues

#### Permission Denied

**Error**: `Permission denied` accessing volumes

**Solution**:
```bash
# Fix volume permissions
sudo chown -R $USER:$USER data/ logs/ models/ checkpoints/

# Or use Docker user mapping
# Add to docker-compose.yml:
# user: "${UID}:${GID}"
```

#### Volume Not Mounting

**Error**: Files not persisting between container restarts

**Solution**:
```bash
# Check volume configuration
docker volume ls
docker volume inspect automl-framework_postgres_data

# Recreate volumes
docker-compose down -v
docker-compose up -d
```

## 🔗 Network and Connectivity

### Port Conflicts

#### Port Already in Use

**Error**: `bind: address already in use`

**Solution**:
```bash
# Find process using port
sudo lsof -i :8000

# Kill process
sudo kill -9 <PID>

# Use different ports
# Edit docker-compose.yml or use environment variables
export API_PORT=8001
export FRONTEND_PORT=3001
```

### DNS Issues

#### Service Discovery

**Error**: `could not resolve hostname`

**Solution**:
```bash
# Check Docker network
docker network ls
docker network inspect automl-framework_automl-network

# Recreate network
docker-compose down
docker network prune -f
docker-compose up -d
```

## 📊 Data and Experiment Issues

### Dataset Problems

#### Upload Failures

**Error**: `Failed to upload dataset`

**Solution**:
```bash
# Check file size limits
# Default limit is 1GB, increase if needed

# Check disk space
df -h

# Check upload directory permissions
ls -la data/uploads/

# Create upload directory
mkdir -p data/uploads
chmod 755 data/uploads
```

#### Data Processing Errors

**Error**: `Failed to process dataset`

**Solution**:
```bash
# Check dataset format
head -5 your_dataset.csv

# Validate CSV format
python -c "
import pandas as pd
df = pd.read_csv('your_dataset.csv')
print(df.info())
print(df.head())
"

# Check for encoding issues
file -bi your_dataset.csv
```

### Experiment Issues

#### Experiments Stuck

**Error**: Experiment status remains "RUNNING" indefinitely

**Solution**:
```bash
# Check worker logs
docker-compose logs training-worker nas-worker hpo-worker

# Check experiment status in database
./scripts/dev-migrate.sh --status

# Restart workers
docker-compose restart training-worker nas-worker hpo-worker

# Cancel stuck experiment
curl -X POST http://localhost:8000/api/v1/experiments/{experiment_id}/stop \
  -H "Authorization: Bearer <token>"
```

#### Poor Model Performance

**Symptoms**: All models perform poorly

**Solution**:
```bash
# Check data quality
python -c "
import pandas as pd
df = pd.read_csv('your_dataset.csv')
print('Missing values:', df.isnull().sum())
print('Data types:', df.dtypes)
print('Target distribution:', df['target_column'].value_counts())
"

# Increase experiment budget
# Edit experiment config:
# - max_trials: 100
# - max_epochs: 200

# Check for data leakage
# Ensure proper train/validation/test splits
```

## 🆘 Getting Help

### Collecting Debug Information

When reporting issues, include:

```bash
# System information
uname -a
docker --version
docker-compose --version
python3 --version
node --version

# Service status
./scripts/dev-health.sh > health_check.txt

# Logs
docker-compose logs > docker_logs.txt

# Resource usage
docker stats --no-stream > docker_stats.txt
df -h > disk_usage.txt
free -h > memory_usage.txt
```

### Log Analysis

#### Enable Debug Logging

```bash
# Set debug level
export LOG_LEVEL=DEBUG

# Restart services
./scripts/dev-start.sh --restart

# Follow logs
./scripts/dev-start.sh --logs
```

#### Common Log Patterns

```bash
# Search for errors
docker-compose logs | grep -i error

# Search for specific service issues
docker-compose logs api | grep -i "database"
docker-compose logs training-worker | grep -i "cuda"

# Monitor real-time logs
docker-compose logs -f --tail=100
```

### Performance Profiling

#### Enable Profiling

```bash
# Enable API profiling
export ENABLE_PROFILING=true

# Enable memory profiling
export ENABLE_MEMORY_PROFILING=true

# Enable GPU profiling
export ENABLE_GPU_PROFILING=true

# Restart services
docker-compose restart api training-worker
```

#### Analyze Performance

```bash
# Check API performance
curl http://localhost:8000/api/v1/metrics

# Monitor GPU usage
nvidia-smi -l 1

# Monitor system resources
htop
iotop
```

### Community Support

1. **GitHub Issues**: Create detailed issue reports
2. **Documentation**: Check latest docs at `/docs`
3. **Health Check**: Always run `./scripts/dev-health.sh` first
4. **Logs**: Include relevant log snippets
5. **Environment**: Specify your OS, Docker version, and hardware

### Emergency Recovery

#### Complete Reset

```bash
# Stop all services
./scripts/dev-stop.sh --force

# Clean everything
docker-compose down --volumes --remove-orphans
docker system prune -a -f
docker volume prune -f

# Remove data (WARNING: destroys all data)
rm -rf data/ logs/ models/ checkpoints/

# Fresh start
./scripts/dev-start.sh --clean
```

#### Backup Before Reset

```bash
# Backup important data
./scripts/backup.sh

# Then proceed with reset
./scripts/dev-stop.sh --clean
./scripts/dev-start.sh
```

This troubleshooting guide covers the most common issues. If you encounter problems not covered here, please check the GitHub issues or create a new issue with detailed information about your problem.