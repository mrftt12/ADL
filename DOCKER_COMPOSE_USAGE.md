# Docker Compose Usage Guide

## Basic Usage (CPU-only)

For development and testing without GPU requirements:

```bash
docker-compose up
```

This will start all services with:
- `DOCKER_CONTAINER=true` - Enables Docker environment detection
- `AUTH_BACKEND=auto` - Automatic authentication backend selection
- CPU-only configurations for all ML workloads
- Resilient service dependencies (services can start even if databases are temporarily unavailable)

## GPU-enabled Usage

If you have NVIDIA Docker runtime and want to enable GPU support:

```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

## Service Dependencies

The configuration has been updated to be more resilient:

- **API Service**: Can start even if database services are not fully healthy
- **Worker Services**: Will restart automatically if they fail
- **Database Services**: Have restart policies to recover from failures
- **Health Checks**: Include appropriate timeouts and start periods

## Environment Variables

All services now include:
- `DOCKER_CONTAINER=true` - For environment detection
- `AUTH_BACKEND=auto` - For automatic authentication backend selection
- `LOG_LEVEL=INFO` - For appropriate logging in Docker environment

## Authentication

The system will automatically:
1. Try to connect to PostgreSQL database
2. Fall back to in-memory authentication if database is unavailable
3. Initialize demo user credentials (demo_user/secret)

## Troubleshooting

If services fail to start:
1. Check logs: `docker-compose logs [service-name]`
2. Services will automatically restart unless stopped manually
3. API service includes a 60-second start period for health checks
4. Database services have health checks with appropriate retry logic