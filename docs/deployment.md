# Deployment Guide

This guide explains how to deploy the Katheryne AI assistant in various environments.

## Local Deployment

### Prerequisites

- Python 3.8+
- Node.js 14+
- 16GB+ RAM
- CUDA-capable GPU (recommended)

### Steps

1. Install dependencies:
```bash
# Python dependencies
pip install -r requirements.txt

# Node.js dependencies
cd node && npm install
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

3. Run services:
```bash
# Start Python backend
python server.py

# Start Node.js frontend
cd node && npm start
```

## Docker Deployment

### Using Docker Compose

1. Build images:
```bash
docker-compose build
```

2. Start services:
```bash
docker-compose up -d
```

### Manual Docker Setup

1. Build backend:
```bash
docker build -t katheryne-backend -f Dockerfile.backend .
```

2. Build frontend:
```bash
docker build -t katheryne-frontend -f Dockerfile.frontend .
```

3. Run containers:
```bash
docker run -d --name katheryne-backend katheryne-backend
docker run -d --name katheryne-frontend katheryne-frontend
```

## Cloud Deployment

### AWS

1. ECR Setup:
```bash
aws ecr create-repository --repository-name katheryne
```

2. Push images:
```bash
docker tag katheryne:latest $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/katheryne
docker push $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/katheryne
```

3. ECS Deployment:
```bash
aws ecs create-service --cluster katheryne --service-name api --task-definition katheryne:1
```

### Google Cloud

1. Enable services:
```bash
gcloud services enable containerregistry.googleapis.com
gcloud services enable run.googleapis.com
```

2. Push images:
```bash
docker tag katheryne:latest gcr.io/$PROJECT_ID/katheryne
docker push gcr.io/$PROJECT_ID/katheryne
```

3. Deploy to Cloud Run:
```bash
gcloud run deploy katheryne --image gcr.io/$PROJECT_ID/katheryne
```

## Monitoring Setup

### Prometheus & Grafana

1. Add Prometheus config:
```yaml
scrape_configs:
  - job_name: 'katheryne'
    static_configs:
      - targets: ['localhost:8000']
```

2. Configure Grafana dashboard:
```bash
curl -X POST -H "Content-Type: application/json" -d @dashboards/katheryne.json \
  http://localhost:3000/api/dashboards/db
```

### Logging

1. Configure logging:
```python
import logging

logging.config.dictConfig({
    'version': 1,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'katheryne.log',
            'level': 'DEBUG'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
})
```

2. Set up log rotation:
```bash
cat > /etc/logrotate.d/katheryne << EOF
/var/log/katheryne/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
EOF
```

## Security Configuration

### SSL/TLS Setup

1. Generate certificates:
```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout private.key -out certificate.crt
```

2. Configure NGINX:
```nginx
server {
    listen 443 ssl;
    server_name api.katheryne.ai;

    ssl_certificate /etc/nginx/certs/certificate.crt;
    ssl_certificate_key /etc/nginx/certs/private.key;

    location / {
        proxy_pass http://localhost:8000;
    }
}
```

### Firewall Configuration

```bash
# Allow HTTP/HTTPS
ufw allow 80/tcp
ufw allow 443/tcp

# Allow API port
ufw allow 8000/tcp
```

## Scaling

### Horizontal Scaling

1. Configure load balancer:
```nginx
upstream katheryne {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}
```

2. Update Docker Compose:
```yaml
services:
  backend:
    image: katheryne-backend
    deploy:
      replicas: 3
```

### Vertical Scaling

1. Update resource limits:
```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
```

## Backup and Recovery

### Database Backup

1. Configure automated backups:
```bash
#!/bin/bash
DATE=$(date +%Y%m%d)
mongodump --out /backups/$DATE
```

2. Set up rotation:
```bash
find /backups -type d -mtime +30 -exec rm -rf {} +
```

### Model Backup

1. Save model checkpoints:
```python
model.save_checkpoint(f'checkpoints/model_{timestamp}.pt')
```

2. Configure S3 backup:
```python
import boto3

s3 = boto3.client('s3')
s3.upload_file(
    f'checkpoints/model_{timestamp}.pt',
    'katheryne-backups',
    f'models/model_{timestamp}.pt'
)
```

## Troubleshooting

### Common Issues

1. Memory Issues
```bash
# Check memory usage
free -h

# Increase swap
sudo fallocate -l 4G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

2. CPU Usage
```bash
# Monitor CPU
top -b -n 1

# Set CPU affinity
taskset -c 0-3 python server.py
```

3. Network Issues
```bash
# Check connections
netstat -tulpn

# Monitor network
iftop -i eth0
```

## Performance Optimization

### Caching

1. Redis setup:
```python
import redis

redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0
)
```

2. Cache configuration:
```python
CACHE_CONFIG = {
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': 'redis://localhost:6379/0',
    'CACHE_DEFAULT_TIMEOUT': 300
}
```

### Query Optimization

1. Batch processing:
```python
async def process_batch(queries: List[str]) -> List[Dict]:
    async with aiohttp.ClientSession() as session:
        tasks = [process_query(q, session) for q in queries]
        return await asyncio.gather(*tasks)
```

2. Response compression:
```python
from flask_compress import Compress

app = Flask(__name__)
Compress(app)
```

## Maintenance

### Regular Tasks

1. Log rotation
2. Database cleanup
3. Model updates
4. Security patches
5. Performance monitoring

### Update Process

1. Backup current state
2. Stop services
3. Apply updates
4. Run tests
5. Restart services
6. Verify functionality

## Health Checks

### Backend Health

```python
@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'version': VERSION,
        'uptime': get_uptime(),
        'memory_usage': get_memory_usage()
    }
```

### Frontend Health

```typescript
async function checkHealth(): Promise<boolean> {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        return data.status === 'healthy';
    } catch (error) {
        return false;
    }
}
```

## Metrics Collection

### Application Metrics

```python
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    'request_total',
    'Total request count',
    ['method', 'endpoint']
)

RESPONSE_TIME = Histogram(
    'response_time_seconds',
    'Response time in seconds',
    ['method', 'endpoint']
)
```

### System Metrics

```python
def collect_metrics():
    return {
        'cpu_usage': psutil.cpu_percent(),
        'memory_usage': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'network_io': psutil.net_io_counters()
    }
```

## Disaster Recovery

### Backup Strategy

1. Daily database backups
2. Weekly full system backups
3. Continuous model checkpoints
4. Configuration version control

### Recovery Steps

1. Assess the situation
2. Restore from latest backup
3. Verify data integrity
4. Test system functionality
5. Resume operations

## Documentation

Keep deployment documentation updated:
```bash
# Generate deployment docs
sphinx-build -b html docs/deployment _build/html
```

## Support

For deployment issues:
1. Check logs
2. Consult documentation
3. Open GitHub issue
4. Contact support team