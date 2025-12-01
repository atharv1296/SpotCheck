# Forbes Marshall SpotCheck - Production Deployment Guide

## Overview
This guide covers the complete deployment process for the Forbes Marshall SpotCheck parking management system in production environments.

**System Requirements:**
- Ubuntu 20.04+ or CentOS 8+ (Linux)
- Python 3.8+
- PostgreSQL 12+ or MySQL 8.0+ (Production database)
- Redis 6.0+ (Caching and sessions)
- Nginx 1.18+ (Web server and reverse proxy)
- SSL certificate (Let's Encrypt recommended)
- Minimum 4GB RAM, 2 CPU cores, 20GB storage

## Pre-Deployment Checklist

### 1. Server Preparation
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3 python3-pip python3-venv postgresql postgresql-contrib redis-server nginx git supervisor certbot python3-certbot-nginx

# Create application user
sudo useradd -m -s /bin/bash spotcheck
sudo usermod -aG sudo spotcheck
```

### 2. Database Setup (PostgreSQL)
```bash
# Switch to postgres user
sudo -u postgres psql

-- Create database and user
CREATE DATABASE spotcheck_prod;
CREATE USER spotcheck_user WITH PASSWORD 'secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE spotcheck_prod TO spotcheck_user;
ALTER USER spotcheck_user CREATEDB;
\q
```

### 3. Redis Configuration
```bash
# Configure Redis for production
sudo nano /etc/redis/redis.conf

# Key settings to modify:
# maxmemory 1gb
# maxmemory-policy allkeys-lru
# save 900 1
# requirepass your_redis_password

# Restart Redis
sudo systemctl restart redis
sudo systemctl enable redis
```

## Application Deployment

### 1. Clone and Setup Application
```bash
# Switch to application user
sudo su - spotcheck

# Clone repository
git clone https://github.com/forbes-marshall/spotcheck.git
cd spotcheck/dashboard

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install gunicorn psycopg2-binary redis django-redis
```

### 2. Production Settings Configuration
```bash
# Create production settings file
cp parking_project/settings.py parking_project/settings_prod.py
```

**Edit `parking_project/settings_prod.py`:**
```python
import os
from .settings import *

# Security Settings
DEBUG = False
ALLOWED_HOSTS = ['your-domain.com', 'www.your-domain.com', 'SERVER_IP']

# Security Headers
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
X_FRAME_OPTIONS = 'DENY'
SECURE_REFERRER_POLICY = 'same-origin'

# SSL Settings (Enable after SSL certificate installation)
# SECURE_SSL_REDIRECT = True
# SESSION_COOKIE_SECURE = True
# CSRF_COOKIE_SECURE = True

# Database Configuration
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'spotcheck_prod',
        'USER': 'spotcheck_user',
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': 'localhost',
        'PORT': '5432',
        'OPTIONS': {
            'sslmode': 'prefer',
        },
    }
}

# Redis Configuration
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'PASSWORD': os.environ.get('REDIS_PASSWORD'),
        }
    }
}

# Session Configuration
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'default'
SESSION_COOKIE_AGE = 3600  # 1 hour

# Email Configuration
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD')
DEFAULT_FROM_EMAIL = 'Forbes Marshall SpotCheck <noreply@forbesmarshall.com>'

# Logging Configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/spotcheck/django.log',
            'maxBytes': 1024*1024*10,  # 10MB
            'backupCount': 5,
            'formatter': 'verbose',
        },
        'error_file': {
            'level': 'ERROR',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/spotcheck/error.log',
            'maxBytes': 1024*1024*10,  # 10MB
            'backupCount': 5,
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file', 'error_file'],
            'level': 'INFO',
            'propagate': True,
        },
        'parking_app': {
            'handlers': ['file', 'error_file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}

# Static and Media Files
STATIC_ROOT = '/var/www/spotcheck/static/'
MEDIA_ROOT = '/var/www/spotcheck/media/'

# Performance Settings
CONN_MAX_AGE = 60
USE_TZ = True
```

### 3. Environment Variables Setup
```bash
# Create environment file
nano ~/.bashrc

# Add environment variables
export DB_PASSWORD='your_secure_db_password'
export REDIS_PASSWORD='your_redis_password'
export EMAIL_HOST_USER='your_email@gmail.com'
export EMAIL_HOST_PASSWORD='your_app_password'
export DJANGO_SECRET_KEY='your_very_secure_secret_key'
export DJANGO_SETTINGS_MODULE='parking_project.settings_prod'

# Reload environment
source ~/.bashrc
```

### 4. Database Migration and Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Run migrations
python manage.py migrate

# Create static files directory
sudo mkdir -p /var/www/spotcheck/static /var/www/spotcheck/media
sudo chown -R spotcheck:spotcheck /var/www/spotcheck

# Collect static files
python manage.py collectstatic --noinput

# Create superuser
python manage.py createsuperuser

# Initialize parking system
python manage.py init_parking_system

# Load sample data (optional)
python manage.py loaddata fixtures/sample_data.json
```

### 5. Log Directory Setup
```bash
# Create log directory
sudo mkdir -p /var/log/spotcheck
sudo chown -R spotcheck:spotcheck /var/log/spotcheck
sudo chmod 755 /var/log/spotcheck
```

## Web Server Configuration

### 1. Gunicorn Configuration
```bash
# Create Gunicorn configuration
nano /home/spotcheck/spotcheck/dashboard/gunicorn.conf.py
```

**Gunicorn configuration:**
```python
# gunicorn.conf.py
import multiprocessing

bind = "127.0.0.1:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True
timeout = 30
keepalive = 2
user = "spotcheck"
group = "spotcheck"
pidfile = "/tmp/gunicorn.pid"
accesslog = "/var/log/spotcheck/gunicorn_access.log"
errorlog = "/var/log/spotcheck/gunicorn_error.log"
loglevel = "info"
```

### 2. Supervisor Configuration
```bash
# Create supervisor configuration
sudo nano /etc/supervisor/conf.d/spotcheck.conf
```

**Supervisor configuration:**
```ini
[program:spotcheck]
command=/home/spotcheck/spotcheck/dashboard/venv/bin/gunicorn parking_project.wsgi:application -c /home/spotcheck/spotcheck/dashboard/gunicorn.conf.py
directory=/home/spotcheck/spotcheck/dashboard
user=spotcheck
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/spotcheck/supervisor.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=5
environment=DJANGO_SETTINGS_MODULE="parking_project.settings_prod"
```

```bash
# Update supervisor
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start spotcheck
```

### 3. Nginx Configuration
```bash
# Create Nginx site configuration
sudo nano /etc/nginx/sites-available/spotcheck
```

**Nginx configuration:**
```nginx
upstream spotcheck_app {
    server 127.0.0.1:8000;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;

server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # File upload limit
    client_max_body_size 10M;
    
    # Static files
    location /static/ {
        alias /var/www/spotcheck/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
        access_log off;
    }
    
    location /media/ {
        alias /var/www/spotcheck/media/;
        expires 7d;
        add_header Cache-Control "public";
    }
    
    # Rate limiting for API endpoints
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://spotcheck_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
    }
    
    # Rate limiting for login
    location /admin/login/ {
        limit_req zone=login burst=5 nodelay;
        proxy_pass http://spotcheck_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Main application
    location / {
        proxy_pass http://spotcheck_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Health check endpoint
    location /health/ {
        access_log off;
        proxy_pass http://spotcheck_app;
        proxy_set_header Host $host;
    }
    
    # Block sensitive files
    location ~ /\. {
        deny all;
        access_log off;
        log_not_found off;
    }
    
    location ~ \.(ini|log|conf)$ {
        deny all;
        access_log off;
        log_not_found off;
    }
}
```

```bash
# Enable site and restart Nginx
sudo ln -s /etc/nginx/sites-available/spotcheck /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx
```

## SSL Certificate Installation

### 1. Let's Encrypt SSL Certificate
```bash
# Install SSL certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Verify auto-renewal
sudo certbot renew --dry-run

# Setup auto-renewal cron job
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 2. Enable SSL in Django Settings
After SSL installation, uncomment SSL settings in `settings_prod.py`:
```python
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
```

## Monitoring and Maintenance

### 1. System Monitoring Script
```bash
# Create monitoring script
nano /home/spotcheck/monitor.sh
```

```bash
#!/bin/bash
# Forbes Marshall SpotCheck System Monitor

LOG_FILE="/var/log/spotcheck/monitor.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$DATE] Starting system check..." >> $LOG_FILE

# Check application status
if ! supervisorctl status spotcheck | grep -q RUNNING; then
    echo "[$DATE] ERROR: SpotCheck application is not running!" >> $LOG_FILE
    supervisorctl restart spotcheck
fi

# Check database connectivity
if ! su - spotcheck -c "cd /home/spotcheck/spotcheck/dashboard && source venv/bin/activate && python manage.py system_health" > /dev/null 2>&1; then
    echo "[$DATE] ERROR: Database connectivity issue!" >> $LOG_FILE
fi

# Check disk space
DISK_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
    echo "[$DATE] WARNING: Disk usage is ${DISK_USAGE}%!" >> $LOG_FILE
fi

# Check memory usage
MEMORY_USAGE=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
if [ $MEMORY_USAGE -gt 90 ]; then
    echo "[$DATE] WARNING: Memory usage is ${MEMORY_USAGE}%!" >> $LOG_FILE
fi

echo "[$DATE] System check completed." >> $LOG_FILE
```

```bash
# Make executable and setup cron job
chmod +x /home/spotcheck/monitor.sh
crontab -e
# Add: */5 * * * * /home/spotcheck/monitor.sh
```

### 2. Backup Script
```bash
# Create backup script
nano /home/spotcheck/backup.sh
```

```bash
#!/bin/bash
# Forbes Marshall SpotCheck Backup Script

BACKUP_DIR="/backup/spotcheck"
DATE=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="/var/log/spotcheck/backup.log"

mkdir -p $BACKUP_DIR

echo "[$(date)] Starting backup..." >> $LOG_FILE

# Database backup
pg_dump -U spotcheck_user -h localhost spotcheck_prod | gzip > $BACKUP_DIR/db_backup_$DATE.sql.gz

# Application backup
tar -czf $BACKUP_DIR/app_backup_$DATE.tar.gz /home/spotcheck/spotcheck/

# Media files backup
tar -czf $BACKUP_DIR/media_backup_$DATE.tar.gz /var/www/spotcheck/media/

# Clean old backups (keep 7 days)
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete

echo "[$(date)] Backup completed successfully." >> $LOG_FILE
```

```bash
# Setup daily backup
chmod +x /home/spotcheck/backup.sh
crontab -e
# Add: 0 2 * * * /home/spotcheck/backup.sh
```

### 3. Log Rotation
```bash
# Create logrotate configuration
sudo nano /etc/logrotate.d/spotcheck
```

```
/var/log/spotcheck/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 644 spotcheck spotcheck
    postrotate
        supervisorctl restart spotcheck
    endscript
}
```

## Performance Optimization

### 1. Database Optimization
```sql
-- Connect to PostgreSQL
sudo -u postgres psql spotcheck_prod

-- Create indexes for better performance
CREATE INDEX idx_parking_slot_status ON parking_app_parkingslot(status);
CREATE INDEX idx_parking_slot_type ON parking_app_parkingslot(slot_type);
CREATE INDEX idx_parking_session_active ON parking_app_parkingsession(is_active);
CREATE INDEX idx_parking_session_entry_time ON parking_app_parkingsession(entry_time);
CREATE INDEX idx_vehicle_license_plate ON parking_app_vehicle(license_plate);

-- Analyze tables
ANALYZE;
```

### 2. Redis Optimization
```bash
# Optimize Redis memory usage
sudo nano /etc/redis/redis.conf

# Add/modify these settings:
# maxmemory-samples 5
# hash-max-ziplist-entries 512
# hash-max-ziplist-value 64
# list-max-ziplist-size -2
# set-max-intset-entries 512
# zset-max-ziplist-entries 128
# zset-max-ziplist-value 64

sudo systemctl restart redis
```

## Security Hardening

### 1. Firewall Configuration
```bash
# Configure UFW firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

### 2. Fail2Ban Configuration
```bash
# Install and configure Fail2Ban
sudo apt install fail2ban

# Create jail configuration
sudo nano /etc/fail2ban/jail.local
```

```ini
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true

[nginx-http-auth]
enabled = true

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
action = iptables-multiport[name=ReqLimit, port="http,https", protocol=tcp]
logpath = /var/log/nginx/error.log
maxretry = 10
```

```bash
sudo systemctl restart fail2ban
sudo systemctl enable fail2ban
```

### 3. System Updates
```bash
# Setup automatic security updates
sudo apt install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

## Deployment Checklist

### Pre-Go-Live
- [ ] Server resources adequate (CPU, RAM, Storage)
- [ ] Database configured and optimized
- [ ] Redis cache configured
- [ ] SSL certificate installed and configured
- [ ] Firewall rules configured
- [ ] Monitoring and alerting setup
- [ ] Backup system configured
- [ ] Log rotation configured
- [ ] Performance testing completed
- [ ] Security scanning completed

### Go-Live
- [ ] DNS records updated
- [ ] Application deployed and running
- [ ] SSL certificate verified
- [ ] All endpoints tested
- [ ] WebSocket connections tested
- [ ] Email notifications tested
- [ ] Backup system tested
- [ ] Monitoring alerts tested

### Post Go-Live
- [ ] Monitor system performance
- [ ] Check log files for errors
- [ ] Verify backup completion
- [ ] Update documentation
- [ ] Train operations team

## Troubleshooting

### Common Issues

**1. Application won't start:**
```bash
# Check logs
sudo tail -f /var/log/spotcheck/supervisor.log
sudo tail -f /var/log/spotcheck/gunicorn_error.log

# Check supervisor status
sudo supervisorctl status spotcheck

# Restart application
sudo supervisorctl restart spotcheck
```

**2. Database connection issues:**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test database connection
sudo -u postgres psql -c "SELECT version();"

# Check connection from application
su - spotcheck -c "cd /home/spotcheck/spotcheck/dashboard && source venv/bin/activate && python manage.py dbshell"
```

**3. Static files not loading:**
```bash
# Check static files collection
su - spotcheck -c "cd /home/spotcheck/spotcheck/dashboard && source venv/bin/activate && python manage.py collectstatic --noinput"

# Check file permissions
sudo chown -R spotcheck:spotcheck /var/www/spotcheck/
sudo chmod -R 755 /var/www/spotcheck/static/
```

**4. High memory usage:**
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Restart services if needed
sudo systemctl restart redis
sudo supervisorctl restart spotcheck
```

## Support and Maintenance

### Regular Maintenance Tasks
- **Daily**: Check system logs, monitor performance
- **Weekly**: Review backup integrity, update security patches
- **Monthly**: Database optimization, capacity planning review
- **Quarterly**: Full security audit, disaster recovery testing

### Contact Information
- **Technical Support**: support@forbesmarshall.com
- **Emergency Contact**: +91-XXX-XXX-XXXX
- **Documentation**: https://docs.forbesmarshall.com/spotcheck/

---

**Forbes Marshall SpotCheck Production Deployment Guide v2.0.0**  
*Intelligent Parking Management System*  
© 2024 Forbes Marshall. All rights reserved.