# Forbes Marshall SpotCheck - Parking Management System

[![Forbes Marshall](https://img.shields.io/badge/Company-Forbes%20Marshall-blue.svg)](https://www.forbesmarshall.com)
[![Version](https://img.shields.io/badge/Version-2.0.0-green.svg)](./core/system_status.py)
[![Django](https://img.shields.io/badge/Django-4.2.7-darkgreen.svg)](https://djangoproject.com)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)

## 🏢 About

**Forbes Marshall SpotCheck** is an advanced parking management system designed for modern corporate environments. Built with Django and featuring real-time monitoring, analytics, and AI-powered vehicle detection.

### ✨ Key Features

- 🚗 **Real-time Parking Monitoring** - Live occupancy tracking with instant updates
- 📊 **Advanced Analytics Dashboard** - Comprehensive insights and reporting
- 🤖 **AI Vehicle Detection** - YOLO-based computer vision for automatic detection  
- 📱 **Responsive Web Interface** - Mobile-friendly dashboard design
- 🔄 **WebSocket Support** - Real-time updates without page refresh
- 🏢 **Corporate Integration** - Designed for Forbes Marshall operations
- 📈 **Performance Metrics** - Detailed usage statistics and trends
- 🔐 **Secure Administration** - Role-based access control

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Django 4.2.7
- SQLite (default) or MySQL/PostgreSQL
- Modern web browser

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/atharv1296/SpotCheck.git
   cd SpotCheck/parking-system/dashboard
   ```

2. **Set up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Database Setup**
   ```bash
   python manage.py makemigrations parking_app
   python manage.py migrate
   ```

5. **Initialize Sample Data** (Optional)
   ```bash
   python manage.py init_parking_system --slots 500 --vehicles 50
   ```

6. **Create Superuser**
   ```bash
   python manage.py createsuperuser
   ```

7. **Run Development Server**
   ```bash
   python manage.py runserver
   ```

8. **Access the System**
   - Dashboard: http://localhost:8000/
   - Admin Panel: http://localhost:8000/admin/
   - Real-time Monitor: http://localhost:8000/realtime/
   - Analytics: http://localhost:8000/analytics/

## 🏗️ Project Structure

```
dashboard/
├── core/                   # Django project configuration
│   ├── settings.py        # Main settings with security & performance config
│   ├── urls.py           # URL routing with health checks
│   ├── wsgi.py           # Production WSGI configuration
│   ├── asgi.py           # WebSocket ASGI configuration
│   └── apps.py           # Core app configuration
├── parking_app/           # Main application
│   ├── models.py         # Database models (Vehicle, ParkingSlot, Session)
│   ├── views.py          # Dashboard and API views
│   ├── urls.py           # App-specific URL patterns
│   ├── admin.py          # Django admin configuration
│   ├── consumers.py      # WebSocket consumers for real-time updates
│   ├── serializers.py    # REST API serializers
│   ├── management/       # Custom management commands
│   │   └── commands/
│   │       ├── init_parking_system.py  # Initialize with sample data
│   │       └── system_health.py        # Health check command
│   └── migrations/       # Database migrations
├── static/               # Static files (CSS, JS, Images)
│   ├── css/
│   │   └── dashboard.css # Professional UI styling
│   ├── js/
│   │   └── dashboard.js  # Interactive functionality
│   └── img/              # System images and icons
├── templates/            # HTML templates
│   └── dashboard/        # Dashboard templates
│       ├── base.html     # Base template with navigation
│       ├── index.html    # Main dashboard
│       ├── realtime.html # Real-time monitoring
│       └── analytics.html # Analytics dashboard
└── manage.py            # Django management script
```

## 📊 Dashboard Features

### Main Dashboard
- **Live Statistics** - Total slots, occupancy rate, available spaces
- **Interactive Parking Grid** - Visual slot representation with real-time updates  
- **Vehicle Type Filtering** - Filter by sedan, SUV, hatchback, truck
- **Recent Activity Feed** - Entry/exit events with timestamps
- **Quick Stats Panel** - Key performance indicators

### Real-time Monitoring
- **Live Activity Stream** - Real-time entry/exit notifications
- **Active Sessions** - Current parking sessions with duration
- **Occupancy Trends** - 24-hour occupancy charts
- **Vehicle Distribution** - Pie chart of vehicle types
- **Auto-refresh Toggle** - Configurable refresh intervals

### Analytics Dashboard
- **Historical Data Analysis** - Trends over time periods
- **Peak Usage Patterns** - Identify busy hours and days
<!-- Revenue Tracking removed (Free parking system) -->
- **Utilization Reports** - Efficiency metrics and insights
- **Export Capabilities** - Download reports in various formats

## 🔧 Management Commands

### Initialize System
```bash
python manage.py init_parking_system --slots 500 --vehicles 50 --clear
```
- Creates parking slots and sample vehicles
- `--clear` flag removes existing data
- Configurable slot and vehicle counts

### Health Check
```bash
python manage.py system_health --detailed --export report.txt
```
- Comprehensive system health analysis
- Database connectivity checks
- Performance metrics
- Export detailed reports

### Other Useful Commands
```bash
# Collect static files for production
python manage.py collectstatic

# Create database migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Run tests
python manage.py test
```

## 🌐 API Endpoints

### System Monitoring
- `GET /health/` - System health check
- `GET /api/info/` - API information and documentation

### Parking Data
- `GET /api/parking-data/` - All parking slot information
- `GET /api/parking-status/` - Current occupancy statistics
- `POST /api/update-slot/` - Update individual slot status

### Analytics
- `GET /api/analytics-data/` - Analytics data for charts and reports

## 🔐 Security Features

- **HTTPS Configuration** - SSL/TLS support for production
- **CSRF Protection** - Cross-site request forgery prevention
- **XSS Protection** - Cross-site scripting prevention
- **Secure Headers** - Security headers for production deployment
- **Session Security** - Secure session management
- **Admin Interface Protection** - Role-based access control

## 🚀 Production Deployment

### Environment Setup
1. Set `DEBUG = False` in settings.py
2. Configure `ALLOWED_HOSTS` with your domain
3. Set up proper database (PostgreSQL/MySQL recommended)
4. Configure static file serving (nginx/Apache)
5. Set up SSL certificates

### Docker Deployment (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
```

## 📱 Mobile Support

The dashboard is fully responsive and optimized for:
- 📱 **Mobile Phones** - iOS and Android browsers
- 📟 **Tablets** - iPad and Android tablets  
- 💻 **Desktop** - All modern browsers
- 🖥️ **Large Displays** - 4K and ultrawide monitors

## 🔧 Configuration

### Parking System Settings
Edit `core/settings.py` to configure:

```python
PARKING_SYSTEM_CONFIG = {
    'COMPANY_NAME': 'Forbes Marshall',
    'SYSTEM_NAME': 'SpotCheck', 
    'AUTO_REFRESH_INTERVAL': 30,  # seconds
    'PARKING_RATES': {
        'hourly': 50,
        'daily': 400, 
        'monthly': 8000,
    },
    'FINE_AMOUNTS': {
        'unauthorized_parking': 500,
        'overtime': 100,
    }
}
```

### WebSocket Configuration
For real-time features, install channels:
```bash
pip install channels
```

## 🐛 Troubleshooting

### Common Issues

1. **Server won't start**
   ```bash
   python manage.py check
   python manage.py system_health
   ```

2. **Database errors**
   ```bash
   python manage.py migrate
   python manage.py init_parking_system
   ```

3. **Static files not loading**
   ```bash
   python manage.py collectstatic
   ```

### Debug Mode
Enable detailed error reporting:
```python
DEBUG = True
LOGGING['loggers']['django']['level'] = 'DEBUG'
```

## 📞 Support

- **Company**: Forbes Marshall Limited
- **System**: SpotCheck Parking Management
- **Version**: 2.0.0
- **Documentation**: [Internal Wiki](https://wiki.forbesmarshall.com/spotcheck)
- **Support**: IT-Support@forbesmarshall.com

## 📄 License

Copyright © 2025 Forbes Marshall Limited. All rights reserved.

This software is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.

---

**Forbes Marshall SpotCheck** - Advanced Parking Management for Modern Enterprises
