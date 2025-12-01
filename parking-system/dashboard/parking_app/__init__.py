"""
Forbes Marshall SpotCheck - Parking Management Application

This package contains the main parking management functionality including:
- Vehicle registration and management
- Parking slot monitoring and control  
- Real-time session tracking
- Analytics and reporting
- REST API endpoints
- WebSocket consumers for live updates

Company: Forbes Marshall Limited
System: SpotCheck v2.0.0
Module: Parking Application Core
"""

default_app_config = 'parking_app.apps.ParkingAppConfig'

# Version information
__version__ = '2.0.0'
__author__ = 'Forbes Marshall IT Team'
__email__ = 'it-support@forbesmarshall.com'

# Application metadata
APP_META = {
    'name': 'Forbes Marshall SpotCheck',
    'version': __version__,
    'description': 'Advanced Parking Management System',
    'company': 'Forbes Marshall Limited',
    'features': [
        'Real-time monitoring',
        'Vehicle detection', 
        'Analytics dashboard',
        'Mobile responsive',
        'WebSocket support',
        'REST API'
    ]
}