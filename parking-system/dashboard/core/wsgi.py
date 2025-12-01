"""
WSGI config for Forbes Marshall Parking System.

This module contains the WSGI application used by Django's development server
and any production WSGI deployments. It exposes the WSGI callable as a 
module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/

Company: Forbes Marshall
System: SpotCheck v2.0.0
"""

import os
import sys
import logging
from django.core.wsgi import get_wsgi_application

# Configure logging for WSGI
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [WSGI] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Set the default Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

try:
    # Get the WSGI application
    application = get_wsgi_application()
    logger.info("🚀 Forbes Marshall SpotCheck WSGI application loaded successfully")
    
except Exception as e:
    logger.error(f"❌ Failed to load WSGI application: {e}")
    raise

# Custom WSGI middleware for additional functionality
class ParkingSystemWSGIMiddleware:
    """Custom WSGI middleware for parking system."""
    
    def __init__(self, application):
        self.application = application
    
    def __call__(self, environ, start_response):
        """Process WSGI request."""
        # Add custom headers
        def custom_start_response(status, headers, exc_info=None):
            # Add system identification headers
            headers.extend([
                ('X-Parking-System', 'Forbes Marshall SpotCheck'),
                ('X-System-Version', '2.0.0'),
            ])
            return start_response(status, headers, exc_info)
        
        return self.application(environ, custom_start_response)

# Wrap application with custom middleware
application = ParkingSystemWSGIMiddleware(application)
