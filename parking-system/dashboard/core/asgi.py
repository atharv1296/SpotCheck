"""
ASGI config for Forbes Marshall Parking System.

This module contains the ASGI application used for handling both HTTP and 
WebSocket connections. It exposes the ASGI callable as a module-level 
variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/

Company: Forbes Marshall
System: SpotCheck v2.0.0
"""

import os
import sys
import logging
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

# Configure logging for ASGI
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [ASGI] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Set the default Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

# Initialize Django ASGI application early to ensure the AppRegistry is populated
django_asgi_app = get_asgi_application()

try:
    # Try to import WebSocket routing (if channels is configured)
    from parking_app.routing import websocket_urlpatterns
    
    application = ProtocolTypeRouter({
        "http": django_asgi_app,
        "websocket": AuthMiddlewareStack(
            URLRouter(websocket_urlpatterns)
        ),
    })
    
    logger.info("🚀 Forbes Marshall SpotCheck ASGI application with WebSocket support loaded")
    
except ImportError:
    # Fallback to HTTP-only ASGI application
    application = django_asgi_app
    logger.info("🚀 Forbes Marshall SpotCheck ASGI application (HTTP only) loaded")
    
except Exception as e:
    logger.error(f"❌ Failed to load ASGI application: {e}")
    # Fallback to basic ASGI application
    application = django_asgi_app
