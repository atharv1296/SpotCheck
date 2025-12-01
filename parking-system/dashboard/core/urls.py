"""
URL Configuration for Forbes Marshall Parking System

This module defines the main URL routing for the parking management system.
It includes admin interface, API endpoints, and static file serving.

Company: Forbes Marshall
System: SpotCheck v2.0.0
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.utils import timezone
import json

# Custom admin configuration
admin.site.site_header = "Forbes Marshall SpotCheck Admin"
admin.site.site_title = "SpotCheck Admin Portal"
admin.site.index_title = "Parking System Administration"

def system_health_check(request):
    """System health check endpoint."""
    try:
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        
        return JsonResponse({
            'status': 'healthy',
            'system': 'Forbes Marshall SpotCheck',
            'version': '2.0.0',
            'database': 'connected',
            'timestamp': str(timezone.now())
        })
    except Exception as e:
        return JsonResponse({
            'status': 'unhealthy',
            'error': str(e),
            'system': 'Forbes Marshall SpotCheck',
            'version': '2.0.0'
        }, status=503)

def api_info(request):
    """API information endpoint."""
    return JsonResponse({
        'name': 'Forbes Marshall SpotCheck API',
        'version': '2.0.0',
        'description': 'Advanced Parking Management System API',
        'company': 'Forbes Marshall',
        'endpoints': {
            'health': '/health/',
            'dashboard': '/',
            'realtime': '/realtime/',
            'analytics': '/analytics/',
            'api': {
                'parking_data': '/api/parking-data/',
                'parking_status': '/api/parking-status/',
                'analytics_data': '/api/analytics-data/',
                'update_slot': '/api/update-slot/'
            }
        }
    })

urlpatterns = [
    # Authentication (must be before other routes)
    path('login/', include('parking_app.auth_urls')),
    
    # Admin interface
    path('admin/', admin.site.urls),
    
    # System monitoring endpoints
    path('health/', system_health_check, name='health_check'),
    path('api/info/', api_info, name='api_info'),
    
    # Main application (protected by login)
    path('', include('parking_app.urls')),
    
    # Driver applications for large vehicles (protected by login)
    path('driver/', include('driver_applications.urls')),
    
    # Favicon redirect
    path('favicon.ico', RedirectView.as_view(url='/static/img/favicon.ico', permanent=True)),
]

# Error handlers
handler404 = 'parking_app.views.custom_404'
handler500 = 'parking_app.views.custom_500'
handler403 = 'parking_app.views.custom_403'

# Serve static and media files during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    
    # Debug toolbar (if installed)
    try:
        import debug_toolbar
        urlpatterns = [
            path('__debug__/', include(debug_toolbar.urls)),
        ] + urlpatterns
    except ImportError:
        pass
