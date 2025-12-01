"""
Parking Application Configuration for Forbes Marshall SpotCheck System.
"""

from django.apps import AppConfig
import logging

logger = logging.getLogger('parking_app')


class ParkingAppConfig(AppConfig):
    """Configuration for the parking management application."""
    
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'parking_app'
    verbose_name = 'Forbes Marshall SpotCheck - Parking Management'
    
    def ready(self):
        """
        Perform initialization tasks when the app is ready.
        """
        logger.info("[CAR] Parking App: Initializing Forbes Marshall SpotCheck...")
        
        # Import signal handlers
        try:
            from . import signals
            logger.info("[OK] Parking App: Signal handlers loaded successfully")
        except ImportError:
            logger.info("[INFO] Parking App: No signal handlers found")
        
        # Register admin customizations
        self.setup_admin_customizations()
        
        # Initialize system checks
        self.perform_system_checks()
        
        logger.info("[SUCCESS] Parking App: Forbes Marshall SpotCheck ready!")
    
    def setup_admin_customizations(self):
        """Setup custom admin interface configurations."""
        try:
            from django.contrib import admin
            
            # Customize admin site
            admin.site.site_header = "Forbes Marshall SpotCheck Administration"
            admin.site.site_title = "SpotCheck Admin"
            admin.site.index_title = "Parking System Management"
            
            logger.info("[OK] Admin interface customized")
            
        except Exception as e:
            logger.warning(f"[WARNING] Admin customization failed: {e}")
    
    def perform_system_checks(self):
        """Perform basic system health checks."""
        try:
            # Check if models are properly configured
            from .models import Vehicle, ParkingSlot, ParkingSession
            
            # Verify model configurations
            models_info = [
                (Vehicle, 'Vehicles'),
                (ParkingSlot, 'Parking Slots'), 
                (ParkingSession, 'Parking Sessions')
            ]
            
            for model, name in models_info:
                try:
                    # Basic model check
                    model._meta.get_field('created_at')
                    logger.info(f"[OK] Model {name}: Configuration verified")
                except Exception:
                    # Some models might not have created_at field
                    logger.info(f"[INFO] Model {name}: Basic configuration checked")
            
        except Exception as e:
            logger.error(f"[ERROR] System check failed: {e}")
