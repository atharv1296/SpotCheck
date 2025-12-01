"""
Core application configuration for Forbes Marshall Parking System.
"""

from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)


class CoreConfig(AppConfig):
    """Core application configuration."""
    
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'
    verbose_name = 'Forbes Marshall Parking System Core'
    
    def ready(self):
        """
        Perform initialization tasks once Django has loaded all apps.
        """
        logger.info("🚀 Forbes Marshall Parking System - Core module initialized")
        
        # Import signal handlers
        try:
            from . import signals
            logger.info("✅ Core signal handlers loaded successfully")
        except ImportError:
            logger.info("ℹ️  No core signal handlers found")
        
        # Verify system configuration
        self.verify_system_config()
        
        logger.info("🏢 Forbes Marshall SpotCheck v2.0.0 - System Ready")
    
    def verify_system_config(self):
        """Verify system configuration on startup."""
        from django.conf import settings
        
        try:
            # Check parking system configuration
            config = getattr(settings, 'PARKING_SYSTEM_CONFIG', {})
            
            required_keys = ['COMPANY_NAME', 'SYSTEM_NAME', 'VERSION']
            missing_keys = [key for key in required_keys if key not in config]
            
            if missing_keys:
                logger.warning(f"⚠️  Missing parking system config keys: {missing_keys}")
            else:
                logger.info(f"✅ System: {config.get('COMPANY_NAME')} {config.get('SYSTEM_NAME')} v{config.get('VERSION')}")
            
            # Check database connection
            from django.db import connection
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                logger.info("✅ Database connection verified")
                
        except Exception as e:
            logger.error(f"❌ System configuration verification failed: {e}")