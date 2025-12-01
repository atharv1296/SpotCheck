"""
System-wide signal handlers for Forbes Marshall Parking System.
"""

import logging
from django.db.models.signals import post_migrate, pre_delete
from django.dispatch import receiver
from django.contrib.auth.signals import user_logged_in, user_logged_out
from django.conf import settings

logger = logging.getLogger('parking_app')


@receiver(post_migrate)
def post_migration_handler(sender, **kwargs):
    """
    Handle post-migration tasks.
    """
    if sender.name == 'parking_app':
        logger.info("🔄 Parking app migrations completed")
        
        # Initialize default data if needed
        try:
            from parking_app.models import ParkingSlot
            if not ParkingSlot.objects.exists():
                logger.info("📋 No parking slots found, consider running seed data command")
        except Exception as e:
            logger.warning(f"⚠️  Could not check parking slots: {e}")


@receiver(user_logged_in)
def user_login_handler(sender, user, request, **kwargs):
    """
    Handle user login events.
    """
    logger.info(f"👤 User {user.username} logged in from {request.META.get('REMOTE_ADDR', 'unknown IP')}")


@receiver(user_logged_out)
def user_logout_handler(sender, user, request, **kwargs):
    """
    Handle user logout events.
    """
    if user:
        logger.info(f"👋 User {user.username} logged out")


# Custom signals for parking system
from django.dispatch import Signal

# Parking-specific signals
parking_slot_occupied = Signal()
parking_slot_vacated = Signal()
vehicle_registered = Signal()
parking_session_started = Signal()
parking_session_ended = Signal()


@receiver(parking_slot_occupied)
def handle_slot_occupied(sender, slot_id, vehicle_info, **kwargs):
    """
    Handle parking slot occupation events.
    """
    logger.info(f"🚗 Slot {slot_id} occupied by vehicle {vehicle_info.get('license_plate', 'Unknown')}")


@receiver(parking_slot_vacated)
def handle_slot_vacated(sender, slot_id, duration, **kwargs):
    """
    Handle parking slot vacation events.
    """
    logger.info(f"🔄 Slot {slot_id} vacated after {duration}")


@receiver(vehicle_registered)
def handle_vehicle_registered(sender, vehicle_data, **kwargs):
    """
    Handle new vehicle registration events.
    """
    logger.info(f"📝 New vehicle registered: {vehicle_data.get('license_plate', 'Unknown')}")


@receiver(parking_session_started)
def handle_session_started(sender, session_id, **kwargs):
    """
    Handle parking session start events.
    """
    logger.info(f"▶️  Parking session {session_id} started")


@receiver(parking_session_ended)
def handle_session_ended(sender, session_id, duration, **kwargs):
    """
    Handle parking session end events.
    """
    logger.info(f"⏹️  Parking session {session_id} ended after {duration}")


# Utility functions for triggering custom signals
def trigger_slot_occupied(slot_id, vehicle_info):
    """Trigger slot occupied signal."""
    parking_slot_occupied.send(sender=None, slot_id=slot_id, vehicle_info=vehicle_info)


def trigger_slot_vacated(slot_id, duration):
    """Trigger slot vacated signal."""
    parking_slot_vacated.send(sender=None, slot_id=slot_id, duration=duration)


def trigger_vehicle_registered(vehicle_data):
    """Trigger vehicle registered signal."""
    vehicle_registered.send(sender=None, vehicle_data=vehicle_data)


def trigger_session_started(session_id):
    """Trigger session started signal."""
    parking_session_started.send(sender=None, session_id=session_id)


def trigger_session_ended(session_id, duration):
    """Trigger session ended signal."""
    parking_session_ended.send(sender=None, session_id=session_id, duration=duration)