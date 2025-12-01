"""
Signal handlers for Forbes Marshall Parking System.

This module contains custom signal handlers for parking-related events
such as vehicle registration, slot occupation, and session management.
"""

import logging
from django.db.models.signals import post_save, pre_delete, post_delete
from django.dispatch import receiver
from django.utils import timezone
from datetime import timedelta

from .models import Vehicle, ParkingSlot, ParkingSession

logger = logging.getLogger('parking_app')


@receiver(post_save, sender=Vehicle)
def vehicle_registered_handler(sender, instance, created, **kwargs):
    """
    Handle new vehicle registration events.
    """
    if created:
        logger.info(f"New vehicle registered: {instance.license_plate} ({instance.vehicle_type})")
        logger.info(f"   Owner: {instance.owner_name}")
        
        # You could add additional logic here like:
        # - Send welcome email to vehicle owner
        # - Update vehicle statistics
        # - Trigger external notifications


@receiver(post_save, sender=ParkingSlot)
def parking_slot_updated_handler(sender, instance, created, **kwargs):
    """
    Handle parking slot status changes.
    """
    if created:
        logger.info(f"New parking slot created: {instance.slot_number} ({instance.slot_type})")
    else:
        # Check if occupancy status changed
        if instance.is_occupied:
            logger.info(f"Slot {instance.slot_number} is now OCCUPIED")
        else:
            logger.info(f"Slot {instance.slot_number} is now AVAILABLE")


@receiver(post_save, sender=ParkingSession)
def parking_session_handler(sender, instance, created, **kwargs):
    """
    Handle parking session events.
    """
    if created:
        logger.info(f"New parking session started: {instance.vehicle.license_plate} in slot {instance.parking_slot.slot_number}")
        
        # Mark slot as occupied
        if not instance.parking_slot.is_occupied:
            instance.parking_slot.is_occupied = True
            instance.parking_slot.save()
            
        # You could add logic here for:
        # - Sending entry confirmation
        # - Starting parking session timer
        # - Updating occupancy statistics
        
    elif not instance.is_active and instance.exit_time:
        # Session ended
        duration = instance.exit_time - instance.entry_time
        logger.info(f"Parking session ended: {instance.vehicle.license_plate} - Duration: {duration}")
        
        # Mark slot as available
        if instance.parking_slot.is_occupied:
            instance.parking_slot.is_occupied = False
            instance.parking_slot.save()


@receiver(pre_delete, sender=ParkingSession)
def parking_session_cleanup_handler(sender, instance, **kwargs):
    """
    Handle parking session deletion cleanup.
    """
    logger.info(f"Parking session being deleted: {instance.vehicle.license_plate} in slot {instance.parking_slot.slot_number}")
    
    # Ensure slot is marked as available if session is deleted
    if instance.parking_slot and instance.parking_slot.is_occupied:
        instance.parking_slot.is_occupied = False
        instance.parking_slot.save()


@receiver(post_delete, sender=Vehicle)
def vehicle_deleted_handler(sender, instance, **kwargs):
    """
    Handle vehicle deletion events.
    """
    logger.info(f"Vehicle deleted: {instance.license_plate}")
    
    # You could add cleanup logic here:
    # - Archive historical data
    # - Send deletion confirmation
    # - Update statistics


# Custom signal handlers for business logic
def handle_slot_occupation(slot_id, vehicle_license_plate):
    """
    Custom handler for slot occupation events.
    """
    try:
        slot = ParkingSlot.objects.get(id=slot_id)
        logger.info(f"🚗➡️  Slot occupation detected: {vehicle_license_plate} -> {slot.slot_number}")
        
        # Update slot status
        slot.is_occupied = True
        slot.last_updated = timezone.now()
        slot.save()
        
        # Log the event for analytics
        logger.info(f"📊 Occupancy analytics: Slot {slot.slot_number} occupied at {timezone.now()}")
        
    except ParkingSlot.DoesNotExist:
        logger.error(f"❌ Slot {slot_id} not found for occupation event")


def handle_slot_vacation(slot_id, duration_minutes=None):
    """
    Custom handler for slot vacation events.
    """
    try:
        slot = ParkingSlot.objects.get(id=slot_id)
        logger.info(f"🚗⬅️  Slot vacation detected: {slot.slot_number}")
        
        # Update slot status
        slot.is_occupied = False
        slot.last_updated = timezone.now()
        slot.save()
        
        # Log duration if provided
        if duration_minutes:
            hours, minutes = divmod(duration_minutes, 60)
            logger.info(f"⏱️  Parking duration: {hours}h {minutes}m for slot {slot.slot_number}")
        
        # Log the event for analytics
        logger.info(f"📊 Occupancy analytics: Slot {slot.slot_number} vacated at {timezone.now()}")
        
    except ParkingSlot.DoesNotExist:
        logger.error(f"❌ Slot {slot_id} not found for vacation event")


def calculate_occupancy_rate():
    """
    Calculate current occupancy rate.
    """
    try:
        total_slots = ParkingSlot.objects.count()
        occupied_slots = ParkingSlot.objects.filter(is_occupied=True).count()
        
        if total_slots > 0:
            occupancy_rate = (occupied_slots / total_slots) * 100
            logger.info(f"📊 Current occupancy: {occupied_slots}/{total_slots} ({occupancy_rate:.1f}%)")
            return occupancy_rate
        else:
            logger.warning("⚠️  No parking slots found for occupancy calculation")
            return 0
            
    except Exception as e:
        logger.error(f"❌ Error calculating occupancy rate: {e}")
        return 0


def generate_session_summary():
    """
    Generate summary of active parking sessions.
    """
    try:
        active_sessions = ParkingSession.objects.filter(is_active=True)
        session_count = active_sessions.count()
        
        logger.info(f"📋 Active sessions summary: {session_count} sessions")
        
        # Categorize by vehicle type
        vehicle_types = {}
        for session in active_sessions:
            v_type = session.vehicle.vehicle_type
            vehicle_types[v_type] = vehicle_types.get(v_type, 0) + 1
        
        for v_type, count in vehicle_types.items():
            logger.info(f"   {v_type.title()}: {count} sessions")
        
        return {
            'total_sessions': session_count,
            'vehicle_types': vehicle_types
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating session summary: {e}")
        return {'total_sessions': 0, 'vehicle_types': {}}


def check_long_parked_vehicles():
    """
    Check for vehicles parked for extended periods.
    """
    try:
        # Find sessions longer than 12 hours
        twelve_hours_ago = timezone.now() - timedelta(hours=12)
        long_sessions = ParkingSession.objects.filter(
            is_active=True,
            entry_time__lt=twelve_hours_ago
        )
        
        if long_sessions.exists():
            logger.warning(f"⚠️  Found {long_sessions.count()} vehicles parked for >12 hours:")
            for session in long_sessions:
                duration = timezone.now() - session.entry_time
                hours = duration.total_seconds() / 3600
                logger.warning(f"   {session.vehicle.license_plate} in {session.parking_slot.slot_number} - {hours:.1f}h")
        else:
            logger.info("✅ No long-parked vehicles found")
        
        return long_sessions.count()
        
    except Exception as e:
        logger.error(f"❌ Error checking long-parked vehicles: {e}")
        return 0


# Utility functions for external triggers
def trigger_entry_event(vehicle_license_plate, slot_number):
    """Trigger entry event for external systems."""
    logger.info(f"🚗🅿️  Entry event triggered: {vehicle_license_plate} -> {slot_number}")


def trigger_exit_event(vehicle_license_plate, slot_number, duration_minutes):
    """Trigger exit event for external systems."""
    hours, minutes = divmod(duration_minutes, 60)
    logger.info(f"🚗🚪 Exit event triggered: {vehicle_license_plate} <- {slot_number} (Duration: {hours}h {minutes}m)")


def trigger_alert(alert_type, message, slot_number=None):
    """Trigger system alerts."""
    slot_info = f" [Slot: {slot_number}]" if slot_number else ""
    logger.warning(f"🚨 ALERT [{alert_type.upper()}]: {message}{slot_info}")


# Analytics helper functions
def get_peak_hours_data():
    """Get peak usage hours data."""
    # This would analyze historical data to find peak hours
    # For now, return a simple message
    logger.info("📈 Peak hours analysis requested - feature coming soon")
    return {"message": "Peak hours analysis feature coming soon"}


def get_revenue_data():
    """Get revenue calculation data."""
    # This would calculate revenue based on parking sessions
    # For now, return a simple message  
    logger.info("💰 Revenue analysis requested - feature coming soon")
    return {"message": "Revenue analysis feature coming soon"}