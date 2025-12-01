#!/usr/bin/env python
"""
Simulate Parking Activity - Create realistic data for dashboard demo
Forbes Marshall SpotCheck - Live Data Simulation
"""
import os
import django
from django.conf import settings
from django.utils import timezone
import random
from datetime import datetime, timedelta

# Set up Oracle environment
os.environ['PATH'] += ';C:\\oracle\\instantclient_19_23'

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from parking_app.models import ParkingSlot, Vehicle, ParkingSession

print("FORBES MARSHALL SPOTCHECK - PARKING ACTIVITY SIMULATOR")
print("=" * 70)

def simulate_parking_activity():
    """Create realistic parking sessions for demo"""
    
    # Get available slots and vehicles
    available_slots = list(ParkingSlot.objects.filter(status='available'))
    vehicles = list(Vehicle.objects.all())
    
    if not vehicles:
        print("❌ No vehicles found! Please run setup_oracle first.")
        return
    
    if not available_slots:
        print("❌ No available slots found!")
        return
    
    # Simulate some parking sessions
    sessions_to_create = min(10, len(available_slots))  # Occupy up to 10 slots
    
    print(f"Simulating {sessions_to_create} parking sessions...")
    
    for i in range(sessions_to_create):
        slot = random.choice(available_slots)
        vehicle = random.choice(vehicles)
        
        # Create entry session (some hours ago)
        hours_ago = random.randint(1, 8)
        entry_time = timezone.now() - timedelta(hours=hours_ago)
        
        # PREVENT DUPLICATE SESSIONS: Deactivate any existing active sessions for this slot
        ParkingSession.objects.filter(parking_slot=slot, is_active=True).update(
            exit_time=timezone.now(),
            status='Completed',
            is_active=False
        )
        
        # Create parking session
        session = ParkingSession.objects.create(
            vehicle=vehicle,
            parking_slot=slot,
            entry_time=entry_time,
            is_active=True
        )
        
        # Update slot status
        slot.status = 'occupied'
        slot.save()
        
        # Remove from available list
        available_slots.remove(slot)
        
        print(f"  OK {vehicle.license_plate} parked in {slot.slot_number} ({hours_ago}h ago)")
    
    # Also create some completed sessions (historical data)
    print(f"\nCreating historical sessions...")
    
    for i in range(5):
        # Pick any slot for historical data
        slot = random.choice(list(ParkingSlot.objects.all()))
        vehicle = random.choice(vehicles)
        
        # Create completed session from yesterday
        days_ago = random.randint(1, 3)
        hours_duration = random.randint(2, 10)
        
        exit_time = timezone.now() - timedelta(days=days_ago)
        entry_time = exit_time - timedelta(hours=hours_duration)
        
        session = ParkingSession.objects.create(
            vehicle=vehicle,
            parking_slot=slot,
            entry_time=entry_time,
            exit_time=exit_time,
            is_active=False
        )
        
        print(f"  Historical: {vehicle.license_plate} in {slot.slot_number} ({days_ago}d ago, {hours_duration}h duration)")

def show_current_status():
    """Display current parking status"""
    total_slots = ParkingSlot.objects.count()
    occupied_slots = ParkingSlot.objects.filter(status='occupied').count()
    available_slots = ParkingSlot.objects.filter(status='available').count()
    active_sessions = ParkingSession.objects.filter(is_active=True).count()
    total_sessions = ParkingSession.objects.count()
    
    print(f"CURRENT PARKING STATUS:")
    print(f"   Total Slots: {total_slots}")
    print(f"   Occupied: {occupied_slots}")
    print(f"   Available: {available_slots}")
    print(f"   Active Sessions: {active_sessions}")
    print(f"   Total Sessions: {total_sessions}")
    
    if total_slots > 0:
        occupancy_rate = round((occupied_slots / total_slots) * 100, 2)
        print(f"   Occupancy Rate: {occupancy_rate}%")
    
    print(f"\nOCCUPANCY BY SLOT TYPE:")
    slot_types = ['two_wheeler', 'sedan', 'suv', 'large']
    for slot_type in slot_types:
        total_type = ParkingSlot.objects.filter(slot_type=slot_type).count()
        occupied_type = ParkingSlot.objects.filter(slot_type=slot_type, status='occupied').count()
        available_type = ParkingSlot.objects.filter(slot_type=slot_type, status='available').count()
        print(f"   {slot_type.title()}: {occupied_type}/{total_type} occupied ({available_type} free)")

if __name__ == '__main__':
    print("Starting parking activity simulation...")
    
    try:
        simulate_parking_activity()
        show_current_status()
        
        print(f"\nSIMULATION COMPLETE!")
        print(f"View live data at: http://127.0.0.1:8000/")
        print(f"Gate interface: http://127.0.0.1:8000/gate/")
        print(f"Admin panel: http://127.0.0.1:8000/admin/")
        print(f"\nYour dashboard is now connected to LIVE ORACLE DATA!")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()