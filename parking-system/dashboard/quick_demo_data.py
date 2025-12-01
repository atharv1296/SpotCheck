#!/usr/bin/env python
"""
Quick Parking Activity Creator - Add some test data
Forbes Marshall SpotCheck - Live Data Demo
"""
import os
import sys
import django
from django.utils import timezone
from datetime import timedelta
import random

# Add the project path
sys.path.append('c:\\Users\\athar\\OneDrive\\Desktop\\TY - Sem 1\\EDI\\parking-system\\dashboard')

# Set up Oracle environment
os.environ['PATH'] += ';C:\\oracle\\instantclient_19_23'

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from parking_app.models import ParkingSlot, Vehicle, ParkingSession

def add_parking_activity():
    """Add some parking sessions to show live data"""
    
    print("Adding parking activity for live dashboard demo...")
    
    # Get some available slots
    available_slots = list(ParkingSlot.objects.filter(status='available'))
    vehicles = list(Vehicle.objects.all())
    
    if not available_slots or not vehicles:
        print("No available slots or vehicles found!")
        return
    
    # Create 5-8 active parking sessions
    num_sessions = min(8, len(available_slots))
    
    for i in range(num_sessions):
        slot = random.choice(available_slots)
        vehicle = random.choice(vehicles)
        
        # Create entry session (1-6 hours ago)
        hours_ago = random.randint(1, 6)
        entry_time = timezone.now() - timedelta(hours=hours_ago)
        
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
        
        # Remove from available list to avoid duplicates
        available_slots.remove(slot)
        
        print(f"✓ {vehicle.license_plate} parked in {slot.slot_number} ({hours_ago}h ago)")
    
    # Add some historical data (completed sessions)
    for i in range(3):
        slot = random.choice(list(ParkingSlot.objects.all()))
        vehicle = random.choice(vehicles)
        
        # Create completed session from 1-2 days ago
        days_ago = random.randint(1, 2)
        duration_hours = random.randint(2, 8)
        
        exit_time = timezone.now() - timedelta(days=days_ago)
        entry_time = exit_time - timedelta(hours=duration_hours)
        
        session = ParkingSession.objects.create(
            vehicle=vehicle,
            parking_slot=slot,
            entry_time=entry_time,
            exit_time=exit_time,
            is_active=False
        )
        
        print(f"✓ Historical: {vehicle.license_plate} ({days_ago}d ago, {duration_hours}h)")
    
    # Show current status
    total = ParkingSlot.objects.count()
    occupied = ParkingSlot.objects.filter(status='occupied').count()
    available = ParkingSlot.objects.filter(status='available').count()
    
    print(f"\nCurrent Status:")
    print(f"  Total Slots: {total}")
    print(f"  Occupied: {occupied}")
    print(f"  Available: {available}")
    print(f"  Occupancy: {round(occupied/total*100, 1)}%")
    
    print(f"\nDashboard updated with live Oracle data!")
    print(f"View at: http://127.0.0.1:8000/")

if __name__ == '__main__':
    try:
        add_parking_activity()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()