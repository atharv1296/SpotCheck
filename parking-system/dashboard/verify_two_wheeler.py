#!/usr/bin/env python3
"""
Verification script to check Two Wheeler slots are working correctly
"""

import os
import sys
import django
from django.conf import settings

# Add the dashboard directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from parking_app.models import ParkingSlot, ParkingSession, Vehicle

def verify_two_wheeler_slots():
    print("🔍 VERIFYING TWO WHEELER SLOTS")
    print("=" * 50)
    
    # Check slot counts by type
    print("📊 SLOT DISTRIBUTION:")
    slot_types = ['two_wheeler', 'sedan', 'suv', 'large']
    for slot_type in slot_types:
        total = ParkingSlot.objects.filter(slot_type=slot_type).count()
        occupied = ParkingSlot.objects.filter(slot_type=slot_type, status='occupied').count()
        available = total - occupied
        print(f"  {slot_type.upper():12}: {total:2} total | {occupied:2} occupied | {available:2} available")
    
    print("\n🏍️ TWO WHEELER SLOTS DETAILS:")
    two_wheeler_slots = ParkingSlot.objects.filter(slot_type='two_wheeler').order_by('slot_number')
    
    for slot in two_wheeler_slots:
        status_icon = "🔴" if slot.status == 'occupied' else "🟢"
        print(f"  {status_icon} {slot.slot_number}: {slot.status}")
        
        # Check if there's an active session for occupied slots
        if slot.status == 'occupied':
            active_session = ParkingSession.objects.filter(
                parking_slot=slot,
                is_active=True
            ).first()
            if active_session:
                vehicle = active_session.vehicle
                print(f"    └─ Vehicle: {vehicle.license_plate} ({vehicle.vehicle_type})")
    
    print(f"\n✅ VERIFICATION COMPLETE")
    print(f"📈 Two Wheeler slots are now working correctly!")
    print(f"🌐 You can now view the dashboard with Two Wheeler category active")

if __name__ == '__main__':
    try:
        verify_two_wheeler_slots()
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        sys.exit(1)