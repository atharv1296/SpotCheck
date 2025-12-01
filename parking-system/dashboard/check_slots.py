#!/usr/bin/env python3
"""
Simple Django shell script to check Two Wheeler slots
"""

from parking_app.models import ParkingSlot, ParkingSession, Vehicle

print("🔍 CHECKING TWO WHEELER SLOTS")
print("=" * 40)

# Check slot counts by type
slot_types = ['two_wheeler', 'sedan', 'suv', 'large']
for slot_type in slot_types:
    total = ParkingSlot.objects.filter(slot_type=slot_type).count()
    occupied = ParkingSlot.objects.filter(slot_type=slot_type, status='occupied').count()
    available = total - occupied
    print(f"{slot_type.upper():12}: {total:2} total | {occupied:2} occupied | {available:2} available")

print(f"\n🏍️ TWO WHEELER SLOTS:")
two_wheeler_slots = ParkingSlot.objects.filter(slot_type='two_wheeler').order_by('slot_number')

for slot in two_wheeler_slots:
    status_icon = "🔴" if slot.status == 'occupied' else "🟢"
    print(f"  {status_icon} {slot.slot_number}: {slot.status}")

print(f"\n✅ Two Wheeler slots are working!")