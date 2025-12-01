"""
Quick Demo: Maintenance Status Feature
Run this to see the maintenance status in action
"""
import os
import sys
import django

# Setup Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from parking_app.models import ParkingSlot

def show_status_summary():
    """Display current slot status summary"""
    print("=" * 60)
    print("🅿️  PARKING SLOT STATUS SUMMARY")
    print("=" * 60)
    
    total = ParkingSlot.objects.count()
    available = ParkingSlot.objects.filter(status='available').count()
    occupied = ParkingSlot.objects.filter(status='occupied').count()
    maintenance = ParkingSlot.objects.filter(status='maintenance').count()
    out_of_service = ParkingSlot.objects.filter(status='out_of_service').count()
    
    print(f"\n📊 Overall Status:")
    print(f"   Total Slots:        {total}")
    print(f"   🟢 Available:       {available}")
    print(f"   🔴 Occupied:        {occupied}")
    print(f"   🟡 Maintenance:     {maintenance}")
    print(f"   ⚫ Out of Service:  {out_of_service}")
    
    if maintenance > 0:
        print(f"\n🔧 Slots Under Maintenance:")
        for slot in ParkingSlot.objects.filter(status='maintenance'):
            print(f"   - {slot.slot_number} ({slot.get_slot_type_display()})")
    
    return total, available, occupied, maintenance

def demo_maintenance_toggle():
    """Demo: Toggle a slot to maintenance and back"""
    print("\n" + "=" * 60)
    print("🔧 DEMO: Toggle Slot Maintenance Status")
    print("=" * 60)
    
    # Find first available slot
    slot = ParkingSlot.objects.filter(status='available').first()
    
    if not slot:
        print("❌ No available slots to demo with")
        return
    
    original_status = slot.status
    print(f"\n1️⃣  Selected slot: {slot.slot_number} ({slot.get_slot_type_display()})")
    print(f"   Current status: {slot.get_status_display()}")
    
    # Mark as maintenance
    print(f"\n2️⃣  Marking {slot.slot_number} as Under Maintenance...")
    slot.status = 'maintenance'
    slot.save()
    print(f"   ✅ Status changed to: {slot.get_status_display()}")
    print(f"   ℹ️  This slot will NOT be assigned to vehicles")
    
    # Check if it's excluded from available slots
    available_count = ParkingSlot.objects.filter(status='available').count()
    print(f"\n3️⃣  Available slots for assignment: {available_count}")
    print(f"   (Note: {slot.slot_number} is excluded)")
    
    # Return to available
    print(f"\n4️⃣  Returning {slot.slot_number} to service...")
    slot.status = 'available'
    slot.save()
    print(f"   ✅ Status changed to: {slot.get_status_display()}")
    print(f"   ℹ️  This slot can now accept vehicles")
    
    available_count = ParkingSlot.objects.filter(status='available').count()
    print(f"\n5️⃣  Available slots for assignment: {available_count}")
    print(f"   (Note: {slot.slot_number} is now included)")

def demo_bulk_maintenance():
    """Demo: Bulk maintenance operation"""
    print("\n" + "=" * 60)
    print("🔧 DEMO: Bulk Maintenance Operation")
    print("=" * 60)
    
    # Get first 3 available two-wheeler slots
    slots = ParkingSlot.objects.filter(
        slot_type='two_wheeler', 
        status='available'
    )[:3]
    
    if not slots:
        print("❌ No available two-wheeler slots to demo with")
        return
    
    slot_numbers = [s.slot_number for s in slots]
    print(f"\n1️⃣  Selected slots: {', '.join(slot_numbers)}")
    
    # Bulk update to maintenance
    print(f"\n2️⃣  Marking all as Under Maintenance (bulk operation)...")
    count = slots.update(status='maintenance')
    print(f"   ✅ Updated {count} slots")
    
    # Show maintenance count
    maintenance_count = ParkingSlot.objects.filter(status='maintenance').count()
    print(f"\n3️⃣  Total slots under maintenance: {maintenance_count}")
    
    # Bulk return to service
    print(f"\n4️⃣  Returning all to service (bulk operation)...")
    count = ParkingSlot.objects.filter(
        slot_number__in=slot_numbers
    ).update(status='available')
    print(f"   ✅ Updated {count} slots back to available")

def main():
    """Run all demos"""
    print("\n🎬 MAINTENANCE STATUS FEATURE DEMO")
    print("=" * 60)
    
    try:
        # Show current status
        show_status_summary()
        
        # Demo single toggle
        demo_maintenance_toggle()
        
        # Demo bulk operation
        demo_bulk_maintenance()
        
        # Final status
        print("\n" + "=" * 60)
        show_status_summary()
        
        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETE!")
        print("=" * 60)
        print("\n📚 See MAINTENANCE_STATUS_GUIDE.md for full documentation")
        print("🌐 Access admin panel: http://127.0.0.1:8000/admin/")
        print("   Navigate to: Parking App → Parking slots")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
