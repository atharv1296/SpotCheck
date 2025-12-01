#!/usr/bin/env python
"""
Quick Database Status Check
Forbes Marshall SpotCheck - Oracle Database Verification
"""

import os
import sys
import django
from pathlib import Path

# Setup Django
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

def check_database_status():
    from parking_app.models import ParkingSlot, Vehicle, ParkingSession
    from driver_applications.models import DriverApplication
    from django.db import connection
    
    print("=" * 60)
    print("📊 FORBES MARSHALL SPOTCHECK - DATABASE STATUS")
    print("=" * 60)
    
    # Test database connection
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 'Oracle Connected' FROM dual")
            result = cursor.fetchone()
            print("✅ Oracle Database Connection: SUCCESS")
            
            # Get database info
            cursor.execute("""
                SELECT 
                    SYS_CONTEXT('USERENV','DB_NAME') as db_name,
                    SYS_CONTEXT('USERENV','CURRENT_USER') as current_user
                FROM dual
            """)
            db_info = cursor.fetchone()
            if db_info:
                print(f"📊 Database: {db_info[0]}")
                print(f"👤 User: {db_info[1]}")
    except Exception as e:
        print(f"❌ Database Connection Error: {e}")
        return
    
    print("\n" + "=" * 60)
    print("📋 TABLE CONTENTS")
    print("=" * 60)
    
    # Check table contents
    try:
        # Parking Slots
        total_slots = ParkingSlot.objects.count()
        print(f"🅿️  Total Parking Slots: {total_slots}")
        
        # Breakdown by type
        slot_types = [('two_wheeler', 'Two Wheeler'), ('sedan', 'Sedan'), ('suv', 'SUV'), ('large', 'Large Vehicle')]
        for slot_key, slot_display in slot_types:
            count = ParkingSlot.objects.filter(slot_type=slot_key).count()
            print(f"   {slot_display}: {count} slots")
        
        # Vehicles
        total_vehicles = Vehicle.objects.count()
        print(f"\n🚗 Total Vehicles: {total_vehicles}")
        
        # Sessions  
        total_sessions = ParkingSession.objects.count()
        active_sessions = ParkingSession.objects.filter(is_active=True).count() if hasattr(ParkingSession, 'is_active') else 0
        print(f"\n📋 Parking Sessions:")
        print(f"   Total: {total_sessions}")
        print(f"   Active: {active_sessions}")
        
        # Applications
        total_applications = DriverApplication.objects.count()
        print(f"\n📝 Driver Applications: {total_applications}")
        
        # Occupancy status
        available = ParkingSlot.objects.filter(status='available').count()
        occupied = ParkingSlot.objects.filter(status='occupied').count()
        
        print(f"\n📊 OCCUPANCY STATUS:")
        print(f"   Available: {available}")
        print(f"   Occupied: {occupied}")
        
        if total_slots > 0:
            occupancy_rate = (occupied / total_slots) * 100
            print(f"   Occupancy Rate: {occupancy_rate:.1f}%")
        
        print("\n" + "=" * 60)
        print("✅ ALL TABLES SUCCESSFULLY CONNECTED TO ORACLE DATABASE!")
        print("🌐 Dashboard URL: http://127.0.0.1:8000/")
        print("🚪 Gate Interface: http://127.0.0.1:8000/gate/")
        print("⚙️  Admin Panel: http://127.0.0.1:8000/admin/")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error reading table data: {e}")

if __name__ == "__main__":
    check_database_status()