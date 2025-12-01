"""
Oracle Database Management Script
Forbes Marshall SpotCheck - Database Utilities

This script provides utilities for managing the Oracle database connection and operations.
"""

import os
import sys
import django
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from django.db import connection
from parking_app.models import ParkingSlot, Vehicle, ParkingSession
from driver_applications.models import DriverApplication

def test_oracle_connection():
    """Test Oracle database connection and show database info"""
    print("=" * 60)
    print("🔌 ORACLE DATABASE CONNECTION TEST")
    print("=" * 60)
    
    try:
        with connection.cursor() as cursor:
            # Test basic connection
            cursor.execute("SELECT 'Oracle Connected Successfully' FROM dual")
            result = cursor.fetchone()
            
            if result:
                print("✅ Oracle Connection: SUCCESS")
                
                # Get database information
                cursor.execute("""
                    SELECT 
                        SYS_CONTEXT('USERENV','DB_NAME') as db_name,
                        SYS_CONTEXT('USERENV','CURRENT_USER') as current_user,
                        SYS_CONTEXT('USERENV','SERVER_HOST') as server_host,
                        SYS_CONTEXT('USERENV','INSTANCE_NAME') as instance_name
                    FROM dual
                """)
                
                db_info = cursor.fetchone()
                if db_info:
                    print(f"📊 Database Name: {db_info[0]}")
                    print(f"👤 Current User: {db_info[1]}")
                    print(f"🖥️  Server Host: {db_info[2]}")
                    print(f"⚙️  Instance: {db_info[3]}")
                
                # Check Oracle version
                cursor.execute("SELECT banner FROM v$version WHERE banner LIKE 'Oracle%'")
                version_info = cursor.fetchone()
                if version_info:
                    print(f"🔢 Oracle Version: {version_info[0]}")
                
                return True
                
    except Exception as e:
        print(f"❌ Oracle Connection Failed: {str(e)}")
        print("\n🔧 Troubleshooting Tips:")
        print("   1. Check if Oracle database is running")
        print("   2. Verify connection details in settings.py")
        print("   3. Ensure cx_Oracle is installed: pip install cx-Oracle")
        print("   4. Check Oracle listener status")
        return False

def show_table_status():
    """Show status of all Django tables"""
    print("\n" + "=" * 60)
    print("📊 DATABASE TABLES STATUS")
    print("=" * 60)
    
    try:
        # Check parking slots
        total_slots = ParkingSlot.objects.count()
        available_slots = ParkingSlot.objects.filter(status='available').count()
        occupied_slots = ParkingSlot.objects.filter(status='occupied').count()
        
        print(f"🅿️  Parking Slots:")
        print(f"   Total: {total_slots}")
        print(f"   Available: {available_slots}")
        print(f"   Occupied: {occupied_slots}")
        
        # Show slots by category
        categories = ['Two Wheeler', 'Sedan', 'SUV', 'Large Vehicle']
        for category in categories:
            count = ParkingSlot.objects.filter(slot_type=category).count()
            print(f"   {category}: {count} slots")
        
        # Check vehicles
        total_vehicles = Vehicle.objects.count()
        print(f"\n🚗 Vehicles: {total_vehicles}")
        
        # Check sessions
        active_sessions = ParkingSession.objects.filter(is_active=True).count()
        total_sessions = ParkingSession.objects.count()
        print(f"\n📋 Parking Sessions:")
        print(f"   Active: {active_sessions}")
        print(f"   Total: {total_sessions}")
        
        # Check driver applications
        driver_apps = DriverApplication.objects.count()
        print(f"\n📝 Driver Applications: {driver_apps}")
        
        # Calculate occupancy rate
        if total_slots > 0:
            occupancy_rate = (occupied_slots / total_slots) * 100
            print(f"\n📈 Current Occupancy Rate: {occupancy_rate:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading table data: {str(e)}")
        return False

def show_recent_activity():
    """Show recent parking activity"""
    print("\n" + "=" * 60)
    print("🕒 RECENT ACTIVITY (Last 10 entries)")
    print("=" * 60)
    
    try:
        recent_sessions = ParkingSession.objects.order_by('-entry_time')[:10]
        
        if recent_sessions:
            for i, session in enumerate(recent_sessions, 1):
                status = "🟢 ACTIVE" if session.is_active else "🔴 COMPLETED"
                print(f"{i:2d}. {session.vehicle.license_plate} | {session.slot.slot_number} | {status}")
                print(f"    Entry: {session.entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
                if not session.is_active and session.exit_time:
                    print(f"    Exit:  {session.exit_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print()
        else:
            print("No parking sessions found.")
            
    except Exception as e:
        print(f"❌ Error reading recent activity: {str(e)}")

def main():
    """Main function to run all checks"""
    print("Forbes Marshall SpotCheck - Oracle Database Management")
    print("🏢 Parking Management System")
    
    # Test connection
    if test_oracle_connection():
        # Show table status
        show_table_status()
        
        # Show recent activity
        show_recent_activity()
        
        print("\n" + "=" * 60)
        print("✅ DATABASE STATUS CHECK COMPLETED")
        print("🌐 Access the system at: http://127.0.0.1:8000/gate/")
        print("=" * 60)
    else:
        print("\n❌ Database connection failed. Please check your Oracle setup.")

if __name__ == "__main__":
    main()