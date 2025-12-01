#!/usr/bin/env python
"""
Oracle Database - All Tables Verification Script
Forbes Marshall SpotCheck - Complete Table Structure
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

def show_all_tables():
    from django.db import connection
    
    print("=" * 80)
    print("📊 ORACLE DATABASE - ALL TABLES VERIFICATION")
    print("Forbes Marshall SpotCheck - Complete Database Structure")
    print("=" * 80)
    
    # Test Oracle connection
    try:
        with connection.cursor() as cursor:
            print("🔌 Testing Oracle Connection...")
            cursor.execute("SELECT 'Oracle Connected Successfully' FROM dual")
            result = cursor.fetchone()
            print("✅ Oracle Database Connection: SUCCESS")
            
            # Get database info
            cursor.execute("""
                SELECT 
                    SYS_CONTEXT('USERENV','DB_NAME') as db_name,
                    SYS_CONTEXT('USERENV','CURRENT_USER') as current_user,
                    SYS_CONTEXT('USERENV','SERVER_HOST') as server_host
                FROM dual
            """)
            db_info = cursor.fetchone()
            if db_info:
                print(f"📊 Database: {db_info[0]}")
                print(f"👤 User: {db_info[1]}")
                print(f"🖥️  Host: {db_info[2]}")
            
            print("\n" + "=" * 80)
            print("🗄️  ALL DATABASE TABLES")
            print("=" * 80)
            
            # Get all user tables
            cursor.execute("""
                SELECT table_name, num_rows, tablespace_name
                FROM user_tables 
                ORDER BY table_name
            """)
            
            tables = cursor.fetchall()
            
            if tables:
                print(f"📋 Total Tables Found: {len(tables)}")
                print("-" * 80)
                
                for i, (table_name, num_rows, tablespace) in enumerate(tables, 1):
                    row_count = num_rows if num_rows else "Unknown"
                    print(f"{i:2d}. ✅ {table_name:<35} | Rows: {row_count:<10} | Tablespace: {tablespace}")
                
                print("-" * 80)
                
                # Get specific table details for parking system
                print("\n🅿️  PARKING SYSTEM TABLES DETAILS:")
                print("-" * 80)
                
                parking_tables = [
                    'PARKING_APP_PARKINGSLOT',
                    'PARKING_APP_VEHICLE', 
                    'PARKING_APP_PARKINGSESSION',
                    'DRIVER_APPLICATIONS_DRIVERAPPLICATION'
                ]
                
                for table in parking_tables:
                    cursor.execute(f"""
                        SELECT COUNT(*) FROM {table}
                    """)
                    count = cursor.fetchone()[0]
                    
                    # Get table structure
                    cursor.execute(f"""
                        SELECT column_name, data_type, data_length, nullable
                        FROM user_tab_columns 
                        WHERE table_name = '{table}'
                        ORDER BY column_id
                    """)
                    columns = cursor.fetchall()
                    
                    print(f"\n📋 {table}:")
                    print(f"   Records: {count}")
                    print(f"   Columns: {len(columns)}")
                    
                    for col_name, data_type, data_length, nullable in columns[:5]:  # Show first 5 columns
                        null_info = "NULL" if nullable == 'Y' else "NOT NULL"
                        if data_length:
                            print(f"     • {col_name:<25} {data_type}({data_length}) {null_info}")
                        else:
                            print(f"     • {col_name:<25} {data_type} {null_info}")
                    
                    if len(columns) > 5:
                        print(f"     ... and {len(columns) - 5} more columns")
                
                # Check data in key tables
                print("\n" + "=" * 80)
                print("📊 DATA SUMMARY")
                print("=" * 80)
                
                from parking_app.models import ParkingSlot, Vehicle, ParkingSession
                from driver_applications.models import DriverApplication
                from django.contrib.auth.models import User
                
                # Parking slots by type
                slot_types = [('two_wheeler', 'Two Wheeler'), ('sedan', 'Sedan'), ('suv', 'SUV'), ('large', 'Large Vehicle')]
                print("🅿️  Parking Slots by Category:")
                total_slots = 0
                for slot_key, slot_display in slot_types:
                    count = ParkingSlot.objects.filter(slot_type=slot_key).count()
                    total_slots += count
                    print(f"   {slot_display:<15}: {count:>3} slots")
                
                print(f"   {'Total':<15}: {total_slots:>3} slots")
                
                # Other data
                vehicle_count = Vehicle.objects.count()
                session_count = ParkingSession.objects.count()
                app_count = DriverApplication.objects.count()
                user_count = User.objects.count()
                
                print(f"\n🚗 Vehicles Registered: {vehicle_count}")
                print(f"📋 Parking Sessions: {session_count}")
                print(f"📝 Driver Applications: {app_count}")
                print(f"👤 System Users: {user_count}")
                
                # Occupancy status
                available = ParkingSlot.objects.filter(status='available').count()
                occupied = ParkingSlot.objects.filter(status='occupied').count()
                
                print(f"\n📊 Current Occupancy:")
                print(f"   Available: {available} slots")
                print(f"   Occupied:  {occupied} slots")
                
                if total_slots > 0:
                    occupancy_rate = (occupied / total_slots) * 100
                    print(f"   Rate:      {occupancy_rate:.1f}%")
                
            else:
                print("❌ No tables found in the database")
                
        print("\n" + "=" * 80)
        print("✅ ALL ORACLE TABLES SUCCESSFULLY VERIFIED!")
        print("🌐 System Access:")
        print("   Dashboard: http://127.0.0.1:8000/")
        print("   Gate:      http://127.0.0.1:8000/gate/") 
        print("   Admin:     http://127.0.0.1:8000/admin/")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    show_all_tables()