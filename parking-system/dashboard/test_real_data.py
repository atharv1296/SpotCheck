#!/usr/bin/env python
"""
Test script to check Oracle database connection and real data
Forbes Marshall SpotCheck - Live Data Test
"""
import os
import django
from django.conf import settings

# Set up Oracle environment
os.environ['PATH'] += ';C:\\oracle\\instantclient_19_23'

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from parking_app.models import ParkingSlot, Vehicle, ParkingSession
from driver_applications.models import DriverApplication
from django.utils import timezone

print("🔥 FORBES MARSHALL SPOTCHECK - REAL DATA TEST")
print("=" * 60)

try:
    # Test Oracle Connection
    print("📊 ORACLE DATABASE CONNECTION TEST:")
    
    # Get real parking data
    total_slots = ParkingSlot.objects.count()
    available_slots = ParkingSlot.objects.filter(status='available').count()
    occupied_slots = ParkingSlot.objects.filter(status='occupied').count()
    maintenance_slots = ParkingSlot.objects.filter(status='maintenance').count()
    
    print(f"✅ Total Parking Slots: {total_slots}")
    print(f"🟢 Available Slots: {available_slots}")
    print(f"🔴 Occupied Slots: {occupied_slots}")
    print(f"🟡 Maintenance Slots: {maintenance_slots}")
    
    # Vehicle data
    total_vehicles = Vehicle.objects.count()
    print(f"🚗 Registered Vehicles: {total_vehicles}")
    
    # Session data
    active_sessions = ParkingSession.objects.filter(is_active=True).count()
    total_sessions = ParkingSession.objects.count()
    print(f"🎯 Active Sessions: {active_sessions}")
    print(f"📈 Total Sessions: {total_sessions}")
    
    # Driver applications
    try:
        pending_apps = DriverApplication.objects.filter(status='pending').count()
        total_apps = DriverApplication.objects.count()
        today_apps = DriverApplication.objects.filter(
            created_at__date=timezone.now().date()
        ).count()
        
        print(f"📋 Pending Applications: {pending_apps}")
        print(f"📊 Total Applications: {total_apps}")
        print(f"📅 Today's Applications: {today_apps}")
    except Exception as e:
        print(f"⚠️ Driver Applications: {e}")
    
    # Slot distribution by type
    print("\n🏗️ SLOT DISTRIBUTION BY TYPE:")
    slot_types = ParkingSlot.objects.values_list('slot_type', flat=True).distinct()
    for slot_type in slot_types:
        count = ParkingSlot.objects.filter(slot_type=slot_type).count()
        available = ParkingSlot.objects.filter(slot_type=slot_type, status='available').count()
        occupied = ParkingSlot.objects.filter(slot_type=slot_type, status='occupied').count()
        print(f"  {slot_type.title()}: {count} total ({available} free, {occupied} occupied)")
    
    # Recent vehicles
    print("\n🚗 REGISTERED VEHICLES:")
    vehicles = Vehicle.objects.all()[:5]
    for vehicle in vehicles:
        print(f"  {vehicle.license_plate} - {vehicle.vehicle_type} ({vehicle.owner_name})")
    
    # Calculate occupancy rate
    if total_slots > 0:
        occupancy_rate = round((occupied_slots / total_slots) * 100, 2)
        print(f"\n📊 Current Occupancy Rate: {occupancy_rate}%")
    
    print("\n✅ DATABASE CONNECTION AND DATA RETRIEVAL SUCCESSFUL!")
    print("🔥 Ready to connect dashboard to real Oracle data!")
    
except Exception as e:
    print(f"❌ Error connecting to Oracle database: {e}")
    print("Make sure Oracle Client is properly configured and database is running.")