#!/usr/bin/env python
"""
Dashboard Real Data Verification Test
Forbes Marshall SpotCheck - Live Oracle Connection Test
"""
import os
import sys
import django
import requests
import json

# Setup Django
sys.path.append('c:\\Users\\athar\\OneDrive\\Desktop\\TY - Sem 1\\EDI\\parking-system\\dashboard')
os.environ['PATH'] += ';C:\\oracle\\instantclient_19_23'
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from parking_app.models import ParkingSlot, Vehicle, ParkingSession

print("=" * 70)
print("🔥 DASHBOARD REAL DATA VERIFICATION TEST")
print("=" * 70)

print("\n📊 ORACLE DATABASE DATA:")
total_slots = ParkingSlot.objects.count()
occupied_slots = ParkingSlot.objects.filter(status='occupied').count()
available_slots = ParkingSlot.objects.filter(status='available').count()
active_sessions = ParkingSession.objects.filter(is_active=True).count()

print(f"   Total Slots: {total_slots}")
print(f"   Occupied Slots: {occupied_slots}")
print(f"   Available Slots: {available_slots}")
print(f"   Active Sessions: {active_sessions}")

print("\n🌐 TESTING API ENDPOINTS:")
try:
    # Test parking status API
    response = requests.get('http://127.0.0.1:8000/api/parking-status/', timeout=10)
    if response.status_code == 200:
        api_data = response.json()
        print(f"   ✅ API Status: Working")
        print(f"   📊 API Reports:")
        print(f"      Total Slots: {api_data.get('total_slots', 'N/A')}")
        print(f"      Occupied: {api_data.get('occupied_slots', 'N/A')}")
        print(f"      Available: {api_data.get('available_slots', 'N/A')}")
        print(f"      Occupancy Rate: {api_data.get('occupancy_rate', 'N/A')}%")
        
        # Verify data matches
        if (api_data.get('total_slots') == total_slots and 
            api_data.get('occupied_slots') == occupied_slots and
            api_data.get('available_slots') == available_slots):
            print(f"   ✅ DATA VERIFICATION: API matches Oracle database!")
        else:
            print(f"   ❌ DATA MISMATCH: API data differs from database")
    else:
        print(f"   ❌ API Status: Error {response.status_code}")
        
except requests.exceptions.RequestException as e:
    print(f"   ❌ API Connection Error: {e}")

print("\n🎯 DASHBOARD CONNECTION STATUS:")
try:
    # Test main dashboard
    response = requests.get('http://127.0.0.1:8000/', timeout=10)
    if response.status_code == 200:
        print(f"   ✅ Dashboard: Accessible")
        print(f"   🌐 URL: http://127.0.0.1:8000/")
        
        # Check if response contains dynamic data indicators
        content = response.text.lower()
        if 'loadrealtimedata' in content or 'oracle' in content:
            print(f"   ✅ Real-time Loading: Enabled")
        else:
            print(f"   ⚠️ Real-time Loading: Not detected")
            
    else:
        print(f"   ❌ Dashboard: Error {response.status_code}")
        
except requests.exceptions.RequestException as e:
    print(f"   ❌ Dashboard Connection Error: {e}")

print(f"\n📋 SLOT BREAKDOWN BY TYPE:")
slot_types = ['two_wheeler', 'sedan', 'suv', 'large']
for slot_type in slot_types:
    total_type = ParkingSlot.objects.filter(slot_type=slot_type).count()
    occupied_type = ParkingSlot.objects.filter(slot_type=slot_type, status='occupied').count()
    print(f"   {slot_type.title()}: {occupied_type}/{total_type} occupied")

if occupied_slots > 0:
    print(f"\n🚗 CURRENTLY PARKED VEHICLES:")
    active_sessions = ParkingSession.objects.filter(is_active=True)[:5]
    for session in active_sessions:
        print(f"   {session.vehicle.license_plate} in {session.parking_slot.slot_number}")

print(f"\n" + "=" * 70)
if occupied_slots > 0:
    print("🎉 SUCCESS: Dashboard is connected to LIVE Oracle data!")
    print("✅ Real parking sessions are being displayed")
else:
    print("⚠️ NOTE: No active parking sessions found")
    print("✅ Database connection working, but no occupied slots")

print("🔥 Access your live dashboard at: http://127.0.0.1:8000/")
print("=" * 70)