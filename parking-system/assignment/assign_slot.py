import mysql.connector
from datetime import datetime, timedelta
from utils.db_connection import get_db_connection
from utils.slot_compatibility import (
    find_best_available_slot, 
    suggest_alternative_slots,
    get_compatibility_rules
)
import re

def validate_plate_number(plate_number):
    """Validate Indian vehicle plate number format"""
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{1,4}$'
    return re.match(pattern, plate_number.upper()) is not None

def validate_vehicle_type(vehicle_type):
    """Validate vehicle type"""
    valid_types = ['Two Wheeler', 'Cars', 'Large']
    return vehicle_type in valid_types

def register_vehicle(plate_number, vehicle_type, owner_name, contact_number=None, registered_state=None):
    """Register a vehicle and assign a parking slot with compatibility checking"""
    
    # Input validation
    if not validate_plate_number(plate_number):
        print("❌ Invalid plate number format. Please use Indian format (e.g., MH12AB1234)")
        return None
    
    if not validate_vehicle_type(vehicle_type):
        print("❌ Invalid vehicle type. Must be one of: Two Wheeler, Cars, Large")
        return None
    
    if not owner_name or len(owner_name.strip()) < 2:
        print("❌ Owner name is required and must be at least 2 characters long")
        return None

    conn = get_db_connection()
    if not conn:
        print("❌ Database connection failed")
        return None
        
    cursor = conn.cursor(dictionary=True)

    try:
        # 1. Insert or get the vehicle
        cursor.execute("""
            INSERT INTO Vehicles (plate_number, vehicle_type, owner_name, contact_number, registered_state)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                owner_name = VALUES(owner_name),
                contact_number = VALUES(contact_number),
                registered_state = VALUES(registered_state),
                vehicle_id = LAST_INSERT_ID(vehicle_id)
        """, (plate_number.upper(), vehicle_type, owner_name, contact_number, registered_state))
        
        vehicle_id = cursor.lastrowid
        if vehicle_id == 0:
            cursor.execute("SELECT vehicle_id FROM Vehicles WHERE plate_number = %s", (plate_number.upper(),))
            result = cursor.fetchone()
            vehicle_id = result['vehicle_id'] if result else None

        if not vehicle_id:
            print("❌ Error: Failed to get vehicle ID")
            conn.rollback()
            return None

        # 2. Check if vehicle already has an active session
        cursor.execute("""
            SELECT ps.session_id, psl.slot_id, psl.slot_number, ps.entry_time
            FROM ParkingSessions ps
            JOIN ParkingSlots psl ON ps.slot_id = psl.slot_id
            WHERE ps.vehicle_id = %s AND ps.status = 'Active'
        """, (vehicle_id,))
        active_session = cursor.fetchone()
        
        if active_session:
            print(f"ℹ️ Vehicle {plate_number} already has an active session in slot {active_session['slot_number']} since {active_session['entry_time']}")
            conn.close()
            return active_session['slot_number']

        # 3. Check if vehicle already has a pending assignment
        cursor.execute("""
            SELECT pa.pending_id, pa.expiry_time, psl.slot_number
            FROM PendingAssignments pa
            JOIN ParkingSlots psl ON pa.assigned_slot_id = psl.slot_id
            WHERE pa.vehicle_id = %s AND pa.expiry_time > NOW()
        """, (vehicle_id,))
        pending_assignment = cursor.fetchone()
        
        if pending_assignment:
            print(f"ℹ️ Vehicle {plate_number} already has a pending assignment for slot {pending_assignment['slot_number']} (expires at {pending_assignment['expiry_time']})")
            conn.close()
            return pending_assignment['slot_number']

        # 4. Find ALL available slots (not just matching type)
        cursor.execute("""
            SELECT slot_id, slot_number, slot_type 
            FROM ParkingSlots 
            WHERE is_occupied = FALSE 
            ORDER BY slot_number
        """)
        all_available_slots = cursor.fetchall()

        if not all_available_slots:
            print("❌ Parking lot is completely full!")
            conn.close()
            return None

        # 5. Use compatibility algorithm to find the best slot
        slot_id, slot_number = find_best_available_slot(vehicle_type, all_available_slots)

        if not slot_id:
            # No compatible slot found, show alternatives
            print("❌ No compatible slots available for your vehicle type.")
            
            # Show compatibility rules
            print("\n📋 Compatibility Rules:")
            rules = get_compatibility_rules()
            for v_type, compatible_slots in rules.items():
                print(f"  {v_type}: Can park in {', '.join(compatible_slots)}")
            
            # Show what's actually available
            print(f"\n📊 Currently Available:")
            available_by_type = {}
            for slot in all_available_slots:
                available_by_type[slot['slot_type']] = available_by_type.get(slot['slot_type'], 0) + 1
            
            for slot_type, count in available_by_type.items():
                print(f"  {slot_type}: {count} slots available")
            
            conn.close()
            return None

        # 6. Get the slot type for verification
        cursor.execute("SELECT slot_type FROM ParkingSlots WHERE slot_id = %s", (slot_id,))
        assigned_slot = cursor.fetchone()
        slot_type = assigned_slot['slot_type']

        # 7. Mark the slot as occupied IMMEDIATELY
        cursor.execute("""
            UPDATE ParkingSlots SET is_occupied = TRUE WHERE slot_id = %s
        """, (slot_id,))

        # 8. Add to PendingAssignments
        expiry_time = datetime.now() + timedelta(minutes=10)  # 10 minutes to park
        cursor.execute("""
            INSERT INTO PendingAssignments (vehicle_id, assigned_slot_id, expiry_time)
            VALUES (%s, %s, %s)
        """, (vehicle_id, slot_id, expiry_time))

        conn.commit()
        
        # Show assignment details with compatibility info
        print(f"✅ Vehicle {plate_number} ({vehicle_type}) assigned to slot {slot_number} ({slot_type})")
        print(f"⏰ Please park within 10 minutes (by {expiry_time.strftime('%H:%M:%S')})")
        
        return slot_number

    except mysql.connector.Error as err:
        print(f"❌ Database error: {err}")
        conn.rollback()
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()

def show_compatibility_rules():
    """Display the vehicle-to-slot compatibility rules"""
    rules = get_compatibility_rules()
    print("\n🚗 Vehicle to Slot Compatibility Rules:")
    print("=" * 50)
    for vehicle_type, compatible_slots in rules.items():
        print(f"{vehicle_type:>10} → {', '.join(compatible_slots)}")
    print("=" * 50)

def interactive_register():
    """Interactive function to register a vehicle"""
    print("\n=== Vehicle Registration ===")
    
    # Show compatibility rules first
    show_compatibility_rules()
    print()
    
    plate_number = input("Enter vehicle plate number: ").strip()
    vehicle_type = input("Enter vehicle type (Two Wheeler/Cars/Large): ").strip()
    owner_name = input("Enter owner name: ").strip()
    contact_number = input("Enter contact number (optional): ").strip() or None
    registered_state = input("Enter registered state (optional): ").strip() or None
    
    result = register_vehicle(plate_number, vehicle_type, owner_name, contact_number, registered_state)
    
    if not result:
        print("Registration failed. Please try again.")
    
    return result

if __name__ == "__main__":
    # Example usage
    result = interactive_register()