from utils.db_connection import get_db_connection
from tabulate import tabulate
import argparse
from utils.slot_compatibility import get_compatibility_rules, get_compatible_slot_types

def show_compatibility_info():
    """Display compatibility information"""
    rules = get_compatibility_rules()
    print("\n🔧 Vehicle to Slot Compatibility Rules:")
    print("=" * 50)
    for vehicle_type, compatible_slots in rules.items():
        print(f"{vehicle_type:>10} → {', '.join(compatible_slots)}")
    print("=" * 50)

def analyze_slot_compatibility(available_slots):
    """Analyze which vehicle types can be accommodated with current available slots"""
    rules = get_compatibility_rules()
    compatibility_analysis = {}
    
    for vehicle_type in rules.keys():
        compatible_slots = []
        for slot in available_slots:
            if slot['slot_type'] in rules[vehicle_type]:
                compatible_slots.append(slot['slot_number'])
        
        compatibility_analysis[vehicle_type] = {
            'can_park': len(compatible_slots) > 0,
            'available_slots': compatible_slots,
            'slot_count': len(compatible_slots)
        }
    
    return compatibility_analysis

def check_parking_status(detailed=False):
    """Check current parking status"""
    conn = get_db_connection()
    if not conn:
        print("❌ Database connection failed")
        return
        
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Get all slots with their status
        cursor.execute("""
            SELECT 
                ps.slot_number,
                ps.slot_type,
                ps.is_occupied,
                ps.last_updated,
                v.plate_number,
                v.owner_name,
                ps2.status,
                ps2.entry_time
            FROM ParkingSlots ps
            LEFT JOIN ParkingSessions ps2 ON ps.slot_id = ps2.slot_id AND ps2.status = 'Active'
            LEFT JOIN Vehicles v ON ps2.vehicle_id = v.vehicle_id
            ORDER BY ps.slot_number
        """)
        
        slots = cursor.fetchall()
        
        # Get available slots for compatibility analysis
        available_slots = [s for s in slots if not s['is_occupied']]
        compatibility_analysis = analyze_slot_compatibility(available_slots)
        
        # Get pending assignments
        cursor.execute("""
            SELECT 
                pa.pending_id,
                v.plate_number,
                v.vehicle_type,
                ps.slot_number,
                pa.expiry_time
            FROM PendingAssignments pa
            JOIN Vehicles v ON pa.vehicle_id = v.vehicle_id
            JOIN ParkingSlots ps ON pa.assigned_slot_id = ps.slot_id
            WHERE pa.expiry_time > NOW()
        """)
        
        pending = cursor.fetchall()
        
        # Get recent fines
        cursor.execute("""
            SELECT 
                f.fine_id,
                ps.slot_number,
                v.plate_number,
                f.expected_slot_type,
                f.fine_amount,
                f.issued_at
            FROM Fines f
            JOIN ParkingSlots ps ON f.actual_slot_id = ps.slot_id
            LEFT JOIN ParkingSessions ps2 ON f.session_id = ps2.session_id
            LEFT JOIN Vehicles v ON ps2.vehicle_id = v.vehicle_id
            ORDER BY f.issued_at DESC
            LIMIT 10
        """)
        
        fines = cursor.fetchall()
        
        # Get statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_slots,
                SUM(is_occupied) as occupied_slots,
                COUNT(*) - SUM(is_occupied) as available_slots
            FROM ParkingSlots
        """)
        
        stats = cursor.fetchone()
        
        # Display results
        print("=== PARKING LOT STATUS ===")
        print(f"Total Slots: {stats['total_slots']} | Occupied: {stats['occupied_slots']} | Available: {stats['available_slots']}")
        
        # Show compatibility information
        show_compatibility_info()
        
        # Show which vehicle types can be accommodated
        print("\n🚗 Vehicle Accommodation Analysis:")
        print("=" * 50)
        for vehicle_type, analysis in compatibility_analysis.items():
            status = "✅ CAN park" if analysis['can_park'] else "❌ CANNOT park"
            slots_info = f" ({analysis['slot_count']} slots)" if analysis['can_park'] else ""
            print(f"{vehicle_type:>10}: {status}{slots_info}")
        
        if detailed:
            print("\n=== SLOT DETAILS ===")
            print(tabulate(slots, headers="keys", tablefmt="grid"))
        else:
            # Summary view
            occupied_slots = [s for s in slots if s['is_occupied']]
            available_slots = [s for s in slots if not s['is_occupied']]
            
            print(f"\nOccupied Slots ({len(occupied_slots)}):")
            for slot in occupied_slots:
                vehicle_info = f"{slot['plate_number']} ({slot['owner_name']})" if slot['plate_number'] else "Unknown"
                print(f"  {slot['slot_number']} ({slot['slot_type']}): {vehicle_info}")
            
            print(f"\nAvailable Slots ({len(available_slots)}):")
            available_by_type = {}
            for slot in available_slots:
                available_by_type[slot['slot_type']] = available_by_type.get(slot['slot_type'], 0) + 1
                print(f"  {slot['slot_number']} ({slot['slot_type']})")
            
            # Show available slots by type
            print(f"\n📊 Available Slots by Type:")
            for slot_type, count in available_by_type.items():
                print(f"  {slot_type}: {count} slots")
        
        print("\n=== PENDING ASSIGNMENTS ===")
        if pending:
            print(tabulate(pending, headers="keys", tablefmt="grid"))
        else:
            print("No pending assignments")
        
        print("\n=== RECENT FINES ===")
        if fines:
            print(tabulate(fines, headers="keys", tablefmt="grid"))
        else:
            print("No fines issued")
            
    except Exception as e:
        print(f"❌ Error retrieving status: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check parking lot status")
    parser.add_argument("--detailed", action="store_true", help="Show detailed slot information")
    args = parser.parse_args()
    
    check_parking_status(detailed=args.detailed)