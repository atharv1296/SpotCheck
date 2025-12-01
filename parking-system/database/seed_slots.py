import mysql.connector
from utils.db_connection import get_db_connection

def seed_slots():
    """Seed database with sample parking slots"""
    conn = get_db_connection()
    if not conn:
        print("❌ Database connection failed")
        return
        
    cursor = conn.cursor()
    
    try:
        # Disable foreign key checks temporarily
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        
        # Clear existing data in correct order to respect foreign key constraints
        cursor.execute("DELETE FROM Fines")
        cursor.execute("DELETE FROM ParkingSessions")
        cursor.execute("DELETE FROM PendingAssignments")
        cursor.execute("DELETE FROM ParkingSlots")
        cursor.execute("DELETE FROM Vehicles")
        
        print("Cleared existing data")
        
        # Define 20 slots (5 per category)
        slots = {
                        \"Two Wheeler\": [\"H1\", \"H2\", \"H3\", \"H4\", \"H5\"],
            "Cars": ["S1", "S2", "S3", "S4", "S5"],

            "Large": ["L1", "L2", "L3", "L4", "L5"]
        }
        
        # Insert slots into ParkingSlots table
        for slot_type, slot_list in slots.items():
            for slot_number in slot_list:
                cursor.execute(
                    "INSERT INTO ParkingSlots (slot_number, slot_type) VALUES (%s, %s)",
                    (slot_number, slot_type)
                )
                print(f"✅ Inserted slot {slot_number} ({slot_type})")
        
        # Re-enable foreign key checks
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        
        conn.commit()
        print("🎉 All 20 slots seeded successfully!")
        
    except mysql.connector.Error as err:
        print(f"❌ Database error: {err}")
        conn.rollback()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    seed_slots()