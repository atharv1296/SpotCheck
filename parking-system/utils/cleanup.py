from utils.db_connection import get_db_connection
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_expired_assignments():
    """Background thread to clean up expired pending assignments"""
    while True:
        try:
            conn = get_db_connection()
            if not conn:
                time.sleep(60)
                continue
                
            cursor = conn.cursor(dictionary=True)
            
            # Check for expired assignments first
            cursor.execute("""
                SELECT pa.pending_id, pa.vehicle_id, v.plate_number, 
                       pa.assigned_slot_id, ps.slot_number, pa.expiry_time
                FROM PendingAssignments pa
                JOIN Vehicles v ON pa.vehicle_id = v.vehicle_id
                JOIN ParkingSlots ps ON pa.assigned_slot_id = ps.slot_id
                WHERE pa.expiry_time < NOW()
            """)
            
            expired_assignments = cursor.fetchall()
            
            if expired_assignments:
                logger.info(f"Found {len(expired_assignments)} expired assignments to clean up")
                
                for assignment in expired_assignments:
                    logger.info(f"Expired: {assignment['plate_number']} in {assignment['slot_number']} (expired at {assignment['expiry_time']})")
                    
                    # Delete the expired assignment
                    cursor.execute("DELETE FROM PendingAssignments WHERE pending_id = %s", 
                                  (assignment['pending_id'],))
                    
                    # Free up the slot
                    cursor.execute("""
                        UPDATE ParkingSlots 
                        SET is_occupied = FALSE 
                        WHERE slot_id = %s
                    """, (assignment['assigned_slot_id'],))
                    
                    logger.info(f"Cleaned up expired assignment for {assignment['plate_number']}")
            
            conn.commit()
            cursor.close()
            conn.close()
            
            # Run cleanup every minute
            time.sleep(60)
            
        except Exception as e:
            logger.error(f"Error in cleanup thread: {e}")
            time.sleep(60)
            
if __name__ == "__main__":
    cleanup_expired_assignments()