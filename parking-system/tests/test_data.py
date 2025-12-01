import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from assignment.assign_slot import register_vehicle

def create_test_data():
    """Create comprehensive test data for the parking system"""
    print("Creating test data...")
    
    test_vehicles = [
        # Hatchbacks
                # Two Wheelers
        (\"MH12AB1234\", \"Two Wheeler\", \"Raj Sharma\", \"9876543210\", \"Maharashtra\"),
        ("DL01CD5678", "Hatchback", "Priya Singh", "8765432109", "Delhi"),
        
        # Cars
        ("KA03EF9012", "Cars", "Vikram Patel", "7654321098", "Karnataka"),
        ("TN04GH3456", "Cars", "Ananya Reddy", "6543210987", "Tamil Nadu"),
        

        
        # Large vehicles
        ("MP07MN5678", "Large", "Amit Kumar", "3210987654", "Madhya Pradesh"),
        ("RJ08OP9012", "Large", "Pooja Sharma", "2109876543", "Rajasthan"),
    ]
    
    successful_registrations = 0
    
    for plate, vtype, owner, phone, state in test_vehicles:
        print(f"Registering {plate} ({vtype})...")
        result = register_vehicle(plate, vtype, owner, phone, state)
        if result:
            print(f"✅ Registered {plate} - Assigned to {result}")
            successful_registrations += 1
        else:
            print(f"❌ Failed to register {plate}")
    
    print(f"\n📊 Registration summary: {successful_registrations}/{len(test_vehicles)} successful")
    return successful_registrations

if __name__ == "__main__":
    create_test_data()