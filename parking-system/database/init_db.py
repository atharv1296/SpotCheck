import mysql.connector
from mysql.connector import errorcode

def init_database():
    try:
        # Connect to MySQL server (without specifying db first)
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root"
        )
        cursor = conn.cursor()

        # Create Database if not exists
        cursor.execute("CREATE DATABASE IF NOT EXISTS spotcheck")
        print("✅ Database 'spotcheck' created (if not already exists)")

        # Switch to 'spotcheck' database
        cursor.execute("USE spotcheck")

        # Table creation queries
        tables = {}

        # Vehicles table
        tables["Vehicles"] = """
        CREATE TABLE IF NOT EXISTS Vehicles (
            vehicle_id INT AUTO_INCREMENT PRIMARY KEY,
            plate_number VARCHAR(20) UNIQUE NOT NULL,
            owner_name VARCHAR(100),
            vehicle_type ENUM('Two Wheeler','Cars','Large') NOT NULL,
            registered_state VARCHAR(50),
            contact_number VARCHAR(15),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_plate (plate_number),
            INDEX idx_vehicle_type (vehicle_type)
        );
        """

        # Parking Slots table
        tables["ParkingSlots"] = """
        CREATE TABLE IF NOT EXISTS ParkingSlots (
            slot_id INT AUTO_INCREMENT PRIMARY KEY,
            slot_number VARCHAR(10) UNIQUE NOT NULL,
            slot_type ENUM('Two Wheeler','Cars','Large') NOT NULL,
            is_occupied BOOLEAN DEFAULT FALSE,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_slot_number (slot_number),
            INDEX idx_slot_type (slot_type),
            INDEX idx_occupied (is_occupied)
        );
        """

        # Parking Sessions table
        tables["ParkingSessions"] = """
        CREATE TABLE IF NOT EXISTS ParkingSessions (
            session_id INT AUTO_INCREMENT PRIMARY KEY,
            vehicle_id INT,
            slot_id INT,
            entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            exit_time TIMESTAMP NULL,
            status ENUM('Active', 'Completed') DEFAULT 'Active',
            FOREIGN KEY (vehicle_id) REFERENCES Vehicles(vehicle_id) ON DELETE CASCADE,
            FOREIGN KEY (slot_id) REFERENCES ParkingSlots(slot_id) ON DELETE CASCADE,
            INDEX idx_status (status),
            INDEX idx_vehicle (vehicle_id),
            INDEX idx_slot (slot_id)
        );
        """

        # Pending Assignments table
        tables["PendingAssignments"] = """
        CREATE TABLE IF NOT EXISTS PendingAssignments (
            pending_id INT AUTO_INCREMENT PRIMARY KEY,
            vehicle_id INT NOT NULL,
            assigned_slot_id INT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expiry_time TIMESTAMP NOT NULL,
            FOREIGN KEY (vehicle_id) REFERENCES Vehicles(vehicle_id) ON DELETE CASCADE,
            FOREIGN KEY (assigned_slot_id) REFERENCES ParkingSlots(slot_id) ON DELETE CASCADE,
            INDEX idx_expiry (expiry_time),
            INDEX idx_vehicle (vehicle_id),
            INDEX idx_slot (assigned_slot_id)
        );
        """

        # Fines table
        tables["Fines"] = """
        CREATE TABLE IF NOT EXISTS Fines (
            fine_id INT AUTO_INCREMENT PRIMARY KEY,
            session_id INT NULL,
            actual_slot_id INT NOT NULL,
            expected_slot_type ENUM('Two Wheeler','Cars','Large'),
            fine_amount DECIMAL(10,2) DEFAULT 500.00,
            issued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status ENUM('Issued', 'Paid', 'Cancelled') DEFAULT 'Issued',
            FOREIGN KEY (session_id) REFERENCES ParkingSessions(session_id) ON DELETE SET NULL,
            FOREIGN KEY (actual_slot_id) REFERENCES ParkingSlots(slot_id) ON DELETE CASCADE,
            INDEX idx_status (status),
            INDEX idx_issued (issued_at)
        );
        """

        # Run all queries
        for name, query in tables.items():
            try:
                cursor.execute(query)
                print(f"✅ Table {name} created successfully")
            except mysql.connector.Error as err:
                print(f"❌ Failed creating table {name}: {err}")

        conn.commit()
        cursor.close()
        conn.close()
        print("🎉 Database initialization completed successfully!")

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("❌ Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("❌ Database does not exist")
        else:
            print(f"❌ Error: {err}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    init_database()