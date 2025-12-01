"""
Django Management Command: Setup Oracle Database
Forbes Marshall SpotCheck - Oracle Database Setup

This command sets up the Oracle database with all required tables and initial data.
"""

from django.core.management.base import BaseCommand
from django.db import connection
from django.utils import timezone
from parking_app.models import ParkingSlot, Vehicle, ParkingSession
from driver_applications.models import DriverApplication
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Setup Oracle database with all required tables and initial data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--reset',
            action='store_true',
            help='Reset all data (WARNING: This will delete all existing data)',
        )
        parser.add_argument(
            '--add-slots',
            type=int,
            default=15,
            help='Number of slots to create per category (default: 15)',
        )

    def handle(self, *args, **options):
        self.stdout.write("=" * 70)
        self.stdout.write(self.style.SUCCESS("📊 ORACLE DATABASE SETUP - FORBES MARSHALL SPOTCHECK"))
        self.stdout.write("=" * 70)
        
        try:
            # Test Oracle connection
            self.test_connection()
            
            # Create/migrate tables
            self.create_tables()
            
            # Reset data if requested
            if options['reset']:
                self.reset_data()
            
            # Create initial parking slots
            self.create_parking_slots(options['add_slots'])
            
            # Create sample vehicles (optional)
            self.create_sample_vehicles()
            
            # Show final status
            self.show_database_status()
            
            self.stdout.write("\n" + "=" * 70)
            self.stdout.write(self.style.SUCCESS("✅ ORACLE DATABASE SETUP COMPLETED SUCCESSFULLY!"))
            self.stdout.write("🌐 Access the system at: http://127.0.0.1:8000/gate/")
            self.stdout.write("=" * 70)
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Setup failed: {str(e)}"))
            raise

    def test_connection(self):
        """Test Oracle database connection"""
        self.stdout.write("🔌 Testing Oracle database connection...")
        
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 'Oracle Connected' FROM dual")
                result = cursor.fetchone()
                if result:
                    self.stdout.write(self.style.SUCCESS("✅ Oracle connection successful"))
                    
                    # Show database info
                    cursor.execute("""
                        SELECT 
                            SYS_CONTEXT('USERENV','DB_NAME') as db_name,
                            SYS_CONTEXT('USERENV','CURRENT_USER') as current_user,
                            SYS_CONTEXT('USERENV','SERVER_HOST') as server_host
                        FROM dual
                    """)
                    db_info = cursor.fetchone()
                    if db_info:
                        self.stdout.write(f"   📊 Database: {db_info[0]}")
                        self.stdout.write(f"   👤 User: {db_info[1]}")
                        self.stdout.write(f"   🖥️  Host: {db_info[2]}")
                        
        except Exception as e:
            raise Exception(f"Oracle connection failed: {str(e)}")

    def create_tables(self):
        """Create all required tables using Django migrations"""
        self.stdout.write("\n🗄️  Creating database tables...")
        
        try:
            # Import Django's migration executor
            from django.core.management import call_command
            
            # Run migrations
            call_command('migrate', verbosity=1, interactive=False)
            self.stdout.write(self.style.SUCCESS("✅ All tables created successfully"))
            
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"⚠️ Migration warning: {str(e)}"))
            # Continue anyway as tables might already exist

    def reset_data(self):
        """Reset all data in the database"""
        self.stdout.write("\n🗑️  Resetting all data...")
        
        try:
            # Delete in order to respect foreign keys
            ParkingSession.objects.all().delete()
            DriverApplication.objects.all().delete()
            Vehicle.objects.all().delete()
            ParkingSlot.objects.all().delete()
            
            self.stdout.write(self.style.SUCCESS("✅ All data cleared"))
            
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"⚠️ Reset warning: {str(e)}"))

    def create_parking_slots(self, slots_per_category):
        """Create parking slots for all categories"""
        self.stdout.write(f"\n🅿️  Creating {slots_per_category} parking slots per category...")
        
        # Define slot categories and their prefixes
        slot_categories = {
            'two_wheeler': 'H',
            'sedan': 'S', 
            'suv': 'U',
            'large': 'L'
        }
        
        total_created = 0
        
        for category, prefix in slot_categories.items():
            category_count = 0
            
            for i in range(1, slots_per_category + 1):
                slot_number = f"{prefix}{i}"
                
                # Check if slot already exists
                if not ParkingSlot.objects.filter(slot_number=slot_number).exists():
                    ParkingSlot.objects.create(
                        slot_number=slot_number,
                        slot_type=category,  # Already lowercase now
                        status='available'
                    )
                    category_count += 1
                    total_created += 1
            
            self.stdout.write(f"   ✅ {category}: {category_count} slots created")
        
        self.stdout.write(self.style.SUCCESS(f"✅ Total {total_created} new parking slots created"))

    def create_sample_vehicles(self):
        """Create some sample vehicles for testing"""
        self.stdout.write("\n🚗 Creating sample vehicles...")
        
        sample_vehicles = [
            {'license_plate': 'MH12AB1234', 'vehicle_type': 'Two Wheeler', 'owner_name': 'John Doe'},
            {'license_plate': 'MH14CD5678', 'vehicle_type': 'Sedan', 'owner_name': 'Jane Smith'},
            {'license_plate': 'MH01EF9012', 'vehicle_type': 'SUV', 'owner_name': 'Bob Johnson'},
            {'license_plate': 'MH09GH3456', 'vehicle_type': 'Large Vehicle', 'owner_name': 'Alice Brown'},
        ]
        
        created_count = 0
        for vehicle_data in sample_vehicles:
            if not Vehicle.objects.filter(license_plate=vehicle_data['license_plate']).exists():
                Vehicle.objects.create(**vehicle_data)
                created_count += 1
        
        self.stdout.write(self.style.SUCCESS(f"✅ {created_count} sample vehicles created"))

    def show_database_status(self):
        """Show current database status and statistics"""
        self.stdout.write("\n📊 DATABASE STATUS:")
        self.stdout.write("-" * 50)
        
        try:
            # Parking slots by category
            slot_categories = [('hatchback', 'Hatchback'), ('sedan', 'Sedan'), ('suv', 'SUV'), ('large', 'Large Vehicle')]
            for category_key, category_display in slot_categories:
                count = ParkingSlot.objects.filter(slot_type=category_key).count()
                self.stdout.write(f"   🅿️  {category_display} slots: {count}")
            
            # Total counts
            total_slots = ParkingSlot.objects.count()
            total_vehicles = Vehicle.objects.count()
            active_sessions = ParkingSession.objects.filter(is_active=True).count()
            
            self.stdout.write("-" * 50)
            self.stdout.write(f"   📊 Total parking slots: {total_slots}")
            self.stdout.write(f"   🚗 Total vehicles: {total_vehicles}")
            self.stdout.write(f"   🟢 Active sessions: {active_sessions}")
            
            # Available vs occupied
            available = ParkingSlot.objects.filter(status='available').count()
            occupied = ParkingSlot.objects.filter(status='occupied').count()
            
            self.stdout.write(f"   ✅ Available slots: {available}")
            self.stdout.write(f"   🔴 Occupied slots: {occupied}")
            
            if total_slots > 0:
                occupancy_rate = (occupied / total_slots) * 100
                self.stdout.write(f"   📈 Occupancy rate: {occupancy_rate:.1f}%")
            
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"⚠️ Could not fetch all statistics: {str(e)}"))

    def success(self, message):
        """Helper method for success messages"""
        self.stdout.write(self.style.SUCCESS(message))

    def warning(self, message):
        """Helper method for warning messages"""
        self.stdout.write(self.style.WARNING(message))

    def error(self, message):
        """Helper method for error messages"""
        self.stdout.write(self.style.ERROR(message))