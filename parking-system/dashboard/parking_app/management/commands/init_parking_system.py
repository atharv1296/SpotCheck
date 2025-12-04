"""
Management command to initialize Forbes Marshall Parking System with sample data.
"""

from django.core.management.base import BaseCommand
from django.db import transaction
from parking_app.models import Vehicle, ParkingSlot, ParkingSession
import logging

logger = logging.getLogger('parking_app')


class Command(BaseCommand):
    help = 'Initialize Forbes Marshall Parking System with sample data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--slots',
            type=int,
            default=500,
            help='Number of parking slots to create (default: 500)'
        )
        parser.add_argument(
            '--vehicles',
            type=int,
            default=50,
            help='Number of sample vehicles to create (default: 50)'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing data before initialization'
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('🏢 Initializing Forbes Marshall Parking System...')
        )

        if options['clear']:
            self.clear_existing_data()

        with transaction.atomic():
            self.create_parking_slots(options['slots'])
            self.create_sample_vehicles(options['vehicles'])
            self.create_sample_sessions()

        self.stdout.write(
            self.style.SUCCESS('✅ Forbes Marshall Parking System initialized successfully!')
        )

    def clear_existing_data(self):
        """Clear existing parking system data."""
        self.stdout.write('🗑️  Clearing existing data...')
        
        ParkingSession.objects.all().delete()
        Vehicle.objects.all().delete()
        ParkingSlot.objects.all().delete()
        
        self.stdout.write(self.style.WARNING('⚠️  Existing data cleared'))

    def create_parking_slots(self, count):
        """Create parking slots."""
        self.stdout.write(f'🅿️  Creating {count} parking slots...')
        
        slots_to_create = []
        slot_types = ['compact', 'standard', 'large', 'electric']
        floors = ['Ground', 'First', 'Second']
        
        for i in range(1, count + 1):
            floor = floors[(i - 1) // (count // len(floors))] if count >= len(floors) else 'Ground'
            slot_type = slot_types[i % len(slot_types)]
            
            slot = ParkingSlot(
                slot_number=f'{floor[0]}{i:03d}',
                slot_type=slot_type,
                floor_level=floor,
                is_occupied=False
            )
            slots_to_create.append(slot)
            
            # Batch create every 100 slots
            if len(slots_to_create) >= 100:
                ParkingSlot.objects.bulk_create(slots_to_create)
                slots_to_create = []
        
        # Create remaining slots
        if slots_to_create:
            ParkingSlot.objects.bulk_create(slots_to_create)
        
        self.stdout.write(
            self.style.SUCCESS(f'✅ Created {count} parking slots')
        )

    def create_sample_vehicles(self, count):
        """Create sample vehicles."""
        self.stdout.write(f'🚗 Creating {count} sample vehicles...')
        
        vehicle_types = ['sedan', 'suv', 'two_wheeler', 'truck']
        states = ['MH', 'KA', 'DL', 'GJ', 'TN', 'UP', 'HR', 'PB']
        
        vehicles_to_create = []
        
        for i in range(1, count + 1):
            state = states[i % len(states)]
            vehicle_type = vehicle_types[i % len(vehicle_types)]
            
            vehicle = Vehicle(
                license_plate=f'{state}{i:02d}AB{1000 + i}',
                vehicle_type=vehicle_type,
                owner_name=f'Owner {i}',
                owner_phone=f'+91{9000000000 + i}',
                owner_email=f'owner{i}@example.com'
            )
            vehicles_to_create.append(vehicle)
        
        Vehicle.objects.bulk_create(vehicles_to_create)
        
        self.stdout.write(
            self.style.SUCCESS(f'✅ Created {count} sample vehicles')
        )

    def create_sample_sessions(self):
        """Create some sample active parking sessions."""
        self.stdout.write('📊 Creating sample parking sessions...')
        
        vehicles = list(Vehicle.objects.all()[:20])  # Use first 20 vehicles
        slots = list(ParkingSlot.objects.filter(is_occupied=False)[:20])  # Get 20 empty slots
        
        sessions_created = 0
        for vehicle, slot in zip(vehicles, slots):
            session = ParkingSession.objects.create(
                vehicle=vehicle,
                parking_slot=slot,
                entry_time=timezone.now(),
                is_active=True
            )
            
            # Mark slot as occupied
            slot.is_occupied = True
            slot.save()
            
            sessions_created += 1
        
        self.stdout.write(
            self.style.SUCCESS(f'✅ Created {sessions_created} active parking sessions')
        )

    def handle_error(self, error):
        """Handle command errors."""
        self.stdout.write(
            self.style.ERROR(f'❌ Error: {error}')
        )
        logger.error(f'Initialization command error: {error}')