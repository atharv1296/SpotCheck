"""
Management command to check Forbes Marshall Parking System health and status.
"""

from django.core.management.base import BaseCommand
from django.db import connection
from django.conf import settings
from parking_app.models import Vehicle, ParkingSlot, ParkingSession
from datetime import datetime, timedelta
import os


class Command(BaseCommand):
    help = 'Check Forbes Marshall Parking System health and status'

    def add_arguments(self, parser):
        parser.add_argument(
            '--detailed',
            action='store_true',
            help='Show detailed system information'
        )
        parser.add_argument(
            '--export',
            type=str,
            help='Export report to file (specify filename)'
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('🏢 Forbes Marshall Parking System Health Check')
        )
        self.stdout.write('=' * 60)

        report_data = []
        
        # System Information
        system_info = self.get_system_info()
        self.display_system_info(system_info)
        report_data.extend(system_info)

        # Database Health
        db_health = self.check_database_health()
        self.display_database_health(db_health)
        report_data.extend(db_health)

        # Parking System Stats
        parking_stats = self.get_parking_stats()
        self.display_parking_stats(parking_stats)
        report_data.extend(parking_stats)

        if options['detailed']:
            # Detailed Analysis
            detailed_stats = self.get_detailed_stats()
            self.display_detailed_stats(detailed_stats)
            report_data.extend(detailed_stats)

        # Export report if requested
        if options['export']:
            self.export_report(report_data, options['export'])

        self.stdout.write(
            self.style.SUCCESS('\n✅ Health check completed successfully!')
        )

    def get_system_info(self):
        """Get basic system information."""
        config = getattr(settings, 'PARKING_SYSTEM_CONFIG', {})
        
        return [
            ('System Information', ''),
            ('Company', config.get('COMPANY_NAME', 'N/A')),
            ('System', config.get('SYSTEM_NAME', 'N/A')),
            ('Version', config.get('VERSION', 'N/A')),
            ('Debug Mode', 'ON' if settings.DEBUG else 'OFF'),
            ('Time Zone', settings.TIME_ZONE),
            ('Current Time', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        ]

    def display_system_info(self, info):
        """Display system information."""
        self.stdout.write('\n📋 System Information:')
        for key, value in info[1:]:  # Skip header
            self.stdout.write(f'   {key}: {value}')

    def check_database_health(self):
        """Check database connectivity and health."""
        health_data = [('Database Health', '')]
        
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                health_data.append(('Connection', '✅ Connected'))
                
                # Check table existence
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name LIKE 'parking_app_%'
                """)
                tables = cursor.fetchall()
                health_data.append(('Tables Found', len(tables)))
                
        except Exception as e:
            health_data.append(('Connection', f'❌ Failed: {e}'))
            
        return health_data

    def display_database_health(self, health):
        """Display database health information."""
        self.stdout.write('\n🗄️  Database Health:')
        for key, value in health[1:]:  # Skip header
            self.stdout.write(f'   {key}: {value}')

    def get_parking_stats(self):
        """Get parking system statistics."""
        try:
            total_slots = ParkingSlot.objects.count()
            occupied_slots = ParkingSlot.objects.filter(status='occupied').count()
            available_slots = total_slots - occupied_slots
            
            total_vehicles = Vehicle.objects.count()
            active_sessions = ParkingSession.objects.filter(is_active=True).count()
            
            # Recent activity (last 24 hours)
            yesterday = datetime.now() - timedelta(days=1)
            recent_entries = ParkingSession.objects.filter(
                entry_time__gte=yesterday
            ).count()
            
            occupancy_rate = (occupied_slots / total_slots * 100) if total_slots > 0 else 0
            
            return [
                ('Parking Statistics', ''),
                ('Total Slots', total_slots),
                ('Occupied Slots', occupied_slots),
                ('Available Slots', available_slots),
                ('Occupancy Rate', f'{occupancy_rate:.1f}%'),
                ('Total Vehicles', total_vehicles),
                ('Active Sessions', active_sessions),
                ('Recent Entries (24h)', recent_entries),
            ]
            
        except Exception as e:
            return [
                ('Parking Statistics', ''),
                ('Status', f'❌ Error: {e}')
            ]

    def display_parking_stats(self, stats):
        """Display parking statistics."""
        self.stdout.write('\n📊 Parking Statistics:')
        for key, value in stats[1:]:  # Skip header
            if key == 'Occupancy Rate':
                color = self.style.SUCCESS if float(value.rstrip('%')) < 90 else self.style.WARNING
                self.stdout.write(f'   {key}: {color(str(value))}')
            else:
                self.stdout.write(f'   {key}: {value}')

    def get_detailed_stats(self):
        """Get detailed system statistics."""
        try:
            # Slot type distribution
            slot_types = ParkingSlot.objects.values('slot_type').distinct()
            detailed_stats = [('Detailed Analysis', '')]
            
            for slot_type in slot_types:
                type_name = slot_type['slot_type']
                total = ParkingSlot.objects.filter(slot_type=type_name).count()
                occupied = ParkingSlot.objects.filter(
                    slot_type=type_name, status='occupied'
                ).count()
                detailed_stats.append((f'{type_name.title()} Slots', f'{occupied}/{total}'))
            
            # Vehicle type distribution
            vehicle_types = Vehicle.objects.values('vehicle_type').distinct()
            for vehicle_type in vehicle_types:
                type_name = vehicle_type['vehicle_type']
                count = Vehicle.objects.filter(vehicle_type=type_name).count()
                detailed_stats.append((f'{type_name.title()} Vehicles', count))
            
            return detailed_stats
            
        except Exception as e:
            return [
                ('Detailed Analysis', ''),
                ('Status', f'❌ Error: {e}')
            ]

    def display_detailed_stats(self, stats):
        """Display detailed statistics."""
        self.stdout.write('\n🔍 Detailed Analysis:')
        for key, value in stats[1:]:  # Skip header
            self.stdout.write(f'   {key}: {value}')

    def export_report(self, data, filename):
        """Export health check report to file."""
        try:
            with open(filename, 'w') as f:
                f.write('Forbes Marshall Parking System Health Report\n')
                f.write('=' * 50 + '\n')
                f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
                
                current_section = None
                for item in data:
                    if len(item) == 2 and item[1] == '':  # Section header
                        if current_section:
                            f.write('\n')
                        current_section = item[0]
                        f.write(f'{current_section}:\n')
                        f.write('-' * len(current_section) + ':\n')
                    else:
                        f.write(f'  {item[0]}: {item[1]}\n')
            
            self.stdout.write(
                self.style.SUCCESS(f'📄 Report exported to: {filename}')
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Export failed: {e}')
            )