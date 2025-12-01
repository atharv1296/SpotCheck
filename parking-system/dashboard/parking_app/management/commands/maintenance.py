"""
Maintenance Management Command
Quick CLI tool to manage slot maintenance status
"""
from django.core.management.base import BaseCommand
from parking_app.models import ParkingSlot


class Command(BaseCommand):
    help = "Manage parking slot maintenance status"

    def add_arguments(self, parser):
        parser.add_argument('action', type=str, help='Action: mark, unmark, list, status')
        parser.add_argument('--slots', nargs='+', help='Slot numbers (e.g., TW1 C5 L1)')
        parser.add_argument('--type', type=str, help='Slot type filter (two_wheeler, car, large)')
        parser.add_argument('--all', action='store_true', help='Apply to all slots')

    def handle(self, *args, **options):
        action = options['action'].lower()
        
        if action == 'list':
            self.list_maintenance_slots()
        elif action == 'status':
            self.show_status()
        elif action == 'mark':
            self.mark_maintenance(options)
        elif action == 'unmark':
            self.unmark_maintenance(options)
        else:
            self.stdout.write(self.style.ERROR(f'Unknown action: {action}'))
            self.stdout.write('Available actions: mark, unmark, list, status')

    def list_maintenance_slots(self):
        """List all slots currently under maintenance"""
        self.stdout.write("=" * 60)
        self.stdout.write(self.style.SUCCESS("SLOTS UNDER MAINTENANCE"))
        self.stdout.write("=" * 60)
        
        maintenance_slots = ParkingSlot.objects.filter(status='maintenance').order_by('slot_number')
        
        if not maintenance_slots:
            self.stdout.write(self.style.WARNING("No slots currently under maintenance"))
            return
        
        for slot in maintenance_slots:
            self.stdout.write(f"  {slot.slot_number:<10} {slot.get_slot_type_display():<20} Floor {slot.floor_level}")
        
        self.stdout.write(f"\nTotal: {maintenance_slots.count()} slots")

    def show_status(self):
        """Show overall status breakdown"""
        self.stdout.write("=" * 60)
        self.stdout.write(self.style.SUCCESS("PARKING SLOT STATUS SUMMARY"))
        self.stdout.write("=" * 60)
        
        total = ParkingSlot.objects.count()
        available = ParkingSlot.objects.filter(status='available').count()
        occupied = ParkingSlot.objects.filter(status='occupied').count()
        maintenance = ParkingSlot.objects.filter(status='maintenance').count()
        out_of_service = ParkingSlot.objects.filter(status='out_of_service').count()
        
        self.stdout.write(f"\nTotal Slots:       {total}")
        self.stdout.write(self.style.SUCCESS(f"Available:         {available}"))
        self.stdout.write(self.style.ERROR(f"Occupied:          {occupied}"))
        self.stdout.write(self.style.WARNING(f"Maintenance:       {maintenance}"))
        self.stdout.write(f"Out of Service:    {out_of_service}")
        
        # By type
        self.stdout.write("\nBy Slot Type:")
        for slot_type in ['two_wheeler', 'car', 'large']:
            type_total = ParkingSlot.objects.filter(slot_type=slot_type).count()
            type_maintenance = ParkingSlot.objects.filter(slot_type=slot_type, status='maintenance').count()
            type_name = slot_type.replace('_', ' ').title()
            self.stdout.write(f"  {type_name:<15} Total: {type_total:<3} Maintenance: {type_maintenance}")

    def mark_maintenance(self, options):
        """Mark slots as under maintenance"""
        slots_to_mark = self._get_slots(options)
        
        if not slots_to_mark:
            self.stdout.write(self.style.ERROR("No slots selected"))
            return
        
        updated = 0
        failed = 0
        
        for slot in slots_to_mark:
            if slot.status == 'occupied':
                self.stdout.write(self.style.WARNING(f"  ⚠️  {slot.slot_number} - Cannot mark occupied slot"))
                failed += 1
                continue
            
            old_status = slot.get_status_display()
            slot.status = 'maintenance'
            slot.save()
            self.stdout.write(self.style.SUCCESS(f"  ✅ {slot.slot_number} - {old_status} → Under Maintenance"))
            updated += 1
        
        self.stdout.write(f"\nMarked {updated} slot(s) as maintenance")
        if failed > 0:
            self.stdout.write(self.style.WARNING(f"Failed: {failed} slot(s)"))

    def unmark_maintenance(self, options):
        """Return slots to available status"""
        slots_to_unmark = self._get_slots(options)
        
        if not slots_to_unmark:
            self.stdout.write(self.style.ERROR("No slots selected"))
            return
        
        updated = 0
        
        for slot in slots_to_unmark:
            if slot.status not in ['maintenance', 'out_of_service']:
                self.stdout.write(self.style.WARNING(f"  ⚠️  {slot.slot_number} - Not in maintenance"))
                continue
            
            old_status = slot.get_status_display()
            slot.status = 'available'
            slot.save()
            self.stdout.write(self.style.SUCCESS(f"  ✅ {slot.slot_number} - {old_status} → Available"))
            updated += 1
        
        self.stdout.write(f"\nReturned {updated} slot(s) to service")

    def _get_slots(self, options):
        """Get queryset based on options"""
        if options['all']:
            queryset = ParkingSlot.objects.all()
        elif options['slots']:
            queryset = ParkingSlot.objects.filter(slot_number__in=options['slots'])
        elif options['type']:
            queryset = ParkingSlot.objects.filter(slot_type=options['type'])
        else:
            return None
        
        return queryset
