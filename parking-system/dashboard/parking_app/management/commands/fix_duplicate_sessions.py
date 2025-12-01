"""
Django management command to fix duplicate active parking sessions
"""
from django.core.management.base import BaseCommand
from django.db.models import Count
from parking_app.models import ParkingSession, ParkingSlot
from django.utils import timezone


class Command(BaseCommand):
    help = 'Fix duplicate active parking sessions for the same slot'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🔧 Starting duplicate session cleanup...'))
        
        # Find slots with multiple active sessions
        duplicate_sessions = ParkingSession.objects.filter(
            is_active=True
        ).values('parking_slot').annotate(
            count=Count('id')
        ).filter(count__gt=1)
        
        total_fixed = 0
        
        for item in duplicate_sessions:
            slot_id = item['parking_slot']
            count = item['count']
            
            try:
                slot = ParkingSlot.objects.get(id=slot_id)
                
                # Get all active sessions for this slot, ordered by entry time
                sessions = ParkingSession.objects.filter(
                    parking_slot=slot,
                    is_active=True
                ).order_by('entry_time')
                
                # Keep the most recent one, deactivate the rest
                sessions_to_deactivate = list(sessions)[:-1]
                
                for session in sessions_to_deactivate:
                    session.is_active = False
                    session.exit_time = timezone.now()
                    session.save()
                    
                    self.stdout.write(
                        self.style.WARNING(
                            f'  ❌ Deactivated old session: {session.vehicle.license_plate} '
                            f'in {slot.slot_number} (Entry: {session.entry_time})'
                        )
                    )
                    total_fixed += 1
                
                # Keep the latest session
                latest_session = sessions.last()
                self.stdout.write(
                    self.style.SUCCESS(
                        f'  ✅ Kept active session: {latest_session.vehicle.license_plate} '
                        f'in {slot.slot_number} (Entry: {latest_session.entry_time})'
                    )
                )
                
            except ParkingSlot.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'  ⚠️ Slot ID {slot_id} not found')
                )
                continue
        
        if total_fixed == 0:
            self.stdout.write(self.style.SUCCESS('✅ No duplicate sessions found!'))
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    f'\n✅ Fixed {total_fixed} duplicate session(s)!'
                )
            )
        
        # Verify no duplicates remain
        remaining_duplicates = ParkingSession.objects.filter(
            is_active=True
        ).values('parking_slot').annotate(
            count=Count('id')
        ).filter(count__gt=1).count()
        
        if remaining_duplicates > 0:
            self.stdout.write(
                self.style.ERROR(
                    f'⚠️ Warning: {remaining_duplicates} slot(s) still have multiple active sessions'
                )
            )
        else:
            self.stdout.write(self.style.SUCCESS('✅ Verification: All slots have at most one active session'))
