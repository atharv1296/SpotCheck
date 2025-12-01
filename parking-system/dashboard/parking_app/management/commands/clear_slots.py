"""
Management command to remove all ParkingSlot rows (and related sessions).

Usage:
  python manage.py clear_slots --confirm

This will:
  1) Delete all ParkingSession rows (safety; also handled by FK cascade)
  2) Delete all ParkingSlot rows
  3) Print before/after counts
"""

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from parking_app.models import ParkingSlot, ParkingSession


class Command(BaseCommand):
    help = 'Delete ALL rows from ParkingSlot (and sessions). Requires --confirm.'

    def add_arguments(self, parser):
        parser.add_argument('--confirm', action='store_true', help='Confirm destructive delete of all slots')

    def handle(self, *args, **options):
        if not options['confirm']:
            raise CommandError('Refusing to delete slots without --confirm')

        total_slots = ParkingSlot.objects.count()
        total_sessions = ParkingSession.objects.count()
        self.stdout.write(f'About to delete: {total_slots} slots and {total_sessions} sessions')

        with transaction.atomic():
            # Delete sessions first (explicit safety), then slots
            sess_deleted, _ = ParkingSession.objects.all().delete()
            self.stdout.write(f'Deleted sessions: {sess_deleted}')

            slots_deleted, _ = ParkingSlot.objects.all().delete()
            self.stdout.write(f'Deleted slots: {slots_deleted}')

        self.stdout.write(self.style.SUCCESS('✅ All slots removed successfully.'))
        self.stdout.write(f'Final counts -> slots: {ParkingSlot.objects.count()}, sessions: {ParkingSession.objects.count()}')
