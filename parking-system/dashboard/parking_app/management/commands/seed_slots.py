from django.core.management.base import BaseCommand
from django.db import transaction
from parking_app.models import ParkingSlot


DEFAULT_COUNTS = {
    'two_wheeler': 20,
    'car': 20,
    'large': 5,
}

PREFIXES = {
    'two_wheeler': 'TW',
    'car': 'C',
    'large': 'L',
}


class Command(BaseCommand):
    help = "Seed ParkingSlot entries into the database. Idempotent by slot_number."

    def add_arguments(self, parser):
        parser.add_argument('--two', type=int, help='Number of Two Wheeler slots', default=DEFAULT_COUNTS['two_wheeler'])
        parser.add_argument('--car', type=int, help='Number of Car slots', default=DEFAULT_COUNTS['car'])
        parser.add_argument('--large', type=int, help='Number of Large Vehicle slots', default=DEFAULT_COUNTS['large'])
        parser.add_argument('--floor', type=int, help='Floor level to assign to new slots', default=1)

    def handle(self, *args, **options):
        plan = {
            'two_wheeler': options['two'],
            'car': options['car'],
            'large': options['large'],
        }
        floor_level = options['floor']

        self.stdout.write("== Seeding Parking Slots ==")
        self.stdout.write(f"Target counts: {plan}")

        created_total = 0
        with transaction.atomic():
            for slot_type, count in plan.items():
                if count <= 0:
                    continue
                prefix = PREFIXES[slot_type]

                # Determine next index by checking existing with prefix
                existing = list(ParkingSlot.objects.filter(slot_type=slot_type, slot_number__startswith=prefix)
                                .values_list('slot_number', flat=True))
                existing_numbers = set()
                for sn in existing:
                    num = sn[len(prefix):]
                    if num.isdigit():
                        existing_numbers.add(int(num))

                created_for_type = 0
                i = 1
                while created_for_type < count:
                    if i not in existing_numbers:
                        slot_number = f"{prefix}{i}"
                        ParkingSlot.objects.get_or_create(
                            slot_number=slot_number,
                            defaults={
                                'slot_type': slot_type,
                                'is_occupied': False,
                                'floor_level': floor_level,
                            }
                        )
                        created_for_type += 1
                        created_total += 1
                        self.stdout.write(self.style.SUCCESS(f"Created slot {slot_number} ({slot_type})"))
                    i += 1

        self.stdout.write(self.style.SUCCESS(f"Done. Created up to {created_total} slots (existing preserved)."))
