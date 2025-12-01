"""
Management command to add N new parking slots per category without duplicates.

Usage:
  python manage.py expand_slots --per-category 20

Categories covered: two_wheeler, sedan, suv, large
Slot numbering prefixes (kept for continuity with existing data):
  two_wheeler -> 'H'  (legacy from Hatchback -> Two Wheeler)
  sedan       -> 'S'
  suv         -> 'U'
  large       -> 'L'
"""

from django.core.management.base import BaseCommand
from django.db import transaction
from parking_app.models import ParkingSlot
import re


PREFIX_MAP = {
	'two_wheeler': 'H',  # legacy prefix retained
	'car': 'C',          # unified Cars category
	'large': 'L',
}


class Command(BaseCommand):
	help = 'Add new parking slots per category using existing numbering scheme'

	def add_arguments(self, parser):
		parser.add_argument(
			'--per-category', type=int, default=20,
			help='Number of new slots to add per category (default: 20)'
		)

	def handle(self, *args, **options):
		per_category = options['per_category']
		self.stdout.write(f"🅿️  Adding {per_category} slots per category...")

		summary = {}

		with transaction.atomic():
			for category, prefix in PREFIX_MAP.items():
				created = self._add_for_category(category, prefix, per_category)
				summary[category] = created

		self.stdout.write("\n📊 Summary:")
		for category, created in summary.items():
			total = ParkingSlot.objects.filter(slot_type=category).count()
			self.stdout.write(
				f"  ✅ {category}: +{created} created | total now: {total}"
			)

		self.stdout.write(self.style.SUCCESS("\n✅ Slot expansion complete."))

	def _add_for_category(self, category: str, prefix: str, count: int) -> int:
		# Find existing numeric suffixes for this prefix
		existing = ParkingSlot.objects.filter(slot_number__startswith=prefix)
		max_n = 0
		pattern = re.compile(rf'^{re.escape(prefix)}(\d+)$')
		for slot in existing:
			m = pattern.match(slot.slot_number)
			if m:
				try:
					n = int(m.group(1))
					if n > max_n:
						max_n = n
				except ValueError:
					continue

		to_create = []
		# Generate next N numbers
		for i in range(1, count + 1):
			slot_number = f"{prefix}{max_n + i}"
			# Double-check uniqueness just in case
			if ParkingSlot.objects.filter(slot_number=slot_number).exists():
				continue
			to_create.append(
				ParkingSlot(
					slot_number=slot_number,
					slot_type=category,
					is_occupied=False,
					floor_level=1,
				)
			)

		if to_create:
			ParkingSlot.objects.bulk_create(to_create)
		self.stdout.write(
			f"   ➕ {category}: created {len(to_create)} new slots (prefix {prefix}, from {max_n+1} to {max_n+len(to_create)})"
		)
		return len(to_create)

