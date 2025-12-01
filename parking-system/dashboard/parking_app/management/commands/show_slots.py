"""
Management command to display entries from the ParkingSlot table.

Usage examples:
  python manage.py show_slots                 # list first 200 via ORM
  python manage.py show_slots --limit 500    # list up to 500 rows
  python manage.py show_slots --raw-sql      # run raw SQL SELECT *
"""

from django.core.management.base import BaseCommand
from django.db import connection
from parking_app.models import ParkingSlot


class Command(BaseCommand):
    help = 'Show entries from the ParkingSlot table (ORM or raw SQL)'

    def add_arguments(self, parser):
        parser.add_argument('--limit', type=int, default=200, help='Max rows to display (default 200)')
        parser.add_argument('--raw-sql', action='store_true', help='Use raw SQL: SELECT * FROM PARKING_APP_PARKINGSLOT')

    def handle(self, *args, **options):
        limit = options['limit']
        raw = options['raw_sql']

        if raw:
            self.stdout.write('📄 Running raw SQL: SELECT * FROM PARKING_APP_PARKINGSLOT')
            sql = f"SELECT * FROM PARKING_APP_PARKINGSLOT FETCH FIRST {limit} ROWS ONLY"
            with connection.cursor() as cursor:
                cursor.execute(sql)
                cols = [col[0] for col in cursor.description]
                rows = cursor.fetchall()
            self._print_table(cols, rows)
            self.stdout.write(self.style.SUCCESS(f"\n✅ {len(rows)} rows shown (raw SQL)"))
            return

        # ORM path (portable and field-selected)
        qs = (
            ParkingSlot.objects
            .all()
            .order_by('slot_number')
            .values('id', 'slot_number', 'slot_type', 'is_occupied', 'last_updated', 'floor_level', 'created_at')[:limit]
        )
        rows = list(qs)
        if not rows:
            self.stdout.write('No slots found.')
            return
        cols = list(rows[0].keys())
        values = [[self._fmt(v) for v in row.values()] for row in rows]
        self._print_table(cols, values)
        self.stdout.write(self.style.SUCCESS(f"\n✅ {len(rows)} rows shown (ORM)"))

    def _fmt(self, v):
        if v is None:
            return ''
        return str(v)

    def _print_table(self, headers, rows):
        # compute widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        def line(char='-'):
            return '+ ' + ' + '.join(char * w for w in widths) + ' +'

        # header
        self.stdout.write(line('='))
        self.stdout.write('| ' + ' | '.join(h.ljust(w) for h, w in zip(headers, widths)) + ' |')
        self.stdout.write(line('='))
        # rows
        for row in rows:
            self.stdout.write('| ' + ' | '.join(str(c).ljust(w) for c, w in zip(row, widths)) + ' |')
        self.stdout.write(line('='))
