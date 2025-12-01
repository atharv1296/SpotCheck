import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from django.db import connection
from driver_applications.models import DriverApplication


def list_columns(table_name):
    with connection.cursor() as cur:
        cur.execute("SELECT column_name FROM user_tab_columns WHERE table_name = :tbl ORDER BY column_id", {'tbl': table_name.upper()})
        return [r[0] for r in cur.fetchall()]


if __name__ == '__main__':
    # Determine actual DB table name for the model (Django may shorten for Oracle)
    tbl = DriverApplication._meta.db_table
    print('Model db_table for DriverApplication:', tbl)
    print('Listing columns for table:', tbl)
    try:
        cols = list_columns(tbl)
    except Exception as e:
        print('Error listing columns for', tbl, ':', repr(e))
        raise

    if not cols:
        print('(no columns returned)')
    for c in cols:
        print(' -', c)

    # Print SQL for the dashboard queryset
    qs = DriverApplication.objects.all()
    try:
        print('\nSQL for DriverApplication.objects.all():')
        print(qs.query)
    except Exception as e:
        print('Error generating SQL for queryset:', repr(e))
        raise
