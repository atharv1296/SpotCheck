import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from django.db import connection

def list_columns(table_name):
    with connection.cursor() as cur:
        cur.execute("SELECT column_name FROM user_tab_columns WHERE table_name = :tbl ORDER BY column_id", {'tbl': table_name.upper()})
        return [r[0] for r in cur.fetchall()]

if __name__ == '__main__':
    tbl = 'driver_applications_driverapplication'
    cols = list_columns(tbl)
    print('Columns on', tbl)
    for c in cols:
        print(' -', c)
