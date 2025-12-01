from django.core.management.base import BaseCommand
from django.db import connection


class Command(BaseCommand):
    help = "Test Oracle DB connectivity and show basic DB info"

    def handle(self, *args, **options):
        self.stdout.write("== Oracle connectivity test ==")
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 'OK' FROM dual")
                ok = cursor.fetchone()
                if ok and ok[0] == 'OK':
                    self.stdout.write(self.style.SUCCESS("Connected to Oracle successfully"))
                cursor.execute(
                    """
                    SELECT 
                        SYS_CONTEXT('USERENV','DB_NAME'),
                        SYS_CONTEXT('USERENV','CURRENT_USER'),
                        SYS_CONTEXT('USERENV','SERVER_HOST'),
                        SYS_CONTEXT('USERENV','INSTANCE_NAME')
                    FROM dual
                    """
                )
                db_name, current_user, server_host, instance_name = cursor.fetchone()
                self.stdout.write(f"DB Name: {db_name}")
                self.stdout.write(f"User: {current_user}")
                self.stdout.write(f"Host: {server_host}")
                self.stdout.write(f"Instance: {instance_name}")

                # Version banner
                try:
                    cursor.execute("SELECT banner FROM v$version WHERE banner LIKE 'Oracle%'")
                    banner = cursor.fetchone()
                    if banner:
                        self.stdout.write(f"Version: {banner[0]}")
                except Exception:
                    pass

        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Connection failed: {e}"))
            self.stderr.write("Troubleshooting:")
            self.stderr.write(" - Ensure environment variables ORACLE_USER/ORACLE_PASSWORD/ORACLE_DSN are set")
            self.stderr.write(" - If using thick mode, set ORACLE_THICK_MODE=1 and ORACLE_CLIENT_LIB_DIR to Instant Client path")
            self.stderr.write(" - Verify the database is reachable (host/port/service)")
