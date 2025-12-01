from django.core.management.base import BaseCommand
from django.conf import settings
from django.db import transaction
from driver_applications.models import DriverApplication
import os
import glob

MEDIA_SUBDIR = os.path.join('driver_applications')

FIELD_MAP = {
    'driver_photo': os.path.join('photos', '{id}_driver_photo.*'),
    'puc_certificate': os.path.join('puc', '{id}_puc.*'),
    'material_receipt': os.path.join('receipts', '{id}_material_receipt.*'),
    'vehicle_rc': os.path.join('rc', '{id}_vehicle_rc.*'),
    'vehicle_insurance': os.path.join('insurance', '{id}_insurance.*'),
}


class Command(BaseCommand):
    help = 'Attach media files from media/driver_applications to DriverApplication records when files exist on disk but not in DB.'

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true', help='Do not save changes; only report')

    def handle(self, *args, **options):
        dry_run = options.get('dry_run', False)

        media_root = getattr(settings, 'MEDIA_ROOT', None)
        if not media_root:
            self.stdout.write(self.style.ERROR('MEDIA_ROOT is not configured.'))
            return

        apps = DriverApplication.objects.all()
        total = apps.count()
        self.stdout.write(f'Found {total} applications; scanning for media files...')

        attached = 0
        missing = 0

        for app in apps:
            app_id = str(app.application_id)
            changed_fields = []

            for field, pattern in FIELD_MAP.items():
                # skip if field already populated
                current = getattr(app, field)
                if current and getattr(current, 'name', None):
                    continue

                rel_pattern = os.path.join(MEDIA_SUBDIR, pattern.format(id=app_id))
                abs_pattern = os.path.join(media_root, rel_pattern)
                matches = glob.glob(abs_pattern)
                if not matches:
                    missing += 1
                    continue

                # pick the first match
                abs_path = matches[0]
                # relative path saved in FileField should be relative to MEDIA_ROOT
                rel_path = os.path.relpath(abs_path, media_root).replace('\\', '/')

                self.stdout.write(f'Will attach {rel_path} → {app_id}.{field}')
                if not dry_run:
                    setattr(app, field, rel_path)
                    changed_fields.append(field)

            if changed_fields and not dry_run:
                try:
                    with transaction.atomic():
                        app.save(update_fields=changed_fields)
                    attached += 1
                except Exception as e:
                    self.stderr.write(f'Failed to save {app_id}: {e}')

        self.stdout.write(self.style.SUCCESS(f'Scan complete. Attached files for {attached} applications.'))
        self.stdout.write(f'Files not found for {missing} field slots (may be normal).')
