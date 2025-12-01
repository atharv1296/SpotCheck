from django.db import migrations


def merge_to_car(apps, schema_editor):
    ParkingSlot = apps.get_model('parking_app', 'ParkingSlot')
    ParkingSlot.objects.filter(slot_type__in=['sedan', 'suv']).update(slot_type='car')


class Migration(migrations.Migration):

    dependencies = [
        ('parking_app', '0004_largevehiclerequest'),
    ]

    operations = [
        migrations.RunPython(merge_to_car, reverse_code=migrations.RunPython.noop),
    ]
