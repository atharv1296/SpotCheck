"""Clear all existing sessions"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from django.contrib.sessions.models import Session

print("\n🗑️  Clearing all existing sessions...")
count = Session.objects.all().count()
Session.objects.all().delete()
print(f"✅ Deleted {count} session(s)")
print("\n💡 All users have been logged out!")
print("   Refresh your browser to test the new 10-minute timeout.")
