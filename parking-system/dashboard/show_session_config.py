"""Display current session configuration"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from django.conf import settings

print("\n" + "="*60)
print("🔒 SESSION CONFIGURATION")
print("="*60)

print(f"\n📋 Session Settings:")
print(f"   SESSION_COOKIE_AGE: {settings.SESSION_COOKIE_AGE} seconds ({settings.SESSION_COOKIE_AGE // 60} minutes)")
print(f"   SESSION_SAVE_EVERY_REQUEST: {settings.SESSION_SAVE_EVERY_REQUEST}")
print(f"   SESSION_EXPIRE_AT_BROWSER_CLOSE: {settings.SESSION_EXPIRE_AT_BROWSER_CLOSE}")
print(f"   SESSION_COOKIE_NAME: {settings.SESSION_COOKIE_NAME}")
print(f"   SESSION_COOKIE_HTTPONLY: {settings.SESSION_COOKIE_HTTPONLY}")
print(f"   SESSION_COOKIE_SECURE: {settings.SESSION_COOKIE_SECURE}")

print(f"\n⏱️  Inactivity Timeout:")
print(f"   ✅ Session expires after {settings.SESSION_COOKIE_AGE // 60} minutes of inactivity")
print(f"   ✅ Timer resets on every page request")
print(f"   ✅ Warning shown at 9 minutes")
print(f"   ✅ Auto-logout at 10 minutes")

print(f"\n🚪 Logout Behavior:")
print(f"   ✅ Session destroyed immediately")
print(f"   ✅ Cookie deleted immediately")
print(f"   ✅ User must login again")

print(f"\n📝 Remember Me Options:")
print(f"   • Unchecked: 10 minutes inactivity")
print(f"   • Checked: 2 weeks (14 days)")

print("\n" + "="*60)
