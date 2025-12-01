"""Show active sessions"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from django.contrib.sessions.models import Session
from django.utils import timezone

print("\n=== ACTIVE SESSIONS IN DATABASE ===")
active_sessions = Session.objects.filter(expire_date__gte=timezone.now())
print(f"Total Active Sessions: {active_sessions.count()}\n")

for session in active_sessions:
    print(f"Session Key: {session.session_key}")
    print(f"Expires: {session.expire_date}")
    
    # Decode session data
    data = session.get_decoded()
    user_id = data.get('_auth_user_id')
    
    if user_id:
        from django.contrib.auth.models import User
        user = User.objects.get(id=user_id)
        print(f"Logged in User: {user.username}")
    
    print(f"Session Data Keys: {list(data.keys())}")
    print("-" * 60)

print("\n💡 This is why you're still logged in after server restart!")
print("   The session is stored in the database AND in your browser cookie.")
