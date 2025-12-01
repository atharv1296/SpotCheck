"""Show all users in database"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from django.contrib.auth.models import User

print("\n=== ALL USERS IN DATABASE ===")
users = User.objects.all()
print(f"Total Users: {users.count()}\n")

for user in users:
    print(f"ID: {user.id}")
    print(f"Username: {user.username}")
    print(f"Email: {user.email}")
    print(f"Superuser: {user.is_superuser}")
    print(f"Staff: {user.is_staff}")
    print(f"Active: {user.is_active}")
    print(f"Password Hash: {user.password}")
    print("-" * 50)
