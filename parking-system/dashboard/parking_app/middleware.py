"""
Custom middleware to ensure only staff users can access the system
"""
from django.shortcuts import redirect
from django.contrib import messages
from django.contrib.auth import logout
from django.urls import reverse

class StaffRequiredMiddleware:
    """
    Middleware to check if user is staff on every request.
    Non-staff users will be logged out and redirected to login.
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        
        # URLs that don't require staff check (login page, static files, etc.)
        self.exempt_urls = [
            '/login/',
            '/static/',
            '/media/',
            '/admin/login/',
        ]
    
    def __call__(self, request):
        # Check if URL is exempt from staff check
        path = request.path
        is_exempt = any(path.startswith(url) for url in self.exempt_urls)
        
        # If user is authenticated but NOT exempt URL
        if request.user.is_authenticated and not is_exempt:
            # Check if user is staff
            if not request.user.is_staff:
                # User is logged in but not staff - logout and redirect
                username = request.user.username
                logout(request)
                request.session.flush()
                
                messages.error(request, f'Access denied for user "{username}". Only staff members can access this system.')
                return redirect('login')
        
        response = self.get_response(request)
        return response
