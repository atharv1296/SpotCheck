"""
Authentication views for Forbes Marshall SpotCheck
"""
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.cache import never_cache

@csrf_protect
@never_cache
def login_view(request):
    """Login page for Forbes Marshall SpotCheck"""
    # Redirect if already logged in
    if request.user.is_authenticated:
        return redirect('parking_app:dashboard')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        remember_me = request.POST.get('remember_me')
        
        # Authenticate user
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            # Check if user is staff/admin
            if not user.is_staff:
                messages.error(request, 'Access denied. Only staff members can login to the system.')
                return render(request, 'auth/login.html')
            
            login(request, user)
            
            # Set session expiry based on remember me
            if not remember_me:
                # Session expires after 10 minutes of inactivity (from settings.py)
                request.session.set_expiry(600)  # 10 minutes = 600 seconds
            else:
                # Remember me: 2 weeks
                request.session.set_expiry(1209600)  # 2 weeks = 1209600 seconds
            
            messages.success(request, f'Welcome back, {user.get_full_name() or user.username}!')
            
            # Redirect to next page or dashboard
            next_url = request.GET.get('next', 'parking_app:dashboard')
            return redirect(next_url)
        else:
            messages.error(request, 'Invalid username or password. Please try again.')
    
    return render(request, 'auth/login.html')

@login_required
def logout_view(request):
    """Logout user and immediately destroy session"""
    username = request.user.username
    
    # Django's logout() already does:
    # 1. Flushes session data
    # 2. Regenerates session key
    # 3. Deletes session from database
    logout(request)
    
    # Extra: Explicitly clear all session data and flush
    request.session.flush()  # Ensure session is completely destroyed
    
    # Set cookie to expire immediately
    response = redirect('login')
    response.set_cookie('spotcheck_sessionid', '', max_age=0)  # Delete cookie immediately
    
    messages.success(request, f'Goodbye, {username}! You have been logged out successfully.')
    return response
