"""
Authentication URLs for Forbes Marshall SpotCheck
"""
from django.urls import path
from . import auth_views

urlpatterns = [
    path('', auth_views.login_view, name='login'),
    path('logout/', auth_views.logout_view, name='logout'),
]
