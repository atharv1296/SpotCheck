from django.urls import path
from . import views

app_name = 'driver_applications'

urlpatterns = [
    # Public URLs
    path('apply/', views.apply_entry, name='apply_entry'),
    path('status/<uuid:application_id>/', views.application_status, name='application_status'),
    
    # Admin URLs (will use main website authentication)
    path('dashboard/', views.applications_dashboard, name='applications_dashboard'),
    path('detail/<uuid:application_id>/', views.application_detail, name='application_detail'),
    path('bulk-action/', views.bulk_action, name='bulk_action'),
    
    # API URLs
    path('api/stats/', views.api_application_stats, name='api_application_stats'),
    path('document/<uuid:application_id>/<str:doc_name>/', views.serve_document, name='serve_document'),
]