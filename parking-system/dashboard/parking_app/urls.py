from django.urls import path, include
from . import views

app_name = 'parking_app'

urlpatterns = [
    # Main dashboard pages
    path('', views.dashboard, name='dashboard'),
    path('realtime/', views.realtime_view, name='realtime'),
    path('analytics/', views.analytics_view, name='analytics'),
    path('maintenance/', views.maintenance_management, name='maintenance_management'),
    
    # API endpoints
    path('api/parking-data/', views.get_parking_data, name='parking_data'),
    path('api/parking-status/', views.api_parking_status, name='parking_status'),
    path('api/analytics-data/', views.get_analytics_data, name='analytics_data'),
    path('api/update-slot/', views.update_slot_status, name='update_slot'),
    path('api/check-vehicle/', views.check_vehicle, name='check_vehicle'),
    path('api/toggle-maintenance/', views.toggle_maintenance, name='toggle_maintenance'),
    # Large vehicle requests API
    path('api/large-vehicle-requests/', views.large_vehicle_requests, name='large_vehicle_requests'),
    path('api/large-vehicle-requests/<int:request_id>/', views.large_vehicle_request_detail, name='large_vehicle_request_detail'),
    path('api/', include('parking_app.api_urls')),
]
