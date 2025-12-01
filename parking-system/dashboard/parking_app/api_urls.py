from django.urls import path
from . import views

app_name = 'parking_app_api'

urlpatterns = [
    path('parking-status/', views.api_parking_status, name='parking_status'),
    path('parking-data/', views.get_parking_data, name='parking_data'),
    path('update-slot/', views.update_slot_status, name='update_slot'),
    path('analytics-data/', views.get_analytics_data, name='analytics_data'),
    path('analytics/export/', views.get_analytics_export, name='analytics_export'),
    path('recent-activity/', views.api_recent_activity, name='recent_activity'),
    path('realtime-monitoring/', views.api_realtime_monitoring, name='realtime_monitoring'),
    path('vehicle-details/<str:slot_number>/', views.api_vehicle_details, name='vehicle_details'),
    path('test-vehicle-details/<str:slot_number>/', views.api_test_vehicle_details, name='test_vehicle_details'),
]
