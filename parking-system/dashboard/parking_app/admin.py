from django.contrib import admin
from .models import Vehicle, ParkingSlot, ParkingSession, LargeVehicleRequest

@admin.register(Vehicle)
class VehicleAdmin(admin.ModelAdmin):
    list_display = ['license_plate', 'vehicle_type', 'registered_state', 'contact_number', 'created_at']
    list_filter = ['vehicle_type', 'registered_state', 'created_at']
    search_fields = ['license_plate', 'owner_name']
    ordering = ['-created_at']

@admin.register(ParkingSlot)
class ParkingSlotAdmin(admin.ModelAdmin):
    list_display = ['slot_number', 'slot_type', 'status', 'floor_level', 'last_updated']
    list_filter = ['slot_type', 'status', 'floor_level']
    search_fields = ['slot_number']
    ordering = ['slot_number']
    actions = ['mark_maintenance', 'mark_available', 'mark_out_of_service']
    
    @admin.action(description='Mark selected slots as Under Maintenance')
    def mark_maintenance(self, request, queryset):
        updated = queryset.update(status='maintenance')
        self.message_user(request, f'{updated} slot(s) marked as Under Maintenance.')
    
    @admin.action(description='Mark selected slots as Available')
    def mark_available(self, request, queryset):
        # Only mark as available if not currently occupied
        non_occupied = queryset.exclude(status='occupied')
        updated = non_occupied.update(status='available')
        self.message_user(request, f'{updated} slot(s) marked as Available.')
    
    @admin.action(description='Mark selected slots as Out of Service')
    def mark_out_of_service(self, request, queryset):
        updated = queryset.update(status='out_of_service')
        self.message_user(request, f'{updated} slot(s) marked as Out of Service.')

@admin.register(ParkingSession)
class ParkingSessionAdmin(admin.ModelAdmin):
    list_display = ['vehicle', 'parking_slot', 'entry_time', 'exit_time', 'status', 'is_active']
    list_filter = ['status', 'is_active', 'entry_time']
    search_fields = ['vehicle__license_plate', 'parking_slot__slot_number']
    ordering = ['-entry_time']


@admin.register(LargeVehicleRequest)
class LargeVehicleRequestAdmin(admin.ModelAdmin):
    list_display = ['license_plate', 'owner_name', 'contact_number', 'registered_state', 'status', 'requested_at']
    list_filter = ['status', 'registered_state', 'requested_at']
    search_fields = ['license_plate', 'owner_name', 'contact_number']
    ordering = ['-requested_at']
