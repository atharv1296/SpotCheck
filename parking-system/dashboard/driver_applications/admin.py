from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils import timezone
from .models import DriverApplication, ApplicationComment, ApplicationStatusHistory

@admin.register(DriverApplication)
class DriverApplicationAdmin(admin.ModelAdmin):
    list_display = [
        'application_id_short', 'driver_name', 'vehicle_number', 
        'source_company_name', 'status_badge', 'urgency_badge',
        'requested_entry_date', 'created_at'
    ]
    list_filter = [
        'status', 'urgency', 'material_type', 'requested_entry_date',
        'created_at', 'source_company_name'
    ]
    search_fields = [
        'driver_name', 'vehicle_number', 'driver_phone', 
        'driver_license_number', 'source_company_name'
    ]
    readonly_fields = [
        'application_id', 'created_at', 'updated_at', 'is_expired'
    ]
    
    fieldsets = (
        ('Application Info', {
            'fields': ('application_id', 'status', 'urgency', 'created_at', 'updated_at')
        }),
        ('Entry Details', {
            'fields': ('requested_entry_date', 'requested_entry_time', 'estimated_duration')
        }),
        ('Driver Information', {
            'fields': (
                    'driver_name', 'driver_phone', 'driver_email', 
                    'driver_license_number', 'driver_photo_name'
            )
        }),
        ('Vehicle Information', {
            'fields': (
                'vehicle_number', 'vehicle_type', 'vehicle_model', 'vehicle_capacity'
            )
        }),
        ('Company Information', {
            'fields': (
                'source_company_name', 'source_company_address', 
                'source_company_contact', 'destination_within_premises'
            )
        }),
        ('Material/Purpose', {
            'fields': (
                'material_type', 'material_description', 'material_weight', 
                'material_value'
            )
        }),
        ('Required Documents', {
            'fields': (
                'vehicle_rc_name', 'vehicle_insurance_name', 'puc_certificate_name', 'material_receipt_name'
            )
        }),
        ('Optional Documents', {
            'fields': ('goods_transport_permit_name', 'customs_clearance_name')
        }),
        ('Review & Status', {
            'fields': (
                'reviewed_by', 'reviewed_at', 'admin_comments', 
                'rejection_reason'
            )
        }),
        ('Entry Tracking', {
            'fields': (
                'approved_entry_slot', 'actual_entry_time', 'actual_exit_time'
            )
        }),
    )
    
    actions = ['approve_applications', 'reject_applications', 'mark_under_review']
    
    def application_id_short(self, obj):
        return str(obj.application_id)[:8] + '...'
    application_id_short.short_description = 'App ID'
    
    def status_badge(self, obj):
        colors = {
            'pending': 'warning',
            'under_review': 'info',
            'approved': 'success',
            'rejected': 'danger',
            'expired': 'secondary'
        }
        color = colors.get(obj.status, 'secondary')
        return format_html(
            '<span class="badge badge-{}">{}</span>',
            color, obj.get_status_display()
        )
    status_badge.short_description = 'Status'
    
    def urgency_badge(self, obj):
        return format_html(
            '<span class="badge badge-{}">{}</span>',
            obj.urgency_color, obj.get_urgency_display()
        )
    urgency_badge.short_description = 'Priority'
    
    def approve_applications(self, request, queryset):
        updated = queryset.filter(status='pending').update(
            status='approved',
            reviewed_by=request.user.username,
            reviewed_at=timezone.now()
        )
        self.message_user(request, f'{updated} applications approved.')
    approve_applications.short_description = "Approve selected applications"
    
    def reject_applications(self, request, queryset):
        updated = queryset.filter(status='pending').update(
            status='rejected',
            reviewed_by=request.user.username,
            reviewed_at=timezone.now()
        )
        self.message_user(request, f'{updated} applications rejected.')
    reject_applications.short_description = "Reject selected applications"
    
    def mark_under_review(self, request, queryset):
        updated = queryset.filter(status='pending').update(
            status='under_review',
            reviewed_by=request.user.username,
            reviewed_at=timezone.now()
        )
        self.message_user(request, f'{updated} applications marked under review.')
    mark_under_review.short_description = "Mark as under review"

@admin.register(ApplicationComment)
class ApplicationCommentAdmin(admin.ModelAdmin):
    list_display = ['application_short', 'comment_by', 'is_internal', 'created_at']
    list_filter = ['is_internal', 'created_at', 'comment_by']
    
    def application_short(self, obj):
        return f"{obj.application.driver_name} - {obj.application.vehicle_number}"
    application_short.short_description = 'Application'

@admin.register(ApplicationStatusHistory)
class ApplicationStatusHistoryAdmin(admin.ModelAdmin):
    list_display = [
        'application_short', 'previous_status', 'new_status', 
        'changed_by', 'changed_at'
    ]
    list_filter = ['previous_status', 'new_status', 'changed_at']
    readonly_fields = ['changed_at']
    
    def application_short(self, obj):
        return f"{obj.application.driver_name} - {obj.application.vehicle_number}"
    application_short.short_description = 'Application'
