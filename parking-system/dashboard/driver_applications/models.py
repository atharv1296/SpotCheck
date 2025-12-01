from django.db import models
from django.core.validators import RegexValidator
from django.utils import timezone
import uuid
import os

def upload_driver_photo(instance, filename):
    """Upload driver photos to organized directory"""
    ext = filename.split('.')[-1]
    filename = f"{instance.application_id}_driver_photo.{ext}"
    return os.path.join('driver_applications', 'photos', filename)

def upload_license(instance, filename):
    """Upload license documents to organized directory"""
    ext = filename.split('.')[-1]
    filename = f"{instance.application_id}_license.{ext}"
    return os.path.join('driver_applications', 'licenses', filename)

def upload_puc(instance, filename):
    """Upload PUC documents to organized directory"""
    ext = filename.split('.')[-1]
    filename = f"{instance.application_id}_puc.{ext}"
    return os.path.join('driver_applications', 'puc', filename)

def upload_insurance(instance, filename):
    """Upload insurance documents to organized directory"""
    ext = filename.split('.')[-1]
    filename = f"{instance.application_id}_insurance.{ext}"
    return os.path.join('driver_applications', 'insurance', filename)

def upload_material_receipt(instance, filename):
    """Upload material receipt documents to organized directory"""
    ext = filename.split('.')[-1]
    filename = f"{instance.application_id}_material_receipt.{ext}"
    return os.path.join('driver_applications', 'receipts', filename)

def upload_vehicle_rc(instance, filename):
    """Upload vehicle RC documents to organized directory"""
    ext = filename.split('.')[-1]
    filename = f"{instance.application_id}_vehicle_rc.{ext}"
    return os.path.join('driver_applications', 'rc', filename)

class DriverApplication(models.Model):
    APPLICATION_STATUS_CHOICES = [
        ('pending', 'Pending Review'),
        ('under_review', 'Under Review'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
        ('expired', 'Expired'),
    ]
    
    MATERIAL_TYPE_CHOICES = [
        ('raw_materials', 'Raw Materials'),
        ('finished_goods', 'Finished Products'),
        ('office_supplies', 'Office Supplies'),
        ('equipment', 'Equipment/Machinery'),
        ('maintenance', 'Maintenance Supplies'),
        ('waste_disposal', 'Waste/Disposal'),
        ('food_beverages', 'Food & Beverages'),
        ('documents', 'Documents/Papers'),
        ('other', 'Other (Specify)'),
    ]
    
    URGENCY_CHOICES = [
        ('low', 'Low Priority'),
        ('medium', 'Medium Priority'),
        ('high', 'High Priority'),
        ('emergency', 'Emergency'),
    ]

    # Application ID and timestamps
    application_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    requested_entry_date = models.DateField()
    requested_entry_time = models.TimeField()
    estimated_duration = models.PositiveIntegerField(help_text="Estimated stay duration in minutes")
    
    # Driver Information
    driver_name = models.CharField(max_length=100)
    driver_phone = models.CharField(
        max_length=15,
        validators=[RegexValidator(r'^\+?91?[6-9]\d{9}$', 'Enter a valid Indian phone number')]
    )
    driver_email = models.EmailField(blank=True, null=True)
    driver_license_number = models.CharField(max_length=20)
    
    # Vehicle Information  
    vehicle_number = models.CharField(
        max_length=15,
        validators=[RegexValidator(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{1,4}$', 'Enter valid vehicle number')]
    )
    vehicle_type = models.CharField(max_length=50, default='large')
    vehicle_model = models.CharField(max_length=100)
    vehicle_capacity = models.CharField(max_length=50, help_text="e.g., 10 tons, 50 passengers")
    
    # Company/Business Information
    source_company_name = models.CharField(max_length=200)
    source_company_address = models.TextField()
    source_company_contact = models.CharField(max_length=15)
    destination_within_premises = models.CharField(max_length=200, help_text="Specific location within Forbes Marshall")
    
    # Material/Purpose Information
    material_type = models.CharField(max_length=20, choices=MATERIAL_TYPE_CHOICES)
    material_description = models.TextField(help_text="Detailed description of materials/purpose")
    material_weight = models.CharField(max_length=50, blank=True, help_text="Approximate weight")
    material_value = models.DecimalField(max_digits=12, decimal_places=2, blank=True, null=True, help_text="Approximate value in INR")
    urgency = models.CharField(max_length=10, choices=URGENCY_CHOICES, default='medium')
    
    # Required Documents
    # FileFields removed: storage moved to DB blob/name fields.
    # Legacy FileField columns were not found in the current Oracle schema; use blob/name fields instead.

    # Binary storage (new): store file bytes and original filename in DB
    driver_photo_blob = models.BinaryField(blank=True, null=True, editable=False)
    driver_photo_name = models.CharField(max_length=255, blank=True, null=True)

    driver_license_photo_blob = models.BinaryField(blank=True, null=True, editable=False)
    driver_license_photo_name = models.CharField(max_length=255, blank=True, null=True)

    vehicle_rc_blob = models.BinaryField(blank=True, null=True, editable=False)
    vehicle_rc_name = models.CharField(max_length=255, blank=True, null=True)

    vehicle_insurance_blob = models.BinaryField(blank=True, null=True, editable=False)
    vehicle_insurance_name = models.CharField(max_length=255, blank=True, null=True)

    puc_certificate_blob = models.BinaryField(blank=True, null=True, editable=False)
    puc_certificate_name = models.CharField(max_length=255, blank=True, null=True)

    material_receipt_blob = models.BinaryField(blank=True, null=True, editable=False)
    material_receipt_name = models.CharField(max_length=255, blank=True, null=True)

    goods_transport_permit_blob = models.BinaryField(blank=True, null=True, editable=False)
    goods_transport_permit_name = models.CharField(max_length=255, blank=True, null=True)

    customs_clearance_blob = models.BinaryField(blank=True, null=True, editable=False)
    customs_clearance_name = models.CharField(max_length=255, blank=True, null=True)
    
    # Optional Documents
    # Optional FileFields removed; use blob/name fields instead.
    
    # Application Status
    status = models.CharField(max_length=15, choices=APPLICATION_STATUS_CHOICES, default='pending')
    reviewed_by = models.CharField(max_length=100, blank=True, null=True)
    reviewed_at = models.DateTimeField(blank=True, null=True)
    admin_comments = models.TextField(blank=True, help_text="Internal comments from security/admin")
    rejection_reason = models.TextField(blank=True)
    
    # Entry Details (filled after approval)
    approved_entry_slot = models.CharField(max_length=10, blank=True, null=True)
    actual_entry_time = models.DateTimeField(blank=True, null=True)
    actual_exit_time = models.DateTimeField(blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Driver Application"
        verbose_name_plural = "Driver Applications"
    
    def __str__(self):
        return f"{self.driver_name} - {self.vehicle_number} ({self.get_status_display()})"
    
    @property
    def is_expired(self):
        """Check if application has expired"""
        if self.status == 'approved':
            # Check if entry date has passed
            return timezone.now().date() > self.requested_entry_date
        return False
    
    @property
    def urgency_color(self):
        """Get color class for urgency level"""
        colors = {
            'low': 'success',
            'medium': 'warning', 
            'high': 'danger',
            'emergency': 'dark'
        }
        return colors.get(self.urgency, 'secondary')

class ApplicationComment(models.Model):
    """Comments/notes on driver applications"""
    application = models.ForeignKey(DriverApplication, on_delete=models.CASCADE, related_name='comments')
    comment_by = models.CharField(max_length=100)
    comment_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_internal = models.BooleanField(default=True, help_text="Internal comment (not visible to driver)")
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Comment by {self.comment_by} on {self.application.application_id}"

class ApplicationStatusHistory(models.Model):
    """Track status changes for applications"""
    application = models.ForeignKey(DriverApplication, on_delete=models.CASCADE, related_name='status_history')
    previous_status = models.CharField(max_length=15)
    new_status = models.CharField(max_length=15)
    changed_by = models.CharField(max_length=100)
    changed_at = models.DateTimeField(auto_now_add=True)
    reason = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-changed_at']
        verbose_name = "Status History"
        verbose_name_plural = "Status Histories"
    
    def __str__(self):
        return f"{self.application.application_id}: {self.previous_status} → {self.new_status}"
