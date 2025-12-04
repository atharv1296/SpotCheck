from django.db import models
from django.utils import timezone

class Vehicle(models.Model):
    VEHICLE_TYPES = [
        ('two_wheeler', 'Two Wheeler'),
        ('sedan', 'Sedan'),
        ('suv', 'SUV'),
        ('large', 'Large Vehicle'),
    ]
    
    STATE_CHOICES = [
        ('AN', 'Andaman and Nicobar Islands'),
        ('AP', 'Andhra Pradesh'),
        ('AR', 'Arunachal Pradesh'),
        ('AS', 'Assam'),
        ('BR', 'Bihar'),
        ('CH', 'Chandigarh'),
        ('CG', 'Chhattisgarh'),
        ('DH', 'Dadra and Nagar Haveli'),
        ('DD', 'Daman and Diu'),
        ('DL', 'Delhi'),
        ('GA', 'Goa'),
        ('GJ', 'Gujarat'),
        ('HR', 'Haryana'),
        ('HP', 'Himachal Pradesh'),
        ('JK', 'Jammu and Kashmir'),
        ('JH', 'Jharkhand'),
        ('KA', 'Karnataka'),
        ('KL', 'Kerala'),
        ('LD', 'Lakshadweep'),
        ('MP', 'Madhya Pradesh'),
        ('MH', 'Maharashtra'),
        ('MN', 'Manipur'),
        ('ML', 'Meghalaya'),
        ('MZ', 'Mizoram'),
        ('NL', 'Nagaland'),
        ('OR', 'Odisha'),
        ('PY', 'Puducherry'),
        ('PB', 'Punjab'),
        ('RJ', 'Rajasthan'),
        ('SK', 'Sikkim'),
        ('TN', 'Tamil Nadu'),
        ('TS', 'Telangana'),
        ('TR', 'Tripura'),
        ('UP', 'Uttar Pradesh'),
        ('UK', 'Uttarakhand'),
        ('WB', 'West Bengal'),
    ]
    
    # vehicle_id is automatically handled by Django's id field
    license_plate = models.CharField(max_length=20, unique=True)  # plateno
    owner_contact = models.CharField(max_length=15, blank=True)  # owner_no (contact)
    vehicle_type = models.CharField(max_length=20, choices=VEHICLE_TYPES)  # vehicle_type
    registered_state = models.CharField(max_length=2, choices=STATE_CHOICES, blank=True)  # registered_state
    contact_number = models.CharField(max_length=15, blank=True)  # contact_no
    created_at = models.DateTimeField(auto_now_add=True)  # created_at
    
    # Keep existing fields for compatibility  
    owner_name = models.CharField(max_length=100, blank=True)
    
    def __str__(self):
        return f"{self.license_plate} - {self.get_vehicle_type_display()}"

class ParkingSlot(models.Model):
    SLOT_TYPES = [
        ('two_wheeler', 'Two Wheeler'),
        ('car', 'Cars'),
        ('large', 'Large Vehicle'),
        ('disabled', 'Disabled'),
        ('vip', 'VIP/Reserved'),
    ]
    
    STATUS_CHOICES = [
        ('available', 'Available'),
        ('occupied', 'Occupied'),
        ('maintenance', 'Under Maintenance'),
        ('out_of_service', 'Out of Service'),
    ]
    
    # slot_id is automatically handled by Django's id field
    slot_number = models.CharField(max_length=10, unique=True)  # slotno
    slot_type = models.CharField(max_length=20, choices=SLOT_TYPES)  # slottype
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='available')  # slot status
    is_occupied = models.BooleanField(default=False)  # isoccupied - kept for backward compatibility
    last_updated = models.DateTimeField(auto_now=True)  # lastupdated
    
    # Keep existing fields for compatibility
    floor_level = models.IntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Slot {self.slot_number} - {self.get_slot_type_display()} ({self.get_status_display()})"
    
    def save(self, *args, **kwargs):
        # Sync is_occupied with status for backward compatibility
        self.is_occupied = (self.status == 'occupied')
        super().save(*args, **kwargs)
    
    @property
    def is_available_for_parking(self):
        """Check if slot can accept new vehicles"""
        return self.status == 'available'

class ParkingSession(models.Model):
    STATUS_CHOICES = [
        ('Active', 'Active'),
        ('Completed', 'Completed'),
    ]
    
    # session_id is automatically handled by Django's id field
    vehicle = models.ForeignKey(Vehicle, on_delete=models.CASCADE)  # vehicle_id (FK)
    parking_slot = models.ForeignKey(ParkingSlot, on_delete=models.CASCADE)  # slot_id (FK)
    entry_time = models.DateTimeField()  # entrytime
    exit_time = models.DateTimeField(null=True, blank=True)  # exittime
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Active')  # status
    
    # Keep existing field for compatibility
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.vehicle.license_plate} in {self.parking_slot.slot_number}"
    
    @property
    def duration(self):
        if self.exit_time:
            return self.exit_time - self.entry_time
        return timezone.now() - self.entry_time


class LargeVehicleRequest(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
    ]

    license_plate = models.CharField(max_length=20)
    owner_name = models.CharField(max_length=100, blank=True)
    contact_number = models.CharField(max_length=15)
    registered_state = models.CharField(max_length=2, choices=Vehicle.STATE_CHOICES, blank=True)
    requested_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-requested_at']
        verbose_name = 'Large Vehicle Request'
        verbose_name_plural = 'Large Vehicle Requests'

    def __str__(self):
        return f"{self.license_plate} - {self.get_status_display()}"
