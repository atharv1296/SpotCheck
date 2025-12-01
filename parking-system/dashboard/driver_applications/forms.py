from django import forms
from django.core.exceptions import ValidationError
from django.utils import timezone
from datetime import date, timedelta
from .models import DriverApplication, ApplicationComment

class DriverApplicationForm(forms.ModelForm):
    # Keep file inputs on the form, but handle saving bytes in the view.
    driver_photo = forms.FileField(required=True)
    driver_license_photo = forms.FileField(required=False)
    vehicle_rc = forms.FileField(required=True)
    vehicle_insurance = forms.FileField(required=False)
    puc_certificate = forms.FileField(required=False)
    material_receipt = forms.FileField(required=True)
    goods_transport_permit = forms.FileField(required=False)
    customs_clearance = forms.FileField(required=False)

    class Meta:
        model = DriverApplication
        # remove the FileField model attributes from the model form fields
        fields = [
            'requested_entry_date', 'requested_entry_time', 'estimated_duration',
            'driver_name', 'driver_phone', 'driver_email', 'driver_license_number',
            'vehicle_number', 'vehicle_model', 'vehicle_capacity',
            'source_company_name', 'source_company_address', 'source_company_contact', 
            'destination_within_premises',
            'material_type', 'material_description', 'material_weight'
        ]
        
        widgets = {
            'requested_entry_date': forms.DateInput(attrs={
                'type': 'date', 
                'class': 'form-control',
                'min': date.today().isoformat()
            }),
            'requested_entry_time': forms.TimeInput(attrs={
                'type': 'time', 
                'class': 'form-control'
            }),
            'estimated_duration': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '30',
                'max': '720',
                'placeholder': 'Duration in minutes (30-720)'
            }),
            'driver_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Full name as per license'
            }),
            'driver_phone': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '+91XXXXXXXXXX'
            }),
            'driver_email': forms.EmailInput(attrs={
                'class': 'form-control',
                'placeholder': 'driver@example.com'
            }),
            'driver_license_number': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'DL number'
            }),
            'driver_photo': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'
            }),
            'driver_license_photo': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*,.pdf'
            }),
            'vehicle_number': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'MH12AB1234',
                'style': 'text-transform: uppercase;'
            }),
            'vehicle_model': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., Tata LPT 1612'
            }),
            'vehicle_capacity': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., 10 tons, 25 passengers'
            }),
            'source_company_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Company sending the vehicle'
            }),
            'source_company_address': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Complete address of source company'
            }),
            'source_company_contact': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '+91XXXXXXXXXX'
            }),
            'destination_within_premises': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., Warehouse Block A, Main Reception'
            }),
            'material_type': forms.Select(attrs={'class': 'form-control'}),
            'material_description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Detailed description of materials/goods being transported'
            }),
            'material_weight': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., 5 tons, 500 kg'
            }),
            'vehicle_rc': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.pdf,.jpg,.jpeg,.png'
            }),
            'vehicle_insurance': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.pdf,.jpg,.jpeg,.png'
            }),
            'puc_certificate': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.pdf,.jpg,.jpeg,.png'
            }),
            'material_receipt': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.pdf,.jpg,.jpeg,.png'
            }),
            'goods_transport_permit': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.pdf,.jpg,.jpeg,.png'
            }),
            'customs_clearance': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.pdf,.jpg,.jpeg,.png'
            }),
        }
    
    def clean_requested_entry_date(self):
        entry_date = self.cleaned_data['requested_entry_date']
        if entry_date < date.today():
            raise ValidationError("Entry date cannot be in the past.")
        if entry_date > date.today() + timedelta(days=30):
            raise ValidationError("Entry date cannot be more than 30 days in advance.")
        return entry_date
    
    def clean_vehicle_number(self):
        vehicle_number = self.cleaned_data['vehicle_number'].upper()
        return vehicle_number
    
    def clean_estimated_duration(self):
        duration = self.cleaned_data['estimated_duration']
        if duration < 30:
            raise ValidationError("Minimum duration is 30 minutes.")
        if duration > 720:  # 12 hours
            raise ValidationError("Maximum duration is 12 hours (720 minutes).")
        return duration
    
    def save(self, commit=True):
        instance = super().save(commit=False)
        # Set default values for removed fields
        instance.urgency = 'medium'  # Default priority
        instance.material_value = None  # No value specified
        if commit:
            instance.save()
        return instance

class ApplicationReviewForm(forms.ModelForm):
    """Form for admin to review and update application status"""
    class Meta:
        model = DriverApplication
        fields = ['status', 'admin_comments', 'rejection_reason', 'approved_entry_slot']
        
        widgets = {
            'status': forms.Select(attrs={'class': 'form-control'}),
            'admin_comments': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Internal comments about the application...'
            }),
            'rejection_reason': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Reason for rejection (visible to applicant)...'
            }),
            'approved_entry_slot': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., L1, L2, L3'
            }),
        }

class ApplicationCommentForm(forms.ModelForm):
    """Form for adding comments to applications"""
    class Meta:
        model = ApplicationComment
        fields = ['comment_text', 'is_internal']
        
        widgets = {
            'comment_text': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Add your comment...'
            }),
            'is_internal': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

class ApplicationSearchForm(forms.Form):
    """Form for searching and filtering applications"""
    search_query = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search by driver name, vehicle number, company...'
        })
    )
    
    status = forms.ChoiceField(
        required=False,
        choices=[('', 'All Status')] + DriverApplication.APPLICATION_STATUS_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    material_type = forms.ChoiceField(
        required=False,
        choices=[('', 'All Materials')] + DriverApplication.MATERIAL_TYPE_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    entry_date_from = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'type': 'date',
            'class': 'form-control'
        })
    )
    
    entry_date_to = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'type': 'date',
            'class': 'form-control'
        })
    )


class ReviewForm(forms.ModelForm):
    """Form for reviewing and updating application status"""
    admin_comments = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Internal admin notes...'
        }),
        label='Admin Comments'
    )
    
    rejection_reason = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Reason for rejection (visible to applicant)...'
        }),
        label='Rejection Reason'
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Disable form if status is final (approved/rejected)
        if self.instance and self.instance.status in ['approved', 'rejected']:
            for field in self.fields:
                self.fields[field].disabled = True
                self.fields[field].help_text = 'Status is final - no changes allowed'
    
    class Meta:
        model = DriverApplication
        fields = ['status', 'approved_entry_slot', 'rejection_reason', 'admin_comments']
        widgets = {
            'status': forms.Select(attrs={
                'class': 'form-select'
            }),
            'approved_entry_slot': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., Slot A-12, Loading Bay 2'
            })
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['status'].choices = [
            ('pending', 'Pending Review'),
            ('under_review', 'Under Review'),
            ('approved', 'Approved'),  
            ('rejected', 'Rejected'),
            ('expired', 'Expired')
        ]