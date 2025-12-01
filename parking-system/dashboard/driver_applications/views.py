from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q
from django.utils import timezone
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import DriverApplication, ApplicationComment, ApplicationStatusHistory
from .forms import DriverApplicationForm, ApplicationSearchForm, ApplicationCommentForm, ReviewForm
from django.http import HttpResponse, Http404
import mimetypes

@login_required
def apply_entry(request):
    """Public form for drivers to apply for entry"""
    if request.method == 'POST':
        form = DriverApplicationForm(request.POST, request.FILES)
        if form.is_valid():
            # STEP 1: Save form to create database record (without blobs first)
            application = form.save(commit=True)
            
            # STEP 2: Read uploaded files and update blob fields
            file_fields = {
                'driver_photo': 'driver_photo_blob',
                'driver_license_photo': 'driver_license_photo_blob',
                'vehicle_rc': 'vehicle_rc_blob',
                'vehicle_insurance': 'vehicle_insurance_blob',
                'puc_certificate': 'puc_certificate_blob',
                'material_receipt': 'material_receipt_blob',
                'goods_transport_permit': 'goods_transport_permit_blob',
                'customs_clearance': 'customs_clearance_blob',
            }
            
            files_updated = False
            for field_name, blob_field_name in file_fields.items():
                uploaded_file = request.FILES.get(field_name)
                if uploaded_file:
                    try:
                        # Read file bytes
                        file_data = uploaded_file.read()
                        if file_data:
                            # Store bytes in blob field
                            setattr(application, blob_field_name, file_data)
                            
                            # Store filename in name field
                            name_field = f"{field_name}_name"
                            setattr(application, name_field, uploaded_file.name)
                            
                            files_updated = True
                    except Exception as e:
                        messages.warning(request, f"Error saving {field_name}: {e}")
            
            # STEP 3: Save again to persist blob data
            if files_updated:
                application.save()
            
            messages.success(
                request, 
                f'Application submitted successfully! Your application ID is: {application.application_id}. '
                f'Please save this ID for tracking your application status.'
            )
            return redirect('driver_applications:application_status', application_id=application.application_id)
    else:
        form = DriverApplicationForm()
    
    return render(request, 'driver_applications/apply_entry.html', {
        'form': form,
        'material_types': DriverApplication.MATERIAL_TYPE_CHOICES
    })

@login_required
def application_status(request, application_id):
    """Public page for checking application status"""
    application = get_object_or_404(DriverApplication, application_id=application_id)
    
    # Get public comments (non-internal)
    public_comments = application.comments.filter(is_internal=False)
    
    return render(request, 'driver_applications/application_status.html', {
        'application': application,
        'comments': public_comments
    })


def serve_document(request, application_id, doc_name):
    """Stream document from database"""
    application = get_object_or_404(DriverApplication, application_id=application_id)
    
    # Map document names to blob fields
    mapping = {
        'driver_photo': 'driver_photo_blob',
        'driver_license_photo': 'driver_license_photo_blob',
        'vehicle_rc': 'vehicle_rc_blob',
        'vehicle_insurance': 'vehicle_insurance_blob',
        'puc_certificate': 'puc_certificate_blob',
        'material_receipt': 'material_receipt_blob',
        'goods_transport_permit': 'goods_transport_permit_blob',
        'customs_clearance': 'customs_clearance_blob',
    }
    
    if doc_name not in mapping:
        raise Http404('Unknown document type')
    
    blob_field = mapping[doc_name]
    blob_data = getattr(application, blob_field, None)
    
    if not blob_data:
        raise Http404('Document not found')
    
    # Get filename for MIME type detection
    name_field = f"{doc_name}_name"
    filename = getattr(application, name_field, None)
    
    # Guess MIME type
    content_type = 'application/octet-stream'
    if filename:
        guessed_type, _ = mimetypes.guess_type(filename)
        if guessed_type:
            content_type = guessed_type
    else:
        # Fallback: detect from blob header
        if blob_data[:2] == b'\xff\xd8':
            content_type = 'image/jpeg'
        elif blob_data[:4] == b'\x89PNG':
            content_type = 'image/png'
        elif blob_data[:4] == b'%PDF':
            content_type = 'application/pdf'
    
    # Create response
    response = HttpResponse(blob_data, content_type=content_type)
    
    # Set disposition (inline for images/PDFs, attachment for others)
    disposition = 'inline' if content_type.startswith('image/') or content_type == 'application/pdf' else 'attachment'
    display_name = filename if filename else f"{application.application_id}_{doc_name}"
    response['Content-Disposition'] = f'{disposition}; filename="{display_name}"'
    
    return response

@login_required
def applications_dashboard(request):
    """Admin dashboard for managing applications"""
    search_form = ApplicationSearchForm(request.GET)
    applications = DriverApplication.objects.all()
    
    # Apply filters
    if search_form.is_valid():
        query = search_form.cleaned_data.get('search_query')
        if query:
            applications = applications.filter(
                Q(driver_name__icontains=query) |
                Q(vehicle_number__icontains=query) |
                Q(source_company_name__icontains=query) |
                Q(driver_phone__icontains=query)
            )
        
        status = search_form.cleaned_data.get('status')
        if status:
            applications = applications.filter(status=status)
        
        material_type = search_form.cleaned_data.get('material_type')
        if material_type:
            applications = applications.filter(material_type=material_type)
        
        entry_from = search_form.cleaned_data.get('entry_date_from')
        if entry_from:
            applications = applications.filter(requested_entry_date__gte=entry_from)
        
        entry_to = search_form.cleaned_data.get('entry_date_to')
        if entry_to:
            applications = applications.filter(requested_entry_date__lte=entry_to)
    
    # Pagination
    paginator = Paginator(applications, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Statistics
    stats = {
        'total': DriverApplication.objects.count(),
        'pending': DriverApplication.objects.filter(status='pending').count(),
        'under_review': DriverApplication.objects.filter(status='under_review').count(),
        'approved': DriverApplication.objects.filter(status='approved').count(),
        'rejected': DriverApplication.objects.filter(status='rejected').count(),
        'today_applications': DriverApplication.objects.filter(
            created_at__date=timezone.now().date()
        ).count(),
    }
    
    return render(request, 'driver_applications/dashboard.html', {
        'applications': page_obj,
        'search_form': search_form,
        'stats': stats
    })

@login_required
def bulk_action(request):
    """Handle bulk actions on applications"""
    if request.method == 'POST':
        action = request.POST.get('action')
        application_ids = request.POST.getlist('application_ids')
        
        if not application_ids:
            messages.error(request, 'No applications selected.')
            return redirect('applications_dashboard')
        
        applications = DriverApplication.objects.filter(application_id__in=application_ids)
        
        if action == 'approve':
            updated = applications.filter(status='pending').update(
                status='approved',
                reviewed_by=request.user.username,
                reviewed_at=timezone.now()
            )
            messages.success(request, f'{updated} applications approved.')
        
        elif action == 'reject':
            updated = applications.filter(status='pending').update(
                status='rejected',
                reviewed_by=request.user.username,
                reviewed_at=timezone.now()
            )
            messages.success(request, f'{updated} applications rejected.')
        
        elif action == 'under_review':
            updated = applications.filter(status='pending').update(
                status='under_review',
                reviewed_by=request.user.username,
                reviewed_at=timezone.now()
            )
            messages.success(request, f'{updated} applications marked under review.')
    
    return redirect('applications_dashboard')

@login_required
def api_application_stats(request):
    """API endpoint for application statistics"""
    stats = {
        'total_applications': DriverApplication.objects.count(),
        'pending': DriverApplication.objects.filter(status='pending').count(),
        'approved': DriverApplication.objects.filter(status='approved').count(),
        'rejected': DriverApplication.objects.filter(status='rejected').count(),
        'today_count': DriverApplication.objects.filter(
            created_at__date=timezone.now().date()
        ).count(),
    }
    return JsonResponse(stats)


@login_required
def application_detail(request, application_id):
    """Detailed view for reviewing individual applications"""
    application = get_object_or_404(DriverApplication, application_id=application_id)
    comments = ApplicationComment.objects.filter(application=application).order_by('-created_at')
    status_history = ApplicationStatusHistory.objects.filter(application=application).order_by('-changed_at')
    
    # Check if status is final (approved/rejected) - prevent further changes
    is_final_status = application.status in ['approved', 'rejected']
    
    # Initialize forms
    comment_form = ApplicationCommentForm()
    review_form = ReviewForm(instance=application)
    
    if request.method == 'POST':
        if 'comment_form' in request.POST:
            # Handle comment submission
            comment_form = ApplicationCommentForm(request.POST)
            if comment_form.is_valid():
                comment = comment_form.save(commit=False)
                comment.application = application
                comment.comment_by = request.user.username
                comment.save()
                messages.success(request, 'Comment added successfully.')
                return redirect('driver_applications:application_detail', application_id=application_id)
        
        elif 'review_form' in request.POST:
            # Prevent status changes if already approved or rejected
            if is_final_status:
                messages.error(request, 'Cannot modify application - it has already been approved or rejected.')
                return redirect('driver_applications:application_detail', application_id=application_id)
            
            # Handle status update
            old_status = application.status
            review_form = ReviewForm(request.POST, instance=application)
            
            if review_form.is_valid():
                application = review_form.save(commit=False)
                application.reviewed_by = request.user.username
                application.reviewed_at = timezone.now()
                
                # Save rejection reason if status is rejected
                if application.status == 'rejected':
                    application.rejection_reason = review_form.cleaned_data.get('rejection_reason', '')
                
                # Save admin comments
                application.admin_comments = review_form.cleaned_data.get('admin_comments', '')
                
                application.save()
                
                # Create status history entry if status changed
                if old_status != application.status:
                    ApplicationStatusHistory.objects.create(
                        application=application,
                        previous_status=old_status,
                        new_status=application.status,
                        changed_by=request.user.username,
                        reason=review_form.cleaned_data.get('admin_comments', '')
                    )
                
                # Add comment for status change
                if review_form.cleaned_data.get('admin_comments'):
                    ApplicationComment.objects.create(
                        application=application,
                        comment_by=request.user.username,
                        comment_text=review_form.cleaned_data['admin_comments'],
                        is_internal=True
                    )
                
                status_msg = f"Application {application.get_status_display().lower()}"
                messages.success(request, f'Application updated successfully - {status_msg}.')
                return redirect('driver_applications:application_detail', application_id=application_id)
    
    context = {
        'application': application,
        'comments': comments,
        'status_history': status_history,
        'comment_form': comment_form,
        'review_form': review_form,
        'is_final_status': is_final_status,
        'title': f'Review Application - {application.driver_name}',
    }
    
    return render(request, 'driver_applications/application_detail.html', context)
