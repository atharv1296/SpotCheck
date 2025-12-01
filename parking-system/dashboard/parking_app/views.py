from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from .models import Vehicle, ParkingSlot, ParkingSession, LargeVehicleRequest
from django.db import transaction, connection
import json
from datetime import datetime, timedelta
from django.utils.timesince import timesince
from django.http import HttpResponse
import csv
from io import BytesIO

try:
    # ReportLab used for basic PDF generation
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

@login_required
def dashboard(request):
    """Forbes Marshall Main Dashboard - Live Oracle Data"""
    # Get real-time data from Oracle database
    total_slots = ParkingSlot.objects.count()
    occupied_slots = ParkingSlot.objects.filter(status='occupied').count()
    available_slots = ParkingSlot.objects.filter(status='available').count()
    maintenance_slots = ParkingSlot.objects.filter(status='maintenance').count()
    
    context = {
        'total_slots': total_slots,
        'occupied_slots': occupied_slots,
        'available_slots': available_slots,
        'maintenance_slots': maintenance_slots,
        'active_sessions': ParkingSession.objects.filter(is_active=True).count(),
        'total_vehicles': Vehicle.objects.count(),
        'occupancy_rate': round((occupied_slots / total_slots) * 100, 2) if total_slots > 0 else 0,
    }
    
    # Add driver applications statistics
    from driver_applications.models import DriverApplication
    context.update({
        'pending_applications': DriverApplication.objects.filter(status='pending').count(),
        'total_applications': DriverApplication.objects.count(),
        'today_applications': DriverApplication.objects.filter(
            created_at__date=timezone.now().date()
        ).count(),
    })
    
    # Add large vehicle requests summary
    try:
        lvr_pending = LargeVehicleRequest.objects.filter(status='pending').count()
        lvr_total = LargeVehicleRequest.objects.count()
    except Exception:
        lvr_pending = 0
        lvr_total = 0
    context['large_vehicle_requests'] = {
        'pending': lvr_pending,
        'total': lvr_total,
    }

    # Add slot distribution by type (3 main categories)
    slot_types = ['two_wheeler', 'car', 'large']
    slot_distribution = {}
    for slot_type in slot_types:
        slot_distribution[slot_type] = {
            'total': ParkingSlot.objects.filter(slot_type=slot_type).count(),
            'available': ParkingSlot.objects.filter(slot_type=slot_type, status='available').count(),
            'occupied': ParkingSlot.objects.filter(slot_type=slot_type, status='occupied').count(),
            'maintenance': ParkingSlot.objects.filter(slot_type=slot_type, status='maintenance').count(),
        }
    context['slot_distribution'] = slot_distribution
    
    # Add recent activity
    context['recent_sessions'] = ParkingSession.objects.order_by('-entry_time')[:5]
    context['recent_vehicles'] = Vehicle.objects.order_by('-created_at')[:5]
    
    return render(request, 'dashboard/professional_dashboard.html', context)

@login_required
def realtime_view(request):
    """Real-time monitoring page - Live Oracle Data"""
    # Get real-time data from Oracle database
    total_slots = ParkingSlot.objects.count()
    occupied_slots = ParkingSlot.objects.filter(status='occupied').count()
    available_slots = ParkingSlot.objects.filter(status='available').count()
    maintenance_slots = ParkingSlot.objects.filter(status='maintenance').count()
    
    context = {
        'total_slots': total_slots,
        'occupied_slots': occupied_slots,
        'available_slots': available_slots,
        'occupancy_rate': round((occupied_slots / total_slots) * 100, 2) if total_slots > 0 else 0,
        'active_sessions': ParkingSession.objects.filter(is_active=True)[:10],
        'recent_entries': ParkingSession.objects.order_by('-entry_time')[:10],
        'recent_exits': ParkingSession.objects.filter(is_active=False).order_by('-exit_time')[:10],
    }
    return render(request, 'dashboard/professional_realtime.html', context)

@login_required
def analytics_view(request):
    """Analytics dashboard for Forbes Marshall"""
    # Get date range from request
    date_range = request.GET.get('range', 'today')
    from_date = request.GET.get('from_date')
    to_date = request.GET.get('to_date')
    
    # Default to today if no dates provided
    today = timezone.now().date()
    if not from_date or not to_date:
        if date_range == 'week':
            from_date = today - timedelta(days=7)
        elif date_range == 'month':
            from_date = today - timedelta(days=30)
        else:
            from_date = today
        to_date = today
    
    # Calculate initial analytics data
    sessions = ParkingSession.objects.filter(
        entry_time__date__range=[from_date, to_date]
    )
    
    # Calculate utilization rate for the period
    total_slots = ParkingSlot.objects.count()
    if sessions.count() > 0 and total_slots > 0:
        avg_utilization = round(sum([
            (sessions.filter(entry_time__date=from_date + timedelta(days=i)).count() / total_slots) * 100 
            for i in range((to_date - from_date).days + 1)
        ]) / ((to_date - from_date).days + 1), 1)
    else:
        avg_utilization = calculate_utilization_rate()
    
    context = {
        'total_sessions': sessions.count(),
        'avg_duration': calculate_avg_duration(sessions),
        'utilization_rate': avg_utilization,
        'active_sessions': ParkingSession.objects.filter(is_active=True).count(),
        'from_date': from_date,
        'to_date': to_date,
        'date_range': date_range,
    }
    return render(request, 'dashboard/professional_analytics.html', context)

@login_required
def get_parking_data(request):
    """API endpoint for parking data - Live Oracle Data"""
    slots = ParkingSlot.objects.all().order_by('slot_number')
    data = []
    
    for slot in slots:
        slot_data = {
            'slot_number': slot.slot_number,
            'slot_type': slot.slot_type,
            'status': slot.status,  # Use actual status field (available/occupied/maintenance/out_of_service)
            'is_occupied': slot.is_occupied,
            'last_updated': slot.last_updated.isoformat() if slot.last_updated else None,
            'floor_level': slot.floor_level,
            'created_at': slot.created_at.isoformat() if slot.created_at else None,
        }
        
        # Add vehicle info if occupied
        if slot.is_occupied:
            try:
                # Get the most recent active session for this slot
                session = ParkingSession.objects.filter(
                    parking_slot=slot, 
                    is_active=True
                ).order_by('-entry_time').first()
                
                if session:
                    # Use contact_number or owner_contact, whichever has data
                    contact = session.vehicle.contact_number or session.vehicle.owner_contact or 'N/A'
                    
                    slot_data['vehicle'] = {
                        'license_plate': session.vehicle.license_plate,
                        'vehicle_type': session.vehicle.vehicle_type,
                        'owner_name': session.vehicle.owner_name or 'Unknown',
                        'owner_contact': contact,
                        'registered_state': session.vehicle.registered_state or 'N/A',
                        'entry_time': session.entry_time.isoformat(),
                        'duration': str(session.duration) if session.duration else 'Active',
                    }
            except Exception as e:
                # Slot marked occupied but no active session found
                slot_data['vehicle'] = {
                    'license_plate': 'Unknown',
                    'vehicle_type': 'Unknown',
                    'owner_name': 'Unknown',
                    'entry_time': None,
                    'duration': 'Unknown',
                }
        
        data.append(slot_data)
    
    # Add summary statistics
    total_slots = len(data)
    occupied_count = len([s for s in data if s['status'] == 'occupied'])
    available_count = len([s for s in data if s['status'] == 'available'])
    maintenance_count = len([s for s in data if s['status'] == 'maintenance'])
    
    return JsonResponse({
        'slots': data,
        'summary': {
            'total_slots': total_slots,
            'occupied_slots': occupied_count,
            'available_slots': available_count,
            'maintenance_slots': maintenance_count,
            'occupancy_rate': round((occupied_count / total_slots) * 100, 2) if total_slots > 0 else 0
        },
        'slot_distribution': {
            'two_wheeler': {
                'total': len([s for s in data if s['slot_type'] == 'two_wheeler']),
                'available': len([s for s in data if s['slot_type'] == 'two_wheeler' and s['status'] == 'available']),
                'occupied': len([s for s in data if s['slot_type'] == 'two_wheeler' and s['status'] == 'occupied']),
                'maintenance': len([s for s in data if s['slot_type'] == 'two_wheeler' and s['status'] == 'maintenance']),
            },
            'car': {
                'total': len([s for s in data if s['slot_type'] == 'car']),
                'available': len([s for s in data if s['slot_type'] == 'car' and s['status'] == 'available']),
                'occupied': len([s for s in data if s['slot_type'] == 'car' and s['status'] == 'occupied']),
                'maintenance': len([s for s in data if s['slot_type'] == 'car' and s['status'] == 'maintenance']),
            },
            'large': {
                'total': len([s for s in data if s['slot_type'] == 'large']),
                'available': len([s for s in data if s['slot_type'] == 'large' and s['status'] == 'available']),
                'occupied': len([s for s in data if s['slot_type'] == 'large' and s['status'] == 'occupied']),
                'maintenance': len([s for s in data if s['slot_type'] == 'large' and s['status'] == 'maintenance']),
            }
        }
    })


@login_required
def get_analytics_export(request):
    """Export analytics data as CSV or PDF. URL: /api/analytics/export/"""
    fmt = request.GET.get('format', 'csv').lower()
    types = request.GET.get('types', 'sessions')
    date_range = request.GET.get('range', 'today')
    start_date_param = request.GET.get('from_date') or request.GET.get('start_date')
    end_date_param = request.GET.get('to_date') or request.GET.get('end_date')

    # Determine date range same as get_analytics_data
    today = timezone.now().date()
    if start_date_param and end_date_param:
        try:
            start_date = datetime.strptime(start_date_param, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date_param, '%Y-%m-%d').date()
        except ValueError:
            start_date = today
            end_date = today
    else:
        if date_range == 'week':
            start_date = today - timedelta(days=7)
        elif date_range == 'month':
            start_date = today - timedelta(days=30)
        elif date_range == 'quarter':
            start_date = today - timedelta(days=90)
        elif date_range == 'year':
            start_date = today - timedelta(days=365)
        else:
            start_date = today
        end_date = today

    sessions = ParkingSession.objects.filter(entry_time__date__range=[start_date, end_date]).order_by('-entry_time')

    # Prepare rows for export: vehicle, slot, vehicle_type, entry_time, exit_time, duration, status
    rows = []
    rows.append(['Vehicle Number', 'Slot', 'Vehicle Type', 'Entry Time', 'Exit Time', 'Duration', 'Status'])
    for s in sessions:
        entry = s.entry_time.strftime('%Y-%m-%d %H:%M:%S') if s.entry_time else ''
        exit_time = s.exit_time.strftime('%Y-%m-%d %H:%M:%S') if s.exit_time else ''
        duration = str(s.duration).split('.')[0] if s.duration else ('Active' if s.is_active else '')
        vtype = s.vehicle.get_vehicle_type_display() if hasattr(s, 'vehicle') else getattr(s, 'vehicle_type', '')
        rows.append([s.vehicle.license_plate if s.vehicle else '', s.parking_slot.slot_number if s.parking_slot else '', vtype, entry, exit_time, duration, 'Active' if s.is_active else 'Completed'])

    # Return CSV
    if fmt == 'csv' or fmt == 'excel':
        output = BytesIO()
        writer = csv.writer(output)
        for row in rows:
            writer.writerow(row)
        content = output.getvalue()
        response = HttpResponse(content, content_type='text/csv')
        filename = f"analytics_{date_range}_{start_date.isoformat()}_{end_date.isoformat()}.csv"
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response

    # PDF fallback
    if fmt == 'pdf':
        if not REPORTLAB_AVAILABLE:
            return JsonResponse({'status': 'error', 'error': 'PDF export not available: reportlab not installed'}, status=500)

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        title = Paragraph(f"Analytics Export: {date_range.title()} ({start_date.isoformat()} - {end_date.isoformat()})", styles['Heading2'])
        elements.append(title)
        elements.append(Spacer(1, 12))

        # Create table
        table = Table(rows, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#003366')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN',(0,0),(-1,-1),'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1, -1), 8),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ]))

        elements.append(table)
        doc.build(elements)
        pdf = buffer.getvalue()
        buffer.close()

        response = HttpResponse(pdf, content_type='application/pdf')
        filename = f"analytics_{date_range}_{start_date.isoformat()}_{end_date.isoformat()}.pdf"
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response

    return JsonResponse({'status': 'error', 'error': 'Unsupported format'}, status=400)

@login_required
@csrf_exempt
def update_slot_status(request):
    """Update parking slot status - Live Oracle Updates (includes maintenance)"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            slot_number = (data.get('slot_number') or '').strip()
            new_status = data.get('status')  # 'occupied', 'available', 'maintenance', 'out_of_service'
            is_occupied = data.get('is_occupied', new_status == 'occupied')
            vehicle_plate = (data.get('vehicle_plate') or '').strip().upper()
            vehicle_type = (data.get('vehicle_type') or '').strip().lower() or None
            owner_name = (data.get('owner_name') or '').strip() or 'Walk-in Customer'
            contact_number = (data.get('contact_number') or '').strip()
            registered_state = (data.get('registered_state') or '').strip().upper()
            
            # Validate input
            if not slot_number and not is_occupied:
                # For release operations, slot number is mandatory
                return JsonResponse({
                    'success': False,
                    'error': 'Slot number required when releasing'
                })
            
            # Determine or obtain target slot
            slot = None
            old_occupied = None
            if slot_number:
                # Explicit slot update
                slot = ParkingSlot.objects.get(slot_number=slot_number)
                old_occupied = slot.is_occupied
            else:
                # Auto-assign path when occupying without a provided slot number
                if not is_occupied:
                    return JsonResponse({'success': False, 'error': 'Slot number required when releasing'})

                # Map requested vehicle_type to a slot_type
                vt_lower = (vehicle_type or '').lower()
                if vt_lower in {'two_wheeler'}:
                    desired_slot_type = 'two_wheeler'
                elif vt_lower in {'car', 'sedan', 'suv'}:
                    desired_slot_type = 'car'
                elif vt_lower == 'large':
                    desired_slot_type = 'large'
                else:
                    # Fallback to standard car slot
                    desired_slot_type = 'car'

                with transaction.atomic():
                    table = ParkingSlot._meta.db_table
                    # 1) Select exactly one free slot with row lock (exclude maintenance/out_of_service)
                    # Note: ROWNUM=1 limits to a single row and avoids locking the entire set.
                    select_sql = f"""
                        SELECT id, slot_number
                        FROM {table}
                        WHERE slot_type = :slot_type 
                          AND status = 'available'
                          AND ROWNUM = 1
                        FOR UPDATE SKIP LOCKED
                    """
                    with connection.cursor() as cursor:
                        cursor.execute(select_sql, {'slot_type': desired_slot_type})
                        row = cursor.fetchone()
                        if not row:
                            return JsonResponse({'success': False, 'error': f'No available {desired_slot_type} slots right now'})
                        assigned_id, assigned_slot_number = row

                        # 2) Mark it occupied and update timestamp
                        update_sql = f"UPDATE {table} SET status = 'occupied', is_occupied = 1, last_updated = :now WHERE id = :id"
                        cursor.execute(update_sql, {'now': timezone.now(), 'id': assigned_id})

                    slot = ParkingSlot.objects.get(id=assigned_id)
                    old_occupied = False
            
            # Update slot occupation status (sync both status and is_occupied)
            if is_occupied:
                slot.status = 'occupied'
                slot.is_occupied = True
            else:
                # Only set to available if not in maintenance or out_of_service
                if slot.status == 'occupied':
                    slot.status = 'available'
                slot.is_occupied = False
            slot.save()
            
            old_status_text = 'occupied' if old_occupied else 'available'
            new_status_text = 'occupied' if is_occupied else 'available'
            
            # Handle parking session logic
            if is_occupied and not old_occupied:
                # Vehicle entering - create new session
                if vehicle_plate:
                    # Check for approved large vehicle request first
                    large_request = LargeVehicleRequest.objects.filter(
                        license_plate=vehicle_plate,
                        status='approved'
                    ).first()

                    if large_request:
                        # If an approved request exists, ensure vehicle type is 'large'
                        # This will influence slot selection if no slot number is provided
                        vehicle_type = 'large'
                        print(f"INFO: Approved large vehicle request found for {vehicle_plate}. Prioritizing large slot.")

                    try:
                        vehicle = Vehicle.objects.get(license_plate=vehicle_plate)
                    except Vehicle.DoesNotExist:
                        # Determine a valid vehicle_type
                        # Map provided types and fallback from slot type
                        vt = None
                        if vehicle_type in {'two_wheeler', 'sedan', 'suv', 'large'}:
                            vt = vehicle_type
                        elif vehicle_type == 'car':
                            vt = 'sedan'  # normalize 'car' to a valid Vehicle type
                        else:
                            # derive from slot type: two_wheeler -> two_wheeler, car -> sedan, large -> large
                            st = (slot.slot_type or '').lower()
                            vt = 'two_wheeler' if st == 'two_wheeler' else ('large' if st == 'large' else 'sedan')

                        # Only keep registered_state if it matches choices, else blank
                        valid_states = {code for code, _ in Vehicle.STATE_CHOICES}
                        rs = registered_state if registered_state in valid_states else ''

                        # Create new vehicle record
                        vehicle = Vehicle.objects.create(
                            license_plate=vehicle_plate,
                            vehicle_type=vt,
                            owner_name=owner_name,
                                contact_number=contact_number,
                            registered_state=rs,
                        )
                    
                    # PREVENT DUPLICATE SESSIONS: 1) Deactivate any existing active sessions for this slot
                    existing_sessions = ParkingSession.objects.filter(
                        parking_slot=slot,
                        is_active=True
                    )
                    if existing_sessions.exists():
                        print(f"⚠️ WARNING: Found {existing_sessions.count()} active session(s) for slot {slot.slot_number}. Deactivating them.")
                        existing_sessions.update(
                            exit_time=timezone.now(),
                            status='Completed',
                            is_active=False
                        )

                    # 2) Prevent reassigning a vehicle that already has an active session elsewhere.
                    #    Instead of forcibly moving it, revert this slot update and inform the caller.
                    vehicle_active_sessions = ParkingSession.objects.filter(
                        vehicle=vehicle,
                        is_active=True
                    ).exclude(parking_slot=slot)
                    if vehicle_active_sessions.exists():
                        other_slot = vehicle_active_sessions.first().parking_slot.slot_number if vehicle_active_sessions.first().parking_slot else 'Unknown'
                        # Revert the slot status change to avoid marking it occupied without a valid session
                        try:
                            slot.status = 'available'
                            slot.is_occupied = False
                            slot.save()
                        except Exception:
                            # If revert fails, log but continue to return error to caller
                            print(f"Warning: failed to revert slot {slot.slot_number} after detecting vehicle active elsewhere")

                        return JsonResponse({
                            'success': False,
                            'error': f'Vehicle already has an active slot ({other_slot}). Please park there instead of reassigning.'
                        })
                    
                    # Create parking session
                    ParkingSession.objects.create(
                        vehicle=vehicle,
                        parking_slot=slot,
                        entry_time=timezone.now(),
                        status='Active',
                        is_active=True
                    )
                else:
                    return JsonResponse({'success': False, 'error': 'Vehicle plate is required to occupy a slot'})
                    
            elif not is_occupied and old_occupied:
                # Vehicle leaving - end active session(s)
                try:
                    # Handle potential duplicate sessions - end the most recent one
                    session = ParkingSession.objects.filter(
                        parking_slot=slot, 
                        is_active=True
                    ).order_by('-entry_time').first()
                    
                    if session:
                        session.exit_time = timezone.now()
                        session.status = 'Completed'
                        session.is_active = False
                        session.save()
                        
                        # Also end any other duplicate active sessions for this slot
                        ParkingSession.objects.filter(
                            parking_slot=slot, 
                            is_active=True
                        ).exclude(id=session.id).update(
                            exit_time=timezone.now(),
                            status='Completed',
                            is_active=False
                        )
                except Exception as e:
                    # Log but don't fail the slot release
                    print(f"Warning: Error ending session for {slot.slot_number}: {e}")
            
            return JsonResponse({
                'success': True, 
                'message': f'Slot {slot.slot_number} updated from {old_status_text} to {new_status_text}',
                'slot_data': {
                    'slot_number': slot.slot_number,
                    'status': new_status_text,
                    'is_occupied': slot.is_occupied,
                    'slot_type': slot.slot_type
                }
            })
            
        except ParkingSlot.DoesNotExist:
            return JsonResponse({
                'success': False, 
                'error': f'Slot {slot_number} not found'
            })
        except Exception as e:
            return JsonResponse({
                'success': False, 
                'error': f'Error updating slot: {str(e)}'
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
@csrf_exempt
def check_vehicle(request):
    """Check if vehicle exists in database"""
    if request.method == 'GET':
        vehicle_number = request.GET.get('vehicle_number', '').strip().upper()
        
        if not vehicle_number:
            return JsonResponse({'success': False, 'error': 'Vehicle number is required'})
        
        try:
            vehicle = Vehicle.objects.get(license_plate=vehicle_number)
            return JsonResponse({
                'success': True,
                'found': True,
                'vehicle': {
                    'license_plate': vehicle.license_plate,
                    'vehicle_type': vehicle.vehicle_type,
                    'owner_name': vehicle.owner_name,
                    'contact_number': vehicle.contact_number,
                    'registered_state': vehicle.registered_state,
                }
            })
        except Vehicle.DoesNotExist:
            # Vehicle not registered, check for an approved large vehicle application
            large_request = LargeVehicleRequest.objects.filter(
                license_plate=vehicle_number,
                status='approved'
            ).first()

            if large_request:
                # Found an approved application for an unregistered vehicle
                return JsonResponse({
                    'success': True,
                    'found': True,  # Treat as found because we have details
                    'from_application': True,  # Flag to indicate the source is an application
                    'vehicle': {
                        'license_plate': large_request.license_plate,
                        'vehicle_type': 'large',  # From application, it's always large
                        'owner_name': large_request.owner_name,
                        'contact_number': large_request.contact_number,
                        'registered_state': large_request.registered_state,
                    }
                })
            else:
                # No registered vehicle and no approved application
                return JsonResponse({
                    'success': True,
                    'found': False,
                    'message': 'Vehicle not registered and no approved application found'
                })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'Error checking vehicle: {str(e)}'
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
@csrf_exempt
def toggle_maintenance(request):
    """Toggle slot maintenance status - Direct endpoint for gate interface"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            slot_numbers = data.get('slot_numbers', [])  # Can be single or multiple
            action = data.get('action')  # 'maintenance', 'available', 'out_of_service'
            
            if not slot_numbers:
                return JsonResponse({'success': False, 'error': 'No slot numbers provided'})
            
            if not isinstance(slot_numbers, list):
                slot_numbers = [slot_numbers]
            
            # Validate action
            valid_statuses = ['maintenance', 'available', 'out_of_service']
            if action not in valid_statuses:
                return JsonResponse({'success': False, 'error': f'Invalid action. Must be one of: {valid_statuses}'})
            
            updated_slots = []
            failed_slots = []
            
            for slot_num in slot_numbers:
                try:
                    slot = ParkingSlot.objects.get(slot_number=slot_num.strip())
                    
                    # Don't change status of occupied slots to maintenance
                    if slot.status == 'occupied' and action in ['maintenance', 'out_of_service']:
                        failed_slots.append({
                            'slot_number': slot_num,
                            'reason': 'Cannot mark occupied slot as maintenance/out_of_service'
                        })
                        continue
                    
                    old_status = slot.get_status_display()
                    slot.status = action
                    slot.save()
                    
                    updated_slots.append({
                        'slot_number': slot.slot_number,
                        'old_status': old_status,
                        'new_status': slot.get_status_display(),
                        'slot_type': slot.get_slot_type_display()
                    })
                    
                except ParkingSlot.DoesNotExist:
                    failed_slots.append({
                        'slot_number': slot_num,
                        'reason': 'Slot not found'
                    })
            
            return JsonResponse({
                'success': True,
                'updated': len(updated_slots),
                'failed': len(failed_slots),
                'slots': updated_slots,
                'failures': failed_slots,
                'message': f'Updated {len(updated_slots)} slot(s) to {action}'
            })
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    elif request.method == 'GET':
        # Get maintenance status overview
        total = ParkingSlot.objects.count()
        maintenance = ParkingSlot.objects.filter(status='maintenance')
        out_of_service = ParkingSlot.objects.filter(status='out_of_service')
        
        return JsonResponse({
            'total_slots': total,
            'maintenance_count': maintenance.count(),
            'out_of_service_count': out_of_service.count(),
            'maintenance_slots': [{
                'slot_number': s.slot_number,
                'slot_type': s.get_slot_type_display(),
                'floor_level': s.floor_level
            } for s in maintenance],
            'out_of_service_slots': [{
                'slot_number': s.slot_number,
                'slot_type': s.get_slot_type_display(),
                'floor_level': s.floor_level
            } for s in out_of_service]
        })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def get_analytics_data(request):
    """API endpoint for analytics data - Live Oracle Data"""
    date_range = request.GET.get('range', 'today')
    start_date_param = request.GET.get('start_date')
    end_date_param = request.GET.get('end_date')
    
    # Calculate date range
    today = timezone.now().date()
    
    if start_date_param and end_date_param:
        # Custom date range
        try:
            start_date = datetime.strptime(start_date_param, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date_param, '%Y-%m-%d').date()
        except ValueError:
            start_date = today
            end_date = today
    else:
        # Predefined ranges
        if date_range == 'week':
            start_date = today - timedelta(days=7)
        elif date_range == 'month':
            start_date = today - timedelta(days=30)
        elif date_range == 'quarter':
            start_date = today - timedelta(days=90)
        elif date_range == 'year':
            start_date = today - timedelta(days=365)
        else:
            start_date = today
        end_date = today
    
    # Get sessions in date range
    sessions = ParkingSession.objects.filter(
        entry_time__date__range=[start_date, end_date]
    )
    
    # Calculate hourly occupancy 
    total_slots = ParkingSlot.objects.count()
    hourly_data = []
    
    if date_range == 'today' or start_date == end_date:
        # For today/single day, show actual hourly pattern
        for hour in range(24):
            # Count sessions that were active during this hour
            hour_sessions = 0
            for session in sessions:
                entry_hour = session.entry_time.hour
                if session.exit_time:
                    exit_hour = session.exit_time.hour
                    # Session active if entry before/at hour and exit after hour
                    if entry_hour <= hour < exit_hour:
                        hour_sessions += 1
                else:
                    # Still active session
                    if entry_hour <= hour:
                        hour_sessions += 1
            
            occupancy_rate = (hour_sessions / total_slots) * 100 if total_slots > 0 else 0
            hourly_data.append(round(occupancy_rate, 1))
    else:
        # For date ranges, show average occupancy per hour across all days
        for hour in range(24):
            hour_sessions = sessions.filter(entry_time__hour=hour).count()
            occupancy_rate = (hour_sessions / total_slots) * 100 if total_slots > 0 else 0
            hourly_data.append(round(occupancy_rate, 1))
    
    # Vehicle type distribution - normalized labels
    vehicle_distribution = []
    vehicle_type_map = {
        'two_wheeler': 'Two Wheeler',
        'sedan': 'Car',
        'suv': 'Car', 
        'car': 'Car',
        'large': 'Large Vehicle'
    }
    
    type_counts = {}
    if sessions.exists():
        # From session data
        for session in sessions:
            vtype = session.vehicle.vehicle_type
            display_type = vehicle_type_map.get(vtype, vtype)
            type_counts[display_type] = type_counts.get(display_type, 0) + 1
    else:
        # From registered vehicles
        for vehicle in Vehicle.objects.all():
            vtype = vehicle.vehicle_type
            display_type = vehicle_type_map.get(vtype, vtype)
            type_counts[display_type] = type_counts.get(display_type, 0) + 1
    
    for vtype, count in type_counts.items():
        vehicle_distribution.append({
            'type': vtype,
            'count': count
        })
    
    # Slot utilization by type - use status field
    slot_utilization = {}
    slot_type_map = {
        'two_wheeler': 'Two Wheeler',
        'car': 'Car',
        'large': 'Large Vehicle'
    }
    
    slot_types = ParkingSlot.objects.values_list('slot_type', flat=True).distinct()
    for slot_type in slot_types:
        total_type = ParkingSlot.objects.filter(slot_type=slot_type).count()
        occupied_type = ParkingSlot.objects.filter(slot_type=slot_type, status='occupied').count()
        utilization_rate = (occupied_type / total_type) * 100 if total_type > 0 else 0
        display_name = slot_type_map.get(slot_type, slot_type)
        slot_utilization[display_name] = round(utilization_rate, 1)
    
    # Calculate peak hours analysis
    peak_hours_data = []
    for hour in range(24):
        hour_sessions = sessions.filter(entry_time__hour=hour).count()
        peak_hours_data.append({
            'hour': f"{hour:02d}:00",
            'count': hour_sessions
        })
    
    # Calculate duration analysis - Oracle compatible
    completed_sessions = sessions.filter(exit_time__isnull=False)
    
    duration_ranges = {
        '0-1h': 0,
        '1-3h': 0,
        '3-6h': 0,
        '6+h': 0,
    }
    
    for session in completed_sessions:
        if session.duration:
            hours = session.duration.total_seconds() / 3600
            if hours < 1:
                duration_ranges['0-1h'] += 1
            elif hours < 3:
                duration_ranges['1-3h'] += 1
            elif hours < 6:
                duration_ranges['3-6h'] += 1
            else:
                duration_ranges['6+h'] += 1
    
    # Get recent session details for table
    recent_sessions = []
    for session in sessions.order_by('-entry_time')[:10]:
        # Normalize vehicle type for display consistency
        vehicle_type = session.vehicle.get_vehicle_type_display()
        if vehicle_type in ['Sedan', 'SUV']:
            vehicle_type = 'Car'
            
        recent_sessions.append({
            'vehicle_number': session.vehicle.license_plate,
            'slot_number': session.parking_slot.slot_number,
            'vehicle_type': vehicle_type,
            'entry_time': session.entry_time.strftime('%I:%M %p'),
            'duration': str(session.duration).split('.')[0] if session.duration else 'Active',
            'status': 'Active' if session.is_active else 'Completed',
            'is_active': session.is_active,
        })
    
    # Calculate utilization rate for the period
    if sessions.count() > 0:
        avg_utilization = round(sum([
            (sessions.filter(entry_time__date=start_date + timedelta(days=i)).count() / total_slots) * 100 
            for i in range((end_date - start_date).days + 1)
        ]) / ((end_date - start_date).days + 1), 1) if total_slots > 0 else 0
    else:
        avg_utilization = 0
    
    return JsonResponse({
        'status': 'success',
        'timestamp': timezone.now().isoformat(),
        'hourly_occupancy': hourly_data,
        'vehicle_distribution': vehicle_distribution,
        'slot_utilization': slot_utilization,
        'peak_hours': peak_hours_data,
        'duration_analysis': duration_ranges,
        'metrics': {
            'total_sessions': sessions.count(),
            'avg_duration': calculate_avg_duration(sessions),
            'avg_utilization': avg_utilization,
            'active_sessions': ParkingSession.objects.filter(is_active=True).count(),
        },
        'recent_sessions': recent_sessions,
        'date_range': {
            'start': start_date.isoformat(),
            'end': end_date.isoformat(),
            'range_type': date_range
        },
        'current_stats': {
            'total_slots': total_slots,
            'occupied_slots': ParkingSlot.objects.filter(is_occupied=True).count(),
            'available_slots': ParkingSlot.objects.filter(is_occupied=False).count(),
            'active_sessions': ParkingSession.objects.filter(is_active=True).count(),
        }
    })

def calculate_avg_duration(sessions):
    """Calculate average parking duration"""
    if not sessions.exists():
        return '0h 0m'
    
    total_duration = timedelta()
    completed_sessions = sessions.filter(is_active=False)
    
    for session in completed_sessions:
        if session.exit_time:
            total_duration += session.exit_time - session.entry_time
    
    if completed_sessions.count() > 0:
        avg_duration = total_duration / completed_sessions.count()
        hours = int(avg_duration.total_seconds() // 3600)
        minutes = int((avg_duration.total_seconds() % 3600) // 60)
        return f'{hours}h {minutes}m'
    
    return '8h 15m'  # Default placeholder

def calculate_utilization_rate():
    """Calculate space utilization rate - Live Oracle Data"""
    total_slots = ParkingSlot.objects.count()
    occupied_slots = ParkingSlot.objects.filter(is_occupied=True).count()
    
    if total_slots > 0:
        return round((occupied_slots / total_slots) * 100, 1)
    return 0.0

def api_parking_status(request):
    """API endpoint for parking status - Live Oracle Data"""
    if request.method == 'GET':
        total_slots = ParkingSlot.objects.count()
        occupied_slots = ParkingSlot.objects.filter(is_occupied=True).count()
        available_slots = ParkingSlot.objects.filter(is_occupied=False).count()
        maintenance_slots = 0  # No maintenance status in boolean model
        
        # Get slot distribution by type
        slot_distribution = {}
        slot_types = ParkingSlot.objects.values_list('slot_type', flat=True).distinct()
        for slot_type in slot_types:
            slot_distribution[slot_type] = {
                'total': ParkingSlot.objects.filter(slot_type=slot_type).count(),
                'available': ParkingSlot.objects.filter(slot_type=slot_type, is_occupied=False).count(),
                'occupied': ParkingSlot.objects.filter(slot_type=slot_type, is_occupied=True).count(),
                'maintenance': 0,  # No maintenance status in boolean model
            }
        
        # Calculate average duration for all completed sessions
        completed_sessions = ParkingSession.objects.filter(is_active=False)
        avg_duration_str = calculate_avg_duration(completed_sessions)

        return JsonResponse({
            'total_slots': total_slots,
            'occupied_slots': occupied_slots,
            'available_slots': available_slots,
            'maintenance_slots': maintenance_slots,
            'occupancy_rate': round((occupied_slots / total_slots) * 100, 2) if total_slots > 0 else 0,
            'slot_distribution': slot_distribution,
            'active_sessions': ParkingSession.objects.filter(is_active=True).count(),
            'total_vehicles': Vehicle.objects.count(),
            'avg_duration': avg_duration_str,
            'last_updated': timezone.now().isoformat(),
        })
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@login_required
@csrf_exempt
def large_vehicle_requests(request):
    """GET: list requests; POST: create new large vehicle request"""
    if request.method == 'GET':
        status_filter = request.GET.copy().get('status')
        qs = LargeVehicleRequest.objects.all()
        if status_filter:
            qs = qs.filter(status=status_filter)
        data = [
            {
                'id': r.id,
                'license_plate': r.license_plate,
                'owner_name': r.owner_name,
                'contact_number': r.contact_number,
                'registered_state': r.registered_state,
                'status': r.status,
                'requested_at': r.requested_at.isoformat(),
                'notes': r.notes,
            }
            for r in qs.order_by('-requested_at')[:200]
        ]
        return JsonResponse({'results': data, 'count': qs.count()})

    if request.method == 'POST':
        try:
            payload = json.loads(request.body or '{}')
        except Exception:
            return JsonResponse({'error': 'Invalid JSON body'}, status=400)

        required = ['license_plate', 'contact_number']
        missing = [k for k in required if not payload.get(k)]
        if missing:
            return JsonResponse({'error': f"Missing required fields: {', '.join(missing)}"}, status=400)

        r = LargeVehicleRequest.objects.create(
            license_plate=payload['license_plate'],
            owner_name=payload.get('owner_name', ''),
            contact_number=payload['contact_number'],
            registered_state=payload.get('registered_state', ''),
            status=payload.get('status', 'pending'),
            notes=payload.get('notes', ''),
        )
        return JsonResponse({'id': r.id}, status=201)

    return JsonResponse({'error': 'Method not allowed'}, status=405)


@login_required
@csrf_exempt
def large_vehicle_request_detail(request, request_id: int):
    """GET single request or PATCH status/notes"""
    try:
        r = LargeVehicleRequest.objects.get(id=request_id)
    except LargeVehicleRequest.DoesNotExist:
        return JsonResponse({'error': 'Not found'}, status=404)

    if request.method == 'GET':
        return JsonResponse({
            'id': r.id,
            'license_plate': r.license_plate,
            'owner_name': r.owner_name,
            'contact_number': r.contact_number,
            'registered_state': r.registered_state,
            'status': r.status,
            'requested_at': r.requested_at.isoformat(),
            'notes': r.notes,
        })

    if request.method in ['PATCH', 'POST']:
        try:
            payload = json.loads(request.body or '{}')
        except Exception:
            return JsonResponse({'error': 'Invalid JSON body'}, status=400)

        updated = False
        if 'status' in payload and payload['status'] in dict(LargeVehicleRequest.STATUS_CHOICES):
            r.status = payload['status']
            updated = True
        if 'notes' in payload:
            r.notes = payload['notes']
            updated = True
        if updated:
            r.save()
        return JsonResponse({'success': True})

    if request.method == 'DELETE':
        r.delete()
        return JsonResponse({'success': True})

    return JsonResponse({'error': 'Method not allowed'}, status=405)


@login_required
def api_recent_activity(request):
    """API endpoint for recent parking activities."""
    try:
        # Get the 5 most recent entries and 5 most recent exits
        recent_entries = ParkingSession.objects.filter(is_active=True).order_by('-entry_time')[:5]
        recent_exits = ParkingSession.objects.filter(is_active=False, exit_time__isnull=False).order_by('-exit_time')[:5]

        activity_list = []

        for session in recent_entries:
            # Normalize vehicle type for display consistency
            vehicle_type = session.vehicle.get_vehicle_type_display()
            if vehicle_type in ['Sedan', 'SUV']:
                vehicle_type = 'Car'
            
            activity_list.append({
                'type': 'entry',
                'vehicle_number': session.vehicle.license_plate,
                'vehicle_type': vehicle_type,
                'slot_number': session.parking_slot.slot_number,
                'time_ago': timesince(session.entry_time).split(',')[0] + ' ago',
                'timestamp': session.entry_time,
                'owner_contact': session.vehicle.contact_number,
            })

        for session in recent_exits:
            # Normalize vehicle type for display consistency
            vehicle_type = session.vehicle.get_vehicle_type_display()
            if vehicle_type in ['Sedan', 'SUV']:
                vehicle_type = 'Car'
                
            activity_list.append({
                'type': 'exit',
                'vehicle_number': session.vehicle.license_plate,
                'vehicle_type': vehicle_type,
                'slot_number': session.parking_slot.slot_number,
                'time_ago': timesince(session.exit_time).split(',')[0] + ' ago',
                'timestamp': session.exit_time,
                'duration': str(session.duration).split('.')[0], # Format duration
            })

        # Sort all activities by timestamp descending and take the latest 5
        sorted_activity = sorted(activity_list, key=lambda x: x['timestamp'], reverse=True)[:5]

        return JsonResponse({'activities': sorted_activity})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def api_realtime_monitoring(request):
    """API endpoint for real-time monitoring data."""
    try:
        # Get current slot statistics
        total_slots = ParkingSlot.objects.count()
        occupied_slots = ParkingSlot.objects.filter(is_occupied=True).count()
        available_slots = ParkingSlot.objects.filter(is_occupied=False).count()
        maintenance_slots = 0  # No maintenance status in current model
        
        # Calculate utilization rate
        utilization_rate = round((occupied_slots / total_slots) * 100, 1) if total_slots > 0 else 0
        
        # Get recent activities for live feed (last 10 activities)
        recent_entries = ParkingSession.objects.filter(is_active=True).order_by('-entry_time')[:5]
        recent_exits = ParkingSession.objects.filter(is_active=False, exit_time__isnull=False).order_by('-exit_time')[:5]
        
        live_activities = []
        
        # Add recent entries
        for session in recent_entries:
            # Normalize vehicle type for display consistency
            vehicle_type = session.vehicle.get_vehicle_type_display()
            if vehicle_type in ['Sedan', 'SUV']:
                vehicle_type = 'Car'
                
            live_activities.append({
                'type': 'entry',
                'text': 'Vehicle Entry Detected',
                'vehicle_number': session.vehicle.license_plate,
                'slot_number': session.parking_slot.slot_number,
                'time_ago': timesince(session.entry_time).split(',')[0] + ' ago',
                'timestamp': session.entry_time,
                'vehicle_type': vehicle_type,
            })
        
        # Add recent exits
        for session in recent_exits:
            # Normalize vehicle type for display consistency
            vehicle_type = session.vehicle.get_vehicle_type_display()
            if vehicle_type in ['Sedan', 'SUV']:
                vehicle_type = 'Car'
                
            live_activities.append({
                'type': 'exit',
                'text': 'Vehicle Exit Confirmed',
                'vehicle_number': session.vehicle.license_plate,
                'slot_number': session.parking_slot.slot_number,
                'time_ago': timesince(session.exit_time).split(',')[0] + ' ago',
                'timestamp': session.exit_time,
                'vehicle_type': vehicle_type,
                'duration': str(session.duration).split('.')[0] if session.duration else 'Unknown',
            })
        
        # Sort activities by timestamp and take the latest 10
        live_activities.sort(key=lambda x: x['timestamp'], reverse=True)
        live_activities = live_activities[:10]
        
        # Get currently occupied slots for detection zones
        occupied_slot_data = []
        for slot in ParkingSlot.objects.filter(is_occupied=True):
            try:
                session = ParkingSession.objects.get(parking_slot=slot, is_active=True)
                # Normalize vehicle type for display consistency
                vehicle_type = session.vehicle.get_vehicle_type_display()
                if vehicle_type in ['Sedan', 'SUV']:
                    vehicle_type = 'Car'
                    
                occupied_slot_data.append({
                    'slot_number': slot.slot_number,
                    'vehicle_number': session.vehicle.license_plate,
                    'vehicle_type': vehicle_type,
                    'entry_time': session.entry_time.isoformat(),
                    'duration': str(session.duration) if session.duration else 'Calculating...',
                })
            except ParkingSession.DoesNotExist:
                occupied_slot_data.append({
                    'slot_number': slot.slot_number,
                    'vehicle_number': 'Unknown',
                    'vehicle_type': 'Unknown',
                    'entry_time': None,
                    'duration': 'Unknown',
                })
        
        return JsonResponse({
            'status': 'success',
            'timestamp': timezone.now().isoformat(),
            'stats': {
                'total_slots': total_slots,
                'available_slots': available_slots,
                'occupied_slots': occupied_slots,
                'maintenance_slots': maintenance_slots,
                'utilization_rate': utilization_rate,
                'detected_vehicles': occupied_slots,
            },
            'live_activities': live_activities,
            'occupied_slots': occupied_slot_data,
            'system_status': {
                'detection_accuracy': 98.7,  # Could be calculated based on manual verification data
                'response_time': 1.2,  # System response time
                'last_update': timezone.now().isoformat(),
            }
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def api_vehicle_details(request, slot_number):
    """API endpoint for detailed vehicle information in a specific slot."""
    print(f"[DEBUG] api_vehicle_details called with slot_number: {slot_number}")
    
    try:
        # Get the parking slot
        try:
            slot = ParkingSlot.objects.get(slot_number=slot_number)
            print(f"[DEBUG] Found slot: {slot.slot_number}, occupied: {slot.is_occupied}")
        except ParkingSlot.DoesNotExist:
            print(f"[ERROR] Slot {slot_number} not found in database")
            return JsonResponse({'error': f'Slot {slot_number} not found'}, status=404)
        
        if not slot.is_occupied:
            slot_type = 'Unknown'
            try:
                slot_type = slot.get_slot_type_display()
            except:
                pass
            
            return JsonResponse({
                'occupied': False,
                'slot_number': slot_number,
                'slot_type': slot_type,
                'status': 'Available'
            })
        
        # Get the active parking session
        try:
            session = ParkingSession.objects.get(parking_slot=slot, is_active=True)
        except ParkingSession.DoesNotExist:
            return JsonResponse({'error': f'No active session found for occupied slot {slot_number}'}, status=404)
        
        # Get vehicle information
        vehicle = session.vehicle
        if not vehicle:
            return JsonResponse({'error': f'No vehicle information found for session in slot {slot_number}'}, status=404)
        
        # Get basic vehicle information safely
        license_plate = vehicle.license_plate if hasattr(vehicle, 'license_plate') else 'Unknown'
        owner_name = vehicle.owner_name if hasattr(vehicle, 'owner_name') else 'Unknown'
        
        # Get contact number (try both fields)
        contact_number = 'N/A'
        if hasattr(vehicle, 'contact_number') and vehicle.contact_number:
            contact_number = vehicle.contact_number
        elif hasattr(vehicle, 'owner_contact') and vehicle.owner_contact:
            contact_number = vehicle.owner_contact
        
        # Get vehicle type and normalize
        vehicle_type = 'Unknown'
        try:
            if hasattr(vehicle, 'get_vehicle_type_display'):
                vehicle_type = vehicle.get_vehicle_type_display()
            elif hasattr(vehicle, 'vehicle_type'):
                vehicle_type = str(vehicle.vehicle_type)
            
            # Normalize for consistency
            if vehicle_type in ['Sedan', 'SUV']:
                vehicle_type = 'Car'
        except:
            vehicle_type = 'Unknown'
        
        # Get state information
        state_display = 'Unknown'
        try:
            if hasattr(vehicle, 'registered_state') and vehicle.registered_state:
                # Try to get display name for registered_state
                for code, name in vehicle.STATE_CHOICES:
                    if code == vehicle.registered_state:
                        state_display = name
                        break
        except:
            state_display = 'Unknown'
        
        # Get slot type
        slot_type = 'Unknown'
        try:
            slot_type = slot.get_slot_type_display()
        except:
            pass
        
        # Format times safely
        entry_time_formatted = 'Unknown'
        entry_time_ago = 'Unknown'
        duration_str = 'Unknown'
        
        if session.entry_time:
            try:
                entry_time_formatted = session.entry_time.strftime('%B %d, %Y at %I:%M %p')
                entry_time_ago = timesince(session.entry_time) + ' ago'
                
                # Calculate duration
                duration = timezone.now() - session.entry_time
                duration_str = str(duration).split('.')[0]  # Remove microseconds
            except Exception as e:
                pass  # Keep default values
        
        # Get registration/creation date
        registration_date = 'Unknown'
        try:
            if hasattr(vehicle, 'created_at') and vehicle.created_at:
                registration_date = vehicle.created_at.strftime('%B %d, %Y')
        except:
            registration_date = 'Unknown'
        
        vehicle_details = {
            'occupied': True,
            'slot_number': slot_number,
            'slot_type': slot_type,
            'vehicle': {
                'license_plate': license_plate,
                'vehicle_type': vehicle_type,
                'owner_name': owner_name,
                'contact_number': contact_number,
                'state': state_display,
                'registration_date': registration_date
            },
            'session': {
                'entry_time': entry_time_formatted,
                'entry_time_ago': entry_time_ago,
                'duration': duration_str,
                'session_id': session.id,
                'is_active': session.is_active
            },
            'status': 'Occupied'
        }
        
        return JsonResponse(vehicle_details)
        
    except Exception as e:
        # More detailed error logging
        import traceback
        error_detail = traceback.format_exc()
        print(f"ERROR in api_vehicle_details for slot '{slot_number}':")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print(f"Full traceback:\n{error_detail}")
        
        return JsonResponse({
            'error': f'Internal server error while fetching details for slot {slot_number}',
            'detail': str(e) if str(e) else 'Unknown error occurred'
        }, status=500)


@login_required
def api_test_vehicle_details(request, slot_number):
    """Simple test endpoint to verify API routing."""
    return JsonResponse({
        'success': True,
        'slot_number': slot_number,
        'message': f'API endpoint working for slot {slot_number}',
        'timestamp': timezone.now().isoformat()
    })


# Custom Error Handlers for Forbes Marshall SpotCheck

def custom_404(request, exception):
    """Custom 404 error handler"""
    context = {
        'error_title': 'Page Not Found',
        'error_message': 'The requested page could not be found.',
        'error_code': '404',
        'support_email': 'support@forbesmarshall.com'
    }
    return render(request, 'errors/404.html', context, status=404)


def custom_500(request):
    """Custom 500 error handler"""
    context = {
        'error_title': 'Internal Server Error',
        'error_message': 'Something went wrong on our end. Please try again later.',
        'error_code': '500',
        'support_email': 'support@forbesmarshall.com'
    }
    return render(request, 'errors/500.html', context, status=500)


@login_required
def maintenance_management(request):
    """Maintenance Management UI - Manage slot maintenance status"""
    # Get all slots grouped by status
    all_slots = ParkingSlot.objects.all().order_by('slot_type', 'slot_number')
    
    # Group by status
    available_slots = ParkingSlot.objects.filter(status='available').order_by('slot_number')
    occupied_slots = ParkingSlot.objects.filter(status='occupied').order_by('slot_number')
    maintenance_slots = ParkingSlot.objects.filter(status='maintenance').order_by('slot_number')
    out_of_service_slots = ParkingSlot.objects.filter(status='out_of_service').order_by('slot_number')
    
    # Statistics
    total_slots = ParkingSlot.objects.count()
    stats = {
        'total': total_slots,
        'available': available_slots.count(),
        'occupied': occupied_slots.count(),
        'maintenance': maintenance_slots.count(),
        'out_of_service': out_of_service_slots.count(),
    }
    
    # Group by type (exclude disabled and VIP)
    slot_types_data = []
    for slot_type_code, slot_type_name in ParkingSlot.SLOT_TYPES:
        # Skip disabled and VIP slots
        if slot_type_code in ['disabled', 'vip']:
            continue
            
        type_slots = ParkingSlot.objects.filter(slot_type=slot_type_code)
        if type_slots.exists():
            slot_types_data.append({
                'code': slot_type_code,
                'name': slot_type_name,
                'type': slot_type_code,
                'total': type_slots.count(),
                'available': type_slots.filter(status='available').count(),
                'occupied': type_slots.filter(status='occupied').count(),
                'maintenance': type_slots.filter(status='maintenance').count(),
                'out_of_service': type_slots.filter(status='out_of_service').count(),
                'slots': type_slots.order_by('slot_number')
            })
    
    context = {
        'stats': stats,
        'slot_types_data': slot_types_data,
        'available_slots': available_slots,
        'occupied_slots': occupied_slots,
        'maintenance_slots': maintenance_slots,
        'out_of_service_slots': out_of_service_slots,
    }
    
    return render(request, 'dashboard/maintenance_management.html', context)

def custom_403(request, exception):
    """Custom 403 error handler"""
    context = {
        'error_title': 'Access Forbidden',
        'error_message': 'You do not have permission to access this resource.',
        'error_code': '403',
        'support_email': 'support@forbesmarshall.com'
    }
    return render(request, 'errors/403.html', context, status=403)
