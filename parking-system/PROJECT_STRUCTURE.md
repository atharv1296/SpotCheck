# SpotCheck - Forbes Marshall Parking Management System
## Complete Project Structure Documentation

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Root Directory](#root-directory)
3. [Dashboard Application](#dashboard-application)
4. [Core Module](#core-module)
5. [Parking App Module](#parking-app-module)
6. [Driver Applications Module](#driver-applications-module)
7. [Static Files](#static-files)
8. [Templates](#templates)
9. [Database & Migrations](#database--migrations)
10. [Management Commands](#management-commands)

---

## 🎯 Project Overview

**Project Name:** SpotCheck v2.0.0  
**Company:** Forbes Marshall  
**Type:** Web-based Parking Management System  
**Framework:** Django (Python)  
**Database Support:** Oracle Database (primary), SQLite (development)

**Purpose:** A comprehensive parking management system for managing vehicle entries, parking slots, analytics, and large vehicle entry applications for Forbes Marshall facilities.

---

## 📁 Root Directory

### `parking-system/`
The main project root containing the Django application.

```
parking-system/
└── dashboard/          # Main Django application directory
```

---

## 🏢 Dashboard Application

### `dashboard/`
The primary Django project directory containing all application modules, configurations, and resources.

#### Key Files in Dashboard Root:

| File | Purpose |
|------|---------|
| `manage.py` | Django's command-line utility for administrative tasks (runserver, migrate, createsuperuser, etc.) |
| `.env.sample` | Sample environment configuration file for Oracle database credentials and settings |
| `dashboard.log` | Application log file capturing system events, errors, and important activities |
| `parking_activity.log` | Dedicated log file for parking-specific activities (entries, exits, slot changes) |

---

## ⚙️ Core Module

### `dashboard/core/`
Django project configuration and settings module.

| File | Purpose |
|------|---------|
| `__init__.py` | Python package initializer |
| `settings.py` | **Main Django settings** - Database configuration, installed apps, middleware, static files, logging, security settings, timezone (Asia/Kolkata) |
| `urls.py` | **Root URL configuration** - URL routing for entire application, includes parking_app, driver_applications, auth URLs, health check endpoints |
| `wsgi.py` | WSGI configuration for production deployment with WSGI servers (Gunicorn, uWSGI) |
| `asgi.py` | ASGI configuration for asynchronous deployment (prepared for WebSocket support, currently not actively used) |

**Key Responsibilities:**
- Application configuration and settings
- URL routing and endpoint mapping
- Production server configuration
- Health monitoring endpoints (`/health/`, `/api/info/`)

---

## 🚗 Parking App Module

### `dashboard/parking_app/`
Core parking management functionality - the heart of the system.

#### Main Files:

| File | Purpose |
|------|---------|
| `__init__.py` | Package initializer with app metadata and feature list |
| `apps.py` | App configuration - Initializes signals and admin customizations on startup |
| `models.py` | **Database models** - Vehicle, ParkingSlot, ParkingSession, LargeVehicleRequest |
| `views.py` | **View functions** - Dashboard, analytics, real-time monitoring, API endpoints, vehicle registration, slot management |
| `urls.py` | URL patterns for parking app - Dashboard, analytics, realtime, maintenance, API endpoints |
| `api_urls.py` | RESTful API endpoints - Parking status, analytics data, slot updates, vehicle details |
| `auth_views.py` | Authentication views - Login, logout with session management |
| `auth_urls.py` | Authentication URL patterns |
| `admin.py` | Django admin customization for parking models |
| `signals.py` | Django signals for automatic slot status updates when parking sessions change |
| `middleware.py` | Custom middleware for session timeout and security |
| `routing.py` | WebSocket routing configuration (prepared but not actively used) |

#### Database Models:

1. **Vehicle** - License plate, owner info, contact, vehicle type, state
2. **ParkingSlot** - Slot number, type (two_wheeler/car/large), status (available/occupied/maintenance)
3. **ParkingSession** - Active parking sessions with entry time, duration tracking
4. **LargeVehicleRequest** - Entry requests for large vehicles requiring approval

#### Key Features:
- Real-time parking slot monitoring
- Vehicle entry/exit management
- Analytics and reporting
- Maintenance slot management
- Large vehicle request handling
- RESTful API for data operations

### `parking_app/management/commands/`
Custom Django management commands for system administration.

| Command | Purpose |
|---------|---------|
| `init_parking_system.py` | Initialize complete parking system with sample data |
| `seed_slots.py` | Seed parking slots (H1-H15: two-wheeler, C1-C10: car, L1-L5: large) |
| `setup_oracle.py` | Configure and setup Oracle database |
| `setup_sqlite.py` | Configure and setup SQLite database for development |
| `clear_slots.py` | Clear all parking sessions and reset slots |
| `expand_slots.py` | Add more parking slots dynamically |
| `show_slots.py` | Display current parking slot status |
| `fix_duplicate_sessions.py` | Fix duplicate active parking sessions |
| `system_health.py` | Comprehensive system health check and diagnostics |
| `test_oracle.py` | Test Oracle database connection |
| `maintenance.py` | Slot maintenance management utilities |

**Usage Example:**
```bash
python manage.py seed_slots          # Create parking slots
python manage.py init_parking_system # Initialize system
python manage.py system_health       # Check system status
```

### `parking_app/migrations/`
Django database migration files tracking schema changes over time.

| Migration | Changes |
|-----------|---------|
| `0001_initial.py` | Initial database schema - Vehicle, ParkingSlot, ParkingSession models |
| `0002_alter_parkingslot_slot_type_and_more.py` | Slot type field modifications |
| `0003_update_table_structure.py` | Table structure updates |
| `0004_largevehiclerequest.py` | Added LargeVehicleRequest model |
| `0005_merge_sedan_suv_to_car.py` | Merged sedan/SUV types into car category |
| `0006_alter_parkingslot_slot_type.py` | Slot type alterations |
| `0007_parkingslot_status.py` | Added status field to parking slots |
| `0008_add_unique_active_session_constraint.py` | Constraint preventing duplicate active sessions |

---

## 🚚 Driver Applications Module

### `dashboard/driver_applications/`
Large vehicle entry application management system.

#### Main Files:

| File | Purpose |
|------|---------|
| `__init__.py` | Package initializer |
| `apps.py` | App configuration |
| `models.py` | **Database models** - DriverApplication, ApplicationComment, ApplicationStatusHistory |
| `views.py` | **View functions** - Application submission, status tracking, admin dashboard, document management |
| `urls.py` | URL patterns for driver applications |
| `forms.py` | Django forms - DriverApplicationForm, ApplicationSearchForm, ReviewForm, CommentForm |
| `admin.py` | Django admin interface customization |

#### Database Models:

1. **DriverApplication** - Driver details, vehicle info, documents (stored as BLOBs), material information
2. **ApplicationComment** - Comments on applications (internal/public)
3. **ApplicationStatusHistory** - Track status changes over time

#### Key Features:
- Public application form for drivers
- Document upload (driver photo, license, RC, insurance, PUC, receipts)
- Admin review dashboard
- Status tracking (pending → approved/rejected)
- Bulk actions on applications
- Comment system for communication

### `driver_applications/management/commands/`

| Command | Purpose |
|---------|---------|
| `fix_media_for_applications.py` | Migrate and fix media file attachments for applications |

### `driver_applications/templates/driver_applications/`
Application-specific templates.

| Template | Purpose |
|----------|---------|
| `public_base.html` | Base template for public-facing pages |
| `apply_entry.html` | Public form for submitting large vehicle entry applications |
| `application_status.html` | Public page for checking application status |
| `dashboard.html` | Admin dashboard for managing applications |
| `application_detail.html` | Detailed view of application for admin review |

---

## 🎨 Static Files

### `dashboard/static/`
Static assets (CSS, JavaScript, images).

#### `static/css/`
| File | Purpose |
|------|---------|
| `professional-theme.css` | **Active theme** - Professional design theme used throughout the application |

**Note:** Previously had multiple unused themes (dashboard.css, enhancements.css, modern-theme.css, clean-professional.css) which have been removed.

#### `static/js/`
Currently empty - JavaScript is embedded in templates for now.

**Note:** Previously had unused files (dashboard.js, enhanced.js, sw.js) which have been removed.

#### `static/images/` & `static/img/`
Application images, logos, and icons.

---

## 🖼️ Templates

### `dashboard/templates/`
HTML templates for the application.

#### `templates/auth/`
Authentication templates.

| Template | Purpose |
|----------|---------|
| `login.html` | **User login page** - Staff authentication with remember me option, session management |

#### `templates/dashboard/`
Main application templates using professional theme.

| Template | Purpose |
|----------|---------|
| `professional_base.html` | **Base template** - Navigation, header, footer, sidebar, used by all dashboard pages |
| `professional_dashboard.html` | **Main dashboard** - Real-time parking slot visualization, statistics, recent activity |
| `professional_realtime.html` | **Real-time monitoring** - Live parking updates, slot status changes |
| `professional_analytics.html` | **Analytics page** - Reports, charts, parking trends, export functionality |
| `maintenance_management.html` | **Maintenance UI** - Manage slot maintenance status |

**Theme Architecture:**
- All pages extend `professional_base.html`
- Consistent Forbes Marshall branding
- Responsive design
- Real-time data updates

#### `templates/errors/`
Error page templates.

| Template | Purpose |
|----------|---------|
| `403.html` | **Access Forbidden** - Displayed when user lacks permissions |
| `404.html` | **Page Not Found** - Displayed for invalid URLs |
| `500.html` | **Server Error** - Displayed for internal server errors |

All error handlers are registered in `core/urls.py` as `handler403`, `handler404`, `handler500`.

---

## 🗄️ Database & Migrations

### Database Support:
1. **Oracle Database** (Production)
   - Configured via `.env` file
   - Connection details in `settings.py`
   - Setup command: `setup_oracle.py`

2. **SQLite** (Development)
   - Default for development
   - File-based database
   - Setup command: `setup_sqlite.py`

### Migration Management:
```bash
python manage.py makemigrations   # Create new migrations
python manage.py migrate           # Apply migrations
python manage.py showmigrations    # View migration status
```

---

## 🔧 Management Commands

### Running Management Commands:
```bash
python manage.py <command_name> [options]
```

### Common Commands:

#### System Setup:
```bash
python manage.py seed_slots              # Create parking slots
python manage.py init_parking_system     # Initialize complete system
python manage.py setup_oracle            # Setup Oracle database
python manage.py setup_sqlite            # Setup SQLite database
```

#### System Maintenance:
```bash
python manage.py system_health           # Check system health
python manage.py show_slots              # Display slot status
python manage.py clear_slots             # Clear all sessions
python manage.py fix_duplicate_sessions  # Fix duplicate sessions
python manage.py maintenance             # Maintenance utilities
```

#### Django Built-in:
```bash
python manage.py runserver               # Start development server
python manage.py createsuperuser         # Create admin user
python manage.py migrate                 # Run migrations
python manage.py collectstatic           # Collect static files
python manage.py shell                   # Django Python shell
```

---

## 🔐 Security & Authentication

### Authentication System:
- Staff-only access (non-staff users denied)
- Session-based authentication
- Session timeout (10 minutes default, 2 weeks with "remember me")
- CSRF protection enabled
- Secure password hashing

### Middleware Stack:
1. Security Middleware
2. Session Middleware
3. CSRF Middleware
4. Authentication Middleware
5. Message Middleware
6. Custom Session Timeout Middleware

---

## 📊 API Endpoints

### RESTful API (`/api/`)

#### Parking Status & Data:
- `GET /api/parking-status/` - Current parking overview
- `GET /api/parking-data/` - Detailed parking data
- `GET /api/recent-activity/` - Recent parking activities
- `GET /api/realtime-monitoring/` - Real-time monitoring data
- `GET /api/vehicle-details/<slot_number>/` - Vehicle details for slot

#### Slot Management:
- `POST /api/update-slot/` - Update slot status

#### Analytics:
- `GET /api/analytics-data/` - Analytics data
- `GET /api/analytics/export/` - Export analytics (CSV/PDF)

#### Large Vehicle Requests:
- `GET /api/large-vehicle-requests/` - List requests (with status filter)
- `POST /api/large-vehicle-requests/` - Create new request
- `GET /api/large-vehicle-request/<id>/` - Get request details
- `PATCH /api/large-vehicle-request/<id>/` - Update request
- `DELETE /api/large-vehicle-request/<id>/` - Delete request

#### Driver Applications:
- `GET /api/application-stats/` - Application statistics

#### System Health:
- `GET /health/` - System health check
- `GET /api/info/` - API information and documentation

---

## 📱 Features Summary

### Core Features:
✅ Real-time parking slot monitoring  
✅ Vehicle entry/exit management  
✅ Analytics and reporting with charts  
✅ Maintenance slot management  
✅ Large vehicle request system  
✅ Driver application workflow  
✅ Document management (BLOB storage)  
✅ Staff authentication with role-based access  
✅ RESTful API for integrations  
✅ Responsive professional UI  
✅ Oracle & SQLite database support  
✅ Comprehensive logging system  
✅ Custom management commands  
✅ Error handling (403, 404, 500)  
✅ Session management with timeout  

### Slot Types:
- **Two Wheeler** (H1-H15): 15 slots
- **Car** (C1-C10): 10 slots  
- **Large Vehicle** (L1-L5): 5 slots

### Vehicle Status:
- Pending - Entry request submitted
- Approved - Entry approved
- Rejected - Entry denied

### Slot Status:
- Available - Ready for parking
- Occupied - Vehicle parked
- Maintenance - Under maintenance

---

## 🚀 Deployment & Running

### Development:
```bash
# Navigate to dashboard directory
cd parking-system/dashboard

# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Seed parking slots
python manage.py seed_slots

# Run development server
python manage.py runserver

# Access application
http://localhost:8000
```

### Production:
- Use WSGI server (Gunicorn)
- Configure Oracle database in `.env`
- Run `collectstatic` for static files
- Set `DEBUG=False` in settings
- Configure proper logging
- Use reverse proxy (Nginx/Apache)

---

## 📝 Logging

### Log Files:
- `dashboard.log` - General application logs
- `parking_activity.log` - Parking-specific activities

### Log Levels:
- INFO - General information
- WARNING - Warning messages
- ERROR - Error messages
- DEBUG - Debug information (development only)

---

## 🔄 Removed Unused Files

The following files were identified as unused and removed from the project:

### Templates:
- ❌ `templates/dashboard/analytics.html`
- ❌ `templates/dashboard/base.html`
- ❌ `templates/dashboard/index.html`
- ❌ `templates/dashboard/realtime.html`
- ❌ `templates/dashboard/parkomate_base.html`
- ❌ `templates/dashboard/parkomate_index.html`
- ❌ `templates/dashboard/parkomate_realtime.html`
- ❌ `templates/errors/professional_404.html`
- ❌ `templates/offline.html`

### Static Files:
- ❌ `static/css/dashboard.css`
- ❌ `static/css/enhancements.css`
- ❌ `static/css/modern-theme.css`
- ❌ `static/css/clean-professional.css`
- ❌ `static/js/dashboard.js`
- ❌ `static/js/enhanced.js`
- ❌ `static/js/sw.js`

### Python Files:
- ❌ `parking_app/serializers.py` (DRF serializers not used)
- ❌ `parking_app/consumers.py` (WebSocket not implemented)
- ❌ `parking_app/routing.py` (WebSocket routing not used)
- ❌ `parking_app/tests.py` (Empty test file)
- ❌ `driver_applications/tests.py` (Empty test file)
- ❌ `core/apps.py` (Core not in INSTALLED_APPS)
- ❌ `core/signals.py` (Signals not registered)
- ❌ `core/system_status.py` (Documentation file not used)

These files were alternative themes, PWA features, or test frameworks that were never fully implemented.

---

## 📞 Support & Contact

**Company:** Forbes Marshall  
**System:** SpotCheck v2.0.0  
**Support Email:** support@forbesmarshall.com  
**Repository:** SpotCheck  
**Owner:** atharv1296

---

## 📄 License

Proprietary - Forbes Marshall  
© 2025 Forbes Marshall. All rights reserved.

---

**Last Updated:** December 5, 2025  
**Documentation Version:** 1.0
