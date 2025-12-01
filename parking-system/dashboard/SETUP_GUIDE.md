# 🚗 Parking Management System - Complete Setup Guide

This guide will help you set up the Forbes Marshal Parking Management System from scratch.

---

## 📋 Prerequisites

Before starting, make sure you have:

1. **Python 3.8+** installed ([Download Python](https://www.python.org/downloads/))
2. **Oracle Database 19c** installed ([Download Oracle](https://www.oracle.com/database/technologies/oracle-database-software-downloads.html))
3. **Git** installed ([Download Git](https://git-scm.com/downloads))
4. **VS Code** or any text editor (optional but recommended)

---

## 🗄️ Step 1: Oracle Database Setup

### 1.1 Install Oracle Database 19c

1. Download Oracle Database 19c from the official website
2. Run the installer and follow the installation wizard
3. During installation, note down:
   - **System Password**: `system123` (or your chosen password)
   - **Port**: `1521` (default)
   - **Service Name**: `orclpdb` (pluggable database)

### 1.2 Verify Oracle Installation

Open **SQL*Plus** or **SQLcl** and connect:

```sql
sqlplus system/system123@localhost:1521/orclpdb
```

If connected successfully, your Oracle database is ready!

### 1.3 Install Oracle Instant Client (for Python connection)

1. Download Oracle Instant Client from [Oracle Downloads](https://www.oracle.com/database/technologies/instant-client/downloads.html)
2. Extract to a folder like `C:\oracle\instantclient_19_x`
3. Add this folder to your **System PATH**:
   - Right-click **This PC** → **Properties** → **Advanced system settings**
   - Click **Environment Variables**
   - Under **System variables**, find **Path** and click **Edit**
   - Add the path: `C:\oracle\instantclient_19_x`
   - Click **OK** and restart your terminal

---

## 📦 Step 2: Clone the Project

Open **PowerShell** or **Command Prompt** and run:

```powershell
cd "C:\Users\YourUsername\Desktop"
git clone https://github.com/atharv1296/SpotCheck.git
cd SpotCheck\parking-system\dashboard
```

---

## 🐍 Step 3: Set Up Python Virtual Environment

### 3.1 Create Virtual Environment

```powershell
python -m venv venv
```

### 3.2 Activate Virtual Environment

**Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### 3.3 Install Python Dependencies

```powershell
pip install --upgrade pip
pip install django==5.1.1
pip install cx_Oracle
pip install channels
pip install daphne
pip install Pillow
```

Or if there's a `requirements.txt` in the dashboard folder:
```powershell
pip install -r requirements.txt
```

---

## ⚙️ Step 4: Configure Database Settings

### 4.1 Update Database Credentials

Open `core/settings.py` and verify the database configuration:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.oracle',
        'NAME': 'localhost:1521/orclpdb',
        'USER': 'system',
        'PASSWORD': 'system123',  # Change this if you used a different password
        'HOST': '',
        'PORT': '',
    }
}
```

**Important:** If you used a different password during Oracle installation, update the `PASSWORD` field.

---

## 🔧 Step 5: Initialize the Database

### 5.1 Run Migrations

```powershell
python manage.py makemigrations
python manage.py migrate
```

This will create all necessary tables in your Oracle database.

### 5.2 Create Parking Slots

```powershell
python manage.py seed_slots
```

This creates:
- 50 two-wheeler slots (T001-T050)
- 30 car slots (C001-C030)
- 20 large vehicle slots (L001-L020)

### 5.3 Create Admin User

```powershell
python manage.py createsuperuser
```

Follow the prompts:
- **Username**: Choose a username (e.g., `admin`)
- **Email**: Your email (optional)
- **Password**: Choose a strong password
- **Confirm password**: Re-enter password

**⚠️ IMPORTANT:** After creating the superuser, you need to mark them as staff:

1. Start the server (see Step 6)
2. Go to Django admin: `http://127.0.0.1:8000/admin/`
3. Log in with superuser credentials
4. Click **Users** → Click on your username
5. Check the box **Staff status** ✅
6. Click **Save**

Alternatively, use SQLcl to update directly:

```sql
sqlplus system/system123@localhost:1521/orclpdb

UPDATE AUTH_USER SET IS_STAFF = 1 WHERE USERNAME = 'admin';
COMMIT;
EXIT;
```

### 5.4 Create Media Directories

```powershell
python create_media_dirs.py
```

This creates folders for storing driver application documents.

---

## 🚀 Step 6: Start the Server

### 6.1 Run the Development Server

```powershell
python manage.py runserver
```

You should see:
```
Starting development server at http://127.0.0.1:8000/
Quit the server with CTRL-BREAK.
```

### 6.2 Access the Application

Open your web browser and go to:

- **Main Application**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Admin Panel**: [http://127.0.0.1:8000/admin/](http://127.0.0.1:8000/admin/)

---

## 👤 Step 7: Log In

1. Go to [http://127.0.0.1:8000/login/](http://127.0.0.1:8000/login/)
2. Enter your superuser credentials
3. You should be redirected to the dashboard

**Note:** Only users with **Staff status** can log in to the system.

---

## 📊 Step 8: Test the System

### 8.1 View Parking Dashboard

After logging in, you should see the **Professional Dashboard** with:

**📈 Key Metrics:**
- Total slots count (two-wheeler, car, large vehicle)
- Available/occupied slots with real-time counts
- Occupancy rate percentage
- Active parking sessions
- Total registered vehicles
- Driver applications statistics (pending, total, today's applications)
- Large vehicle requests (pending and total)

**🎨 Live Parking Grid:**
- 🟢 **Green** = Available
- 🔴 **Red** = Occupied
- 🟡 **Yellow** = Under maintenance
- ⚫ **Gray** = Out of service

**📊 Slot Distribution:**
- Two-wheeler slots status breakdown
- Car slots status breakdown
- Large vehicle slots status breakdown

**🕐 Recent Activity:**
- Recent parking sessions (last 5)
- Recently registered vehicles (last 5)

### 8.2 Navigation Menu

The system includes multiple sections:

1. **Dashboard** - Main overview with statistics
2. **Real-time Monitoring** - Live parking activity
3. **Analytics** - Detailed reports and charts
4. **Maintenance Management** - Slot maintenance control
5. **Driver Applications** - Large vehicle entry applications

### 8.3 Register a Vehicle (Smart Registration)

The system has **intelligent vehicle registration** that checks driver applications:

**Method 1: Via Slot Grid**
1. Click on any **available** slot (green)
2. Enter vehicle number (e.g., `MH12BN0003`)
3. The system automatically checks:
   - ✅ If vehicle is already registered → Assigns slot directly
   - ✅ If driver application exists and is **approved** → Auto-registers from application data
   - ❌ If driver application is **rejected** → Shows rejection reason
   - ❌ If driver application is **pending/under review** → Prompts to wait
   - ✅ If no application exists → Allows manual registration

**Method 2: Manual Vehicle Registration**
1. Click "Check Vehicle" button
2. Enter vehicle details:
   - Vehicle number (format: `MH12AB1234`)
   - Owner name
   - Contact number
   - Vehicle type (two-wheeler/sedan/SUV/large)
   - Registered state (select from dropdown)
3. Click **Register Vehicle**

**Method 3: From Approved Driver Application**
- When admin approves a large vehicle application
- Vehicle is automatically registered in the system
- Can be assigned to any compatible slot

### 8.4 Check Out a Vehicle

1. Click on any **occupied** slot (red)
2. Review parking details:
   - Vehicle number and type
   - Driver name and contact
   - Entry time and duration
   - Parking slot details
3. Click **End Session** or **Check Out**
4. System records exit time
5. Slot status changes to available (green)

### 8.5 Driver Applications (Large Vehicles)

**For Drivers/External Users:**

1. Go to `/driver-applications/apply/` (public URL, no login required)
2. Fill out comprehensive application form:
   
   **Driver Information:**
   - Driver name, phone, email
   - Driver license number
   
   **Vehicle Information:**
   - Vehicle number (format: `MH12BN0003`)
   - Vehicle type (default: large)
   - Vehicle model and capacity
   
   **Company/Business Details:**
   - Source company name and address
   - Source company contact
   - Destination within Forbes Marshall premises
   
   **Material/Purpose Information:**
   - Material type (raw materials, finished goods, equipment, etc.)
   - Detailed material description
   - Approximate weight and value
   - Urgency level (low/medium/high/emergency)
   
   **Requested Entry Details:**
   - Requested entry date
   - Requested entry time
   - Estimated duration (in minutes)

3. **Upload Required Documents** (stored as BLOBs in Oracle database):
   - Driver photo
   - Driver license photo
   - Vehicle RC (Registration Certificate)
   - Vehicle insurance
   - PUC certificate
   - Material receipt/invoice
   - Goods transport permit (if applicable)
   - Customs clearance (for imports, if applicable)

4. Submit application
5. Receive unique **Application ID** for tracking

**For Admin/Security Staff:**

1. Go to **Driver Applications** section
2. View all applications with filters:
   - Status: Pending / Under Review / Approved / Rejected / Expired
   - Search by vehicle number, driver name, company
   - Date range filters

3. Click on any application to view details:
   - Complete driver and vehicle information
   - Material details with urgency indicators
   - View/download uploaded documents
   - Application status history

4. **Review Application:**
   - Change status to "Under Review"
   - View all uploaded documents (download from Oracle BLOB)
   - Add internal notes (not visible to driver)
   
5. **Approve Application:**
   - Select "Approved" status
   - Vehicle is **automatically registered** in the system
   - Driver can now be assigned a parking slot
   - Status becomes **immutable** (cannot be changed after approval)

6. **Reject Application:**
   - Select "Rejected" status
   - **Must provide rejection reason** (visible to applicant)
   - Driver sees rejection reason in application status
   - Status becomes **immutable** (cannot be changed after rejection)

**Application Status Tracking:**
- ⏳ **Pending** - Just submitted, awaiting review
- 🔍 **Under Review** - Being reviewed by admin
- ✅ **Approved** - Approved, vehicle registered
- ❌ **Rejected** - Rejected with reason
- ⏰ **Expired** - Entry date has passed

### 8.6 Real-time Monitoring

Access via **Real-time Monitoring** menu:

**Features:**
- Live parking status updates
- Recent entries (last 10)
- Recent exits (last 10)
- Active sessions with duration
- Real-time occupancy percentage
- Auto-refresh capability

### 8.7 Analytics Dashboard

Access via **Analytics** menu:

**Available Reports:**
- Peak hours analysis
- Vehicle type distribution
- Average parking duration
- Slot utilization rates
- Daily/weekly/monthly trends
- Occupancy heatmaps
- Revenue calculations (if applicable)

**Export Options:**
- Export to CSV
- Export to PDF (if ReportLab installed)
- Custom date range selection

### 8.8 Maintenance Management

Access via **Maintenance Management** menu:

**Features:**
1. View all slots with maintenance status
2. **Toggle Maintenance Mode:**
   - Click on any slot
   - Switch between:
     - ✅ Available
     - 🛠️ Under Maintenance
     - 🚫 Out of Service
3. Slots under maintenance cannot accept new vehicles
4. Occupied slots cannot be set to maintenance (must checkout first)
5. Maintenance history tracking

### 8.9 Large Vehicle Requests

For vehicles without driver applications:

1. Quick request form for large vehicles
2. Admin can view and approve/reject
3. Lighter process than full driver application
4. Useful for emergency or one-time entries

---

## 🛠️ Troubleshooting

### Issue 1: Oracle Connection Error

**Error:** `DPI-1047: Cannot locate a 64-bit Oracle Client library`

**Solution:**
- Install Oracle Instant Client
- Add to System PATH
- Restart terminal/VS Code

### Issue 2: Migration Errors

**Error:** `ORA-00955: name is already used by an existing object`

**Solution:**
```powershell
python manage.py migrate --fake
```

### Issue 3: Login Not Working

**Error:** "Only staff members can access this system"

**Solution:**
- Log in to Django admin: `/admin/`
- Set **Staff status** = ✅ for your user
- Or run SQL: `UPDATE AUTH_USER SET IS_STAFF = 1 WHERE USERNAME = 'yourusername';`

### Issue 4: Slots Not Showing

**Solution:**
```powershell
python manage.py seed_slots
```

### Issue 5: Media Files Not Uploading

**Solution:**
```powershell
python create_media_dirs.py
```

---

## 📁 Project Structure

```
parking-system/
└── dashboard/
    ├── manage.py                    # Django management script
    ├── core/                        # Project settings
    │   ├── settings.py              # Main configuration (Oracle DB settings)
    │   ├── urls.py                  # URL routing
    │   ├── wsgi.py                  # WSGI server configuration
    │   ├── asgi.py                  # ASGI server configuration
    │   ├── middleware.py            # Staff verification middleware
    │   └── signals.py               # Signal handlers
    │
    ├── parking_app/                 # Main parking management app
    │   ├── models.py                # Database models (Vehicle, ParkingSlot, ParkingSession, LargeVehicleRequest)
    │   ├── views.py                 # Business logic (20+ views)
    │   ├── urls.py                  # App URL patterns
    │   ├── api_urls.py              # API endpoints
    │   ├── auth_views.py            # Authentication (staff-only login)
    │   ├── auth_urls.py             # Auth URL patterns
    │   ├── serializers.py           # API serializers
    │   ├── consumers.py             # WebSocket consumers
    │   ├── routing.py               # WebSocket routing
    │   ├── signals.py               # Signal handlers
    │   ├── admin.py                 # Django admin configuration
    │   └── management/
    │       └── commands/            # Management commands
    │           ├── seed_slots.py    # Create parking slots
    │           ├── init_parking_system.py
    │           ├── maintenance.py   # Maintenance utilities
    │           ├── system_health.py # System health check
    │           ├── clear_slots.py   # Clear all slots
    │           ├── expand_slots.py  # Add more slots
    │           └── fix_duplicate_sessions.py
    │
    ├── driver_applications/         # Driver application module
    │   ├── models.py                # Application models (DriverApplication, ApplicationComment, ApplicationStatusHistory)
    │   ├── views.py                 # Application views (apply, review, approve/reject)
    │   ├── forms.py                 # Application forms (with document upload)
    │   ├── urls.py                  # Application URL patterns
    │   ├── admin.py                 # Admin configuration
    │   └── management/
    │       └── commands/
    │           └── fix_media_for_applications.py
    │
    ├── static/                      # Static files
    │   ├── css/                     # Stylesheets
    │   ├── js/                      # JavaScript files
    │   ├── images/                  # Images and icons
    │   └── img/                     # Additional images
    │
    ├── templates/                   # HTML templates
    │   ├── dashboard/               # Dashboard templates
    │   │   ├── professional_dashboard.html
    │   │   ├── realtime.html
    │   │   ├── analytics.html
    │   │   └── maintenance_management.html
    │   ├── driver_applications/     # Application templates
    │   │   ├── apply.html
    │   │   ├── application_list.html
    │   │   ├── application_detail.html
    │   │   └── status_check.html
    │   ├── auth/                    # Authentication templates
    │   │   ├── login.html
    │   │   └── logout.html
    │   ├── errors/                  # Error page templates
    │   │   ├── 403.html
    │   │   ├── 404.html
    │   │   └── 500.html
    │   └── offline.html
    │
    ├── media/                       # Uploaded files (organized structure)
    │   └── driver_applications/
    │       ├── photos/              # Driver photos
    │       ├── licenses/            # Driver licenses
    │       ├── rc/                  # Vehicle RC documents
    │       ├── insurance/           # Insurance documents
    │       ├── puc/                 # PUC certificates
    │       ├── receipts/            # Material receipts
    │       ├── permits/             # Transport permits
    │       └── customs/             # Customs clearance docs
    │
    ├── scripts/                     # Utility scripts
    │   ├── db_debug.py              # Database debugging
    │   └── inspect_columns.py       # Column inspection
    │
    ├── create_media_dirs.py         # Create media directory structure
    ├── clear_sessions.py            # Clear parking sessions
    ├── oracle_utils.py              # Oracle database utilities
    └── SETUP_GUIDE.md               # This file
```

---

## 🔐 Default Credentials

After setup, use these credentials:

- **Oracle Database:**
  - Username: `system`
  - Password: `system123`
  - Connection: `localhost:1521/orclpdb`

- **Django Admin:**
  - Username: (what you created in Step 5.3)
  - Password: (what you set)

---

## 🎯 Key Features

### 🔐 Security Features
- **Staff-Only Access:** All users must have `is_staff=True` to log in
- **Session-Based Verification:** StaffRequiredMiddleware checks on every request
- **Auto-Logout:** Non-staff users are automatically logged out
- **Exempt URLs:** Login, static files, and admin login are accessible without staff check

### 🚗 Smart Vehicle Management
- **Automatic Registration:** Vehicles from approved driver applications are auto-registered
- **Case-Insensitive Search:** Vehicle lookup works regardless of case (MH12BN0003 = mh12bn0003)
- **Driver Application Integration:** System checks if vehicle has pending/approved/rejected application
- **State-Based Validation:** Indian state codes for vehicle registration
- **Multiple Vehicle Types:** Two-wheeler, Sedan, SUV, Large vehicles

### 📋 Driver Application System
- **Public Access:** Drivers can apply without login
- **Document Management:** All documents stored as BLOBs in Oracle (no file system storage)
- **Immutable Status:** Approved/rejected applications cannot be modified
- **Rejection Reasons:** Applicants can see why their application was rejected
- **Status History:** Complete audit trail of all status changes
- **Urgency Levels:** Low, Medium, High, Emergency with color indicators
- **Material Categories:** 9 different material types supported

### 📊 Real-Time Monitoring
- **Live Updates:** Real-time parking status across all slots
- **Active Sessions:** Track all ongoing parking sessions with duration
- **Recent Activity:** View last 10 entries and exits
- **Occupancy Metrics:** Real-time occupancy rate calculations
- **Slot Distribution:** Breakdown by vehicle type

### 🛠️ Maintenance Management
- **Flexible Status Control:** Available, Maintenance, Out of Service
- **Occupancy Prevention:** Maintenance slots cannot accept new vehicles
- **Status Protection:** Occupied slots cannot be set to maintenance
- **Real-Time Updates:** Changes reflect immediately across system

### 📈 Analytics & Reporting
- **Peak Hours Analysis:** Identify busiest times
- **Duration Reports:** Average parking duration by vehicle type
- **Utilization Rates:** Slot usage efficiency metrics
- **Export Options:** CSV and PDF export (with ReportLab)
- **Custom Date Ranges:** Filter reports by specific periods

### 🗄️ Database Architecture
- **Oracle 19c Backend:** Enterprise-grade database
- **BLOB Storage:** Documents stored directly in database
- **Normalized Schema:** Proper foreign key relationships
- **Transaction Management:** Atomic operations for data integrity
- **Audit Trail:** Created/updated timestamps on all records

## 📝 Important Notes

1. **Staff Status Required:** All users must have `is_staff=True` to log in (enforced at login + middleware)
2. **Oracle Client Required:** Python needs Oracle Instant Client to connect to the database
3. **Virtual Environment:** Always activate the virtual environment before running commands
4. **Port 8000:** Make sure port 8000 is not in use by another application
5. **File Uploads:** Driver applications are stored as BLOBs in Oracle database (not file system)
6. **Vehicle Number Format:** Use Indian format (e.g., MH12BN0003) for proper validation
7. **Slot Types:** Two-wheeler (T001-T050), Car (C001-C030), Large (L001-L020)
8. **Status Immutability:** Once approved/rejected, driver applications cannot be changed
9. **Case-Insensitive:** Vehicle number lookups work regardless of uppercase/lowercase

---

## 🎯 Quick Start Commands (After Initial Setup)

```powershell
# Navigate to project
cd "C:\Users\YourUsername\Desktop\SpotCheck\parking-system\dashboard"

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start server
python manage.py runserver
```

---

## 🎓 Management Commands Reference

The system includes several management commands for common tasks:

```powershell
# Parking Slot Management
python manage.py seed_slots              # Create default parking slots
python manage.py expand_slots            # Add more slots to existing setup
python manage.py clear_slots             # Clear all slots (WARNING: deletes all)
python manage.py show_slots              # Display all slots and their status

# System Maintenance
python manage.py maintenance             # Interactive maintenance mode
python manage.py system_health           # Check system health and connectivity
python manage.py fix_duplicate_sessions  # Fix duplicate active sessions
python manage.py init_parking_system     # Initialize complete parking system

# Database Operations
python manage.py test_oracle             # Test Oracle database connection
python manage.py setup_oracle            # Setup Oracle database
python manage.py makemigrations          # Create database migrations
python manage.py migrate                 # Apply database migrations

# User Management
python manage.py createsuperuser         # Create admin user
```

## 📊 Database Tables

The system uses the following Oracle tables:

**Core Tables:**
- `PARKING_APP_VEHICLE` - Registered vehicles
- `PARKING_APP_PARKINGSLOT` - Parking slot inventory
- `PARKING_APP_PARKINGSESSION` - Parking sessions (entry/exit tracking)
- `PARKING_APP_LARGEVEHICLEREQUEST` - Quick large vehicle requests

**Driver Application Tables:**
- `DRIVER_APPLICATIONS_DRIVERAPPLICATION` - Application details with BLOB documents
- `DRIVER_APPLICATIONS_APPLICATIONCOMMENT` - Comments on applications
- `DRIVER_APPLICATIONS_APPLICATIONSTATUSHISTORY` - Status change audit trail

**Authentication Tables:**
- `AUTH_USER` - Django user accounts
- `DJANGO_SESSION` - User sessions

## 🔧 Common Configuration Changes

### Change Oracle Database Password

Edit `core/settings.py`:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.oracle',
        'NAME': 'localhost:1521/orclpdb',
        'USER': 'system',
        'PASSWORD': 'YOUR_NEW_PASSWORD',  # Change here
        'HOST': '',
        'PORT': '',
    }
}
```

### Change Session Timeout

Edit `core/settings.py`:
```python
SESSION_COOKIE_AGE = 600  # 10 minutes (default)
# Change to:
SESSION_COOKIE_AGE = 1800  # 30 minutes
```

### Enable Debug Mode

Edit `core/settings.py`:
```python
DEBUG = True  # For development only
DEBUG = False  # For production
```

### Add More Parking Slots

```powershell
python manage.py expand_slots
# Follow prompts to add slots by type
```

## 📱 API Endpoints

The system provides REST APIs for integration:

**Parking Status:**
- `GET /api/parking-data/` - Get all parking data
- `GET /api/parking-status/` - Get current parking status
- `POST /api/update-slot/` - Update slot status
- `POST /api/check-vehicle/` - Check vehicle registration

**Analytics:**
- `GET /api/analytics-data/` - Get analytics data
- `GET /api/recent-activity/` - Get recent parking activity
- `GET /api/realtime-monitoring/` - Real-time monitoring data

**Vehicle Management:**
- `GET /api/vehicle-details/<slot_number>/` - Get vehicle in specific slot

**Large Vehicle Requests:**
- `GET /api/large-vehicle-requests/` - List all requests
- `GET /api/large-vehicle-requests/<id>/` - Get specific request details

**Maintenance:**
- `POST /api/toggle-maintenance/` - Toggle maintenance status

## 🌐 URL Structure

**Public URLs (No Login Required):**
- `/driver-applications/apply/` - Submit driver application
- `/driver-applications/status-check/` - Check application status

**Staff-Only URLs (Login Required):**
- `/` - Main dashboard
- `/realtime/` - Real-time monitoring
- `/analytics/` - Analytics dashboard
- `/maintenance/` - Maintenance management
- `/driver-applications/` - Application list
- `/driver-applications/<id>/` - Application detail/review
- `/admin/` - Django admin panel

**Authentication URLs:**
- `/login/` - Staff login
- `/logout/` - Logout
- `/admin/login/` - Django admin login

## 📞 Need Help?

If you encounter any issues:

1. Check the **Troubleshooting** section above
2. Verify Oracle is running: `sqlplus system/system123@localhost:1521/orclpdb`
3. Check if virtual environment is activated (you should see `(venv)` in terminal)
4. Ensure all dependencies are installed: `pip list`
5. Check Django logs for errors
6. Verify staff status: `python manage.py shell` → `from django.contrib.auth.models import User` → `User.objects.filter(is_staff=True)`

## 🔍 Debugging Tips

### Check Database Connection
```powershell
python manage.py test_oracle
```

### Check Current User Status
```powershell
python manage.py shell
```
```python
from django.contrib.auth.models import User
User.objects.all().values('username', 'is_staff', 'is_superuser')
```

### View All Parking Slots
```powershell
python manage.py show_slots
```

### Check System Health
```powershell
python manage.py system_health
```

### View Active Sessions
```powershell
python manage.py shell
```
```python
from parking_app.models import ParkingSession
ParkingSession.objects.filter(is_active=True).count()
```

---

## 🎉 You're All Set!

Your parking management system is now ready to use. This comprehensive guide covers:

✅ Complete installation and setup  
✅ All major features and functionalities  
✅ Database architecture and management  
✅ API endpoints for integration  
✅ Troubleshooting and debugging  
✅ Security features and access control  

Enjoy managing your parking facility with this enterprise-grade system! 🚗🅿️

---

**Last Updated:** December 2024  
**Version:** 2.0  
**Developer:** Atharv  
**Repository:** [github.com/atharv1296/SpotCheck](https://github.com/atharv1296/SpotCheck)
