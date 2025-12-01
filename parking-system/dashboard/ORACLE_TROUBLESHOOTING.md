# Oracle Client PATH Issue - Troubleshooting Guide
# Forbes Marshall SpotCheck - Database Connection Fix

## Problem Description:
The error "DPI-1047: Cannot locate a 64-bit Oracle Client library" occurs when the Oracle Instant Client is not in the system PATH.

## Quick Fixes:

### Method 1: PowerShell Session Fix (Temporary)
```powershell
# Add Oracle Client to current session PATH
$env:PATH += ";C:\oracle\instantclient_19_23"

# Navigate to project and start server
cd "C:\Users\athar\OneDrive\Desktop\TY - Sem 1\EDI\parking-system\dashboard"
python manage.py runserver
```

### Method 2: Use Startup Scripts (Recommended)
```bash
# Windows Batch File
.\start_server.bat

# PowerShell Script
.\start_server.ps1
```

### Method 3: Permanent PATH Update (System-wide)
1. Open System Properties → Advanced → Environment Variables
2. Edit System PATH variable
3. Add: `C:\oracle\instantclient_19_23`
4. Restart PowerShell/CMD
5. Run: `python manage.py runserver`

### Method 4: One-Line Command
```powershell
$env:PATH += ";C:\oracle\instantclient_19_23" ; cd "C:\Users\athar\OneDrive\Desktop\TY - Sem 1\EDI\parking-system\dashboard" ; python manage.py runserver
```

## Verification Commands:
```powershell
# Check if Oracle Client is accessible
Test-Path "C:\oracle\instantclient_19_23\oci.dll"

# Check current PATH
$env:PATH -split ";" | Select-String "oracle"

# Test Django Oracle connection
python manage.py shell -c "from django.db import connection; connection.cursor(); print('Oracle connection successful!')"
```

## Success Indicators:
✅ Server starts without errors
✅ "Forbes Marshall SpotCheck ready!" message appears
✅ No DPI-1047 errors in console
✅ Dashboard accessible at http://127.0.0.1:8000/

## System Status:
- Oracle Database: ORCLPDB (localhost:1521) ✅
- Oracle Client: instantclient_19_23 ✅  
- Django: 5.1.1 with cx_Oracle 8.3.0 ✅
- Parking Slots: 60 slots created ✅
- Sample Data: 4 vehicles registered ✅

## Access Points:
- Main Dashboard: http://127.0.0.1:8000/
- Gate Interface: http://127.0.0.1:8000/gate/
- Admin Panel: http://127.0.0.1:8000/admin/ (admin/admin123)
- API Endpoints: http://127.0.0.1:8000/api/

## Notes:
- The Oracle Client path must be added to PATH before starting Django
- Virtual environment must be activated
- Oracle database service must be running
- Use startup scripts to avoid manual PATH configuration