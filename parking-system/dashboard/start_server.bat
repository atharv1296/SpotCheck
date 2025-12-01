@echo off
echo ================================================================
echo Forbes Marshall SpotCheck - Oracle Parking Management System
echo ================================================================
echo Starting Django server with Oracle database...
echo.

REM Add Oracle Client to PATH
set "PATH=%PATH%;C:\oracle\instantclient_19_23"

REM Navigate to project directory
cd /d "C:\Users\athar\OneDrive\Desktop\TY - Sem 1\EDI\parking-system\dashboard"

REM Activate virtual environment and start server
call "C:\Users\athar\OneDrive\Desktop\TY - Sem 1\EDI\venv\Scripts\activate.bat"

echo Oracle Client added to PATH
echo Project directory: %cd%
echo Virtual environment activated
echo.
echo ================================================================
echo Starting Django Development Server...
echo Access your system at: http://127.0.0.1:8000/
echo Gate Interface: http://127.0.0.1:8000/gate/
echo Admin Panel: http://127.0.0.1:8000/admin/
echo ================================================================
echo.

python manage.py runserver

pause