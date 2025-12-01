@echo off
echo ======================================
echo Oracle Instant Client Setup Script
echo Forbes Marshall SpotCheck Setup
echo ======================================

:: Create oracle directory
if not exist "C:\oracle" mkdir "C:\oracle"
cd /d "C:\oracle"

echo [1/4] Downloading Oracle Instant Client...
:: Download using curl (available in Windows 10+)
curl -L -o instantclient-basic.zip "https://download.oracle.com/otn_software/nt/instantclient/1923000/instantclient-basic-windows.x64-19.23.0.0.0dbru.zip"

if errorlevel 1 (
    echo ERROR: Failed to download Oracle Instant Client
    echo Please manually download from: https://www.oracle.com/database/technologies/instant-client/winx64-64-downloads.html
    pause
    exit /b 1
)

echo [2/4] Extracting Oracle Instant Client...
:: Extract using PowerShell
powershell -Command "Expand-Archive -Path 'instantclient-basic.zip' -DestinationPath 'C:\oracle' -Force"

if errorlevel 1 (
    echo ERROR: Failed to extract Oracle Instant Client
    pause
    exit /b 1
)

echo [3/4] Setting up PATH environment variable...
:: Add to PATH
set "ORACLE_PATH=C:\oracle\instantclient_19_23"
setx PATH "%PATH%;%ORACLE_PATH%" /M

echo [4/4] Verifying installation...
dir "C:\oracle\instantclient_19_23\oci.dll"

if exist "C:\oracle\instantclient_19_23\oci.dll" (
    echo ✅ SUCCESS: Oracle Instant Client installed successfully!
    echo ✅ Location: C:\oracle\instantclient_19_23
    echo ✅ PATH updated
    echo.
    echo ⚠️  IMPORTANT: Please restart your terminal/PowerShell
    echo ⚠️  for PATH changes to take effect!
    echo.
    echo After restarting terminal, test with:
    echo python manage.py migrate
) else (
    echo ❌ ERROR: Oracle Instant Client installation failed
    echo Please check the installation manually
)

echo.
pause