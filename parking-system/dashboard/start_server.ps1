# Forbes Marshall SpotCheck - Oracle Server Startup Script
# PowerShell version for starting the parking management system

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "Forbes Marshall SpotCheck - Oracle Parking Management System" -ForegroundColor Cyan  
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "Starting Django server with Oracle database..." -ForegroundColor Green
Write-Host ""

# Add Oracle Client to PATH
$env:PATH += ";C:\oracle\instantclient_19_23"
Write-Host "✅ Oracle Client added to PATH" -ForegroundColor Green

# Navigate to project directory  
Set-Location "C:\Users\athar\OneDrive\Desktop\TY - Sem 1\EDI\parking-system\dashboard"
Write-Host "✅ Project directory: $(Get-Location)" -ForegroundColor Green

# Activate virtual environment
& "C:\Users\athar\OneDrive\Desktop\TY - Sem 1\EDI\venv\Scripts\Activate.ps1"
Write-Host "✅ Virtual environment activated" -ForegroundColor Green

Write-Host ""
Write-Host "================================================================" -ForegroundColor Yellow
Write-Host "Starting Django Development Server..." -ForegroundColor Yellow
Write-Host "Access your system at: http://127.0.0.1:8000/" -ForegroundColor White
Write-Host "Gate Interface: http://127.0.0.1:8000/gate/" -ForegroundColor White  
Write-Host "Admin Panel: http://127.0.0.1:8000/admin/" -ForegroundColor White
Write-Host "================================================================" -ForegroundColor Yellow
Write-Host ""

# Start Django server
python manage.py runserver

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")