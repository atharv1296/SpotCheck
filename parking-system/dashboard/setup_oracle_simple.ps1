# Simple Oracle Instant Client Installation Script
Write-Host "Oracle Instant Client Setup for Forbes Marshall SpotCheck" -ForegroundColor Cyan

try {
    # Create directory
    Write-Host "Creating Oracle directory..." -ForegroundColor Green
    New-Item -ItemType Directory -Path "C:\oracle" -Force | Out-Null
    
    # Download Oracle Instant Client
    Write-Host "Downloading Oracle Instant Client..." -ForegroundColor Green
    $url = "https://download.oracle.com/otn_software/nt/instantclient/1923000/instantclient-basic-windows.x64-19.23.0.0.0dbru.zip"
    $output = "C:\oracle\instantclient.zip"
    
    Invoke-WebRequest -Uri $url -OutFile $output -UseBasicParsing
    Write-Host "Download completed!" -ForegroundColor Green
    
    # Extract
    Write-Host "Extracting files..." -ForegroundColor Green
    Expand-Archive -Path $output -DestinationPath "C:\oracle" -Force
    
    # Find instant client directory
    $clientDir = Get-ChildItem "C:\oracle" -Directory | Where-Object { $_.Name -like "*instant*" } | Select-Object -First 1
    
    if ($clientDir) {
        Write-Host "Oracle Instant Client installed to: $($clientDir.FullName)" -ForegroundColor Green
        
        # Add to PATH
        $currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
        if ($currentPath -notlike "*$($clientDir.FullName)*") {
            $newPath = "$currentPath;$($clientDir.FullName)"
            [Environment]::SetEnvironmentVariable("PATH", $newPath, "Machine")
            $env:PATH += ";$($clientDir.FullName)"
            Write-Host "Added to PATH successfully!" -ForegroundColor Green
        }
        
        Write-Host "Setup completed! Please restart your terminal and try: python manage.py migrate" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "Error occurred: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please download manually from Oracle website" -ForegroundColor Yellow
}

Read-Host "Press Enter to continue"