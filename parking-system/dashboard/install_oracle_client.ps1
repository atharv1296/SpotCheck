# Oracle Instant Client Setup Script
# Forbes Marshall SpotCheck - Oracle Integration

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Oracle Instant Client Setup Script" -ForegroundColor Cyan  
Write-Host "Forbes Marshall SpotCheck Setup" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "⚠️  WARNING: Running without Administrator privileges" -ForegroundColor Yellow
    Write-Host "   Some operations may fail. Consider running PowerShell as Administrator." -ForegroundColor Yellow
    Write-Host ""
}

try {
    # Step 1: Create oracle directory
    Write-Host "[1/5] Creating Oracle directory..." -ForegroundColor Green
    $oracleDir = "C:\oracle"
    if (-not (Test-Path $oracleDir)) {
        New-Item -ItemType Directory -Path $oracleDir -Force | Out-Null
        Write-Host "✅ Created: $oracleDir" -ForegroundColor Green
    } else {
        Write-Host "✅ Directory exists: $oracleDir" -ForegroundColor Green
    }

    # Step 2: Download Oracle Instant Client
    Write-Host "[2/5] Downloading Oracle Instant Client..." -ForegroundColor Green
    $downloadUrl = "https://download.oracle.com/otn_software/nt/instantclient/1923000/instantclient-basic-windows.x64-19.23.0.0.0dbru.zip"
    $zipFile = "$oracleDir\instantclient-basic.zip"
    
    if (-not (Test-Path $zipFile)) {
        Write-Host "   Downloading from Oracle..." -ForegroundColor Yellow
        try {
            Invoke-WebRequest -Uri $downloadUrl -OutFile $zipFile -UseBasicParsing
            Write-Host "✅ Downloaded: $(Split-Path $zipFile -Leaf)" -ForegroundColor Green
        } catch {
            Write-Host "❌ Download failed with built-in method" -ForegroundColor Red
            Write-Host "   Trying alternative method..." -ForegroundColor Yellow
            
            # Try with curl
            & curl -L -o $zipFile $downloadUrl
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to download with curl"
            }
            Write-Host "✅ Downloaded with curl" -ForegroundColor Green
        }
    } else {
        Write-Host "✅ File already exists: $(Split-Path $zipFile -Leaf)" -ForegroundColor Green
    }

    # Step 3: Extract the ZIP file
    Write-Host "[3/5] Extracting Oracle Instant Client..." -ForegroundColor Green
    $extractPath = $oracleDir
    try {
        Expand-Archive -Path $zipFile -DestinationPath $extractPath -Force
        Write-Host "✅ Extracted to: $extractPath" -ForegroundColor Green
    } catch {
        Write-Host "❌ Extraction failed: $($_.Exception.Message)" -ForegroundColor Red
        throw
    }

    # Step 4: Find the instantclient directory
    Write-Host "[4/5] Locating Instant Client directory..." -ForegroundColor Green
    $instantClientDir = Get-ChildItem -Path $oracleDir -Directory | Where-Object { $_.Name -like "*instantclient*" } | Select-Object -First 1
    
    if ($instantClientDir) {
        $oracleClientPath = $instantClientDir.FullName
        Write-Host "✅ Found: $oracleClientPath" -ForegroundColor Green
        
        # Verify critical files exist
        $criticalFiles = @("oci.dll", "orannzsbb19.dll", "oraociei19.dll")
        foreach ($file in $criticalFiles) {
            if (Test-Path "$oracleClientPath\$file") {
                Write-Host "   ✅ $file" -ForegroundColor Green
            } else {
                Write-Host "   ❌ Missing: $file" -ForegroundColor Red
            }
        }
    } else {
        throw "Instant Client directory not found after extraction"
    }

    # Step 5: Update PATH environment variable
    Write-Host "[5/5] Updating PATH environment variable..." -ForegroundColor Green
    
    # Get current PATH
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
    
    if ($currentPath -notlike "*$oracleClientPath*") {
        try {
            $newPath = "$currentPath;$oracleClientPath"
            [Environment]::SetEnvironmentVariable("PATH", $newPath, "Machine")
            Write-Host "✅ Added to system PATH: $oracleClientPath" -ForegroundColor Green
            
            # Also update current session PATH
            $env:PATH += ";$oracleClientPath"
            Write-Host "✅ Updated current session PATH" -ForegroundColor Green
            
        } catch {
            Write-Host "❌ Failed to update system PATH (requires Administrator)" -ForegroundColor Red
            Write-Host "   Manually add this to your PATH: $oracleClientPath" -ForegroundColor Yellow
            
            # Update only current session
            $env:PATH += ";$oracleClientPath"
            Write-Host "✅ Updated current session PATH only" -ForegroundColor Yellow
        }
    } else {
        Write-Host "✅ PATH already contains Oracle Client directory" -ForegroundColor Green
    }

    # Final verification
    Write-Host "" -ForegroundColor White
    Write-Host "🎉 ORACLE INSTANT CLIENT SETUP COMPLETE!" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host "📍 Installation Location: $oracleClientPath" -ForegroundColor White
    Write-Host "📍 Added to PATH: YES" -ForegroundColor White
    Write-Host "" -ForegroundColor White
    Write-Host "🔄 NEXT STEPS:" -ForegroundColor Yellow
    Write-Host "1. Restart your PowerShell/Terminal" -ForegroundColor White
    Write-Host "2. Navigate to your Django project" -ForegroundColor White
    Write-Host "3. Run: python manage.py migrate" -ForegroundColor White
    Write-Host "4. Run: python manage.py seed_slots --two 15 --car 15 --large 5" -ForegroundColor White
    Write-Host "" -ForegroundColor White
    Write-Host "✅ Oracle Database connection should now work!" -ForegroundColor Green

} catch {
    Write-Host "" -ForegroundColor White
    Write-Host "❌ SETUP FAILED" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "" -ForegroundColor White
    Write-Host "📋 MANUAL INSTALLATION STEPS:" -ForegroundColor Yellow
    Write-Host "1. Visit: https://www.oracle.com/database/technologies/instant-client/winx64-64-downloads.html" -ForegroundColor White
    Write-Host "2. Download 'Basic Package (64-bit)'" -ForegroundColor White
    Write-Host "3. Extract to C:\oracle\instantclient_XX_X" -ForegroundColor White
    Write-Host "4. Add the path to your system PATH environment variable" -ForegroundColor White
    Write-Host "5. Restart PowerShell and try again" -ForegroundColor White
}

Write-Host "" -ForegroundColor White
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')